"""StateGraph build, invoke, routing, validation, budgets, streaming."""

import time
from typing import List, TypedDict

import pytest

from flowgentra_ai.graph import StateGraph, END
from flowgentra_ai.exceptions import (
    ValidationError,
    WorkflowTimeoutError,
)


class ChainState(TypedDict):
    x: int
    log: List[str]


class RouteState(TypedDict):
    kind: str
    out: str


def test_invoke_linear_chain(chain_graph):
    result = chain_graph.invoke({"x": 1, "log": []})
    assert result["x"] == 4
    assert result["log"] == ["inc", "double"]


def test_invoke_returns_plain_dict(chain_graph):
    result = chain_graph.invoke({"x": 0, "log": []})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"x", "log"}


def test_conditional_routing():
    def classify(s):
        return {**s, "out": ""}

    def hi(s):
        return {**s, "out": "hi"}

    def bye(s):
        return {**s, "out": "bye"}

    def router(s):
        return "hi" if s["kind"] == "greeting" else "bye"

    b = StateGraph(RouteState)
    b.add_node("classify", classify)
    b.add_node("hi", hi)
    b.add_node("bye", bye)
    b.set_entry_point("classify")
    b.add_conditional_edge("classify", router)
    b.add_edge("hi", END)
    b.add_edge("bye", END)
    g = b.compile()

    assert g.invoke({"kind": "greeting", "out": ""})["out"] == "hi"
    assert g.invoke({"kind": "question", "out": ""})["out"] == "bye"


def test_add_conditional_edges_plural_alias():
    def one(s):
        return {**s, "x": s["x"] + 1}

    b = StateGraph(ChainState)
    b.add_node("one", one)
    b.set_entry_point("one")
    b.add_conditional_edges("one", lambda s: END)
    g = b.compile()
    assert g.invoke({"x": 0, "log": []})["x"] == 1


def test_compile_requires_entry_point():
    b = StateGraph(ChainState)
    b.add_node("inc", lambda s: s)
    with pytest.raises(Exception, match="[Ee]ntry point"):
        b.compile()


def test_invoke_rejects_unknown_key(chain_graph):
    with pytest.raises(KeyError):
        chain_graph.invoke({"x": 1, "log": [], "bogus": True})


def test_invoke_rejects_missing_required_key(chain_graph):
    with pytest.raises(ValidationError):
        chain_graph.invoke({"x": 1})


def test_node_exception_propagates():
    class BoomError(RuntimeError):
        pass

    def boom(s):
        raise BoomError("node blew up")

    b = StateGraph(ChainState)
    b.add_node("boom", boom)
    b.set_entry_point("boom")
    b.add_edge("boom", END)
    g = b.compile()
    with pytest.raises(Exception, match="node blew up"):
        g.invoke({"x": 1, "log": []})


def test_max_steps_limit():
    def spin(s):
        return {**s, "x": s["x"] + 1}

    b = StateGraph(ChainState)
    b.add_node("spin", spin)
    b.set_entry_point("spin")
    b.add_conditional_edge("spin", lambda s: "spin")
    b.set_max_steps(5)
    g = b.compile()
    with pytest.raises(Exception, match="[Mm]ax steps"):
        g.invoke({"x": 0, "log": []})


def test_wall_clock_budget():
    def slow(s):
        time.sleep(0.05)
        return {**s, "x": s["x"] + 1}

    b = StateGraph(ChainState)
    b.add_node("slow", slow)
    b.set_entry_point("slow")
    b.add_conditional_edge("slow", lambda s: "slow")
    b.set_max_duration(0.15)
    g = b.compile()
    with pytest.raises(WorkflowTimeoutError):
        g.invoke({"x": 0, "log": []})


def test_stream_yields_events_and_final_values(chain_graph):
    events = list(chain_graph.stream({"x": 1, "log": []}))
    types = [e["type"] for e in events]
    assert types[0] == "graph_started"
    assert types.count("node_completed") == 2
    assert types[-1] == "values"
    assert events[-1]["state"]["x"] == 4


def test_stream_raises_on_node_failure():
    def boom(s):
        raise ValueError("stream boom")

    b = StateGraph(ChainState)
    b.add_node("boom", boom)
    b.set_entry_point("boom")
    b.add_edge("boom", END)
    g = b.compile()
    with pytest.raises(Exception, match="stream boom"):
        list(g.stream({"x": 1, "log": []}))


def test_reentrant_node_invoking_subgraph(chain_graph):
    """A Python node that itself invokes another compiled graph (reentrancy)."""

    def outer(s):
        inner_result = chain_graph.invoke({"x": s["x"], "log": []})
        return {**s, "x": inner_result["x"], "log": s["log"] + ["outer"]}

    b = StateGraph(ChainState)
    b.add_node("outer", outer)
    b.set_entry_point("outer")
    b.add_edge("outer", END)
    g = b.compile()
    result = g.invoke({"x": 1, "log": []})
    assert result["x"] == 4
    assert result["log"] == ["outer"]


class MsgState(TypedDict):
    messages: List[str]


def test_add_subgraph_merges_result_into_parent():
    """builder.add_subgraph() wires a compiled graph in as a single node —
    its final state must merge into the parent, not get discarded."""

    def inner_echo(s):
        return {"messages": s["messages"] + ["from_subgraph"]}

    inner_builder = StateGraph(MsgState)
    inner_builder.add_node("inner_echo", inner_echo)
    inner_builder.set_entry_point("inner_echo")
    inner_builder.add_edge("inner_echo", END)
    subgraph = inner_builder.compile()

    def outer_start(s):
        return {"messages": s["messages"] + ["outer"]}

    outer_builder = StateGraph(MsgState)
    outer_builder.add_node("outer_start", outer_start)
    outer_builder.add_subgraph("sub", subgraph)
    outer_builder.set_entry_point("outer_start")
    outer_builder.add_edge("outer_start", "sub")
    outer_builder.add_edge("sub", END)
    graph = outer_builder.compile()

    result = graph.invoke({"messages": []})
    # Both the outer node's message and the subgraph's message, exactly once —
    # no drop and no double-count of the Append-reduced "messages" field.
    assert result["messages"] == ["outer", "from_subgraph"]
