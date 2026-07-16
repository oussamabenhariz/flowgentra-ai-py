"""Stage-8 features: SQLite checkpointer, cached nodes, in-node interrupt,
per-key parallel merge, OTel export."""

import json
from typing import Annotated, List, TypedDict

import pytest

from flowgentra_ai.graph import StateGraph, END, NodeInterrupt


class PubState(TypedDict):
    topic: str
    log: List[str]


def draft(s):
    return {**s, "log": s["log"] + ["draft"]}


def publish(s):
    return {**s, "log": s["log"] + ["publish"]}


# ── SQLite checkpointer ───────────────────────────────────────────────────────

def make_sqlite_graph(db_path, interrupt_before=None):
    b = StateGraph(PubState)
    b.add_node("draft", draft)
    b.add_node("publish", publish)
    b.set_entry_point("draft")
    b.add_edge("draft", "publish")
    b.add_edge("publish", END)
    if interrupt_before:
        b.interrupt_before(interrupt_before)
    b.set_sqlite_checkpointer(f"sqlite://{db_path}")
    return b.compile()


def test_sqlite_checkpointer_persists(tmp_path):
    db = (tmp_path / "cp.db").as_posix()
    g = make_sqlite_graph(db)
    result = g.invoke_with_thread("t1", {"topic": "AI", "log": []})
    assert result["log"] == ["draft", "publish"]
    assert (tmp_path / "cp.db").exists()


def test_sqlite_interrupt_resume(tmp_path):
    db = (tmp_path / "cp.db").as_posix()
    g = make_sqlite_graph(db, interrupt_before="publish")
    with pytest.raises(Exception, match="breakpoint"):
        g.invoke_with_thread("t1", {"topic": "AI", "log": []})
    result = g.resume("t1")
    assert result["log"] == ["draft", "publish"]


def test_sqlite_thread_survives_new_graph_instance(tmp_path):
    """Durable execution: a fresh process/graph resumes from the DB."""
    db = (tmp_path / "cp.db").as_posix()
    g1 = make_sqlite_graph(db, interrupt_before="publish")
    with pytest.raises(Exception):
        g1.invoke_with_thread("t1", {"topic": "AI", "log": []})
    # New compiled graph, same database — state must still be there.
    g2 = make_sqlite_graph(db, interrupt_before="publish")
    result = g2.resume("t1")
    assert result["log"] == ["draft", "publish"]


# ── Cached nodes ──────────────────────────────────────────────────────────────

def test_cached_node_skips_reexecution():
    calls = {"n": 0}

    class CacheState(TypedDict):
        x: int
        out: int

    def expensive(s):
        calls["n"] += 1
        return {**s, "out": s["x"] * 10}

    b = StateGraph(CacheState)
    b.add_cached_node("expensive", expensive, max_entries=8)
    b.set_entry_point("expensive")
    b.add_edge("expensive", END)
    g = b.compile()

    r1 = g.invoke({"x": 3, "out": 0})
    r2 = g.invoke({"x": 3, "out": 0})  # identical input state → cache hit
    r3 = g.invoke({"x": 4, "out": 0})  # different state → miss
    assert r1["out"] == r2["out"] == 30
    assert r3["out"] == 40
    assert calls["n"] == 2, f"expected 2 executions, got {calls['n']}"


# ── In-node interrupt (human-in-the-loop) ────────────────────────────────────

class ApprovalState(TypedDict):
    doc: str
    approval: str
    log: List[str]


def test_node_interrupt_pauses_and_resume_injects_answer(tmp_path):
    def gate(s):
        if not s["approval"]:
            raise NodeInterrupt({"question": "approve?", "doc": s["doc"]})
        return {**s, "log": s["log"] + [f"gate:{s['approval']}"]}

    b = StateGraph(ApprovalState)
    b.add_node("gate", gate)
    b.set_entry_point("gate")
    b.add_edge("gate", END)
    b.set_checkpointer(str(tmp_path))
    g = b.compile()

    with pytest.raises(NodeInterrupt) as excinfo:
        g.invoke_with_thread("t1", {"doc": "draft-1", "approval": "", "log": []})
    payload = excinfo.value.args[0]
    assert payload["question"] == "approve?"
    assert payload["doc"] == "draft-1"

    result = g.resume_with_state("t1", {"approval": "yes"})
    assert result["log"] == ["gate:yes"]


def test_node_interrupt_inherits_base_exception():
    from flowgentra_ai.exceptions import FlowgentraAIError, NodeInterrupt as NI

    assert issubclass(NI, FlowgentraAIError)
    assert NI is NodeInterrupt


# ── Per-key reducers in parallel supersteps ──────────────────────────────────

def test_parallel_branches_merge_with_append_reducer():
    import operator

    class FanState(TypedDict):
        query: str
        findings: Annotated[List[str], operator.add]
        winner: str

    def branch_a(s):
        return {"findings": ["from-a"], "winner": "a"}

    def branch_b(s):
        return {"findings": ["from-b"], "winner": "b"}

    b = StateGraph(FanState)
    b.add_parallel_node("fan", {"a": branch_a, "b": branch_b})
    b.set_entry_point("fan")
    b.add_edge("fan", END)
    g = b.compile()

    result = g.invoke({"query": "q", "findings": ["seed"], "winner": ""})
    # Append-reduced field accumulates every branch's contribution.
    assert result["findings"] == ["seed", "from-a", "from-b"], result["findings"]
    # LastValue field: last branch in declaration order wins.
    assert result["winner"] == "b"


# ── OpenTelemetry export ──────────────────────────────────────────────────────

def test_trace_converts_to_otlp_spans():
    from flowgentra_ai.observability import (
        ExecutionTrace,
        trace_to_otel_spans,
        spans_to_otlp_json,
    )

    trace_json = {
        "trace_id": "test-trace-1",
        "agent_name": "unit",
        "start_time": "2026-07-16T00:00:00Z",
        "end_time": "2026-07-16T00:00:01Z",
        "status": "Completed",
        "node_timings": [
            {
                "node_name": "draft",
                "duration_ms": 12,
                "start_time": "2026-07-16T00:00:00Z",
                "state_snapshot": None,
            }
        ],
        "path_segments": [],
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "metadata": {},
    }
    trace = ExecutionTrace.from_json(json.dumps(trace_json))
    spans = trace_to_otel_spans(trace)
    assert len(spans) >= 2  # root span + one node span
    ops = [s.operation_name for s in spans]
    assert any("draft" in op for op in ops), ops

    payload = spans_to_otlp_json(spans)
    assert "resourceSpans" in payload
