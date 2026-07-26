"""Feature batch: node context arg, structured output, multimodal messages,
MockLLM.when, agent event subscription, langgraph_compat."""

from typing import List, TypedDict

import pytest

from flowgentra_ai.graph import StateGraph, END, Command, NodeInterrupt
from flowgentra_ai.llm import LLM, Message, MockLLM


# ── #3 Python node context arg ────────────────────────────────────────────────

class GateState(TypedDict):
    messages: List[str]
    answer: str


def test_two_arg_node_receives_context_resume_value(tmp_path):
    """A node declaring (state, ctx) reads ctx.resume_value on resume via
    Command(resume=...)."""
    def gate(state, ctx):
        rv = ctx.resume_value
        if rv is not None:
            return {"messages": state["messages"] + [f"got:{rv}"]}
        raise NodeInterrupt({"q": "approve?"})

    b = StateGraph(GateState)
    b.add_node("gate", gate)
    b.set_entry_point("gate")
    b.add_edge("gate", END)
    b.set_checkpointer(str(tmp_path))
    g = b.compile()

    with pytest.raises(NodeInterrupt):
        g.invoke_with_thread("t1", {"messages": [], "answer": ""})

    result = g.resume_command("t1", Command(resume="yes"))
    assert result["messages"] == ["got:yes"]


def test_one_arg_node_still_works():
    """Classic single-arg (state) nodes are unaffected."""
    class S(TypedDict):
        x: int

    b = StateGraph(S)
    b.add_node("inc", lambda state: {"x": state["x"] + 1})
    b.set_entry_point("inc")
    b.add_edge("inc", END)
    g = b.compile()
    assert g.invoke({"x": 1})["x"] == 2


def test_node_context_exposes_node_name():
    class S(TypedDict):
        name: str

    def n(state, ctx):
        return {"name": ctx.node_name}

    b = StateGraph(S)
    b.add_node("mynode", n)
    b.set_entry_point("mynode")
    b.add_edge("mynode", END)
    g = b.compile()
    assert g.invoke({"name": ""})["name"] == "mynode"


# ── #2 Structured output ──────────────────────────────────────────────────────

def test_with_structured_output_dict_schema():
    llm = MockLLM.always('{"sentiment": "positive", "score": 0.9}').as_llm()
    schema = {"type": "object", "properties": {"sentiment": {"type": "string"}}}
    structured = llm.with_structured_output(schema)
    result = structured.invoke([Message.user("Classify: I love it")])
    assert result["sentiment"] == "positive"


def test_with_structured_output_strips_markdown_fence():
    llm = MockLLM.always('```json\n{"ok": true}\n```').as_llm()
    structured = llm.with_structured_output({"type": "object"})
    assert structured.invoke("go") == {"ok": True}


def test_with_structured_output_pydantic():
    pydantic = pytest.importorskip("pydantic")

    class Sentiment(pydantic.BaseModel):
        label: str
        score: float

    llm = MockLLM.always('{"label": "pos", "score": 0.8}').as_llm()
    structured = llm.with_structured_output(Sentiment)
    result = structured.invoke("Classify: great")
    assert isinstance(result, Sentiment)
    assert result.label == "pos"
    assert result.score == 0.8


# ── #1 Multimodal messages ────────────────────────────────────────────────────

def test_message_with_image_url():
    msg = Message.user("what's this?", images=["https://ex.com/cat.jpg"])
    imgs = msg.images
    assert len(imgs) == 1
    assert imgs[0]["url"] == "https://ex.com/cat.jpg"


def test_message_with_image_dict_and_detail():
    msg = Message.user("describe", images=[{"url": "data:image/png;base64,AAAA", "detail": "high"}])
    assert msg.images[0]["detail"] == "high"


def test_message_with_image_builder():
    msg = Message.user("x").with_image("https://ex.com/a.png", detail="low")
    assert msg.images[0]["url"] == "https://ex.com/a.png"
    assert msg.images[0]["detail"] == "low"


def test_text_only_message_has_no_images():
    assert Message.user("hello").images == []


# ── #8 MockLLM.when(predicate) ────────────────────────────────────────────────

def test_mock_llm_when_predicate():
    mock = MockLLM()
    mock.when(lambda msgs: "escalate" if len(msgs) > 2 else None)
    mock.otherwise("normal")
    llm = mock.as_llm()

    one = llm.chat([Message.user("hi")])
    assert one.content == "normal"

    many = llm.chat([Message.user("a"), Message.assistant("b"), Message.user("c")])
    assert many.content == "escalate"


def test_mock_llm_when_reads_content():
    mock = MockLLM()
    mock.when(lambda msgs: "yes" if msgs[-1]["content"] == "ping" else None)
    mock.otherwise("no")
    llm = mock.as_llm()
    assert llm.chat([Message.user("ping")]).content == "yes"
    assert llm.chat([Message.user("other")]).content == "no"


# ── langgraph_compat ──────────────────────────────────────────────────────────

def test_langgraph_compat_start_edge_and_invoke_config():
    from flowgentra_ai.langgraph_compat import StateGraph as LGStateGraph, START, END as LGEND

    class S(TypedDict):
        x: int

    g = LGStateGraph(S)
    g.add_node("work", lambda state: {"x": state["x"] + 10})
    g.add_edge(START, "work")   # START edge -> entry point
    g.add_edge("work", LGEND)
    app = g.compile()

    # LangGraph-style config with thread_id
    result = app.invoke({"x": 1}, config={"configurable": {"thread_id": "t1"}})
    assert result["x"] == 11


def test_langgraph_compat_memorysaver_and_url_checkpointer(tmp_path):
    from flowgentra_ai.langgraph_compat import StateGraph as LGStateGraph, START, END as LGEND, MemorySaver

    class S(TypedDict):
        x: int

    g = LGStateGraph(S)
    g.add_node("work", lambda state: {"x": state["x"] + 1})
    g.add_edge(START, "work")
    g.add_edge("work", LGEND)
    # MemorySaver accepted for source-compat; path string maps to file checkpointer.
    app = g.compile(checkpointer=MemorySaver())
    assert app.invoke({"x": 0})["x"] == 1
