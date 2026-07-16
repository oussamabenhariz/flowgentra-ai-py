"""Checkpointing: persistence round-trip, interrupt/resume, corruption."""

import json
import os
from typing import List, TypedDict

import pytest

from flowgentra_ai.graph import StateGraph, END
from flowgentra_ai.exceptions import AgentExecutionError


class PubState(TypedDict):
    topic: str
    log: List[str]


def draft(s):
    return {**s, "log": s["log"] + ["draft"]}


def publish(s):
    return {**s, "log": s["log"] + ["publish"]}


def make_graph(tmp_path, interrupt_before=None):
    b = StateGraph(PubState)
    b.add_node("draft", draft)
    b.add_node("publish", publish)
    b.set_entry_point("draft")
    b.add_edge("draft", "publish")
    b.add_edge("publish", END)
    if interrupt_before:
        b.interrupt_before(interrupt_before)
    b.set_checkpointer(str(tmp_path))
    return b.compile()


def test_invoke_with_thread_persists_state(tmp_path):
    g = make_graph(tmp_path)
    result = g.invoke_with_thread("t1", {"topic": "AI", "log": []})
    assert result["log"] == ["draft", "publish"]
    # Checkpoint files exist on disk
    files = list((tmp_path / "t1").glob("*.json"))
    assert files, "no checkpoint files written"


def test_interrupt_before_pauses(tmp_path):
    g = make_graph(tmp_path, interrupt_before="publish")
    with pytest.raises(AgentExecutionError, match="breakpoint"):
        g.invoke_with_thread("t1", {"topic": "AI", "log": []})


def test_resume_continues_past_breakpoint(tmp_path):
    """Regression: resume() used to restart from the entry point and
    re-trigger the same breakpoint forever."""
    g = make_graph(tmp_path, interrupt_before="publish")
    with pytest.raises(AgentExecutionError):
        g.invoke_with_thread("t1", {"topic": "AI", "log": []})

    result = g.resume("t1")
    # draft ran exactly once; publish actually ran after resume.
    assert result["log"] == ["draft", "publish"]


def test_resume_with_state_injects_updates(tmp_path):
    g = make_graph(tmp_path, interrupt_before="publish")
    with pytest.raises(AgentExecutionError):
        g.invoke_with_thread("t1", {"topic": "AI", "log": []})

    result = g.resume_with_state("t1", {"topic": "AI (reviewed)"})
    assert result["topic"] == "AI (reviewed)"
    assert result["log"][-1] == "publish"


def test_checkpoint_files_have_schema_version(tmp_path):
    g = make_graph(tmp_path)
    g.invoke_with_thread("t1", {"topic": "AI", "log": []})
    files = sorted((tmp_path / "t1").glob("step_*.json"))
    data = json.loads(files[0].read_text())
    assert data["schema_version"] == "1.0"


def test_corrupt_checkpoint_raises_actionable_error(tmp_path):
    g = make_graph(tmp_path)
    g.invoke_with_thread("t1", {"topic": "AI", "log": []})
    # Truncate every checkpoint to simulate crash damage.
    for f in (tmp_path / "t1").glob("step_*.json"):
        f.write_text('{"thread_id": "t1", "ste')
    with pytest.raises(Exception, match="corrupt"):
        g.resume("t1")


def test_path_traversal_thread_id_rejected(tmp_path):
    g = make_graph(tmp_path)
    with pytest.raises(Exception, match="thread_id"):
        g.invoke_with_thread("../escape", {"topic": "AI", "log": []})
    # Nothing escaped the checkpoint root.
    assert not (tmp_path.parent / "escape").exists()


def test_thread_isolation(tmp_path):
    g = make_graph(tmp_path)
    r1 = g.invoke_with_thread("alpha", {"topic": "A", "log": []})
    r2 = g.invoke_with_thread("beta", {"topic": "B", "log": []})
    assert r1["topic"] == "A"
    assert r2["topic"] == "B"
    assert (tmp_path / "alpha").exists() and (tmp_path / "beta").exists()
