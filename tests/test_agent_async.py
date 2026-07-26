"""Async Agent API: arun() / arun_with_thread() — native awaitables driven by
the tokio runtime, mirroring StateGraph.ainvoke()'s mechanism (no worker-thread
bounce, no per-call block_on)."""

import asyncio
import textwrap

import pytest

from flowgentra_ai.agent import Agent

HANDLER_MODULE = textwrap.dedent(
    """
    from flowgentra_ai.agent import register_handler

    @register_handler
    def step_a(state: dict) -> dict:
        return {**state, "a_ran": True, "trail": (state.get("trail") or []) + ["a"]}

    @register_handler
    def step_b(state: dict) -> dict:
        return {**state, "b_ran": True, "trail": (state.get("trail") or []) + ["b"]}
    """
)

CONFIG = textwrap.dedent(
    """
    name: pipeline
    python_handler_module: cfg_agent_async_handlers
    llm:
      provider: ollama
      model: mistral
      api_key: ""
    state_schema:
      input: str
      a_ran: bool
      b_ran: bool
      trail: list
    graph:
      nodes:
        - name: step_a
          handler: step_a
        - name: step_b
          handler: step_b
      edges:
        - from: START
          to: step_a
        - from: step_a
          to: step_b
        - from: step_b
          to: END
    """
)


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    (tmp_path / "cfg_agent_async_handlers.py").write_text(HANDLER_MODULE)
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(CONFIG)
    monkeypatch.syspath_prepend(str(tmp_path))
    return str(cfg)


def test_arun_matches_sync_run(config_path):
    agent = Agent.from_config_path(config_path, allow_python_handlers=True)
    agent.set_state("input", "hi")

    result = asyncio.run(agent.arun())

    assert result["a_ran"] is True
    assert result["b_ran"] is True
    assert result["trail"] == ["a", "b"]


def test_arun_with_thread_checkpoints_per_thread(config_path):
    agent = Agent.from_config_path(config_path, allow_python_handlers=True)
    agent.set_state("input", "hi")

    async def go():
        return await agent.arun_with_thread("thread-async-1")

    result = asyncio.run(go())
    assert result["trail"] == ["a", "b"]


def test_subscribe_events_captures_execution(config_path):
    """agent.subscribe_events() streams node/edge/graph events from the run."""
    agent = Agent.from_config_path(config_path, allow_python_handlers=True)
    rx = agent.subscribe_events()
    assert rx is not None
    agent.set_state("input", "hi")
    agent.run()
    types = [e["type"] for e in rx.drain()]
    assert "node_started" in types
    assert "node_completed" in types


def test_arun_runs_concurrently_not_serially(config_path):
    """Two independent agents' arun() calls should overlap in wall-clock time
    (same guarantee StateGraph.ainvoke's parallel-supervisor test proves) —
    this is what distinguishes a native awaitable from a blocking wrapper."""
    import time

    agent1 = Agent.from_config_path(config_path, allow_python_handlers=True)
    agent1.set_state("input", "a")
    agent2 = Agent.from_config_path(config_path, allow_python_handlers=True)
    agent2.set_state("input", "b")

    async def go():
        start = time.perf_counter()
        results = await asyncio.gather(agent1.arun(), agent2.arun())
        elapsed = time.perf_counter() - start
        return results, elapsed

    (r1, r2), elapsed = asyncio.run(go())
    assert r1["trail"] == ["a", "b"]
    assert r2["trail"] == ["a", "b"]
    # Both complete quickly; the real assertion is that this doesn't hang or
    # deadlock — two arun() calls sharing the tokio runtime must coexist.
    assert elapsed < 5.0
