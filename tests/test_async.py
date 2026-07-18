"""Async API: ainvoke, astream, concurrency."""

import asyncio
from typing import List, TypedDict

import pytest

from flowgentra_ai.graph import StateGraph, END


class S(TypedDict):
    x: int
    log: List[str]


def build(delay: float = 0.0):
    def work(state):
        if delay:
            import time

            time.sleep(delay)
        return {**state, "x": state["x"] + 1, "log": state["log"] + ["work"]}

    b = StateGraph(S)
    b.add_node("work", work)
    b.set_entry_point("work")
    b.add_edge("work", END)
    return b.compile()


def test_ainvoke_returns_result():
    g = build()
    result = asyncio.run(g.ainvoke({"x": 1, "log": []}))
    assert result["x"] == 2


def test_astream_yields_events():
    g = build()

    async def collect():
        return [e["type"] async for e in g.astream({"x": 1, "log": []})]

    types = asyncio.run(collect())
    assert "node_completed" in types
    assert types[-1] == "values"


def test_astream_early_break_does_not_hang():
    """Breaking out of `async for` mid-stream must not deadlock or leak the
    native stream (the classic hazard of bridged async iterators)."""
    g = build()

    async def first_only():
        agen = g.astream({"x": 1, "log": []})
        async for event in agen:
            return event["type"]

    assert asyncio.run(first_only()) == "graph_started"


def test_astream_exhausted_raises_stop_async_iteration():
    g = build()

    async def drain_then_next():
        agen = g.astream({"x": 1, "log": []})
        async for _ in agen:
            pass
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()

    asyncio.run(drain_then_next())


def test_concurrent_ainvoke_actually_parallelizes():
    """Two 100ms graphs concurrently should take well under 200ms
    (nodes release the GIL only around Rust work, but asyncio.to_thread
    lets the sleeps overlap since time.sleep releases the GIL)."""
    import time

    g = build(delay=0.1)

    async def both():
        t0 = time.perf_counter()
        await asyncio.gather(
            g.ainvoke({"x": 1, "log": []}),
            g.ainvoke({"x": 2, "log": []}),
        )
        return time.perf_counter() - t0

    elapsed = asyncio.run(both())
    assert elapsed < 0.19, f"concurrent invokes serialized: {elapsed:.3f}s"


def test_ainvoke_propagates_node_error():
    def boom(state):
        raise ValueError("async boom")

    b = StateGraph(S)
    b.add_node("boom", boom)
    b.set_entry_point("boom")
    b.add_edge("boom", END)
    g = b.compile()

    with pytest.raises(Exception, match="async boom"):
        asyncio.run(g.ainvoke({"x": 1, "log": []}))
