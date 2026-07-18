"""Graph and state management for workflow execution.

This module provides the core graph building and execution APIs compatible with LangGraph.

Examples:
    Build a simple state graph:

        from flowgentra_ai.graph import StateGraph, END

        class MyState(TypedDict):
            messages: List[str]
            score: float

        def process_node(state: dict) -> dict:
            return {"messages": state["messages"] + ["processed"]}

        builder = StateGraph(MyState)
        builder.add_node("process", process_node)
        builder.set_entry_point("process")
        builder.add_edge("process", END)
        graph = builder.compile()
"""

from __future__ import annotations

import asyncio

from flowgentra_ai._native import graph as _g, nodes as _n
from flowgentra_ai._native import NodeInterrupt

# Re-export compiled types directly.
CompiledGraph = _g.CompiledGraph
GraphStream = _g.GraphStream
END = _g.END
MessageGraph = _n.MessageGraph
MessageGraphBuilder = _n.MessageGraphBuilder


async def _ainvoke(self, input_dict: dict) -> dict:
    """Async variant of invoke().

    Drives the graph on the tokio runtime via a native awaitable
    (pyo3-async-runtimes) — no worker-thread bounce and no per-call block_on.
    This ``async def`` wrapper defers the native call until it runs inside the
    awaiting event loop, so both ``await g.ainvoke(x)`` and
    ``asyncio.run(g.ainvoke(x))`` work.

    Example:
        result = await graph.ainvoke({"messages": []})
    """
    return await self._ainvoke_native(input_dict)


async def _astream(self, input_dict: dict):
    """Async variant of stream(): an async iterator of event dicts.

    Example:
        async for event in graph.astream({"messages": []}):
            print(event["type"])
    """
    stream = self.stream(input_dict)
    sentinel = object()
    while True:
        event = await asyncio.to_thread(next, stream, sentinel)
        if event is sentinel:
            return
        yield event


# `ainvoke` wraps the native `_ainvoke_native` awaitable in an `async def` so
# the native future is created inside the running loop (see _ainvoke). `astream`
# stays a thin async generator over the GIL-releasing sync stream (pull-based; a
# native async generator would need pyo3-async-runtimes' unstable-streams).
CompiledGraph.ainvoke = _ainvoke
CompiledGraph.astream = _astream

# Default matches the Rust runtime default (config/mod.rs `default_recursion_limit`).
_DEFAULT_MAX_STEPS: int = 25
_MAX_STEPS_UPPER: int = 10_000


class StateGraph(_g.StateGraph):
    """Thin wrapper around the compiled Rust StateGraph.

    Adds Python-level validation so that ``set_max_steps()`` enforces the same
    1–10 000 bounds as the Rust config validator, making cross-language behaviour
    consistent and giving developers an early error instead of a hard-to-debug
    runtime abort.
    """

    def set_max_steps(self, max_steps: int) -> None:
        """Set the maximum number of execution iterations (default: 25, range: 1–10 000).

        Matches the Rust ``graph.recursion_limit`` field.  Values outside the
        accepted range raise ``ValueError`` immediately so the error surface is
        at configuration time, not mid-execution.

        Args:
            max_steps: Maximum loop iterations before the graph raises
                ``RecursionLimitExceeded``.
        """
        if not isinstance(max_steps, int):
            raise TypeError(f"max_steps must be an int, got {type(max_steps).__name__}")
        if max_steps < 1 or max_steps > _MAX_STEPS_UPPER:
            raise ValueError(
                f"max_steps must be between 1 and {_MAX_STEPS_UPPER}, got {max_steps}"
            )
        super().set_max_steps(max_steps)


__all__ = [
    "StateGraph",
    "CompiledGraph",
    "GraphStream",
    "NodeInterrupt",
    "MessageGraph",
    "MessageGraphBuilder",
    "END",
]
