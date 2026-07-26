"""LangGraph compatibility shim.

Flowgentra's graph API already mirrors LangGraph's (``StateGraph``,
``add_node`` / ``add_edge`` / ``add_conditional_edges``, ``compile``,
``invoke``, ``END``, ``Command``, ``interrupt_before`` / ``interrupt_after``).
This module smooths over the small remaining differences so that most LangGraph
code runs against Flowgentra's Rust core with minimal edits:

- adds the ``START`` sentinel (Flowgentra uses ``set_entry_point`` natively),
- lets ``compile(checkpointer=...)`` set the checkpointer the LangGraph way,
- lets ``invoke(state, config={"configurable": {"thread_id": ...}})`` carry a
  thread id the LangGraph way,
- provides a ``MemorySaver`` marker and an ``interrupt(value)`` helper.

Usage — change your import line and (mostly) nothing else::

    # from langgraph.graph import StateGraph, START, END
    from flowgentra_ai.langgraph_compat import StateGraph, START, END, MemorySaver

    g = StateGraph(MyState)
    g.add_node("work", work)
    g.add_edge(START, "work")
    g.add_edge("work", END)
    app = g.compile(checkpointer=MemorySaver())
    app.invoke({"x": 1}, config={"configurable": {"thread_id": "t1"}})

See the "Migrating from LangGraph" guide for the full mapping and the handful
of behaviors that intentionally differ.
"""

from __future__ import annotations

from typing import Any, Optional

from flowgentra_ai.graph import StateGraph as _StateGraph
from flowgentra_ai.graph import END, Command
from flowgentra_ai import NodeInterrupt

# LangGraph's START sentinel. Flowgentra's native entry point is set with
# `set_entry_point`; here we accept edges `from START` and translate them.
START = "__start__"

__all__ = ["StateGraph", "CompiledGraph", "START", "END", "Command", "MemorySaver", "interrupt"]


class MemorySaver:
    """LangGraph-compatible in-memory checkpointer marker.

    Flowgentra's default (no checkpointer) is already in-memory per run; pass
    this to ``compile(checkpointer=...)`` for source compatibility. For
    persistence across processes use ``SqliteSaver`` / ``PostgresSaver`` in
    LangGraph — the Flowgentra equivalent is a path or URL string (see the
    migration guide) passed to ``compile(checkpointer=...)``.
    """

    kind = "memory"


def interrupt(value: Any):
    """LangGraph-style ``interrupt(value)`` — pause the run for human input.

    Equivalent to ``raise NodeInterrupt(value)`` inside a node. Resume with
    ``app.resume_command(thread_id, Command(update={...}))`` (Flowgentra) —
    Python nodes read the injected answer from state.
    """
    raise NodeInterrupt(value)


class _CompiledGraph:
    """Wraps Flowgentra's native compiled graph so ``invoke`` / ``stream``
    accept a LangGraph-style ``config={"configurable": {"thread_id": ...}}``."""

    def __init__(self, inner):
        self._inner = inner

    @staticmethod
    def _thread_id(config: Optional[dict]) -> Optional[str]:
        if not config:
            return None
        return (config.get("configurable") or {}).get("thread_id")

    def invoke(self, state: dict, config: Optional[dict] = None):
        thread_id = self._thread_id(config)
        if thread_id is not None:
            return self._inner.invoke_with_thread(thread_id, state)
        return self._inner.invoke(state)

    def stream(self, state: dict, config: Optional[dict] = None):
        # Flowgentra's stream() does not take a thread id; thread-scoped
        # streaming isn't wired through the compat layer. Falls back to a
        # plain stream.
        return self._inner.stream(state)

    def resume(self, thread_id: str):
        return self._inner.resume(thread_id)

    def resume_command(self, thread_id: str, command: "Command"):
        return self._inner.resume_command(thread_id, command)

    def __getattr__(self, name):
        # Delegate everything else (to_mermaid, get_state_history, serve_dev, ...).
        return getattr(self._inner, name)


class StateGraph(_StateGraph):
    """LangGraph-compatible ``StateGraph``.

    Adds: ``START``-aware ``add_edge`` (an edge ``from START`` becomes the entry
    point), and ``compile(checkpointer=...)`` the LangGraph way.
    """

    def add_edge(self, start, end):
        if start == START:
            self.set_entry_point(end)
            return
        return super().add_edge(start, end)

    def compile(self, checkpointer: Any = None, **_ignored):
        if checkpointer is not None:
            if isinstance(checkpointer, MemorySaver):
                pass  # default in-memory behavior; nothing to set
            elif isinstance(checkpointer, str):
                # A path ("./cp") or URL ("sqlite://...", "postgres://...").
                if checkpointer.startswith("postgres://"):
                    self.set_postgres_checkpointer(checkpointer)
                elif checkpointer.startswith(("sqlite://", "sqlite::")):
                    self.set_sqlite_checkpointer(checkpointer)
                elif checkpointer.startswith("mysql://"):
                    self.set_mysql_checkpointer(checkpointer)
                elif checkpointer.startswith("redis://"):
                    self.set_redis_checkpointer(checkpointer)
                else:
                    self.set_checkpointer(checkpointer)
            else:
                raise TypeError(
                    "compile(checkpointer=): pass a MemorySaver, or a path/URL string "
                    "('./cp', 'sqlite://x.db', 'postgres://...'). LangGraph saver "
                    "objects other than MemorySaver aren't supported — see the migration guide."
                )
        return _CompiledGraph(super().compile())


# Public alias mirroring LangGraph's name.
CompiledGraph = _CompiledGraph
