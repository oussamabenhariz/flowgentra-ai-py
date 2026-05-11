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

from flowgentra_ai._native import graph as _g, nodes as _n

# Re-export compiled types directly.
CompiledGraph = _g.CompiledGraph
END = _g.END
MessageGraph = _n.MessageGraph
MessageGraphBuilder = _n.MessageGraphBuilder

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
    "MessageGraph",
    "MessageGraphBuilder",
    "END",
]
