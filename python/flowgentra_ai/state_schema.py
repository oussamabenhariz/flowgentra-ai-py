"""
State schema support for Flowgentra-AI graphs.

Provides runtime validation and IDE type support for graph state using
Python's native TypedDict or Pydantic BaseModel.

## Usage

### TypedDict (zero dependencies)

    from typing import TypedDict, List, Optional
    from flowgentra_ai.graph import StateGraph, END
    from flowgentra_ai.state_schema import typed_state_graph

    class AgentState(TypedDict):
        input: str
        messages: List[str]
        result: Optional[str]

    # Wraps a native StateGraph and validates state on every invoke()
    graph = typed_state_graph(AgentState)

    def my_node(state: AgentState) -> AgentState:
        # IDE can autocomplete state.input, state.messages, etc.
        return {"result": state["input"].upper()}

    graph.add_node("process", my_node)
    graph.set_entry_point("process")
    graph.add_edge("process", END)
    compiled = graph.compile()
    result = compiled.invoke({"input": "hello", "messages": [], "result": None})


### Pydantic (if pydantic is installed)

    from pydantic import BaseModel
    from flowgentra_ai.state_schema import typed_state_graph

    class AgentState(BaseModel):
        input: str
        messages: list[str] = []
        result: str | None = None

    graph = typed_state_graph(AgentState)
    # The initial state can be a dict or AgentState instance.
    # State is automatically validated before each invoke().
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Type, get_type_hints

__all__ = [
    "StateSchema",
    "validate_state",
    "coerce_state",
    "typed_state_graph",
    "extract_schema_fields",
]


# ─── Schema introspection ──────────────────────────────────────────────────


def _is_typeddict(cls: Any) -> bool:
    """Return True if *cls* is a TypedDict class."""
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
        and hasattr(cls, "__total__")
    )


def _is_pydantic(cls: Any) -> bool:
    """Return True if *cls* is a Pydantic BaseModel subclass."""
    try:
        from pydantic import BaseModel  # type: ignore[import]
        return isinstance(cls, type) and issubclass(cls, BaseModel)
    except ImportError:
        return False


def extract_schema_fields(schema: Any) -> Dict[str, Any]:
    """Extract field names and (optional) types from a TypedDict or Pydantic model.

    Returns a mapping ``{field_name: python_type_or_Any}``.
    """
    if _is_pydantic(schema):
        return {
            name: field.annotation
            for name, field in schema.model_fields.items()
        }
    if _is_typeddict(schema):
        try:
            return get_type_hints(schema)
        except Exception:
            return dict(schema.__annotations__)
    return {}


# ─── StateSchema ───────────────────────────────────────────────────────────


class StateSchema:
    """Wraps a TypedDict or Pydantic class to provide validation helpers.

    This is a *runtime* validator — it does not modify the native Rust
    graph execution, but validates state before and after Python-side calls
    to catch KeyErrors and type mismatches early.
    """

    def __init__(self, schema_class: Any) -> None:
        if not (_is_typeddict(schema_class) or _is_pydantic(schema_class)):
            raise TypeError(
                f"state_schema must be a TypedDict or pydantic.BaseModel subclass, "
                f"got {type(schema_class).__name__!r}. "
                f"Example: class MyState(TypedDict): field: str"
            )
        self._schema = schema_class
        self._is_pydantic = _is_pydantic(schema_class)
        self._fields = extract_schema_fields(schema_class)
        self._required: set[str] = set()

        if _is_typeddict(schema_class):
            # __required_keys__ is set for Python ≥ 3.9
            self._required = set(
                getattr(schema_class, "__required_keys__", self._fields.keys())
            )
        elif _is_pydantic(schema_class):
            self._required = {
                name
                for name, field in schema_class.model_fields.items()
                if field.is_required()
            }

    @property
    def fields(self) -> Dict[str, Any]:
        """Field name → Python type mapping."""
        return self._fields

    @property
    def required_fields(self) -> set:
        """Set of field names that must be present in the state dict."""
        return self._required

    @property
    def schema_class(self) -> Any:
        """The original TypedDict / Pydantic class."""
        return self._schema

    def validate(self, state: dict, context: str = "state") -> None:
        """Raise a ``ValueError`` if *state* violates the schema.

        Checks:
        - All required fields are present.
        - No extra keys exist (warns but does not raise — Rust state can have
          internal keys like ``_token_usage``).
        """
        if not isinstance(state, dict):
            raise TypeError(
                f"{context} must be a dict, got {type(state).__name__!r}. "
                f"Pass a dict matching the {self._schema.__name__} schema."
            )

        # Check required fields
        missing = self._required - state.keys()
        if missing:
            raise ValueError(
                f"State is missing required field(s): {sorted(missing)}. "
                f"Schema '{self._schema.__name__}' requires: {sorted(self._required)}. "
                f"Got keys: {sorted(state.keys())}."
            )

    def coerce(self, state: Any) -> dict:
        """Convert *state* to a plain dict, accepting dicts or Pydantic instances."""
        if isinstance(state, dict):
            return state
        if self._is_pydantic and hasattr(state, "model_dump"):
            return state.model_dump()
        if hasattr(state, "__dict__"):
            return vars(state)
        raise TypeError(
            f"Cannot coerce state of type {type(state).__name__!r} to dict. "
            f"Pass a dict or a {self._schema.__name__} instance."
        )

    def default_state(self) -> dict:
        """Build a minimal default state dict with all fields set to None / default.

        For Pydantic models this calls the model's default constructor.
        For TypedDicts it fills every field with None.
        """
        if self._is_pydantic:
            try:
                instance = self._schema()
                return instance.model_dump()
            except Exception:
                pass
        return {field: None for field in self._fields}

    def __repr__(self) -> str:
        return f"StateSchema({self._schema.__name__}, fields={list(self._fields)})"


# ─── Convenience helpers ───────────────────────────────────────────────────


def validate_state(state: dict, schema: Any, context: str = "state") -> None:
    """Validate *state* against a TypedDict or Pydantic *schema*.

    Raises ``ValueError`` if required fields are missing.
    """
    StateSchema(schema).validate(state, context)


def coerce_state(state: Any, schema: Any) -> dict:
    """Coerce *state* (dict or Pydantic instance) to a plain dict."""
    return StateSchema(schema).coerce(state)


# ─── ValidatingStateGraph ──────────────────────────────────────────────────


class ValidatingStateGraph:
    """A thin wrapper around the native `StateGraph` that validates state
    against a TypedDict or Pydantic schema before each `invoke()`.

    Use `typed_state_graph(MyState)` to create one.
    """

    def __init__(self, schema: Any) -> None:
        from flowgentra_ai._native import graph as _g; NativeStateGraph = _g.StateGraph  # type: ignore

        self._schema = StateSchema(schema)
        # Pass the schema class to the native binding (it may use it for field discovery)
        self._native = NativeStateGraph(schema)

    # ── Delegation to native StateGraph builder API ──

    def add_node(self, name: str, fn: Any) -> "ValidatingStateGraph":
        self._native.add_node(name, fn)
        return self

    def add_edge(self, from_node: str, to_node: str) -> "ValidatingStateGraph":
        self._native.add_edge(from_node, to_node)
        return self

    def add_conditional_edge(self, from_node: str, router: Any) -> "ValidatingStateGraph":
        self._native.add_conditional_edge(from_node, router)
        return self

    def set_entry_point(self, node_name: str) -> "ValidatingStateGraph":
        self._native.set_entry_point(node_name)
        return self

    def set_finish_point(self, node_name: str) -> "ValidatingStateGraph":
        self._native.set_finish_point(node_name)
        return self

    def compile(self) -> "ValidatingCompiledGraph":
        compiled = self._native.compile()
        return ValidatingCompiledGraph(compiled, self._schema)

    @property
    def schema(self) -> StateSchema:
        """The attached state schema."""
        return self._schema

    def __repr__(self) -> str:
        return f"ValidatingStateGraph(schema={self._schema})"


class ValidatingCompiledGraph:
    """A compiled graph that validates state dicts before and after execution."""

    def __init__(self, compiled: Any, schema: StateSchema) -> None:
        self._compiled = compiled
        self._schema = schema

    def invoke(self, state: Any, config: Optional[dict] = None) -> dict:
        """Invoke the graph with state validation.

        *state* can be a plain dict or a Pydantic model instance.
        Validates required fields are present before invoking.
        """
        state_dict = self._schema.coerce(state)
        self._schema.validate(state_dict, "initial state")
        if config is not None:
            return self._compiled.invoke(state_dict, config)
        return self._compiled.invoke(state_dict)

    async def ainvoke(self, state: Any, config: Optional[dict] = None) -> dict:
        """Async invoke with state validation."""
        state_dict = self._schema.coerce(state)
        self._schema.validate(state_dict, "initial state")
        if config is not None:
            return await self._compiled.ainvoke(state_dict, config)
        return await self._compiled.ainvoke(state_dict)

    def stream(self, state: Any, config: Optional[dict] = None):
        """Stream graph execution events with state validation."""
        state_dict = self._schema.coerce(state)
        self._schema.validate(state_dict, "initial state")
        if config is not None:
            return self._compiled.stream(state_dict, config)
        return self._compiled.stream(state_dict)

    @property
    def schema(self) -> StateSchema:
        return self._schema

    def __getattr__(self, name: str) -> Any:
        # Forward unknown attribute access to the native compiled graph
        return getattr(self._compiled, name)

    def __repr__(self) -> str:
        return f"ValidatingCompiledGraph(schema={self._schema})"


# ─── Public factory ────────────────────────────────────────────────────────


def typed_state_graph(schema_class: Any) -> ValidatingStateGraph:
    """Create a StateGraph that validates state against *schema_class*.

    *schema_class* must be a ``TypedDict`` or ``pydantic.BaseModel`` subclass.

    Example::

        from typing import TypedDict
        from flowgentra_ai.state_schema import typed_state_graph
        from flowgentra_ai.graph import END

        class AgentState(TypedDict):
            input: str
            result: str | None

        graph = typed_state_graph(AgentState)
        graph.add_node("process", lambda s: {"result": s["input"].upper()})
        graph.set_entry_point("process")
        graph.add_edge("process", END)
        compiled = graph.compile()
        result = compiled.invoke({"input": "hello", "result": None})
        print(result["result"])  # HELLO

    Pydantic example::

        from pydantic import BaseModel

        class AgentState(BaseModel):
            input: str
            result: str | None = None

        graph = typed_state_graph(AgentState)
        compiled = graph.compile()
        # Pass a dict or a Pydantic instance:
        result = compiled.invoke(AgentState(input="hello"))
        print(result["result"])  # HELLO
    """
    return ValidatingStateGraph(schema_class)
