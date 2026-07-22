"""LCEL-style pipe composition: ``prompt | llm | parser``.

This is sugar over Flowgentra's existing prompt/LLM/parser primitives, not a
new execution engine — :class:`~flowgentra_ai.graph.StateGraph` remains the
tool for anything with branching, loops, retries, or persistence. ``Chain``
exists because building a whole graph for the extremely common "format a
prompt, call the LLM, parse the result" pipeline is pure boilerplate.

Any of the following can be a stage: :class:`~flowgentra_ai.llm.PromptTemplate`,
:class:`~flowgentra_ai.llm.LLM` (including a :class:`~flowgentra_ai.llm.MockLLM`
via ``.as_llm()``), :class:`~flowgentra_ai.llm.JsonOutputParser` /
:class:`~flowgentra_ai.llm.ListOutputParser`, or any plain callable ``f(x) -> y``.

Example — pipe operator::

    from flowgentra_ai.llm import LLM, PromptTemplate, JsonOutputParser

    prompt = PromptTemplate("List 3 colors related to {topic}, as JSON.")
    llm = LLM(provider="anthropic", model="claude-opus-4-6")
    pipeline = prompt | llm | JsonOutputParser()

    result = pipeline.invoke({"topic": "the ocean"})  # -> parsed JSON

Example — explicit, no operator overloading::

    from flowgentra_ai.chain import Chain

    pipeline = Chain.sequence([prompt, llm, JsonOutputParser()])
    result = pipeline.invoke({"topic": "the ocean"})
"""

from __future__ import annotations

from typing import Any, Callable, List, Union

from flowgentra_ai.llm import Message

__all__ = ["Runnable", "Chain"]


def _has_method(obj: Any, name: str) -> bool:
    return callable(getattr(obj, name, None))


def _invoke_stage(stage: Any, value: Any) -> Any:
    """Normalize any supported stage type into a single `invoke(value) -> value` call."""
    if isinstance(stage, Runnable):
        return stage.invoke(value)

    # PromptTemplate-like: .format(dict) -> str
    if _has_method(stage, "format") and not _has_method(stage, "chat"):
        variables = value if isinstance(value, dict) else {}
        return stage.format(variables)

    # LLM-like: .chat(list[Message]) -> Message
    if _has_method(stage, "chat"):
        if isinstance(value, str):
            messages = [Message.user(value)]
        elif isinstance(value, list):
            messages = value
        elif isinstance(value, Message):
            messages = [value]
        else:
            raise TypeError(
                f"Chain: LLM stage {stage!r} received {type(value).__name__}; "
                "expected str, Message, or list[Message] from the previous stage."
            )
        return stage.chat(messages)

    # Output-parser-like: .parse(str) -> Any
    if _has_method(stage, "parse"):
        text = value.content if isinstance(value, Message) else value
        return stage.parse(text)

    # Plain callable — the escape hatch for anything else.
    if callable(stage):
        return stage(value)

    raise TypeError(
        f"Chain: don't know how to invoke stage {stage!r} of type {type(stage).__name__}. "
        "Supported: PromptTemplate, LLM, JsonOutputParser/ListOutputParser, or a plain callable."
    )


Stage = Union["Runnable", Any]


class Runnable:
    """Wraps a single stage (prompt, LLM, parser, or plain callable) so it can
    be composed with `|`. You normally don't construct this directly — `|`
    wraps bare stages automatically."""

    __slots__ = ("_stage",)

    def __init__(self, stage: Any) -> None:
        self._stage = stage

    def invoke(self, value: Any) -> Any:
        return _invoke_stage(self._stage, value)

    def __or__(self, other: Stage) -> "Chain":
        return Chain([self, _as_runnable(other)])

    def __repr__(self) -> str:
        return f"Runnable({self._stage!r})"


def _as_runnable(stage: Stage) -> Runnable:
    return stage if isinstance(stage, Runnable) else Runnable(stage)


class Chain(Runnable):
    """A sequence of stages run in order — each stage's output feeds the next
    stage's input. Build with `|` (see module docs) or `Chain.sequence(...)`.
    """

    __slots__ = ("_stages",)

    def __init__(self, stages: List[Runnable]) -> None:
        self._stages = stages

    @classmethod
    def sequence(cls, stages: List[Stage]) -> "Chain":
        """Explicit alternative to `|` — no operator overloading.

        Example:
            pipeline = Chain.sequence([prompt, llm, parser])
            result = pipeline.invoke({"topic": "rust"})
        """
        return cls([_as_runnable(s) for s in stages])

    def invoke(self, value: Any) -> Any:
        for stage in self._stages:
            value = stage.invoke(value)
        return value

    def __or__(self, other: Stage) -> "Chain":
        return Chain(self._stages + [_as_runnable(other)])

    def __repr__(self) -> str:
        return f"Chain({self._stages!r})"


def _patch_pipe_operator(cls: type) -> None:
    """Add `__or__` to a native (pyo3) class so `stage | other` starts a Chain
    without requiring `Runnable(stage) | other`. Idempotent — safe to call
    more than once (e.g. if this module is imported multiple times)."""
    if "__or__" in cls.__dict__:
        return

    def __or__(self: Any, other: Stage) -> "Chain":
        return Chain([_as_runnable(self), _as_runnable(other)])

    cls.__or__ = __or__


def _patch_known_stage_types() -> None:
    from flowgentra_ai.llm import LLM, JsonOutputParser, ListOutputParser, PromptTemplate

    for cls in (PromptTemplate, LLM, JsonOutputParser, ListOutputParser):
        _patch_pipe_operator(cls)


_patch_known_stage_types()
