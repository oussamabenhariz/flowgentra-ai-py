"""Language model interfaces and utilities.

This module provides LLM abstractions and configuration for using various
language models with your workflows.

Examples:
    Create and use an LLM:

        from flowgentra_ai.llm import LLM, Message

        client = LLM(provider="openai", model="gpt-4o")  # key from OPENAI_API_KEY

        response = client.chat([Message.user("Hello")])
        print(response.content)

    Stream tokens as they arrive:

        for chunk in client.chat_stream([Message.user("Tell me a story")]):
            print(chunk, end="", flush=True)

    Offline unit tests with no network/credentials:

        from flowgentra_ai.llm import MockLLM

        mock = MockLLM()
        mock.when_contains("weather", "It is sunny")
        mock.otherwise("I don't know")
        llm = mock.as_llm()  # -> LLM, drop-in anywhere a real LLM is expected

    A single prompt -> LLM step, without building a graph. This `Chain` is
    fixed at exactly two stages (prompt, then LLM) — for more stages (parsers,
    plain functions) or `|`-operator composition, see `flowgentra_ai.chain`,
    whose (differently-scoped) `Chain` supports an arbitrary number of stages:

        from flowgentra_ai.llm import Chain, PromptTemplate

        prompt = PromptTemplate("Translate '{text}' to French.")
        chain = Chain(prompt, client)
        reply = chain.invoke({"text": "Hello"})
"""

from flowgentra_ai._native import llm as _l, text as _t

LLMConfig = _l.LLMConfig
Message = _l.Message
ToolCall = _l.ToolCall
ToolDefinition = _l.ToolDefinition
TokenUsage = _l.TokenUsage
LLM = _l.LLM
LLMClient = LLM  # alias used in SKILLS_PROPOSAL examples
MockLLM = _l.MockLLM
Chain = _l.Chain
create_llm = _l.py_create_llm

# Prompt templates and output parsers — previously only reachable via the
# internal `_native.text` module, never re-exported at package level.
PromptTemplate = _t.PromptTemplate
JsonOutputParser = _t.JsonOutputParser
ListOutputParser = _t.ListOutputParser


def _schema_for(target):
    """Extract a JSON Schema (dict) from a dict, a Pydantic model class, or a
    TypedDict. Returns ``(json_schema, validator)`` where ``validator(data)``
    turns the parsed dict into the caller's preferred type (a Pydantic instance,
    or the dict unchanged)."""
    # A plain JSON Schema dict — used as-is.
    if isinstance(target, dict):
        return target, (lambda data: data)

    # A Pydantic model class (v2 preferred, v1 fallback).
    if hasattr(target, "model_json_schema"):  # pydantic v2
        return target.model_json_schema(), (lambda data: target.model_validate(data))
    if hasattr(target, "schema") and hasattr(target, "parse_obj"):  # pydantic v1
        return target.schema(), (lambda data: target.parse_obj(data))

    # A TypedDict (or any class with __annotations__): build a minimal schema.
    annotations = getattr(target, "__annotations__", None)
    if annotations:
        _PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean",
                       list: "array", dict: "object"}
        props = {name: {"type": _PY_TO_JSON.get(typ, "string")} for name, typ in annotations.items()}
        schema = {"type": "object", "properties": props,
                  "required": list(getattr(target, "__required_keys__", annotations.keys()))}
        return schema, (lambda data: data)

    raise TypeError(
        f"with_structured_output: {target!r} must be a JSON Schema dict, a Pydantic "
        "model class, or a TypedDict."
    )


class _StructuredLLM:
    """Wraps an LLM so `.invoke(messages)` returns output conforming to a schema.

    Returned by :meth:`LLM.with_structured_output`. Sends the schema to the model,
    parses the JSON reply, and (for a Pydantic model) validates it into an instance.
    """

    __slots__ = ("_llm", "_schema", "_validate", "_target")

    def __init__(self, llm, target):
        self._llm = llm
        self._target = target
        self._schema, self._validate = _schema_for(target)

    def invoke(self, messages):
        """Run the LLM and return the schema-conforming result (a dict, or a
        Pydantic instance when a model class was given)."""
        if isinstance(messages, str):
            messages = [Message.user(messages)]
        data = self._llm.chat_structured_with_schema(messages, self._schema)
        return self._validate(data)

    def __repr__(self):
        name = getattr(self._target, "__name__", "schema")
        return f"StructuredLLM({name})"


def _with_structured_output(self, target):
    """Return a wrapper whose `.invoke(messages)` yields output matching `target`.

    `target` may be a JSON Schema dict, a Pydantic model class (the result is a
    validated model instance), or a TypedDict. Mirrors LangChain's
    `with_structured_output`.

    Example:
        from pydantic import BaseModel

        class Sentiment(BaseModel):
            label: str
            score: float

        structured = client.with_structured_output(Sentiment)
        result = structured.invoke("Classify: 'I love this!'")
        print(result.label, result.score)   # -> a Sentiment instance
    """
    return _StructuredLLM(self, target)


# Attach to the native LLM class (monkeypatch, same pattern as Agent.arun /
# CompiledGraph.ainvoke) so it reads as `client.with_structured_output(Model)`.
LLM.with_structured_output = _with_structured_output

__all__ = [
    "LLMConfig",
    "Message",
    "ToolCall",
    "ToolDefinition",
    "TokenUsage",
    "LLM",
    "LLMClient",
    "MockLLM",
    "Chain",
    "PromptTemplate",
    "JsonOutputParser",
    "ListOutputParser",
    "create_llm",
]
