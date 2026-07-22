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
