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
"""

from flowgentra_ai._native import llm as _l

LLMConfig = _l.LLMConfig
Message = _l.Message
ToolCall = _l.ToolCall
ToolDefinition = _l.ToolDefinition
TokenUsage = _l.TokenUsage
LLM = _l.LLM
LLMClient = LLM  # alias used in SKILLS_PROPOSAL examples
create_llm = _l.py_create_llm

__all__ = [
    "LLMConfig",
    "Message",
    "ToolCall",
    "ToolDefinition",
    "TokenUsage",
    "LLM",
    "LLMClient",
    "create_llm",
]
