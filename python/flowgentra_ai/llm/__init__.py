"""Language model interfaces and utilities.

This module provides LLM abstractions and configuration for using various
language models with your workflows.

Examples:
    Create and use an LLM:

        from flowgentra_ai.llm import LLM, LLMConfig, Message

        config = LLMConfig(model="gpt-4", temperature=0.7)
        client = LLM(config)

        messages = [Message(role="user", content="Hello")]
        response = client.call(messages)
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
