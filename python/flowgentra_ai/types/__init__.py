"""Common types and utilities shared across modules.

This module provides shared types and utility functions that do not belong
to a single domain.  Domain-specific types live in their own submodule:

    - MCP types → ``flowgentra_ai.mcp``
    - DB types  → ``flowgentra_ai.db``
    - LLM types → ``flowgentra_ai.llm``
    - Agent loading → ``flowgentra_ai.agent``

Examples:
    Conditional routing predicates::

        from flowgentra_ai.types import Condition, ConditionBuilder, ComparisonOp

        cond = ConditionBuilder.field("score").gt(0.5)

    LLM response format::

        from flowgentra_ai.types import ResponseFormat

    Text processing utilities::

        from flowgentra_ai.types import chunk_text, extract_text, estimate_tokens

        chunks = chunk_text(long_text, chunk_size=1000)
        tokens = estimate_tokens(text, model="gpt-4")
"""

from flowgentra_ai._native import routing as _ro, llm as _l, text as _tx, rag as _r

# ── Routing / conditional logic ───────────────────────────────────────────────
Condition = _ro.Condition
ConditionBuilder = _ro.ConditionBuilder
ComparisonOp = _ro.ComparisonOp

# ── LLM response types ────────────────────────────────────────────────────────
ResponseFormat = _l.ResponseFormat

# ── LLM utility functions ─────────────────────────────────────────────────────
estimate_tokens = _l.py_estimate_tokens
model_pricing = _l.py_model_pricing

# ── Text processing utilities ─────────────────────────────────────────────────
chunk_text = _tx.py_chunk_text
extract_text = _tx.py_extract_text
chunk_text_by_tokens = _r.py_chunk_text_by_tokens
extract_and_chunk = _r.py_extract_and_chunk

__all__ = [
    # Routing / conditions
    "Condition",
    "ConditionBuilder",
    "ComparisonOp",
    # Response types
    "ResponseFormat",
    # LLM utilities
    "estimate_tokens",
    "model_pricing",
    # Text processing
    "chunk_text",
    "extract_text",
    "chunk_text_by_tokens",
    "extract_and_chunk",
]
