"""Custom exception hierarchy for FlowgentraAI.

All exceptions inherit from :class:`FlowgentraAIError`, which is a subclass
of the built-in :class:`Exception`.  Catch the base class to handle any
library error, or catch a specific subclass for fine-grained control.

Hierarchy::

    Exception
    └── FlowgentraAIError
        ├── ConfigurationError     invalid config, YAML parse errors
        ├── ValidationError        schema / value validation failures
        ├── GraphError             graph structure problems
        │   ├── NodeNotFoundError  referenced node does not exist
        │   └── CycleError         infinite loop / recursion limit reached
        ├── LLMError               LLM API failures
        ├── MCPError               MCP server / transport failures
        ├── ToolExecutionError     tool execution failures
        ├── AgentExecutionError    agent / node execution failures
        ├── WorkflowTimeoutError   execution timeout
        ├── SerializationError     JSON / serialization failures
        ├── CheckpointError        state persistence failures
        └── InternalError          unexpected internal failures

Note: :class:`OSError` is still raised directly for low-level file / network
I/O errors (``IoError`` on the Rust side) to preserve standard Python semantics.

Examples::

    from flowgentra_ai.exceptions import FlowgentraAIError, LLMError, WorkflowTimeoutError

    try:
        result = agent.run(prompt)
    except WorkflowTimeoutError:
        # retry with a shorter prompt or increase the timeout
        ...
    except LLMError as e:
        # check API key / rate limits
        print(f"LLM failed: {e}")
    except FlowgentraAIError as e:
        # catch-all for any other library error
        print(f"Unexpected error: {e}")
"""

from flowgentra_ai._native import (  # type: ignore[import]
    FlowgentraAIError,
    ConfigurationError,
    ValidationError,
    GraphError,
    NodeNotFoundError,
    CycleError,
    LLMError,
    MCPError,
    ToolExecutionError,
    AgentExecutionError,
    WorkflowTimeoutError,
    SerializationError,
    CheckpointError,
    InternalError,
)

__all__ = [
    "FlowgentraAIError",
    "ConfigurationError",
    "ValidationError",
    "GraphError",
    "NodeNotFoundError",
    "CycleError",
    "LLMError",
    "MCPError",
    "ToolExecutionError",
    "AgentExecutionError",
    "WorkflowTimeoutError",
    "SerializationError",
    "CheckpointError",
    "InternalError",
]
