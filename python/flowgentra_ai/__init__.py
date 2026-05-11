"""
FlowgentraAI - Build AI agent workflows with graphs

Python bindings for the FlowgentraAI Rust library via PyO3.

This package provides modular imports organized by domain:

    from flowgentra_ai.graph import StateGraph, END
    from flowgentra_ai.llm import LLMConfig, LLM, Message, create_llm
    from flowgentra_ai.tools import ToolRegistry, ToolNode
    from flowgentra_ai.memory import ConversationMemory, FileCheckpointer
    from flowgentra_ai.agent import Agent, AgentConfig, from_config_path
    from flowgentra_ai.rag import InMemoryVectorStore, Embeddings
    from flowgentra_ai.db import SqliteDatabase, PostgresDatabase, MongoDocumentStore
    from flowgentra_ai.evaluation import hit_rate, mrr, rag_evaluate
    from flowgentra_ai.observability import ExecutionTracer, visualize_graph

Quick Start (LangGraph-compatible API):

    from typing import TypedDict, List
    from flowgentra_ai.graph import StateGraph, END

    class State(TypedDict):
        messages: List[str]
        score: float

    def greet(state: dict) -> dict:
        return {"messages": state["messages"] + ["Hello!"]}

    def router(state: dict) -> str:
        return END if state["score"] > 0.5 else "greet"

    builder = StateGraph(State)
    builder.add_node("greet", greet)
    builder.set_entry_point("greet")
    builder.add_conditional_edge("greet", router)
    graph = builder.compile()

    result = graph.invoke({"messages": [], "score": 0.8})
    print(result["messages"])  # ["Hello!"]

Submodules:
    - graph: Graph construction and execution (StateGraph, CompiledGraph, END)
    - llm: LLM interfaces (LLMConfig, LLM, Message, create_llm)
    - tools: Tool registry and execution (ToolRegistry, ToolNode)
    - memory: Memory and persistence (ConversationMemory, FileCheckpointer, async checkpointers)
    - agent: Agent definitions (Agent, AgentConfig, MemoryAwareAgent)
    - rag: Retrieval-augmented generation (RAGConfig, InMemoryVectorStore, Embeddings)
    - db: SQL and NoSQL databases (SqliteDatabase, PostgresDatabase, MongoDocumentStore)
    - evaluation: Evaluation and metrics (hit_rate, mrr, rag_evaluate)
    - supervision: Multi-agent orchestration (Supervisor, PlannerNode)
    - nodes: Advanced graph nodes (RetryNode, TimeoutNode, SubgraphNodeConfig)
    - document_loaders: Document loading (load_document, load_directory, WebLoader)
    - rerankers: Search result reranking (RRFReranker, CrossEncoderReranker)
    - observability: Tracing and visualization (ExecutionTracer, visualize_graph)
    - mcp: Model Context Protocol client (MCPConfig, MCPClient, create_client)
    - types: Common types and utilities (Condition, chunk_text, estimate_tokens)
    - skills: Skill system (SkillRegistry, SkillAgent, Skill)
"""

# Minimal re-exports for convenience — import from submodules for the full API
from flowgentra_ai.graph import StateGraph, CompiledGraph, END
from flowgentra_ai.llm import LLMConfig, LLM, Message, create_llm
from flowgentra_ai.agent import Agent, AgentConfig
from flowgentra_ai.types import ResponseFormat
from flowgentra_ai._native import state as _st
State = _st.State
from flowgentra_ai.state_schema import typed_state_graph, validate_state, coerce_state, StateSchema

from flowgentra_ai.exceptions import (
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

__version__ = "0.1.5"

__all__ = [
    # Core graph
    "StateGraph",
    "CompiledGraph",
    "END",
    # LLM
    "LLMConfig",
    "LLM",
    "Message",
    "create_llm",
    # Agent
    "Agent",
    "AgentConfig",
    # Config
    "ResponseFormat",
    # Typed state schema helpers
    "typed_state_graph",
    "validate_state",
    "coerce_state",
    "StateSchema",
    # Exceptions
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
