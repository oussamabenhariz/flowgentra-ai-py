# FlowgentraAI

**Build AI agent workflows with graphs** -- Python bindings for the [FlowgentraAI](https://github.com/oussamabenhariz/FlowgentraAI) Rust engine, powered by [PyO3](https://pyo3.rs).

[![PyPI](https://img.shields.io/pypi/v/flowgentra-ai)](https://pypi.org/project/flowgentra-ai/)
[![Python](https://img.shields.io/pypi/pyversions/flowgentra-ai)](https://pypi.org/project/flowgentra-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Installation

```bash
pip install flowgentra-ai
```

### From source (development)

```bash
cd flowgentra-ai-py
pip install maturin
maturin develop
```

## Quick Start

### Build a graph workflow (recommended)

The primary way to use FlowgentraAI in Python is the **StateGraph** API.
Define nodes as Python functions, wire them with edges, compile, and `invoke()`:

```python
from flowgentra_ai.graph import StateGraph, END
from flowgentra_ai import State

# 1. Define node functions (receive and return State)
def greet(state):
    name = state["name"]
    state["greeting"] = f"Hello, {name}!"
    return state

def uppercase(state):
    state["greeting"] = state["greeting"].upper()
    return state

# 2. Build the graph
builder = StateGraph()
builder.add_node("greet", greet)
builder.add_node("uppercase", uppercase)
builder.set_entry_point("greet")
builder.add_edge("greet", "uppercase")
builder.add_edge("uppercase", END)
graph = builder.compile()

# 3. Invoke with initial state
result = graph.invoke(State({"name": "World"}))
print(result.to_dict())
# {"name": "World", "greeting": "HELLO, WORLD!"}
```

### Config-driven agent

For YAML-configured agents with auto-discovered handlers:

```python
from flowgentra_ai.agent import Agent

agent = Agent.from_config_path("config.yaml")
agent.state["input"] = "Hello, world!"
result = agent.run()
print(result.to_dict())
```

## Core Concepts

### State

State is a thread-safe dict-like container that flows through your graph:

```python
from flowgentra_ai import State

state = State({"key": "value"})
state["name"] = "FlowgentraAI"
print(state["name"])        # "FlowgentraAI"
print("name" in state)      # True
print(state.to_dict())      # {"key": "value", "name": "FlowgentraAI"}
print(len(state))           # 2

# Serialization
json_str = state.to_json()
state = State.from_json('{"a": 1}')
state = State.from_dict({"a": 1})
```

### Conditional Routing

Route dynamically based on state:

```python
from flowgentra_ai.graph import StateGraph as StateGraphBuilder, END
from flowgentra_ai import State

def classify(state):
    text = state["input"]
    state["category"] = "greeting" if "hello" in text.lower() else "question"
    return state

def handle_greeting(state):
    state["output"] = "Hi there!"
    return state

def handle_question(state):
    state["output"] = "Let me think about that..."
    return state

def router(state):
    if state["category"] == "greeting":
        return "handle_greeting"
    return "handle_question"

builder = StateGraphBuilder()
builder.add_node("classify", classify)
builder.add_node("handle_greeting", handle_greeting)
builder.add_node("handle_question", handle_question)
builder.set_entry_point("classify")
builder.add_conditional_edge("classify", router)
builder.add_edge("handle_greeting", END)
builder.add_edge("handle_question", END)
graph = builder.compile()

result = graph.invoke(State({"input": "hello world"}))
print(result["output"])  # "Hi there!"
```

### LLM

Call LLM providers directly:

```python
from flowgentra_ai.llm import LLMConfig, LLM, Message

config = LLMConfig("openai", "gpt-4", api_key="sk-...")
client = LLM.from_config(config)

# Simple chat
response = client.chat([
    Message.system("You are a helpful assistant."),
    Message.user("What is Rust?"),
])
print(response.content)

# With token usage tracking
response, usage = client.chat_with_usage([Message.user("Hello!")])
if usage:
    print(f"Tokens: {usage.total_tokens}, Cost: ${usage.estimated_cost('gpt-4'):.4f}")

# With retry and caching
robust_client = client.with_retry(max_retries=3).cached(max_entries=100)
```

### Multi-Agent Supervisor

Orchestrate multiple agent graphs with 11 strategies:

```python
from flowgentra_ai.supervision import (
    Supervisor, SupervisorNodeConfig, OrchestrationStrategy,
    ParallelMergeStrategy,
)
from flowgentra_ai import State

# Simple mode (router function)
def router(state):
    if "research" in (state.get_string("task") or ""):
        return "researcher"
    return "FINISH"

sup = Supervisor(router)
sup.add_agent("researcher", research_graph)
result = sup.run(State({"task": "research AI trends"}))

# Strategy mode (parallel, sequential, retry_fallback, debate, etc.)
config = SupervisorNodeConfig(
    "coordinator",
    ["researcher", "writer"],
    OrchestrationStrategy.parallel(),
)
config.set_child_timeout_ms(30000)
config.set_merge_strategy(ParallelMergeStrategy.latest())

sup = Supervisor.from_config(config)
sup.add_agent("researcher", research_graph)
sup.add_agent("writer", writer_graph)
result = sup.run(State({"task": "write article"}))
```

Strategies: `sequential`, `parallel`, `autonomous`, `dynamic`, `round_robin`, `hierarchical`, `broadcast`, `map_reduce`, `conditional_routing`, `retry_fallback`, `debate`.

### Built-in Node Types

```python
# Retry with backoff
builder.add_retry_node("fetch", fetch_fn, max_retries=3, backoff_ms=1000)

# Timeout enforcement
builder.add_timeout_node("slow_op", slow_fn, timeout_ms=5000)

# LLM-integrated node
builder.add_llm_node("generate", client, prompt_key="query", output_key="response")

# LLM-driven planner (dynamic routing)
builder.add_planner_node("planner", client)

# Iterative evaluation/refinement
from flowgentra_ai.evaluation import EvaluationNodeConfig
config = EvaluationNodeConfig("refine", field_state="draft", min_confidence=0.8)
builder.add_evaluation_node(handler=refine_fn, config=config)
```

### Conversation Memory

Track multi-turn conversations:

```python
from flowgentra_ai.memory import ConversationMemory
from flowgentra_ai.llm import Message

mem = ConversationMemory(max_messages=100)
mem.add_message("thread-1", Message.user("Hello"))
mem.add_message("thread-1", Message.assistant("Hi! How can I help?"))

messages = mem.messages("thread-1")
print(len(messages))  # 2
```

Or use the high-level `MemoryAwareAgent`:

```python
from flowgentra_ai.agent import MemoryAwareAgent

agent = MemoryAwareAgent.from_config("config.yaml")
agent.set_thread_id("user_123")
answer = agent.run_turn("What is Rust?")
answer2 = agent.run_turn("What are its main features?")  # remembers context
```

### RAG (Retrieval-Augmented Generation)

```python
from flowgentra_ai.rag import (
    Document, Embeddings, InMemoryVectorStore,
    Retriever, RetrievalConfig,
    chunk_text, extract_text, estimate_tokens,
)

# Text utilities
chunks = chunk_text("long text...", chunk_size=500, overlap=50)
text = extract_text("document.pdf")
tokens = estimate_tokens("some text")

# Vector store + retrieval
emb = Embeddings.openai("sk-...", "text-embedding-3-small")
store = InMemoryVectorStore()

doc = Document("doc1", "Rust is a systems programming language.")
store.index(doc, emb.embed(doc.text))

config = RetrievalConfig.semantic(top_k=5, threshold=0.7)
retriever = Retriever(store, emb, config)
results = retriever.retrieve("What is Rust?")

for r in results:
    print(f"{r.id}: {r.text} (score: {r.score:.3f})")
```

### Prebuilt Agents

Use ready-made agent patterns:

```python
from flowgentra_ai.agent import ZeroShotReAct, ToolSpec
from flowgentra_ai.llm import LLMConfig

tool = ToolSpec("search", "Search the web")
tool.add_parameter("query", "string")
tool.set_required("query")

agent = ZeroShotReAct(
    name="my_agent",
    llm=LLMConfig("openai", "gpt-4"),
    system_prompt="You are a helpful assistant.",
    tools=[tool],
)

result = agent.execute_input("What is the weather today?")
print(result)
```

### Human-in-the-Loop

Interrupt graph execution for human review:

```python
from flowgentra_ai.graph import StateGraph as StateGraphBuilder, END
from flowgentra_ai import State

builder = StateGraphBuilder()
builder.add_node("draft", draft_fn)
builder.add_node("publish", publish_fn)
builder.set_entry_point("draft")
builder.add_edge("draft", "publish")
builder.add_edge("publish", END)
builder.interrupt_before("publish")  # pause before publishing
builder.set_checkpointer("./checkpoints")
graph = builder.compile()

# First run -- pauses before "publish"
result = graph.invoke_with_thread("thread-1", State({"topic": "AI"}))

# Human reviews, then resumes
result = graph.resume("thread-1")

# Or resume with modifications
result = graph.resume_with_state("thread-1", State({"approved": True}))
```

### Visualization

Export graphs as DOT or Mermaid diagrams:

```python
graph = builder.compile()
print(graph.to_mermaid())
print(graph.to_dot())
print(graph.node_names())
```

### Tool Registry

Use built-in tools or register custom ones:

```python
from flowgentra_ai.tools import ToolRegistry

registry = ToolRegistry.with_builtins()
print(registry.list_names())  # ["calculator", "search", ...]

result = registry.call_tool("calculator", {"operation": "add", "a": 2, "b": 3})
print(result)  # 5
```

### Model Pricing

```python
from flowgentra_ai.rag import model_pricing
from flowgentra_ai.llm import TokenUsage

pricing = model_pricing("gpt-4")  # (input_price, output_price) per million tokens
usage = TokenUsage(1000, 500)
cost = usage.estimated_cost("gpt-4")
```

## Supported LLM Providers

| Provider | Config name |
|----------|-------------|
| OpenAI | `"openai"` |
| Anthropic | `"anthropic"` |
| Mistral | `"mistral"` |
| Groq | `"groq"` |
| Ollama (local) | `"ollama"` |
| HuggingFace | `"huggingface"` |
| Azure OpenAI | `"azure"` |

## Full Documentation

For complete API reference and guides, visit the [documentation](https://oussamabenhariz.github.io/flowgentra-ai/).

## License

MIT
