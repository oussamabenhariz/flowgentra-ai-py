# What is Flowgentra?

Flowgentra is a framework for building **AI agent workflows**. It gives you the building blocks to create anything from a simple LLM-powered function to a complex multi-agent system with memory, tools, evaluation, and observability.

The core idea is simple: your agent logic is a **graph**. Nodes are functions. Edges connect them. State flows through.

---

## The mental model

Think of your agent as a flowchart:

```
          ┌─────────────┐
Input ───►│  Classify   │
          └──────┬──────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌─────────────┐   ┌─────────────┐
│  Use Tools  │   │  Answer LLM │
└──────┬──────┘   └──────┬──────┘
       │                 │
       └────────┬────────┘
                │
                ▼
           ┌─────────┐
           │  Output │
           └─────────┘
```

Each box is a **node** — a function that takes state in and returns state out. The arrows are **edges** — connections that tell the engine what to run next. Some edges are fixed; others are **conditional** (the router function decides at runtime).

---

## Core primitives

### State

State is the data container that travels through your graph. Every node reads from it and writes to it.

```python
# Python
state["user_query"] = "What is Rust?"
state["answer"]     # read it later
```

```rust
// Rust
state.set("user_query", "What is Rust?");
state.get_string("answer");
```

State is designed to be safe across threads and serializable to JSON — so you can checkpoint it, pass it between agents, and inspect it in traces.

### Nodes

A node is just a function. It receives state, does something (call an LLM, run a query, format text), and returns updated state.

```python
# Python
def my_node(state):
    state["result"] = process(state["input"])
    return state
```

```rust
// Rust
async fn my_node(mut state: DynState) -> Result<DynState> {
    let input = state.get_string("input").unwrap_or_default();
    state.set("result", process(&input));
    Ok(state)
}
```

That's really it. Nodes have no special interface — they're plain functions.

### Graphs

A graph wires nodes together and tells the engine the execution order.

```python
# Python
builder = StateGraph(dict)
builder.add_node("classify", classify)
builder.add_node("respond", respond)
builder.set_entry_point("classify")
builder.add_edge("classify", "respond")
builder.add_edge("respond", END)
graph = builder.compile()
result = graph.invoke(State({"input": "Hello"}))
```

The engine runs `classify`, then `respond`, then stops. If `classify` had a conditional edge, the engine would call your router function to decide which node to run next.

---

## What the engine gives you

On top of this simple model, Flowgentra's engine adds:

| Feature | What it does |
|---------|--------------|
| **Checkpointing** | Persist state to disk between nodes — resume after crashes or for human review |
| **Retries** | Automatically retry failed nodes with exponential backoff |
| **Timeouts** | Kill slow nodes after a deadline |
| **Tracing** | Record every node execution, state change, and timing |
| **Visualization** | Export your graph as Mermaid or Graphviz diagrams |
| **Parallel execution** | Run multiple branches simultaneously and merge results |
| **Evaluation** | Auto-grade output and re-run nodes until quality thresholds are met |

---

## The two patterns

There are two ways to use Flowgentra. Both use the same engine.

### 1. Code-first (direct API)

Build the graph programmatically. Best for complex logic with non-trivial routing.

```python
builder = StateGraph(dict)
builder.add_node("step_a", fn_a)
builder.add_node("step_b", fn_b)
builder.set_entry_point("step_a")
builder.add_conditional_edge("step_a", router)
builder.add_edge("step_b", END)
graph = builder.compile()
```

### 2. Config-driven (YAML + handlers)

Define the graph in YAML; write handlers in code. Best for production deployments where non-engineers need to adjust the flow.

```yaml
# agent.yaml
name: my-agent
graph:
  entry: fetch
  edges:
    - from: fetch
      to: process
    - from: process
      to: __end__
```

```python
# Python
from flowgentra_ai.agent import Agent
agent = Agent.from_config_path("agent.yaml")
result = agent.run()
```

```rust
// Rust
let agent = Agent::from_config_path("agent.yaml").await?;
let result = agent.run().await?;
```

Handlers are auto-discovered from your codebase by name — you mark them with `@register_handler` (Python) or `#[register_handler]` (Rust).

---

## How Rust and Python relate

The Python library is a thin PyO3 wrapper around the Rust engine. When you call `graph.invoke()` in Python, you're calling the same Rust execution loop that you'd use directly in Rust.

| Rust | Python |
|------|--------|
| `StateGraph::<S>::builder()` | `StateGraph(MyState)` |
| `DynState` | `State` |
| `#[derive(State)]` | TypedDict / dict |
| `#[register_handler]` | `@register_handler` |
| `Agent::from_config_path()` | `Agent.from_config_path()` |

The key difference: Rust uses generics for compile-time type safety; Python uses dynamic dispatch for flexibility. Both are full-featured.

---

## Next steps

- [Installation](../getting-started/installation.md) — set up Rust or Python
- [Quick Start](../getting-started/quickstart.md) — build your first graph in 5 minutes
- [State Management](state.md) — understand how state works
- [Graph Execution](graphs.md) — understand how graphs execute
