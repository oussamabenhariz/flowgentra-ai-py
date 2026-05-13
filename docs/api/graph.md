# Graph API Reference

## StateGraph (builder)

Builds a state graph with Python callables as nodes. After calling `compile()`, you get a runnable `StateGraph`.

```python
from flowgentra_ai.graph import StateGraph, END
```

### Constructor

```python
StateGraph(schema)
```

Creates an empty builder. Pass a TypedDict class or `dict` for untyped use:

```python
# Untyped (accepts any dict keys)
builder = StateGraph(dict)

# Typed (preferred for larger graphs)
from typing import TypedDict

class MyState(TypedDict):
    input: str
    output: str

builder = StateGraph(MyState)
```

---

### `add_node(name, func)` → `StateGraph`

Add a node to the graph.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Node name (must be unique) |
| `func` | `Callable[[State], State]` | Node function |

The function must accept a `State` and return a `State`.

```python
def my_node(state):
    state["result"] = process(state["input"])
    return state

builder.add_node("process", my_node)
```

---

### `set_entry_point(name)` → `StateGraph`

Set the first node to execute when the graph is invoked.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Name of the entry node |

---

### `add_edge(from_node, to_node)` → `StateGraph`

Add a fixed edge. After `from_node` runs, always go to `to_node`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `from_node` | `str` | Source node name |
| `to_node` | `str \| END` | Destination node name, or `END` to terminate |

```python
builder.add_edge("process", "respond")
builder.add_edge("respond", END)       # END = graph terminates here
```

---

### `add_conditional_edge(from_node, router)` → `StateGraph`

Add a dynamic edge. After `from_node` runs, call `router(state)` to decide the next node.

| Parameter | Type | Description |
|-----------|------|-------------|
| `from_node` | `str` | Source node name |
| `router` | `Callable[[State], str]` | Returns the name of the next node (or `"__end__"`) |

```python
def router(state):
    if state["score"] > 0.8:
        return "accept"
    return "reject"

builder.add_conditional_edge("evaluate", router)
```

---

### `add_retry_node(name, func, ...)` → `StateGraph`

Add a node that automatically retries with exponential backoff on failure.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Node name |
| `func` | callable | required | Node function |
| `max_retries` | `int` | `3` | Maximum number of attempts |
| `backoff_ms` | `int` | `1000` | Wait time before first retry (ms) |
| `backoff_multiplier` | `float` | `2.0` | Multiply wait time by this after each retry |
| `max_backoff_ms` | `int` | `30000` | Maximum wait time cap (ms) |

```python
builder.add_retry_node(
    "fetch_api",
    fetch_fn,
    max_retries=5,
    backoff_ms=500,
    backoff_multiplier=2.0,
    max_backoff_ms=30000,
)
```

---

### `add_timeout_node(name, func, timeout_ms, ...)` → `StateGraph`

Add a node that enforces a time limit.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Node name |
| `func` | callable | required | Node function |
| `timeout_ms` | `int` | required | Time limit in milliseconds |
| `on_timeout` | `str` | `"error"` | `"error"` raises, `"skip"` returns original state |

```python
builder.add_timeout_node("slow_op", slow_fn, timeout_ms=5000, on_timeout="skip")
```

---

### `add_llm_node(name, llm, ...)` → `StateGraph`

Add a node that reads a prompt from state, calls an LLM, and writes the response back.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Node name |
| `llm` | `LLM` | required | The LLM to use |
| `prompt_key` | `str` | `"prompt"` | State key to read the prompt from |
| `output_key` | `str` | `"llm_response"` | State key to write the response to |
| `system_prompt` | `str \| None` | `None` | Optional system prompt |

```python
builder.add_llm_node(
    "generate",
    client,
    prompt_key="user_question",
    output_key="llm_answer",
    system_prompt="You are a concise assistant.",
)
```

---

### `add_planner_node(name, llm, ...)` → `StateGraph`

Add an LLM-driven planner that dynamically selects the next node at runtime.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Node name |
| `llm` | `LLM` | required | LLM to use for planning decisions |
| `prompt` | `str \| None` | `None` | Custom planner prompt |

The planner reads `_reachable_nodes` from state and writes `_next_node`.

```python
builder.add_planner_node("planner", client)
```

---

### `add_evaluation_node(handler, config, scorer)` → `StateGraph`

Add an iterative evaluation node that refines output until it meets a quality threshold.

| Parameter | Type | Description |
|-----------|------|-------------|
| `handler` | callable | The node function to evaluate and retry |
| `config` | `EvaluationNodeConfig` | Configuration for the evaluation loop |
| `scorer` | `Callable \| None` | Optional custom scorer: `(output, attempt) -> (float, str)` |

```python
from flowgentra_ai.evaluation import EvaluationNodeConfig

config = EvaluationNodeConfig(
    name="refine",
    field_state="draft",
    min_confidence=0.8,
    max_retries=3,
    rubric="Is the output clear and well-structured?",
)

builder.add_evaluation_node(
    handler=draft_fn,
    config=config,
    scorer=my_scorer,   # optional
)
```

---

### `add_subgraph(name, graph)` → `StateGraph`

Embed a compiled `StateGraph` as a single node.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Node name for the subgraph |
| `graph` | `StateGraph` | A compiled graph |

```python
inner_graph = inner_builder.compile()
outer_builder.add_subgraph("inner", inner_graph)
```

---

### `interrupt_before(name)` → `StateGraph`

Pause execution before this node runs. Requires checkpointing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Node name to interrupt before |

---

### `interrupt_after(name)` → `StateGraph`

Pause execution after this node runs. Requires checkpointing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Node name to interrupt after |

---

### `set_checkpointer(base_dir)` → `StateGraph`

Enable file-based checkpointing. State is saved after each node.

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_dir` | `str` | Directory path for checkpoint files |

---

### `set_max_steps(n)` → `StateGraph`

Cap the number of node executions per run. Prevents infinite loops.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | `1000` | Maximum steps |

---

### `compile()` → `StateGraph` (compiled)

Build and lock the graph. Returns a runnable `StateGraph`.

---

## StateGraph (compiled)

A compiled, runnable graph. Returned by `StateGraphBuilder.compile()`.

---

### `invoke(state)` → `State`

Run the graph with an initial state.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `State` | Initial state |

```python
result = graph.invoke(State({"input": "hello"}))
```

---

### `invoke_with_thread(thread_id, state)` → `State`

Run with checkpointing. State is saved after each node.

| Parameter | Type | Description |
|-----------|------|-------------|
| `thread_id` | `str` | Unique session identifier |
| `state` | `State` | Initial state |

---

### `resume(thread_id)` → `State`

Resume a previously interrupted execution.

| Parameter | Type | Description |
|-----------|------|-------------|
| `thread_id` | `str` | Thread to resume |

---

### `resume_with_state(thread_id, updates)` → `State`

Resume with additional state modifications. Useful for injecting human edits.

| Parameter | Type | Description |
|-----------|------|-------------|
| `thread_id` | `str` | Thread to resume |
| `updates` | `State` | State changes to merge before resuming |

---

### `node_names()` → `list[str]`

Returns names of all nodes in the graph.

### `entry_point()` → `str`

Returns the name of the entry node.

### `to_mermaid()` → `str`

Export as a Mermaid diagram string.

### `to_dot()` → `str`

Export as Graphviz DOT format.

### `to_json()` → `dict`

Export graph structure as a JSON-compatible dict.

---

## END

Sentinel value for terminating edges. Import and use as `to_node` in `add_edge`.

```python
from flowgentra_ai.graph import END

builder.add_edge("last_node", END)
```

---

## MessageGraphBuilder

Convenience builder for chat workflows. Nodes receive and return `list[Message]` instead of `State`.

```python
from flowgentra_ai.graph import MessageGraphBuilder
```

### Constructor

```python
MessageGraphBuilder()
```

### Methods

| Method | Description |
|--------|-------------|
| `add_node(name, func)` | `func(messages: list[Message]) -> list[Message]` |
| `add_edge(from_node, to_node)` | Fixed edge |
| `set_entry_point(name)` | Set entry point |
| `compile()` | Build into a `MessageGraph` |

```python
def chat(messages):
    response = client.chat(messages)
    return messages + [response]

builder = MessageGraphBuilder()
builder.add_node("chat", chat)
builder.set_entry_point("chat")
builder.add_edge("chat", "__end__")
graph = builder.compile()

result = graph.invoke([Message.user("Hello!")])
```

---

## MessageGraph

Compiled message-centric graph.

### `invoke(messages)` → `list[Message]`

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `list[Message]` | Initial message list |
