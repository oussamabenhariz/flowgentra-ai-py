# Nodes

A node is the fundamental unit of work in a Flowgentra graph. It's just a function.

---

## The node contract

Every node must:

1. Accept a single `State` argument (Python) or `DynState` / your typed state (Rust)
2. Return a `State` (usually the same object, modified)

=== "Python"

    ```python
    def my_node(state):
        # Read from state
        input_data = state["input"]

        # Do work
        result = process(input_data)

        # Write to state
        state["output"] = result

        return state   # always return state
    ```

=== "Rust"

    ```rust
    async fn my_node(mut state: DynState) -> Result<DynState> {
        let input = state.get_string("input").unwrap_or_default();
        let result = process(&input).await?;
        state.set("output", result);
        Ok(state)
    }
    ```

---

## Node types

### Regular nodes

The default. A plain function added with `add_node`.

```python
builder.add_node("process", process_fn)
```

### Retry nodes

Automatically retries with exponential backoff when the function raises an exception.

```python
builder.add_retry_node("fetch", fetch_fn, max_retries=3, backoff_ms=1000)
```

Use this for: API calls, database queries, network requests — anything that can fail transiently.

### Timeout nodes

Terminates the function if it exceeds a time limit.

```python
builder.add_timeout_node("slow_op", slow_fn, timeout_ms=5000, on_timeout="skip")
```

- `on_timeout="error"` — raises an exception (default)
- `on_timeout="skip"` — returns the original state unchanged

### LLM nodes

A pre-wired node that reads a prompt from state, calls an LLM, and writes the response.

```python
builder.add_llm_node(
    "generate",
    client,
    prompt_key="user_query",
    output_key="llm_response",
    system_prompt="You are a helpful assistant.",
)
```

### Planner nodes

Uses an LLM to decide the next node dynamically at runtime.

```python
builder.add_planner_node("planner", client)
```

The planner reads `state["_reachable_nodes"]` (a list of node names) and writes `state["_next_node"]` (the chosen next node).

### Evaluation nodes

Wraps a node in an iterative refinement loop — runs the node, evaluates the output, and re-runs until quality meets a threshold.

```python
config = EvaluationNodeConfig(
    name="refine",
    field_state="draft",
    min_confidence=0.8,
    max_retries=3,
    rubric="Is the output clear and accurate?",
)
builder.add_evaluation_node(handler=draft_fn, config=config)
```

### Subgraph nodes

An entire graph embedded as a single node. From the outer graph's perspective, it's just a node.

```python
inner_graph = inner_builder.compile()
outer_builder.add_subgraph("inner", inner_graph)
```

---

## What happens when a node fails?

By default, an exception in a node propagates up and terminates the graph run. To handle errors gracefully:

1. Use `add_retry_node` for transient failures
2. Catch the exception inside the node and store it in state, then route to an error handler

```python
def safe_node(state):
    try:
        state["result"] = risky_operation()
    except ValueError as e:
        state["error"] = str(e)
    return state

def router(state):
    return "error_handler" if state.get("error") else "next_node"
```

---

## Node naming

- Use **verb-noun** style: `fetch_data`, `classify_intent`, `generate_response`
- Use **underscores**, not hyphens (hyphens cause issues in some routing logic)
- Make names unique within a graph
- The special string `"__end__"` is reserved — don't use it as a node name

---

## Testing a node

Nodes are just functions, so testing is straightforward:

```python
from flowgentra_ai import State

def test_classify_node():
    state = State({"input": "What is the capital of France?"})
    result = classify_node(state)
    assert result["is_question"] == True
    assert "error" not in result
```
