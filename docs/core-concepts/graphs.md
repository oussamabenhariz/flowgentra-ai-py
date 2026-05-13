# Graph Execution

A graph is the core execution unit in Flowgentra. It defines which nodes run, in which order, and under what conditions.

---

## Anatomy of a graph

```
          entry point
               │
               ▼
          [Node A]
         /        \
   (cond)          (cond)
       │                │
       ▼                ▼
  [Node B]          [Node C]
       │                │
       └────────┬───────┘
                ▼
             __end__
```

Every graph has:

- An **entry point** — the first node to run
- **Nodes** — functions that transform state
- **Edges** — connections between nodes (fixed or conditional)
- A **terminal** — `END` / `"__end__"` signals the graph to stop

---

## Building a graph

=== "Python"

    ```python
    from flowgentra_ai.graph import StateGraph, END
    from flowgentra_ai import State

    builder = StateGraph(dict)

    # Add nodes
    builder.add_node("fetch",   fetch_fn)
    builder.add_node("process", process_fn)
    builder.add_node("respond", respond_fn)

    # Set entry point
    builder.set_entry_point("fetch")

    # Fixed edges
    builder.add_edge("fetch", "process")
    builder.add_edge("process", "respond")
    builder.add_edge("respond", END)

    # Compile — after this the graph is immutable
    graph = builder.compile()

    # Run
    result = graph.invoke(State({"input": "hello"}))
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::{StateGraph, DynState};

    let graph = StateGraph::builder()
        .add_node("fetch",   fetch_fn)
        .add_node("process", process_fn)
        .add_node("respond", respond_fn)
        .entry("fetch")
        .edge("fetch",   "process")
        .edge("process", "respond")
        .edge("respond", "__end__")
        .build();

    let mut state = DynState::new();
    state.set("input", "hello");
    let result = graph.invoke(state).await?;
    ```

---

## Conditional routing

When you need different nodes to run based on state, use a **conditional edge**. Your router function receives the current state and returns the name of the next node.

=== "Python"

    ```python
    def router(state):
        score = state.get("confidence") or 0
        if score > 0.8:
            return "accept"
        elif score > 0.5:
            return "review"
        return "reject"

    builder.add_conditional_edge("evaluate", router)
    # Each possible return value must be a node name
    ```

=== "Rust"

    ```rust
    .conditional_edge("evaluate", |state: &DynState| {
        let score = state.get_float("confidence").unwrap_or(0.0);
        if score > 0.8 { "accept" }
        else if score > 0.5 { "review" }
        else { "reject" }
    })
    ```

!!! tip
    Your router's return values must be node names (or `"__end__"`). If a router returns an unknown name, the engine will panic at compile/build time — not silently at runtime.

---

## Invoking a graph

=== "Python"

    ```python
    # Basic — no persistence
    result = graph.invoke(State({"input": "data"}))

    # With thread ID — enables checkpointing
    result = graph.invoke_with_thread("thread-1", State({"input": "data"}))
    ```

=== "Rust"

    ```rust
    // Basic
    let result = graph.invoke(state).await?;

    // With thread ID
    let result = graph.invoke_with_thread("thread-1", state).await?;
    ```

---

## Special nodes

The builder has convenience methods for common patterns. Use these instead of implementing the logic manually.

### Retry node

Automatically retries with exponential backoff when the node function fails.

=== "Python"

    ```python
    builder.add_retry_node(
        "fetch_api",
        fetch_fn,
        max_retries=3,         # attempt count (default: 3)
        backoff_ms=1000,       # first wait (default: 1000ms)
        backoff_multiplier=2.0, # doubles each retry (default: 2.0)
        max_backoff_ms=30000,   # cap (default: 30s)
    )
    ```

=== "Rust"

    ```rust
    .add_retry_node("fetch_api", fetch_fn, RetryConfig {
        max_retries: 3,
        backoff_ms: 1000,
        backoff_multiplier: 2.0,
        max_backoff_ms: 30_000,
    })
    ```

### Timeout node

Kills the function if it takes too long.

=== "Python"

    ```python
    builder.add_timeout_node(
        "slow_op",
        slow_fn,
        timeout_ms=5000,
        on_timeout="error",   # "error" (default) or "skip"
    )
    # on_timeout="skip" returns the original state unchanged
    ```

=== "Rust"

    ```rust
    .add_timeout_node("slow_op", slow_fn, TimeoutConfig {
        timeout_ms: 5000,
        on_timeout: OnTimeout::Error,
    })
    ```

### LLM node

Reads a prompt from state, calls an LLM, writes the response back.

=== "Python"

    ```python
    builder.add_llm_node(
        "generate",
        client,
        prompt_key="user_query",       # state key to read prompt from
        output_key="llm_response",     # state key to write response to
        system_prompt="You are a helpful assistant.",
    )
    ```

### Planner node

LLM-driven dynamic routing — the LLM decides the next node at runtime.

=== "Python"

    ```python
    builder.add_planner_node("planner", client)
    # Reads "_reachable_nodes" from state, sets "_next_node"
    ```

### Evaluation node

Iteratively refines a node's output until a quality threshold is met.

=== "Python"

    ```python
    from flowgentra_ai.evaluation import EvaluationNodeConfig

    config = EvaluationNodeConfig(
        name="refine",
        field_state="draft",       # state key holding the output
        min_confidence=0.8,        # stop refining when score >= 0.8
        max_retries=3,
        rubric="Is the output clear and accurate?",
    )

    def scorer(output, attempt):
        # Return (score 0–1, feedback string)
        words = len(str(output).split())
        if words > 50:
            return (0.9, "Good detail level")
        return (0.4, "Too brief — expand more")

    builder.add_evaluation_node(
        handler=draft_fn,
        config=config,
        scorer=scorer,            # optional custom scorer
    )
    ```

---

## Subgraphs

Embed a compiled graph inside another graph. The inner graph runs as a single node from the outer graph's perspective.

=== "Python"

    ```python
    # Build inner graph
    inner = StateGraph(dict)
    inner.add_node("step", step_fn)
    inner.set_entry_point("step")
    inner.add_edge("step", END)
    inner_graph = inner.compile()

    # Use it in outer graph
    outer = StateGraph(dict)
    outer.add_node("prepare", prepare_fn)
    outer.add_subgraph("inner", inner_graph)   # ← subgraph as a node
    outer.set_entry_point("prepare")
    outer.add_edge("prepare", "inner")
    outer.add_edge("inner", END)
    outer_graph = outer.compile()
    ```

=== "Rust"

    ```rust
    let inner_graph = StateGraph::builder()
        .add_node("step", step_fn)
        .entry("step")
        .edge("step", "__end__")
        .build();

    let outer_graph = StateGraph::builder()
        .add_node("prepare", prepare_fn)
        .add_subgraph("inner", inner_graph)
        .entry("prepare")
        .edge("prepare", "inner")
        .edge("inner", "__end__")
        .build();
    ```

---

## Parallel execution

Run multiple nodes at the same time and merge their results.

=== "Python"

    ```python
    from flowgentra_ai.nodes import ParallelNodeConfig, MergeStrategy

    config = ParallelNodeConfig(
        branches=["fetch_a", "fetch_b", "fetch_c"],
        merge_strategy=MergeStrategy.deep_merge(),
    )
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::node::ParallelExecutor;

    let results = ParallelExecutor::new()
        .add_branch("fetch_a", fetch_a_fn)
        .add_branch("fetch_b", fetch_b_fn)
        .merge(MergeStrategy::DeepMerge)
        .execute(state)
        .await?;
    ```

---

## Checkpointing

Persist graph state to disk between node executions. Required for human-in-the-loop and crash recovery.

=== "Python"

    ```python
    builder.set_checkpointer("./checkpoints")   # directory path
    graph = builder.compile()

    result = graph.invoke_with_thread("session-abc", initial_state)
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::memory::FileCheckpointer;

    let graph = StateGraph::builder()
        .with_checkpointer(FileCheckpointer::new("./checkpoints"))
        // ... nodes and edges
        .build();
    ```

---

## Max steps

Prevent runaway loops by capping the number of node executions.

=== "Python"

    ```python
    builder.set_max_steps(50)   # default: 1000
    ```

=== "Rust"

    ```rust
    .max_steps(50)
    ```

---

## Exporting / visualizing

=== "Python"

    ```python
    graph = builder.compile()

    mermaid = graph.to_mermaid()   # renders in GitHub, GitLab, Notion
    dot     = graph.to_dot()       # Graphviz format
    struct  = graph.to_json()      # JSON structure

    # Metadata
    graph.node_names()    # ["fetch", "process", "respond"]
    graph.entry_point()   # "fetch"
    ```

---

## Message graph

For chat-based workflows, `MessageGraphBuilder` pre-configures message accumulation. Nodes receive and return `list[Message]` instead of `State`.

=== "Python"

    ```python
    from flowgentra_ai.graph import MessageGraphBuilder
    from flowgentra_ai.llm import LLM, Message

    client = LLM(provider="openai", model="gpt-4o", api_key="sk-...")

    def chat_node(messages):
        response = client.chat(messages)
        return messages + [response]

    builder = MessageGraphBuilder()
    builder.add_node("chat", chat_node)
    builder.set_entry_point("chat")
    builder.add_edge("chat", "__end__")
    graph = builder.compile()

    result = graph.invoke([Message.user("Hello!")])
    for msg in result:
        print(f"{msg.role}: {msg.content}")
    ```
