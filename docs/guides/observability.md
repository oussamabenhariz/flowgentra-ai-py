# Observability

Understanding what your agent did — which nodes ran, in what order, how long they took — is essential for debugging and production monitoring.

---

## Structured logging

Initialize at the start of your program to get structured logs from the Rust engine.

=== "Python"

    ```python
    from flowgentra_ai.observability import init_tracing

    init_tracing()          # default: "info" level
    init_tracing("debug")   # verbose — logs every node entry/exit and state change
    init_tracing("warn")    # quiet — only warnings and errors
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::observability::init_tracing;

    init_tracing("info");   // or "debug", "warn", "error"
    ```

Call this once before building any graphs.

---

## Execution tracing

Record every event during a graph execution for post-hoc analysis.

=== "Python"

    ```python
    from flowgentra_ai.observability import ExecutionTracer

    tracer = ExecutionTracer()

    # Record events manually (usually you'd pass the tracer to the graph)
    tracer.trace_node_start("fetch")
    tracer.trace_node_end("fetch", duration_ms=120, success=True)
    tracer.trace_edge_traversal("fetch", "process", condition_met=True)
    tracer.trace_state_update("data", "fetched_value")
    tracer.trace_custom("cache_hit", details="key: query_abc")

    # Export all events as JSON
    events_json = tracer.get_events_json()
    print(events_json)

    # Clear for next run
    tracer.clear()
    ```

### ExecutionTrace

A snapshot of a completed execution.

=== "Python"

    ```python
    from flowgentra_ai.observability import ExecutionTrace

    trace = ExecutionTrace(agent_name="my-agent")

    path     = trace.execution_path()      # ["fetch", "process", "respond"]
    duration = trace.total_duration_ms()   # total runtime in ms

    # Serialize and restore
    json_str = trace.to_json()
    restored = ExecutionTrace.from_json(json_str)
    ```

---

## Graph visualization

Export your graph structure as a diagram to understand the flow visually.

=== "Python"

    ```python
    graph = builder.compile()

    # Mermaid — renders in GitHub, GitLab, Notion, Obsidian
    mermaid = graph.to_mermaid()
    print(mermaid)
    # graph LR
    #   fetch --> process
    #   process --> respond
    #   respond --> __end__

    # Graphviz DOT — render with `dot -Tpng graph.dot -o graph.png`
    dot = graph.to_dot()

    # JSON structure — for programmatic inspection
    structure = graph.to_json()
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::observability::{graph_to_mermaid, graph_to_dot};

    let mermaid = graph_to_mermaid(&graph);
    let dot     = graph_to_dot(&graph);
    ```

### Visualization utilities

=== "Python"

    ```python
    from flowgentra_ai.observability import (
        visualize_graph,
        graph_to_dot,
        graph_to_mermaid,
        VisualizationConfig,
    )

    config = VisualizationConfig()
    visualize_graph(graph, config)
    ```

---

## OpenTelemetry export (Rust)

For production systems, export traces to any OTLP-compatible backend (Jaeger, Tempo, Datadog, etc.).

```rust
use flowgentra_ai::observability::OtelExporter;

OtelExporter::new()
    .endpoint("http://localhost:4317")   // OTLP gRPC endpoint
    .service_name("my-agent")
    .init()?;
```

With OpenTelemetry enabled, every node execution, state update, and LLM call is exported as a span.

---

## Node timing

The engine tracks timing for each node automatically. Access it via the execution trace.

=== "Rust"

    ```rust
    let result = graph.invoke_with_trace(state).await?;
    let (final_state, trace) = result;

    for event in trace.events() {
        match event {
            ExecutionEvent::NodeEnd { node, duration_ms, .. } => {
                println!("{node}: {duration_ms}ms");
            }
            _ => {}
        }
    }
    ```

---

## Health monitoring patterns

A common production pattern is to wrap your graph invocation with error tracking:

=== "Python"

    ```python
    import time

    def invoke_with_monitoring(graph, state, thread_id=None):
        start = time.time()
        try:
            if thread_id:
                result = graph.invoke_with_thread(thread_id, state)
            else:
                result = graph.invoke(state)
            duration_ms = (time.time() - start) * 1000
            print(f"Graph completed in {duration_ms:.0f}ms")
            return result
        except Exception as e:
            print(f"Graph failed: {e}")
            raise
    ```

---

## Debugging tips

**Use `init_tracing("debug")`** during development. It logs every state transition and shows exactly where things went wrong.

**Visualize before you debug.** Render `graph.to_mermaid()` first — sometimes the issue is a wrong edge, which is instantly obvious in a diagram.

**Inspect intermediate state.** Add `print(state.to_dict())` at the start of a node to see exactly what data it received.

**Replay with fixed state.** If a run fails, the checkpoint lets you restart from just before the failure — you don't have to re-run everything.
