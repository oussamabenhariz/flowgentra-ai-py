# Observability API Reference

## init_tracing

Initialize structured logging. Call once at program start.

```python
from flowgentra_ai.observability import init_tracing
```

### Signature

```python
init_tracing(level: str = "info") -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | `str` | `"info"` | Log level: `"debug"`, `"info"`, `"warn"`, `"error"` |

```python
init_tracing()          # info level (default)
init_tracing("debug")  # verbose — logs every state change
init_tracing("warn")   # quiet — only warnings and errors
```

---

## ExecutionTrace

A recorded trace of a complete agent execution.

```python
from flowgentra_ai.observability import ExecutionTrace
```

### Constructor

```python
ExecutionTrace(agent_name: str | None = None)
```

### Class Methods

#### `ExecutionTrace.from_json(json_str)` → `ExecutionTrace`

Deserialize from a JSON string.

| Parameter | Type | Description |
|-----------|------|-------------|
| `json_str` | `str` | Serialized trace |

### Methods

#### `execution_path()` → `list[str]`

Returns the ordered list of node names that executed.

```python
path = trace.execution_path()
# ["fetch", "process", "respond"]
```

#### `total_duration_ms()` → `int | None`

Total execution time in milliseconds.

#### `to_json()` → `str`

Serialize the trace to JSON.

```python
json_str = trace.to_json()
# Store it, send it, or restore it later
restored = ExecutionTrace.from_json(json_str)
```

---

## ExecutionTracer

Records execution events during a graph run.

```python
from flowgentra_ai.observability import ExecutionTracer
```

### Constructor

```python
ExecutionTracer()
```

### Methods

| Method | Parameters | Description |
|--------|-----------|-------------|
| `trace_node_start(node_id)` | `str` | Record that a node started |
| `trace_node_end(node_id, duration_ms, success)` | `str, int, bool` | Record node completion |
| `trace_edge_traversal(from_node, to_node, condition_met)` | `str, str, bool` | Record edge traversal |
| `trace_state_update(key, value)` | `str, Any` | Record a state change |
| `trace_custom(event_name, details=None)` | `str, str\|None` | Record a custom event |
| `get_events_json()` | → `str` | Export all events as JSON |
| `clear()` | — | Clear all recorded events |

```python
tracer = ExecutionTracer()

tracer.trace_node_start("fetch_data")
tracer.trace_state_update("data", "fetched_value")
tracer.trace_node_end("fetch_data", duration_ms=85, success=True)
tracer.trace_edge_traversal("fetch_data", "process", condition_met=True)

events = tracer.get_events_json()
print(events)
```

---

## Visualization Functions

```python
from flowgentra_ai.observability import (
    visualize_graph,
    graph_to_dot,
    graph_to_mermaid,
    VisualizationConfig,
)
```

### `graph_to_dot(graph)` → `str`

Export graph as Graphviz DOT format. Render with `dot -Tpng graph.dot -o graph.png`.

### `graph_to_mermaid(graph)` → `str`

Export as Mermaid diagram. Renders in GitHub, GitLab, Notion, Obsidian.

### `visualize_graph(graph, config)` → `None`

Visualize with custom configuration.

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `StateGraph` | Compiled graph |
| `config` | `VisualizationConfig` | Visualization options |

### VisualizationConfig

```python
config = VisualizationConfig()
```

Configuration options for graph visualization (layout, node styles, etc.).

---

Note: Graphs also have these methods directly:

```python
graph.to_mermaid()   # Mermaid string
graph.to_dot()       # DOT string
graph.to_json()      # JSON structure
```
