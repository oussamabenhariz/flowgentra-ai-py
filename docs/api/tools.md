# Tools API Reference

## ToolRegistry

Central registry for managing and executing tools.

```python
from flowgentra_ai.tools import ToolRegistry
```

### Constructor / Factory

```python
ToolRegistry()                 # empty registry
ToolRegistry.with_builtins()  # pre-loaded with CalculatorTool, SearchTool, WebRequestTool, FilesTool
```

### Methods

#### `call_tool(name, input)` → `Any`

Call a tool by name.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tool name |
| `input` | `dict` | Input arguments (must match the tool's schema) |

```python
result = registry.call_tool("calculator", {"operation": "add", "a": 17, "b": 8})
# 25
```

#### `validate_input(name, input)` → `None`

Validate input without calling the tool. Raises if input is invalid.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tool name |
| `input` | `dict` | Input to validate |

#### `list_names()` → `list[str]`

Get all registered tool names.

#### `__len__()` → `int`

Number of registered tools.

---

## Built-in Tools

```python
from flowgentra_ai.tools import CalculatorTool, SearchTool, WebRequestTool, FilesTool
```

| Class | Description | Key Operations |
|-------|-------------|----------------|
| `CalculatorTool` | Basic arithmetic | `add`, `subtract`, `multiply`, `divide` |
| `SearchTool` | Web search | Returns text results for a query |
| `WebRequestTool` | HTTP requests | `GET`, `POST` to any URL |
| `FilesTool` | File system | `read`, `write`, `list` directory |

---

## JsonSchema

Build JSON Schema objects programmatically for tool validation.

```python
from flowgentra_ai.tools import JsonSchema
```

### Factory Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `JsonSchema.object()` | `JsonSchema` | Object type |
| `JsonSchema.string()` | `JsonSchema` | String type |
| `JsonSchema.number()` | `JsonSchema` | Number (float) type |
| `JsonSchema.integer()` | `JsonSchema` | Integer type |
| `JsonSchema.boolean()` | `JsonSchema` | Boolean type |
| `JsonSchema.array()` | `JsonSchema` | Array type |

### Methods

| Method | Parameter | Description |
|--------|-----------|-------------|
| `with_description(desc)` | `str` | Set a description for this schema |
| `with_required(fields)` | `list[str]` | Mark fields as required (object schemas) |
| `validate(value)` | — | Validate a value. Raises if invalid. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `schema_type` | `str` | The type string (`"object"`, `"string"`, etc.) |
| `description` | `str \| None` | Schema description |

```python
schema = (
    JsonSchema.object()
    .with_description("Calculator input")
    .with_required(["operation", "a", "b"])
)
schema.validate({"operation": "add", "a": 1, "b": 2})  # passes
schema.validate({"operation": "add"})                   # raises — a and b are required
```

---

## ToolNode / create_tool_node

Integrates a `ToolRegistry` into a graph as a node. It automatically finds tool calls in the last message of state and executes them.

```python
from flowgentra_ai.tools import create_tool_node
```

### `create_tool_node(registry)` → callable

| Parameter | Type | Description |
|-----------|------|-------------|
| `registry` | `ToolRegistry` | Registry to call tools from |

Returns a node function that reads `state["messages"]`, executes any tool calls in the last message, and appends tool result messages.

```python
registry  = ToolRegistry.with_builtins()
tool_node = create_tool_node(registry)

builder.add_node("tools", tool_node)
```

### `store_tool_calls(state, results)` → `State`

Helper to store tool call results in state messages.

### `check_tools_condition(state)` → `str`

Router helper — returns `"tools"` if the last message has tool calls, or `"__end__"` otherwise.

```python
builder.add_conditional_edge("llm", check_tools_condition)
```

---

## ToolCallRequest / ToolCallResult

Types for the tool call request/response flow.

```python
from flowgentra_ai.llm import ToolCallRequest, ToolCallResult
```

| Type | Description |
|------|-------------|
| `ToolCallRequest` | A request to call a tool (from the LLM) |
| `ToolCallResult` | The result of calling a tool |
