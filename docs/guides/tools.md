# Tools

Tools give your agent the ability to act — search the web, run calculations, call APIs, read files. Flowgentra supports **local tools** you define yourself and **MCP tools** from external tool servers.

---

## ToolRegistry: Complete API

```python
registry = ToolRegistry() -> ToolRegistry
registry = ToolRegistry.with_builtins() -> ToolRegistry  # Pre-populated
```

### Creating a Registry

**Python:**
```python
from flowgentra_ai.tools import ToolRegistry

# Empty registry
registry = ToolRegistry()

# With all built-in tools pre-registered
registry = ToolRegistry.with_builtins()  # Includes: calculator, search, web_request, files
```

### Registering Tools

```python
registry.register(
    name: str,
    func: Callable,
    description: str,
    parameters: Dict[str, Any]
) -> ToolRegistry
```

**Parameters:**
- `name` (str): Unique tool identifier
- `func` (Callable): Function to execute  
- `description` (str): Human-readable description
- `parameters` (Dict): JSON Schema describing input format

**Returns:** Self (for chaining)

**Example:**
```python
def multiply(a: float, b: float) -> float:
    return a * b

registry.register(
    name="multiply",
    func=multiply,
    description="Multiply two numbers",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"}
        },
        "required": ["a", "b"]
    }
)
```

### Calling Tools

```python
result = registry.call_tool(
    name: str,
    arguments: Dict[str, Any]
) -> Any
```

**Parameters:**
- `name` (str): Tool name
- `arguments` (Dict): Tool parameters

**Returns:** Tool result

**Example:**
```python
result = registry.call_tool("calculator", {"operation": "add", "a": 5, "b": 3})
# result: 8
```

### Validating Input

```python
registry.validate_input(name: str, arguments: Dict[str, Any]) -> bool
```

Checks if arguments match the tool's schema without executing.

**Example:**
```python
is_valid = registry.validate_input("calculator", {"operation": "add", "a": 1, "b": 2})
# True

try:
    registry.validate_input("calculator", {})  # Missing required parameters
except ValidationError as e:
    print(f"Invalid: {e}")
```

### Listing Tools

```python
names: List[str] = registry.list_names()
schemas: Dict[str, Dict] = registry.get_schemas()
```

**Example:**
```python
print(registry.list_names())  # ["calculator", "search", "web_request", "files"]

for name, schema in registry.get_schemas().items():
    print(f"{name}: {schema['description']}")
```

### Async Tool Execution

```python
result = await registry.call_tool_async(
    name: str,
    arguments: Dict[str, Any]
) -> Any
```

**Example:**
```python
# Support both sync and async tools
result = await registry.call_tool_async("search", {"query": "Python async"})
```

---

## Built-in Tools: Complete Reference

### 1. Calculator Tool

```python
registry.call_tool("calculator", {
    "operation": str,  # "add", "subtract", "multiply", "divide", "power", "sqrt", "log"
    "a": float,
    "b": float  # Optional for sqrt/log
}) -> float
```

**Operations:**

| Operation | Parameters | Returns | Example |
|-----------|-----------|---------|---------|
| `add` | a, b | a + b | 5 + 3 = 8 |
| `subtract` | a, b | a - b | 5 - 3 = 2 |
| `multiply` | a, b | a * b | 5 * 3 = 15 |
| `divide` | a, b | a / b | 9 / 3 = 3 |
| `power` | a, b | a ^ b | 2 ^ 8 = 256 |
| `sqrt` | a | √a | √16 = 4 |
| `log` | a, b | logₐ(b) | log₂(8) = 3 |

**Example:**
```python
# Basic arithmetic
registry.call_tool("calculator", {"operation": "multiply", "a": 17, "b": 8})
# → 136

# Powers
registry.call_tool("calculator", {"operation": "power", "a": 2, "b": 10})
# → 1024

# Square root
registry.call_tool("calculator", {"operation": "sqrt", "a": 144})
# → 12.0
```

### 2. Search Tool

```python
registry.call_tool("search", {
    "query": str,              # Search terms
    "max_results": int = 10,   # Number of results
    "language": str = "en"     # Language code
}) -> List[SearchResult]
```

**Returns:** List of search results with title, url, snippet

**Example:**
```python
results = registry.call_tool("search", {
    "query": "Python machine learning",
    "max_results": 5
})

# Results structure:
# [
#   {
#     "title": "scikit-learn: Machine Learning in Python",
#     "url": "https://scikit-learn.org",
#     "snippet": "Simple and efficient tools for data analysis...",
#     "position": 1
#   },
#   ...
# ]
```

### 3. Web Request Tool

```python
registry.call_tool("web_request", {
    "url": str,                              # Full URL
    "method": str = "GET",                   # HTTP method
    "headers": Dict[str, str] = None,        # Custom headers
    "body": str = None,                      # Request body (for POST/PUT)
    "timeout_seconds": int = 30              # Request timeout
}) -> Dict[str, Any]
```

**Returns:** Response with status_code, headers, body, duration_ms

**Response Structure:**
```python
{
    "status_code": 200,
    "headers": {"content-type": "application/json"},
    "body": "...",
    "duration_ms": 245,
    "url": "https://..."
}
```

**Example:**
```python
# GET request
response = registry.call_tool("web_request", {
    "url": "https://api.github.com/users/octocat",
    "method": "GET",
    "headers": {"Accept": "application/vnd.github.v3+json"}
})

# POST request with JSON
response = registry.call_tool("web_request", {
    "url": "https://api.example.com/submit",
    "method": "POST",
    "headers": {"Content-Type": "application/json"},
    "body": '{"name": "Alice", "email": "alice@example.com"}'
})

status = response["status_code"]
body = response["body"]
```

### 4. Files Tool

```python
registry.call_tool("files", {
    "operation": str,           # "read", "write", "list", "delete"
    "path": str,                # File or directory path
    "content": str = None,      # For write operations
    "encoding": str = "utf-8"   # File encoding
}) -> Union[str, List[str], bool]
```

**Operations:**

| Operation | Parameters | Returns | Purpose |
|-----------|-----------|---------|---------|
| `read` | path, encoding | str | Read file contents |
| `write` | path, content, encoding | bool | Write to file |
| `list` | path | List[str] | List directory contents |
| `delete` | path | bool | Delete file/directory |

**Example:**
```python
# Read a file
content = registry.call_tool("files", {
    "operation": "read",
    "path": "/path/to/config.json"
})

# Write a file
registry.call_tool("files", {
    "operation": "write",
    "path": "/path/to/output.txt",
    "content": "Hello, World!",
    "encoding": "utf-8"
})

# List directory
contents = registry.call_tool("files", {
    "operation": "list",
    "path": "/path/to/directory"
})
# → ["file1.txt", "file2.py", "subdir/"]

# Delete file
registry.call_tool("files", {
    "operation": "delete",
    "path": "/path/to/old_file.txt"
})
```

---

## Custom Tools: Complete Pattern

### Simple Function Tool

```python
def my_tool(x: int, y: int) -> int:
    """Adds two numbers. Always provide both arguments."""
    return x + y

registry.register(
    name="my_tool",
    func=my_tool,
    description="Add two integers",
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "First number"},
            "y": {"type": "integer", "description": "Second number"}
        },
        "required": ["x", "y"]
    }
)
```

### Async Tool

```python
import asyncio

async def async_fetch(url: str) -> str:
    """Fetch content from a URL asynchronously."""
    # Simulate async work
    await asyncio.sleep(0.1)
    return f"Content from {url}"

registry.register(
    name="async_fetch",
    func=async_fetch,
    description="Fetch URL content",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string"}
        },
        "required": ["url"]
    }
)

# Call it
result = await registry.call_tool_async("async_fetch", {"url": "https://example.com"})
```

### Class-Based Tool

```python
class DatabaseQuery:
    def __init__(self, connection_string: str):
        self.conn = connection_string
    
    def __call__(self, query: str, params: dict = None) -> list:
        # Execute query
        return [{"result": "..."}]

db_tool = DatabaseQuery("postgresql://localhost/mydb")

registry.register(
    name="database_query",
    func=db_tool,
    description="Execute SQL queries",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "params": {"type": "object"}
        },
        "required": ["query"]
    }
)
```

---

## Tools in a graph (LLM function calling loop)

The most common pattern is an LLM-tool loop: the LLM decides which tools to call, your code executes them, and the result goes back to the LLM.

=== "Python"

    ```python
    from flowgentra_ai.graph import StateGraph, END
    from flowgentra_ai.tools import ToolRegistry, create_tool_node
    from flowgentra_ai.llm import LLMConfig, LLM, Message, ToolDefinition
    from flowgentra_ai import State

    registry = ToolRegistry.with_builtins()
    client   = LLM.from_config(LLMConfig("openai", "gpt-4", api_key="sk-..."))

    # Describe tools to the LLM
    tools = [
        ToolDefinition("calculator", "Perform arithmetic", {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add","subtract","multiply","divide"]},
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        })
    ]

    def llm_node(state):
        messages = state.get("messages") or []
        response = client.chat_with_tools(messages, tools)
        state["messages"] = messages + [response]
        return state

    def tool_executor(state):
        messages = state.get("messages") or []
        last = messages[-1]
        results = []
        for tc in last.tool_calls():
            result = registry.call_tool(tc.name, tc.arguments)
            results.append(Message.tool(str(result), tool_call_id=tc.id))
        state["messages"] = messages + results
        return state

    def router(state):
        messages = state.get("messages") or []
        if messages and messages[-1].has_tool_calls():
            return "tools"
        return "__end__"

    builder = StateGraph()
    builder.add_node("llm",   llm_node)
    builder.add_node("tools", tool_executor)
    builder.set_entry_point("llm")
    builder.add_conditional_edge("llm", router)
    builder.add_edge("tools", "llm")   # loop back for final answer
    graph = builder.compile()

    result = graph.invoke(State({
        "messages": [
            Message.system("You are a helpful math assistant."),
            Message.user("What is (17 * 8) + 45?"),
        ]
    }))

    # Print final assistant response
    for msg in result["messages"]:
        if msg.is_assistant() and not msg.has_tool_calls():
            print(msg.content)
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::{StateGraph, DynState};
    use flowgentra_ai::llm::{LLM, LLMConfig, Message, ToolDefinition};
    use flowgentra_ai::tools::ToolRegistry;
    use serde_json::json;

    let registry = ToolRegistry::with_builtins();
    let client   = LLM::from_config(LLMConfig::openai("gpt-4", "sk-..."));

    let tools = vec![
        ToolDefinition {
            name: "calculator".to_string(),
            description: "Perform arithmetic".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation","a","b"]
            }),
        }
    ];

    let graph = StateGraph::builder()
        .add_node("llm", move |mut state: DynState| {
            let client = client.clone();
            let tools  = tools.clone();
            async move {
                let messages: Vec<Message> = state.get_array("messages").unwrap_or_default();
                let response = client.chat_with_tools(messages.clone(), tools).await?;
                let mut all = messages;
                all.push(response);
                state.set("messages", all);
                Ok(state)
            }
        })
        .add_node("tools", move |mut state: DynState| {
            let registry = registry.clone();
            async move {
                let messages: Vec<Message> = state.get_array("messages").unwrap_or_default();
                let last = messages.last().unwrap();
                let mut tool_results = vec![];
                for tc in last.tool_calls() {
                    let result = registry.call(&tc.name, tc.arguments.clone()).await?;
                    tool_results.push(Message::tool(result.to_string(), &tc.id));
                }
                let mut all = messages;
                all.extend(tool_results);
                state.set("messages", all);
                Ok(state)
            }
        })
        .entry("llm")
        .conditional_edge("llm", |state: &DynState| {
            let msgs: Vec<Message> = state.get_array("messages").unwrap_or_default();
            if msgs.last().map(|m| m.has_tool_calls()).unwrap_or(false) {
                "tools"
            } else {
                "__end__"
            }
        })
        .edge("tools", "llm")
        .build();
    ```

---

## JSON Schema for tools

Use `JsonSchema` to build tool input schemas programmatically instead of writing raw JSON.

=== "Python"

    ```python
    from flowgentra_ai.tools import JsonSchema

    schema = (
        JsonSchema.object()
        .with_description("Weather query parameters")
        .with_required(["city"])
    )

    # All primitive types
    JsonSchema.string()
    JsonSchema.number()
    JsonSchema.integer()
    JsonSchema.boolean()
    JsonSchema.array()

    # Validate a value against the schema
    schema.validate({"city": "Paris"})   # passes
    schema.validate({})                  # raises — city is required
    ```

---

## MCP tools (external tool servers)

MCP (Model Context Protocol) lets you connect to external tool servers. This is useful when your tools live in a separate process or are provided by a third party.

### SSE transport (HTTP-based)

=== "Python"

    ```python
    from flowgentra_ai.types import MCPConfig

    config = MCPConfig(
        transport="sse",
        url="http://localhost:8080/sse",
    )
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::mcp::{MCPConfig, MCPConnectionType};

    let config = MCPConfig {
        transport: MCPConnectionType::Sse {
            url: "http://localhost:8080/sse".to_string(),
        },
        ..Default::default()
    };
    ```

### Stdio transport (subprocess)

```rust
// Rust
let config = MCPConfig {
    transport: MCPConnectionType::Stdio {
        command: "python".to_string(),
        args: vec!["tools_server.py".to_string()],
    },
    ..Default::default()
};
```

### Auto-reconnecting client

For long-running agents, use the reconnecting client — it automatically recovers from dropped connections.

=== "Rust"

    ```rust
    use flowgentra_ai::mcp::ReconnectingMCPClient;

    let client = ReconnectingMCPClient::new(config)
        .connect()
        .await?;

    // List available tools
    let tools = client.list_tools().await?;

    // Call a tool
    let result = client.call_tool("my_tool", json!({"param": "value"})).await?;
    ```

### MCP in YAML config

```yaml
# agent.yaml
tools:
  mcp:
    - name: my_tools
      transport: sse
      url: "http://localhost:8080/sse"
```

---

## ToolNode (convenience helper)

`create_tool_node` wraps a registry into a graph node. It automatically finds tool calls in the last message and executes them.

=== "Python"

    ```python
    from flowgentra_ai.tools import create_tool_node

    registry = ToolRegistry.with_builtins()
    tool_node = create_tool_node(registry)

    builder.add_node("tools", tool_node)
    ```

This is equivalent to the manual executor in the LLM loop example above, but shorter.

---

## Concurrent tool execution

When the LLM makes multiple tool calls at once, you can execute them in parallel:

=== "Rust"

    ```rust
    use futures::future::join_all;

    let futures: Vec<_> = tool_calls
        .iter()
        .map(|tc| registry.call(&tc.name, tc.arguments.clone()))
        .collect();

    let results = join_all(futures).await;
    ```

=== "Python"

    ```python
    import asyncio

    async def execute_tools_parallel(tool_calls, registry):
        tasks = [
            asyncio.create_task(registry.call_tool_async(tc.name, tc.arguments))
            for tc in tool_calls
        ]
        return await asyncio.gather(*tasks)
    ```
