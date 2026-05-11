# Model Context Protocol (MCP)

**Connect your agents to external tools and services** through the Model Context Protocol. MCP enables seamless integration with web APIs, local processes, and containerized services.

## What is MCP?

The Model Context Protocol (MCP) is a standard for connecting AI agents to external tools and data sources. It provides a unified interface for accessing web services, databases, local applications, and specialized tools.

Unlike local tools (functions defined in your code), MCP tools are external services that run independently. This allows you to:

- Access web APIs and cloud services
- Integrate with existing infrastructure
- Use specialized tools without rewriting them
- Scale tool execution across different environments
- Maintain tool isolation and security boundaries

## Installation

All MCP functionality lives under `flowgentra_ai.mcp`:

```python
from flowgentra_ai.mcp import (
    MCPConfig,
    MCPAuth,
    MCPConnectionSettings,
    MCPClient,
    MCPTool,
    MCPResource,
    MCPResourceContent,
    MCPPrompt,
    MCPPromptResult,
    create_client,
    merge_tool_lists,
)
```

Types are also re-exported from `flowgentra_ai.types` for convenience.

## Connection Types

MCP supports three connection types:

### SSE (Server-Sent Events)

Connect to HTTP-based MCP servers. Ideal for web services and cloud APIs.

```python
config = MCPConfig.sse("http://api.example.com/mcp", name="web_api")
```

### Stdio (Standard I/O)

Launch a local process and communicate via stdin/stdout. Perfect for CLI tools.

```python
config = MCPConfig.stdio(
    "npx",
    ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    name="filesystem"
)
```

### Docker

Run a tool in an isolated Docker container.

```python
config = MCPConfig.docker("my-org/tool-image:latest", name="container_tool")
```

## Creating a Client

Use `create_client()` to create an `MCPClient` from any config. The client is automatically wrapped with retry logic and tool-list caching.

```python
from flowgentra_ai.mcp import MCPConfig, create_client

config = MCPConfig.sse("http://localhost:8080/sse", name="my-server")
client = create_client(config)

# Always shut down when done (stops processes/containers for stdio/docker)
client.shutdown()
```

## Discovering Tools

```python
# Initialize the MCP handshake (optional — called automatically on first use)
protocol_version = client.initialize()

# List all available tools
tools = client.list_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}")
    print(f"  Input schema: {tool.input_schema}")

# Check if server is reachable
if client.health_check():
    print("Server is up")
```

## Calling Tools

```python
# Call a single tool
result = client.call_tool("search", {"query": "python asyncio", "max_results": 5})
print(result)  # Python dict/list/str depending on the tool

# Call multiple tools in parallel
results = client.call_tools_parallel([
    ("search", {"query": "MCP protocol"}),
    ("fetch", {"url": "https://example.com"}),
])
for r in results:
    print(r)
```

## Resources

Some MCP servers expose resources (files, database records, etc.):

```python
# List available resources
resources = client.list_resources()
for r in resources:
    print(f"{r.uri} ({r.mime_type}): {r.description}")

# Read a resource
content = client.read_resource("file:///path/to/document.txt")
print(content.text)      # text content
print(content.blob)      # base64 blob (for binary)
print(content.mime_type) # e.g. "text/plain"
```

## Prompts

Some MCP servers expose reusable prompt templates:

```python
# List prompt templates
prompts = client.list_prompts()
for p in prompts:
    print(f"{p.name}: {p.description}")
    for arg in p.arguments:
        print(f"  {arg.name} (required={arg.required}): {arg.description}")

# Render a prompt with arguments
result = client.get_prompt("summarize", {"text": "Long document content here..."})
for message in result.messages:
    print(f"{message.role}: {message.content}")
```

## Authentication

### Bearer Token

```python
from flowgentra_ai.mcp import MCPAuth

auth = MCPAuth("bearer", {"token": "my-api-token"})
config = MCPConfig.sse("http://api.example.com/mcp", name="api").with_auth(auth)
```

### API Key

```python
auth = MCPAuth("api_key", {"header": "X-API-Key", "key": "abc123"})
config = MCPConfig.sse("http://api.example.com/mcp", name="api").with_auth(auth)
```

### Basic Auth

```python
auth = MCPAuth("basic", {"username": "user", "password": "pass"})
config = MCPConfig.sse("http://api.example.com/mcp", name="api").with_auth(auth)
```

## Connection Settings

Fine-tune timeouts, retries, and environment:

```python
from flowgentra_ai.mcp import MCPConnectionSettings

settings = MCPConnectionSettings(
    timeout=30,           # general fallback timeout (seconds)
    connect_timeout=5,    # timeout for establishing connection
    call_timeout=60,      # timeout for individual tool calls
    max_retries=3,        # retry failed calls up to N times
    # Stdio/Docker specific:
    working_dir="/app",
    env_vars={"API_KEY": "secret"},
    # Docker specific:
    container_name="my-tool",
    port=8080,
    host_port=9090,
)

config = MCPConfig.stdio("python", ["-m", "my_tool"], name="tool") \
                  .with_connection_settings(settings)
```

## Tool Namespacing and Filtering

When using multiple MCP servers you can namespace and filter tools to avoid collisions:

```python
# Add a namespace prefix: "search.query", "search.fetch", etc.
config = MCPConfig.sse("http://search.example.com/mcp", name="search") \
                  .with_namespace("search")

# Only expose specific tools
config = MCPConfig.sse("http://api.example.com/mcp", name="api") \
                  .with_tool_include(["search", "fetch"])

# Hide specific tools
config = MCPConfig.sse("http://api.example.com/mcp", name="api") \
                  .with_tool_exclude(["dangerous_tool"])
```

## Multiple Clients

Merge tool lists from multiple MCP servers into one:

```python
from flowgentra_ai.mcp import MCPConfig, create_client, merge_tool_lists

client1 = create_client(MCPConfig.sse("http://server1/sse", name="s1"))
client2 = create_client(MCPConfig.stdio("python", ["-m", "tool"], name="s2"))

all_tools = merge_tool_lists([client1, client2])
print(f"Total tools: {len(all_tools)}")
```

## Using MCP in Agent Nodes

```python
from flowgentra_ai.graph import StateGraph, END
from flowgentra_ai.mcp import MCPConfig, create_client
from typing import TypedDict, Any

class AgentState(TypedDict):
    query: str
    result: Any

# Create client once at startup
mcp_client = create_client(MCPConfig.sse("http://localhost:8080/sse", name="tools"))

def search_node(state: AgentState) -> AgentState:
    result = mcp_client.call_tool("search", {"query": state["query"]})
    return {"result": result}

builder = StateGraph(AgentState)
builder.add_node("search", search_node)
builder.set_entry_point("search")
builder.add_edge("search", END)

graph = builder.compile()
output = graph.invoke({"query": "hello world", "result": None})
```

## YAML Configuration

Configure MCP connections in your agent configuration file:

```yaml
graph:
  mcps:
    web_search:
      type: sse
      url: "http://api.example.com/search"
      timeout: 30
      auth:
        type: bearer
        token: "${API_TOKEN}"

    calculator:
      type: stdio
      command: "python"
      args: ["-c", "import sys; exec(sys.stdin.read())"]
      timeout: 5

    data_processor:
      type: docker
      image: "myregistry.com/data-tools:latest"
      timeout: 120
```

## API Reference

### `MCPConfig`

| Method | Description |
|--------|-------------|
| `MCPConfig.sse(url, name=None)` | Create SSE config |
| `MCPConfig.stdio(command, args=None, name=None)` | Create Stdio config |
| `MCPConfig.docker(image, name=None)` | Create Docker config |
| `.with_auth(auth)` | Return copy with authentication |
| `.with_namespace(ns)` | Return copy with tool namespace prefix |
| `.with_tool_include(tools)` | Return copy that only exposes listed tools |
| `.with_tool_exclude(tools)` | Return copy that hides listed tools |
| `.with_connection_settings(settings)` | Return copy with custom connection settings |
| `.is_remote()` | True for SSE connections |
| `.is_local()` | True for Stdio/Docker connections |

### `MCPClient`

| Method | Description |
|--------|-------------|
| `initialize()` | MCP protocol handshake, returns protocol version string |
| `list_tools()` → `list[MCPTool]` | List all available tools |
| `call_tool(name, args)` → `Any` | Call a tool with a dict of arguments |
| `call_tools_parallel(calls)` → `list[Any]` | Call multiple tools in parallel |
| `health_check()` → `bool` | Check if the server is reachable |
| `shutdown()` | Gracefully stop the connection |
| `list_resources()` → `list[MCPResource]` | List available resources |
| `read_resource(uri)` → `MCPResourceContent` | Read a resource by URI |
| `list_prompts()` → `list[MCPPrompt]` | List prompt templates |
| `get_prompt(name, args)` → `MCPPromptResult` | Render a prompt template |

### `create_client(config)` → `MCPClient`

Factory function. Creates the appropriate client (SSE/Stdio/Docker) from the config, automatically wrapped with retry logic and tool-list caching.

### `merge_tool_lists(clients)` → `list[MCPTool]`

Merge tool lists from multiple `MCPClient` instances into a single flat list.

### Data Types

| Type | Key Attributes |
|------|---------------|
| `MCPTool` | `name`, `description`, `input_schema` |
| `MCPResource` | `uri`, `name`, `description`, `mime_type` |
| `MCPResourceContent` | `uri`, `mime_type`, `text`, `blob` |
| `MCPPrompt` | `name`, `description`, `arguments` |
| `MCPPromptArgument` | `name`, `description`, `required` |
| `MCPPromptResult` | `description`, `messages` |
| `MCPPromptMessage` | `role`, `content` |
| `MCPAuth` | `auth_type`, `credentials` |
| `MCPConnectionSettings` | `timeout`, `connect_timeout`, `call_timeout`, `max_retries`, `env_vars`, … |

## Best Practices

**Configuration**
- Use environment variables for credentials: `MCPAuth("bearer", {"token": os.environ["API_TOKEN"]})`
- Set `call_timeout` on long-running tools and `connect_timeout` separately for faster failure detection
- Use namespaces when combining tools from multiple servers to prevent name collisions

**Performance**
- Create clients once at startup and reuse them — `list_tools()` results are cached automatically
- Use `call_tools_parallel()` for independent tool calls instead of sequential `call_tool()` calls

**Resource Management**
- Always call `client.shutdown()` when done, especially for Stdio and Docker clients
- Use a `try/finally` block to ensure cleanup:

  ```python
  client = create_client(config)
  try:
      result = client.call_tool("my_tool", {"arg": "value"})
  finally:
      client.shutdown()
  ```

**Security**
- Use Docker connections for untrusted or third-party tools
- Use `with_tool_include()` to whitelist only the tools your agent actually needs
- Validate tool results before passing them to an LLM
