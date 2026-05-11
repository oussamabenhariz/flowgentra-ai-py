"""
Model Context Protocol (MCP) support for FlowgentraAI.

Connect agents to external tools and services through the MCP standard.
Supports SSE (HTTP), Stdio (local processes), and Docker connections.

Quick start:

    from flowgentra_ai.mcp import MCPConfig, create_client

    # Connect to an SSE server
    config = MCPConfig.sse("http://localhost:8080/sse", name="my-server")
    client = create_client(config)

    # Discover and call tools
    tools = client.list_tools()
    result = client.call_tool("search", {"query": "hello world"})

    client.shutdown()

Advanced usage:

    from flowgentra_ai.mcp import (
        MCPConfig, MCPAuth, MCPConnectionSettings,
        MCPClient, create_client, merge_tool_lists
    )

    auth = MCPAuth("bearer", {"token": "my-token"})
    settings = MCPConnectionSettings(timeout=30, max_retries=3)

    config = (
        MCPConfig.sse("http://api.example.com/mcp", name="api")
              .with_auth(auth)
              .with_namespace("api")
              .with_tool_include(["search", "fetch"])
              .with_connection_settings(settings)
    )

    client = create_client(config)

    # Resources and prompts
    resources = client.list_resources()
    content = client.read_resource("file:///path/to/resource")

    prompts = client.list_prompts()
    rendered = client.get_prompt("summarize", {"text": "..."})

    # Multiple clients
    all_tools = merge_tool_lists([client1, client2])
"""

from flowgentra_ai._native import mcp as _mcp

# Connection configuration
MCPConnectionType = _mcp.MCPConnectionType
MCPAuth = _mcp.MCPAuth
MCPConnectionSettings = _mcp.MCPConnectionSettings
MCPConfig = _mcp.MCPConfig

# Result / data types
MCPTool = _mcp.MCPTool
MCPResource = _mcp.MCPResource
MCPResourceContent = _mcp.MCPResourceContent
MCPPromptArgument = _mcp.MCPPromptArgument
MCPPrompt = _mcp.MCPPrompt
MCPPromptMessage = _mcp.MCPPromptMessage
MCPPromptResult = _mcp.MCPPromptResult

# Client
MCPClient = _mcp.MCPClient

# Factory / utility functions
create_client = _mcp.create_client
merge_tool_lists = _mcp.merge_tool_lists

__all__ = [
    # Config
    "MCPConnectionType",
    "MCPAuth",
    "MCPConnectionSettings",
    "MCPConfig",
    # Data types
    "MCPTool",
    "MCPResource",
    "MCPResourceContent",
    "MCPPromptArgument",
    "MCPPrompt",
    "MCPPromptMessage",
    "MCPPromptResult",
    # Client
    "MCPClient",
    # Functions
    "create_client",
    "merge_tool_lists",
]
