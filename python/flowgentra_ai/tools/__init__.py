"""Tool registry, execution, and all predefined tools.

Examples:
    from flowgentra_ai.tools import ToolRegistry, CalculatorTool, DuckDuckGoSearchTool

    registry = ToolRegistry.with_builtins()
    result = registry.call_tool("duckduckgo_search", {"query": "Rust language"})

    # Tools with API keys
    from flowgentra_ai.tools import TavilySearchTool, WikipediaTool
    wiki = WikipediaTool()
    result = wiki.call({"title": "Python (programming language)"})
"""

from __future__ import annotations
from typing import Any, Callable

from flowgentra_ai._native import tools as _t

# ── @tool decorator ───────────────────────────────────────────────────────────

def tool(
    name: str,
    description: str,
    parameters: dict[str, str] | None = None,
    required: list[str] | None = None,
) -> Callable:
    """Mark a Python function as a custom tool.

    Use this decorator for both **skill-specific tools** (inside a skill's
    ``scripts/`` directory) and **shared custom tools** (registered globally
    via ``ToolRegistry.register()``).

    The decorator adds metadata attributes that ``SkillRegistry`` and
    ``ToolRegistry`` use to build ``ToolSpec`` objects for the LLM.

    Args:
        name:        Tool name the LLM uses to invoke it.
        description: One-sentence description shown to the LLM.
        parameters:  Mapping of parameter name → JSON Schema type string
                     (e.g. ``{"query": "string", "max_results": "integer"}``).
        required:    Parameter names the LLM must always provide.

    Example::

        from flowgentra_ai.tools import tool

        @tool(
            name="html_parser",
            description="Extract clean text from raw HTML.",
            parameters={"html": "string"},
            required=["html"],
        )
        def html_parser(html: str) -> str:
            from bs4 import BeautifulSoup
            return BeautifulSoup(html, "html.parser").get_text()
    """
    def decorator(func: Callable) -> Callable:
        func._is_tool = True
        func._tool_name = name
        func._tool_description = description
        func._tool_parameters = parameters or {}
        func._tool_required = required or []
        return func
    return decorator


# ── ToolRegistry (Python wrapper extending the Rust registry) ─────────────────

class ToolRegistry:
    """Tool registry supporting both built-in Rust tools and Python callables.

    Built-in tools live in the Rust core. Custom Python tools decorated with
    ``@tool`` are registered via :meth:`register` and stored in a Python dict.

    Example::

        from flowgentra_ai.tools import ToolRegistry, tool

        @tool(name="my_formatter", description="Format output as a report.")
        def my_formatter(data: dict) -> str:
            ...

        registry = ToolRegistry.with_builtins()
        registry.register(my_formatter)       # shared custom tool
        registry.list_names()                 # includes my_formatter
    """

    def __init__(self, tools: Any = None) -> None:
        self._rust = _t.ToolRegistry(tools)
        self._python: dict[str, Callable] = {}

    @classmethod
    def with_builtins(cls) -> "ToolRegistry":
        """Create a registry pre-loaded with all keyless built-in tools."""
        instance = cls.__new__(cls)
        instance._rust = _t.ToolRegistry.with_builtins()
        instance._python = {}
        return instance

    def register(self, tool_func: Callable) -> None:
        """Register a ``@tool`` decorated Python callable as a custom tool.

        Args:
            tool_func: A callable decorated with ``@tool``.

        Raises:
            TypeError: If the callable is not decorated with ``@tool``.
        """
        if not getattr(tool_func, "_is_tool", False):
            raise TypeError(
                f"{tool_func!r} must be decorated with @tool before registering."
            )
        self._python[tool_func._tool_name] = tool_func

    def list_names(self) -> list[str]:
        """Return all registered tool names (built-in + custom Python tools)."""
        return list(self._rust.list_names()) + list(self._python.keys())

    def has(self, name: str) -> bool:
        """Return True if a tool with this name is registered."""
        return self._rust.has(name) or name in self._python

    def get(self, name: str) -> dict:
        """Return ``{"name": ..., "description": ...}`` for any registered tool."""
        if self._rust.has(name):
            return self._rust.get(name)
        if name in self._python:
            func = self._python[name]
            return {"name": func._tool_name, "description": func._tool_description}
        raise KeyError(f"Tool '{name}' not found in registry.")

    def call_tool(self, name: str, input: dict) -> Any:
        """Execute a tool by name. Routes to Rust or Python callable."""
        if name in self._python:
            return self._python[name](**input)
        return self._rust.call_tool(name, input)

    def get_python_tool(self, name: str) -> Callable | None:
        """Return the Python callable for a custom tool, or None for built-ins."""
        return self._python.get(name)

    def python_tools(self) -> dict[str, Callable]:
        """Return all registered Python-callable tools."""
        return dict(self._python)

    def validate_input(self, name: str, input: dict) -> None:
        if self._rust.has(name):
            self._rust.validate_input(name, input)

    def __len__(self) -> int:
        return len(self._rust) + len(self._python)

    def __repr__(self) -> str:
        return f"ToolRegistry(builtin={len(self._rust)}, custom={len(self._python)})"


# ── Core infrastructure ───────────────────────────────────────────────────────
ToolCallRequest = _t.ToolCallRequest
ToolCallResult = _t.ToolCallResult
JsonSchema = _t.JsonSchema
ToolNode = _t.ToolNode
create_tool_node = _t.py_create_tool_node
store_tool_calls = _t.py_store_tool_calls
check_tools_condition = _t.py_check_tools_condition

# ── Core built-ins ────────────────────────────────────────────────────────────
CalculatorTool = _t.CalculatorTool
WebRequestTool = _t.WebRequestTool
FilesTool = _t.FilesTool

# ── Search tools ──────────────────────────────────────────────────────────────
DuckDuckGoSearchTool = _t.DuckDuckGoSearchTool
TavilySearchTool = _t.TavilySearchTool
SerpApiSearchTool = _t.SerpApiSearchTool
GoogleSerperTool = _t.GoogleSerperTool
BraveSearchTool = _t.BraveSearchTool

# ── Knowledge tools ───────────────────────────────────────────────────────────
WikipediaTool = _t.WikipediaTool
ArxivTool = _t.ArxivTool
PubMedTool = _t.PubMedTool
WolframAlphaTool = _t.WolframAlphaTool

# ── Code execution tools ──────────────────────────────────────────────────────
PythonReplTool = _t.PythonReplTool
NodeJsReplTool = _t.NodeJsReplTool
ShellTool = _t.ShellTool

# ── Extended file tools ───────────────────────────────────────────────────────
CopyFileTool = _t.CopyFileTool
DeleteFileTool = _t.DeleteFileTool
MoveFileTool = _t.MoveFileTool
FileSearchTool = _t.FileSearchTool

# ── Data tools ────────────────────────────────────────────────────────────────
JsonGetValueTool = _t.JsonGetValueTool
JsonListKeysTool = _t.JsonListKeysTool
CsvQueryTool = _t.CsvQueryTool

# ── Human-in-the-loop ─────────────────────────────────────────────────────────
HumanInputTool = _t.HumanInputTool

# ── Communication tools ───────────────────────────────────────────────────────
GmailTool = _t.GmailTool
SlackTool = _t.SlackTool

# ── External API tools ────────────────────────────────────────────────────────
OpenWeatherMapTool = _t.OpenWeatherMapTool
NewsApiTool = _t.NewsApiTool
AlphaVantageTool = _t.AlphaVantageTool

__all__ = [
    # Infrastructure
    "tool",
    "ToolCallRequest", "ToolCallResult", "ToolRegistry", "JsonSchema",
    "ToolNode", "create_tool_node", "store_tool_calls", "check_tools_condition",
    # Core built-ins
    "CalculatorTool", "WebRequestTool", "FilesTool",
    # Search
    "DuckDuckGoSearchTool", "TavilySearchTool", "SerpApiSearchTool",
    "GoogleSerperTool", "BraveSearchTool",
    # Knowledge
    "WikipediaTool", "ArxivTool", "PubMedTool", "WolframAlphaTool",
    # Code execution
    "PythonReplTool", "NodeJsReplTool", "ShellTool",
    # Extended file ops
    "CopyFileTool", "DeleteFileTool", "MoveFileTool", "FileSearchTool",
    # Data
    "JsonGetValueTool", "JsonListKeysTool", "CsvQueryTool",
    # Human
    "HumanInputTool",
    # Communication
    "GmailTool", "SlackTool",
    # External APIs
    "OpenWeatherMapTool", "NewsApiTool", "AlphaVantageTool",
]
