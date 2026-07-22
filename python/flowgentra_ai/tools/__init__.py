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
import inspect
import re
import typing
from typing import Any, Callable, get_args, get_origin

from flowgentra_ai._native import tools as _t

# ── Schema inference from type hints ──────────────────────────────────────────

_PY_TYPE_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_SIMPLE_TYPE_NAME_TO_JSON_TYPE: dict[str, str] = {
    "string": "string",
    "str": "string",
    "integer": "integer",
    "int": "integer",
    "number": "number",
    "float": "number",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
}


def _json_type_for_annotation(annotation: Any) -> tuple[str, bool]:
    """Map a Python type hint to a `(json_schema_type, is_optional)` pair.

    Unrecognized or missing annotations fall back to `"string"` rather than
    raising — a tool with an un-hinted parameter should still work, just with
    a looser schema.
    """
    if annotation is inspect.Parameter.empty:
        return "string", False

    origin = get_origin(annotation)

    # Optional[T] / T | None → unwrap to T, mark optional.
    if origin is typing.Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1 and len(get_args(annotation)) == 2:
            inner_type, _ = _json_type_for_annotation(args[0])
            return inner_type, True
        return "string", True

    if origin in (list, typing.List):
        return "array", False
    if origin in (dict, typing.Dict):
        return "object", False

    if annotation in _PY_TYPE_TO_JSON_TYPE:
        return _PY_TYPE_TO_JSON_TYPE[annotation], False

    return "string", False


_GOOGLE_ARGS_LINE = re.compile(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def _parse_docstring_arg_descriptions(doc: str | None) -> dict[str, str]:
    """Pull per-parameter descriptions out of a Google-style ``Args:`` block.

    Best-effort: an unparseable or missing docstring just yields no
    descriptions, never an error — schema inference must never fail just
    because a docstring wasn't written yet.
    """
    if not doc:
        return {}
    lines = doc.splitlines()
    descriptions: dict[str, str] = {}
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped.rstrip(":").lower() == "args":
            in_args = True
            continue
        if in_args:
            if not stripped:
                continue
            # A new top-level section (e.g. "Returns:") ends the Args block.
            if stripped.rstrip(":").lower() in ("returns", "raises", "yields", "example", "examples"):
                break
            match = _GOOGLE_ARGS_LINE.match(line)
            if match:
                descriptions[match.group(1)] = match.group(2).strip()
    return descriptions


def _infer_schema_from_signature(func: Callable) -> tuple[dict[str, dict], list[str]]:
    """Build `(properties, required)` from `func`'s signature, type hints, and
    Google-style docstring — the same shape LangChain's `@tool` infers."""
    sig = inspect.signature(func)
    arg_descriptions = _parse_docstring_arg_descriptions(inspect.getdoc(func))

    properties: dict[str, dict] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        json_type, optional_hint = _json_type_for_annotation(param.annotation)
        prop: dict[str, Any] = {"type": json_type}
        if pname in arg_descriptions:
            prop["description"] = arg_descriptions[pname]
        properties[pname] = prop
        if param.default is inspect.Parameter.empty and not optional_hint:
            required.append(pname)
    return properties, required


def _normalize_explicit_parameters(
    parameters: dict[str, str], required: list[str] | None
) -> tuple[dict[str, dict], list[str]]:
    """Convert the legacy `{"name": "string"}` shorthand into full JSON Schema
    property objects, so explicit and inferred schemas end up in the same shape."""
    properties = {
        name: {"type": _SIMPLE_TYPE_NAME_TO_JSON_TYPE.get(type_name, type_name)}
        for name, type_name in parameters.items()
    }
    return properties, list(required or [])


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
    ``ToolRegistry`` use to build ``ToolDefinition`` objects for the LLM.

    Args:
        name:        Tool name the LLM uses to invoke it.
        description: One-sentence description shown to the LLM.
        parameters:  Optional. Mapping of parameter name → JSON Schema type
                     string (e.g. ``{"query": "string", "max_results": "integer"}``).
                     **When omitted, the schema is inferred** from the function's
                     type hints (parameter → JSON type, `Optional[T]`/defaulted
                     params → not required) and its Google-style docstring
                     ``Args:`` block (per-parameter descriptions) — no need to
                     duplicate the signature by hand.
        required:    Optional. Parameter names the LLM must always provide.
                     Inferred from params with no default and a non-Optional
                     type hint when `parameters` is also omitted.

    Example — inferred (recommended)::

        from flowgentra_ai.tools import tool

        @tool(name="html_parser", description="Extract clean text from raw HTML.")
        def html_parser(html: str, strip_scripts: bool = True) -> str:
            \"\"\"Args:
                html: Raw HTML source to clean.
                strip_scripts: Remove <script> tags before extracting text.
            \"\"\"
            from bs4 import BeautifulSoup
            return BeautifulSoup(html, "html.parser").get_text()

        # func._tool_schema == {
        #     "type": "object",
        #     "properties": {
        #         "html": {"type": "string", "description": "Raw HTML source to clean."},
        #         "strip_scripts": {"type": "boolean", "description": "Remove <script> tags before extracting text."},
        #     },
        #     "required": ["html"],   # strip_scripts has a default -> not required
        # }

    Example — explicit (still supported, e.g. when a param can't be type-hinted)::

        @tool(
            name="html_parser",
            description="Extract clean text from raw HTML.",
            parameters={"html": "string"},
            required=["html"],
        )
        def html_parser(html):
            ...
    """
    def decorator(func: Callable) -> Callable:
        if parameters is None:
            properties, inferred_required = _infer_schema_from_signature(func)
            resolved_required = required if required is not None else inferred_required
        else:
            properties, _ = _normalize_explicit_parameters(parameters, required)
            resolved_required = required or []

        func._is_tool = True
        func._tool_name = name
        func._tool_description = description
        func._tool_parameters = {k: v.get("type", "string") for k, v in properties.items()}
        func._tool_required = resolved_required
        func._tool_schema = {
            "type": "object",
            "properties": properties,
            "required": resolved_required,
        }
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
        """Return ``{"name": ..., "description": ..., "parameters": ...}`` for
        any registered tool — built-in or custom Python, inferred or explicit."""
        if self._rust.has(name):
            return self._rust.get(name)
        if name in self._python:
            func = self._python[name]
            return {
                "name": func._tool_name,
                "description": func._tool_description,
                "parameters": func._tool_schema,
            }
        raise KeyError(f"Tool '{name}' not found in registry.")

    def to_tool_definitions(self) -> list:
        """Every registered tool (built-in + custom Python) as a `ToolDefinition`
        list — ready to pass straight to `LLM.chat_with_tools(messages, defs)`.

        Example::

            registry = ToolRegistry.with_builtins()
            registry.register(my_custom_tool)
            response = llm.chat_with_tools(messages, registry.to_tool_definitions())
        """
        from flowgentra_ai.llm import ToolDefinition

        defs = list(self._rust.list_definitions())
        for func in self._python.values():
            defs.append(
                ToolDefinition(func._tool_name, func._tool_description, func._tool_schema)
            )
        return defs

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
