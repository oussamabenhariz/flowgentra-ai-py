"""@tool schema inference from type hints/docstring, and ToolRegistry.to_tool_definitions()."""

from typing import Optional

from flowgentra_ai.tools import ToolRegistry, tool


def test_schema_inferred_from_type_hints_and_docstring():
    @tool(name="html_parser", description="Extract clean text from raw HTML.")
    def html_parser(html: str, strip_scripts: bool = True, max_len: Optional[int] = None) -> str:
        """Args:
            html: Raw HTML source to clean.
            strip_scripts: Remove <script> tags before extracting text.
            max_len: Optional max output length.
        """
        return html.upper()

    schema = html_parser._tool_schema
    assert schema["properties"]["html"] == {"type": "string", "description": "Raw HTML source to clean."}
    assert schema["properties"]["strip_scripts"]["type"] == "boolean"
    assert schema["properties"]["max_len"]["type"] == "integer"
    # strip_scripts/max_len have defaults -> not required; html has none -> required.
    assert schema["required"] == ["html"]


def test_schema_inference_handles_missing_docstring_and_hints():
    @tool(name="noop", description="does nothing")
    def noop(x, y: int = 0):
        return x

    schema = noop._tool_schema
    # No type hint on x -> falls back to "string", not an error.
    assert schema["properties"]["x"]["type"] == "string"
    assert "description" not in schema["properties"]["x"]
    assert schema["properties"]["y"]["type"] == "integer"
    assert schema["required"] == ["x"]


def test_explicit_parameters_still_supported():
    @tool(
        name="legacy_tool",
        description="legacy",
        parameters={"q": "string", "n": "integer"},
        required=["q"],
    )
    def legacy_tool(q, n=1):
        return q

    schema = legacy_tool._tool_schema
    assert schema["properties"]["q"]["type"] == "string"
    assert schema["properties"]["n"]["type"] == "integer"
    assert schema["required"] == ["q"]


def test_registry_get_includes_parameters_for_python_tools():
    @tool(name="upper", description="uppercase a string")
    def upper(text: str) -> str:
        return text.upper()

    registry = ToolRegistry.with_builtins()
    registry.register(upper)
    info = registry.get("upper")
    assert info["parameters"]["properties"]["text"]["type"] == "string"
    assert info["parameters"]["required"] == ["text"]


def test_to_tool_definitions_merges_builtin_and_custom():
    @tool(name="upper", description="uppercase a string")
    def upper(text: str) -> str:
        return text.upper()

    registry = ToolRegistry.with_builtins()
    registry.register(upper)
    defs = registry.to_tool_definitions()
    names = {d.name for d in defs}

    assert "calculator" in names  # built-in, from Rust
    assert "upper" in names  # custom Python tool
    custom = next(d for d in defs if d.name == "upper")
    assert custom.parameters["properties"]["text"]["type"] == "string"
