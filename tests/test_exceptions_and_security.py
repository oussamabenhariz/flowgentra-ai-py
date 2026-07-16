"""Exception hierarchy, config trust boundary, tool registry safety."""

import textwrap
import warnings

import pytest

import flowgentra_ai.exceptions as ex
from flowgentra_ai.agent import Agent
from flowgentra_ai.tools import ToolRegistry


# ── Exception hierarchy ───────────────────────────────────────────────────────

ALL_EXCEPTIONS = [
    ex.ConfigurationError,
    ex.ValidationError,
    ex.GraphError,
    ex.NodeNotFoundError,
    ex.CycleError,
    ex.LLMError,
    ex.MCPError,
    ex.ToolExecutionError,
    ex.AgentExecutionError,
    ex.WorkflowTimeoutError,
    ex.SerializationError,
    ex.CheckpointError,
    ex.InternalError,
]


@pytest.mark.parametrize("exc", ALL_EXCEPTIONS)
def test_exception_inherits_base(exc):
    assert issubclass(exc, ex.FlowgentraAIError)


def test_graph_error_subclasses():
    assert issubclass(ex.NodeNotFoundError, ex.GraphError)
    assert issubclass(ex.CycleError, ex.GraphError)


# ── Config trust boundary (allow_python_handlers) ────────────────────────────

HANDLER_MODULE = textwrap.dedent(
    """
    from flowgentra_ai.agent import register_handler

    @register_handler
    def start(state: dict) -> dict:
        return {**state, "ran": True}
    """
)

CONFIG_WITH_HANDLERS = textwrap.dedent(
    """
    name: test-agent
    python_handler_module: trust_handlers_mod
    llm:
      provider: ollama
      model: mistral
      api_key: ""
    graph:
      nodes:
        - name: start
          handler: start
      edges:
        - from: START
          to: start
        - from: start
          to: END
    """
)


@pytest.fixture
def handler_config(tmp_path, monkeypatch):
    (tmp_path / "trust_handlers_mod.py").write_text(HANDLER_MODULE)
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(CONFIG_WITH_HANDLERS)
    monkeypatch.syspath_prepend(str(tmp_path))
    return str(cfg)


def test_config_handlers_rejected_when_unset(handler_config):
    """Since 0.3.0 the default rejects handler imports (0.2.x warned)."""
    with pytest.raises(ex.ValidationError, match="allow_python_handlers"):
        Agent.from_config_path(handler_config)


def test_config_handlers_rejected_when_false(handler_config):
    with pytest.raises(ex.ValidationError, match="allow_python_handlers"):
        Agent.from_config_path(handler_config, allow_python_handlers=False)


def test_config_handlers_silent_when_true(handler_config):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Agent.from_config_path(handler_config, allow_python_handlers=True)
    assert not [w for w in caught if "IMPORTED" in str(w.message)]


# ── Tool registry ─────────────────────────────────────────────────────────────

def test_builtin_calculator_is_structured_not_eval():
    reg = ToolRegistry.with_builtins()
    result = reg.call_tool("calculator", {"operation": "add", "a": 2, "b": 3})
    assert result["result"] == 5.0
    # An eval-style payload must be rejected, not evaluated.
    with pytest.raises(Exception):
        reg.call_tool("calculator", {"operation": "__import__('os')", "a": 1, "b": 2})


def test_calculator_division_by_zero_is_typed_error():
    reg = ToolRegistry.with_builtins()
    with pytest.raises(Exception, match="[Dd]ivision by zero"):
        reg.call_tool("calculator", {"operation": "divide", "a": 1, "b": 0})


def test_unknown_tool_raises():
    reg = ToolRegistry.with_builtins()
    with pytest.raises(Exception):
        reg.call_tool("no_such_tool", {})


# ── Deprecation vocabulary ────────────────────────────────────────────────────

def test_execute_input_emits_deprecation_warning():
    """All predefined agents: execute_input warns, run() does not.

    Uses ZeroShotReAct with an Ollama config (no network call happens —
    the warning fires before execution; the LLM call itself fails fast)."""
    from flowgentra_ai.agent import ZeroShotReAct
    from flowgentra_ai.llm import LLM

    agent = ZeroShotReAct(
        name="t",
        llm=LLM(provider="ollama", model="does-not-exist"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            agent.execute_input("hi")
        except Exception:
            pass  # offline: the LLM call fails; we only assert the warning
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
