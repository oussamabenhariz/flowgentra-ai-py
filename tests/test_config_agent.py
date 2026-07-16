"""Config-driven Agent: end-to-end run through the state_graph bridge.

A plain-handler config now auto-selects the state_graph engine
(Agent::run auto-select). This exercises that path through the binding:
build a config + handler module, run the agent, assert the handlers ran.
"""

import textwrap

import pytest

from flowgentra_ai.agent import Agent


HANDLER_MODULE = textwrap.dedent(
    """
    from flowgentra_ai.agent import register_handler

    @register_handler
    def step_a(state: dict) -> dict:
        return {**state, "a_ran": True, "trail": (state.get("trail") or []) + ["a"]}

    @register_handler
    def step_b(state: dict) -> dict:
        return {**state, "b_ran": True, "trail": (state.get("trail") or []) + ["b"]}
    """
)

CONFIG = textwrap.dedent(
    """
    name: pipeline
    python_handler_module: cfg_agent_handlers
    llm:
      provider: ollama
      model: mistral
      api_key: ""
    state_schema:
      input: str
      a_ran: bool
      b_ran: bool
      trail: list
    graph:
      nodes:
        - name: step_a
          handler: step_a
        - name: step_b
          handler: step_b
      edges:
        - from: START
          to: step_a
        - from: step_a
          to: step_b
        - from: step_b
          to: END
    """
)


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    (tmp_path / "cfg_agent_handlers.py").write_text(HANDLER_MODULE)
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(CONFIG)
    monkeypatch.syspath_prepend(str(tmp_path))
    return str(cfg)


def test_config_agent_runs_handlers_via_bridge(config_path):
    agent = Agent.from_config_path(config_path, allow_python_handlers=True)
    agent.set_state("input", "hi")
    result = agent.run()
    assert result["a_ran"] is True
    assert result["b_ran"] is True
    assert result["trail"] == ["a", "b"]
