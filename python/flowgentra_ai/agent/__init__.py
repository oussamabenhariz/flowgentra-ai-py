"""Agent definitions and builders.

This module provides high-level agent interfaces for building autonomous agents with
different execution models (code-driven, config-driven, memory-aware).

Examples:
    Build a config-driven agent with Python handlers (decorator approach):

        # handlers.py
        from flowgentra_ai.agent import register_handler

        @register_handler
        def validate_input(state: dict) -> dict:
            if not state.get("input"):
                raise ValueError(
                    "State field 'input' is required but missing or empty. "
                    "Pass {'input': '...'} as the initial state when invoking the agent."
                )
            return state

        # config.yaml includes: python_handler_module: handlers

        # main.py
        from flowgentra_ai.agent import Agent
        agent = Agent.from_config_path("config.yaml")

    Build a skill-enabled agent:

        from flowgentra_ai.agent import Conversational
        from flowgentra_ai.skills import SkillRegistry
        from flowgentra_ai.llm import LLM

        llm = LLM(provider="anthropic", model="claude-sonnet-4-6")
        skills = SkillRegistry.from_directory("skills/")

        agent = Conversational(name="assistant", llm=llm, skills=skills)
        print(agent.execute_input("Research the latest Rust news"))

    Load an agent from a YAML config file:

        from flowgentra_ai.agent import from_config_path

        agent = from_config_path("agent.yaml")
"""

from typing import Callable

from flowgentra_ai._native import agent as _a, advanced as _adv, utils as _u

Agent = _a.Agent
ToolSpec = _a.ToolSpec
GraphBasedAgent = _a.GraphBasedAgent
AgentConfig = _a.AgentConfig
StateField = _a.StateField
MemoryAwareAgent = _adv.MemoryAwareAgent
MemoryStats = _adv.MemoryStats

# Typed agent constructors (no skills support — use Conversational for that)
ZeroShotReAct = _a.ZeroShotReAct
FewShotReAct = _a.FewShotReAct
ToolCalling = _a.ToolCalling
StructuredChat = _a.StructuredChat
SelfAskWithSearch = _a.SelfAskWithSearch
ReactDocstore = _a.ReactDocstore


# ── Conversational (skills-aware wrapper) ─────────────────────────────────────

class Conversational:
    """Conversational agent — multi-turn dialogue with persistent conversation history.

    When ``skills=`` is provided the agent uses a two-phase interaction as
    specified in SKILLS_PROPOSAL:

    **Phase 1 — Discovery**: The LLM sees only skill names + descriptions and
    selects one via the built-in ``activate_skill`` tool.

    **Phase 2 — Execution**: The selected skill's full instructions and
    **only** its ``allowed-tools`` are presented to the LLM. All other tools
    are hidden.

    The ``skills=`` parameter replaces manually passing ``system_prompt=`` and
    ``tools=`` — the agent handles the two-phase interaction internally.

    When ``skills=`` is omitted the agent behaves as a plain conversational
    agent (``system_prompt=`` and ``tools=`` apply directly).

    Args:
        name:          Agent name.
        llm:           ``LLM`` instance to use.
        system_prompt: Static system prompt (ignored when ``skills=`` is set).
        tools:         Tools passed to the LLM (ignored when ``skills=`` is set).
        retries:       LLM call retries (default: 3).
        memory_steps:  Conversation history window size.
        skills:        ``SkillRegistry`` — enables two-phase skill routing.

    Example with skills::

        from flowgentra_ai.agent import Conversational
        from flowgentra_ai.skills import SkillRegistry
        from flowgentra_ai.llm import LLM

        llm    = LLM(provider="anthropic", model="claude-sonnet-4-6")
        skills = SkillRegistry.from_directory("skills/")

        agent = Conversational(name="assistant", llm=llm, skills=skills)
        print(agent.execute_input("Research the latest Rust news"))

    Example without skills (plain conversational)::

        from flowgentra_ai.agent import Conversational, ToolSpec
        from flowgentra_ai.llm import LLM

        llm   = LLM(provider="openai", model="gpt-4")
        agent = Conversational(name="chat", llm=llm, system_prompt="Be concise.")
        print(agent.execute_input("Hello!"))
    """

    def __init__(
        self,
        name: str,
        llm,
        system_prompt: str | None = None,
        tools=None,
        retries: int = 3,
        memory_steps: int | None = None,
        skills=None,
        tool_registry=None,
    ) -> None:
        if skills is not None:
            # Two-phase skill routing — delegate to SkillAgent
            from flowgentra_ai.skills import SkillAgent
            self._inner = SkillAgent(name=name, llm=llm, skills=skills, retries=retries)
            self._skill_mode = True
        else:
            # Plain conversational — use Rust implementation directly
            self._inner = _a.Conversational(
                name=name,
                llm=llm,
                system_prompt=system_prompt,
                tools=tools,
                retries=retries,
                memory_steps=memory_steps,
                tool_registry=tool_registry,
            )
            self._skill_mode = False

    def execute_input(self, user_input: str) -> str:
        """Run the agent with a string input and return the response."""
        return self._inner.execute_input(user_input)

    @property
    def name(self) -> str:
        return self._inner.name

    def node_names(self) -> list[str]:
        if self._skill_mode:
            return []
        return self._inner.node_names()

    def __repr__(self) -> str:
        return f"Conversational(name={self.name!r})"


# ── Config-driven loading ─────────────────────────────────────────────────────

def from_config_path(config_path: str):
    """Create an agent from a YAML config file.

    Supports a ``skills:`` block for two-phase skill routing::

        name: assistant
        llm:
          provider: anthropic
          model: claude-sonnet-4-6

        tools:
          - module: shared_tools
            names: [my_formatter]

        skills:
          directory: skills/

    Without a ``skills:`` block the config is processed by the Rust engine
    (which supports Python handler modules and all graph-based config options).

    Args:
        config_path: Path to the YAML config file.

    Returns:
        An agent instance — either a ``SkillAgent`` (when ``skills:`` is
        present) or an ``Agent`` (for standard graph-based configs).
    """
    with open(config_path) as f:
        raw = f.read()

    # Fast path: no skills block — delegate entirely to Rust
    if "skills:" not in raw:
        return _u.py_from_config_path(config_path)

    try:
        import yaml
    except ImportError:
        raise ImportError(
            "pyyaml is required to use the 'skills:' block in agent config YAML. "
            "Install it with: pip install pyyaml"
        )

    config = yaml.safe_load(raw)

    # Double-check: 'skills:' might have appeared inside a string/comment
    if "skills" not in config:
        return _u.py_from_config_path(config_path)

    # ── Build global tool registry ────────────────────────────────────────────
    from flowgentra_ai.tools import ToolRegistry
    import importlib

    tool_registry = ToolRegistry.with_builtins()
    for tool_cfg in config.get("tools", []) or []:
        mod = importlib.import_module(tool_cfg["module"])
        for func_name in tool_cfg.get("names", []):
            tool_registry.register(getattr(mod, func_name))

    # ── Build skill registry ──────────────────────────────────────────────────
    from flowgentra_ai.skills import SkillRegistry, SkillAgent

    skills_cfg = config["skills"]
    if isinstance(skills_cfg, dict):
        if "directory" in skills_cfg:
            skill_registry = SkillRegistry.from_directory(
                skills_cfg["directory"],
                tool_registry=tool_registry,
            )
        else:
            skill_registry = SkillRegistry(tool_registry=tool_registry)
            for entry in skills_cfg.get("paths", []) or []:
                path = entry["path"] if isinstance(entry, dict) else entry
                skill_registry.load(path)
    elif isinstance(skills_cfg, list):
        skill_registry = SkillRegistry(tool_registry=tool_registry)
        for entry in skills_cfg:
            path = entry["path"] if isinstance(entry, dict) else entry
            skill_registry.load(path)
    else:
        raise ValueError(
            f"Invalid 'skills:' value in {config_path!r}: expected a dict "
            f"(with 'directory:' or 'paths:') or a list of path entries."
        )

    # ── Build LLM ─────────────────────────────────────────────────────────────
    from flowgentra_ai.llm import LLM

    llm_cfg = config.get("llm", {})
    llm = LLM(**llm_cfg)

    return SkillAgent(
        name=config.get("name", "assistant"),
        llm=llm,
        skills=skill_registry,
    )


def register_handler(func: Callable) -> Callable:
    """Decorator to mark a Python function as an agent handler.

    Decorated functions are auto-discovered when the containing module is
    specified via ``python_handler_module`` in the agent config YAML.

    The function must accept a ``dict`` (the agent state) and return a
    ``dict`` (full state or partial update merged into the current state).

    Example::

        from flowgentra_ai.agent import register_handler

        @register_handler
        def process_input(state: dict) -> dict:
            return {**state, "output": state["input"].upper()}

    Then in ``config.yaml``::

        python_handler_module: handlers  # module containing the above function
        graph:
          nodes:
            - name: process
              handler: process_input
    """
    func._is_handler = True  # type: ignore[attr-defined]
    return func


__all__ = [
    # Core agent classes
    "Agent",
    "GraphBasedAgent",
    # Typed agent constructors (preferred)
    "ZeroShotReAct",
    "FewShotReAct",
    "Conversational",
    "ToolCalling",
    "StructuredChat",
    "SelfAskWithSearch",
    "ReactDocstore",
    # Agent configuration
    "ToolSpec",
    "AgentConfig",
    "StateField",
    # Memory-aware agents
    "MemoryAwareAgent",
    "MemoryStats",
    # Python handler registration
    "register_handler",
    # Config-driven loading
    "from_config_path",
]
