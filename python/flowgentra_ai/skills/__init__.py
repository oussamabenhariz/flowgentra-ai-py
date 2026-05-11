"""Skill system for flowgentra_ai — follows the SKILLS_PROPOSAL design.

Two-phase interaction:

  **Phase 1 — Discovery**: ``build_menu()`` → LLM sees only skill names + descriptions,
  picks a skill via the ``activate_skill`` tool.

  **Phase 2 — Execution**: ``build_system_prompt(skill_name)`` + ``resolve_tools(skill_name)``
  → LLM sees full instructions and **only** that skill's scoped tools.

Tool scoping:
  When a skill is active the LLM sees **only** its ``allowed-tools``.
  All other tools (from other skills or the global registry) are hidden.

Skill folder layout (agentskills.io standard)::

    skills/
    └── web-research/
        ├── SKILL.md       ← required: frontmatter + instructions
        ├── scripts/       ← optional: @tool decorated skill-specific tools
        │   └── tools.py
        ├── references/    ← optional: extra context appended to instructions
        │   └── guide.md
        └── assets/        ← optional: static templates / data files

Example::

    # skills/web-research/scripts/tools.py
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

    # main.py
    from flowgentra_ai.tools import ToolRegistry, tool
    from flowgentra_ai.skills import SkillRegistry, SkillAgent
    from flowgentra_ai.llm import LLMClient

    @tool(name="my_formatter", description="Format output as a structured report.")
    def my_formatter(data: dict) -> str: ...

    tool_registry = ToolRegistry.with_builtins()
    tool_registry.register(my_formatter)    # shared custom tool

    skill_registry = SkillRegistry.from_directory(
        "skills/",
        tool_registry=tool_registry,
    )

    agent = SkillAgent(
        name="assistant",
        llm=LLMClient(provider="anthropic", model="claude-sonnet-4-6"),
        skills=skill_registry,
    )

    response = agent.execute_input("Research the latest Rust news")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# ── Rust-backed core (parsing, registry, prompt generation) ───────────────────
# Use attribute access (from _native import skills as _s) rather than dotted
# submodule import (from _native.skills import ...) — PyO3 submodules are
# attributes of the parent extension, not sys.modules entries, so dotted import
# fails with "not a package" until the module is explicitly registered.
from flowgentra_ai._native import skills as _s
Skill = _s.Skill
SkillRegistry = _s.SkillRegistry

if TYPE_CHECKING:
    from flowgentra_ai.llm import LLM
    from flowgentra_ai.agent import Conversational


# ── SkillAgent ────────────────────────────────────────────────────────────────

class SkillAgent:
    """Two-phase skill-enabled conversational agent.

    **Phase 1 — Discovery**: The LLM sees only the skill menu (names +
    descriptions) and selects a skill via the built-in ``activate_skill``
    tool.  If the registry contains a single skill this phase is skipped.

    **Phase 2 — Execution**: A ``Conversational`` agent is created (and
    cached) with the selected skill's full instructions as its system
    prompt and **only** that skill's ``allowed-tools`` visible to the LLM.

    Args:
        name:    Agent name (used for logging / repr).
        llm:     ``LLM`` instance to use for all calls.
        skills:  ``SkillRegistry`` loaded with one or more skills.
        retries: Number of retries passed to the inner ``Conversational``
                 agents (default: 3).

    Example::

        from flowgentra_ai.skills import SkillRegistry, SkillAgent
        from flowgentra_ai.llm import LLMClient

        skills = SkillRegistry.from_directory("skills/")

        agent = SkillAgent(
            name="assistant",
            llm=LLMClient(provider="anthropic", model="claude-sonnet-4-6"),
            skills=skills,
        )
        print(agent.execute_input("Research the latest Rust news"))
    """

    def __init__(
        self,
        name: str,
        llm: "LLM",
        skills: SkillRegistry,
        retries: int = 3,
    ) -> None:
        self._name = name
        self._llm = llm
        self._skills = skills
        self._retries = retries
        # Cache: skill_name → Conversational agent
        self._skill_agents: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def _activate_skill_definition(self) -> Any:
        from flowgentra_ai.llm import ToolDefinition

        skill_names = self._skills.list()
        return ToolDefinition(
            "activate_skill",
            "Select the most relevant skill for the user's request.",
            {
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "enum": skill_names,
                        "description": "The name of the skill to activate.",
                    }
                },
                "required": ["skill"],
            },
        )

    def _select_skill(self, user_input: str) -> str:
        """Phase 1: ask the LLM to choose a skill via activate_skill."""
        from flowgentra_ai.llm import Message

        skill_names = self._skills.list()

        if not skill_names:
            raise ValueError("SkillRegistry is empty — load at least one skill.")

        if len(skill_names) == 1:
            return skill_names[0]

        menu = self._skills.build_menu()
        response = self._llm.chat_with_tools(
            [Message.system(menu), Message.user(user_input)],
            [self._activate_skill_definition()],
        )

        if response.has_tool_calls():
            for tc in response.tool_calls():
                if tc.name == "activate_skill":
                    args = tc.arguments
                    if isinstance(args, dict):
                        chosen = args.get("skill")
                        if chosen and chosen in skill_names:
                            return chosen

        # Fallback: first skill in registry
        return skill_names[0]

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def _get_or_create_skill_agent(self, skill_name: str) -> Any:
        """Phase 2: return a cached Conversational agent for the skill."""
        if skill_name not in self._skill_agents:
            from flowgentra_ai.agent import Conversational

            system_prompt = self._skills.build_system_prompt(skill_name)
            tools = self._skills.resolve_tools(skill_name)

            self._skill_agents[skill_name] = Conversational(
                name=f"{self._name}/{skill_name}",
                llm=self._llm,
                system_prompt=system_prompt,
                tools=tools or None,
                retries=self._retries,
            )

        return self._skill_agents[skill_name]

    # ── Public API ────────────────────────────────────────────────────────────

    def execute_input(self, user_input: str) -> str:
        """Run the two-phase interaction and return the agent's response.

        1. Phase 1 — select a skill (or skip if only one skill exists).
        2. Phase 2 — execute the user's input with the selected skill's
           instructions and scoped tools.

        Args:
            user_input: The user's message.

        Returns:
            The agent's response string.
        """
        skill_name = self._select_skill(user_input)
        agent = self._get_or_create_skill_agent(skill_name)
        return agent.execute_input(user_input)

    def active_skills(self) -> list[str]:
        """Return the names of skills that have been activated so far."""
        return list(self._skill_agents.keys())

    def __repr__(self) -> str:
        return (
            f"SkillAgent(name={self._name!r}, "
            f"skills={self._skills.list()})"
        )


__all__ = [
    "Skill",
    "SkillRegistry",
    "SkillAgent",
]
