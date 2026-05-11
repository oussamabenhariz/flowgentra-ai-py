# Skills

Skills are self-contained agent personas loaded from a directory. Each skill bundles its own instructions, a curated tool list, and optional reference documents. The **SkillAgent** uses a two-phase model: first the LLM picks a skill, then it executes with that skill's scoped prompt and tools.

---

## Skill Directory Layout

```
skills/
└── math-solver/
    ├── SKILL.md          ← required: frontmatter + instructions
    ├── scripts/          ← optional: @tool decorated skill-specific tools
    │   └── tools.py
    └── references/       ← optional: extra context appended to instructions
        └── formulas.md
```

### SKILL.md Format

```markdown
---
name: math-solver
description: "Solve mathematical problems step-by-step."
version: "1.0.0"
allowed-tools:
  - calculator          # built-in (from ToolRegistry.with_builtins())
  - expression_eval     # skill-specific (defined in scripts/tools.py)
---

# Math Solver Skill

You are a precise math solver who shows full working for every problem.
```

`allowed-tools` controls **exactly** which tools the LLM sees during execution. Tools not listed are hidden from the agent.

---

## Loading Skills

### Multiple skills from a directory

```python
from flowgentra_ai.skills import SkillRegistry
from flowgentra_ai.tools import ToolRegistry

# Build-in tools (calculator, http_get, file, …) must be passed explicitly.
# Without tool_registry=, any built-in listed in allowed-tools raises ValueError.
tool_registry = ToolRegistry.with_builtins()

registry = SkillRegistry.from_directory("skills/", tool_registry=tool_registry)
print(registry.list())  # ['math-solver', 'tech-writer']
```

### Single skill

```python
single = SkillRegistry(tool_registry=tool_registry)
single.load("skills/math-solver")
```

`SkillRegistry.from_directory` scans for **subdirectories** that contain a `SKILL.md`. Do **not** pass the skill folder itself — pass its parent.

### With shared custom tools

```python
from flowgentra_ai.tools import ToolRegistry, tool

@tool(name="my_formatter", description="Format output as a report.")
def my_formatter(data: dict) -> str: ...

tool_registry = ToolRegistry.with_builtins()
tool_registry.register(my_formatter)  # now usable in any skill's allowed-tools

registry = SkillRegistry.from_directory("skills/", tool_registry=tool_registry)
```

---

## Skill-Specific Tools

Define tools in `skills/<name>/scripts/tools.py` using `@tool`:

```python
from flowgentra_ai.tools import tool

@tool(
    name="expression_eval",
    description="Evaluate a math expression string. Example: '2**10 + sqrt(144)'",
    parameters={"expression": "string"},
    required=["expression"],
)
def expression_eval(expression: str) -> dict:
    import math
    safe = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    safe["abs"] = abs
    try:
        return {"result": float(eval(expression, {"__builtins__": {}}, safe))}
    except Exception as e:
        return {"error": str(e)}
```

These are auto-discovered when `SkillRegistry.from_directory` or `.load()` runs. No manual registration needed.

To call a skill tool directly (e.g. in tests), retrieve it from the registry:

```python
expression_eval = registry.get_callable("expression_eval")
result = expression_eval("2**10 + sqrt(144)")
print(result)  # {'result': 1036.0, 'expression': '2**10 + sqrt(144)'}
```

`get_callable` returns `None` for built-in tools (`calculator`, `http_get`, etc.) — only skill-specific tools defined in `scripts/tools.py` return a Python callable.

---

## SkillAgent: Two-Phase Routing

```python
from flowgentra_ai.skills import SkillRegistry, SkillAgent
from flowgentra_ai.tools import ToolRegistry
from flowgentra_ai.llm import LLMClient

tool_registry = ToolRegistry.with_builtins()
registry = SkillRegistry.from_directory("skills/", tool_registry=tool_registry)

agent = SkillAgent(
    name="assistant",
    llm=LLMClient(provider="mistral", model="mistral-small-latest"),
    skills=registry,
)

response = agent.execute_input("Solve: if 3x + 7 = 22, what is x?")
print(agent.active_skills())  # ['math-solver']
```

**Phase 1 — Discovery**: LLM sees only skill names + descriptions and calls `activate_skill`. Skipped automatically when the registry has exactly one skill.

**Phase 2 — Execution**: LLM runs with the selected skill's full instructions and only its `allowed-tools`.

---

## Conversational Agent with Skills

`Conversational` also accepts a `skills=` parameter:

```python
from flowgentra_ai.agent import Conversational

conv = Conversational(
    name="conv-assistant",
    llm=LLMClient(provider="mistral", model="mistral-small-latest"),
    skills=registry,
)
result = conv.execute_input("What is the derivative of x^3 + 2x?")
```

---

## API Reference

### `SkillRegistry`

```python
SkillRegistry(tool_registry: ToolRegistry | None = None)
```

| Method | Description |
|--------|-------------|
| `from_directory(path, tool_registry=None, allow_override=False)` | Load all skills from subdirs with `SKILL.md` |
| `load(path, allow_override=False)` | Load a single skill from its directory |
| `list()` | Return list of skill names |
| `get(name)` | Return `Skill` object |
| `build_menu()` | Phase 1 system prompt (names + descriptions only) |
| `build_system_prompt(skill_name)` | Phase 2 system prompt with full instructions |
| `resolve_tools(skill_name)` | Phase 2 scoped tool list |
| `get_callable(tool_name)` | Python callable for a skill-specific tool (from `scripts/tools.py`). Returns `None` for built-ins. |

### `SkillAgent`

```python
SkillAgent(name: str, llm: LLM, skills: SkillRegistry, retries: int = 3)
```

| Method | Description |
|--------|-------------|
| `execute_input(user_input)` | Run two-phase interaction, return response |
| `active_skills()` | Names of skills activated so far |

---

## Common Errors

**`ValueError: Skill 'X' lists tool 'Y' in allowed-tools but it could not be resolved`**

`Y` is a built-in tool but no `ToolRegistry` was provided. Fix:

```python
# Wrong — no tool_registry, built-ins invisible
registry = SkillRegistry.from_directory("skills/")

# Correct
registry = SkillRegistry.from_directory(
    "skills/",
    tool_registry=ToolRegistry.with_builtins(),
)

# Also correct for single skill
registry = SkillRegistry(tool_registry=ToolRegistry.with_builtins())
registry.load("skills/math-solver")
```

**`ValueError: SkillRegistry is empty`**

`from_directory` was pointed at the skill folder instead of its parent:

```python
# Wrong — math-solver/ has no subdirs with SKILL.md
registry = SkillRegistry.from_directory("skills/math-solver")

# Correct — use parent + .load() for a single skill
registry = SkillRegistry(tool_registry=ToolRegistry.with_builtins())
registry.load("skills/math-solver")
```
