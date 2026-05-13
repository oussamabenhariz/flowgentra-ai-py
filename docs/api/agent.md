# Agent API Reference

## Agent

Config-driven agent. Loads its graph structure from a YAML file and auto-discovers handler functions by name.

```python
from flowgentra_ai.agent import Agent
```

### Class Methods

#### `Agent.from_config_path(path)` → `Agent`

Load an agent from a YAML configuration file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to the YAML config file |

```python
agent = Agent.from_config_path("agent.yaml")
```

#### `Agent.from_config(config)` → `Agent`

Load from an already-constructed `AgentConfig` object.

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `AgentConfig` | Pre-loaded config |

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Agent name from config |
| `config` | `AgentConfig` | The full configuration |
| `state` | `State` | Current state (after `run()`) |

---

### Methods

#### `run()` → `State`

Execute the agent from the start. Returns the final state.

```python
agent.set_state("query", "What is Rust?")
result = agent.run()
print(result["response"])
```

#### `run_with_thread(thread_id)` → `State`

Execute with checkpointing. State is persisted to disk — calling this again with the same `thread_id` resumes from where it left off.

| Parameter | Type | Description |
|-----------|------|-------------|
| `thread_id` | `str` | Unique ID for this session |

```python
result = agent.run_with_thread("session-abc")
```

#### `set_state(key, value)` → `None`

Set an initial state value before running.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | State key |
| `value` | `Any` | Value (must be JSON-serializable) |

---

## AgentConfig

Configuration loaded from a YAML file.

```python
from flowgentra_ai.agent import AgentConfig
```

### Class Methods

#### `AgentConfig.from_file(path)` → `AgentConfig`

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to YAML file |

#### `AgentConfig.from_yaml(yaml_str)` → `AgentConfig`

| Parameter | Type | Description |
|-----------|------|-------------|
| `yaml_str` | `str` | YAML content as a string |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Agent name |
| `description` | `str \| None` | Optional description |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `validate()` | `None` | Validate the config; raises on error |
| `to_json()` | `str` | Serialize to JSON |

---

## Typed Agent Constructors

Build a prebuilt agent using one of the typed constructor classes. Each class uses the same signature:

```python
AgentClass(
    name: str,
    llm: LLM,
    system_prompt: str = "",
    tools: list[ToolSpec] = [],
    retries: int = 3,
    memory_steps: int | None = None,
)
```

Available classes: `ZeroShotReAct`, `FewShotReAct`, `Conversational`, `ToolCalling`,
`StructuredChat`, `SelfAskWithSearch`, `ReactDocstore`.

```python
from flowgentra_ai.agent import ZeroShotReAct, ToolSpec
from flowgentra_ai.llm import LLM

agent = ZeroShotReAct(
    name="assistant",
    llm=LLM(provider="openai", model="gpt-4o", temperature=0.2),
    system_prompt="You are a helpful assistant.",
    tools=[my_tool],
    retries=3,
)
answer = agent.execute_input("What is the population of Japan?")
```

### FewShotReAct example

```python
from flowgentra_ai.agent import FewShotReAct
from flowgentra_ai.llm import LLM

agent = FewShotReAct(
    name="specialist",
    llm=LLM(provider="openai", model="gpt-4o"),
    system_prompt="Example 1: ...\nExample 2: ...",
    tools=[data_tool],
    memory_steps=5,
)
```

### Conversational example

```python
from flowgentra_ai.agent import Conversational
from flowgentra_ai.llm import LLM

agent = Conversational(
    name="chatbot",
    llm=LLM(provider="openai", model="gpt-4o"),
    memory_steps=20,
)
r1 = agent.execute_input("My name is Alice.")
r2 = agent.execute_input("What's my name?")  # "Alice"
```

---

## GraphBasedAgent

A config-driven agent. Created by `Agent.from_config_path()`.

### Methods

#### `execute_input(input)` → `str`

Run the agent with a text input. Returns the final text response.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `str` | The user's input text |

```python
answer = agent.execute_input("What is 17 * 8?")
print(answer)  # "136"
```

#### `node_names()` → `list[str]`

Returns the names of all nodes in the underlying graph. Useful for debugging.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Agent name |

---

## ToolSpec

Describes a tool that a prebuilt agent can call. You define the name, description, and parameter schema — Flowgentra passes this to the LLM so it knows when and how to call the tool.

```python
from flowgentra_ai.agent import ToolSpec
```

### Constructor

```python
ToolSpec(name: str, description: str)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tool name (used by LLM to call it) |
| `description` | `str` | Plain-English description of what the tool does |

### Methods

#### `add_parameter(name, param_type)` → `None`

Add a parameter to the tool's input schema.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Parameter name |
| `param_type` | `str` | JSON type: `"string"`, `"number"`, `"integer"`, `"boolean"`, `"array"`, `"object"` |

#### `set_required(param_name)` → `None`

Mark a parameter as required. The LLM will always include it.

| Parameter | Type | Description |
|-----------|------|-------------|
| `param_name` | `str` | Parameter name to mark required |

```python
tool = ToolSpec("calculator", "Perform arithmetic calculations")
tool.add_parameter("operation", "string")   # "add", "subtract", etc.
tool.add_parameter("a", "number")
tool.add_parameter("b", "number")
tool.set_required("operation")
tool.set_required("a")
tool.set_required("b")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Tool name |
| `description` | `str` | Tool description |

---

## MemoryAwareAgent

High-level agent wrapper with automatic per-user memory management. Each thread ID gets its own isolated conversation history.

```python
from flowgentra_ai.agent import MemoryAwareAgent
```

### Class Methods

#### `MemoryAwareAgent.from_config(path)` → `MemoryAwareAgent`

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to YAML config file |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `set_thread_id(thread_id)` | `None` | Switch to this user's conversation thread |
| `thread_id()` | `str` | Get the currently active thread ID |
| `run_turn(input)` | `str` | Run one conversation turn, returns agent response |
| `clear_memory()` | `None` | Clear conversation history for the current thread |
| `memory_stats()` | `MemoryStats` | Get memory usage stats for the current thread |

```python
agent = MemoryAwareAgent.from_config("agent.yaml")
agent.set_thread_id("user_alice")
r1 = agent.run_turn("Hello, I'm Alice.")
r2 = agent.run_turn("What's my name?")   # "Alice"
stats = agent.memory_stats()
```

---

## MemoryStats

Memory usage statistics for a thread.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `message_count` | `int` | Total messages stored |
| `user_messages` | `int` | Number of user messages |
| `assistant_messages` | `int` | Number of assistant messages |
| `approximate_tokens` | `int` | Approximate token count across all messages |

---

## StateField

Defines a field in the state schema for config-driven agents.

```python
from flowgentra_ai.agent import StateField
```

### Constructor

```python
StateField(field_type: str, description: str)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `field_type` | `str` | Type of the field (`"string"`, `"number"`, etc.) |
| `description` | `str` | Human-readable description |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `field_type` | `str` | Field type |
| `description` | `str` | Field description |
