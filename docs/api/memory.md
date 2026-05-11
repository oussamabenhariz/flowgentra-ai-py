# Memory API Reference

## ConversationMemory

In-memory conversation message storage. Each thread is isolated. Supports optional sliding window to cap the number of messages.

```python
from flowgentra_ai.memory import ConversationMemory
```

### Constructor

```python
ConversationMemory(max_messages: int | None = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_messages` | `int \| None` | `None` | Maximum messages per thread. Oldest messages are dropped when exceeded. `None` = unlimited. |

```python
mem = ConversationMemory()              # unlimited
mem = ConversationMemory(max_messages=50)  # keep last 50 messages per thread
```

### Methods

#### `add_message(thread_id, message)` → `None`

Add a message to a thread.

| Parameter | Type | Description |
|-----------|------|-------------|
| `thread_id` | `str` | Thread identifier (e.g., a user ID) |
| `message` | `Message` | The message to add |

```python
mem.add_message("user-alice", Message.user("Hello!"))
mem.add_message("user-alice", Message.assistant("Hi! How can I help?"))
```

#### `messages(thread_id, limit=None)` → `list[Message]`

Get messages for a thread, oldest first.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thread_id` | `str` | required | Thread identifier |
| `limit` | `int \| None` | `None` | Return only the last N messages |

```python
all_messages  = mem.messages("user-alice")
last_10       = mem.messages("user-alice", limit=10)
```

#### `clear(thread_id)` → `None`

Delete all messages for a thread.

| Parameter | Type | Description |
|-----------|------|-------------|
| `thread_id` | `str` | Thread to clear |

---

## TokenBufferMemory

Conversation memory with a token budget. When accumulated tokens exceed the budget, oldest messages are dropped.

```python
from flowgentra_ai.memory import TokenBufferMemory
```

### Constructor

```python
TokenBufferMemory(max_tokens: int)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_tokens` | `int` | Maximum token budget per thread |

```python
mem = TokenBufferMemory(max_tokens=2000)
```

### Methods

Same interface as `ConversationMemory`: `add_message`, `messages`, `clear`.

---

## SummaryMemory

Conversation memory that automatically summarizes old messages using an LLM. Unlike `TokenBufferMemory`, context is preserved (via the summary) rather than dropped.

```python
from flowgentra_ai.memory import SummaryMemory, SummaryConfig
```

### SummaryConfig

```python
SummaryConfig(
    llm_config: LLMConfig,
    summary_threshold: int = 20,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_config` | `LLMConfig` | required | LLM to use for summarization |
| `summary_threshold` | `int` | `20` | Summarize when history exceeds this many messages |

### SummaryMemory Constructor

```python
SummaryMemory(config: SummaryConfig)
```

```python
from flowgentra_ai.llm import LLMConfig

config = SummaryConfig(
    llm_config=LLMConfig("openai", "gpt-3.5-turbo", api_key="sk-..."),
    summary_threshold=25,
)
mem = SummaryMemory(config)
```

### Methods

Same interface as `ConversationMemory`: `add_message`, `messages`, `clear`.

---

## Checkpoint

A persisted snapshot of graph state at a point in execution.

```python
from flowgentra_ai.memory import Checkpoint
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `state` | `dict` | The full state dict at this checkpoint |
| `metadata` | `CheckpointMetadata` | Metadata about the checkpoint |

---

## CheckpointMetadata

Metadata associated with a checkpoint.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `last_node` | `str \| None` | Name of the last node that executed |
| `execution_path` | `list[str]` | Ordered list of all executed node names |
| `extra` | `dict` | Any additional metadata |

---

## FileCheckpointer

File-based checkpoint storage. Typically configured via `StateGraphBuilder.set_checkpointer()` rather than directly.

```python
from flowgentra_ai.memory import FileCheckpointer
```

The checkpointer serializes state to JSON files in the specified directory.

```python
# Usually you just do this on the builder:
builder.set_checkpointer("./checkpoints")

# Direct use (advanced):
checkpointer = FileCheckpointer("./checkpoints")
```
