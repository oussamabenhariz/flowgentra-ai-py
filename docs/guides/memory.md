# Memory & Conversations

Memory in Flowgentra handles two distinct concerns:

- **Conversation memory** — storing and retrieving messages in chat sessions
- **Checkpointing** — persisting graph execution state for recovery and resumption

---

## Conversation Memory: Complete API

All memory types implement the same interface for adding and retrieving messages.

### ConversationMemory (In-Memory, Simple)

Basic in-memory message storage with optional message limit.

**Constructor:**
```python
mem = ConversationMemory(
    max_messages: Optional[int] = None
) -> ConversationMemory
```

**Parameters:**
- `max_messages` (Optional[int]): Max messages per thread (default: None = unlimited)

**Methods:**

```python
# Add a message
mem.add_message(thread_id: str, message: Message) -> None

# Retrieve messages
messages: List[Message] = mem.messages(
    thread_id: str,
    limit: Optional[int] = None
) -> List[Message]

# Get count
count: int = mem.message_count(thread_id: str) -> int

# Clear a thread
mem.clear(thread_id: str) -> None

# Clear all
mem.clear_all() -> None
```

**Token Cost:** Grows unbounded
**Best for:** Short sessions, < 50 messages

**Example:**
```python
from flowgentra_ai.memory import ConversationMemory
from flowgentra_ai.llm import Message

mem = ConversationMemory(max_messages=100)

# Store conversation
mem.add_message("user-alice", Message.user("Hello!"))
mem.add_message("user-alice", Message.assistant("Hi Alice!"))
mem.add_message("user-alice", Message.user("What is Rust?"))

# Retrieve with limit
recent = mem.messages("user-alice", limit=2)  # Last 2 messages

# Check size
count = mem.message_count("user-alice")  # 3

# Clean up
mem.clear("user-alice")
```

---

### TokenBufferMemory (Token Budget)

Maintains messages within a maximum token budget. When history exceeds the limit, oldest messages are dropped.

**Constructor:**
```python
mem = TokenBufferMemory(
    max_tokens: int,
    encoding: str = "cl100k_base"  # OpenAI encoding
) -> TokenBufferMemory
```

**Parameters:**
- `max_tokens` (int): Maximum tokens to store
- `encoding` (str): Token encoding model (default: "cl100k_base" for GPT-3.5/4)

**Methods:**

```python
# Add message (drops old ones if over budget)
mem.add_message(thread_id: str, message: Message) -> None

# Retrieve messages
messages: List[Message] = mem.messages(thread_id: str) -> List[Message]

# Get token usage
tokens_used: int = mem.token_count(thread_id: str) -> int
token_budget: int = mem.max_tokens() -> int

# Clear
mem.clear(thread_id: str) -> None
```

**Token Cost:** Capped at max_tokens
**Best for:** Cost sensitivity, recent messages matter most

**Example:**
```python
from flowgentra_ai.memory import TokenBufferMemory

# Keep budget to 2000 tokens
mem = TokenBufferMemory(max_tokens=2000)

# Add messages until budget exceeded
mem.add_message("thread-1", Message.user("Tell me about quantum computing... [long text]"))
mem.add_message("thread-1", Message.assistant("Quantum computing is... [long response]"))
# ... many more messages ...

# Oldest messages automatically dropped to stay within budget
messages = mem.messages("thread-1")
print(mem.token_count("thread-1"))  # ≤ 2000
```

---

### SummaryMemory (Semantic Compression)

Summarizes old messages with an LLM instead of dropping them. Keeps semantic content while saving tokens.

**Constructor:**
```python
mem = SummaryMemory(
    llm_config: LLMConfig,
    summary_threshold: int = 20,
    max_tokens_per_summary: int = 500
) -> SummaryMemory
```

**Parameters:**
- `llm_config` (LLMConfig): LLM for summarization (see llm.md)
- `summary_threshold` (int): Summarize when history exceeds N messages (default: 20)
- `max_tokens_per_summary` (int): Token limit per summary (default: 500)

**Methods:**

```python
# Add message (triggers summarization when threshold exceeded)
mem.add_message(thread_id: str, message: Message) -> None

# Retrieve (includes summaries for old messages)
messages: List[Message] = mem.messages(thread_id: str) -> List[Message]

# Get stats
stats: SummaryStats = mem.stats(thread_id: str) -> SummaryStats
# SummaryStats: message_count, summary_count, total_tokens

# Clear
mem.clear(thread_id: str) -> None
```

**Token Cost:** Capped (messages + summaries)
**Best for:** Long conversations, need full context

**Example:**
```python
from flowgentra_ai.memory import SummaryMemory
from flowgentra_ai.llm import LLMConfig, Message

llm_config = LLMConfig("openai", "gpt-3.5-turbo", api_key="sk-...")

mem = SummaryMemory(
    llm_config=llm_config,
    summary_threshold=20,
    max_tokens_per_summary=500
)

# Add many messages
for i in range(30):
    mem.add_message("thread-1", Message.user(f"Question {i}"))
    mem.add_message("thread-1", Message.assistant(f"Answer {i}"))

# Messages 0-10 are now summarized, 11-30 remain verbatim
messages = mem.messages("thread-1")
print(len(messages))  # ~21 (summary + recent messages)

# Inspect
stats = mem.stats("thread-1")
print(f"{stats.message_count} original, {stats.summary_count} summaries")
```

---

### Hybrid: TokenBufferMemory + SummaryMemory

Combine both for maximum efficiency:

```python
# First: Keep recent messages within token budget
buffer_mem = TokenBufferMemory(max_tokens=2000)

# Second: When buffer fills, summarize into a separate "archive"
summary_mem = SummaryMemory(llm_config=llm, summary_threshold=50)

def add_message_hybrid(thread_id, message):
    buffer_mem.add_message(thread_id, message)
    if buffer_mem.token_count(thread_id) > 1800:
        # Move old messages to summary
        summary_mem.add_message(thread_id, message)
```

---

## Memory Methods: Complete Reference

| Method | Signature | Returns | Purpose |
|--------|-----------|---------|---------|
| **Add** | `add_message(thread_id, msg)` | None | Store message |
| **Get** | `messages(thread_id, limit)` | List[Message] | Retrieve messages |
| **Get Recent** | `messages(thread_id, limit=5)` | List[Message] | Get last N messages |
| **Count** | `message_count(thread_id)` | int | Get total messages |
| **Tokens** | `token_count(thread_id)` | int | Token usage |
| **Clear** | `clear(thread_id)` | None | Clear thread |

---

## Using Memory in a Graph

Wire conversation memory into a chatbot node:

```python
from flowgentra_ai.graph import StateGraph, END
from flowgentra_ai.memory import ConversationMemory
from flowgentra_ai.llm import LLMConfig, LLM, Message
from flowgentra_ai import State

mem = ConversationMemory(max_messages=50)
client = LLM.from_config(LLMConfig("openai", "gpt-4", api_key="sk-..."))

def chat_node(state):
    thread_id = state["thread_id"]
    user_input = state["user_input"]
    
    # Add user message to memory
    user_msg = Message.user(user_input)
    mem.add_message(thread_id, user_msg)
    
    # Get full history for LLM context
    history = mem.messages(thread_id)
    
    # Call LLM with context
    response = client.chat([
        Message.system("You are a helpful assistant."),
        *history,
    ])
    
    # Store assistant response
    mem.add_message(thread_id, response)
    
    state["reply"] = response.content
    return state

# Build graph
builder = StateGraph()
builder.add_node("chat", chat_node)
builder.set_entry_point("chat")
builder.add_edge("chat", END)
graph = builder.compile()

# Use it
r1 = graph.invoke(State({
    "thread_id": "user-alice",
    "user_input": "My name is Alice."
}))

# Memory persists
r2 = graph.invoke(State({
    "thread_id": "user-alice",
    "user_input": "What's my name?"  # Agent remembers!
}))
print(r2["reply"])
```

---

## Checkpointing: Graph State Persistence

Checkpointing saves the entire graph state after each node execution. This enables:

- **Recovery:** Resume from crash
- **Human-in-the-loop:** Pause for review, resume with edits
- **Debugging:** Replay execution with modifications

### FileCheckpointer

```python
checkpointer = FileCheckpointer(
    checkpoint_dir: str,
    create_if_missing: bool = True
) -> FileCheckpointer
```

**Parameters:**
- `checkpoint_dir` (str): Directory to store checkpoints
- `create_if_missing` (bool): Auto-create directory (default: True)

**Example:**
```python
from flowgentra_ai.memory import FileCheckpointer
from flowgentra_ai.graph import StateGraph

# Create graph with checkpointing
checkpointer = FileCheckpointer("./my_checkpoints")

builder = StateGraph()
# ... add nodes ...
graph = builder.compile(checkpointer=checkpointer)

# Run with thread ID — state saved after each node
result = graph.invoke_with_thread("session-123", State({"data": "..."}))

# Resume (continues from last checkpoint)
result = graph.resume_thread("session-123")

# Resume with modification (e.g., human edit)
modified_state = State({"draft": "user-edited-version"})
result = graph.resume_thread_with_state("session-123", modified_state)
```

### InMemoryCheckpointer (Testing)

For testing without disk I/O:

```python
from flowgentra_ai.memory import InMemoryCheckpointer

checkpointer = InMemoryCheckpointer()
graph = builder.compile(checkpointer=checkpointer)
```

---

## Checkpoint Files: Structure

Each checkpoint file contains:

```json
{
  "thread_id": "session-123",
  "checkpoint_id": "1234567890",
  "node": "last_node_executed",
  "state": {
    "key1": "value1",
    "key2": "value2"
  },
  "execution_path": ["node_a", "node_b", "node_c"],
  "timestamp": "2024-01-15T10:30:45Z",
  "metadata": {}
}
```

**Fields:**
- `thread_id` (str): Unique conversation/thread identifier
- `checkpoint_id` (str): Checkpoint sequence number
- `node` (str): Last executed node name
- `state` (dict): Complete graph state
- `execution_path` (List[str]): Sequence of nodes executed
- `timestamp` (str): When checkpoint was created
- `metadata` (dict): Custom metadata

---

## Common Memory Patterns

### Pattern 1: Short Chats (ConversationMemory)
        return state

    builder = StateGraph()
    builder.add_node("chat", chat)
    builder.set_entry_point("chat")
    builder.add_edge("chat", END)
    graph = builder.compile()

    # Turn 1
    r1 = graph.invoke(State({"thread_id": "u1", "user_input": "My name is Alice."}))

    # Turn 2 — agent remembers Alice
    r2 = graph.invoke(State({"thread_id": "u1", "user_input": "What's my name?"}))
    print(r2["reply"])   # "Your name is Alice."
    ```

---

## Checkpointing (graph-level persistence)

Checkpointing saves the entire graph state to disk after each node. This lets you:

- Resume a graph after a crash
- Implement human-in-the-loop review (pause the graph and resume later)
- Debug by replaying an execution

=== "Python"

    ```python
    # Enable checkpointing
    builder.set_checkpointer("./checkpoints")
    graph = builder.compile()

    # Run with a thread ID — state is saved after each node
    result = graph.invoke_with_thread("session-abc", State({"input": "data"}))

    # Resume (picks up from where it left off)
    result = graph.resume("session-abc")

    # Resume with state modifications (e.g., after human edit)
    result = graph.resume_with_state("session-abc", State({"draft": "edited draft"}))
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::memory::FileCheckpointer;

    let graph = StateGraph::builder()
        .with_checkpointer(FileCheckpointer::new("./checkpoints"))
        // ... nodes
        .build();

    let result = graph.invoke_with_thread("session-abc", state).await?;
    let result = graph.resume("session-abc").await?;
    ```

### Checkpoint structure

Each checkpoint stores:

| Field | Description |
|-------|-------------|
| `state` | The full state dict at that point |
| `last_node` | Which node was last executed |
| `execution_path` | Full sequence of nodes executed |
| `extra` | Any metadata you attach |

### InMemoryCheckpointer (for testing)

=== "Rust"

    ```rust
    use flowgentra_ai::memory::InMemoryCheckpointer;

    let graph = StateGraph::builder()
        .with_checkpointer(InMemoryCheckpointer::new())
        .build();
    ```

---

## MemoryAwareAgent (high-level wrapper)

If you're building a multi-user chatbot, `MemoryAwareAgent` handles threading and memory management for you.

=== "Python"

    ```python
    from flowgentra_ai.agent import MemoryAwareAgent

    agent = MemoryAwareAgent.from_config("agent.yaml")

    # Each user_id gets isolated memory
    agent.set_thread_id("user_alice")
    r1 = agent.run_turn("Hello, my name is Alice.")
    r2 = agent.run_turn("What is my name?")   # "Alice"

    agent.set_thread_id("user_bob")
    r3 = agent.run_turn("Hi, I'm Bob.")       # fresh thread

    # Inspect memory usage
    stats = agent.memory_stats()
    print(f"{stats.message_count} total messages")
    print(f"{stats.user_messages} from user")
    print(f"{stats.assistant_messages} from assistant")
    print(f"~{stats.approximate_tokens} tokens")

    # Clear a user's memory
    agent.clear_memory()
    ```

---

## Common Memory Patterns

### Pattern 1: Short Chats

```python
from flowgentra_ai.memory import ConversationMemory

mem = ConversationMemory(max_messages=50)

def chat(state):
    mem.add_message(state["user"], Message.user(state["text"]))
    history = mem.messages(state["user"])
    response = llm.chat(history)
    mem.add_message(state["user"], response)
    return state
```

### Pattern 2: Cost-Sensitive

```python
from flowgentra_ai.memory import TokenBufferMemory

mem = TokenBufferMemory(max_tokens=2000)

def chat(state):
    mem.add_message(state["user"], Message.user(state["text"]))
    history = mem.messages(state["user"])  # Only recent
    response = llm.chat(history)
    mem.add_message(state["user"], response)
    return state
```

### Pattern 3: Long-Term Context

```python
from flowgentra_ai.memory import SummaryMemory

mem = SummaryMemory(
    llm_config=LLMConfig("openai", "gpt-3.5-turbo"),
    summary_threshold=30
)

def chat(state):
    mem.add_message(state["user"], Message.user(state["text"]))
    history = mem.messages(state["user"])  # Includes summaries
    response = llm.chat(history)
    mem.add_message(state["user"], response)
    return state
```

---

## Memory Best Practices

### 1. Choose Memory Type Wisely

✓ **ConversationMemory** for short chats
✓ **TokenBufferMemory** for cost control
✓ **SummaryMemory** for long conversations

### 2. Always Use Thread IDs

```python
# ✓ Good
graph.invoke_with_thread("user-alice", state)
graph.invoke_with_thread("user-bob", state)

# ✗ Bad - mixes conversations
graph.invoke_with_thread("user", state_alice)
graph.invoke_with_thread("user", state_bob)
```

### 3. Use Checkpointing in Production

```python
# ✓ Persists state — call set_checkpointer() before compile()
builder.set_checkpointer("./checkpoints")
graph = builder.compile()

# ✗ Lost on crash
graph = builder.compile()
```

### 4. Monitor Memory Usage

```python
count = mem.message_count(thread_id)
if count > 200:
    logger.warn("Memory growth detected")

tokens = mem.token_count(thread_id)
if tokens > max_tokens * 0.9:
    logger.warn("Approaching token limit")
```

### 5. Clean Up Old Threads

```python
def cleanup_old():
    for thread in mem.list_threads():
        last_active = mem.last_activity(thread)
        if (now() - last_active).days > 30:
            mem.clear(thread)  # Archive to DB first
```

---

## Memory Comparison

| Feature | Conversation | TokenBuffer | Summary |
|---------|--------------|-------------|---------|
| **Token Cost** | Grows | Fixed | Fixed |
| **Old Messages** | Kept | Dropped | Summarized |
| **Context Quality** | Perfect | Partial | Full |
| **Setup** | Simple | Simple | Medium |
| **For** | Short chats | Cost-sensitive | Long sessions |
