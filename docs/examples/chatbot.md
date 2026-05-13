# Example: Multi-User Chatbot

A production-ready chatbot that supports multiple users simultaneously, each with their own conversation history.

---

## What we're building

- Multi-turn conversation with memory
- Per-user thread isolation
- Token-bounded history (no runaway costs)
- Structured response with metadata

---

## Python

```python
# chatbot.py
from flowgentra_ai.graph import StateGraph, END
from flowgentra_ai.memory import ConversationMemory
from flowgentra_ai.llm import LLM, Message
from flowgentra_ai import State

# ── Setup ──────────────────────────────────────────────────────────────────────

client = (
    LLM(provider="openai", model="gpt-4o", api_key="sk-...")
    # Retry on transient failures, cache repeated identical questions
    .with_retry(max_retries=3).cached(max_entries=1000)
)

# 50 messages per user — oldest dropped when exceeded
memory = ConversationMemory(max_messages=50)

SYSTEM_PROMPT = """You are a helpful, friendly assistant.
Be concise — answer in 2–3 sentences unless the user asks for detail.
If you don't know something, say so."""

# ── Graph nodes ────────────────────────────────────────────────────────────────

def load_history(state):
    """Load this user's conversation history."""
    thread = state["thread_id"]
    history = memory.messages(thread)
    state["history"] = history
    return state

def call_llm(state):
    """Call the LLM with the full conversation context."""
    history  = state["history"]
    user_msg = Message.user(state["user_input"])

    messages = [Message.system(SYSTEM_PROMPT)] + history + [user_msg]
    response = client.chat(messages)

    state["reply"] = response.content
    state["user_message"]  = user_msg
    state["bot_message"]   = response
    return state

def save_history(state):
    """Persist this turn to memory."""
    thread = state["thread_id"]
    memory.add_message(thread, state["user_message"])
    memory.add_message(thread, state["bot_message"])
    return state

# ── Graph ──────────────────────────────────────────────────────────────────────

builder = StateGraph(dict)
builder.add_node("load",   load_history)
builder.add_node("llm",    call_llm)
builder.add_node("save",   save_history)
builder.set_entry_point("load")
builder.add_edge("load",  "llm")
builder.add_edge("llm",   "save")
builder.add_edge("save",  END)
graph = builder.compile()

# ── Public API ─────────────────────────────────────────────────────────────────

def chat(user_id: str, message: str) -> str:
    """Send a message and get a reply. Thread-safe."""
    result = graph.invoke(State({
        "thread_id":  user_id,
        "user_input": message,
    }))
    return result["reply"]

def clear_history(user_id: str) -> None:
    """Wipe a user's conversation history."""
    memory.clear(user_id)

# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Two users, completely isolated
    print(chat("alice", "Hi! My name is Alice and I love Rust."))
    print(chat("bob",   "Hi! I'm Bob and I work in Python."))

    print(chat("alice", "What language did I say I love?"))   # "Rust"
    print(chat("bob",   "What language do I work in?"))        # "Python"

    # Clear one user without affecting the other
    clear_history("alice")
    print(chat("alice", "What's my name?"))   # won't know — history was cleared
```

---

## Rust

```rust
// src/chatbot.rs
use flowgentra_ai::{StateGraph, DynState};
use flowgentra_ai::llm::{LLMConfig, LLM, Message};
use flowgentra_ai::memory::InMemoryConversationMemory;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct Chatbot {
    graph:  Arc<StateGraph>,
    memory: Arc<Mutex<InMemoryConversationMemory>>,
}

impl Chatbot {
    pub async fn new(api_key: &str) -> Self {
        let client = LLM::from_config(LLMConfig::openai("gpt-4", api_key))
            .with_retry(3)
            .cached(1000);

        let memory = Arc::new(Mutex::new(InMemoryConversationMemory::new(Some(50))));

        let client = Arc::new(client);
        let mem_clone = memory.clone();

        let graph = StateGraph::builder()
            .add_node("load", {
                let mem = mem_clone.clone();
                move |mut state: DynState| {
                    let mem = mem.clone();
                    async move {
                        let thread = state.get_string("thread_id").unwrap_or_default();
                        let history = mem.lock().await.get(&thread, None).await;
                        state.set("history", history);
                        Ok(state)
                    }
                }
            })
            .add_node("llm", {
                let client = client.clone();
                move |mut state: DynState| {
                    let client = client.clone();
                    async move {
                        let history: Vec<Message> = state.get_array("history").unwrap_or_default();
                        let input = state.get_string("user_input").unwrap_or_default();
                        let user_msg = Message::user(&input);

                        let mut messages = vec![Message::system(
                            "You are a helpful, concise assistant."
                        )];
                        messages.extend(history);
                        messages.push(user_msg.clone());

                        let response = client.chat(messages).await?;
                        state.set("reply", response.content.clone());
                        state.set("user_message", serde_json::to_value(&user_msg).unwrap());
                        state.set("bot_message", serde_json::to_value(&response).unwrap());
                        Ok(state)
                    }
                }
            })
            .add_node("save", {
                let mem = mem_clone;
                move |state: DynState| {
                    let mem = mem.clone();
                    async move {
                        let thread   = state.get_string("thread_id").unwrap_or_default();
                        let user_msg: Message = serde_json::from_value(state.get("user_message").unwrap().clone()).unwrap();
                        let bot_msg: Message  = serde_json::from_value(state.get("bot_message").unwrap().clone()).unwrap();
                        let mut mem = mem.lock().await;
                        mem.add(&thread, user_msg).await;
                        mem.add(&thread, bot_msg).await;
                        Ok(state)
                    }
                }
            })
            .entry("load")
            .edge("load", "llm")
            .edge("llm",  "save")
            .edge("save", "__end__")
            .build();

        Chatbot { graph: Arc::new(graph), memory }
    }

    pub async fn chat(&self, user_id: &str, message: &str) -> anyhow::Result<String> {
        let mut state = DynState::new();
        state.set("thread_id",  user_id);
        state.set("user_input", message);
        let result = self.graph.invoke(state).await?;
        Ok(result.get_string("reply").unwrap_or_default())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let bot = Chatbot::new("sk-...").await;

    println!("{}", bot.chat("alice", "Hi! My name is Alice.").await?);
    println!("{}", bot.chat("bob",   "I'm Bob.").await?);
    println!("{}", bot.chat("alice", "What's my name?").await?);   // "Alice"
    Ok(())
}
```

---

## What's happening

1. **`load_history`** — fetch this user's previous messages from `ConversationMemory`
2. **`call_llm`** — prepend the system prompt and history, then call the LLM
3. **`save_history`** — persist the new user + assistant messages to memory

The key insight: state carries everything needed for one turn (`thread_id`, `user_input`, `reply`). The memory object lives outside the graph and is shared across invocations.
