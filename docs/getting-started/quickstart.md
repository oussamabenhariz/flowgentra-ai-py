# Quick Start

Build a real, working agent workflow in under 5 minutes.

---

## Python { #python }

### Step 1 — Install

```bash
pip install flowgentra-ai
```

### Step 2 — Your first graph

A graph has three parts: **nodes** (functions), **edges** (connections), and **state** (shared data).

```python
from flowgentra_ai.graph import StateGraph, END
from flowgentra_ai import State

# Nodes are plain functions
def greet(state):
    name = state["name"]
    state["message"] = f"Hello, {name}!"
    return state

def shout(state):
    state["message"] = state["message"].upper()
    return state

# Wire them together
builder = StateGraph(dict)
builder.add_node("greet", greet)
builder.add_node("shout", shout)
builder.set_entry_point("greet")
builder.add_edge("greet", "shout")
builder.add_edge("shout", END)
graph = builder.compile()

# Run it
result = graph.invoke(State({"name": "Alice"}))
print(result["message"])   # "HELLO, ALICE!"
```

### Step 3 — Add an LLM

```python
from flowgentra_ai.llm import LLM, Message

client = LLM(provider="openai", model="gpt-4o", api_key="sk-...")

def ask_llm(state):
    response = client.chat([
        Message.system("You are a helpful assistant."),
        Message.user(state["question"]),
    ])
    state["answer"] = response.content
    return state

builder = StateGraph(dict)
builder.add_node("ask", ask_llm)
builder.set_entry_point("ask")
builder.add_edge("ask", END)
graph = builder.compile()

result = graph.invoke(State({"question": "What is Rust in one sentence?"}))
print(result["answer"])
```

### Step 4 — Add conditional routing

Route to different nodes based on state:

```python
def classify(state):
    text = state["input"].lower()
    state["is_question"] = "?" in text
    return state

def answer_question(state):
    state["output"] = "Great question! The answer is..."
    return state

def make_statement(state):
    state["output"] = "Interesting! Tell me more."
    return state

def router(state):
    return "answer_question" if state["is_question"] else "make_statement"

builder = StateGraph(dict)
builder.add_node("classify", classify)
builder.add_node("answer_question", answer_question)
builder.add_node("make_statement", make_statement)
builder.set_entry_point("classify")
builder.add_conditional_edge("classify", router)
builder.add_edge("answer_question", END)
builder.add_edge("make_statement", END)
graph = builder.compile()

result = graph.invoke(State({"input": "What is the capital of France?"}))
print(result["output"])   # "Great question! The answer is..."
```

### Step 5 — Use a prebuilt agent

For common patterns (ReAct, conversational), use a typed agent class:

```python
from flowgentra_ai.agent import ZeroShotReAct, ToolSpec
from flowgentra_ai.llm import LLM

calc = ToolSpec("calculator", "Perform arithmetic")
calc.add_parameter("expression", "string")
calc.set_required("expression")

agent = ZeroShotReAct(
    name="math-assistant",
    llm=LLM(provider="openai", model="gpt-4o"),
    system_prompt="You are a math assistant.",
    tools=[calc],
)

print(agent.execute_input("What is 17 * 8?"))
```

---

## Rust { #rust }

### Step 1 — Create a project

```bash
cargo new my-agent
cd my-agent
```

Add to `Cargo.toml`:

```toml
[dependencies]
flowgentra-ai = { version = "0.1", features = ["macros"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1"
```

### Step 2 — Your first graph

```rust
use flowgentra_ai::{StateGraph, DynState};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let graph = StateGraph::builder()
        .add_node("greet", |mut state: DynState| async move {
            let name = state.get_string("name").unwrap_or("World".to_string());
            state.set("message", format!("Hello, {name}!"));
            Ok(state)
        })
        .add_node("shout", |mut state: DynState| async move {
            let msg = state.get_string("message").unwrap_or_default();
            state.set("message", msg.to_uppercase());
            Ok(state)
        })
        .entry("greet")
        .edge("greet", "shout")
        .edge("shout", "__end__")
        .build();

    let mut initial = DynState::new();
    initial.set("name", "Alice");

    let result = graph.invoke(initial).await?;
    println!("{}", result.get_string("message").unwrap());
    // "HELLO, ALICE!"
    Ok(())
}
```

### Step 3 — Add an LLM

```rust
use flowgentra_ai::llm::{LLMConfig, LLM, Message};

let client = LLM::from_config(LLMConfig::openai("gpt-4", "sk-..."));

let graph = StateGraph::builder()
    .add_node("ask", move |mut state: DynState| {
        let client = client.clone();
        async move {
            let question = state.get_string("question").unwrap_or_default();
            let response = client.chat(vec![
                Message::system("You are a helpful assistant."),
                Message::user(question),
            ]).await?;
            state.set("answer", response.content);
            Ok(state)
        }
    })
    .entry("ask")
    .edge("ask", "__end__")
    .build();
```

### Step 4 — Add conditional routing

```rust
let graph = StateGraph::builder()
    .add_node("classify", |mut state: DynState| async move {
        let input = state.get_string("input").unwrap_or_default();
        state.set("is_question", input.contains('?'));
        Ok(state)
    })
    .add_node("answer_question", |mut state: DynState| async move {
        state.set("output", "Great question! The answer is...");
        Ok(state)
    })
    .add_node("make_statement", |mut state: DynState| async move {
        state.set("output", "Interesting! Tell me more.");
        Ok(state)
    })
    .entry("classify")
    .conditional_edge("classify", |state: &DynState| {
        if state.get_bool("is_question").unwrap_or(false) {
            "answer_question"
        } else {
            "make_statement"
        }
    })
    .edge("answer_question", "__end__")
    .edge("make_statement", "__end__")
    .build();
```

### Step 5 — Use a prebuilt agent

```rust
use flowgentra_ai::agents::{AgentBuilder, AgentType};

let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_llm_config("gpt-4", "sk-...")
    .with_system_prompt("You are a math assistant.")
    .build()
    .await?;

let answer = agent.execute("What is 17 * 8?").await?;
println!("{answer}");
```

---

## What's next?

You've seen the basics. Here's where to go depending on what you want to build:

| Goal | Guide |
|------|-------|
| Understand the fundamentals | [Core Concepts](../core-concepts/what-is-flowgentra.md) |
| Build a chatbot with memory | [Memory & Conversations](../guides/memory.md) |
| Give your agent tools to use | [Tools](../guides/tools.md) |
| Search documents (RAG) | [RAG Pipeline](../guides/rag.md) |
| Orchestrate multiple agents | [Supervisor](../guides/supervisor.md) |
| Add human review steps | [Human-in-the-Loop](../guides/human-in-the-loop.md) |
| Full API reference | [API Reference](../api/state.md) |
| Complete examples | [Examples](../examples/chatbot.md) |
