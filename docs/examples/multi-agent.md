# Example: Multi-Agent Research System

A supervisor that orchestrates three specialist agents to research a topic, analyze the data, and write a structured report.

---

## What we're building

- **Researcher** — searches and retrieves raw information
- **Analyst** — processes and structures the data
- **Writer** — produces a polished final report
- **Supervisor** — runs them in sequence, passes output between them

---

## Python

```python
# multi_agent.py
from flowgentra_ai.supervision import (
    Supervisor, SupervisorNodeConfig, OrchestrationStrategy,
)
from flowgentra_ai.graph import StateGraph, END
from flowgentra_ai.llm import LLMConfig, LLM, Message
from flowgentra_ai import State
import os

api_key = os.environ["OPENAI_API_KEY"]
client  = LLM.from_config(LLMConfig("openai", "gpt-4", api_key=api_key))

# ── Researcher ─────────────────────────────────────────────────────────────────

def researcher_node(state):
    topic = state.get("topic") or "unknown"
    response = client.chat([
        Message.system(
            "You are a research specialist. Search your knowledge and return raw factual "
            "information. Include key statistics, dates, and sources where known."
        ),
        Message.user(f"Research this topic comprehensively: {topic}"),
    ])
    state["raw_research"] = response.content
    state["research_done"] = True
    return state

researcher = StateGraph()
researcher.add_node("research", researcher_node)
researcher.set_entry_point("research")
researcher.add_edge("research", END)
researcher_graph = researcher.compile()

# ── Analyst ────────────────────────────────────────────────────────────────────

def analyst_node(state):
    raw = state.get("raw_research") or ""
    response = client.chat([
        Message.system(
            "You are a data analyst. Take raw research and structure it into "
            "clear sections: Key Facts, Statistics, Timeline, and Implications."
        ),
        Message.user(f"Analyze and structure this research:\n\n{raw}"),
    ])
    state["analysis"] = response.content
    state["analysis_done"] = True
    return state

analyst = StateGraph()
analyst.add_node("analyze", analyst_node)
analyst.set_entry_point("analyze")
analyst.add_edge("analyze", END)
analyst_graph = analyst.compile()

# ── Writer ─────────────────────────────────────────────────────────────────────

def writer_node(state):
    topic    = state.get("topic") or ""
    analysis = state.get("analysis") or ""
    response = client.chat([
        Message.system(
            "You are a technical writer. Transform structured analysis into a polished, "
            "well-written report with an executive summary, body, and conclusion. "
            "Use clear headings and professional language."
        ),
        Message.user(
            f"Write a comprehensive report on '{topic}' based on this analysis:\n\n{analysis}"
        ),
    ])
    state["report"] = response.content
    state["writing_done"] = True
    return state

writer = StateGraph()
writer.add_node("write", writer_node)
writer.set_entry_point("write")
writer.add_edge("write", END)
writer_graph = writer.compile()

# ── Supervisor: sequential pipeline ───────────────────────────────────────────

config = SupervisorNodeConfig(
    "research-pipeline",
    children=["researcher", "analyst", "writer"],
    strategy=OrchestrationStrategy.sequential(),
)
config.set_fail_fast(True)     # abort if any agent fails
config.set_child_timeout_ms(60_000)  # 60s per agent

sup = Supervisor.from_config(config)
sup.add_agent("researcher", researcher_graph)
sup.add_agent("analyst",    analyst_graph)
sup.add_agent("writer",     writer_graph)

# ── Run it ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    topic  = "The impact of large language models on software development in 2024"
    result = sup.run(State({"topic": topic}))

    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["report"])
```

---

## Parallel variant (independent tasks)

When the tasks don't depend on each other, run them in parallel for speed:

```python
# Run all three at once and merge results
config = SupervisorNodeConfig(
    "parallel-research",
    children=["rust_agent", "python_agent", "go_agent"],
    strategy=OrchestrationStrategy.parallel(),
)
config.set_child_timeout_ms(30_000)
config.set_max_concurrent(3)

sup = Supervisor.from_config(config)
sup.add_agent("rust_agent",   rust_expert_graph)
sup.add_agent("python_agent", python_expert_graph)
sup.add_agent("go_agent",     go_expert_graph)

result = sup.run(State({"question": "What are the best use cases for each language?"}))
```

---

## Rust

```rust
// src/multi_agent.rs
use flowgentra_ai::{StateGraph, DynState};
use flowgentra_ai::agents::{Supervisor, SupervisorConfig, OrchestrationStrategy};
use flowgentra_ai::llm::{LLMConfig, LLM, Message};
use std::sync::Arc;

async fn build_researcher(client: Arc<LLM>) -> StateGraph {
    StateGraph::builder()
        .add_node("research", move |mut state: DynState| {
            let client = client.clone();
            async move {
                let topic    = state.get_string("topic").unwrap_or_default();
                let response = client.chat(vec![
                    Message::system("You are a research specialist. Return factual information."),
                    Message::user(&format!("Research: {topic}")),
                ]).await?;
                state.set("raw_research",   response.content);
                state.set("research_done", true);
                Ok(state)
            }
        })
        .entry("research")
        .edge("research", "__end__")
        .build()
}

async fn build_analyst(client: Arc<LLM>) -> StateGraph {
    StateGraph::builder()
        .add_node("analyze", move |mut state: DynState| {
            let client = client.clone();
            async move {
                let raw = state.get_string("raw_research").unwrap_or_default();
                let response = client.chat(vec![
                    Message::system("Structure raw research into Key Facts, Statistics, Timeline, Implications."),
                    Message::user(&raw),
                ]).await?;
                state.set("analysis",      response.content);
                state.set("analysis_done", true);
                Ok(state)
            }
        })
        .entry("analyze")
        .edge("analyze", "__end__")
        .build()
}

async fn build_writer(client: Arc<LLM>) -> StateGraph {
    StateGraph::builder()
        .add_node("write", move |mut state: DynState| {
            let client = client.clone();
            async move {
                let topic    = state.get_string("topic").unwrap_or_default();
                let analysis = state.get_string("analysis").unwrap_or_default();
                let response = client.chat(vec![
                    Message::system("Write a polished report with executive summary and conclusion."),
                    Message::user(&format!("Report on '{topic}':\n\n{analysis}")),
                ]).await?;
                state.set("report",       response.content);
                state.set("writing_done", true);
                Ok(state)
            }
        })
        .entry("write")
        .edge("write", "__end__")
        .build()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Arc::new(LLM::from_config(
        LLMConfig::openai("gpt-4", &std::env::var("OPENAI_API_KEY")?)
    ));

    let sup = Supervisor::from_config(SupervisorConfig {
        name: "pipeline".to_string(),
        children: vec!["researcher".to_string(), "analyst".to_string(), "writer".to_string()],
        strategy: OrchestrationStrategy::Sequential,
        fail_fast: true,
        child_timeout_ms: Some(60_000),
        ..Default::default()
    })
    .add_agent("researcher", build_researcher(client.clone()).await)
    .add_agent("analyst",    build_analyst(client.clone()).await)
    .add_agent("writer",     build_writer(client.clone()).await);

    let mut state = DynState::new();
    state.set("topic", "The impact of LLMs on software development in 2024");

    let result = sup.run(state).await?;
    println!("{}", result.get_string("report").unwrap_or_default());

    Ok(())
}
```

---

## What's happening

1. **Sequential strategy** — the supervisor calls `researcher` first, waits for it to finish, then calls `analyst` with the enriched state, then `writer`
2. **State flows through** — each agent receives the full state, including outputs from previous agents (`raw_research` → `analysis` → `report`)
3. **fail_fast** — if any agent errors, the pipeline stops immediately instead of producing a partial result
4. **Timeouts** — each agent has 60 seconds, after which an error is raised (use `on_timeout="skip"` if you want to continue)
