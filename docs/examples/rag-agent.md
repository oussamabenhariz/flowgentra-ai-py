# Example: RAG Agent

A knowledge-base Q&A agent that retrieves relevant documents before answering. Uses the full RAG pipeline: split → embed → index → retrieve → answer.

---

## What we're building

- PDF ingestion and chunking
- Vector indexing with OpenAI embeddings
- Hybrid search (semantic + keyword)
- LLM answer generation with retrieved context
- Source attribution in the response

---

## Python

```python
# rag_agent.py
import os
from flowgentra_ai.graph import StateGraph, END
from flowgentra_ai.rag import (
    Embeddings, InMemoryVectorStore, Retriever, RetrievalConfig,
    Document, extract_and_chunk, load_directory,
)
from flowgentra_ai.llm import LLM, Message
from flowgentra_ai import State

# ── Build the knowledge base ───────────────────────────────────────────────────

def build_index(docs_dir: str) -> tuple:
    """Load documents, chunk them, and index them."""
    emb   = Embeddings.openai(os.environ["OPENAI_API_KEY"])
    store = InMemoryVectorStore()

    loaded_docs = load_directory(docs_dir)
    print(f"Loaded {len(loaded_docs)} documents")

    chunk_count = 0
    for loaded in loaded_docs:
        # Split into ~500-char chunks with 50-char overlap
        chunks = loaded.text.split("\n\n")   # simple paragraph split
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            doc_id  = f"{loaded.source}::{i}"
            doc     = Document(doc_id, chunk.strip(), {"source": loaded.source})
            embedding = emb.embed(chunk)
            store.index(doc, embedding)
            chunk_count += 1

    print(f"Indexed {chunk_count} chunks")

    retriever = Retriever(
        store, emb,
        RetrievalConfig.hybrid(keyword_weight=0.3, top_k=5, threshold=0.5),
    )
    retriever.with_dedup(threshold=0.85)
    return store, emb, retriever

# ── Graph nodes ────────────────────────────────────────────────────────────────

def retrieve(state):
    """Find relevant chunks for the question."""
    retriever = state["_retriever"]   # injected at graph build time
    question  = state["question"]

    results = retriever.retrieve(question)
    state["context_docs"] = [
        {"text": r.text, "source": r.metadata.get("source", ""), "score": r.score}
        for r in results
    ]
    return state

def generate_answer(state):
    """Generate an answer grounded in the retrieved context."""
    client   = state["_client"]
    question = state["question"]
    docs     = state["context_docs"]

    if not docs:
        state["answer"]  = "I couldn't find relevant information in the knowledge base."
        state["sources"] = []
        return state

    context = "\n\n".join(
        f"[Source: {d['source']} | Score: {d['score']:.2f}]\n{d['text']}"
        for d in docs
    )

    response = client.chat([
        Message.system(f"""You are a knowledgeable assistant. Answer questions using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Always cite your sources.

Context:
{context}"""),
        Message.user(question),
    ])

    state["answer"]  = response.content
    state["sources"] = list({d["source"] for d in docs})
    return state

# ── Build the graph ────────────────────────────────────────────────────────────

def build_rag_graph(retriever, client):
    """Build a RAG graph with the retriever and client injected."""
    def retrieve_node(state):
        state["_retriever"] = retriever
        return retrieve(state)

    def answer_node(state):
        state["_client"] = client
        return generate_answer(state)

    builder = StateGraph(dict)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer",   answer_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer",   END)
    return builder.compile()

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    api_key = os.environ["OPENAI_API_KEY"]
    client  = LLM(provider="openai", model="gpt-4o", api_key=api_key)

    # Index your docs (this takes a while the first time)
    store, emb, retriever = build_index("./knowledge_base/")

    graph = build_rag_graph(retriever, client)

    # Ask questions
    questions = [
        "What is Rust's ownership model?",
        "How does Python's GIL work?",
        "What are the key differences between Rust and C++?",
    ]

    for q in questions:
        result = graph.invoke(State({"question": q}))
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")
        print("-" * 60)
```

---

## Rust

```rust
// src/rag_agent.rs
use flowgentra_ai::{StateGraph, DynState};
use flowgentra_ai::rag::{
    OpenAIEmbeddings, InMemoryVectorStore, Retriever, RetrievalConfig,
    Document, load_directory, EmbeddingsProvider,
};
use flowgentra_ai::llm::{LLMConfig, LLM, Message};
use std::sync::Arc;

async fn build_index(docs_dir: &str, api_key: &str) -> (Arc<InMemoryVectorStore>, Retriever) {
    let emb   = Arc::new(OpenAIEmbeddings::new(api_key, "text-embedding-3-small"));
    let store = Arc::new(InMemoryVectorStore::new());

    let docs = load_directory(docs_dir).await.unwrap();
    println!("Loaded {} documents", docs.len());

    for doc in &docs {
        let chunks: Vec<&str> = doc.text.split("\n\n").collect();
        for (i, chunk) in chunks.iter().enumerate() {
            if chunk.trim().is_empty() { continue; }
            let doc_id    = format!("{}::{}", doc.source, i);
            let embedding = emb.embed(chunk.trim()).await.unwrap();
            store.index(
                Document::new(&doc_id, chunk.trim()).with_metadata("source", &doc.source),
                embedding,
            ).await.unwrap();
        }
    }

    let retriever = Retriever::new(
        store.clone(),
        emb.clone(),
        RetrievalConfig::hybrid(0.3, 5, 0.5),
    ).with_dedup(0.85);

    (store, retriever)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let client  = Arc::new(LLM::from_config(LLMConfig::openai("gpt-4", &api_key)));

    let (_, retriever) = build_index("./knowledge_base/", &api_key).await;
    let retriever = Arc::new(retriever);

    let graph = StateGraph::builder()
        .add_node("retrieve", {
            let retriever = retriever.clone();
            move |mut state: DynState| {
                let retriever = retriever.clone();
                async move {
                    let question = state.get_string("question").unwrap_or_default();
                    let results  = retriever.retrieve(&question).await?;
                    let context: Vec<serde_json::Value> = results.iter().map(|r| {
                        serde_json::json!({
                            "text":   r.text,
                            "source": r.metadata.get("source").cloned().unwrap_or_default(),
                            "score":  r.score,
                        })
                    }).collect();
                    state.set("context_docs", context);
                    Ok(state)
                }
            }
        })
        .add_node("answer", {
            let client = client.clone();
            move |mut state: DynState| {
                let client = client.clone();
                async move {
                    let question = state.get_string("question").unwrap_or_default();
                    let docs: Vec<serde_json::Value> = state.get_array("context_docs").unwrap_or_default();

                    let context = docs.iter()
                        .map(|d| format!("[{}] {}", d["source"].as_str().unwrap_or(""), d["text"].as_str().unwrap_or("")))
                        .collect::<Vec<_>>()
                        .join("\n\n");

                    let response = client.chat(vec![
                        Message::system(&format!("Answer using only this context:\n\n{context}")),
                        Message::user(&question),
                    ]).await?;

                    state.set("answer", response.content);
                    Ok(state)
                }
            }
        })
        .entry("retrieve")
        .edge("retrieve", "answer")
        .edge("answer",   "__end__")
        .build();

    let questions = [
        "What is Rust's ownership model?",
        "How does Python's GIL work?",
    ];

    for q in &questions {
        let mut state = DynState::new();
        state.set("question", *q);
        let result = graph.invoke(state).await?;
        println!("\nQ: {q}");
        println!("A: {}", result.get_string("answer").unwrap_or_default());
    }

    Ok(())
}
```

---

## Key decisions explained

**Why hybrid search?** Pure semantic search misses exact keyword matches (e.g., "GIL" won't match semantically if the docs use "Global Interpreter Lock"). Hybrid gives you the best of both.

**Why deduplication?** When you ask about "Rust ownership", many similar chunks may match. Deduplication at 0.85 similarity removes near-identical results so the LLM doesn't see the same content twice.

**Why inject the retriever through state?** It keeps the node functions pure (no global state) and makes them easier to test in isolation — just pass a mock state.

**Source attribution** is important for trust. Always include where the answer came from.
