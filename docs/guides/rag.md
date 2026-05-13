# RAG Pipeline

RAG (Retrieval-Augmented Generation) is the pattern of fetching relevant documents and injecting them into the LLM's context before answering. This lets you give your agent access to a private knowledge base without fine-tuning.

The Flowgentra RAG pipeline covers the full lifecycle: **split → embed → index → retrieve → rerank → answer**.

## Why RAG?

Traditional LLMs are trained on public data up to a certain point. They can't access:
- Your private documents
- Recent information after training
- Domain-specific knowledge

RAG solves this by retrieving relevant information from your knowledge base and adding it to the LLM's context.

---

## Full pipeline example

Here's a complete RAG system that loads documents, indexes them, and answers questions:

=== "Python"

    ```python
    from flowgentra_ai.rag import (
        Embeddings, InMemoryVectorStore, Retriever, RetrievalConfig,
        Document, chunk_text,
    )
    from flowgentra_ai.llm import LLM, Message
    from flowgentra_ai.graph import StateGraph, END
    from flowgentra_ai import State

    # 1. Set up embeddings (converts text to vectors)
    emb = Embeddings.openai("sk-...")

    # 2. Create vector store (stores documents + their vectors)
    store = InMemoryVectorStore()

    # 3. Index your documents
    docs = [
        "Rust is a systems programming language focused on safety and performance.",
        "Python is a high-level language known for simplicity and readability.",
        "Both Rust and Python can be used for AI and machine learning.",
    ]
    for i, text in enumerate(docs):
        embedding = emb.embed(text)  # Convert text to vector
        store.index(Document(f"doc-{i}", text), embedding)  # Store both

    # 4. Set up retriever (handles search + retrieval)
    retriever = Retriever(
        store,
        emb,
        RetrievalConfig.semantic(top_k=3, threshold=0.6),  # Get top 3 similar docs
    )

    # 5. LLM (generates answers)
    client = LLM(provider="openai", model="gpt-4o", api_key="sk-...")

    # 6. RAG graph (coordinates the pipeline)
    def retrieve_node(state):
        query = state["question"]
        results = retriever.retrieve(query)  # Find relevant docs
        context = "\n".join(r.text for r in results)  # Combine their text
        state["context"] = context
        return state

    def answer_node(state):
        response = client.chat([
            Message.system(f"Answer based on this context:\n{state['context']}"),
            Message.user(state["question"]),
        ])
        state["answer"] = response.content
        return state

    builder = StateGraph(dict)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer",   answer_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)
    graph = builder.compile()

    result = graph.invoke(State({"question": "What language is focused on safety?"}))
    print(result["answer"])  # Should mention Rust
    ```

This example shows the complete flow: documents → vectors → storage → retrieval → generation.

---

## Text splitting

Before embedding, split large documents into smaller chunks. Why?

- **Context limits**: LLMs have maximum context windows (e.g., 4K-128K tokens)
- **Relevance**: Smaller chunks are more likely to be entirely relevant
- **Cost**: Smaller chunks = fewer tokens = lower API costs

Flowgentra provides multiple splitting strategies:

=== "Python"

    ```python
    from flowgentra_ai.rag import (
        chunk_text,
        chunk_text_by_tokens,
        estimate_tokens,
        RecursiveCharacterTextSplitter,
        MarkdownTextSplitter,
        HTMLTextSplitter,
        TokenTextSplitter,
        CodeTextSplitter,
    )

    # Simple character-based splitting
    chunks = chunk_text("long text...", chunk_size=500, overlap=50)

    # Token-based splitting (more accurate for LLM limits)
    chunks = chunk_text_by_tokens("long text...", max_tokens=200, overlap_tokens=20)

    # Estimate token count (approximate)
    count = estimate_tokens("some text")

    # Smart splitters that respect content structure
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)
    chunks = splitter.split("long document text...")  # Tries paragraph → sentence → word boundaries

    splitter = MarkdownTextSplitter(chunk_size=500, overlap=50)
    chunks = splitter.split(markdown_content)  # Splits on headers/sections

    splitter = HTMLTextSplitter(chunk_size=500, overlap=50)
    chunks = splitter.split(html_content)  # Strips tags, respects structure

    splitter = TokenTextSplitter(max_tokens=200, overlap_tokens=20)
    chunks = splitter.split(text)  # Splits by token count

    splitter = CodeTextSplitter(chunk_size=500, overlap=50)
    chunks = splitter.split(source_code)  # Respects function/class boundaries
    ```

**Choosing chunk size:**
- Too small: Loses context, may miss important information
- Too large: May exceed context limits, less precise retrieval
- Good starting point: 500-1000 characters, 50-200 character overlap

---

## Embeddings

Embeddings convert text into numerical vectors that capture semantic meaning. Similar texts have similar vectors.

### Providers

Flowgentra supports multiple embedding providers:

=== "Python"

    ```python
    from flowgentra_ai.rag import Embeddings

    # OpenAI (recommended for quality) - costs money but very good
    emb = Embeddings.openai("sk-...", "text-embedding-3-small")

    # With custom dimension (reduce storage for speed)
    emb = Embeddings.openai_with_dimension("sk-...", "text-embedding-3-small", 256)

    # With caching (avoids re-embedding the same text)
    emb = Embeddings.openai_cached("sk-...")

    # Ollama (free, runs locally) - requires Ollama installed
    emb = Embeddings.ollama("nomic-embed-text")

    # Mistral AI
    emb = Embeddings.mistral("api-key-...")

    # HuggingFace (via their API)
    emb = Embeddings.huggingface(
        "sentence-transformers/all-MiniLM-L6-v2",
        api_key="hf_...",
    )

    # Mock (for testing - deterministic, no API calls)
    emb = Embeddings.mock(dimension=128)

    # Usage
    vector   = emb.embed("Hello world")          # Single vector
    vectors  = emb.embed_batch(["Hello", "World"])  # Multiple at once (faster)
    dim      = emb.get_dimension()               # Vector size (e.g., 1536)
    ```

**Choosing an embedding model:**
- **OpenAI text-embedding-3-small**: Good balance of quality/cost/speed
- **Local models (Ollama)**: Free, private, but slower and lower quality
- **Mock**: For testing your pipeline without API costs

=== "Rust"

    ```rust
    use flowgentra_ai::rag::{OpenAIEmbeddings, CachedEmbeddings, EmbeddingsProvider};

    let emb = OpenAIEmbeddings::new("sk-...", "text-embedding-3-small");
    let cached = CachedEmbeddings::new(emb);  // wrap with cache

    let vector = cached.embed("Hello world").await?;
    ```

---

## Vector stores

### InMemoryVectorStore

Fast, no dependencies, not persistent. Great for development and small datasets.

=== "Python"

    ```python
    from flowgentra_ai.rag import InMemoryVectorStore, Document

    store = InMemoryVectorStore()

    # Index
    doc = Document("doc-1", "Rust is a systems language.", metadata={"source": "wiki"})
    store.index(doc, emb.embed(doc.text))

    # Search
    query_vec = emb.embed("What is Rust?")
    results = store.search(query_vec, top_k=5)
    results = store.search(query_vec, top_k=5, filter={"source": "wiki"})  # with metadata filter

    # Manage
    doc    = store.get("doc-1")
    all    = store.list()
    store.delete("doc-1")
    store.clear()
    ```

### ChromaDB Store

Persistent vector store backed by ChromaDB. Survives restarts.

```python
from flowgentra_ai.rag import ChromaStore

store = ChromaStore(
    collection_name="my_docs",
    persist_directory="./chroma_db",  # None = in-memory
)
# Same API as InMemoryVectorStore
store.index(doc, embedding)
results = store.search(query_emb, top_k=5)
```

### External stores (Rust)

```rust
use flowgentra_ai::rag::{PineconeStore, QdrantStore, ChromaStore, VectorStore};

// Pinecone (cloud-hosted)
let store = PineconeStore::new("api-key", "index-name", "us-east-1").await?;

// Qdrant (self-hosted vector DB)
let store = QdrantStore::new("http://localhost:6333", "my_collection").await?;
```

**Choosing a vector store:**
- **InMemoryVectorStore**: Development, small datasets (< 10K docs)
- **ChromaDB**: Production, persistent, good performance
- **Pinecone/Qdrant**: Large scale, distributed, advanced features

---

## Retriever

The `Retriever` combines embedding, search, hybrid scoring, and deduplication into a single pipeline call.

=== "Python"

    ```python
    from flowgentra_ai.rag import Retriever, RetrievalConfig

    # Pure semantic search (vector similarity only)
    config = RetrievalConfig.semantic(top_k=5, threshold=0.7)
    retriever = Retriever(store, emb, config)

    # Hybrid search (semantic + keyword matching)
    config = RetrievalConfig.hybrid(keyword_weight=0.3, top_k=10, threshold=0.5)
    retriever = Retriever(store, emb, config)

    # Enable deduplication (removes near-identical results)
    retriever.with_dedup(threshold=0.85)

    # Retrieve relevant documents
    results = retriever.retrieve("What is Rust?")
    for r in results:
        print(f"[{r.score:.2f}] {r.text}")
    ```

**Retrieval strategies:**
- **Semantic**: Finds conceptually similar content
- **Hybrid**: Combines semantic with keyword matching (better for exact terms)
- **Threshold**: Filters out low-relevance results
- **Deduplication**: Removes redundant results

---

## Hybrid search utilities

=== "Python"

    ```python
    from flowgentra_ai.rag import bm25_score, hybrid_merge, dedup_by_id, dedup_by_similarity

    # BM25 keyword scores
    scores = bm25_score("rust language", ["Rust is fast", "Python is easy"])

    # Merge semantic results with keyword scores
    merged = hybrid_merge(semantic_results, "rust language", keyword_weight=0.3)

    # Deduplication
    unique = dedup_by_id(results)                     # by exact ID
    unique = dedup_by_similarity(results, threshold=0.85)  # by semantic similarity
    ```

---

## Document loaders

=== "Python"

    ```python
    from flowgentra_ai.rag import load_document, load_directory, extract_text, extract_pdf, extract_and_chunk

    # Load a single file (PDF, text, markdown, JSON, CSV, HTML)
    doc = load_document("report.pdf")

    # Load all files in a directory
    docs = load_directory("./knowledge_base/")

    # Extract text from PDF
    text = extract_text("document.pdf")

    # Extract PDF as object (with metadata)
    pdf = extract_pdf("document.pdf")
    print(pdf.source)      # "document.pdf"
    print(pdf.page_count)  # 42
    print(pdf.text)        # full extracted text

    # One-shot: extract + chunk
    chunks = extract_and_chunk("document.pdf", chunk_size=500, overlap=50)
    # Returns list of (chunk_id, chunk_text) tuples
    ```

---

## Ingestion pipeline

Load, split, embed, and index an entire directory in one call.

=== "Python"

    ```python
    from flowgentra_ai.rag import IngestionPipeline

    pipeline = IngestionPipeline(
        directory="./knowledge_base",
        embeddings=emb,
        store=store,
        chunk_size=500,
        overlap=50,
    )

    stats = pipeline.run()
    print(f"Loaded {stats.document_count} documents, {stats.chunk_count} chunks")
    # store is now ready for retrieval
    ```

---

## Rerankers

After retrieval, rerankers re-score results for better relevance.

=== "Python"

    ```python
    from flowgentra_ai.rerankers import NoopReranker, RRFReranker, CrossEncoderReranker, LLMReranker

    results = retriever.retrieve("my query")

    # No reranking (pass-through)
    reranker = NoopReranker()

    # Reciprocal Rank Fusion — fast, score-based
    reranker = RRFReranker()

    # Neural cross-encoder — slow but high quality
    reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LLM-based — uses your LLM to score relevance
    reranker = LLMReranker(client=client)

    reranked = reranker.rerank(results, query="my query")
    ```

**When to use reranking:**
- **Basic retrieval is good enough**: Skip reranking
- **Need higher precision**: Use cross-encoder or LLM reranker
- **Performance matters**: Use RRF (fastest)

---

## Query decomposition

Break compound queries into sub-queries for better recall.

=== "Python"

    ```python
    from flowgentra_ai.rag import decompose_query

    queries = decompose_query("What are Rust's safety and performance features?", max_depth=2)
    # ["What are Rust's safety and performance features?",
    #  "Rust safety features",
    #  "Rust performance features"]

    # Retrieve for each sub-query, then merge
    all_results = []
    for q in queries:
        all_results.extend(retriever.retrieve(q))
    unique = dedup_by_id(all_results)
    ```

---

## RAG evaluation

Measure how well your retrieval system works.

=== "Rust"

    ```rust
    use flowgentra_ai::rag::evaluate;

    let metrics = evaluate(&retrieval_results, &ground_truth).await?;
    println!("Hit Rate: {:.2}", metrics.hit_rate());
    println!("MRR:      {:.2}", metrics.mrr());
    println!("NDCG:     {:.2}", metrics.mean_ndcg());
    ```

**Key metrics:**
- **Hit Rate**: Fraction of queries that found at least one relevant document
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of first relevant result
- **NDCG**: Normalized Discounted Cumulative Gain — accounts for ranking quality

    let metrics = evaluate(&retrieval_results, &ground_truth).await?;
    println!("Hit Rate: {:.2}", metrics.hit_rate());
    println!("MRR:      {:.2}", metrics.mrr());
    println!("NDCG:     {:.2}", metrics.mean_ndcg());
    ```
