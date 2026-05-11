# RAG API Reference

## Document

A document stored in a vector store.

```python
from flowgentra_ai.rag import Document
```

### Constructor

```python
Document(id: str, text: str, metadata: dict | None = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id` | `str` | required | Unique identifier for this document |
| `text` | `str` | required | Document text content |
| `metadata` | `dict \| None` | `None` | Optional metadata (e.g., `{"source": "wiki", "page": 3}`) |

```python
doc = Document("doc-1", "Rust is a systems programming language.", {"source": "wiki"})
```

### Properties

| Property | Type | Notes |
|----------|------|-------|
| `id` | `str` | Document ID |
| `text` | `str` | Document text (read/write) |
| `metadata` | `dict` | Metadata dict |
| `embedding` | `list[float] \| None` | Embedding vector (set after indexing) |

---

## SearchResult

A result from a vector store search.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Document ID |
| `text` | `str` | Document text |
| `score` | `float` | Similarity score (0.0–1.0, higher = more similar) |
| `metadata` | `dict` | Document metadata |

---

## TextChunk

A chunk from a text splitter.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `text` | `str` | Chunk text content |
| `metadata` | `ChunkMetadata` | Source file, chunk index, character offsets |

---

## Embeddings

Generate embedding vectors from text.

```python
from flowgentra_ai.rag import Embeddings
```

### Factory Methods

| Method | Description |
|--------|-------------|
| `Embeddings.mock(dimension)` | Hash-based mock embeddings — no API needed, good for testing |
| `Embeddings.openai(api_key, model="text-embedding-3-small")` | OpenAI embeddings |
| `Embeddings.openai_with_dimension(api_key, model, dimension)` | OpenAI with reduced dimension |
| `Embeddings.openai_cached(api_key, model="text-embedding-3-small")` | OpenAI with in-memory cache |
| `Embeddings.ollama(model, base_url=None, dimension=None)` | Ollama local model |
| `Embeddings.mistral(api_key, model=None)` | Mistral AI embeddings |
| `Embeddings.huggingface(model, api_key, endpoint=None, dimension=None)` | HuggingFace Inference API |

```python
# OpenAI (most common)
emb = Embeddings.openai("sk-...")

# Local with Ollama
emb = Embeddings.ollama("nomic-embed-text")

# Testing (no API)
emb = Embeddings.mock(dimension=128)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `embed(text)` | `list[float]` | Embed a single string |
| `embed_batch(texts)` | `list[list[float]]` | Embed multiple strings efficiently |
| `get_dimension()` | `int` | Embedding vector length (e.g., 1536 for text-embedding-3-small) |

---

## InMemoryVectorStore

In-memory vector store with cosine similarity search. Not persistent — data is lost on restart.

```python
from flowgentra_ai.rag import InMemoryVectorStore
```

### Constructor

```python
InMemoryVectorStore()
```

### Methods

#### `index(doc, embedding)` → `None`

Store a document with its embedding.

| Parameter | Type | Description |
|-----------|------|-------------|
| `doc` | `Document` | The document to store |
| `embedding` | `list[float]` | Pre-computed embedding vector |

#### `search(query_embedding, top_k=5, filter=None)` → `list[SearchResult]`

Find the most similar documents to a query vector.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query_embedding` | `list[float]` | required | Query vector |
| `top_k` | `int` | `5` | Number of results to return |
| `filter` | `dict \| None` | `None` | Metadata filter (only return docs whose metadata matches) |

```python
results = store.search(query_emb, top_k=10, filter={"source": "wiki"})
```

#### `get(doc_id)` → `Document`

Retrieve a document by ID.

#### `delete(doc_id)` → `None`

Remove a document by ID.

#### `list()` → `list[Document]`

Get all stored documents.

#### `clear()` → `None`

Remove all documents.

---

## ChromaStore

ChromaDB-backed vector store. Persistent across restarts. Requires `chromadb` package.

```python
from flowgentra_ai.rag import ChromaStore
```

### Constructor

```python
ChromaStore(
    collection_name: str,
    persist_directory: str | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | `str` | required | ChromaDB collection name |
| `persist_directory` | `str \| None` | `None` | Path to persist data. `None` = in-memory. |

```python
store = ChromaStore("my_docs", persist_directory="./chroma_db")
```

Has the same methods as `InMemoryVectorStore`.

---

## IngestionPipeline

Loads, splits, embeds, and indexes an entire directory of documents in one step.

```python
from flowgentra_ai.rag import IngestionPipeline
```

### Constructor

```python
IngestionPipeline(
    directory: str,
    embeddings: Embeddings,
    store: InMemoryVectorStore,
    chunk_size: int = 500,
    overlap: int = 50,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | `str` | required | Path to the directory with documents |
| `embeddings` | `Embeddings` | required | Embeddings provider |
| `store` | `InMemoryVectorStore` | required | Vector store to index into |
| `chunk_size` | `int` | `500` | Characters per chunk |
| `overlap` | `int` | `50` | Character overlap between chunks |

### Methods

#### `run()` → `IngestionStats`

Execute the full pipeline. Returns statistics when complete.

```python
stats = pipeline.run()
print(f"Indexed {stats.document_count} documents in {stats.chunk_count} chunks")
```

### IngestionStats Properties

| Property | Type | Description |
|----------|------|-------------|
| `document_count` | `int` | Total documents loaded |
| `chunk_count` | `int` | Total chunks indexed |

---

## RetrievalConfig

Configuration for the retrieval strategy.

```python
from flowgentra_ai.rag import RetrievalConfig
```

### Factory Methods

#### `RetrievalConfig.semantic(top_k=5, threshold=0.7)` → `RetrievalConfig`

Semantic similarity search only.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | `int` | `5` | Number of results |
| `threshold` | `float` | `0.7` | Minimum similarity score (0.0–1.0) |

#### `RetrievalConfig.hybrid(keyword_weight=0.3, top_k=5, threshold=0.7)` → `RetrievalConfig`

Combines semantic search with BM25 keyword matching.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keyword_weight` | `float` | `0.3` | Weight of keyword score (0.0 = pure semantic, 1.0 = pure keyword) |
| `top_k` | `int` | `5` | Number of results |
| `threshold` | `float` | `0.7` | Minimum score |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `top_k` | `int` | Number of results |
| `similarity_threshold` | `float` | Minimum score threshold |

---

## Retriever

Full retrieval pipeline: embed query → search → optional hybrid merge → optional rerank → optional dedup.

```python
from flowgentra_ai.rag import Retriever
```

### Constructor

```python
Retriever(store: InMemoryVectorStore, embeddings: Embeddings, config: RetrievalConfig)
```

### Methods

#### `retrieve(query)` → `list[SearchResult]`

Execute the full pipeline.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Natural language query |

```python
results = retriever.retrieve("What is Rust?")
for r in results:
    print(f"[{r.score:.2f}] {r.text}")
```

#### `with_dedup(threshold)` → `None`

Enable deduplication of near-identical results.

| Parameter | Type | Description |
|-----------|------|-------------|
| `threshold` | `float` | Similarity threshold above which results are considered duplicates (0.0–1.0) |

---

## PdfDocument

An extracted PDF document.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `source` | `str` | File path |
| `page_count` | `int` | Number of pages |
| `text` | `str` | Full extracted text |

---

## Free Functions

```python
from flowgentra_ai.rag import (
    chunk_text, chunk_text_by_tokens, estimate_tokens,
    extract_text, extract_pdf, extract_and_chunk,
    load_document, load_directory,
    bm25_score, hybrid_merge,
    dedup_by_id, dedup_by_similarity,
    decompose_query,
)
```

| Function | Returns | Description |
|----------|---------|-------------|
| `chunk_text(text, chunk_size, overlap=0)` | `list[str]` | Split text by character count |
| `chunk_text_by_tokens(text, max_tokens, overlap_tokens=0)` | `list[str]` | Split by token count |
| `estimate_tokens(text)` | `int` | Approximate token count |
| `extract_text(path)` | `str` | Extract text from a PDF |
| `extract_pdf(path)` | `PdfDocument` | Extract PDF as structured object |
| `extract_and_chunk(path, chunk_size=1000, overlap=200)` | `list[(str, str)]` | Extract + chunk in one call |
| `load_document(path)` | `LoadedDocument` | Load a file (PDF, text, markdown, JSON, CSV, HTML) |
| `load_directory(path)` | `list[LoadedDocument]` | Load all files in a directory |
| `bm25_score(query, documents)` | `list[float]` | BM25 keyword relevance scores |
| `hybrid_merge(results, query, keyword_weight=0.3)` | `list[SearchResult]` | Merge semantic + keyword scores |
| `dedup_by_id(results)` | `list[SearchResult]` | Remove duplicates by exact ID |
| `dedup_by_similarity(results, threshold=0.85)` | `list[SearchResult]` | Remove near-identical results |
| `decompose_query(query, max_depth=2)` | `list[str]` | Break compound queries into sub-queries |
