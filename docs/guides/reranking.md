# Reranking

Reranking improves search result quality by reordering retrieved documents. Flowgentra provides multiple reranking strategies optimized for different use cases.

## Why Rerank?

Initial retrieval (like vector similarity search) is fast but may not capture all relevance signals. Reranking applies more sophisticated scoring to improve result quality.

## Available Rerankers

### Reciprocal Rank Fusion (RRF)

Combines multiple ranking sources mathematically. Excellent for hybrid search (semantic + keyword).

=== "Python"

    ```python
    from flowgentra_ai.rerankers import RRFReranker

    # Create RRF reranker
    reranker = RRFReranker(k=60)  # k controls influence of original ranking

    # Rerank results
    reranked_results = reranker.rerank(search_results)

    # Results are reordered by combined score
    for result in reranked_results:
        print(f"Score: {result.score:.3f}, Text: {result.text[:50]}...")
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::reranking::RRFReranker;

    let reranker = RRFReranker::new(60);
    let reranked = reranker.rerank(search_results)?;
    ```

**When to use RRF:**
- Hybrid search (combining semantic and keyword results)
- Multiple retrieval sources
- Need fast, deterministic reranking

### Cross-Encoder Reranking

Uses a transformer model to score query-document pairs directly. More accurate but slower.

=== "Python"

    ```python
    from flowgentra_ai.rerankers import CrossEncoderReranker

    # Use pre-trained cross-encoder model
    reranker = CrossEncoderReranker(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Rerank with higher quality scores
    reranked = reranker.rerank(search_results, query="What is machine learning?")

    # Much slower but more accurate than RRF
    ```

**When to use Cross-Encoder:**
- Maximum accuracy needed
- Small result sets (<100 documents)
- Can afford 10-100ms per query latency

### LLM-Based Reranking

Use an LLM to evaluate document relevance. Most flexible but slowest.

=== "Python"

    ```python
    from flowgentra_ai.rerankers import LLMReranker
    from flowgentra_ai.llm import LLM

    # Configure LLM for reranking
    llm = LLM(provider="openai", model="gpt-4o", temperature=0.1)

    reranker = LLMReranker(
        llm=llm,
        prompt_template="""
        Rate how relevant this document is to the query on a scale of 0-10.
        Query: {query}
        Document: {document}
        Score (0-10): """
    )

    reranked = reranker.rerank(search_results, query="machine learning basics")
    ```

**When to use LLM reranking:**
- Complex relevance criteria
- Need explainable rankings
- Custom scoring logic required

### No-Op Reranker

Pass-through reranker for when you don't want reranking.

=== "Python"

    ```python
    from flowgentra_ai.rerankers import NoopReranker

    # No reranking - preserves original order
    reranker = NoopReranker()
    results = reranker.rerank(search_results)  # Returns unchanged
    ```

## Integration with RAG

Rerankers integrate seamlessly with the RAG pipeline:

=== "Python"

    ```python
    from flowgentra_ai.rag import RAGConfig, Retriever
    from flowgentra_ai.rerankers import RRFReranker

    # Configure RAG with reranking
    rag_config = RAGConfig(
        retriever=Retriever(
            vectorstore=vectorstore,
            reranker=RRFReranker(k=60),  # Add reranking
            top_k=20  # Retrieve more, then rerank
        )
    )

    # Retrieval now includes reranking
    results = rag_config.retriever.retrieve("What is AI?", top_k=5)
    ```

## Performance Comparison

| Reranker | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| RRF | ⚡ Fast | ⭐ Good | Hybrid search, production |
| Cross-Encoder | 🐌 Slow | ⭐⭐⭐ Excellent | Quality-critical |
| LLM | 🐌🐌 Very Slow | ⭐⭐⭐⭐ Custom | Complex criteria |
| No-Op | ⚡ Instant | ❌ None | Testing, baseline |

## Configuration Tips

### Choosing RRF Parameters

```python
# Conservative (preserves original ranking more)
reranker = RRFReranker(k=100)

# Aggressive (more influence from fusion)
reranker = RRFReranker(k=20)
```

### Cross-Encoder Models

```python
# Fast but less accurate
reranker = CrossEncoderReranker("cross-encoder/ms-marco-TinyBERT-L-2-v2")

# Balanced speed/accuracy
reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

# High accuracy (slower)
reranker = CrossEncoderReranker("cross-encoder/ms-marco-electra-base")
```

### LLM Prompt Engineering

```python
# More detailed scoring criteria
prompt = """
Rate relevance 0-10 considering:
- Direct answer to query
- Related concepts
- Background context

Query: {query}
Document: {document}
Score: """

reranker = LLMReranker(llm=llm, prompt_template=prompt)
```

## Best Practices

1. **Start with RRF** - Fast and effective for most use cases
2. **Use cross-encoders** - When accuracy is critical and latency allows
3. **Retrieve more than needed** - Rerank from larger candidate set (20-50 docs)
4. **Tune thresholds** - Set minimum scores to filter irrelevant results
5. **Monitor performance** - Track reranking impact on your metrics
6. **Consider cost** - LLM reranking can be expensive at scale</content>
<parameter name="filePath">c:\Users\OussamaBenHariz\Desktop\agentflow-rs\flowgentra-ai-py\docs\guides\rerankers.md