# Evaluation & Metrics

Evaluate your RAG systems and agent workflows with comprehensive metrics and automated evaluation pipelines.

## Why Evaluate?

Evaluation helps you:
- Measure retrieval quality
- Assess generation accuracy
- Compare different configurations
- Monitor production performance
- Debug issues systematically

## Retrieval Metrics

### Hit Rate

Fraction of queries that retrieved at least one relevant document.

=== "Python"

    ```python
    from flowgentra_ai.evaluation import hit_rate

    # retrieval_results: List[List[SearchResult]]
    # ground_truth: List[List[str]] (document IDs)

    score = hit_rate(retrieval_results, ground_truth)
    print(f"Hit Rate: {score:.3f}")  # e.g., 0.85
    ```

**Interpretation:**
- 1.0 = Perfect (every query found relevant docs)
- 0.0 = Terrible (no queries found relevant docs)
- Good baseline: > 0.8 for most applications

### Mean Reciprocal Rank (MRR)

Measures how highly relevant documents are ranked.

=== "Python"

    ```python
    from flowgentra_ai.evaluation import mrr

    score = mrr(retrieval_results, ground_truth)
    print(f"MRR: {score:.3f}")  # e.g., 0.72
    ```

**Interpretation:**
- Rewards high rankings of relevant documents
- Perfect score = 1.0 (relevant doc always first)
- Good score: > 0.7 for information retrieval

### Normalized Discounted Cumulative Gain (NDCG)

Sophisticated ranking metric that considers position and relevance grades.

=== "Python"

    ```python
    from flowgentra_ai.evaluation import mean_ndcg

    # Evaluate at different cutoffs
    ndcg_5 = mean_ndcg(retrieval_results, ground_truth, k=5)
    ndcg_10 = mean_ndcg(retrieval_results, ground_truth, k=10)

    print(f"NDCG@5: {ndcg_5:.3f}")
    print(f"NDCG@10: {ndcg_10:.3f}")
    ```

**Interpretation:**
- Considers graded relevance (not just binary)
- Perfect = 1.0
- Good for comparing ranking quality

## Full Evaluation Pipeline

Run comprehensive evaluations with multiple metrics:

=== "Python"

    ```python
    from flowgentra_ai.evaluation import rag_evaluate, EvaluationConfig, GradingConfig

    # Configure evaluation
    config = EvaluationConfig(
        metrics=["hit_rate", "mrr", "ndcg@5", "ndcg@10"],
        grading_config=GradingConfig(
            temperature=0.1,  # Low temperature for consistent grading
            model="gpt-4"     # Use strong model for grading
        )
    )

    # Run evaluation
    results = rag_evaluate(
        queries=queries,
        results=retrieval_results,
        ground_truth=ground_truth,
        config=config
    )

    # Results summary
    print(f"Mean Hit Rate: {results.hit_rate:.3f}")
    print(f"Mean MRR: {results.mrr:.3f}")
    print(f"Mean NDCG@5: {results.ndcg_at_5:.3f}")

    # Per-query results
    for i, query_result in enumerate(results.query_results):
        print(f"Query {i}: HR={query_result.hit_rate}, MRR={query_result.mrr}")
    ```

## Generation Evaluation

Evaluate answer quality beyond retrieval:

=== "Python"

    ```python
    from flowgentra_ai.evaluation import evaluate_output_score

    # Evaluate generated answers
    scores = []
    for query, answer, context in zip(queries, answers, contexts):
        score = evaluate_output_score(
            query=query,
            output=answer,
            context=context,
            grading_config=GradingConfig(model="gpt-4")
        )
        scores.append(score)

    print(f"Average answer quality: {sum(scores)/len(scores):.3f}")
    ```

## Custom Metrics

Create your own evaluation metrics:

=== "Python"

    ```python
    from flowgentra_ai.evaluation import EvaluationResult

    def custom_metric(results, ground_truth):
        """Custom evaluation logic"""
        total_score = 0.0
        for retrieved, truth in zip(results, ground_truth):
            # Your custom scoring logic
            score = calculate_custom_score(retrieved, truth)
            total_score += score
        return total_score / len(results)

    # Use in evaluation
    config = EvaluationConfig(
        metrics=["hit_rate", "mrr"],  # Built-in metrics
        custom_metrics={"my_metric": custom_metric}
    )
    ```

## A/B Testing Configurations

Compare different RAG configurations:

=== "Python"

    ```python
    from flowgentra_ai.rag import RAGConfig
    from flowgentra_ai.evaluation import rag_evaluate

    # Configuration A: Basic setup
    config_a = RAGConfig(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000),
        embeddings=Embeddings.openai("text-embedding-3-small"),
        retriever=Retriever(top_k=5)
    )

    # Configuration B: Optimized setup
    config_b = RAGConfig(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, overlap=50),
        embeddings=Embeddings.openai("text-embedding-3-large"),
        retriever=Retriever(top_k=10, reranker=RRFReranker(k=60))
    )

    # Evaluate both
    results_a = rag_evaluate(queries, config_a.retrieve_all(queries), ground_truth)
    results_b = rag_evaluate(queries, config_b.retrieve_all(queries), ground_truth)

    print(f"Config A - Hit Rate: {results_a.hit_rate:.3f}")
    print(f"Config B - Hit Rate: {results_b.hit_rate:.3f}")
    ```

## Evaluation Best Practices

### Dataset Creation

1. **Diverse queries** - Cover different types of questions
2. **Realistic ground truth** - Use actual relevant documents
3. **Sufficient volume** - 100+ queries for reliable metrics
4. **Balanced difficulty** - Mix easy and hard queries

### Metric Selection

- **Hit Rate** - Simple effectiveness check
- **MRR** - Ranking quality for single relevant docs
- **NDCG** - Graded relevance and ranking quality
- **Custom metrics** - Domain-specific evaluation

### Statistical Significance

```python
# Use confidence intervals for reliable comparisons
import numpy as np
from scipy import stats

def confidence_interval(scores, confidence=0.95):
    mean = np.mean(scores)
    std = np.std(scores)
    n = len(scores)
    h = std * stats.t.ppf((1 + confidence) / 2, n - 1) / np.sqrt(n)
    return mean - h, mean + h

# Check if difference is significant
ci_a = confidence_interval(scores_a)
ci_b = confidence_interval(scores_b)

if ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]:
    print("Results are statistically different")
else:
    print("Results are not significantly different")
```

### Continuous Monitoring

Set up automated evaluation in production:

```python
# Evaluate on live traffic sample
def evaluate_live_traffic(sample_queries, sample_results):
    """Monitor production performance"""
    results = rag_evaluate(sample_queries, sample_results, ground_truth)
    return results.hit_rate, results.mrr

# Alert if performance drops
baseline_hit_rate = 0.85
current_hit_rate, _ = evaluate_live_traffic(live_queries, live_results)

if current_hit_rate < baseline_hit_rate * 0.9:  # 10% drop
    alert_team("RAG performance degraded!")
```

## Common Pitfalls

1. **Small test sets** - Need 100+ queries for reliable results
2. **Biased ground truth** - Ensure relevance judgments are accurate
3. **Over-optimization** - Don't tune too closely to test set
4. **Ignoring context** - Consider query type and user intent
5. **Single metrics** - Use multiple complementary metrics</content>
<parameter name="filePath">c:\Users\OussamaBenHariz\Desktop\agentflow-rs\flowgentra-ai-py\docs\guides\evaluation.md