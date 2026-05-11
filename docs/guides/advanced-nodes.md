# Advanced Node Types

Flowgentra provides sophisticated node types for complex workflow patterns including retry logic, timeouts, branching, parallel execution, and subgraph composition.

## Retry Nodes

Automatically retry failed operations with configurable backoff strategies.

=== "Python"

    ```python
    from flowgentra_ai.nodes import RetryNode
    from flowgentra_ai.graph import StateGraph

    def unreliable_api_call(state):
        # Simulate occasional failures
        if random.random() < 0.3:
            raise Exception("API temporarily unavailable")
        state["result"] = "Success!"
        return state

    # Wrap with retry logic
    retry_node = RetryNode(
        max_retries=3,
        backoff_strategy="exponential",  # or "linear", "fixed"
        base_delay=1.0  # seconds
    )

    builder = StateGraph(MyState)
    builder.add_node("api_call", retry_node.wrap(unreliable_api_call))
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::nodes::RetryNode;

    let retry_node = RetryNode::new(3, BackoffStrategy::Exponential);
    graph.add_node("unreliable_task", retry_node.wrap(my_fallible_node))?;
    ```

**Configuration Options:**
- `max_retries`: Maximum retry attempts (default: 3)
- `backoff_strategy`: "exponential", "linear", or "fixed"
- `base_delay`: Initial delay between retries (default: 1.0 seconds)
- `max_delay`: Maximum delay cap (default: 60.0 seconds)

## Timeout Nodes

Protect against hanging operations with configurable timeouts.

=== "Python"

    ```python
    from flowgentra_ai.nodes import TimeoutNode

    def slow_operation(state):
        time.sleep(30)  # This might hang
        state["result"] = "Completed"
        return state

    # Add timeout protection
    timeout_node = TimeoutNode(timeout_seconds=10.0)

    builder.add_node("slow_task", timeout_node.wrap(slow_operation))
    ```

**When to use timeouts:**
- External API calls
- File I/O operations
- Long-running computations
- Network requests

## Branching and Conditional Logic

Execute different paths based on state conditions.

=== "Python"

    ```python
    from flowgentra_ai.nodes import BranchConfig
    from flowgentra_ai.types import Condition

    def classify_query(state):
        query = state["query"].lower()
        if "calculate" in query:
            state["category"] = "math"
        elif "search" in query:
            state["category"] = "search"
        else:
            state["category"] = "general"
        return state

    def math_handler(state):
        # Handle math queries
        return state

    def search_handler(state):
        # Handle search queries
        return state

    # Configure branching
    branch_config = BranchConfig(
        branches=[
            (Condition.field_equals("category", "math"), "math_handler"),
            (Condition.field_equals("category", "search"), "search_handler")
        ],
        default_branch="general_handler"
    )

    builder.add_node("classify", classify_query)
    builder.add_node("math", math_handler)
    builder.add_node("search", search_handler)
    builder.add_node("general", general_handler)
    builder.add_node("branch", branch_config)

    builder.add_edge("classify", "branch")
    # Branch node automatically routes to appropriate handler
    ```

## Parallel Execution

Run multiple nodes concurrently and merge results.

=== "Python"

    ```python
    from flowgentra_ai.nodes import ParallelNodeConfig, MergeStrategy

    # Define parallel tasks
    def analyze_sentiment(state):
        state["sentiment"] = analyze_text_sentiment(state["text"])
        return state

    def extract_keywords(state):
        state["keywords"] = extract_text_keywords(state["text"])
        return state

    def summarize_text(state):
        state["summary"] = summarize_text_content(state["text"])
        return state

    # Configure parallel execution
    parallel_config = ParallelNodeConfig(
        nodes=["sentiment_analysis", "keyword_extraction", "summarization"],
        merge_strategy=MergeStrategy.CONCATENATE  # or OVERWRITE, MERGE_MAP
    )

    builder.add_node("sentiment_analysis", analyze_sentiment)
    builder.add_node("keyword_extraction", extract_keywords)
    builder.add_node("summarization", summarize_text)
    builder.add_node("parallel_processor", parallel_config)

    # All three nodes run in parallel
    builder.add_edge("input_processor", "parallel_processor")
    ```

**Merge Strategies:**
- `CONCATENATE`: Combine list fields
- `OVERWRITE`: Last writer wins
- `MERGE_MAP`: Deep merge dictionaries
- `CUSTOM`: Provide custom merge function

## Loop Nodes

Repeat operations until a condition is met.

=== "Python"

    ```python
    from flowgentra_ai.nodes import LoopNodeConfig
    from flowgentra_ai.types import Condition

    def iterative_refinement(state):
        # Improve result iteratively
        current_score = evaluate_quality(state["result"])
        state["iterations"] = state.get("iterations", 0) + 1

        if current_score > 0.8 or state["iterations"] > 5:
            state["complete"] = True
        else:
            # Refine the result
            state["result"] = improve_result(state["result"])

        return state

    # Configure loop
    loop_config = LoopNodeConfig(
        body_node="refinement_step",
        exit_condition=Condition.field_equals("complete", True),
        max_iterations=10
    )

    builder.add_node("refinement_step", iterative_refinement)
    builder.add_node("quality_loop", loop_config)
    ```

## Subgraph Nodes

Compose complex workflows from smaller graphs.

=== "Python"

    ```python
    from flowgentra_ai.nodes import SubgraphNodeConfig

    # Define a subgraph for document processing
    doc_builder = StateGraph(DocumentState)
    doc_builder.add_node("load", load_document_node)
    doc_builder.add_node("chunk", chunk_document_node)
    doc_builder.add_node("embed", embed_chunks_node)
    doc_builder.set_entry_point("load")
    doc_builder.add_edge("load", "chunk")
    doc_builder.add_edge("chunk", "embed")

    document_processor = doc_builder.compile()

    # Use as a node in larger workflow
    subgraph_config = SubgraphNodeConfig(
        subgraph=document_processor,
        input_mapping={"document_path": "path"},  # Map outer state to inner
        output_mapping={"processed_chunks": "chunks"}  # Map inner results to outer
    )

    main_builder.add_node("process_documents", subgraph_config)
    ```

## Join Nodes

Synchronize multiple parallel branches.

=== "Python"

    ```python
    from flowgentra_ai.nodes import JoinNodeConfig, JoinType

    # After parallel processing, join results
    join_config = JoinNodeConfig(
        join_type=JoinType.ALL,  # Wait for all branches
        merge_strategy=MergeStrategy.MERGE_MAP
    )

    builder.add_node("join_results", join_config)

    # Connect parallel branches to join
    builder.add_edge("branch_a", "join_results")
    builder.add_edge("branch_b", "join_results")
    builder.add_edge("branch_c", "join_results")
    ```

**Join Types:**
- `ALL`: Wait for all incoming branches
- `ANY`: Continue when any branch completes
- `RACE`: Use result from first completed branch

## Error Handling Patterns

Combine nodes for robust error handling:

=== "Python"

    ```python
    from flowgentra_ai.nodes import RetryNode, TimeoutNode

    # Chain error handling strategies
    robust_node = RetryNode(
        max_retries=3,
        backoff_strategy="exponential"
    ).wrap(
        TimeoutNode(timeout_seconds=30.0).wrap(
            my_unreliable_operation
        )
    )

    builder.add_node("robust_operation", robust_node)
    ```

## Best Practices

### 1. Start Simple
Begin with basic nodes and add complexity gradually. Use advanced nodes only when needed.

### 2. Test Error Cases
Thoroughly test retry logic, timeouts, and error conditions.

### 3. Monitor Performance
Parallel execution and retries can impact performance. Monitor resource usage.

### 4. Clear State Management
Ensure state updates are predictable when using parallel execution and merging.

### 5. Use Appropriate Timeouts
Set realistic timeouts based on expected operation duration.

### 6. Handle Partial Failures
Design workflows to handle partial failures in parallel execution.

### 7. Document Complex Logic
Use comments and clear naming for complex node configurations.

## Common Patterns

### Circuit Breaker
```python
def circuit_breaker_wrapper(node_func, failure_threshold=5):
    failures = 0

    def wrapped(state):
        nonlocal failures
        if failures >= failure_threshold:
            raise Exception("Circuit breaker open")

        try:
            result = node_func(state)
            failures = 0  # Reset on success
            return result
        except Exception as e:
            failures += 1
            raise e

    return wrapped
```

### Progressive Retry
```python
# Start fast, then slow down
progressive_retry = RetryNode(
    max_retries=5,
    backoff_strategy="exponential",
    base_delay=0.1,  # Quick first retry
    max_delay=10.0   # Cap at 10 seconds
)
```

### Conditional Parallelism
```python
# Only run expensive operations if needed
def should_parallelize(state):
    return len(state["documents"]) > 10

parallel_config = ParallelNodeConfig(
    nodes=["expensive_analysis", "basic_processing"],
    condition=should_parallelize  # Only run in parallel when worthwhile
)
```</content>
<parameter name="filePath">c:\Users\OussamaBenHariz\Desktop\agentflow-rs\flowgentra-ai-py\docs\guides\nodes.md