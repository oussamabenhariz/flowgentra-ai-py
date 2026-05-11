# Best Practices

Patterns that work well in production Flowgentra deployments.

---

## State design

**Keep state flat.** Nested dicts are harder to read and update. Prefer `state["user_name"]` over `state["user"]["name"]`.

**Use descriptive key names.** `state["llm_response"]` is clearer than `state["res"]`. You'll thank yourself when debugging a trace.

**Don't store large objects in state.** State is serialized to JSON for checkpointing. Keep it to strings, numbers, and small lists. Store large data (embeddings, binary files) outside state and reference them by ID.

**Document your state schema.** Add a comment at the top of your graph file explaining what keys each node reads and writes. This becomes essential as graphs grow.

```python
# State schema:
# "question"     (str)  — user's input question
# "context_docs" (list) — retrieved documents
# "answer"       (str)  — final LLM response
# "sources"      (list) — source file paths
```

---

## Graph design

**One node, one responsibility.** If a node does three things, split it into three nodes. Smaller nodes are easier to retry, test, and replace.

**Name nodes as verbs.** `fetch_data`, `classify_intent`, `generate_response` — not `node_1`, `handler`, `step_3`.

**Make routers explicit.** Your conditional edge router should be a named function, not an inline lambda. Give it a docstring explaining the routing logic.

```python
def quality_router(state):
    """Route to 'retry' if score is below threshold, else 'finalize'."""
    if (state.get("score") or 0) < 0.8:
        return "retry"
    return "finalize"
```

**Always set `max_steps`.** The default (1000) is high. For production agents, set it to a reasonable value (50–100) to catch runaway loops early.

---

## LLM calls

**Don't call the LLM in a loop without a termination condition.** ReAct agents must have a maximum step count. Set it via `builder.set_max_steps(n)`.

**Use low temperature for tool-calling and reasoning.** `temperature=0.0–0.3` produces more reliable function calls. Use `0.7+` only for creative writing tasks.

**Add retries to all LLMs.** Networks fail. Rate limits happen. `client.with_retry(max_retries=3)` costs almost nothing and prevents a lot of pain.

**Use caching in development.** `client.cached()` prevents re-running the same prompts while you're iterating. Disable it in production if you need fresh responses.

**Check token usage.** Use `client.chat_with_usage()` to track costs during development. It's easy to accidentally build a workflow that costs $5 per run.

---

## RAG

**Chunk size matters.** Too small (< 200 chars) → each chunk lacks context. Too large (> 1000 chars) → chunks contain irrelevant content. Start at 400–600 chars.

**Always use overlap.** A 50–100 char overlap between chunks prevents splitting a sentence mid-thought.

**Match embedding model with your content type.** `text-embedding-3-small` works for most text. For code, consider a code-specific model.

**Hybrid search > pure semantic for most cases.** Pure semantic can miss exact keyword matches. Add 20–40% BM25 weight as a baseline.

**Test retrieval quality independently.** Before building the full RAG pipeline, manually test that `retriever.retrieve(question)` returns the right chunks. If retrieval is wrong, the LLM can't fix it.

---

## Memory

**Use `ConversationMemory` with a cap.** Unlimited memory is dangerous — 100+ messages will exceed the LLM's context window. Set `max_messages=50` as a safe default.

**`SummaryMemory` for long sessions.** If users have conversations that run for hours, `TokenBufferMemory` will drop important early context. `SummaryMemory` preserves it in compressed form.

**Don't store memory in state.** Conversation history shouldn't be in the graph state. Keep it in a `ConversationMemory` object that lives outside the graph and is loaded at the start of each turn.

---

## Error handling

**Use state for errors, not exceptions.** In a graph, raising an exception terminates the run. Use a state key (`state["error"]`) so the graph can route to an error-handling node.

```python
def risky_node(state):
    try:
        state["result"] = do_something()
    except Exception as e:
        state["error"] = str(e)
    return state

def router(state):
    return "error_node" if state.get("error") else "next_node"
```

**Use retry nodes for flaky operations.** `builder.add_retry_node(...)` is cleaner than wrapping every node function in try/except with sleep.

**Log at the start and end of each node.** During debugging, knowing which node succeeded and which failed is invaluable. Even a simple `print(f"[fetch] done, got {len(state['data'])} records")` helps.

---

## Testing

**Test nodes in isolation.** Each node is just a function. Create a test state, call the node, check the output state. No graph needed.

```python
def test_fetch_node():
    state = State({"query": "test query"})
    result = fetch_node(state)
    assert "data" in result
    assert result["data"] is not None
```

**Use mock embeddings and LLMs in tests.** `Embeddings.mock(128)` and a cached LLM prevent expensive API calls during CI.

**Test the router logic separately.** Routers are where bugs hide. Test every branch of every router function with explicit state examples.

---

## Production deployment

**Enable checkpointing.** Even if you don't need human-in-the-loop, checkpointing gives you crash recovery for free.

**Set child timeouts on supervisors.** Without timeouts, a hung agent will block the entire supervisor indefinitely.

**Monitor token usage.** Set up alerting on `TokenUsage.total_tokens` so you know when a workflow starts consuming dramatically more tokens than expected.

**Use `init_tracing("info")` in production.** The structured logs from the engine include node names, durations, and state keys — exactly what you need for debugging production incidents.
