# FAQ & Troubleshooting

---

## General

**What's the difference between Flowgentra and LangChain/LangGraph?**

LangGraph is the closest comparison. Key differences:
- Flowgentra's execution engine is written in Rust — significantly faster for CPU-bound operations and concurrent workloads
- Flowgentra has a native Python API (via PyO3) rather than a pure-Python reimplementation
- Both use the same graph-based model with state, nodes, and edges
- LangGraph has a larger ecosystem of integrations; Flowgentra is newer

**Do I need to know Rust to use the Python library?**

No. The Python library is a complete, standalone package. You install it with `pip` and write Python. You only need Rust if you're building from source or contributing to the library itself.

**Can I use both Rust and Python in the same project?**

Yes. The Rust engine is used by both, but you can't directly share graphs between them at runtime. You can, however, use them side-by-side: run performance-critical parts in Rust and prototyping/orchestration in Python.

---

## Installation

**`pip install flowgentra-ai` fails with "no matching distribution found"**

Check your Python version: `python --version`. Flowgentra requires Python 3.9+. If you're on an older version or an unsupported platform (e.g., 32-bit Linux), you'll need to build from source.

**`maturin develop` fails with "linker not found"**

You need a C linker. On Ubuntu/Debian: `sudo apt install build-essential`. On macOS: `xcode-select --install`.

**ImportError: `flowgentra_ai._native` not found**

The native extension wasn't compiled. Run `maturin develop` again from the `flowgentra-ai-py` directory.

---

## Graphs

**My graph runs forever / hits max_steps**

Your graph has a loop with no exit condition. Check that:
1. Your conditional edge router can return `"__end__"` (or use `END`)
2. The condition that breaks the loop can actually be reached
3. You've set `builder.set_max_steps(n)` to a reasonable value

**`KeyError: "my_key"` inside a node**

You used `state["my_key"]` but the key wasn't set. Use `state.get("my_key")` (returns `None` instead of raising) or check `"my_key" in state` first.

**"Unknown node: xyz" error when building the graph**

Your router returned a node name that doesn't exist. Common causes:
- Typo in the node name
- The node was added to the builder but with a different name
- The router returns a string that varies at runtime — check all possible return values

**Graph compiles but never terminates (no error)**

You have a cycle with no exit. Make sure at least one path from every node eventually reaches `END` / `"__end__"`.

---

## LLM

**LLM calls fail with rate limit errors**

Add retries: `client.with_retry(max_retries=5)`. The retry client uses exponential backoff by default.

**Responses are inconsistent / not following instructions**

1. Lower the temperature (try `0.1–0.3` for structured tasks)
2. Check your system prompt — it should be clear and specific
3. For structured output, use `ResponseFormat.json_schema(...)` instead of asking "respond in JSON"

**Tool calls are malformed / arguments are missing**

Make sure your `ToolDefinition.parameters` has the correct JSON Schema, including `"required"` for mandatory fields. The LLM ignores schema constraints it doesn't know about.

**`chat_with_tools` returns a message without tool calls**

The LLM decided it doesn't need the tools. This is expected behavior — it might answer directly. Check `response.has_tool_calls()` before processing tool calls.

---

## Memory & Checkpointing

**How do I clear a user's conversation history?**

```python
memory.clear("user-thread-id")
```

**Checkpoints are growing unboundedly on disk**

Implement periodic cleanup. Delete old checkpoint files from the checkpoint directory based on age or thread ID. The checkpoint format is plain JSON files — you can safely delete them.

**`graph.resume("thread-1")` raises "thread not found"**

The thread was never started with `invoke_with_thread("thread-1", ...)`, or the checkpoint directory was changed/deleted.

---

## RAG

**Retrieval quality is poor — wrong chunks are returned**

1. Check your chunk size — too large includes irrelevant content, too small loses context
2. Try hybrid search: `RetrievalConfig.hybrid(keyword_weight=0.3)`
3. Lower the `threshold` — maybe your similarity scores are just lower than expected
4. Use `with_dedup()` to remove near-duplicate results

**Embeddings are slow**

Use `Embeddings.openai_cached(api_key)` to cache results. If the bottleneck is the API, use `embed_batch` instead of individual `embed()` calls.

**Out of memory when indexing large document sets**

Use a persistent vector store (`ChromaStore`) instead of `InMemoryVectorStore`. Also make sure you're chunking documents before embedding — don't try to embed a 100-page PDF as a single text.

---

## Multi-Agent

**A supervisor child agent times out**

Increase `set_child_timeout_ms(ms)`. For LLM-heavy agents, 60 seconds is often not enough on slow models. Alternatively, use `OrchestrationStrategy.retry_fallback()` to have a backup agent.

**Parallel supervisor: state from one child overwrites another**

Switch from `ParallelMergeStrategy.latest()` to `ParallelMergeStrategy.deep_merge()`. Or design your agents to write to different state keys so there's no collision.

---

## Rust-specific

**"cannot infer type for type parameter `S`" when building a StateGraph**

Annotate the state type explicitly:
```rust
let graph = StateGraph::<MyState>::builder()...
// or
let graph: StateGraph<DynState> = StateGraph::builder()...
```

**Handler registered with `#[register_handler]` isn't found**

Make sure the module containing the handler is loaded before calling `from_config_path()`. In Rust, `#[register_handler]` uses a startup initializer — all handlers must be in modules that are compiled and linked.

**Async closures in nodes give type errors**

The pattern for capturing mutable state in async closures:
```rust
.add_node("name", {
    let captured_value = value.clone();
    move |mut state: DynState| {
        let captured_value = captured_value.clone();
        async move {
            // use captured_value here
            Ok(state)
        }
    }
})
```
The double `clone()` is necessary because the outer closure (`move`) must be `Fn` (called multiple times), so inner data must be cloned each invocation.

---

## Still stuck?

- Check the [examples](examples/chatbot.md) for complete working code
- Open an issue at [github.com/oussamabenhariz/FlowgentraAI](https://github.com/oussamabenhariz/FlowgentraAI/issues)
- Enable `init_tracing("debug")` and check the logs — they often reveal the issue immediately
