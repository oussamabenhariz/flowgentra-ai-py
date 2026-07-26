# Changelog

All notable changes to the `flowgentra-ai` Python package. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows SemVer
(0.x: minor bumps may break).

## [0.3.4] - Unreleased

Tracks `flowgentra-ai` 0.3.4. Capability + DX batch aimed at LangGraph/LangChain
parity. No breaking changes.

### Added
- **Multimodal (vision) messages** — `Message.user(content, images=[...])` where
  each image is a URL / `data:` URI string or a `{"url", "detail"}` dict; plus
  `message.with_image(url, detail=None)` and a `message.images` accessor.
  Serialized for OpenAI/Anthropic automatically.
- **Typed structured output** — `llm.with_structured_output(target)` returns a
  wrapper whose `.invoke(messages)` yields output conforming to `target`, which
  may be a JSON Schema dict, a Pydantic model class (returns a validated
  instance), or a TypedDict. Mirrors LangChain.
- **Node context argument** — a node function may declare a second parameter
  (`def node(state, ctx)`) to receive a `NodeContext` exposing `resume_value`
  (so `Command(resume=)` reaches Python nodes) and `node_name`. Single-argument
  `(state)` nodes are unchanged.
- **More graph checkpointers** — `builder.set_mysql_checkpointer(url)`,
  `set_redis_checkpointer(url, ttl_secs=None)`, and
  `set_mongo_checkpointer(url, db, collection)`, joining file/SQLite/Postgres.
- **`MockLLM.when(predicate)`** — a Python predicate over the message history
  (previously Rust-only).
- **`agent.subscribe_events()`** — stream node/edge/LLM execution events from a
  config-driven agent (subscribe before `run()`, drain during/after).
- **`flowgentra_ai.langgraph_compat`** — a compatibility shim (StateGraph with a
  `START` sentinel, `compile(checkpointer=...)`, `invoke(state, config=...)`,
  `MemorySaver`, `interrupt`) so most LangGraph code runs by changing the import
  line. See the "Migrating from LangGraph" guide.

## [0.3.3] - 2026-07-22

Tracks `flowgentra-ai` 0.3.3. Human-in-the-loop, checkpointing, and subgraphs
reach parity with LangGraph's core primitives, plus ergonomics features aimed
at closing the day-to-day UX gap. No breaking changes.

### Added
- **`Command` + `CompiledGraph.resume_command()`** (from `flowgentra_ai.graph`)
  — unifies resume with `update=` (schema-validated state update) and
  `goto=` (resume at an arbitrary node). Mirrors LangGraph's
  `Command(resume=, update=, goto=)`. Note: `resume=` only reaches
  Rust-authored nodes; Python node functions receive `state: dict` with no
  context, so hand answers to Python nodes via `update=`.
- **`StateGraphBuilder.set_postgres_checkpointer(url)`** — Postgres-backed
  graph checkpointing for multi-process/replica deployments, joining
  `set_checkpointer` (file) and `set_sqlite_checkpointer`.
- **`@tool` schema inference** — the decorator now infers a full JSON Schema
  from the function's type hints (`Optional[T]`/defaulted params → not
  required) and its Google-style docstring `Args:` block (per-parameter
  descriptions). `parameters=` may still be passed explicitly. New
  `ToolRegistry.to_tool_definitions()` returns built-in + custom tools as a
  `ToolDefinition` list ready for `LLM.chat_with_tools()`; `.get()` now
  includes the parameter schema.
- **`MockLLM`** (`flowgentra_ai.llm`) — a scripted, offline LLM for tests
  (`.always()` / `.sequence()` / `.when_contains()` / `.otherwise()`), no
  network or credentials. `.as_llm()` returns a drop-in `LLM`.
- **Async `Agent`** — `arun()`, `arun_with_input()`, and `arun_with_thread()`
  are native awaitables driven by the tokio runtime (same mechanism as
  `CompiledGraph.ainvoke`), no worker-thread bounce.
- **`CompiledGraph.serve_dev(port)`** — a local dev viewer showing the graph's
  nodes and a live execution-event feed in the browser; returns a
  `DevServerHandle` (`.url` / `.shutdown()`).
- **Chain composition** — `flowgentra_ai.llm.Chain(prompt, llm)` for the
  fixed two-stage prompt → LLM case, and `flowgentra_ai.chain` for LCEL-style
  composition: `prompt | llm | parser | fn` via the `|` operator, or the
  explicit `Chain.sequence([...])`. `PromptTemplate`, `JsonOutputParser`, and
  `ListOutputParser` are now re-exported from `flowgentra_ai.llm` (previously
  reachable only via the internal `_native.text`).

### Fixed
- `builder.add_subgraph(...)` is now covered by an end-to-end test asserting
  the subgraph's result merges into the parent state exactly once (the
  underlying Rust no-op bug affected only pure-Rust graphs; the Python path
  was already correct).

## [0.3.2] - 2026-07-19

### Fixed
- `Conversational` (the skills-aware wrapper) was missing the canonical
  `run()` method introduced in 0.3.0 — it only exposed the deprecated
  `execute_input()`. All seven agent types now share the `run()` vocabulary.
- `flowgentra_ai.document_loaders` was unimportable: the wrapper read the
  loader classes from `_native.rag` after they moved to `_native.loaders`,
  so importing the subpackage raised `AttributeError`. The never-exposed
  `*Config` placeholder names now degrade to `None` instead of crashing the
  import.

### Added
- Regression test suite that imports every public wrapper module and asserts
  the unified `run()` vocabulary across all agent types.

## [0.3.1] - 2026-07-17

Tracks `flowgentra-ai` 0.3.1. No Python API changes.

### Changed
- Config-driven agents (`Agent.from_config_path`, `MemoryAwareAgent.from_config`)
  now execute on the unified `state_graph` engine in core. This is transparent:
  every node type behaves as before, by design. The one observable difference is
  that parallel fan-out merges by per-field reducer in sorted node order rather
  than by legacy BFS-wave last-write-wins, so a config that (accidentally)
  depended on wave ordering can produce different output.

### Changed
- `CompiledGraph.ainvoke` is now a native awaitable (pyo3-async-runtimes): the
  graph runs on the tokio runtime bridged to the asyncio loop, replacing the
  `asyncio.to_thread(invoke)` wrapper — no worker-thread bounce, no per-call
  `block_on` (F-22). Behavior is unchanged for `await g.ainvoke(x)` and
  `asyncio.run(g.ainvoke(x))`.
- `CompiledGraph.astream` is also native: returns `AsyncGraphStream`, whose
  `__anext__` awaits the next event on the tokio runtime — no per-event thread
  bounce. Built on the stable `future_into_py` API only (the `unstable-streams`
  feature is deliberately not enabled); `pyo3-async-runtimes` is exact-pinned
  at 0.22.0.

### Added
- End-to-end test running a config-driven agent through the bridge from Python.
- `Graph.set_max_cost(usd)` — USD cost budget, mirroring `set_max_tokens`;
  breach raises `WorkflowTimeoutError`. `llm.set_model_price(model, input, output)`
  registers a price override. Stubs added (including the previously missing
  `set_max_tokens`). Requires a wheel rebuild to be importable.

## [0.3.0] - 2026-07-16

### Security (breaking)
- `Agent.from_config_path`, `MemoryAwareAgent.from_config`, and
  `from_config_path` now **reject by default** configs that name Python
  modules to import (`python_handler_module`, `handler: python.module:function`,
  `tools: - module: ...`). Importing a module executes its code, so a config
  file is as powerful as a script. Pass `allow_python_handlers=True` for
  configs you trust. (0.2.6 warned and proceeded.)
- API keys are wrapped in a redacting `Secret` type in the Rust core: they no
  longer appear in debug output, error messages, or serialized state — which
  also means **checkpoints no longer contain raw API keys** (0.2.x wrote the
  key into every checkpoint file via `_llm_config`). Handlers using
  `state.get_llm()` now resolve the key from the provider's environment
  variable.

### Added
- `StateGraph.set_max_tokens(n)` — total-token budget.
- Stub coverage raised to 100% (scripts/check_stubs.py gate + gen_stubs.py);
  py.typed already shipped.
- RAG/embeddings config API keys no longer written into checkpoints.
- **Ctrl+C works**: `KeyboardInterrupt` cancels a running graph at the next
  node boundary instead of hanging until completion.
- `CompiledGraph.stream(input)` — iterate live execution events
  (node/edge/tool/LLM-chunk), ending with a `values` event carrying the final
  state. `GraphStream` type.
- `CompiledGraph.ainvoke(...)` / `astream(...)` — async API; graphs run
  concurrently under asyncio.
- `NodeInterrupt` — raise from inside a node to pause for human input;
  `resume_with_state(thread_id, {...})` injects the answer and re-runs the
  interrupted node.
- `StateGraph.set_sqlite_checkpointer("sqlite://file.db")` — durable,
  transactional checkpointing; survives process restarts.
- `StateGraph.add_cached_node(name, fn, max_entries=, ttl_secs=)` — memoize
  expensive deterministic nodes by input-state hash.
- `StateGraph.set_max_duration(seconds)` — wall-clock budget; breach raises
  `WorkflowTimeoutError`.
- `StateGraph.add_conditional_edges` — LangGraph-compatible plural alias.
- `run(input)` on all predefined agents (`ZeroShotReAct`, …) and
  `MemoryAwareAgent.run` — one vocabulary; `run()` also releases the GIL.
- `State` is dict-like: `get(k, default)`, `items()`, `values()`, `update()`,
  `pop()`, iteration, `==` against dicts.
- `TokenUsage(prompt, completion)` constructor.
- Parallel nodes merge per key by channel reducer: `Annotated[list, operator.add]`
  fields accumulate every branch's contribution (was last-write-wins).
- OpenTelemetry helpers re-exported from `flowgentra_ai.observability`:
  `trace_to_otel_spans`, `spans_to_otlp_json`, `export_to_otlp`.
- `py.typed` shipped; stubs updated.

### Fixed
- `resume()` continued from the entry point and re-triggered its own
  breakpoint forever; it now continues after the last completed node.
- `State.from_json` bounded (64 MB) against memory exhaustion.
- Checkpoint `thread_id` path traversal (`"../escape"`) rejected.
- README examples: `model_pricing`/`extract_text` import paths, non-runnable
  snippets.

### Deprecated
- `execute_input()` on predefined agents — use `run()`. Emits
  `DeprecationWarning`.
