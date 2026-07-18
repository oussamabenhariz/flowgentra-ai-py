# Changelog

All notable changes to the `flowgentra-ai` Python package. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows SemVer
(0.x: minor bumps may break).

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
  `asyncio.run(g.ainvoke(x))`. `astream` still uses `to_thread`.

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
