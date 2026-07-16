# Changelog

All notable changes to the `flowgentra-ai` Python package. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows SemVer
(0.x: minor bumps may break).

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
