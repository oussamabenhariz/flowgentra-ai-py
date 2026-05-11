# Supervisor API Reference

## Supervisor

Orchestrates multiple agent graphs. Two modes: **simple** (router function) and **strategy** (config-based).

```python
from flowgentra_ai.supervision import Supervisor
```

### Constructor / Factory

| Signature | Mode | Description |
|-----------|------|-------------|
| `Supervisor(router)` | Simple | Router function decides the next agent |
| `Supervisor.from_config(config)` | Strategy | Uses a `SupervisorNodeConfig` |

```python
# Simple mode
sup = Supervisor(lambda state: "agent_a" if state.get("task_type") == "a" else "FINISH")

# Strategy mode
sup = Supervisor.from_config(config)
```

### Methods

#### `add_agent(name, agent)` → `None`

Register an agent (a compiled `StateGraph`, callable, or any object with `invoke`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Agent name (must match what the router returns) |
| `agent` | `StateGraph \| Callable` | The agent implementation |

```python
sup.add_agent("researcher", research_graph)
sup.add_agent("writer",     writer_graph)
sup.add_agent("simple",     lambda state: (state.__setitem__("done", True), state)[1])
```

#### `max_rounds(rounds)` → `None`

Set maximum routing rounds (simple mode). Prevents infinite loops.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rounds` | `int` | Maximum iterations |

#### `finish_marker(marker)` → `None`

Set the string that signals the router is done (simple mode). Default: `"FINISH"`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `marker` | `str` | `"FINISH"` | Termination signal |

#### `run(state)` → `State`

Execute the supervisor.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `State` | Initial state passed to the first agent |

#### `agent_names()` → `list[str]`

List all registered agent names.

---

## SupervisorNodeConfig

Full configuration for strategy-based orchestration.

```python
from flowgentra_ai.supervision import SupervisorNodeConfig
```

### Constructor

```python
SupervisorNodeConfig(
    name: str,
    children: list[str],
    strategy: OrchestrationStrategy | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Supervisor name |
| `children` | `list[str]` | required | List of child agent names |
| `strategy` | `OrchestrationStrategy \| None` | Sequential | Orchestration strategy |

```python
config = SupervisorNodeConfig(
    "coordinator",
    ["researcher", "writer"],
    OrchestrationStrategy.parallel(),
)
```

### General Methods

| Method | Parameter | Description |
|--------|-----------|-------------|
| `set_strategy(strategy)` | `OrchestrationStrategy` | Set the orchestration strategy |
| `set_fail_fast(enabled)` | `bool` | Stop on first child error |
| `set_child_timeout_ms(ms)` | `int` | Timeout per child agent (ms) |
| `set_timeout_ms(ms)` | `int` | Global timeout for all children (ms) |
| `set_merge_strategy(strategy)` | `ParallelMergeStrategy` | How to merge parallel state results |
| `set_parallel_aggregation(agg)` | `ParallelAggregation` | Success aggregation for parallel mode |
| `set_collect_stats(enabled)` | `bool` | Collect per-child timing stats |
| `set_max_retries_per_child(n)` | `int` | Retry each child up to N times |
| `set_max_concurrent(n)` | `int` | Max concurrent children in parallel mode |
| `add_skip_condition(child, expr)` | `str, str` | Skip a child when a condition is true |

### Strategy-Specific Methods

| Method | Strategy | Parameter | Description |
|--------|----------|-----------|-------------|
| `set_goal(goal)` | Autonomous | `str` | Human-readable goal |
| `set_required_outputs(keys)` | Autonomous | `list[str]` | State keys that must be present to stop |
| `add_output_owner(key, child)` | Autonomous | `str, str` | Map output key to responsible child |
| `set_max_iterations(n)` | Autonomous, Dynamic, Debate | `int` | Max iterations |
| `set_selector_prompt(prompt)` | Dynamic | `str` | Prompt for LLM agent selector |
| `set_tasks_key(key)` | RoundRobin | `str` | State key with tasks array |
| `set_selection_criteria(c)` | Broadcast | `str` | `"first_success"` or `"highest_score"` |
| `set_score_key(key)` | Broadcast | `str` | State key for quality score (0–1) |
| `set_map_key(key)` | MapReduce | `str` | State key with input array |
| `set_reduce_key(key)` | MapReduce | `str` | State key for merged output |
| `add_routing_rule(cond, child)` | ConditionalRouting | `str, str` | Condition expression → child name |
| `set_fallback_order(order)` | RetryFallback | `list[str]` | Ordered list of agents to try |
| `set_debate_rounds(n)` | Debate | `int` | Number of critique rounds |
| `set_debate_key(key)` | Debate | `str` | State key with debate topic |

---

## OrchestrationStrategy

Selects how children are orchestrated.

```python
from flowgentra_ai.supervision import OrchestrationStrategy
```

| Factory Method | Description |
|----------------|-------------|
| `OrchestrationStrategy.sequential()` | Children run one after another; output flows through each |
| `OrchestrationStrategy.parallel()` | All children run simultaneously |
| `OrchestrationStrategy.autonomous()` | Loop until all required outputs are present |
| `OrchestrationStrategy.dynamic()` | LLM picks the next child at runtime |
| `OrchestrationStrategy.round_robin()` | Distribute tasks from an array across children in rotation |
| `OrchestrationStrategy.hierarchical()` | Children may be sub-supervisors |
| `OrchestrationStrategy.broadcast()` | Send same task to all children; pick best result |
| `OrchestrationStrategy.map_reduce()` | Split input, process in parallel, merge output |
| `OrchestrationStrategy.conditional_routing()` | Route to child based on state conditions |
| `OrchestrationStrategy.retry_fallback()` | Try children in order until one succeeds |
| `OrchestrationStrategy.debate()` | Children generate and critique each other's work |
| `OrchestrationStrategy.custom(name)` | Placeholder for user-defined strategy |

---

## ParallelAggregation

How to determine success when running children in parallel.

```python
from flowgentra_ai.supervision import ParallelAggregation
```

| Factory Method | Description |
|----------------|-------------|
| `ParallelAggregation.first_success()` | Succeed if any child succeeds |
| `ParallelAggregation.all()` | Succeed only if ALL children succeed |
| `ParallelAggregation.majority()` | Succeed if more than half succeed |

---

## ParallelMergeStrategy

How to merge state from multiple parallel child executions.

```python
from flowgentra_ai.supervision import ParallelMergeStrategy
```

| Factory Method | Description |
|----------------|-------------|
| `ParallelMergeStrategy.first_success()` | Use state from the first successful child |
| `ParallelMergeStrategy.latest()` | Use state from the last successful child |
| `ParallelMergeStrategy.deep_merge()` | Deep-merge state from all successful children |
| `ParallelMergeStrategy.custom(name)` | Placeholder for user-defined merge |

---

## ChildExecutionStats

Per-child execution statistics. Available when `set_collect_stats(True)` is set.

```python
from flowgentra_ai.supervision import ChildExecutionStats
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Child agent name |
| `duration_ms` | `int` | Execution time in milliseconds |
| `success` | `bool` | Whether the child succeeded |
| `error` | `str \| None` | Error message if it failed |
| `timeout` | `bool` | Whether the child timed out |
