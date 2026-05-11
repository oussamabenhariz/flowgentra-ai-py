# Multi-Agent Supervisor

When one agent isn't enough, you can compose multiple agent graphs under a **Supervisor** that decides which agent to call and when to stop. Flowgentra provides 11 orchestration strategies for different coordination patterns.

---

## Quick Start: Simple Router

**Best for:** Custom routing logic and simple multi-agent coordination

Write a function that looks at state and returns the next agent's name, or `"FINISH"` to stop.

=== "Python"

    ```python
    from flowgentra_ai.supervision import Supervisor
    from flowgentra_ai import State

    def router(state):
        task = state.get_string("task") or ""
        if "research" in task:
            return "researcher"
        if "write" in task:
            return "writer"
        if state.get("research_done") and state.get("draft_done"):
            return "FINISH"
        return "researcher"   # default

    sup = Supervisor(router)
    sup.add_agent("researcher", research_graph)   # StateGraph or callable
    sup.add_agent("writer",     writer_graph)
    sup.max_rounds(10)          # prevent infinite loops
    sup.finish_marker("FINISH") # default is "FINISH"

    result = sup.run(State({"task": "research AI trends and write a summary"}))
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::agents::Supervisor;

    let sup = Supervisor::new(|state: &DynState| {
        let task = state.get_string("task").unwrap_or_default();
        if task.contains("research") { "researcher" }
        else if task.contains("write") { "writer" }
        else { "FINISH" }
    })
    .add_agent("researcher", research_graph)
    .add_agent("writer",     writer_graph)
    .max_rounds(10);

    let result = sup.run(initial_state).await?;
    ```

---

## 11 Orchestration Strategies (Complete Reference)

### Strategy 1: Sequential

**Run agents one after another as a pipeline. Each agent receives the output of the previous.**

**When to use:**
- Data processing pipelines (extract → transform → load)
- Order matters and each stage depends on previous
- Strict ordering guarantees needed

**Function Signature:**
```python
config = SupervisorNodeConfig(
    name: str,
    children: List[str],
    strategy=OrchestrationStrategy.sequential()
)
config.set_fail_fast(bool)  # Stop on first failure (default: False)
config.set_order(List[str]) # Explicit agent order (default: definition order)
```

**Returns:** `State` with results from all agents

**Example:**

```python
config = SupervisorNodeConfig(
    name="parallel_analysis",
    children=["sentiment", "entities", "topics"],
    strategy=OrchestrationStrategy.parallel()
)
config.set_timeout_ms(30000)  # 30s per agent
config.set_merge_strategy("deep_merge")  # Combine all results
config.set_collect_stats(True)  # Track timing
```

**Parameters:**
- `timeout_ms` (int): Per-agent timeout in milliseconds (default: 60000)
- `merge_strategy` (str): "latest" | "deep_merge" | "concat" (default: "latest")
- `collect_stats` (bool): Track execution stats per agent (default: False)

**Merge strategies:**
- `latest()`: Last update wins (overwrites previous values)
- `deep_merge()`: Recursively merge dicts, combine lists
- `concatenate()`: Collect all results into lists

---

### Strategy 3: Autonomous

**Loop until all required outputs are present. Agents decide what to do next.**

**When to use:**
- Goal-oriented workflows with multiple stages
- Agents know their responsibilities
- Need to ensure all outputs are eventually produced
- Self-supervising teams

**Function Signature:**
```python
config = SupervisorNodeConfig(
    name: str,
    children: List[str],
    strategy=OrchestrationStrategy.autonomous()
)
config.set_goal(str)                      # Overall goal description
config.set_required_outputs(List[str])    # Keys that must be in state
config.add_output_owner(key, agent_name)  # Which agent produces each output
config.set_max_iterations(int)            # Max loops (default: 100)
```

**Returns:** `State` with all required outputs

**Example:**
```python
config = SupervisorNodeConfig(
    name="research_team",
    children=["researcher", "analyst",  "writer"],
    strategy=OrchestrationStrategy.autonomous()
)
config.set_goal("Produce comprehensive market analysis")
config.set_required_outputs(["data", "analysis", "report"])
config.add_output_owner("data", "researcher")
config.add_output_owner("analysis", "analyst")
config.add_output_owner("report", "writer")
config.set_max_iterations(20)
```

---

### Strategy 4: Dynamic

**LLM decides which agent to call next based on current state.**

**When to use:**
- Complex decision logic better handled by LLM
- Adaptive routing needed
- Want interpretable decisions from LLM

**Function Signature:**
```python
config = SupervisorNodeConfig(
    name: str,
    children: List[str],
    strategy=OrchestrationStrategy.dynamic()
)
config.set_selector_prompt(str)   # Custom prompt for routing decision
config.set_llm_config(str)        # LLM to use (default: "gpt-4")
config.set_max_iterations(int)    # Max routing decisions
```

**Returns:** `State` after LLM-directed agent executions

---

### Strategy 5: Round Robin

**Distribute tasks across agents in sequence. Each gets a turn.**

**When to use:**
- Load balancing across identical workers
- One agent shouldn't become bottleneck
- Fair task distribution

**Function Signature:**
```python
config = SupervisorNodeConfig(
    name: str,
    children: List[str],
    strategy=OrchestrationStrategy.round_robin()
)
config.set_tasks_key(str)           # State key with task list/queue
config.set_skip_on_error(bool)      # Skip agent if it fails
config.set_rotation_strategy(str)   # "sequential" or "random"
```

---

### Strategy 6: Hierarchical

**Multi-level supervision. Top supervisor delegates to sub-supervisors.**

**When to use:**
- Organizing related agents into groups
- Different teams for different domains
- Independent sub-coordinators per team

**Function Signature:**
```python
config = SupervisorNodeConfig(
    name: str,
    children: List[str],  # Names of other supervisors
    strategy=OrchestrationStrategy.hierarchical()
)
config.set_level_depth(int)           # Max nesting depth
config.set_parallel_subteams(bool)    # Run sub-supervisors in parallel
```

---

### Strategy 7: Broadcast

**Send to all agents, pick the best result.**

**When to use:**
- High-accuracy critical tasks
- Multiple solution approaches
- Want to compare results

**Function Signature:**
```python
config = SupervisorNodeConfig(
    name: str,
    children: List[str],
    strategy=OrchestrationStrategy.broadcast()
)
config.set_selection_criteria(str)  # "highest_score" or "first_success"
config.set_score_key(str)           # State key with quality score
```

**Returns:** `State` with best result selected

---

### Strategy 8: MapReduce

**Split input, process in parallel across agents, merge results.**

**When to use:**
- Batch processing large datasets
- Embarrassingly parallel problems
- Need to aggregate results

**Function Signature:**
```python
config = SupervisorNodeConfig(
    name: str,
    children: List[str],
    strategy=OrchestrationStrategy.map_reduce()
)
config.set_map_key(str)         # Key with array to split
config.set_reduce_key(str)      # Key to store merged results
config.set_batch_size(int)      # Items per agent (auto if not set)
config.set_reduce_function(fn)  # Custom aggregation function
```

---

### Strategy 9: Conditional Routing

**Route based on conditions in state.**

**When to use:**
- Route by task type or category
- Multiple conditional paths
- Complex branching logic

**Function Signature:**
```python
config = SupervisorNodeConfig(
    name: str,
    children: List[str],
    strategy=OrchestrationStrategy.conditional_routing()
)
config.add_routing_rule(condition_str, agent_name)
config.set_default_agent(agent_name)
```

**Example:**
```python
config.add_routing_rule("task_type == 'code'", "code_agent")
config.add_routing_rule("task_type == 'writing'", "writing_agent")
config.add_routing_rule("task_type == 'math'", "math_agent")
config.set_default_agent("general_agent")
```

---

### Strategy 10: Retry Fallback

**Try agents sequentially until one succeeds (with retries).**

**When to use:**
- Reliability and fault tolerance critical
- Different agents have different failure modes
- Graceful degradation

**Function Signature:**
```python
config = SupervisorNodeConfig(
    name: str,
    children: List[str],  # Ordered list of fallbacks
    strategy=OrchestrationStrategy.retry_fallback()
)
config.set_max_retries(int)           # Retries per agent
config.set_fallback_order(List[str])  # Explicit order
config.set_on_all_failed(agent_name)  # Last resort agent
```

**Returns:** `State` from first successful agent, or error if all fail

---

### Strategy 11: Debate

**Agents with different perspectives evaluate each other.**

**When to use:**
- High-stakes decisions need validation
- Want multiple viewpoints
- Iterative improvement through debate

**Function Signature:**
```python
config = SupervisorNodeConfig(
    name: str,
    children: List[str],
    strategy=OrchestrationStrategy.debate()
)
config.set_debate_rounds(int)   # Number of critique rounds
config.set_debate_key(str)      # State key with topic/question
config.set_selection_criteria(str)  # How to pick final answer
```

---

## SupervisorNodeConfig: Complete API Reference

### Constructor

```python
SupervisorNodeConfig(
    name: str,                              # Supervisor identifier
    children: List[str],                    # Child agent names
    strategy: OrchestrationStrategy,        # Strategy instance
    checkpointer: Optional[str] = "memory", # "memory" or "file"
    max_rounds: int = 100,                  # Max execution cycles
    timeout_ms: int = 300000,               # Total timeout (5 min)
    collect_stats: bool = False,            # Track metrics
    debug: bool = False                     # Debug logging
)
```

### Core Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `add_agent` | `name: str, agent: Graph` | `Self` | Register an agent/graph |
| `add_agent_callable` | `name: str, fn: Callable` | `Self` | Register callable as agent |
| `set_timeout_ms` | `ms: int` | `Self` | Total execution timeout |
| `set_child_timeout_ms` | `ms: int` | `Self` | Per-child timeout |
| `set_max_rounds` | `n: int` | `Self` | Max execution cycles |
| `set_collect_stats` | `bool` | `Self` | Enable/disable metrics |
| `collect_stats` | (none) | `ChildExecutionStats` | Get execution statistics |

### Strategy-Specific Methods

| Strategy | Methods |
|----------|---------|
| `sequential` | `set_fail_fast(bool)`, `set_order(List[str])` |
| `parallel` | `set_merge_strategy(str)`, `set_timeout_ms(int)` |
| `conditional_routing` | `add_routing_rule(str, str)`, `set_default_agent(str)` |
| `map_reduce` | `set_map_key(str)`, `set_reduce_key(str)`, `set_batch_size(int)` |
| `retry_fallback` | `set_max_retries(int)`, `set_fallback_order(List[str])` |

---

## Supervisor Class: Complete API

### Constructor

```python
Supervisor(
    router_fn: Union[Callable, OrchestrationStrategy],
    checkpointer: Optional[Checkpointer] = None,
    max_rounds: int = 100,
    finish_marker: str = "FINISH"
)
```

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `add_agent` | `(name: str, agent: Union[Graph, Callable])` | `Supervisor` | Register agent |
| `max_rounds` | `(n: int)` | `Supervisor` | Set max cycles |
| `finish_marker` | `(marker: str)` | `Supervisor` | Set finish token |
| `from_config` | `(config: SupervisorNodeConfig)` → `Supervisor` | `Supervisor` | Create from config |
| `run` | `(state: State)` → async | `State` | Execute supervisor |
| `run_with_thread` | `(thread_id: str, state: State)` → async | `State` | Execute with checkpointing |

### Example: Complete Supervisor Setup

```python
from flowgentra_ai.supervision import Supervisor, SupervisorNodeConfig, OrchestrationStrategy

# Create agents
research_agent = build_research_graph()
writing_agent = build_writing_graph()
editor_agent = build_editor_graph()

# Method 1: Simple Router Function
def route_next(state):
    if not state.get("has_research"):
        return "researcher"
    elif not state.get("has_draft"):
        return "writer"
    elif not state.get("is_edited"):
        return "editor"
    else:
        return "FINISH"

sup = Supervisor(route_next)
sup.add_agent("researcher", research_agent)
sup.add_agent("writer", writing_agent)
sup.add_agent("editor", editor_agent)
sup.max_rounds(10)

# Execute
result = sup.run(State({"topic": "AI trends"}))

# Method 2: Strategy-Based (Sequential Pipeline)
config = SupervisorNodeConfig(
    "content_pipeline",
    children=["researcher", "writer", "editor"],
    strategy=OrchestrationStrategy.sequential()
)
config.set_fail_fast(True)

sup2 = Supervisor.from_config(config)
sup2.add_agent("researcher", research_agent)
sup2.add_agent("writer", writing_agent)
sup2.add_agent("editor", editor_agent)
```

---

## MemoryAwareAgent with Supervision

For multi-user scenarios with isolated memory per user:

```python
from flowgentra_ai.agent import MemoryAwareAgent

# Create base supervisor
supervisor = Supervisor(router_fn)
supervisor.add_agent("agent1", agent1)
supervisor.add_agent("agent2", agent2)

# Wrap with memory
memory_supervisor = MemoryAwareAgent(
    base_agent=supervisor,
    memory_type="conversation",
    max_messages=100,
    per_user_isolation=True
)

# Each user gets isolated memory
result = memory_supervisor.run_with_user_id(
    user_id="user-123",
    state=State({"query": "Summarize what we discussed"})
)

# Memory automatically tracked per user
memory_stats = memory_supervisor.get_stats("user-123")
print(f"Total turns: {memory_stats.total_turns}")
```

---

## Error Handling & Edge Cases

### Handling Agent Failures

```python
config = SupervisorNodeConfig(
    name="resilient",
    children=["primary", "backup"],
    strategy=OrchestrationStrategy.retry_fallback()
)
config.set_max_retries(2)

try:
    result = sup.run(state)
except Exception as e:
    print(f"All agents failed: {e}")
```

### Preventing Infinite Loops

```python
sup.max_rounds(50)  # Hard limit

# Also check state for termination
def should_stop(state):
    return state.get("confidence", 0) > 0.95 or state.get("attempts", 0) > 10

# Custom termination
config.set_termination_condition(should_stop)
```

### Timeout Protection

```python
config.set_timeout_ms(300000)  # 5 minute total
config.set_child_timeout_ms(60000)  # 1 minute per agent
```

---

## Collecting & Analyzing Statistics

```python
config.set_collect_stats(True)

result = sup.run(state)

# Access statistics
stats = config.collect_stats()

for agent_stat in stats.per_agent:
    print(f"{agent_stat.name}:")
    print(f"  Duration: {agent_stat.duration_ms}ms")
    print(f"  Success: {agent_stat.success}")
    print(f"  Calls: {agent_stat.call_count}")
```

---

## Strategy Comparison: Decision Matrix

| Need | Strategy | Speed | Complexity |
|------|----------|-------|-----------|
| Pipeline | Sequential | ⭐ | ⭐ |
| Independence | Parallel | ⭐⭐⭐ | ⭐⭐ |
| Best answer | Broadcast | ⭐ | ⭐⭐⭐ |
| Load balance | Round Robin | ⭐⭐ | ⭐ |
| Fault tolerance | Retry Fallback | ⭐ | ⭐⭐ |
| Goal-driven | Autonomous | ⭐ | ⭐⭐⭐⭐ |
| Routing logic | Conditional | ⭐⭐ | ⭐⭐ |
| Batch processing | MapReduce | ⭐⭐⭐ | ⭐⭐⭐ |
| Complex decisions | Dynamic | ⭐⭐ | ⭐⭐⭐⭐ |
| Validation | Debate | ⭐ | ⭐⭐⭐⭐ |
| Organization | Hierarchical | ⭐ | ⭐⭐⭐⭐⭐ |

---

## Best Practices

### 1. Test Agents Independently First
Always validate each agent works correctly before supervising.

### 2. Start Simple, Add Complexity
Begin with Sequential or Parallel, graduate to advanced strategies.

### 3. Handle Failures Explicitly
Define fallback behavior, retries, and timeout strategies.

### 4. Document State Contracts
Be clear about what state each agent expects and produces.

### 5. Monitor Production Supervisors
Collect stats and alert on failures or slow performance.

```python
# Production example
config.set_collect_stats(True)
result = sup.run(state)
stats = config.collect_stats()

if stats.total_duration_ms > 10000:
    alert("Supervisor took too long")

if not stats.success:
    alert(f"Supervisor failed: {stats.error}")
```
