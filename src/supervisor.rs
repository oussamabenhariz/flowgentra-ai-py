//! Python bindings for Supervisor (multi-agent orchestration)
//!
//! Exposes the full Rust SupervisorNode with all 11 orchestration strategies,
//! configuration types, and execution stats.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use flowgentra_ai::core::node::nodes_trait::{NodeOutput, PluggableNode};
use flowgentra_ai::core::node::orchestrator_node::{
    ChildExecutionStats, OrchestrationStrategy, ParallelAggregation, ParallelMergeStrategy,
    SupervisorNode, SupervisorNodeConfig,
};
use flowgentra_ai::core::state::DynState;
use flowgentra_ai::core::state_graph::StateGraph;

use crate::graph::PyStateGraph;
use crate::state::PyState;

// ─── Bridge: StateGraph → PluggableNode ─────────────────────────────────────

/// Adapter that wraps a StateGraph as a PluggableNode so it can be used
/// as a child in the Rust SupervisorNode.
struct StateGraphAsPluggable {
    name: String,
    graph: Arc<StateGraph<DynState>>,
}

#[async_trait::async_trait]
impl PluggableNode<DynState> for StateGraphAsPluggable {
    async fn run(
        &self,
        state: DynState,
    ) -> flowgentra_ai::core::error::Result<NodeOutput<DynState>> {
        let start = std::time::Instant::now();
        match self.graph.invoke(state.clone()).await {
            Ok(result_state) => Ok(NodeOutput {
                state: result_state,
                metadata: HashMap::new(),
                success: true,
                error: None,
                execution_time_ms: start.elapsed().as_millis(),
            }),
            Err(e) => Ok(NodeOutput {
                state,
                metadata: HashMap::new(),
                success: false,
                error: Some(format!("{}", e)),
                execution_time_ms: start.elapsed().as_millis(),
            }),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn node_type(&self) -> &str {
        "state_graph"
    }

    fn config(&self) -> &HashMap<String, serde_json::Value> {
        // Return empty config — the graph is opaque
        static EMPTY: std::sync::OnceLock<HashMap<String, serde_json::Value>> =
            std::sync::OnceLock::new();
        EMPTY.get_or_init(HashMap::new)
    }

    fn clone_box(&self) -> Box<dyn PluggableNode<DynState>> {
        Box::new(StateGraphAsPluggable {
            name: self.name.clone(),
            graph: self.graph.clone(),
        })
    }
}

/// Adapter that wraps a Python callable as a PluggableNode.
struct PyCallableAsPluggable {
    name: String,
    func: PyObject,
}

#[async_trait::async_trait]
impl PluggableNode<DynState> for PyCallableAsPluggable {
    async fn run(
        &self,
        state: DynState,
    ) -> flowgentra_ai::core::error::Result<NodeOutput<DynState>> {
        let start = std::time::Instant::now();
        let result = Python::with_gil(|py| -> PyResult<DynState> {
            let func = self.func.clone_ref(py);
            let py_state = PyState {
                inner: state.clone(),
            };
            let py_result = func.call1(py, (py_state,))?;
            let result_state: PyState = py_result.extract(py)?;
            Ok(result_state.inner)
        });

        match result {
            Ok(result_state) => Ok(NodeOutput {
                state: result_state,
                metadata: HashMap::new(),
                success: true,
                error: None,
                execution_time_ms: start.elapsed().as_millis(),
            }),
            Err(e) => Ok(NodeOutput {
                state,
                metadata: HashMap::new(),
                success: false,
                error: Some(format!("Python callable error: {}", e)),
                execution_time_ms: start.elapsed().as_millis(),
            }),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn node_type(&self) -> &str {
        "python_callable"
    }

    fn config(&self) -> &HashMap<String, serde_json::Value> {
        static EMPTY: std::sync::OnceLock<HashMap<String, serde_json::Value>> =
            std::sync::OnceLock::new();
        EMPTY.get_or_init(HashMap::new)
    }

    fn clone_box(&self) -> Box<dyn PluggableNode<DynState>> {
        Python::with_gil(|py| {
            Box::new(PyCallableAsPluggable {
                name: self.name.clone(),
                func: self.func.clone_ref(py),
            }) as Box<dyn PluggableNode<DynState>>
        })
    }
}

// ─── PyOrchestrationStrategy ────────────────────────────────────────────────

/// Orchestration strategy for the Supervisor.
///
/// Available strategies:
///   - Sequential: children run one after another
///   - Parallel: all children run simultaneously
///   - Autonomous: loop-based, runs agents until required outputs are present
///   - Dynamic: LLM decides which agents to call at runtime
///   - RoundRobin: distributes tasks across agents in rotation
///   - Hierarchical: delegates to sub-supervisors
///   - Broadcast: sends same task to all, picks best result
///   - MapReduce: splits input, processes in parallel, merges results
///   - ConditionalRouting: routes to agent based on state conditions
///   - RetryFallback: tries agents in order until one succeeds
///   - Debate: agents generate and critique each other's outputs
#[pyclass(name = "OrchestrationStrategy")]
#[derive(Clone)]
pub struct PyOrchestrationStrategy {
    pub(crate) inner: OrchestrationStrategy,
}

#[pymethods]
impl PyOrchestrationStrategy {
    #[staticmethod]
    fn sequential() -> Self {
        PyOrchestrationStrategy {
            inner: OrchestrationStrategy::Sequential,
        }
    }
    #[staticmethod]
    fn parallel() -> Self {
        PyOrchestrationStrategy {
            inner: OrchestrationStrategy::Parallel,
        }
    }
    #[staticmethod]
    fn autonomous() -> Self {
        PyOrchestrationStrategy {
            inner: OrchestrationStrategy::Autonomous,
        }
    }
    #[staticmethod]
    fn dynamic() -> Self {
        PyOrchestrationStrategy {
            inner: OrchestrationStrategy::Dynamic,
        }
    }
    #[staticmethod]
    fn round_robin() -> Self {
        PyOrchestrationStrategy {
            inner: OrchestrationStrategy::RoundRobin,
        }
    }
    #[staticmethod]
    fn hierarchical() -> Self {
        PyOrchestrationStrategy {
            inner: OrchestrationStrategy::Hierarchical,
        }
    }
    #[staticmethod]
    fn broadcast() -> Self {
        PyOrchestrationStrategy {
            inner: OrchestrationStrategy::Broadcast,
        }
    }
    #[staticmethod]
    fn map_reduce() -> Self {
        PyOrchestrationStrategy {
            inner: OrchestrationStrategy::MapReduce,
        }
    }
    #[staticmethod]
    fn conditional_routing() -> Self {
        PyOrchestrationStrategy {
            inner: OrchestrationStrategy::ConditionalRouting,
        }
    }
    #[staticmethod]
    fn retry_fallback() -> Self {
        PyOrchestrationStrategy {
            inner: OrchestrationStrategy::RetryFallback,
        }
    }
    #[staticmethod]
    fn debate() -> Self {
        PyOrchestrationStrategy {
            inner: OrchestrationStrategy::Debate,
        }
    }
    #[staticmethod]
    fn custom(name: &str) -> Self {
        PyOrchestrationStrategy {
            inner: OrchestrationStrategy::Custom(name.to_string()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OrchestrationStrategy.{}",
            match &self.inner {
                OrchestrationStrategy::Sequential => "sequential".to_string(),
                OrchestrationStrategy::Parallel => "parallel".to_string(),
                OrchestrationStrategy::Autonomous => "autonomous".to_string(),
                OrchestrationStrategy::Dynamic => "dynamic".to_string(),
                OrchestrationStrategy::RoundRobin => "round_robin".to_string(),
                OrchestrationStrategy::Hierarchical => "hierarchical".to_string(),
                OrchestrationStrategy::Broadcast => "broadcast".to_string(),
                OrchestrationStrategy::MapReduce => "map_reduce".to_string(),
                OrchestrationStrategy::ConditionalRouting => "conditional_routing".to_string(),
                OrchestrationStrategy::RetryFallback => "retry_fallback".to_string(),
                OrchestrationStrategy::Debate => "debate".to_string(),
                OrchestrationStrategy::Custom(name) => format!("custom({})", name),
            }
        )
    }
}

// ─── PyParallelAggregation ──────────────────────────────────────────────────

/// How to aggregate results from parallel child executions.
///
///   - FirstSuccess: use the first child that succeeds
///   - All: require all children to succeed
///   - Majority: succeed if more than half succeed
#[pyclass(name = "ParallelAggregation")]
#[derive(Clone)]
pub struct PyParallelAggregation {
    pub(crate) inner: ParallelAggregation,
}

#[pymethods]
impl PyParallelAggregation {
    #[staticmethod]
    fn first_success() -> Self {
        PyParallelAggregation {
            inner: ParallelAggregation::FirstSuccess,
        }
    }
    #[staticmethod]
    fn all() -> Self {
        PyParallelAggregation {
            inner: ParallelAggregation::All,
        }
    }
    #[staticmethod]
    fn majority() -> Self {
        PyParallelAggregation {
            inner: ParallelAggregation::Majority,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ParallelAggregation.{}",
            match &self.inner {
                ParallelAggregation::FirstSuccess => "first_success",
                ParallelAggregation::All => "all",
                ParallelAggregation::Majority => "majority",
            }
        )
    }
}

// ─── PyParallelMergeStrategy ────────────────────────────────────────────────

/// How to merge state from parallel child executions.
///
///   - FirstSuccess: use the first successful child's state
///   - Latest: use the last successful child's state
///   - DeepMerge: deep-merge all successful states
///   - Custom(name): placeholder for user-defined merge logic
#[pyclass(name = "ParallelMergeStrategy")]
#[derive(Clone)]
pub struct PyParallelMergeStrategy {
    pub(crate) inner: ParallelMergeStrategy,
}

#[pymethods]
impl PyParallelMergeStrategy {
    #[staticmethod]
    fn first_success() -> Self {
        PyParallelMergeStrategy {
            inner: ParallelMergeStrategy::FirstSuccess,
        }
    }
    #[staticmethod]
    fn latest() -> Self {
        PyParallelMergeStrategy {
            inner: ParallelMergeStrategy::Latest,
        }
    }
    #[staticmethod]
    fn deep_merge() -> Self {
        PyParallelMergeStrategy {
            inner: ParallelMergeStrategy::DeepMerge,
        }
    }
    #[staticmethod]
    fn custom(name: &str) -> Self {
        PyParallelMergeStrategy {
            inner: ParallelMergeStrategy::Custom(name.to_string()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ParallelMergeStrategy.{}",
            match &self.inner {
                ParallelMergeStrategy::FirstSuccess => "first_success".to_string(),
                ParallelMergeStrategy::Latest => "latest".to_string(),
                ParallelMergeStrategy::DeepMerge => "deep_merge".to_string(),
                ParallelMergeStrategy::Custom(name) => format!("custom({})", name),
            }
        )
    }
}

// ─── PyChildExecutionStats ──────────────────────────────────────────────────

/// Per-child execution statistics collected when `collect_stats=True`.
#[pyclass(name = "ChildExecutionStats")]
#[derive(Clone)]
pub struct PyChildExecutionStats {
    inner: ChildExecutionStats,
}

#[pymethods]
impl PyChildExecutionStats {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn duration_ms(&self) -> u128 {
        self.inner.duration_ms
    }

    #[getter]
    fn success(&self) -> bool {
        self.inner.success
    }

    #[getter]
    fn error(&self) -> Option<String> {
        self.inner.error.clone()
    }

    #[getter]
    fn timeout(&self) -> bool {
        self.inner.timeout
    }

    fn __repr__(&self) -> String {
        format!(
            "ChildExecutionStats(name='{}', duration_ms={}, success={}, timeout={})",
            self.inner.name, self.inner.duration_ms, self.inner.success, self.inner.timeout
        )
    }
}

// ─── PySupervisorNodeConfig ─────────────────────────────────────────────────

/// Configuration for the SupervisorNode.
///
/// Example:
///     config = SupervisorNodeConfig(
///         name="coordinator",
///         children=["researcher", "writer"],
///         strategy=OrchestrationStrategy.parallel(),
///     )
#[pyclass(name = "SupervisorNodeConfig")]
#[derive(Clone)]
pub struct PySupervisorNodeConfig {
    pub(crate) inner: SupervisorNodeConfig,
}

#[pymethods]
impl PySupervisorNodeConfig {
    #[new]
    #[pyo3(signature = (name, children, strategy=None))]
    fn new(
        name: &str,
        children: Vec<String>,
        strategy: Option<PyOrchestrationStrategy>,
    ) -> PyResult<Self> {
        if children.is_empty() {
            return Err(crate::error::AgentExecutionError::new_err(
                "children list must not be empty",
            ));
        }
        Ok(PySupervisorNodeConfig {
            inner: SupervisorNodeConfig::new(name, children)
                .strategy(strategy.map(|s| s.inner).unwrap_or_default()),
        })
    }

    /// Set the orchestration strategy.
    fn set_strategy(&mut self, strategy: PyOrchestrationStrategy) {
        self.inner.strategy = strategy.inner;
    }

    /// Set whether to stop on first child error.
    fn set_fail_fast(&mut self, fail_fast: bool) {
        self.inner.fail_fast = fail_fast;
    }

    /// Set per-child timeout in milliseconds.
    fn set_child_timeout_ms(&mut self, timeout_ms: u64) {
        self.inner.child_timeout_ms = Some(timeout_ms);
    }

    /// Set global timeout for the entire supervisor execution.
    fn set_timeout_ms(&mut self, timeout_ms: u64) {
        self.inner.timeout_ms = Some(timeout_ms);
    }

    /// Set how to merge state from parallel executions.
    fn set_merge_strategy(&mut self, strategy: PyParallelMergeStrategy) {
        self.inner.merge_strategy = strategy.inner;
    }

    /// Set parallel aggregation policy.
    fn set_parallel_aggregation(&mut self, agg: PyParallelAggregation) {
        self.inner.parallel_aggregation = Some(agg.inner);
    }

    /// Enable or disable per-child stats collection.
    fn set_collect_stats(&mut self, collect: bool) {
        self.inner.collect_stats = collect;
    }

    /// Set max retries per child before giving up.
    fn set_max_retries_per_child(&mut self, retries: usize) {
        self.inner.max_retries_per_child = retries;
    }

    /// Set max concurrent children in parallel mode.
    fn set_max_concurrent(&mut self, max: usize) {
        self.inner.max_concurrent = Some(max);
    }

    /// Add a skip condition for a child.
    /// Example: `config.add_skip_condition("analyst", "analysis != null")`
    fn add_skip_condition(&mut self, child_name: &str, condition: &str) {
        self.inner
            .skip_conditions
            .insert(child_name.to_string(), condition.to_string());
    }

    // ── Autonomous strategy fields ──

    /// Set the goal description for autonomous mode.
    fn set_goal(&mut self, goal: &str) {
        self.inner.goal = Some(goal.to_string());
    }

    /// Set required output keys for autonomous mode.
    fn set_required_outputs(&mut self, outputs: Vec<String>) {
        self.inner.required_outputs = outputs;
    }

    /// Map a required output key to the child responsible for producing it.
    fn add_output_owner(&mut self, output_key: &str, child_name: &str) {
        self.inner
            .output_owners
            .insert(output_key.to_string(), child_name.to_string());
    }

    /// Set maximum iterations for autonomous/dynamic/debate mode.
    fn set_max_iterations(&mut self, max: usize) {
        self.inner.max_iterations = max;
    }

    // ── Dynamic strategy fields ──

    /// Set the LLM selector prompt for dynamic strategy.
    fn set_selector_prompt(&mut self, prompt: &str) {
        self.inner.selector_prompt = Some(prompt.to_string());
    }

    // ── RoundRobin strategy fields ──

    /// Set the state key containing tasks array for round-robin.
    fn set_tasks_key(&mut self, key: &str) {
        self.inner.tasks_key = Some(key.to_string());
    }

    // ── Broadcast strategy fields ──

    /// Set selection criteria for broadcast ("first_success", "highest_score", "llm_judge").
    fn set_selection_criteria(&mut self, criteria: &str) {
        self.inner.selection_criteria = Some(criteria.to_string());
    }

    /// Set the state key for score comparison in broadcast mode.
    fn set_score_key(&mut self, key: &str) {
        self.inner.score_key = Some(key.to_string());
    }

    // ── MapReduce strategy fields ──

    /// Set the state key for input chunks in map-reduce.
    fn set_map_key(&mut self, key: &str) {
        self.inner.map_key = Some(key.to_string());
    }

    /// Set the state key for reduced output in map-reduce.
    fn set_reduce_key(&mut self, key: &str) {
        self.inner.reduce_key = Some(key.to_string());
    }

    // ── ConditionalRouting strategy fields ──

    /// Add a routing rule mapping a condition to a child agent.
    /// Example: `config.add_routing_rule("task_type == code", "code_agent")`
    fn add_routing_rule(&mut self, condition: &str, child_name: &str) {
        self.inner
            .routing_rules
            .insert(condition.to_string(), child_name.to_string());
    }

    // ── RetryFallback strategy fields ──

    /// Set the ordered fallback list of agent names.
    fn set_fallback_order(&mut self, order: Vec<String>) {
        self.inner.fallback_order = order;
    }

    // ── Debate strategy fields ──

    /// Set the number of debate rounds.
    fn set_debate_rounds(&mut self, rounds: usize) {
        self.inner.debate_rounds = rounds;
    }

    /// Set the state key for the debate topic.
    fn set_debate_key(&mut self, key: &str) {
        self.inner.debate_key = Some(key.to_string());
    }

    fn __repr__(&self) -> String {
        format!(
            "SupervisorNodeConfig(name='{}', children={:?}, strategy={:?})",
            self.inner.name, self.inner.children, self.inner.strategy
        )
    }
}

// ─── PySupervisor ───────────────────────────────────────────────────────────

/// Multi-agent supervisor that orchestrates named agent graphs.
///
/// Supports all 11 orchestration strategies from the Rust engine:
/// Sequential, Parallel, Autonomous, Dynamic, RoundRobin, Hierarchical,
/// Broadcast, MapReduce, ConditionalRouting, RetryFallback, Debate.
///
/// **Strategy mode (inline):**
///     sup = Supervisor("coordinator", children=["researcher", "writer"],
///                      strategy=OrchestrationStrategy.parallel(),
///                      fail_fast=True, child_timeout_ms=30_000)
///     sup.add_agent("researcher", researcher_graph)
///
/// **Strategy mode (config object):**
///     config = SupervisorNodeConfig("coordinator", ["researcher", "writer"],
///                                    OrchestrationStrategy.parallel())
///     config.set_child_timeout_ms(30_000)
///     sup = Supervisor.from_config(config)
///     sup.add_agent("researcher", researcher_graph)
///
/// **Router mode:**
///     sup = Supervisor(lambda state: "researcher" if ... else "FINISH")
///     sup.add_agent("researcher", research_graph)
#[pyclass(name = "Supervisor")]
pub struct PySupervisor {
    /// Router function (for simple mode)
    router: Option<PyObject>,
    /// Config (for strategy-based mode)
    config: Option<SupervisorNodeConfig>,
    /// Registered agent graphs
    agents: Vec<(String, AgentChild)>,
    /// Max rounds for router-based mode
    max_rounds_val: usize,
    /// Finish marker for router-based mode
    finish_marker_val: String,
}

enum AgentChild {
    Graph(Arc<StateGraph<DynState>>),
    Callable(PyObject),
}

#[pymethods]
impl PySupervisor {
    /// Create a new supervisor.
    ///
    /// **Strategy mode** (pass name + children):
    ///     sup = Supervisor("coordinator", children=["a", "b"],
    ///                      strategy=OrchestrationStrategy.sequential(),
    ///                      fail_fast=True, child_timeout_ms=60_000)
    ///
    /// **Router mode** (pass a callable router):
    ///     sup = Supervisor(router_fn)
    #[new]
    #[pyo3(signature = (
        name_or_router,
        children=None,
        strategy=None,
        fail_fast=false,
        child_timeout_ms=None,
        timeout_ms=None,
        merge_strategy=None,
        parallel_aggregation=None,
        max_retries_per_child=0,
        max_concurrent=None,
        collect_stats=true,
        max_iterations=10,
        goal=None,
        required_outputs=None,
        output_owners=None,
        selector_prompt=None,
        tasks_key=None,
        selection_criteria=None,
        score_key=None,
        map_key=None,
        reduce_key=None,
        fallback_order=None,
        debate_rounds=2,
        debate_key=None,
        skip_conditions=None,
        routing_rules=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name_or_router: PyObject,
        children: Option<Vec<String>>,
        strategy: Option<PyOrchestrationStrategy>,
        fail_fast: bool,
        child_timeout_ms: Option<u64>,
        timeout_ms: Option<u64>,
        merge_strategy: Option<PyParallelMergeStrategy>,
        parallel_aggregation: Option<PyParallelAggregation>,
        max_retries_per_child: usize,
        max_concurrent: Option<usize>,
        collect_stats: bool,
        max_iterations: usize,
        goal: Option<String>,
        required_outputs: Option<Vec<String>>,
        output_owners: Option<HashMap<String, String>>,
        selector_prompt: Option<String>,
        tasks_key: Option<String>,
        selection_criteria: Option<String>,
        score_key: Option<String>,
        map_key: Option<String>,
        reduce_key: Option<String>,
        fallback_order: Option<Vec<String>>,
        debate_rounds: usize,
        debate_key: Option<String>,
        skip_conditions: Option<HashMap<String, String>>,
        routing_rules: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        if let Some(children_list) = children {
            let name: String = Python::with_gil(|py| name_or_router.extract::<String>(py))
                .map_err(|_| {
                    crate::error::AgentExecutionError::new_err(
                "When children are provided, the first argument must be the supervisor name (str)"
            )
                })?;
            if children_list.is_empty() {
                return Err(crate::error::AgentExecutionError::new_err(
                    "children list must not be empty",
                ));
            }
            let mut config = SupervisorNodeConfig::new(name, children_list)
                .strategy(strategy.map(|s| s.inner).unwrap_or_default())
                .fail_fast(fail_fast)
                .max_retries_per_child(max_retries_per_child)
                .collect_stats(collect_stats)
                .max_iterations(max_iterations)
                .debate_rounds(debate_rounds);
            if let Some(ms) = child_timeout_ms {
                config = config.child_timeout_ms(ms);
            }
            if let Some(ms) = timeout_ms {
                config = config.timeout_ms(ms);
            }
            if let Some(s) = merge_strategy {
                config = config.merge_strategy(s.inner);
            }
            if let Some(a) = parallel_aggregation {
                config = config.parallel_aggregation(a.inner);
            }
            if let Some(n) = max_concurrent {
                config = config.max_concurrent(n);
            }
            if let Some(g) = goal {
                config = config.goal(g);
            }
            if let Some(ro) = required_outputs {
                config = config.required_outputs(ro);
            }
            if let Some(oo) = output_owners {
                for (k, v) in oo {
                    config = config.add_output_owner(k, v);
                }
            }
            if let Some(sp) = selector_prompt {
                config = config.selector_prompt(sp);
            }
            if let Some(tk) = tasks_key {
                config = config.tasks_key(tk);
            }
            if let Some(sc) = selection_criteria {
                config = config.selection_criteria(sc);
            }
            if let Some(sk) = score_key {
                config = config.score_key(sk);
            }
            if let Some(mk) = map_key {
                config = config.map_key(mk);
            }
            if let Some(rk) = reduce_key {
                config = config.reduce_key(rk);
            }
            if let Some(fo) = fallback_order {
                config = config.fallback_order(fo);
            }
            if let Some(dk) = debate_key {
                config = config.debate_key(dk);
            }
            if let Some(sc) = skip_conditions {
                for (k, v) in sc {
                    config = config.add_skip_condition(k, v);
                }
            }
            if let Some(rr) = routing_rules {
                for (k, v) in rr {
                    config = config.add_routing_rule(k, v);
                }
            }
            Ok(PySupervisor {
                router: None,
                config: Some(config),
                agents: Vec::new(),
                max_rounds_val: 10,
                finish_marker_val: "FINISH".to_string(),
            })
        } else {
            let is_callable = Python::with_gil(|py| name_or_router.bind(py).is_callable());
            if !is_callable {
                return Err(crate::error::AgentExecutionError::new_err(
                    "First argument must be a callable router (state -> str) or a supervisor name (str) when children= is provided"
                ));
            }
            Ok(PySupervisor {
                router: Some(name_or_router),
                config: None,
                agents: Vec::new(),
                max_rounds_val: 10,
                finish_marker_val: "FINISH".to_string(),
            })
        }
    }

    /// Create a supervisor from a SupervisorNodeConfig (strategy-based mode).
    ///
    /// This enables all 11 orchestration strategies.
    #[staticmethod]
    fn from_config(config: &PySupervisorNodeConfig) -> Self {
        PySupervisor {
            router: None,
            config: Some(config.inner.clone()),
            agents: Vec::new(),
            max_rounds_val: 10,
            finish_marker_val: "FINISH".to_string(),
        }
    }

    /// Add a named agent (StateGraph or Python callable).
    fn add_agent(&mut self, py: Python<'_>, name: &str, agent: PyObject) -> PyResult<()> {
        // Try to extract as PyStateGraph first
        if let Ok(graph) = agent.extract::<PyStateGraph>(py) {
            self.agents
                .push((name.to_string(), AgentChild::Graph(graph.inner.clone())));
        } else if agent.bind(py).is_callable() {
            self.agents
                .push((name.to_string(), AgentChild::Callable(agent.clone_ref(py))));
        } else {
            return Err(crate::error::AgentExecutionError::new_err(
                "agent must be a StateGraph or a callable (state) -> state",
            ));
        }
        Ok(())
    }

    /// Set the maximum number of routing rounds (simple mode only).
    fn max_rounds(&mut self, rounds: usize) {
        self.max_rounds_val = rounds;
    }

    /// Set the finish marker string (default: "FINISH", simple mode only).
    fn finish_marker(&mut self, marker: &str) {
        self.finish_marker_val = marker.to_string();
    }

    /// Run the supervisor loop.
    ///
    /// Accepts either a native `State` object or a plain Python `dict`.
    /// In simple mode (created with router), uses the Python router function.
    /// In strategy mode (created with from_config), delegates to the Rust SupervisorNode.
    fn run(&self, state: &Bound<'_, PyAny>) -> PyResult<PyState> {
        let py_state = if let Ok(dict) = state.downcast::<pyo3::types::PyDict>() {
            let inner = crate::graph::pydict_to_dynstate(dict)?;
            PyState { inner }
        } else {
            state.extract::<PyState>()?
        };

        if let Some(ref config) = self.config {
            self.run_strategy(config, &py_state)
        } else if let Some(ref router) = self.router {
            self.run_router(router, &py_state)
        } else {
            Err(crate::error::AgentExecutionError::new_err(
                "Supervisor has no router or config",
            ))
        }
    }

    /// Get the list of registered agent names.
    fn agent_names(&self) -> Vec<String> {
        self.agents.iter().map(|(n, _)| n.clone()).collect()
    }

    fn __repr__(&self) -> String {
        let names: Vec<&str> = self.agents.iter().map(|(n, _)| n.as_str()).collect();
        let mode = if self.config.is_some() {
            "strategy"
        } else {
            "router"
        };
        format!("Supervisor(mode={}, agents={:?})", mode, names)
    }
}

impl PySupervisor {
    /// Strategy-based execution using the real Rust SupervisorNode.
    fn run_strategy(&self, config: &SupervisorNodeConfig, state: &PyState) -> PyResult<PyState> {
        // Issue #14: validate ALL declared children exist before executing any of
        // them. Collect every missing name so the error lists them all at once
        // rather than stopping at the first mismatch.
        let registered: Vec<&str> = self.agents.iter().map(|(n, _)| n.as_str()).collect();
        let missing: Vec<&str> = config
            .children
            .iter()
            .filter(|c| !self.agents.iter().any(|(n, _)| n == *c))
            .map(|c| c.as_str())
            .collect();
        if !missing.is_empty() {
            return Err(crate::error::AgentExecutionError::new_err(format!(
                "Supervisor: the following children declared in the config were not \
                 registered via add_agent(): {:?}. Registered agents: {:?}",
                missing, registered
            )));
        }

        // Build PluggableNode children from registered agents
        let mut children: Vec<Arc<dyn PluggableNode<DynState>>> = Vec::new();
        for child_name in &config.children {
            let agent = self.agents.iter().find(|(n, _)| n == child_name);
            match agent {
                Some((name, AgentChild::Graph(graph))) => {
                    children.push(Arc::new(StateGraphAsPluggable {
                        name: name.clone(),
                        graph: graph.clone(),
                    }));
                }
                Some((name, AgentChild::Callable(func))) => {
                    let func_clone = Python::with_gil(|py| func.clone_ref(py));
                    children.push(Arc::new(PyCallableAsPluggable {
                        name: name.clone(),
                        func: func_clone,
                    }));
                }
                None => {
                    return Err(crate::error::AgentExecutionError::new_err(format!(
                        "Supervisor internal error: child '{}' was not found in the agents \
                         registry after pre-flight validation. This is a bug — please report it.",
                        child_name
                    )));
                }
            }
        }

        // Create the real Rust SupervisorNode
        let supervisor = SupervisorNode::from_config(config.clone(), children).map_err(|e| {
            crate::error::AgentExecutionError::new_err(format!(
                "SupervisorNode config error: {}",
                e
            ))
        })?;

        // Execute — release the GIL before entering the async runtime so that
        // Python node callbacks invoked via spawn_blocking can acquire it.
        // Without this, spawn_blocking threads deadlock waiting for the GIL
        // while the calling thread holds it blocked on run_async.
        let state_clone = state.inner.clone();
        let result = Python::with_gil(|py| {
            py.allow_threads(|| {
                crate::run_async(async move {
                    use flowgentra_ai::core::node::nodes_trait::PluggableNode;
                    supervisor.run(state_clone).await
                })
            })
        });

        match result {
            Ok(output) => {
                if output.success {
                    Ok(PyState {
                        inner: output.state,
                    })
                } else {
                    // Store error in state but still return it
                    let s = output.state;
                    if let Some(err) = &output.error {
                        s.set("__supervisor_error__", serde_json::json!(err));
                    }
                    Ok(PyState { inner: s })
                }
            }
            Err(e) => Err(crate::error::AgentExecutionError::new_err(format!(
                "Supervisor execution error: {}",
                e
            ))),
        }
    }

    /// Simple router-based execution (backwards compatible with original API).
    fn run_router(&self, router: &PyObject, state: &PyState) -> PyResult<PyState> {
        let router = Python::with_gil(|py| router.clone_ref(py));
        let finish_marker = self.finish_marker_val.clone();
        let max_rounds = self.max_rounds_val;
        let mut current_state = state.inner.clone();

        for _round in 0..max_rounds {
            let next_agent = Python::with_gil(|py| -> PyResult<String> {
                let py_state = PyState {
                    inner: current_state.clone(),
                };
                let result = router.call1(py, (py_state,))?;
                result.extract::<String>(py)
            })?;

            if next_agent == finish_marker {
                return Ok(PyState {
                    inner: current_state,
                });
            }

            let agent = self.agents.iter().find(|(n, _)| n == &next_agent);
            match agent {
                Some((_, AgentChild::Graph(graph))) => {
                    let state_for_invoke = current_state.clone();
                    current_state = Python::with_gil(|py| {
                        py.allow_threads(|| crate::run_async(graph.invoke(state_for_invoke)))
                    })
                    .map_err(|e| crate::error::AgentExecutionError::new_err(format!("{}", e)))?;
                }
                Some((_, AgentChild::Callable(func))) => {
                    current_state = Python::with_gil(|py| -> PyResult<DynState> {
                        let f = func.clone_ref(py);
                        let py_state = PyState {
                            inner: current_state.clone(),
                        };
                        let py_result = f.call1(py, (py_state,))?;
                        let result_state: PyState = py_result.extract(py)?;
                        Ok(result_state.inner)
                    })?;
                }
                None => {
                    let names: Vec<&str> = self.agents.iter().map(|(n, _)| n.as_str()).collect();
                    return Err(crate::error::AgentExecutionError::new_err(format!(
                        "Supervisor: agent '{}' not found. Available: {:?}",
                        next_agent, names
                    )));
                }
            }
        }

        Err(crate::error::AgentExecutionError::new_err(format!(
            "Supervisor exceeded max rounds ({})",
            max_rounds
        )))
    }
}
