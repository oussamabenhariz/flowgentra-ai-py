//! Python bindings for advanced node configs (Loop, Parallel, Subgraph, Join)

use pyo3::prelude::*;

use flowgentra_ai::core::node::advanced_nodes::{
    BranchConfig, JoinNodeConfig, JoinType, LoopNodeConfig, MergeStrategy, ParallelNodeConfig,
    SubgraphNodeConfig,
};

use crate::py_to_json;

// ─── PyJoinType ──────────────────────────────────────────────────────────────

/// How to join parallel branches.
#[pyclass(name = "JoinType")]
#[derive(Clone)]
pub struct PyJoinType {
    pub(crate) inner: JoinType,
}

#[pymethods]
impl PyJoinType {
    /// Wait for all branches to complete.
    #[staticmethod]
    fn wait_all() -> Self {
        PyJoinType { inner: JoinType::WaitAll }
    }

    /// Continue as soon as any branch completes.
    #[staticmethod]
    fn wait_any() -> Self {
        PyJoinType { inner: JoinType::WaitAny }
    }

    /// Wait for a specific number of branches.
    #[staticmethod]
    fn wait_count(n: usize) -> Self {
        PyJoinType { inner: JoinType::WaitCount(n) }
    }

    /// Timeout-based join.
    #[staticmethod]
    fn wait_timeout() -> Self {
        PyJoinType { inner: JoinType::WaitTimeout }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            JoinType::WaitAll => "JoinType.WaitAll".to_string(),
            JoinType::WaitAny => "JoinType.WaitAny".to_string(),
            JoinType::WaitCount(n) => format!("JoinType.WaitCount({})", n),
            JoinType::WaitTimeout => "JoinType.WaitTimeout".to_string(),
        }
    }
}

// ─── PyMergeStrategy ─────────────────────────────────────────────────────────

/// Strategy for merging parallel branch results.
#[pyclass(name = "MergeStrategy")]
#[derive(Clone)]
pub struct PyMergeStrategy {
    pub(crate) inner: MergeStrategy,
}

#[pymethods]
impl PyMergeStrategy {
    #[staticmethod]
    fn combine() -> Self {
        PyMergeStrategy { inner: MergeStrategy::Combine }
    }

    #[staticmethod]
    fn first() -> Self {
        PyMergeStrategy { inner: MergeStrategy::First }
    }

    #[staticmethod]
    fn last() -> Self {
        PyMergeStrategy { inner: MergeStrategy::Last }
    }

    #[staticmethod]
    fn by_branch() -> Self {
        PyMergeStrategy { inner: MergeStrategy::ByBranch }
    }

    #[staticmethod]
    fn custom() -> Self {
        PyMergeStrategy { inner: MergeStrategy::Custom }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            MergeStrategy::Combine => "MergeStrategy.Combine",
            MergeStrategy::First => "MergeStrategy.First",
            MergeStrategy::Last => "MergeStrategy.Last",
            MergeStrategy::ByBranch => "MergeStrategy.ByBranch",
            MergeStrategy::Custom => "MergeStrategy.Custom",
        }
        .to_string()
    }
}

// ─── PyBranchConfig ──────────────────────────────────────────────────────────

/// Configuration for a single branch in parallel execution.
#[pyclass(name = "BranchConfig")]
#[derive(Clone)]
pub struct PyBranchConfig {
    pub(crate) inner: BranchConfig,
}

#[pymethods]
impl PyBranchConfig {
    #[new]
    fn new(name: &str, handler: &str) -> Self {
        PyBranchConfig {
            inner: BranchConfig::new(name, handler),
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn handler(&self) -> String {
        self.inner.handler.clone()
    }

    fn __repr__(&self) -> String {
        format!("BranchConfig(name='{}', handler='{}')", self.inner.name, self.inner.handler)
    }
}

// ─── PyLoopNodeConfig ────────────────────────────────────────────────────────

/// Configuration for a loop node.
///
/// Example:
///     loop_cfg = LoopNodeConfig("retry_handler", max_iterations=3)
///     loop_cfg.break_condition = "is_done"
#[pyclass(name = "LoopNodeConfig")]
#[derive(Clone)]
pub struct PyLoopNodeConfig {
    pub(crate) inner: LoopNodeConfig,
}

#[pymethods]
impl PyLoopNodeConfig {
    #[new]
    #[pyo3(signature = (handler, max_iterations=3, break_condition=None, accumulate=false))]
    fn new(
        handler: &str,
        max_iterations: usize,
        break_condition: Option<String>,
        accumulate: bool,
    ) -> Self {
        let mut cfg = LoopNodeConfig::new(handler).with_max_iterations(max_iterations);
        if let Some(cond) = break_condition {
            cfg = cfg.with_break_condition(cond);
        }
        cfg = cfg.with_accumulation(accumulate);
        PyLoopNodeConfig { inner: cfg }
    }

    #[getter]
    fn handler(&self) -> String {
        self.inner.handler.clone()
    }

    #[getter]
    fn max_iterations(&self) -> usize {
        self.inner.max_iterations
    }

    #[getter]
    fn break_condition(&self) -> Option<String> {
        self.inner.break_condition.clone()
    }

    #[getter]
    fn accumulate_results(&self) -> bool {
        self.inner.accumulate_results
    }

    fn __repr__(&self) -> String {
        format!(
            "LoopNodeConfig(handler='{}', max_iterations={})",
            self.inner.handler, self.inner.max_iterations
        )
    }
}

// ─── PyParallelNodeConfig ────────────────────────────────────────────────────

/// Configuration for parallel branch execution.
///
/// Example:
///     parallel = ParallelNodeConfig("parallel_step")
///     parallel.add_branch("handler_a")
///     parallel.add_named_branch("b", "handler_b")
#[pyclass(name = "ParallelNodeConfig")]
#[derive(Clone)]
pub struct PyParallelNodeConfig {
    pub(crate) inner: ParallelNodeConfig,
}

#[pymethods]
impl PyParallelNodeConfig {
    #[new]
    fn new(name: &str) -> Self {
        PyParallelNodeConfig {
            inner: ParallelNodeConfig::new(name),
        }
    }

    /// Add an auto-named branch.
    fn add_branch(&mut self, handler: &str) {
        self.inner = std::mem::take(&mut self.inner).add_branch(handler);
    }

    /// Add a named branch.
    fn add_named_branch(&mut self, name: &str, handler: &str) {
        self.inner = std::mem::take(&mut self.inner).add_named_branch(name, handler);
    }

    /// Set join type.
    fn set_join_type(&mut self, join_type: &PyJoinType) {
        self.inner = std::mem::take(&mut self.inner).with_join_type(join_type.inner);
    }

    /// Set timeout in milliseconds.
    fn set_timeout(&mut self, timeout_ms: u64) {
        self.inner = std::mem::take(&mut self.inner).with_timeout(timeout_ms);
    }

    /// Set continue on error.
    fn set_continue_on_error(&mut self, value: bool) {
        self.inner = std::mem::take(&mut self.inner).with_continue_on_error(value);
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn branches(&self) -> Vec<PyBranchConfig> {
        self.inner
            .branches
            .iter()
            .map(|b| PyBranchConfig { inner: b.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ParallelNodeConfig(name='{}', branches={})",
            self.inner.name,
            self.inner.branches.len()
        )
    }
}

// ─── PySubgraphNodeConfig ────────────────────────────────────────────────────

/// Configuration for subgraph nodes.
///
/// Example:
///     sub = SubgraphNodeConfig("validation", "validation.yaml")
///     sub.set_parameter("strict", True)
#[pyclass(name = "SubgraphNodeConfig")]
#[derive(Clone)]
pub struct PySubgraphNodeConfig {
    pub(crate) inner: SubgraphNodeConfig,
}

#[pymethods]
impl PySubgraphNodeConfig {
    #[new]
    fn new(name: &str, subgraph_path: &str) -> Self {
        PySubgraphNodeConfig {
            inner: SubgraphNodeConfig::new(name, subgraph_path),
        }
    }

    /// Add a parameter.
    fn set_parameter(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let json_val = py_to_json(value)?;
        self.inner.parameters.insert(key.to_string(), json_val);
        Ok(())
    }

    /// Set whether to inherit parent state.
    fn set_inherit_state(&mut self, inherit: bool) {
        self.inner.inherit_state = inherit;
    }

    /// Add input key mapping.
    fn add_input_mapping(&mut self, key: &str) {
        self.inner.map_input_keys.push(key.to_string());
    }

    /// Add output key mapping.
    fn add_output_mapping(&mut self, key: &str) {
        self.inner.map_output_keys.push(key.to_string());
    }

    /// Set timeout in ms.
    fn set_timeout(&mut self, timeout_ms: u64) {
        self.inner.timeout_ms = Some(timeout_ms);
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn subgraph_path(&self) -> String {
        self.inner.subgraph_path.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SubgraphNodeConfig(name='{}', path='{}')",
            self.inner.name, self.inner.subgraph_path
        )
    }
}

// ─── PyJoinNodeConfig ────────────────────────────────────────────────────────

/// Configuration for join points in parallel execution.
///
/// Example:
///     join = JoinNodeConfig("merge_results")
///     join.set_strategy(MergeStrategy.by_branch())
#[pyclass(name = "JoinNodeConfig")]
#[derive(Clone)]
pub struct PyJoinNodeConfig {
    pub(crate) inner: JoinNodeConfig,
}

#[pymethods]
impl PyJoinNodeConfig {
    #[new]
    fn new(name: &str) -> Self {
        PyJoinNodeConfig {
            inner: JoinNodeConfig::new(name),
        }
    }

    /// Set merge strategy.
    fn set_strategy(&mut self, strategy: &PyMergeStrategy) {
        self.inner.merge_strategy = strategy.inner.clone();
    }

    /// Add a key to merge.
    fn add_merge_key(&mut self, key: &str) {
        self.inner.merge_keys.push(key.to_string());
    }

    /// Set fail on error.
    fn set_fail_on_error(&mut self, fail: bool) {
        self.inner.fail_on_error = fail;
    }

    /// Set custom merge function name.
    fn set_merge_function(&mut self, function: &str) {
        self.inner.merge_function = Some(function.to_string());
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    fn __repr__(&self) -> String {
        format!("JoinNodeConfig(name='{}')", self.inner.name)
    }
}

