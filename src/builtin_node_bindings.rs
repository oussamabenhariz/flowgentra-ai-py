//! Python bindings for built-in node types (Retry, Timeout, LLM Node)
//!
//! RetryGraphNode and TimeoutGraphNode wrap Python callables with retry/timeout
//! logic. They use the LangGraph-style dict I/O pattern: the Python callable
//! receives the full state dict and returns a partial update dict.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{AgentExecutionError, ValidationError};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use flowgentra_ai::core::state::{Context, DynState, DynStateUpdate};
use flowgentra_ai::core::state_graph::node::Node;
use flowgentra_ai::core::state_graph::StateGraphError;

use crate::graph::{dynstate_to_pydict, pydict_to_dynstate};

// ─── PyRetryNode (standalone Python class) ──────────────────────────────────

/// A standalone retry wrapper for a Python callable (not graph-embedded).
///
/// For graph use, prefer builder.add_retry_node(...) instead.
///
/// The inner callable receives the current state dict and returns a partial
/// update dict. Retries up to `max_retries` times with exponential backoff.
///
/// Example:
///     def fetch(state: dict) -> dict:
///         return {"result": call_api(state["query"])}
///
///     retry = RetryNode("fetch", fetch, max_retries=3)
///     result = retry.run({"query": "hello", "result": None})
#[pyclass(name = "RetryNode")]
pub struct PyRetryNode {
    name: String,
    func: PyObject,
    max_retries: usize,
    backoff_ms: u64,
    backoff_multiplier: f64,
    max_backoff_ms: u64,
}

#[pymethods]
impl PyRetryNode {
    #[new]
    #[pyo3(signature = (name, func, max_retries=3, backoff_ms=1000, backoff_multiplier=2.0, max_backoff_ms=30000))]
    fn new(
        name: &str,
        func: PyObject,
        max_retries: usize,
        backoff_ms: u64,
        backoff_multiplier: f64,
        max_backoff_ms: u64,
    ) -> Self {
        PyRetryNode {
            name: name.to_string(),
            func,
            max_retries,
            backoff_ms,
            backoff_multiplier,
            max_backoff_ms,
        }
    }

    /// Execute the retry node with the given state dict.
    ///
    /// Args:
    ///     state_dict: Full current state as a plain dict.
    ///
    /// Returns:
    ///     Merged state dict after applying the partial update from the callable.
    fn run(&self, py: Python<'_>, state_dict: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let func = self.func.clone_ref(py);
        let max_retries = self.max_retries;
        let mut backoff = self.backoff_ms;
        let multiplier = self.backoff_multiplier;
        let max_backoff = self.max_backoff_ms;
        let name = self.name.clone();

        // Build a DynState from the input dict (no schema — standalone mode)
        let current_state = pydict_to_dynstate(state_dict)?;

        let result = crate::run_async(async move {
            let mut last_error = String::new();
            for attempt in 0..=max_retries {
                let call_result = Python::with_gil(|py| -> PyResult<DynState> {
                    let f = func.clone_ref(py);
                    let state_d = dynstate_to_pydict(py, &current_state)?;
                    let py_result = f.call1(py, (state_d,))?;
                    let update = py_result.bind(py).downcast::<PyDict>().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "RetryNode '{}' callable must return a dict", name
                        ))
                    })?.clone();
                    // Merge partial update (no schema validation in standalone mode)
                    let merged = current_state.clone();
                    for (k, v) in update.iter() {
                        let key: String = k.extract()?;
                        let val = crate::py_to_json(&v)?;
                        merged.set(key, val);
                    }
                    Ok(merged)
                });

                match call_result {
                    Ok(new_state) => return Ok(new_state),
                    Err(e) => {
                        last_error = format!("{}", e);
                        if attempt < max_retries {
                            tokio::time::sleep(Duration::from_millis(backoff)).await;
                            backoff = ((backoff as f64 * multiplier) as u64).min(max_backoff);
                        }
                    }
                }
            }
            Err(last_error)
        });

        match result {
            Ok(new_state) => Python::with_gil(|py| {
                dynstate_to_pydict(py, &new_state).map(|d| d.into())
            }),
            Err(e) => Err(AgentExecutionError::new_err(format!(
                "RetryNode '{}' failed after {} retries: {}",
                self.name, self.max_retries, e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RetryNode(name='{}', max_retries={}, backoff_ms={})",
            self.name, self.max_retries, self.backoff_ms
        )
    }
}

// ─── PyTimeoutNode (standalone Python class) ────────────────────────────────

/// A standalone timeout wrapper for a Python callable (not graph-embedded).
///
/// For graph use, prefer builder.add_timeout_node(...) instead.
///
/// The inner callable receives the current state dict and returns a partial
/// update dict. Raises an error or skips based on `on_timeout`.
///
/// Example:
///     timeout = TimeoutNode("slow_op", slow_fn, timeout_ms=5000)
///     result = timeout.run({"query": "hello", "result": None})
#[pyclass(name = "TimeoutNode")]
pub struct PyTimeoutNode {
    name: String,
    func: PyObject,
    timeout_ms: u64,
    on_timeout: String,
}

#[pymethods]
impl PyTimeoutNode {
    #[new]
    #[pyo3(signature = (name, func, timeout_ms, on_timeout="error"))]
    fn new(name: &str, func: PyObject, timeout_ms: u64, on_timeout: &str) -> Self {
        PyTimeoutNode {
            name: name.to_string(),
            func,
            timeout_ms,
            on_timeout: on_timeout.to_string(),
        }
    }

    /// Execute the timeout node with the given state dict.
    ///
    /// Args:
    ///     state_dict: Full current state as a plain dict.
    ///
    /// Returns:
    ///     Merged state dict, or original state if on_timeout="skip".
    fn run(&self, py: Python<'_>, state_dict: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let func = self.func.clone_ref(py);
        let timeout = Duration::from_millis(self.timeout_ms);
        let on_timeout = self.on_timeout.clone();
        let name = self.name.clone();
        let current_state = pydict_to_dynstate(state_dict)?;

        let result = crate::run_async(async move {
            let inner_future = async {
                Python::with_gil(|py| -> PyResult<DynState> {
                    let f = func.clone_ref(py);
                    let state_d = dynstate_to_pydict(py, &current_state)?;
                    let py_result = f.call1(py, (state_d,))?;
                    let update = py_result.downcast_bound::<PyDict>(py).map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "TimeoutNode '{}' callable must return a dict", name
                        ))
                    })?;
                    let merged = current_state.clone();
                    for (k, v) in update.iter() {
                        let key: String = k.extract()?;
                        let val = crate::py_to_json(&v)?;
                        merged.set(key, val);
                    }
                    Ok(merged)
                })
            };

            match tokio::time::timeout(timeout, inner_future).await {
                Ok(Ok(new_state)) => Ok(new_state),
                Ok(Err(e)) => Err(format!("Node '{}' error: {}", name, e)),
                Err(_) => match on_timeout.as_str() {
                    "skip" => Ok(current_state),
                    _ => Err(format!(
                        "Node '{}' timed out after {}ms",
                        name,
                        timeout.as_millis()
                    )),
                },
            }
        });

        match result {
            Ok(new_state) => Python::with_gil(|py| {
                dynstate_to_pydict(py, &new_state).map(|d| d.into())
            }),
            Err(e) => Err(AgentExecutionError::new_err(e)),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TimeoutNode(name='{}', timeout_ms={}, on_timeout='{}')",
            self.name, self.timeout_ms, self.on_timeout
        )
    }
}

// ─── Graph node adapters ────────────────────────────────────────────────────

/// Wraps a Python callable with retry logic as a Node<DynState>.
pub(crate) struct RetryGraphNode {
    pub name: String,
    pub func: PyObject,
    pub max_retries: usize,
    pub backoff_ms: u64,
    pub backoff_multiplier: f64,
    pub max_backoff_ms: u64,
    pub schema_fields: Arc<Vec<String>>,
    /// O(1) lookup set (issue #16).
    pub schema_set: Arc<HashSet<String>>,
}

#[async_trait::async_trait]
impl Node<DynState> for RetryGraphNode {
    async fn execute(&self, state: &DynState, _ctx: &Context) -> Result<DynStateUpdate, StateGraphError> {
        let schema_set = self.schema_set.clone();
        let schema_fields = self.schema_fields.clone();
        let node_name = self.name.clone();
        let state_snap = state.clone();
        let func = Python::with_gil(|py| self.func.clone_ref(py));
        let mut backoff = self.backoff_ms;
        let mut last_error = String::new();

        for attempt in 0..=self.max_retries {
            // Issue #9: run the Python call on a blocking thread so the Tokio
            // pool is not stalled by GIL acquisition.
            //
            // Issue #13: the Python *execution* is retryable (network errors,
            // transient failures). Schema validation is NOT — a wrong return key
            // will be wrong on every attempt, so we validate outside the retry
            // loop and short-circuit immediately on schema errors.
            let func_clone = Python::with_gil(|py| func.clone_ref(py));
            let state_clone = state_snap.clone();
            let nn = node_name.clone();

            // Phase 1 (retryable): call the Python callable.
            let phase1 = tokio::task::spawn_blocking(move || {
                Python::with_gil(|py| -> PyResult<pyo3::Py<PyDict>> {
                    let state_d = dynstate_to_pydict(py, &state_clone)?;
                    let py_result = func_clone.call1(py, (state_d,))?;
                    py_result.downcast_bound::<PyDict>(py)
                        .map(|d| d.clone().unbind())
                        .map_err(|_| {
                            pyo3::exceptions::PyTypeError::new_err(format!(
                                "RetryNode '{}' callable must return a dict", nn
                            ))
                        })
                })
            })
            .await;

            match phase1 {
                Ok(Ok(py_dict)) => {
                    // Phase 2 (non-retryable): schema validation and update build.
                    return Python::with_gil(|py| {
                        let dict = py_dict.bind(py);
                        for (k, _) in dict.iter() {
                            let key: String = k.extract()?;
                            if !schema_set.contains(&key) {
                                return Err(ValidationError::new_err(format!(
                                    "RetryNode '{}' returned unknown key '{}'. \
                                     Valid keys: {:?}",
                                    node_name, key, schema_fields
                                )));
                            }
                        }
                        let mut su = DynStateUpdate::new();
                        for (k, v) in dict.iter() {
                            let key: String = k.extract()?;
                            let val = crate::py_to_json(&v)?;
                            su.insert(key, val);
                        }
                        Ok(su)
                    })
                    .map_err(|e| StateGraphError::ExecutionError {
                        node: self.name.clone(),
                        reason: format!("Schema validation: {}", e),
                    });
                }
                Ok(Err(py_err)) => {
                    // Python raised an exception — retry if attempts remain.
                    last_error = format!("{}", py_err);
                    if attempt < self.max_retries {
                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                        backoff = ((backoff as f64 * self.backoff_multiplier) as u64)
                            .min(self.max_backoff_ms);
                    }
                }
                Err(join_err) => {
                    // The blocking thread panicked — don't retry.
                    return Err(StateGraphError::ExecutionError {
                        node: self.name.clone(),
                        reason: format!("spawn_blocking panic: {}", join_err),
                    });
                }
            }
        }

        Err(StateGraphError::ExecutionError {
            node: self.name.clone(),
            reason: format!("Failed after {} retries: {}", self.max_retries, last_error),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Wraps a Python callable with a timeout as a Node<DynState>.
pub(crate) struct TimeoutGraphNode {
    pub name: String,
    pub func: PyObject,
    pub timeout_ms: u64,
    pub on_timeout: String,
    pub schema_fields: Arc<Vec<String>>,
    /// O(1) lookup set (issue #16).
    pub schema_set: Arc<HashSet<String>>,
}

#[async_trait::async_trait]
impl Node<DynState> for TimeoutGraphNode {
    async fn execute(&self, state: &DynState, _ctx: &Context) -> Result<DynStateUpdate, StateGraphError> {
        // Issue #1: the previous implementation wrapped `Python::with_gil(...)` —
        // a *synchronous* call — in an async block and then passed it to
        // `tokio::time::timeout`.  Because there are no `.await` points inside
        // `with_gil`, the timeout could never fire while Python was executing.
        //
        // Fix: run the Python call on a blocking thread via `spawn_blocking`.
        // `tokio::time::timeout` can now cancel the *future returned by
        // spawn_blocking* at its internal poll boundary, so the deadline is
        // actually enforced.
        //
        // Issue #9: offloading to spawn_blocking also prevents the GIL from
        // stalling the Tokio worker pool.
        let func = Python::with_gil(|py| self.func.clone_ref(py));
        let timeout = Duration::from_millis(self.timeout_ms);
        let state_clone = state.clone();
        let schema_set = self.schema_set.clone();
        let schema_fields = self.schema_fields.clone();
        let node_name = self.name.clone();
        let on_timeout = self.on_timeout.clone();

        let blocking_task = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<DynStateUpdate> {
                let state_dict = dynstate_to_pydict(py, &state_clone)?;
                let py_result = func.call1(py, (state_dict,))?;
                let update = py_result.downcast_bound::<PyDict>(py).map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "TimeoutNode '{}' callable must return a dict", node_name
                    ))
                })?;
                // O(1) schema validation (issue #16)
                for (k, _) in update.iter() {
                    let key: String = k.extract()?;
                    if !schema_set.contains(&key) {
                        return Err(ValidationError::new_err(format!(
                            "TimeoutNode '{}' returned unknown key '{}'. \
                             Valid keys: {:?}",
                            node_name, key, schema_fields
                        )));
                    }
                }
                let mut su = DynStateUpdate::new();
                for (k, v) in update.iter() {
                    let key: String = k.extract()?;
                    let val = crate::py_to_json(&v)?;
                    su.insert(key, val);
                }
                Ok(su)
            })
        });

        match tokio::time::timeout(timeout, blocking_task).await {
            Ok(Ok(Ok(update))) => Ok(update),
            Ok(Ok(Err(py_err))) => Err(StateGraphError::ExecutionError {
                node: self.name.clone(),
                reason: format!("Python node error: {}", py_err),
            }),
            Ok(Err(join_err)) => Err(StateGraphError::ExecutionError {
                node: self.name.clone(),
                reason: format!("spawn_blocking panic: {}", join_err),
            }),
            Err(_elapsed) => match on_timeout.as_str() {
                "skip" => Ok(DynStateUpdate::new()),
                _ => Err(StateGraphError::ExecutionError {
                    node: self.name.clone(),
                    reason: format!("Timed out after {}ms", self.timeout_ms),
                }),
            },
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ─── LoopGraphNode ──────────────────────────────────────────────────────────

/// Wraps a Python callable with loop logic as a Node<DynState>.
///
/// Runs `func` up to `max_iterations` times.  After each iteration, if
/// `break_condition` is set, calls it with the current state; the loop exits
/// early when it returns `True`.  Returns the changed fields as a partial
/// `DynStateUpdate`.
pub(crate) struct LoopGraphNode {
    pub name: String,
    pub func: PyObject,
    pub max_iterations: usize,
    /// Optional `(state: dict) -> bool` callable.
    pub break_condition: Option<PyObject>,
    pub schema_set: Arc<HashSet<String>>,
}

#[async_trait::async_trait]
impl Node<DynState> for LoopGraphNode {
    async fn execute(&self, state: &DynState, _ctx: &Context) -> Result<DynStateUpdate, StateGraphError> {
        // Snapshot as HashMap so we can mutate between iterations.
        let mut running: HashMap<String, serde_json::Value> = state
            .keys()
            .into_iter()
            .filter_map(|k| state.get(&k).map(|v| (k, v)))
            .collect();
        let original = running.clone();

        let func = Python::with_gil(|py| self.func.clone_ref(py));
        let break_cond = self.break_condition.as_ref()
            .map(|bc| Python::with_gil(|py| bc.clone_ref(py)));
        let schema_set = self.schema_set.clone();
        let node_name = self.name.clone();

        for _ in 0..self.max_iterations {
            let func_clone = Python::with_gil(|py| func.clone_ref(py));
            let current = running.clone();
            let ss = schema_set.clone();
            let nn = node_name.clone();

            let update: HashMap<String, serde_json::Value> =
                tokio::task::spawn_blocking(move || {
                    Python::with_gil(|py| -> PyResult<HashMap<String, serde_json::Value>> {
                        let state_dict = pyo3::types::PyDict::new_bound(py);
                        for (k, v) in &current {
                            state_dict.set_item(k, crate::json_to_py(py, v)?)?;
                        }
                        let py_result = func_clone.call1(py, (state_dict,))?;
                        let update_dict = py_result.downcast_bound::<pyo3::types::PyDict>(py)
                            .map_err(|_| pyo3::exceptions::PyTypeError::new_err(format!(
                                "LoopNode '{}' callable must return a dict", nn
                            )))?;
                        let mut result = HashMap::new();
                        for (k, v) in update_dict.iter() {
                            let key: String = k.extract()?;
                            if !ss.contains(&key) {
                                return Err(ValidationError::new_err(format!(
                                    "LoopNode '{}' returned unknown key '{}'", nn, key
                                )));
                            }
                            result.insert(key, crate::py_to_json(&v)?);
                        }
                        Ok(result)
                    })
                })
                .await
                .map_err(|e| StateGraphError::ExecutionError {
                    node: node_name.clone(),
                    reason: format!("spawn_blocking panic: {}", e),
                })?
                .map_err(|e| StateGraphError::ExecutionError {
                    node: node_name.clone(),
                    reason: format!("Python loop error: {}", e),
                })?;

            for (k, v) in update {
                running.insert(k, v);
            }

            // Check break condition after applying the update.
            if let Some(ref bc) = break_cond {
                let bc_clone = Python::with_gil(|py| bc.clone_ref(py));
                let check_state = running.clone();
                let nn2 = node_name.clone();

                let should_break: bool = tokio::task::spawn_blocking(move || {
                    Python::with_gil(|py| -> PyResult<bool> {
                        let state_dict = pyo3::types::PyDict::new_bound(py);
                        for (k, v) in &check_state {
                            state_dict.set_item(k, crate::json_to_py(py, v)?)?;
                        }
                        let result = bc_clone.call1(py, (state_dict,))?;
                        result.extract(py)
                    })
                })
                .await
                .map_err(|e| StateGraphError::ExecutionError {
                    node: node_name.clone(),
                    reason: format!("break_condition panic: {}", e),
                })?
                .map_err(|e| StateGraphError::ExecutionError {
                    node: nn2,
                    reason: format!("break_condition error: {}", e),
                })?;

                if should_break {
                    break;
                }
            }
        }

        // Return only the fields that changed from the original state.
        let mut final_update = DynStateUpdate::new();
        for (key, new_val) in &running {
            if original.get(key) != Some::<serde_json::Value>(new_val.clone()).as_ref() {
                final_update.insert(key.clone(), new_val.clone());
            }
        }
        Ok(final_update)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ─── ParallelGraphNode ──────────────────────────────────────────────────────

/// Runs multiple Python callables concurrently as a single graph node.
///
/// All branches receive the same initial state.  Their returned partial updates
/// are merged with last-write-wins semantics (branches are awaited in declaration
/// order; later branches override earlier ones for the same key).
pub(crate) struct ParallelGraphNode {
    pub name: String,
    /// `(branch_name, callable)` pairs.
    pub branches: Vec<(String, PyObject)>,
    pub schema_set: Arc<HashSet<String>>,
}

#[async_trait::async_trait]
impl Node<DynState> for ParallelGraphNode {
    async fn execute(&self, state: &DynState, _ctx: &Context) -> Result<DynStateUpdate, StateGraphError> {
        let schema_set = self.schema_set.clone();
        let node_name = self.name.clone();

        // Spawn all branches onto blocking threads so they run in parallel.
        let mut handles = Vec::with_capacity(self.branches.len());
        for (branch_name, func) in &self.branches {
            let func_clone = Python::with_gil(|py| func.clone_ref(py));
            let state_clone = state.clone();
            let ss = schema_set.clone();
            let bn = branch_name.clone();
            let nn = node_name.clone();

            let handle = tokio::task::spawn_blocking(move || {
                Python::with_gil(|py| -> PyResult<HashMap<String, serde_json::Value>> {
                    let state_dict = dynstate_to_pydict(py, &state_clone)?;
                    let py_result = func_clone.call1(py, (state_dict,))?;
                    let update_dict = py_result.downcast_bound::<pyo3::types::PyDict>(py)
                        .map_err(|_| pyo3::exceptions::PyTypeError::new_err(format!(
                            "ParallelNode '{}' branch '{}' must return a dict", nn, bn
                        )))?;
                    let mut result = HashMap::new();
                    for (k, v) in update_dict.iter() {
                        let key: String = k.extract()?;
                        if !ss.contains(&key) {
                            return Err(ValidationError::new_err(format!(
                                "ParallelNode '{}' branch '{}' returned unknown key '{}'",
                                nn, bn, key
                            )));
                        }
                        result.insert(key, crate::py_to_json(&v)?);
                    }
                    Ok(result)
                })
            });
            handles.push((branch_name.clone(), handle));
        }

        // All blocking tasks are running. Collect results; last-write-wins merge.
        let mut merged = DynStateUpdate::new();
        for (branch_name, handle) in handles {
            let result = handle
                .await
                .map_err(|e| StateGraphError::ExecutionError {
                    node: node_name.clone(),
                    reason: format!("Branch '{}' spawn_blocking panic: {}", branch_name, e),
                })?
                .map_err(|e| StateGraphError::ExecutionError {
                    node: node_name.clone(),
                    reason: format!("Branch '{}' error: {}", branch_name, e),
                })?;

            for (k, v) in result {
                merged.insert(k, v);
            }
        }

        Ok(merged)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Wraps an LLM as a graph node that reads a prompt key and writes a response key.
pub(crate) struct LLMGraphNode {
    pub name: String,
    pub llm: Arc<dyn flowgentra_ai::core::llm::LLM>,
    pub system_prompt: Option<String>,
    pub prompt_key: String,
    pub output_key: String,
}

#[async_trait::async_trait]
impl Node<DynState> for LLMGraphNode {
    async fn execute(&self, state: &DynState, _ctx: &Context) -> Result<DynStateUpdate, StateGraphError> {
        use flowgentra_ai::core::llm::Message;

        let user_prompt = state
            .get(&self.prompt_key)
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default();

        if user_prompt.is_empty() {
            return Err(StateGraphError::ExecutionError {
                node: self.name.clone(),
                reason: format!("No prompt found at state key '{}'", self.prompt_key),
            });
        }

        let mut messages = Vec::new();
        if let Some(ref sys) = self.system_prompt {
            messages.push(Message::system(sys));
        }
        messages.push(Message::user(&user_prompt));

        let response = self.llm.chat(messages).await.map_err(|e| {
            StateGraphError::ExecutionError {
                node: self.name.clone(),
                reason: format!("LLM call failed: {}", e),
            }
        })?;

        let mut update = DynStateUpdate::new();
        update.insert(self.output_key.clone(), serde_json::json!(response.content));
        Ok(update)
    }

    fn name(&self) -> &str {
        &self.name
    }
}
