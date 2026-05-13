//! Python bindings for StateGraph (LangGraph-compatible API)
//!
//! StateGraph(State) takes a TypedDict subclass defining the state schema.
//! Node callables receive the full state as a plain dict and return a partial
//! update dict. The graph merges partial updates back into the state.
//! graph.invoke({...}) accepts and returns plain Python dicts.

use crate::error::ValidationError;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use flowgentra_ai::core::middleware::Middleware;
use flowgentra_ai::core::observability::events::EventBroadcaster;
use flowgentra_ai::core::state::{Context, DynState, DynStateUpdate};
use flowgentra_ai::core::state_graph::{
    edge::END, node::Node, FileCheckpointer, StateGraph, StateGraphBuilder, StateGraphError,
};

use flowgentra_ai::core::observability::visualization::ExecutionTracer;

use crate::builtin_node_bindings::{LLMGraphNode, RetryGraphNode, TimeoutGraphNode};
use crate::channel::{apply_channel_reducer, ChannelType};
use crate::evaluation_node::{EvaluationGraphNode, PyEvaluationNodeConfig};
use crate::llm::PyLLM;
use crate::observability::PyExecutionTracer;
use crate::planner_node::create_planner_graph_node;
use crate::py_reducers::extract_reducers_from_class;

// ─── Error conversion ──────────────────────────────────────────────────────

fn sg_err_to_py(e: StateGraphError) -> PyErr {
    crate::error::to_py_err_state_graph(e)
}

// ─── Schema extraction ─────────────────────────────────────────────────────

/// Extract field names from a TypedDict class's __annotations__.
/// Walks the MRO to capture inherited fields.
fn extract_schema_fields(state_class: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut fields: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Walk __mro__ to collect annotations from this class and all parents
    if let Ok(mro) = state_class.getattr("__mro__") {
        if let Ok(list) = mro.downcast::<pyo3::types::PyList>() {
            for cls in list.iter() {
                // Skip built-in base types
                let skip = cls
                    .getattr("__name__")
                    .ok()
                    .and_then(|n: Bound<'_, PyAny>| n.extract::<String>().ok())
                    .map(|n: String| matches!(n.as_str(), "object" | "TypedDict" | "dict"))
                    .unwrap_or(false);
                if skip {
                    continue;
                }
                if let Ok(ann) = cls.getattr("__annotations__") {
                    if let Ok(dict) = ann.downcast::<PyDict>() {
                        for (k, _) in dict.iter() {
                            let key: String = k.extract()?;
                            if seen.insert(key.clone()) {
                                fields.push(key);
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: just check __annotations__ on the class itself
    if fields.is_empty() {
        let ann = state_class.getattr("__annotations__").map_err(|_| {
            ValidationError::new_err(
                "StateGraph requires a TypedDict subclass with type annotations.\n\
                 Example:\n\
                 \n\
                 from typing import TypedDict, List\n\
                 class State(TypedDict):\n\
                     messages: List[str]\n\
                     score: float\n\
                 \n\
                 builder = StateGraph(State)",
            )
        })?;
        let dict = ann
            .downcast::<PyDict>()
            .map_err(|_| ValidationError::new_err("__annotations__ must be a dict"))?;
        for (k, _) in dict.iter() {
            let key: String = k.extract()?;
            if seen.insert(key.clone()) {
                fields.push(key);
            }
        }
    }

    if fields.is_empty() {
        return Err(ValidationError::new_err(
            "State TypedDict must declare at least one field.",
        ));
    }

    Ok(fields)
}

// ─── Required-fields extraction ───────────────────────────────────────────

/// Extract the set of *required* fields from a TypedDict class.
///
/// TypedDict sets `__required_keys__` (a frozenset) that correctly reflects
/// `total=False`, `total=True`, and mixed `Required[…]`/`NotRequired[…]`
/// annotations.  We use this instead of treating every annotated field as
/// required, which would break `total=False` schemas.
///
/// Falls back to treating all annotated fields as required for non-TypedDict
/// classes that lack the attribute.
fn extract_required_fields(
    state_class: &Bound<'_, PyAny>,
    all_fields: &[String],
) -> HashSet<String> {
    if let Ok(required_keys) = state_class.getattr("__required_keys__") {
        if let Ok(iter) = required_keys.iter() {
            let mut set = HashSet::new();
            for item in iter.flatten() {
                if let Ok(key) = item.extract::<String>() {
                    set.insert(key);
                }
            }
            return set;
        }
    }
    // Fallback: class doesn't expose __required_keys__ — treat all as required.
    all_fields.iter().cloned().collect()
}

// ─── Subgraph delta helper ─────────────────────────────────────────────────

/// Compute what a subgraph node should emit for a single field.
///
/// Issue #2: returning ALL subgraph result keys as a raw update causes
/// double-counting for Append/Sum reducers — the parent re-applies the
/// reducer on top of already-accumulated values.
///
/// Strategy:
/// * Unchanged fields  → `None` (excluded from the update)
/// * Array extended at the tail (pure append) → only the new elements
/// * Any other changed field → the full new value (LastValue replacement)
fn compute_subgraph_field_delta(
    old: Option<serde_json::Value>,
    new: serde_json::Value,
) -> Option<serde_json::Value> {
    use serde_json::Value;
    match &old {
        None => Some(new),                        // new key produced by subgraph
        Some(old_val) if old_val == &new => None, // unchanged — skip
        Some(Value::Array(old_arr)) => {
            if let Value::Array(ref new_arr) = new {
                if new_arr.len() > old_arr.len() && new_arr[..old_arr.len()] == old_arr[..] {
                    // Pure append: return only the freshly added items so the
                    // parent's Append reducer extends the list correctly.
                    Some(Value::Array(new_arr[old_arr.len()..].to_vec()))
                } else {
                    // Non-append modification — full replacement.
                    Some(new)
                }
            } else {
                Some(new) // type changed
            }
        }
        Some(_) => Some(new), // changed non-array field
    }
}

// ─── State ↔ dict helpers ──────────────────────────────────────────────────

/// Convert all keys in a DynState into a Python dict.
pub(crate) fn dynstate_to_pydict<'py>(
    py: Python<'py>,
    state: &DynState,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    for key in state.keys() {
        if let Some(val) = state.get(&key) {
            dict.set_item(&key, crate::json_to_py(py, &val)?)?;
        }
    }
    Ok(dict)
}

/// Convert a Python dict into a DynState.
pub(crate) fn pydict_to_dynstate(dict: &Bound<'_, PyDict>) -> PyResult<DynState> {
    let state = DynState::new();
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val = crate::py_to_json(&v)?;
        state.set(key, val);
    }
    Ok(state)
}

/// Validate that every key in `update` is present in `schema_set` (O(1) lookup).
fn validate_update_keys(
    node_name: &str,
    update: &Bound<'_, PyDict>,
    schema_set: &HashSet<String>,
) -> PyResult<()> {
    for (k, _) in update.iter() {
        let key: String = k.extract()?;
        if !schema_set.contains(&key) {
            return Err(PyKeyError::new_err(format!(
                "Node '{}' returned key '{}' which is not declared in the state schema.",
                node_name, key
            )));
        }
    }
    Ok(())
}

// ─── Subgraph wrapper ──────────────────────────────────────────────────────

struct ArcSubgraphNode {
    name: String,
    subgraph: Arc<StateGraph<DynState>>,
}

#[async_trait::async_trait]
impl Node<DynState> for ArcSubgraphNode {
    async fn execute(
        &self,
        state: &DynState,
        _ctx: &Context,
    ) -> Result<DynStateUpdate, StateGraphError> {
        let input = state.clone();
        let result = self.subgraph.invoke(input.clone()).await?;
        // Issue #2: returning ALL result keys caused double-counting for Append/Sum
        // reducers (the parent re-applied the reducer on already-accumulated values).
        // Instead, compute per-field deltas so parent reducers receive correct inputs.
        let mut update = DynStateUpdate::new();
        for key in result.keys() {
            if let Some(new_val) = result.get(&key) {
                let old_val = input.get(&key);
                if let Some(delta) = compute_subgraph_field_delta(old_val, new_val) {
                    update.insert(key, delta);
                }
            }
        }
        Ok(update)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ─── Tracing node wrapper ──────────────────────────────────────────────────

/// Wraps any Node and emits trace_node_start / trace_node_end events into an
/// ExecutionTracer for each node execution. Used when compile(tracer=...) is set.
struct TracingNode {
    inner: Arc<dyn Node<DynState>>,
    tracer: Arc<ExecutionTracer>,
    node_name: String,
}

#[async_trait::async_trait]
impl Node<DynState> for TracingNode {
    async fn execute(
        &self,
        state: &DynState,
        ctx: &Context,
    ) -> Result<DynStateUpdate, StateGraphError> {
        let start = std::time::Instant::now();
        self.tracer.trace_node_start(&self.node_name);
        let result = self.inner.execute(state, ctx).await;
        let success = result.is_ok();
        self.tracer
            .trace_node_end(&self.node_name, start.elapsed(), success);
        result
    }

    fn name(&self) -> &str {
        &self.node_name
    }
}

// ─── Python-callable node ──────────────────────────────────────────────────

/// Wraps a Python callable as a graph node (LangGraph-style reducer pattern).
///
/// The callable receives the full state as a plain dict and must return a
/// partial update dict. Unknown keys raise a KeyError at runtime.
///
/// Each field in the update is merged using its registered `ChannelType`:
/// - `LastValue`        — plain overwrite (default)
/// - `Topic`            — list append
/// - `BinaryOperator`   — custom merge fn
///
/// # GIL safety
///
/// The callable is invoked inside `tokio::task::spawn_blocking`, which runs
/// on a dedicated OS thread pool separate from the Tokio async executor.
/// The GIL is acquired only for the duration of the Python call and is
/// released immediately after.
///
/// **Important constraints for node functions:**
/// - Do **not** rely on Python `threading.local()` state — the blocking thread
///   that runs your function is chosen by Tokio and may differ across calls.
/// - Do **not** hold Python objects (e.g. locks, generators, file handles) that
///   depend on which OS thread holds the GIL between calls.
/// - Async Python functions are **not** supported; the callable must be
///   synchronous.  Wrap async Python work in `asyncio.run()` if needed.
struct PyFunctionNode {
    name: String,
    func: PyObject,
    #[allow(dead_code)]
    schema_fields: Arc<Vec<String>>,
    /// O(1) lookup set built from schema_fields at construction time (issue #16).
    schema_set: Arc<HashSet<String>>,
    /// Per-field reducer strategies extracted from the state class schema.
    channel_schemas: Arc<HashMap<String, ChannelType>>,
}

#[async_trait::async_trait]
impl Node<DynState> for PyFunctionNode {
    async fn execute(
        &self,
        state: &DynState,
        _ctx: &Context,
    ) -> Result<DynStateUpdate, StateGraphError> {
        let state_clone = state.clone();
        let schema_set = self.schema_set.clone();
        let channel_schemas = self.channel_schemas.clone();
        let node_name = self.name.clone();
        // Clone the PyObject before entering spawn_blocking (requires GIL token).
        let func = Python::with_gil(|py| self.func.clone_ref(py));

        // Issue #9: move the synchronous Python call onto a dedicated blocking
        // thread so the Tokio async pool is never stalled by GIL acquisition.
        // This also makes tokio::time::timeout in TimeoutGraphNode effective.
        let result = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<DynStateUpdate> {
                // Pass full state as a plain dict
                let state_dict = dynstate_to_pydict(py, &state_clone)?;

                // Call Python function
                let py_result = func.call1(py, (state_dict,))?;

                // Expect a plain dict back (partial update)
                let py_result_bound = py_result.bind(py);
                let type_name = py_result_bound
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_else(|_| "unknown".to_string());
                let update_dict = py_result_bound.downcast::<PyDict>().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "Node '{}' must return a dict (partial state update), got: {}",
                        node_name, type_name
                    ))
                })?;

                // Validate keys against schema — O(1) HashSet lookup (issue #16)
                validate_update_keys(&node_name, update_dict, &schema_set)?;

                // Build a DynStateUpdate, applying channel reducers.
                let mut update = DynStateUpdate::new();
                for (k, v) in update_dict.iter() {
                    let key: String = k.extract()?;
                    let new_val = crate::py_to_json(&v)?;

                    let channel_type = channel_schemas.get(&key).unwrap_or(&ChannelType::LastValue);

                    let merged_val = match channel_type {
                        ChannelType::LastValue => new_val,
                        _ => {
                            let current = state_clone.get(&key).unwrap_or(serde_json::Value::Null);
                            apply_channel_reducer(current, new_val, channel_type)
                        }
                    };
                    update.insert(key, merged_val);
                }

                Ok(update)
            })
        })
        .await
        .map_err(|e| StateGraphError::ExecutionError {
            node: self.name.clone(),
            reason: format!("Thread join error: {}", e),
        })?;

        result.map_err(|e| StateGraphError::ExecutionError {
            node: self.name.clone(),
            reason: format!("Python node error: {}", e),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ─── Python-callable router ────────────────────────────────────────────────

/// Wraps a Python callable as a sync router function.
///
/// The callable receives the current state as a plain dict and must return a
/// str — either a node name or `"__end__"`.
#[allow(clippy::type_complexity)]
fn make_router_fn(
    py_func: PyObject,
) -> Box<dyn Fn(&DynState) -> Result<String, StateGraphError> + Send + Sync> {
    Box::new(move |state: &DynState| {
        Python::with_gil(|py| -> Result<String, StateGraphError> {
            let func = py_func.clone_ref(py);

            let state_dict =
                dynstate_to_pydict(py, state).map_err(|e| StateGraphError::ExecutionError {
                    node: "router".into(),
                    reason: format!("Failed to convert state to dict: {}", e),
                })?;

            let py_result =
                func.call1(py, (state_dict,))
                    .map_err(|e| StateGraphError::ExecutionError {
                        node: "router".into(),
                        reason: format!("Python router error: {}", e),
                    })?;

            py_result
                .extract(py)
                .map_err(|e| StateGraphError::ExecutionError {
                    node: "router".into(),
                    reason: format!("Router must return a str (node name): {}", e),
                })
        })
    })
}

// ─── StateGraph (builder) ──────────────────────────────────────────────────

/// Build a state graph from a TypedDict schema (LangGraph-compatible API).
///
/// Example:
///     from typing import TypedDict, List
///
///     class State(TypedDict):
///         messages: List[str]
///         score: float
///
///     def greet(state: dict) -> dict:
///         return {"messages": state["messages"] + ["Hello!"]}
///
///     def router(state: dict) -> str:
///         return "greet" if state["score"] > 0.5 else END
///
///     builder = StateGraph(State)
///     builder.add_node("greet", greet)
///     builder.set_entry_point("greet")
///     builder.add_conditional_edge("greet", router)
///     graph = builder.compile()
///
///     result = graph.invoke({"messages": [], "score": 0.8})
///     print(result["messages"])  # ["Hello!"]
#[pyclass(name = "StateGraph", subclass)]
pub struct PyStateGraphBuilder {
    schema_fields: Arc<Vec<String>>,
    /// O(1) lookup set for schema validation (issue #16).
    schema_set: Arc<HashSet<String>>,
    /// Only the *required* fields from the TypedDict schema (issue #3).
    /// Derived from `__required_keys__` so that `total=False` fields are optional.
    required_fields: Arc<HashSet<String>>,
    /// Per-field reducer strategies extracted from `__reducers__` or `Annotated` annotations.
    channel_schemas: Arc<HashMap<String, ChannelType>>,
    nodes: Vec<(String, Arc<dyn Node<DynState>>)>,
    edges: Vec<(String, String)>,
    conditional_edges: Vec<(String, PyObject)>,
    entry_point: Option<String>,
    max_steps: usize,
    interrupt_before: Vec<String>,
    interrupt_after: Vec<String>,
    subgraphs: Vec<(String, Arc<StateGraph<DynState>>)>,
    checkpointer_path: Option<String>,
    middleware: Vec<Arc<dyn Middleware<DynState>>>,
    broadcaster: Option<Arc<EventBroadcaster>>,
}

#[pymethods]
impl PyStateGraphBuilder {
    /// Create a new StateGraph from a TypedDict subclass.
    ///
    /// Args:
    ///     state_class: A TypedDict subclass defining the state schema.
    ///
    /// Raises:
    ///     ValueError: If the class has no type annotations.
    ///
    /// Example:
    ///     class MyState(TypedDict):
    ///         messages: List[str]
    ///         score: float
    ///
    ///     builder = StateGraph(MyState)
    #[new]
    fn new(py: Python<'_>, state_class: &Bound<'_, PyAny>) -> PyResult<Self> {
        let schema_fields = extract_schema_fields(state_class)?;
        let schema_set: HashSet<String> = schema_fields.iter().cloned().collect();
        // Issue #3: respect total=False by reading __required_keys__ from TypedDict.
        let required_fields = extract_required_fields(state_class, &schema_fields);
        // Extract per-field reducer strategies from __reducers__ or Annotated hints.
        let channel_schemas = extract_reducers_from_class(py, state_class).unwrap_or_default();
        Ok(PyStateGraphBuilder {
            schema_fields: Arc::new(schema_fields),
            schema_set: Arc::new(schema_set),
            required_fields: Arc::new(required_fields),
            channel_schemas: Arc::new(channel_schemas),
            nodes: Vec::new(),
            edges: Vec::new(),
            conditional_edges: Vec::new(),
            entry_point: None,
            max_steps: 1000,
            interrupt_before: Vec::new(),
            interrupt_after: Vec::new(),
            subgraphs: Vec::new(),
            checkpointer_path: None,
            middleware: Vec::new(),
            broadcaster: None,
        })
    }

    /// Add a node backed by a Python callable.
    ///
    /// Callable signature: `(state: dict) -> dict`
    /// - Receives: full current state as a plain dict
    /// - Returns: partial update dict (only the keys you want to change)
    ///
    /// Unknown keys in the return value raise a KeyError at runtime.
    fn add_node(&mut self, name: &str, func: PyObject) {
        let node = Arc::new(PyFunctionNode {
            name: name.to_string(),
            func,
            schema_fields: self.schema_fields.clone(),
            schema_set: self.schema_set.clone(),
            channel_schemas: self.channel_schemas.clone(),
        }) as Arc<dyn Node<DynState>>;
        self.nodes.push((name.to_string(), node));
    }

    /// Add a fixed edge: from → to.
    ///
    /// Use `END` (or `"__end__"`) as `to` to terminate execution.
    fn add_edge(&mut self, from: &str, to: &str) {
        self.edges.push((from.to_string(), to.to_string()));
    }

    /// Add a conditional edge with a Python router callable.
    ///
    /// Router signature: `(state: dict) -> str`
    /// - Receives: full current state as a plain dict
    /// - Returns: name of the next node, or `"__end__"` / END
    fn add_conditional_edge(&mut self, from: &str, router: PyObject) {
        self.conditional_edges.push((from.to_string(), router));
    }

    /// Set the entry point node (first node executed).
    fn set_entry_point(&mut self, name: &str) {
        self.entry_point = Some(name.to_string());
    }

    /// Set the maximum number of execution steps (default: 1000).
    fn set_max_steps(&mut self, max_steps: usize) {
        self.max_steps = max_steps;
    }

    /// Pause execution BEFORE this node runs (human-in-the-loop).
    fn interrupt_before(&mut self, name: &str) {
        self.interrupt_before.push(name.to_string());
    }

    /// Pause execution AFTER this node runs.
    fn interrupt_after(&mut self, name: &str) {
        self.interrupt_after.push(name.to_string());
    }

    /// Add a compiled subgraph as a node.
    fn add_subgraph(&mut self, name: &str, subgraph: &PyCompiledGraph) {
        self.subgraphs
            .push((name.to_string(), subgraph.inner.clone()));
    }

    /// Add a retry node backed by a Python callable.
    ///
    /// Retries the callable up to `max_retries` times with exponential backoff.
    /// Callable signature: `(state: dict) -> dict`
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, func, max_retries=3, backoff_ms=1000, backoff_multiplier=2.0, max_backoff_ms=30000))]
    fn add_retry_node(
        &mut self,
        py: Python<'_>,
        name: &str,
        func: PyObject,
        max_retries: usize,
        backoff_ms: u64,
        backoff_multiplier: f64,
        max_backoff_ms: u64,
    ) {
        let node = Arc::new(RetryGraphNode {
            name: name.to_string(),
            func: func.clone_ref(py),
            max_retries,
            backoff_ms,
            backoff_multiplier,
            max_backoff_ms,
            schema_fields: self.schema_fields.clone(),
            schema_set: self.schema_set.clone(),
        }) as Arc<dyn Node<DynState>>;
        self.nodes.push((name.to_string(), node));
    }

    /// Add a timeout node backed by a Python callable.
    ///
    /// Callable signature: `(state: dict) -> dict`
    #[pyo3(signature = (name, func, timeout_ms, on_timeout="error"))]
    fn add_timeout_node(
        &mut self,
        py: Python<'_>,
        name: &str,
        func: PyObject,
        timeout_ms: u64,
        on_timeout: &str,
    ) {
        let node = Arc::new(TimeoutGraphNode {
            name: name.to_string(),
            func: func.clone_ref(py),
            timeout_ms,
            on_timeout: on_timeout.to_string(),
            schema_fields: self.schema_fields.clone(),
            schema_set: self.schema_set.clone(),
        }) as Arc<dyn Node<DynState>>;
        self.nodes.push((name.to_string(), node));
    }

    /// Add an LLM node that reads a prompt from state and writes the response.
    ///
    /// The `prompt_key` and `output_key` must be declared in the state schema.
    #[pyo3(signature = (name, llm, prompt_key="prompt", output_key="llm_response", system_prompt=None))]
    fn add_llm_node(
        &mut self,
        name: &str,
        llm: &PyLLM,
        prompt_key: &str,
        output_key: &str,
        system_prompt: Option<String>,
    ) {
        let node = Arc::new(LLMGraphNode {
            name: name.to_string(),
            llm: llm.inner.clone(),
            system_prompt,
            prompt_key: prompt_key.to_string(),
            output_key: output_key.to_string(),
        }) as Arc<dyn Node<DynState>>;
        self.nodes.push((name.to_string(), node));
    }

    /// Add a planner node (LLM-driven dynamic routing).
    #[pyo3(signature = (name, llm, prompt=None))]
    fn add_planner_node(&mut self, name: &str, llm: &PyLLM, prompt: Option<String>) {
        let node = create_planner_graph_node(name, llm.inner.clone(), prompt);
        self.nodes.push((name.to_string(), node));
    }

    /// Add a loop node that runs a Python callable up to `max_iterations` times.
    ///
    /// Callable signature: `(state: dict) -> dict`
    /// break_condition signature: `(state: dict) -> bool`  (optional — exits early when True)
    ///
    /// Example:
    ///     def body(state):
    ///         n = state["counter"] + 1
    ///         return {"counter": n, "done": n >= 5}
    ///
    ///     builder.add_loop_node("loop", body, max_iterations=10,
    ///                           break_condition=lambda s: s["done"])
    #[pyo3(signature = (name, func, max_iterations=10, break_condition=None))]
    fn add_loop_node(
        &mut self,
        py: Python<'_>,
        name: &str,
        func: PyObject,
        max_iterations: usize,
        break_condition: Option<PyObject>,
    ) {
        let node = Arc::new(crate::builtin_node_bindings::LoopGraphNode {
            name: name.to_string(),
            func: func.clone_ref(py),
            max_iterations,
            break_condition: break_condition.map(|bc| bc.clone_ref(py)),
            schema_set: self.schema_set.clone(),
        }) as Arc<dyn Node<DynState>>;
        self.nodes.push((name.to_string(), node));
    }

    /// Add a parallel node that runs multiple Python callables concurrently.
    ///
    /// `branches` must be either:
    ///   - a **list** of callables (auto-named branch_0, branch_1, …)
    ///   - a **dict** `{name: callable}` for named branches
    ///
    /// All branches receive the same initial state.  Their partial updates are
    /// merged with last-write-wins semantics (dict insertion order).
    ///
    /// Example (list):
    ///     builder.add_parallel_node("analyze", [sentiment_fn, keywords_fn, summary_fn])
    ///
    /// Example (dict):
    ///     builder.add_parallel_node("analyze", {
    ///         "sentiment": sentiment_fn,
    ///         "keywords": keywords_fn,
    ///     })
    fn add_parallel_node(
        &mut self,
        py: Python<'_>,
        name: &str,
        branches: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut branch_list: Vec<(String, PyObject)> = Vec::new();

        if let Ok(list) = branches.downcast::<pyo3::types::PyList>() {
            for (i, item) in list.iter().enumerate() {
                branch_list.push((format!("branch_{}", i), item.into_py(py)));
            }
        } else if let Ok(dict) = branches.downcast::<pyo3::types::PyDict>() {
            for (k, v) in dict.iter() {
                let branch_name: String = k.extract()?;
                branch_list.push((branch_name, v.into_py(py)));
            }
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "add_parallel_node: branches must be a list or dict of callables",
            ));
        }

        let node = Arc::new(crate::builtin_node_bindings::ParallelGraphNode {
            name: name.to_string(),
            branches: branch_list,
            schema_set: self.schema_set.clone(),
        }) as Arc<dyn Node<DynState>>;
        self.nodes.push((name.to_string(), node));
        Ok(())
    }

    /// Add an evaluation node (iterative quality refinement).
    #[pyo3(signature = (handler, config, scorer=None))]
    fn add_evaluation_node(
        &mut self,
        py: Python<'_>,
        handler: PyObject,
        config: &PyEvaluationNodeConfig,
        scorer: Option<PyObject>,
    ) {
        let node = Arc::new(EvaluationGraphNode {
            name: config.inner.name.clone(),
            handler: handler.clone_ref(py),
            scorer: scorer.map(|s| s.clone_ref(py)),
            config: config.inner.clone(),
            schema_fields: self.schema_fields.clone(),
        }) as Arc<dyn Node<DynState>>;
        self.nodes.push((config.inner.name.clone(), node));
    }

    /// Set a file checkpointer for persistent state across invocations.
    fn set_checkpointer(&mut self, base_dir: &str) -> PyResult<()> {
        self.checkpointer_path = Some(base_dir.to_string());
        Ok(())
    }

    /// Attach an EventBroadcaster so the compiled graph emits execution events.
    ///
    /// Subscribe from the broadcaster before invoking the graph to receive
    /// node-start/complete/failed, edge-traversed, and LLM-streaming events.
    ///
    /// Example:
    ///     bc = EventBroadcaster()
    ///     rx = bc.subscribe()
    ///     builder.set_broadcaster(bc)
    ///     graph = builder.compile()
    ///     graph.invoke({...})
    ///     for event in rx.drain():
    ///         print(event["type"])
    fn set_broadcaster(&mut self, broadcaster: &crate::observability::PyEventBroadcaster) {
        self.broadcaster = Some(broadcaster.inner.clone());
    }

    /// Add middleware to the graph execution pipeline.
    ///
    /// Middleware intercepts each node execution. Supported types:
    /// - LoggingMiddleware — logs node start/end via tracing
    /// - MetricsMiddleware — collects per-node timing and error counts
    /// - Any Python object with before_node(node_name, state) and/or
    ///   after_node(node_name, state) methods that return "continue", "skip",
    ///   or "abort:<reason>".
    ///
    /// Example — built-in:
    ///     mw = LoggingMiddleware(verbose=True)
    ///     builder.use_middleware(mw)
    ///
    /// Example — custom Python class:
    ///     class MyMW:
    ///         def before_node(self, node_name, state):
    ///             print(f"entering {node_name}")
    ///             return "continue"
    ///     builder.use_middleware(MyMW())
    fn use_middleware(&mut self, mw: &Bound<'_, PyAny>) -> PyResult<()> {
        let arc: Arc<dyn Middleware<DynState>> = if let Ok(logging) =
            mw.extract::<PyRef<crate::middleware::PyLoggingMiddleware>>()
        {
            logging.as_dyn()
        } else if let Ok(metrics) = mw.extract::<PyRef<crate::middleware::PyMetricsMiddleware>>() {
            metrics.as_dyn()
        } else if mw.hasattr("before_node")? || mw.hasattr("after_node")? {
            let name = mw
                .getattr("__class__")
                .and_then(|c| c.getattr("__name__"))
                .and_then(|n| n.extract::<String>())
                .ok();
            Arc::new(crate::middleware::PyObjectMiddleware::new(
                mw.clone().unbind(),
                name,
            ))
        } else {
            return Err(crate::error::ValidationError::new_err(
                "use_middleware: expected LoggingMiddleware, MetricsMiddleware, \
                     or an object with before_node / after_node methods",
            ));
        };
        self.middleware.push(arc);
        Ok(())
    }

    /// Compile the builder into a runnable CompiledGraph.
    ///
    /// Args:
    ///     tracer: Optional ExecutionTracer. When provided, every node execution
    ///             records trace_node_start / trace_node_end events into the tracer.
    ///
    /// Raises RuntimeError if the graph is invalid (missing entry point, etc.).
    #[pyo3(signature = (tracer=None))]
    fn compile(
        &self,
        py: Python<'_>,
        tracer: Option<&PyExecutionTracer>,
    ) -> PyResult<PyCompiledGraph> {
        // Issue #15: validate interrupt node names at compile time — a typo
        // causes silent failure at runtime (the interrupt never fires).
        let node_name_set: HashSet<&str> = self.nodes.iter().map(|(n, _)| n.as_str()).collect();

        for name in &self.interrupt_before {
            if !node_name_set.contains(name.as_str()) {
                return Err(ValidationError::new_err(format!(
                    "interrupt_before: node '{}' does not exist in this graph. \
                     Declared nodes: {:?}",
                    name,
                    node_name_set.iter().collect::<Vec<_>>()
                )));
            }
        }
        for name in &self.interrupt_after {
            if !node_name_set.contains(name.as_str()) {
                return Err(ValidationError::new_err(format!(
                    "interrupt_after: node '{}' does not exist in this graph. \
                     Declared nodes: {:?}",
                    name,
                    node_name_set.iter().collect::<Vec<_>>()
                )));
            }
        }

        let mut builder = StateGraphBuilder::<DynState>::new();

        let tracer_arc: Option<Arc<ExecutionTracer>> = tracer.map(|t| t.tracer_arc());

        for (name, node) in &self.nodes {
            let wrapped: Arc<dyn Node<DynState>> = if let Some(ref t) = tracer_arc {
                Arc::new(TracingNode {
                    inner: node.clone(),
                    tracer: t.clone(),
                    node_name: name.clone(),
                })
            } else {
                node.clone()
            };
            builder = builder.add_node(name.clone(), wrapped);
        }

        for (from, to) in &self.edges {
            builder = builder.add_edge(from.clone(), to.clone());
        }

        for (from, router) in &self.conditional_edges {
            let router_clone = router.clone_ref(py);
            builder = builder.add_conditional_edge(from.clone(), make_router_fn(router_clone));
        }

        if let Some(ref ep) = self.entry_point {
            builder = builder.set_entry_point(ep.clone());
        }

        builder = builder.set_max_steps(self.max_steps);

        for name in &self.interrupt_before {
            builder = builder.interrupt_before(name.clone());
        }
        for name in &self.interrupt_after {
            builder = builder.interrupt_after(name.clone());
        }

        for (name, subgraph) in &self.subgraphs {
            let node = Arc::new(ArcSubgraphNode {
                name: name.clone(),
                subgraph: subgraph.clone(),
            }) as Arc<dyn Node<DynState>>;
            builder = builder.add_node(name.clone(), node);
        }

        if let Some(ref path) = self.checkpointer_path {
            let cp = FileCheckpointer::new(path).map_err(|e| {
                crate::error::InternalError::new_err(format!(
                    "Failed to create checkpointer: {}",
                    e
                ))
            })?;
            builder = builder.set_checkpointer(Arc::new(cp));
        }

        for mw in &self.middleware {
            builder = builder.use_middleware(mw.clone());
        }

        if let Some(ref bc) = self.broadcaster {
            builder = builder.set_broadcaster(bc.clone());
        }

        let graph = builder
            .compile()
            .map_err(|e| crate::error::InternalError::new_err(format!("{}", e)))?;

        Ok(PyCompiledGraph {
            inner: Arc::new(graph),
            schema_fields: self.schema_fields.clone(),
            schema_set: self.schema_set.clone(),
            required_fields: self.required_fields.clone(),
            channel_schemas: self.channel_schemas.clone(),
        })
    }

    fn __repr__(&self) -> String {
        let node_names: Vec<&str> = self.nodes.iter().map(|(n, _)| n.as_str()).collect();
        format!(
            "StateGraph(schema={:?}, nodes={:?}, entry={:?})",
            *self.schema_fields, node_names, self.entry_point
        )
    }
}

// ─── CompiledGraph ─────────────────────────────────────────────────────────

/// A compiled, runnable state graph returned by StateGraph.compile().
///
/// Execute with invoke() passing a plain dict that matches the state schema.
///
/// Example:
///     graph = builder.compile()
///     result = graph.invoke({"messages": [], "score": 0.0})
///     print(result["messages"])
#[pyclass(name = "CompiledGraph")]
#[derive(Clone)]
pub struct PyCompiledGraph {
    pub(crate) inner: Arc<StateGraph<DynState>>,
    pub(crate) schema_fields: Arc<Vec<String>>,
    /// O(1) schema lookup set (issue #16).
    pub(crate) schema_set: Arc<HashSet<String>>,
    /// Required fields only — respects TypedDict `total=False` (issue #3).
    pub(crate) required_fields: Arc<HashSet<String>>,
    /// Per-field reducer strategies — stored for potential future use by external tooling.
    #[allow(dead_code)]
    pub(crate) channel_schemas: Arc<HashMap<String, ChannelType>>,
}

/// Rust-internal backwards-compat alias so that supervisor.rs / visualization.rs
/// can still import `crate::graph::PyStateGraph` without changes.
pub type PyStateGraph = PyCompiledGraph;

#[pymethods]
impl PyCompiledGraph {
    /// Execute the graph with the given initial state dict.
    ///
    /// Args:
    ///     input_dict: A dict whose keys must match the state schema exactly.
    ///                 All schema keys are required.
    ///
    /// Returns:
    ///     A dict with the final state after all nodes have executed.
    ///     Only schema-declared keys are included.
    ///
    /// Raises:
    ///     KeyError:   Input contains a key not in the schema.
    ///     ValueError: A required schema key is missing from the input.
    fn invoke(&self, py: Python<'_>, input_dict: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        self.validate_input(input_dict)?;
        let initial = pydict_to_dynstate(input_dict)?;
        let fut = self.inner.invoke(initial);
        let result = py
            .allow_threads(|| crate::run_async(fut))
            .map_err(sg_err_to_py)?;
        self.state_to_output_dict(py, &result)
    }

    /// Execute the graph with a thread ID for checkpointing.
    fn invoke_with_thread(
        &self,
        py: Python<'_>,
        thread_id: &str,
        input_dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        self.validate_input(input_dict)?;
        let initial = pydict_to_dynstate(input_dict)?;
        let fut = self.inner.invoke_with_id(thread_id.to_string(), initial);
        let result = py
            .allow_threads(|| crate::run_async(fut))
            .map_err(sg_err_to_py)?;
        self.state_to_output_dict(py, &result)
    }

    /// Resume a previously interrupted graph from its checkpoint.
    fn resume(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let fut = self.inner.resume(thread_id);
        let result = py
            .allow_threads(|| crate::run_async(fut))
            .map_err(sg_err_to_py)?;
        self.state_to_output_dict(py, &result)
    }

    /// Resume with injected state updates (human-in-the-loop).
    fn resume_with_state(
        &self,
        py: Python<'_>,
        thread_id: &str,
        updates: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        // Issue #11: validate update keys — resume_with_state previously skipped
        // schema validation, allowing arbitrary keys to corrupt the state.
        for (k, _) in updates.iter() {
            let key: String = k.extract()?;
            if !self.schema_set.contains(&key) {
                return Err(PyKeyError::new_err(format!(
                    "resume_with_state: key '{}' is not declared in the state schema. \
                     Valid keys: {:?}",
                    key, *self.schema_fields
                )));
            }
        }

        // Convert the Python dict to a DynStateUpdate for resume_with_update
        let mut state_update = DynStateUpdate::new();
        for (k, v) in updates.iter() {
            let key: String = k.extract()?;
            let val = crate::py_to_json(&v)?;
            state_update.insert(key, val);
        }
        let fut = self.inner.resume_with_update(thread_id, state_update);
        let result = py
            .allow_threads(|| crate::run_async(fut))
            .map_err(sg_err_to_py)?;
        self.state_to_output_dict(py, &result)
    }

    /// Subscribe to execution events emitted during graph.invoke().
    ///
    /// Returns an EventReceiver whose drain() / try_recv() can be polled
    /// after invoke() completes to inspect what happened.
    ///
    /// Note: subscribe BEFORE calling invoke() — events emitted during invoke
    /// are only received by subscribers that were registered beforehand.
    fn subscribe_events(&self) -> crate::observability::PyEventReceiver {
        crate::observability::PyEventReceiver {
            inner: Some(self.inner.subscribe()),
        }
    }

    /// Get the list of node names in this graph.
    fn node_names(&self) -> Vec<String> {
        self.inner.node_names()
    }

    /// Get the entry point node name.
    fn entry_point(&self) -> String {
        self.inner.entry_point().to_string()
    }

    /// Export the graph as a Graphviz DOT string.
    fn to_dot(&self) -> String {
        self.inner.to_dot()
    }

    /// Export the graph as a Mermaid diagram string.
    fn to_mermaid(&self) -> String {
        self.inner.to_mermaid()
    }

    /// Export the graph structure as JSON.
    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let val = self.inner.to_json();
        crate::json_to_py(py, &val)
    }

    /// Return a list of state snapshots from execution history (requires checkpointer).
    ///
    /// Each entry is a dict with keys: step_id, state, created_at, metadata.
    fn get_state_history(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let fut = self.inner.history(thread_id);
        let history = py
            .allow_threads(|| crate::run_async(fut))
            .map_err(|e| crate::error::InternalError::new_err(format!("{}", e)))?;

        let list = pyo3::types::PyList::empty_bound(py);
        for (step_idx, node_name) in history.iter() {
            let dict = PyDict::new_bound(py);
            dict.set_item("step_id", step_idx)?;
            dict.set_item("node", node_name)?;
            list.append(dict)?;
        }
        Ok(list.into())
    }

    /// Return the schema field names for this graph.
    fn schema_fields(&self) -> Vec<String> {
        self.schema_fields.iter().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "CompiledGraph(entry='{}', nodes={:?})",
            self.inner.entry_point(),
            self.inner.node_names()
        )
    }
}

impl PyCompiledGraph {
    /// Validate that `input_dict` matches the schema:
    /// - No extra keys (O(1) HashSet lookup, issue #16)
    /// - No missing *required* keys (respects total=False, issue #3)
    fn validate_input(&self, input_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        for (k, _) in input_dict.iter() {
            let key: String = k.extract()?;
            if !self.schema_set.contains(&key) {
                return Err(PyKeyError::new_err(format!(
                    "Input key '{}' is not declared in the state schema. Valid keys: {:?}",
                    key, *self.schema_fields
                )));
            }
        }
        // Only check fields that are actually required (honours total=False).
        for field in self.required_fields.iter() {
            if !input_dict.contains(field.as_str())? {
                return Err(ValidationError::new_err(format!(
                    "Missing required key '{}' in input dict. \
                     Required fields: {:?}",
                    field,
                    self.required_fields.iter().collect::<Vec<_>>()
                )));
            }
        }
        Ok(())
    }

    /// Convert a DynState result to a plain Python dict (schema keys only).
    fn state_to_output_dict(&self, py: Python<'_>, state: &DynState) -> PyResult<PyObject> {
        let result_dict = PyDict::new_bound(py);
        for field in self.schema_fields.iter() {
            let val = state.get(field).unwrap_or(serde_json::Value::Null);
            result_dict.set_item(field, crate::json_to_py(py, &val)?)?;
        }
        Ok(result_dict.into())
    }
}

// ─── Constants ─────────────────────────────────────────────────────────────

/// The END sentinel — use as edge target to terminate the graph.
pub const PY_END: &str = END;

/// Free function to get the END constant from Python.
#[allow(dead_code)]
#[pyfunction]
pub fn graph_end() -> String {
    END.to_string()
}
