//! Python bindings for observability — ExecutionTracer, ExecutionTrace

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::observability::ExecutionTrace;
use flowgentra_ai::core::observability::visualization::ExecutionTracer;
use flowgentra_ai::core::utils::tracing::init_tracing;

// ─── py_init_tracing ────────────────────────────────────────────────────────

/// Initialize the tracing subscriber with structured logging.
///
/// Call once at program start to enable tracing output.
///
/// Example:
///     init_tracing()
#[pyfunction]
#[pyo3(signature = (log_level="info"))]
pub fn py_init_tracing(log_level: &str) {
    init_tracing(log_level);
}

// ─── PyExecutionTrace ───────────────────────────────────────────────────────

/// A recorded execution trace of graph execution.
///
/// Captures node timings, paths, and token usage.
#[pyclass(name = "ExecutionTrace")]
pub struct PyExecutionTrace {
    pub(crate) inner: ExecutionTrace,
}

#[pymethods]
impl PyExecutionTrace {
    /// Create a new empty trace.
    #[new]
    #[pyo3(signature = (agent_name=None))]
    fn new(agent_name: Option<String>) -> Self {
        PyExecutionTrace {
            inner: ExecutionTrace::new(agent_name),
        }
    }

    /// Get the execution path (list of node names in order).
    fn execution_path(&self) -> Vec<String> {
        self.inner.execution_path()
    }

    /// Get total duration in milliseconds.
    fn total_duration_ms(&self) -> Option<u64> {
        self.inner.total_duration_ms()
    }

    /// Serialize to JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| crate::error::InternalError::new_err(format!("{}", e)))
    }

    /// Deserialize from JSON.
    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let trace = ExecutionTrace::from_json(json_str)
            .map_err(|e| crate::error::SerializationError::new_err(format!("{}", e)))?;
        Ok(PyExecutionTrace { inner: trace })
    }

    fn __repr__(&self) -> String {
        let path = self.inner.execution_path();
        format!("ExecutionTrace(nodes={})", path.len())
    }
}

// ─── PyExecutionTracer ──────────────────────────────────────────────────────

/// Records execution events for graph visualization and debugging.
///
/// Example:
///     tracer = ExecutionTracer()
///     tracer.trace_node_start("process")
///     tracer.trace_node_end("process", duration_ms=150, success=True)
///     print(tracer.get_events_json())
///
/// Pass to StateGraph.compile(tracer=tracer) for automatic per-node tracing.
#[pyclass(name = "ExecutionTracer")]
pub struct PyExecutionTracer {
    inner: Arc<ExecutionTracer>,
}

impl PyExecutionTracer {
    pub(crate) fn tracer_arc(&self) -> Arc<ExecutionTracer> {
        self.inner.clone()
    }
}

#[pymethods]
impl PyExecutionTracer {
    #[new]
    fn new() -> Self {
        PyExecutionTracer {
            inner: Arc::new(ExecutionTracer::new()),
        }
    }

    /// Record a node starting execution.
    fn trace_node_start(&self, node_id: &str) {
        self.inner.trace_node_start(node_id);
    }

    /// Record a node finishing execution.
    fn trace_node_end(&self, node_id: &str, duration_ms: u64, success: bool) {
        self.inner.trace_node_end(
            node_id,
            std::time::Duration::from_millis(duration_ms),
            success,
        );
    }

    /// Record an edge traversal.
    fn trace_edge_traversal(&self, from: &str, to: &str, condition_met: bool) {
        self.inner.trace_edge_traversal(from, to, condition_met);
    }

    /// Record a state update.
    fn trace_state_update(&self, key: &str, value: &str) {
        self.inner.trace_state_update(key, value);
    }

    /// Record a custom event.
    #[pyo3(signature = (event_name, details=None))]
    fn trace_custom(&self, event_name: &str, details: Option<&str>) {
        self.inner.trace_custom(event_name, details);
    }

    /// Get all recorded events as JSON.
    fn get_events_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| crate::error::InternalError::new_err(format!("{}", e)))
    }

    /// Clear all recorded events.
    fn clear(&self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        let events = self.inner.get_events();
        format!("ExecutionTracer(events={})", events.len())
    }
}
