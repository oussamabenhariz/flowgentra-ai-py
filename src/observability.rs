//! Python bindings for observability — ExecutionTracer, ExecutionTrace,
//! EventBroadcaster, EventReceiver, ReplayMode

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use flowgentra_ai::core::observability::ExecutionTrace;
use flowgentra_ai::core::observability::visualization::ExecutionTracer;
use flowgentra_ai::core::observability::events::{EventBroadcaster, ExecutionEvent};
use flowgentra_ai::core::observability::ReplayMode;
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

// ─── ExecutionEvent → Python dict helper ───────────────────────────────────

fn execution_event_to_dict(py: Python<'_>, event: &ExecutionEvent) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    match event {
        ExecutionEvent::GraphStarted { graph_id } => {
            d.set_item("type", "graph_started")?;
            d.set_item("graph_id", graph_id)?;
        }
        ExecutionEvent::NodeStarted { node_name, step } => {
            d.set_item("type", "node_started")?;
            d.set_item("node_name", node_name)?;
            d.set_item("step", step)?;
        }
        ExecutionEvent::NodeCompleted { node_name, step, duration_ms, state_snapshot } => {
            d.set_item("type", "node_completed")?;
            d.set_item("node_name", node_name)?;
            d.set_item("step", step)?;
            d.set_item("duration_ms", duration_ms)?;
            match state_snapshot {
                Some(snap) => d.set_item("state_snapshot", crate::json_to_py(py, snap)?)?,
                None => d.set_item("state_snapshot", py.None())?,
            }
        }
        ExecutionEvent::NodeFailed { node_name, step, error } => {
            d.set_item("type", "node_failed")?;
            d.set_item("node_name", node_name)?;
            d.set_item("step", step)?;
            d.set_item("error", error)?;
        }
        ExecutionEvent::EdgeTraversed { from, to, condition } => {
            d.set_item("type", "edge_traversed")?;
            d.set_item("from", from)?;
            d.set_item("to", to)?;
            match condition {
                Some(c) => d.set_item("condition", c)?,
                None => d.set_item("condition", py.None())?,
            }
        }
        ExecutionEvent::GraphCompleted { total_steps, total_duration_ms } => {
            d.set_item("type", "graph_completed")?;
            d.set_item("total_steps", total_steps)?;
            d.set_item("total_duration_ms", total_duration_ms)?;
        }
        ExecutionEvent::GraphFailed { error, last_node } => {
            d.set_item("type", "graph_failed")?;
            d.set_item("error", error)?;
            match last_node {
                Some(n) => d.set_item("last_node", n)?,
                None => d.set_item("last_node", py.None())?,
            }
        }
        ExecutionEvent::LLMStreaming { node_name, chunk, chunk_index } => {
            d.set_item("type", "llm_streaming")?;
            d.set_item("node_name", node_name)?;
            d.set_item("chunk", chunk)?;
            d.set_item("chunk_index", chunk_index)?;
        }
        ExecutionEvent::LLMStreamingCompleted { node_name, total_chunks } => {
            d.set_item("type", "llm_streaming_completed")?;
            d.set_item("node_name", node_name)?;
            d.set_item("total_chunks", total_chunks)?;
        }
        ExecutionEvent::ToolCalled { node_name, tool_name, args } => {
            d.set_item("type", "tool_called")?;
            d.set_item("node_name", node_name)?;
            d.set_item("tool_name", tool_name)?;
            d.set_item("args", crate::json_to_py(py, args)?)?;
        }
        ExecutionEvent::ToolResult { node_name, tool_name, result, success } => {
            d.set_item("type", "tool_result")?;
            d.set_item("node_name", node_name)?;
            d.set_item("tool_name", tool_name)?;
            d.set_item("result", crate::json_to_py(py, result)?)?;
            d.set_item("success", success)?;
        }
    }
    Ok(d.into())
}

// ─── PyEventBroadcaster ────────────────────────────────────────────────────

/// Real-time event broadcaster for graph execution monitoring.
///
/// Create a broadcaster, pass it to StateGraph.set_broadcaster(), then
/// subscribe to receive events as plain dicts.
///
/// Example:
///     broadcaster = EventBroadcaster()
///     receiver = broadcaster.subscribe()
///
///     builder.set_broadcaster(broadcaster)
///     graph = builder.compile()
///     graph.invoke({...})
///
///     for event in receiver.drain():
///         print(event["type"], event.get("node_name"))
#[pyclass(name = "EventBroadcaster")]
#[derive(Clone)]
pub struct PyEventBroadcaster {
    pub(crate) inner: Arc<EventBroadcaster>,
}

#[pymethods]
impl PyEventBroadcaster {
    #[new]
    #[pyo3(signature = (capacity = 256))]
    fn new(capacity: usize) -> Self {
        PyEventBroadcaster {
            inner: Arc::new(EventBroadcaster::new(capacity)),
        }
    }

    /// Subscribe to execution events. Returns a PyEventReceiver.
    fn subscribe(&self) -> PyEventReceiver {
        PyEventReceiver {
            inner: Some(self.inner.subscribe()),
        }
    }

    /// Number of active subscribers.
    fn subscriber_count(&self) -> usize {
        self.inner.subscriber_count()
    }

    /// Manually emit a GraphStarted event.
    fn emit_graph_started(&self, graph_id: &str) {
        self.inner.emit(ExecutionEvent::GraphStarted { graph_id: graph_id.to_string() });
    }

    /// Manually emit a NodeStarted event.
    fn emit_node_started(&self, node_name: &str, step: usize) {
        self.inner.node_started(node_name, step);
    }

    /// Manually emit a NodeCompleted event.
    fn emit_node_completed(&self, node_name: &str, step: usize, duration_ms: u64) {
        self.inner.node_completed(node_name, step, duration_ms, None);
    }

    /// Manually emit a NodeFailed event.
    fn emit_node_failed(&self, node_name: &str, step: usize, error: &str) {
        self.inner.node_failed(node_name, step, error);
    }

    fn __repr__(&self) -> String {
        format!("EventBroadcaster(subscribers={})", self.inner.subscriber_count())
    }
}

// ─── PyEventReceiver ────────────────────────────────────────────────────────

/// Subscription handle for receiving graph execution events.
///
/// Events are returned as plain Python dicts with a "type" key plus
/// event-specific fields.
///
/// Example:
///     receiver = broadcaster.subscribe()
///     # After running the graph:
///     for event in receiver.drain():
///         if event["type"] == "node_completed":
///             print(event["node_name"], event["duration_ms"], "ms")
#[pyclass(name = "EventReceiver")]
pub struct PyEventReceiver {
    pub(crate) inner: Option<tokio::sync::broadcast::Receiver<ExecutionEvent>>,
}

#[pymethods]
impl PyEventReceiver {
    /// Try to receive a single pending event. Returns None if no events are ready.
    fn try_recv(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let rx = self.inner.as_mut().ok_or_else(|| {
            crate::error::InternalError::new_err("EventReceiver has been consumed")
        })?;
        match rx.try_recv() {
            Ok(event) => execution_event_to_dict(py, &event),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => Ok(py.None()),
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => Ok(py.None()),
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => Ok(py.None()),
        }
    }

    /// Drain all pending events, return as a list of dicts. Does not block.
    fn drain(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let rx = self.inner.as_mut().ok_or_else(|| {
            crate::error::InternalError::new_err("EventReceiver has been consumed")
        })?;
        let list = pyo3::types::PyList::empty_bound(py);
        loop {
            match rx.try_recv() {
                Ok(event) => {
                    let d = execution_event_to_dict(py, &event)?;
                    list.append(d)?;
                }
                Err(_) => break,
            }
        }
        Ok(list.into())
    }

    fn __repr__(&self) -> &'static str {
        "EventReceiver()"
    }
}

// ─── PyReplayMode ───────────────────────────────────────────────────────────

/// Step through a saved execution trace for debugging.
///
/// Load from a trace object or JSON string, then step forward/back to inspect
/// state snapshots at each node.
///
/// Example:
///     trace = tracer.get_events_json()
///     replay = ReplayMode.from_json(trace)
///     while not replay.is_complete():
///         print(f"Step {replay.current_step()}: {replay.current_node()}")
///         snapshot = replay.current_state()
///         replay.step_forward()
#[pyclass(name = "ReplayMode")]
pub struct PyReplayMode {
    inner: ReplayMode,
}

#[pymethods]
impl PyReplayMode {
    /// Create a ReplayMode from an ExecutionTrace.
    #[staticmethod]
    fn from_trace(trace: &crate::observability::PyExecutionTrace) -> Self {
        PyReplayMode {
            inner: ReplayMode::from_trace(trace.inner.clone()),
        }
    }

    /// Create a ReplayMode from a JSON string (as produced by ExecutionTrace.to_json()).
    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let mode = ReplayMode::from_json(json_str)
            .map_err(|e| crate::error::SerializationError::new_err(format!("{}", e)))?;
        Ok(PyReplayMode { inner: mode })
    }

    /// Current position in the replay (0-indexed).
    fn current_step(&self) -> usize {
        self.inner.current_step()
    }

    /// Total number of steps recorded.
    fn total_steps(&self) -> usize {
        self.inner.total_steps()
    }

    /// Node name at the current position, or None if past the end.
    fn current_node(&self) -> Option<&str> {
        self.inner.current_node()
    }

    /// Whether the replay has passed the last step.
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Advance one step forward. Returns True if moved, False if already at end.
    fn step_forward(&mut self) -> bool {
        self.inner.step_forward()
    }

    /// Step back one position. Returns True if moved, False if already at start.
    fn step_back(&mut self) -> bool {
        self.inner.step_back()
    }

    /// Jump to a specific step index.
    fn go_to_step(&mut self, step: usize) {
        self.inner.go_to_step(step);
    }

    /// List of node names in execution order.
    fn execution_path(&self) -> Vec<String> {
        self.inner.execution_path()
    }

    /// State snapshot dict at the current step, or None if none was captured.
    fn current_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.inner.current_state() {
            Some(snap) => crate::json_to_py(py, snap),
            None => Ok(py.None()),
        }
    }

    /// State snapshot dict at a specific step, or None.
    fn state_at(&self, py: Python<'_>, step: usize) -> PyResult<PyObject> {
        match self.inner.state_at(step) {
            Some(snap) => crate::json_to_py(py, snap),
            None => Ok(py.None()),
        }
    }

    /// List of state keys that changed between two steps.
    fn diff_states(&self, step_a: usize, step_b: usize) -> Option<Vec<String>> {
        self.inner.diff_states(step_a, step_b)
    }

    fn __repr__(&self) -> String {
        format!(
            "ReplayMode(step={}/{}, node={:?})",
            self.inner.current_step(),
            self.inner.total_steps(),
            self.inner.current_node(),
        )
    }
}
