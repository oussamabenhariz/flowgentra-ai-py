//! Python bindings for the middleware system.
//!
//! Exposes built-in middleware (LoggingMiddleware, MetricsMiddleware) and a
//! Python-callable wrapper (pass any object with before_node/after_node methods).

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use flowgentra_ai::core::middleware::{
    ExecutionContext, LoggingMiddleware, MetricsMiddleware, Middleware, MiddlewareResult,
};
use flowgentra_ai::core::state::DynState;

use crate::graph::dynstate_to_pydict;

// ─── PyExecutionMetrics ──────────────────────────────────────────────────────

/// Aggregated execution metrics collected by MetricsMiddleware.
#[pyclass(name = "ExecutionMetrics")]
#[derive(Clone)]
pub struct PyExecutionMetrics {
    pub nodes_executed: usize,
    pub errors: usize,
    pub node_timings: HashMap<String, Vec<u128>>,
}

#[pymethods]
impl PyExecutionMetrics {
    #[getter]
    fn get_nodes_executed(&self) -> usize { self.nodes_executed }

    #[getter]
    fn get_errors(&self) -> usize { self.errors }

    #[getter]
    fn get_node_timings(&self, py: Python<'_>) -> PyObject {
        let d = pyo3::types::PyDict::new_bound(py);
        for (k, v) in &self.node_timings {
            let list = pyo3::types::PyList::empty_bound(py);
            for &t in v {
                let _ = list.append(t);
            }
            let _ = d.set_item(k, list);
        }
        d.into()
    }

    /// Average timing in milliseconds for a specific node, or None.
    fn avg_timing(&self, node: &str) -> Option<u128> {
        self.node_timings.get(node).and_then(|t| {
            if t.is_empty() { None } else { Some(t.iter().sum::<u128>() / t.len() as u128) }
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ExecutionMetrics(nodes_executed={}, errors={})",
            self.nodes_executed, self.errors,
        )
    }
}

// ─── PyLoggingMiddleware ─────────────────────────────────────────────────────

/// Middleware that logs node start/end/error events via the tracing crate.
///
/// Pass to StateGraph.use_middleware() to enable execution logging.
///
/// Example:
///     mw = LoggingMiddleware(verbose=True)
///     builder.use_middleware(mw)
///     graph = builder.compile()
#[pyclass(name = "LoggingMiddleware")]
pub struct PyLoggingMiddleware {
    pub(crate) inner: Arc<LoggingMiddleware>,
}

#[pymethods]
impl PyLoggingMiddleware {
    #[new]
    #[pyo3(signature = (verbose = false))]
    fn new(verbose: bool) -> Self {
        let mw = if verbose {
            LoggingMiddleware::new().verbose()
        } else {
            LoggingMiddleware::new()
        };
        PyLoggingMiddleware { inner: Arc::new(mw) }
    }

    fn __repr__(&self) -> &'static str {
        "LoggingMiddleware()"
    }
}

impl PyLoggingMiddleware {
    pub(crate) fn as_dyn(&self) -> Arc<dyn Middleware<DynState>> {
        self.inner.clone() as Arc<dyn Middleware<DynState>>
    }
}

// ─── PyMetricsMiddleware ─────────────────────────────────────────────────────

/// Middleware that collects per-node timing and error counts.
///
/// Call get_metrics() after graph execution to retrieve the collected data.
///
/// Example:
///     mw = MetricsMiddleware()
///     builder.use_middleware(mw)
///     graph = builder.compile()
///     graph.invoke({...})
///     print(mw.get_metrics())
#[pyclass(name = "MetricsMiddleware")]
pub struct PyMetricsMiddleware {
    pub(crate) inner: Arc<MetricsMiddleware>,
}

#[pymethods]
impl PyMetricsMiddleware {
    #[new]
    fn new() -> Self {
        PyMetricsMiddleware { inner: Arc::new(MetricsMiddleware::new()) }
    }

    /// Return the collected metrics (blocking — safe from Python call context).
    fn get_metrics(&self) -> PyExecutionMetrics {
        let metrics = crate::run_async(self.inner.metrics());
        PyExecutionMetrics {
            nodes_executed: metrics.nodes_executed,
            errors: metrics.errors,
            node_timings: metrics.node_timings,
        }
    }

    fn __repr__(&self) -> &'static str {
        "MetricsMiddleware()"
    }
}

impl PyMetricsMiddleware {
    pub(crate) fn as_dyn(&self) -> Arc<dyn Middleware<DynState>> {
        self.inner.clone() as Arc<dyn Middleware<DynState>>
    }
}

// ─── PyObjectMiddleware ──────────────────────────────────────────────────────

/// Wraps a Python object with before_node / after_node methods as Rust middleware.
///
/// Called from graph.rs use_middleware() when the user passes an arbitrary Python object.
pub(crate) struct PyObjectMiddleware {
    obj: PyObject,
    middleware_name: String,
}

impl PyObjectMiddleware {
    pub fn new(obj: PyObject, name: Option<String>) -> Self {
        let middleware_name = name.unwrap_or_else(|| "PyMiddleware".to_string());
        PyObjectMiddleware { obj, middleware_name }
    }
}

/// Call a Python middleware hook that receives (node_name: str, state: dict) -> str.
///
/// Accepted return values: "continue", "skip", "abort:<reason>".
/// Anything else is treated as "continue".
async fn call_py_hook(
    obj: &PyObject,
    method: &'static str,
    node_name: String,
    state: DynState,
) -> MiddlewareResult<DynState> {
    let result = tokio::task::spawn_blocking({
        let obj = Python::with_gil(|py| obj.clone_ref(py));
        let node_name = node_name.clone();
        let state = state.clone();
        move || -> Option<String> {
            Python::with_gil(|py| -> PyResult<String> {
                let state_dict = dynstate_to_pydict(py, &state)?;
                let m = obj.getattr(py, method)?;
                let result = m.call1(py, (&node_name, state_dict))?;
                result.extract(py)
            }).ok()
        }
    })
    .await
    .ok()
    .flatten()
    .unwrap_or_else(|| "continue".to_string());

    match result.as_str() {
        "skip" => MiddlewareResult::Skip,
        s if s.starts_with("abort:") => MiddlewareResult::Abort(s[6..].to_string()),
        _ => MiddlewareResult::Continue,
    }
}

#[async_trait]
impl Middleware<DynState> for PyObjectMiddleware {
    async fn before_node(&self, ctx: &mut ExecutionContext<DynState>) -> MiddlewareResult<DynState> {
        let has_method = Python::with_gil(|py| {
            self.obj.bind(py).hasattr("before_node").unwrap_or(false)
        });
        if !has_method {
            return MiddlewareResult::Continue;
        }
        call_py_hook(&self.obj, "before_node", ctx.node_name.clone(), ctx.state.clone()).await
    }

    async fn after_node(&self, ctx: &mut ExecutionContext<DynState>) -> MiddlewareResult<DynState> {
        let has_method = Python::with_gil(|py| {
            self.obj.bind(py).hasattr("after_node").unwrap_or(false)
        });
        if !has_method {
            return MiddlewareResult::Continue;
        }
        call_py_hook(&self.obj, "after_node", ctx.node_name.clone(), ctx.state.clone()).await
    }

    fn name(&self) -> &str {
        &self.middleware_name
    }
}
