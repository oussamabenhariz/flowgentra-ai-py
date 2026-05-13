//! Python bindings for Prometheus metrics export.
//!
//! Exposes PrometheusExporter, MetricsCollector, and record_llm_tokens.

use flowgentra_ai::core::observability::{record_llm_tokens, MetricsCollector, PrometheusExporter};
use pyo3::prelude::*;

// ─── PyPrometheusExporter ──────────────────────────────────────────────────

/// Prometheus /metrics HTTP endpoint.
///
/// After install(), Prometheus can scrape the configured address.
///
/// Example:
///     exporter = PrometheusExporter("0.0.0.0:9090")
///     exporter.install()
///     # Scrape: http://localhost:9090/metrics
#[pyclass(name = "PrometheusExporter")]
pub struct PyPrometheusExporter {
    addr: String,
}

#[pymethods]
impl PyPrometheusExporter {
    #[new]
    #[pyo3(signature = (addr = "0.0.0.0:9090"))]
    fn new(addr: &str) -> Self {
        PyPrometheusExporter {
            addr: addr.to_string(),
        }
    }

    /// Install the Prometheus recorder and start the HTTP listener.
    ///
    /// One-time call — returns an error if called twice.
    fn install(&self) -> PyResult<()> {
        PrometheusExporter::new(&self.addr).install().map_err(|e| {
            crate::error::InternalError::new_err(format!("Prometheus install failed: {}", e))
        })
    }

    fn __repr__(&self) -> String {
        format!("PrometheusExporter(addr='{}')", self.addr)
    }
}

// ─── PyMetricsCollector ────────────────────────────────────────────────────

/// Subscribes to an EventBroadcaster and records Prometheus metrics.
///
/// Call run_background() to start collecting before running the graph.
///
/// Example:
///     broadcaster = EventBroadcaster()
///     builder.set_broadcaster(broadcaster)
///     graph = builder.compile()
///
///     collector = MetricsCollector(broadcaster)
///     collector.run_background()
///
///     graph.invoke({...})
#[pyclass(name = "MetricsCollector")]
pub struct PyMetricsCollector {
    inner: std::sync::Mutex<Option<MetricsCollector>>,
}

#[pymethods]
impl PyMetricsCollector {
    #[new]
    fn new(broadcaster: &crate::observability::PyEventBroadcaster) -> Self {
        PyMetricsCollector {
            inner: std::sync::Mutex::new(Some(MetricsCollector::new(&broadcaster.inner))),
        }
    }

    /// Spawn the metrics collector as a background tokio task.
    ///
    /// The collector runs until the graph emits GraphCompleted or GraphFailed.
    fn run_background(&self) -> PyResult<()> {
        let collector = {
            let mut guard = self.inner.lock().unwrap();
            guard.take().ok_or_else(|| {
                crate::error::InternalError::new_err(
                    "MetricsCollector already running or consumed — create a new instance",
                )
            })?
        };
        crate::get_runtime().spawn(collector.run());
        Ok(())
    }

    fn __repr__(&self) -> &'static str {
        "MetricsCollector()"
    }
}

// ─── py_record_llm_tokens ──────────────────────────────────────────────────

/// Record LLM prompt and completion token counts as Prometheus metrics.
///
/// Call this after each LLM call to feed the flowgentra_llm_tokens_total counter.
///
/// Example:
///     msg, usage = client.chat_with_usage(messages)
///     if usage:
///         record_llm_tokens(usage.prompt_tokens, usage.completion_tokens)
#[pyfunction]
pub fn py_record_llm_tokens(prompt_tokens: u64, completion_tokens: u64) {
    record_llm_tokens(prompt_tokens, completion_tokens);
}
