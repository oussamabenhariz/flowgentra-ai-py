//! Python bindings for OpenTelemetry-compatible trace export.
//!
//! Converts ExecutionTrace to OTLP spans for Jaeger, Datadog, Honeycomb, etc.

use pyo3::prelude::*;

use flowgentra_ai::core::observability::otel::{
    export_to_otlp, spans_to_otlp_json, trace_to_otel_spans, OtelAttribute, OtelSpan, OtelStatus,
};

// ─── PyOtelStatus ─────────────────────────────────────────────────────────

/// OTLP span status (code: 0=Unset, 1=Ok, 2=Error).
#[pyclass(name = "OtelStatus")]
#[derive(Clone)]
pub struct PyOtelStatus {
    pub(crate) inner: OtelStatus,
}

#[pymethods]
impl PyOtelStatus {
    #[new]
    #[pyo3(signature = (code, message = None))]
    fn new(code: u32, message: Option<String>) -> Self {
        PyOtelStatus {
            inner: OtelStatus { code, message },
        }
    }

    #[getter]
    fn code(&self) -> u32 {
        self.inner.code
    }

    #[getter]
    fn message(&self) -> Option<&str> {
        self.inner.message.as_deref()
    }

    fn __repr__(&self) -> String {
        format!("OtelStatus(code={})", self.inner.code)
    }
}

// ─── PyOtelAttribute ──────────────────────────────────────────────────────

/// A key-value attribute on an OTLP span.
#[pyclass(name = "OtelAttribute")]
#[derive(Clone)]
pub struct PyOtelAttribute {
    pub(crate) inner: OtelAttribute,
}

#[pymethods]
impl PyOtelAttribute {
    #[new]
    fn new(key: String, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let val = crate::py_to_json(value)?;
        Ok(PyOtelAttribute {
            inner: OtelAttribute { key, value: val },
        })
    }

    #[getter]
    fn key(&self) -> &str {
        &self.inner.key
    }

    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        crate::json_to_py(py, &self.inner.value)
    }

    fn __repr__(&self) -> String {
        format!("OtelAttribute(key='{}')", self.inner.key)
    }
}

// ─── PyOtelSpan ───────────────────────────────────────────────────────────

/// An OTLP-compatible span produced from an ExecutionTrace.
///
/// Contains trace/span IDs, timing, attributes, and status.
#[pyclass(name = "OtelSpan")]
#[derive(Clone)]
pub struct PyOtelSpan {
    pub(crate) inner: OtelSpan,
}

#[pymethods]
impl PyOtelSpan {
    #[getter]
    fn trace_id(&self) -> &str {
        &self.inner.trace_id
    }

    #[getter]
    fn span_id(&self) -> &str {
        &self.inner.span_id
    }

    #[getter]
    fn parent_span_id(&self) -> Option<&str> {
        self.inner.parent_span_id.as_deref()
    }

    #[getter]
    fn operation_name(&self) -> &str {
        &self.inner.operation_name
    }

    #[getter]
    fn start_time_unix_nano(&self) -> u64 {
        self.inner.start_time_unix_nano
    }

    #[getter]
    fn end_time_unix_nano(&self) -> u64 {
        self.inner.end_time_unix_nano
    }

    #[getter]
    fn attributes(&self) -> Vec<PyOtelAttribute> {
        self.inner
            .attributes
            .iter()
            .map(|a| PyOtelAttribute { inner: a.clone() })
            .collect()
    }

    #[getter]
    fn status(&self) -> PyOtelStatus {
        PyOtelStatus {
            inner: self.inner.status.clone(),
        }
    }

    /// Serialize this span to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| crate::error::SerializationError::new_err(format!("{}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "OtelSpan(op='{}', trace_id='{}')",
            self.inner.operation_name, self.inner.trace_id
        )
    }
}

// ─── py_trace_to_otel_spans ───────────────────────────────────────────────

/// Convert an ExecutionTrace into a list of OTLP-compatible OtelSpan objects.
///
/// Creates a root span for the full graph execution and child spans per node.
///
/// Example:
///     spans = trace_to_otel_spans(trace)
///     print(f"Generated {len(spans)} spans")
#[pyfunction]
pub fn py_trace_to_otel_spans(trace: &crate::observability::PyExecutionTrace) -> Vec<PyOtelSpan> {
    trace_to_otel_spans(&trace.inner)
        .into_iter()
        .map(|s| PyOtelSpan { inner: s })
        .collect()
}

// ─── py_spans_to_otlp_json ────────────────────────────────────────────────

/// Convert a list of OtelSpan objects to OTLP JSON payload as a dict.
///
/// The returned dict can be sent to any OTLP HTTP collector.
///
/// Example:
///     payload = spans_to_otlp_json(spans)
///     import json
///     print(json.dumps(payload, indent=2))
#[pyfunction]
pub fn py_spans_to_otlp_json(py: Python<'_>, spans: Vec<PyOtelSpan>) -> PyResult<PyObject> {
    let rust_spans: Vec<OtelSpan> = spans.into_iter().map(|s| s.inner).collect();
    let json_val = spans_to_otlp_json(&rust_spans);
    crate::json_to_py(py, &json_val)
}

// ─── py_export_to_otlp ────────────────────────────────────────────────────

/// Send spans to an OTLP HTTP collector endpoint (blocking).
///
/// Args:
///     endpoint: Base URL of the OTLP collector (e.g. "http://localhost:4318")
///     spans: List of OtelSpan objects to export
///
/// Example:
///     export_to_otlp("http://localhost:4318", spans)
#[pyfunction]
pub fn py_export_to_otlp(endpoint: &str, spans: Vec<PyOtelSpan>) -> PyResult<()> {
    let rust_spans: Vec<OtelSpan> = spans.into_iter().map(|s| s.inner).collect();
    crate::run_async(export_to_otlp(endpoint, &rust_spans))
        .map_err(|e| crate::error::InternalError::new_err(format!("OTLP export failed: {}", e)))
}
