//! Python bindings for core built-in tools (Calculator, WebRequest, Files).

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::tools::builtin::{CalculatorTool, FilesTool};
use flowgentra_ai::core::tools::web_extended::WebRequestTool;
use flowgentra_ai::core::tools::Tool;

use crate::error::{to_py_err, SerializationError};
use crate::{json_to_py, py_to_json};

// ─── PyCalculatorTool ────────────────────────────────────────────────────────

#[pyclass(name = "CalculatorTool")]
pub struct PyCalculatorTool {
    pub inner: Arc<CalculatorTool>,
}

#[pymethods]
impl PyCalculatorTool {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(CalculatorTool::new()) }
    }

    fn __repr__(&self) -> String { "CalculatorTool()".to_string() }

    fn call(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let v = py_to_json(input)?;
        let r = crate::run_async(self.inner.call(v)).map_err(to_py_err)?;
        json_to_py(py, &r)
    }

    fn definition(&self, py: Python<'_>) -> PyResult<PyObject> {
        let val = serde_json::to_value(&self.inner.definition())
            .map_err(|e| SerializationError::new_err(e.to_string()))?;
        json_to_py(py, &val)
    }
}

// ─── PyWebRequestTool ────────────────────────────────────────────────────────

/// Real HTTP request tool supporting all methods, custom headers and body.
///
/// Example:
///     web = WebRequestTool()
///     result = web.call({"url": "https://api.example.com", "method": "POST", "body": "{}"})
#[pyclass(name = "WebRequestTool")]
pub struct PyWebRequestTool {
    pub inner: Arc<WebRequestTool>,
}

#[pymethods]
impl PyWebRequestTool {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(WebRequestTool::new()) }
    }

    fn __repr__(&self) -> String { "WebRequestTool()".to_string() }

    fn call(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let v = py_to_json(input)?;
        let r = crate::run_async(self.inner.call(v)).map_err(to_py_err)?;
        json_to_py(py, &r)
    }

    fn definition(&self, py: Python<'_>) -> PyResult<PyObject> {
        let val = serde_json::to_value(&self.inner.definition())
            .map_err(|e| SerializationError::new_err(e.to_string()))?;
        json_to_py(py, &val)
    }
}

// ─── PyFilesTool ──────────────────────────────────────────────────────────────

/// Sandboxed file operations tool (read, write, list).
#[pyclass(name = "FilesTool")]
pub struct PyFilesTool {
    pub inner: Arc<FilesTool>,
}

#[pymethods]
impl PyFilesTool {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(FilesTool::default()) }
    }

    fn __repr__(&self) -> String { "FilesTool()".to_string() }

    fn call(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let v = py_to_json(input)?;
        let r = crate::run_async(self.inner.call(v)).map_err(to_py_err)?;
        json_to_py(py, &r)
    }

    fn definition(&self, py: Python<'_>) -> PyResult<PyObject> {
        let val = serde_json::to_value(&self.inner.definition())
            .map_err(|e| SerializationError::new_err(e.to_string()))?;
        json_to_py(py, &val)
    }
}
