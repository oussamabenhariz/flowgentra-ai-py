//! Python bindings for knowledge tools (Wikipedia, ArXiv, PubMed, WolframAlpha).

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::tools::knowledge::{
    ArxivTool, PubMedTool, WikipediaTool, WolframAlphaTool,
};
use flowgentra_ai::core::tools::Tool;

use crate::error::to_py_err;
use crate::{json_to_py, py_to_json};

macro_rules! impl_tool {
    ($py_name:ident, $repr:expr) => {
        #[pymethods]
        impl $py_name {
            fn call(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<PyObject> {
                let v = py_to_json(input)?;
                let r = crate::run_async(self.inner.call(v)).map_err(to_py_err)?;
                json_to_py(py, &r)
            }
            fn definition(&self, py: Python<'_>) -> PyResult<PyObject> {
                let val = serde_json::to_value(&self.inner.definition())
                    .map_err(|e| crate::error::ToolExecutionError::new_err(e.to_string()))?;
                json_to_py(py, &val)
            }
            fn __repr__(&self) -> String { $repr.to_string() }
        }
    };
}

// ─── WikipediaTool ──────────────────────────────────────────────────────────

#[pyclass(name = "WikipediaTool")]
pub struct PyWikipediaTool {
    pub inner: Arc<WikipediaTool>,
}

#[pymethods]
impl PyWikipediaTool {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(WikipediaTool::new()) }
    }
}
impl_tool!(PyWikipediaTool, "WikipediaTool()");

// ─── ArxivTool ──────────────────────────────────────────────────────────────

#[pyclass(name = "ArxivTool")]
pub struct PyArxivTool {
    pub inner: Arc<ArxivTool>,
}

#[pymethods]
impl PyArxivTool {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(ArxivTool::new()) }
    }
}
impl_tool!(PyArxivTool, "ArxivTool()");

// ─── PubMedTool ─────────────────────────────────────────────────────────────

#[pyclass(name = "PubMedTool")]
pub struct PyPubMedTool {
    pub inner: Arc<PubMedTool>,
}

#[pymethods]
impl PyPubMedTool {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(PubMedTool::new()) }
    }
}
impl_tool!(PyPubMedTool, "PubMedTool()");

// ─── WolframAlphaTool ───────────────────────────────────────────────────────

#[pyclass(name = "WolframAlphaTool")]
pub struct PyWolframAlphaTool {
    pub inner: Arc<WolframAlphaTool>,
}

#[pymethods]
impl PyWolframAlphaTool {
    #[new]
    #[pyo3(signature = (app_id=None))]
    fn new(app_id: Option<String>) -> PyResult<Self> {
        let tool = match app_id {
            Some(k) => WolframAlphaTool::new(k),
            None => WolframAlphaTool::from_env().map_err(to_py_err)?,
        };
        Ok(Self { inner: Arc::new(tool) })
    }
}
impl_tool!(PyWolframAlphaTool, "WolframAlphaTool()");
