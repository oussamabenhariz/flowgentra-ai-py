//! Python bindings for web search tools.

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::tools::search::{
    BraveSearchTool, DuckDuckGoSearchTool, GoogleSerperTool, SerpApiSearchTool, TavilySearchTool,
};
use flowgentra_ai::core::tools::Tool;

use crate::error::to_py_err;
use crate::{json_to_py, py_to_json};

macro_rules! impl_tool {
    ($py_name:ident, $rust_type:ty, $repr:expr) => {
        #[pymethods]
        impl $py_name {
            fn call(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<PyObject> {
                let v = py_to_json(input)?;
                let r = crate::run_async(self.inner.call(v)).map_err(to_py_err)?;
                json_to_py(py, &r)
            }
            fn definition(&self, py: Python<'_>) -> PyResult<PyObject> {
                let val = serde_json::to_value(&self.inner.definition())
                    .map_err(|e| pyo3::exceptions::crate::error::ToolExecutionError::new_err(e.to_string()))?;
                json_to_py(py, &val)
            }
            fn __repr__(&self) -> String { $repr.to_string() }
        }
    };
}

// ─── DuckDuckGoSearchTool ────────────────────────────────────────────────────

/// Web search via DuckDuckGo — no API key required.
#[pyclass(name = "DuckDuckGoSearchTool")]
pub struct PyDuckDuckGoSearchTool {
    pub inner: Arc<DuckDuckGoSearchTool>,
}

#[pymethods]
impl PyDuckDuckGoSearchTool {
    #[new]
    #[pyo3(signature = (max_results=None))]
    fn new(max_results: Option<usize>) -> Self {
        Self { inner: Arc::new(DuckDuckGoSearchTool::new(max_results.unwrap_or(5))) }
    }
}
impl_tool!(PyDuckDuckGoSearchTool, DuckDuckGoSearchTool, "DuckDuckGoSearchTool()");

// ─── TavilySearchTool ───────────────────────────────────────────────────────

/// AI-powered web search via Tavily (requires API key or TAVILY_API_KEY env var).
#[pyclass(name = "TavilySearchTool")]
pub struct PyTavilySearchTool {
    pub inner: Arc<TavilySearchTool>,
}

#[pymethods]
impl PyTavilySearchTool {
    #[new]
    #[pyo3(signature = (api_key=None))]
    fn new(api_key: Option<String>) -> PyResult<Self> {
        let tool = match api_key {
            Some(k) => TavilySearchTool::new(k),
            None => TavilySearchTool::from_env().map_err(to_py_err)?,
        };
        Ok(Self { inner: Arc::new(tool) })
    }
}
impl_tool!(PyTavilySearchTool, TavilySearchTool, "TavilySearchTool()");

// ─── SerpApiSearchTool ──────────────────────────────────────────────────────

/// Google search via SerpApi (requires API key or SERPAPI_API_KEY env var).
#[pyclass(name = "SerpApiSearchTool")]
pub struct PySerpApiSearchTool {
    pub inner: Arc<SerpApiSearchTool>,
}

#[pymethods]
impl PySerpApiSearchTool {
    #[new]
    #[pyo3(signature = (api_key=None))]
    fn new(api_key: Option<String>) -> PyResult<Self> {
        let tool = match api_key {
            Some(k) => SerpApiSearchTool::new(k),
            None => SerpApiSearchTool::from_env().map_err(to_py_err)?,
        };
        Ok(Self { inner: Arc::new(tool) })
    }
}
impl_tool!(PySerpApiSearchTool, SerpApiSearchTool, "SerpApiSearchTool()");

// ─── GoogleSerperTool ───────────────────────────────────────────────────────

/// Google search via Serper.dev (requires API key or SERPER_API_KEY env var).
#[pyclass(name = "GoogleSerperTool")]
pub struct PyGoogleSerperTool {
    pub inner: Arc<GoogleSerperTool>,
}

#[pymethods]
impl PyGoogleSerperTool {
    #[new]
    #[pyo3(signature = (api_key=None))]
    fn new(api_key: Option<String>) -> PyResult<Self> {
        let tool = match api_key {
            Some(k) => GoogleSerperTool::new(k),
            None => GoogleSerperTool::from_env().map_err(to_py_err)?,
        };
        Ok(Self { inner: Arc::new(tool) })
    }
}
impl_tool!(PyGoogleSerperTool, GoogleSerperTool, "GoogleSerperTool()");

// ─── BraveSearchTool ────────────────────────────────────────────────────────

/// Web search via Brave Search API (requires API key or BRAVE_API_KEY env var).
#[pyclass(name = "BraveSearchTool")]
pub struct PyBraveSearchTool {
    pub inner: Arc<BraveSearchTool>,
}

#[pymethods]
impl PyBraveSearchTool {
    #[new]
    #[pyo3(signature = (api_key=None))]
    fn new(api_key: Option<String>) -> PyResult<Self> {
        let tool = match api_key {
            Some(k) => BraveSearchTool::new(k),
            None => BraveSearchTool::from_env().map_err(to_py_err)?,
        };
        Ok(Self { inner: Arc::new(tool) })
    }
}
impl_tool!(PyBraveSearchTool, BraveSearchTool, "BraveSearchTool()");
