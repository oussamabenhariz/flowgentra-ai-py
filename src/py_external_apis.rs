//! Python bindings for external API tools (weather, news, finance).

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::tools::finance::AlphaVantageTool;
use flowgentra_ai::core::tools::news::NewsApiTool;
use flowgentra_ai::core::tools::weather::OpenWeatherMapTool;
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

// ─── OpenWeatherMapTool ──────────────────────────────────────────────────────

#[pyclass(name = "OpenWeatherMapTool")]
pub struct PyOpenWeatherMapTool {
    pub inner: Arc<OpenWeatherMapTool>,
}

#[pymethods]
impl PyOpenWeatherMapTool {
    #[new]
    #[pyo3(signature = (api_key=None))]
    fn new(api_key: Option<String>) -> PyResult<Self> {
        let tool = match api_key {
            Some(k) => OpenWeatherMapTool::new(k),
            None => OpenWeatherMapTool::from_env().map_err(to_py_err)?,
        };
        Ok(Self { inner: Arc::new(tool) })
    }
}
impl_tool!(PyOpenWeatherMapTool, "OpenWeatherMapTool()");

// ─── NewsApiTool ─────────────────────────────────────────────────────────────

#[pyclass(name = "NewsApiTool")]
pub struct PyNewsApiTool {
    pub inner: Arc<NewsApiTool>,
}

#[pymethods]
impl PyNewsApiTool {
    #[new]
    #[pyo3(signature = (api_key=None))]
    fn new(api_key: Option<String>) -> PyResult<Self> {
        let tool = match api_key {
            Some(k) => NewsApiTool::new(k),
            None => NewsApiTool::from_env().map_err(to_py_err)?,
        };
        Ok(Self { inner: Arc::new(tool) })
    }
}
impl_tool!(PyNewsApiTool, "NewsApiTool()");

// ─── AlphaVantageTool ────────────────────────────────────────────────────────

#[pyclass(name = "AlphaVantageTool")]
pub struct PyAlphaVantageTool {
    pub inner: Arc<AlphaVantageTool>,
}

#[pymethods]
impl PyAlphaVantageTool {
    #[new]
    #[pyo3(signature = (api_key=None))]
    fn new(api_key: Option<String>) -> PyResult<Self> {
        let tool = match api_key {
            Some(k) => AlphaVantageTool::new(k),
            None => AlphaVantageTool::from_env().map_err(to_py_err)?,
        };
        Ok(Self { inner: Arc::new(tool) })
    }
}
impl_tool!(PyAlphaVantageTool, "AlphaVantageTool()");
