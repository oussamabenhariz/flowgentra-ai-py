//! Python bindings for communication tools (Gmail, Slack).

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::tools::communication::{GmailTool, SlackTool};
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
                    .map_err(|e| pyo3::exceptions::crate::error::ToolExecutionError::new_err(e.to_string()))?;
                json_to_py(py, &val)
            }
            fn __repr__(&self) -> String { $repr.to_string() }
        }
    };
}

#[pyclass(name = "GmailTool")]
pub struct PyGmailTool {
    pub inner: Arc<GmailTool>,
}

#[pymethods]
impl PyGmailTool {
    #[new]
    #[pyo3(signature = (access_token=None))]
    fn new(access_token: Option<String>) -> PyResult<Self> {
        let tool = match access_token {
            Some(t) => GmailTool::new(t),
            None => GmailTool::from_env().map_err(to_py_err)?,
        };
        Ok(Self { inner: Arc::new(tool) })
    }
}
impl_tool!(PyGmailTool, "GmailTool()");

#[pyclass(name = "SlackTool")]
pub struct PySlackTool {
    pub inner: Arc<SlackTool>,
}

#[pymethods]
impl PySlackTool {
    #[new]
    #[pyo3(signature = (bot_token=None))]
    fn new(bot_token: Option<String>) -> PyResult<Self> {
        let tool = match bot_token {
            Some(t) => SlackTool::new(t),
            None => SlackTool::from_env().map_err(to_py_err)?,
        };
        Ok(Self { inner: Arc::new(tool) })
    }
}
impl_tool!(PySlackTool, "SlackTool()");
