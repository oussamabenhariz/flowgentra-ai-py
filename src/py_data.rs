//! Python bindings for data tools (JSON, CSV).

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::tools::data::{CsvQueryTool, JsonGetValueTool, JsonListKeysTool};
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

#[pyclass(name = "JsonGetValueTool")]
pub struct PyJsonGetValueTool {
    pub inner: Arc<JsonGetValueTool>,
}

#[pymethods]
impl PyJsonGetValueTool {
    #[new]
    fn new() -> Self { Self { inner: Arc::new(JsonGetValueTool) } }
}
impl_tool!(PyJsonGetValueTool, "JsonGetValueTool()");

#[pyclass(name = "JsonListKeysTool")]
pub struct PyJsonListKeysTool {
    pub inner: Arc<JsonListKeysTool>,
}

#[pymethods]
impl PyJsonListKeysTool {
    #[new]
    fn new() -> Self { Self { inner: Arc::new(JsonListKeysTool) } }
}
impl_tool!(PyJsonListKeysTool, "JsonListKeysTool()");

#[pyclass(name = "CsvQueryTool")]
pub struct PyCsvQueryTool {
    pub inner: Arc<CsvQueryTool>,
}

#[pymethods]
impl PyCsvQueryTool {
    #[new]
    fn new() -> Self { Self { inner: Arc::new(CsvQueryTool) } }
}
impl_tool!(PyCsvQueryTool, "CsvQueryTool()");
