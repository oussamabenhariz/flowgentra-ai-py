//! Python binding for HumanInputTool.

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::tools::human::HumanInputTool;
use flowgentra_ai::core::tools::Tool;

use crate::error::to_py_err;
use crate::{json_to_py, py_to_json};

#[pyclass(name = "HumanInputTool")]
pub struct PyHumanInputTool {
    pub inner: Arc<HumanInputTool>,
}

#[pymethods]
impl PyHumanInputTool {
    #[new]
    fn new() -> Self { Self { inner: Arc::new(HumanInputTool) } }

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

    fn __repr__(&self) -> String { "HumanInputTool()".to_string() }
}
