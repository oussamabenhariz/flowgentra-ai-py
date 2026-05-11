//! Python bindings for Tool types

use pyo3::prelude::*;

use crate::{json_to_py, py_to_json};

// ─── PyToolCallRequest ──────────────────────────────────────────────────────

/// A request to call a tool
#[pyclass(name = "ToolCallRequest")]
#[derive(Clone)]
pub struct PyToolCallRequest {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[pymethods]
impl PyToolCallRequest {
    #[new]
    fn new(name: String, arguments: &Bound<'_, PyAny>) -> PyResult<Self> {
        let args = py_to_json(arguments)?;
        Ok(PyToolCallRequest {
            name,
            arguments: args,
        })
    }

    #[getter]
    fn get_name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    fn get_arguments(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_to_py(py, &self.arguments)
    }

    fn __repr__(&self) -> String {
        format!("ToolCallRequest(name='{}')", self.name)
    }
}

// ─── PyToolCallResult ───────────────────────────────────────────────────────

/// Result from a tool call
#[pyclass(name = "ToolCallResult")]
#[derive(Clone)]
pub struct PyToolCallResult {
    pub tool_call_id: String,
    pub content: String,
    pub success: bool,
}

#[pymethods]
impl PyToolCallResult {
    #[new]
    fn new(tool_call_id: String, content: String, success: bool) -> Self {
        PyToolCallResult {
            tool_call_id,
            content,
            success,
        }
    }

    #[getter]
    fn get_tool_call_id(&self) -> String {
        self.tool_call_id.clone()
    }

    #[getter]
    fn get_content(&self) -> String {
        self.content.clone()
    }

    #[getter]
    fn get_success(&self) -> bool {
        self.success
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolCallResult(id='{}', success={})",
            self.tool_call_id, self.success
        )
    }
}
