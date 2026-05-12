//! Python bindings for extended file tools (Copy, Delete, Move, FileSearch).

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::tools::files_extended::{
    CopyFileTool, DeleteFileTool, FileSearchTool, MoveFileTool,
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

// ─── CopyFileTool ────────────────────────────────────────────────────────────

#[pyclass(name = "CopyFileTool")]
pub struct PyCopyFileTool {
    pub inner: Arc<CopyFileTool>,
}

#[pymethods]
impl PyCopyFileTool {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(CopyFileTool::default()) }
    }
}
impl_tool!(PyCopyFileTool, "CopyFileTool()");

// ─── DeleteFileTool ──────────────────────────────────────────────────────────

#[pyclass(name = "DeleteFileTool")]
pub struct PyDeleteFileTool {
    pub inner: Arc<DeleteFileTool>,
}

#[pymethods]
impl PyDeleteFileTool {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(DeleteFileTool::default()) }
    }
}
impl_tool!(PyDeleteFileTool, "DeleteFileTool()");

// ─── MoveFileTool ────────────────────────────────────────────────────────────

#[pyclass(name = "MoveFileTool")]
pub struct PyMoveFileTool {
    pub inner: Arc<MoveFileTool>,
}

#[pymethods]
impl PyMoveFileTool {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(MoveFileTool::default()) }
    }
}
impl_tool!(PyMoveFileTool, "MoveFileTool()");

// ─── FileSearchTool ──────────────────────────────────────────────────────────

#[pyclass(name = "FileSearchTool")]
pub struct PyFileSearchTool {
    pub inner: Arc<FileSearchTool>,
}

#[pymethods]
impl PyFileSearchTool {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(FileSearchTool::default()) }
    }
}
impl_tool!(PyFileSearchTool, "FileSearchTool()");
