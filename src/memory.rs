//! Python bindings for Memory types (Checkpoint, CheckpointMetadata)

use pyo3::prelude::*;

use flowgentra_ai::core::memory::{Checkpoint, CheckpointMetadata};
use flowgentra_ai::core::state::DynState;

use crate::json_to_py;

// ─── PyCheckpointMetadata ───────────────────────────────────────────────────

/// Metadata for a checkpoint
#[pyclass(name = "CheckpointMetadata")]
#[derive(Clone)]
pub struct PyCheckpointMetadata {
    pub(crate) inner: CheckpointMetadata,
}

#[pymethods]
impl PyCheckpointMetadata {
    #[getter]
    fn last_node(&self) -> Option<String> {
        self.inner.last_node.clone()
    }

    #[getter]
    fn execution_path(&self) -> Vec<String> {
        self.inner.execution_path.clone()
    }

    #[getter]
    fn extra(&self, py: Python<'_>) -> PyResult<PyObject> {
        let val = serde_json::to_value(&self.inner.extra)
            .map_err(|e| crate::error::CheckpointError::new_err(format!("{}", e)))?;
        json_to_py(py, &val)
    }

    fn __repr__(&self) -> String {
        format!(
            "CheckpointMetadata(last_node={:?}, path={:?})",
            self.inner.last_node, self.inner.execution_path
        )
    }
}

// ─── PyCheckpoint ───────────────────────────────────────────────────────────

/// A state checkpoint for persistence/resume
#[pyclass(name = "Checkpoint")]
#[derive(Clone)]
pub struct PyCheckpoint {
    pub(crate) inner: Checkpoint,
}

#[pymethods]
impl PyCheckpoint {
    #[getter]
    fn metadata(&self) -> PyCheckpointMetadata {
        PyCheckpointMetadata {
            inner: self.inner.metadata.clone(),
        }
    }

    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Use the state() method which deserializes from the internal Value
        let shared: Result<DynState, _> = self.inner.state();
        match shared {
            Ok(s) => {
                let val = s.to_value();
                json_to_py(py, &val)
            }
            Err(e) => Err(crate::error::CheckpointError::new_err(format!("{}", e))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Checkpoint(last_node={:?}, path={:?})",
            self.inner.metadata.last_node, self.inner.metadata.execution_path
        )
    }
}
