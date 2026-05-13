//! Python bindings for FileCheckpointer

use pyo3::prelude::*;
use pyo3::types::PyDict;

use flowgentra_ai::core::state::DynState;
use flowgentra_ai::core::state_graph::{Checkpoint, Checkpointer, FileCheckpointer};

use crate::error::to_py_err_generic;
use crate::graph::pydict_to_dynstate;
use crate::json_to_py;

// ─── PyFileCheckpointer ────────────────────────────────────────────────────

/// File-based checkpointer for persisting graph state to disk.
///
/// Saves checkpoints as JSON files organized by thread ID.
///
/// Example:
///     cp = FileCheckpointer("./checkpoints")
///     # Used with StateGraph for persistent workflows
#[pyclass(name = "FileCheckpointer")]
pub struct PyFileCheckpointer {
    inner: FileCheckpointer,
}

#[pymethods]
impl PyFileCheckpointer {
    /// Create a new file checkpointer at the given directory.
    /// Creates the directory if it doesn't exist.
    #[new]
    fn new(base_dir: &str) -> PyResult<Self> {
        let cp = FileCheckpointer::new(base_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(PyFileCheckpointer { inner: cp })
    }

    /// Save a checkpoint for a thread.
    ///
    /// Args:
    ///     state: Current state as a plain dict.
    fn save(
        &self,
        thread_id: &str,
        step: usize,
        node_name: &str,
        state: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let shared_state = pydict_to_dynstate(state)?;
        let checkpoint = Checkpoint {
            thread_id: thread_id.to_string(),
            step,
            node_name: node_name.to_string(),
            state: shared_state,
            timestamp: chrono_timestamp(),
            metadata: std::collections::HashMap::new(),
            schema_version: "1.0".to_string(),
        };
        crate::run_async(Checkpointer::<DynState>::save(&self.inner, &checkpoint))
            .map_err(to_py_err_generic)
    }

    /// Load a specific checkpoint by thread ID and step.
    fn load(&self, thread_id: &str, step: usize) -> PyResult<Option<PyObject>> {
        let result = crate::run_async(Checkpointer::<DynState>::load(&self.inner, thread_id, step))
            .map_err(to_py_err_generic)?;
        match result {
            Some(cp) => Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("thread_id", &cp.thread_id)?;
                dict.set_item("step", cp.step)?;
                dict.set_item("node_name", &cp.node_name)?;
                dict.set_item("timestamp", cp.timestamp)?;
                let state_val = cp.state.to_value();
                dict.set_item("state", json_to_py(py, &state_val)?)?;
                Ok(Some(dict.into()))
            }),
            None => Ok(None),
        }
    }

    /// Load the latest checkpoint for a thread.
    fn load_latest(&self, thread_id: &str) -> PyResult<Option<PyObject>> {
        let result = crate::run_async(Checkpointer::<DynState>::load_latest(
            &self.inner,
            thread_id,
        ))
        .map_err(to_py_err_generic)?;
        match result {
            Some(cp) => Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("thread_id", &cp.thread_id)?;
                dict.set_item("step", cp.step)?;
                dict.set_item("node_name", &cp.node_name)?;
                dict.set_item("timestamp", cp.timestamp)?;
                let state_val = cp.state.to_value();
                dict.set_item("state", json_to_py(py, &state_val)?)?;
                Ok(Some(dict.into()))
            }),
            None => Ok(None),
        }
    }

    /// List all checkpoints for a thread as [(step, timestamp), ...].
    fn list_checkpoints(&self, thread_id: &str) -> PyResult<Vec<(usize, i64)>> {
        crate::run_async(Checkpointer::<DynState>::list_checkpoints(
            &self.inner,
            thread_id,
        ))
        .map_err(to_py_err_generic)
    }

    /// Delete a specific checkpoint.
    fn delete(&self, thread_id: &str, step: usize) -> PyResult<()> {
        crate::run_async(Checkpointer::<DynState>::delete(
            &self.inner,
            thread_id,
            step,
        ))
        .map_err(to_py_err_generic)
    }

    /// Delete all checkpoints for a thread.
    fn delete_thread(&self, thread_id: &str) -> PyResult<()> {
        crate::run_async(Checkpointer::<DynState>::delete_thread(
            &self.inner,
            thread_id,
        ))
        .map_err(to_py_err_generic)
    }

    fn __repr__(&self) -> String {
        "FileCheckpointer(...)".to_string()
    }
}

fn chrono_timestamp() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
