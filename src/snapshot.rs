//! StateSnapshot — captures the full state at a given execution step.
//!
//! Mirrors LangGraph's checkpoint/snapshot concept:
//! each step produces a snapshot that can be stored and replayed.
//!
//! Python API:
//!     snap = state.snapshot("step-1")
//!     snap.step_id        # "step-1"
//!     snap.state          # dict of all fields
//!     snap.created_at     # unix timestamp (int)
//!     snap.metadata       # arbitrary metadata dict
//!     state.restore(snap) # replay this snapshot

use pyo3::prelude::*;
use pyo3::types::PyDict;

pub use flowgentra_ai::core::state::StateSnapshot;

use crate::json_to_py;

// ── PyStateSnapshot (#[pyclass]) ─────────────────────────────────────────────

/// A snapshot of graph state at a single execution step.
///
/// Produced by `state.snapshot(step_id)` or returned from checkpointers.
///
/// Example:
///     snap = state.snapshot("after-node-a")
///     print(snap.step_id)       # "after-node-a"
///     print(snap.state)         # {"messages": [...], "steps": 3}
///     print(snap.created_at)    # 1720000000
///     state.restore(snap)       # roll back to this point
#[pyclass(name = "StateSnapshot")]
#[derive(Clone)]
pub struct PyStateSnapshot {
    pub(crate) inner: StateSnapshot,
}

#[pymethods]
impl PyStateSnapshot {
    /// The unique identifier for this snapshot.
    #[getter]
    fn step_id(&self) -> &str {
        &self.inner.step_id
    }

    /// The state at the time of snapshot, as a Python dict.
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (k, v) in &self.inner.state {
            dict.set_item(k, json_to_py(py, v)?)?;
        }
        Ok(dict.into())
    }

    /// Unix timestamp (seconds) when this snapshot was taken.
    #[getter]
    fn created_at(&self) -> u64 {
        self.inner.created_at
    }

    /// Metadata dict attached to this snapshot.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (k, v) in &self.inner.metadata {
            dict.set_item(k, json_to_py(py, v)?)?;
        }
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "StateSnapshot(step_id={:?}, fields={:?}, created_at={})",
            self.inner.step_id,
            self.inner.state.keys().collect::<Vec<_>>(),
            self.inner.created_at,
        )
    }
}
