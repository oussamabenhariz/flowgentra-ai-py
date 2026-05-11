//! Python bindings for the new LangGraph-style State system.
//!
//! # What changed
//!
//! The old `PyState` wrapped `DynState` (a raw Arc<RwLock<JSON map>>) with no
//! concept of per-field reducers.  The new `PyState` wraps `DynState` — a
//! channel-based store where every field has a `ChannelType`:
//!
//! - `LastValue`  — overwrite (default, identical to the old behaviour)
//! - `Topic`      — list-append (like `operator.add` on lists in LangGraph)
//! - `BinaryOperator(fn)` — custom merge function
//!
//! Additionally, `PyState` now exposes `snapshot(step_id)` and `restore(snap)`
//! for time-travel / checkpointing workflows.
//!
//! Python API:
//!     state = State({"messages": [], "steps": 0})
//!     state["messages"] = ["hello"]   # raw set (no reducer)
//!     snap = state.snapshot("before-node-b")
//!     state["steps"] = 1
//!     state.restore(snap)             # roll back
//!     print(state["steps"])           # 0

use pyo3::prelude::*;
use pyo3::exceptions::PyKeyError;
use crate::error::SerializationError;
use pyo3::types::PyDict;
use serde_json::Value;

use flowgentra_ai::core::state::DynState;

use crate::snapshot::PyStateSnapshot;
use crate::{json_to_py, py_to_json};

// ── PyState (#[pyclass]) ─────────────────────────────────────────────────────

/// LangGraph-style state object with channel-based reducers.
///
/// Each field is backed by a Channel (LastValue, Topic, or BinaryOperator).
/// Channels control how partial updates are merged into the state when a
/// graph node returns a patch dict.
///
/// Standalone usage:
///     state = State({"messages": [], "steps": 0})
///     state["steps"] = 1
///     snap = state.snapshot("checkpoint-1")
///     state["steps"] = 99
///     state.restore(snap)
///     print(state["steps"])  # 1
///
/// Inside graphs, State objects are created automatically by `StateGraph.invoke()`.
#[pyclass(name = "State")]
#[derive(Clone)]
pub struct PyState {
    pub(crate) inner: DynState,
}

#[pymethods]
impl PyState {
    /// Create a new State, optionally initialised from a dict.
    ///
    /// All fields default to `LastValue` (plain overwrite) channels.
    /// To use reducers, go through `StateGraph` which reads the class schema.
    #[new]
    #[pyo3(signature = (initial=None))]
    pub fn new(initial: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let state = DynState::new();
        if let Some(dict) = initial {
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                let val = py_to_json(&v)?;
                state.set_raw(key, val);
            }
        }
        Ok(PyState { inner: state })
    }

    // ── Read ─────────────────────────────────────────────────────────────────

    /// Get the value for `key`, or `None` if absent.
    fn get(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.get(key) {
            Some(v) => json_to_py(py, &v),
            None => Ok(py.None()),
        }
    }

    /// `True` if the key exists.
    fn contains_key(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    /// All field names.
    fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }

    /// Get a string value or `None`.
    fn get_string(&self, key: &str) -> Option<String> {
        self.inner.get_string(key)
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    /// Set a field value (raw overwrite — does not apply reducers).
    fn set(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let val = py_to_json(value)?;
        self.inner.set_raw(key, val);
        Ok(())
    }

    /// Remove a key and return its former value (or `None`).
    fn remove(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.remove(key) {
            Some(v) => json_to_py(py, &v),
            None => Ok(py.None()),
        }
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    /// Return a plain Python dict of all fields.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (k, v) in self.inner.to_map() {
            dict.set_item(&k, json_to_py(py, &v)?)?;
        }
        Ok(dict.into())
    }

    /// Return a JSON string of the state.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json_string()
            .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
    }

    /// Create a State from a Python dict (all channels become LastValue).
    #[staticmethod]
    fn from_dict(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let state = DynState::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            let val = py_to_json(&v)?;
            state.set_raw(key, val);
        }
        Ok(PyState { inner: state })
    }

    /// Create a State from a JSON string.
    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let value: Value = serde_json::from_str(json_str)
            .map_err(|e| SerializationError::new_err(format!("Invalid JSON: {}", e)))?;
        match value {
            Value::Object(map) => {
                let state = DynState::new();
                for (k, v) in map {
                    state.set_raw(k, v);
                }
                Ok(PyState { inner: state })
            }
            _ => Err(SerializationError::new_err("JSON root must be an object")),
        }
    }

    // ── Snapshot / restore ───────────────────────────────────────────────────

    /// Capture the current state as a `StateSnapshot` with the given `step_id`.
    ///
    ///     snap = state.snapshot("before-node-b")
    ///     # ... run more nodes ...
    ///     state.restore(snap)  # roll back
    fn snapshot(&self, step_id: &str) -> PyStateSnapshot {
        PyStateSnapshot {
            inner: self.inner.snapshot(step_id),
        }
    }

    /// Restore field values from a `StateSnapshot` (raw overwrite).
    ///
    /// Fields NOT present in the snapshot are left unchanged.
    fn restore(&self, snapshot: &PyStateSnapshot) {
        self.inner.restore(&snapshot.inner);
    }

    // ── Clone ─────────────────────────────────────────────────────────────────

    /// Return an independent copy (changes do not affect the original).
    fn deep_clone(&self) -> Self {
        PyState {
            inner: self.inner.deep_clone(),
        }
    }

    // ── Dunder methods ────────────────────────────────────────────────────────

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.get(key) {
            Some(v) => json_to_py(py, &v),
            None => Err(PyKeyError::new_err(key.to_string())),
        }
    }

    fn __setitem__(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let val = py_to_json(value)?;
        self.inner.set_raw(key, val);
        Ok(())
    }

    fn __delitem__(&self, key: &str) -> PyResult<()> {
        if self.inner.contains_key(key) {
            self.inner.remove(key);
            Ok(())
        } else {
            Err(PyKeyError::new_err(key.to_string()))
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "State({})",
            self.inner
                .to_json_string()
                .unwrap_or_else(|_| "{}".into())
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}
