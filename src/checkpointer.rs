//! Checkpointer implementations for LangGraph-style state history.
//!
//! A Checkpointer stores ordered state snapshots keyed by `thread_id`.
//! This mirrors LangGraph's `MemorySaver` and `SqliteSaver` pattern.
//!
//! Python API:
//!     cp = MemoryCheckpointer()
//!     cp.save("thread-1", state.to_dict())
//!     snaps = cp.list("thread-1")        # all snapshots, oldest first
//!     latest = cp.load_latest("thread-1") # most recent snapshot
//!     snap   = cp.load("thread-1", "step-0")
//!     cp.delete_thread("thread-1")
//!
//!     # FileCheckpointer persists to disk as JSON
//!     fcp = FileCheckpointer("./checkpoints")

use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
use pyo3::types::PyDict;
use serde_json::Value;

use crate::error::{CheckpointError, SerializationError, ValidationError};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

use crate::json_to_py;
use crate::snapshot::PyStateSnapshot;
use flowgentra_ai::core::state::StateSnapshot;

// ── helper: Python dict → HashMap<String, Value> ─────────────────────────────

fn dict_to_map(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, Value>> {
    let mut map = HashMap::new();
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let val = crate::py_to_json(&v)?;
        map.insert(key, val);
    }
    Ok(map)
}

fn map_to_py(py: Python<'_>, map: &HashMap<String, Value>) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    for (k, v) in map {
        dict.set_item(k, json_to_py(py, v)?)?;
    }
    Ok(dict.into())
}

// ── MemoryCheckpointer ────────────────────────────────────────────────────────

/// In-memory checkpointer — stores snapshots in a `HashMap<thread_id, Vec<snapshot>>`.
///
/// Fast and zero-config, but does not survive process restart.
///
/// Example:
///     cp = MemoryCheckpointer()
///     app = graph.compile(checkpointer=cp)
///
///     result = app.invoke({"messages": []}, thread_id="user-42")
///     history = cp.list("user-42")   # all steps
///     last    = cp.load_latest("user-42")
#[pyclass(name = "MemoryCheckpointer")]
#[derive(Clone)]
pub struct PyMemoryCheckpointer {
    /// thread_id → ordered list of snapshots (oldest first)
    storage: Arc<RwLock<HashMap<String, Vec<StateSnapshot>>>>,
}

#[pymethods]
impl PyMemoryCheckpointer {
    #[new]
    fn new() -> Self {
        PyMemoryCheckpointer {
            storage: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Save a state dict as a new snapshot under `thread_id`.
    ///
    /// Returns the generated `step_id` string.
    fn save(&self, thread_id: &str, state_dict: &Bound<'_, PyDict>) -> PyResult<String> {
        let state = dict_to_map(state_dict)?;
        let step_id = Uuid::new_v4().to_string();
        let snap = StateSnapshot::new(step_id.clone(), state);

        self.storage
            .write()
            .map_err(|e| PyIOError::new_err(format!("lock poisoned: {}", e)))?
            .entry(thread_id.to_string())
            .or_default()
            .push(snap);

        Ok(step_id)
    }

    /// Save a `StateSnapshot` object directly.
    fn save_snapshot(&self, thread_id: &str, snapshot: &PyStateSnapshot) -> PyResult<()> {
        self.storage
            .write()
            .map_err(|e| PyIOError::new_err(format!("lock poisoned: {}", e)))?
            .entry(thread_id.to_string())
            .or_default()
            .push(snapshot.inner.clone());
        Ok(())
    }

    /// Return the most recent snapshot for a thread, or `None` if empty.
    fn load_latest(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let guard = self.storage.read()
            .map_err(|e| PyIOError::new_err(format!("lock poisoned: {}", e)))?;
        match guard.get(thread_id).and_then(|v| v.last()) {
            Some(snap) => Ok(PyStateSnapshot { inner: snap.clone() }.into_py(py)),
            None => Ok(py.None()),
        }
    }

    /// Return the snapshot with the given `step_id`, or `None`.
    fn load(&self, py: Python<'_>, thread_id: &str, step_id: &str) -> PyResult<PyObject> {
        let guard = self.storage.read()
            .map_err(|e| PyIOError::new_err(format!("lock poisoned: {}", e)))?;
        match guard.get(thread_id).and_then(|v| v.iter().find(|s| s.step_id == step_id)) {
            Some(snap) => Ok(PyStateSnapshot { inner: snap.clone() }.into_py(py)),
            None => Ok(py.None()),
        }
    }

    /// Return all snapshots for a thread (oldest first).
    fn list(&self, thread_id: &str) -> PyResult<Vec<PyStateSnapshot>> {
        let guard = self.storage.read()
            .map_err(|e| PyIOError::new_err(format!("lock poisoned: {}", e)))?;
        Ok(guard
            .get(thread_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|inner| PyStateSnapshot { inner })
            .collect())
    }

    /// Delete all snapshots for a thread.
    fn delete_thread(&self, thread_id: &str) -> PyResult<()> {
        self.storage
            .write()
            .map_err(|e| PyIOError::new_err(format!("lock poisoned: {}", e)))?
            .remove(thread_id);
        Ok(())
    }

    /// Return a list of all thread IDs that have at least one snapshot.
    fn thread_ids(&self) -> PyResult<Vec<String>> {
        Ok(self.storage.read()
            .map_err(|e| PyIOError::new_err(format!("lock poisoned: {}", e)))?
            .keys()
            .cloned()
            .collect())
    }

    /// Return the number of snapshots stored for a thread.
    fn len(&self, thread_id: &str) -> PyResult<usize> {
        Ok(self.storage.read()
            .map_err(|e| PyIOError::new_err(format!("lock poisoned: {}", e)))?
            .get(thread_id)
            .map(|v| v.len())
            .unwrap_or(0))
    }

    /// Update state fields inside an existing snapshot identified by `step_id`.
    ///
    /// The patch dict is merged into the snapshot's state using `LastValue` semantics
    /// (plain overwrite — reducers are not applied here, this is a direct override).
    fn update_state(
        &self,
        thread_id: &str,
        step_id: &str,
        patch: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let updates = dict_to_map(patch)?;
        let mut guard = self.storage.write()
            .map_err(|e| PyIOError::new_err(format!("lock poisoned: {}", e)))?;

        let snaps = guard
            .get_mut(thread_id)
            .ok_or_else(|| CheckpointError::new_err(format!("thread '{}' not found", thread_id)))?;

        let snap = snaps
            .iter_mut()
            .find(|s| s.step_id == step_id)
            .ok_or_else(|| CheckpointError::new_err(format!("step_id '{}' not found", step_id)))?;

        for (k, v) in updates {
            snap.state.insert(k, v);
        }
        Ok(())
    }

    /// Get a state dict for the given `step_id`.
    fn get_state(&self, py: Python<'_>, thread_id: &str, step_id: &str) -> PyResult<PyObject> {
        self.load(py, thread_id, step_id)
    }

    /// Return the full state history as a list of dicts (for debugging).
    fn get_state_history(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let snaps = self.list(thread_id)?;
        let list = pyo3::types::PyList::empty_bound(py);
        for snap in snaps {
            let dict = PyDict::new_bound(py);
            dict.set_item("step_id", snap.inner.step_id.as_str())?;
            dict.set_item("state", map_to_py(py, &snap.inner.state)?)?;
            dict.set_item("created_at", snap.inner.created_at)?;
            dict.set_item("metadata", map_to_py(py, &snap.inner.metadata)?)?;
            list.append(dict)?;
        }
        Ok(list.into())
    }

    fn __repr__(&self) -> PyResult<String> {
        let guard = self.storage.read().map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(format!("MemoryCheckpointer(threads={})", guard.len()))
    }
}

// ── FileCheckpointer ──────────────────────────────────────────────────────────

/// File-backed checkpointer — persists each snapshot as a JSON file on disk.
///
/// Layout:
///   `base_dir/<thread_id>/<step_id>.json`
///
/// Example:
///     fcp = FileCheckpointer("./checkpoints")
///     app  = graph.compile(checkpointer=fcp)
///     result = app.invoke({"messages": []}, thread_id="user-42")
///     # Snapshots are now on disk and survive restarts.
#[pyclass(name = "FileCheckpointer")]
#[derive(Clone)]
pub struct PyFileCheckpointer {
    base_dir: PathBuf,
}

#[pymethods]
impl PyFileCheckpointer {
    /// Create a new FileCheckpointer rooted at `base_dir`.
    ///
    /// The directory is created if it does not exist.
    #[new]
    fn new(base_dir: &str) -> PyResult<Self> {
        let path = PathBuf::from(base_dir);
        std::fs::create_dir_all(&path)
            .map_err(|e| PyIOError::new_err(format!("cannot create dir {}: {}", base_dir, e)))?;
        Ok(PyFileCheckpointer { base_dir: path })
    }

    /// Save a state dict as a new snapshot under `thread_id`.
    fn save(&self, thread_id: &str, state_dict: &Bound<'_, PyDict>) -> PyResult<String> {
        let state = dict_to_map(state_dict)?;
        let step_id = Uuid::new_v4().to_string();
        let snap = StateSnapshot::new(step_id.clone(), state);
        self.write_snap(thread_id, &snap)?;
        Ok(step_id)
    }

    /// Save a `StateSnapshot` object directly.
    fn save_snapshot(&self, thread_id: &str, snapshot: &PyStateSnapshot) -> PyResult<()> {
        self.write_snap(thread_id, &snapshot.inner)
    }

    /// Return the most recent snapshot for a thread (largest timestamp), or `None`.
    fn load_latest(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let snaps = self.read_all(thread_id)?;
        match snaps.into_iter().max_by_key(|s| s.created_at) {
            Some(snap) => Ok(PyStateSnapshot { inner: snap }.into_py(py)),
            None => Ok(py.None()),
        }
    }

    /// Return the snapshot with the given `step_id`, or `None`.
    fn load(&self, py: Python<'_>, thread_id: &str, step_id: &str) -> PyResult<PyObject> {
        let path = self.snap_path(thread_id, step_id)?;
        if !path.exists() {
            return Ok(py.None());
        }
        let data = std::fs::read_to_string(&path)
            .map_err(|e| PyIOError::new_err(format!("read error: {}", e)))?;
        let snap: StateSnapshot = serde_json::from_str(&data)
            .map_err(|e| SerializationError::new_err(format!("JSON decode error: {}", e)))?;
        Ok(PyStateSnapshot { inner: snap }.into_py(py))
    }

    /// Return all snapshots for a thread (sorted oldest-first).
    fn list(&self, thread_id: &str) -> PyResult<Vec<PyStateSnapshot>> {
        let mut snaps = self.read_all(thread_id)?;
        snaps.sort_by_key(|s| s.created_at);
        Ok(snaps.into_iter().map(|inner| PyStateSnapshot { inner }).collect())
    }

    /// Delete all snapshot files for a thread.
    fn delete_thread(&self, thread_id: &str) -> PyResult<()> {
        let dir = self.thread_dir(thread_id)?;
        if dir.exists() {
            std::fs::remove_dir_all(&dir)
                .map_err(|e| PyIOError::new_err(format!("remove error: {}", e)))?;
        }
        Ok(())
    }

    /// Update state fields inside an existing snapshot on disk.
    fn update_state(
        &self,
        thread_id: &str,
        step_id: &str,
        patch: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let path = self.snap_path(thread_id, step_id)?;
        if !path.exists() {
            return Err(CheckpointError::new_err(format!(
                "step_id '{}' not found for thread '{}'",
                step_id, thread_id
            )));
        }
        let data = std::fs::read_to_string(&path)
            .map_err(|e| PyIOError::new_err(format!("read error: {}", e)))?;
        let mut snap: StateSnapshot = serde_json::from_str(&data)
            .map_err(|e| SerializationError::new_err(format!("JSON decode error: {}", e)))?;

        let updates = dict_to_map(patch)?;
        for (k, v) in updates {
            snap.state.insert(k, v);
        }
        self.write_snap(thread_id, &snap)?;
        Ok(())
    }

    /// Return full state history as a list of dicts.
    fn get_state_history(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let snaps = self.list(thread_id)?;
        let list = pyo3::types::PyList::empty_bound(py);
        for snap in snaps {
            let dict = PyDict::new_bound(py);
            dict.set_item("step_id", snap.inner.step_id.as_str())?;
            dict.set_item("state", map_to_py(py, &snap.inner.state)?)?;
            dict.set_item("created_at", snap.inner.created_at)?;
            dict.set_item("metadata", map_to_py(py, &snap.inner.metadata)?)?;
            list.append(dict)?;
        }
        Ok(list.into())
    }

    fn __repr__(&self) -> String {
        format!("FileCheckpointer(base_dir={:?})", self.base_dir)
    }
}

impl PyFileCheckpointer {
    /// Validate that `component` (a thread_id or step_id segment) does not
    /// contain path-traversal sequences or other forbidden characters.
    fn validate_path_component(component: &str, label: &str) -> PyResult<()> {
        if component.is_empty() {
            return Err(ValidationError::new_err(format!("{} must not be empty", label)));
        }
        if component.contains("..") || component.contains('/') || component.contains('\\') || component.contains('\0') {
            return Err(ValidationError::new_err(format!(
                "{} '{}' contains invalid characters (path traversal is not allowed)",
                label, component
            )));
        }
        Ok(())
    }

    fn thread_dir(&self, thread_id: &str) -> PyResult<PathBuf> {
        Self::validate_path_component(thread_id, "thread_id")?;
        Ok(self.base_dir.join(thread_id))
    }

    fn snap_path(&self, thread_id: &str, step_id: &str) -> PyResult<PathBuf> {
        Self::validate_path_component(step_id, "step_id")?;
        Ok(self.thread_dir(thread_id)?.join(format!("{}.json", step_id)))
    }

    fn write_snap(&self, thread_id: &str, snap: &StateSnapshot) -> PyResult<()> {
        let dir = self.thread_dir(thread_id)?;
        std::fs::create_dir_all(&dir)
            .map_err(|e| PyIOError::new_err(format!("create dir error: {}", e)))?;
        // step_id is generated internally (UUID), but validate defensively.
        Self::validate_path_component(&snap.step_id, "step_id")?;
        let path = dir.join(format!("{}.json", snap.step_id));
        let data = serde_json::to_string_pretty(snap)
            .map_err(|e| PyIOError::new_err(format!("serialize error: {}", e)))?;
        std::fs::write(&path, data)
            .map_err(|e| PyIOError::new_err(format!("write error: {}", e)))?;
        Ok(())
    }

    fn read_all(&self, thread_id: &str) -> PyResult<Vec<StateSnapshot>> {
        let dir = self.thread_dir(thread_id)?;
        if !dir.exists() {
            return Ok(vec![]);
        }
        let mut snaps = vec![];
        for entry in std::fs::read_dir(&dir)
            .map_err(|e| PyIOError::new_err(format!("read_dir error: {}", e)))?
        {
            let entry = entry.map_err(|e| PyIOError::new_err(format!("entry error: {}", e)))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let data = std::fs::read_to_string(&path)
                    .map_err(|e| PyIOError::new_err(format!("read error: {}", e)))?;
                let snap: StateSnapshot = serde_json::from_str(&data)
                    .map_err(|e| SerializationError::new_err(format!("JSON decode error: {}", e)))?;
                snaps.push(snap);
            }
        }
        Ok(snaps)
    }
}
