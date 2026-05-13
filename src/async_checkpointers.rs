//! Python bindings for async checkpointer backends.
//!
//! All backends expose the same interface:
//!
//! ```python
//! cp = SqliteAsyncCheckpointer("sqlite:///checkpoints.db")
//! cp.save("user-42", {"messages": [...], "step": 3})
//! cp.save("user-42", {"step": 4}, last_node="summarize")
//! latest = cp.load("user-42")          # Checkpoint | None
//! history = cp.list_history("user-42") # List[CheckpointHistoryEntry]
//! threads = cp.list_threads()           # List[str]
//! cp.delete_thread("user-42")
//!
//! # Namespace isolation (multi-tenant)
//! namespaced = NamespacedCheckpointer(cp, "tenant-1")
//! namespaced.save("thread-1", state)
//! # stored as "tenant-1:thread-1" internally
//! ```

use pyo3::exceptions::PyConnectionError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{CheckpointError, SerializationError};

use flowgentra_ai::core::memory::async_checkpointer::{AsyncCheckpointer, CheckpointHistoryEntry};
use flowgentra_ai::core::memory::CheckpointMetadata;
use flowgentra_ai::core::state::DynState;

use crate::memory::{PyCheckpoint, PyCheckpointMetadata};

// ── helpers ───────────────────────────────────────────────────────────────────

fn dict_to_dynstate(dict: &Bound<'_, PyDict>) -> PyResult<DynState> {
    let val = crate::py_to_json(dict.as_any())?;
    DynState::from_json(val).map_err(|e| SerializationError::new_err(e.to_string()))
}

fn make_metadata(last_node: Option<&str>) -> CheckpointMetadata {
    CheckpointMetadata {
        last_node: last_node.map(str::to_string),
        ..Default::default()
    }
}

fn entry_to_py(entry: CheckpointHistoryEntry) -> PyCheckpointHistoryEntry {
    PyCheckpointHistoryEntry {
        thread_id: entry.thread_id,
        namespace: entry.namespace,
        saved_at: entry.saved_at,
        checkpoint: PyCheckpoint {
            inner: entry.checkpoint,
        },
    }
}

// ── CheckpointHistoryEntry ────────────────────────────────────────────────────

/// A single checkpoint record returned by `list_history()`.
///
/// Attributes:
///     thread_id (str):               Thread this checkpoint belongs to.
///     namespace (Optional[str]):     Namespace prefix (set by NamespacedCheckpointer).
///     saved_at (int):                Unix timestamp (seconds since epoch).
///     checkpoint (Checkpoint):       The actual state + metadata snapshot.
///     metadata (CheckpointMetadata): Short-cut to ``checkpoint.metadata``.
#[pyclass(name = "CheckpointHistoryEntry")]
#[derive(Clone)]
pub struct PyCheckpointHistoryEntry {
    pub thread_id: String,
    pub namespace: Option<String>,
    pub saved_at: i64,
    pub checkpoint: PyCheckpoint,
}

#[pymethods]
impl PyCheckpointHistoryEntry {
    #[getter]
    fn thread_id(&self) -> &str {
        &self.thread_id
    }

    #[getter]
    fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    #[getter]
    fn saved_at(&self) -> i64 {
        self.saved_at
    }

    #[getter]
    fn checkpoint(&self) -> PyCheckpoint {
        self.checkpoint.clone()
    }

    #[getter]
    fn metadata(&self) -> PyCheckpointMetadata {
        PyCheckpointMetadata {
            inner: self.checkpoint.inner.metadata.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CheckpointHistoryEntry(thread_id={:?}, saved_at={}, namespace={:?})",
            self.thread_id, self.saved_at, self.namespace
        )
    }
}

// ── SqliteAsyncCheckpointer ───────────────────────────────────────────────────

/// Async SQLite checkpointer with full history support.
///
/// Persists checkpoints in a local SQLite database. Each ``save()`` appends a
/// new record; ``load()`` returns the latest. Survives process restarts.
/// Single-process safe (SQLite WAL mode handles concurrent reads).
///
/// Example::
///
///     cp = SqliteAsyncCheckpointer("sqlite:///checkpoints.db")
///     # Or in-memory:
///     cp = SqliteAsyncCheckpointer("sqlite::memory:")
///     cp.save("thread-1", {"messages": [...]})
///     last = cp.load("thread-1")         # Checkpoint | None
///     history = cp.list_history("thread-1")
#[pyclass(name = "SqliteAsyncCheckpointer")]
pub struct PySqliteAsyncCheckpointer {
    inner: flowgentra_ai::core::memory::AsyncSqliteCheckpointer,
}

#[pymethods]
impl PySqliteAsyncCheckpointer {
    /// Open (or create) a SQLite checkpoint store.
    ///
    /// Args:
    ///     url: SQLite URL.  ``"sqlite:///path/to/file.db"`` or
    ///          ``"sqlite::memory:"`` for a temporary in-memory store.
    #[new]
    fn new(url: &str) -> PyResult<Self> {
        let inner = crate::run_async(flowgentra_ai::core::memory::AsyncSqliteCheckpointer::new(
            url,
        ))
        .map_err(|e| PyConnectionError::new_err(format!("SQLite connect failed: {e}")))?;
        Ok(Self { inner })
    }

    /// Save the current state for a thread.
    ///
    /// Args:
    ///     thread_id: Unique identifier for the conversation / run.
    ///     state_dict: Dict of state fields to persist.
    ///     last_node:  Optional name of the last executed node (stored in metadata).
    #[pyo3(signature = (thread_id, state_dict, last_node = None))]
    fn save(
        &self,
        thread_id: &str,
        state_dict: &Bound<'_, PyDict>,
        last_node: Option<&str>,
    ) -> PyResult<()> {
        let state = dict_to_dynstate(state_dict)?;
        let meta = make_metadata(last_node);
        crate::run_async(self.inner.save(thread_id, &state, &meta))
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    /// Load the latest checkpoint for a thread.
    ///
    /// Returns:
    ///     ``Checkpoint`` if the thread has at least one checkpoint, ``None`` otherwise.
    fn load(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let result = crate::run_async(self.inner.load(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))?;
        match result {
            None => Ok(py.None()),
            Some(cp) => Ok(PyCheckpoint { inner: cp }.into_py(py)),
        }
    }

    /// Return all thread IDs that have at least one checkpoint.
    fn list_threads(&self) -> PyResult<Vec<String>> {
        crate::run_async(self.inner.list_threads())
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    /// Return the full checkpoint history for a thread, newest first.
    fn list_history(&self, thread_id: &str) -> PyResult<Vec<PyCheckpointHistoryEntry>> {
        let entries = crate::run_async(self.inner.list_history(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))?;
        Ok(entries.into_iter().map(entry_to_py).collect())
    }

    /// Delete all checkpoints for a thread.
    fn delete_thread(&self, thread_id: &str) -> PyResult<()> {
        crate::run_async(self.inner.delete_thread(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "SqliteAsyncCheckpointer(...)".to_string()
    }
}

// ── PostgresAsyncCheckpointer ─────────────────────────────────────────────────

/// Async PostgreSQL checkpointer with full history.
///
/// Production-grade distributed checkpointing. Multiple workers can share the
/// same table. Stores all checkpoints so time-travel is possible.
///
/// Example::
///
///     cp = PostgresAsyncCheckpointer("postgres://user:pass@localhost/mydb")
///     cp.save("thread-1", {"messages": [...]})
///     latest = cp.load("thread-1")
#[pyclass(name = "PostgresAsyncCheckpointer")]
pub struct PyPostgresAsyncCheckpointer {
    inner: flowgentra_ai::core::memory::AsyncPostgresCheckpointer,
}

#[pymethods]
impl PyPostgresAsyncCheckpointer {
    /// Connect to a PostgreSQL database.
    ///
    /// Args:
    ///     url: PostgreSQL connection URL, e.g.
    ///          ``"postgres://user:pass@localhost/mydb"``.
    #[new]
    fn new(url: &str) -> PyResult<Self> {
        let inner = crate::run_async(flowgentra_ai::core::memory::AsyncPostgresCheckpointer::new(
            url,
        ))
        .map_err(|e| PyConnectionError::new_err(format!("Postgres connect failed: {e}")))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (thread_id, state_dict, last_node = None))]
    fn save(
        &self,
        thread_id: &str,
        state_dict: &Bound<'_, PyDict>,
        last_node: Option<&str>,
    ) -> PyResult<()> {
        let state = dict_to_dynstate(state_dict)?;
        let meta = make_metadata(last_node);
        crate::run_async(self.inner.save(thread_id, &state, &meta))
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn load(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let result = crate::run_async(self.inner.load(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))?;
        match result {
            None => Ok(py.None()),
            Some(cp) => Ok(PyCheckpoint { inner: cp }.into_py(py)),
        }
    }

    fn list_threads(&self) -> PyResult<Vec<String>> {
        crate::run_async(self.inner.list_threads())
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn list_history(&self, thread_id: &str) -> PyResult<Vec<PyCheckpointHistoryEntry>> {
        let entries = crate::run_async(self.inner.list_history(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))?;
        Ok(entries.into_iter().map(entry_to_py).collect())
    }

    fn delete_thread(&self, thread_id: &str) -> PyResult<()> {
        crate::run_async(self.inner.delete_thread(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "PostgresAsyncCheckpointer(...)".to_string()
    }
}

// ── RedisAsyncCheckpointer ────────────────────────────────────────────────────

/// Async Redis checkpointer (latest checkpoint per thread).
///
/// Fast, low-latency, and distributed. Stores only the latest checkpoint per
/// thread (no full history). Optionally expires checkpoints via TTL.
///
/// Example::
///
///     cp = RedisAsyncCheckpointer("redis://localhost:6379")
///     cp = RedisAsyncCheckpointer("redis://localhost:6379", ttl_secs=3600)
///     cp.save("thread-1", {"messages": [...]})
///     latest = cp.load("thread-1")
#[pyclass(name = "RedisAsyncCheckpointer")]
pub struct PyRedisAsyncCheckpointer {
    inner: flowgentra_ai::core::memory::AsyncRedisCheckpointer,
}

#[pymethods]
impl PyRedisAsyncCheckpointer {
    /// Connect to a Redis server.
    ///
    /// Args:
    ///     url:      Redis URL, e.g. ``"redis://localhost:6379"``.
    ///     ttl_secs: Optional TTL in seconds. After this duration the
    ///               checkpoint is automatically deleted by Redis.
    #[new]
    #[pyo3(signature = (url, ttl_secs = None))]
    fn new(url: &str, ttl_secs: Option<u64>) -> PyResult<Self> {
        let inner = crate::run_async(flowgentra_ai::core::memory::AsyncRedisCheckpointer::new(
            url, ttl_secs,
        ))
        .map_err(|e| PyConnectionError::new_err(format!("Redis connect failed: {e}")))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (thread_id, state_dict, last_node = None))]
    fn save(
        &self,
        thread_id: &str,
        state_dict: &Bound<'_, PyDict>,
        last_node: Option<&str>,
    ) -> PyResult<()> {
        let state = dict_to_dynstate(state_dict)?;
        let meta = make_metadata(last_node);
        crate::run_async(self.inner.save(thread_id, &state, &meta))
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn load(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let result = crate::run_async(self.inner.load(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))?;
        match result {
            None => Ok(py.None()),
            Some(cp) => Ok(PyCheckpoint { inner: cp }.into_py(py)),
        }
    }

    fn list_threads(&self) -> PyResult<Vec<String>> {
        crate::run_async(self.inner.list_threads())
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn list_history(&self, thread_id: &str) -> PyResult<Vec<PyCheckpointHistoryEntry>> {
        let entries = crate::run_async(self.inner.list_history(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))?;
        Ok(entries.into_iter().map(entry_to_py).collect())
    }

    fn delete_thread(&self, thread_id: &str) -> PyResult<()> {
        crate::run_async(self.inner.delete_thread(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "RedisAsyncCheckpointer(...)".to_string()
    }
}

// ── MongoAsyncCheckpointer ────────────────────────────────────────────────────

/// Async MongoDB checkpointer with full history.
///
/// Stores checkpoints as documents in a MongoDB collection. Supports full
/// history and multi-replica deployments. Automatically creates an index on
/// ``(thread_id, saved_at)`` for efficient queries.
///
/// Example::
///
///     cp = MongoAsyncCheckpointer(
///         "mongodb://localhost:27017",
///         db_name="myapp",
///         collection="checkpoints",
///     )
///     cp.save("thread-1", {"messages": [...]})
///     latest = cp.load("thread-1")
#[pyclass(name = "MongoAsyncCheckpointer")]
pub struct PyMongoAsyncCheckpointer {
    inner: flowgentra_ai::core::memory::AsyncMongoCheckpointer,
}

#[pymethods]
impl PyMongoAsyncCheckpointer {
    /// Connect to MongoDB.
    ///
    /// Args:
    ///     url:        MongoDB connection string.
    ///     db_name:    Database name (default ``"flowgentra"``).
    ///     collection: Collection name (default ``"checkpoints"``).
    #[new]
    #[pyo3(signature = (url, db_name = "flowgentra", collection = "checkpoints"))]
    fn new(url: &str, db_name: &str, collection: &str) -> PyResult<Self> {
        let inner = crate::run_async(flowgentra_ai::core::memory::AsyncMongoCheckpointer::new(
            url, db_name, collection,
        ))
        .map_err(|e| PyConnectionError::new_err(format!("MongoDB connect failed: {e}")))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (thread_id, state_dict, last_node = None))]
    fn save(
        &self,
        thread_id: &str,
        state_dict: &Bound<'_, PyDict>,
        last_node: Option<&str>,
    ) -> PyResult<()> {
        let state = dict_to_dynstate(state_dict)?;
        let meta = make_metadata(last_node);
        crate::run_async(self.inner.save(thread_id, &state, &meta))
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn load(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let result = crate::run_async(self.inner.load(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))?;
        match result {
            None => Ok(py.None()),
            Some(cp) => Ok(PyCheckpoint { inner: cp }.into_py(py)),
        }
    }

    fn list_threads(&self) -> PyResult<Vec<String>> {
        crate::run_async(self.inner.list_threads())
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn list_history(&self, thread_id: &str) -> PyResult<Vec<PyCheckpointHistoryEntry>> {
        let entries = crate::run_async(self.inner.list_history(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))?;
        Ok(entries.into_iter().map(entry_to_py).collect())
    }

    fn delete_thread(&self, thread_id: &str) -> PyResult<()> {
        crate::run_async(self.inner.delete_thread(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "MongoAsyncCheckpointer(...)".to_string()
    }
}

// ── MySqlAsyncCheckpointer ────────────────────────────────────────────────────

/// Async MySQL checkpointer with full history.
///
/// Stores checkpoints in a MySQL / MariaDB table. The schema is created
/// automatically on first connection.
///
/// Example::
///
///     cp = MySqlAsyncCheckpointer("mysql://user:pass@localhost/mydb")
///     cp.save("thread-1", {"messages": [...]})
///     latest = cp.load("thread-1")
#[pyclass(name = "MySqlAsyncCheckpointer")]
pub struct PyMySqlAsyncCheckpointer {
    inner: flowgentra_ai::core::memory::AsyncMysqlCheckpointer,
}

#[pymethods]
impl PyMySqlAsyncCheckpointer {
    /// Connect to MySQL / MariaDB.
    ///
    /// Args:
    ///     url: MySQL connection URL, e.g. ``"mysql://user:pass@localhost/mydb"``.
    #[new]
    fn new(url: &str) -> PyResult<Self> {
        let inner = crate::run_async(flowgentra_ai::core::memory::AsyncMysqlCheckpointer::new(
            url,
        ))
        .map_err(|e| PyConnectionError::new_err(format!("MySQL connect failed: {e}")))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (thread_id, state_dict, last_node = None))]
    fn save(
        &self,
        thread_id: &str,
        state_dict: &Bound<'_, PyDict>,
        last_node: Option<&str>,
    ) -> PyResult<()> {
        let state = dict_to_dynstate(state_dict)?;
        let meta = make_metadata(last_node);
        crate::run_async(self.inner.save(thread_id, &state, &meta))
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn load(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let result = crate::run_async(self.inner.load(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))?;
        match result {
            None => Ok(py.None()),
            Some(cp) => Ok(PyCheckpoint { inner: cp }.into_py(py)),
        }
    }

    fn list_threads(&self) -> PyResult<Vec<String>> {
        crate::run_async(self.inner.list_threads())
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn list_history(&self, thread_id: &str) -> PyResult<Vec<PyCheckpointHistoryEntry>> {
        let entries = crate::run_async(self.inner.list_history(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))?;
        Ok(entries.into_iter().map(entry_to_py).collect())
    }

    fn delete_thread(&self, thread_id: &str) -> PyResult<()> {
        crate::run_async(self.inner.delete_thread(thread_id))
            .map_err(|e| CheckpointError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "MySqlAsyncCheckpointer(...)".to_string()
    }
}

// ── NamespacedCheckpointer ────────────────────────────────────────────────────

/// Wraps any async checkpointer and scopes all thread IDs to a namespace.
///
/// Useful for multi-tenant deployments where multiple tenants or agents share
/// the same backing store. Thread IDs are stored as ``<namespace>:<thread_id>``
/// internally, preventing key collisions between tenants.
///
/// Example::
///
///     # Shared backing store
///     shared = PostgresAsyncCheckpointer("postgres://...")
///
///     # Per-tenant views — no key collisions
///     tenant_a = NamespacedCheckpointer(shared, "tenant-a")
///     tenant_b = NamespacedCheckpointer(shared, "tenant-b")
///
///     tenant_a.save("thread-1", state_a)  # stored as "tenant-a:thread-1"
///     tenant_b.save("thread-1", state_b)  # stored as "tenant-b:thread-1"
///
///     # list_threads() strips the prefix automatically
///     assert tenant_a.list_threads() == ["thread-1"]
#[pyclass(name = "NamespacedCheckpointer")]
pub struct PyNamespacedCheckpointer {
    inner: PyObject,
    namespace: String,
}

#[pymethods]
impl PyNamespacedCheckpointer {
    /// Wrap an async checkpointer with namespace scoping.
    ///
    /// Args:
    ///     inner:     Any async checkpointer instance.
    ///     namespace: Namespace prefix (e.g. tenant ID, agent name).
    #[new]
    fn new(inner: PyObject, namespace: &str) -> Self {
        Self {
            inner,
            namespace: namespace.to_string(),
        }
    }

    fn scoped_key(&self, thread_id: &str) -> String {
        format!("{}:{}", self.namespace, thread_id)
    }

    /// Namespace prefix used for all thread IDs.
    #[getter]
    fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Save the current state under ``<namespace>:<thread_id>``.
    #[pyo3(signature = (thread_id, state_dict, last_node = None))]
    fn save(
        &self,
        py: Python<'_>,
        thread_id: &str,
        state_dict: &Bound<'_, PyDict>,
        last_node: Option<&str>,
    ) -> PyResult<()> {
        let scoped = self.scoped_key(thread_id);
        let kwargs = PyDict::new_bound(py);
        if let Some(node) = last_node {
            kwargs.set_item("last_node", node)?;
        }
        self.inner
            .bind(py)
            .call_method("save", (scoped.as_str(), state_dict), Some(&kwargs))?;
        Ok(())
    }

    /// Load the latest checkpoint for the scoped thread ID.
    fn load(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let scoped = self.scoped_key(thread_id);
        self.inner.call_method1(py, "load", (scoped.as_str(),))
    }

    /// List thread IDs in this namespace (prefix stripped).
    fn list_threads(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let result = self.inner.call_method0(py, "list_threads")?;
        let all: Vec<String> = result.extract(py)?;
        let prefix = format!("{}:", self.namespace);
        Ok(all
            .into_iter()
            .filter_map(|t| t.strip_prefix(&prefix).map(str::to_string))
            .collect())
    }

    /// Return checkpoint history for the scoped thread ID.
    fn list_history(&self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        let scoped = self.scoped_key(thread_id);
        self.inner
            .call_method1(py, "list_history", (scoped.as_str(),))
    }

    /// Delete all checkpoints under the scoped thread ID.
    fn delete_thread(&self, py: Python<'_>, thread_id: &str) -> PyResult<()> {
        let scoped = self.scoped_key(thread_id);
        self.inner
            .call_method1(py, "delete_thread", (scoped.as_str(),))?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("NamespacedCheckpointer(namespace={:?})", self.namespace)
    }
}
