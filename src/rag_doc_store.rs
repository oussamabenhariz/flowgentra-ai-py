//! Python bindings for the RAG-level DocStore trait and its implementations.
//!
//! These are the document stores used by ``ParentDocumentRetriever`` and
//! ``MultiVectorRetriever`` to persist full parent documents.  They are distinct
//! from the NoSQL document stores in ``db.py`` (those store arbitrary JSON).
//!
//! ```python
//! from flowgentra_ai import rag
//!
//! store = rag.InMemoryDocStore()
//! store.set("d1", "Hello world", {"source": "test"})
//! doc   = store.get("d1")
//! print(doc["text"])   # "Hello world"
//!
//! store2 = rag.LocalFileDocStore("/tmp/docs")
//! store2.set("d1", "Hello", {})
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;

use flowgentra_ai::core::rag::doc_store::{
    DocStore, DocStoreError, InMemoryDocStore, LocalFileDocStore, StoredDocument,
};

use crate::{json_to_py, py_to_json, run_async};

// ── helpers ───────────────────────────────────────────────────────────────────

fn to_py_err(e: DocStoreError) -> PyErr {
    crate::error::InternalError::new_err(e.to_string())
}

fn stored_doc_to_py(py: Python<'_>, doc: &StoredDocument) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", &doc.id)?;
    dict.set_item("text", &doc.text)?;
    let meta_val = serde_json::to_value(&doc.metadata)
        .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
    dict.set_item("metadata", json_to_py(py, &meta_val)?)?;
    Ok(dict.into())
}

fn extract_meta(meta: Option<&Bound<'_, PyDict>>) -> PyResult<HashMap<String, Value>> {
    match meta {
        None => Ok(HashMap::new()),
        Some(d) => {
            let mut map = HashMap::new();
            for (k, v) in d.iter() {
                let key: String = k.extract()?;
                map.insert(key, py_to_json(&v)?);
            }
            Ok(map)
        }
    }
}

// ── PyStoredDocument ──────────────────────────────────────────────────────────

/// A document retrieved from a ``DocStore``.
///
/// Attributes:
///     id (str):       Document identifier.
///     text (str):     Document text content.
///     metadata (dict): Arbitrary key-value metadata.
#[pyclass(name = "StoredDocument")]
#[derive(Clone)]
pub struct PyStoredDocument {
    pub(crate) inner: StoredDocument,
}

#[pymethods]
impl PyStoredDocument {
    #[new]
    #[pyo3(signature = (id, text, metadata=None))]
    fn new(id: String, text: String, metadata: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: StoredDocument {
                id,
                text,
                metadata: extract_meta(metadata)?,
            },
        })
    }

    #[getter]
    fn id(&self) -> &str { &self.inner.id }

    #[getter]
    fn text(&self) -> &str { &self.inner.text }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        let val = serde_json::to_value(&self.inner.metadata)
            .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
        json_to_py(py, &val)
    }

    fn __repr__(&self) -> String {
        format!("StoredDocument(id='{}', len={})", self.inner.id, self.inner.text.len())
    }
}

// ── PyInMemoryDocStore ────────────────────────────────────────────────────────

/// Thread-safe in-memory document store.
///
/// Fast, zero-persistence store suitable for development and testing.
/// Used by ``ParentDocumentRetriever`` and ``MultiVectorRetriever``.
///
/// Example::
///
///     store = InMemoryDocStore()
///     store.set("d1", "Hello world", {"source": "test"})
///     doc = store.get("d1")   # {"id": "d1", "text": "Hello world", ...}
///     keys = store.keys()     # ["d1"]
///     store.delete("d1")
#[pyclass(name = "InMemoryDocStore")]
pub struct PyInMemoryDocStore {
    inner: Arc<InMemoryDocStore>,
}

#[pymethods]
impl PyInMemoryDocStore {
    /// Create an empty in-memory document store.
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(InMemoryDocStore::new()) }
    }

    /// Store a single document.
    ///
    /// Args:
    ///     doc_id:   Unique document identifier.
    ///     text:     Document text.
    ///     metadata: Optional dict of metadata.
    #[pyo3(signature = (doc_id, text, metadata=None))]
    fn set(&self, doc_id: &str, text: &str, metadata: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let doc = StoredDocument {
            id: doc_id.to_string(),
            text: text.to_string(),
            metadata: extract_meta(metadata)?,
        };
        let inner = self.inner.clone();
        run_async(async move { inner.set(doc).await }).map_err(to_py_err)
    }

    /// Store multiple documents at once.
    fn mset(&self, docs: Vec<PyRef<PyStoredDocument>>) -> PyResult<()> {
        let rust_docs: Vec<StoredDocument> = docs.iter().map(|d| d.inner.clone()).collect();
        let inner = self.inner.clone();
        run_async(async move { inner.mset(rust_docs).await }).map_err(to_py_err)
    }

    /// Retrieve a document by id. Returns ``None`` if not found.
    fn get(&self, py: Python<'_>, doc_id: &str) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let id = doc_id.to_string();
        let result = run_async(async move { inner.get(&id).await }).map_err(to_py_err)?;
        match result {
            None => Ok(py.None()),
            Some(doc) => stored_doc_to_py(py, &doc),
        }
    }

    /// Retrieve multiple documents by id. Missing ids return ``None`` entries.
    fn mget(&self, py: Python<'_>, ids: Vec<String>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let id_refs: Vec<String> = ids;
        let results = run_async(async move {
            let refs: Vec<&str> = id_refs.iter().map(|s| s.as_str()).collect();
            inner.mget(&refs).await
        })
        .map_err(to_py_err)?;
        let list = PyList::empty_bound(py);
        for opt_doc in &results {
            match opt_doc {
                None => list.append(py.None())?,
                Some(doc) => list.append(stored_doc_to_py(py, doc)?)?,
            }
        }
        Ok(list.into())
    }

    /// Delete a document by id.
    fn delete(&self, doc_id: &str) -> PyResult<()> {
        let inner = self.inner.clone();
        let id = doc_id.to_string();
        run_async(async move { inner.delete(&id).await }).map_err(to_py_err)
    }

    /// Delete multiple documents by id.
    fn mdelete(&self, ids: Vec<String>) -> PyResult<()> {
        let inner = self.inner.clone();
        run_async(async move {
            let refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
            inner.mdelete(&refs).await
        })
        .map_err(to_py_err)
    }

    /// List all stored document ids.
    fn keys(&self) -> PyResult<Vec<String>> {
        let inner = self.inner.clone();
        run_async(async move { inner.yield_keys().await }).map_err(to_py_err)
    }

    /// Number of stored documents.
    fn __len__(&self) -> usize { self.inner.len() }

    fn __repr__(&self) -> String {
        format!("InMemoryDocStore(n={})", self.inner.len())
    }
}

// ── PyLocalFileDocStore ───────────────────────────────────────────────────────

/// File-system document store — one JSON file per document.
///
/// Files are written to ``{dir}/{id}.json``. The directory is created if it
/// does not exist. Suitable for small-scale persistence without a database.
///
/// Example::
///
///     store = LocalFileDocStore("/tmp/my-docs")
///     store.set("d1", "Hello world", {"source": "test"})
///     doc = store.get("d1")
///     print(doc["text"])  # "Hello world"
#[pyclass(name = "LocalFileDocStore")]
pub struct PyLocalFileDocStore {
    inner: Arc<LocalFileDocStore>,
}

#[pymethods]
impl PyLocalFileDocStore {
    /// Create a file-backed document store.
    ///
    /// Args:
    ///     dir: Directory path. Created if it does not exist.
    #[new]
    fn new(dir: &str) -> Self {
        Self { inner: Arc::new(LocalFileDocStore::new(dir)) }
    }

    /// Store a single document.
    #[pyo3(signature = (doc_id, text, metadata=None))]
    fn set(&self, doc_id: &str, text: &str, metadata: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let doc = StoredDocument {
            id: doc_id.to_string(),
            text: text.to_string(),
            metadata: extract_meta(metadata)?,
        };
        let inner = self.inner.clone();
        run_async(async move { inner.set(doc).await }).map_err(to_py_err)
    }

    /// Store multiple documents at once.
    fn mset(&self, docs: Vec<PyRef<PyStoredDocument>>) -> PyResult<()> {
        let rust_docs: Vec<StoredDocument> = docs.iter().map(|d| d.inner.clone()).collect();
        let inner = self.inner.clone();
        run_async(async move { inner.mset(rust_docs).await }).map_err(to_py_err)
    }

    /// Retrieve a document by id. Returns ``None`` if not found.
    fn get(&self, py: Python<'_>, doc_id: &str) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let id = doc_id.to_string();
        let result = run_async(async move { inner.get(&id).await }).map_err(to_py_err)?;
        match result {
            None => Ok(py.None()),
            Some(doc) => stored_doc_to_py(py, &doc),
        }
    }

    /// Retrieve multiple documents by id.
    fn mget(&self, py: Python<'_>, ids: Vec<String>) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let results = run_async(async move {
            let refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
            inner.mget(&refs).await
        })
        .map_err(to_py_err)?;
        let list = PyList::empty_bound(py);
        for opt_doc in &results {
            match opt_doc {
                None => list.append(py.None())?,
                Some(doc) => list.append(stored_doc_to_py(py, doc)?)?,
            }
        }
        Ok(list.into())
    }

    /// Delete a document by id.
    fn delete(&self, doc_id: &str) -> PyResult<()> {
        let inner = self.inner.clone();
        let id = doc_id.to_string();
        run_async(async move { inner.delete(&id).await }).map_err(to_py_err)
    }

    /// List all stored document ids.
    fn keys(&self) -> PyResult<Vec<String>> {
        let inner = self.inner.clone();
        run_async(async move { inner.yield_keys().await }).map_err(to_py_err)
    }

    fn __repr__(&self) -> String { "LocalFileDocStore(...)".to_string() }
}
