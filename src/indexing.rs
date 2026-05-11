//! Python bindings for the indexing pipeline (LangChain-style RecordManager).
//!
//! Deduplication-aware indexing: documents are hashed and re-indexing is skipped
//! if the same document was already indexed under the same namespace.
//!
//! ```python
//! from flowgentra_ai.rag import (
//!     InMemoryRecordManager, CleanupMode, IndexStats, index_documents,
//! )
//! from flowgentra_ai.rag import InMemoryVectorStore, Embeddings, Document
//!
//! store = InMemoryVectorStore()
//! rm    = InMemoryRecordManager("my_namespace")
//!
//! docs  = [Document("d1", "Rust ownership", {"source": "blog"})]
//! emb   = Embeddings("mock", dimension=128)
//! stats = index_documents(docs, rm, store, emb, cleanup=CleanupMode.incremental())
//! print(stats.added)    # 1
//!
//! # Re-index same docs → skipped
//! stats2 = index_documents(docs, rm, store, emb, cleanup=CleanupMode.incremental())
//! print(stats2.skipped) # 1
//! ```

use std::sync::Arc;

use pyo3::prelude::*;

use flowgentra_ai::core::rag::{
    indexing::{CleanupMode, InMemoryRecordManager, IndexStats, RecordEntry, RecordManager, index},
    vector_db::VectorStoreError,
};

use crate::run_async;
use crate::rag::PyDocument;
use crate::vector_store::{PyEmbeddings, PyInMemoryVectorStore};

fn to_py_err(e: VectorStoreError) -> PyErr {
    crate::error::InternalError::new_err(e.to_string())
}

// ── PyCleanupMode ─────────────────────────────────────────────────────────────

/// Cleanup strategy used by ``index_documents()``.
///
/// - ``none()`` — add new, skip duplicates; never delete.
/// - ``incremental()`` — delete source documents no longer in the input batch.
/// - ``full()`` — wipe the entire namespace, then re-index from scratch.
#[pyclass(name = "CleanupMode")]
#[derive(Clone)]
pub struct PyCleanupMode {
    inner: CleanupMode,
}

#[pymethods]
impl PyCleanupMode {
    /// Never delete; only skip already-indexed duplicates.
    #[staticmethod]
    fn none() -> Self { Self { inner: CleanupMode::None } }

    /// Delete source docs no longer present in the input batch.
    #[staticmethod]
    fn incremental() -> Self { Self { inner: CleanupMode::Incremental } }

    /// Wipe the entire namespace then re-index from scratch.
    #[staticmethod]
    fn full() -> Self { Self { inner: CleanupMode::Full } }

    fn __repr__(&self) -> String {
        match self.inner {
            CleanupMode::None => "CleanupMode.None",
            CleanupMode::Incremental => "CleanupMode.Incremental",
            CleanupMode::Full => "CleanupMode.Full",
        }
        .to_string()
    }
}

// ── PyIndexStats ──────────────────────────────────────────────────────────────

/// Statistics returned by ``index_documents()``.
///
/// Attributes:
///     added (int):   Documents newly indexed.
///     updated (int): Documents whose content changed (re-indexed).
///     skipped (int): Documents already indexed with identical content.
///     deleted (int): Documents removed by cleanup.
#[pyclass(name = "IndexStats")]
#[derive(Clone)]
pub struct PyIndexStats {
    pub(crate) inner: IndexStats,
}

#[pymethods]
impl PyIndexStats {
    #[getter]
    fn added(&self) -> usize { self.inner.added }

    #[getter]
    fn updated(&self) -> usize { self.inner.updated }

    #[getter]
    fn skipped(&self) -> usize { self.inner.skipped }

    #[getter]
    fn deleted(&self) -> usize { self.inner.deleted }

    fn __repr__(&self) -> String {
        format!(
            "IndexStats(added={}, updated={}, skipped={}, deleted={})",
            self.inner.added, self.inner.updated, self.inner.skipped, self.inner.deleted,
        )
    }
}

// ── PyRecordEntry ─────────────────────────────────────────────────────────────

/// A record in the ``InMemoryRecordManager``.
///
/// Attributes:
///     doc_id (str):     Document id in the vector store.
///     hash (str):       blake3 hash of (text + metadata + source).
///     indexed_at (int): Unix timestamp of indexing.
///     source (str):     Source identifier (file path, URL, etc.).
#[pyclass(name = "RecordEntry")]
#[derive(Clone)]
pub struct PyRecordEntry {
    inner: RecordEntry,
}

#[pymethods]
impl PyRecordEntry {
    #[getter]
    fn doc_id(&self) -> &str { &self.inner.doc_id }

    #[getter]
    fn hash(&self) -> &str { &self.inner.hash }

    #[getter]
    fn indexed_at(&self) -> i64 { self.inner.indexed_at }

    #[getter]
    fn source(&self) -> &str { &self.inner.source }

    fn __repr__(&self) -> String {
        format!("RecordEntry(doc_id='{}', source='{}')", self.inner.doc_id, self.inner.source)
    }
}

// ── PyInMemoryRecordManager ───────────────────────────────────────────────────

/// In-memory record manager for deduplication-aware indexing.
///
/// Tracks which documents have been indexed via their content hash. Fast,
/// zero-persistence — state is lost when the process exits. Use ``SqliteRecordManager``
/// (from ``flowgentra_ai.rag``) for durable persistence.
///
/// Example::
///
///     rm = InMemoryRecordManager("my_namespace")
///     stats = index_documents(docs, rm, store, embeddings)
///     records = rm.list_records()
///     rm.clear()
#[pyclass(name = "InMemoryRecordManager")]
pub struct PyInMemoryRecordManager {
    pub(crate) inner: Arc<InMemoryRecordManager>,
}

#[pymethods]
impl PyInMemoryRecordManager {
    /// Create a new in-memory record manager.
    ///
    /// Args:
    ///     namespace: Logical namespace for this manager (e.g. index name).
    #[new]
    fn new(namespace: &str) -> Self {
        Self {
            inner: Arc::new(InMemoryRecordManager::new(namespace)),
        }
    }

    /// Return the namespace this manager operates under.
    #[getter]
    fn namespace(&self) -> &str { self.inner.namespace() }

    /// Check if a hash already exists in the namespace.
    fn exists(&self, hash: &str) -> PyResult<bool> {
        let inner = self.inner.clone();
        let h = hash.to_string();
        run_async(async move { inner.exists(&h).await }).map_err(to_py_err)
    }

    /// List all records in this namespace.
    fn list_records(&self) -> PyResult<Vec<PyRecordEntry>> {
        let inner = self.inner.clone();
        let records = run_async(async move { inner.list_records().await }).map_err(to_py_err)?;
        Ok(records.into_iter().map(|r| PyRecordEntry { inner: r }).collect())
    }

    /// Delete all records in this namespace.
    fn clear(&self) -> PyResult<()> {
        let inner = self.inner.clone();
        run_async(async move { inner.clear().await }).map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!("InMemoryRecordManager(namespace='{}')", self.inner.namespace())
    }
}

// ── index_documents() ─────────────────────────────────────────────────────────

/// Index documents with deduplication and optional cleanup.
///
/// Documents are hashed (text + metadata + source). Documents whose hash is
/// already present in the ``record_manager`` are skipped. Embeddings are
/// computed for new documents only.
///
/// Args:
///     documents:      List of ``Document`` objects. Add ``{"source": "..."}``
///                     metadata for proper incremental cleanup.
///     record_manager: ``InMemoryRecordManager`` instance.
///     store:          ``InMemoryVectorStore`` instance.
///     embeddings:     ``Embeddings`` instance used to embed new documents.
///     cleanup:        ``CleanupMode`` — None / Incremental / Full (default None).
///
/// Returns:
///     ``IndexStats`` with added / skipped / deleted counts.
///
/// Example::
///
///     stats = index_documents(docs, rm, store, emb, cleanup=CleanupMode.incremental())
///     print(f"Added: {stats.added}, Skipped: {stats.skipped}")
#[pyfunction]
#[pyo3(signature = (documents, record_manager, store, embeddings, cleanup=None))]
pub fn py_index_documents(
    documents: Vec<PyRef<PyDocument>>,
    record_manager: &PyInMemoryRecordManager,
    store: &PyInMemoryVectorStore,
    embeddings: &PyEmbeddings,
    cleanup: Option<&PyCleanupMode>,
) -> PyResult<PyIndexStats> {
    let cleanup_mode = cleanup.map(|c| c.inner).unwrap_or(CleanupMode::None);

    // Build documents with embeddings for those that lack them
    let emb_arc = embeddings.inner.clone();
    let mut rust_docs: Vec<flowgentra_ai::core::rag::vector_db::Document> = Vec::new();

    for doc_ref in &documents {
        let mut doc = doc_ref.inner.clone();
        if doc.embedding.is_none() {
            let emb = emb_arc.clone();
            let text = doc.text.clone();
            let vec = run_async(async move { emb.embed(&text).await })
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            doc.embedding = Some(vec);
        }
        rust_docs.push(doc);
    }

    let rm = record_manager.inner.clone();
    let store_arc = store.inner.clone();
    let stats = run_async(async move {
        index(rust_docs, rm.as_ref(), store_arc.as_ref(), cleanup_mode).await
    })
    .map_err(to_py_err)?;

    Ok(PyIndexStats { inner: stats })
}
