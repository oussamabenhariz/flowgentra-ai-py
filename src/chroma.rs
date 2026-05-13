//! Python bindings for ChromaDB vector store

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use flowgentra_ai::core::rag::{
    ChromaStore, FilterExpr, RAGConfig, VectorStoreBackend, VectorStoreType,
};

use crate::error::to_py_err_generic;
use crate::py_to_json;
use crate::rag::{PyDocument, PySearchResult};

// ─── PyChromaStore ──────────────────────────────────────────────────────────

/// ChromaDB vector store backend.
///
/// Connects to a ChromaDB instance via REST API.
/// API version (v1 / v2) is detected automatically — no configuration needed.
///
/// Example::
///
///     # v1 or v2 — auto-detected
///     store = ChromaStore("my_collection", endpoint="http://localhost:8000")
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
///
///     # v2 with custom tenant/database
///     store = ChromaStore("my_collection", tenant="my_org", database="prod")
#[pyclass(name = "ChromaStore")]
pub struct PyChromaStore {
    pub(crate) inner: Arc<ChromaStore>,
}

#[pymethods]
impl PyChromaStore {
    /// Create a new ChromaStore connected to a ChromaDB instance.
    ///
    /// API version is detected automatically by probing ``/api/v2/heartbeat``.
    ///
    /// Args:
    ///     collection_name: Name of the collection to use/create.
    ///     endpoint:        ChromaDB REST API endpoint (default: ``"http://localhost:8000"``).
    ///     embedding_dim:   Embedding dimension (default: 1536).
    ///     tenant:          ChromaDB tenant name — v2 only (default: ``"default_tenant"``).
    ///     database:        ChromaDB database name — v2 only (default: ``"default_database"``).
    #[new]
    #[pyo3(signature = (
        collection_name,
        endpoint="http://localhost:8000",
        embedding_dim=1536,
        tenant="default_tenant",
        database="default_database",
    ))]
    fn new(
        collection_name: &str,
        endpoint: &str,
        embedding_dim: usize,
        tenant: &str,
        database: &str,
    ) -> PyResult<Self> {
        let config = RAGConfig {
            store_type: VectorStoreType::Chroma,
            api_key: None,
            endpoint: Some(endpoint.to_string()),
            index_name: collection_name.to_string(),
            embedding_dim,
        };
        let store = crate::run_async(ChromaStore::new_with_tenant(&config, tenant, database))
            .map_err(to_py_err_generic)?;
        Ok(PyChromaStore {
            inner: Arc::new(store),
        })
    }

    /// Index a document with its embedding.
    fn index(&self, doc: &PyDocument, embedding: Vec<f32>) -> PyResult<()> {
        let mut document = doc.inner.clone();
        document.embedding = Some(embedding);
        crate::run_async(self.inner.index(document)).map_err(to_py_err_generic)
    }

    /// Search for similar documents by embedding vector.
    #[pyo3(signature = (query_embedding, top_k=5, filter=None))]
    fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<PySearchResult>> {
        let metadata_filter: Option<FilterExpr> = if let Some(dict) = filter {
            if dict.is_empty() {
                None
            } else {
                let mut exprs = Vec::new();
                for (k, v) in dict.iter() {
                    let key: String = k.extract()?;
                    let val = py_to_json(&v)?;
                    exprs.push(FilterExpr::Eq(key, val));
                }
                Some(if exprs.len() == 1 {
                    exprs.remove(0)
                } else {
                    FilterExpr::And(exprs)
                })
            }
        } else {
            None
        };

        let results = crate::run_async(self.inner.search(query_embedding, top_k, metadata_filter))
            .map_err(to_py_err_generic)?;

        Ok(results
            .into_iter()
            .map(|r| PySearchResult { inner: r })
            .collect())
    }

    /// Delete a document by ID.
    fn delete(&self, doc_id: &str) -> PyResult<()> {
        crate::run_async(self.inner.delete(doc_id)).map_err(to_py_err_generic)
    }

    /// Update a document.
    #[pyo3(signature = (doc, embedding=None))]
    fn update(&self, doc: &PyDocument, embedding: Option<Vec<f32>>) -> PyResult<()> {
        let mut document = doc.inner.clone();
        document.embedding = embedding;
        crate::run_async(self.inner.update(document)).map_err(to_py_err_generic)
    }

    /// Get a document by ID.
    fn get(&self, doc_id: &str) -> PyResult<PyDocument> {
        let doc = crate::run_async(self.inner.get(doc_id)).map_err(to_py_err_generic)?;
        Ok(PyDocument { inner: doc })
    }

    /// List all documents.
    fn list(&self) -> PyResult<Vec<PyDocument>> {
        let docs = crate::run_async(self.inner.list()).map_err(to_py_err_generic)?;
        Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect())
    }

    /// Clear all documents.
    fn clear(&self) -> PyResult<()> {
        crate::run_async(self.inner.clear()).map_err(to_py_err_generic)
    }

    fn __repr__(&self) -> String {
        format!(
            "ChromaStore(collection='{}', ...)",
            self.inner.collection_name()
        )
    }
}
