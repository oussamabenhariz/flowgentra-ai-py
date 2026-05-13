//! Python bindings for all vector store backends.
//!
//! All backends expose the same Python API:
//!
//! ```python
//! store = PineconeStore("my-namespace", api_key="...", endpoint="https://...")
//! store.index(doc, embedding)
//! results = store.search(query_embedding, top_k=5, filter={"source": "pdf"})
//! results = store.search(query_embedding, top_k=5, filter={"age": {"$gt": 18}})
//! store.delete("doc-id")
//! store.update(doc, embedding)
//! doc = store.get("doc-id")
//! docs = store.list()
//! store.clear()
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use flowgentra_ai::core::rag::{
    AstraDbConfig, AstraDbStore, ElasticsearchConfig, ElasticsearchStore, FilterExpr, MilvusStore,
    OpenSearchConfig, OpenSearchStore, PineconeStore, QdrantStore, RAGConfig, UpstashVectorConfig,
    UpstashVectorStore, VectorStoreBackend, VectorStoreType, WeaviateStore,
};
use flowgentra_ai::core::rag::{MongoAtlasConfig, MongoAtlasVectorStore};
use flowgentra_ai::core::rag::{PgVectorConfig, PgVectorStore};
use flowgentra_ai::core::rag::{RedisVectorConfig, RedisVectorStore};

use crate::error::to_py_err_generic;
use crate::py_to_json;
use crate::rag::{PyDocument, PySearchResult};

// ─── Shared filter helper ─────────────────────────────────────────────────────

/// Convert an optional Python dict into a typed `FilterExpr`.
///
/// Supports two forms:
/// - Simple equality: `{"field": value}` → `FilterExpr::Eq`
/// - Operator form:  `{"field": {"$gt": 5, "$lt": 10}}` → `FilterExpr::And([Gt, Lt])`
///
/// Multiple top-level keys are combined with `And`.
fn extract_filter(filter: Option<&Bound<'_, PyDict>>) -> PyResult<Option<FilterExpr>> {
    let Some(dict) = filter else { return Ok(None) };
    if dict.is_empty() {
        return Ok(None);
    }

    let mut top_exprs: Vec<FilterExpr> = Vec::new();

    for (k, v) in dict.iter() {
        let key: String = k.extract()?;

        if let Ok(inner) = v.downcast::<PyDict>() {
            // Operator form: {"field": {"$gt": 5}}
            for (op_k, op_v) in inner.iter() {
                let op: String = op_k.extract()?;
                let val = py_to_json(&op_v)?;
                let expr = match op.as_str() {
                    "$eq" => FilterExpr::Eq(key.clone(), val),
                    "$ne" => FilterExpr::Ne(key.clone(), val),
                    "$gt" => FilterExpr::Gt(key.clone(), val),
                    "$gte" => FilterExpr::Gte(key.clone(), val),
                    "$lt" => FilterExpr::Lt(key.clone(), val),
                    "$lte" => FilterExpr::Lte(key.clone(), val),
                    "$in" => {
                        let arr = val.as_array().cloned().unwrap_or_default();
                        FilterExpr::In(key.clone(), arr)
                    }
                    _ => FilterExpr::Eq(key.clone(), val),
                };
                top_exprs.push(expr);
            }
        } else {
            // Simple equality: {"field": value}
            top_exprs.push(FilterExpr::Eq(key, py_to_json(&v)?));
        }
    }

    Ok(Some(if top_exprs.len() == 1 {
        top_exprs.remove(0)
    } else {
        FilterExpr::And(top_exprs)
    }))
}

// ─── PineconeStore ────────────────────────────────────────────────────────────

/// Pinecone vector store backend.
///
/// Connects to a Pinecone index via the Pinecone REST API.
///
/// Example::
///
///     store = PineconeStore(
///         namespace="my-namespace",
///         api_key="pc-...",
///         endpoint="https://my-index-abc123.svc.pinecone.io",
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "PineconeStore")]
pub struct PyPineconeStore {
    inner: Arc<PineconeStore>,
}

#[pymethods]
impl PyPineconeStore {
    /// Create a Pinecone vector store.
    ///
    /// Args:
    ///     namespace:     Pinecone namespace (logical grouping within an index).
    ///     api_key:       Pinecone API key.
    ///     endpoint:      Index host URL (from the Pinecone dashboard).
    ///     embedding_dim: Vector dimension (default 1536).
    #[new]
    #[pyo3(signature = (namespace, api_key, endpoint, embedding_dim = 1536))]
    fn new(namespace: &str, api_key: &str, endpoint: &str, embedding_dim: usize) -> PyResult<Self> {
        let config = RAGConfig {
            store_type: VectorStoreType::Pinecone,
            api_key: Some(api_key.to_string()),
            endpoint: Some(endpoint.to_string()),
            index_name: namespace.to_string(),
            embedding_dim,
        };
        let store = crate::run_async(PineconeStore::new(config)).map_err(to_py_err_generic)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    /// Index a document with its embedding vector.
    fn index(&self, doc: &PyDocument, embedding: Vec<f32>) -> PyResult<()> {
        let mut document = doc.inner.clone();
        document.embedding = Some(embedding);
        crate::run_async(self.inner.index(document)).map_err(to_py_err_generic)
    }

    /// Search for the most similar documents.
    #[pyo3(signature = (query_embedding, top_k = 5, filter = None))]
    fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<PySearchResult>> {
        let mf = extract_filter(filter)?;
        let results = crate::run_async(self.inner.search(query_embedding, top_k, mf))
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

    /// Update a document (pass `None` for embedding to leave it unchanged).
    #[pyo3(signature = (doc, embedding = None))]
    fn update(&self, doc: &PyDocument, embedding: Option<Vec<f32>>) -> PyResult<()> {
        let mut document = doc.inner.clone();
        document.embedding = embedding;
        crate::run_async(self.inner.update(document)).map_err(to_py_err_generic)
    }

    /// Retrieve a document by ID.
    fn get(&self, doc_id: &str) -> PyResult<PyDocument> {
        let doc = crate::run_async(self.inner.get(doc_id)).map_err(to_py_err_generic)?;
        Ok(PyDocument { inner: doc })
    }

    /// List all documents (not supported by Pinecone — raises RuntimeError).
    fn list(&self) -> PyResult<Vec<PyDocument>> {
        let docs = crate::run_async(self.inner.list()).map_err(to_py_err_generic)?;
        Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect())
    }

    /// Delete all documents in the namespace.
    fn clear(&self) -> PyResult<()> {
        crate::run_async(self.inner.clear()).map_err(to_py_err_generic)
    }

    /// Search with multiple query embeddings in a single call.
    #[pyo3(signature = (query_embeddings, top_k = 5, filter = None))]
    fn search_batch(
        &self,
        query_embeddings: Vec<Vec<f32>>,
        top_k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Vec<PySearchResult>>> {
        let mf = extract_filter(filter)?;
        let all_results = crate::run_async(self.inner.search_batch(query_embeddings, top_k, mf))
            .map_err(to_py_err_generic)?;
        Ok(all_results
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .map(|r| PySearchResult { inner: r })
                    .collect()
            })
            .collect())
    }

    fn __repr__(&self) -> String {
        "PineconeStore(...)".to_string()
    }
}

// ─── QdrantStore ──────────────────────────────────────────────────────────────

/// Qdrant vector store backend.
///
/// Connects to a Qdrant instance via its REST API.
///
/// Example::
///
///     store = QdrantStore(
///         collection="my_docs",
///         endpoint="http://localhost:6333",
///         embedding_dim=1536,
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "QdrantStore")]
pub struct PyQdrantStore {
    inner: Arc<QdrantStore>,
}

#[pymethods]
impl PyQdrantStore {
    /// Create a Qdrant vector store.
    ///
    /// Args:
    ///     collection:    Collection name.
    ///     endpoint:      Qdrant REST endpoint.
    ///     api_key:       Optional API key / bearer token.
    ///     embedding_dim: Vector dimension (default 1536).
    #[new]
    #[pyo3(signature = (collection, endpoint = "http://localhost:6333", api_key = None, embedding_dim = 1536))]
    fn new(
        collection: &str,
        endpoint: &str,
        api_key: Option<String>,
        embedding_dim: usize,
    ) -> PyResult<Self> {
        let config = RAGConfig {
            store_type: VectorStoreType::Qdrant,
            api_key,
            endpoint: Some(endpoint.to_string()),
            index_name: collection.to_string(),
            embedding_dim,
        };
        let store = crate::run_async(QdrantStore::new(config)).map_err(to_py_err_generic)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    fn index(&self, doc: &PyDocument, embedding: Vec<f32>) -> PyResult<()> {
        let mut document = doc.inner.clone();
        document.embedding = Some(embedding);
        crate::run_async(self.inner.index(document)).map_err(to_py_err_generic)
    }

    #[pyo3(signature = (query_embedding, top_k = 5, filter = None))]
    fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<PySearchResult>> {
        let mf = extract_filter(filter)?;
        let results = crate::run_async(self.inner.search(query_embedding, top_k, mf))
            .map_err(to_py_err_generic)?;
        Ok(results
            .into_iter()
            .map(|r| PySearchResult { inner: r })
            .collect())
    }

    fn delete(&self, doc_id: &str) -> PyResult<()> {
        crate::run_async(self.inner.delete(doc_id)).map_err(to_py_err_generic)
    }

    #[pyo3(signature = (doc, embedding = None))]
    fn update(&self, doc: &PyDocument, embedding: Option<Vec<f32>>) -> PyResult<()> {
        let mut document = doc.inner.clone();
        document.embedding = embedding;
        crate::run_async(self.inner.update(document)).map_err(to_py_err_generic)
    }

    fn get(&self, doc_id: &str) -> PyResult<PyDocument> {
        let doc = crate::run_async(self.inner.get(doc_id)).map_err(to_py_err_generic)?;
        Ok(PyDocument { inner: doc })
    }

    fn list(&self) -> PyResult<Vec<PyDocument>> {
        let docs = crate::run_async(self.inner.list()).map_err(to_py_err_generic)?;
        Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect())
    }

    fn clear(&self) -> PyResult<()> {
        crate::run_async(self.inner.clear()).map_err(to_py_err_generic)
    }

    #[pyo3(signature = (query_embeddings, top_k = 5, filter = None))]
    fn search_batch(
        &self,
        query_embeddings: Vec<Vec<f32>>,
        top_k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Vec<PySearchResult>>> {
        let mf = extract_filter(filter)?;
        let all_results = crate::run_async(self.inner.search_batch(query_embeddings, top_k, mf))
            .map_err(to_py_err_generic)?;
        Ok(all_results
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .map(|r| PySearchResult { inner: r })
                    .collect()
            })
            .collect())
    }

    fn __repr__(&self) -> String {
        "QdrantStore(...)".to_string()
    }
}

// ─── WeaviateStore ────────────────────────────────────────────────────────────

/// Weaviate vector store backend.
///
/// Connects to a Weaviate instance via its REST v1 API.
///
/// Example::
///
///     store = WeaviateStore(
///         class_name="Documents",
///         endpoint="http://localhost:8080",
///         embedding_dim=1536,
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "WeaviateStore")]
pub struct PyWeaviateStore {
    inner: Arc<WeaviateStore>,
}

#[pymethods]
impl PyWeaviateStore {
    /// Create a Weaviate vector store.
    ///
    /// Args:
    ///     class_name:    Weaviate class name (auto-capitalised if needed).
    ///     endpoint:      Weaviate REST endpoint.
    ///     api_key:       Optional API key / bearer token.
    ///     embedding_dim: Vector dimension (default 1536).
    #[new]
    #[pyo3(signature = (class_name, endpoint = "http://localhost:8080", api_key = None, embedding_dim = 1536))]
    fn new(
        class_name: &str,
        endpoint: &str,
        api_key: Option<String>,
        embedding_dim: usize,
    ) -> PyResult<Self> {
        let config = RAGConfig {
            store_type: VectorStoreType::Weaviate,
            api_key,
            endpoint: Some(endpoint.to_string()),
            index_name: class_name.to_string(),
            embedding_dim,
        };
        let store = crate::run_async(WeaviateStore::new(config)).map_err(to_py_err_generic)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    fn index(&self, doc: &PyDocument, embedding: Vec<f32>) -> PyResult<()> {
        let mut document = doc.inner.clone();
        document.embedding = Some(embedding);
        crate::run_async(self.inner.index(document)).map_err(to_py_err_generic)
    }

    #[pyo3(signature = (query_embedding, top_k = 5, filter = None))]
    fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<PySearchResult>> {
        let mf = extract_filter(filter)?;
        let results = crate::run_async(self.inner.search(query_embedding, top_k, mf))
            .map_err(to_py_err_generic)?;
        Ok(results
            .into_iter()
            .map(|r| PySearchResult { inner: r })
            .collect())
    }

    fn delete(&self, doc_id: &str) -> PyResult<()> {
        crate::run_async(self.inner.delete(doc_id)).map_err(to_py_err_generic)
    }

    #[pyo3(signature = (doc, embedding = None))]
    fn update(&self, doc: &PyDocument, embedding: Option<Vec<f32>>) -> PyResult<()> {
        let mut document = doc.inner.clone();
        document.embedding = embedding;
        crate::run_async(self.inner.update(document)).map_err(to_py_err_generic)
    }

    fn get(&self, doc_id: &str) -> PyResult<PyDocument> {
        let doc = crate::run_async(self.inner.get(doc_id)).map_err(to_py_err_generic)?;
        Ok(PyDocument { inner: doc })
    }

    fn list(&self) -> PyResult<Vec<PyDocument>> {
        let docs = crate::run_async(self.inner.list()).map_err(to_py_err_generic)?;
        Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect())
    }

    fn clear(&self) -> PyResult<()> {
        crate::run_async(self.inner.clear()).map_err(to_py_err_generic)
    }

    #[pyo3(signature = (query_embeddings, top_k = 5, filter = None))]
    fn search_batch(
        &self,
        query_embeddings: Vec<Vec<f32>>,
        top_k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Vec<PySearchResult>>> {
        let mf = extract_filter(filter)?;
        let all_results = crate::run_async(self.inner.search_batch(query_embeddings, top_k, mf))
            .map_err(to_py_err_generic)?;
        Ok(all_results
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .map(|r| PySearchResult { inner: r })
                    .collect()
            })
            .collect())
    }

    fn __repr__(&self) -> String {
        "WeaviateStore(...)".to_string()
    }
}

// ─── MilvusStore ─────────────────────────────────────────────────────────────

/// Milvus vector store backend.
///
/// Connects to a Milvus instance via its RESTful API v2.
///
/// Example::
///
///     store = MilvusStore(
///         collection="my_docs",
///         endpoint="http://localhost:19530",
///         embedding_dim=1536,
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "MilvusStore")]
pub struct PyMilvusStore {
    inner: Arc<MilvusStore>,
}

#[pymethods]
impl PyMilvusStore {
    /// Create a Milvus vector store.
    ///
    /// Args:
    ///     collection:    Collection name.
    ///     endpoint:      Milvus REST endpoint.
    ///     api_key:       Optional bearer token (for Zilliz Cloud).
    ///     embedding_dim: Vector dimension (default 1536).
    #[new]
    #[pyo3(signature = (collection, endpoint = "http://localhost:19530", api_key = None, embedding_dim = 1536))]
    fn new(
        collection: &str,
        endpoint: &str,
        api_key: Option<String>,
        embedding_dim: usize,
    ) -> PyResult<Self> {
        let config = RAGConfig {
            store_type: VectorStoreType::Milvus,
            api_key,
            endpoint: Some(endpoint.to_string()),
            index_name: collection.to_string(),
            embedding_dim,
        };
        let store = crate::run_async(MilvusStore::new(config)).map_err(to_py_err_generic)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    fn index(&self, doc: &PyDocument, embedding: Vec<f32>) -> PyResult<()> {
        let mut document = doc.inner.clone();
        document.embedding = Some(embedding);
        crate::run_async(self.inner.index(document)).map_err(to_py_err_generic)
    }

    #[pyo3(signature = (query_embedding, top_k = 5, filter = None))]
    fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<PySearchResult>> {
        let mf = extract_filter(filter)?;
        let results = crate::run_async(self.inner.search(query_embedding, top_k, mf))
            .map_err(to_py_err_generic)?;
        Ok(results
            .into_iter()
            .map(|r| PySearchResult { inner: r })
            .collect())
    }

    fn delete(&self, doc_id: &str) -> PyResult<()> {
        crate::run_async(self.inner.delete(doc_id)).map_err(to_py_err_generic)
    }

    #[pyo3(signature = (doc, embedding = None))]
    fn update(&self, doc: &PyDocument, embedding: Option<Vec<f32>>) -> PyResult<()> {
        let mut document = doc.inner.clone();
        document.embedding = embedding;
        crate::run_async(self.inner.update(document)).map_err(to_py_err_generic)
    }

    fn get(&self, doc_id: &str) -> PyResult<PyDocument> {
        let doc = crate::run_async(self.inner.get(doc_id)).map_err(to_py_err_generic)?;
        Ok(PyDocument { inner: doc })
    }

    fn list(&self) -> PyResult<Vec<PyDocument>> {
        let docs = crate::run_async(self.inner.list()).map_err(to_py_err_generic)?;
        Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect())
    }

    fn clear(&self) -> PyResult<()> {
        crate::run_async(self.inner.clear()).map_err(to_py_err_generic)
    }

    #[pyo3(signature = (query_embeddings, top_k = 5, filter = None))]
    fn search_batch(
        &self,
        query_embeddings: Vec<Vec<f32>>,
        top_k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Vec<PySearchResult>>> {
        let mf = extract_filter(filter)?;
        let all_results = crate::run_async(self.inner.search_batch(query_embeddings, top_k, mf))
            .map_err(to_py_err_generic)?;
        Ok(all_results
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .map(|r| PySearchResult { inner: r })
                    .collect()
            })
            .collect())
    }

    fn __repr__(&self) -> String {
        "MilvusStore(...)".to_string()
    }
}

// ─── Macro: common VectorStoreBackend methods ─────────────────────────────────

// In PyO3 0.22 macros cannot be used as items inside #[pymethods] blocks.
// Instead the macro generates a *separate* #[pymethods] impl block so that it
// is placed at the module (item) level.
macro_rules! impl_vector_store_pymethods {
    ($struct:ty) => {
        #[pymethods]
        impl $struct {
            fn index(&self, doc: &PyDocument, embedding: Vec<f32>) -> PyResult<()> {
                let mut document = doc.inner.clone();
                document.embedding = Some(embedding);
                crate::run_async(self.inner.index(document)).map_err(to_py_err_generic)
            }

            #[pyo3(signature = (query_embedding, top_k = 5, filter = None))]
            fn search(
                &self,
                query_embedding: Vec<f32>,
                top_k: usize,
                filter: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<Vec<PySearchResult>> {
                let mf = extract_filter(filter)?;
                let results = crate::run_async(self.inner.search(query_embedding, top_k, mf))
                    .map_err(to_py_err_generic)?;
                Ok(results
                    .into_iter()
                    .map(|r| PySearchResult { inner: r })
                    .collect())
            }

            fn delete(&self, doc_id: &str) -> PyResult<()> {
                crate::run_async(self.inner.delete(doc_id)).map_err(to_py_err_generic)
            }

            #[pyo3(signature = (doc, embedding = None))]
            fn update(&self, doc: &PyDocument, embedding: Option<Vec<f32>>) -> PyResult<()> {
                let mut document = doc.inner.clone();
                document.embedding = embedding;
                crate::run_async(self.inner.update(document)).map_err(to_py_err_generic)
            }

            fn get(&self, doc_id: &str) -> PyResult<PyDocument> {
                let doc = crate::run_async(self.inner.get(doc_id)).map_err(to_py_err_generic)?;
                Ok(PyDocument { inner: doc })
            }

            fn list(&self) -> PyResult<Vec<PyDocument>> {
                let docs = crate::run_async(self.inner.list()).map_err(to_py_err_generic)?;
                Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect())
            }

            fn clear(&self) -> PyResult<()> {
                crate::run_async(self.inner.clear()).map_err(to_py_err_generic)
            }

            /// Search with multiple query embeddings in a single call.
            ///
            /// Args:
            ///     query_embeddings: List of query vectors.
            ///     top_k:            Results per query (default 5).
            ///     filter:           Optional metadata filter dict applied to all queries.
            ///
            /// Returns:
            ///     List of result lists — one per input query embedding.
            #[pyo3(signature = (query_embeddings, top_k = 5, filter = None))]
            fn search_batch(
                &self,
                query_embeddings: Vec<Vec<f32>>,
                top_k: usize,
                filter: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<Vec<Vec<PySearchResult>>> {
                let mf = extract_filter(filter)?;
                let all_results =
                    crate::run_async(self.inner.search_batch(query_embeddings, top_k, mf))
                        .map_err(to_py_err_generic)?;
                Ok(all_results
                    .into_iter()
                    .map(|batch| {
                        batch
                            .into_iter()
                            .map(|r| PySearchResult { inner: r })
                            .collect()
                    })
                    .collect())
            }
        }
    };
}

// ─── PgVectorStore (requires pgvector-store feature) ─────────────────────────

/// PostgreSQL + pgvector vector store.
///
/// Requires the `pgvector` extension to be installed in your PostgreSQL
/// database. Works with Supabase out of the box.
///
/// Example::
///
///     store = PgVectorStore(
///         url="postgres://user:pass@localhost/mydb",
///         table="documents",
///         embedding_dim=1536,
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "PgVectorStore")]
pub struct PyPgVectorStore {
    inner: Arc<PgVectorStore>,
}

#[pymethods]
impl PyPgVectorStore {
    /// Connect to a PostgreSQL database with pgvector.
    ///
    /// Args:
    ///     url:           PostgreSQL connection URL.
    ///     table:         Table name (created automatically if absent).
    ///     embedding_dim: Vector dimension (default 1536).
    #[new]
    #[pyo3(signature = (url, table = "documents", embedding_dim = 1536))]
    fn new(url: &str, table: &str, embedding_dim: usize) -> PyResult<Self> {
        let store = crate::run_async(PgVectorStore::connect(PgVectorConfig {
            url: url.to_string(),
            table: table.to_string(),
            embedding_dim,
        }))
        .map_err(to_py_err_generic)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    fn __repr__(&self) -> String {
        "PgVectorStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PyPgVectorStore);

// ─── RedisVectorStore (requires redis-vector feature) ────────────────────────

/// Redis vector store using RediSearch (Redis Stack).
///
/// Example::
///
///     store = RedisVectorStore(
///         url="redis://localhost:6379",
///         index_name="docs",
///         embedding_dim=1536,
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "RedisVectorStore")]
pub struct PyRedisVectorStore {
    inner: Arc<RedisVectorStore>,
}

#[pymethods]
impl PyRedisVectorStore {
    /// Connect to a Redis instance with RediSearch.
    ///
    /// Args:
    ///     url:           Redis URL (e.g. ``redis://localhost:6379``).
    ///     index_name:    RediSearch index name.
    ///     key_prefix:    Key prefix for stored documents (default ``"doc"``).
    ///     embedding_dim: Vector dimension (default 1536).
    #[new]
    #[pyo3(signature = (url, index_name = "docs", key_prefix = "doc", embedding_dim = 1536))]
    fn new(url: &str, index_name: &str, key_prefix: &str, embedding_dim: usize) -> PyResult<Self> {
        let store = crate::run_async(RedisVectorStore::connect(RedisVectorConfig {
            url: url.to_string(),
            index_name: index_name.to_string(),
            key_prefix: key_prefix.to_string(),
            embedding_dim,
        }))
        .map_err(to_py_err_generic)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    fn __repr__(&self) -> String {
        "RedisVectorStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PyRedisVectorStore);

// ─── ElasticsearchVectorStore ─────────────────────────────────────────────────

/// Elasticsearch vector store using dense_vector kNN search.
///
/// Example::
///
///     store = ElasticsearchVectorStore(
///         endpoint="http://localhost:9200",
///         index="documents",
///         embedding_dim=1536,
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "ElasticsearchVectorStore")]
pub struct PyElasticsearchVectorStore {
    inner: Arc<ElasticsearchStore>,
}

#[pymethods]
impl PyElasticsearchVectorStore {
    /// Connect to Elasticsearch.
    ///
    /// Args:
    ///     endpoint:      Elasticsearch base URL.
    ///     index:         Index name.
    ///     embedding_dim: Vector dimension (default 1536).
    ///     api_key:       Optional API key for Elastic Cloud.
    #[new]
    #[pyo3(signature = (endpoint, index = "documents", embedding_dim = 1536, api_key = None))]
    fn new(
        endpoint: &str,
        index: &str,
        embedding_dim: usize,
        api_key: Option<String>,
    ) -> PyResult<Self> {
        let store = crate::run_async(ElasticsearchStore::new(ElasticsearchConfig {
            endpoint: endpoint.to_string(),
            index: index.to_string(),
            embedding_dim,
            api_key,
        }))
        .map_err(to_py_err_generic)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    fn __repr__(&self) -> String {
        "ElasticsearchVectorStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PyElasticsearchVectorStore);

// ─── OpenSearchVectorStore ────────────────────────────────────────────────────

/// OpenSearch vector store using k-NN search.
///
/// Example::
///
///     store = OpenSearchVectorStore(
///         endpoint="http://localhost:9200",
///         index="documents",
///         embedding_dim=1536,
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "OpenSearchVectorStore")]
pub struct PyOpenSearchVectorStore {
    inner: Arc<OpenSearchStore>,
}

#[pymethods]
impl PyOpenSearchVectorStore {
    /// Connect to OpenSearch.
    ///
    /// Args:
    ///     endpoint:      OpenSearch base URL.
    ///     index:         Index name.
    ///     embedding_dim: Vector dimension (default 1536).
    ///     username:      Basic auth username (default ``"admin"``).
    ///     password:      Basic auth password (default ``"admin"``).
    #[new]
    #[pyo3(signature = (endpoint, index = "documents", embedding_dim = 1536, username = "admin", password = "admin"))]
    fn new(
        endpoint: &str,
        index: &str,
        embedding_dim: usize,
        username: &str,
        password: &str,
    ) -> PyResult<Self> {
        let store = crate::run_async(OpenSearchStore::new(OpenSearchConfig {
            endpoint: endpoint.to_string(),
            index: index.to_string(),
            embedding_dim,
            username: Some(username.to_string()),
            password: Some(password.to_string()),
        }))
        .map_err(to_py_err_generic)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    fn __repr__(&self) -> String {
        "OpenSearchVectorStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PyOpenSearchVectorStore);

// ─── UpstashVectorStore ───────────────────────────────────────────────────────

/// Upstash serverless vector store.
///
/// Example::
///
///     store = UpstashVectorStore(
///         url="https://<id>.upstash.io",
///         token="your-token",
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "UpstashVectorStore")]
pub struct PyUpstashVectorStore {
    inner: Arc<UpstashVectorStore>,
}

#[pymethods]
impl PyUpstashVectorStore {
    /// Connect to Upstash Vector.
    ///
    /// Args:
    ///     url:   Upstash Vector REST URL (from your Upstash console).
    ///     token: Upstash REST token.
    #[new]
    fn new(url: &str, token: &str) -> PyResult<Self> {
        let store = UpstashVectorStore::new(UpstashVectorConfig {
            url: url.to_string(),
            token: token.to_string(),
        });
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    fn __repr__(&self) -> String {
        "UpstashVectorStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PyUpstashVectorStore);

// ─── AstraDbVectorStore ───────────────────────────────────────────────────────

/// DataStax Astra DB vector store.
///
/// Example::
///
///     store = AstraDbVectorStore(
///         endpoint="https://<id>-<region>.apps.astra.datastax.com",
///         token="AstraCS:...",
///         keyspace="default_keyspace",
///         collection="documents",
///         embedding_dim=1536,
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "AstraDbVectorStore")]
pub struct PyAstraDbVectorStore {
    inner: Arc<AstraDbStore>,
}

#[pymethods]
impl PyAstraDbVectorStore {
    /// Connect to Astra DB.
    ///
    /// Args:
    ///     endpoint:      Astra DB REST endpoint.
    ///     token:         Astra application token (``AstraCS:...``).
    ///     keyspace:      Keyspace name (default ``"default_keyspace"``).
    ///     collection:    Collection name.
    ///     embedding_dim: Vector dimension (default 1536).
    #[new]
    #[pyo3(signature = (endpoint, token, keyspace = "default_keyspace", collection = "documents", embedding_dim = 1536))]
    fn new(
        endpoint: &str,
        token: &str,
        keyspace: &str,
        collection: &str,
        embedding_dim: usize,
    ) -> PyResult<Self> {
        let store = crate::run_async(AstraDbStore::new(AstraDbConfig {
            endpoint: endpoint.to_string(),
            token: token.to_string(),
            keyspace: keyspace.to_string(),
            collection: collection.to_string(),
            embedding_dim,
        }))
        .map_err(to_py_err_generic)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    fn __repr__(&self) -> String {
        "AstraDbVectorStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PyAstraDbVectorStore);

// ─── MongoAtlasVectorStore (requires mongodb-atlas-store feature) ────────────

/// MongoDB Atlas Vector Search backend.
///
/// Requires a MongoDB Atlas M10+ cluster with a vector search index on the
/// ``embedding`` field (create once via the Atlas UI).
///
/// Example::
///
///     store = MongoAtlasVectorStore(
///         uri="mongodb+srv://user:pass@cluster.mongodb.net",
///         database="mydb",
///         collection="documents",
///         index_name="vector_index",
///         embedding_dim=1536,
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "MongoAtlasVectorStore")]
pub struct PyMongoAtlasVectorStore {
    inner: Arc<MongoAtlasVectorStore>,
}

#[pymethods]
impl PyMongoAtlasVectorStore {
    /// Connect to MongoDB Atlas.
    ///
    /// Args:
    ///     uri:           MongoDB connection string.
    ///     database:      Database name.
    ///     collection:    Collection name.
    ///     index_name:    Atlas vector search index name.
    ///     embedding_dim: Vector dimension (default 1536).
    #[new]
    #[pyo3(signature = (uri, database, collection, index_name = "vector_index", embedding_dim = 1536))]
    fn new(
        uri: &str,
        database: &str,
        collection: &str,
        index_name: &str,
        embedding_dim: usize,
    ) -> PyResult<Self> {
        let store = crate::run_async(MongoAtlasVectorStore::connect(MongoAtlasConfig {
            uri: uri.to_string(),
            database: database.to_string(),
            collection: collection.to_string(),
            index_name: index_name.to_string(),
            embedding_dim,
        }))
        .map_err(to_py_err_generic)?;
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    fn __repr__(&self) -> String {
        "MongoAtlasVectorStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PyMongoAtlasVectorStore);

// ─── HnswVectorStore ──────────────────────────────────────────────────────────

/// In-process HNSW (Hierarchical Navigable Small World) vector store.
///
/// Pure-Rust, zero-dependency approximate nearest-neighbour search. No server
/// required. Ideal for local development, testing, and small to medium datasets
/// (up to ~1M vectors). Does not support filters.
///
/// Example::
///
///     store = HnswVectorStore(embedding_dim=1536)
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
///     # Save/load to avoid rebuilding the index
///     store.save_to_file("my_index.bin")
///     store2 = HnswVectorStore.load_from_file("my_index.bin")
#[pyclass(name = "HnswVectorStore")]
pub struct PyHnswVectorStore {
    inner: Arc<flowgentra_ai::core::rag::HnswVectorStore>,
}

#[pymethods]
impl PyHnswVectorStore {
    /// Create a new in-memory HNSW index.
    ///
    /// Args:
    ///     embedding_dim: Dimensionality of the vectors to store.
    #[new]
    fn new(embedding_dim: usize) -> Self {
        Self {
            inner: Arc::new(flowgentra_ai::core::rag::HnswVectorStore::new(
                embedding_dim,
            )),
        }
    }

    fn __repr__(&self) -> String {
        "HnswVectorStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PyHnswVectorStore);

// ─── SingleStoreVectorStore ───────────────────────────────────────────────────

/// Vector store backed by SingleStore's native vector similarity search.
///
/// Requires a SingleStore instance with a table that has columns:
/// ``id TEXT``, ``text TEXT``, ``embedding VECTOR(dim)``, ``metadata JSON``.
///
/// Example::
///
///     store = SingleStoreVectorStore(
///         host="https://my-singlestore.example.com",
///         api_key="...",
///         database="mydb",
///         table="documents",
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "SingleStoreVectorStore")]
pub struct PySingleStoreVectorStore {
    inner: Arc<flowgentra_ai::core::rag::extra_vector_stores::SingleStoreVectorStore>,
}

#[pymethods]
impl PySingleStoreVectorStore {
    /// Connect to SingleStore.
    ///
    /// Args:
    ///     host:     HTTP endpoint of the SingleStore API.
    ///     api_key:  API key or authentication token.
    ///     database: Database name.
    ///     table:    Table name (must have the required schema).
    #[new]
    fn new(host: &str, api_key: &str, database: &str, table: &str) -> Self {
        Self {
            inner: Arc::new(
                flowgentra_ai::core::rag::extra_vector_stores::SingleStoreVectorStore::new(
                    host, api_key, database, table,
                ),
            ),
        }
    }

    fn __repr__(&self) -> String {
        "SingleStoreVectorStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PySingleStoreVectorStore);

// ─── AzureAISearchStore ───────────────────────────────────────────────────────

/// Vector store backed by Azure AI Search (formerly Azure Cognitive Search).
///
/// Requires an Azure AI Search resource with a vector-enabled index. Supports
/// the Azure AI Search 2024-07-01 REST API.
///
/// Example::
///
///     store = AzureAISearchStore(
///         endpoint="https://my-search.search.windows.net",
///         index_name="documents",
///         api_key="...",
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "AzureAISearchStore")]
pub struct PyAzureAISearchStore {
    inner: Arc<flowgentra_ai::core::rag::extra_vector_stores::AzureAISearchStore>,
}

#[pymethods]
impl PyAzureAISearchStore {
    /// Connect to Azure AI Search.
    ///
    /// Args:
    ///     endpoint:   Azure AI Search service endpoint URL.
    ///     index_name: Name of the search index.
    ///     api_key:    Azure AI Search admin API key.
    #[new]
    fn new(endpoint: &str, index_name: &str, api_key: &str) -> Self {
        Self {
            inner: Arc::new(
                flowgentra_ai::core::rag::extra_vector_stores::AzureAISearchStore::new(
                    endpoint, index_name, api_key,
                ),
            ),
        }
    }

    fn __repr__(&self) -> String {
        "AzureAISearchStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PyAzureAISearchStore);

// ─── VectaraStore ─────────────────────────────────────────────────────────────

/// Vector store backed by Vectara's semantic search platform.
///
/// Vectara manages embeddings internally — indexing takes plain text.
/// ``search()`` requires a text query; use ``search_text()`` for best results.
///
/// Example::
///
///     store = VectaraStore(
///         customer_id="12345678",
///         corpus_id=1,
///         api_key="...",
///     )
///     store.index(doc, embedding=[])    # embedding ignored by Vectara
///     results = store.search_text("my query", top_k=5)
#[pyclass(name = "VectaraStore")]
pub struct PyVectaraStore {
    inner: Arc<flowgentra_ai::core::rag::extra_vector_stores::VectaraStore>,
}

#[pymethods]
impl PyVectaraStore {
    /// Connect to Vectara.
    ///
    /// Args:
    ///     customer_id: Vectara customer ID (numeric string).
    ///     corpus_id:   ID of the corpus to query.
    ///     api_key:     Vectara API key.
    #[new]
    fn new(customer_id: &str, corpus_id: u64, api_key: &str) -> Self {
        Self {
            inner: Arc::new(
                flowgentra_ai::core::rag::extra_vector_stores::VectaraStore::new(
                    customer_id,
                    corpus_id,
                    api_key,
                ),
            ),
        }
    }

    /// Text-native search (recommended over ``search()`` for Vectara).
    ///
    /// Vectara encodes the query internally — no embedding step needed.
    ///
    /// Args:
    ///     query:  Plain-text search query.
    ///     top_k:  Maximum number of results to return.
    #[pyo3(signature = (query, top_k = 5))]
    fn search_text(&self, query: &str, top_k: usize) -> PyResult<Vec<PySearchResult>> {
        let results =
            crate::run_async(self.inner.search_text(query, top_k)).map_err(to_py_err_generic)?;
        Ok(results
            .into_iter()
            .map(|r| PySearchResult { inner: r })
            .collect())
    }

    fn __repr__(&self) -> String {
        "VectaraStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PyVectaraStore);

// ─── TurbopufferStore ─────────────────────────────────────────────────────────

/// Vector store backed by Turbopuffer's serverless vector database.
///
/// Turbopuffer is a cloud-native vector store with sub-millisecond query
/// latency. No infrastructure to manage.
///
/// Example::
///
///     store = TurbopufferStore(
///         api_key="tpuf_...",
///         namespace="my-docs",
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "TurbopufferStore")]
pub struct PyTurbopufferStore {
    inner: Arc<flowgentra_ai::core::rag::extra_vector_stores::TurbopufferStore>,
}

#[pymethods]
impl PyTurbopufferStore {
    /// Connect to Turbopuffer.
    ///
    /// Args:
    ///     api_key:   Turbopuffer API key (``tpuf_...``).
    ///     namespace: Logical namespace / index name.
    #[new]
    fn new(api_key: &str, namespace: &str) -> Self {
        Self {
            inner: Arc::new(
                flowgentra_ai::core::rag::extra_vector_stores::TurbopufferStore::new(
                    api_key, namespace,
                ),
            ),
        }
    }

    fn __repr__(&self) -> String {
        "TurbopufferStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PyTurbopufferStore);

// ─── Neo4jVectorStore ─────────────────────────────────────────────────────────

/// Vector store backed by Neo4j's native vector index (Neo4j 5.11+).
///
/// Uses the Neo4j HTTP API via Cypher queries. Supports both indexing and
/// vector-similarity search via ``db.index.vector.queryNodes``.
///
/// Example::
///
///     store = Neo4jVectorStore(
///         url="http://localhost:7474",
///         username="neo4j",
///         password="password",
///         index_name="vector_index",
///         node_label="Document",
///     )
///     store.index(doc, embedding)
///     results = store.search(query_embedding, top_k=5)
#[pyclass(name = "Neo4jVectorStore")]
pub struct PyNeo4jVectorStore {
    inner: Arc<flowgentra_ai::core::rag::extra_vector_stores::Neo4jVectorStore>,
}

#[pymethods]
impl PyNeo4jVectorStore {
    /// Connect to Neo4j.
    ///
    /// Args:
    ///     url:        Neo4j HTTP URL (e.g. ``"http://localhost:7474"``).
    ///     username:   Neo4j username.
    ///     password:   Neo4j password.
    ///     index_name: Name of the vector index (default ``"vector_index"``).
    ///     node_label: Node label for documents (default ``"Document"``).
    #[new]
    #[pyo3(signature = (url, username, password, index_name = "vector_index", node_label = "Document"))]
    fn new(url: &str, username: &str, password: &str, index_name: &str, node_label: &str) -> Self {
        let store = flowgentra_ai::core::rag::extra_vector_stores::Neo4jVectorStore::new(
            url, username, password,
        )
        .with_index(index_name)
        .with_node_label(node_label);
        Self {
            inner: Arc::new(store),
        }
    }

    fn __repr__(&self) -> String {
        "Neo4jVectorStore(...)".to_string()
    }
}

impl_vector_store_pymethods!(PyNeo4jVectorStore);
