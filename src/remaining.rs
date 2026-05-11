//! Python bindings for remaining prelude types:
//! ChunkMetadata, RetrieverStrategy, RerankStrategy, VectorStore, evaluate_output_score

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::Arc;

use flowgentra_ai::core::rag::{
    ChunkMetadata, InMemoryVectorStore, RerankStrategy, RetrieverStrategy,
    VectorStore, VectorStoreBackend,
};
use flowgentra_ai::core::node::evaluation_node::evaluate_output_score;

use crate::error::to_py_err_generic;
use crate::rag::{PyDocument, PySearchResult};
use crate::rag_config::PyRAGConfig;
use crate::py_to_json;

// ─── PyChunkMetadata ────────────────────────────────────────────────────────

/// Metadata for a text chunk produced by text splitters.
#[pyclass(name = "ChunkMetadata")]
#[derive(Clone)]
pub struct PyChunkMetadata {
    pub(crate) inner: ChunkMetadata,
}

#[pymethods]
impl PyChunkMetadata {
    #[new]
    #[pyo3(signature = (source=None, chunk_index=0, start_char=0, end_char=0))]
    fn new(source: Option<String>, chunk_index: usize, start_char: usize, end_char: usize) -> Self {
        PyChunkMetadata {
            inner: ChunkMetadata {
                source,
                chunk_index,
                start_char,
                end_char,
                extra: HashMap::new(),
            },
        }
    }

    #[getter]
    fn source(&self) -> Option<String> {
        self.inner.source.clone()
    }

    #[getter]
    fn chunk_index(&self) -> usize {
        self.inner.chunk_index
    }

    #[getter]
    fn start_char(&self) -> usize {
        self.inner.start_char
    }

    #[getter]
    fn end_char(&self) -> usize {
        self.inner.end_char
    }

    #[getter]
    fn extra(&self) -> HashMap<String, String> {
        self.inner.extra.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChunkMetadata(index={}, chars={}..{})",
            self.inner.chunk_index, self.inner.start_char, self.inner.end_char
        )
    }
}

// ─── PyRetrieverStrategy ────────────────────────────────────────────────────

/// Strategy for document retrieval.
///
/// Example:
///     strategy = RetrieverStrategy.semantic()
///     strategy = RetrieverStrategy.hybrid(keyword_weight=0.3)
///     strategy = RetrieverStrategy.multiquery(variants=3)
#[pyclass(name = "RetrieverStrategy")]
#[derive(Clone)]
pub struct PyRetrieverStrategy {
    pub(crate) inner: RetrieverStrategy,
}

#[pymethods]
impl PyRetrieverStrategy {
    /// Pure semantic similarity search.
    #[staticmethod]
    fn semantic() -> Self {
        PyRetrieverStrategy { inner: RetrieverStrategy::semantic() }
    }

    /// Hybrid: combine semantic + keyword search.
    #[staticmethod]
    #[pyo3(signature = (keyword_weight=0.3))]
    fn hybrid(keyword_weight: f32) -> Self {
        PyRetrieverStrategy { inner: RetrieverStrategy::hybrid(keyword_weight) }
    }

    /// Multi-query: expand query and retrieve for each variant.
    #[staticmethod]
    #[pyo3(signature = (variants=3))]
    fn multiquery(variants: usize) -> Self {
        PyRetrieverStrategy { inner: RetrieverStrategy::multiquery(variants) }
    }

    /// Decomposed: break complex queries into sub-queries.
    #[staticmethod]
    #[pyo3(signature = (depth=2))]
    fn decomposed(depth: usize) -> Self {
        PyRetrieverStrategy { inner: RetrieverStrategy::decomposed(depth) }
    }

    /// Custom chain strategy.
    #[staticmethod]
    fn custom(chain_name: &str) -> Self {
        PyRetrieverStrategy {
            inner: RetrieverStrategy::Custom {
                chain_name: chain_name.to_string(),
            },
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            RetrieverStrategy::SemanticSearch => "RetrieverStrategy.Semantic".to_string(),
            RetrieverStrategy::Hybrid { keyword_weight } => {
                format!("RetrieverStrategy.Hybrid(weight={})", keyword_weight)
            }
            RetrieverStrategy::MultiQuery { query_variants } => {
                format!("RetrieverStrategy.MultiQuery(variants={})", query_variants)
            }
            RetrieverStrategy::Decomposed { decomposition_depth } => {
                format!("RetrieverStrategy.Decomposed(depth={})", decomposition_depth)
            }
            RetrieverStrategy::Custom { chain_name } => {
                format!("RetrieverStrategy.Custom('{}')", chain_name)
            }
        }
    }
}

// ─── PyRerankStrategy ───────────────────────────────────────────────────────

/// Strategy for re-ranking search results.
///
/// Example:
///     strategy = RerankStrategy.none()
///     strategy = RerankStrategy.llm()
///     strategy = RerankStrategy.rrf(k=60)
#[pyclass(name = "RerankStrategy")]
#[derive(Clone)]
pub struct PyRerankStrategy {
    pub(crate) inner: RerankStrategy,
}

#[pymethods]
impl PyRerankStrategy {
    /// No re-ranking (pass-through).
    #[staticmethod]
    fn none() -> Self {
        PyRerankStrategy { inner: RerankStrategy::None }
    }

    /// Re-rank using an LLM to score relevance.
    #[staticmethod]
    fn llm() -> Self {
        PyRerankStrategy { inner: RerankStrategy::LLM }
    }

    /// Re-rank using a cross-encoder model.
    #[staticmethod]
    fn cross_encoder(model: &str, endpoint: &str) -> Self {
        PyRerankStrategy {
            inner: RerankStrategy::CrossEncoder {
                model: model.to_string(),
                endpoint: endpoint.to_string(),
            },
        }
    }

    /// Reciprocal Rank Fusion.
    #[staticmethod]
    #[pyo3(signature = (k=60))]
    fn rrf(k: usize) -> Self {
        PyRerankStrategy {
            inner: RerankStrategy::ReciprocalRankFusion { k },
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            RerankStrategy::None => "RerankStrategy.None".to_string(),
            RerankStrategy::LLM => "RerankStrategy.LLM".to_string(),
            RerankStrategy::CrossEncoder { model, .. } => {
                format!("RerankStrategy.CrossEncoder(model='{}')", model)
            }
            RerankStrategy::ReciprocalRankFusion { k } => {
                format!("RerankStrategy.RRF(k={})", k)
            }
        }
    }
}

// ─── PyVectorStore ──────────────────────────────────────────────────────────

/// Vector store wrapper for indexing and searching documents.
///
/// Example:
///     store = VectorStore.in_memory(RAGConfig.memory())
///     store.index_document("doc1", "Hello world", {})
///     results = store.search([0.1, 0.2, ...], top_k=5)
#[pyclass(name = "VectorStore")]
pub struct PyVectorStore {
    inner: VectorStore,
}

#[pymethods]
impl PyVectorStore {
    /// Create a vector store with in-memory backend.
    #[staticmethod]
    fn in_memory(config: &PyRAGConfig) -> Self {
        let backend = Arc::new(InMemoryVectorStore::new()) as Arc<dyn VectorStoreBackend>;
        PyVectorStore {
            inner: VectorStore::new(backend, config.inner.clone()),
        }
    }

    /// Index a document with text and metadata.
    fn index_document(
        &self,
        id: &str,
        text: &str,
        metadata: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let mut map = serde_json::Map::new();
        for (k, v) in metadata.iter() {
            let key: String = k.extract()?;
            let val = py_to_json(&v)?;
            map.insert(key, val);
        }
        crate::run_async(
            self.inner
                .index_document(id, text, serde_json::Value::Object(map)),
        )
        .map_err(to_py_err_generic)
    }

    /// Search by embedding vector.
    #[pyo3(signature = (query_embedding, top_k=5))]
    fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
    ) -> PyResult<Vec<PySearchResult>> {
        let results = 
            crate::run_async(self.inner.search(query_embedding, top_k, None))
            .map_err(to_py_err_generic)?;
        Ok(results
            .into_iter()
            .map(|r| PySearchResult { inner: r })
            .collect())
    }

    /// Delete a document by ID.
    fn delete(&self, doc_id: &str) -> PyResult<()> {
        crate::run_async(self.inner.delete(doc_id))
            .map_err(to_py_err_generic)
    }

    /// Update a document's text.
    fn update(&self, id: &str, text: &str) -> PyResult<()> {
        crate::run_async(self.inner.update(id, text))
            .map_err(to_py_err_generic)
    }

    /// Get a document by ID.
    fn get(&self, doc_id: &str) -> PyResult<PyDocument> {
        let doc = 
            crate::run_async(self.inner.get(doc_id))
            .map_err(to_py_err_generic)?;
        Ok(PyDocument { inner: doc })
    }

    /// List all documents.
    fn list(&self) -> PyResult<Vec<PyDocument>> {
        let docs = 
            crate::run_async(self.inner.list())
            .map_err(to_py_err_generic)?;
        Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect())
    }

    /// Clear all documents.
    fn clear(&self) -> PyResult<()> {
        crate::run_async(self.inner.clear()).map_err(to_py_err_generic)
    }

    fn __repr__(&self) -> String {
        format!("VectorStore(index='{}')", self.inner.config().index_name)
    }
}

// ─── evaluate_output_score ──────────────────────────────────────────────────

/// Score an output for quality evaluation.
///
/// Args:
///     output: The output value (string or dict) to score
///     attempt: The attempt number (1-based)
///
/// Returns:
///     Tuple of (score: float, feedback: str)
#[pyfunction]
pub fn py_evaluate_output_score(
    output: &Bound<'_, PyAny>,
    attempt: u32,
) -> PyResult<(f64, String)> {
    let val = py_to_json(output)?;
    let (score, feedback) = evaluate_output_score(&val, attempt);
    Ok((score, feedback))
}
