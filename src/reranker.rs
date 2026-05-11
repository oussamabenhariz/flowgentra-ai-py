//! Python bindings for rerankers

use pyo3::prelude::*;

use flowgentra_ai::core::rag::{
    NoopReranker, RRFReranker, CrossEncoderReranker,
    Reranker, SearchResult,
};

use crate::error::to_py_err_generic;
use crate::rag::PySearchResult;

// ─── PyNoopReranker ────────────────────────────────────────────────────────

/// A no-op reranker that passes results through unchanged.
#[pyclass(name = "NoopReranker")]
pub struct PyNoopReranker {
    inner: NoopReranker,
}

#[pymethods]
impl PyNoopReranker {
    #[new]
    fn new() -> Self {
        PyNoopReranker {
            inner: NoopReranker,
        }
    }

    /// Rerank results (no-op, returns as-is).
    fn rerank(&self, query: &str, results: Vec<PySearchResult>) -> PyResult<Vec<PySearchResult>> {
        let rs: Vec<SearchResult> = results.into_iter().map(|r| r.inner).collect();
        let out = 
            crate::run_async(self.inner.rerank(query, rs))
            .map_err(to_py_err_generic)?;
        Ok(out.into_iter().map(|r| PySearchResult { inner: r }).collect())
    }

    fn __repr__(&self) -> String {
        "NoopReranker()".to_string()
    }
}

// ─── PyRRFReranker ─────────────────────────────────────────────────────────

/// Reciprocal Rank Fusion reranker for merging multiple result lists.
///
/// Example:
///     rrf = RRFReranker(k=60)
///     merged = rrf.fuse([results1, results2])
#[pyclass(name = "RRFReranker")]
pub struct PyRRFReranker {
    inner: RRFReranker,
}

#[pymethods]
impl PyRRFReranker {
    #[new]
    #[pyo3(signature = (k=60))]
    fn new(k: usize) -> Self {
        PyRRFReranker {
            inner: RRFReranker::new(k),
        }
    }

    /// Fuse multiple result lists using Reciprocal Rank Fusion.
    fn fuse(&self, result_lists: Vec<Vec<PySearchResult>>) -> Vec<PySearchResult> {
        let lists: Vec<Vec<SearchResult>> = result_lists
            .into_iter()
            .map(|l| l.into_iter().map(|r| r.inner).collect())
            .collect();
        let out = self.inner.fuse(lists);
        out.into_iter().map(|r| PySearchResult { inner: r }).collect()
    }

    fn __repr__(&self) -> String {
        "RRFReranker(...)".to_string()
    }
}

// ─── PyCrossEncoderReranker ────────────────────────────────────────────────

/// Cross-encoder reranker using an external API endpoint.
///
/// Example:
///     reranker = CrossEncoderReranker("https://api.example.com/rerank", api_key="sk-...")
///     reranked = reranker.rerank("query", results)
#[pyclass(name = "CrossEncoderReranker")]
pub struct PyCrossEncoderReranker {
    inner: CrossEncoderReranker,
}

#[pymethods]
impl PyCrossEncoderReranker {
    #[new]
    #[pyo3(signature = (endpoint, api_key=None, top_k=None))]
    fn new(endpoint: &str, api_key: Option<String>, top_k: Option<usize>) -> Self {
        let mut reranker = CrossEncoderReranker::new(endpoint, api_key);
        if let Some(k) = top_k {
            reranker = reranker.with_top_k(k);
        }
        PyCrossEncoderReranker { inner: reranker }
    }

    /// Rerank search results using the cross-encoder model.
    fn rerank(&self, query: &str, results: Vec<PySearchResult>) -> PyResult<Vec<PySearchResult>> {
        let rs: Vec<SearchResult> = results.into_iter().map(|r| r.inner).collect();
        let out = 
            crate::run_async(self.inner.rerank(query, rs))
            .map_err(to_py_err_generic)?;
        Ok(out.into_iter().map(|r| PySearchResult { inner: r }).collect())
    }

    fn __repr__(&self) -> String {
        "CrossEncoderReranker(...)".to_string()
    }
}
