//! Python bindings for RAG types

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use std::collections::HashMap;

use flowgentra_ai::core::rag::text_splitter::TextChunk;
use flowgentra_ai::core::rag::{
    bm25_score, chunk_text, chunk_text_by_tokens, dedup_by_id, dedup_by_similarity,
    estimate_tokens, extract_and_chunk, extract_text, hybrid_merge, Document, PdfDocument,
    QueryExpander, SearchResult,
};

use crate::{json_to_py, py_to_json};

// ─── PyDocument ─────────────────────────────────────────────────────────────

/// A document for RAG (Retrieval-Augmented Generation)
///
/// Example:
///     doc = Document("doc-1", "Hello world", {"source": "example"})
#[pyclass(name = "Document")]
#[derive(Clone)]
pub struct PyDocument {
    pub(crate) inner: Document,
}

#[pymethods]
impl PyDocument {
    #[new]
    #[pyo3(signature = (id, text, metadata=None))]
    fn new(id: String, text: String, metadata: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let meta: HashMap<String, Value> = if let Some(dict) = metadata {
            let mut map = HashMap::new();
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                let val = py_to_json(&v)?;
                map.insert(key, val);
            }
            map
        } else {
            HashMap::new()
        };

        Ok(PyDocument {
            inner: Document {
                id,
                text,
                metadata: meta,
                embedding: None,
            },
        })
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    #[setter]
    fn set_text(&mut self, text: String) {
        self.inner.text = text;
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        let val = serde_json::to_value(&self.inner.metadata)
            .map_err(|e| crate::error::InternalError::new_err(format!("{}", e)))?;
        json_to_py(py, &val)
    }

    #[getter]
    fn embedding(&self) -> Option<Vec<f32>> {
        self.inner.embedding.clone()
    }

    fn __repr__(&self) -> String {
        let preview_len = self.inner.text.len().min(50);
        format!(
            "Document(id='{}', text='{}...')",
            self.inner.id,
            &self.inner.text[..preview_len]
        )
    }
}

// ─── PySearchResult ─────────────────────────────────────────────────────────

/// A search result from a vector store query
#[pyclass(name = "SearchResult")]
#[derive(Clone)]
pub struct PySearchResult {
    pub(crate) inner: SearchResult,
}

#[pymethods]
impl PySearchResult {
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        let val = serde_json::to_value(&self.inner.metadata)
            .map_err(|e| crate::error::InternalError::new_err(format!("{}", e)))?;
        json_to_py(py, &val)
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(id='{}', score={:.4})",
            self.inner.id, self.inner.score
        )
    }
}

// ─── PyTextChunk ────────────────────────────────────────────────────────────

/// A chunk of text from text splitting
#[pyclass(name = "TextChunk")]
#[derive(Clone)]
pub struct PyTextChunk {
    pub(crate) inner: TextChunk,
}

#[pymethods]
impl PyTextChunk {
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// Get the chunk metadata (source, index, char offsets).
    #[getter]
    fn metadata(&self) -> crate::remaining::PyChunkMetadata {
        crate::remaining::PyChunkMetadata {
            inner: self.inner.metadata.clone(),
        }
    }

    fn __repr__(&self) -> String {
        let preview_len = self.inner.text.len().min(40);
        format!("TextChunk(text='{}...')", &self.inner.text[..preview_len])
    }

    fn __len__(&self) -> usize {
        self.inner.text.len()
    }
}

// ─── Free functions ─────────────────────────────────────────────────────────

/// Split text into chunks of approximately `chunk_size` characters.
#[pyfunction]
#[pyo3(signature = (text, chunk_size, overlap=0))]
pub fn py_chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    chunk_text(text, chunk_size, overlap)
}

/// Extract text content from a PDF file (async, blocks until done).
#[pyfunction]
pub fn py_extract_text(path: &str) -> PyResult<String> {
    let result = crate::run_async(extract_text(path))
        .map_err(|e| crate::error::InternalError::new_err(format!("{}", e)))?;
    Ok(result.text)
}

/// Estimate the number of tokens in a string.
#[pyfunction]
pub fn py_estimate_tokens(text: &str) -> usize {
    estimate_tokens(text)
}

/// Split text into chunks by token count.
#[pyfunction]
#[pyo3(signature = (text, max_tokens, overlap_tokens=0))]
pub fn py_chunk_text_by_tokens(
    text: &str,
    max_tokens: usize,
    overlap_tokens: usize,
) -> Vec<String> {
    chunk_text_by_tokens(text, max_tokens, overlap_tokens)
}

/// Extract text from a PDF and split into (id, text) chunks.
#[pyfunction]
#[pyo3(signature = (path, chunk_size=1000, overlap=200))]
pub fn py_extract_and_chunk(
    path: &str,
    chunk_size: usize,
    overlap: usize,
) -> PyResult<Vec<(String, String)>> {
    crate::run_async(extract_and_chunk(path, chunk_size, overlap))
        .map_err(|e| crate::error::InternalError::new_err(format!("{}", e)))
}

// ─── PyPdfDocument ─────────────────────────────────────────────────────────

/// Represents an extracted PDF document.
#[pyclass(name = "PdfDocument")]
pub struct PyPdfDocument {
    inner: PdfDocument,
}

#[pymethods]
impl PyPdfDocument {
    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    #[getter]
    fn page_count(&self) -> usize {
        self.inner.page_count
    }

    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PdfDocument(source='{}', pages={}, len={})",
            self.inner.source,
            self.inner.page_count,
            self.inner.text.len()
        )
    }
}

/// Extract a PDF into a PdfDocument object.
#[pyfunction]
pub fn py_extract_pdf(path: &str) -> PyResult<PyPdfDocument> {
    let doc = crate::run_async(extract_text(path))
        .map_err(|e| crate::error::InternalError::new_err(format!("{}", e)))?;
    Ok(PyPdfDocument { inner: doc })
}

// ─── Hybrid search functions ───────────────────────────────────────────────

/// Compute BM25 scores for a query against a list of documents.
#[pyfunction]
pub fn py_bm25_score(query: &str, documents: Vec<String>) -> Vec<f32> {
    let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
    bm25_score(query, &refs)
}

/// Merge semantic results with keyword scores using hybrid weighting.
#[pyfunction]
#[pyo3(signature = (results, query, keyword_weight=0.3))]
pub fn py_hybrid_merge(
    results: Vec<PySearchResult>,
    query: &str,
    keyword_weight: f32,
) -> Vec<PySearchResult> {
    let rs: Vec<SearchResult> = results.into_iter().map(|r| r.inner).collect();
    let merged = hybrid_merge(rs, query, keyword_weight);
    merged
        .into_iter()
        .map(|r| PySearchResult { inner: r })
        .collect()
}

// ─── Dedup functions ───────────────────────────────────────────────────────

/// Deduplicate search results by document ID.
#[pyfunction]
pub fn py_dedup_by_id(results: Vec<PySearchResult>) -> Vec<PySearchResult> {
    let rs: Vec<SearchResult> = results.into_iter().map(|r| r.inner).collect();
    let deduped = dedup_by_id(rs);
    deduped
        .into_iter()
        .map(|r| PySearchResult { inner: r })
        .collect()
}

/// Deduplicate search results by text similarity threshold.
#[pyfunction]
#[pyo3(signature = (results, threshold=0.85))]
pub fn py_dedup_by_similarity(results: Vec<PySearchResult>, threshold: f32) -> Vec<PySearchResult> {
    let rs: Vec<SearchResult> = results.into_iter().map(|r| r.inner).collect();
    let deduped = dedup_by_similarity(rs, threshold);
    deduped
        .into_iter()
        .map(|r| PySearchResult { inner: r })
        .collect()
}

// ─── Query expansion ───────────────────────────────────────────────────────

/// Decompose a compound query into sub-queries.
///
/// Splits on "and", "or", ";" to generate query variants.
///
/// Example:
///     queries = decompose_query("Rust safety and performance", max_depth=2)
///     # ["Rust safety and performance", "Rust safety", "performance"]
#[pyfunction]
#[pyo3(signature = (query, max_depth=2))]
pub fn py_decompose_query(query: &str, max_depth: usize) -> Vec<String> {
    QueryExpander::decompose_query(query, max_depth)
}
