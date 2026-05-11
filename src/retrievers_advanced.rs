//! Python bindings for advanced retrievers.
//!
//! Exposes:
//! - ``Bm25Retriever`` — pure keyword BM25 retrieval
//! - ``VectorRetriever`` — dense semantic retrieval
//! - ``EnsembleRetriever`` — RRF fusion of multiple retrievers (duck-typed)
//! - ``MultiQueryRetriever`` — LLM-generated query variants
//! - ``ScoreThresholdRetriever`` — filter by minimum similarity score
//! - ``EmbeddingsFilter`` — keep docs by cosine similarity to query
//! - ``ContextualCompressionRetriever`` — compress/filter after retrieval
//! - ``TimeWeightedRetriever`` — semantic + recency blend
//! - ``MultiVectorRetriever`` — multiple embeddings per parent document
//! - ``ParentDocumentRetriever`` — index child chunks, return full parents
//! - ``ReorderStrategy`` / ``reorder_for_long_context`` — lost-in-the-middle fix

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;

use flowgentra_ai::core::rag::{
    bm25_retriever::{Bm25Config, Bm25Document, Bm25Retriever},
    compression_retriever::{DocumentCompressor, EmbeddingsFilter},
    ensemble_retriever::{AsyncRetriever, VectorRetriever},
    multi_vector_retriever::{MultiVectorConfig, MultiVectorParent, MultiVectorRetriever, VectorView},
    parent_doc_retriever::{ParentDocConfig, ParentDocument, ParentDocumentRetriever},
    reorder::{reorder_for_long_context, ReorderStrategy},
    time_weighted_retriever::{TimeWeightedConfig, TimeWeightedRetriever},
    vector_db::{SearchResult, VectorStoreError},
};

use crate::run_async;
use crate::rag::PySearchResult;
use crate::vector_store::{PyEmbeddings, PyInMemoryVectorStore};

// ── helpers ───────────────────────────────────────────────────────────────────

fn to_py_results(results: Vec<SearchResult>) -> Vec<PySearchResult> {
    results.into_iter().map(|r| PySearchResult { inner: r }).collect()
}

fn to_py_err(e: VectorStoreError) -> PyErr {
    crate::error::InternalError::new_err(e.to_string())
}

// ── PyBm25Retriever ───────────────────────────────────────────────────────────

/// Pure keyword BM25 retriever — no embeddings needed.
///
/// Example::
///
///     r = Bm25Retriever(top_k=5)
///     r.add_texts([("doc1", "Rust ownership"), ("doc2", "Python pandas")])
///     results = r.retrieve("rust")
#[pyclass(name = "Bm25Retriever")]
pub struct PyBm25Retriever {
    inner: Bm25Retriever,
}

#[pymethods]
impl PyBm25Retriever {
    /// Create a BM25 retriever.
    ///
    /// Args:
    ///     top_k:           Maximum results (default 5).
    ///     score_threshold: Minimum normalised score 0–1 (default 0.0).
    ///     preprocess:      Lowercase + strip punctuation (default True).
    #[new]
    #[pyo3(signature = (top_k=5, score_threshold=0.0, preprocess=true))]
    fn new(top_k: usize, score_threshold: f32, preprocess: bool) -> Self {
        Self {
            inner: Bm25Retriever::new(Bm25Config { top_k, score_threshold, preprocess }),
        }
    }

    /// Create from a list of ``(id, text)`` tuples.
    #[staticmethod]
    #[pyo3(signature = (texts, top_k=5, score_threshold=0.0))]
    fn from_texts(texts: Vec<(String, String)>, top_k: usize, score_threshold: f32) -> Self {
        Self {
            inner: Bm25Retriever::from_texts(
                texts,
                Bm25Config { top_k, score_threshold, preprocess: true },
            ),
        }
    }

    /// Add ``(id, text)`` pairs to the BM25 corpus.
    fn add_texts(&mut self, texts: Vec<(String, String)>) {
        let docs = texts
            .into_iter()
            .map(|(id, text)| Bm25Document { id, text, metadata: HashMap::new() })
            .collect();
        self.inner.add_documents(docs);
    }

    /// Retrieve top-k documents by BM25 score.
    fn retrieve(&self, query: &str) -> Vec<PySearchResult> {
        to_py_results(self.inner.retrieve(query))
    }

    fn __repr__(&self) -> String { "Bm25Retriever(...)".to_string() }
}

// ── PyVectorRetriever ─────────────────────────────────────────────────────────

/// Dense semantic retriever backed by a vector store and embeddings.
///
/// Example::
///
///     store = InMemoryVectorStore()
///     emb   = Embeddings("mock", dimension=128)
///     r = VectorRetriever(store, emb, top_k=5)
///     results = r.retrieve("rust ownership")
#[pyclass(name = "VectorRetriever")]
pub struct PyVectorRetriever {
    inner: Arc<VectorRetriever>,
}

#[pymethods]
impl PyVectorRetriever {
    /// Args:
    ///     store:      ``InMemoryVectorStore`` instance.
    ///     embeddings: ``Embeddings`` instance.
    ///     top_k:      Results to return (default 5).
    #[new]
    #[pyo3(signature = (store, embeddings, top_k=5))]
    fn new(store: &PyInMemoryVectorStore, embeddings: &PyEmbeddings, top_k: usize) -> Self {
        Self {
            inner: Arc::new(VectorRetriever::new(
                store.inner.clone(),
                embeddings.inner.clone(),
                top_k,
            )),
        }
    }

    /// Retrieve semantically similar documents for ``query``.
    fn retrieve(&self, query: &str) -> PyResult<Vec<PySearchResult>> {
        let r = self.inner.clone();
        let q = query.to_string();
        let results = run_async(async move { r.as_ref().retrieve(&q).await }).map_err(to_py_err)?;
        Ok(to_py_results(results))
    }

    fn __repr__(&self) -> String { "VectorRetriever(...)".to_string() }
}

// ── PyEnsembleRetriever ───────────────────────────────────────────────────────

/// Fuses multiple retrievers using Reciprocal Rank Fusion (RRF).
///
/// Accepts any Python object with a ``retrieve(query) -> list[SearchResult]``
/// method — including ``Bm25Retriever``, ``VectorRetriever``, or custom classes.
///
/// Example::
///
///     bm25   = Bm25Retriever.from_texts([("d1", "Rust ownership")])
///     vector = VectorRetriever(store, emb, top_k=10)
///     ens = EnsembleRetriever(
///         retrievers=[bm25, vector],
///         weights=[0.3, 0.7],
///         top_k=5,
///     )
///     results = ens.retrieve("rust")
#[pyclass(name = "EnsembleRetriever")]
pub struct PyEnsembleRetriever {
    retrievers: Vec<(PyObject, f32)>,
    top_k: usize,
    rrf_k: usize,
}

#[pymethods]
impl PyEnsembleRetriever {
    /// Args:
    ///     retrievers: List of retriever objects (any with ``.retrieve()``).
    ///     weights:    Per-retriever weights (defaults to equal weights).
    ///     top_k:      Final number of results (default 5).
    ///     rrf_k:      RRF constant (default 60).
    #[new]
    #[pyo3(signature = (retrievers, weights=None, top_k=5, rrf_k=60))]
    fn new(
        retrievers: Vec<PyObject>,
        weights: Option<Vec<f32>>,
        top_k: usize,
        rrf_k: usize,
    ) -> Self {
        let n = retrievers.len();
        let w = weights.unwrap_or_else(|| vec![1.0 / n.max(1) as f32; n]);
        Self {
            retrievers: retrievers.into_iter().zip(w).collect(),
            top_k,
            rrf_k,
        }
    }

    /// Retrieve and fuse results via weighted RRF.
    fn retrieve(&self, py: Python<'_>, query: &str) -> PyResult<Vec<PySearchResult>> {
        let mut scores: HashMap<String, (f32, String, HashMap<String, serde_json::Value>)> =
            HashMap::new();

        for (retriever, weight) in &self.retrievers {
            let py_results = retriever.call_method1(py, "retrieve", (query,))?;
            let results: Vec<PyRef<PySearchResult>> = py_results.extract(py)?;
            let rrf_k = self.rrf_k as f32;
            for (rank, r) in results.iter().enumerate() {
                let rrf = weight / (rrf_k + rank as f32 + 1.0);
                scores
                    .entry(r.inner.id.clone())
                    .and_modify(|(s, _, _)| *s += rrf)
                    .or_insert_with(|| (rrf, r.inner.text.clone(), r.inner.metadata.clone()));
            }
        }

        let mut out: Vec<SearchResult> = scores
            .into_iter()
            .map(|(id, (score, text, metadata))| SearchResult { id, text, score, metadata })
            .collect();
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(self.top_k);
        Ok(to_py_results(out))
    }

    fn __repr__(&self) -> String {
        format!("EnsembleRetriever(n={}, top_k={})", self.retrievers.len(), self.top_k)
    }
}

// ── PyMultiQueryRetriever ─────────────────────────────────────────────────────

/// Generates LLM query variants, retrieves for each, then deduplicates.
///
/// The ``base_retriever`` can be any Python object with ``.retrieve(query)``.
///
/// Example::
///
///     r = MultiQueryRetriever(
///         base_retriever=vector_ret,
///         api_url="https://api.openai.com/v1/chat/completions",
///         api_key="sk-...",
///         model="gpt-4o-mini",
///         num_queries=3,
///         top_k=10,
///     )
///     results = r.retrieve("What is Rust ownership?")
#[pyclass(name = "MultiQueryRetriever")]
pub struct PyMultiQueryRetriever {
    base: PyObject,
    api_url: String,
    api_key: String,
    model: String,
    num_queries: usize,
    top_k: usize,
}

#[pymethods]
impl PyMultiQueryRetriever {
    /// Args:
    ///     base_retriever: Any retriever with ``.retrieve(query)``.
    ///     api_url:        OpenAI-compatible chat completions URL.
    ///     api_key:        API key.
    ///     model:          LLM model (default ``"gpt-4o-mini"``).
    ///     num_queries:    Variants to generate (default 3).
    ///     top_k:          Max results after dedup (default 10).
    #[new]
    #[pyo3(signature = (base_retriever, api_url, api_key, model="gpt-4o-mini", num_queries=3, top_k=10))]
    fn new(
        base_retriever: PyObject,
        api_url: &str,
        api_key: &str,
        model: &str,
        num_queries: usize,
        top_k: usize,
    ) -> Self {
        Self {
            base: base_retriever,
            api_url: api_url.to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            num_queries,
            top_k,
        }
    }

    /// Retrieve with LLM-expanded query variants.
    fn retrieve(&self, py: Python<'_>, query: &str) -> PyResult<Vec<PySearchResult>> {
        // Generate query variants via LLM
        let queries: Vec<String> = {
            let api_url = self.api_url.clone();
            let api_key = self.api_key.clone();
            let model = self.model.clone();
            let n = self.num_queries;
            let q = query.to_string();
            run_async(async move {
                let client = reqwest::Client::new();
                let prompt = format!(
                    "Generate {n} different search query variants for: \"{q}\"\n\
                     Return one query per line, no numbering or bullets."
                );
                let resp = client
                    .post(&api_url)
                    .bearer_auth(&api_key)
                    .json(&serde_json::json!({
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                    }))
                    .send()
                    .await;
                match resp {
                    Ok(r) => {
                        if let Ok(json) = r.json::<serde_json::Value>().await {
                            let text = json["choices"][0]["message"]["content"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();
                            let mut variants: Vec<String> = text
                                .lines()
                                .map(|l| l.trim().to_string())
                                .filter(|l| !l.is_empty())
                                .take(n)
                                .collect();
                            variants.insert(0, q);
                            Ok::<Vec<String>, ()>(variants)
                        } else {
                            Ok(vec![q])
                        }
                    }
                    Err(_) => Ok(vec![q]),
                }
            })
            .unwrap_or_else(|_| vec![query.to_string()])
        };

        // Retrieve with each variant and deduplicate
        let mut seen = std::collections::HashSet::new();
        let mut merged: Vec<PySearchResult> = Vec::new();
        for q in queries {
            let py_res = self.base.call_method1(py, "retrieve", (q.as_str(),))?;
            let results: Vec<PyRef<PySearchResult>> = py_res.extract(py)?;
            for r in results.iter() {
                if seen.insert(r.inner.id.clone()) {
                    merged.push(PySearchResult { inner: r.inner.clone() });
                }
            }
        }
        merged.truncate(self.top_k);
        Ok(merged)
    }

    fn __repr__(&self) -> String {
        format!("MultiQueryRetriever(model='{}', num_queries={})", self.model, self.num_queries)
    }
}

// ── PyScoreThresholdRetriever ─────────────────────────────────────────────────

/// Wraps any retriever and drops results below a minimum similarity score.
///
/// Example::
///
///     r = ScoreThresholdRetriever(base_retriever=vector_ret, min_score=0.75, top_k=5)
///     results = r.retrieve("rust ownership")
#[pyclass(name = "ScoreThresholdRetriever")]
pub struct PyScoreThresholdRetriever {
    base: PyObject,
    min_score: f32,
    top_k: usize,
}

#[pymethods]
impl PyScoreThresholdRetriever {
    /// Args:
    ///     base_retriever: Any retriever with ``.retrieve(query)``.
    ///     min_score:      Minimum score to keep (default 0.75).
    ///     top_k:          Hard cap on results (default 5).
    #[new]
    #[pyo3(signature = (base_retriever, min_score=0.75, top_k=5))]
    fn new(base_retriever: PyObject, min_score: f32, top_k: usize) -> Self {
        Self { base: base_retriever, min_score, top_k }
    }

    /// Retrieve and filter by minimum similarity score.
    fn retrieve(&self, py: Python<'_>, query: &str) -> PyResult<Vec<PySearchResult>> {
        let py_res = self.base.call_method1(py, "retrieve", (query,))?;
        let results: Vec<PyRef<PySearchResult>> = py_res.extract(py)?;
        let mut filtered: Vec<PySearchResult> = results
            .iter()
            .filter(|r| r.inner.score >= self.min_score)
            .map(|r| PySearchResult { inner: r.inner.clone() })
            .collect();
        filtered.truncate(self.top_k);
        Ok(filtered)
    }

    fn __repr__(&self) -> String {
        format!("ScoreThresholdRetriever(min_score={}, top_k={})", self.min_score, self.top_k)
    }
}

// ── PyEmbeddingsFilter ────────────────────────────────────────────────────────

/// Filters documents by cosine similarity to the query embedding.
///
/// Used as the compressor inside ``ContextualCompressionRetriever``.
///
/// Example::
///
///     f = EmbeddingsFilter(embeddings=emb, threshold=0.75)
#[pyclass(name = "EmbeddingsFilter")]
pub struct PyEmbeddingsFilter {
    pub(crate) inner: Arc<EmbeddingsFilter>,
}

#[pymethods]
impl PyEmbeddingsFilter {
    /// Args:
    ///     embeddings: ``Embeddings`` instance for re-scoring.
    ///     threshold:  Minimum cosine similarity 0–1 (default 0.75).
    #[new]
    #[pyo3(signature = (embeddings, threshold=0.75))]
    fn new(embeddings: &PyEmbeddings, threshold: f32) -> Self {
        Self {
            inner: Arc::new(EmbeddingsFilter::new(embeddings.inner.clone(), threshold)),
        }
    }

    fn __repr__(&self) -> String { "EmbeddingsFilter(...)".to_string() }
}

// ── PyContextualCompressionRetriever ─────────────────────────────────────────

/// Retrieves candidates from a base retriever, then filters with an ``EmbeddingsFilter``.
///
/// Example::
///
///     f = EmbeddingsFilter(emb, threshold=0.75)
///     r = ContextualCompressionRetriever(
///         base_retriever=vector_ret, compressor=f, top_k=5,
///     )
///     results = r.retrieve("rust ownership")
#[pyclass(name = "ContextualCompressionRetriever")]
pub struct PyContextualCompressionRetriever {
    base: PyObject,
    compressor: Arc<EmbeddingsFilter>,
    top_k: usize,
}

#[pymethods]
impl PyContextualCompressionRetriever {
    /// Args:
    ///     base_retriever: Any retriever with ``.retrieve(query)``.
    ///     compressor:     ``EmbeddingsFilter`` instance.
    ///     top_k:          Max results after compression (default 5).
    #[new]
    #[pyo3(signature = (base_retriever, compressor, top_k=5))]
    fn new(base_retriever: PyObject, compressor: &PyEmbeddingsFilter, top_k: usize) -> Self {
        Self { base: base_retriever, compressor: compressor.inner.clone(), top_k }
    }

    /// Retrieve then filter by embedding similarity.
    fn retrieve(&self, py: Python<'_>, query: &str) -> PyResult<Vec<PySearchResult>> {
        let py_res = self.base.call_method1(py, "retrieve", (query,))?;
        let raw: Vec<PyRef<PySearchResult>> = py_res.extract(py)?;
        let docs: Vec<SearchResult> = raw.iter().map(|r| r.inner.clone()).collect();
        let compressor = self.compressor.clone();
        let q = query.to_string();
        let top_k = self.top_k;
        let mut compressed =
            run_async(async move { compressor.compress(&q, docs).await }).map_err(to_py_err)?;
        compressed.truncate(top_k);
        Ok(to_py_results(compressed))
    }

    fn __repr__(&self) -> String { "ContextualCompressionRetriever(...)".to_string() }
}

// ── PyTimeWeightedRetriever ───────────────────────────────────────────────────

/// Blends semantic similarity with document recency (exponential decay).
///
/// Documents need a Unix timestamp under the ``last_accessed_at`` metadata key
/// (or the configured ``timestamp_key``).
///
/// Example::
///
///     r = TimeWeightedRetriever(store=store, embeddings=emb, decay_rate=0.01)
///     results = r.retrieve("rust ownership")
#[pyclass(name = "TimeWeightedRetriever")]
pub struct PyTimeWeightedRetriever {
    inner: Arc<TimeWeightedRetriever>,
}

#[pymethods]
impl PyTimeWeightedRetriever {
    /// Args:
    ///     store:             ``InMemoryVectorStore``.
    ///     embeddings:        ``Embeddings``.
    ///     decay_rate:        Hourly decay rate 0–1 (default 0.01).
    ///     fetch_k:           Candidates fetched pre-rerank (default 50).
    ///     top_k:             Final results (default 5).
    ///     score_threshold:   Minimum combined score (default 0.0).
    ///     timestamp_key:     Metadata key for Unix timestamp (default ``"last_accessed_at"``).
    ///     update_on_access:  Update timestamp on retrieval (default True).
    #[new]
    #[pyo3(signature = (store, embeddings, decay_rate=0.01, fetch_k=50, top_k=5, score_threshold=0.0, timestamp_key="last_accessed_at", update_on_access=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        store: &PyInMemoryVectorStore,
        embeddings: &PyEmbeddings,
        decay_rate: f32,
        fetch_k: usize,
        top_k: usize,
        score_threshold: f32,
        timestamp_key: &str,
        update_on_access: bool,
    ) -> Self {
        let config = TimeWeightedConfig {
            decay_rate,
            fetch_k,
            top_k,
            score_threshold,
            timestamp_key: timestamp_key.to_string(),
            update_on_access,
        };
        Self {
            inner: Arc::new(TimeWeightedRetriever::new(
                store.inner.clone(),
                embeddings.inner.clone(),
                config,
            )),
        }
    }

    /// Retrieve with recency-weighted scores.
    fn retrieve(&self, query: &str) -> PyResult<Vec<PySearchResult>> {
        let inner = self.inner.clone();
        let q = query.to_string();
        let results = run_async(async move { inner.retrieve(&q).await }).map_err(to_py_err)?;
        Ok(to_py_results(results))
    }

    fn __repr__(&self) -> String { "TimeWeightedRetriever(...)".to_string() }
}

// ── PyVectorView ──────────────────────────────────────────────────────────────

/// Type of vector representation stored per parent document.
///
/// Pass one or more to ``MultiVectorRetriever.add_with_views()``.
#[pyclass(name = "VectorView")]
#[derive(Clone)]
pub struct PyVectorView {
    pub(crate) inner: VectorView,
}

#[pymethods]
impl PyVectorView {
    /// Small chunks of the parent's raw text (auto-split).
    #[staticmethod]
    fn chunk() -> Self { Self { inner: VectorView::Chunk } }

    /// Hand-written or LLM-generated summary.
    #[staticmethod]
    fn summary(text: String) -> Self { Self { inner: VectorView::Summary(text) } }

    /// Hypothetical questions the document answers.
    #[staticmethod]
    fn hypothetical_questions(questions: Vec<String>) -> Self {
        Self { inner: VectorView::HypotheticalQuestions(questions) }
    }

    /// Custom tagged text representation.
    #[staticmethod]
    fn custom(tag: String, text: String) -> Self {
        Self { inner: VectorView::Custom { tag, text } }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            VectorView::Chunk => "VectorView.Chunk".to_string(),
            VectorView::Summary(_) => "VectorView.Summary(...)".to_string(),
            VectorView::HypotheticalQuestions(_) => "VectorView.HypotheticalQuestions(...)".to_string(),
            VectorView::Custom { tag, .. } => format!("VectorView.Custom(tag='{tag}')"),
        }
    }
}

// ── PyMultiVectorRetriever ────────────────────────────────────────────────────

/// Stores multiple vector representations per parent document.
///
/// Example::
///
///     r = MultiVectorRetriever(store=store, embeddings=emb, top_k=5)
///     r.add_with_views("d1", "Long document...", [VectorView.chunk()])
///     results = r.retrieve("memory safety")
#[pyclass(name = "MultiVectorRetriever")]
pub struct PyMultiVectorRetriever {
    inner: Arc<MultiVectorRetriever>,
}

#[pymethods]
impl PyMultiVectorRetriever {
    /// Args:
    ///     store:            ``InMemoryVectorStore``.
    ///     embeddings:       ``Embeddings``.
    ///     top_k:            Parent documents to return (default 5).
    ///     child_chunk_size: Chars per chunk for ``VectorView.chunk()`` (default 400).
    #[new]
    #[pyo3(signature = (store, embeddings, top_k=5, child_chunk_size=400))]
    fn new(
        store: &PyInMemoryVectorStore,
        embeddings: &PyEmbeddings,
        top_k: usize,
        child_chunk_size: usize,
    ) -> Self {
        let config = MultiVectorConfig {
            top_k,
            chunk_size: child_chunk_size,
            ..Default::default()
        };
        Self {
            inner: Arc::new(MultiVectorRetriever::new(
                store.inner.clone(),
                embeddings.inner.clone(),
                config,
            )),
        }
    }

    /// Index a document with one or more vector views.
    ///
    /// Args:
    ///     doc_id:   Unique document identifier.
    ///     text:     Full document text.
    ///     views:    List of ``VectorView`` instances.
    ///     metadata: Optional dict of metadata.
    #[pyo3(signature = (doc_id, text, views, metadata=None))]
    fn add_with_views(
        &self,
        doc_id: String,
        text: String,
        views: Vec<PyRef<PyVectorView>>,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<()> {
        let inner = self.inner.clone();
        let parent = MultiVectorParent {
            id: doc_id,
            text,
            metadata: metadata
                .unwrap_or_default()
                .into_iter()
                .map(|(k, v)| (k, serde_json::Value::String(v)))
                .collect(),
        };
        let rust_views: Vec<VectorView> = views.iter().map(|v| v.inner.clone()).collect();
        run_async(async move { inner.add_with_views(parent, rust_views).await })
            .map_err(to_py_err)
    }

    /// Retrieve parent documents matching ``query``.
    fn retrieve(&self, query: &str) -> PyResult<Vec<PySearchResult>> {
        let inner = self.inner.clone();
        let q = query.to_string();
        let results = run_async(async move { inner.retrieve_as_results(&q).await }).map_err(to_py_err)?;
        Ok(to_py_results(results))
    }

    fn __repr__(&self) -> String { "MultiVectorRetriever(...)".to_string() }
}

// ── PyParentDocumentRetriever ─────────────────────────────────────────────────

/// Indexes child chunks in a vector store but returns full parent documents.
///
/// Example::
///
///     r = ParentDocumentRetriever(store=store, embeddings=emb, max_parents=5)
///     r.add_documents([("d1", "Long document...", {})])
///     results = r.retrieve("What is Rust?")
#[pyclass(name = "ParentDocumentRetriever")]
pub struct PyParentDocumentRetriever {
    inner: Arc<ParentDocumentRetriever>,
}

#[pymethods]
impl PyParentDocumentRetriever {
    /// Args:
    ///     store:                ``InMemoryVectorStore``.
    ///     embeddings:           ``Embeddings``.
    ///     child_chunk_size:     Chars per child chunk (default 400).
    ///     child_chunk_overlap:  Overlap between chunks (default 50).
    ///     child_top_k:          Child matches before dedup (default 20).
    ///     similarity_threshold: Min child score (default 0.0).
    ///     max_parents:          Max distinct parents returned (default 5).
    #[new]
    #[pyo3(signature = (store, embeddings, child_chunk_size=400, child_chunk_overlap=50, child_top_k=20, similarity_threshold=0.0, max_parents=5))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        store: &PyInMemoryVectorStore,
        embeddings: &PyEmbeddings,
        child_chunk_size: usize,
        child_chunk_overlap: usize,
        child_top_k: usize,
        similarity_threshold: f32,
        max_parents: usize,
    ) -> Self {
        let config = ParentDocConfig {
            child_chunk_size,
            child_chunk_overlap,
            child_top_k,
            similarity_threshold,
            max_parents,
        };
        Self {
            inner: Arc::new(ParentDocumentRetriever::new(
                store.inner.clone(),
                embeddings.inner.clone(),
                config,
            )),
        }
    }

    /// Add parent documents (child chunks created internally).
    ///
    /// Args:
    ///     documents: List of ``(id, text, metadata_dict)`` tuples.
    fn add_documents(&self, documents: Vec<(String, String, HashMap<String, String>)>) -> PyResult<()> {
        let inner = self.inner.clone();
        let parent_docs: Vec<ParentDocument> = documents
            .into_iter()
            .map(|(id, text, meta)| ParentDocument {
                id,
                text,
                metadata: meta
                    .into_iter()
                    .map(|(k, v)| (k, serde_json::Value::String(v)))
                    .collect(),
            })
            .collect();
        run_async(async move { inner.add_documents(parent_docs).await }).map_err(to_py_err)
    }

    /// Retrieve full parent documents matching ``query``.
    fn retrieve(&self, query: &str) -> PyResult<Vec<PySearchResult>> {
        let inner = self.inner.clone();
        let q = query.to_string();
        let results = run_async(async move { inner.retrieve_as_results(&q).await }).map_err(to_py_err)?;
        Ok(to_py_results(results))
    }

    fn __repr__(&self) -> String { "ParentDocumentRetriever(...)".to_string() }
}

// ── PyReorderStrategy ─────────────────────────────────────────────────────────

/// Strategy for reordering results before passing to the LLM context.
///
/// Mitigates the "lost in the middle" problem (Liu et al., 2023).
#[pyclass(name = "ReorderStrategy")]
#[derive(Clone)]
pub struct PyReorderStrategy {
    pub(crate) inner: ReorderStrategy,
}

#[pymethods]
impl PyReorderStrategy {
    /// Best docs at start/end, worst in middle (recommended).
    #[staticmethod]
    fn lost_in_the_middle() -> Self { Self { inner: ReorderStrategy::LostInTheMiddle } }

    /// Leave in original score order (no-op).
    #[staticmethod]
    fn none() -> Self { Self { inner: ReorderStrategy::None } }

    /// Reverse order (worst first).
    #[staticmethod]
    fn reverse() -> Self { Self { inner: ReorderStrategy::Reverse } }

    fn __repr__(&self) -> String {
        match self.inner {
            ReorderStrategy::LostInTheMiddle => "ReorderStrategy.LostInTheMiddle",
            ReorderStrategy::None => "ReorderStrategy.None",
            ReorderStrategy::Reverse => "ReorderStrategy.Reverse",
        }
        .to_string()
    }
}

// ── Free function ─────────────────────────────────────────────────────────────

/// Reorder retrieved documents to mitigate "lost in the middle" attention bias.
///
/// Args:
///     results:  ``SearchResult`` list sorted by score descending.
///     strategy: ``ReorderStrategy`` (default: ``LostInTheMiddle``).
///
/// Returns:
///     Reordered list of ``SearchResult``.
///
/// Example::
///
///     reordered = reorder_for_long_context(results, ReorderStrategy.lost_in_the_middle())
#[pyfunction]
#[pyo3(signature = (results, strategy=None))]
pub fn py_reorder_for_long_context(
    results: Vec<PyRef<PySearchResult>>,
    strategy: Option<&PyReorderStrategy>,
) -> Vec<PySearchResult> {
    let strat = strategy.map(|s| s.inner).unwrap_or(ReorderStrategy::LostInTheMiddle);
    let docs: Vec<SearchResult> = results.iter().map(|r| r.inner.clone()).collect();
    to_py_results(reorder_for_long_context(docs, strat))
}
