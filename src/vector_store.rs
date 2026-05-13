//! Python bindings for VectorStore, InMemoryVectorStore, Embeddings, Retriever

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use flowgentra_ai::core::rag::{
    CachedEmbeddings, Embeddings, FilterExpr, HuggingFaceEmbeddings, InMemoryVectorStore,
    MistralEmbeddings, OllamaEmbeddings, OpenAIEmbeddings, RetrievalConfig, Retriever,
    RetrieverStrategy, VectorStoreBackend,
};

use crate::error::to_py_err_generic;
use crate::py_to_json;
use crate::rag::{PyDocument, PySearchResult};

// ─── PyEmbeddings ───────────────────────────────────────────────────────────

/// Embeddings manager for generating vector embeddings from text.
///
/// Example:
///     emb = Embeddings.mock(128)
///     vector = emb.embed("Hello world")
///     print(len(vector))  # 128
#[pyclass(name = "Embeddings")]
pub struct PyEmbeddings {
    pub(crate) inner: Arc<Embeddings>,
}

#[pymethods]
impl PyEmbeddings {
    /// Create embeddings with flexible provider configuration (Option A).
    ///
    /// Args:
    ///     provider: "openai", "mistral", "huggingface", "ollama", or "mock"
    ///     model: Model name (e.g., "text-embedding-3-small", "nomic-embed-text")
    ///     api_key: Optional API key. If None, reads from PROVIDER_API_KEY env var
    ///     dimension: Optional custom dimension
    ///     batch_size: Batch size for processing (default: 100)
    ///     cache: Enable caching for this provider (default: False)
    ///
    /// Example:
    ///     # OpenAI with env var fallback
    ///     emb = Embeddings(
    ///         provider="openai",
    ///         model="text-embedding-3-small",
    ///         dimension=1536,
    ///         batch_size=100,
    ///         cache=True
    ///     )
    ///     # Reads OPENAI_API_KEY from environment
    ///
    ///     # Ollama (local, no API key needed)
    ///     emb = Embeddings(
    ///         provider="ollama",
    ///         model="nomic-embed-text"
    ///     )
    ///
    ///     # Mock for testing
    ///     emb = Embeddings(provider="mock", dimension=128)
    #[new]
    #[allow(unused_variables)]
    #[pyo3(signature = (provider, model=None, api_key=None, dimension=None, batch_size=100, cache=false))]
    fn new(
        provider: &str,
        model: Option<String>,
        api_key: Option<String>,
        dimension: Option<usize>,
        batch_size: usize,
        cache: bool,
    ) -> PyResult<Self> {
        let model = model.unwrap_or_else(|| match provider {
            "openai" => "text-embedding-3-small".to_string(),
            "mistral" => "mistral-embed".to_string(),
            "ollama" => "nomic-embed-text".to_string(),
            "huggingface" => "BAAI/bge-small-en-v1.5".to_string(),
            _ => "mock".to_string(),
        });

        match provider.to_lowercase().as_str() {
            "mock" => {
                let dim = dimension.unwrap_or(128);
                Ok(PyEmbeddings {
                    inner: Arc::new(Embeddings::mock(dim)),
                })
            }

            "openai" => {
                let key = api_key.ok_or_else(|| {
                    crate::error::ConfigurationError::new_err(
                        "api_key is required for OpenAI embeddings",
                    )
                })?;

                let mut openai_provider = OpenAIEmbeddings::new(&key, &model);
                if let Some(dim) = dimension {
                    openai_provider = openai_provider.with_dimension(dim);
                }

                let embeddings = if cache {
                    let cached = CachedEmbeddings::new(Arc::new(openai_provider));
                    Arc::new(Embeddings::new(Arc::new(cached)))
                } else {
                    Arc::new(Embeddings::new(Arc::new(openai_provider)))
                };

                Ok(PyEmbeddings { inner: embeddings })
            }

            "mistral" => {
                let key = api_key.ok_or_else(|| {
                    crate::error::ConfigurationError::new_err(
                        "api_key is required for Mistral embeddings",
                    )
                })?;

                let mistral_provider = MistralEmbeddings::new(&key, Some(model));
                Ok(PyEmbeddings {
                    inner: Arc::new(Embeddings::new(Arc::new(mistral_provider))),
                })
            }

            "ollama" => {
                let base_url = api_key.unwrap_or_else(|| "http://localhost:11434".to_string());

                let mut ollama_provider = OllamaEmbeddings::new(&model, Some(base_url));
                if let Some(dim) = dimension {
                    ollama_provider = ollama_provider.with_dimension(dim);
                }

                Ok(PyEmbeddings {
                    inner: Arc::new(Embeddings::new(Arc::new(ollama_provider))),
                })
            }

            "huggingface" => {
                // HuggingFace local mode works without a key; cloud mode requires one.
                let key = api_key.unwrap_or_default();

                let mut hf_provider = HuggingFaceEmbeddings::new(&model, &key);
                if let Some(dim) = dimension {
                    hf_provider = hf_provider.with_dimension(dim);
                }

                Ok(PyEmbeddings {
                    inner: Arc::new(Embeddings::new(Arc::new(hf_provider))),
                })
            }

            _ => Err(crate::error::ValidationError::new_err(format!(
                "Unknown provider: {}. Supported: openai, mistral, ollama, huggingface, mock",
                provider
            ))),
        }
    }

    /// Create mock embeddings for testing (hash-based, no API needed).
    #[staticmethod]
    fn mock(dimension: usize) -> Self {
        PyEmbeddings {
            inner: Arc::new(Embeddings::mock(dimension)),
        }
    }

    /// Create OpenAI embeddings.
    ///
    /// Example:
    ///     emb = Embeddings.openai("sk-...", "text-embedding-3-small")
    #[staticmethod]
    #[pyo3(signature = (api_key, model="text-embedding-3-small"))]
    fn openai(api_key: &str, model: &str) -> Self {
        let provider = OpenAIEmbeddings::new(api_key, model);
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(Arc::new(provider))),
        }
    }

    /// Create OpenAI embeddings with custom dimension.
    #[staticmethod]
    fn openai_with_dimension(api_key: &str, model: &str, dimension: usize) -> Self {
        let provider = OpenAIEmbeddings::new(api_key, model).with_dimension(dimension);
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(Arc::new(provider))),
        }
    }

    /// Create HuggingFace embeddings.
    #[staticmethod]
    #[pyo3(signature = (model, api_key, endpoint=None, dimension=None))]
    fn huggingface(
        model: &str,
        api_key: &str,
        endpoint: Option<&str>,
        dimension: Option<usize>,
    ) -> Self {
        let mut provider = HuggingFaceEmbeddings::new(model, api_key);
        if let Some(ep) = endpoint {
            provider = provider.with_endpoint(ep);
        }
        if let Some(dim) = dimension {
            provider = provider.with_dimension(dim);
        }
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(Arc::new(provider))),
        }
    }

    /// Create Ollama embeddings (local).
    #[staticmethod]
    #[pyo3(signature = (model, base_url=None, dimension=None))]
    fn ollama(model: &str, base_url: Option<String>, dimension: Option<usize>) -> Self {
        let mut provider = OllamaEmbeddings::new(model, base_url);
        if let Some(dim) = dimension {
            provider = provider.with_dimension(dim);
        }
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(Arc::new(provider))),
        }
    }

    /// Create Mistral embeddings.
    #[staticmethod]
    #[pyo3(signature = (api_key, model=None))]
    fn mistral(api_key: &str, model: Option<String>) -> Self {
        let provider = MistralEmbeddings::new(api_key, model);
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(Arc::new(provider))),
        }
    }

    /// Create OpenAI embeddings with a cache layer.
    #[staticmethod]
    #[pyo3(signature = (api_key, model="text-embedding-3-small"))]
    fn openai_cached(api_key: &str, model: &str) -> Self {
        let provider = OpenAIEmbeddings::new(api_key, model);
        let cached = CachedEmbeddings::new(Arc::new(provider));
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(Arc::new(cached))),
        }
    }

    /// Generate embedding for a single text.
    fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        crate::run_async(self.inner.embed(text))
            .map_err(|e| crate::error::InternalError::new_err(format!("{}", e)))
    }

    /// Generate embeddings for multiple texts (batch).
    fn embed_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        crate::run_async(self.inner.embed_batch(refs))
            .map_err(|e| crate::error::InternalError::new_err(format!("{}", e)))
    }

    /// Get embedding dimension.
    fn get_dimension(&self) -> usize {
        self.inner.get_dimension()
    }

    fn __repr__(&self) -> String {
        format!("Embeddings(dim={})", self.inner.get_dimension())
    }
}

// ─── PyInMemoryVectorStore ──────────────────────────────────────────────────

/// In-memory vector store with cosine similarity search.
///
/// Example:
///     store = InMemoryVectorStore()
///     emb = Embeddings.mock(128)
///     doc = Document("doc1", "Hello world")
///     doc_emb = emb.embed("Hello world")
///     store.index(doc, doc_emb)
///     results = store.search(emb.embed("Hello"), top_k=5)
#[pyclass(name = "InMemoryVectorStore")]
pub struct PyInMemoryVectorStore {
    pub(crate) inner: Arc<InMemoryVectorStore>,
}

#[pymethods]
impl PyInMemoryVectorStore {
    /// Create an empty in-memory vector store or with documents.
    ///
    /// Example:
    ///     # Empty store
    ///     store = InMemoryVectorStore()
    ///
    ///     # With documents and auto-embedding
    ///     store = InMemoryVectorStore(
    ///         documents=[doc1, doc2],
    ///         auto_embed=True,
    ///         embeddings_model="mock"  # or "openai", "ollama", etc.
    ///     )
    #[new]
    #[pyo3(signature = (documents=None, auto_embed=false, embeddings_model=None))]
    fn new(
        documents: Option<Vec<PyDocument>>,
        auto_embed: bool,
        embeddings_model: Option<&str>,
    ) -> PyResult<Self> {
        let store = Arc::new(InMemoryVectorStore::new());
        let result = PyInMemoryVectorStore { inner: store };

        // If documents provided, batch add them
        if let Some(docs) = documents {
            if docs.is_empty() {
                return Ok(result);
            }

            // If auto-embed is enabled, create embeddings and index documents
            if auto_embed {
                let model = embeddings_model.unwrap_or("mock");

                // Create appropriate embeddings
                let embeddings = match model {
                    "mock" => Arc::new(Embeddings::mock(128)),
                    "openai" => {
                        // For OpenAI, we'd need an API key - using mock for now
                        // In production, this should be configurable
                        Arc::new(Embeddings::mock(1536))
                    }
                    "ollama" => {
                        // Default Ollama setup
                        Arc::new(Embeddings::new(Arc::new(OllamaEmbeddings::new(
                            "nomic-embed-text",
                            None,
                        ))))
                    }
                    _ => Arc::new(Embeddings::mock(128)), // Default to mock
                };

                // Embed all documents
                let texts_owned: Vec<String> = docs.iter().map(|d| d.inner.text.clone()).collect();
                let texts: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
                let embeddings_result = crate::run_async(embeddings.embed_batch(texts));

                if let Ok(vectors) = embeddings_result {
                    // Index all documents with their embeddings
                    for (doc, embedding) in docs.iter().zip(vectors.iter()) {
                        let mut document = doc.inner.clone();
                        document.embedding = Some(embedding.clone());

                        let index_result = crate::run_async(result.inner.index(document));
                        if let Err(e) = index_result {
                            return Err(to_py_err_generic(e));
                        }
                    }
                } else {
                    return Err(crate::error::InternalError::new_err(
                        "Failed to generate embeddings for documents",
                    ));
                }
            } else {
                // Without auto-embed, just add documents as-is (requires pre-computed embeddings)
                for doc in docs {
                    let index_result = crate::run_async(result.inner.index(doc.inner.clone()));
                    if let Err(e) = index_result {
                        return Err(to_py_err_generic(e));
                    }
                }
            }
        }

        Ok(result)
    }

    /// Add a single document (legacy API, kept for compatibility).
    /// For batch operations, use the constructor with documents parameter.
    fn add(&self, doc: &PyDocument) -> PyResult<()> {
        crate::run_async(self.inner.index(doc.inner.clone())).map_err(to_py_err_generic)
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

    /// Search with multiple query embeddings in a single call.
    ///
    /// Args:
    ///     query_embeddings: List of query vectors.
    ///     top_k:            Results per query (default 5).
    ///     filter:           Optional metadata filter applied to all queries.
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
        let all_results = crate::run_async(self.inner.search_batch(
            query_embeddings,
            top_k,
            metadata_filter,
        ))
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
        "InMemoryVectorStore(...)".to_string()
    }
}

// ─── PyRetrievalConfig ──────────────────────────────────────────────────────

/// Configuration for retrieval strategies.
///
/// Example:
///     config = RetrievalConfig.semantic(top_k=5, threshold=0.7)
///     config = RetrievalConfig.hybrid(keyword_weight=0.3, top_k=10)
///     config = RetrievalConfig.multi_query(query_variants=3, top_k=5)
///     config = RetrievalConfig.decomposed(decomposition_depth=2, top_k=5)
///     config = RetrievalConfig.custom("my_chain", top_k=5)
#[pyclass(name = "RetrievalConfig")]
#[derive(Clone)]
pub struct PyRetrievalConfig {
    pub(crate) inner: RetrievalConfig,
}

#[pymethods]
impl PyRetrievalConfig {
    /// Create a semantic search config.
    #[staticmethod]
    #[pyo3(signature = (top_k=5, threshold=0.7))]
    fn semantic(top_k: usize, threshold: f32) -> Self {
        PyRetrievalConfig {
            inner: RetrievalConfig::new(RetrieverStrategy::SemanticSearch)
                .with_top_k(top_k)
                .with_threshold(threshold),
        }
    }

    /// Create a hybrid (semantic + keyword) search config.
    #[staticmethod]
    #[pyo3(signature = (keyword_weight=0.3, top_k=5, threshold=0.7))]
    fn hybrid(keyword_weight: f32, top_k: usize, threshold: f32) -> Self {
        PyRetrievalConfig {
            inner: RetrievalConfig::new(RetrieverStrategy::hybrid(keyword_weight))
                .with_top_k(top_k)
                .with_threshold(threshold),
        }
    }

    /// Create a multi-query config (LLM expands query into variants).
    #[staticmethod]
    #[pyo3(signature = (query_variants=3, top_k=5, threshold=0.7))]
    fn multi_query(query_variants: usize, top_k: usize, threshold: f32) -> Self {
        PyRetrievalConfig {
            inner: RetrievalConfig::new(RetrieverStrategy::multiquery(query_variants))
                .with_top_k(top_k)
                .with_threshold(threshold),
        }
    }

    /// Create a decomposed query config (splits complex queries into sub-queries).
    #[staticmethod]
    #[pyo3(signature = (decomposition_depth=2, top_k=5, threshold=0.7))]
    fn decomposed(decomposition_depth: usize, top_k: usize, threshold: f32) -> Self {
        PyRetrievalConfig {
            inner: RetrievalConfig::new(RetrieverStrategy::decomposed(decomposition_depth))
                .with_top_k(top_k)
                .with_threshold(threshold),
        }
    }

    /// Create a custom chain strategy config.
    #[staticmethod]
    #[pyo3(signature = (chain_name, top_k=5, threshold=0.7))]
    fn custom(chain_name: String, top_k: usize, threshold: f32) -> Self {
        PyRetrievalConfig {
            inner: RetrievalConfig::new(RetrieverStrategy::Custom { chain_name })
                .with_top_k(top_k)
                .with_threshold(threshold),
        }
    }

    #[getter]
    fn top_k(&self) -> usize {
        self.inner.top_k
    }

    #[getter]
    fn similarity_threshold(&self) -> f32 {
        self.inner.similarity_threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "RetrievalConfig(top_k={}, threshold={})",
            self.inner.top_k, self.inner.similarity_threshold
        )
    }
}

// ─── PyRetriever ────────────────────────────────────────────────────────────

/// Full retrieval pipeline: embed → search → hybrid → rerank → dedup.
///
/// Example:
///     store = InMemoryVectorStore()
///     emb = Embeddings.mock(128)
///     config = RetrievalConfig.semantic(top_k=3, threshold=0.0)
///     retriever = Retriever(store, emb, config)
///     results = retriever.retrieve("What is Rust?")
#[pyclass(name = "Retriever")]
pub struct PyRetriever {
    inner: Retriever,
}

#[pymethods]
impl PyRetriever {
    #[new]
    fn new(
        store: &PyInMemoryVectorStore,
        embeddings: &PyEmbeddings,
        config: &PyRetrievalConfig,
    ) -> Self {
        PyRetriever {
            inner: Retriever::new(
                store.inner.clone() as Arc<dyn VectorStoreBackend>,
                embeddings.inner.clone(),
                config.inner.clone(),
            ),
        }
    }

    /// Enable deduplication with a similarity threshold (e.g., 0.85).
    fn with_dedup(&mut self, threshold: f32) {
        // Reconstruct since with_dedup consumes self
        let inner = std::mem::replace(
            &mut self.inner,
            Retriever::new(
                Arc::new(InMemoryVectorStore::new()),
                Arc::new(Embeddings::mock(1)),
                RetrievalConfig::default(),
            ),
        );
        self.inner = inner.with_dedup(threshold);
    }

    /// Execute the full retrieval pipeline for a query.
    fn retrieve(&self, query: &str) -> PyResult<Vec<PySearchResult>> {
        let results = crate::run_async(self.inner.retrieve(query)).map_err(to_py_err_generic)?;
        Ok(results
            .into_iter()
            .map(|r| PySearchResult { inner: r })
            .collect())
    }

    fn __repr__(&self) -> String {
        "Retriever(...)".to_string()
    }
}
