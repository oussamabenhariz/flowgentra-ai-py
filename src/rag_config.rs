//! Python bindings for RAG configuration types

use pyo3::prelude::*;

use flowgentra_ai::core::config::{
    EmbeddingsConfig, PdfSettings, RAGGraphConfig, RetrievalSettings, VectorStoreConfig,
};
use flowgentra_ai::core::rag::VectorStoreType;

// ─── PyVectorStoreType ──────────────────────────────────────────────────────

/// Vector store backend type.
///
/// Available types: pinecone, weaviate, chroma, milvus, qdrant, memory.
#[pyclass(name = "VectorStoreType")]
#[derive(Clone)]
pub struct PyVectorStoreType {
    pub(crate) inner: VectorStoreType,
}

#[pymethods]
impl PyVectorStoreType {
    #[staticmethod]
    fn pinecone() -> Self {
        PyVectorStoreType { inner: VectorStoreType::Pinecone }
    }
    #[staticmethod]
    fn weaviate() -> Self {
        PyVectorStoreType { inner: VectorStoreType::Weaviate }
    }
    #[staticmethod]
    fn chroma() -> Self {
        PyVectorStoreType { inner: VectorStoreType::Chroma }
    }
    #[staticmethod]
    fn milvus() -> Self {
        PyVectorStoreType { inner: VectorStoreType::Milvus }
    }
    #[staticmethod]
    fn qdrant() -> Self {
        PyVectorStoreType { inner: VectorStoreType::Qdrant }
    }
    #[staticmethod]
    fn memory() -> Self {
        PyVectorStoreType { inner: VectorStoreType::Memory }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            VectorStoreType::Pinecone => "VectorStoreType.Pinecone",
            VectorStoreType::Weaviate => "VectorStoreType.Weaviate",
            VectorStoreType::Chroma => "VectorStoreType.Chroma",
            VectorStoreType::Milvus => "VectorStoreType.Milvus",
            VectorStoreType::Qdrant => "VectorStoreType.Qdrant",
            VectorStoreType::Memory => "VectorStoreType.Memory",
        }
        .to_string()
    }
}

// ─── PyRAGConfig ────────────────────────────────────────────────────────────

/// Low-level RAG configuration for vector store connections.
///
/// Example:
///     config = RAGConfig.memory(embedding_dim=1536)
///     config = RAGConfig.chroma("http://localhost:8000")
#[pyclass(name = "RAGConfig")]
#[derive(Clone)]
pub struct PyRAGConfig {
    pub(crate) inner: flowgentra_ai::core::rag::RAGConfig,
}

#[pymethods]
impl PyRAGConfig {
    /// Create a config for in-memory vector store.
    #[staticmethod]
    #[pyo3(signature = (embedding_dim=1536))]
    fn memory(embedding_dim: usize) -> PyResult<Self> {
        let config = flowgentra_ai::core::rag::RAGConfig::memory(embedding_dim)
            .map_err(|e| pyo3::exceptions::crate::error::ConfigurationError::new_err(format!("{}", e)))?;
        Ok(PyRAGConfig { inner: config })
    }

    /// Create a config for ChromaDB.
    #[staticmethod]
    #[pyo3(signature = (endpoint="http://localhost:8000"))]
    fn chroma(endpoint: &str) -> PyResult<Self> {
        let config = flowgentra_ai::core::rag::RAGConfig::chroma(endpoint)
            .map_err(|e| pyo3::exceptions::crate::error::ConfigurationError::new_err(format!("{}", e)))?;
        Ok(PyRAGConfig { inner: config })
    }

    /// Create a config for Pinecone.
    #[staticmethod]
    fn pinecone(index: &str, api_key: &str) -> PyResult<Self> {
        let config = flowgentra_ai::core::rag::RAGConfig::pinecone(index, api_key)
            .map_err(|e| pyo3::exceptions::crate::error::ConfigurationError::new_err(format!("{}", e)))?;
        Ok(PyRAGConfig { inner: config })
    }

    /// Create a config for Qdrant.
    #[staticmethod]
    #[pyo3(signature = (endpoint, collection, embedding_dim=1536))]
    fn qdrant(endpoint: &str, collection: &str, embedding_dim: usize) -> PyResult<Self> {
        let config = flowgentra_ai::core::rag::RAGConfig::qdrant(endpoint, collection, embedding_dim)
            .map_err(|e| pyo3::exceptions::crate::error::ConfigurationError::new_err(format!("{}", e)))?;
        Ok(PyRAGConfig { inner: config })
    }

    /// Create a config for Weaviate.
    #[staticmethod]
    fn weaviate(endpoint: &str) -> PyResult<Self> {
        let config = flowgentra_ai::core::rag::RAGConfig::weaviate(endpoint)
            .map_err(|e| pyo3::exceptions::crate::error::ConfigurationError::new_err(format!("{}", e)))?;
        Ok(PyRAGConfig { inner: config })
    }

    /// Create a config for Milvus.
    #[staticmethod]
    #[pyo3(signature = (endpoint, collection, embedding_dim=1536))]
    fn milvus(endpoint: &str, collection: &str, embedding_dim: usize) -> PyResult<Self> {
        let config = flowgentra_ai::core::rag::RAGConfig::milvus(endpoint, collection, embedding_dim)
            .map_err(|e| pyo3::exceptions::crate::error::ConfigurationError::new_err(format!("{}", e)))?;
        Ok(PyRAGConfig { inner: config })
    }

    #[getter]
    fn store_type(&self) -> PyVectorStoreType {
        PyVectorStoreType { inner: self.inner.store_type.clone() }
    }

    #[getter]
    fn index_name(&self) -> String {
        self.inner.index_name.clone()
    }

    #[getter]
    fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim
    }

    #[getter]
    fn endpoint(&self) -> Option<String> {
        self.inner.endpoint.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RAGConfig(index='{}', dim={})",
            self.inner.index_name, self.inner.embedding_dim
        )
    }
}

// ─── PyVectorStoreConfig ────────────────────────────────────────────────────

/// Vector store configuration for agent config YAML.
///
/// Example:
///     config = VectorStoreConfig(VectorStoreType.chroma(), "my_docs")
#[pyclass(name = "VectorStoreConfig")]
#[derive(Clone)]
pub struct PyVectorStoreConfig {
    pub(crate) inner: VectorStoreConfig,
}

#[pymethods]
impl PyVectorStoreConfig {
    #[new]
    #[pyo3(signature = (store_type, collection="documents", endpoint=None, api_key=None, namespace=None, embedding_dim=None))]
    fn new(
        store_type: &PyVectorStoreType,
        collection: &str,
        endpoint: Option<String>,
        api_key: Option<String>,
        namespace: Option<String>,
        embedding_dim: Option<usize>,
    ) -> Self {
        PyVectorStoreConfig {
            inner: VectorStoreConfig {
                store_type: store_type.inner.clone(),
                endpoint,
                collection: collection.to_string(),
                api_key,
                namespace,
                embedding_dim,
            },
        }
    }

    #[getter]
    fn store_type(&self) -> PyVectorStoreType {
        PyVectorStoreType { inner: self.inner.store_type.clone() }
    }

    #[getter]
    fn collection(&self) -> String {
        self.inner.collection.clone()
    }

    #[getter]
    fn endpoint(&self) -> Option<String> {
        self.inner.endpoint.clone()
    }

    #[getter]
    fn namespace(&self) -> Option<String> {
        self.inner.namespace.clone()
    }

    #[getter]
    fn embedding_dim(&self) -> Option<usize> {
        self.inner.embedding_dim
    }

    /// Return a copy with all `${ENV_VAR}` tokens resolved from the environment.
    fn resolved(&self) -> Self {
        PyVectorStoreConfig { inner: self.inner.resolved() }
    }

    fn __repr__(&self) -> String {
        format!("VectorStoreConfig(collection='{}')", self.inner.collection)
    }
}

// ─── PyEmbeddingsConfig ─────────────────────────────────────────────────────

/// Embeddings provider configuration.
///
/// Example:
///     config = EmbeddingsConfig("openai", model="text-embedding-3-small")
#[pyclass(name = "EmbeddingsConfig")]
#[derive(Clone)]
pub struct PyEmbeddingsConfig {
    pub(crate) inner: EmbeddingsConfig,
}

#[pymethods]
impl PyEmbeddingsConfig {
    #[new]
    #[pyo3(signature = (provider, model=None, api_key=None, dimension=None))]
    fn new(
        provider: &str,
        model: Option<String>,
        api_key: Option<String>,
        dimension: Option<usize>,
    ) -> Self {
        PyEmbeddingsConfig {
            inner: EmbeddingsConfig {
                provider: provider.to_string(),
                model,
                api_key,
                dimension,
            },
        }
    }

    #[getter]
    fn provider(&self) -> String {
        self.inner.provider.clone()
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    #[getter]
    fn dimension(&self) -> Option<usize> {
        self.inner.dimension
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingsConfig(provider='{}', model={:?})",
            self.inner.provider, self.inner.model
        )
    }
}

// ─── PyRetrievalSettings ────────────────────────────────────────────────────

/// Retrieval settings for RAG.
///
/// Example:
///     settings = RetrievalSettings(top_k=10, similarity_threshold=0.8)
#[pyclass(name = "RetrievalSettings")]
#[derive(Clone)]
pub struct PyRetrievalSettings {
    pub(crate) inner: RetrievalSettings,
}

#[pymethods]
impl PyRetrievalSettings {
    #[new]
    #[pyo3(signature = (top_k=5, similarity_threshold=0.7))]
    fn new(top_k: usize, similarity_threshold: f32) -> Self {
        PyRetrievalSettings {
            inner: RetrievalSettings {
                top_k,
                similarity_threshold,
            },
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
            "RetrievalSettings(top_k={}, threshold={:.2})",
            self.inner.top_k, self.inner.similarity_threshold
        )
    }
}

// ─── PyPdfSettings ──────────────────────────────────────────────────────────

/// PDF processing settings.
///
/// Example:
///     settings = PdfSettings(chunk_size=500, chunk_overlap=100)
#[pyclass(name = "PdfSettings")]
#[derive(Clone)]
pub struct PyPdfSettings {
    pub(crate) inner: PdfSettings,
}

#[pymethods]
impl PyPdfSettings {
    #[new]
    #[pyo3(signature = (chunk_size=1000, chunk_overlap=200))]
    fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        PyPdfSettings {
            inner: PdfSettings {
                chunk_size,
                chunk_overlap,
            },
        }
    }

    #[getter]
    fn chunk_size(&self) -> usize {
        self.inner.chunk_size
    }

    #[getter]
    fn chunk_overlap(&self) -> usize {
        self.inner.chunk_overlap
    }

    fn __repr__(&self) -> String {
        format!(
            "PdfSettings(chunk_size={}, chunk_overlap={})",
            self.inner.chunk_size, self.inner.chunk_overlap
        )
    }
}

// ─── PyRAGGraphConfig ───────────────────────────────────────────────────────

/// Complete RAG configuration for agent graph config.
///
/// Example:
///     rag = RAGGraphConfig(
///         vector_store=VectorStoreConfig(VectorStoreType.chroma(), "docs"),
///         embeddings=EmbeddingsConfig("openai"),
///     )
#[pyclass(name = "RAGGraphConfig")]
#[derive(Clone)]
pub struct PyRAGGraphConfig {
    pub(crate) inner: RAGGraphConfig,
}

#[pymethods]
impl PyRAGGraphConfig {
    #[new]
    #[pyo3(signature = (vector_store, embeddings, retrieval=None, pdf=None))]
    fn new(
        vector_store: &PyVectorStoreConfig,
        embeddings: &PyEmbeddingsConfig,
        retrieval: Option<&PyRetrievalSettings>,
        pdf: Option<&PyPdfSettings>,
    ) -> Self {
        PyRAGGraphConfig {
            inner: RAGGraphConfig {
                vector_store: vector_store.inner.clone(),
                embeddings: embeddings.inner.clone(),
                retrieval: retrieval.map(|r| r.inner.clone()).unwrap_or_default(),
                pdf: pdf.map(|p| p.inner.clone()).unwrap_or_default(),
            },
        }
    }

    #[getter]
    fn vector_store(&self) -> PyVectorStoreConfig {
        PyVectorStoreConfig { inner: self.inner.vector_store.clone() }
    }

    #[getter]
    fn embeddings(&self) -> PyEmbeddingsConfig {
        PyEmbeddingsConfig { inner: self.inner.embeddings.clone() }
    }

    #[getter]
    fn retrieval(&self) -> PyRetrievalSettings {
        PyRetrievalSettings { inner: self.inner.retrieval.clone() }
    }

    #[getter]
    fn pdf(&self) -> PyPdfSettings {
        PyPdfSettings { inner: self.inner.pdf.clone() }
    }

    /// Get the resolved embedding dimension.
    fn embedding_dimension(&self) -> usize {
        self.inner.embedding_dimension()
    }

    fn __repr__(&self) -> String {
        format!(
            "RAGGraphConfig(collection='{}', provider='{}')",
            self.inner.vector_store.collection, self.inner.embeddings.provider
        )
    }
}
