//! Python bindings for additional embedding providers.
//!
//! All providers expose the same ``Embeddings``-compatible interface:
//!
//! ```python
//! from flowgentra_ai.rag import (
//!     CohereEmbeddings, AzureOpenAIEmbeddings, GoogleVertexEmbeddings,
//!     BedrockEmbeddings, VoyageEmbeddings, JinaEmbeddings,
//!     TogetherEmbeddings, NomicEmbeddings,
//! )
//!
//! emb = CohereEmbeddings(api_key="...", model="embed-english-v3.0")
//! vec = emb.embed("Hello world")
//! vecs = emb.embed_batch(["Hello", "World"])
//! ```

use std::sync::Arc;

use pyo3::prelude::*;

use flowgentra_ai::core::rag::embeddings::{Embeddings, EmbeddingsProvider};
use flowgentra_ai::core::rag::extra_embeddings::{
    AzureOpenAIEmbeddings, BedrockEmbeddings, CohereEmbeddings, GoogleVertexEmbeddings,
    JinaEmbeddings, NomicEmbeddings, TogetherEmbeddings, VoyageEmbeddings,
};

use crate::run_async;
use crate::vector_store::PyEmbeddings;

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    crate::error::InternalError::new_err(e.to_string())
}

// ── PyCohereEmbeddings ────────────────────────────────────────────────────────

/// Embeddings via Cohere's ``/v1/embed`` endpoint.
///
/// Example::
///
///     emb = CohereEmbeddings(api_key="co-...", model="embed-english-v3.0")
///     vec = emb.embed("Hello world")
///     # Use as_embeddings() to pass to retrievers that need an Embeddings object
///     retriever = VectorRetriever(store, emb.as_embeddings(), top_k=5)
#[pyclass(name = "CohereEmbeddings")]
pub struct PyCohereEmbeddings {
    inner: Arc<CohereEmbeddings>,
}

#[pymethods]
impl PyCohereEmbeddings {
    /// Args:
    ///     api_key: Cohere API key.
    ///     model:   Embedding model ID (default ``"embed-english-v3.0"``).
    ///     for_query: If True, use ``search_query`` input type (default False = ``search_document``).
    #[new]
    #[pyo3(signature = (api_key, model="embed-english-v3.0", for_query=false))]
    fn new(api_key: &str, model: &str, for_query: bool) -> Self {
        let mut inner = CohereEmbeddings::new(api_key, model);
        if for_query {
            inner = inner.for_query();
        }
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Embed a single text string. Returns a ``list[float]``.
    fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        let t = text.to_string();
        run_async(async move { inner.embed(&t).await }).map_err(to_py_err)
    }

    /// Embed multiple texts. Returns ``list[list[float]]``.
    fn embed_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        run_async(async move {
            inner
                .embed_batch(texts.iter().map(|s| s.as_str()).collect())
                .await
        })
        .map_err(to_py_err)
    }

    /// Wrap as an ``Embeddings`` object for use with retrievers.
    fn as_embeddings(&self) -> PyEmbeddings {
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(self.inner.clone())),
        }
    }

    fn __repr__(&self) -> String {
        format!("CohereEmbeddings(model='{}')", self.inner.model)
    }
}

// ── PyAzureOpenAIEmbeddings ───────────────────────────────────────────────────

/// Embeddings via Azure OpenAI deployment endpoint.
///
/// Example::
///
///     emb = AzureOpenAIEmbeddings(
///         endpoint="https://my-resource.openai.azure.com",
///         deployment="text-embedding-ada-002",
///         api_key="...",
///     )
///     vec = emb.embed("Hello world")
#[pyclass(name = "AzureOpenAIEmbeddings")]
pub struct PyAzureOpenAIEmbeddings {
    inner: Arc<AzureOpenAIEmbeddings>,
}

#[pymethods]
impl PyAzureOpenAIEmbeddings {
    /// Args:
    ///     endpoint:    Azure OpenAI resource endpoint URL.
    ///     deployment:  Deployment name (model).
    ///     api_key:     Azure OpenAI API key.
    ///     api_version: API version (default ``"2024-02-01"``).
    #[new]
    #[pyo3(signature = (endpoint, deployment, api_key, api_version="2024-02-01"))]
    fn new(endpoint: &str, deployment: &str, api_key: &str, api_version: &str) -> Self {
        let mut inner = AzureOpenAIEmbeddings::new(endpoint, deployment, api_key);
        inner.api_version = api_version.to_string();
        Self {
            inner: Arc::new(inner),
        }
    }

    fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        let t = text.to_string();
        run_async(async move { inner.embed(&t).await }).map_err(to_py_err)
    }

    fn embed_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        run_async(async move {
            inner
                .embed_batch(texts.iter().map(|s| s.as_str()).collect())
                .await
        })
        .map_err(to_py_err)
    }

    fn as_embeddings(&self) -> PyEmbeddings {
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(self.inner.clone())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AzureOpenAIEmbeddings(deployment='{}')",
            self.inner.deployment
        )
    }
}

// ── PyGoogleVertexEmbeddings ──────────────────────────────────────────────────

/// Embeddings via Google Vertex AI text-embedding models.
///
/// Example::
///
///     emb = GoogleVertexEmbeddings(
///         endpoint="https://us-central1-aiplatform.googleapis.com/v1/projects/MY_PROJECT/locations/us-central1/publishers/google/models/textembedding-gecko:predict",
///         access_token="ya29...",
///     )
///     vec = emb.embed("Hello world")
#[pyclass(name = "GoogleVertexEmbeddings")]
pub struct PyGoogleVertexEmbeddings {
    inner: Arc<GoogleVertexEmbeddings>,
}

#[pymethods]
impl PyGoogleVertexEmbeddings {
    /// Args:
    ///     endpoint:     Full Vertex AI predict endpoint URL.
    ///     access_token: Short-lived Google OAuth2 access token.
    #[new]
    fn new(endpoint: &str, access_token: &str) -> Self {
        Self {
            inner: Arc::new(GoogleVertexEmbeddings::new(endpoint, access_token)),
        }
    }

    fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        let t = text.to_string();
        run_async(async move { inner.embed(&t).await }).map_err(to_py_err)
    }

    fn embed_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        run_async(async move {
            inner
                .embed_batch(texts.iter().map(|s| s.as_str()).collect())
                .await
        })
        .map_err(to_py_err)
    }

    fn as_embeddings(&self) -> PyEmbeddings {
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(self.inner.clone())),
        }
    }

    fn __repr__(&self) -> String {
        "GoogleVertexEmbeddings(...)".to_string()
    }
}

// ── PyBedrockEmbeddings ───────────────────────────────────────────────────────

/// Embeddings via AWS Bedrock (Amazon Titan Embed or Cohere on Bedrock).
///
/// Example::
///
///     emb = BedrockEmbeddings(
///         region="us-east-1",
///         model_id="amazon.titan-embed-text-v2:0",
///         access_key="AKIA...",
///         secret_key="...",
///     )
///     vec = emb.embed("Hello world")
#[pyclass(name = "BedrockEmbeddings")]
pub struct PyBedrockEmbeddings {
    inner: Arc<BedrockEmbeddings>,
}

#[pymethods]
impl PyBedrockEmbeddings {
    /// Args:
    ///     region:     AWS region (e.g. ``"us-east-1"``).
    ///     model_id:   Bedrock model ID (e.g. ``"amazon.titan-embed-text-v2:0"``).
    ///     access_key: AWS Access Key ID.
    ///     secret_key: AWS Secret Access Key.
    #[new]
    fn new(region: &str, model_id: &str, access_key: &str, secret_key: &str) -> Self {
        Self {
            inner: Arc::new(BedrockEmbeddings::new(
                region, model_id, access_key, secret_key,
            )),
        }
    }

    fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        let t = text.to_string();
        run_async(async move { inner.embed(&t).await }).map_err(to_py_err)
    }

    fn embed_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        run_async(async move {
            inner
                .embed_batch(texts.iter().map(|s| s.as_str()).collect())
                .await
        })
        .map_err(to_py_err)
    }

    fn as_embeddings(&self) -> PyEmbeddings {
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(self.inner.clone())),
        }
    }

    fn __repr__(&self) -> String {
        format!("BedrockEmbeddings(model_id='{}')", self.inner.model_id)
    }
}

// ── PyVoyageEmbeddings ────────────────────────────────────────────────────────

/// Embeddings via Voyage AI (``voyage-2``, ``voyage-code-2``, etc.).
///
/// Example::
///
///     emb = VoyageEmbeddings(api_key="pa-...", model="voyage-2")
///     vec = emb.embed("Hello world")
#[pyclass(name = "VoyageEmbeddings")]
pub struct PyVoyageEmbeddings {
    inner: Arc<VoyageEmbeddings>,
}

#[pymethods]
impl PyVoyageEmbeddings {
    /// Args:
    ///     api_key:    Voyage AI API key.
    ///     model:      Model name (default ``"voyage-2"``).
    ///     input_type: Optional input type hint (``"query"`` or ``"document"``).
    #[new]
    #[pyo3(signature = (api_key, model="voyage-2", input_type=None))]
    fn new(api_key: &str, model: &str, input_type: Option<String>) -> Self {
        let mut inner = VoyageEmbeddings::new(api_key, model);
        if let Some(t) = input_type {
            inner = inner.with_input_type(t);
        }
        Self {
            inner: Arc::new(inner),
        }
    }

    fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        let t = text.to_string();
        run_async(async move { inner.embed(&t).await }).map_err(to_py_err)
    }

    fn embed_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        run_async(async move {
            inner
                .embed_batch(texts.iter().map(|s| s.as_str()).collect())
                .await
        })
        .map_err(to_py_err)
    }

    fn as_embeddings(&self) -> PyEmbeddings {
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(self.inner.clone())),
        }
    }

    fn __repr__(&self) -> String {
        format!("VoyageEmbeddings(model='{}')", self.inner.model)
    }
}

// ── PyJinaEmbeddings ──────────────────────────────────────────────────────────

/// Embeddings via Jina AI (``jina-embeddings-v2-base-en``, etc.).
///
/// Example::
///
///     emb = JinaEmbeddings(api_key="jina_...", model="jina-embeddings-v2-base-en")
///     vec = emb.embed("Hello world")
#[pyclass(name = "JinaEmbeddings")]
pub struct PyJinaEmbeddings {
    inner: Arc<JinaEmbeddings>,
}

#[pymethods]
impl PyJinaEmbeddings {
    /// Args:
    ///     api_key: Jina AI API key.
    ///     model:   Model name (default ``"jina-embeddings-v2-base-en"``).
    #[new]
    #[pyo3(signature = (api_key, model="jina-embeddings-v2-base-en"))]
    fn new(api_key: &str, model: &str) -> Self {
        Self {
            inner: Arc::new(JinaEmbeddings::new(api_key, model)),
        }
    }

    fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        let t = text.to_string();
        run_async(async move { inner.embed(&t).await }).map_err(to_py_err)
    }

    fn embed_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        run_async(async move {
            inner
                .embed_batch(texts.iter().map(|s| s.as_str()).collect())
                .await
        })
        .map_err(to_py_err)
    }

    fn as_embeddings(&self) -> PyEmbeddings {
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(self.inner.clone())),
        }
    }

    fn __repr__(&self) -> String {
        format!("JinaEmbeddings(model='{}')", self.inner.model)
    }
}

// ── PyTogetherEmbeddings ──────────────────────────────────────────────────────

/// Embeddings via Together AI (OpenAI-compatible endpoint).
///
/// Example::
///
///     emb = TogetherEmbeddings(api_key="...", model="togethercomputer/m2-bert-80M-8k-retrieval")
///     vec = emb.embed("Hello world")
#[pyclass(name = "TogetherEmbeddings")]
pub struct PyTogetherEmbeddings {
    inner: Arc<TogetherEmbeddings>,
}

#[pymethods]
impl PyTogetherEmbeddings {
    /// Args:
    ///     api_key: Together AI API key.
    ///     model:   Model name.
    #[new]
    fn new(api_key: &str, model: &str) -> Self {
        Self {
            inner: Arc::new(TogetherEmbeddings::new(api_key, model)),
        }
    }

    fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        let t = text.to_string();
        run_async(async move { inner.embed(&t).await }).map_err(to_py_err)
    }

    fn embed_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        run_async(async move {
            inner
                .embed_batch(texts.iter().map(|s| s.as_str()).collect())
                .await
        })
        .map_err(to_py_err)
    }

    fn as_embeddings(&self) -> PyEmbeddings {
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(self.inner.clone())),
        }
    }

    fn __repr__(&self) -> String {
        format!("TogetherEmbeddings(model='{}')", self.inner.model)
    }
}

// ── PyNomicEmbeddings ─────────────────────────────────────────────────────────

/// Embeddings via Nomic Atlas (``nomic-embed-text-v1``, etc.).
///
/// Example::
///
///     emb = NomicEmbeddings(api_key="nk-...", model="nomic-embed-text-v1")
///     vec = emb.embed("Hello world")
///     # For query encoding:
///     emb_q = NomicEmbeddings(api_key="nk-...", model="nomic-embed-text-v1", for_query=True)
#[pyclass(name = "NomicEmbeddings")]
pub struct PyNomicEmbeddings {
    inner: Arc<NomicEmbeddings>,
}

#[pymethods]
impl PyNomicEmbeddings {
    /// Args:
    ///     api_key:   Nomic API key.
    ///     model:     Model name (default ``"nomic-embed-text-v1"``).
    ///     for_query: If True, use ``search_query`` task type (default False).
    #[new]
    #[pyo3(signature = (api_key, model="nomic-embed-text-v1", for_query=false))]
    fn new(api_key: &str, model: &str, for_query: bool) -> Self {
        let mut inner = NomicEmbeddings::new(api_key, model);
        if for_query {
            inner = inner.for_query();
        }
        Self {
            inner: Arc::new(inner),
        }
    }

    fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        let t = text.to_string();
        run_async(async move { inner.embed(&t).await }).map_err(to_py_err)
    }

    fn embed_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        run_async(async move {
            inner
                .embed_batch(texts.iter().map(|s| s.as_str()).collect())
                .await
        })
        .map_err(to_py_err)
    }

    fn as_embeddings(&self) -> PyEmbeddings {
        PyEmbeddings {
            inner: Arc::new(Embeddings::new(self.inner.clone())),
        }
    }

    fn __repr__(&self) -> String {
        format!("NomicEmbeddings(model='{}')", self.inner.model)
    }
}
