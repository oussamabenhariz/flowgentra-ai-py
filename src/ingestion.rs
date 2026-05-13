//! Python bindings for IngestionPipeline

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::rag::{
    IngestionPipeline, IngestionStats, RAGConfig, VectorStore, VectorStoreBackend, VectorStoreType,
};

use crate::error::to_py_err_generic;
use crate::py_to_json;
use crate::vector_store::{PyEmbeddings, PyInMemoryVectorStore};

// ─── PyIngestionStats ──────────────────────────────────────────────────────

/// Statistics from an ingestion run.
///
/// Example:
///     stats = pipeline.ingest([("doc1", "Hello"), ("doc2", "World")])
///     print(stats.documents_processed, stats.chunks_indexed)
#[pyclass(name = "IngestionStats")]
pub struct PyIngestionStats {
    inner: IngestionStats,
}

#[pymethods]
impl PyIngestionStats {
    #[getter]
    fn documents_processed(&self) -> usize {
        self.inner.documents_processed
    }

    #[getter]
    fn chunks_indexed(&self) -> usize {
        self.inner.chunks_indexed
    }

    #[getter]
    fn errors(&self) -> Vec<String> {
        self.inner.errors.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "IngestionStats(docs={}, chunks={}, errors={})",
            self.inner.documents_processed,
            self.inner.chunks_indexed,
            self.inner.errors.len()
        )
    }
}

// ─── PyIngestionPipeline ───────────────────────────────────────────────────

/// Pipeline for ingesting documents into a vector store.
///
/// Example:
///     store = InMemoryVectorStore()
///     emb = Embeddings.mock(128)
///     pipeline = IngestionPipeline(store, emb, batch_size=10)
///     stats = pipeline.ingest([("doc1", "Hello world")])
#[pyclass(name = "IngestionPipeline")]
pub struct PyIngestionPipeline {
    inner: IngestionPipeline,
}

#[pymethods]
impl PyIngestionPipeline {
    #[new]
    #[pyo3(signature = (store, embeddings, batch_size=10))]
    fn new(store: &PyInMemoryVectorStore, embeddings: &PyEmbeddings, batch_size: usize) -> Self {
        let config = RAGConfig {
            store_type: VectorStoreType::Memory,
            api_key: None,
            endpoint: None,
            index_name: String::new(),
            embedding_dim: embeddings.inner.get_dimension(),
        };
        let vs = Arc::new(VectorStore::new(
            store.inner.clone() as Arc<dyn VectorStoreBackend>,
            config,
        ));
        PyIngestionPipeline {
            inner: IngestionPipeline::new(vs, embeddings.inner.clone(), batch_size),
        }
    }

    /// Ingest a list of (id, text) tuples.
    fn ingest(&self, documents: Vec<(String, String)>) -> PyResult<PyIngestionStats> {
        let stats = crate::run_async(self.inner.ingest(documents)).map_err(to_py_err_generic)?;
        Ok(PyIngestionStats { inner: stats })
    }

    /// Ingest a list of (id, text, metadata_dict) tuples.
    fn ingest_with_metadata(
        &self,
        documents: Vec<(String, String, PyObject)>,
    ) -> PyResult<PyIngestionStats> {
        let mut docs = Vec::with_capacity(documents.len());
        Python::with_gil(|py| -> PyResult<()> {
            for (id, text, meta) in &documents {
                let bound = meta.bind(py);
                let json_val = py_to_json(bound)?;
                docs.push((id.clone(), text.clone(), json_val));
            }
            Ok(())
        })?;
        let stats =
            crate::run_async(self.inner.ingest_with_metadata(docs)).map_err(to_py_err_generic)?;
        Ok(PyIngestionStats { inner: stats })
    }

    fn __repr__(&self) -> String {
        "IngestionPipeline(...)".to_string()
    }
}
