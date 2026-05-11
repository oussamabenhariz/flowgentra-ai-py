//! Python bindings for LLMReranker

use pyo3::prelude::*;

use flowgentra_ai::core::rag::{LLMReranker, Reranker, SearchResult};

use crate::error::to_py_err_generic;
use crate::rag::PySearchResult;

// Type alias for the boxed scoring function
type ScoreFn = Box<
    dyn Fn(
            String,
            String,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<f32, String>> + Send>,
        > + Send
        + Sync,
>;

/// LLM-based reranker that scores query-document pairs.
///
/// Takes a Python scoring function that receives (query, document_text)
/// and returns a relevance score (float).
///
/// Example:
///     def scorer(query, doc_text):
///         # Call your LLM to score relevance
///         return 0.85
///
///     reranker = LLMReranker(scorer)
///     reranked = reranker.rerank("What is Rust?", results)
#[pyclass(name = "LLMReranker")]
pub struct PyLLMReranker {
    inner: LLMReranker<ScoreFn>,
}

#[pymethods]
impl PyLLMReranker {
    #[new]
    fn new(score_fn: PyObject) -> Self {
        let func = Python::with_gil(|py| score_fn.clone_ref(py));

        let rust_fn: ScoreFn = Box::new(move |query: String, doc_text: String| {
            let func = Python::with_gil(|py| func.clone_ref(py));
            Box::pin(async move {
                Python::with_gil(|py| -> Result<f32, String> {
                    let result = func
                        .call1(py, (query, doc_text))
                        .map_err(|e| format!("Score function error: {}", e))?;
                    let score: f32 = result
                        .extract(py)
                        .map_err(|e| format!("Score function must return float: {}", e))?;
                    Ok(score)
                })
            })
        });

        PyLLMReranker {
            inner: LLMReranker::new(rust_fn),
        }
    }

    /// Rerank search results using the LLM scoring function.
    fn rerank(&self, query: &str, results: Vec<PySearchResult>) -> PyResult<Vec<PySearchResult>> {
        let rs: Vec<SearchResult> = results.into_iter().map(|r| r.inner).collect();
        let out = 
            crate::run_async(self.inner.rerank(query, rs))
            .map_err(to_py_err_generic)?;
        Ok(out
            .into_iter()
            .map(|r| PySearchResult { inner: r })
            .collect())
    }

    fn __repr__(&self) -> String {
        "LLMReranker(...)".to_string()
    }
}
