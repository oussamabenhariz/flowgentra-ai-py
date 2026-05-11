//! Python bindings for TokenBufferMemory and SummaryMemory

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::memory::{
    BoxedSummarizeFn, ConversationMemory, SummaryConfig, SummaryMemory, TokenBufferMemory,
};

use crate::error::to_py_err;
use crate::llm::{PyLLM, PyMessage};

// ─── PyTokenBufferMemory ──────────────────────────────────────────────────────

/// Token-aware conversation memory that trims to a token budget.
///
/// Example:
///     mem = TokenBufferMemory(max_tokens=4000)
///     mem.add_message("thread-1", Message.user("Hello"))
///     messages = mem.messages("thread-1")
///     print(mem.token_count("thread-1"), "/", mem.max_tokens())
#[pyclass(name = "TokenBufferMemory")]
pub struct PyTokenBufferMemory {
    inner: TokenBufferMemory,
}

#[pymethods]
impl PyTokenBufferMemory {
    #[new]
    #[pyo3(signature = (max_tokens=4000))]
    fn new(max_tokens: usize) -> Self {
        PyTokenBufferMemory {
            inner: TokenBufferMemory::new(max_tokens),
        }
    }

    /// Add a message to a thread.
    fn add_message(&self, thread_id: &str, message: &PyMessage) -> PyResult<()> {
        self.inner
            .add_message(thread_id, message.inner.clone())
            .map_err(to_py_err)
    }

    /// Get messages for a thread.
    #[pyo3(signature = (thread_id, limit=None))]
    fn messages(&self, thread_id: &str, limit: Option<usize>) -> PyResult<Vec<PyMessage>> {
        let msgs = self.inner.messages(thread_id, limit).map_err(to_py_err)?;
        Ok(msgs.into_iter().map(|m| PyMessage { inner: m }).collect())
    }

    /// Clear all messages for a thread.
    fn clear(&self, thread_id: &str) -> PyResult<()> {
        self.inner.clear(thread_id).map_err(to_py_err)
    }

    /// Current estimated token count for a thread.
    fn token_count(&self, thread_id: &str) -> PyResult<usize> {
        self.inner.token_count(thread_id).map_err(to_py_err)
    }

    /// The configured token budget.
    fn max_tokens(&self) -> usize {
        self.inner.max_tokens()
    }

    fn __repr__(&self) -> String {
        format!("TokenBufferMemory(max_tokens={})", self.inner.max_tokens())
    }
}

// ─── PySummaryConfig ──────────────────────────────────────────────────────────

/// Configuration for SummaryMemory.
///
/// Example:
///     from flowgentra_ai.llm import LLM
///     config = SummaryConfig(llm=LLM("mistral", "mistral-small-latest"), summary_threshold=6)
///     config = SummaryConfig(buffer_size=10, max_summary_tokens=200)
#[pyclass(name = "SummaryConfig")]
#[derive(Clone)]
pub struct PySummaryConfig {
    pub(crate) inner: SummaryConfig,
    /// Optional LLM for auto-summarization (Arc is Clone without needing the GIL)
    pub(crate) llm: Option<Arc<dyn flowgentra_ai::core::llm::LLM>>,
    pub(crate) summary_threshold: usize,
}

#[pymethods]
impl PySummaryConfig {
    #[new]
    #[pyo3(signature = (buffer_size=None, max_summary_tokens=200, llm=None, summary_threshold=None))]
    fn new(
        buffer_size: Option<usize>,
        max_summary_tokens: usize,
        llm: Option<&PyLLM>,
        summary_threshold: Option<usize>,
    ) -> Self {
        let effective_buffer = buffer_size.or(summary_threshold).unwrap_or(10);
        PySummaryConfig {
            inner: SummaryConfig {
                buffer_size: effective_buffer,
                max_summary_tokens,
            },
            llm: llm.map(|l| l.inner.clone()),
            summary_threshold: effective_buffer,
        }
    }

    #[getter]
    fn buffer_size(&self) -> usize {
        self.inner.buffer_size
    }

    #[getter]
    fn max_summary_tokens(&self) -> usize {
        self.inner.max_summary_tokens
    }

    #[getter]
    fn summary_threshold(&self) -> usize {
        self.summary_threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "SummaryConfig(buffer_size={}, max_summary_tokens={})",
            self.inner.buffer_size, self.inner.max_summary_tokens
        )
    }
}

// ─── PySummaryMemory ──────────────────────────────────────────────────────────

/// Conversation memory with automatic summarization.
///
/// When the buffer exceeds `summary_threshold`, older messages are summarized
/// by the provided LLM. Call `summarize_if_needed(thread_id)` to trigger it.
///
/// Example (LLM-based):
///     from flowgentra_ai.llm import LLM
///     mem = SummaryMemory(llm=LLM("mistral", "mistral-small-latest"), summary_threshold=6)
///
/// Example (via SummaryConfig):
///     cfg = SummaryConfig(llm=LLM(...), summary_threshold=6)
///     mem = SummaryMemory(config=cfg)
///
/// Example (custom Python function):
///     mem = SummaryMemory(summarize_fn=lambda text: text[:200])
#[pyclass(name = "SummaryMemory")]
pub struct PySummaryMemory {
    inner: SummaryMemory<BoxedSummarizeFn>,
}

#[pymethods]
impl PySummaryMemory {
    #[new]
    #[pyo3(signature = (config=None, summarize_fn=None, llm=None, summary_threshold=None))]
    fn new(
        py: Python,
        config: Option<&PySummaryConfig>,
        summarize_fn: Option<PyObject>,
        llm: Option<&PyLLM>,
        summary_threshold: Option<usize>,
    ) -> PyResult<Self> {
        let rust_config = config.map(|c| c.inner.clone()).unwrap_or_else(|| SummaryConfig {
            buffer_size: summary_threshold.unwrap_or(10),
            max_summary_tokens: 200,
        });

        // Priority: explicit llm arg > explicit summarize_fn > llm from config > no-op
        let effective_llm: Option<Arc<dyn flowgentra_ai::core::llm::LLM>> = llm
            .map(|l| l.inner.clone())
            .or_else(|| config.and_then(|c| c.llm.clone()));

        let inner = if let Some(llm_arc) = effective_llm {
            // Use Rust LLM directly — no Python GIL dance needed
            SummaryMemory::with_llm(rust_config, llm_arc)
        } else if let Some(fn_obj) = summarize_fn {
            // User-provided Python callable
            let func = fn_obj.clone_ref(py);
            let fn_box: BoxedSummarizeFn = Box::new(move |text: String| {
                let func = Python::with_gil(|py| func.clone_ref(py));
                Box::pin(async move {
                    Python::with_gil(
                        |py| -> Result<String, flowgentra_ai::core::error::FlowgentraError> {
                            let res = func.call1(py, (text,)).map_err(|e| {
                                flowgentra_ai::core::error::FlowgentraError::ToolError(
                                    format!("summarize_fn error: {}", e),
                                )
                            })?;
                            res.extract::<String>(py).map_err(|e| {
                                flowgentra_ai::core::error::FlowgentraError::ToolError(
                                    format!("summarize_fn must return str: {}", e),
                                )
                            })
                        },
                    )
                })
            });
            SummaryMemory::new(rust_config, fn_box)
        } else {
            // No summarizer: messages kept verbatim, summarize_if_needed is a no-op
            let fn_box: BoxedSummarizeFn = Box::new(|_text: String| {
                Box::pin(async {
                    Ok::<String, flowgentra_ai::core::error::FlowgentraError>(String::new())
                })
            });
            SummaryMemory::new(rust_config, fn_box)
        };

        Ok(PySummaryMemory { inner })
    }

    /// Add a message to a thread.
    fn add_message(&self, thread_id: &str, message: &PyMessage) -> PyResult<()> {
        self.inner
            .add_message(thread_id, message.inner.clone())
            .map_err(to_py_err)
    }

    /// Get messages for a thread.
    #[pyo3(signature = (thread_id, limit=None))]
    fn messages(&self, thread_id: &str, limit: Option<usize>) -> PyResult<Vec<PyMessage>> {
        let msgs = self.inner.messages(thread_id, limit).map_err(to_py_err)?;
        Ok(msgs.into_iter().map(|m| PyMessage { inner: m }).collect())
    }

    /// Clear all messages for a thread.
    fn clear(&self, thread_id: &str) -> PyResult<()> {
        self.inner.clear(thread_id).map_err(to_py_err)
    }

    /// Get the current summary for a thread (if any).
    fn get_summary(&self, thread_id: &str) -> Option<String> {
        self.inner.get_summary(thread_id)
    }

    /// Trigger summarization if the buffer exceeds the configured size.
    fn summarize_if_needed(&self, thread_id: &str) -> PyResult<()> {
        crate::run_async(self.inner.summarize_if_needed(thread_id)).map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        "SummaryMemory(...)".to_string()
    }
}
