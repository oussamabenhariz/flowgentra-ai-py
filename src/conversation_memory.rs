//! Python bindings for ConversationMemory

use pyo3::prelude::*;

use flowgentra_ai::core::memory::{ConversationMemory, InMemoryConversationMemory};

use crate::error::to_py_err;
use crate::llm::PyMessage;

// ─── PyConversationMemory ──────────────────────────────────────────────────

/// In-memory conversation memory with optional sliding window.
///
/// Stores message history per thread ID.
///
/// Example:
///     mem = ConversationMemory()
///     mem.add_message("thread-1", Message.user("Hello"))
///     mem.add_message("thread-1", Message.assistant("Hi!"))
///     messages = mem.messages("thread-1")
///     print(len(messages))  # 2
#[pyclass(name = "ConversationMemory")]
pub struct PyConversationMemory {
    inner: InMemoryConversationMemory,
}

#[pymethods]
impl PyConversationMemory {
    /// Create a new conversation memory with optional configuration.
    ///
    /// Args:
    ///     max_messages: Optional sliding window size (keep last N messages per thread)
    ///     system_prompt: Optional system prompt to include at the start of each conversation
    ///     initial_messages: Optional list of initial messages (e.g., examples, context)
    ///     summarize_threshold: Optional token threshold for auto-summarization
    ///
    /// Example:
    ///     memory = ConversationMemory(
    ///         max_messages=50,
    ///         system_prompt="You are a helpful assistant",
    ///         initial_messages=[Message.system("Context...")],
    ///         summarize_threshold=5000
    ///     )
    #[new]
    #[pyo3(signature = (max_messages=None, system_prompt=None, initial_messages=None, summarize_threshold=None))]
    fn new(
        max_messages: Option<usize>,
        system_prompt: Option<String>,
        initial_messages: Option<Vec<PyMessage>>,
        summarize_threshold: Option<usize>,
    ) -> Self {
        let mut mem = InMemoryConversationMemory::new();

        if let Some(max) = max_messages {
            mem = mem.with_max_messages(max);
        }

        if let Some(prompt) = system_prompt {
            mem = mem.with_system_prompt(prompt);
        }

        if let Some(msgs) = initial_messages {
            let rust_msgs: Vec<_> = msgs.into_iter().map(|m| m.inner).collect();
            mem = mem.with_initial_messages(rust_msgs);
        }

        if let Some(threshold) = summarize_threshold {
            mem = mem.with_summarize_threshold(threshold);
        }

        PyConversationMemory { inner: mem }
    }

    /// Add a message for a thread.
    fn add_message(&self, thread_id: &str, message: &PyMessage) -> PyResult<()> {
        self.inner
            .add_message(thread_id, message.inner.clone())
            .map_err(to_py_err)
    }

    /// Get messages for a thread (oldest first).
    ///
    /// Args:
    ///     thread_id: The thread identifier
    ///     limit: Optional max number of recent messages to return
    #[pyo3(signature = (thread_id, limit=None))]
    fn messages(&self, thread_id: &str, limit: Option<usize>) -> PyResult<Vec<PyMessage>> {
        let msgs = self.inner.messages(thread_id, limit).map_err(to_py_err)?;
        Ok(msgs.into_iter().map(|m| PyMessage { inner: m }).collect())
    }

    /// Clear all messages for a thread.
    fn clear(&self, thread_id: &str) -> PyResult<()> {
        self.inner.clear(thread_id).map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        "ConversationMemory(...)".to_string()
    }
}
