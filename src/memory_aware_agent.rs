//! Python bindings for MemoryAwareAgent

use pyo3::prelude::*;
use std::collections::HashMap;

use flowgentra_ai::core::agent::MemoryAwareAgent;
use flowgentra_ai::ArcHandler;
use flowgentra_ai::core::state::DynState;

use crate::agent::{scan_module_for_handlers, wrap_python_callable};
use crate::error::to_py_err;

// ─── PyMemoryStats ──────────────────────────────────────────────────────────

/// Statistics about memory usage in a MemoryAwareAgent.
#[pyclass(name = "MemoryStats")]
#[derive(Clone)]
pub struct PyMemoryStats {
    /// Total messages stored
    #[pyo3(get)]
    pub message_count: usize,
    /// User messages
    #[pyo3(get)]
    pub user_messages: usize,
    /// Assistant messages
    #[pyo3(get)]
    pub assistant_messages: usize,
    /// Approximate token count
    #[pyo3(get)]
    pub approximate_tokens: usize,
}

#[pymethods]
impl PyMemoryStats {
    fn __repr__(&self) -> String {
        format!(
            "MemoryStats(messages={}, user={}, assistant={}, ~tokens={})",
            self.message_count, self.user_messages, self.assistant_messages, self.approximate_tokens
        )
    }
}

// ─── PyMemoryAwareAgent ─────────────────────────────────────────────────────

/// Agent with automatic conversation memory management.
///
/// Example:
///     agent = MemoryAwareAgent.from_config("config.yaml")
///     agent.set_thread_id("user_123")
///     answer = agent.run_turn("What is Rust?")
///     answer2 = agent.run_turn("What are its features?")
#[pyclass(name = "MemoryAwareAgent")]
pub struct PyMemoryAwareAgent {
    inner: MemoryAwareAgent,
}

#[pymethods]
impl PyMemoryAwareAgent {
    /// Create from a YAML config file.
    ///
    /// Supports Python handlers via `python_handler_module` in config YAML +
    /// `@register_handler` decorator — same mechanism as `Agent.from_config_path()`.
    #[staticmethod]
    fn from_config(py: Python<'_>, config_path: &str) -> PyResult<Self> {
        // Parse YAML to find python_handler_module
        let yaml_content = std::fs::read_to_string(config_path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Cannot read config file '{}': {}",
                config_path, e
            ))
        })?;

        let yaml_val: serde_yml::Value = serde_yml::from_str(&yaml_content).map_err(|e| {
            crate::error::ConfigurationError::new_err(format!(
                "Invalid YAML in '{}': {}",
                config_path, e
            ))
        })?;

        // Collect Python handlers from python_handler_module
        let mut python_callables: HashMap<String, PyObject> = HashMap::new();
        if let Some(module_name) = yaml_val
            .get("python_handler_module")
            .and_then(|v| v.as_str())
        {
            let discovered = scan_module_for_handlers(py, module_name)?;
            python_callables.extend(discovered);
        }

        let agent = if python_callables.is_empty() {
            MemoryAwareAgent::from_config(config_path).map_err(to_py_err)?
        } else {
            let extra_handlers: HashMap<String, ArcHandler<DynState>> = python_callables
                .into_iter()
                .map(|(name, func)| (name, wrap_python_callable(func)))
                .collect();
            MemoryAwareAgent::from_config_with_extra_handlers(config_path, extra_handlers)
                .map_err(to_py_err)?
        };

        Ok(PyMemoryAwareAgent { inner: agent })
    }

    /// Set the thread ID for conversation memory.
    fn set_thread_id(&mut self, thread_id: &str) {
        self.inner.set_thread_id(thread_id);
    }

    /// Get current thread ID.
    fn thread_id(&self) -> String {
        self.inner.thread_id().to_string()
    }

    /// Run a single conversation turn with automatic memory.
    ///
    /// Args:
    ///     input: User input text
    ///
    /// Returns:
    ///     The agent's response string
    fn run_turn(&mut self, input: &str) -> PyResult<String> {
        let result = 
            crate::run_async(self.inner.run_turn(input))
            .map_err(to_py_err)?;
        Ok(result)
    }

    /// Clear conversation memory for the current thread.
    fn clear_memory(&self) -> PyResult<()> {
        self.inner.clear_memory().map_err(to_py_err)
    }

    /// Get memory usage statistics.
    fn memory_stats(&self) -> PyResult<PyMemoryStats> {
        let stats = self.inner.memory_stats().map_err(to_py_err)?;
        Ok(PyMemoryStats {
            message_count: stats.message_count,
            user_messages: stats.user_messages,
            assistant_messages: stats.assistant_messages,
            approximate_tokens: stats.approximate_tokens,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryAwareAgent(thread_id='{}')",
            self.inner.thread_id()
        )
    }
}
