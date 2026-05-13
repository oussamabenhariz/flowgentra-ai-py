//! Python bindings for LLM — create_llm(), chat(), chat_with_tools()

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::llm::{
    create_llm, model_pricing, CachedLLM, FallbackLLM, LLMConfig, LLMProvider, Message,
    MessageRole, ResponseFormat, RetryLLM, TokenUsage, ToolCall, ToolDefinition, LLM,
};

use crate::error::to_py_err;
use crate::{json_to_py, py_to_json};

// ─── PyTokenUsage ─────────────────────────────────────────────────────────────

#[pyclass(name = "TokenUsage")]
#[derive(Clone)]
pub struct PyTokenUsage {
    pub(crate) inner: TokenUsage,
}

#[pymethods]
impl PyTokenUsage {
    #[getter]
    fn get_prompt_tokens(&self) -> u64 {
        self.inner.prompt_tokens
    }
    #[getter]
    fn get_completion_tokens(&self) -> u64 {
        self.inner.completion_tokens
    }
    #[getter]
    fn get_total_tokens(&self) -> u64 {
        self.inner.total_tokens
    }

    fn estimated_cost(&self, model: &str) -> Option<f64> {
        self.inner.estimated_cost(model)
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenUsage(prompt={}, completion={}, total={})",
            self.inner.prompt_tokens, self.inner.completion_tokens, self.inner.total_tokens
        )
    }
}

// ─── PyToolCall ───────────────────────────────────────────────────────────────

#[pyclass(name = "ToolCall")]
#[derive(Clone)]
pub struct PyToolCall {
    pub(crate) inner: ToolCall,
}

#[pymethods]
impl PyToolCall {
    #[getter]
    fn get_id(&self) -> String {
        self.inner.id.clone()
    }
    #[getter]
    fn get_name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn get_arguments(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_to_py(py, &self.inner.arguments)
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolCall(id='{}', name='{}')",
            self.inner.id, self.inner.name
        )
    }
}

// ─── PyToolDefinition ─────────────────────────────────────────────────────────

#[pyclass(name = "ToolDefinition")]
#[derive(Clone)]
pub struct PyToolDefinition {
    pub(crate) inner: ToolDefinition,
}

#[pymethods]
impl PyToolDefinition {
    #[new]
    fn new(name: String, description: String, parameters: &Bound<'_, PyAny>) -> PyResult<Self> {
        let params = py_to_json(parameters)?;
        Ok(PyToolDefinition {
            inner: ToolDefinition::new(name, description, params),
        })
    }

    #[getter]
    fn get_name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn get_description(&self) -> String {
        self.inner.description.clone()
    }
    #[getter]
    fn get_parameters(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_to_py(py, &self.inner.parameters)
    }

    fn __repr__(&self) -> String {
        format!("ToolDefinition(name='{}')", self.inner.name)
    }
}

// ─── PyMessage ────────────────────────────────────────────────────────────────

#[pyclass(name = "Message")]
#[derive(Clone)]
pub struct PyMessage {
    pub(crate) inner: Message,
}

#[pymethods]
impl PyMessage {
    #[staticmethod]
    fn user(content: String) -> Self {
        PyMessage {
            inner: Message::user(content),
        }
    }
    #[staticmethod]
    fn system(content: String) -> Self {
        PyMessage {
            inner: Message::system(content),
        }
    }
    #[staticmethod]
    fn assistant(content: String) -> Self {
        PyMessage {
            inner: Message::assistant(content),
        }
    }
    #[staticmethod]
    fn tool(content: String) -> Self {
        PyMessage {
            inner: Message::tool(content),
        }
    }
    #[staticmethod]
    fn tool_result(tool_call_id: String, content: String) -> Self {
        PyMessage {
            inner: Message::tool_result(tool_call_id, content),
        }
    }

    #[getter]
    fn get_role(&self) -> &str {
        match self.inner.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        }
    }
    #[getter]
    fn get_content(&self) -> String {
        self.inner.content.clone()
    }
    #[getter]
    fn get_tool_calls(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner.tool_calls {
            None => Ok(py.None()),
            Some(calls) => {
                let list = pyo3::types::PyList::empty_bound(py);
                for c in calls {
                    let py_call = PyToolCall { inner: c.clone() };
                    list.append(py_call.into_py(py))?;
                }
                Ok(list.into())
            }
        }
    }
    #[getter]
    fn get_tool_call_id(&self) -> Option<String> {
        self.inner.tool_call_id.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Message(role='{}', content='{}')",
            self.get_role(),
            &self.inner.content
        )
    }
}

// ─── PyResponseFormat ─────────────────────────────────────────────────────────

#[pyclass(name = "ResponseFormat")]
#[derive(Clone)]
pub struct PyResponseFormat {
    pub(crate) inner: ResponseFormat,
}

#[pymethods]
impl PyResponseFormat {
    #[staticmethod]
    fn text() -> Self {
        PyResponseFormat {
            inner: ResponseFormat::Text,
        }
    }
    #[staticmethod]
    fn json() -> Self {
        PyResponseFormat {
            inner: ResponseFormat::Json,
        }
    }
    #[staticmethod]
    fn json_schema(name: String, schema: &Bound<'_, PyAny>) -> PyResult<Self> {
        let schema_val = py_to_json(schema)?;
        Ok(PyResponseFormat {
            inner: ResponseFormat::JsonSchema {
                name,
                schema: schema_val,
            },
        })
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ResponseFormat::Text => "ResponseFormat.text()".to_string(),
            ResponseFormat::Json => "ResponseFormat.json()".to_string(),
            ResponseFormat::JsonSchema { name, .. } => {
                format!("ResponseFormat.json_schema(name='{}')", name)
            }
        }
    }
}

// ─── PyLLMConfig ──────────────────────────────────────────────────────────────

#[pyclass(name = "LLMConfig")]
#[derive(Clone)]
pub struct PyLLMConfig {
    pub(crate) inner: LLMConfig,
}

#[pymethods]
impl PyLLMConfig {
    #[new]
    #[pyo3(signature = (provider, model, api_key=String::new(), temperature=None, max_tokens=None, top_p=None))]
    fn new(
        provider: &str,
        model: String,
        api_key: String,
        temperature: Option<f32>,
        max_tokens: Option<usize>,
        top_p: Option<f32>,
    ) -> Self {
        let llm_provider = match provider.to_lowercase().as_str() {
            "openai" => LLMProvider::OpenAI,
            "anthropic" => LLMProvider::Anthropic,
            "mistral" => LLMProvider::Mistral,
            "groq" => LLMProvider::Groq,
            "huggingface" => LLMProvider::HuggingFace,
            "ollama" => LLMProvider::Ollama,
            "azure" => LLMProvider::Azure,
            other => LLMProvider::Custom(other.to_string()),
        };
        let mut config = LLMConfig::new(llm_provider, model, api_key);
        config.temperature = temperature;
        config.max_tokens = max_tokens;
        config.top_p = top_p;
        PyLLMConfig { inner: config }
    }

    #[getter]
    fn get_model(&self) -> String {
        self.inner.model.clone()
    }
    #[getter]
    fn get_temperature(&self) -> Option<f32> {
        self.inner.temperature
    }
    #[getter]
    fn get_max_tokens(&self) -> Option<usize> {
        self.inner.max_tokens
    }

    fn with_response_format(&self, fmt: &PyResponseFormat) -> Self {
        PyLLMConfig {
            inner: self.inner.clone().with_response_format(fmt.inner.clone()),
        }
    }

    /// Add a provider-specific extra parameter.
    ///
    /// Example:
    ///     config = LLMConfig("huggingface", "mistralai/Mistral-7B", api_key=...)
    ///     config = config.with_extra_param("mode", "tgi").with_extra_param("endpoint", "http://localhost:8080")
    fn with_extra_param(&self, key: String, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let val = crate::py_to_json(value)?;
        Ok(PyLLMConfig {
            inner: self.inner.clone().with_extra_param(key, val),
        })
    }

    fn __repr__(&self) -> String {
        format!("LLMConfig(model='{}')", self.inner.model)
    }
}

// ─── py_model_pricing ────────────────────────────────────────────────────────

#[pyfunction]
pub fn py_model_pricing(model: &str) -> Option<(f64, f64)> {
    model_pricing(model)
}

// ─── PyLLM ────────────────────────────────────────────────────────────────────

/// An LLM for sending messages and receiving responses.
///
/// Create from an LLMConfig:
///     config = LLMConfig("openai", "gpt-4", api_key="sk-...")
///     client = LLM.from_config(config)
///     response = client.chat([Message.user("Hello!")])
///     print(response.content)
#[pyclass(name = "LLM")]
pub struct PyLLM {
    pub(crate) inner: Arc<dyn LLM>,
}

#[pymethods]
impl PyLLM {
    /// Create an LLM directly with provider and model parameters.
    ///
    /// If ``api_key`` is not provided the constructor resolves it automatically:
    ///
    /// 1. Checks the provider's conventional environment variable
    ///    (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``MISTRAL_API_KEY``,
    ///    ``GROQ_API_KEY``, ``HUGGINGFACEHUB_API_TOKEN``, ``AZURE_OPENAI_KEY``).
    /// 2. If not found, loads a ``.env`` file from the working directory and retries.
    ///
    /// A ``ValueError`` is raised only when the key is still missing after both steps.
    ///
    /// Args:
    ///     provider: Provider name ("openai", "anthropic", "mistral", "groq", "ollama", "huggingface", "azure")
    ///     model: Model identifier (e.g. "gpt-4", "claude-3-opus-20240229")
    ///     api_key: API key for the provider (optional — falls back to env var / .env)
    ///     temperature: Response randomness 0.0-2.0 (optional, default: 0.7)
    ///     max_tokens: Maximum response tokens (optional)
    ///     top_p: Nucleus sampling parameter 0.0-1.0 (optional)
    ///
    /// Example:
    ///     client = LLM(provider="anthropic", model="claude-opus-4-6")
    ///     response = client.chat([Message.user("Hello!")])
    #[new]
    #[pyo3(signature = (provider, model, api_key=None, temperature=None, max_tokens=None, top_p=None))]
    fn new(
        provider: &str,
        model: String,
        api_key: Option<String>,
        temperature: Option<f32>,
        max_tokens: Option<usize>,
        top_p: Option<f32>,
    ) -> PyResult<Self> {
        let llm_provider = match provider.to_lowercase().as_str() {
            "openai" => LLMProvider::OpenAI,
            "anthropic" => LLMProvider::Anthropic,
            "mistral" => LLMProvider::Mistral,
            "groq" => LLMProvider::Groq,
            "huggingface" => LLMProvider::HuggingFace,
            "ollama" => LLMProvider::Ollama,
            "azure" => LLMProvider::Azure,
            other => LLMProvider::Custom(other.to_string()),
        };

        // Resolve key: explicit arg → env var → .env file → error
        let key = match api_key {
            Some(k) if !k.is_empty() => k,
            _ => match llm_provider.env_var() {
                None => String::new(), // Ollama / custom — no key needed
                Some(var) => {
                    // 1. Process environment
                    std::env::var(var).unwrap_or_else(|_| {
                        // 2. .env file in the working directory
                        let _ = dotenv::dotenv();
                        std::env::var(var).unwrap_or_default()
                    })
                }
            },
        };

        let requires_key = !matches!(llm_provider, LLMProvider::Ollama | LLMProvider::Custom(_));
        if requires_key && key.is_empty() {
            let var_name = llm_provider.env_var().unwrap_or("API_KEY");
            return Err(crate::error::ConfigurationError::new_err(format!(
                "No API key found for provider '{}'. \
                 Pass api_key=... or set the {} environment variable (or add it to a .env file).",
                provider, var_name
            )));
        }

        let mut config = LLMConfig::new(llm_provider, model, key);
        config.temperature = temperature;
        config.max_tokens = max_tokens;
        config.top_p = top_p;
        let client = create_llm(&config).map_err(to_py_err)?;
        Ok(PyLLM { inner: client })
    }

    /// Create an LLM from config.
    ///
    /// This is useful if you want to create and reuse the same config.
    ///
    /// Example:
    ///     config = LLMConfig("openai", "gpt-4", api_key="sk-...")
    ///     client = LLM.from_config(config)
    #[staticmethod]
    fn from_config(config: &PyLLMConfig) -> PyResult<Self> {
        let client = create_llm(&config.inner).map_err(to_py_err)?;
        Ok(PyLLM { inner: client })
    }

    /// Send messages and get a response.
    ///
    /// Args:
    ///     messages: List of Message objects
    ///
    /// Returns:
    ///     A Message with the LLM's response
    fn chat(&self, messages: Vec<PyMessage>) -> PyResult<PyMessage> {
        let msgs: Vec<Message> = messages.into_iter().map(|m| m.inner).collect();
        let result = crate::run_async(self.inner.chat(msgs)).map_err(to_py_err)?;
        Ok(PyMessage { inner: result })
    }

    /// Send messages and get a response with token usage stats.
    ///
    /// Returns:
    ///     Tuple of (Message, optional TokenUsage)
    fn chat_with_usage(
        &self,
        py: Python<'_>,
        messages: Vec<PyMessage>,
    ) -> PyResult<(PyMessage, PyObject)> {
        let msgs: Vec<Message> = messages.into_iter().map(|m| m.inner).collect();

        let (msg, usage) = crate::run_async(self.inner.chat_with_usage(msgs)).map_err(to_py_err)?;

        let py_usage = match usage {
            Some(u) => PyTokenUsage { inner: u }.into_py(py),
            None => py.None(),
        };
        Ok((PyMessage { inner: msg }, py_usage))
    }

    /// Send messages with tool definitions and get a response (function calling).
    ///
    /// Args:
    ///     messages: List of Message objects
    ///     tools: List of ToolDefinition objects
    ///
    /// Returns:
    ///     A Message (may contain tool_calls)
    fn chat_with_tools(
        &self,
        messages: Vec<PyMessage>,
        tools: Vec<PyToolDefinition>,
    ) -> PyResult<PyMessage> {
        let msgs: Vec<Message> = messages.into_iter().map(|m| m.inner).collect();
        let tool_defs: Vec<ToolDefinition> = tools.into_iter().map(|t| t.inner).collect();

        let result =
            crate::run_async(self.inner.chat_with_tools(msgs, &tool_defs)).map_err(to_py_err)?;

        Ok(PyMessage { inner: result })
    }

    /// Stream the LLM response chunk by chunk. Returns an iterable LLMStream.
    ///
    /// Example:
    ///     stream = client.chat_stream([Message.user("Tell me a long story")])
    ///     for chunk in stream:
    ///         print(chunk, end="", flush=True)
    fn chat_stream(&self, messages: Vec<PyMessage>) -> PyResult<PyLLMStream> {
        let msgs: Vec<Message> = messages.into_iter().map(|m| m.inner).collect();
        let rx = crate::run_async(self.inner.chat_stream(msgs)).map_err(to_py_err)?;
        Ok(PyLLMStream { inner: Some(rx) })
    }

    /// Send messages and get a structured JSON response.
    ///
    /// Returns:
    ///     The parsed response as a Python dict (or list/scalar).
    fn chat_structured(&self, messages: Vec<PyMessage>) -> PyResult<PyObject> {
        let msgs: Vec<Message> = messages.into_iter().map(|m| m.inner).collect();
        let val = crate::run_async(self.inner.chat_structured(msgs)).map_err(to_py_err)?;
        Python::with_gil(|py| crate::json_to_py(py, &val))
    }

    /// Wrap this client with a response cache.
    ///
    /// Example:
    ///     cached = client.cached(max_entries=1000)
    #[pyo3(signature = (max_entries=100))]
    fn cached(&self, max_entries: usize) -> Self {
        let cached = CachedLLM::new(self.inner.clone()).with_max_entries(max_entries);
        PyLLM {
            inner: Arc::new(cached),
        }
    }

    /// Create a fallback client that tries this client first, then falls back.
    ///
    /// Example:
    ///     robust = primary.with_fallback(secondary)
    fn with_fallback(&self, fallback: &PyLLM) -> Self {
        let fb = FallbackLLM::new(self.inner.clone()).with_fallback(fallback.inner.clone());
        PyLLM {
            inner: Arc::new(fb),
        }
    }

    /// Wrap this client with automatic retry and exponential backoff.
    ///
    /// Example:
    ///     retrying = client.with_retry(max_retries=3)
    #[pyo3(signature = (max_retries=3))]
    fn with_retry(&self, max_retries: u32) -> Self {
        let retry = RetryLLM::new(self.inner.clone(), max_retries);
        PyLLM {
            inner: Arc::new(retry),
        }
    }

    fn __repr__(&self) -> String {
        "LLM(...)".to_string()
    }
}

// ─── PyLLMStream ─────────────────────────────────────────────────────────────

/// Iterable stream of LLM response chunks.
///
/// Returned by LLM.chat_stream() — iterate to receive tokens one by one.
///
/// Example:
///     stream = client.chat_stream([Message.user("Tell me a story")])
///     for chunk in stream:
///         print(chunk, end="", flush=True)
///     print()
#[pyclass(name = "LLMStream")]
pub struct PyLLMStream {
    inner: Option<tokio::sync::mpsc::Receiver<String>>,
}

#[pymethods]
impl PyLLMStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<String> {
        let rx = self.inner.as_mut()?;
        let chunk = match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                use tokio::runtime::RuntimeFlavor;
                match handle.runtime_flavor() {
                    RuntimeFlavor::CurrentThread => crate::get_runtime().block_on(rx.recv()),
                    _ => tokio::task::block_in_place(|| handle.block_on(rx.recv())),
                }
            }
            Err(_) => crate::get_runtime().block_on(rx.recv()),
        };
        if chunk.is_none() {
            self.inner = None;
        }
        chunk
    }

    fn __repr__(&self) -> &'static str {
        "LLMStream()"
    }
}

// ─── Standalone create_llm function ────────────────────────────────────────

/// Create an LLM from a config object.
///
/// This mirrors the Rust ``create_llm()`` function and is the
/// recommended factory when you already have an ``LLMConfig``.
///
/// Example::
///
///     from flowgentra_ai.llm import LLMConfig, create_llm
///
///     config = LLMConfig(model="gpt-4", provider="openai", api_key="sk-...")
///     client = create_llm(config)
///     response = client.chat([Message.user("Hello!")])
#[pyfunction]
pub fn py_create_llm(config: &PyLLMConfig) -> PyResult<PyLLM> {
    let client = create_llm(&config.inner).map_err(to_py_err)?;
    Ok(PyLLM { inner: client })
}
