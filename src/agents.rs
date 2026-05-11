//! Python bindings for prebuilt agents (AgentType, GraphBasedAgent,
//! and typed constructors ZeroShotReAct / FewShotReAct / Conversational / …)
//!
//! ## Unified tool interface
//!
//! All typed agent constructors now accept tools the same way `LLM.chat_with_tools` does:
//!
//! ```python
//! from flowgentra_ai.llm import LLMClient, ToolDefinition
//! from flowgentra_ai.agent import ZeroShotReAct
//! from flowgentra_ai.tools import ToolRegistry, tool
//!
//! # --- Built-in tool via ToolDefinition (same type as chat_with_tools) ---
//! calc = ToolDefinition("calculator", "Perform arithmetic", {
//!     "type": "object",
//!     "properties": {
//!         "operation": {"type": "string"},
//!         "a": {"type": "number"},
//!         "b": {"type": "number"},
//!     },
//!     "required": ["operation", "a", "b"],
//! })
//!
//! # --- Custom Python tool ---
//! @tool("greet", "Greet a person")
//! def greet(name: str) -> dict:
//!     return {"greeting": f"Hello, {name}!"}
//!
//! registry = ToolRegistry.with_builtins()
//! registry.register(greet)
//!
//! agent = ZeroShotReAct(
//!     name="my-agent",
//!     llm=llm,
//!     tools=[calc],            # ToolDefinition or ToolSpec — both accepted
//!     tool_registry=registry,  # executes tools; supports custom @tool functions
//! )
//! ```
//!
//! `ToolSpec` is still accepted in `tools=` for backward compatibility.
//! When `tool_registry` is omitted the agent falls back to the built-in Rust
//! `ToolRegistry` (built-in tools only, no custom Python callables).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use flowgentra_ai::core::agents::{
    AgentConfig, AgentType, Conversational as RsConversational, FewShotReAct as RsFewShotReAct,
    GraphBasedAgent, ReactDocstore as RsReactDocstore, SelfAskWithSearch as RsSelfAskWithSearch,
    StructuredChat as RsStructuredChat, ToolCalling as RsToolCalling, ToolSpec,
    ZeroShotReAct as RsZeroShotReAct,
};
use flowgentra_ai::core::llm::ToolDefinition;
use flowgentra_ai::core::tools::ToolRegistry;

use crate::error::to_py_err;
use crate::llm::{PyLLM, PyToolDefinition};
use crate::tool_registry::PyToolRegistry;

// ─── PyAgentType ────────────────────────────────────────────────────────────

/// Agent type: "zero_shot_react", "few_shot_react", or "conversational".
#[pyclass(name = "AgentType")]
#[derive(Clone)]
pub struct PyAgentType {
    pub(crate) inner: AgentType,
}

#[pymethods]
impl PyAgentType {
    /// Zero-shot ReAct agent (reasoning + action without examples).
    #[staticmethod]
    fn zero_shot_react() -> Self {
        PyAgentType { inner: AgentType::ZeroShotReAct }
    }

    /// Few-shot ReAct agent (with example demonstrations).
    #[staticmethod]
    fn few_shot_react() -> Self {
        PyAgentType { inner: AgentType::FewShotReAct }
    }

    /// Conversational agent (multi-turn dialogue with memory).
    #[staticmethod]
    fn conversational() -> Self {
        PyAgentType { inner: AgentType::Conversational }
    }

    /// Tool Calling agent — uses the provider's native function/tool-calling API.
    #[staticmethod]
    fn tool_calling() -> Self {
        PyAgentType { inner: AgentType::ToolCalling }
    }

    /// Structured Chat Zero-Shot ReAct agent — ReAct with JSON-blob actions.
    #[staticmethod]
    fn structured_chat_zero_shot_react() -> Self {
        PyAgentType {
            inner: AgentType::StructuredChatZeroShotReAct,
        }
    }

    /// Self Ask With Search agent — decomposes questions into sub-questions.
    #[staticmethod]
    fn self_ask_with_search() -> Self {
        PyAgentType { inner: AgentType::SelfAskWithSearch }
    }

    /// ReAct Docstore agent — Search + Lookup loop over a document store.
    #[staticmethod]
    fn react_docstore() -> Self {
        PyAgentType { inner: AgentType::ReactDocstore }
    }

    fn __repr__(&self) -> String {
        format!("AgentType({})", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

// ─── PyToolSpec ─────────────────────────────────────────────────────────────

/// Tool specification for prebuilt agents.
///
/// **Prefer `ToolDefinition` for new code** — it is the same type accepted by
/// `LLM.chat_with_tools` and provides a full JSON Schema for parameters.
/// `ToolSpec` is kept for backward compatibility.
///
/// Example:
///     tool = ToolSpec("search", "Search the web")
///     tool.add_parameter("query", "string")
///     tool.set_required("query")
#[pyclass(name = "ToolSpec")]
#[derive(Clone)]
pub struct PyToolSpec {
    pub(crate) inner: ToolSpec,
}

#[pymethods]
impl PyToolSpec {
    #[new]
    fn new(name: &str, description: &str) -> Self {
        PyToolSpec {
            inner: ToolSpec::new(name, description),
        }
    }

    /// Add a parameter with its type.
    fn add_parameter(&mut self, name: &str, param_type: &str) {
        self.inner.parameters.insert(name.to_string(), param_type.to_string());
    }

    /// Mark a parameter as required.
    fn set_required(&mut self, param: &str) {
        self.inner.required.push(param.to_string());
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn description(&self) -> String {
        self.inner.description.clone()
    }

    fn __repr__(&self) -> String {
        format!("ToolSpec(name='{}', desc='{}')", self.inner.name, self.inner.description)
    }
}

// ─── PyGraphBasedAgent ──────────────────────────────────────────────────────

/// A graph-based prebuilt agent (ReAct, Conversational, etc.).
///
/// Execute with `execute_input()` for simple string in/out.
#[pyclass(name = "GraphBasedAgent")]
pub struct PyGraphBasedAgent {
    inner: GraphBasedAgent,
}

#[pymethods]
impl PyGraphBasedAgent {
    /// Execute the agent with a text input and get a text response.
    fn execute_input(&self, input: &str) -> PyResult<String> {
        crate::run_async(self.inner.execute_input(input)).map_err(to_py_err)
    }

    /// Get the agent name.
    #[getter]
    fn name(&self) -> String {
        self.inner.config().name.clone()
    }

    /// Get node names of the underlying graph.
    fn node_names(&self) -> Vec<String> {
        self.inner.graph().node_names()
    }

    fn __repr__(&self) -> String {
        format!("GraphBasedAgent(name='{}')", self.inner.config().name)
    }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Extract a `ToolDefinition` from a Python object that is either a
/// `PyToolDefinition` or a `PyToolSpec` (for backward compat).
fn extract_tool_definition(obj: &Bound<'_, PyAny>) -> PyResult<ToolDefinition> {
    if let Ok(td) = obj.extract::<PyRef<PyToolDefinition>>() {
        return Ok(td.inner.clone());
    }
    if let Ok(ts) = obj.extract::<PyRef<PyToolSpec>>() {
        return Ok(ts.inner.clone().into());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "tools= items must be ToolDefinition or ToolSpec",
    ))
}

/// Build a `ToolExecutorFn` that dispatches tool calls through a Python
/// `ToolRegistry` object (supports both built-in Rust tools and custom
/// `@tool`-decorated Python callables).
///
/// Falls back to the built-in Rust `ToolRegistry` when no Python registry is
/// provided so that built-in tools (calculator, file, duckduckgo_search, …)
/// work without any extra setup.
fn build_executor(py_registry: Option<PyObject>) -> flowgentra_ai::core::agents::graph_nodes::ToolExecutorFn {
    match py_registry {
        Some(registry_obj) => {
            // The Python-level ToolRegistry handles both Rust built-ins and
            // @tool-decorated custom callables. We call it via the GIL.
            let registry_arc = Arc::new(registry_obj);
            Arc::new(move |name: &str, args: &str| {
                let registry = registry_arc.clone();
                let name_owned = name.to_string();
                let args_owned = args.to_string();
                Python::with_gil(|py| {
                    // Parse args JSON string into a Python dict via json.loads
                    let json_mod = match py.import_bound("json") {
                        Ok(m) => m,
                        Err(e) => return format!("Tool error (import json): {}", e),
                    };
                    let py_args = json_mod
                        .call_method1("loads", (&args_owned,))
                        .unwrap_or_else(|_| PyDict::new_bound(py).into_any());

                    match registry.call_method1(py, "call_tool", (&name_owned, &py_args)) {
                        Ok(result) => {
                            // Serialize result back to JSON string
                            match json_mod.call_method1("dumps", (&result,)) {
                                Ok(json_str) => json_str
                                    .extract::<String>()
                                    .unwrap_or_else(|_| result.to_string()),
                                Err(_) => result
                                    .extract::<String>(py)
                                    .unwrap_or_else(|_| result.to_string()),
                            }
                        }
                        Err(e) => format!("Tool error: {}", e),
                    }
                })
            })
        }
        None => {
            // No Python registry — fall back to built-in Rust ToolRegistry.
            let registry = Arc::new(ToolRegistry::with_builtins());
            Arc::new(move |name: &str, args: &str| {
                let registry = registry.clone();
                let name_owned = name.to_string();
                let args_val: serde_json::Value =
                    serde_json::from_str(args).unwrap_or(serde_json::json!({}));
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        match registry.call_tool(&name_owned, args_val).await {
                            Ok(result) => serde_json::to_string(&result)
                                .unwrap_or_else(|_| result.to_string()),
                            Err(e) => format!("Tool error: {}", e),
                        }
                    })
                })
            })
        }
    }
}

/// Build `AgentConfig` from the common constructor arguments.
///
/// Returns the config so the caller can attach `tool_executor` before
/// calling `$rs_type::new(config)`.
macro_rules! build_agent_config {
    ($name:expr, $llm:expr, $system_prompt:expr,
     $tools:expr, $retries:expr, $memory_steps:expr) => {{
        let mut config = AgentConfig {
            name: $name.to_string(),
            llm: $llm.inner.clone(),
            retries: $retries,
            ..Default::default()
        };
        if let Some(prompt) = $system_prompt {
            config.system_prompt = Some(prompt.to_string());
        }
        if let Some(tool_items) = $tools {
            let mut defs = Vec::new();
            for item in &tool_items {
                defs.push(extract_tool_definition(item)?);
            }
            config.tools = defs;
        }
        if let Some(steps) = $memory_steps {
            config.memory_steps = Some(steps);
        }
        config
    }};
}

// ── Typed agent constructors ──────────────────────────────────────────────────
//
// Each class accepts keyword arguments and wraps the corresponding Rust typed agent.
//
//   agent = ZeroShotReAct(name="my-agent", llm=llm, tools=[calc], tool_registry=registry)
//   result = agent.execute_input("What is 2+2?")

// ─── ZeroShotReAct ──────────────────────────────────────────────────────────

/// Zero-shot ReAct agent — general-purpose reasoning + action without examples.
///
/// Example::
///
///     from flowgentra_ai.agent import ZeroShotReAct
///     from flowgentra_ai.llm import LLMClient, ToolDefinition
///     from flowgentra_ai.tools import ToolRegistry, tool
///
///     @tool("greet", "Greet a person")
///     def greet(name: str) -> dict:
///         return {"greeting": f"Hello, {name}!"}
///
///     registry = ToolRegistry.with_builtins()
///     registry.register(greet)
///
///     calc = ToolDefinition("calculator", "Arithmetic", {...})
///
///     agent = ZeroShotReAct(
///         name="assistant",
///         llm=llm,
///         tools=[calc],
///         tool_registry=registry,
///     )
///     result = agent.execute_input("What is 17 * 8?")
#[pyclass(name = "ZeroShotReAct")]
pub struct PyZeroShotReAct {
    inner: RsZeroShotReAct,
}

#[pymethods]
impl PyZeroShotReAct {
    #[new]
    #[pyo3(signature = (name, llm, system_prompt=None, tools=None, retries=3, memory_steps=None, tool_registry=None))]
    fn new(
        py: Python<'_>,
        name: &str,
        llm: &PyLLM,
        system_prompt: Option<&str>,
        tools: Option<Vec<Bound<'_, PyAny>>>,
        retries: usize,
        memory_steps: Option<usize>,
        tool_registry: Option<PyObject>,
    ) -> PyResult<Self> {
        let mut config = build_agent_config!(name, llm, system_prompt, tools, retries, memory_steps);
        config.tool_executor = Some(build_executor(tool_registry));
        let agent = RsZeroShotReAct::new(config).map_err(to_py_err)?;
        Ok(Self { inner: agent })
    }

    fn execute_input(&self, input: &str) -> PyResult<String> {
        crate::run_async(self.inner.execute_input(input)).map_err(to_py_err)
    }

    #[getter]
    fn name(&self) -> String { self.inner.name().to_string() }

    fn node_names(&self) -> Vec<String> { self.inner.graph().node_names() }

    fn __repr__(&self) -> String {
        format!("ZeroShotReAct(name='{}')", self.inner.name())
    }
}

// ─── FewShotReAct ───────────────────────────────────────────────────────────

/// Few-shot ReAct agent — same as ZeroShotReAct with example demonstrations.
#[pyclass(name = "FewShotReAct")]
pub struct PyFewShotReAct {
    inner: RsFewShotReAct,
}

#[pymethods]
impl PyFewShotReAct {
    #[new]
    #[pyo3(signature = (name, llm, system_prompt=None, tools=None, retries=3, memory_steps=None, tool_registry=None))]
    fn new(
        py: Python<'_>,
        name: &str,
        llm: &PyLLM,
        system_prompt: Option<&str>,
        tools: Option<Vec<Bound<'_, PyAny>>>,
        retries: usize,
        memory_steps: Option<usize>,
        tool_registry: Option<PyObject>,
    ) -> PyResult<Self> {
        let mut config = build_agent_config!(name, llm, system_prompt, tools, retries, memory_steps);
        config.tool_executor = Some(build_executor(tool_registry));
        let agent = RsFewShotReAct::new(config).map_err(to_py_err)?;
        Ok(Self { inner: agent })
    }

    fn execute_input(&self, input: &str) -> PyResult<String> {
        crate::run_async(self.inner.execute_input(input)).map_err(to_py_err)
    }

    #[getter]
    fn name(&self) -> String { self.inner.name().to_string() }

    fn node_names(&self) -> Vec<String> { self.inner.graph().node_names() }

    fn __repr__(&self) -> String {
        format!("FewShotReAct(name='{}')", self.inner.name())
    }
}

// ─── Conversational ─────────────────────────────────────────────────────────

/// Conversational agent — multi-turn dialogue with persistent conversation history.
#[pyclass(name = "Conversational")]
pub struct PyConversational {
    inner: RsConversational,
}

#[pymethods]
impl PyConversational {
    #[new]
    #[pyo3(signature = (name, llm, system_prompt=None, tools=None, retries=3, memory_steps=None, tool_registry=None))]
    fn new(
        py: Python<'_>,
        name: &str,
        llm: &PyLLM,
        system_prompt: Option<&str>,
        tools: Option<Vec<Bound<'_, PyAny>>>,
        retries: usize,
        memory_steps: Option<usize>,
        tool_registry: Option<PyObject>,
    ) -> PyResult<Self> {
        let mut config = build_agent_config!(name, llm, system_prompt, tools, retries, memory_steps);
        config.tool_executor = Some(build_executor(tool_registry));
        let agent = RsConversational::new(config).map_err(to_py_err)?;
        Ok(Self { inner: agent })
    }

    fn execute_input(&self, input: &str) -> PyResult<String> {
        crate::run_async(self.inner.execute_input(input)).map_err(to_py_err)
    }

    #[getter]
    fn name(&self) -> String { self.inner.name().to_string() }

    fn node_names(&self) -> Vec<String> { self.inner.graph().node_names() }

    fn __repr__(&self) -> String {
        format!("Conversational(name='{}')", self.inner.name())
    }
}

// ─── ToolCalling ────────────────────────────────────────────────────────────

/// Tool-calling agent — uses the provider's native function-calling API.
#[pyclass(name = "ToolCalling")]
pub struct PyToolCalling {
    inner: RsToolCalling,
}

#[pymethods]
impl PyToolCalling {
    #[new]
    #[pyo3(signature = (name, llm, system_prompt=None, tools=None, retries=3, memory_steps=None, tool_registry=None))]
    fn new(
        py: Python<'_>,
        name: &str,
        llm: &PyLLM,
        system_prompt: Option<&str>,
        tools: Option<Vec<Bound<'_, PyAny>>>,
        retries: usize,
        memory_steps: Option<usize>,
        tool_registry: Option<PyObject>,
    ) -> PyResult<Self> {
        let mut config = build_agent_config!(name, llm, system_prompt, tools, retries, memory_steps);
        config.tool_executor = Some(build_executor(tool_registry));
        let agent = RsToolCalling::new(config).map_err(to_py_err)?;
        Ok(Self { inner: agent })
    }

    fn execute_input(&self, input: &str) -> PyResult<String> {
        crate::run_async(self.inner.execute_input(input)).map_err(to_py_err)
    }

    #[getter]
    fn name(&self) -> String { self.inner.name().to_string() }

    fn node_names(&self) -> Vec<String> { self.inner.graph().node_names() }

    fn __repr__(&self) -> String {
        format!("ToolCalling(name='{}')", self.inner.name())
    }
}

// ─── StructuredChat ─────────────────────────────────────────────────────────

/// Structured-chat agent — ReAct with JSON-blob tool actions and a JSON final answer.
#[pyclass(name = "StructuredChat")]
pub struct PyStructuredChat {
    inner: RsStructuredChat,
}

#[pymethods]
impl PyStructuredChat {
    #[new]
    #[pyo3(signature = (name, llm, system_prompt=None, tools=None, retries=3, memory_steps=None, tool_registry=None))]
    fn new(
        py: Python<'_>,
        name: &str,
        llm: &PyLLM,
        system_prompt: Option<&str>,
        tools: Option<Vec<Bound<'_, PyAny>>>,
        retries: usize,
        memory_steps: Option<usize>,
        tool_registry: Option<PyObject>,
    ) -> PyResult<Self> {
        let mut config = build_agent_config!(name, llm, system_prompt, tools, retries, memory_steps);
        config.tool_executor = Some(build_executor(tool_registry));
        let agent = RsStructuredChat::new(config).map_err(to_py_err)?;
        Ok(Self { inner: agent })
    }

    fn execute_input(&self, input: &str) -> PyResult<String> {
        crate::run_async(self.inner.execute_input(input)).map_err(to_py_err)
    }

    #[getter]
    fn name(&self) -> String { self.inner.name().to_string() }

    fn node_names(&self) -> Vec<String> { self.inner.graph().node_names() }

    fn __repr__(&self) -> String {
        format!("StructuredChat(name='{}')", self.inner.name())
    }
}

// ─── SelfAskWithSearch ──────────────────────────────────────────────────────

/// Self-ask-with-search agent — decomposes questions via a `search` tool.
#[pyclass(name = "SelfAskWithSearch")]
pub struct PySelfAskWithSearch {
    inner: RsSelfAskWithSearch,
}

#[pymethods]
impl PySelfAskWithSearch {
    #[new]
    #[pyo3(signature = (name, llm, system_prompt=None, tools=None, retries=3, memory_steps=None, tool_registry=None))]
    fn new(
        py: Python<'_>,
        name: &str,
        llm: &PyLLM,
        system_prompt: Option<&str>,
        tools: Option<Vec<Bound<'_, PyAny>>>,
        retries: usize,
        memory_steps: Option<usize>,
        tool_registry: Option<PyObject>,
    ) -> PyResult<Self> {
        let mut config = build_agent_config!(name, llm, system_prompt, tools, retries, memory_steps);
        config.tool_executor = Some(build_executor(tool_registry));
        let agent = RsSelfAskWithSearch::new(config).map_err(to_py_err)?;
        Ok(Self { inner: agent })
    }

    fn execute_input(&self, input: &str) -> PyResult<String> {
        crate::run_async(self.inner.execute_input(input)).map_err(to_py_err)
    }

    #[getter]
    fn name(&self) -> String { self.inner.name().to_string() }

    fn node_names(&self) -> Vec<String> { self.inner.graph().node_names() }

    fn __repr__(&self) -> String {
        format!("SelfAskWithSearch(name='{}')", self.inner.name())
    }
}

// ─── ReactDocstore ──────────────────────────────────────────────────────────

/// ReAct docstore agent — Search + Lookup loop over a document store.
#[pyclass(name = "ReactDocstore")]
pub struct PyReactDocstore {
    inner: RsReactDocstore,
}

#[pymethods]
impl PyReactDocstore {
    #[new]
    #[pyo3(signature = (name, llm, system_prompt=None, tools=None, retries=3, memory_steps=None, tool_registry=None))]
    fn new(
        py: Python<'_>,
        name: &str,
        llm: &PyLLM,
        system_prompt: Option<&str>,
        tools: Option<Vec<Bound<'_, PyAny>>>,
        retries: usize,
        memory_steps: Option<usize>,
        tool_registry: Option<PyObject>,
    ) -> PyResult<Self> {
        let mut config = build_agent_config!(name, llm, system_prompt, tools, retries, memory_steps);
        config.tool_executor = Some(build_executor(tool_registry));
        let agent = RsReactDocstore::new(config).map_err(to_py_err)?;
        Ok(Self { inner: agent })
    }

    fn execute_input(&self, input: &str) -> PyResult<String> {
        crate::run_async(self.inner.execute_input(input)).map_err(to_py_err)
    }

    #[getter]
    fn name(&self) -> String { self.inner.name().to_string() }

    fn node_names(&self) -> Vec<String> { self.inner.graph().node_names() }

    fn __repr__(&self) -> String {
        format!("ReactDocstore(name='{}')", self.inner.name())
    }
}
