//! Python bindings for Agent

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::Arc;

use flowgentra_ai::core::agent::Agent;
use flowgentra_ai::core::error::FlowgentraError;
use crate::error::{AgentExecutionError, ConfigurationError, ValidationError};
use flowgentra_ai::core::agents::{AgentConfig, AgentType, GraphBasedAgent};
use flowgentra_ai::{ArcHandler, from_config_path_with_extra_handlers};
use flowgentra_ai::core::state::DynState;

use crate::error::to_py_err;
use crate::config::PyAgentConfig;
use crate::llm::PyLLMConfig;
use crate::py_to_json;
use crate::graph::dynstate_to_pydict;

// ─── Internal enum for different agent backends ───────────────────────────────

enum PyAgentInner {
    ConfigBased(Agent),
    GraphBased(GraphBasedAgent),
}

// ─── Python handler wrapping ─────────────────────────────────────────────────

/// Wrap a Python callable (dict → dict) as a Rust ArcHandler<DynState>.
///
/// The Python function receives the full state as a plain dict and must return
/// a dict (full state or partial update). The returned dict becomes the new state.
pub(crate) fn wrap_python_callable(py_func: PyObject) -> ArcHandler<DynState> {
    let func = Arc::new(py_func);
    Arc::new(move |state: DynState| {
        let func = func.clone();
        Box::pin(async move {
            Python::with_gil(|py| -> Result<DynState, FlowgentraError> {
                let state_dict = dynstate_to_pydict(py, &state)
                    .map_err(|e| FlowgentraError::ExecutionError(e.to_string()))?;

                let py_result = func
                    .as_ref()
                    .call1(py, (state_dict,))
                    .map_err(|e| FlowgentraError::ExecutionError(
                        format!("Python handler error: {}", e)
                    ))?;

                let result_bound = py_result.bind(py);
                let result_dict = result_bound
                    .downcast::<PyDict>()
                    .map_err(|_| FlowgentraError::ExecutionError(
                        "Python handler must return a dict".to_string()
                    ))?;

                // Build new state: start from current state, apply returned dict as update
                let new_state = state.clone();
                for (k, v) in result_dict.iter() {
                    let key: String = k
                        .extract()
                        .map_err(|e| FlowgentraError::ExecutionError(e.to_string()))?;
                    let val = py_to_json(&v)
                        .map_err(|e| FlowgentraError::ExecutionError(e.to_string()))?;
                    new_state.set(key, val);
                }
                Ok(new_state)
            })
        })
    })
}

/// Resolve a Python callable from a `module:function` spec string.
fn resolve_python_handler_path(py: Python<'_>, spec: &str) -> PyResult<PyObject> {
    let (module_path, func_name) = spec.split_once(':').ok_or_else(|| {
        ValidationError::new_err(format!(
            "Python handler spec '{}' must use format 'module:function'",
            spec
        ))
    })?;
    let module = py.import_bound(module_path)?;
    let func = module.getattr(func_name)?;
    Ok(func.to_object(py))
}

/// Import a module and collect all functions decorated with `@register_handler`.
pub(crate) fn scan_module_for_handlers(
    py: Python<'_>,
    module_name: &str,
) -> PyResult<HashMap<String, PyObject>> {
    let module = py.import_bound(module_name).map_err(|e| {
        pyo3::exceptions::PyImportError::new_err(format!(
            "Failed to import python_handler_module '{}': {}",
            module_name, e
        ))
    })?;

    let inspect = py.import_bound("inspect")?;
    let members = inspect
        .getattr("getmembers")?
        .call1((&module, inspect.getattr("isfunction")?))?;

    let mut handlers = HashMap::new();
    for item in members.iter()? {
        let pair = item?;
        let name: String = pair.get_item(0)?.extract()?;
        let func = pair.get_item(1)?;
        // Check for _is_handler attribute (set by @register_handler decorator)
        if func.getattr("_is_handler").map(|v| v.is_truthy().unwrap_or(false)).unwrap_or(false) {
            handlers.insert(name, func.to_object(py));
        }
    }
    Ok(handlers)
}

// ─── PyAgent ────────────────────────────────────────────────────────────────

/// The Agent — main interface to FlowgentraAI.
///
/// Create via:
///     1. Agent.create() - Direct API with LLMConfig object
///     2. Agent.from_config_path() - YAML config-driven
///
/// Example:
///     # Option A: Direct API
///     llm = LLMConfig(provider="anthropic", model="claude-opus-4-6", api_key="...")
///     agent = Agent.create(
///         agent_type="conversational",
///         llm=llm,
///         memory_steps=10
///     )
///     result = agent.run_with_input("Hello!")
///
///     # Option B: Config file
///     agent = Agent.from_config_path("config.yaml")
///     result = agent.run()
#[pyclass(name = "Agent")]
pub struct PyAgent {
    inner: PyAgentInner,
}

#[pymethods]
impl PyAgent {
    /// Option A: Create agent directly with an LLMConfig object.
    ///
    /// Args:
    ///     agent_type: "zero_shot_react" | "few_shot_react" | "conversational"
    ///     llm: LLMConfig object (provider, model, api_key, temperature, etc.)
    ///     memory_steps: Number of previous messages to keep in memory (default 10)
    ///
    /// Returns:
    ///     Agent instance ready to run via run_with_input()
    ///
    /// Example:
    ///     llm = LLMConfig("anthropic", "claude-opus-4-6", api_key="sk-ant-...")
    ///     agent = Agent.create(
    ///         agent_type="conversational",
    ///         llm=llm,
    ///         memory_steps=10
    ///     )
    ///     result = agent.run_with_input("Hello!")
    #[staticmethod]
    #[pyo3(signature = (agent_type, llm, memory_steps=10))]
    fn create(
        agent_type: &str,
        llm: PyLLMConfig,
        memory_steps: usize,
    ) -> PyResult<Self> {
        let agent_type_inner = match agent_type.to_lowercase().as_str() {
            "zero_shot_react" => AgentType::ZeroShotReAct,
            "few_shot_react" => AgentType::FewShotReAct,
            "conversational" => AgentType::Conversational,
            _ => return Err(ValidationError::new_err(format!(
                "Unknown agent type: '{}'. Expected: 'zero_shot_react', 'few_shot_react', or 'conversational'",
                agent_type
            ))),
        };

        let config = AgentConfig {
            name: "agent".into(),
            llm: llm.inner.create_client().map_err(to_py_err)?,
            memory_steps: Some(memory_steps),
            ..Default::default()
        };
        let prebuilt = config.into_prebuilt(agent_type_inner);
        let graph_based = GraphBasedAgent::new(prebuilt, None).map_err(to_py_err)?;

        Ok(PyAgent { inner: PyAgentInner::GraphBased(graph_based) })
    }

    /// Option B: Create agent from YAML config file path.
    ///
    /// Supports Python handlers via two mechanisms (Solution C - Hybrid):
    ///
    /// 1. `python_handler_module: handlers` in config + `@register_handler` decorator:
    ///    ```yaml
    ///    python_handler_module: handlers
    ///    graph:
    ///      nodes:
    ///        - name: validate
    ///          handler: validate_input   # auto-discovered from handlers.py
    ///    ```
    ///    ```python
    ///    from flowgentra_ai.agent import register_handler
    ///    @register_handler
    ///    def validate_input(state: dict) -> dict:
    ///        return state
    ///    ```
    ///
    /// 2. Explicit `python.module:function` path in handler config:
    ///    ```yaml
    ///    graph:
    ///      nodes:
    ///        - name: process
    ///          handler: python.other_module:process_func
    ///    ```
    ///    ```python
    ///    # other_module.py — no decorator needed
    ///    def process_func(state: dict) -> dict:
    ///        return state
    ///    ```
    ///
    /// Both approaches can be mixed. Rust handlers registered with `#[register_handler]`
    /// continue to work as before.
    #[staticmethod]
    fn from_config_path(py: Python<'_>, config_path: &str) -> PyResult<Self> {
        // ── Step 1: Parse YAML to find Python-specific config ──────────────────
        let yaml_content = std::fs::read_to_string(config_path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Cannot read config file '{}': {}",
                config_path, e
            ))
        })?;

        let yaml_val: serde_yml::Value = serde_yml::from_str(&yaml_content).map_err(|e| {
            ConfigurationError::new_err(format!(
                "Invalid YAML in '{}': {}",
                config_path, e
            ))
        })?;

        // ── Step 2: Collect Python handlers ───────────────────────────────────
        let mut python_callables: HashMap<String, PyObject> = HashMap::new();

        // Option A: python_handler_module — scan module for @register_handler functions
        if let Some(module_name) = yaml_val
            .get("python_handler_module")
            .and_then(|v| v.as_str())
        {
            let discovered = scan_module_for_handlers(py, module_name)?;
            python_callables.extend(discovered);
        }

        // Option B: node handlers with `python.module:function` prefix
        if let Some(nodes) = yaml_val
            .get("graph")
            .and_then(|g| g.get("nodes"))
            .and_then(|n| n.as_sequence())
        {
            for node in nodes {
                let handler_spec = node
                    .get("handler")
                    .and_then(|h| h.as_str())
                    .unwrap_or("");
                let node_name = node.get("name").and_then(|n| n.as_str()).unwrap_or("");

                if let Some(py_spec) = handler_spec.strip_prefix("python.") {
                    let callable = resolve_python_handler_path(py, py_spec)?;
                    // Also store under node name as fallback
                    if !node_name.is_empty() {
                        let callable2 = Python::with_gil(|p| callable.clone_ref(p));
                        python_callables.insert(node_name.to_string(), callable2);
                    }
                    // Store under the full spec key so Rust lookup (by node_config.handler) matches
                    python_callables.insert(handler_spec.to_string(), callable);
                }
            }
        }

        // ── Step 3: Wrap Python callables as Rust ArcHandlers ─────────────────
        let extra_handlers: HashMap<String, ArcHandler<DynState>> = python_callables
            .into_iter()
            .map(|(name, func)| (name, wrap_python_callable(func)))
            .collect();

        // ── Step 4: Build agent with merged handler registry ──────────────────
        let agent =
            from_config_path_with_extra_handlers(config_path, extra_handlers).map_err(to_py_err)?;

        Ok(PyAgent { inner: PyAgentInner::ConfigBased(agent) })
    }

    /// Get the agent's current state as a plain dict.
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner {
            PyAgentInner::ConfigBased(agent) => {
                dynstate_to_pydict(py, &agent.state).map(|d| d.into())
            }
            PyAgentInner::GraphBased(_) => {
                Ok(pyo3::types::PyDict::new_bound(py).into())
            }
        }
    }

    /// Set a value in the agent's state.
    fn set_state(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let val = py_to_json(value)?;
        match &mut self.inner {
            PyAgentInner::ConfigBased(agent) => { agent.state.set(key, val); }
            PyAgentInner::GraphBased(_) => {} // state not applicable
        }
        Ok(())
    }

    /// Run the agent (config-based) and return the final state as a plain dict.
    fn run(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        match &mut self.inner {
            PyAgentInner::ConfigBased(agent) => {
                let fut = agent.run();
                let result = py.allow_threads(|| crate::run_async(fut)).map_err(to_py_err)?;
                dynstate_to_pydict(py, &result).map(|d| d.into())
            }
            PyAgentInner::GraphBased(_) => {
                Err(ValidationError::new_err(
                    "Use run_with_input(input) for agents created via Agent.create()"
                ))
            }
        }
    }

    /// Run the agent with a string input (for agents created via Agent.create()).
    fn run_with_input(&self, py: Python<'_>, input: &str) -> PyResult<PyObject> {
        match &self.inner {
            PyAgentInner::ConfigBased(_) => {
                Err(ValidationError::new_err(
                    "Use run() for config-based agents created via Agent.from_config_path()"
                ))
            }
            PyAgentInner::GraphBased(agent) => {
                let fut = agent.execute_input(input);
                let result = py.allow_threads(|| crate::run_async(fut)).map_err(to_py_err)?;
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("result", result)?;
                Ok(dict.into())
            }
        }
    }

    /// Run the agent with a thread ID (for checkpointing) and return the final state dict.
    fn run_with_thread(&mut self, py: Python<'_>, thread_id: &str) -> PyResult<PyObject> {
        match &mut self.inner {
            PyAgentInner::ConfigBased(agent) => {
                let fut = agent.run_with_thread(thread_id);
                let result = py.allow_threads(|| crate::run_async(fut)).map_err(to_py_err)?;
                dynstate_to_pydict(py, &result).map(|d| d.into())
            }
            PyAgentInner::GraphBased(_) => {
                Err(ValidationError::new_err(
                    "run_with_thread is not supported for agents created via Agent.create()"
                ))
            }
        }
    }

    /// Get the agent's configuration.
    #[getter]
    fn config(&self) -> PyResult<PyAgentConfig> {
        match &self.inner {
            PyAgentInner::ConfigBased(agent) => Ok(PyAgentConfig { inner: agent.config().clone() }),
            PyAgentInner::GraphBased(_) => Err(pyo3::exceptions::PyAttributeError::new_err(
                "config not available for agents created via Agent.create()"
            )),
        }
    }

    /// Get the agent name from config.
    #[getter]
    fn name(&self) -> String {
        match &self.inner {
            PyAgentInner::ConfigBased(agent) => agent.config().name.clone(),
            PyAgentInner::GraphBased(agent) => {
                use flowgentra_ai::core::agents::Agent as AgentTrait;
                agent.name().to_string()
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("Agent(name='{}')", self.name())
    }
}

// ─── Free function ──────────────────────────────────────────────────────────

/// Create an agent from a YAML config file with auto-discovered handlers.
#[pyfunction]
pub fn py_from_config_path(py: Python<'_>, config_path: &str) -> PyResult<PyAgent> {
    PyAgent::from_config_path(py, config_path)
}
