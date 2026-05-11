//! Python bindings for MCP (Model Context Protocol)

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use flowgentra_ai::core::mcp::{
    MCPAuth, MCPClient, MCPConfig, MCPConnectionSettings, MCPConnectionType, MCPPromptArgument,
    MCPResource, MCPResourceContent, MCPTool, MCPClientFactory,
};

use crate::error::to_py_err;
use crate::{json_to_py, py_to_json};

// ─── PyMCPConnectionType ────────────────────────────────────────────────────

/// MCP connection type.
#[pyclass(name = "MCPConnectionType", eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum PyMCPConnectionType {
    Sse,
    Stdio,
    Docker,
}

#[pymethods]
impl PyMCPConnectionType {
    fn is_remote(&self) -> bool {
        matches!(self, PyMCPConnectionType::Sse)
    }

    fn is_local(&self) -> bool {
        matches!(self, PyMCPConnectionType::Stdio | PyMCPConnectionType::Docker)
    }

    fn as_str(&self) -> &str {
        match self {
            PyMCPConnectionType::Sse => "sse",
            PyMCPConnectionType::Stdio => "stdio",
            PyMCPConnectionType::Docker => "docker",
        }
    }

    fn __repr__(&self) -> String {
        format!("MCPConnectionType.{}", match self {
            PyMCPConnectionType::Sse => "Sse",
            PyMCPConnectionType::Stdio => "Stdio",
            PyMCPConnectionType::Docker => "Docker",
        })
    }
}

// ─── PyMCPAuth ───────────────────────────────────────────────────────────────

/// MCP authentication credentials.
///
/// Example:
///     # Bearer token
///     auth = MCPAuth(auth_type="bearer", credentials={"token": "my-token"})
///
///     # API key
///     auth = MCPAuth(auth_type="api_key", credentials={"header": "X-API-Key", "key": "abc123"})
///
///     # Basic auth
///     auth = MCPAuth(auth_type="basic", credentials={"username": "user", "password": "pass"})
#[pyclass(name = "MCPAuth")]
#[derive(Clone)]
pub struct PyMCPAuth {
    pub(crate) inner: MCPAuth,
}

#[pymethods]
impl PyMCPAuth {
    #[new]
    #[pyo3(signature = (auth_type, credentials))]
    fn new(auth_type: String, credentials: HashMap<String, String>) -> Self {
        PyMCPAuth {
            inner: MCPAuth { auth_type, credentials },
        }
    }

    #[getter]
    fn auth_type(&self) -> &str {
        &self.inner.auth_type
    }

    /// Return the credentials dictionary.
    ///
    /// .. warning::
    ///    This dict contains sensitive values such as bearer tokens, API keys,
    ///    or passwords.  **Never** log, print, or include this value in
    ///    responses, tracebacks, or error messages.  Store credentials in
    ///    environment variables and load them at runtime rather than
    ///    hard-coding them in source files or configs.
    #[getter]
    fn credentials(&self) -> HashMap<String, String> {
        self.inner.credentials.clone()
    }

    fn __repr__(&self) -> String {
        format!("MCPAuth(type='{}')", self.inner.auth_type)
    }
}

// ─── PyMCPConnectionSettings ────────────────────────────────────────────────

/// Connection-specific settings for MCP.
///
/// Example:
///     settings = MCPConnectionSettings(
///         timeout=30,
///         connect_timeout=5,
///         call_timeout=60,
///         max_retries=3
///     )
#[pyclass(name = "MCPConnectionSettings")]
#[derive(Clone)]
pub struct PyMCPConnectionSettings {
    pub(crate) inner: MCPConnectionSettings,
}

#[pymethods]
impl PyMCPConnectionSettings {
    #[new]
    #[pyo3(signature = (
        timeout=None,
        connect_timeout=None,
        call_timeout=None,
        container_name=None,
        port=None,
        host_port=None,
        working_dir=None,
        env_vars=None,
        max_retries=None
    ))]
    fn new(
        timeout: Option<u64>,
        connect_timeout: Option<u64>,
        call_timeout: Option<u64>,
        container_name: Option<String>,
        port: Option<u16>,
        host_port: Option<u16>,
        working_dir: Option<String>,
        env_vars: Option<HashMap<String, String>>,
        max_retries: Option<u32>,
    ) -> Self {
        PyMCPConnectionSettings {
            inner: MCPConnectionSettings {
                timeout,
                connect_timeout,
                call_timeout,
                container_name,
                port,
                host_port,
                working_dir,
                env_vars: env_vars.unwrap_or_default(),
                max_retries,
            },
        }
    }

    #[getter]
    fn timeout(&self) -> Option<u64> { self.inner.timeout }
    #[getter]
    fn connect_timeout(&self) -> Option<u64> { self.inner.connect_timeout }
    #[getter]
    fn call_timeout(&self) -> Option<u64> { self.inner.call_timeout }
    #[getter]
    fn max_retries(&self) -> Option<u32> { self.inner.max_retries }
    #[getter]
    fn container_name(&self) -> Option<String> { self.inner.container_name.clone() }
    #[getter]
    fn port(&self) -> Option<u16> { self.inner.port }
    #[getter]
    fn host_port(&self) -> Option<u16> { self.inner.host_port }
    #[getter]
    fn working_dir(&self) -> Option<String> { self.inner.working_dir.clone() }
    #[getter]
    fn env_vars(&self) -> HashMap<String, String> { self.inner.env_vars.clone() }

    fn __repr__(&self) -> String {
        format!("MCPConnectionSettings(timeout={:?}, max_retries={:?})", self.inner.timeout, self.inner.max_retries)
    }
}

// ─── PyMCPConfig ────────────────────────────────────────────────────────────

/// MCP (Model Context Protocol) configuration.
///
/// Example:
///     # SSE connection
///     config = MCPConfig.sse("http://localhost:8080/sse", name="my-server")
///
///     # Stdio connection with args
///     config = MCPConfig.stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
///
///     # Docker connection
///     config = MCPConfig.docker("my-org/tool-image:latest", name="container_tool")
///
///     # Advanced: with auth, namespace, filtering
///     from flowgentra_ai.mcp import MCPAuth
///     auth = MCPAuth("bearer", {"token": "my-token"})
///     config = MCPConfig.sse("http://api.example.com/mcp", name="api")
///     config = config.with_auth(auth).with_namespace("api").with_tool_include(["search", "fetch"])
#[pyclass(name = "MCPConfig")]
#[derive(Clone)]
pub struct PyMCPConfig {
    pub(crate) inner: MCPConfig,
}

#[pymethods]
impl PyMCPConfig {
    /// Create an SSE (Server-Sent Events) connection config.
    #[staticmethod]
    #[pyo3(signature = (url, name=None))]
    fn sse(url: &str, name: Option<&str>) -> PyResult<Self> {
        let mut builder = MCPConfig::builder().sse(url);
        if let Some(n) = name { builder = builder.name(n); }
        let config = builder.build().map_err(to_py_err)?;
        Ok(PyMCPConfig { inner: config })
    }

    /// Create a Stdio connection config.
    #[staticmethod]
    #[pyo3(signature = (command, args=None, name=None))]
    fn stdio(command: &str, args: Option<Vec<String>>, name: Option<&str>) -> PyResult<Self> {
        let mut builder = MCPConfig::builder().stdio(command);
        if let Some(a) = args { builder = builder.args(a); }
        if let Some(n) = name { builder = builder.name(n); }
        let config = builder.build().map_err(to_py_err)?;
        Ok(PyMCPConfig { inner: config })
    }

    /// Create a Docker connection config.
    #[staticmethod]
    #[pyo3(signature = (image, name=None))]
    fn docker(image: &str, name: Option<&str>) -> PyResult<Self> {
        let mut builder = MCPConfig::builder().docker(image);
        if let Some(n) = name { builder = builder.name(n); }
        let config = builder.build().map_err(to_py_err)?;
        Ok(PyMCPConfig { inner: config })
    }

    /// Return a copy with authentication set.
    fn with_auth(&self, auth: &PyMCPAuth) -> PyMCPConfig {
        let mut inner = self.inner.clone();
        inner.auth = Some(auth.inner.clone());
        PyMCPConfig { inner }
    }

    /// Return a copy with a namespace prefix for tools (e.g. "math" → tools become "math.add").
    fn with_namespace(&self, ns: &str) -> PyMCPConfig {
        let mut inner = self.inner.clone();
        inner.namespace = Some(ns.to_string());
        PyMCPConfig { inner }
    }

    /// Return a copy that only exposes tools whose names are in the given list.
    fn with_tool_include(&self, tools: Vec<String>) -> PyMCPConfig {
        let mut inner = self.inner.clone();
        inner.tool_include = Some(tools);
        PyMCPConfig { inner }
    }

    /// Return a copy that hides tools whose names are in the given list.
    fn with_tool_exclude(&self, tools: Vec<String>) -> PyMCPConfig {
        let mut inner = self.inner.clone();
        inner.tool_exclude = Some(tools);
        PyMCPConfig { inner }
    }

    /// Return a copy with custom connection settings.
    fn with_connection_settings(&self, settings: &PyMCPConnectionSettings) -> PyMCPConfig {
        let mut inner = self.inner.clone();
        inner.connection_settings = settings.inner.clone();
        PyMCPConfig { inner }
    }

    /// Check if this is a remote connection.
    fn is_remote(&self) -> bool {
        self.inner.connection_type.is_remote()
    }

    /// Check if this is a local connection.
    fn is_local(&self) -> bool {
        self.inner.connection_type.is_local()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn connection_type(&self) -> PyMCPConnectionType {
        match self.inner.connection_type {
            MCPConnectionType::Sse => PyMCPConnectionType::Sse,
            MCPConnectionType::Stdio => PyMCPConnectionType::Stdio,
            MCPConnectionType::Docker => PyMCPConnectionType::Docker,
        }
    }

    #[getter]
    fn uri(&self) -> String {
        self.inner.uri.clone()
    }

    #[getter]
    fn command(&self) -> Option<String> {
        self.inner.command.clone()
    }

    #[getter]
    fn args(&self) -> Vec<String> {
        self.inner.args.clone()
    }

    #[getter]
    fn image(&self) -> Option<String> {
        self.inner.image.clone()
    }

    #[getter]
    fn namespace(&self) -> Option<String> {
        self.inner.namespace.clone()
    }

    #[getter]
    fn tool_include(&self) -> Option<Vec<String>> {
        self.inner.tool_include.clone()
    }

    #[getter]
    fn tool_exclude(&self) -> Option<Vec<String>> {
        self.inner.tool_exclude.clone()
    }

    #[getter]
    fn auth(&self) -> Option<PyMCPAuth> {
        self.inner.auth.as_ref().map(|a| PyMCPAuth { inner: a.clone() })
    }

    #[getter]
    fn connection_settings(&self) -> PyMCPConnectionSettings {
        PyMCPConnectionSettings { inner: self.inner.connection_settings.clone() }
    }

    fn __repr__(&self) -> String {
        let conn_type = match self.inner.connection_type {
            MCPConnectionType::Sse => "sse",
            MCPConnectionType::Stdio => "stdio",
            MCPConnectionType::Docker => "docker",
        };
        format!("MCPConfig(type='{}', name='{}')", conn_type, self.inner.name)
    }
}

// ─── PyMCPTool ───────────────────────────────────────────────────────────────

/// An MCP tool descriptor returned by list_tools().
#[pyclass(name = "MCPTool")]
#[derive(Clone)]
pub struct PyMCPTool {
    pub(crate) inner: MCPTool,
}

#[pymethods]
impl PyMCPTool {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn description(&self) -> Option<&str> {
        self.inner.description.as_deref()
    }

    #[getter]
    fn input_schema<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        json_to_py(py, &self.inner.input_schema)
    }

    fn __repr__(&self) -> String {
        format!(
            "MCPTool(name='{}', description={:?})",
            self.inner.name,
            self.inner.description.as_deref().unwrap_or("")
        )
    }
}

// ─── PyMCPResource ───────────────────────────────────────────────────────────

/// An MCP resource descriptor returned by list_resources().
#[pyclass(name = "MCPResource")]
#[derive(Clone)]
pub struct PyMCPResource {
    pub(crate) inner: MCPResource,
}

#[pymethods]
impl PyMCPResource {
    #[getter]
    fn uri(&self) -> &str {
        &self.inner.uri
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
    }

    #[getter]
    fn description(&self) -> Option<&str> {
        self.inner.description.as_deref()
    }

    #[getter]
    fn mime_type(&self) -> Option<&str> {
        self.inner.mime_type.as_deref()
    }

    fn __repr__(&self) -> String {
        format!("MCPResource(uri='{}', name={:?})", self.inner.uri, self.inner.name)
    }
}

// ─── PyMCPResourceContent ────────────────────────────────────────────────────

/// Content of a resource returned by read_resource().
#[pyclass(name = "MCPResourceContent")]
#[derive(Clone)]
pub struct PyMCPResourceContent {
    pub(crate) inner: MCPResourceContent,
}

#[pymethods]
impl PyMCPResourceContent {
    #[getter]
    fn uri(&self) -> &str {
        &self.inner.uri
    }

    #[getter]
    fn mime_type(&self) -> Option<&str> {
        self.inner.mime_type.as_deref()
    }

    #[getter]
    fn text(&self) -> Option<&str> {
        self.inner.text.as_deref()
    }

    #[getter]
    fn blob(&self) -> Option<&str> {
        self.inner.blob.as_deref()
    }

    fn __repr__(&self) -> String {
        format!("MCPResourceContent(uri='{}', mime_type={:?})", self.inner.uri, self.inner.mime_type)
    }
}

// ─── PyMCPPromptArgument ─────────────────────────────────────────────────────

/// An argument definition for an MCP prompt template.
#[pyclass(name = "MCPPromptArgument")]
#[derive(Clone)]
pub struct PyMCPPromptArgument {
    pub(crate) inner: MCPPromptArgument,
}

#[pymethods]
impl PyMCPPromptArgument {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn description(&self) -> Option<&str> {
        self.inner.description.as_deref()
    }

    #[getter]
    fn required(&self) -> bool {
        self.inner.required
    }

    fn __repr__(&self) -> String {
        format!("MCPPromptArgument(name='{}', required={})", self.inner.name, self.inner.required)
    }
}

// ─── PyMCPPrompt ─────────────────────────────────────────────────────────────

/// An MCP prompt template returned by list_prompts().
#[pyclass(name = "MCPPrompt")]
#[derive(Clone)]
pub struct PyMCPPrompt {
    name: String,
    description: Option<String>,
    arguments: Vec<PyMCPPromptArgument>,
}

#[pymethods]
impl PyMCPPrompt {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    #[getter]
    fn arguments(&self) -> Vec<PyMCPPromptArgument> {
        self.arguments.clone()
    }

    fn __repr__(&self) -> String {
        format!("MCPPrompt(name='{}', args={})", self.name, self.arguments.len())
    }
}

// ─── PyMCPPromptMessage ──────────────────────────────────────────────────────

/// A message in a rendered prompt result.
#[pyclass(name = "MCPPromptMessage")]
#[derive(Clone)]
pub struct PyMCPPromptMessage {
    role: String,
    content: serde_json::Value,
}

#[pymethods]
impl PyMCPPromptMessage {
    #[getter]
    fn role(&self) -> &str {
        &self.role
    }

    #[getter]
    fn content<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        json_to_py(py, &self.content)
    }

    fn __repr__(&self) -> String {
        format!("MCPPromptMessage(role='{}')", self.role)
    }
}

// ─── PyMCPPromptResult ───────────────────────────────────────────────────────

/// Result of get_prompt() — a rendered prompt with messages.
#[pyclass(name = "MCPPromptResult")]
#[derive(Clone)]
pub struct PyMCPPromptResult {
    description: Option<String>,
    messages: Vec<PyMCPPromptMessage>,
}

#[pymethods]
impl PyMCPPromptResult {
    #[getter]
    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    #[getter]
    fn messages(&self) -> Vec<PyMCPPromptMessage> {
        self.messages.clone()
    }

    fn __repr__(&self) -> String {
        format!("MCPPromptResult(messages={})", self.messages.len())
    }
}

// ─── PyMCPClient ─────────────────────────────────────────────────────────────

/// MCP client for interacting with an external MCP server.
///
/// Create via the factory function:
///     from flowgentra_ai.mcp import MCPConfig, create_client
///
///     config = MCPConfig.sse("http://localhost:8080/sse", name="my-server")
///     client = create_client(config)
///
///     # Discover tools
///     tools = client.list_tools()
///     for tool in tools:
///         print(tool.name, tool.description)
///
///     # Call a tool
///     result = client.call_tool("my_tool", {"param": "value"})
///
///     # Cleanup
///     client.shutdown()
#[pyclass(name = "MCPClient")]
pub struct PyMCPClient {
    inner: Arc<dyn MCPClient>,
}

#[pymethods]
impl PyMCPClient {
    /// Perform the MCP initialize handshake. Returns the negotiated protocol version.
    fn initialize(&self) -> PyResult<String> {
        crate::run_async(self.inner.initialize()).map_err(to_py_err)
    }

    /// List all tools available on the MCP server.
    fn list_tools(&self) -> PyResult<Vec<PyMCPTool>> {
        let tools = crate::run_async(self.inner.list_tools()).map_err(to_py_err)?;
        Ok(tools.into_iter().map(|t| PyMCPTool { inner: t }).collect())
    }

    /// Call an MCP tool by name with the given arguments dict.
    ///
    /// Args:
    ///     tool_name: Name of the tool to call
    ///     arguments: Dict of arguments to pass to the tool
    ///
    /// Returns:
    ///     The tool result as a Python object (dict, list, str, etc.)
    fn call_tool(&self, tool_name: &str, arguments: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let args_json = py_to_json(arguments)?;
        let result = crate::run_async(self.inner.call_tool(tool_name, args_json)).map_err(to_py_err)?;
        Python::with_gil(|py| json_to_py(py, &result))
    }

    /// Call multiple tools in parallel.
    ///
    /// Args:
    ///     calls: List of (tool_name, arguments_dict) tuples
    ///
    /// Returns:
    ///     List of results in the same order as the input calls
    fn call_tools_parallel(&self, calls: &Bound<'_, pyo3::types::PyList>) -> PyResult<Vec<PyObject>> {
        let mut calls_json: Vec<(String, serde_json::Value)> = Vec::new();
        for item in calls.iter() {
            let tuple = item.downcast::<pyo3::types::PyTuple>()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err("Each call must be a (name, args) tuple"))?;
            let name: String = tuple.get_item(0)?.extract()?;
            let args = tuple.get_item(1)?;
            calls_json.push((name, py_to_json(&args)?));
        }

        let results = crate::run_async(self.inner.call_tools_parallel(calls_json)).map_err(to_py_err)?;
        Python::with_gil(|py| results.iter().map(|v| json_to_py(py, v)).collect())
    }

    /// Check if the MCP server is reachable.
    fn health_check(&self) -> PyResult<bool> {
        crate::run_async(self.inner.health_check()).map_err(to_py_err)
    }

    /// Gracefully shut down the connection (stops subprocess, removes container, etc.).
    fn shutdown(&self) -> PyResult<()> {
        crate::run_async(self.inner.shutdown()).map_err(to_py_err)
    }

    /// List available resources from the MCP server.
    fn list_resources(&self) -> PyResult<Vec<PyMCPResource>> {
        let resources = crate::run_async(self.inner.list_resources()).map_err(to_py_err)?;
        Ok(resources.into_iter().map(|r| PyMCPResource { inner: r }).collect())
    }

    /// Read the content of a resource by URI.
    fn read_resource(&self, uri: &str) -> PyResult<PyMCPResourceContent> {
        let content = crate::run_async(self.inner.read_resource(uri)).map_err(to_py_err)?;
        Ok(PyMCPResourceContent { inner: content })
    }

    /// List available prompt templates from the MCP server.
    fn list_prompts(&self) -> PyResult<Vec<PyMCPPrompt>> {
        let prompts = crate::run_async(self.inner.list_prompts()).map_err(to_py_err)?;
        Ok(prompts
            .into_iter()
            .map(|p| PyMCPPrompt {
                name: p.name,
                description: p.description,
                arguments: p
                    .arguments
                    .into_iter()
                    .map(|a| PyMCPPromptArgument { inner: a })
                    .collect(),
            })
            .collect())
    }

    /// Get a rendered prompt by name with the given arguments.
    ///
    /// Args:
    ///     name: Prompt template name
    ///     arguments: Dict of argument values
    ///
    /// Returns:
    ///     MCPPromptResult with rendered messages
    fn get_prompt(&self, name: &str, arguments: &Bound<'_, PyAny>) -> PyResult<PyMCPPromptResult> {
        let args_json = py_to_json(arguments)?;
        let result = crate::run_async(self.inner.get_prompt(name, args_json)).map_err(to_py_err)?;
        Ok(PyMCPPromptResult {
            description: result.description,
            messages: result
                .messages
                .into_iter()
                .map(|m| PyMCPPromptMessage { role: m.role, content: m.content })
                .collect(),
        })
    }

    fn __repr__(&self) -> String {
        "MCPClient()".to_string()
    }
}

// ─── Factory function ────────────────────────────────────────────────────────

/// Create an MCPClient from an MCPConfig.
///
/// The returned client is automatically wrapped with retry and tool-list caching.
///
/// Example:
///     from flowgentra_ai.mcp import MCPConfig, create_client
///
///     config = MCPConfig.sse("http://localhost:8080/sse", name="my-server")
///     client = create_client(config)
///     tools = client.list_tools()
#[pyfunction(name = "create_client")]
pub fn py_create_mcp_client(config: &PyMCPConfig) -> PyResult<PyMCPClient> {
    let client = MCPClientFactory::create(config.inner.clone()).map_err(to_py_err)?;
    Ok(PyMCPClient { inner: client })
}

/// Merge tool lists from multiple MCPClient instances into one deduplicated list.
///
/// Example:
///     from flowgentra_ai.mcp import create_client, merge_tool_lists
///
///     client1 = create_client(MCPConfig.sse("http://server1/sse", name="s1"))
///     client2 = create_client(MCPConfig.stdio("python", ["-m", "tool"], name="s2"))
///     all_tools = merge_tool_lists([client1, client2])
#[pyfunction(name = "merge_tool_lists")]
pub fn py_merge_tool_lists(clients: Vec<PyRef<'_, PyMCPClient>>) -> PyResult<Vec<PyMCPTool>> {
    let arcs: Vec<Arc<dyn MCPClient>> = clients.iter().map(|c| c.inner.clone()).collect();
    let tools = 
        crate::run_async(flowgentra_ai::core::mcp::merge_tool_lists(&arcs))
        .map_err(to_py_err)?;
    Ok(tools.into_iter().map(|t| PyMCPTool { inner: t }).collect())
}
