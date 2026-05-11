//! Python bindings for code execution tools (PythonRepl, NodeJsRepl, Shell).

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::tools::code_exec::{NodeJsReplTool, PythonReplTool, ShellTool};
use flowgentra_ai::core::tools::Tool;

use crate::error::to_py_err;
use crate::{json_to_py, py_to_json};

macro_rules! impl_tool {
    ($py_name:ident, $repr:expr) => {
        #[pymethods]
        impl $py_name {
            fn call(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<PyObject> {
                let v = py_to_json(input)?;
                let r = crate::run_async(self.inner.call(v)).map_err(to_py_err)?;
                json_to_py(py, &r)
            }
            fn definition(&self, py: Python<'_>) -> PyResult<PyObject> {
                let val = serde_json::to_value(&self.inner.definition())
                    .map_err(|e| pyo3::exceptions::crate::error::ToolExecutionError::new_err(e.to_string()))?;
                json_to_py(py, &val)
            }
            fn __repr__(&self) -> String { $repr.to_string() }
        }
    };
}

// ─── PythonReplTool ─────────────────────────────────────────────────────────

/// Execute Python code in a subprocess and return stdout/stderr/exit_code.
///
/// Default subprocess timeout: **30 seconds** (matches the Rust ``PythonReplTool``
/// default).  Override per-call via the ``timeout_secs`` field in the input dict,
/// or at construction time via the ``timeout_secs`` argument.
#[pyclass(name = "PythonReplTool")]
pub struct PyPythonReplTool {
    pub inner: Arc<PythonReplTool>,
}

#[pymethods]
impl PyPythonReplTool {
    #[new]
    #[pyo3(signature = (python_path=None, timeout_secs=None))]
    fn new(python_path: Option<String>, timeout_secs: Option<u64>) -> Self {
        Self {
            inner: Arc::new(PythonReplTool::new(
                python_path.unwrap_or_else(|| "python3".to_string()),
                timeout_secs.unwrap_or(30),
            )),
        }
    }
}
impl_tool!(PyPythonReplTool, "PythonReplTool()");

// ─── NodeJsReplTool ─────────────────────────────────────────────────────────

/// Execute JavaScript code via Node.js and return stdout/stderr/exit_code.
///
/// Default subprocess timeout: **30 seconds**.  Override per-call via the
/// ``timeout_secs`` field in the input dict, or at construction time via the
/// ``timeout_secs`` argument.
#[pyclass(name = "NodeJsReplTool")]
pub struct PyNodeJsReplTool {
    pub inner: Arc<NodeJsReplTool>,
}

#[pymethods]
impl PyNodeJsReplTool {
    #[new]
    #[pyo3(signature = (timeout_secs=None))]
    fn new(timeout_secs: Option<u64>) -> Self {
        Self { inner: Arc::new(NodeJsReplTool::new(timeout_secs.unwrap_or(30))) }
    }
}
impl_tool!(PyNodeJsReplTool, "NodeJsReplTool()");

// ─── ShellTool ──────────────────────────────────────────────────────────────

/// Execute shell commands.
///
/// Default subprocess timeout: **30 seconds**.  This matches the Rust-side
/// ``ShellTool`` default; both can be overridden via the ``timeout_secs``
/// constructor argument or the ``timeout_secs`` field in the per-call input dict.
///
/// **Security**: when ``allowed_commands`` is provided (restricted mode) the
/// command is split into tokens and executed directly without a shell, so shell
/// metacharacters cannot be used for injection.  When ``allowed_commands`` is
/// omitted the tool falls back to ``sh -c`` (unrestricted mode) — only safe
/// when the command string is fully under developer control.
#[pyclass(name = "ShellTool")]
pub struct PyShellTool {
    pub inner: Arc<ShellTool>,
}

#[pymethods]
impl PyShellTool {
    /// Create a restricted shell tool.
    ///
    /// Pass ``allowed_commands=["echo", "ls"]`` etc. to whitelist programs.
    /// Pass ``allowed_commands=[]`` to block all commands (safe default).
    /// Omit ``allowed_commands`` (or pass ``None``) for unrestricted ``sh -c`` mode
    /// — see the class docstring for the security implications.
    #[new]
    #[pyo3(signature = (allowed_commands=None, timeout_secs=None))]
    fn new(allowed_commands: Option<Vec<String>>, timeout_secs: Option<u64>) -> Self {
        let secs = timeout_secs.unwrap_or(30);
        let tool = match allowed_commands {
            Some(cmds) => ShellTool::new(cmds, secs),
            None => ShellTool::unrestricted(secs),
        };
        Self { inner: Arc::new(tool) }
    }

    /// Create an unrestricted shell tool that passes commands to ``sh -c``.
    ///
    /// .. warning::
    ///    Never pass agent-generated or user-supplied input to this tool.
    ///    Use ``ShellTool(allowed_commands=[...])`` for any environment where
    ///    the command string is not fully under your control.
    #[staticmethod]
    #[pyo3(signature = (timeout_secs=None))]
    fn unrestricted(timeout_secs: Option<u64>) -> Self {
        Self { inner: Arc::new(ShellTool::unrestricted(timeout_secs.unwrap_or(30))) }
    }
}
impl_tool!(PyShellTool, "ShellTool()");
