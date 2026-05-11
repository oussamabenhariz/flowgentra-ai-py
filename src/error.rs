//! Error conversion from FlowgentraError to typed Python exceptions.
//!
//! ## Exception hierarchy
//!
//! ```text
//! Exception
//! └── FlowgentraAIError          (base for all library errors)
//!     ├── ConfigurationError     (ConfigError, YamlError)
//!     ├── ValidationError        (ValidationError, InvalidStateTransition)
//!     ├── GraphError             (InvalidEdge, RoutingError, GraphError)
//!     │   ├── NodeNotFoundError  (NodeNotFound)
//!     │   └── CycleError         (CycleDetected, RecursionLimitExceeded, NoTerminationPath)
//!     ├── LLMError               (LLMError)
//!     ├── MCPError               (MCPError, MCPTransportError, MCPServerError)
//!     ├── ToolExecutionError     (ToolError)
//!     ├── AgentExecutionError    (ExecutionError, NodeExecutionError, ExecutionAborted, ParallelExecutionError)
//!     ├── WorkflowTimeoutError   (TimeoutError, ExecutionTimeout)
//!     ├── SerializationError     (SerializationError)
//!     ├── CheckpointError        (checkpoint persistence failures)
//!     └── InternalError          (StateError, RuntimeError, unexpected failures)
//! ```
//!
//! `IoError` maps to `OSError` (standard Python I/O exception) rather than
//! `FlowgentraAIError` to preserve built-in semantics.

use pyo3::prelude::*;
use pyo3::exceptions::PyOSError;
use flowgentra_ai::core::error::FlowgentraError;
use flowgentra_ai::core::state_graph::StateGraphError;

// ─── Custom exception hierarchy ──────────────────────────────────────────────

pyo3::create_exception!(_native, FlowgentraAIError, pyo3::exceptions::PyException);

pyo3::create_exception!(_native, ConfigurationError, FlowgentraAIError);
pyo3::create_exception!(_native, ValidationError, FlowgentraAIError);
pyo3::create_exception!(_native, GraphError, FlowgentraAIError);
pyo3::create_exception!(_native, NodeNotFoundError, GraphError);
pyo3::create_exception!(_native, CycleError, GraphError);
pyo3::create_exception!(_native, LLMError, FlowgentraAIError);
pyo3::create_exception!(_native, MCPError, FlowgentraAIError);
pyo3::create_exception!(_native, ToolExecutionError, FlowgentraAIError);
pyo3::create_exception!(_native, AgentExecutionError, FlowgentraAIError);
pyo3::create_exception!(_native, WorkflowTimeoutError, FlowgentraAIError);
pyo3::create_exception!(_native, SerializationError, FlowgentraAIError);
pyo3::create_exception!(_native, CheckpointError, FlowgentraAIError);
pyo3::create_exception!(_native, InternalError, FlowgentraAIError);

/// Register all custom exception types on the given module.
///
/// Call this once from the `_native` module initialiser so that
/// `from flowgentra_ai._native import LLMError` works as expected.
pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("FlowgentraAIError", py.get_type_bound::<FlowgentraAIError>())?;
    m.add("ConfigurationError", py.get_type_bound::<ConfigurationError>())?;
    m.add("ValidationError", py.get_type_bound::<ValidationError>())?;
    m.add("GraphError", py.get_type_bound::<GraphError>())?;
    m.add("NodeNotFoundError", py.get_type_bound::<NodeNotFoundError>())?;
    m.add("CycleError", py.get_type_bound::<CycleError>())?;
    m.add("LLMError", py.get_type_bound::<LLMError>())?;
    m.add("MCPError", py.get_type_bound::<MCPError>())?;
    m.add("ToolExecutionError", py.get_type_bound::<ToolExecutionError>())?;
    m.add("AgentExecutionError", py.get_type_bound::<AgentExecutionError>())?;
    m.add("WorkflowTimeoutError", py.get_type_bound::<WorkflowTimeoutError>())?;
    m.add("SerializationError", py.get_type_bound::<SerializationError>())?;
    m.add("CheckpointError", py.get_type_bound::<CheckpointError>())?;
    m.add("InternalError", py.get_type_bound::<InternalError>())?;
    Ok(())
}

// ─── Message sanitization ─────────────────────────────────────────────────────

/// Truncate and sanitize an error message before surfacing it to Python.
///
/// - Caps length to 2 048 chars to prevent log flooding.
/// - Strips any line that looks like it contains a secret header value
///   (e.g. lines echoed back in proxy error responses).
fn sanitize_msg(raw: &str) -> String {
    const MAX_LEN: usize = 2048;
    let sensitive = ["authorization:", "x-api-key:", "bearer ", "x-auth-token:"];
    let filtered: String = raw
        .lines()
        .filter(|line| {
            let lower = line.to_lowercase();
            !sensitive.iter().any(|p| lower.contains(p))
        })
        .collect::<Vec<_>>()
        .join("\n");
    if filtered.len() <= MAX_LEN {
        filtered
    } else {
        format!("{}… [truncated]", &filtered[..MAX_LEN])
    }
}

// ─── Error classification ─────────────────────────────────────────────────────

/// Map a `FlowgentraError` variant to the correct Python exception type.
///
/// Recurses through `Context` wrappers to find the semantic root error and
/// preserves the full (already-formatted) `msg` for the exception value.
fn classify_error(e: &FlowgentraError, msg: String) -> PyErr {
    match e {
        FlowgentraError::NodeNotFound(_) => NodeNotFoundError::new_err(msg),

        FlowgentraError::ConfigError(_) | FlowgentraError::YamlError(_) => {
            ConfigurationError::new_err(msg)
        }

        FlowgentraError::ValidationError(_) | FlowgentraError::InvalidStateTransition(_) => {
            ValidationError::new_err(msg)
        }

        FlowgentraError::InvalidEdge(_)
        | FlowgentraError::GraphError(_)
        | FlowgentraError::RoutingError(_) => GraphError::new_err(msg),

        FlowgentraError::TimeoutError | FlowgentraError::ExecutionTimeout(_) => {
            WorkflowTimeoutError::new_err(msg)
        }

        // Keep OS-level I/O mapped to the standard Python OSError so that
        // callers using `except OSError` continue to work.
        FlowgentraError::IoError(_) => PyOSError::new_err(msg),

        FlowgentraError::CycleDetected { .. }
        | FlowgentraError::RecursionLimitExceeded { .. }
        | FlowgentraError::NoTerminationPath { .. } => CycleError::new_err(msg),

        FlowgentraError::SerializationError(_) => SerializationError::new_err(msg),

        FlowgentraError::LLMError(_) => LLMError::new_err(msg),

        FlowgentraError::MCPError(_)
        | FlowgentraError::MCPTransportError(_)
        | FlowgentraError::MCPServerError(_) => MCPError::new_err(msg),

        FlowgentraError::ToolError(_) => ToolExecutionError::new_err(msg),

        FlowgentraError::ExecutionError(_)
        | FlowgentraError::NodeExecutionError(_)
        | FlowgentraError::ExecutionAborted(_)
        | FlowgentraError::ParallelExecutionError(_) => AgentExecutionError::new_err(msg),

        // Recurse through context wrappers to find the semantic root type.
        // The caller already built `msg` from the full error chain, so we
        // only need the inner variant for classification.
        FlowgentraError::Context(_, inner) => classify_error(inner, msg),

        // StateError / RuntimeError / anything new added to FlowgentraError.
        _ => InternalError::new_err(msg),
    }
}

// ─── Public conversion helpers ────────────────────────────────────────────────

/// Convert a `FlowgentraError` to the most appropriate Python exception.
///
/// Messages are sanitized (length-capped, credential lines stripped) and
/// developer hints are appended before the exception is raised.
pub fn to_py_err(e: FlowgentraError) -> PyErr {
    let hint = e.hint();
    let raw = match hint {
        Some(h) => format!("{}\nHint: {}", e, h),
        None => format!("{}", e),
    };
    let msg = sanitize_msg(&raw);
    classify_error(&e, msg)
}

/// Convert a `StateGraphError` to the appropriate Python exception.
///
/// `StateGraphError` is the graph-executor-level error type; it maps to the
/// same custom exceptions as `FlowgentraError` for a consistent Python surface.
pub fn to_py_err_state_graph(e: StateGraphError) -> PyErr {
    let msg = e.to_string();
    match &e {
        StateGraphError::NodeNotFound(_) => NodeNotFoundError::new_err(msg),
        StateGraphError::InvalidGraph(_) | StateGraphError::TypeError(_) => {
            GraphError::new_err(msg)
        }
        StateGraphError::SerializationError(_) => SerializationError::new_err(msg),
        StateGraphError::Timeout(_) => WorkflowTimeoutError::new_err(msg),
        StateGraphError::RecursionLimitExceeded { .. } | StateGraphError::UnterminatedCycle => {
            CycleError::new_err(msg)
        }
        StateGraphError::RouterError(_) | StateGraphError::EdgeNotFound { .. } => {
            GraphError::new_err(msg)
        }
        StateGraphError::CheckpointError(_) => CheckpointError::new_err(msg),
        StateGraphError::ExecutionError { .. }
        | StateGraphError::InterruptedAtBreakpoint { .. }
        | StateGraphError::ResumeFailed(_) => AgentExecutionError::new_err(msg),
    }
}

/// Convert any `Display` error to a Python `InternalError`.
///
/// Use for non-`FlowgentraError` types (e.g. `VectorStoreError`, `sqlx::Error`)
/// where no semantic mapping is possible.
pub fn to_py_err_generic(e: impl std::fmt::Display) -> PyErr {
    InternalError::new_err(format!("{}", e))
}
