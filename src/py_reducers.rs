//! Python-facing reducer helpers for LangGraph-style annotations.
//!
//! Users declare reducers directly on their state class:
//!
//!     from flowgentra_ai import AppendField
//!
//!     class AgentState:
//!         messages: list = []
//!         steps: int = 0
//!         __reducers__ = {"messages": AppendField()}
//!
//! Or with the Annotated pattern (via __reducers__ inspection):
//!
//!     import operator
//!     from typing import Annotated
//!
//!     class AgentState:
//!         messages: Annotated[list, operator.add] = []
//!
//! When `StateGraph(AgentState)` is constructed, the binding layer reads
//! `__reducers__` (or inspects Annotated metadata) and maps each entry to
//! a `ChannelType`, which controls how partial updates are merged.

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ── AppendField ───────────────────────────────────────────────────────────────

/// Marks a field as accumulating (Topic channel / list-append reducer).
///
/// When a node returns `{"messages": ["hello"]}`, the new items are
/// appended to the existing list rather than replacing it.
///
/// Example:
///     from flowgentra_ai import AppendField
///
///     class MyState:
///         messages: list = []
///         __reducers__ = {"messages": AppendField()}
#[pyclass(name = "AppendField")]
#[derive(Clone)]
pub struct PyAppendField;

#[pymethods]
impl PyAppendField {
    #[new]
    fn new() -> Self {
        PyAppendField
    }

    fn __repr__(&self) -> &'static str {
        "AppendField()"
    }

    fn __str__(&self) -> &'static str {
        "AppendField()"
    }
}

// ── BinaryOperatorField ───────────────────────────────────────────────────────

/// Marks a field as using a custom binary merge function.
///
/// The function receives `(current_value, new_value)` and returns the merged result.
/// It must be pure (no side effects) and handle the JSON value types correctly.
///
/// Example:
///     from flowgentra_ai import BinaryOperatorField
///     import operator
///
///     class MyState:
///         score: float = 0.0
///         __reducers__ = {
///             "score": BinaryOperatorField(lambda a, b: max(a, b))
///         }
#[pyclass(name = "BinaryOperatorField")]
pub struct PyBinaryOperatorField {
    /// The Python callable: (current, new) -> merged
    pub func: PyObject,
}

impl Clone for PyBinaryOperatorField {
    fn clone(&self) -> Self {
        Python::with_gil(|py| PyBinaryOperatorField {
            func: self.func.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyBinaryOperatorField {
    #[new]
    fn new(func: PyObject) -> Self {
        PyBinaryOperatorField { func }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!("BinaryOperatorField(func={})", self.func.bind(py).repr().map(|s| s.to_string()).unwrap_or_else(|_| "<fn>".into()))
    }

    /// Invoke the merge function from Python (useful for testing).
    fn merge(&self, py: Python<'_>, current: PyObject, new_val: PyObject) -> PyResult<PyObject> {
        self.func.call1(py, (current, new_val))
    }
}

// ── Utility: extract reducer map from a Python state class ───────────────────

/// Extract the `__reducers__` dict from a Python class, if present.
///
/// Returns a `HashMap<field_name, ChannelType>`.  The caller is responsible
/// for calling this during `StateGraph.__init__` and storing the result.
pub fn extract_reducers_from_class(
    py: Python<'_>,
    state_class: &Bound<'_, PyAny>,
) -> PyResult<std::collections::HashMap<String, crate::channel::ChannelType>> {
    use crate::channel::ChannelType;
    use std::collections::HashMap;

    let mut result: HashMap<String, ChannelType> = HashMap::new();

    // ── Strategy 1: explicit __reducers__ dict ──────────────────────────────
    if let Ok(reducers_attr) = state_class.getattr("__reducers__") {
        if let Ok(reducers_dict) = reducers_attr.downcast::<PyDict>() {
            for (k, v) in reducers_dict.iter() {
                let field_name: String = k.extract()?;
                let channel_type = py_reducer_to_channel_type(py, &v)?;
                result.insert(field_name, channel_type);
            }
        }
    }

    // ── Strategy 2: Annotated[T, operator.add] or Annotated[T, AppendField()] ─
    // Inspect __annotations__ for Annotated types with reducer metadata.
    if let Ok(annotations) = state_class.getattr("__annotations__") {
        if let Ok(ann_dict) = annotations.downcast::<PyDict>() {
            for (field, type_hint) in ann_dict.iter() {
                let field_name: String = field.extract()?;
                // Skip if already specified in __reducers__
                if result.contains_key(&field_name) {
                    continue;
                }
                // Check if it's typing.Annotated
                if let Some(channel_type) = try_extract_annotated_reducer(py, &type_hint)? {
                    result.insert(field_name, channel_type);
                }
            }
        }
    }

    Ok(result)
}

/// Convert a Python reducer object (AppendField, BinaryOperatorField, callable)
/// to a `ChannelType`.
fn py_reducer_to_channel_type(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<crate::channel::ChannelType> {
    use crate::channel::ChannelType;
    use std::sync::Arc;

    // AppendField instance
    if obj.is_instance_of::<PyAppendField>() {
        return Ok(ChannelType::Topic);
    }

    // BinaryOperatorField instance
    if obj.is_instance_of::<PyBinaryOperatorField>() {
        let bof: PyBinaryOperatorField = obj.extract()?;
        let func = bof.func.clone_ref(py);
        return Ok(ChannelType::BinaryOperator(Arc::new(move |a, b| {
            Python::with_gil(|py| -> serde_json::Value {
                let py_a = crate::json_to_py(py, &a).unwrap_or_else(|_| py.None());
                let py_b = crate::json_to_py(py, &b).unwrap_or_else(|_| py.None());
                match func.call1(py, (py_a, py_b)) {
                    Ok(result) => crate::py_to_json(result.bind(py)).unwrap_or(serde_json::Value::Null),
                    Err(_) => b,  // on error, fall back to new value
                }
            })
        })));
    }

    // Plain Python callable (e.g. operator.add, lambda a, b: ...)
    if obj.is_callable() {
        let func = obj.to_object(py);
        return Ok(ChannelType::BinaryOperator(Arc::new(move |a, b| {
            Python::with_gil(|py| -> serde_json::Value {
                let py_a = crate::json_to_py(py, &a).unwrap_or_else(|_| py.None());
                let py_b = crate::json_to_py(py, &b).unwrap_or_else(|_| py.None());
                match func.call1(py, (py_a, py_b)) {
                    Ok(result) => crate::py_to_json(result.bind(py)).unwrap_or(serde_json::Value::Null),
                    Err(_) => b,
                }
            })
        })));
    }

    // String shorthand: "append", "last_value"
    if let Ok(s) = obj.extract::<String>() {
        return match s.as_str() {
            "append" | "topic" => Ok(ChannelType::Topic),
            "last_value" | "replace" | "overwrite" => Ok(ChannelType::LastValue),
            other => Err(crate::error::ValidationError::new_err(format!(
                "Unknown reducer string '{}'. Use 'append' or 'last_value'.",
                other
            ))),
        };
    }

    let type_name = obj.get_type().name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Unsupported reducer type: {}. Use AppendField(), BinaryOperatorField(fn), \
         or a callable (current, new) -> merged.",
        type_name
    )))
}

/// Try to extract a `ChannelType` from a `typing.Annotated[T, meta, ...]` annotation.
/// Returns `None` if the annotation is not Annotated or has no recognised reducer.
fn try_extract_annotated_reducer(
    py: Python<'_>,
    type_hint: &Bound<'_, PyAny>,
) -> PyResult<Option<crate::channel::ChannelType>> {
    // typing.get_args(type_hint) returns () for non-Annotated types
    let typing = py.import_bound("typing")?;
    let get_args = typing.getattr("get_args")?;
    let args = get_args.call1((type_hint,))?;
    let args_tuple = match args.downcast::<pyo3::types::PyTuple>() {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    // Annotated[T, meta1, meta2, ...] → args = (T, meta1, meta2, ...)
    // We need at least 2 elements (T and one metadata item)
    if args_tuple.len() < 2 {
        return Ok(None);
    }

    // Check if this is actually an Annotated type by seeing if typing.get_origin returns Annotated
    let get_origin = typing.getattr("get_origin")?;
    let origin = get_origin.call1((type_hint,))?;
    let annotated_type = typing.getattr("Annotated")?;
    if !origin.eq(&annotated_type)? {
        return Ok(None);
    }

    // Inspect metadata items (args[1..])
    for i in 1..args_tuple.len() {
        let meta = args_tuple.get_item(i)?;
        if let Ok(ct) = py_reducer_to_channel_type(py, &meta) {
            return Ok(Some(ct));
        }
    }

    Ok(None)
}
