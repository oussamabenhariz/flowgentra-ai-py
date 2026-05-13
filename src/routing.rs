//! Python bindings for Routing Conditions DSL

use crate::error::ValidationError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use flowgentra_ai::core::graph::routing::{
    ComparisonOp, Condition, ConditionBuilder, FieldTypeCheck,
};

use crate::graph::pydict_to_dynstate;
use crate::py_to_json;

// ─── PyComparisonOp ─────────────────────────────────────────────────────────

/// Comparison operator for field comparisons.
///
/// Example:
///     op = ComparisonOp.greater_than()
#[pyclass(name = "ComparisonOp")]
#[derive(Clone)]
pub struct PyComparisonOp {
    pub(crate) inner: ComparisonOp,
}

#[pymethods]
impl PyComparisonOp {
    #[staticmethod]
    fn equal() -> Self {
        PyComparisonOp {
            inner: ComparisonOp::Equal,
        }
    }

    #[staticmethod]
    fn not_equal() -> Self {
        PyComparisonOp {
            inner: ComparisonOp::NotEqual,
        }
    }

    #[staticmethod]
    fn less_than() -> Self {
        PyComparisonOp {
            inner: ComparisonOp::LessThan,
        }
    }

    #[staticmethod]
    fn less_or_equal() -> Self {
        PyComparisonOp {
            inner: ComparisonOp::LessOrEqual,
        }
    }

    #[staticmethod]
    fn greater_than() -> Self {
        PyComparisonOp {
            inner: ComparisonOp::GreaterThan,
        }
    }

    #[staticmethod]
    fn greater_or_equal() -> Self {
        PyComparisonOp {
            inner: ComparisonOp::GreaterOrEqual,
        }
    }

    fn __repr__(&self) -> String {
        format!("ComparisonOp('{}')", self.inner.as_str())
    }

    fn __str__(&self) -> String {
        self.inner.as_str().to_string()
    }
}

// ─── PyCondition ────────────────────────────────────────────────────────────

/// Type-safe routing condition for graph edges.
///
/// Example:
///     cond = Condition.compare("confidence", ComparisonOp.greater_than(), 0.8)
///     cond = Condition.field_exists("result")
///     cond = Condition.and_conditions([cond1, cond2])
#[pyclass(name = "Condition")]
#[derive(Clone)]
pub struct PyCondition {
    pub(crate) inner: Condition,
}

#[pymethods]
impl PyCondition {
    /// Create a comparison condition: field op value.
    #[staticmethod]
    fn compare(field: &str, op: &PyComparisonOp, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let json_val = py_to_json(value)?;
        Ok(PyCondition {
            inner: Condition::compare(field, op.inner, json_val),
        })
    }

    /// Check if a field exists in state.
    #[staticmethod]
    fn field_exists(field: &str) -> Self {
        PyCondition {
            inner: Condition::field_exists(field),
        }
    }

    /// Check if a field has a specific type.
    ///
    /// Types: "string", "number", "boolean", "array", "object", "null"
    #[staticmethod]
    fn field_type(field: &str, expected_type: &str) -> PyResult<Self> {
        let ft = match expected_type {
            "string" => FieldTypeCheck::String,
            "number" => FieldTypeCheck::Number,
            "boolean" | "bool" => FieldTypeCheck::Boolean,
            "array" | "list" => FieldTypeCheck::Array,
            "object" | "dict" => FieldTypeCheck::Object,
            "null" | "none" => FieldTypeCheck::Null,
            _ => {
                return Err(ValidationError::new_err(format!(
                    "Unknown type: '{}'. Use: string, number, boolean, array, object, null",
                    expected_type
                )))
            }
        };
        Ok(PyCondition {
            inner: Condition::field_type(field, ft),
        })
    }

    /// Logical AND of multiple conditions.
    #[staticmethod]
    fn and_conditions(conditions: Vec<PyCondition>) -> Self {
        PyCondition {
            inner: Condition::and(conditions.into_iter().map(|c| c.inner).collect()),
        }
    }

    /// Logical OR of multiple conditions.
    #[staticmethod]
    fn or_conditions(conditions: Vec<PyCondition>) -> Self {
        PyCondition {
            inner: Condition::or(conditions.into_iter().map(|c| c.inner).collect()),
        }
    }

    /// Logical NOT of a condition.
    #[staticmethod]
    fn not_condition(condition: &PyCondition) -> Self {
        PyCondition {
            inner: Condition::not_condition(condition.inner.clone()),
        }
    }

    /// Evaluate the condition against a state dict.
    ///
    /// Args:
    ///     state: The current state as a plain dict.
    ///
    /// Example:
    ///     cond = Condition.compare("score", ComparisonOp.greater_than(), 0.8)
    ///     if cond.evaluate({"score": 0.9, "messages": []}):
    ///         print("high score")
    fn evaluate(&self, state: &Bound<'_, PyDict>) -> PyResult<bool> {
        let shared = pydict_to_dynstate(state)?;
        Ok(self.inner.evaluate(&shared))
    }

    /// Simplify the condition (remove double negations, flatten nested AND/OR).
    fn simplify(&self) -> Self {
        PyCondition {
            inner: self.inner.clone().simplify(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Condition({})", self.inner.to_string_representation())
    }

    fn __str__(&self) -> String {
        self.inner.to_string_representation()
    }
}

// ─── PyConditionBuilder ─────────────────────────────────────────────────────

/// Builder for constructing complex conditions.
///
/// Example:
///     cond = (ConditionBuilder.and_builder()
///         .compare("confidence", ComparisonOp.greater_than(), 0.8)
///         .compare("attempts", ComparisonOp.less_than(), 3)
///         .field_exists("result")
///         .build())
#[pyclass(name = "ConditionBuilder")]
pub struct PyConditionBuilder {
    inner: ConditionBuilder,
}

#[pymethods]
impl PyConditionBuilder {
    /// Create a builder in AND mode.
    #[staticmethod]
    fn and_builder() -> Self {
        PyConditionBuilder {
            inner: ConditionBuilder::and(),
        }
    }

    /// Create a builder in OR mode.
    #[staticmethod]
    fn or_builder() -> Self {
        PyConditionBuilder {
            inner: ConditionBuilder::or(),
        }
    }

    /// Add a comparison condition.
    fn compare(
        &mut self,
        field: &str,
        op: &PyComparisonOp,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let json_val = py_to_json(value)?;
        let builder = std::mem::replace(&mut self.inner, ConditionBuilder::and());
        self.inner = builder.compare(field, op.inner, json_val);
        Ok(())
    }

    /// Add a field existence check.
    fn field_exists(&mut self, field: &str) {
        let builder = std::mem::replace(&mut self.inner, ConditionBuilder::and());
        self.inner = builder.field_exists(field);
    }

    /// Add an arbitrary condition.
    fn add_condition(&mut self, condition: &PyCondition) {
        let builder = std::mem::replace(&mut self.inner, ConditionBuilder::and());
        self.inner = builder.add_condition(condition.inner.clone());
    }

    /// Build the final condition.
    fn build(&mut self) -> PyCondition {
        let builder = std::mem::replace(&mut self.inner, ConditionBuilder::and());
        PyCondition {
            inner: builder.build(),
        }
    }

    fn __repr__(&self) -> String {
        "ConditionBuilder(...)".to_string()
    }
}
