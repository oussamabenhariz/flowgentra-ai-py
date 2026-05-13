//! Python bindings for the EvaluationNode (iterative quality refinement)
//!
//! The evaluation node runs a handler function repeatedly, scoring the output
//! each time, until a confidence threshold is met or max retries are exhausted.
//! Handler callables use the LangGraph dict I/O pattern.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use flowgentra_ai::core::node::evaluation_node::EvaluationNodeConfig;
use flowgentra_ai::core::state::{Context, DynState, DynStateUpdate};
use flowgentra_ai::core::state_graph::node::Node;
use flowgentra_ai::core::state_graph::StateGraphError;

use crate::graph::dynstate_to_pydict;

// ─── PyEvaluationNodeConfig ─────────────────────────────────────────────────

/// Configuration for an evaluation node.
///
/// Example:
///     config = EvaluationNodeConfig(
///         name="refine",
///         field_state="llm_output",
///         min_confidence=0.8,
///         max_retries=3,
///         rubric="Is the output clear and accurate?",
///     )
#[pyclass(name = "EvaluationNodeConfig")]
#[derive(Clone)]
pub struct PyEvaluationNodeConfig {
    pub(crate) inner: EvaluationNodeConfig,
}

#[pymethods]
impl PyEvaluationNodeConfig {
    #[new]
    #[pyo3(signature = (name, field_state=None, min_confidence=0.8, max_retries=3, rubric=None))]
    fn new(
        name: &str,
        field_state: Option<String>,
        min_confidence: f64,
        max_retries: u32,
        rubric: Option<String>,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&min_confidence) {
            return Err(crate::error::AgentExecutionError::new_err(
                "min_confidence must be between 0.0 and 1.0",
            ));
        }
        Ok(PyEvaluationNodeConfig {
            inner: EvaluationNodeConfig {
                name: name.to_string(),
                handler: String::new(),
                field_state,
                min_confidence,
                max_retries,
                rubric,
                config: std::collections::HashMap::new(),
            },
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn field_state(&self) -> Option<String> {
        self.inner.field_state.clone()
    }

    #[getter]
    fn min_confidence(&self) -> f64 {
        self.inner.min_confidence
    }

    #[getter]
    fn max_retries(&self) -> u32 {
        self.inner.max_retries
    }

    #[getter]
    fn rubric(&self) -> Option<String> {
        self.inner.rubric.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "EvaluationNodeConfig(name='{}', field_state={:?}, min_confidence={}, max_retries={})",
            self.inner.name,
            self.inner.field_state,
            self.inner.min_confidence,
            self.inner.max_retries
        )
    }
}

// ─── EvaluationGraphNode ────────────────────────────────────────────────────

/// Graph-integrated evaluation node.
///
/// 1. Calls the handler with the current state dict → receives partial update dict.
/// 2. Scores the output using the scorer (or a built-in heuristic).
/// 3. If score >= min_confidence, commits the update and returns.
/// 4. Otherwise retries, passing feedback in `__eval_feedback__<name>`.
pub(crate) struct EvaluationGraphNode {
    pub name: String,
    pub handler: PyObject,
    pub scorer: Option<PyObject>,
    pub config: EvaluationNodeConfig,
    pub schema_fields: Arc<Vec<String>>,
}

#[async_trait::async_trait]
impl Node<DynState> for EvaluationGraphNode {
    async fn execute(
        &self,
        state: &DynState,
        _ctx: &Context,
    ) -> Result<DynStateUpdate, StateGraphError> {
        let handler = Python::with_gil(|py| self.handler.clone_ref(py));
        let scorer = self
            .scorer
            .as_ref()
            .map(|s| Python::with_gil(|py| s.clone_ref(py)));
        let field_state = self.config.get_field_name();
        let min_confidence = self.config.min_confidence;
        let max_retries = self.config.max_retries;
        let schema_fields = self.schema_fields.clone();
        let node_name = self.name.clone();
        let current_state = state.clone();

        for attempt in 0..=max_retries {
            // --- Call the handler (dict in, partial dict out) ---
            let call_result = Python::with_gil(|py| -> PyResult<()> {
                let f = handler.clone_ref(py);
                let state_dict = dynstate_to_pydict(py, &current_state)?;
                let py_result = f.call1(py, (state_dict,))?;

                let update = py_result.downcast_bound::<PyDict>(py).map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "EvaluationNode '{}' handler must return a dict",
                        node_name
                    ))
                })?;

                // Validate returned keys against schema
                for (k, _) in update.iter() {
                    let key: String = k.extract()?;
                    if !schema_fields.contains(&key) {
                        return Err(crate::error::AgentExecutionError::new_err(format!(
                            "EvaluationNode '{}' handler returned unknown key '{}'. \
                             Valid keys: {:?}",
                            node_name, key, schema_fields
                        )));
                    }
                }

                // Merge partial update into current state (for scoring and retry logic)
                for (k, v) in update.iter() {
                    let key: String = k.extract()?;
                    let val = crate::py_to_json(&v)?;
                    current_state.set(key, val);
                }
                Ok(())
            });

            call_result.map_err(|e| StateGraphError::ExecutionError {
                node: self.name.clone(),
                reason: format!("Handler error on attempt {}: {}", attempt, e),
            })?;

            // --- Score the output ---
            let (score, feedback) = if let Some(ref scorer_fn) = scorer {
                let output_val = field_state
                    .as_ref()
                    .and_then(|k| current_state.get(k))
                    .unwrap_or(serde_json::json!(null));

                Python::with_gil(|py| -> PyResult<(f64, String)> {
                    let s = scorer_fn.clone_ref(py);
                    let py_result =
                        s.call1(py, (crate::json_to_py(py, &output_val)?, attempt + 1))?;
                    py_result.extract(py)
                })
                .unwrap_or((0.0, "Scorer error".to_string()))
            } else {
                // Built-in heuristic: field must be non-null and non-empty
                let output_val = field_state.as_ref().and_then(|k| current_state.get(k));
                let score: f64 = match output_val {
                    Some(v) if !v.is_null() => {
                        if let Some(s) = v.as_str() {
                            if s.is_empty() {
                                0.3
                            } else {
                                0.85
                            }
                        } else {
                            0.85
                        }
                    }
                    _ => 0.0,
                };
                let feedback: String = if score >= min_confidence {
                    "OK".to_string()
                } else {
                    "Output missing or empty".to_string()
                };
                (score, feedback)
            };

            // Store internal evaluation metadata into the working state
            current_state.set(
                format!("__eval_score__{}", self.name),
                serde_json::json!(score),
            );
            current_state.set(
                format!("__eval_attempt__{}", self.name),
                serde_json::json!(attempt),
            );

            if score >= min_confidence {
                current_state.set(
                    format!("__eval_needs_retry__{}", self.name),
                    serde_json::json!(false),
                );
                // Return all keys from the accumulated state as the update
                let mut update = DynStateUpdate::new();
                for key in current_state.keys() {
                    if let Some(val) = current_state.get(&key) {
                        update.insert(key, val);
                    }
                }
                return Ok(update);
            }

            // Inject feedback for next attempt (internal key — handlers can read it)
            current_state.set(
                format!("__eval_feedback__{}", self.name),
                serde_json::json!(feedback),
            );
        }

        // Max retries exhausted — return all accumulated state as update
        current_state.set(
            format!("__eval_needs_retry__{}", self.name),
            serde_json::json!(false),
        );
        let mut update = DynStateUpdate::new();
        for key in current_state.keys() {
            if let Some(val) = current_state.get(&key) {
                update.insert(key, val);
            }
        }
        Ok(update)
    }

    fn name(&self) -> &str {
        &self.name
    }
}
