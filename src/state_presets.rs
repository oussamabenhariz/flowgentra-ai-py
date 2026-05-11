//! Python bindings for preset state types
//!
//! Provides PyO3 wrappers for SimpleState, MessageState, AgentState, RAGState, etc.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use flowgentra_ai::core::state::{
    SimpleState, MessageState, AgentState, RAGState, EvaluationState, SupervisorState, AgentResult, State
};

use crate::{json_to_py, py_to_json};

// ─── PySimpleState ──────────────────────────────────────────────────────────

/// SimpleState - minimal state with input and output
#[pyclass(name = "SimpleState")]
#[derive(Clone)]
pub struct PySimpleState {
    pub(crate) inner: SimpleState,
}

#[pymethods]
impl PySimpleState {
    #[new]
    #[pyo3(signature = (input="", output=None))]
    fn new(input: &str, output: Option<&str>) -> Self {
        PySimpleState {
            inner: SimpleState {
                input: input.to_string(),
                output: output.map(|s| s.to_string()),
            },
        }
    }

    #[getter]
    fn input(&self) -> String {
        self.inner.input.clone()
    }

    #[setter]
    fn set_input(&mut self, value: String) {
        self.inner.input = value;
    }

    #[getter]
    fn output(&self) -> Option<String> {
        self.inner.output.clone()
    }

    #[setter]
    fn set_output(&mut self, value: Option<String>) {
        self.inner.output = value;
    }

    fn __repr__(&self) -> String {
        format!("SimpleState(input={:?}, output={:?})", self.inner.input, self.inner.output)
    }

    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("input", &self.inner.input)?;
            dict.set_item("output", &self.inner.output)?;
            Ok(dict.into())
        })
    }
}

// ─── PyMessageState ─────────────────────────────────────────────────────────

/// MessageState - state for message-based workflows
#[pyclass(name = "MessageState")]
#[derive(Clone)]
pub struct PyMessageState {
    pub(crate) inner: MessageState,
}

#[pymethods]
impl PyMessageState {
    #[new]
    #[pyo3(signature = ())]
    fn new() -> Self {
        PyMessageState {
            inner: MessageState::empty(),
        }
    }

    fn add_message(&mut self, msg: &crate::llm::PyMessage) -> PyResult<()> {
        self.inner.messages.push(msg.inner.clone());
        Ok(())
    }

    fn get_messages(&self, py: Python<'_>) -> PyResult<Vec<crate::llm::PyMessage>> {
        Ok(self.inner.messages
            .iter()
            .map(|m| crate::llm::PyMessage { inner: m.clone() })
            .collect())
    }

    #[getter]
    fn summary(&self) -> Option<String> {
        self.inner.summary.clone()
    }

    #[setter]
    fn set_summary(&mut self, value: Option<String>) {
        self.inner.summary = value;
    }

    fn __repr__(&self) -> String {
        format!("MessageState(messages_count={})", self.inner.messages.len())
    }
}

// ─── PyAgentState ───────────────────────────────────────────────────────────

/// AgentState - state for agent-based workflows
#[pyclass(name = "AgentState")]
#[derive(Clone)]
pub struct PyAgentState {
    pub(crate) inner: AgentState,
}

#[pymethods]
impl PyAgentState {
    #[new]
    fn new(query: &str) -> Self {
        PyAgentState {
            inner: AgentState::new(query),
        }
    }

    #[getter]
    fn query(&self) -> String {
        self.inner.query.clone()
    }

    #[setter]
    fn set_query(&mut self, value: String) {
        self.inner.query = value;
    }

    fn add_message(&mut self, msg: &crate::llm::PyMessage) -> PyResult<()> {
        self.inner.messages.push(msg.inner.clone());
        Ok(())
    }

    fn get_messages(&self, py: Python<'_>) -> PyResult<Vec<crate::llm::PyMessage>> {
        Ok(self.inner.messages
            .iter()
            .map(|m| crate::llm::PyMessage { inner: m.clone() })
            .collect())
    }

    #[getter]
    fn tools_used(&self) -> Vec<String> {
        self.inner.tools_used.clone()
    }

    fn add_tool_use(&mut self, tool_name: String) {
        self.inner.tools_used.push(tool_name);
    }

    #[getter]
    fn result(&self) -> Option<String> {
        self.inner.result.clone()
    }

    #[setter]
    fn set_result(&mut self, value: Option<String>) {
        self.inner.result = value;
    }

    fn __repr__(&self) -> String {
        format!("AgentState(query={:?}, messages={}, result={:?})",
            self.inner.query, self.inner.messages.len(), self.inner.result.is_some())
    }
}

// ─── PyRAGState ─────────────────────────────────────────────────────────────

/// RAGState - state for RAG-based workflows
#[pyclass(name = "RAGState")]
#[derive(Clone)]
pub struct PyRAGState {
    pub(crate) inner: RAGState,
}

#[pymethods]
impl PyRAGState {
    #[new]
    fn new(query: &str) -> Self {
        PyRAGState {
            inner: RAGState::new(query),
        }
    }

    #[getter]
    fn query(&self) -> String {
        self.inner.query.clone()
    }

    #[setter]
    fn set_query(&mut self, value: String) {
        self.inner.query = value;
    }

    #[getter]
    fn answer(&self) -> Option<String> {
        self.inner.answer.clone()
    }

    #[setter]
    fn set_answer(&mut self, value: Option<String>) {
        self.inner.answer = value;
    }

    fn __repr__(&self) -> String {
        format!("RAGState(query={:?}, documents={})", self.inner.query, self.inner.documents.len())
    }
}

// ─── PyEvaluationState ───────────────────────────────────────────────────────

/// EvaluationState - state for evaluation workflows
#[pyclass(name = "EvaluationState")]
#[derive(Clone)]
pub struct PyEvaluationState {
    pub(crate) inner: EvaluationState,
}

#[pymethods]
impl PyEvaluationState {
    #[new]
    fn new(input: &str, output: &str) -> Self {
        PyEvaluationState {
            inner: EvaluationState::new(input, output),
        }
    }

    #[getter]
    fn input(&self) -> String {
        self.inner.input.clone()
    }

    #[getter]
    fn output(&self) -> String {
        self.inner.output.clone()
    }

    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }

    #[setter]
    fn set_score(&mut self, value: f64) {
        self.inner.score = value;
    }

    #[getter]
    fn feedback(&self) -> String {
        self.inner.feedback.clone()
    }

    #[setter]
    fn set_feedback(&mut self, value: String) {
        self.inner.feedback = value;
    }

    #[getter]
    fn passed(&self) -> bool {
        self.inner.passed
    }

    #[setter]
    fn set_passed(&mut self, value: bool) {
        self.inner.passed = value;
    }

    fn __repr__(&self) -> String {
        format!("EvaluationState(score={:.2}, passed={})", self.inner.score, self.inner.passed)
    }
}

// ─── PySupervisorState ───────────────────────────────────────────────────────

/// SupervisorState - state for multi-agent orchestration
#[pyclass(name = "SupervisorState")]
#[derive(Clone)]
pub struct PySupervisorState {
    pub(crate) inner: SupervisorState,
}

#[pymethods]
impl PySupervisorState {
    #[new]
    fn new(task: &str) -> Self {
        PySupervisorState {
            inner: SupervisorState::new(task),
        }
    }

    #[getter]
    fn task(&self) -> String {
        self.inner.task.clone()
    }

    #[getter]
    fn final_result(&self) -> Option<String> {
        self.inner.final_result.clone()
    }

    #[setter]
    fn set_final_result(&mut self, value: Option<String>) {
        self.inner.final_result = value;
    }

    #[getter]
    fn status(&self) -> String {
        self.inner.status.clone()
    }

    #[setter]
    fn set_status(&mut self, value: String) {
        self.inner.status = value;
    }

    fn __repr__(&self) -> String {
        format!("SupervisorState(task={:?}, status={:?})", self.inner.task, self.inner.status)
    }
}
