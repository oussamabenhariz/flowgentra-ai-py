//! Python bindings for Evaluation types

use pyo3::prelude::*;

// ─── PyEvaluationResult ─────────────────────────────────────────────────────

/// Result of an auto-evaluation node
#[pyclass(name = "EvaluationResult")]
#[derive(Clone)]
pub struct PyEvaluationResult {
    pub score: f64,
    pub passed: bool,
    pub feedback: String,
}

#[pymethods]
impl PyEvaluationResult {
    #[new]
    fn new(score: f64, passed: bool, feedback: String) -> Self {
        PyEvaluationResult {
            score,
            passed,
            feedback,
        }
    }

    #[getter]
    fn get_score(&self) -> f64 {
        self.score
    }

    #[getter]
    fn get_passed(&self) -> bool {
        self.passed
    }

    #[getter]
    fn get_feedback(&self) -> String {
        self.feedback.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "EvaluationResult(score={:.2}, passed={}, feedback='{}')",
            self.score, self.passed, self.feedback
        )
    }
}
