//! Python bindings for evaluation configuration and reporting types

use pyo3::prelude::*;

use flowgentra_ai::core::config::{EvaluationConfig, GradingConfig, ScoringConfig};
use flowgentra_ai::core::evaluation::reporting::{EvaluationReport, NodeResult};
use crate::state::PyState;
use crate::error::to_py_err;

// ─── PyScoringConfig ────────────────────────────────────────────────────────

/// Scoring sub-configuration for evaluation.
///
/// Example:
///     scoring = ScoringConfig(
///         metrics=["relevance", "completeness"],
///         weights=[0.6, 0.4],
///     )
#[pyclass(name = "ScoringConfig")]
#[derive(Clone)]
pub struct PyScoringConfig {
    pub(crate) inner: ScoringConfig,
}

#[pymethods]
impl PyScoringConfig {
    #[new]
    #[pyo3(signature = (metrics=vec![], weights=vec![]))]
    fn new(metrics: Vec<String>, weights: Vec<f64>) -> Self {
        PyScoringConfig {
            inner: ScoringConfig { metrics, weights },
        }
    }

    #[getter]
    fn metrics(&self) -> Vec<String> {
        self.inner.metrics.clone()
    }

    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.inner.weights.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ScoringConfig(metrics={:?}, weights={:?})",
            self.inner.metrics, self.inner.weights
        )
    }
}

// ─── PyGradingConfig ────────────────────────────────────────────────────────

/// Grading sub-configuration for evaluation.
///
/// Example:
///     grading = GradingConfig(enabled=True, rubric="Grade on accuracy and clarity")
#[pyclass(name = "GradingConfig")]
#[derive(Clone)]
pub struct PyGradingConfig {
    pub(crate) inner: GradingConfig,
}

#[pymethods]
impl PyGradingConfig {
    #[new]
    #[pyo3(signature = (enabled=false, rubric=None))]
    fn new(enabled: bool, rubric: Option<String>) -> Self {
        PyGradingConfig {
            inner: GradingConfig { enabled, rubric },
        }
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
    }

    #[getter]
    fn rubric(&self) -> Option<String> {
        self.inner.rubric.clone()
    }

    fn __repr__(&self) -> String {
        format!("GradingConfig(enabled={})", self.inner.enabled)
    }
}

// ─── PyEvaluationConfig ─────────────────────────────────────────────────────

/// Evaluation configuration for automatic agent evaluation.
///
/// Example:
///     eval_cfg = EvaluationConfig(
///         enabled=True,
///         min_confidence=0.8,
///         max_retries=3,
///     )
#[pyclass(name = "EvaluationConfig")]
#[derive(Clone)]
pub struct PyEvaluationConfig {
    pub(crate) inner: EvaluationConfig,
}

#[pymethods]
impl PyEvaluationConfig {
    #[new]
    #[pyo3(signature = (
        enabled=true,
        min_confidence=0.8,
        max_retries=3,
        scoring=None,
        grading=None,
        retry_policy=None,
        retry_delay_ms=None,
    ))]
    fn new(
        enabled: bool,
        min_confidence: f64,
        max_retries: u32,
        scoring: Option<&PyScoringConfig>,
        grading: Option<&PyGradingConfig>,
        retry_policy: Option<String>,
        retry_delay_ms: Option<u64>,
    ) -> Self {
        PyEvaluationConfig {
            inner: EvaluationConfig {
                enabled,
                min_confidence,
                max_retries,
                scoring: scoring.map(|s| s.inner.clone()),
                grading: grading.map(|g| g.inner.clone()),
                retry_policy,
                retry_delay_ms,
            },
        }
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
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
    fn scoring(&self) -> Option<PyScoringConfig> {
        self.inner.scoring.as_ref().map(|s| PyScoringConfig { inner: s.clone() })
    }

    #[getter]
    fn grading(&self) -> Option<PyGradingConfig> {
        self.inner.grading.as_ref().map(|g| PyGradingConfig { inner: g.clone() })
    }

    #[getter]
    fn retry_policy(&self) -> Option<String> {
        self.inner.retry_policy.clone()
    }

    #[getter]
    fn retry_delay_ms(&self) -> Option<u64> {
        self.inner.retry_delay_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "EvaluationConfig(enabled={}, min_confidence={}, max_retries={})",
            self.inner.enabled, self.inner.min_confidence, self.inner.max_retries
        )
    }
}

// ─── PyNodeResult ───────────────────────────────────────────────────────────

/// Result for a single node in an evaluation report.
#[pyclass(name = "NodeResult")]
#[derive(Clone)]
pub struct PyNodeResult {
    inner: NodeResult,
}

#[pymethods]
impl PyNodeResult {
    #[getter]
    fn node_name(&self) -> String {
        self.inner.node_name.clone()
    }

    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }

    #[getter]
    fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    #[getter]
    fn retries(&self) -> u32 {
        self.inner.retries
    }

    #[getter]
    fn issues(&self) -> Vec<String> {
        self.inner.issues.clone()
    }

    #[getter]
    fn suggestions(&self) -> Vec<String> {
        self.inner.suggestions.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "NodeResult(node='{}', score={:.2}, confidence={:.2})",
            self.inner.node_name, self.inner.score, self.inner.confidence
        )
    }
}

// ─── PyEvaluationReport ─────────────────────────────────────────────────────

/// Complete evaluation report extracted from agent state.
///
/// Example:
///     report = EvaluationReport.from_state(state)
///     print(report.overall_score)
///     print(report.passed)
///     report.save_json("report.json")
#[pyclass(name = "EvaluationReport")]
#[derive(Clone)]
pub struct PyEvaluationReport {
    inner: EvaluationReport,
}

#[pymethods]
impl PyEvaluationReport {
    /// Extract an evaluation report from agent state.
    #[staticmethod]
    fn from_state(state: &PyState) -> Self {
        let report = EvaluationReport::from_state(&state.inner);
        PyEvaluationReport { inner: report }
    }

    #[getter]
    fn nodes(&self) -> Vec<PyNodeResult> {
        self.inner
            .nodes
            .iter()
            .map(|n| PyNodeResult { inner: n.clone() })
            .collect()
    }

    #[getter]
    fn overall_score(&self) -> f64 {
        self.inner.overall_score
    }

    #[getter]
    fn total_retries(&self) -> u32 {
        self.inner.total_retries
    }

    #[getter]
    fn passed(&self) -> bool {
        self.inner.passed
    }

    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.clone()
    }

    /// Save report to a JSON file.
    fn save_json(&self, path: &str) -> PyResult<()> {
        self.inner.save_json(path).map_err(to_py_err)
    }

    /// Get report as a JSON-compatible Python dict.
    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let val = self.inner.to_json();
        crate::json_to_py(py, &val)
    }

    /// Print formatted report to console.
    fn print(&self) {
        self.inner.print();
    }

    /// Print a one-line summary.
    fn print_summary(&self) {
        self.inner.print_summary();
    }

    fn __repr__(&self) -> String {
        format!(
            "EvaluationReport(score={:.2}, passed={}, retries={})",
            self.inner.overall_score, self.inner.passed, self.inner.total_retries
        )
    }
}
