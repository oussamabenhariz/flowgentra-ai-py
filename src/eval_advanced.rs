//! Python bindings for advanced evaluation:
//! ConfidenceScorer, NodeScorer, RetryPolicy, SmartFallback

use pyo3::prelude::*;

use flowgentra_ai::core::evaluation::{
    ConfidenceConfig, ConfidenceLevel, ConfidenceScore, ConfidenceScorer, FallbackLevel, NodeScore,
    NodeScorer, RetryConfig, RetryPolicy, RetryResult, ScoringCriteria, SmartFallback,
};
use flowgentra_ai::core::state::DynState;

use crate::py_to_json;

// ─── PyConfidenceLevel ───────────────────────────────────────────────────────

#[pyclass(name = "ConfidenceLevel")]
#[derive(Clone)]
pub struct PyConfidenceLevel {
    pub(crate) inner: ConfidenceLevel,
}

#[pymethods]
impl PyConfidenceLevel {
    #[staticmethod]
    fn very_low() -> Self {
        PyConfidenceLevel {
            inner: ConfidenceLevel::VeryLow,
        }
    }
    #[staticmethod]
    fn low() -> Self {
        PyConfidenceLevel {
            inner: ConfidenceLevel::Low,
        }
    }
    #[staticmethod]
    fn medium() -> Self {
        PyConfidenceLevel {
            inner: ConfidenceLevel::Medium,
        }
    }
    #[staticmethod]
    fn high() -> Self {
        PyConfidenceLevel {
            inner: ConfidenceLevel::High,
        }
    }
    #[staticmethod]
    fn very_high() -> Self {
        PyConfidenceLevel {
            inner: ConfidenceLevel::VeryHigh,
        }
    }

    fn __eq__(&self, other: &PyConfidenceLevel) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            ConfidenceLevel::VeryLow => "ConfidenceLevel.VeryLow",
            ConfidenceLevel::Low => "ConfidenceLevel.Low",
            ConfidenceLevel::Medium => "ConfidenceLevel.Medium",
            ConfidenceLevel::High => "ConfidenceLevel.High",
            ConfidenceLevel::VeryHigh => "ConfidenceLevel.VeryHigh",
        }
    }

    fn __str__(&self) -> &'static str {
        match self.inner {
            ConfidenceLevel::VeryLow => "very_low",
            ConfidenceLevel::Low => "low",
            ConfidenceLevel::Medium => "medium",
            ConfidenceLevel::High => "high",
            ConfidenceLevel::VeryHigh => "very_high",
        }
    }
}

// ─── PyConfidenceConfig ──────────────────────────────────────────────────────

#[pyclass(name = "ConfidenceConfig")]
#[derive(Clone)]
pub struct PyConfidenceConfig {
    pub(crate) inner: ConfidenceConfig,
}

#[pymethods]
impl PyConfidenceConfig {
    #[new]
    #[pyo3(signature = (
        clarity_weight = 0.3,
        relevance_weight = 0.4,
        completeness_weight = 0.3,
        low_threshold = 0.5,
        high_threshold = 0.8,
    ))]
    fn new(
        clarity_weight: f64,
        relevance_weight: f64,
        completeness_weight: f64,
        low_threshold: f64,
        high_threshold: f64,
    ) -> Self {
        PyConfidenceConfig {
            inner: ConfidenceConfig {
                clarity_weight,
                relevance_weight,
                completeness_weight,
                low_threshold,
                high_threshold,
            },
        }
    }

    #[staticmethod]
    fn defaults() -> Self {
        PyConfidenceConfig {
            inner: ConfidenceConfig::default(),
        }
    }

    #[getter]
    fn get_clarity_weight(&self) -> f64 {
        self.inner.clarity_weight
    }
    #[getter]
    fn get_relevance_weight(&self) -> f64 {
        self.inner.relevance_weight
    }
    #[getter]
    fn get_completeness_weight(&self) -> f64 {
        self.inner.completeness_weight
    }
    #[getter]
    fn get_low_threshold(&self) -> f64 {
        self.inner.low_threshold
    }
    #[getter]
    fn get_high_threshold(&self) -> f64 {
        self.inner.high_threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "ConfidenceConfig(clarity={}, relevance={}, completeness={}, low={}, high={})",
            self.inner.clarity_weight,
            self.inner.relevance_weight,
            self.inner.completeness_weight,
            self.inner.low_threshold,
            self.inner.high_threshold,
        )
    }
}

// ─── PyConfidenceScore ───────────────────────────────────────────────────────

#[pyclass(name = "ConfidenceScore")]
#[derive(Clone)]
pub struct PyConfidenceScore {
    pub(crate) inner: ConfidenceScore,
}

#[pymethods]
impl PyConfidenceScore {
    #[getter]
    fn get_overall(&self) -> f64 {
        self.inner.overall
    }
    #[getter]
    fn get_clarity(&self) -> f64 {
        self.inner.clarity
    }
    #[getter]
    fn get_relevance(&self) -> f64 {
        self.inner.relevance
    }
    #[getter]
    fn get_completeness(&self) -> f64 {
        self.inner.completeness
    }
    #[getter]
    fn get_level(&self) -> PyConfidenceLevel {
        PyConfidenceLevel {
            inner: self.inner.level.clone(),
        }
    }
    #[getter]
    fn get_indicators(&self) -> Vec<String> {
        self.inner.indicators.clone()
    }

    fn is_high_confidence(&self) -> bool {
        matches!(
            self.inner.level,
            ConfidenceLevel::High | ConfidenceLevel::VeryHigh
        )
    }

    fn is_low_confidence(&self) -> bool {
        matches!(
            self.inner.level,
            ConfidenceLevel::VeryLow | ConfidenceLevel::Low
        )
    }

    fn passes(&self, threshold: f64) -> bool {
        self.inner.overall >= threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "ConfidenceScore(overall={:.3}, level={})",
            self.inner.overall,
            PyConfidenceLevel {
                inner: self.inner.level.clone()
            }
            .__str__(),
        )
    }
}

// ─── py_score_confidence ─────────────────────────────────────────────────────

/// Score confidence in a value. Returns a ConfidenceScore with overall, clarity,
/// relevance, completeness dimensions.
///
/// Example:
///     score = score_confidence("This is a well-structured response.", task="Explain X")
///     print(score.overall, score.level)
#[pyfunction]
#[pyo3(signature = (output, task = None, node_name = "node", config = None))]
pub fn py_score_confidence(
    output: &Bound<'_, PyAny>,
    task: Option<&str>,
    node_name: &str,
    config: Option<&PyConfidenceConfig>,
) -> PyResult<PyConfidenceScore> {
    let val = py_to_json(output)?;
    let state = DynState::new();
    let default_cfg;
    let cfg = match config {
        Some(c) => &c.inner,
        None => {
            default_cfg = ConfidenceConfig::default();
            &default_cfg
        }
    };
    let score = ConfidenceScorer::score(&val, task, &state, node_name, cfg);
    Ok(PyConfidenceScore { inner: score })
}

// ─── PyScoringCriteria ───────────────────────────────────────────────────────

#[pyclass(name = "ScoringCriteria")]
#[derive(Clone)]
pub struct PyScoringCriteria {
    pub(crate) inner: ScoringCriteria,
}

#[pymethods]
impl PyScoringCriteria {
    #[new]
    #[pyo3(signature = (
        check_empty = true,
        check_validity = true,
        check_usefulness = true,
        check_consistency = true,
        min_length = 1,
        max_length = 0,
    ))]
    fn new(
        check_empty: bool,
        check_validity: bool,
        check_usefulness: bool,
        check_consistency: bool,
        min_length: usize,
        max_length: usize,
    ) -> Self {
        PyScoringCriteria {
            inner: ScoringCriteria {
                check_empty,
                check_validity,
                check_usefulness,
                check_consistency,
                min_length,
                max_length,
            },
        }
    }

    #[staticmethod]
    fn defaults() -> Self {
        PyScoringCriteria {
            inner: ScoringCriteria::default(),
        }
    }

    #[getter]
    fn get_check_empty(&self) -> bool {
        self.inner.check_empty
    }
    #[getter]
    fn get_check_validity(&self) -> bool {
        self.inner.check_validity
    }
    #[getter]
    fn get_check_usefulness(&self) -> bool {
        self.inner.check_usefulness
    }
    #[getter]
    fn get_check_consistency(&self) -> bool {
        self.inner.check_consistency
    }
    #[getter]
    fn get_min_length(&self) -> usize {
        self.inner.min_length
    }
    #[getter]
    fn get_max_length(&self) -> usize {
        self.inner.max_length
    }

    fn __repr__(&self) -> String {
        format!(
            "ScoringCriteria(check_empty={}, check_validity={}, check_usefulness={}, check_consistency={})",
            self.inner.check_empty,
            self.inner.check_validity,
            self.inner.check_usefulness,
            self.inner.check_consistency,
        )
    }
}

// ─── PyNodeScore ─────────────────────────────────────────────────────────────

#[pyclass(name = "NodeScore")]
#[derive(Clone)]
pub struct PyNodeScore {
    pub(crate) inner: NodeScore,
}

#[pymethods]
impl PyNodeScore {
    #[getter]
    fn get_overall(&self) -> f64 {
        self.inner.overall
    }
    #[getter]
    fn get_completeness(&self) -> f64 {
        self.inner.completeness
    }
    #[getter]
    fn get_validity(&self) -> f64 {
        self.inner.validity
    }
    #[getter]
    fn get_usefulness(&self) -> f64 {
        self.inner.usefulness
    }
    #[getter]
    fn get_consistency(&self) -> f64 {
        self.inner.consistency
    }
    #[getter]
    fn get_explanation(&self) -> String {
        self.inner.explanation.clone()
    }

    fn passes(&self, threshold: f64) -> bool {
        self.inner.overall >= threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "NodeScore(overall={:.3}, completeness={:.3}, validity={:.3}, usefulness={:.3})",
            self.inner.overall, self.inner.completeness, self.inner.validity, self.inner.usefulness,
        )
    }
}

// ─── py_score_node ───────────────────────────────────────────────────────────

/// Score a node output on completeness, validity, usefulness, and consistency.
///
/// Example:
///     score = score_node({"result": "success", "count": 42})
///     print(score.overall, score.explanation)
#[pyfunction]
#[pyo3(signature = (output, criteria = None, node_name = "node"))]
pub fn py_score_node(
    output: &Bound<'_, PyAny>,
    criteria: Option<&PyScoringCriteria>,
    node_name: &str,
) -> PyResult<PyNodeScore> {
    let val = py_to_json(output)?;
    let state = DynState::new();
    let default_criteria;
    let c = match criteria {
        Some(c) => &c.inner,
        None => {
            default_criteria = ScoringCriteria::default();
            &default_criteria
        }
    };
    let score = NodeScorer::score(&val, c, &state, node_name);
    Ok(PyNodeScore { inner: score })
}

// ─── PyRetryConfig ───────────────────────────────────────────────────────────

#[pyclass(name = "RetryConfig")]
#[derive(Clone)]
pub struct PyRetryConfig {
    pub(crate) inner: RetryConfig,
}

#[pymethods]
impl PyRetryConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        max_retries = 3,
        initial_delay_ms = 100,
        backoff_multiplier = 2.0,
        max_delay_ms = 5000,
        confidence_threshold = 0.65,
        include_feedback = true,
        increase_temperature = true,
        enable_circuit_breaker = true,
        circuit_breaker_threshold = 3,
    ))]
    fn new(
        max_retries: u32,
        initial_delay_ms: u64,
        backoff_multiplier: f64,
        max_delay_ms: u64,
        confidence_threshold: f64,
        include_feedback: bool,
        increase_temperature: bool,
        enable_circuit_breaker: bool,
        circuit_breaker_threshold: u32,
    ) -> Self {
        PyRetryConfig {
            inner: RetryConfig {
                max_retries,
                initial_delay_ms,
                backoff_multiplier,
                max_delay_ms,
                confidence_threshold,
                include_feedback,
                increase_temperature,
                enable_circuit_breaker,
                circuit_breaker_threshold,
            },
        }
    }

    #[staticmethod]
    fn defaults() -> Self {
        PyRetryConfig {
            inner: RetryConfig::default(),
        }
    }

    #[getter]
    fn get_max_retries(&self) -> u32 {
        self.inner.max_retries
    }
    #[getter]
    fn get_initial_delay_ms(&self) -> u64 {
        self.inner.initial_delay_ms
    }
    #[getter]
    fn get_backoff_multiplier(&self) -> f64 {
        self.inner.backoff_multiplier
    }
    #[getter]
    fn get_max_delay_ms(&self) -> u64 {
        self.inner.max_delay_ms
    }
    #[getter]
    fn get_confidence_threshold(&self) -> f64 {
        self.inner.confidence_threshold
    }
    #[getter]
    fn get_include_feedback(&self) -> bool {
        self.inner.include_feedback
    }
    #[getter]
    fn get_increase_temperature(&self) -> bool {
        self.inner.increase_temperature
    }
    #[getter]
    fn get_enable_circuit_breaker(&self) -> bool {
        self.inner.enable_circuit_breaker
    }
    #[getter]
    fn get_circuit_breaker_threshold(&self) -> u32 {
        self.inner.circuit_breaker_threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "RetryConfig(max_retries={}, confidence_threshold={:.2})",
            self.inner.max_retries, self.inner.confidence_threshold,
        )
    }
}

// ─── PyRetryResult ───────────────────────────────────────────────────────────

#[pyclass(name = "RetryResult")]
#[derive(Clone)]
pub struct PyRetryResult {
    pub(crate) inner: RetryResult,
}

#[pymethods]
impl PyRetryResult {
    #[staticmethod]
    fn no_retry() -> Self {
        PyRetryResult {
            inner: RetryResult::new_no_retry(),
        }
    }

    #[staticmethod]
    fn retried(
        retry_count: u32,
        confidence_history: Vec<f64>,
        success: bool,
        stop_reason: String,
    ) -> Self {
        PyRetryResult {
            inner: RetryResult::new_retried(retry_count, confidence_history, success, stop_reason),
        }
    }

    #[getter]
    fn get_was_retried(&self) -> bool {
        self.inner.was_retried
    }
    #[getter]
    fn get_retry_count(&self) -> u32 {
        self.inner.retry_count
    }
    #[getter]
    fn get_confidence_history(&self) -> Vec<f64> {
        self.inner.confidence_history.clone()
    }
    #[getter]
    fn get_success(&self) -> bool {
        self.inner.success
    }
    #[getter]
    fn get_improvement(&self) -> f64 {
        self.inner.improvement
    }
    #[getter]
    fn get_stop_reason(&self) -> String {
        self.inner.stop_reason.clone()
    }

    fn generate_report(&self) -> String {
        RetryPolicy::generate_report(&self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "RetryResult(was_retried={}, retry_count={}, success={}, improvement={:.3})",
            self.inner.was_retried,
            self.inner.retry_count,
            self.inner.success,
            self.inner.improvement,
        )
    }
}

// ─── RetryPolicy functions ───────────────────────────────────────────────────

/// Check if output should be retried given confidence and retry count.
#[pyfunction]
pub fn py_retry_should_retry(confidence: f64, retry_count: u32, config: &PyRetryConfig) -> bool {
    RetryPolicy::should_retry(confidence, retry_count, &config.inner)
}

/// Get the delay in milliseconds before the next retry attempt (exponential backoff).
#[pyfunction]
pub fn py_retry_delay_ms(retry_count: u32, config: &PyRetryConfig) -> u64 {
    RetryPolicy::get_retry_delay(retry_count, &config.inner)
}

/// Get the recommended LLM temperature for a retry attempt (increases slightly each retry).
#[pyfunction]
pub fn py_retry_temperature(retry_count: u32) -> f64 {
    RetryPolicy::get_temperature_adjustment(retry_count)
}

/// Build a retry feedback prompt to inject into the next LLM call.
#[pyfunction]
pub fn py_retry_feedback(feedback: &str, issues: Vec<String>, suggestions: Vec<String>) -> String {
    RetryPolicy::build_retry_feedback(feedback, &issues, &suggestions)
}

/// Check whether the circuit breaker should stop retries.
#[pyfunction]
pub fn py_check_circuit_breaker(consecutive_failures: u32, config: &PyRetryConfig) -> bool {
    RetryPolicy::check_circuit_breaker(consecutive_failures, &config.inner)
}

/// Generate a human-readable report for a RetryResult.
#[pyfunction]
pub fn py_retry_generate_report(result: &PyRetryResult) -> String {
    RetryPolicy::generate_report(&result.inner)
}

// ─── PyFallbackLevel ─────────────────────────────────────────────────────────

#[pyclass(name = "FallbackLevel")]
#[derive(Clone)]
pub struct PyFallbackLevel {
    pub(crate) inner: FallbackLevel,
}

#[pymethods]
impl PyFallbackLevel {
    #[staticmethod]
    fn initial() -> Self {
        PyFallbackLevel {
            inner: FallbackLevel::Initial,
        }
    }
    #[staticmethod]
    fn degraded() -> Self {
        PyFallbackLevel {
            inner: FallbackLevel::Degraded,
        }
    }
    #[staticmethod]
    fn minimal() -> Self {
        PyFallbackLevel {
            inner: FallbackLevel::Minimal,
        }
    }
    #[staticmethod]
    fn template() -> Self {
        PyFallbackLevel {
            inner: FallbackLevel::Template,
        }
    }

    /// Get the fallback level for a given number of retries (0→Initial, 1→Degraded, …).
    #[staticmethod]
    fn from_retries(retries: u32) -> Self {
        PyFallbackLevel {
            inner: FallbackLevel::from_retries(retries),
        }
    }

    fn __repr__(&self) -> String {
        format!("FallbackLevel({:?})", self.inner)
    }
}

// ─── SmartFallback functions ─────────────────────────────────────────────────

/// Generate fallback content for a topic based on how many retries have been attempted.
///
/// Example:
///     level = FallbackLevel.from_retries(2)
///     content = generate_content_fallback("Rust programming", level)
#[pyfunction]
#[pyo3(signature = (topic, level, previous = None))]
pub fn py_generate_content_fallback(
    topic: &str,
    level: &PyFallbackLevel,
    previous: Option<&str>,
) -> String {
    SmartFallback::generate_content_fallback(topic, level.inner.clone(), previous)
}

/// Progressively simplify existing content based on retry level.
#[pyfunction]
pub fn py_refine_content_fallback(content: &str, level: &PyFallbackLevel) -> String {
    SmartFallback::refine_content_fallback(content, level.inner.clone())
}

/// Generate a retry instruction message for the LLM based on fallback level.
#[pyfunction]
#[pyo3(signature = (level, previous_feedback = None))]
pub fn py_fallback_retry_message(
    level: &PyFallbackLevel,
    previous_feedback: Option<&str>,
) -> String {
    SmartFallback::retry_message(level.inner.clone(), previous_feedback)
}

/// Decide whether to stop retrying and use a fallback response.
#[pyfunction]
pub fn py_should_fallback(retry_count: u32, max_retries: u32) -> bool {
    SmartFallback::should_fallback(retry_count, max_retries)
}
