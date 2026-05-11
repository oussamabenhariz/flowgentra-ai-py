//! Python bindings for AgentConfig

use pyo3::prelude::*;

use flowgentra_ai::core::config::{AgentConfig, StateField};

use crate::error::to_py_err;

// ─── PyAgentConfig ──────────────────────────────────────────────────────────

/// Agent configuration loaded from YAML.
///
/// Example:
///     config = AgentConfig.from_file("config.yaml")
///     print(config.name)
///     print(config.description)
#[pyclass(name = "AgentConfig")]
#[derive(Clone)]
pub struct PyAgentConfig {
    pub(crate) inner: AgentConfig,
}

#[pymethods]
impl PyAgentConfig {
    /// Load config from a YAML file
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let config = AgentConfig::from_file(path).map_err(to_py_err)?;
        Ok(PyAgentConfig { inner: config })
    }

    /// Load config from a YAML string
    #[staticmethod]
    fn from_yaml(yaml_str: &str) -> PyResult<Self> {
        let config = AgentConfig::from_yaml_str(yaml_str).map_err(to_py_err)?;
        Ok(PyAgentConfig { inner: config })
    }

    /// Validate the config
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(to_py_err)
    }

    /// Agent name
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Agent description
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }

    /// Convert config to JSON string
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| crate::error::ConfigurationError::new_err(format!("{}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentConfig(name='{}', description={:?})",
            self.inner.name, self.inner.description
        )
    }
}

// ─── PyStateField ───────────────────────────────────────────────────────────

/// A field in the state schema
#[pyclass(name = "StateField")]
#[derive(Clone)]
pub struct PyStateField {
    pub(crate) inner: StateField,
}

#[pymethods]
impl PyStateField {
    #[new]
    fn new(field_type: &str, description: &str) -> Self {
        PyStateField {
            inner: StateField::new(field_type, description),
        }
    }

    #[getter]
    fn field_type(&self) -> String {
        self.inner.field_type.clone()
    }

    #[getter]
    fn description(&self) -> String {
        self.inner.description.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "StateField(type='{}', description='{}')",
            self.inner.field_type, self.inner.description
        )
    }
}
