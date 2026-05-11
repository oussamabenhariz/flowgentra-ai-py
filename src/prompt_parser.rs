//! Python bindings for PromptTemplate and output parsers

use pyo3::prelude::*;
use crate::error::ValidationError;
use std::collections::HashMap;

use flowgentra_ai::core::llm::prompt_template::PromptTemplate;
use flowgentra_ai::core::llm::output_parser::{
    JsonOutputParser, ListOutputParser, OutputParser,
};

use crate::json_to_py;

// ─── PyPromptTemplate ──────────────────────────────────────────────────────

/// A prompt template with variable substitution.
///
/// Variables are delimited by curly braces: {variable_name}
///
/// Example:
///     tmpl = PromptTemplate("Hello {name}, you are a {role}")
///     result = tmpl.format({"name": "Alice", "role": "developer"})
///     print(result)  # "Hello Alice, you are a developer"
#[pyclass(name = "PromptTemplate")]
pub struct PyPromptTemplate {
    inner: PromptTemplate,
}

#[pymethods]
impl PyPromptTemplate {
    #[new]
    fn new(template: &str) -> Self {
        PyPromptTemplate {
            inner: PromptTemplate::new(template),
        }
    }

    /// Get the list of variable names found in the template.
    fn input_variables(&self) -> Vec<String> {
        self.inner.input_variables().to_vec()
    }

    /// Format the template with the given variables dict.
    fn format(&self, variables: HashMap<String, String>) -> PyResult<String> {
        let pairs: Vec<(&str, &str)> = variables
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        self.inner
            .format(&pairs)
            .map_err(|e| ValidationError::new_err(format!("{}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "PromptTemplate(vars={:?})",
            self.inner.input_variables()
        )
    }
}

// ─── PyJsonOutputParser ────────────────────────────────────────────────────

/// Parse JSON output from LLM responses.
///
/// Handles markdown code blocks, extra text around JSON, etc.
///
/// Example:
///     parser = JsonOutputParser()
///     data = parser.parse('Here is the result: ```json\n{"key": "value"}\n```')
///     print(data)  # {"key": "value"}
#[pyclass(name = "JsonOutputParser")]
pub struct PyJsonOutputParser {
    inner: JsonOutputParser,
}

#[pymethods]
impl PyJsonOutputParser {
    #[new]
    #[pyo3(signature = (schema_hint=None))]
    fn new(schema_hint: Option<String>) -> Self {
        let mut parser = JsonOutputParser::new();
        if let Some(hint) = schema_hint {
            parser = parser.with_schema(hint);
        }
        PyJsonOutputParser { inner: parser }
    }

    /// Parse JSON from text (handles code blocks, extra text, etc.)
    fn parse(&self, py: Python<'_>, text: &str) -> PyResult<PyObject> {
        let val = self
            .inner
            .parse(text)
            .map_err(|e| ValidationError::new_err(format!("{}", e)))?;
        json_to_py(py, &val)
    }

    /// Get format instructions to include in your prompt.
    fn format_instructions(&self) -> String {
        self.inner.format_instructions()
    }

    fn __repr__(&self) -> String {
        "JsonOutputParser(...)".to_string()
    }
}

// ─── PyListOutputParser ────────────────────────────────────────────────────

/// Parse a list from LLM output.
///
/// Example:
///     parser = ListOutputParser("comma")
///     items = parser.parse("apple, banana, cherry")
///     print(items)  # ["apple", "banana", "cherry"]
#[pyclass(name = "ListOutputParser")]
pub struct PyListOutputParser {
    inner: ListOutputParser,
}

#[pymethods]
impl PyListOutputParser {
    /// Create a list output parser.
    ///
    /// Args:
    ///     separator: "comma", "newline", or "numbered"
    #[new]
    #[pyo3(signature = (separator="comma"))]
    fn new(separator: &str) -> Self {
        let parser = match separator {
            "newline" => ListOutputParser::newline_separated(),
            "numbered" => ListOutputParser::numbered(),
            _ => ListOutputParser::comma_separated(),
        };
        PyListOutputParser {
            inner: parser,
        }
    }

    /// Parse a list from text.
    fn parse(&self, text: &str) -> PyResult<Vec<String>> {
        self.inner
            .parse(text)
            .map_err(|e| ValidationError::new_err(format!("{}", e)))
    }

    /// Get format instructions to include in your prompt.
    fn format_instructions(&self) -> String {
        self.inner.format_instructions()
    }

    fn __repr__(&self) -> String {
        "ListOutputParser(...)".to_string()
    }
}
