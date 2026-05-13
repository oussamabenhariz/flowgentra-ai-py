//! Python bindings for ToolRegistry and related types

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

use flowgentra_ai::core::tools::{JsonSchema, Tool, ToolRegistry};

use crate::builtin_tools::{PyCalculatorTool, PyFilesTool, PyWebRequestTool};
use crate::error::to_py_err;
use crate::py_code_exec::{PyNodeJsReplTool, PyPythonReplTool, PyShellTool};
use crate::py_communication::{PyGmailTool, PySlackTool};
use crate::py_data::{PyCsvQueryTool, PyJsonGetValueTool, PyJsonListKeysTool};
use crate::py_external_apis::{PyAlphaVantageTool, PyNewsApiTool, PyOpenWeatherMapTool};
use crate::py_files_extended::{
    PyCopyFileTool, PyDeleteFileTool, PyFileSearchTool, PyMoveFileTool,
};
use crate::py_human::PyHumanInputTool;
use crate::py_knowledge::{PyArxivTool, PyPubMedTool, PyWikipediaTool, PyWolframAlphaTool};
use crate::py_search::{
    PyBraveSearchTool, PyDuckDuckGoSearchTool, PyGoogleSerperTool, PySerpApiSearchTool,
    PyTavilySearchTool,
};
use crate::{json_to_py, py_to_json};

// ─── Helper Function for Tool Extraction ────────────────────────────────────

/// Extract `Arc<dyn Tool>` from any registered Python tool type.
fn extract_tool_arc(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn Tool>> {
    macro_rules! try_extract {
        ($($ty:ty),* $(,)?) => {
            $(
                if let Ok(t) = obj.extract::<PyRef<$ty>>() {
                    return Ok(t.inner.clone() as Arc<dyn Tool>);
                }
            )*
        };
    }

    try_extract!(
        // Core built-ins
        PyCalculatorTool,
        PyWebRequestTool,
        PyFilesTool,
        // Search
        PyDuckDuckGoSearchTool,
        PyTavilySearchTool,
        PySerpApiSearchTool,
        PyGoogleSerperTool,
        PyBraveSearchTool,
        // Knowledge
        PyWikipediaTool,
        PyArxivTool,
        PyPubMedTool,
        PyWolframAlphaTool,
        // Code execution
        PyPythonReplTool,
        PyNodeJsReplTool,
        PyShellTool,
        // Extended file ops
        PyCopyFileTool,
        PyDeleteFileTool,
        PyMoveFileTool,
        PyFileSearchTool,
        // Data
        PyJsonGetValueTool,
        PyJsonListKeysTool,
        PyCsvQueryTool,
        // Human
        PyHumanInputTool,
        // Communication
        PyGmailTool,
        PySlackTool,
        // External APIs
        PyOpenWeatherMapTool,
        PyNewsApiTool,
        PyAlphaVantageTool,
    );

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Object is not a recognized Flowgentra tool type",
    ))
}

// ─── PyJsonSchema ──────────────────────────────────────────────────────────

#[pyclass(name = "JsonSchema")]
#[derive(Clone)]
pub struct PyJsonSchema {
    pub(crate) inner: JsonSchema,
}

#[pymethods]
impl PyJsonSchema {
    #[staticmethod]
    fn object() -> Self {
        PyJsonSchema {
            inner: JsonSchema::object(),
        }
    }
    #[staticmethod]
    fn string() -> Self {
        PyJsonSchema {
            inner: JsonSchema::string(),
        }
    }
    #[staticmethod]
    fn number() -> Self {
        PyJsonSchema {
            inner: JsonSchema::number(),
        }
    }
    #[staticmethod]
    fn integer() -> Self {
        PyJsonSchema {
            inner: JsonSchema::integer(),
        }
    }
    #[staticmethod]
    fn boolean() -> Self {
        PyJsonSchema {
            inner: JsonSchema::boolean(),
        }
    }
    #[staticmethod]
    fn array() -> Self {
        PyJsonSchema {
            inner: JsonSchema::array(),
        }
    }

    fn with_description(&mut self, desc: &str) {
        self.inner.description = Some(desc.to_string());
    }

    fn with_required(&mut self, fields: Vec<String>) {
        self.inner.required = Some(fields);
    }

    fn validate(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let val = py_to_json(value)?;
        self.inner.validate(&val).map_err(to_py_err)
    }

    #[getter]
    fn schema_type(&self) -> String {
        self.inner.schema_type.clone()
    }

    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }

    fn __repr__(&self) -> String {
        format!("JsonSchema(type='{}')", self.inner.schema_type)
    }
}

// ─── PyToolRegistry ────────────────────────────────────────────────────────

#[pyclass(name = "ToolRegistry")]
pub struct PyToolRegistry {
    inner: ToolRegistry,
}

#[pymethods]
impl PyToolRegistry {
    /// Create a registry, optionally pre-seeded with tools.
    ///
    /// Args:
    ///     tools: dict mapping name → tool, or list of tools (name taken from definition)
    #[new]
    #[pyo3(signature = (tools=None))]
    fn new(tools: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut registry = PyToolRegistry {
            inner: ToolRegistry::new(),
        };

        if let Some(tools_arg) = tools {
            if let Ok(dict) = tools_arg.downcast::<PyDict>() {
                for (key, value) in dict.iter() {
                    let name: String = key.extract()?;
                    let arc = extract_tool_arc(&value)?;
                    registry.inner.register(&name, arc).map_err(to_py_err)?;
                }
            } else if let Ok(list) = tools_arg.downcast::<PyList>() {
                for item in list.iter() {
                    let arc = extract_tool_arc(&item)?;
                    let def = arc.definition();
                    registry.inner.register(&def.name, arc).map_err(to_py_err)?;
                }
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "tools must be a dict or list",
                ));
            }
        }

        Ok(registry)
    }

    /// Create a registry with all keyless, non-destructive built-in tools pre-registered.
    #[staticmethod]
    fn with_builtins() -> Self {
        PyToolRegistry {
            inner: ToolRegistry::with_builtins(),
        }
    }

    /// Call a tool by name with a dict input.
    fn call_tool(
        &self,
        py: Python<'_>,
        name: &str,
        input: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let val = py_to_json(input)?;
        let result = crate::run_async(self.inner.call_tool(name, val)).map_err(to_py_err)?;
        json_to_py(py, &result)
    }

    /// Validate tool input without executing it.
    fn validate_input(&self, name: &str, input: &Bound<'_, PyAny>) -> PyResult<()> {
        let val = py_to_json(input)?;
        self.inner.validate_input(name, &val).map_err(to_py_err)
    }

    /// List all registered tool names.
    fn list_names(&self) -> Vec<String> {
        self.inner
            .list_definitions()
            .into_iter()
            .map(|d| d.name)
            .collect()
    }

    /// Check if a tool is registered.
    fn has(&self, name: &str) -> bool {
        self.inner.has(name)
    }

    /// Get tool definition as a dict.
    fn get(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyDict>> {
        let definition = self.inner.get_definition(name).map_err(to_py_err)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("name", &definition.name)?;
        dict.set_item("description", &definition.description)?;
        Ok(dict.into())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("ToolRegistry(tools={})", self.inner.len())
    }
}
