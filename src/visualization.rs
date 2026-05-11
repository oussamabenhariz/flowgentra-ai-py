//! Python bindings for graph visualization

use pyo3::prelude::*;

use flowgentra_ai::core::utils::visualization::VisualizationConfig;

use crate::graph::PyStateGraph;

// ─── PyVisualizationConfig ──────────────────────────────────────────────────

/// Configuration for graph visualization (SVG output).
///
/// Example:
///     config = VisualizationConfig("my_graph.svg")
///     visualize(graph, config)
#[pyclass(name = "VisualizationConfig")]
#[derive(Clone)]
pub struct PyVisualizationConfig {
    inner: VisualizationConfig,
}

#[pymethods]
impl PyVisualizationConfig {
    #[new]
    #[pyo3(signature = (output_path="agent_graph.svg"))]
    fn new(output_path: &str) -> Self {
        PyVisualizationConfig {
            inner: VisualizationConfig::new(output_path),
        }
    }

    #[getter]
    fn output_path(&self) -> String {
        self.inner.output_path.clone()
    }

    fn __repr__(&self) -> String {
        format!("VisualizationConfig(output_path='{}')", self.inner.output_path)
    }
}

// ─── Free functions ──────────────────────────────────────────────────────────

/// Visualize a state graph, writing output to a file.
///
/// If the path ends with .dot, writes Graphviz DOT format.
/// Otherwise writes Mermaid diagram format.
///
/// Example:
///     visualize_graph_svg(graph, VisualizationConfig("my_graph.dot"))
#[pyfunction]
pub fn py_visualize_graph(graph: &PyStateGraph, config: &PyVisualizationConfig) -> PyResult<()> {
    let path = &config.inner.output_path;

    if path.ends_with(".dot") {
        let dot = graph.inner.to_dot();
        std::fs::write(path, &dot)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
    } else {
        let mermaid = graph.inner.to_mermaid();
        std::fs::write(path, &mermaid)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
    }

    Ok(())
}

/// Export graph as a DOT string (Graphviz format).
#[pyfunction]
pub fn py_graph_to_dot(graph: &PyStateGraph) -> String {
    graph.inner.to_dot()
}

/// Export graph as a Mermaid diagram string.
#[pyfunction]
pub fn py_graph_to_mermaid(graph: &PyStateGraph) -> String {
    graph.inner.to_mermaid()
}
