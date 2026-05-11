//! Python bindings for MessageGraphBuilder

use std::sync::Arc;

use pyo3::prelude::*;

use flowgentra_ai::core::llm::Message;
use flowgentra_ai::core::state::Context;
use flowgentra_ai::core::state_graph::message_graph::{
    MessageGraphBuilder, MessageState, MessageStateUpdate,
};
use flowgentra_ai::core::state_graph::node::FunctionNode;
use flowgentra_ai::core::state_graph::StateGraph;

use crate::error::to_py_err_generic;
use crate::llm::PyMessage;

// ─── PyMessageGraph (compiled) ──────────────────────────────────────────────

/// A compiled message-centric graph.
#[pyclass(name = "MessageGraph")]
pub struct PyMessageGraph {
    inner: StateGraph<MessageState>,
}

#[pymethods]
impl PyMessageGraph {
    /// Invoke the graph with a list of initial messages.
    fn invoke(&self, messages: Vec<PyMessage>) -> PyResult<Vec<PyMessage>> {
        let msgs: Vec<Message> = messages.into_iter().map(|m| m.inner).collect();
        let initial = MessageState::new(msgs);
        let result = 
            crate::run_async(self.inner.invoke(initial))
            .map_err(to_py_err_generic)?;
        Ok(result
            .messages
            .into_iter()
            .map(|m| PyMessage { inner: m })
            .collect())
    }

    fn __repr__(&self) -> String {
        "MessageGraph(...)".to_string()
    }
}

// ─── PyMessageGraphBuilder ──────────────────────────────────────────────────

/// Convenience builder for chat-focused workflows.
///
/// Pre-configures a graph with message accumulation.
/// Node callables receive a list of `Message` objects and must return a list of `Message`.
/// Returned messages are appended to the conversation via the `Append` reducer.
///
/// Example:
///     builder = MessageGraphBuilder()
///     builder.add_node("echo", echo_fn)  # echo_fn(messages: list[Message]) -> list[Message]
///     builder.set_entry_point("echo")
///     builder.add_edge("echo", "__end__")
///     graph = builder.compile()
///     result = graph.invoke([Message.user("Hello")])
#[pyclass(name = "MessageGraphBuilder")]
pub struct PyMessageGraphBuilder {
    inner: Option<MessageGraphBuilder>,
}

#[pymethods]
impl PyMessageGraphBuilder {
    #[new]
    fn new() -> Self {
        PyMessageGraphBuilder {
            inner: Some(MessageGraphBuilder::new()),
        }
    }

    /// Add a node with a Python callable.
    ///
    /// The callable receives a list of Messages and must return a list of Messages.
    /// Returned messages are appended to the conversation.
    fn add_node(&mut self, name: &str, func: PyObject) {
        let builder = self.inner.take().unwrap_or_else(MessageGraphBuilder::new);
        let func_clone = Python::with_gil(|py| func.clone_ref(py));
        let node_name = name.to_string();

        let node = Arc::new(FunctionNode::new(
            node_name.clone(),
            move |state: &MessageState, _ctx: &Context| {
                let messages = state.messages.clone();
                let func = Python::with_gil(|py| func_clone.clone_ref(py));
                let name = node_name.clone();

                Box::pin(async move {
                    Python::with_gil(
                        |py| -> Result<MessageStateUpdate, flowgentra_ai::core::state_graph::StateGraphError> {
                            let py_messages: Vec<PyMessage> = messages
                                .into_iter()
                                .map(|m| PyMessage { inner: m })
                                .collect();

                            let result = func.call1(py, (py_messages,)).map_err(|e| {
                                flowgentra_ai::core::state_graph::StateGraphError::ExecutionError {
                                    node: name.clone(),
                                    reason: format!("{}", e),
                                }
                            })?;

                            let returned: Vec<PyMessage> = result.extract(py).map_err(|e| {
                                flowgentra_ai::core::state_graph::StateGraphError::ExecutionError {
                                    node: name.clone(),
                                    reason: format!("Node must return list of Messages: {}", e),
                                }
                            })?;

                            let rust_msgs: Vec<Message> =
                                returned.into_iter().map(|m| m.inner).collect();
                            Ok(MessageStateUpdate::new().messages(rust_msgs))
                        },
                    )
                })
            },
        ));

        self.inner = Some(builder.add_node(name, node));
    }

    /// Add a fixed edge.
    fn add_edge(&mut self, from_node: &str, to_node: &str) {
        let builder = self.inner.take().unwrap_or_else(MessageGraphBuilder::new);
        self.inner = Some(builder.add_edge(from_node, to_node));
    }

    /// Set the entry point.
    fn set_entry_point(&mut self, name: &str) {
        let builder = self.inner.take().unwrap_or_else(MessageGraphBuilder::new);
        self.inner = Some(builder.set_entry_point(name));
    }

    /// Compile the graph.
    fn compile(&mut self) -> PyResult<PyMessageGraph> {
        let builder = self.inner.take().unwrap_or_else(MessageGraphBuilder::new);
        let graph = builder.compile().map_err(to_py_err_generic)?;
        Ok(PyMessageGraph { inner: graph })
    }

    fn __repr__(&self) -> String {
        "MessageGraphBuilder(...)".to_string()
    }
}
