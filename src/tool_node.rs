//! Python bindings for tool_node helpers (create_tool_node, store_tool_calls, tools_condition)

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::state::DynState;
use flowgentra_ai::core::state_graph::node::Node;

use crate::llm::PyMessage;
use crate::state::PyState;

/// Create a tool execution node from a Python callable.
///
/// The callable signature is: `(tool_name: str, arguments: dict) -> str`
/// It should return the tool result as a string, or raise an exception on error.
///
/// Example:
///     def executor(name, args):
///         if name == "calculator":
///             return str(args["a"] + args["b"])
///         raise ValueError(f"Unknown tool: {name}")
///
///     tool_node = create_tool_node_py(executor)
#[pyfunction]
pub fn py_create_tool_node(executor: PyObject) -> PyToolNode {
    let func = Python::with_gil(|py| executor.clone_ref(py));

    let node: Arc<dyn Node<DynState>> = Arc::new(DynToolNode { func });
    PyToolNode { inner: node }
}

/// Internal DynState-based tool node that executes tool calls stored in state.
struct DynToolNode {
    func: PyObject,
}

#[async_trait::async_trait]
impl Node<DynState> for DynToolNode {
    async fn execute(
        &self,
        state: &DynState,
        _ctx: &flowgentra_ai::core::state::Context,
    ) -> flowgentra_ai::core::state_graph::error::Result<flowgentra_ai::core::state::DynStateUpdate>
    {
        use flowgentra_ai::core::state::DynStateUpdate;

        // Read tool_calls from state (array of {id, name, arguments})
        let tool_calls = state
            .get("tool_calls")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();

        let func = Python::with_gil(|py| self.func.clone_ref(py));
        let mut results: Vec<serde_json::Value> = Vec::new();
        let mut messages: Vec<serde_json::Value> = Vec::new();

        for tc in &tool_calls {
            let name = tc
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args = tc
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let result = Python::with_gil(|py| -> Result<String, String> {
                let py_args = crate::json_to_py(py, &args).map_err(|e| format!("{}", e))?;
                let ret = func
                    .call1(py, (name.clone(), py_args))
                    .map_err(|e| format!("Tool executor error: {}", e))?;
                let s: String = ret
                    .extract(py)
                    .map_err(|e| format!("Tool executor must return str: {}", e))?;
                Ok(s)
            });

            match result {
                Ok(output) => {
                    results.push(serde_json::json!({
                        "tool_call_id": id,
                        "name": name,
                        "result": output,
                        "error": null
                    }));
                    messages.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": id,
                        "content": output
                    }));
                }
                Err(err) => {
                    results.push(serde_json::json!({
                        "tool_call_id": id,
                        "name": name,
                        "result": null,
                        "error": err.clone()
                    }));
                    messages.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": id,
                        "content": format!("Error: {}", err)
                    }));
                }
            }
        }

        let mut update = DynStateUpdate::new();
        update.insert("tool_results", serde_json::json!(results));
        // Clear pending tool_calls after execution
        update.insert("tool_calls", serde_json::json!([]));
        // Append tool messages
        let existing_messages = state
            .get("messages")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let mut all_messages = existing_messages;
        all_messages.extend(messages);
        update.insert("messages", serde_json::json!(all_messages));
        Ok(update)
    }

    fn name(&self) -> &str {
        "tool_executor"
    }
}

/// A tool execution node for use with StateGraphBuilder.
///
/// Created via `create_tool_node()`. Pass to `StateGraphBuilder.add_tool_node()`.
#[pyclass(name = "ToolNode")]
pub struct PyToolNode {
    #[allow(dead_code)]
    pub(crate) inner: Arc<dyn Node<DynState>>,
}

/// Extract tool calls from an LLM response and store them in a DynState.
///
/// Example:
///     response = client.chat_with_tools(messages, tools)
///     state = store_tool_calls_py(state, response)
#[pyfunction]
pub fn py_store_tool_calls(state: &PyState, message: &PyMessage) -> PyState {
    // Extract tool calls from the message and store in state
    let tool_calls: Vec<serde_json::Value> = message
        .inner
        .tool_calls
        .as_ref()
        .map(|tcs| {
            tcs.iter()
                .map(|tc| {
                    serde_json::json!({
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let new_state = state.inner.deep_clone();
    new_state.set("tool_calls", serde_json::json!(tool_calls));
    new_state.set("last_response", serde_json::json!(message.inner.content));
    PyState { inner: new_state }
}

/// Get the name of the tool node or "__end__" based on whether tool_calls exist.
///
/// This is a simple helper — returns tool_node_name if state has tool_calls,
/// or "__end__" otherwise. Use in a Python router function.
///
/// Example:
///     def tools_router(state):
///         return check_tools_condition(state, "tools")
///     builder.add_conditional_edge("agent", tools_router)
#[pyfunction]
pub fn py_check_tools_condition(state: &PyState, tool_node_name: &str) -> PyResult<String> {
    let tool_calls = state
        .inner
        .get("tool_calls")
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();

    if !tool_calls.is_empty() {
        Ok(tool_node_name.to_string())
    } else {
        Ok("__end__".to_string())
    }
}
