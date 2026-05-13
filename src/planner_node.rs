//! Python bindings for the Planner Node (LLM-driven dynamic routing)
//!
//! The planner node uses an LLM to decide the next node to execute
//! based on the current state and reachable nodes.

use pyo3::prelude::*;
use std::sync::Arc;

use flowgentra_ai::core::llm::LLM;
use flowgentra_ai::core::node::planner::create_planner_handler;
use flowgentra_ai::core::state::{Context, DynState, DynStateUpdate};
use flowgentra_ai::core::state_graph::node::Node;
use flowgentra_ai::core::state_graph::StateGraphError;

use crate::llm::PyLLM;
use crate::state::PyState;

// ─── Planner as a Node for StateGraphBuilder ────────────────────────────────

/// Wraps the Rust planner handler as a Node<DynState> so it can be
/// plugged into a StateGraphBuilder.
struct PlannerHandlerNode {
    name: String,
    llm: Arc<dyn LLM>,
    prompt_template: Option<String>,
}

#[async_trait::async_trait]
impl Node<DynState> for PlannerHandlerNode {
    async fn execute(
        &self,
        state: &DynState,
        _ctx: &Context,
    ) -> Result<DynStateUpdate, StateGraphError> {
        // Create a fresh handler each time (the handler is cheap — it's just a closure)
        let handler = create_planner_handler(Arc::clone(&self.llm), self.prompt_template.clone());
        // The handler takes ownership of state, so clone it
        let result_state =
            handler(state.clone())
                .await
                .map_err(|e| StateGraphError::ExecutionError {
                    node: self.name.clone(),
                    reason: format!("Planner error: {}", e),
                })?;
        // Return all keys from result as a DynStateUpdate (planner sets _next_node etc.)
        let mut update = DynStateUpdate::new();
        for key in result_state.keys() {
            if let Some(val) = result_state.get(&key) {
                update.insert(key, val);
            }
        }
        Ok(update)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ─── PyPlannerNode ──────────────────────────────────────────────────────────

/// LLM-driven planner node for dynamic routing in a StateGraph.
///
/// The planner reads `_reachable_nodes` from state and uses an LLM
/// to decide which node should run next, setting `_next_node` in state.
///
/// Usage:
///     planner = PlannerNode("planner", llm)
///     planner.set_prompt("You are a task planner...")
///
///     # Add to graph via StateGraphBuilder
///     builder.add_planner_node("planner", llm)
///     # Or manually:
///     result = planner.run(state)  # sets state["_next_node"]
///
/// The planner expects these state keys:
///   - `_current_node`: name of the current node (set by runtime)
///   - `_reachable_nodes`: JSON array of valid next node names
///
/// It sets:
///   - `_next_node`: the chosen next node name (or "END")
#[pyclass(name = "PlannerNode")]
pub struct PyPlannerNode {
    name: String,
    llm: Arc<dyn LLM>,
    prompt_template: Option<String>,
}

#[pymethods]
impl PyPlannerNode {
    /// Create a new planner node.
    ///
    /// Args:
    ///     name: Node name for the graph
    ///     llm: LLM instance for making LLM calls
    ///     prompt: Optional custom system prompt (replaces default planner prompt)
    #[new]
    #[pyo3(signature = (name, llm, prompt=None))]
    fn new(name: &str, llm: &PyLLM, prompt: Option<String>) -> Self {
        PyPlannerNode {
            name: name.to_string(),
            llm: llm.inner.clone(),
            prompt_template: prompt,
        }
    }

    /// Set a custom system prompt for the planner LLM.
    fn set_prompt(&mut self, prompt: &str) {
        self.prompt_template = Some(prompt.to_string());
    }

    /// Run the planner on the given state.
    ///
    /// Sets `_next_node` in the returned state.
    fn run(&self, state: &PyState) -> PyResult<PyState> {
        let handler = create_planner_handler(Arc::clone(&self.llm), self.prompt_template.clone());

        let result = crate::run_async(handler(state.inner.clone()));
        match result {
            Ok(new_state) => Ok(PyState { inner: new_state }),
            Err(e) => Err(crate::error::AgentExecutionError::new_err(format!(
                "Planner error: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PlannerNode(name='{}', custom_prompt={})",
            self.name,
            self.prompt_template.is_some()
        )
    }
}

/// Helper: create a Node<DynState> from a PlannerNode for use in StateGraphBuilder.
pub(crate) fn create_planner_graph_node(
    name: &str,
    llm: Arc<dyn LLM>,
    prompt_template: Option<String>,
) -> Arc<dyn Node<DynState>> {
    Arc::new(PlannerHandlerNode {
        name: name.to_string(),
        llm,
        prompt_template,
    })
}
