//! Python bindings for FlowgentraAI via PyO3
//!
//! Exposes the core Rust library to Python through native extension module.

#![allow(clippy::useless_conversion)]
#![allow(unexpected_cfgs)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;

mod advanced_nodes;
mod agent;
mod agents;
mod async_checkpointers;
mod builtin_node_bindings;
mod builtin_tools;
mod channel;
mod checkpointer;
mod chroma;
mod config;
mod conversation_memory;
mod doc_store;
mod document_loader;
mod error;
mod eval_advanced;
mod eval_config;
mod evaluation;
mod evaluation_node;
mod extra_embeddings;
mod file_checkpointer;
mod graph;
mod indexing;
mod ingestion;
mod llm;
mod llm_reranker;
mod loaders_extra;
mod mcp;
mod memory;
mod memory_aware_agent;
mod message_graph;
mod middleware;
mod observability;
mod otel;
mod planner_node;
mod prometheus;
mod prompt_parser;
mod py_code_exec;
mod py_communication;
mod py_data;
mod py_external_apis;
mod py_files_extended;
mod py_human;
mod py_knowledge;
mod py_reducers;
mod py_search;
mod rag;
mod rag_config;
mod rag_doc_store;
mod rag_eval;
mod remaining;
mod reranker;
mod retrievers_advanced;
mod routing;
mod skills;
mod snapshot;
mod sql_db;
mod state;
mod supervisor;
mod text_splitter;
mod token_buffer_memory;
mod tool_node;
mod tool_registry;
mod tools;
mod vector_store;
mod vector_stores;
mod visualization;
mod web_retrievers;

use advanced_nodes::*;
use agent::*;
use agents::*;
use builtin_node_bindings::*;
use builtin_tools::*;
use chroma::*;
use config::*;
use conversation_memory::*;
use doc_store::*;
use document_loader::*;
use eval_advanced::*;
use eval_config::*;
use evaluation::*;
use evaluation_node::*;
use extra_embeddings::*;
use graph::*;
use indexing::*;
use ingestion::*;
use llm::*;
use llm_reranker::*;
use loaders_extra::*;
use mcp::*;
use memory::*;
use memory_aware_agent::*;
use message_graph::*;
use observability::*;
use planner_node::*;
use prompt_parser::*;
use rag::*;
use rag_config::*;
use rag_doc_store::*;
use rag_eval::*;
use remaining::*;
use reranker::*;
use retrievers_advanced::*;
use routing::*;
use sql_db::*;
use supervisor::*;
use text_splitter::*;
use token_buffer_memory::*;
use tool_node::*;
use tool_registry::*;
use tools::*;
use vector_store::*;
use vector_stores::*;
use visualization::*;
use web_retrievers::*;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Convert a serde_json::Value to a Python object
pub fn json_to_py(py: Python<'_>, val: &Value) -> PyResult<PyObject> {
    match val {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.to_object(py)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None())
            }
        }
        Value::String(s) => Ok(s.to_object(py)),
        Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into())
        }
        Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into())
        }
    }
}

/// Convert a Python object to serde_json::Value
pub fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        Ok(Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(Value::Number(i.into()))
    } else if let Ok(f) = obj.extract::<f64>() {
        // Issue #6: NaN/Inf are not JSON-serializable; surface an explicit error
        // instead of silently coercing to null and corrupting the state.
        serde_json::Number::from_f64(f)
            .map(Value::Number)
            .ok_or_else(|| {
                crate::error::SerializationError::new_err(format!(
                    "Cannot serialize non-finite float to JSON (got: {}). \
                     Use a finite value, or convert to string/None explicitly.",
                    f
                ))
            })
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(Value::String(s))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let mut arr = Vec::new();
        for item in list.iter() {
            arr.push(py_to_json(&item)?);
        }
        Ok(Value::Array(arr))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(&v)?);
        }
        Ok(Value::Object(map))
    } else {
        // Fallback: try string representation
        let s = obj.str()?.to_string();
        Ok(Value::String(s))
    }
}

/// Get or create a tokio runtime for blocking async calls.
///
/// Initialised once (lazily) and reused for the lifetime of the process.
/// Panics with a descriptive message if the OS refuses to allocate the runtime
/// (e.g. extreme memory pressure or hard thread-count limits); this is
/// unrecoverable and a process-level panic is the appropriate response.
pub fn get_runtime() -> &'static tokio::runtime::Runtime {
    use std::sync::OnceLock;
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Runtime::new().expect(
            "Failed to create the FlowgentraAI Tokio runtime. \
             This usually means the OS has exhausted threads or memory. \
             Try reducing concurrent workloads or increasing system limits.",
        )
    })
}

/// Run an async future from any context — sync or already inside a Tokio runtime.
///
/// When called from within a multi-thread Tokio runtime, `block_in_place` parks
/// the current worker thread and allows `handle.block_on` to run the future.
///
/// On a `current_thread` runtime (e.g. `#[tokio::test]` default flavor),
/// `block_in_place` would panic. We detect this via `runtime_flavor()` and fall
/// back to our own dedicated multi-thread runtime, avoiding the panic entirely.
pub fn run_async<F: std::future::Future>(fut: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            use tokio::runtime::RuntimeFlavor;
            match handle.runtime_flavor() {
                // Issue #10: block_in_place panics on current_thread runtimes
                // (no worker threads to block). Fall back to our own runtime.
                RuntimeFlavor::CurrentThread => get_runtime().block_on(fut),
                _ => tokio::task::block_in_place(|| handle.block_on(fut)),
            }
        }
        Err(_) => get_runtime().block_on(fut),
    }
}

// ─── Python Module ──────────────────────────────────────────────────────────

/// FlowgentraAI Python module
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Exception hierarchy ────────────────────────────────────────────────────
    error::register_exceptions(m)?;

    // ── State Submodule ────────────────────────────────────────────────────────
    let state_module = PyModule::new_bound(m.py(), "state")?;
    state_module.add_class::<state::PyState>()?;
    state_module.add_class::<snapshot::PyStateSnapshot>()?;
    state_module.add_class::<checkpointer::PyMemoryCheckpointer>()?;
    state_module.add_class::<checkpointer::PyFileCheckpointer>()?;
    // Async checkpointers
    state_module.add_class::<async_checkpointers::PyCheckpointHistoryEntry>()?;
    state_module.add_class::<async_checkpointers::PySqliteAsyncCheckpointer>()?;
    state_module.add_class::<async_checkpointers::PyPostgresAsyncCheckpointer>()?;
    state_module.add_class::<async_checkpointers::PyRedisAsyncCheckpointer>()?;
    state_module.add_class::<async_checkpointers::PyMongoAsyncCheckpointer>()?;
    state_module.add_class::<async_checkpointers::PyMySqlAsyncCheckpointer>()?;
    state_module.add_class::<async_checkpointers::PyNamespacedCheckpointer>()?;
    state_module.add_class::<py_reducers::PyAppendField>()?;
    state_module.add_class::<py_reducers::PyBinaryOperatorField>()?;
    m.add_submodule(&state_module)?;

    // ── Agent Submodule ────────────────────────────────────────────────────────
    let agent_module = PyModule::new_bound(m.py(), "agent")?;
    agent_module.add_class::<PyAgent>()?;
    agent_module.add_class::<PyAgentType>()?;
    agent_module.add_class::<PyToolSpec>()?;
    agent_module.add_class::<PyGraphBasedAgent>()?;
    agent_module.add_class::<PyAgentConfig>()?;
    agent_module.add_class::<PyStateField>()?;
    // Typed agent constructors
    agent_module.add_class::<agents::PyZeroShotReAct>()?;
    agent_module.add_class::<agents::PyFewShotReAct>()?;
    agent_module.add_class::<agents::PyConversational>()?;
    agent_module.add_class::<agents::PyToolCalling>()?;
    agent_module.add_class::<agents::PyStructuredChat>()?;
    agent_module.add_class::<agents::PySelfAskWithSearch>()?;
    agent_module.add_class::<agents::PyReactDocstore>()?;
    m.add_submodule(&agent_module)?;

    // ── Graph Submodule ────────────────────────────────────────────────────────
    let graph_module = PyModule::new_bound(m.py(), "graph")?;
    graph_module.add_class::<PyStateGraphBuilder>()?;
    graph_module.add_class::<PyCompiledGraph>()?;
    graph_module.add("END", graph::PY_END)?;
    m.add_submodule(&graph_module)?;

    // ── LLM Submodule ─────────────────────────────────────────────────────────
    let llm_module = PyModule::new_bound(m.py(), "llm")?;
    llm_module.add_class::<PyLLMConfig>()?;
    llm_module.add_class::<PyResponseFormat>()?;
    llm_module.add_class::<PyMessage>()?;
    llm_module.add_class::<PyToolCall>()?;
    llm_module.add_class::<PyToolDefinition>()?;
    llm_module.add_class::<PyTokenUsage>()?;
    llm_module.add_class::<PyLLM>()?;
    llm_module.add_class::<llm::PyLLMStream>()?;
    llm_module.add_function(wrap_pyfunction!(llm::py_create_llm, &llm_module)?)?;
    llm_module.add_function(wrap_pyfunction!(py_estimate_tokens, &llm_module)?)?;
    llm_module.add_function(wrap_pyfunction!(py_model_pricing, &llm_module)?)?;
    m.add_submodule(&llm_module)?;

    // ── Tools Submodule ────────────────────────────────────────────────────────
    let tools_module = PyModule::new_bound(m.py(), "tools")?;
    tools_module.add_class::<PyToolCallRequest>()?;
    tools_module.add_class::<PyToolCallResult>()?;
    tools_module.add_class::<PyToolRegistry>()?;
    tools_module.add_class::<PyJsonSchema>()?;
    // Core built-ins
    tools_module.add_class::<PyCalculatorTool>()?;
    tools_module.add_class::<PyWebRequestTool>()?;
    tools_module.add_class::<PyFilesTool>()?;
    // Search tools
    tools_module.add_class::<py_search::PyDuckDuckGoSearchTool>()?;
    tools_module.add_class::<py_search::PyTavilySearchTool>()?;
    tools_module.add_class::<py_search::PySerpApiSearchTool>()?;
    tools_module.add_class::<py_search::PyGoogleSerperTool>()?;
    tools_module.add_class::<py_search::PyBraveSearchTool>()?;
    // Knowledge tools
    tools_module.add_class::<py_knowledge::PyWikipediaTool>()?;
    tools_module.add_class::<py_knowledge::PyArxivTool>()?;
    tools_module.add_class::<py_knowledge::PyPubMedTool>()?;
    tools_module.add_class::<py_knowledge::PyWolframAlphaTool>()?;
    // Code execution tools
    tools_module.add_class::<py_code_exec::PyPythonReplTool>()?;
    tools_module.add_class::<py_code_exec::PyNodeJsReplTool>()?;
    tools_module.add_class::<py_code_exec::PyShellTool>()?;
    // Extended file tools
    tools_module.add_class::<py_files_extended::PyCopyFileTool>()?;
    tools_module.add_class::<py_files_extended::PyDeleteFileTool>()?;
    tools_module.add_class::<py_files_extended::PyMoveFileTool>()?;
    tools_module.add_class::<py_files_extended::PyFileSearchTool>()?;
    // Data tools
    tools_module.add_class::<py_data::PyJsonGetValueTool>()?;
    tools_module.add_class::<py_data::PyJsonListKeysTool>()?;
    tools_module.add_class::<py_data::PyCsvQueryTool>()?;
    // Human-in-the-loop
    tools_module.add_class::<py_human::PyHumanInputTool>()?;
    // Communication
    tools_module.add_class::<py_communication::PyGmailTool>()?;
    tools_module.add_class::<py_communication::PySlackTool>()?;
    // External APIs
    tools_module.add_class::<py_external_apis::PyOpenWeatherMapTool>()?;
    tools_module.add_class::<py_external_apis::PyNewsApiTool>()?;
    tools_module.add_class::<py_external_apis::PyAlphaVantageTool>()?;
    // ToolNode helper
    tools_module.add_class::<PyToolNode>()?;
    tools_module.add_function(wrap_pyfunction!(
        tool_node::py_create_tool_node,
        &tools_module
    )?)?;
    tools_module.add_function(wrap_pyfunction!(
        tool_node::py_store_tool_calls,
        &tools_module
    )?)?;
    tools_module.add_function(wrap_pyfunction!(
        tool_node::py_check_tools_condition,
        &tools_module
    )?)?;
    m.add_submodule(&tools_module)?;

    // ── RAG Submodule ──────────────────────────────────────────────────────────
    let rag_module = PyModule::new_bound(m.py(), "rag")?;
    rag_module.add_class::<PyDocument>()?;
    rag_module.add_class::<PySearchResult>()?;
    rag_module.add_class::<PyTextChunk>()?;
    rag_module.add_class::<PyEmbeddings>()?;
    rag_module.add_class::<PyInMemoryVectorStore>()?;
    rag_module.add_class::<PyRetrievalConfig>()?;
    rag_module.add_class::<PyRetriever>()?;
    rag_module.add_class::<rag::PyPdfDocument>()?;
    rag_module.add_class::<PyVectorStoreType>()?;
    rag_module.add_class::<PyRAGConfig>()?;
    rag_module.add_class::<PyVectorStoreConfig>()?;
    rag_module.add_class::<PyEmbeddingsConfig>()?;
    rag_module.add_class::<PyRetrievalSettings>()?;
    rag_module.add_class::<PyPdfSettings>()?;
    rag_module.add_class::<PyRAGGraphConfig>()?;
    rag_module.add_class::<PyChromaStore>()?;
    rag_module.add_class::<PyPineconeStore>()?;
    rag_module.add_class::<PyQdrantStore>()?;
    rag_module.add_class::<PyWeaviateStore>()?;
    rag_module.add_class::<PyMilvusStore>()?;
    rag_module.add_class::<vector_stores::PyPgVectorStore>()?;
    rag_module.add_class::<vector_stores::PyRedisVectorStore>()?;
    rag_module.add_class::<vector_stores::PyElasticsearchVectorStore>()?;
    rag_module.add_class::<vector_stores::PyOpenSearchVectorStore>()?;
    rag_module.add_class::<vector_stores::PyUpstashVectorStore>()?;
    rag_module.add_class::<vector_stores::PyAstraDbVectorStore>()?;
    rag_module.add_class::<vector_stores::PyMongoAtlasVectorStore>()?;
    // Extra vector stores
    rag_module.add_class::<vector_stores::PyHnswVectorStore>()?;
    rag_module.add_class::<vector_stores::PySingleStoreVectorStore>()?;
    rag_module.add_class::<vector_stores::PyAzureAISearchStore>()?;
    rag_module.add_class::<vector_stores::PyVectaraStore>()?;
    rag_module.add_class::<vector_stores::PyTurbopufferStore>()?;
    rag_module.add_class::<vector_stores::PyNeo4jVectorStore>()?;
    rag_module.add_function(wrap_pyfunction!(rag::py_chunk_text_by_tokens, &rag_module)?)?;
    rag_module.add_function(wrap_pyfunction!(rag::py_extract_and_chunk, &rag_module)?)?;
    rag_module.add_function(wrap_pyfunction!(rag::py_extract_pdf, &rag_module)?)?;
    rag_module.add_function(wrap_pyfunction!(rag::py_bm25_score, &rag_module)?)?;
    rag_module.add_function(wrap_pyfunction!(rag::py_hybrid_merge, &rag_module)?)?;
    rag_module.add_function(wrap_pyfunction!(rag::py_dedup_by_id, &rag_module)?)?;
    rag_module.add_function(wrap_pyfunction!(rag::py_dedup_by_similarity, &rag_module)?)?;
    rag_module.add_function(wrap_pyfunction!(rag::py_decompose_query, &rag_module)?)?;

    // ── Advanced Retrievers ────────────────────────────────────────────────────
    rag_module.add_class::<PyBm25Retriever>()?;
    rag_module.add_class::<PyVectorRetriever>()?;
    rag_module.add_class::<PyEnsembleRetriever>()?;
    rag_module.add_class::<PyMultiQueryRetriever>()?;
    rag_module.add_class::<PyScoreThresholdRetriever>()?;
    rag_module.add_class::<PyEmbeddingsFilter>()?;
    rag_module.add_class::<PyContextualCompressionRetriever>()?;
    rag_module.add_class::<PyTimeWeightedRetriever>()?;
    rag_module.add_class::<PyVectorView>()?;
    rag_module.add_class::<PyMultiVectorRetriever>()?;
    rag_module.add_class::<PyParentDocumentRetriever>()?;
    rag_module.add_class::<PyReorderStrategy>()?;
    rag_module.add_function(wrap_pyfunction!(
        retrievers_advanced::py_reorder_for_long_context,
        &rag_module
    )?)?;
    rag_module.add_class::<PySelfQueryRetriever>()?;
    rag_module.add_class::<PyLLMCompressor>()?;
    rag_module.add_class::<PyDocumentCompressorPipeline>()?;

    // ── Web Retrievers ─────────────────────────────────────────────────────────
    rag_module.add_class::<PyWikipediaRetriever>()?;
    rag_module.add_class::<PyArxivRetriever>()?;
    rag_module.add_class::<PyTavilySearchRetriever>()?;

    // ── RAG DocStore ───────────────────────────────────────────────────────────
    rag_module.add_class::<PyStoredDocument>()?;
    rag_module.add_class::<PyInMemoryDocStore>()?;
    rag_module.add_class::<PyLocalFileDocStore>()?;

    // ── Indexing Pipeline ──────────────────────────────────────────────────────
    rag_module.add_class::<PyCleanupMode>()?;
    rag_module.add_class::<PyIndexStats>()?;
    rag_module.add_class::<PyRecordEntry>()?;
    rag_module.add_class::<PyInMemoryRecordManager>()?;
    rag_module.add_function(wrap_pyfunction!(indexing::py_index_documents, &rag_module)?)?;

    // ── Extra Embeddings ───────────────────────────────────────────────────────
    rag_module.add_class::<PyCohereEmbeddings>()?;
    rag_module.add_class::<PyAzureOpenAIEmbeddings>()?;
    rag_module.add_class::<PyGoogleVertexEmbeddings>()?;
    rag_module.add_class::<PyBedrockEmbeddings>()?;
    rag_module.add_class::<PyVoyageEmbeddings>()?;
    rag_module.add_class::<PyJinaEmbeddings>()?;
    rag_module.add_class::<PyTogetherEmbeddings>()?;
    rag_module.add_class::<PyNomicEmbeddings>()?;

    m.add_submodule(&rag_module)?;

    // ── Memory Submodule ───────────────────────────────────────────────────────
    let memory_module = PyModule::new_bound(m.py(), "memory")?;
    memory_module.add_class::<PyCheckpoint>()?;
    memory_module.add_class::<PyCheckpointMetadata>()?;
    memory_module.add_class::<PyConversationMemory>()?;
    memory_module.add_class::<PyTokenBufferMemory>()?;
    memory_module.add_class::<PySummaryConfig>()?;
    memory_module.add_class::<PySummaryMemory>()?;
    m.add_submodule(&memory_module)?;

    // ── Text Processing Submodule ──────────────────────────────────────────────
    let text_module = PyModule::new_bound(m.py(), "text")?;
    text_module.add_class::<PyRecursiveCharacterTextSplitter>()?;
    text_module.add_class::<PyMarkdownTextSplitter>()?;
    text_module.add_class::<PyHTMLTextSplitter>()?;
    text_module.add_class::<PyTokenTextSplitter>()?;
    text_module.add_class::<PyCodeTextSplitter>()?;
    text_module.add_class::<PyPromptTemplate>()?;
    text_module.add_class::<PyJsonOutputParser>()?;
    text_module.add_class::<PyListOutputParser>()?;
    text_module.add_function(wrap_pyfunction!(py_chunk_text, &text_module)?)?;
    text_module.add_function(wrap_pyfunction!(py_extract_text, &text_module)?)?;
    m.add_submodule(&text_module)?;

    // ── Evaluation Submodule ───────────────────────────────────────────────────
    let eval_module = PyModule::new_bound(m.py(), "evaluation")?;
    eval_module.add_class::<PyEvaluationResult>()?;
    eval_module.add_class::<PyEvalQuery>()?;
    eval_module.add_class::<PyEvalResults>()?;
    eval_module.add_class::<PyQueryResult>()?;
    eval_module.add_class::<PyScoringConfig>()?;
    eval_module.add_class::<PyGradingConfig>()?;
    eval_module.add_class::<PyEvaluationConfig>()?;
    eval_module.add_class::<PyNodeResult>()?;
    eval_module.add_class::<PyEvaluationReport>()?;
    eval_module.add_function(wrap_pyfunction!(rag_eval::py_hit_rate, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(rag_eval::py_mrr, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(rag_eval::py_mean_ndcg, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(rag_eval::py_rag_evaluate, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(
        remaining::py_evaluate_output_score,
        &eval_module
    )?)?;
    // Advanced evaluation: confidence, node scoring, retry, smart fallback
    eval_module.add_class::<PyConfidenceLevel>()?;
    eval_module.add_class::<PyConfidenceConfig>()?;
    eval_module.add_class::<PyConfidenceScore>()?;
    eval_module.add_class::<PyScoringCriteria>()?;
    eval_module.add_class::<PyNodeScore>()?;
    eval_module.add_class::<PyRetryConfig>()?;
    eval_module.add_class::<PyRetryResult>()?;
    eval_module.add_class::<PyFallbackLevel>()?;
    eval_module.add_function(wrap_pyfunction!(py_score_confidence, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(py_score_node, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(py_retry_should_retry, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(py_retry_delay_ms, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(py_retry_temperature, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(py_retry_feedback, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(py_check_circuit_breaker, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(py_retry_generate_report, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(
        py_generate_content_fallback,
        &eval_module
    )?)?;
    eval_module.add_function(wrap_pyfunction!(py_refine_content_fallback, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(py_fallback_retry_message, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(py_should_fallback, &eval_module)?)?;
    m.add_submodule(&eval_module)?;

    // ── Observability Submodule ────────────────────────────────────────────────
    let obs_module = PyModule::new_bound(m.py(), "observability")?;
    obs_module.add_class::<PyExecutionTrace>()?;
    obs_module.add_class::<PyExecutionTracer>()?;
    obs_module.add_class::<observability::PyEventBroadcaster>()?;
    obs_module.add_class::<observability::PyEventReceiver>()?;
    obs_module.add_class::<observability::PyReplayMode>()?;
    obs_module.add_function(wrap_pyfunction!(
        observability::py_init_tracing,
        &obs_module
    )?)?;
    obs_module.add_function(wrap_pyfunction!(
        visualization::py_visualize_graph,
        &obs_module
    )?)?;
    obs_module.add_function(wrap_pyfunction!(
        visualization::py_graph_to_dot,
        &obs_module
    )?)?;
    obs_module.add_function(wrap_pyfunction!(
        visualization::py_graph_to_mermaid,
        &obs_module
    )?)?;
    // Prometheus metrics
    obs_module.add_class::<prometheus::PyPrometheusExporter>()?;
    obs_module.add_class::<prometheus::PyMetricsCollector>()?;
    obs_module.add_function(wrap_pyfunction!(
        prometheus::py_record_llm_tokens,
        &obs_module
    )?)?;
    // OpenTelemetry spans
    obs_module.add_class::<otel::PyOtelStatus>()?;
    obs_module.add_class::<otel::PyOtelAttribute>()?;
    obs_module.add_class::<otel::PyOtelSpan>()?;
    obs_module.add_function(wrap_pyfunction!(otel::py_trace_to_otel_spans, &obs_module)?)?;
    obs_module.add_function(wrap_pyfunction!(otel::py_spans_to_otlp_json, &obs_module)?)?;
    obs_module.add_function(wrap_pyfunction!(otel::py_export_to_otlp, &obs_module)?)?;
    m.add_submodule(&obs_module)?;

    // ── Advanced Nodes Submodule ───────────────────────────────────────────────
    let nodes_module = PyModule::new_bound(m.py(), "nodes")?;
    nodes_module.add_class::<PyMessageGraph>()?;
    nodes_module.add_class::<PyMessageGraphBuilder>()?;
    nodes_module.add_class::<PySupervisor>()?;
    nodes_module.add_class::<PyOrchestrationStrategy>()?;
    nodes_module.add_class::<PySupervisorNodeConfig>()?;
    nodes_module.add_class::<PyParallelAggregation>()?;
    nodes_module.add_class::<PyParallelMergeStrategy>()?;
    nodes_module.add_class::<PyChildExecutionStats>()?;
    nodes_module.add_class::<PyPlannerNode>()?;
    nodes_module.add_class::<PyRetryNode>()?;
    nodes_module.add_class::<PyTimeoutNode>()?;
    nodes_module.add_class::<PyEvaluationNodeConfig>()?;
    nodes_module.add_class::<PyJoinType>()?;
    nodes_module.add_class::<PyMergeStrategy>()?;
    nodes_module.add_class::<PyBranchConfig>()?;
    nodes_module.add_class::<PyLoopNodeConfig>()?;
    nodes_module.add_class::<PyParallelNodeConfig>()?;
    nodes_module.add_class::<PySubgraphNodeConfig>()?;
    nodes_module.add_class::<PyJoinNodeConfig>()?;
    m.add_submodule(&nodes_module)?;

    // ── Routing Submodule ──────────────────────────────────────────────────────
    let routing_module = PyModule::new_bound(m.py(), "routing")?;
    routing_module.add_class::<PyComparisonOp>()?;
    routing_module.add_class::<PyCondition>()?;
    routing_module.add_class::<PyConditionBuilder>()?;
    m.add_submodule(&routing_module)?;

    // ── Data Loading Submodule ────────────────────────────────────────────────
    let data_module = PyModule::new_bound(m.py(), "data")?;
    data_module.add_class::<PyFileType>()?;
    data_module.add_class::<PyLoadedDocument>()?;
    data_module.add_class::<PyIngestionPipeline>()?;
    data_module.add_class::<PyIngestionStats>()?;
    data_module.add_function(wrap_pyfunction!(
        document_loader::py_load_document,
        &data_module
    )?)?;
    data_module.add_function(wrap_pyfunction!(
        document_loader::py_load_directory,
        &data_module
    )?)?;
    m.add_submodule(&data_module)?;

    // ── Reranking Submodule ────────────────────────────────────────────────────
    let rerank_module = PyModule::new_bound(m.py(), "reranking")?;
    rerank_module.add_class::<PyNoopReranker>()?;
    rerank_module.add_class::<PyRRFReranker>()?;
    rerank_module.add_class::<PyCrossEncoderReranker>()?;
    rerank_module.add_class::<PyLLMReranker>()?;
    m.add_submodule(&rerank_module)?;

    // ── Advanced Agents Submodule ──────────────────────────────────────────────
    let advanced_module = PyModule::new_bound(m.py(), "advanced")?;
    advanced_module.add_class::<PyMemoryAwareAgent>()?;
    advanced_module.add_class::<PyMemoryStats>()?;
    m.add_submodule(&advanced_module)?;

    // ── MCP Submodule ─────────────────────────────────────────────────────────
    let mcp_module = PyModule::new_bound(m.py(), "mcp")?;
    mcp_module.add_class::<mcp::PyMCPConnectionType>()?;
    mcp_module.add_class::<mcp::PyMCPAuth>()?;
    mcp_module.add_class::<mcp::PyMCPConnectionSettings>()?;
    mcp_module.add_class::<PyMCPConfig>()?;
    mcp_module.add_class::<mcp::PyMCPTool>()?;
    mcp_module.add_class::<mcp::PyMCPResource>()?;
    mcp_module.add_class::<mcp::PyMCPResourceContent>()?;
    mcp_module.add_class::<mcp::PyMCPPromptArgument>()?;
    mcp_module.add_class::<mcp::PyMCPPrompt>()?;
    mcp_module.add_class::<mcp::PyMCPPromptMessage>()?;
    mcp_module.add_class::<mcp::PyMCPPromptResult>()?;
    mcp_module.add_class::<mcp::PyMCPClient>()?;
    mcp_module.add_function(wrap_pyfunction!(mcp::py_create_mcp_client, &mcp_module)?)?;
    mcp_module.add_function(wrap_pyfunction!(mcp::py_merge_tool_lists, &mcp_module)?)?;
    // SSE streaming
    mcp_module.add_class::<mcp::PySSEMessage>()?;
    mcp_module.add_class::<mcp::PySSEStreamReceiver>()?;
    mcp_module.add_class::<mcp::PySSEConnection>()?;
    // Stdio process connections
    mcp_module.add_class::<mcp::PyStdioConnection>()?;
    mcp_module.add_class::<mcp::PyStdioConnectionBuilder>()?;
    // Docker connections
    mcp_module.add_class::<mcp::PyContainerState>()?;
    mcp_module.add_class::<mcp::PyDockerConfig>()?;
    mcp_module.add_class::<mcp::PyDockerConnection>()?;
    mcp_module.add_class::<mcp::PyDockerConnectionBuilder>()?;
    m.add_submodule(&mcp_module)?;

    // ── Utilities Submodule ────────────────────────────────────────────────────
    let utils_module = PyModule::new_bound(m.py(), "utils")?;
    utils_module.add_class::<PyVisualizationConfig>()?;
    utils_module.add_class::<PyChunkMetadata>()?;
    utils_module.add_class::<PyRetrieverStrategy>()?;
    utils_module.add_class::<PyRerankStrategy>()?;
    utils_module.add_class::<PyVectorStore>()?;
    utils_module.add_function(wrap_pyfunction!(py_from_config_path, &utils_module)?)?;
    m.add_submodule(&utils_module)?;

    // ── Loaders Submodule ──────────────────────────────────────────────────────
    let loaders_module = PyModule::new_bound(m.py(), "loaders")?;
    loaders_module.add_class::<PyCsvLoader>()?;
    loaders_module.add_class::<PyWebLoader>()?;
    loaders_module.add_class::<PyJsonLoader>()?;
    loaders_module.add_class::<PyJsonlLoader>()?;
    loaders_module.add_class::<PyDocxLoader>()?;
    loaders_module.add_class::<PyEpubLoader>()?;
    loaders_module.add_class::<PyExcelLoader>()?;
    loaders_module.add_class::<PyDirectoryLoader>()?;
    loaders_module.add_class::<PySitemapLoader>()?;
    loaders_module.add_class::<PyRecursiveUrlLoader>()?;
    loaders_module.add_class::<PyWikipediaLoader>()?;
    loaders_module.add_class::<PyArxivLoader>()?;
    loaders_module.add_class::<PyRssFeedLoader>()?;
    loaders_module.add_class::<PyYouTubeLoader>()?;
    loaders_module.add_class::<PyS3Loader>()?;
    loaders_module.add_class::<PyDataFrameLoader>()?;
    loaders_module.add_class::<PyGitLoader>()?;
    m.add_submodule(&loaders_module)?;

    // ── Database Submodule ─────────────────────────────────────────────────────
    let db_module = PyModule::new_bound(m.py(), "db")?;
    // SQL backends
    db_module.add_class::<PySqliteDatabase>()?;
    db_module.add_class::<PyPostgresDatabase>()?;
    db_module.add_class::<PyMySqlDatabase>()?;
    db_module.add_class::<PyMssqlDatabase>()?;
    db_module.add_class::<PyBigQueryDatabase>()?;
    db_module.add_class::<PyDatabricksDatabase>()?;
    // Document store backends
    db_module.add_class::<PyMongoDocumentStore>()?;
    db_module.add_class::<PyRedisDocumentStore>()?;
    db_module.add_class::<PyNeo4jDocumentStore>()?;
    db_module.add_class::<PyCassandraDocumentStore>()?;
    db_module.add_class::<PyElasticsearchDocumentStore>()?;
    m.add_submodule(&db_module)?;

    // ── Middleware Submodule ───────────────────────────────────────────────────
    let mw_module = PyModule::new_bound(m.py(), "middleware")?;
    mw_module.add_class::<middleware::PyExecutionMetrics>()?;
    mw_module.add_class::<middleware::PyLoggingMiddleware>()?;
    mw_module.add_class::<middleware::PyMetricsMiddleware>()?;
    m.add_submodule(&mw_module)?;

    // ── Skills Submodule ───────────────────────────────────────────────────────
    let skills_module = PyModule::new_bound(m.py(), "skills")?;
    skills_module.add_class::<skills::PySkill>()?;
    skills_module.add_class::<skills::PySkillRegistry>()?;
    m.add_submodule(&skills_module)?;
    // Register in sys.modules so both `from _native import skills as _s` and
    // `from _native.skills import X` work — PyO3 add_submodule alone only adds
    // the attribute, not a sys.modules entry.
    m.py()
        .import_bound("sys")?
        .getattr("modules")?
        .set_item("flowgentra_ai._native.skills", &skills_module)?;

    Ok(())
}
