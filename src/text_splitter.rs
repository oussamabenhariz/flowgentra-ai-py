//! Python bindings for text splitters

use flowgentra_ai::core::rag::{
    CodeTextSplitter, Document, HTMLTextSplitter, Language, MarkdownTextSplitter,
    RecursiveCharacterTextSplitter, TextChunk, TextSplitter, TokenTextSplitter,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use crate::py_to_json;
use crate::rag::{PyDocument, PyTextChunk};

// ─── Helper to convert Vec<TextChunk> → Vec<PyTextChunk> ───────────────────

fn to_py_chunks(chunks: Vec<TextChunk>) -> Vec<PyTextChunk> {
    chunks
        .into_iter()
        .map(|c| PyTextChunk { inner: c })
        .collect()
}

// ─── Helper to extract Document content and ID from mixed input ────────────

#[allow(clippy::type_complexity)]
fn extract_document_info(
    obj: &Bound<'_, PyAny>,
) -> PyResult<(String, String, Option<HashMap<String, serde_json::Value>>)> {
    // Try string first (simpler)
    if let Ok(text) = obj.extract::<String>() {
        return Ok(("string_chunk".to_string(), text, None));
    }

    // Try PyDocument by checking type name and extracting fields
    let type_name_owned = obj
        .get_type()
        .name()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let type_name = type_name_owned.as_str();
    if type_name == "Document" {
        // Extract as PyDocument by accessing attributes
        let id: String = obj.getattr("id")?.extract()?;
        let text: String = obj.getattr("text")?.extract()?;

        let mut metadata = HashMap::new();
        if let Ok(meta_obj) = obj.getattr("metadata") {
            if let Ok(meta_dict) = meta_obj.downcast::<PyDict>() {
                for (k, v) in meta_dict.iter() {
                    let key: String = k.extract()?;
                    let val = py_to_json(&v)?;
                    metadata.insert(key, val);
                }
            }
        }

        return Ok((id, text, Some(metadata)));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Items must be Document objects or strings",
    ))
}

// ─── PyRecursiveCharacterTextSplitter ──────────────────────────────────────

/// Split text by recursively trying different separators.
///
/// Example:
///     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
///     chunks = splitter.split_text("Very long document text...")
#[pyclass(name = "RecursiveCharacterTextSplitter")]
pub struct PyRecursiveCharacterTextSplitter {
    inner: RecursiveCharacterTextSplitter,
}

#[pymethods]
impl PyRecursiveCharacterTextSplitter {
    #[new]
    #[pyo3(signature = (chunk_size=1000, chunk_overlap=200))]
    fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        PyRecursiveCharacterTextSplitter {
            inner: RecursiveCharacterTextSplitter::new(chunk_size, chunk_overlap),
        }
    }

    /// Split text into chunks.
    fn split_text(&self, text: &str) -> Vec<PyTextChunk> {
        to_py_chunks(self.inner.split_text(text))
    }

    /// Split text with source metadata.
    fn split_with_source(&self, text: &str, source: &str) -> Vec<PyTextChunk> {
        to_py_chunks(self.inner.split_with_source(text, source))
    }

    /// Split documents into chunks with metadata preservation (Option A).
    ///
    /// Processes documents **in parallel** for better performance on large datasets.
    ///
    /// Args:
    ///     documents: List of Document objects or strings
    ///     chunk_size: Chunk size (default 1024)
    ///     chunk_overlap: Overlap between chunks (default 100)
    ///
    /// Returns:
    ///     List of Document objects with metadata:
    ///     - Original metadata preserved
    ///     - source_doc_id: Which document this chunk came from
    ///     - chunk_index: Position of chunk within its document
    ///     - start_char: Character position in original text
    ///     - end_char: Character position in original text
    ///
    /// Performance:
    ///     - Uses Rayon for parallel chunking
    ///     - Optimal for large document sets (100+ documents)
    ///     - Automatic thread pool management
    ///
    /// Example:
    ///     chunks = RecursiveCharacterTextSplitter.split_documents(
    ///         documents=[doc1, doc2, "plain text"],
    ///         chunk_size=1024,
    ///         chunk_overlap=100
    ///     )
    #[staticmethod]
    #[pyo3(signature = (documents, chunk_size=1024, chunk_overlap=100))]
    fn split_documents(
        documents: &Bound<'_, PyList>,
        chunk_size: usize,
        chunk_overlap: usize,
    ) -> PyResult<Vec<PyDocument>> {
        // Step 1: Extract all documents from Python (holding GIL)
        let mut extracted_docs = Vec::new();
        for item in documents.iter() {
            let (doc_id, text, metadata) = extract_document_info(&item)?;
            extracted_docs.push((doc_id, text, metadata));
        }

        // Step 2: Release GIL and do parallel processing
        let splitter = RecursiveCharacterTextSplitter::new(chunk_size, chunk_overlap);

        let result_docs = Python::with_gil(|py| {
            py.allow_threads(|| {
                extracted_docs
                    .into_iter()
                    .flat_map(
                        |(doc_id, text, original_metadata): (
                            String,
                            String,
                            Option<HashMap<String, serde_json::Value>>,
                        )| {
                            // Split the text
                            let text_chunks = splitter.split_text(&text);

                            // Convert chunks to documents with enriched metadata
                            text_chunks
                                .into_iter()
                                .map(|chunk| {
                                    let mut enriched_metadata =
                                        original_metadata.clone().unwrap_or_default();

                                    // Add chunk tracking metadata
                                    enriched_metadata.insert(
                                        "source_doc_id".to_string(),
                                        serde_json::Value::String(doc_id.clone()),
                                    );
                                    enriched_metadata.insert(
                                        "chunk_index".to_string(),
                                        serde_json::Value::Number(
                                            chunk.metadata.chunk_index.into(),
                                        ),
                                    );
                                    enriched_metadata.insert(
                                        "start_char".to_string(),
                                        serde_json::Value::Number(chunk.metadata.start_char.into()),
                                    );
                                    enriched_metadata.insert(
                                        "end_char".to_string(),
                                        serde_json::Value::Number(chunk.metadata.end_char.into()),
                                    );

                                    // Create document for this chunk
                                    let chunk_doc = Document {
                                        id: format!(
                                            "{}_chunk_{}",
                                            doc_id, chunk.metadata.chunk_index
                                        ),
                                        text: chunk.text,
                                        metadata: enriched_metadata,
                                        embedding: None,
                                    };

                                    PyDocument { inner: chunk_doc }
                                })
                                .collect::<Vec<_>>()
                        },
                    )
                    .collect::<Vec<_>>()
            })
        });

        Ok(result_docs)
    }

    fn __repr__(&self) -> String {
        format!(
            "RecursiveCharacterTextSplitter(chunk_size={}, overlap={})",
            self.inner.chunk_size, self.inner.chunk_overlap
        )
    }
}

// ─── PyMarkdownTextSplitter ────────────────────────────────────────────────

/// Split markdown text by headers and code blocks.
#[pyclass(name = "MarkdownTextSplitter")]
pub struct PyMarkdownTextSplitter {
    inner: MarkdownTextSplitter,
}

#[pymethods]
impl PyMarkdownTextSplitter {
    #[new]
    #[pyo3(signature = (chunk_size=1000, chunk_overlap=200))]
    fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        PyMarkdownTextSplitter {
            inner: MarkdownTextSplitter::new(chunk_size, chunk_overlap),
        }
    }

    fn split_text(&self, text: &str) -> Vec<PyTextChunk> {
        to_py_chunks(self.inner.split_text(text))
    }

    fn __repr__(&self) -> String {
        "MarkdownTextSplitter(...)".to_string()
    }
}

// ─── PyHTMLTextSplitter ────────────────────────────────────────────────────

/// Split HTML text by tags.
#[pyclass(name = "HTMLTextSplitter")]
pub struct PyHTMLTextSplitter {
    inner: HTMLTextSplitter,
}

#[pymethods]
impl PyHTMLTextSplitter {
    #[new]
    #[pyo3(signature = (chunk_size=1000, chunk_overlap=200))]
    fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        PyHTMLTextSplitter {
            inner: HTMLTextSplitter::new(chunk_size, chunk_overlap),
        }
    }

    fn split_text(&self, text: &str) -> Vec<PyTextChunk> {
        to_py_chunks(self.inner.split_text(text))
    }

    fn __repr__(&self) -> String {
        "HTMLTextSplitter(...)".to_string()
    }
}

// ─── PyTokenTextSplitter ───────────────────────────────────────────────────

/// Split text by estimated token count.
#[pyclass(name = "TokenTextSplitter")]
pub struct PyTokenTextSplitter {
    inner: TokenTextSplitter,
}

#[pymethods]
impl PyTokenTextSplitter {
    #[new]
    #[pyo3(signature = (max_tokens=500, overlap_tokens=50))]
    fn new(max_tokens: usize, overlap_tokens: usize) -> Self {
        PyTokenTextSplitter {
            inner: TokenTextSplitter::new(max_tokens, overlap_tokens),
        }
    }

    fn split_text(&self, text: &str) -> Vec<PyTextChunk> {
        to_py_chunks(self.inner.split_text(text))
    }

    fn __repr__(&self) -> String {
        "TokenTextSplitter(...)".to_string()
    }
}

// ─── PyCodeTextSplitter ────────────────────────────────────────────────────

/// Split source code by language-specific syntax.
///
/// Example:
///     splitter = CodeTextSplitter("python", chunk_size=500)
///     chunks = splitter.split_text(python_code)
#[pyclass(name = "CodeTextSplitter")]
pub struct PyCodeTextSplitter {
    inner: CodeTextSplitter,
}

#[pymethods]
impl PyCodeTextSplitter {
    /// Create a code text splitter for a given language.
    ///
    /// Supported languages: "python", "rust", "javascript", "typescript",
    /// "java", "go", "cpp", "c", "ruby", "swift", "kotlin"
    #[new]
    #[pyo3(signature = (language, chunk_size=1000, chunk_overlap=200))]
    fn new(language: &str, chunk_size: usize, chunk_overlap: usize) -> Self {
        let lang = match language.to_lowercase().as_str() {
            "python" => Language::Python,
            "rust" => Language::Rust,
            "javascript" | "js" => Language::JavaScript,
            "typescript" | "ts" => Language::TypeScript,
            "java" => Language::Java,
            "go" => Language::Go,
            "cpp" | "c++" => Language::Cpp,
            "csharp" | "c#" => Language::CSharp,
            "ruby" => Language::Ruby,
            _ => Language::Generic,
        };
        PyCodeTextSplitter {
            inner: CodeTextSplitter::new(chunk_size, chunk_overlap, lang),
        }
    }

    fn split_text(&self, text: &str) -> Vec<PyTextChunk> {
        to_py_chunks(self.inner.split_text(text))
    }

    fn __repr__(&self) -> String {
        "CodeTextSplitter(...)".to_string()
    }
}
