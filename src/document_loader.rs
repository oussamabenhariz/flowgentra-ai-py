//! Python bindings for document loaders

use pyo3::prelude::*;

use flowgentra_ai::core::rag::{load_document, load_directory, FileType, LoadedDocument};

use crate::error::to_py_err_generic;

// ─── PyFileType ────────────────────────────────────────────────────────────

/// File type detected from extension.
///
/// Example:
///     ft = FileType.from_path("readme.md")  # FileType.Markdown
#[pyclass(name = "FileType")]
#[derive(Clone)]
pub struct PyFileType {
    inner: FileType,
}

#[pymethods]
impl PyFileType {
    /// Detect file type from a file path.
    #[staticmethod]
    fn from_path(path: &str) -> Self {
        PyFileType {
            inner: FileType::from_path(std::path::Path::new(path)),
        }
    }

    #[staticmethod]
    fn plain_text() -> Self {
        PyFileType { inner: FileType::PlainText }
    }

    #[staticmethod]
    fn markdown() -> Self {
        PyFileType { inner: FileType::Markdown }
    }

    #[staticmethod]
    fn html() -> Self {
        PyFileType { inner: FileType::Html }
    }

    #[staticmethod]
    fn pdf() -> Self {
        PyFileType { inner: FileType::Pdf }
    }

    #[staticmethod]
    fn unknown() -> Self {
        PyFileType { inner: FileType::Unknown }
    }

    fn __repr__(&self) -> String {
        let name = match &self.inner {
            FileType::PlainText => "PlainText",
            FileType::Markdown => "Markdown",
            FileType::Html => "Html",
            FileType::Pdf => "Pdf",
            FileType::Unknown => "Unknown",
        };
        format!("FileType.{}", name)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ─── PyLoadedDocument ──────────────────────────────────────────────────────

/// A document loaded from a file.
///
/// Example:
///     doc = load_document("readme.md")
///     print(doc.text)
#[pyclass(name = "LoadedDocument")]
pub struct PyLoadedDocument {
    pub(crate) inner: LoadedDocument,
}

#[pymethods]
impl PyLoadedDocument {
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    #[getter]
    fn file_type(&self) -> PyFileType {
        PyFileType {
            inner: self.inner.file_type.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LoadedDocument(id='{}', source='{}', len={})",
            self.inner.id,
            self.inner.source,
            self.inner.text.len()
        )
    }
}

// ─── Free functions ────────────────────────────────────────────────────────

/// Load a single document from a file path.
#[pyfunction]
pub fn py_load_document(path: &str) -> PyResult<PyLoadedDocument> {
    let doc = crate::run_async(load_document(path)).map_err(to_py_err_generic)?;
    Ok(PyLoadedDocument { inner: doc })
}

/// Load all documents from a directory.
#[pyfunction]
pub fn py_load_directory(dir: &str) -> PyResult<Vec<PyLoadedDocument>> {
    let docs = crate::run_async(load_directory(dir)).map_err(to_py_err_generic)?;
    Ok(docs
        .into_iter()
        .map(|d| PyLoadedDocument { inner: d })
        .collect())
}
