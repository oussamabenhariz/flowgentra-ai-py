//! Python bindings for the extended document loaders.
//!
//! All loaders return ``LoadedDocument`` instances compatible with
//! ``IngestionPipeline``.
//!
//! ```python
//! from flowgentra_ai.data import (
//!     WebLoader, CsvLoader, JsonLoader, JsonlLoader, DocxLoader,
//!     EpubLoader, ExcelLoader, DirectoryLoader, SitemapLoader,
//!     RecursiveUrlLoader, WikipediaLoader, ArxivLoader, RssFeedLoader,
//!     YouTubeLoader, S3Loader, DataFrameLoader, GitLoader,
//! )
//!
//! docs = WebLoader().load("https://example.com")
//! docs = CsvLoader().load("/path/to/data.csv")
//! docs = WikipediaLoader(top_k=3).load("Rust programming language")
//! ```

use pyo3::prelude::*;

use flowgentra_ai::core::rag::document_loader::LoadedDocument;
use flowgentra_ai::core::rag::loaders::{
    ArxivLoader, CsvLoader, DataFrameLoader, DirectoryLoader, DirectoryLoaderConfig, DocxLoader,
    EpubLoader, ExcelLoader, GitLoader, GitLoaderConfig, JsonLoader, JsonlLoader,
    RecursiveUrlConfig, RecursiveUrlLoader, RssFeedLoader, S3Loader, SitemapConfig, SitemapLoader,
    WebLoader, WebLoaderConfig, WikipediaLoader, YouTubeLoader,
};

use crate::document_loader::PyLoadedDocument;
use crate::run_async;

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    crate::error::InternalError::new_err(e.to_string())
}

fn to_py_loaded(docs: Vec<LoadedDocument>) -> Vec<PyLoadedDocument> {
    docs.into_iter().map(|d| PyLoadedDocument { inner: d }).collect()
}

// ── PyCsvLoader ───────────────────────────────────────────────────────────────

/// Loads CSV files. Each non-header row becomes a ``LoadedDocument``.
///
/// Example::
///
///     docs = CsvLoader().load("/path/to/data.csv")
///     docs = CsvLoader(text_column="body").load("/path/to/data.csv")
#[pyclass(name = "CsvLoader")]
pub struct PyCsvLoader {
    inner: CsvLoader,
}

#[pymethods]
impl PyCsvLoader {
    /// Args:
    ///     text_column: Column to use as document text (default auto-detect).
    #[new]
    #[pyo3(signature = (text_column=None))]
    fn new(text_column: Option<String>) -> Self {
        let mut inner = CsvLoader::new();
        if let Some(col) = text_column {
            inner = inner.with_text_column(col);
        }
        Self { inner }
    }

    /// Load all rows from ``path`` as documents.
    fn load(&self, path: &str) -> PyResult<Vec<PyLoadedDocument>> {
        run_async(async { self.inner.load(path).await })
            .map(to_py_loaded)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String { "CsvLoader(...)".to_string() }
}

// ── PyWebLoader ───────────────────────────────────────────────────────────────

/// Loads content from HTTP/HTTPS URLs.
///
/// Example::
///
///     loader = WebLoader()
///     doc  = loader.load("https://example.com")
///     docs = loader.load_many(["https://a.com", "https://b.com"])
#[pyclass(name = "WebLoader")]
pub struct PyWebLoader {
    inner: WebLoader,
}

#[pymethods]
impl PyWebLoader {
    /// Args:
    ///     timeout_secs:      Request timeout in seconds (default 30).
    ///     user_agent:        Custom User-Agent header.
    ///     decode_entities:   Decode HTML entities in text (default True).
    #[new]
    #[pyo3(signature = (timeout_secs=30, user_agent=None, decode_entities=true))]
    fn new(timeout_secs: u64, user_agent: Option<String>, decode_entities: bool) -> Self {
        let mut config = WebLoaderConfig::default();
        config.timeout_secs = timeout_secs;
        config.decode_entities = decode_entities;
        if let Some(ua) = user_agent { config.user_agent = ua; }
        Self { inner: WebLoader::with_config(config) }
    }

    /// Load a single URL. Returns a single ``LoadedDocument``.
    fn load(&self, url: &str) -> PyResult<PyLoadedDocument> {
        run_async(async { self.inner.load(url).await })
            .map(|d| PyLoadedDocument { inner: d })
            .map_err(to_py_err)
    }

    /// Load multiple URLs concurrently. Returns a list of ``LoadedDocument``.
    fn load_many(&self, urls: Vec<String>) -> PyResult<Vec<PyLoadedDocument>> {
        let results = run_async(async { self.inner.load_many(urls).await });
        results
            .into_iter()
            .map(|r| r.map(|d| PyLoadedDocument { inner: d }).map_err(to_py_err))
            .collect()
    }

    fn __repr__(&self) -> String { "WebLoader(...)".to_string() }
}

// ── PyJsonLoader ──────────────────────────────────────────────────────────────

/// Loads JSON files (array of objects). Extracts one field as document text.
///
/// Example::
///
///     docs = JsonLoader(text_field="content").load("/path/to/data.json")
#[pyclass(name = "JsonLoader")]
pub struct PyJsonLoader {
    inner: JsonLoader,
}

#[pymethods]
impl PyJsonLoader {
    /// Args:
    ///     text_field: Object field to use as text (default ``"text"``).
    #[new]
    #[pyo3(signature = (text_field="text"))]
    fn new(text_field: &str) -> Self {
        Self { inner: JsonLoader::new(text_field) }
    }

    /// Load documents from ``path``.
    fn load(&self, path: &str) -> PyResult<Vec<PyLoadedDocument>> {
        run_async(async { self.inner.load(path).await })
            .map(to_py_loaded)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String { "JsonLoader(...)".to_string() }
}

// ── PyJsonlLoader ─────────────────────────────────────────────────────────────

/// Loads JSONL (newline-delimited JSON) files.
///
/// Example::
///
///     docs = JsonlLoader(text_field="body").load("/path/to/data.jsonl")
#[pyclass(name = "JsonlLoader")]
pub struct PyJsonlLoader {
    inner: JsonlLoader,
}

#[pymethods]
impl PyJsonlLoader {
    /// Args:
    ///     text_field: Field to use as text (default ``"text"``).
    #[new]
    #[pyo3(signature = (text_field="text"))]
    fn new(text_field: &str) -> Self {
        Self { inner: JsonlLoader::new(text_field) }
    }

    fn load(&self, path: &str) -> PyResult<Vec<PyLoadedDocument>> {
        run_async(async { self.inner.load(path).await })
            .map(to_py_loaded)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String { "JsonlLoader(...)".to_string() }
}

// ── PyDocxLoader ──────────────────────────────────────────────────────────────

/// Loads Microsoft Word ``.docx`` files.
///
/// Example::
///
///     doc = DocxLoader().load("/path/to/report.docx")
#[pyclass(name = "DocxLoader")]
pub struct PyDocxLoader;

#[pymethods]
impl PyDocxLoader {
    #[new]
    fn new() -> Self { Self }

    /// Load a single ``.docx`` file. Returns a ``LoadedDocument``.
    fn load(&self, path: &str) -> PyResult<PyLoadedDocument> {
        let loader = DocxLoader::new();
        run_async(async move { loader.load(path).await })
            .map(|d| PyLoadedDocument { inner: d })
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String { "DocxLoader()".to_string() }
}

// ── PyEpubLoader ──────────────────────────────────────────────────────────────

/// Loads ``.epub`` e-book files. Each chapter becomes a ``LoadedDocument``.
///
/// Example::
///
///     docs = EpubLoader("/path/to/book.epub").load()
#[pyclass(name = "EpubLoader")]
pub struct PyEpubLoader {
    path: String,
}

#[pymethods]
impl PyEpubLoader {
    /// Args:
    ///     path: Path to the ``.epub`` file.
    #[new]
    fn new(path: &str) -> Self {
        Self { path: path.to_string() }
    }

    /// Load all chapters.
    fn load(&self) -> PyResult<Vec<PyLoadedDocument>> {
        let loader = EpubLoader::new(&self.path);
        loader.load().map(to_py_loaded).map_err(to_py_err)
    }

    fn __repr__(&self) -> String { format!("EpubLoader('{}')", self.path) }
}

// ── PyExcelLoader ─────────────────────────────────────────────────────────────

/// Loads ``.xlsx`` Excel spreadsheet files. Each row becomes a ``LoadedDocument``.
///
/// Example::
///
///     docs = ExcelLoader("/path/to/data.xlsx").load()
#[pyclass(name = "ExcelLoader")]
pub struct PyExcelLoader {
    path: String,
}

#[pymethods]
impl PyExcelLoader {
    /// Args:
    ///     path: Path to the ``.xlsx`` file.
    #[new]
    fn new(path: &str) -> Self {
        Self { path: path.to_string() }
    }

    fn load(&self) -> PyResult<Vec<PyLoadedDocument>> {
        let loader = ExcelLoader::new(&self.path);
        loader.load().map(to_py_loaded).map_err(to_py_err)
    }

    fn __repr__(&self) -> String { format!("ExcelLoader('{}')", self.path) }
}

// ── PyDirectoryLoader ─────────────────────────────────────────────────────────

/// Recursively loads documents from a directory tree.
///
/// Example::
///
///     # All supported file types, recursive
///     docs = DirectoryLoader("/data").load()
///
///     # Only .txt and .md files
///     docs = DirectoryLoader("/data", extensions=["txt", "md"]).load()
#[pyclass(name = "DirectoryLoader")]
pub struct PyDirectoryLoader {
    root: String,
    config: DirectoryLoaderConfig,
}

#[pymethods]
impl PyDirectoryLoader {
    /// Args:
    ///     root:        Root directory path.
    ///     extensions:  File extensions to include (default all supported).
    ///     recursive:   Recurse into subdirectories (default True).
    ///     skip_errors: Skip unreadable files instead of raising (default True).
    #[new]
    #[pyo3(signature = (root, extensions=None, recursive=true, skip_errors=true))]
    fn new(
        root: &str,
        extensions: Option<Vec<String>>,
        recursive: bool,
        skip_errors: bool,
    ) -> Self {
        let mut config = DirectoryLoaderConfig::default();
        if let Some(exts) = extensions { config.extensions = exts; }
        config.recursive = recursive;
        config.skip_errors = skip_errors;
        Self { root: root.to_string(), config }
    }

    /// Load all matching documents.
    fn load(&self) -> PyResult<Vec<PyLoadedDocument>> {
        let loader = DirectoryLoader::new(&self.root, self.config.clone());
        run_async(async move { loader.load().await })
            .map(to_py_loaded)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String { format!("DirectoryLoader('{}')", self.root) }
}

// ── PySitemapLoader ───────────────────────────────────────────────────────────

/// Loads URLs listed in an XML sitemap, fetching and parsing each page.
///
/// Example::
///
///     docs = SitemapLoader("https://example.com/sitemap.xml").load()
///     docs = SitemapLoader("https://example.com/sitemap.xml", max_pages=50).load()
#[pyclass(name = "SitemapLoader")]
pub struct PySitemapLoader {
    url: String,
    max_pages: usize,
}

#[pymethods]
impl PySitemapLoader {
    /// Args:
    ///     sitemap_url: URL of the XML sitemap.
    ///     max_pages:   Maximum pages to fetch (default 100).
    #[new]
    #[pyo3(signature = (sitemap_url, max_pages=100))]
    fn new(sitemap_url: &str, max_pages: usize) -> Self {
        Self { url: sitemap_url.to_string(), max_pages }
    }

    fn load(&self) -> PyResult<Vec<PyLoadedDocument>> {
        let loader = SitemapLoader::new(&self.url).with_config(SitemapConfig {
            max_pages: Some(self.max_pages),
            ..Default::default()
        });
        run_async(async move { loader.load().await })
            .map(to_py_loaded)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String { format!("SitemapLoader('{}')", self.url) }
}

// ── PyRecursiveUrlLoader ──────────────────────────────────────────────────────

/// Crawls a website starting from a root URL, following internal links.
///
/// Example::
///
///     docs = RecursiveUrlLoader("https://docs.example.com", max_depth=2).load()
#[pyclass(name = "RecursiveUrlLoader")]
pub struct PyRecursiveUrlLoader {
    root_url: String,
    max_depth: usize,
    max_pages: usize,
}

#[pymethods]
impl PyRecursiveUrlLoader {
    /// Args:
    ///     root_url:  Starting URL.
    ///     max_depth: Link-following depth (default 2).
    ///     max_pages: Maximum pages to crawl (default 50).
    #[new]
    #[pyo3(signature = (root_url, max_depth=2, max_pages=50))]
    fn new(root_url: &str, max_depth: usize, max_pages: usize) -> Self {
        Self { root_url: root_url.to_string(), max_depth, max_pages }
    }

    fn load(&self) -> PyResult<Vec<PyLoadedDocument>> {
        let loader = RecursiveUrlLoader::new(&self.root_url)
            .with_config(RecursiveUrlConfig {
                max_depth: self.max_depth,
                max_pages: self.max_pages,
                ..Default::default()
            });
        run_async(async move { loader.load().await })
            .map(to_py_loaded)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String { format!("RecursiveUrlLoader('{}')", self.root_url) }
}

// ── PyWikipediaLoader ─────────────────────────────────────────────────────────

/// Fetches Wikipedia article summaries matching a search query.
///
/// Example::
///
///     docs = WikipediaLoader(top_k=3).load("Rust programming language")
#[pyclass(name = "WikipediaLoader")]
pub struct PyWikipediaLoader {
    top_k: usize,
    lang: String,
}

#[pymethods]
impl PyWikipediaLoader {
    /// Args:
    ///     top_k: Number of articles to fetch (default 3).
    ///     lang:  Wikipedia language code (default ``"en"``).
    #[new]
    #[pyo3(signature = (top_k=3, lang="en"))]
    fn new(top_k: usize, lang: &str) -> Self {
        Self { top_k, lang: lang.to_string() }
    }

    /// Fetch Wikipedia articles for ``query``.
    fn load(&self, query: &str) -> PyResult<Vec<PyLoadedDocument>> {
        let loader = WikipediaLoader::new(self.top_k).with_lang(&self.lang);
        let q = query.to_string();
        run_async(async move { loader.load(&q).await })
            .map(to_py_loaded)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String { format!("WikipediaLoader(top_k={})", self.top_k) }
}

// ── PyArxivLoader ─────────────────────────────────────────────────────────────

/// Fetches arXiv paper abstracts matching a search query.
///
/// Example::
///
///     docs = ArxivLoader(max_results=5).load("attention is all you need")
#[pyclass(name = "ArxivLoader")]
pub struct PyArxivLoader {
    max_results: usize,
}

#[pymethods]
impl PyArxivLoader {
    /// Args:
    ///     max_results: Maximum papers to fetch (default 5).
    #[new]
    #[pyo3(signature = (max_results=5))]
    fn new(max_results: usize) -> Self {
        Self { max_results }
    }

    /// Fetch arXiv papers matching ``query``.
    fn load(&self, query: &str) -> PyResult<Vec<PyLoadedDocument>> {
        let loader = ArxivLoader::new(self.max_results);
        let q = query.to_string();
        run_async(async move { loader.load(&q).await })
            .map(to_py_loaded)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String { format!("ArxivLoader(max_results={})", self.max_results) }
}

// ── PyRssFeedLoader ───────────────────────────────────────────────────────────

/// Loads RSS / Atom feed entries as documents.
///
/// Example::
///
///     docs = RssFeedLoader("https://feeds.example.com/rss.xml").load()
#[pyclass(name = "RssFeedLoader")]
pub struct PyRssFeedLoader {
    feed_url: String,
}

#[pymethods]
impl PyRssFeedLoader {
    /// Args:
    ///     feed_url: URL of the RSS or Atom feed.
    #[new]
    fn new(feed_url: &str) -> Self {
        Self { feed_url: feed_url.to_string() }
    }

    fn load(&self) -> PyResult<Vec<PyLoadedDocument>> {
        let loader = RssFeedLoader::new(&self.feed_url);
        run_async(async move { loader.load().await })
            .map(to_py_loaded)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String { format!("RssFeedLoader('{}')", self.feed_url) }
}

// ── PyYouTubeLoader ───────────────────────────────────────────────────────────

/// Fetches YouTube video metadata (title, description, channel) as documents.
///
/// Requires a YouTube Data API v3 key.
///
/// Example::
///
///     loader = YouTubeLoader(api_key="AIza...")
///     docs = loader.load_by_query("Rust tutorial", max_results=5)
///     doc  = loader.load_by_id("dQw4w9WgXcQ")
#[pyclass(name = "YouTubeLoader")]
pub struct PyYouTubeLoader {
    api_key: String,
}

#[pymethods]
impl PyYouTubeLoader {
    /// Args:
    ///     api_key: YouTube Data API v3 key.
    #[new]
    fn new(api_key: &str) -> Self {
        Self { api_key: api_key.to_string() }
    }

    /// Search YouTube and load video metadata.
    #[pyo3(signature = (query, max_results=5))]
    fn load_by_query(&self, query: &str, max_results: usize) -> PyResult<Vec<PyLoadedDocument>> {
        let loader = YouTubeLoader::new(&self.api_key).with_max_results(max_results);
        let q = query.to_string();
        run_async(async move { loader.load_by_query(&q).await })
            .map(to_py_loaded)
            .map_err(to_py_err)
    }

    /// Load metadata for one or more videos by ID.
    fn load_by_id(&self, video_id: &str) -> PyResult<Vec<PyLoadedDocument>> {
        let loader = YouTubeLoader::new(&self.api_key);
        let vid = video_id.to_string();
        run_async(async move { loader.load_by_id(&vid).await })
            .map(to_py_loaded)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String { "YouTubeLoader(...)".to_string() }
}

// ── PyS3Loader ────────────────────────────────────────────────────────────────

/// Loads documents from an AWS S3 bucket.
///
/// Example::
///
///     docs = S3Loader(
///         bucket="my-bucket",
///         region="us-east-1",
///         access_key="AKIA...",
///         secret_key="...",
///         prefix="docs/",
///     ).load()
#[pyclass(name = "S3Loader")]
pub struct PyS3Loader {
    bucket: String,
    region: String,
    access_key: String,
    secret_key: String,
    prefix: Option<String>,
}

#[pymethods]
impl PyS3Loader {
    /// Args:
    ///     bucket:     S3 bucket name.
    ///     region:     AWS region (e.g. ``"us-east-1"``).
    ///     access_key: AWS Access Key ID.
    ///     secret_key: AWS Secret Access Key.
    ///     prefix:     Optional key prefix filter (e.g. ``"docs/"``).
    #[new]
    #[pyo3(signature = (bucket, region, access_key, secret_key, prefix=None))]
    fn new(
        bucket: &str,
        region: &str,
        access_key: &str,
        secret_key: &str,
        prefix: Option<String>,
    ) -> Self {
        Self {
            bucket: bucket.to_string(),
            region: region.to_string(),
            access_key: access_key.to_string(),
            secret_key: secret_key.to_string(),
            prefix,
        }
    }

    fn load(&self) -> PyResult<Vec<PyLoadedDocument>> {
        let mut loader = S3Loader::new(&self.bucket, &self.region, &self.access_key, &self.secret_key);
        if let Some(p) = &self.prefix {
            loader = loader.with_prefix(p);
        }
        run_async(async move { loader.load().await })
            .map(to_py_loaded)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!("S3Loader(bucket='{}')", self.bucket)
    }
}

// ── PyDataFrameLoader ─────────────────────────────────────────────────────────

/// Loads tabular data (list of row dicts) as documents.
///
/// Each row becomes one ``LoadedDocument`` with the ``page_content_column``
/// field as the document text and the remaining columns as metadata.
///
/// Example::
///
///     rows = [{"title": "Rust", "body": "Rust is a systems language."}, ...]
///     docs = DataFrameLoader(rows, page_content_column="body").load()
#[pyclass(name = "DataFrameLoader")]
pub struct PyDataFrameLoader {
    rows: Vec<std::collections::HashMap<String, String>>,
    page_content_column: String,
}

#[pymethods]
impl PyDataFrameLoader {
    /// Args:
    ///     rows:                List of dicts (string values only).
    ///     page_content_column: Field to use as document text (default ``"text"``).
    #[new]
    #[pyo3(signature = (rows, page_content_column="text"))]
    fn new(
        rows: Vec<std::collections::HashMap<String, String>>,
        page_content_column: &str,
    ) -> Self {
        Self { rows, page_content_column: page_content_column.to_string() }
    }

    fn load(&self) -> Vec<PyLoadedDocument> {
        let loader = DataFrameLoader::new(self.rows.clone(), &self.page_content_column);
        to_py_loaded(loader.load())
    }

    fn __repr__(&self) -> String {
        format!("DataFrameLoader(rows={}, column='{}')", self.rows.len(), self.page_content_column)
    }
}

// ── PyGitLoader ───────────────────────────────────────────────────────────────

/// Loads source files from a local git repository.
///
/// Example::
///
///     docs = GitLoader("/path/to/repo").load()
///     docs = GitLoader("/path/to/repo", extensions=["rs", "py"]).load()
#[pyclass(name = "GitLoader")]
pub struct PyGitLoader {
    repo_path: String,
    extensions: Vec<String>,
    branch: Option<String>,
}

#[pymethods]
impl PyGitLoader {
    /// Args:
    ///     repo_path:   Path to the local git repository.
    ///     extensions:  File extensions to include (e.g. ``["rs", "py"]``). Empty = all.
    ///     branch:      Git branch to load (default current branch).
    #[new]
    #[pyo3(signature = (repo_path, extensions=None, branch=None))]
    fn new(repo_path: &str, extensions: Option<Vec<String>>, branch: Option<String>) -> Self {
        Self {
            repo_path: repo_path.to_string(),
            extensions: extensions.unwrap_or_default(),
            branch,
        }
    }

    fn load(&self) -> PyResult<Vec<PyLoadedDocument>> {
        let mut config = GitLoaderConfig::default();
        config.extensions = self.extensions.clone();
        if let Some(b) = &self.branch { config.branch = Some(b.clone()); }
        let loader = GitLoader::new(&self.repo_path).with_config(config);
        loader.load().map(to_py_loaded).map_err(to_py_err)
    }

    fn __repr__(&self) -> String { format!("GitLoader('{}')", self.repo_path) }
}
