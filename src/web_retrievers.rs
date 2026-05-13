//! Python bindings for web-based retrievers.
//!
//! These retrievers fetch results from public web APIs and return them as
//! ``SearchResult`` objects, compatible with the retriever duck-typing
//! interface used by ``EnsembleRetriever`` and similar composable retrievers.
//!
//! ```python
//! from flowgentra_ai.rag import WikipediaRetriever, ArxivRetriever, TavilySearchRetriever
//!
//! wiki = WikipediaRetriever(top_k=3, lang="en")
//! results = wiki.retrieve("Rust programming language")
//! for r in results:
//!     print(r.text[:200])
//!
//! arxiv = ArxivRetriever(top_k=5, sort_by_date=True)
//! results = arxiv.retrieve("transformer neural networks")
//!
//! tavily = TavilySearchRetriever(api_key="tvly-...", top_k=5, advanced=True)
//! results = tavily.retrieve("latest AI research 2024")
//! ```

use pyo3::prelude::*;

use flowgentra_ai::core::rag::{
    ensemble_retriever::AsyncRetriever as _,
    web_retrievers::{ArxivRetriever, TavilySearchRetriever, WikipediaRetriever},
};

use crate::rag::PySearchResult;
use crate::run_async;

fn to_py_err(e: flowgentra_ai::core::rag::vector_db::VectorStoreError) -> PyErr {
    crate::error::to_py_err_generic(e)
}

// ── PyWikipediaRetriever ──────────────────────────────────────────────────────

/// Retrieves Wikipedia article summaries for a query.
///
/// Uses the Wikipedia OpenSearch API to find matching article titles, then
/// fetches each article's summary via the REST API. Free — no API key required.
///
/// Args:
///     top_k:   Maximum number of results to return (default 3).
///     lang:    Wikipedia language code (default ``"en"``).
///
/// Example::
///
///     wiki = WikipediaRetriever(top_k=3, lang="en")
///     results = wiki.retrieve("Rust programming language")
///     for r in results:
///         print(r.score, r.text[:100])
#[pyclass(name = "WikipediaRetriever")]
pub struct PyWikipediaRetriever {
    inner: WikipediaRetriever,
}

#[pymethods]
impl PyWikipediaRetriever {
    /// Create a Wikipedia retriever.
    ///
    /// Args:
    ///     top_k:   Maximum number of results.
    ///     lang:    Wikipedia language code (e.g. ``"en"``, ``"fr"``).
    #[new]
    #[pyo3(signature = (top_k = 3, lang = "en"))]
    fn new(top_k: usize, lang: &str) -> Self {
        let retriever = WikipediaRetriever::new(top_k).with_lang(lang);
        Self { inner: retriever }
    }

    /// Retrieve Wikipedia summaries for a query.
    ///
    /// Args:
    ///     query: Search string.
    ///
    /// Returns:
    ///     List of ``SearchResult`` objects ordered by relevance rank.
    fn retrieve(&self, query: &str) -> PyResult<Vec<PySearchResult>> {
        let q = query.to_string();
        let results = run_async(async { self.inner.retrieve(&q).await }).map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|r| PySearchResult { inner: r })
            .collect())
    }

    fn __repr__(&self) -> String {
        "WikipediaRetriever(...)".to_string()
    }
}

// ── PyArxivRetriever ──────────────────────────────────────────────────────────

/// Retrieves arXiv paper abstracts for a query.
///
/// Uses the arXiv Atom feed API. Free — no API key required.
///
/// Args:
///     top_k:        Maximum number of results (default 5).
///     sort_by_date: Sort by submission date instead of relevance (default False).
///
/// Example::
///
///     arxiv = ArxivRetriever(top_k=5, sort_by_date=True)
///     results = arxiv.retrieve("transformer attention mechanism")
///     for r in results:
///         print(r.metadata["title"], r.score)
#[pyclass(name = "ArxivRetriever")]
pub struct PyArxivRetriever {
    inner: ArxivRetriever,
}

#[pymethods]
impl PyArxivRetriever {
    /// Create an arXiv retriever.
    ///
    /// Args:
    ///     top_k:        Maximum number of paper abstracts to return.
    ///     sort_by_date: If ``True``, sort by submission date (newest first).
    #[new]
    #[pyo3(signature = (top_k = 5, sort_by_date = false))]
    fn new(top_k: usize, sort_by_date: bool) -> Self {
        let mut retriever = ArxivRetriever::new(top_k);
        if sort_by_date {
            retriever = retriever.sort_by_date();
        }
        Self { inner: retriever }
    }

    /// Retrieve arXiv paper abstracts for a query.
    ///
    /// Args:
    ///     query: Full-text search string (supports arXiv query syntax).
    ///
    /// Returns:
    ///     List of ``SearchResult`` objects. ``metadata`` includes ``title``,
    ///     ``url``, and ``authors``.
    fn retrieve(&self, query: &str) -> PyResult<Vec<PySearchResult>> {
        let q = query.to_string();
        let results = run_async(async { self.inner.retrieve(&q).await }).map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|r| PySearchResult { inner: r })
            .collect())
    }

    fn __repr__(&self) -> String {
        "ArxivRetriever(...)".to_string()
    }
}

// ── PyTavilySearchRetriever ───────────────────────────────────────────────────

/// Retrieves web search results using the Tavily AI Search API.
///
/// Requires a Tavily API key. Set ``TAVILY_API_KEY`` in your environment or
/// pass it directly. Advanced search mode produces deeper, higher-quality
/// results but uses more credits.
///
/// Args:
///     api_key:  Tavily API key (``"tvly-..."``).
///     top_k:    Maximum number of results (default 5).
///     advanced: Use advanced search depth (default False).
///
/// Example::
///
///     import os
///     tavily = TavilySearchRetriever(
///         api_key=os.environ["TAVILY_API_KEY"],
///         top_k=5,
///         advanced=True,
///     )
///     results = tavily.retrieve("Claude AI capabilities 2024")
///     for r in results:
///         print(r.score, r.metadata["url"])
#[pyclass(name = "TavilySearchRetriever")]
pub struct PyTavilySearchRetriever {
    inner: TavilySearchRetriever,
}

#[pymethods]
impl PyTavilySearchRetriever {
    /// Create a Tavily search retriever.
    ///
    /// Args:
    ///     api_key:  Tavily API key.
    ///     top_k:    Maximum number of results.
    ///     advanced: Enable advanced search depth for higher quality results.
    #[new]
    #[pyo3(signature = (api_key, top_k = 5, advanced = false))]
    fn new(api_key: &str, top_k: usize, advanced: bool) -> Self {
        let mut retriever = TavilySearchRetriever::new(api_key, top_k);
        if advanced {
            retriever = retriever.with_advanced_search();
        }
        Self { inner: retriever }
    }

    /// Retrieve web search results for a query.
    ///
    /// Args:
    ///     query: Natural language search query.
    ///
    /// Returns:
    ///     List of ``SearchResult`` objects ordered by relevance score.
    ///     ``metadata`` includes ``title`` and ``url``.
    fn retrieve(&self, query: &str) -> PyResult<Vec<PySearchResult>> {
        let q = query.to_string();
        let results = run_async(async { self.inner.retrieve(&q).await }).map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|r| PySearchResult { inner: r })
            .collect())
    }

    fn __repr__(&self) -> String {
        "TavilySearchRetriever(...)".to_string()
    }
}
