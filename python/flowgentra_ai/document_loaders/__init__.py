"""Document loading and processing utilities.

This module provides tools for loading documents from various sources — files, URLs,
CSV/JSONL/DOCX, entire directory trees, spreadsheets, EPUBs, sitemaps, Wikipedia,
arXiv, RSS feeds, YouTube, S3, DataFrames, and git repositories.

Examples:
    Load basic file types:

        from flowgentra_ai.document_loaders import load_document, load_directory

        doc = load_document("./document.pdf")
        docs = load_directory("./docs")

    Load from a URL (web scraping):

        from flowgentra_ai.document_loaders import WebLoader

        loader = WebLoader()
        doc = await loader.load("https://www.rust-lang.org")

    Load Excel spreadsheets:

        from flowgentra_ai.document_loaders import ExcelLoader

        docs = ExcelLoader("data.xlsx").load()

    Load EPUB e-books:

        from flowgentra_ai.document_loaders import EpubLoader

        docs = EpubLoader("book.epub").load()

    Load from XML sitemap:

        from flowgentra_ai.document_loaders import SitemapLoader

        docs = await SitemapLoader("https://example.com/sitemap.xml").load()

    Load Wikipedia articles:

        from flowgentra_ai.document_loaders import WikipediaLoader

        docs = await WikipediaLoader(top_k=3).load("Rust programming language")

    Load arXiv papers:

        from flowgentra_ai.document_loaders import ArxivLoader

        docs = await ArxivLoader(max_results=5).load("transformer attention mechanism")

    Load from DataFrame:

        from flowgentra_ai.document_loaders import DataFrameLoader

        loader = DataFrameLoader.from_json(df.to_json(orient="records"), "content")
        docs = loader.load()

    Load from git repository:

        from flowgentra_ai.document_loaders import GitLoader

        docs = GitLoader("./my_repo").with_extensions(["rs", "py"]).load()
"""

from flowgentra_ai._native import data as _d, rag as _r

def _try(attr, default=None):
    return getattr(_r, attr, default)

# ── Basic file loaders ────────────────────────────────────────────────────────
FileType = _d.FileType
LoadedDocument = _d.LoadedDocument
load_document = _d.py_load_document
load_directory = _d.py_load_directory

# ── PDF ───────────────────────────────────────────────────────────────────────
PdfDocument = _r.PdfDocument
extract_pdf = _r.py_extract_pdf

# ── Ingestion pipeline ────────────────────────────────────────────────────────
IngestionPipeline = _d.IngestionPipeline
IngestionStats = _d.IngestionStats

# ── Web loader ────────────────────────────────────────────────────────────────
WebLoader = _r.WebLoader
WebLoaderConfig = _r.WebLoaderConfig

# ── CSV / JSON / JSONL loaders ────────────────────────────────────────────────
CsvLoader = _r.CsvLoader
JsonLoader = _r.JsonLoader
JsonlLoader = _r.JsonlLoader

# ── DOCX loader ───────────────────────────────────────────────────────────────
DocxLoader = _r.DocxLoader

# ── Recursive directory loader ────────────────────────────────────────────────
DirectoryLoader = _r.DirectoryLoader
DirectoryLoaderConfig = _r.DirectoryLoaderConfig

# ── Excel (XLSX) loader ───────────────────────────────────────────────────────
ExcelLoader = _try("ExcelLoader")

# ── EPUB loader ───────────────────────────────────────────────────────────────
EpubLoader = _try("EpubLoader")

# ── Sitemap loader ────────────────────────────────────────────────────────────
SitemapLoader = _try("SitemapLoader")
SitemapConfig = _try("SitemapConfig")

# ── Recursive URL loader ──────────────────────────────────────────────────────
RecursiveUrlLoader = _try("RecursiveUrlLoader")
RecursiveUrlConfig = _try("RecursiveUrlConfig")

# ── Wikipedia loader ──────────────────────────────────────────────────────────
WikipediaLoader = _try("WikipediaLoader")

# ── ArXiv loader ──────────────────────────────────────────────────────────────
ArxivLoader = _try("ArxivLoader")

# ── RSS / Atom feed loader ────────────────────────────────────────────────────
RssFeedLoader = _try("RssFeedLoader")

# ── YouTube loader ────────────────────────────────────────────────────────────
YouTubeLoader = _try("YouTubeLoader")

# ── S3 loader ─────────────────────────────────────────────────────────────────
S3Loader = _try("S3Loader")

# ── DataFrame loader ──────────────────────────────────────────────────────────
DataFrameLoader = _try("DataFrameLoader")

# ── Git repository loader ─────────────────────────────────────────────────────
GitLoader = _try("GitLoader")
GitLoaderConfig = _try("GitLoaderConfig")

__all__ = [
    # Types
    "FileType",
    "LoadedDocument",
    "PdfDocument",
    # Basic loading functions
    "load_document",
    "load_directory",
    "extract_pdf",
    # Ingestion pipeline
    "IngestionPipeline",
    "IngestionStats",
    # Original loaders
    "WebLoader",
    "WebLoaderConfig",
    "CsvLoader",
    "JsonLoader",
    "JsonlLoader",
    "DocxLoader",
    "DirectoryLoader",
    "DirectoryLoaderConfig",
    # New loaders
    "ExcelLoader",
    "EpubLoader",
    "SitemapLoader",
    "SitemapConfig",
    "RecursiveUrlLoader",
    "RecursiveUrlConfig",
    "WikipediaLoader",
    "ArxivLoader",
    "RssFeedLoader",
    "YouTubeLoader",
    "S3Loader",
    "DataFrameLoader",
    "GitLoader",
    "GitLoaderConfig",
]
