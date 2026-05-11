"""Retrieval-augmented generation (RAG) components.

This module provides document retrieval, embedding, vector stores, and text processing
for RAG workflows.

Examples:
    Set up RAG with vector stores:

        from flowgentra_ai.rag import (
            InMemoryVectorStore,
            HnswVectorStore,
            Embeddings,
            RecursiveCharacterTextSplitter
        )

    Advanced retrievers:

        from flowgentra_ai.rag import (
            ParentDocumentRetriever, ParentDocConfig, ParentDocument,
            MultiVectorRetriever,  MultiVectorConfig, VectorView,
            TimeWeightedRetriever, TimeWeightedConfig,
            Bm25Retriever, Bm25Config,
            EnsembleRetriever, VectorRetriever, Bm25RetrieverAdapter, EnsembleConfig,
            ContextualCompressionRetriever, EmbeddingsFilter,
            ScoreThresholdRetriever,
            MultiQueryRetriever, MultiQueryConfig,
            WikipediaRetriever, ArxivRetriever, TavilySearchRetriever,
            reorder, ReorderStrategy,
        )

    Additional vector stores:

        from flowgentra_ai.rag import (
            SingleStoreVectorStore,
            AzureAISearchStore,
            VectaraStore,
            Neo4jVectorStore,
            TurbopufferStore,
        )

    Additional embedding providers:

        from flowgentra_ai.rag import (
            CohereEmbeddings,
            AzureOpenAIEmbeddings,
            GoogleVertexEmbeddings,
            VoyageEmbeddings,
            JinaEmbeddings,
            TogetherEmbeddings,
            NomicEmbeddings,
        )

    SQL database agent tools:

        from flowgentra_ai.rag import (
            SqlDatabaseWrapper,
            ListSQLDatabaseTool,
            InfoSQLDatabaseTool,
            QuerySQLDatabaseTool,
            QuerySQLCheckerTool,
            SqlDatabaseLoader,
        )

    Document store:

        from flowgentra_ai.rag import InMemoryDocStore, LocalFileDocStore

    Indexing with deduplication:

        from flowgentra_ai.rag import index, InMemoryRecordManager, CleanupMode

    Configure RAG retrieval:

        from flowgentra_ai.rag import RAGConfig, RetrievalConfig
"""

from flowgentra_ai._native import rag as _r, text as _tx, utils as _u

def _try(attr, default=None):
    return getattr(_r, attr, default)

# ── Core document/search types ────────────────────────────────────────────────
Document = _r.Document
SearchResult = _r.SearchResult
TextChunk = _r.TextChunk

# ── Embeddings ────────────────────────────────────────────────────────────────
Embeddings = _r.Embeddings

# ── Standard retriever ────────────────────────────────────────────────────────
Retriever = _r.Retriever
RetrievalConfig = _r.RetrievalConfig

# ── Vector stores ─────────────────────────────────────────────────────────────
VectorStoreType = _r.VectorStoreType
RAGConfig = _r.RAGConfig
VectorStoreConfig = _r.VectorStoreConfig
EmbeddingsConfig = _r.EmbeddingsConfig
RetrievalSettings = _r.RetrievalSettings
PdfSettings = _r.PdfSettings
RAGGraphConfig = _r.RAGGraphConfig
ChromaStore = _r.ChromaStore
InMemoryVectorStore = _r.InMemoryVectorStore

# ── Python embedded Chroma (no server) ───────────────────────────────────────
from flowgentra_ai.rag.chroma_local import Chroma

# ── HNSW / local persistent vector store ─────────────────────────────────────
HnswVectorStore = _r.HnswVectorStore

# ── Advanced retrievers (round 1) ─────────────────────────────────────────────
# Config/data classes are not yet exposed as PyO3 bindings; guard to avoid ImportError.
_unbound = ["SelfQueryConfig", "ParentDocConfig", "ParentDocument",
            "MultiVectorConfig", "MultiVectorParent", "TimeWeightedConfig"]
for _name in _unbound:
    try:
        globals()[_name] = getattr(_r, _name)
    except AttributeError:
        pass
del _unbound, _name

ParentDocumentRetriever = _r.ParentDocumentRetriever
MultiVectorRetriever = _r.MultiVectorRetriever
VectorView = _r.VectorView
TimeWeightedRetriever = _r.TimeWeightedRetriever

# ── BM25 / Ensemble / Reorder ─────────────────────────────────────────────────
Bm25Config = _try("Bm25Config")
Bm25Document = _try("Bm25Document")
Bm25Retriever = _r.Bm25Retriever
EnsembleConfig = _try("EnsembleConfig")
EnsembleRetriever = _r.EnsembleRetriever
VectorRetriever = _r.VectorRetriever
Bm25RetrieverAdapter = _try("Bm25RetrieverAdapter")
ReorderStrategy = _r.ReorderStrategy
reorder = _try("py_reorder")
reorder_for_long_context = _r.py_reorder_for_long_context
filter_from_json_object = _try("py_filter_from_json_object")

# ── Advanced retrievers (round 2) ─────────────────────────────────────────────
ContextualCompressionRetriever = _try("ContextualCompressionRetriever")
EmbeddingsFilter = _try("EmbeddingsFilter")
LLMCompressor = _try("LLMCompressor")
DocumentCompressorPipeline = _try("DocumentCompressorPipeline")
ScoreThresholdRetriever = _try("ScoreThresholdRetriever")
MultiQueryConfig = _try("MultiQueryConfig")
MultiQueryRetriever = _try("MultiQueryRetriever")

# ── Web retrievers ────────────────────────────────────────────────────────────
WikipediaRetriever = _try("WikipediaRetriever")
ArxivRetriever = _try("ArxivRetriever")
TavilySearchRetriever = _try("TavilySearchRetriever")

# ── Additional vector stores ──────────────────────────────────────────────────
SingleStoreVectorStore = _try("SingleStoreVectorStore")
AzureAISearchStore = _try("AzureAISearchStore")
VectaraStore = _try("VectaraStore")
Neo4jVectorStore = _try("Neo4jVectorStore")
TurbopufferStore = _try("TurbopufferStore")

# ── Additional embedding providers ───────────────────────────────────────────
CohereEmbeddings = _try("CohereEmbeddings")
AzureOpenAIEmbeddings = _try("AzureOpenAIEmbeddings")
GoogleVertexEmbeddings = _try("GoogleVertexEmbeddings")
BedrockEmbeddings = _try("BedrockEmbeddings")
VoyageEmbeddings = _try("VoyageEmbeddings")
JinaEmbeddings = _try("JinaEmbeddings")
TogetherEmbeddings = _try("TogetherEmbeddings")
NomicEmbeddings = _try("NomicEmbeddings")

# ── SQL database agent toolkit ────────────────────────────────────────────────
SqlDatabaseWrapper = _try("SqlDatabaseWrapper")
ListSQLDatabaseTool = _try("ListSQLDatabaseTool")
InfoSQLDatabaseTool = _try("InfoSQLDatabaseTool")
QuerySQLDatabaseTool = _try("QuerySQLDatabaseTool")
QuerySQLCheckerTool = _try("QuerySQLCheckerTool")
SqlDatabaseLoader = _try("SqlDatabaseLoader")

# ── Document store ────────────────────────────────────────────────────────────
StoredDocument = _try("StoredDocument")
InMemoryDocStore = _try("InMemoryDocStore")
LocalFileDocStore = _try("LocalFileDocStore")
RedisDocStore = _try("RedisDocStore")
MongoDocStore = _try("MongoDocStore")

# ── Indexing ──────────────────────────────────────────────────────────────────
index = _try("py_index")
InMemoryRecordManager = _try("InMemoryRecordManager")
SqliteRecordManager = _try("SqliteRecordManager")
CleanupMode = _try("CleanupMode")
IndexStats = _try("IndexStats")
hash_document = _try("py_hash_document")

# ── Extended document loaders ─────────────────────────────────────────────────
WebLoader = _try("WebLoader")
WebLoaderConfig = _try("WebLoaderConfig")
CsvLoader = _try("CsvLoader")
JsonLoader = _try("JsonLoader")
JsonlLoader = _try("JsonlLoader")
DocxLoader = _try("DocxLoader")
DirectoryLoader = _try("DirectoryLoader")
DirectoryLoaderConfig = _try("DirectoryLoaderConfig")

ExcelLoader = _try("ExcelLoader")
EpubLoader = _try("EpubLoader")
SitemapLoader = _try("SitemapLoader")
SitemapConfig = _try("SitemapConfig")
RecursiveUrlLoader = _try("RecursiveUrlLoader")
RecursiveUrlConfig = _try("RecursiveUrlConfig")
WikipediaLoader = _try("WikipediaLoader")
ArxivLoader = _try("ArxivLoader")
RssFeedLoader = _try("RssFeedLoader")
YouTubeLoader = _try("YouTubeLoader")
S3Loader = _try("S3Loader")
DataFrameLoader = _try("DataFrameLoader")
GitLoader = _try("GitLoader")
GitLoaderConfig = _try("GitLoaderConfig")

# ── Utility functions ─────────────────────────────────────────────────────────
bm25_score = _r.py_bm25_score
hybrid_merge = _r.py_hybrid_merge
dedup_by_id = _r.py_dedup_by_id
dedup_by_similarity = _r.py_dedup_by_similarity
decompose_query = _try("py_decompose_query")

# ── Text chunking helpers ─────────────────────────────────────────────────────
chunk_text = _tx.py_chunk_text


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int = 0) -> list:
    chunk_size = max_tokens * 4
    overlap = overlap_tokens * 4
    return _tx.py_chunk_text(text, chunk_size, overlap)

# ── Text splitters ────────────────────────────────────────────────────────────
RecursiveCharacterTextSplitter = _tx.RecursiveCharacterTextSplitter
MarkdownTextSplitter = _tx.MarkdownTextSplitter
HTMLTextSplitter = _tx.HTMLTextSplitter
TokenTextSplitter = _tx.TokenTextSplitter
CodeTextSplitter = _tx.CodeTextSplitter

# ── Misc ─────────────────────────────────────────────────────────────────────
VectorStore = _u.VectorStore
ChunkMetadata = _u.ChunkMetadata
RetrieverStrategy = _u.RetrieverStrategy

__all__ = [
    # Document types
    "Document", "SearchResult", "TextChunk", "ChunkMetadata",
    # Embeddings
    "Embeddings",
    # Retrieval
    "Retriever", "RetrievalConfig", "RetrieverStrategy",
    # Vector stores
    "VectorStore", "InMemoryVectorStore", "HnswVectorStore", "ChromaStore", "Chroma",
    "SingleStoreVectorStore", "AzureAISearchStore", "VectaraStore",
    "Neo4jVectorStore", "TurbopufferStore",
    # Embedding providers
    "CohereEmbeddings", "AzureOpenAIEmbeddings", "GoogleVertexEmbeddings",
    "BedrockEmbeddings", "VoyageEmbeddings", "JinaEmbeddings",
    "TogetherEmbeddings", "NomicEmbeddings",
    # Advanced retrievers (round 1)
    "SelfQueryConfig",
    "ParentDocConfig", "ParentDocument", "ParentDocumentRetriever",
    "MultiVectorConfig", "MultiVectorParent", "MultiVectorRetriever", "VectorView",
    "TimeWeightedConfig", "TimeWeightedRetriever",
    # BM25 / Ensemble / Reorder
    "Bm25Config", "Bm25Document", "Bm25Retriever",
    "EnsembleConfig", "EnsembleRetriever", "VectorRetriever", "Bm25RetrieverAdapter",
    "ReorderStrategy", "reorder", "reorder_for_long_context", "filter_from_json_object",
    # Advanced retrievers (round 2)
    "ContextualCompressionRetriever", "EmbeddingsFilter", "LLMCompressor",
    "DocumentCompressorPipeline", "ScoreThresholdRetriever",
    "MultiQueryConfig", "MultiQueryRetriever",
    # Web retrievers
    "WikipediaRetriever", "ArxivRetriever", "TavilySearchRetriever",
    # SQL database toolkit
    "SqlDatabaseWrapper",
    "ListSQLDatabaseTool", "InfoSQLDatabaseTool",
    "QuerySQLDatabaseTool", "QuerySQLCheckerTool",
    "SqlDatabaseLoader",
    # Document store
    "StoredDocument", "InMemoryDocStore", "LocalFileDocStore",
    "RedisDocStore", "MongoDocStore",
    # Indexing
    "index", "InMemoryRecordManager", "SqliteRecordManager",
    "CleanupMode", "IndexStats", "hash_document",
    # Extended document loaders
    "WebLoader", "WebLoaderConfig",
    "CsvLoader", "JsonLoader", "JsonlLoader", "DocxLoader",
    "DirectoryLoader", "DirectoryLoaderConfig",
    "ExcelLoader", "EpubLoader",
    "SitemapLoader", "SitemapConfig",
    "RecursiveUrlLoader", "RecursiveUrlConfig",
    "WikipediaLoader", "ArxivLoader", "RssFeedLoader",
    "YouTubeLoader", "S3Loader", "DataFrameLoader",
    "GitLoader", "GitLoaderConfig",
    # Text splitters
    "RecursiveCharacterTextSplitter", "MarkdownTextSplitter",
    "HTMLTextSplitter", "TokenTextSplitter", "CodeTextSplitter",
    # RAG configuration
    "VectorStoreType", "RAGConfig", "VectorStoreConfig",
    "EmbeddingsConfig", "RetrievalSettings", "PdfSettings", "RAGGraphConfig",
    # Utility functions
    "bm25_score", "hybrid_merge", "dedup_by_id", "dedup_by_similarity", "decompose_query",
    # Text chunking helpers
    "chunk_text", "chunk_text_by_tokens", "estimate_tokens",
]
