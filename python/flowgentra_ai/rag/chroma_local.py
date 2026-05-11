"""Pure-Python embedded ChromaDB store.

Drop-in replacement for ``ChromaStore`` (Rust/HTTP) that uses
``chromadb.PersistentClient`` — no server required, data persists to disk.

Usage::

    from flowgentra_ai.rag import Chroma, Document, Embeddings

    emb   = Embeddings.mistral(api_key)
    store = Chroma("my_collection", persist_directory="./chroma_data")

    # index one doc
    store.index(doc, emb.embed(doc.text))

    # index many docs (batched automatically)
    store.index_many(docs, [emb.embed(d.text) for d in docs])

    # search by vector
    results = store.search(query_vec, top_k=5)

    # auto-embed query (requires embeddings= at construction)
    store2 = Chroma("col", persist_directory="./data", embeddings=emb)
    results = store2.retrieve("what is Rust ownership?")

Limitation
----------
``search()`` / ``retrieve()`` return Python ``SearchResult`` objects, NOT
Rust ``PySearchResult``.  They cannot be passed to ``EnsembleRetriever``
(which requires Rust types).  Use ``ChromaStore`` (HTTP) or
``InMemoryVectorStore`` when ``EnsembleRetriever`` compatibility is needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

DistanceMetric = Literal["cosine", "l2", "ip"]


def _require_chroma() -> None:
    if not _CHROMA_AVAILABLE:
        raise ImportError(
            "chromadb is required. Install it: pip install chromadb"
        )


def _to_score(distance: float, metric: DistanceMetric) -> float:
    """Convert ChromaDB distance to similarity score 0–1."""
    if metric == "cosine":
        return 1.0 - distance          # cosine distance in [0, 2] → clamp to [0, 1]
    if metric == "l2":
        return 1.0 / (1.0 + distance)  # L2 in [0, ∞) → map to (0, 1]
    # ip (inner product): higher = more similar, but range is unbounded
    return float(distance)


class SearchResult:
    """Search result returned by ``Chroma.search()`` and ``Chroma.retrieve()``."""

    __slots__ = ("id", "text", "score", "metadata")

    def __init__(self, id: str, text: str, score: float, metadata: Dict[str, Any]) -> None:
        self.id = id
        self.text = text
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"SearchResult(id='{self.id}', score={self.score:.4f})"


class Chroma:
    """Embedded ChromaDB vector store — no server required.

    Args:
        collection_name:   Name of the ChromaDB collection.
        persist_directory: Path to persist data on disk.
                           Pass ``None`` for ephemeral in-memory store.
        embeddings:        Optional ``Embeddings`` instance.  When provided,
                           ``retrieve(query_text)`` embeds the query
                           automatically.
        distance_metric:   Distance function: ``"cosine"`` (default),
                           ``"l2"``, or ``"ip"`` (inner product).
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[str] = "./chroma_data",
        embeddings=None,
        distance_metric: DistanceMetric = "cosine",
    ) -> None:
        _require_chroma()
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._metric: DistanceMetric = distance_metric
        self._persist_directory = persist_directory

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        self._col = self._client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": distance_metric},
        )

    # ── store API (mirrors ChromaStore) ────────────────────────────────────────

    def index(self, doc, embedding: List[float]) -> None:
        """Index a single document with its embedding vector."""
        self._col.upsert(
            ids=[doc.id],
            documents=[doc.text],
            embeddings=[embedding],
            metadatas=[dict(doc.metadata) if doc.metadata else {}],
        )

    def index_many(self, docs: list, embeddings: List[List[float]]) -> None:
        """Index multiple documents in optimally-sized batches.

        Uses ``client.max_batch_size`` (SQLite-dependent, ~5461) to avoid
        batch-size errors on large corpora.
        """
        ids       = [d.id for d in docs]
        texts     = [d.text for d in docs]
        metadatas = [dict(d.metadata) if d.metadata else {} for d in docs]

        try:
            from chromadb.utils.batch_utils import create_batches  # type: ignore[import]
            batches = create_batches(
                api=self._client,
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            for batch in batches:
                # batch = (ids, embeddings, metadatas, documents)
                self._col.upsert(
                    ids=batch[0],
                    embeddings=batch[1],
                    metadatas=batch[2],
                    documents=batch[3],
                )
        except ImportError:
            # fallback: manual chunking using max_batch_size
            max_b = getattr(self._client, "max_batch_size", 5000)
            for start in range(0, len(ids), max_b):
                s, e = start, start + max_b
                self._col.upsert(
                    ids=ids[s:e],
                    embeddings=embeddings[s:e],
                    metadatas=metadatas[s:e],
                    documents=texts[s:e],
                )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Search by embedding vector.

        Args:
            query_embedding: Query vector.
            top_k:           Max results (clamped to collection size).
            filter:          ChromaDB ``where`` clause dict.
                             Supports operators: ``$eq``, ``$ne``, ``$gt``,
                             ``$gte``, ``$lt``, ``$lte``, ``$in``, ``$nin``,
                             ``$and``, ``$or``.
                             Example: ``{"topic": "rust"}``
                             Example: ``{"$and": [{"topic": {"$eq": "rust"}}, {"score": {"$gt": 0.5}}]}``
            score_threshold: Drop results below this similarity score (0–1).
        """
        n = self._col.count()
        if n == 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, n),
            "include": ["documents", "distances", "metadatas"],
        }
        if filter:
            kwargs["where"] = filter

        raw = self._col.query(**kwargs)

        results = []
        for doc_id, text, dist, meta in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["distances"][0],
            raw["metadatas"][0],
        ):
            score = _to_score(dist, self._metric)
            if score < score_threshold:
                continue
            results.append(SearchResult(
                id=doc_id,
                text=text,
                score=score,
                metadata=meta or {},
            ))
        return results

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""
        self._col.delete(ids=[doc_id])

    def update(self, doc, embedding: Optional[List[float]] = None) -> None:
        """Update document text/metadata and optionally its embedding."""
        kwargs: Dict[str, Any] = {
            "ids": [doc.id],
            "documents": [doc.text],
            "metadatas": [dict(doc.metadata) if doc.metadata else {}],
        }
        if embedding is not None:
            kwargs["embeddings"] = [embedding]
        self._col.update(**kwargs)

    def get(self, doc_id: str):
        """Fetch single document by ID.  Returns a ``Document`` instance."""
        from flowgentra_ai.rag import Document
        raw = self._col.get(ids=[doc_id], include=["documents", "metadatas"])
        if not raw["ids"]:
            raise KeyError(f"Document not found: {doc_id}")
        return Document(
            id=raw["ids"][0],
            text=raw["documents"][0],
            metadata=raw["metadatas"][0] or {},
        )

    def list(self) -> list:
        """Return all documents in the collection as ``Document`` instances."""
        from flowgentra_ai.rag import Document
        raw = self._col.get(include=["documents", "metadatas"])
        return [
            Document(id=doc_id, text=text, metadata=meta or {})
            for doc_id, text, meta in zip(
                raw["ids"], raw["documents"], raw["metadatas"]
            )
        ]

    def count(self) -> int:
        """Return number of documents in the collection."""
        return self._col.count()

    def clear(self) -> None:
        """Delete all documents (recreates collection, preserving distance metric)."""
        self._client.delete_collection(self._collection_name)
        self._col = self._client.get_or_create_collection(
            self._collection_name,
            metadata={"hnsw:space": self._metric},
        )

    # ── retriever API ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Embed ``query`` then search.  Requires ``embeddings=`` at init.

        Note: returns Python ``SearchResult``, NOT Rust ``PySearchResult``.
        Not compatible with ``EnsembleRetriever``.
        """
        if self._embeddings is None:
            raise ValueError(
                "Pass embeddings=<Embeddings> to Chroma(...) to use retrieve()."
            )
        vec = self._embeddings.embed(query)
        return self.search(vec, top_k=top_k, score_threshold=score_threshold)

    # ── convenience constructors ───────────────────────────────────────────────

    @classmethod
    def from_documents(
        cls,
        docs: list,
        embeddings,
        collection_name: str = "default",
        persist_directory: Optional[str] = "./chroma_data",
        distance_metric: DistanceMetric = "cosine",
    ) -> "Chroma":
        """Build a ``Chroma`` store from a list of documents in one call.

        Embeddings are computed and indexed automatically.

        Example::

            store = Chroma.from_documents(
                docs=DOCUMENTS,
                embeddings=Embeddings.mistral(api_key),
                collection_name="my_col",
            )
        """
        store = cls(
            collection_name,
            persist_directory=persist_directory,
            embeddings=embeddings,
            distance_metric=distance_metric,
        )
        vecs = [embeddings.embed(d.text) for d in docs]
        store.index_many(docs, vecs)
        return store

    def __repr__(self) -> str:
        loc = f"'{self._persist_directory}'" if self._persist_directory else "ephemeral"
        return (
            f"Chroma(collection='{self._collection_name}', "
            f"metric='{self._metric}', persist={loc}, docs={self._col.count()})"
        )
