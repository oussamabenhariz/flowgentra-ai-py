"""SQL and NoSQL database backends.

This module provides database clients for querying SQL and document stores
from Python agent handlers.

All SQL backends expose the same interface::

    db.query(sql, params)    -> list[dict]
    db.execute(sql, params)  -> int  (rows affected)

All document store backends expose::

    store.insert(collection, doc)   -> str  (id)
    store.find(collection, filter)  -> list[dict]
    store.delete(collection, id)    -> None

SQL Examples::

    from flowgentra_ai.db import SqliteDatabase, PostgresDatabase

    # SQLite (in-memory or file)
    db = SqliteDatabase(":memory:")
    db.execute("CREATE TABLE users (id INTEGER, name TEXT)", [])
    db.execute("INSERT INTO users VALUES (1, 'Alice')", [])
    rows = db.query("SELECT * FROM users WHERE id = ?", [1])
    # [{"id": 1, "name": "Alice"}]

    # PostgreSQL
    db = PostgresDatabase("postgres://user:pass@localhost/mydb")
    rows = db.query("SELECT * FROM users WHERE id = $1", [42])

    # MySQL
    db = MySqlDatabase("mysql://user:pass@localhost/mydb")

    # BigQuery (REST API — no driver needed)
    import os
    db = BigQueryDatabase(
        project_id="my-project",
        dataset_id="my_dataset",
        access_token=os.environ["BIGQUERY_TOKEN"],
    )

    # Databricks
    db = DatabricksDatabase(
        host="https://<workspace>.azuredatabricks.net",
        warehouse_id="abc123",
        token=os.environ["DATABRICKS_TOKEN"],
    )

Document Store Examples::

    from flowgentra_ai.db import MongoDocumentStore, RedisDocumentStore

    store = MongoDocumentStore("mongodb://localhost:27017", "mydb")
    doc_id = store.insert("users", {"name": "Alice", "age": 30})
    docs = store.find("users", {"name": "Alice"})
    store.delete("users", doc_id)

    store = RedisDocumentStore("redis://localhost:6379", "my_prefix")
    store = Neo4jDocumentStore("bolt://localhost:7687", "neo4j", "password")
    store = CassandraDocumentStore("https://...astra.datastax.com", "keyspace", token="...")
    store = ElasticsearchDocumentStore("https://localhost:9200", api_key="...")
"""

from flowgentra_ai._native import db as _db


def _try_import(attr):
    """Return the attribute if the native extension exposes it, else None.

    Backend classes are only present when the wheel was compiled with the
    matching Cargo feature (e.g. ``mssql``, ``mongodb-store``).  Callers
    that try to instantiate a ``None`` value will get a clear ``TypeError``
    rather than a cryptic ``AttributeError`` at import time.
    """
    try:
        return getattr(_db, attr)
    except AttributeError:
        return None


# ── SQL backends (feature-gated) ──────────────────────────────────────────────
SqliteDatabase = _try_import("SqliteDatabase")
PostgresDatabase = _try_import("PostgresDatabase")
MySqlDatabase = _try_import("MySqlDatabase")
MssqlDatabase = _try_import("MssqlDatabase")
BigQueryDatabase = _try_import("BigQueryDatabase")
DatabricksDatabase = _try_import("DatabricksDatabase")

# ── Document store backends (feature-gated) ───────────────────────────────────
MongoDocumentStore = _try_import("MongoDocumentStore")
RedisDocumentStore = _try_import("RedisDocumentStore")
Neo4jDocumentStore = _try_import("Neo4jDocumentStore")
CassandraDocumentStore = _try_import("CassandraDocumentStore")
ElasticsearchDocumentStore = _try_import("ElasticsearchDocumentStore")

__all__ = [
    # SQL
    "SqliteDatabase",
    "PostgresDatabase",
    "MySqlDatabase",
    "MssqlDatabase",
    "BigQueryDatabase",
    "DatabricksDatabase",
    # Document stores
    "MongoDocumentStore",
    "RedisDocumentStore",
    "Neo4jDocumentStore",
    "CassandraDocumentStore",
    "ElasticsearchDocumentStore",
]
