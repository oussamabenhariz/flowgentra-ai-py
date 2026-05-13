//! Python bindings for NoSQL / document store backends.
//!
//! All backends implement the same ``DocumentStore`` interface:
//!
//! ```python
//! from flowgentra_ai import db
//!
//! store = db.MongoDocumentStore("mongodb://localhost:27017", "mydb")
//! id = store.insert("users", {"name": "Alice", "age": 30})
//! docs = store.find("users", {"name": "Alice"})
//! store.delete("users", id)
//! ```

use pyo3::prelude::*;
use pyo3::types::PyList;
use serde_json::Value;
use std::sync::Arc;

use crate::{json_to_py, py_to_json, run_async};

fn pyobj_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    py_to_json(obj)
}

// ─── MongoDocumentStore ───────────────────────────────────────────────────────

/// MongoDB document store.
///
/// Example::
///
///     store = MongoDocumentStore("mongodb://localhost:27017", "mydb")
///     id = store.insert("users", {"name": "Alice", "age": 30})
///     docs = store.find("users", {"name": "Alice"})
///     # [{"_id": "...", "name": "Alice", "age": 30}]
///     store.delete("users", id)
#[pyclass(name = "MongoDocumentStore")]
pub struct PyMongoDocumentStore {
    #[cfg(feature = "mongodb-store")]
    inner: Arc<flowgentra_ai::core::db::document::mongodb::MongoDocumentStore>,
}

#[pymethods]
impl PyMongoDocumentStore {
    /// Connect to MongoDB.
    ///
    /// Args:
    ///     uri:      MongoDB connection URI (e.g. ``"mongodb://localhost:27017"``).
    ///     database: Database name.
    #[new]
    fn new(uri: &str, database: &str) -> PyResult<Self> {
        #[cfg(feature = "mongodb-store")]
        {
            use flowgentra_ai::core::db::document::mongodb::MongoDocumentStore;
            let store = run_async(MongoDocumentStore::connect(uri, database))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            Ok(Self {
                inner: Arc::new(store),
            })
        }
        #[cfg(not(feature = "mongodb-store"))]
        Err(crate::error::ConfigurationError::new_err(
            "mongodb-store feature not enabled",
        ))
    }

    /// Insert a document. Returns the inserted document ID.
    fn insert(&self, collection: &str, doc: &Bound<'_, PyAny>) -> PyResult<String> {
        #[cfg(feature = "mongodb-store")]
        {
            use flowgentra_ai::core::db::document::DocumentStore;
            let val = pyobj_to_json(doc)?;
            run_async(self.inner.insert(collection, val))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
        }
        #[cfg(not(feature = "mongodb-store"))]
        Err(crate::error::ConfigurationError::new_err(
            "mongodb-store feature not enabled",
        ))
    }

    /// Find documents matching filter. Returns a list of dicts.
    fn find(
        &self,
        py: Python<'_>,
        collection: &str,
        filter: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        #[cfg(feature = "mongodb-store")]
        {
            use flowgentra_ai::core::db::document::DocumentStore;
            let filter_val = pyobj_to_json(filter)?;
            let docs = run_async(self.inner.find(collection, filter_val))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            let list = PyList::empty_bound(py);
            for doc in docs {
                list.append(json_to_py(py, &doc)?)?;
            }
            Ok(list.into())
        }
        #[cfg(not(feature = "mongodb-store"))]
        Err(crate::error::ConfigurationError::new_err(
            "mongodb-store feature not enabled",
        ))
    }

    /// Delete a document by ID.
    fn delete(&self, collection: &str, id: &str) -> PyResult<()> {
        #[cfg(feature = "mongodb-store")]
        {
            use flowgentra_ai::core::db::document::DocumentStore;
            run_async(self.inner.delete(collection, id))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
        }
        #[cfg(not(feature = "mongodb-store"))]
        Err(crate::error::ConfigurationError::new_err(
            "mongodb-store feature not enabled",
        ))
    }

    fn __repr__(&self) -> String {
        "MongoDocumentStore(...)".to_string()
    }
}

// ─── RedisDocumentStore ───────────────────────────────────────────────────────

/// Redis document store (JSON docs stored as hash fields).
///
/// Example::
///
///     store = RedisDocumentStore("redis://127.0.0.1/")
///     id = store.insert("users", {"name": "Alice", "age": 30})
///     docs = store.find("users", {"name": "Alice"})
///     store.delete("users", id)
#[pyclass(name = "RedisDocumentStore")]
pub struct PyRedisDocumentStore {
    #[cfg(feature = "redis-store")]
    inner: Arc<flowgentra_ai::core::db::document::redis::RedisDocumentStore>,
}

#[pymethods]
impl PyRedisDocumentStore {
    /// Connect to Redis.
    ///
    /// Args:
    ///     url: Redis URL (e.g. ``"redis://127.0.0.1/"``).
    #[new]
    fn new(url: &str) -> PyResult<Self> {
        #[cfg(feature = "redis-store")]
        {
            use flowgentra_ai::core::db::document::redis::RedisDocumentStore;
            let store = run_async(RedisDocumentStore::connect(url))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            Ok(Self {
                inner: Arc::new(store),
            })
        }
        #[cfg(not(feature = "redis-store"))]
        Err(crate::error::ConfigurationError::new_err(
            "redis-store feature not enabled",
        ))
    }

    fn insert(&self, collection: &str, doc: &Bound<'_, PyAny>) -> PyResult<String> {
        #[cfg(feature = "redis-store")]
        {
            use flowgentra_ai::core::db::document::DocumentStore;
            run_async(self.inner.insert(collection, pyobj_to_json(doc)?))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
        }
        #[cfg(not(feature = "redis-store"))]
        Err(crate::error::ConfigurationError::new_err(
            "redis-store feature not enabled",
        ))
    }

    fn find(
        &self,
        py: Python<'_>,
        collection: &str,
        filter: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        #[cfg(feature = "redis-store")]
        {
            use flowgentra_ai::core::db::document::DocumentStore;
            let docs = run_async(self.inner.find(collection, pyobj_to_json(filter)?))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            let list = PyList::empty_bound(py);
            for doc in docs {
                list.append(json_to_py(py, &doc)?)?;
            }
            Ok(list.into())
        }
        #[cfg(not(feature = "redis-store"))]
        Err(crate::error::ConfigurationError::new_err(
            "redis-store feature not enabled",
        ))
    }

    fn delete(&self, collection: &str, id: &str) -> PyResult<()> {
        #[cfg(feature = "redis-store")]
        {
            use flowgentra_ai::core::db::document::DocumentStore;
            run_async(self.inner.delete(collection, id))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
        }
        #[cfg(not(feature = "redis-store"))]
        Err(crate::error::ConfigurationError::new_err(
            "redis-store feature not enabled",
        ))
    }

    fn __repr__(&self) -> String {
        "RedisDocumentStore(...)".to_string()
    }
}

// ─── Neo4jDocumentStore ───────────────────────────────────────────────────────

/// Neo4j graph database document store.
///
/// Stores JSON documents as labelled graph nodes. The ``collection`` argument
/// maps to the node label. Simple equality filters are applied via Cypher
/// ``WHERE`` clauses.
///
/// Example::
///
///     store = Neo4jDocumentStore(
///         uri="bolt://localhost:7687",
///         user="neo4j",
///         password="password",
///     )
///     id = store.insert("Person", {"name": "Alice", "age": 30})
///     docs = store.find("Person", {"name": "Alice"})
///     store.delete("Person", id)
#[pyclass(name = "Neo4jDocumentStore")]
pub struct PyNeo4jDocumentStore {
    #[cfg(feature = "neo4j-store")]
    inner: Arc<flowgentra_ai::core::db::document::neo4j::Neo4jDocumentStore>,
}

#[pymethods]
impl PyNeo4jDocumentStore {
    /// Connect to Neo4j via the Bolt protocol.
    ///
    /// Args:
    ///     uri:      Bolt URI (e.g. ``"bolt://localhost:7687"`` or ``"bolt+s://host:7687"`` for TLS).
    ///     user:     Neo4j username (default ``"neo4j"``).
    ///     password: Neo4j password.
    #[new]
    #[pyo3(signature = (uri, user = "neo4j", password = "neo4j"))]
    fn new(uri: &str, user: &str, password: &str) -> PyResult<Self> {
        #[cfg(feature = "neo4j-store")]
        {
            use flowgentra_ai::core::db::document::neo4j::Neo4jDocumentStore;
            let store = run_async(Neo4jDocumentStore::connect(uri, user, password))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            Ok(Self {
                inner: Arc::new(store),
            })
        }
        #[cfg(not(feature = "neo4j-store"))]
        Err(crate::error::ConfigurationError::new_err(
            "neo4j-store feature not enabled",
        ))
    }

    fn insert(&self, collection: &str, doc: &Bound<'_, PyAny>) -> PyResult<String> {
        #[cfg(feature = "neo4j-store")]
        {
            use flowgentra_ai::core::db::document::DocumentStore;
            run_async(self.inner.insert(collection, pyobj_to_json(doc)?))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
        }
        #[cfg(not(feature = "neo4j-store"))]
        Err(crate::error::ConfigurationError::new_err(
            "neo4j-store feature not enabled",
        ))
    }

    fn find(
        &self,
        py: Python<'_>,
        collection: &str,
        filter: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        #[cfg(feature = "neo4j-store")]
        {
            use flowgentra_ai::core::db::document::DocumentStore;
            let docs = run_async(self.inner.find(collection, pyobj_to_json(filter)?))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            let list = PyList::empty_bound(py);
            for doc in docs {
                list.append(json_to_py(py, &doc)?)?;
            }
            Ok(list.into())
        }
        #[cfg(not(feature = "neo4j-store"))]
        Err(crate::error::ConfigurationError::new_err(
            "neo4j-store feature not enabled",
        ))
    }

    fn delete(&self, collection: &str, id: &str) -> PyResult<()> {
        #[cfg(feature = "neo4j-store")]
        {
            use flowgentra_ai::core::db::document::DocumentStore;
            run_async(self.inner.delete(collection, id))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
        }
        #[cfg(not(feature = "neo4j-store"))]
        Err(crate::error::ConfigurationError::new_err(
            "neo4j-store feature not enabled",
        ))
    }

    fn __repr__(&self) -> String {
        "Neo4jDocumentStore(...)".to_string()
    }
}

// ─── CassandraDocumentStore ───────────────────────────────────────────────────

/// Cassandra / Astra DB document store via the Stargate Document REST API.
///
/// Example::
///
///     import os
///     store = CassandraDocumentStore(
///         endpoint="https://<id>-<region>.apps.astra.datastax.com",
///         keyspace="default_keyspace",
///         token=os.environ["ASTRA_TOKEN"],
///     )
///     id = store.insert("users", {"name": "Alice"})
///     docs = store.find("users", {"name": {"$eq": "Alice"}})
///     store.delete("users", id)
#[pyclass(name = "CassandraDocumentStore")]
pub struct PyCassandraDocumentStore {
    inner: Arc<flowgentra_ai::core::db::document::cassandra::CassandraDocumentStore>,
}

#[pymethods]
impl PyCassandraDocumentStore {
    /// Connect to Cassandra/Astra DB via Stargate.
    ///
    /// Args:
    ///     endpoint: Stargate REST base URL.
    ///     keyspace: Cassandra keyspace name.
    ///     token:    Stargate/Astra authentication token.
    #[new]
    fn new(endpoint: &str, keyspace: &str, token: &str) -> PyResult<Self> {
        use flowgentra_ai::core::db::document::cassandra::{
            CassandraConfig, CassandraDocumentStore,
        };
        let store = CassandraDocumentStore::new(CassandraConfig {
            endpoint: endpoint.to_string(),
            keyspace: keyspace.to_string(),
            token: token.to_string(),
        });
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    fn insert(&self, collection: &str, doc: &Bound<'_, PyAny>) -> PyResult<String> {
        use flowgentra_ai::core::db::document::DocumentStore;
        run_async(self.inner.insert(collection, pyobj_to_json(doc)?))
            .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
    }

    fn find(
        &self,
        py: Python<'_>,
        collection: &str,
        filter: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        use flowgentra_ai::core::db::document::DocumentStore;
        let docs = run_async(self.inner.find(collection, pyobj_to_json(filter)?))
            .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
        let list = PyList::empty_bound(py);
        for doc in docs {
            list.append(json_to_py(py, &doc)?)?;
        }
        Ok(list.into())
    }

    fn delete(&self, collection: &str, id: &str) -> PyResult<()> {
        use flowgentra_ai::core::db::document::DocumentStore;
        run_async(self.inner.delete(collection, id))
            .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "CassandraDocumentStore(...)".to_string()
    }
}

// ─── ElasticsearchDocumentStore ───────────────────────────────────────────────

/// Elasticsearch document store (full-text + structured queries).
///
/// Example::
///
///     store = ElasticsearchDocumentStore("http://localhost:9200")
///     id = store.insert("my_index", {"title": "Hello", "body": "World"})
///     # Full Elasticsearch DSL:
///     docs = store.find("my_index", {"query": {"match": {"title": "Hello"}}})
///     # Simple equality (shorthand):
///     docs = store.find("my_index", {"title": "Hello"})
///     store.delete("my_index", id)
#[pyclass(name = "ElasticsearchDocumentStore")]
pub struct PyElasticsearchDocumentStore {
    inner: Arc<flowgentra_ai::core::db::document::elasticsearch::ElasticsearchDocumentStore>,
}

#[pymethods]
impl PyElasticsearchDocumentStore {
    /// Connect to Elasticsearch.
    ///
    /// Args:
    ///     endpoint: Elasticsearch base URL (e.g. ``"http://localhost:9200"``).
    ///     api_key:  Optional API key for Elastic Cloud.
    #[new]
    #[pyo3(signature = (endpoint, api_key = None))]
    fn new(endpoint: &str, api_key: Option<String>) -> PyResult<Self> {
        use flowgentra_ai::core::db::document::elasticsearch::{
            ElasticsearchDocConfig, ElasticsearchDocumentStore,
        };
        let store = ElasticsearchDocumentStore::new(ElasticsearchDocConfig {
            endpoint: endpoint.to_string(),
            api_key,
        });
        Ok(Self {
            inner: Arc::new(store),
        })
    }

    fn insert(&self, collection: &str, doc: &Bound<'_, PyAny>) -> PyResult<String> {
        use flowgentra_ai::core::db::document::DocumentStore;
        run_async(self.inner.insert(collection, pyobj_to_json(doc)?))
            .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
    }

    fn find(
        &self,
        py: Python<'_>,
        collection: &str,
        filter: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        use flowgentra_ai::core::db::document::DocumentStore;
        let docs = run_async(self.inner.find(collection, pyobj_to_json(filter)?))
            .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
        let list = PyList::empty_bound(py);
        for doc in docs {
            list.append(json_to_py(py, &doc)?)?;
        }
        Ok(list.into())
    }

    fn delete(&self, collection: &str, id: &str) -> PyResult<()> {
        use flowgentra_ai::core::db::document::DocumentStore;
        run_async(self.inner.delete(collection, id))
            .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "ElasticsearchDocumentStore(...)".to_string()
    }
}
