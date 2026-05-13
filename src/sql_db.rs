//! Python bindings for SQL database backends.
//!
//! All backends implement the same ``SqlDatabase`` interface:
//!
//! ```python
//! from flowgentra_ai import db
//!
//! # SQLite (in-memory)
//! db = db.SqliteDatabase(":memory:")
//!
//! # PostgreSQL
//! db = db.PostgresDatabase("postgres://user:pass@localhost/mydb")
//!
//! # MySQL
//! db = db.MySqlDatabase("mysql://user:pass@localhost/mydb")
//!
//! # BigQuery
//! db = db.BigQueryDatabase("my-project", "my_dataset", access_token="...")
//!
//! # Databricks
//! db = db.DatabricksDatabase("https://...azuredatabricks.net", "warehouse-id", token="...")
//!
//! rows = db.query("SELECT * FROM users WHERE age > ?", [18])
//! db.execute("INSERT INTO users (name) VALUES (?)", ["Alice"])
//! ```

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;
use std::sync::Arc;

use crate::{json_to_py, py_to_json, run_async};

// ─── Row → Python dict ────────────────────────────────────────────────────────

fn row_to_pydict(
    py: Python<'_>,
    row: std::collections::HashMap<String, Value>,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    for (k, v) in row {
        dict.set_item(k, json_to_py(py, &v)?)?;
    }
    Ok(dict.into())
}

fn pylist_to_params(params: Option<&Bound<'_, PyList>>) -> PyResult<Vec<Value>> {
    match params {
        None => Ok(vec![]),
        Some(list) => list.iter().map(|item| py_to_json(&item)).collect(),
    }
}

// ─── SqliteDatabase ───────────────────────────────────────────────────────────

/// SQLite database backend.
///
/// Example::
///
///     db = SqliteDatabase(":memory:")
///     db.execute("CREATE TABLE t (id INTEGER, name TEXT)", [])
///     db.execute("INSERT INTO t VALUES (1, 'Alice')", [])
///     rows = db.query("SELECT * FROM t", [])
///     # [{"id": 1, "name": "Alice"}]
#[pyclass(name = "SqliteDatabase")]
pub struct PySqliteDatabase {
    #[cfg(feature = "sqlite")]
    inner: Arc<flowgentra_ai::core::db::sql::sqlite::SqliteDatabase>,
}

#[pymethods]
impl PySqliteDatabase {
    /// Open a SQLite database.
    ///
    /// Args:
    ///     url: SQLite URL — ``":memory:"`` for in-memory, or a file path like
    ///          ``"sqlite:///path/to/db.sqlite3"``.
    #[new]
    fn new(url: &str) -> PyResult<Self> {
        #[cfg(feature = "sqlite")]
        {
            use flowgentra_ai::core::db::sql::sqlite::SqliteDatabase;
            let db = run_async(SqliteDatabase::connect(url))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            Ok(Self {
                inner: Arc::new(db),
            })
        }
        #[cfg(not(feature = "sqlite"))]
        Err(crate::error::InternalError::new_err(
            "SqliteDatabase requires the 'sqlite' feature. Recompile with --features sqlite",
        ))
    }

    /// Run a SELECT query and return rows as a list of dicts.
    #[pyo3(signature = (sql, params = None))]
    fn query(
        &self,
        py: Python<'_>,
        sql: &str,
        params: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyObject> {
        #[cfg(feature = "sqlite")]
        {
            use flowgentra_ai::core::db::sql::SqlDatabase;
            let p = pylist_to_params(params)?;
            let rows = run_async(self.inner.query(sql, &p))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            let list = PyList::empty_bound(py);
            for row in rows {
                list.append(row_to_pydict(py, row)?)?;
            }
            Ok(list.into())
        }
        #[cfg(not(feature = "sqlite"))]
        Err(crate::error::InternalError::new_err(
            "sqlite feature not enabled",
        ))
    }

    /// Run an INSERT / UPDATE / DELETE / DDL statement. Returns rows affected.
    #[pyo3(signature = (sql, params = None))]
    fn execute(&self, sql: &str, params: Option<&Bound<'_, PyList>>) -> PyResult<u64> {
        #[cfg(feature = "sqlite")]
        {
            use flowgentra_ai::core::db::sql::SqlDatabase;
            let p = pylist_to_params(params)?;
            run_async(self.inner.execute(sql, &p))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
        }
        #[cfg(not(feature = "sqlite"))]
        Err(crate::error::InternalError::new_err(
            "sqlite feature not enabled",
        ))
    }

    fn __repr__(&self) -> String {
        "SqliteDatabase(...)".to_string()
    }
}

// ─── PostgresDatabase ─────────────────────────────────────────────────────────

/// PostgreSQL database backend.
///
/// Example::
///
///     db = PostgresDatabase("postgres://user:pass@localhost/mydb")
///     rows = db.query("SELECT * FROM users WHERE id = $1", [42])
#[pyclass(name = "PostgresDatabase")]
pub struct PyPostgresDatabase {
    #[cfg(feature = "postgres")]
    inner: Arc<flowgentra_ai::core::db::sql::postgres::PostgresDatabase>,
}

#[pymethods]
impl PyPostgresDatabase {
    #[new]
    fn new(url: &str) -> PyResult<Self> {
        #[cfg(feature = "postgres")]
        {
            use flowgentra_ai::core::db::sql::postgres::PostgresDatabase;
            let db = run_async(PostgresDatabase::connect(url))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            Ok(Self {
                inner: Arc::new(db),
            })
        }
        #[cfg(not(feature = "postgres"))]
        Err(crate::error::InternalError::new_err(
            "postgres feature not enabled",
        ))
    }

    #[pyo3(signature = (sql, params = None))]
    fn query(
        &self,
        py: Python<'_>,
        sql: &str,
        params: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyObject> {
        #[cfg(feature = "postgres")]
        {
            use flowgentra_ai::core::db::sql::SqlDatabase;
            let p = pylist_to_params(params)?;
            let rows = run_async(self.inner.query(sql, &p))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            let list = PyList::empty_bound(py);
            for row in rows {
                list.append(row_to_pydict(py, row)?)?;
            }
            Ok(list.into())
        }
        #[cfg(not(feature = "postgres"))]
        Err(crate::error::InternalError::new_err(
            "postgres feature not enabled",
        ))
    }

    #[pyo3(signature = (sql, params = None))]
    fn execute(&self, sql: &str, params: Option<&Bound<'_, PyList>>) -> PyResult<u64> {
        #[cfg(feature = "postgres")]
        {
            use flowgentra_ai::core::db::sql::SqlDatabase;
            let p = pylist_to_params(params)?;
            run_async(self.inner.execute(sql, &p))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
        }
        #[cfg(not(feature = "postgres"))]
        Err(crate::error::InternalError::new_err(
            "postgres feature not enabled",
        ))
    }

    fn __repr__(&self) -> String {
        "PostgresDatabase(...)".to_string()
    }
}

// ─── MySqlDatabase ────────────────────────────────────────────────────────────

/// MySQL database backend.
///
/// Example::
///
///     db = MySqlDatabase("mysql://user:pass@localhost/mydb")
///     rows = db.query("SELECT * FROM users WHERE age > ?", [18])
#[pyclass(name = "MySqlDatabase")]
pub struct PyMySqlDatabase {
    #[cfg(feature = "mysql")]
    inner: Arc<flowgentra_ai::core::db::sql::mysql::MySqlDatabase>,
}

#[pymethods]
impl PyMySqlDatabase {
    #[new]
    fn new(url: &str) -> PyResult<Self> {
        #[cfg(feature = "mysql")]
        {
            use flowgentra_ai::core::db::sql::mysql::MySqlDatabase;
            let db = run_async(MySqlDatabase::connect(url))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            Ok(Self {
                inner: Arc::new(db),
            })
        }
        #[cfg(not(feature = "mysql"))]
        Err(crate::error::InternalError::new_err(
            "mysql feature not enabled",
        ))
    }

    #[pyo3(signature = (sql, params = None))]
    fn query(
        &self,
        py: Python<'_>,
        sql: &str,
        params: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyObject> {
        #[cfg(feature = "mysql")]
        {
            use flowgentra_ai::core::db::sql::SqlDatabase;
            let p = pylist_to_params(params)?;
            let rows = run_async(self.inner.query(sql, &p))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            let list = PyList::empty_bound(py);
            for row in rows {
                list.append(row_to_pydict(py, row)?)?;
            }
            Ok(list.into())
        }
        #[cfg(not(feature = "mysql"))]
        Err(crate::error::InternalError::new_err(
            "mysql feature not enabled",
        ))
    }

    #[pyo3(signature = (sql, params = None))]
    fn execute(&self, sql: &str, params: Option<&Bound<'_, PyList>>) -> PyResult<u64> {
        #[cfg(feature = "mysql")]
        {
            use flowgentra_ai::core::db::sql::SqlDatabase;
            let p = pylist_to_params(params)?;
            run_async(self.inner.execute(sql, &p))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
        }
        #[cfg(not(feature = "mysql"))]
        Err(crate::error::InternalError::new_err(
            "mysql feature not enabled",
        ))
    }

    fn __repr__(&self) -> String {
        "MySqlDatabase(...)".to_string()
    }
}

// ─── MssqlDatabase ────────────────────────────────────────────────────────────

/// Microsoft SQL Server database backend.
///
/// Uses ``@p1``, ``@p2`` bind placeholders in your SQL.
///
/// Example::
///
///     db = MssqlDatabase("mssql://user:pass@localhost/mydb")
///     db.execute("INSERT INTO users (name) VALUES (@p1)", ["Alice"])
///     rows = db.query("SELECT * FROM users WHERE name = @p1", ["Alice"])
#[pyclass(name = "MssqlDatabase")]
pub struct PyMssqlDatabase {
    #[cfg(feature = "mssql")]
    inner: Arc<flowgentra_ai::core::db::sql::mssql::MssqlDatabase>,
}

#[pymethods]
impl PyMssqlDatabase {
    /// Connect to SQL Server.
    ///
    /// Args:
    ///     url: Connection URL — ``mssql://user:password@host/database``.
    ///          For Windows auth: ``mssql://host/database?trusted_connection=true``.
    #[new]
    fn new(url: &str) -> PyResult<Self> {
        #[cfg(feature = "mssql")]
        {
            use flowgentra_ai::core::db::sql::mssql::MssqlDatabase;
            let db = run_async(MssqlDatabase::connect(url))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            Ok(Self {
                inner: Arc::new(db),
            })
        }
        #[cfg(not(feature = "mssql"))]
        Err(crate::error::InternalError::new_err(
            "mssql feature not enabled",
        ))
    }

    #[pyo3(signature = (sql, params = None))]
    fn query(
        &self,
        py: Python<'_>,
        sql: &str,
        params: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyObject> {
        #[cfg(feature = "mssql")]
        {
            use flowgentra_ai::core::db::sql::SqlDatabase;
            let p = pylist_to_params(params)?;
            let rows = run_async(self.inner.query(sql, &p))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
            let list = PyList::empty_bound(py);
            for row in rows {
                list.append(row_to_pydict(py, row)?)?;
            }
            Ok(list.into())
        }
        #[cfg(not(feature = "mssql"))]
        Err(crate::error::InternalError::new_err(
            "mssql feature not enabled",
        ))
    }

    #[pyo3(signature = (sql, params = None))]
    fn execute(&self, sql: &str, params: Option<&Bound<'_, PyList>>) -> PyResult<u64> {
        #[cfg(feature = "mssql")]
        {
            use flowgentra_ai::core::db::sql::SqlDatabase;
            let p = pylist_to_params(params)?;
            run_async(self.inner.execute(sql, &p))
                .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
        }
        #[cfg(not(feature = "mssql"))]
        Err(crate::error::InternalError::new_err(
            "mssql feature not enabled",
        ))
    }

    fn __repr__(&self) -> String {
        "MssqlDatabase(...)".to_string()
    }
}

// ─── BigQueryDatabase ─────────────────────────────────────────────────────────

/// Google BigQuery SQL backend (REST API — no extra driver needed).
///
/// Example::
///
///     import os
///     db = BigQueryDatabase(
///         project_id="my-project",
///         dataset_id="my_dataset",
///         access_token=os.environ["BIGQUERY_TOKEN"],
///     )
///     rows = db.query("SELECT name FROM `my-project.my_dataset.users` LIMIT 10", [])
#[pyclass(name = "BigQueryDatabase")]
pub struct PyBigQueryDatabase {
    inner: Arc<flowgentra_ai::core::db::sql::bigquery::BigQueryDatabase>,
}

#[pymethods]
impl PyBigQueryDatabase {
    /// Create a BigQuery client.
    ///
    /// Args:
    ///     project_id:    GCP project ID.
    ///     dataset_id:    Default dataset ID.
    ///     access_token:  OAuth2 Bearer token (``gcloud auth print-access-token``).
    #[new]
    fn new(project_id: &str, dataset_id: &str, access_token: &str) -> PyResult<Self> {
        use flowgentra_ai::core::db::sql::bigquery::{BigQueryConfig, BigQueryDatabase};
        let db = BigQueryDatabase::new(BigQueryConfig {
            project_id: project_id.to_string(),
            dataset_id: dataset_id.to_string(),
            access_token: access_token.to_string(),
        });
        Ok(Self {
            inner: Arc::new(db),
        })
    }

    #[pyo3(signature = (sql, params = None))]
    fn query(
        &self,
        py: Python<'_>,
        sql: &str,
        params: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyObject> {
        use flowgentra_ai::core::db::sql::SqlDatabase;
        let p = pylist_to_params(params)?;
        let rows = run_async(self.inner.query(sql, &p))
            .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
        let list = PyList::empty_bound(py);
        for row in rows {
            list.append(row_to_pydict(py, row)?)?;
        }
        Ok(list.into())
    }

    #[pyo3(signature = (sql, params = None))]
    fn execute(&self, sql: &str, params: Option<&Bound<'_, PyList>>) -> PyResult<u64> {
        use flowgentra_ai::core::db::sql::SqlDatabase;
        let p = pylist_to_params(params)?;
        run_async(self.inner.execute(sql, &p))
            .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "BigQueryDatabase(...)".to_string()
    }
}

// ─── DatabricksDatabase ───────────────────────────────────────────────────────

/// Databricks SQL backend using the Statement Execution REST API.
///
/// Example::
///
///     import os
///     db = DatabricksDatabase(
///         host="https://<workspace>.azuredatabricks.net",
///         warehouse_id="abc123",
///         token=os.environ["DATABRICKS_TOKEN"],
///         catalog="main",
///         schema="default",
///     )
///     rows = db.query("SELECT * FROM my_table LIMIT 10", [])
#[pyclass(name = "DatabricksDatabase")]
pub struct PyDatabricksDatabase {
    inner: Arc<flowgentra_ai::core::db::sql::databricks::DatabricksDatabase>,
}

#[pymethods]
impl PyDatabricksDatabase {
    /// Create a Databricks SQL client.
    ///
    /// Args:
    ///     host:         Databricks workspace URL.
    ///     warehouse_id: SQL Warehouse ID.
    ///     token:        Personal access token or service principal token.
    ///     catalog:      Optional Unity Catalog name.
    ///     schema:       Optional schema name.
    #[new]
    #[pyo3(signature = (host, warehouse_id, token, catalog = None, schema = None))]
    fn new(
        host: &str,
        warehouse_id: &str,
        token: &str,
        catalog: Option<String>,
        schema: Option<String>,
    ) -> PyResult<Self> {
        use flowgentra_ai::core::db::sql::databricks::{DatabricksConfig, DatabricksDatabase};
        let db = DatabricksDatabase::new(DatabricksConfig {
            host: host.to_string(),
            warehouse_id: warehouse_id.to_string(),
            token: token.to_string(),
            catalog,
            schema,
        });
        Ok(Self {
            inner: Arc::new(db),
        })
    }

    #[pyo3(signature = (sql, params = None))]
    fn query(
        &self,
        py: Python<'_>,
        sql: &str,
        params: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyObject> {
        use flowgentra_ai::core::db::sql::SqlDatabase;
        let p = pylist_to_params(params)?;
        let rows = run_async(self.inner.query(sql, &p))
            .map_err(|e| crate::error::InternalError::new_err(e.to_string()))?;
        let list = PyList::empty_bound(py);
        for row in rows {
            list.append(row_to_pydict(py, row)?)?;
        }
        Ok(list.into())
    }

    #[pyo3(signature = (sql, params = None))]
    fn execute(&self, sql: &str, params: Option<&Bound<'_, PyList>>) -> PyResult<u64> {
        use flowgentra_ai::core::db::sql::SqlDatabase;
        let p = pylist_to_params(params)?;
        run_async(self.inner.execute(sql, &p))
            .map_err(|e| crate::error::InternalError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "DatabricksDatabase(...)".to_string()
    }
}
