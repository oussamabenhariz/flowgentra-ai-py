//! Python bindings for RAG evaluation metrics

use pyo3::prelude::*;

use flowgentra_ai::core::rag::{
    evaluate, hit_rate, mean_ndcg, mrr, EvalQuery, EvalResults, QueryResult,
};

// ─── PyEvalQuery ───────────────────────────────────────────────────────────

/// A query with known relevant document IDs for evaluation.
///
/// Example:
///     q = EvalQuery("What is Rust?", ["doc1", "doc3"])
#[pyclass(name = "EvalQuery")]
#[derive(Clone)]
pub struct PyEvalQuery {
    inner: EvalQuery,
}

#[pymethods]
impl PyEvalQuery {
    #[new]
    fn new(query: String, relevant_doc_ids: Vec<String>) -> Self {
        PyEvalQuery {
            inner: EvalQuery {
                query,
                relevant_doc_ids,
            },
        }
    }

    #[getter]
    fn query(&self) -> String {
        self.inner.query.clone()
    }

    #[getter]
    fn relevant_doc_ids(&self) -> Vec<String> {
        self.inner.relevant_doc_ids.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "EvalQuery(query='{}', relevant={})",
            self.inner.query,
            self.inner.relevant_doc_ids.len()
        )
    }
}

// ─── PyQueryResult ─────────────────────────────────────────────────────────

/// Per-query evaluation result.
#[pyclass(name = "QueryResult")]
pub struct PyQueryResult {
    inner: QueryResult,
}

#[pymethods]
impl PyQueryResult {
    #[getter]
    fn query(&self) -> String {
        self.inner.query.clone()
    }

    #[getter]
    fn hit(&self) -> bool {
        self.inner.hit
    }

    #[getter]
    fn reciprocal_rank(&self) -> f64 {
        self.inner.reciprocal_rank
    }

    #[getter]
    fn ndcg(&self) -> f64 {
        self.inner.ndcg
    }

    #[getter]
    fn retrieved_ids(&self) -> Vec<String> {
        self.inner.retrieved_ids.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "QueryResult(query='{}', hit={}, rr={:.4}, ndcg={:.4})",
            self.inner.query, self.inner.hit, self.inner.reciprocal_rank, self.inner.ndcg
        )
    }
}

// ─── PyEvalResults ─────────────────────────────────────────────────────────

/// Aggregate evaluation results across all queries.
///
/// Example:
///     results = rag_evaluate([q1, q2], [["doc1", "doc2"], ["doc3"]])
///     print(results.hit_rate, results.mrr, results.ndcg)
#[pyclass(name = "EvalResults")]
pub struct PyEvalResults {
    inner: EvalResults,
}

#[pymethods]
impl PyEvalResults {
    #[getter]
    fn hit_rate(&self) -> f64 {
        self.inner.hit_rate
    }

    #[getter]
    fn mrr(&self) -> f64 {
        self.inner.mrr
    }

    #[getter]
    fn ndcg(&self) -> f64 {
        self.inner.ndcg
    }

    #[getter]
    fn num_queries(&self) -> usize {
        self.inner.num_queries
    }

    #[getter]
    fn per_query(&self) -> Vec<PyQueryResult> {
        self.inner
            .per_query
            .iter()
            .map(|q| PyQueryResult { inner: q.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "EvalResults(hit_rate={:.4}, mrr={:.4}, ndcg={:.4}, queries={})",
            self.inner.hit_rate, self.inner.mrr, self.inner.ndcg, self.inner.num_queries
        )
    }
}

// ─── Free functions ────────────────────────────────────────────────────────

/// Compute hit rate metric.
#[pyfunction]
pub fn py_hit_rate(queries: Vec<PyEvalQuery>, retrieved: Vec<Vec<String>>) -> f64 {
    let qs: Vec<EvalQuery> = queries.into_iter().map(|q| q.inner).collect();
    let refs: Vec<Vec<String>> = retrieved;
    hit_rate(&qs, &refs)
}

/// Compute mean reciprocal rank (MRR).
#[pyfunction]
pub fn py_mrr(queries: Vec<PyEvalQuery>, retrieved: Vec<Vec<String>>) -> f64 {
    let qs: Vec<EvalQuery> = queries.into_iter().map(|q| q.inner).collect();
    mrr(&qs, &retrieved)
}

/// Compute mean NDCG.
#[pyfunction]
pub fn py_mean_ndcg(queries: Vec<PyEvalQuery>, retrieved: Vec<Vec<String>>) -> f64 {
    let qs: Vec<EvalQuery> = queries.into_iter().map(|q| q.inner).collect();
    mean_ndcg(&qs, &retrieved)
}

/// Run full evaluation (hit_rate, mrr, ndcg) with per-query breakdown.
#[pyfunction]
pub fn py_rag_evaluate(queries: Vec<PyEvalQuery>, retrieved: Vec<Vec<String>>) -> PyEvalResults {
    let qs: Vec<EvalQuery> = queries.into_iter().map(|q| q.inner).collect();
    let results = evaluate(&qs, &retrieved);
    PyEvalResults { inner: results }
}
