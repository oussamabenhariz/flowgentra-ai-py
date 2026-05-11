"""Evaluation and metrics for agent workflows and RAG systems.

This module provides evaluation frameworks, metrics, and scoring for assessing
agent performance, RAG retrieval quality, and generation quality.

Examples:
    Evaluate RAG retrieval:

        from flowgentra_ai.evaluation import hit_rate, mrr, mean_ndcg

        score = hit_rate(retrieval_results, ground_truth)
        
    Create and run evaluations:
    
        from flowgentra_ai.evaluation import (
            EvaluationConfig,
            GradingConfig,
            rag_evaluate
        )
        
        config = EvaluationConfig(...)
        results = rag_evaluate(queries, results, ground_truth, config)
"""

from flowgentra_ai._native import evaluation as _e, nodes as _n

EvaluationResult = _e.EvaluationResult
EvalQuery = _e.EvalQuery
EvalResults = _e.EvalResults
QueryResult = _e.QueryResult
ScoringConfig = _e.ScoringConfig
GradingConfig = _e.GradingConfig
EvaluationConfig = _e.EvaluationConfig
NodeResult = _e.NodeResult
EvaluationReport = _e.EvaluationReport
hit_rate = _e.py_hit_rate
mrr = _e.py_mrr
mean_ndcg = _e.py_mean_ndcg
rag_evaluate = _e.py_rag_evaluate
EvaluationNodeConfig = _n.EvaluationNodeConfig

__all__ = [
    # Core evaluation types
    "EvaluationResult",
    "EvaluationNodeConfig",
    # RAG evaluation types
    "EvalQuery",
    "EvalResults",
    "QueryResult",
    # Metrics functions
    "hit_rate",
    "mrr",
    "mean_ndcg",
    "rag_evaluate",
    # Configuration and reporting
    "ScoringConfig",
    "GradingConfig",
    "EvaluationConfig",
    "NodeResult",
    "EvaluationReport",
]
