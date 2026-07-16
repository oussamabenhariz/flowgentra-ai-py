"""Execution tracing, observability, and visualization.

This module provides tools for tracing execution, collecting metrics, and visualizing
workflows and execution traces.

Examples:
    Set up observability:

        from flowgentra_ai.observability import ExecutionTracer, init_tracing

        tracer = ExecutionTracer()
        init_tracing()
        
    Visualize graphs:
    
        from flowgentra_ai.observability import (
            visualize_graph,
            graph_to_mermaid,
            evaluate_output_score,
            VisualizationConfig
        )
        
        mermaid_str = graph_to_mermaid(compiled_graph)
"""

from flowgentra_ai._native import observability as _o, utils as _u, evaluation as _e

ExecutionTrace = _o.ExecutionTrace
ExecutionTracer = _o.ExecutionTracer
init_tracing = _o.py_init_tracing
visualize_graph = _o.py_visualize_graph
graph_to_dot = _o.py_graph_to_dot
graph_to_mermaid = _o.py_graph_to_mermaid
VisualizationConfig = _u.VisualizationConfig
evaluate_output_score = _e.py_evaluate_output_score

# OpenTelemetry export: convert an ExecutionTrace to OTLP spans and ship them
# to any OTLP HTTP collector (Jaeger, Datadog, Honeycomb, Grafana Tempo, ...).
#
#     tracer = ExecutionTracer()
#     graph = builder.compile(tracer=tracer)
#     graph.invoke({...})
#     spans = trace_to_otel_spans(tracer.get_trace())
#     export_to_otlp("http://localhost:4318", spans)
OtelSpan = _o.OtelSpan
OtelAttribute = _o.OtelAttribute
OtelStatus = _o.OtelStatus
trace_to_otel_spans = _o.py_trace_to_otel_spans
spans_to_otlp_json = _o.py_spans_to_otlp_json
export_to_otlp = _o.py_export_to_otlp

__all__ = [
    # Tracing
    "ExecutionTrace",
    "ExecutionTracer",
    "init_tracing",
    # OpenTelemetry export
    "OtelSpan",
    "OtelAttribute",
    "OtelStatus",
    "trace_to_otel_spans",
    "spans_to_otlp_json",
    "export_to_otlp",
    # Visualization
    "VisualizationConfig",
    "visualize_graph",
    "graph_to_dot",
    "graph_to_mermaid",
    "evaluate_output_score",
]
