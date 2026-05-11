"""Multi-agent orchestration and supervision.

This module provides components for coordinating multiple agents, distributing work,
and aggregating results in complex workflows.

Examples:
    Create a supervised multi-agent workflow:

        from flowgentra_ai.supervision import (
            Supervisor,
            OrchestrationStrategy,
            SupervisorNodeConfig,
            PlannerNode
        )

        supervisor = Supervisor(agents=[agent1, agent2])
        supervisor.add_node("supervisor", supervisor)
        
    Use parallel processing:
    
        from flowgentra_ai.supervision import (
            ParallelAggregation,
            ParallelMergeStrategy,
            ChildExecutionStats
        )
"""

from flowgentra_ai._native import nodes as _n

Supervisor = _n.Supervisor
OrchestrationStrategy = _n.OrchestrationStrategy
SupervisorNodeConfig = _n.SupervisorNodeConfig
ParallelAggregation = _n.ParallelAggregation
ParallelMergeStrategy = _n.ParallelMergeStrategy
ChildExecutionStats = _n.ChildExecutionStats
PlannerNode = _n.PlannerNode

__all__ = [
    "Supervisor",
    "OrchestrationStrategy",
    "SupervisorNodeConfig",
    "ParallelAggregation",
    "ParallelMergeStrategy",
    "ChildExecutionStats",
    "PlannerNode",
]
