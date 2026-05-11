"""Advanced graph nodes for workflows.

This module provides sophisticated node types for complex workflow patterns including
branching, looping, joining, and subgraph composition.

Examples:
    Use advanced nodes:

        from flowgentra_ai.nodes import (
            RetryNode,
            TimeoutNode,
            BranchConfig,
            LoopNodeConfig,
            ParallelNodeConfig
        )

        retry_node = RetryNode(max_retries=3)
        builder.add_node("retry", retry_node)
        
    Configure complex node behavior:
    
        from flowgentra_ai.nodes import (
            MergeStrategy,
            SubgraphNodeConfig,
            JoinNodeConfig
        )
"""

from flowgentra_ai._native import nodes as _n

RetryNode = _n.RetryNode
TimeoutNode = _n.TimeoutNode
JoinType = _n.JoinType
MergeStrategy = _n.MergeStrategy
BranchConfig = _n.BranchConfig
LoopNodeConfig = _n.LoopNodeConfig
ParallelNodeConfig = _n.ParallelNodeConfig
SubgraphNodeConfig = _n.SubgraphNodeConfig
JoinNodeConfig = _n.JoinNodeConfig

__all__ = [
    # Built-in node types
    "RetryNode",
    "TimeoutNode",
    # Configuration types
    "JoinType",
    "MergeStrategy",
    "BranchConfig",
    "LoopNodeConfig",
    "ParallelNodeConfig",
    "SubgraphNodeConfig",
    "JoinNodeConfig",
]
