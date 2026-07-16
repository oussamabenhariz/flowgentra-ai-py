"""Shared fixtures for the binding test suite.

The suite is fully offline and deterministic: no network, no API keys.
It exercises the FFI boundary — where binding bugs live — rather than
re-testing Rust internals (those are covered by cargo test).
"""

from typing import List, TypedDict

import pytest

from flowgentra_ai.graph import StateGraph, END


class ChainState(TypedDict):
    x: int
    log: List[str]


def inc(state: dict) -> dict:
    return {**state, "x": state["x"] + 1, "log": state["log"] + ["inc"]}


def double(state: dict) -> dict:
    return {**state, "x": state["x"] * 2, "log": state["log"] + ["double"]}


@pytest.fixture
def chain_graph():
    """inc -> double -> END."""
    b = StateGraph(ChainState)
    b.add_node("inc", inc)
    b.add_node("double", double)
    b.set_entry_point("inc")
    b.add_edge("inc", "double")
    b.add_edge("double", END)
    return b.compile()


@pytest.fixture
def chain_builder():
    """Uncompiled builder with the inc node registered."""
    b = StateGraph(ChainState)
    b.add_node("inc", inc)
    b.set_entry_point("inc")
    return b
