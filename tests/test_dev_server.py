"""CompiledGraph.serve_dev() — minimal local dev server (graph structure + live SSE events)."""

import json
import time
import urllib.request
from typing import List, TypedDict

import pytest

from flowgentra_ai.graph import END, StateGraph


class DevState(TypedDict):
    messages: List[str]


def _build_graph():
    def echo(state):
        return {"messages": state["messages"] + ["hi"]}

    b = StateGraph(DevState)
    b.add_node("echo", echo)
    b.set_entry_point("echo")
    b.add_edge("echo", END)
    return b.compile()


@pytest.fixture
def dev_server():
    graph = _build_graph()
    handle = graph.serve_dev(7881)
    time.sleep(0.2)
    yield graph, handle
    handle.shutdown()


def test_index_page_served(dev_server):
    _, handle = dev_server
    with urllib.request.urlopen(handle.url) as r:
        assert r.status == 200
        assert "Flowgentra" in r.read().decode()


def test_graph_endpoint_reflects_structure(dev_server):
    _, handle = dev_server
    with urllib.request.urlopen(handle.url + "graph") as r:
        data = json.loads(r.read())
    assert data["entry_point"] == "echo"
    assert data["nodes"] == ["echo"]


def test_shutdown_invalidates_url(dev_server):
    _, handle = dev_server
    handle.shutdown()
    with pytest.raises(Exception):
        _ = handle.url
