"""Every public wrapper module must import cleanly.

Regression guard for the class of bug where a pure-Python wrapper references
a native symbol that moved or was renamed (e.g. document_loaders reading
WebLoader from _native.rag after it moved to _native.loaders): the module
then raises AttributeError at import time and the whole subpackage is dead.
"""

import importlib

import pytest

WRAPPER_MODULES = [
    "flowgentra_ai",
    "flowgentra_ai.agent",
    "flowgentra_ai.graph",
    "flowgentra_ai.llm",
    "flowgentra_ai.memory",
    "flowgentra_ai.tools",
    "flowgentra_ai.rag",
    "flowgentra_ai.document_loaders",
    "flowgentra_ai.supervision",
    "flowgentra_ai.observability",
    "flowgentra_ai.skills",
    "flowgentra_ai.state_schema",
]


@pytest.mark.parametrize("module", WRAPPER_MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_document_loaders_key_symbols():
    from flowgentra_ai.document_loaders import (  # noqa: F401
        CsvLoader,
        DirectoryLoader,
        WebLoader,
        load_directory,
        load_document,
    )


def test_all_agents_have_run():
    """The 0.3.0 unified vocabulary: every agent type exposes run()."""
    from flowgentra_ai import agent as a

    for cls_name in (
        "ZeroShotReAct",
        "FewShotReAct",
        "Conversational",
        "ToolCalling",
        "StructuredChat",
        "SelfAskWithSearch",
        "ReactDocstore",
    ):
        assert hasattr(getattr(a, cls_name), "run"), f"{cls_name} lacks run()"
