"""State: dict parity, serialization bounds, snapshot/restore."""

import pytest

from flowgentra_ai import State
from flowgentra_ai.exceptions import SerializationError


def test_basic_get_set():
    s = State({"a": 1})
    s["b"] = [1, 2, 3]
    assert s["a"] == 1
    assert s["b"] == [1, 2, 3]
    assert len(s) == 2
    assert "a" in s and "missing" not in s


def test_getitem_keyerror():
    s = State()
    with pytest.raises(KeyError):
        s["nope"]


def test_get_with_default():
    s = State({"a": 1})
    assert s.get("a") == 1
    assert s.get("missing") is None
    assert s.get("missing", 42) == 42


def test_items_values_iter():
    s = State({"a": 1, "b": 2})
    assert sorted(s.items()) == [("a", 1), ("b", 2)]
    assert sorted(s.values()) == [1, 2]
    assert sorted(iter(s)) == ["a", "b"]


def test_update_and_pop():
    s = State({"a": 1})
    s.update({"b": 2, "c": 3})
    assert s["b"] == 2 and s["c"] == 3
    assert s.pop("c") == 3
    assert "c" not in s
    assert s.pop("gone", "fallback") == "fallback"
    with pytest.raises(KeyError):
        s.pop("gone")


def test_equality_with_dict_and_state():
    s1 = State({"a": 1, "b": [1, 2]})
    s2 = State({"a": 1, "b": [1, 2]})
    assert s1 == s2
    assert s1 == {"a": 1, "b": [1, 2]}
    assert not (s1 == {"a": 999})
    assert not (s1 == "not a mapping")


def test_json_round_trip():
    s = State({"nested": {"list": [1, {"deep": True}]}, "n": 3.5})
    restored = State.from_json(s.to_json())
    assert restored == s


def test_from_json_rejects_non_object():
    with pytest.raises(SerializationError):
        State.from_json("[1, 2, 3]")


def test_from_json_rejects_invalid_json():
    with pytest.raises(SerializationError):
        State.from_json("{broken")


def test_from_json_rejects_oversized_input():
    # 64 MB + 1 byte of JSON. Built as a single huge string value.
    big = '{"k": "' + "x" * (64 * 1024 * 1024) + '"}'
    with pytest.raises(SerializationError, match="64 MB"):
        State.from_json(big)


def test_snapshot_restore():
    s = State({"steps": 0})
    snap = s.snapshot("before")
    s["steps"] = 99
    s.restore(snap)
    assert s["steps"] == 0


def test_deep_clone_is_independent():
    s = State({"a": 1})
    c = s.deep_clone()
    c["a"] = 2
    assert s["a"] == 1
    assert c["a"] == 2


def test_unicode_and_special_values_round_trip():
    s = State({"emoji": "🦀", "none": None, "bool": True, "neg": -1.5})
    d = s.to_dict()
    assert d["emoji"] == "🦀"
    assert d["none"] is None
    assert d["bool"] is True
    assert d["neg"] == -1.5
