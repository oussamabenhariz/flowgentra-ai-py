# State API Reference

## State

Thread-safe shared state container. This is the primary state type used in all graph workflows.

```python
from flowgentra_ai import State
```

### Constructor

```python
State(initial: dict | None = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial` | `dict \| None` | `None` | Optional initial data. All values must be JSON-serializable. |

```python
state = State()                            # empty
state = State({"key": "value", "n": 42})  # with initial data
```

### Class Methods

#### `State.from_dict(d)` → `State`

Create from a Python dict.

| Parameter | Type | Description |
|-----------|------|-------------|
| `d` | `dict` | Source dict |

```python
state = State.from_dict({"key": "value"})
```

#### `State.from_json(json_str)` → `State`

Create from a JSON string.

| Parameter | Type | Description |
|-----------|------|-------------|
| `json_str` | `str` | Valid JSON string |

```python
state = State.from_json('{"key": "value"}')
```

---

### Instance Methods

#### `get(key)` → `Any | None`

Get a value by key. Returns `None` if the key doesn't exist (never raises `KeyError`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Key to look up |

```python
val = state.get("missing_key")   # None, not an error
```

#### `set(key, value)` → `None`

Set a value. The value must be JSON-serializable (str, int, float, bool, list, dict, None).

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Key |
| `value` | `Any` | JSON-serializable value |

#### `contains_key(key)` → `bool`

Check if a key exists.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Key to check |

#### `remove(key)` → `Any | None`

Remove a key and return its value (or `None` if it didn't exist).

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Key to remove |

#### `keys()` → `list[str]`

Return all keys in the state.

#### `get_string(key)` → `str | None`

Get a value as a string. Returns `None` if the key doesn't exist or the value is not a string.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Key |

#### `to_dict()` → `dict`

Convert the entire state to a Python dict.

#### `to_json()` → `str`

Serialize to a JSON string.

#### `deep_clone()` → `State`

Create an independent copy. Modifying the clone does not affect the original.

```python
original = State({"x": 1})
clone = original.deep_clone()
clone["x"] = 99
print(original["x"])   # still 1
```

---

### Dict Protocol

`State` supports standard Python dict operations:

```python
state["key"] = "value"    # set
val = state["key"]        # get (raises KeyError if missing)
del state["key"]          # delete
"key" in state            # check existence
len(state)                # number of keys
```

!!! tip
    Prefer `state.get("key")` over `state["key"]` in node functions — it returns `None` instead of raising `KeyError`, which makes nodes more robust.

---

## PlainState

Non-thread-safe state for advanced use cases where you own the state exclusively. Has extra typed getters compared to `State`.

```python
from flowgentra_ai._native import PlainState
```

!!! note
    Use `State` in graph nodes. `PlainState` is for special cases where you need typed getters and know the state won't be accessed concurrently.

### Constructor

```python
PlainState(initial: dict | None = None)
```

### Additional Typed Getters

| Method | Returns | Description |
|--------|---------|-------------|
| `get_string(key)` | `str \| None` | Get as string, or `None` |
| `get_int(key)` | `int \| None` | Get as integer, or `None` |
| `get_float(key)` | `float \| None` | Get as float, or `None` |
| `get_bool(key)` | `bool \| None` | Get as boolean, or `None` |

```python
plain = PlainState({"score": 42, "ratio": 3.14, "active": True})
plain.get_int("score")     # 42
plain.get_float("ratio")   # 3.14
plain.get_bool("active")   # True
plain.get_string("score")  # None (it's an int, not a string)
```

`PlainState` supports the same dict protocol as `State`.
