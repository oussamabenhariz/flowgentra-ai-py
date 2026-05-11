# State Management

State is the data container that flows through your graph. Every node receives the current state and returns an updated state.

## State

`State` is the default, thread-safe state type. It behaves like a Python dict:

```python
from flowgentra_ai import State

# Create with initial data
state = State({"name": "FlowgentraAI", "version": 1})

# Dict-like access
state["count"] = 42
print(state["name"])          # "FlowgentraAI"
print("name" in state)        # True
print(len(state))             # 3
del state["version"]

# Methods
state.set("key", "value")
val = state.get("key")        # "value" (returns None if missing)
val = state.get_string("key") # "value" (returns None if not a string)
keys = state.keys()           # ["name", "count", "key"]
state.remove("key")
```

## Serialization

```python
# To/from dict
d = state.to_dict()           # {"name": "FlowgentraAI", "count": 42}
state = State.from_dict({"a": 1, "b": 2})

# To/from JSON
json_str = state.to_json()
state = State.from_json('{"a": 1, "b": 2}')
```

## Deep Clone

`State` is reference-counted internally. Assigning it to another variable shares the same data. To get an independent copy:

```python
original = State({"x": 1})
clone = original.deep_clone()
clone["x"] = 99
print(original["x"])  # 1 (unchanged)
```

## PlainState

`PlainState` is a non-thread-safe state for advanced use cases where you need direct ownership:

```python
from flowgentra_ai._native import PlainState

state = PlainState({"x": 1})
state["y"] = 2

# Additional typed getters
state.get_string("x")   # None (x is int)
state.get_int("x")      # 1
state.get_float("x")    # None
state.get_bool("x")     # None
```

!!! tip
    Use `State` for all graph workflows.
