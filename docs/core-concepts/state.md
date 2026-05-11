# State Management

State is the data container that flows through your graph. Every node receives the current state, modifies it, and returns it. Understanding state is key to understanding how Flowgentra works.

---

## The basics

=== "Python"

    ```python
    from flowgentra_ai import State

    state = State({"user": "Alice", "score": 0})

    # Read
    state["user"]           # "Alice"
    state.get("score")      # 0
    state.get("missing")    # None (no KeyError)

    # Write
    state["score"] = 42
    state.set("tags", ["rust", "ai"])

    # Check existence
    "user" in state         # True
    state.contains_key("x") # False

    # Delete
    del state["score"]
    state.remove("tags")    # returns the value

    # Iterate
    state.keys()            # ["user"]
    len(state)              # 1
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::DynState;

    let mut state = DynState::new();
    state.set("user", "Alice");
    state.set("score", 0i64);

    // Typed getters
    state.get_string("user")    // Some("Alice")
    state.get_int("score")      // Some(0)
    state.get_bool("flag")      // None if missing or wrong type
    state.get_float("ratio")    // None

    // Check and remove
    state.contains_key("user")  // true
    state.remove("score");
    ```

---

## Creating state

=== "Python"

    ```python
    # From a dict
    state = State({"key": "value"})
    state = State.from_dict({"key": "value"})

    # From JSON
    state = State.from_json('{"key": "value"}')

    # Empty
    state = State()
    ```

=== "Rust"

    ```rust
    // Empty
    let state = DynState::new();

    // From a JSON value
    let state = DynState::from_value(serde_json::json!({
        "key": "value",
    }));
    ```

---

## Serialization

=== "Python"

    ```python
    # To dict
    d = state.to_dict()    # {"key": "value", ...}

    # To JSON string
    j = state.to_json()    # '{"key":"value"}'

    # Deep clone (independent copy)
    copy = state.deep_clone()
    copy["key"] = "new"
    state["key"]           # still "value"
    ```

=== "Rust"

    ```rust
    // To JSON value
    let json_val = state.to_value();

    // To JSON string
    let json_str = serde_json::to_string(&state).unwrap();
    ```

---

## Typed state (Rust only)

In Rust, you can define a typed state struct with compile-time guarantees. This is the preferred pattern for non-trivial graphs.

```rust
use flowgentra_ai::State;

#[derive(State, Debug, Clone)]
pub struct MyState {
    pub query: String,

    #[reducer(Append)]
    pub messages: Vec<String>,   // appended, never overwritten

    #[reducer(Sum)]
    pub token_count: u64,        // accumulated across nodes

    pub result: Option<String>,
}
```

The `#[derive(State)]` macro generates a companion `MyStateUpdate` type. Nodes return partial updates, and the engine merges them using the reducer for each field.

### Reducers

Reducers control how a field is merged when a node returns an update:

| Reducer | Behavior |
|---------|----------|
| `Overwrite` | Last value wins (default) |
| `Append` | Extends lists/strings |
| `Sum` | Adds numbers together |
| `Min` | Keeps the smaller value |
| `Max` | Keeps the larger value |
| `MergeMap` | Deep-merges dictionaries |
| `AppendUnique` | Appends, but deduplicates |

```rust
#[derive(State, Debug, Clone)]
pub struct AgentState {
    pub query: String,

    #[reducer(Append)]
    pub history: Vec<Message>,   // accumulate conversation

    #[reducer(MergeMap)]
    pub context: HashMap<String, String>,  // merge context from multiple nodes

    #[reducer(Max)]
    pub confidence: f64,         // keep highest confidence score
}
```

### Using typed state in a graph

```rust
let graph = StateGraph::<MyState>::builder()
    .add_node("process", |state: MyState| async move {
        // `state` is your typed struct — no string keys, full IDE support
        let result = process(&state.query).await?;
        Ok(MyStateUpdate {
            result: Some(result),
            token_count: 42,
            ..Default::default()
        })
    })
    .entry("process")
    .edge("process", "__end__")
    .build();
```

---

## State in Python graphs

Python uses `State` (the dynamic version) for all graphs. You can also use `PlainState` for non-threaded contexts.

```python
# State — thread-safe, used in all graph workflows
from flowgentra_ai import State

# PlainState — non-thread-safe, has extra typed getters
from flowgentra_ai._native import PlainState

plain = PlainState({"x": 42, "ratio": 3.14, "flag": True})
plain.get_int("x")      # 42
plain.get_float("ratio") # 3.14
plain.get_bool("flag")   # True
```

!!! tip
    Use `State` for everything unless you have a specific reason to use `PlainState`. The thread-safe version is what graph nodes receive automatically.

---

## Common patterns

### Accumulating a list

=== "Python"

    ```python
    def collect(state):
        items = state.get("items") or []
        items.append(compute_item())
        state["items"] = items
        return state
    ```

=== "Rust"

    ```rust
    // With typed state and Append reducer — handled automatically
    // With DynState:
    async fn collect(mut state: DynState) -> Result<DynState> {
        let mut items: Vec<String> = state.get_array("items")
            .unwrap_or_default();
        items.push(compute_item());
        state.set("items", items);
        Ok(state)
    }
    ```

### Passing context between nodes

State is global to the graph — any node can read what any other node wrote.

```python
def fetch(state):
    state["raw_data"] = call_api()
    return state

def process(state):
    # can read what fetch wrote
    data = state["raw_data"]
    state["processed"] = transform(data)
    return state
```

### Error handling via state

A common pattern is to use a state key for errors rather than raising exceptions, so the graph can route to an error handler:

```python
def risky_node(state):
    try:
        state["result"] = do_thing()
    except Exception as e:
        state["error"] = str(e)
    return state

def router(state):
    return "error_handler" if state.get("error") else "next_node"
```
