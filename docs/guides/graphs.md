# Graph Workflows

Graphs are the core abstraction in Flowgentra. You define nodes (functions), connect them with edges, compile the graph, and invoke it with state.

---

## Core Concepts

### What is a Graph?

A graph is a directed acyclic workflow where:
- **Nodes** are functions that process state
- **Edges** connect nodes to define flow
- **State** flows through nodes and is updated at each step
- **Entry point** is where execution begins
- **Exit/END** is where execution terminates

### Graph Execution Flow

```
    Input State
        │
        ▼
    Entry Point
        │
        ▼
    [Node A] ──► update state
        │
        ▼
    [Decision] ──┬─► [Node B] ──┐
                 │              │
                 └─► [Node C] ──┤
        │
        ▼
    [Node D]
        │
        ▼
      END
```

---

## Building a Graph: Complete API

### StateGraph Constructor

```python
from flowgentra_ai.graph import StateGraph, END

builder = StateGraph(dict)  # Untyped (accepts any state)
```

### Adding Nodes

```python
def my_node(state):
    """Node must accept State and return State."""
    state["result"] = process_data(state["input"])
    return state

# Add node to graph
builder.add_node(name: str, func: Callable[[State], State]) -> StateGraph
```

**Parameters:**
- `name` (str): Unique node identifier
- `func` (Callable): Function that takes State and returns State

**Returns:** StateGraph (for method chaining)

**Example:**
```python
def process_node(state):
    state["processed"] = True
    state["output"] = state["input"].upper()
    return state

builder.add_node("process", process_node)
```

### Adding Edges

```python
# Fixed edge: Always goes from source to destination
builder.add_edge(
    source: str,
    destination: str
) -> StateGraph
```

**Example:**
```python
builder.add_edge("node_a", "node_b")  # node_a always leads to node_b
builder.add_edge("node_b", END)       # node_b ends the workflow
```

### Conditional Edges

```python
# Router function receives state, returns next node name or "__end__"
def my_router(state):
    if state.get("score", 0) > 0.8:
        return "accept"
    elif state.get("score", 0) > 0.5:
        return "review"
    else:
        return "reject"

builder.add_conditional_edge(
    source: str,
    router: Callable[[State], str],
    path_map: Optional[Dict[str, str]] = None
) -> StateGraph
```

**Parameters:**
- `source` (str): Node name to route from
- `router` (Callable): Function returning next node name
- `path_map` (Dict): Optional mapping of router output to node names

**Returns:** StateGraph

**Example:**
```python
def score_router(state):
    score = state["confidence"]
    if score > 0.85:
        return "high_confidence"
    elif score > 0.70:
        return "medium_confidence"
    else:
        return "low_confidence"

builder.add_conditional_edge("evaluate", score_router)
```

### Setting Entry Point

```python
builder.set_entry_point(node_name: str) -> StateGraph
```

**Parameters:**
- `node_name` (str): Name of first node to execute

**Returns:** StateGraph

**Example:**
```python
builder.set_entry_point("classify")  # Start at classify node
```

### Compiling the Graph

```python
graph = builder.compile() -> CompiledGraph
```

**Returns:** CompiledGraph (executable graph instance)

**Example:**
```python
# Basic compilation
graph = builder.compile()

# With file-based checkpointing — call set_checkpointer() BEFORE compile()
builder.set_checkpointer("./checkpoints")
graph = builder.compile()
```

### Setting Max Execution Steps

Prevent infinite loops by setting a maximum number of execution steps:

```python
builder.set_max_steps(steps: int) -> StateGraph
```

**Parameters:**
- `steps` (int): Maximum number of steps before execution halts (default: 1000)

**Returns:** StateGraph

**Example:**
```python
builder.set_max_steps(100)  # Prevent runaway graphs
```

---

## Invoking Compiled Graphs: Complete API

### Basic Invocation

```python
result: State = graph.invoke(
    input_state: State,
    config: Optional[RunnableConfig] = None
)
```

**Parameters:**
- `input_state` (State): Initial state to start execution
- `config` (Optional[RunnableConfig]): Optional execution configuration

**Returns:** Final state after graph completes

**Example:**
```python
initial = State({"input": "hello", "context": []})
result = graph.invoke(initial)
print(result.get_string("output"))  # Print final output
```

### Blocking Invocation (Synchronous)

```python
result: State = graph.invoke_sync(
    input_state: State,
    config: Optional[RunnableConfig] = None
)
```

**Use when:** You're in a synchronous context and don't have an event loop

**Example:**
```python
# Traditional synchronous code
result = graph.invoke_sync(State({"query": "test"}))
```

### Invocation with Thread ID (Checkpointing)

```python
result: State = graph.invoke_with_thread(
    thread_id: str,
    input_state: State,
    config: Optional[RunnableConfig] = None
)
```

**Parameters:**
- `thread_id` (str): Unique identifier for this execution thread
- `input_state` (State): Initial state
- `config` (Optional[RunnableConfig]): Execution configuration

**Returns:** Final state (automatically saved to checkpoint)

**Important:** Requires checkpointer to be configured during compilation

**Example:**
```python
# First turn of conversation
result1 = graph.invoke_with_thread(
    "user-123-session",
    State({"user_message": "Hello"})
)

# Second turn - graph resumes from checkpointed state
result2 = graph.invoke_with_thread(
    "user-123-session",
    State({"user_message": "Tell me more"})
)
```

### Stream Execution

```python
async def stream(
    input_state: State,
    config: Optional[RunnableConfig] = None
) -> AsyncIterator[Tuple[str, State]]
```

**Parameters:**
- `input_state` (State): Initial state
- `config` (Optional[RunnableConfig]): Execution configuration

**Yields:** Tuples of (node_name, state_at_step)

**Use when:** You need to monitor step-by-step execution

**Example:**
```python
async for node_name, state in graph.stream(initial_state):
    print(f"✓ Executed: {node_name}")
    print(f"  State keys: {list(state.keys())}")
    print(f"  ---")
```

---

## Graph Inspection API

After compiling a graph, you can inspect its structure:

### Get All Node Names

```python
nodes: List[str] = graph.node_names()
```

**Example:**
```python
names = graph.node_names()  # ["classify", "process", "validate"]
```

### Get Entry Point

```python
entry: str = graph.entry_point()
```

**Example:**
```python
start = graph.entry_point()  # "classify"
```

### Get All Edges

```python
edges: List[Tuple[str, str]] = graph.edges()
```

**Returns:** List of (source, destination) tuples

**Example:**
```python
for src, dest in graph.edges():
    print(f"{src} → {dest}")
```

### Get Edges From Node

```python
next_nodes: List[str] = graph.get_edges_from_node(node_name: str)
```

**Example:**
```python
what_comes_after = graph.get_edges_from_node("classify")
# ["high_priority", "normal"]
```

### Visualization: Mermaid Diagram

```python
mermaid: str = graph.to_mermaid()
```

**Example:**
```python
diagram = graph.to_mermaid()
print(diagram)  # Mermaid-compatible diagram string
# Can render in markdown or web interfaces
```

### Visualization: Graphviz DOT

```python
dot: str = graph.to_dot()
```

**Example:**
```python
dot_format = graph.to_dot()
# Can save to file and render with Graphviz
with open("graph.dot", "w") as f:
    f.write(dot_format)
```

### Visualization: JSON

```python
json_repr: dict = graph.to_json()
```

**Example:**
```python
import json
graph_dict = graph.to_json()
json_str = json.dumps(graph_dict, indent=2)
```

---

## Advanced Graph Patterns

### Subgraphs: Composing Graphs

Embed one graph inside another as a single node:

```python
# Build inner graph
inner_builder = StateGraph(dict)
inner_builder.add_node("process", process_fn)
inner_builder.add_node("validate", validate_fn)
inner_builder.set_entry_point("process")
inner_builder.add_edge("process", "validate")
inner_builder.add_edge("validate", END)
inner_graph = inner_builder.compile()

# Use as a node in outer graph
def subgraph_wrapper(state):
    result = await inner_graph.invoke(state)
    return result

outer_builder = StateGraph(dict)
outer_builder.add_node("prepare", prepare_fn)
outer_builder.add_node("subprocess", subgraph_wrapper)
outer_builder.set_entry_point("prepare")
outer_builder.add_edge("prepare", "subprocess")
outer_builder.add_edge("subprocess", END)
outer_graph = outer_builder.compile()

result = outer_graph.invoke(State({"data": "..."}))
```

### Checkpointing and Persistence

Persist graph state for:
- **Recovery:** Resume after interrupts
- **Audit:** Track all execution steps
- **Resumable workflows:** Human-in-the-loop patterns

See [Human-in-the-Loop](human-in-the-loop.md) for interrupt/resume patterns.

## Built-in Node Types

The builder offers specialized methods for common patterns:

### Retry Node

```python
builder.add_retry_node(
    name: str,
    func: Callable,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    backoff_ms: int = 1000
) -> StateGraph
```

Automatically retries a function with exponential backoff:

**Parameters:**
- `name` (str): Node identifier
- `func` (Callable): Function to retry
- `max_retries` (int): Number of retry attempts (default: 3)
- `backoff_factor` (float): Multiplier for exponential backoff (default: 2.0)
- `backoff_ms` (int): Initial backoff milliseconds (default: 1000)

**Example:**
```python
def flaky_fetch(state):
    data = fetch_api_that_might_fail(state["url"])
    state["data"] = data
    return state

builder.add_retry_node("fetch", flaky_fetch, max_retries=5, backoff_ms=500)
```

### Timeout Node

```python
builder.add_timeout_node(
    name: str,
    func: Callable,
    timeout_ms: int,
    on_timeout: str = "error"  # or "skip"
) -> StateGraph
```

Enforces a time limit on a function:

**Parameters:**
- `name` (str): Node identifier
- `func` (Callable): Function to timeout
- `timeout_ms` (int): Timeout in milliseconds
- `on_timeout` (str): "error" raises exception, "skip" returns original state

**Example:**
```python
def slow_operation(state):
    result = very_slow_api_call()
    state["result"] = result
    return state

builder.add_timeout_node("slow_op", slow_operation, timeout_ms=5000, on_timeout="skip")
```

### LLM Node

```python
builder.add_llm_node(
    name: str,
    llm: LLM,
    prompt_key: str = "prompt",
    output_key: str = "response",
    system_prompt: Optional[str] = None,
    model_args: Optional[dict] = None
) -> StateGraph
```

Reads a prompt from state, calls the LLM, writes the response:

**Parameters:**
- `name` (str): Node identifier
- `llm` (LLM): Configured LLM
- `prompt_key` (str): State key for input prompt (default: "prompt")
- `output_key` (str): State key for LLM response (default: "response")
- `system_prompt` (Optional[str]): System prompt to prepend (default: None)
- `model_args` (Optional[dict]): Additional model parameters (default: None)

**Example:**
```python
from flowgentra_ai.llm import LLM

client = LLM(provider="openai", model="gpt-4o", api_key="sk-...")

builder.add_llm_node(
    "generate",
    client,
    prompt_key="user_query",
    output_key="llm_response",
    system_prompt="You are a helpful assistant.",
)
```

### Planner Node

```python
builder.add_planner_node(
    name: str,
    llm: LLM,
    system_prompt: Optional[str] = None
) -> StateGraph
```

LLM-driven dynamic routing. The planner analyzes available nodes and routes intelligently:

**How it works:**
1. Reads `_reachable_nodes` from state (list of available next nodes)
2. Uses LLM to decide which node to execute next
3. Sets `_next_node` in state
4. Router automatically routes to selected node

**Example:**
```python
builder.add_planner_node("intelligent_router", llm)
```

### Evaluation Node

```python
builder.add_evaluation_node(
    handler: Callable,
    config: EvaluationNodeConfig,
    scorer: Callable[[Any, int], Tuple[float, str]]
) -> StateGraph
```

Iteratively refines output until a quality threshold:

**Parameters:**
- `handler` (Callable): Function to refine output
- `config` (EvaluationNodeConfig): Configuration (see below)
- `scorer` (Callable): Function returning (score: float 0-1, feedback: str)

**EvaluationNodeConfig:**
```python
EvaluationNodeConfig(
    name: str,                    # Node identifier
    field_state: str,             # State key to evaluate
    min_confidence: float = 0.8,  # Quality threshold
    max_retries: int = 3,         # Max refinement attempts
    rubric: str = ""              # Scoring guidance
)
```

**Example:**
```python
from flowgentra_ai.graph import EvaluationNodeConfig

def refine_essay(state):
    # Refine the essay in state["draft"]
    state["draft"] = run_llm_refinement(state["draft"])
    return state

def score_essay(output, attempt):
    # Evaluate quality
    clarity = evaluate_clarity(output)
    structure = evaluate_structure(output)
    score = (clarity + structure) / 2
    feedback = f"Clarity: {clarity}, Structure: {structure}"
    return (score, feedback)

config = EvaluationNodeConfig(
    name="refine_essay",
    field_state="draft",
    min_confidence=0.85,
    max_retries=3,
    rubric="Is the essay clear, well-structured, and persuasive?"
)

builder.add_evaluation_node(refine_essay, config, score_essay)
```

---

## State Manipulation: Complete API

State is the data structure flowing through your graph. Here are all operations:

### Read Operations

```python
# Get value with type
value = state.get("key", default=None)

# Get as string
text = state.get_string("key", default="")

# Get as int
num = state.get_int("key", default=0)

# Get as dict
obj = state.get_dict("key", default={})

# Check if key exists
has_key = "key" in state

# Get all keys
keys = state.keys()

# Get all values
values = state.values()
```

### Write Operations

```python
# Set value
state["key"] = "value"
state.set("key", "value")

# Update multiple
state.update({"key1": "val1", "key2": "val2"})

# Remove key
del state["key"]
state.pop("key", default=None)
```

### Serialization

```python
# Convert to dict
dict_form = state.to_dict()

# Convert to JSON string
json_str = state.to_json()

# Convert to string representation
str_repr = str(state)
```

### Example Usage in Nodes

```python
def complex_node(state):
    # Read
    user_input = state.get_string("user_message")
    history = state.get("message_history", default=[])
    config = state.get_dict("config", default={})
    
    # Process
    result = process(user_input, history, config)
    
    # Write
    state["output"] = result
    state["message_history"] = history + [{"input": user_input, "output": result}]
    
    # Check
    if "error" in state:
        state["has_error"] = True
    
    return state
```

---

## Complete Workflow Example

```python
from flowgentra_ai.graph import StateGraph, END
from flowgentra_ai import State
import asyncio

# Define Nodes
def classify_task(state):
    """Classify user input by urgency."""
    text = state.get_string("user_input", "")
    urgency_keywords = ["urgent", "asap", "critical", "emergency"]
    
    is_urgent = any(kw in text.lower() for kw in urgency_keywords)
    state["urgency"] = "high" if is_urgent else "normal"
    return state

def handle_urgent(state):
    """Handle high-urgency tasks."""
    state["priority_queue_position"] = "front"
    state["output"] = f"URGENT: {state['user_input']}"
    return state

def handle_normal(state):
    """Handle normal tasks."""
    state["priority_queue_position"] = "back"
    state["output"] = f"Standard: {state['user_input']}"
    return state

def log_result(state):
    """Log the final result."""
    state["logged"] = True
    print(f"[LOG] {state['output']}")
    return state

# Build Graph
builder = StateGraph(dict)

# Add nodes
builder.add_node("classify", classify_task)
builder.add_node("urgent", handle_urgent)
builder.add_node("normal", handle_normal)
builder.add_node("log", log_result)

# Define entry point
builder.set_entry_point("classify")

# Define routing
def route_by_urgency(state):
    return "urgent" if state.get("urgency") == "high" else "normal"

# Add edges
builder.add_conditional_edge("classify", route_by_urgency)
builder.add_edge("urgent", "log")
builder.add_edge("normal", "log")
builder.add_edge("log", END)

# Compile
graph = builder.compile()

# Execute
async def main():
    # Test 1: Urgent task
    result1 = graph.invoke(State({"user_input": "URGENT: Fix the database"}))
    print(f"Result 1: {result1.get_string('output')}\n")
    
    # Test 2: Normal task
    result2 = graph.invoke(State({"user_input": "Update the documentation"}))
    print(f"Result 2: {result2.get_string('output')}\n")
    
    # Stream execution
    print("Streaming execution:")
    async for node_name, state in graph.stream(State({"user_input": "URGENT: Critical issue"})):
        print(f"  ✓ {node_name}")

asyncio.run(main())
```

---

## Complete API Reference Tables

### StateGraph Methods

| Method | Signature | Returns | Purpose |
|--------|-----------|---------|---------|
| `add_node` | `(name, fn)` | StateGraph | Add computation node |
| `add_edge` | `(source, dest)` | StateGraph | Fixed routing |
| `add_conditional_edge` | `(source, router)` | StateGraph | Dynamic routing |
| `add_retry_node` | `(name, fn, max_retries, backoff_ms)` | StateGraph | Retrying node |
| `add_timeout_node` | `(name, fn, timeout_ms, on_timeout)` | StateGraph | Timeout enforcement |
| `add_llm_node` | `(name, client, prompt_key, output_key)` | StateGraph | LLM integration |
| `add_planner_node` | `(name, client, system_prompt)` | StateGraph | Intelligent routing |
| `add_evaluation_node` | `(handler, config, scorer)` | StateGraph | Quality refinement |
| `set_entry_point` | `(node_name)` | StateGraph | Set start node |
| `set_max_steps` | `(steps)` | StateGraph | Execution limit |
| `compile` | `(checkpointer)` | CompiledGraph | Create executable |

### CompiledGraph Methods

| Method | Signature | Returns | Purpose |
|--------|-----------|---------|---------|
| `invoke` | `(state, config)` | State | Execute graph |
| `invoke_sync` | `(state, config)` | State | Synchronous execution |
| `invoke_with_thread` | `(thread_id, state, config)` | State | With checkpointing |
| `stream` | `(state, config)` | AsyncIterator | Stream execution |
| `node_names` | `()` | List[str] | Get all nodes |
| `entry_point` | `()` | str | Get start node |
| `edges` | `()` | List[Tuple] | Get all edges |
| `get_edges_from_node` | `(node_name)` | List[str] | Get next nodes |
| `to_mermaid` | `()` | str | Mermaid diagram |
| `to_dot` | `()` | str | Graphviz DOT |
| `to_json` | `()` | dict | JSON representation |

### State Methods

| Method | Signature | Returns | Purpose |
|--------|-----------|---------|---------|
| `get` | `(key, default=None)` | Any | Get value |
| `get_string` | `(key, default="")` | str | Get as string |
| `get_int` | `(key, default=0)` | int | Get as int |
| `get_dict` | `(key, default={})` | dict | Get as dict |
| `set` | `(key, value)` | None | Set value |
| `update` | `(dict)` | None | Update multiple |
| `keys` | `()` | List[str] | Get all keys |
| `values` | `()` | List[Any] | Get all values |
| `to_dict` | `()` | dict | Serialize to dict |
| `to_json` | `()` | str | Serialize to JSON |

---

## Best Practices

### 1. Keep Nodes Focused

Each node should do one thing well:

```python
# Good ✓
def validate_email(state):
    is_valid = check_email(state["email"])
    state["email_valid"] = is_valid
    return state

# Avoid ✗
def do_everything(state):
    # validate, transform, call API, log, update score...
    pass
```

### 2. Use Descriptive Node Names

```python
# Good ✓
builder.add_node("extract_entities", extract_entities_fn)

# Avoid ✗
builder.add_node("step_2", process_fn)
```

### 3. Handle Missing Keys Gracefully

```python
# Good ✓
def safe_node(state):
    value = state.get("optional_key", "default")
    config = state.get_dict("config", {})  # Safe even if missing
    # ...

# Avoid ✗
def unsafe_node(state):
    value = state["optional_key"]  # KeyError if missing
```

### 4. Test Nodes Before Graphing

```python
# Test node in isolation
initial_state = State({"input": "test"})
output = my_node(initial_state)
assert output.get("result") == expected
```

### 5. Use Conditional Edges for Complex Logic

```python
# Good ✓
def intelligent_router(state):
    score = state.get("score", 0)
    if score > 0.8:
        return "accept"
    elif score > 0.5:
        return "review"
    else:
        return "reject"

builder.add_conditional_edge("evaluate", intelligent_router)

# Avoid ✗ - don't embed routing in nodes
def node_with_internal_routing(state):
    if state["should_route"]:
        # ... execute path 1
    else:
        # ... execute path 2
```

### 6. Visualize During Development

```python
# So you can see the graph structure
graph = builder.compile()
print(graph.to_mermaid())
```

### 7. Use Streaming for Long Operations

```python
# Good ✓ - see progress
async for node_name, state in graph.stream(initial_state):
    print(f"✓ {node_name}")

# Less ideal ✗ - wait for entire execution
result = graph.invoke(initial_state)
```

### 8. Persist with Checkpointing

```python
# For resumable workflows — set checkpointer before compile
builder.set_checkpointer("./state_checkpoints")
graph = builder.compile()

# Later, resume from thread
state = graph.invoke_with_thread("user-123", new_input)
```

---

## Message Graph

For chat-focused workflows, `MessageGraphBuilder` pre-configures message accumulation:

```python
from flowgentra_ai.graph import MessageGraphBuilder
from flowgentra_ai.llm import Message

def echo(messages):
    """Receives list of Messages, returns list of Messages."""
    last = messages[-1]
    return messages + [Message.assistant(f"Echo: {last.content}")]

builder = MessageGraphBuilder()
builder.add_node("echo", echo)
builder.set_entry_point("echo")
builder.add_edge("echo", "__end__")
graph = builder.compile()

result = graph.invoke([Message.user("Hello")])
for msg in result:
    print(f"{msg.role}: {msg.content}")
```
