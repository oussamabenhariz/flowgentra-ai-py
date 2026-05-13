# Agents

An **agent** is a graph that loops: it reasons, picks an action (usually a tool call or an LLM call), observes the result, and repeats — until it has a final answer.

Flowgentra provides multiple ways to create agents, from simple builder patterns to advanced memory-aware multi-user systems.

---

## Creating Agents: Overview

### Method 1: Typed Agent Constructor (Recommended)

Each agent type is its own class. Pass the LLM config directly — no builder chain needed.

```python
from flowgentra_ai.agent import ZeroShotReAct, ToolSpec
from flowgentra_ai.llm import LLM

llm = LLM(provider="openai", model="gpt-4o")  # or any other provider

agent = ZeroShotReAct(
    name="my-agent",
    llm=llm,
    system_prompt="You are a helpful assistant.",
    tools=[search_tool],
    retries=2,
    memory_steps=12,
)
result = agent.execute_input("What is the capital of France?")
```

Available agent classes: `ZeroShotReAct`, `FewShotReAct`, `Conversational`, `ToolCalling`,
`StructuredChat`, `SelfAskWithSearch`, `ReactDocstore`.

### Method 2: Agent.from_config_path() (Production)

YAML-based configuration for production deployments. Best for complex workflows.

```python
from flowgentra_ai.agent import Agent

agent = Agent.from_config_path("./agent_config.yaml")
```

### Method 3: MemoryAwareAgent (Multi-User)

Automatically manages per-user memory and conversation history. Load from a YAML config file.

```python
from flowgentra_ai.agent import MemoryAwareAgent

agent = MemoryAwareAgent.from_config("agent_memory.yaml")
agent.set_thread_id("user-123")
result = agent.run_turn("What is quantum computing?")
```

### Method 4: Custom Graph-Based Agent

Build custom logic by composing a StateGraph manually.

```python
from flowgentra_ai.graph import StateGraph, END
from flowgentra_ai.llm import LLM, Message

llm = LLM(provider="openai", model="gpt-4")

def llm_node(state):
    response = llm.chat([Message.user(state["query"])])
    return {"response": response.content}

builder = StateGraph(dict)
builder.add_node("llm", llm_node)
builder.set_entry_point("llm")
builder.add_edge("llm", END)
graph = builder.compile()
```

---

## Predefined Agent Types

Flowgentra provides seven built-in agent strategies optimized for different scenarios.

### ZeroShotReAct: Reasoning without Examples

**Constructor:**
```python
ZeroShotReAct(
    name: str,
    llm: LLM,
    system_prompt: str = "",
    tools: list[ToolSpec] = [],
    retries: int = 3,
    memory_steps: int | None = None,
)
```

The classic ReAct (Reason + Act) loop. The agent reasons step-by-step, decides which tool to call, observes the result, and loops until it has an answer — **without** needing example demonstrations.

**Internal Loop:**
```
Input Query
    │
    ▼
[LLM: Reason]  ◄──────────────────────────────┐
    │ "I need to search for..."               │
    ▼                                          │
[Decide]                                       │
    │                                          │
    ├─► Tool call? YES ──► [Execute Tools]    │
    │       │              │ (get observation)│
    │       └──────────────┴──────────────────┘
    │
    └─► Tool call? NO ───► [Generate Answer]
```

**Parameters:**
- `name` (str): Agent identifier
- `llm` (LLM): Configured LLM — set `temperature` and `max_tokens` on the instance
- `system_prompt` (str): Override the default ReAct system prompt
- `tools` (list[ToolSpec]): Tools the agent can call
- `retries` (int): Retry failed LLM calls, default 3
- `memory_steps` (int | None): Steps of conversation history to retain

**Best for:**
- ✅ General-purpose tasks with tools
- ✅ Information gathering (search, APIs, databases)
- ✅ Calculations and text processing
- ✅ When you don't have examples to learn from

**Example:**
```python
from flowgentra_ai.agent import ZeroShotReAct, ToolSpec
from flowgentra_ai.llm import LLM

# Define tools
search = ToolSpec("web_search", "Search the web for current information")
search.add_parameter("query", "string")
search.set_required("query")

calc = ToolSpec("calculator", "Solve math problems")
calc.add_parameter("expression", "string")
calc.set_required("expression")

# Configure LLM with desired settings
llm = LLM(provider="openai", model="gpt-4o", temperature=0.2, max_tokens=2000)

# Build agent
agent = ZeroShotReAct(
    name="research-assistant",
    llm=llm,
    system_prompt="You are a research assistant. Think step by step and cite sources.",
    tools=[search, calc],
    retries=2,
)

# Execute
result = agent.execute_input("What is 2847 × 3.14? Show your reasoning.")
print(result)
```

---

### FewShotReAct: Improved Accuracy with Examples

**Constructor:**
```python
FewShotReAct(
    name: str,
    llm: LLM,
    system_prompt: str = "",   # include worked examples here
    tools: list[ToolSpec] = [],
    retries: int = 3,
    memory_steps: int | None = None,
)
```

Same loop as ZeroShotReAct, but the system prompt includes worked examples. This dramatically improves performance for specialized domains where you can provide 2-5 demonstrations.

**Parameters:**
- `name` (str): Agent identifier
- `llm` (LLM): Configured LLM (set temperature/max_tokens on the instance)
- `system_prompt` (str): Include worked examples in this prompt
- `tools` (list): ToolSpec objects available to the agent
- `retries` (int): Retry failed LLM calls, default 3
- `memory_steps` (int | None): Steps of conversation history to retain

**Example Structure:**
```python
class Example:
    query: str          # User question
    reasoning: str      # Step-by-step thinking
    tool_calls: List[str]  # Which tools were used
    answer: str         # Final answer
```

**Best for:**
- ✅ Specialized domains (finance, legal, medical)
- ✅ When you have 2-5 good reference examples
- ✅ Tasks requiring consistent reasoning patterns
- ✅ Improving accuracy over ZeroShot

**Example:**
```python
from flowgentra_ai.agent import FewShotReAct, ToolSpec
from flowgentra_ai.llm import LLM

# Craft examples directly inside the system prompt string
examples_prompt = """
Here are example reasoning chains:

Example 1:
Question: What was Tesla's Q3 revenue in 2023?
Thought: I need to find Tesla's Q3 2023 financial results.
Action: search_api("Tesla Q3 2023 revenue")
Observation: Tesla reported $23.85B in Q3 2023 revenue
Thought: I have the answer.
Answer: Tesla's Q3 2023 revenue was $23.85 billion.

Example 2:
Question: Calculate the revenue growth rate.
Thought: I need Q3 2023 revenue ($23.85B) and previous data.
Action: calculator("23.85 - 22.5")
Observation: 1.35
Thought: Growth rate = (1.35/22.5) * 100 = 6%
Answer: Revenue grew approximately 6% compared to Q3 2022.
"""

# (legacy – kept for reference, not actually used at runtime)
examples_legacy = [
    Example(
        query="What was Tesla's Q3 revenue in 2023?",
        reasoning="""
        Thought: I need to find Tesla's Q3 2023 financial results.
        Action: Call search_api with query "Tesla Q3 2023 revenue"
        Observation: Tesla reported $23.85B in Q3 2023 revenue
        Thought: I have the answer.
        """,
        tool_calls=["search_api"],
        answer="Tesla's Q3 2023 revenue was $23.85 billion."
    ),
    Example(
        query="Calculate the revenue growth rate",
        reasoning="""
        Thought: I need Q3 2023 revenue ($23.85B) and previous data
        Action: Call calculator with "23.85 - 22.5"
        Observation: 1.35
        Thought: Growth rate = (1.35/22.5) * 100 = 6%
        """,
        tool_calls=["calculator"],
        answer="Revenue grew approximately 6% compared to Q3 2022."
    )
]

# Build agent – pass examples via system_prompt
llm = LLM(provider="openai", model="gpt-4o")
agent = FewShotReAct(
    name="finance-analyzer",
    llm=llm,
    system_prompt=examples_prompt,
    tools=[data_tool],
    memory_steps=12,
)

result = agent.execute_input("What was Apple's revenue growth in 2023?")
```

---

### Conversational: Multi-Turn Dialogue

**Constructor:**
```python
Conversational(
    name: str,
    llm: LLM,
    system_prompt: str = "",
    tools: list[ToolSpec] = [],
    retries: int = 3,
    memory_steps: int | None = None,
)
```

Multi-turn dialogue with persistent conversation history. The agent remembers previous messages and uses context from earlier turns.

**Parameters:**
- `name` (str): Agent identifier
- `llm` (LLM): Configured LLM
- `system_prompt` (str): Initial system instruction
- `tools` (list): Optional tools the agent can call
- `retries` (int): Retry failed LLM calls, default 3
- `memory_steps` (int | None): Steps of history to retain

**Best for:**
- ✅ Chatbots and virtual assistants
- ✅ Multi-turn customer support
- ✅ Interactive learning systems
- ✅ When conversation context matters

**Example:**
```python
from flowgentra_ai.agent import Conversational
from flowgentra_ai.llm import LLM

llm = LLM(provider="openai", model="gpt-4o", temperature=0.7, max_tokens=1000)

agent = Conversational(
    name="support-bot",
    llm=llm,
    system_prompt="""
        You are a friendly customer support representative for TechCorp.
        - Be helpful and professional
        - Remember what the user told you earlier
        - Offer solutions based on their situation
    """,
)

# Multi-turn conversation
turn1 = agent.execute_input("Hi, I can't log into my account")
# → "I'll help you regain access. Can you tell me your email?"

turn2 = agent.execute_input("It's alice@example.com")
# → "Thanks Alice! Have you tried resetting your password?"

turn3 = agent.execute_input("I tried but I didn't get the reset email")
# Agent remembers: email is alice@example.com, can't log in, reset didn't work
# → "Let me check if your email is on our system..."
```

**With Tools and Conversation:**
```python
agent = Conversational(
    name="support-agent",
    llm=LLM(provider="openai", model="gpt-4o"),
    tools=[lookup_account_tool, check_email_tool, create_ticket_tool],
)

result = agent.execute_input("My name is Bob and I want to close my account")
result = agent.execute_input("I'm moving to a new provider")
# Agent remembers Bob's context AND can call tools
```

---

### ToolCalling: Native API Function Calling

**Constructor:**
```python
ToolCalling(
    name: str,
    llm: LLM,
    system_prompt: str = "",
    tools: list[ToolSpec] = [],
    retries: int = 3,
    memory_steps: int | None = None,
)
```

Uses the provider's **native tool/function-calling API** rather than text-based `<action>` tags.  
Tools are passed as structured JSON schemas; the LLM returns structured `tool_calls` in the response.

Supported by: OpenAI, Anthropic, Mistral, Groq, Azure OpenAI, and any OpenAI-compatible endpoint.

**Internal Loop:**
```
Input
  │
  ▼
[LLM: chat_with_tools(messages, tool_defs)]
  │
  ├─► response.tool_calls present → [Execute tool]
  │       │ (result added as tool_result message)
  │       └──────────────────────────────────────► (loop back to LLM)
  │
  └─► no tool_calls → [Return response content]
```

**Key Difference from ZeroShotReAct:**

| | ZeroShotReAct | ToolCalling |
|---|---|---|
| Tool signal | `<action>tool(args)</action>` text | Native `tool_calls` in API response |
| Args format | Free-text string | Structured JSON |
| Provider support | All text models | Models with function-calling support |
| Reliability | Prompt-dependent | API-enforced |

**Best for:**
- ✅ Providers with native function-calling (OpenAI, Anthropic, Mistral)
- ✅ Complex tool schemas with structured arguments
- ✅ When you need reliable, schema-validated tool calls
- ✅ Production systems where prompt-based parsing is too fragile

**Example:**
```python
from flowgentra_ai.agent import ToolCalling, ToolSpec
from flowgentra_ai.llm import LLM

# Define tools with rich schemas
get_weather = ToolSpec("get_weather", "Get current weather for a location")
get_weather.add_parameter("location", "string")
get_weather.add_parameter("unit", "string")   # "celsius" or "fahrenheit"
get_weather.set_required("location")

search = ToolSpec("web_search", "Search the web for current information")
search.add_parameter("query", "string")
search.set_required("query")

agent = ToolCalling(
    name="tool-calling-agent",
    llm=LLM(provider="openai", model="gpt-4o"),
    tools=[get_weather, search],
)

result = agent.execute_input("What's the weather in Paris right now?")
print(result)
```

**With Anthropic:**
```python
agent = ToolCalling(
    name="claude-tool-agent",
    llm=LLM(provider="anthropic", model="claude-3-5-sonnet-20241022"),
    tools=[search],
)
```

---

### StructuredChat: JSON-Structured Actions

ReAct agent that communicates via **JSON blobs** instead of free-text `<action>` tags.  
Each LLM turn produces a fenced JSON block specifying the next action:

```json
{
  "action": "tool_name",
  "action_input": "query or value"
}
```

The final answer is signalled with `"action": "Final Answer"`:

```json
{
  "action": "Final Answer",
  "action_input": "The answer to the user's question."
}
```

**Internal Loop:**
```
Input
  │
  ▼
[LLM: Thought + JSON action blob]
  │
  ├─► action ≠ "Final Answer" → parse action_input → [Execute tool]
  │       │ (result added as Observation)
  │       └──────────────────────────────────────────► (loop with new Thought)
  │
  └─► action == "Final Answer" → return action_input
```

**Best for:**
- ✅ Models that reliably produce JSON (GPT-4, Claude, Mistral Large)
- ✅ Tools with complex, structured `action_input` (objects, not just strings)
- ✅ When you want consistent machine-parseable intermediate reasoning
- ✅ Pipelines that log and inspect each reasoning step

**Constructor:**
```python
StructuredChat(
    name: str,
    llm: LLM,
    system_prompt: str = "",
    tools: list[ToolSpec] = [],
    retries: int = 3,
    memory_steps: int | None = None,
)
```

**Example:**
```python
from flowgentra_ai.agent import StructuredChat, ToolSpec
from flowgentra_ai.llm import LLM

calculator = ToolSpec("calculator", "Evaluate a mathematical expression")
calculator.add_parameter("expression", "string")
calculator.set_required("expression")

lookup = ToolSpec("db_lookup", "Look up a record by ID")
lookup.add_parameter("table", "string")
lookup.add_parameter("id", "integer")
lookup.set_required("table")
lookup.set_required("id")

agent = StructuredChat(
    name="structured-agent",
    llm=LLM(provider="openai", model="gpt-4o", temperature=0.0),
    tools=[calculator, lookup],
)

result = agent.execute_input("What is 15% of 2847?")
print(result)
```

---

### SelfAskWithSearch: Question Decomposition

**Constructor:**
```python
SelfAskWithSearch(
    name: str,
    llm: LLM,
    system_prompt: str = "",
    tools: list[ToolSpec] = [],
    retries: int = 3,
    memory_steps: int | None = None,
)
```

Decomposes complex questions into a **chain of simpler sub-questions**, each
answered by a single `search` tool.  The agent continues decomposing until it
has all the information it needs, then emits `"So the final answer is: ..."`.

**Required:** A tool named `"search"` must be registered.

**Output format:**
```
Question: Are both directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who directed Jaws?
Intermediate answer: Steven Spielberg
Follow up: Where is Steven Spielberg from?
Intermediate answer: The United States
Follow up: Who directed Casino Royale?
Intermediate answer: Martin Campbell
Follow up: Where is Martin Campbell from?
Intermediate answer: New Zealand
So the final answer is: No
```

**Best for:**
- ✅ Multi-hop questions requiring chained lookups
- ✅ Research tasks where you have a single reliable search tool
- ✅ Questions with factual sub-components (Who, What, When)
- ✅ When you want transparent step-by-step reasoning visible in the output

**Parameters:**
- Requires exactly one tool: `search` (any name works, but "search" is conventional)
- The tool executor receives `(tool_name="search", args=<follow_up_question>)`

**Example:**
```python
from flowgentra_ai.agent import SelfAskWithSearch, ToolSpec
from flowgentra_ai.llm import LLM

search = ToolSpec("search", "Search for factual information about a query")
search.add_parameter("query", "string")
search.set_required("query")

agent = SelfAskWithSearch(
    name="research-decomposer",
    llm=LLM(provider="openai", model="gpt-4o", temperature=0.0),
    tools=[search],
)

result = agent.execute_input(
    "Who was the maternal grandfather of George Washington?"
)
print(result)
# → "Joseph Ball"
```

**Multi-hop example:**
```python
result = agent.execute_input(
    "What year was the director of Schindler's List born?"
)
# Agent decomposes:
#   Follow up: Who directed Schindler's List? → Steven Spielberg
#   Follow up: When was Steven Spielberg born? → 1946
#   So the final answer is: 1946
```

---

### ReactDocstore: Search + Lookup Loop

**Constructor:**
```python
ReactDocstore(
    name: str,
    llm: LLM,
    system_prompt: str = "",
    tools: list[ToolSpec] = [],
    retries: int = 3,
    memory_steps: int | None = None,
)
```

ReAct loop specialised for **document-store retrieval**.  Uses two operations:

- `Search[query]`  — finds a document or passage about the query
- `Lookup[term]`   — looks up a specific term in the most recently found document
- `Finish[answer]` — returns the final answer

**Required:** Tools named `"search"` and `"lookup"` (or a single executor that dispatches by name).

**Internal Loop:**
```
Input
  │
  ▼
[LLM: Thought → Action: Search/Lookup/Finish]
  │
  ├─► Action: Search[q]  → call "search" tool → Observation
  ├─► Action: Lookup[t]  → call "lookup" tool → Observation
  │       └─────────────────────────────────────► (loop back to LLM)
  │
  └─► Action: Finish[a]  → return answer
```

**Best for:**
- ✅ Wikipedia-style document stores with Search + Lookup
- ✅ Knowledge bases where you first find a document, then drill into details
- ✅ Multi-hop factual questions over a fixed corpus
- ✅ When Lookup semantics are important (finding terms in a specific document)

**Example:**
```python
from flowgentra_ai.agent import ReactDocstore, ToolSpec
from flowgentra_ai.llm import LLM

search_tool = ToolSpec("search", "Search the document store")
search_tool.add_parameter("query", "string")
search_tool.set_required("query")

lookup_tool = ToolSpec("lookup", "Look up a term in the most recently found document")
lookup_tool.add_parameter("term", "string")
lookup_tool.set_required("term")

agent = ReactDocstore(
    name="docstore-agent",
    llm=LLM(provider="openai", model="gpt-4o", temperature=0.0),
    tools=[search_tool, lookup_tool],
)

result = agent.execute_input(
    "What is the elevation range for the area that the eastern sector "
    "of the Colorado orogeny extends into?"
)
print(result)
# → "1,800 to 7,000 ft (550 to 2,130 m)."
```

**Implementing a Docstore:**
```python
class WikipediaDocstore:
    def __init__(self):
        import wikipedia
        self._current_doc = ""

    def search(self, query: str) -> str:
        try:
            self._current_doc = wikipedia.summary(query, sentences=5)
            return self._current_doc
        except Exception as e:
            return f"Search failed: {e}"

    def lookup(self, term: str, doc: str = "") -> str:
        # Find sentences containing the term
        source = doc or self._current_doc
        sentences = source.split(". ")
        matches = [s for s in sentences if term.lower() in s.lower()]
        if matches:
            return ". ".join(matches[:2])
        return f"Term '{term}' not found in current document."
```

---

### Custom Agent Graph

For full control over agent behavior, build a custom StateGraph with your own nodes and routing logic.

```python
from typing import TypedDict
from flowgentra_ai.graph import StateGraph, END
from flowgentra_ai.llm import LLM, LLMConfig, Message

llm = LLM(provider="openai", model="gpt-4")

class AgentState(TypedDict):
    query: str
    response: str
    done: bool

def llm_node(state: dict) -> dict:
    response = llm.chat([Message.user(state["query"])])
    return {"response": response.content, "done": True}

def router(state: dict) -> str:
    return END if state.get("done") else "llm"

builder = StateGraph(AgentState)
builder.add_node("llm", llm_node)
builder.set_entry_point("llm")
builder.add_conditional_edge("llm", router)
graph = builder.compile()

result = graph.invoke({"query": "What's 15 × 27?", "response": "", "done": False})
print(result["response"])
```

---

## Typed Agent Constructor API

All seven agent classes share the same keyword-argument signature:

```python
AgentClass(
    name: str,                     # Agent identifier
    llm: LLM,                     # Configured LLM (provider, model, temperature, …)
    system_prompt: str = "",       # Override default system instruction
    tools: list[ToolSpec] = [],    # Tools the agent can call
    retries: int = 3,             # Retry failed LLM calls
    memory_steps: int | None = None,  # Steps of conversation history to retain
)
```

**LLM carries temperature and max_tokens:**

```python
from flowgentra_ai.llm import LLM

llm = LLM(
    provider="openai",
    model="gpt-4o",
    temperature=0.3,
    max_tokens=1500,
)
```

**Example — ZeroShotReAct:**
```python
from flowgentra_ai.agent import ZeroShotReAct, ToolSpec
from flowgentra_ai.llm import LLM

agent = ZeroShotReAct(
    name="my-agent",
    llm=LLM(provider="openai", model="gpt-4o", temperature=0.3, max_tokens=1500),
    system_prompt="You are helpful.",
    tools=[search_tool, calc_tool],
    retries=2,
    memory_steps=10,
)
result = agent.execute_input("What is 2847 × 3.14?")
```

### execute_input

```python
agent.execute_input(input: str) -> str
```

Run the agent on a single string and return its final answer.


---

## Agent: from_config_path

For production deployments, define agents in YAML configuration files.

```python
agent = Agent.from_config_path(
    config_path: str,
    handlers_module: Optional[str] = None
) -> Agent
```

**Parameters:**
- `config_path` (str): Path to agent configuration YAML file
- `handlers_module` (Optional[str]): Python module with handler functions

**YAML Configuration:**
```yaml
name: my-agent
version: "1.0"
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
  max_tokens: 2000
memory:
  type: rolling_window
  max_messages: 20
tools:
  - web_search
  - calculator
graph:
  entry_point: classify_input
  nodes:
    - name: classify_input
      handler: classify_input_handler
    - name: search
      handler: search_handler
    - name: final_answer
      handler: final_answer_handler
  edges:
    - from: classify_input
      to: search
      condition: needs_search
    - from: classify_input
      to: final_answer
      condition: direct_answer
    - from: search
      to: final_answer
```

---

## MemoryAwareAgent: Multi-User Support

**Factory:**
```python
MemoryAwareAgent.from_config(config_path: str) -> MemoryAwareAgent
```

Per-user memory management for multi-user systems. Each user gets isolated conversation history.
Load the agent from a YAML config file that specifies the LLM, memory settings, and graph.

**Methods:**

```python
# Set current user thread
agent.set_thread_id(user_id: str) -> None

# Execute one turn
result = agent.run_turn(input: str) -> str

# Get current thread
current = agent.thread_id() -> str

# Get memory statistics
stats = agent.memory_stats() -> MemoryStats

# Clear current user's memory
agent.clear_memory() -> None
```

**Example:**
```python
from flowgentra_ai.agent import MemoryAwareAgent

agent = MemoryAwareAgent.from_config("agent_memory.yaml")

# User Alice
agent.set_thread_id("alice")
r1 = agent.run_turn("Hi! I'm Alice and I live in Paris.")
r2 = agent.run_turn("What city do I live in?")  # Remembers "Paris"

# User Bob - completely separate memory
agent.set_thread_id("bob")
r3 = agent.run_turn("I'm Bob, from New York")
r4 = agent.run_turn("What's my city?")  # Remembers "New York", not "Paris"

# Memory usage
stats = agent.memory_stats()
print(f"Messages: {stats.message_count}, Tokens: ~{stats.tokens}")
```

---

## ToolSpec: Defining Tools

```python
tool = ToolSpec(name: str, description: str) -> ToolSpec
```

Describe a tool's interface so agents know how to call it.

**Methods:**
```python
tool.add_parameter(name: str, param_type: str) -> ToolSpec
tool.set_required(name: str) -> ToolSpec
tool.set_optional(name: str, default: Any = None) -> ToolSpec
tool.set_description(desc: str) -> ToolSpec
```

**Parameter Types:** `"string"`, `"number"`, `"integer"`, `"boolean"`, `"array"`, `"object"`

**Example:**
```python
calculator = ToolSpec("calculator", "Solve mathematical expressions")
calculator.add_parameter("expression", "string")
calculator.set_required("expression")
calculator.set_description("Mathematical expression like '2 + 3 * 4'")

file_reader = ToolSpec("read_file", "Read contents of a text file")
file_reader.add_parameter("path", "string")
file_reader.add_parameter("encoding", "string")
file_reader.set_required("path")
file_reader.set_optional("encoding", "utf-8")

api_caller = ToolSpec("make_request", "Make HTTP requests")
api_caller.add_parameter("url", "string")
api_caller.add_parameter("method", "string")
api_caller.add_parameter("headers", "object")
api_caller.add_parameter("body", "object")
api_caller.set_required("url")
api_caller.set_optional("method", "GET")
```

---

## Agent Execution API

### From StateGraph

```python
result = graph.invoke(state: State) -> State
result = graph.invoke_with_thread(thread_id: str, state: State) -> State

async for node_name, state in graph.stream(state: State):
    print(f"✓ {node_name}")
```

### From Agent (Config-based)

```python
result = agent.run() -> dict
result = agent.run_with_thread(thread_id: str) -> dict
result = agent.get_state(key: str) -> Any
agent.set_state(key: str, value: Any) -> None
```

### From MemoryAwareAgent

```python
agent.set_thread_id(user_id: str) -> None
result = agent.run_turn(input: str) -> str
thread = agent.thread_id() -> str
stats = agent.memory_stats() -> MemoryStats
agent.clear_memory() -> None
```

---

## Best Practices for Agent Design

### 1. Choose the Right Agent Type

| Your Use Case | Recommended Type |
|---------------|-----------------|
| General Q&A with tools | **ZeroShotReAct** |
| Specialized domain (finance, legal) | **FewShotReAct** with examples |
| Chatbot / multi-turn conversation | **Conversational** |
| Provider with native function calling | **ToolCalling** |
| Structured JSON tool arguments | **StructuredChatZeroShotReAct** |
| Multi-hop factual decomposition | **SelfAskWithSearch** |
| Document store Search + Lookup | **ReactDocstore** |
| Complex production workflow | **Agent.from_config_path** |
| Multi-user system | **MemoryAwareAgent** wrapper |

### 2. Design Tools Carefully

**Good tool design:**
```python
# ✓ Simple, single purpose
search = ToolSpec("web_search", "Search the web for factual information")
search.add_parameter("query", "string")
search.set_required("query")

# ✓ Clear descriptions
weather = ToolSpec("get_current_weather", "Get real-time weather for a location")
weather.add_parameter("location", "string")
weather.add_parameter("unit", "string")  # "celsius" or "fahrenheit"
```

**Poor tool design:**
```python
# ✗ Too broad/ambiguous
do_everything = ToolSpec("api_call", "Call APIs")

# ✗ Missing required parameters
search = ToolSpec("search", "Search")  # What kind of search?

# ✗ Confusing names
tool = ToolSpec("foo", "Does stuff")
```

### 3. Use Temperature Appropriately

Set temperature and max_tokens on the `LLM` instance passed to the agent:

```python
from flowgentra_ai.llm import LLM

# For analytical/deterministic reasoning
llm = LLM(provider="openai", model="gpt-4o", temperature=0.1)  # Very focused

# For creative tasks
llm = LLM(provider="openai", model="gpt-4o", temperature=0.9)  # More varied

# Default/balanced
llm = LLM(provider="openai", model="gpt-4o", temperature=0.7)  # Most scenarios
```

### 4. Set Reasonable Limits

```python
from flowgentra_ai.llm import LLM
from flowgentra_ai.agent import ZeroShotReAct

llm = LLM(provider="openai", model="gpt-4o", max_tokens=2000)

agent = ZeroShotReAct(
    name="my-agent",
    llm=llm,
    retries=2,     # Don't retry forever
)
```

### 5. Provide Good System Prompts

```python
# Good: Specific role and task
system_prompt = """You are a customer support agent for TechCorp.
- Be professional and empathetic
- Escalate to human if issue is complex
- Reference customer's account history when available"""

# Poor: Too vague
system_prompt = "You are helpful"
```

### 6. Use Few-Shot Examples for Specialized Domains

```python
# If you have domain-specific reasoning patterns, show examples in system_prompt
agent = FewShotReAct(
    name="specialist",
    llm=LLM(provider="openai", model="gpt-4o"),
    system_prompt="""
    Example 1:
    Question: ...  Thought: ...  Action: ...  Answer: ...

    Example 2:
    Question: ...  Thought: ...  Action: ...  Answer: ...
    """,
)
```

### 7. Monitor for Common Issues

```python
# Infinite loops: Agent keeps calling tools without progress
→ Use max_steps or timeout constraints

# Hallucination: Agent makes up tool responses
→ Implement strict tool result validation

# Context overload: Too many messages in memory
→ Use memory compression or rolling window

# Tool misuse: Agent calls wrong tool repeatedly
→ Improve tool descriptions and examples
```

### 8. Test Agents Thoroughly

```python
# Test with edge cases
test_queries = [
    "What is 2+2?",              # Simple math
    "Tell me about quantum physics",  # General knowledge
    "What's the weather tomorrow in Tokyo?",  # Tool usage
    "Who is the current president of Mars?",  # Should refuse
]

for query in test_queries:
    result = agent.execute_input(query)
    print(f"Q: {query}")
    print(f"A: {result}\n")
```

---

## Common Agent Patterns

### Research Assistant (ZeroShotReAct)

```python
from flowgentra_ai.agent import ZeroShotReAct
from flowgentra_ai.llm import LLM

research_agent = ZeroShotReAct(
    name="researcher",
    llm=LLM(provider="openai", model="gpt-4o"),
    system_prompt="""You are a research assistant.
        1. Search for relevant information
        2. Fetch and read sources
        3. Summarize findings with citations
    """,
    tools=[search_web_tool, fetch_url_tool, summarize_tool],
)
```

### Customer Support Chatbot (Conversational + Tools)

```python
from flowgentra_ai.agent import Conversational

support_agent = Conversational(
    name="support-bot",
    llm=LLM(provider="openai", model="gpt-4o"),
    system_prompt="""You are a helpful support agent.
        - Greet customers warmly
        - Look up their account when needed
        - Resolve issues or escalate appropriately
    """,
    tools=[lookup_customer_tool, check_order_tool, create_refund_tool],
    memory_steps=20,
)
```

### Code Analysis Agent (ZeroShotReAct)

```python
code_agent = ZeroShotReAct(
    name="code-analyzer",
    llm=LLM(provider="openai", model="gpt-4o", temperature=0.2),
    system_prompt="""You are a code analysis expert.
        - Read relevant source files
        - Run tests to understand behavior
        - Identify issues and suggest improvements
    """,
    tools=[read_file_tool, run_tests_tool, lint_tool],
)
```

### Multi-User Chatbot (MemoryAwareAgent)

```python
from flowgentra_ai.agent import MemoryAwareAgent

memory_agent = MemoryAwareAgent.from_config("chatbot_memory.yaml")

memory_agent.set_thread_id("user-alice")
response = memory_agent.run_turn("Hi, I'm Alice")
response = memory_agent.run_turn("What's my name?")  # Remembers "Alice"
```

---
