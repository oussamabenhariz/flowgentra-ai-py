# LLM API Reference

## LLMConfig

Configuration for an LLM provider.

```python
from flowgentra_ai.llm import LLMConfig
```

### Constructor

```python
LLMConfig(
    provider: str,
    model: str,
    api_key: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | required | Provider name: `"openai"`, `"anthropic"`, `"mistral"`, `"groq"`, `"ollama"`, `"huggingface"`, `"azure"` |
| `model` | `str` | required | Model identifier (e.g., `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`) |
| `api_key` | `str` | `""` | API key. Not needed for Ollama. |
| `temperature` | `float \| None` | provider default | Randomness of responses, 0.0 (deterministic) to 2.0 (creative) |
| `max_tokens` | `int \| None` | provider default | Maximum tokens in the response |
| `top_p` | `float \| None` | provider default | Nucleus sampling parameter, 0.0–1.0 |

```python
# Minimal
config = LLMConfig("openai", "gpt-4", api_key="sk-...")

# With options
config = LLMConfig(
    "anthropic", "claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",
    temperature=0.3,
    max_tokens=1000,
)
```

### Properties

All parameters are readable as properties.

| Property | Type | Notes |
|----------|------|-------|
| `provider` | `str` | Read-only |
| `model` | `str` | Read-only |
| `api_key` | `str` | Read-only |
| `temperature` | `float \| None` | Read/write |
| `max_tokens` | `int \| None` | Read/write |
| `top_p` | `float \| None` | Read/write |

### Methods

#### `set_response_format(format)` → `None`

Force the LLM to return structured output.

| Parameter | Type | Description |
|-----------|------|-------------|
| `format` | `ResponseFormat` | Output format constraint |

```python
from flowgentra_ai.types import ResponseFormat
config.set_response_format(ResponseFormat.json())
```

---

## LLM

Sends messages to an LLM provider.

```python
from flowgentra_ai.llm import LLM
```

### Class Methods

#### `LLM(provider, model, ...)` — preferred constructor

Create a client directly (recommended):

```python
client = LLM(provider="openai", model="gpt-4o", api_key="sk-...")
```

#### `LLM.from_config(config)` → `LLM`

Create a client from an `LLMConfig` object.

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `LLMConfig` | Provider configuration |

```python
config = LLMConfig("openai", "gpt-4o", api_key="sk-...")
client = LLM.from_config(config)
```

---

### `chat(messages)` → `Message`

Send a list of messages and get the assistant's response.

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `list[Message]` | Conversation history |

Returns a `Message` with `role="assistant"`.

```python
response = client.chat([
    Message.system("You are a helpful assistant."),
    Message.user("What is Python?"),
])
print(response.content)
```

---

### `chat_with_usage(messages)` → `(Message, TokenUsage | None)`

Same as `chat()`, but also returns token usage statistics.

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `list[Message]` | Conversation history |

```python
response, usage = client.chat_with_usage([Message.user("Hello!")])
if usage:
    print(f"Total tokens: {usage.total_tokens}")
    print(f"Cost: ${usage.estimated_cost('gpt-4'):.4f}")
```

---

### `chat_with_tools(messages, tools)` → `Message`

Send messages with tool definitions. The LLM may respond with tool calls.

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `list[Message]` | Conversation history |
| `tools` | `list[ToolDefinition]` | Tools the LLM can call |

```python
response = client.chat_with_tools(messages, tools)
if response.has_tool_calls():
    for tc in response.tool_calls():
        print(f"Call: {tc.name}({tc.arguments})")
```

---

### `cached(max_entries)` → `LLM`

Wrap with a response cache. Identical inputs return cached outputs without an API call.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_entries` | `int` | `100` | Maximum number of cached responses |

```python
fast_client = client.cached(max_entries=500)
```

---

### `with_fallback(client)` → `LLM`

Add a fallback provider. If this client fails, the fallback is tried.

| Parameter | Type | Description |
|-----------|------|-------------|
| `client` | `LLM` | Backup client |

```python
backup = LLM(provider="anthropic", model="claude-3-5-haiku-20241022", api_key="...")
robust = client.with_fallback(backup)
```

---

### `with_retry(max_retries)` → `LLM`

Wrap with automatic retry on failure (exponential backoff).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | `int` | `3` | Maximum retry attempts |

```python
reliable = client.with_retry(max_retries=5)
```

---

## Message

A message in a conversation.

```python
from flowgentra_ai.llm import Message
```

### Constructor

```python
Message(role: str, content: str, tool_call_id: str | None = None)
```

### Factory Methods (recommended)

| Method | Description |
|--------|-------------|
| `Message.system(content)` | System message — sets the assistant's behavior |
| `Message.user(content)` | User message |
| `Message.assistant(content)` | Assistant message |
| `Message.tool(content, tool_call_id=None)` | Tool result message |

```python
Message.system("You are a helpful assistant.")
Message.user("What is Rust?")
Message.assistant("Rust is a systems programming language.")
Message.tool('{"result": 42}', tool_call_id="call_abc123")
```

### Properties

| Property | Type | Notes |
|----------|------|-------|
| `role` | `str` | `"system"`, `"user"`, `"assistant"`, or `"tool"` |
| `content` | `str` | Message text (read/write) |
| `tool_call_id` | `str \| None` | ID linking a tool result to a tool call |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `is_system()` | `bool` | True if role is "system" |
| `is_user()` | `bool` | True if role is "user" |
| `is_assistant()` | `bool` | True if role is "assistant" |
| `is_tool()` | `bool` | True if role is "tool" |
| `has_tool_calls()` | `bool` | True if this assistant message contains tool calls |
| `tool_calls()` | `list[ToolCall]` | Get list of tool calls (empty if none) |

---

## ToolCall

A tool call from an LLM response.

```python
from flowgentra_ai.llm import ToolCall
```

### Constructor

```python
ToolCall(id: str, name: str, arguments: Any)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique call identifier (use as `tool_call_id` in the result message) |
| `name` | `str` | Name of the tool to call |
| `arguments` | `dict` | Arguments the LLM chose for this call |

---

## ToolDefinition

Describes a tool to the LLM (used in `chat_with_tools`).

```python
from flowgentra_ai.llm import ToolDefinition
```

### Constructor

```python
ToolDefinition(name: str, description: str, parameters: Any)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tool name |
| `description` | `str` | What the tool does (shown to the LLM) |
| `parameters` | `dict` | JSON Schema describing the input parameters |

```python
ToolDefinition(
    "get_weather",
    "Get current weather for a city",
    {
        "type": "object",
        "properties": {
            "city":  {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Tool name |
| `description` | `str` | Tool description |
| `parameters` | `Any` | JSON Schema |

---

## TokenUsage

Token usage from a chat request.

```python
from flowgentra_ai.llm import TokenUsage
```

### Constructor

```python
TokenUsage(prompt_tokens: int, completion_tokens: int)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `prompt_tokens` | `int` | Tokens consumed by the input messages |
| `completion_tokens` | `int` | Tokens in the response |
| `total_tokens` | `int` | Sum of prompt + completion |

### Methods

#### `estimated_cost(model)` → `float | None`

Returns the estimated cost in USD, or `None` if the model's pricing is not known.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model name |

```python
cost = usage.estimated_cost("gpt-4")
if cost:
    print(f"${cost:.4f}")
```

---

## ResponseFormat

Controls structured output mode.

```python
from flowgentra_ai.types import ResponseFormat   # note: types module, not llm
```

### Factory Methods

| Method | Description |
|--------|-------------|
| `ResponseFormat.text()` | Plain text response (default) |
| `ResponseFormat.json()` | Force any valid JSON |
| `ResponseFormat.json_schema(name, schema)` | Force JSON matching a specific schema |

```python
# Force JSON
config.set_response_format(ResponseFormat.json())

# Force JSON matching a schema
config.set_response_format(ResponseFormat.json_schema("person", {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}))
```

!!! note
    Structured output requires provider support. Works with OpenAI GPT-4 and later. For other providers, use `JsonOutputParser` to parse responses manually.

---

## model_pricing(model)

Returns pricing for a model.

```python
from flowgentra_ai.llm import model_pricing

pricing = model_pricing("gpt-4")
if pricing:
    input_price, output_price = pricing
    # Prices are per million tokens
    print(f"Input:  ${input_price}/M tokens")
    print(f"Output: ${output_price}/M tokens")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model name |

Returns `(float, float) | None` — `(input_price, output_price)` per million tokens, or `None` if unknown.
