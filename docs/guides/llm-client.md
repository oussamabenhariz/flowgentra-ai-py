# LLM

Flowgentra provides a unified LLM that works with 7 providers. You configure it once and use the same API regardless of whether you're using OpenAI, Anthropic, or a local Ollama model.

---

## Creating a client

=== "Python"

    ```python
    from flowgentra_ai.llm import LLMConfig, LLM

    # OpenAI
    client = LLM.from_config(
        LLMConfig("openai", "gpt-4", api_key="sk-...")
    )

    # Anthropic
    client = LLM.from_config(
        LLMConfig("anthropic", "claude-3-opus-20240229", api_key="sk-ant-...")
    )

    # Mistral
    client = LLM.from_config(
        LLMConfig("mistral", "mistral-large-latest", api_key="...")
    )

    # Groq (fast inference)
    client = LLM.from_config(
        LLMConfig("groq", "llama3-70b-8192", api_key="gsk_...")
    )

    # Ollama (local — no API key needed)
    client = LLM.from_config(
        LLMConfig("ollama", "llama3")
    )

    # HuggingFace Inference API
    client = LLM.from_config(
        LLMConfig("huggingface", "meta-llama/Meta-Llama-3-8B-Instruct", api_key="hf_...")
    )

    # Azure OpenAI
    client = LLM.from_config(
        LLMConfig("azure", "gpt-4", api_key="...")
    )
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::llm::{LLMConfig, LLM};

    // Convenience constructors
    let client = LLM::from_config(LLMConfig::openai("gpt-4", "sk-..."));
    let client = LLM::from_config(LLMConfig::anthropic("claude-3-opus-20240229", "sk-ant-..."));
    let client = LLM::from_config(LLMConfig::ollama("llama3"));

    // Full config
    let config = LLMConfig {
        provider: "openai".to_string(),
        model: "gpt-4".to_string(),
        api_key: "sk-...".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        ..Default::default()
    };
    let client = LLM::from_config(config);
    ```

### Config options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | required | Provider name (see table below) |
| `model` | `str` | required | Model identifier |
| `api_key` | `str` | `""` | API key (not needed for Ollama) |
| `temperature` | `float` | provider default | Response randomness (0.0–2.0) |
| `max_tokens` | `int` | provider default | Max response length |
| `top_p` | `float` | provider default | Nucleus sampling (0.0–1.0) |

---

## Sending messages

=== "Python"

    ```python
    from flowgentra_ai.llm import Message

    response = client.chat([
        Message.system("You are a helpful assistant."),
        Message.user("What is the capital of France?"),
    ])

    print(response.content)   # "The capital of France is Paris."
    print(response.role)      # "assistant"
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::llm::Message;

    let response = client.chat(vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the capital of France?"),
    ]).await?;

    println!("{}", response.content);
    ```

---

## Token usage & cost

=== "Python"

    ```python
    response, usage = client.chat_with_usage([Message.user("Hello!")])

    if usage:
        print(f"Prompt tokens:     {usage.prompt_tokens}")
        print(f"Completion tokens: {usage.completion_tokens}")
        print(f"Total tokens:      {usage.total_tokens}")

        cost = usage.estimated_cost("gpt-4")
        if cost:
            print(f"Estimated cost: ${cost:.4f}")
    ```

=== "Rust"

    ```rust
    let (response, usage) = client.chat_with_usage(messages).await?;

    if let Some(u) = usage {
        println!("Tokens: {}", u.total_tokens);
        if let Some(cost) = u.estimated_cost("gpt-4") {
            println!("Cost: ${cost:.4f}");
        }
    }
    ```

### Check pricing for a model

=== "Python"

    ```python
    from flowgentra_ai.llm import model_pricing

    pricing = model_pricing("gpt-4")
    if pricing:
        input_price, output_price = pricing
        print(f"Input:  ${input_price}/M tokens")
        print(f"Output: ${output_price}/M tokens")
    ```

---

## Function calling (tools)

=== "Python"

    ```python
    from flowgentra_ai.llm import ToolDefinition

    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get the current weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city":  {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        )
    ]

    response = client.chat_with_tools(
        [Message.user("What's the weather in Paris?")],
        tools,
    )

    if response.has_tool_calls():
        for tc in response.tool_calls():
            print(f"Tool:      {tc.name}")
            print(f"Arguments: {tc.arguments}")
            # tc.arguments is a dict like {"city": "Paris"}
            # You call your actual implementation here:
            result = my_weather_api(tc.arguments["city"])
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::llm::{ToolDefinition, JsonSchema};
    use serde_json::json;

    let tools = vec![
        ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get current weather".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
        }
    ];

    let response = client.chat_with_tools(messages, tools).await?;
    if response.has_tool_calls() {
        for tc in response.tool_calls() {
            println!("Call: {}({:?})", tc.name, tc.arguments);
        }
    }
    ```

---

## Retry, cache, fallback

These return new clients that wrap the original — they're composable.

=== "Python"

    ```python
    # Retry with exponential backoff
    reliable = client.with_retry(max_retries=3)

    # Cache responses (same input = same output, no API call)
    fast = client.cached(max_entries=500)

    # Fallback to another provider if the first fails
    backup = LLM.from_config(LLMConfig("anthropic", "claude-3-haiku-20240307", api_key="..."))
    robust = client.with_fallback(backup)

    # Combine: retry, then cache the successful result
    production = client.with_retry(max_retries=3).cached(max_entries=500)
    ```

=== "Rust"

    ```rust
    // Retry
    let reliable = client.with_retry(RetryConfig { max_retries: 3, ..Default::default() });

    // Cache
    let fast = client.cached(500);

    // Fallback
    let backup = LLM::from_config(LLMConfig::anthropic("claude-3-haiku-20240307", "..."));
    let robust = client.with_fallback(backup);
    ```

---

## Structured output

Force the LLM to return valid JSON or JSON matching a specific schema. This is much more reliable than asking the LLM to "respond in JSON" in the system prompt.

=== "Python"

    ```python
    from flowgentra_ai.types import ResponseFormat

    config = LLMConfig("openai", "gpt-4", api_key="sk-...")

    # Option 1: Force JSON (any valid JSON)
    config.set_response_format(ResponseFormat.json())

    # Option 2: Force JSON matching a specific schema
    config.set_response_format(ResponseFormat.json_schema("person", {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age":  {"type": "integer"},
            "city": {"type": "string"},
        },
        "required": ["name", "age"],
    }))

    client = LLM.from_config(config)
    response = client.chat([Message.user("Extract: John is 30 years old from London")])
    import json
    data = json.loads(response.content)
    print(data["name"])   # "John"
    print(data["age"])    # 30
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::llm::ResponseFormat;
    use serde_json::json;

    let config = LLMConfig::openai("gpt-4", "sk-...")
        .with_response_format(ResponseFormat::JsonSchema {
            name: "person".to_string(),
            schema: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age":  {"type": "integer"},
                },
                "required": ["name", "age"]
            }),
        });
    ```

!!! note
    Structured output is supported by OpenAI and some other providers. For others, use `JsonOutputParser` to parse the response manually.

---

## Prompt templates

Use `PromptTemplate` to avoid string formatting boilerplate.

=== "Python"

    ```python
    from flowgentra_ai import PromptTemplate

    template = PromptTemplate(
        "You are a {role}.\n\nUser asked: {question}\n\nContext: {context}"
    )

    prompt = template.format(
        role="financial analyst",
        question="What was Apple's revenue in 2023?",
        context="Apple reported $383B in total revenue for FY2023.",
    )

    response = client.chat([Message.user(prompt)])
    ```

---

## Output parsers

Parse structured data out of LLM responses.

=== "Python"

    ```python
    from flowgentra_ai import JsonOutputParser, ListOutputParser

    # JSON parser — handles code fences and extra text
    parser = JsonOutputParser()
    data = parser.parse("""
    Here is the result:
    ```json
    {"score": 0.92, "label": "positive"}
    ```
    """)
    # {"score": 0.92, "label": "positive"}

    # List parser — handles bullet points and newlines
    parser = ListOutputParser()
    items = parser.parse("- Rust\n- Python\n- Go")
    # ["Rust", "Python", "Go"]
    ```

---

## Supported providers

| Provider | Config name | Notes |
|----------|-------------|-------|
| OpenAI | `"openai"` | GPT-4, GPT-3.5, O1, O3-mini |
| Anthropic | `"anthropic"` | Claude 3 Opus, Sonnet, Haiku |
| Mistral | `"mistral"` | Mistral Large, Medium, Small |
| Groq | `"groq"` | Fast inference, Llama/Mixtral models |
| Ollama | `"ollama"` | Local models, no API key needed |
| HuggingFace | `"huggingface"` | Inference API, any supported model |
| Azure | `"azure"` | Azure OpenAI Service |
