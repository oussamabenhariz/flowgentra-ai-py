# Installation

## Python

### From PyPI

```bash
pip install flowgentra-ai
```

Requires **Python 3.9+**. Pre-built wheels are available for Linux, macOS, and Windows (x86-64 and ARM).

### Verify

```python
import flowgentra_ai
print(flowgentra_ai.__version__)
```

### Build from source

You'll only need this if you're on an unsupported platform or want to contribute.

**Prerequisites:**

- Python 3.9+
- Rust toolchain — install from [rustup.rs](https://rustup.rs)
- `maturin` build tool

```bash
git clone https://github.com/oussamabenhariz/FlowgentraAI.git
cd FlowgentraAI/flowgentra-ai-py

pip install maturin
maturin develop          # development install (editable)
# or
maturin build --release  # build a wheel
pip install target/wheels/flowgentra_ai-*.whl
```

---

## Rust

### Add to Cargo.toml

```toml
[dependencies]
flowgentra-ai = "0.1"
tokio = { version = "1", features = ["full"] }
```

Flowgentra is fully async — you'll need a Tokio runtime.

### Optional features

```toml
[dependencies]
flowgentra-ai = { version = "0.1", features = ["rag", "mcp", "macros"] }
```

| Feature | What it enables |
|---------|-----------------|
| `rag` | RAG pipeline, vector stores, text splitters |
| `mcp` | Model Context Protocol (MCP) tool servers |
| `macros` | `#[derive(State)]`, `#[node]`, `#[register_handler]` proc macros |
| `otel` | OpenTelemetry tracing export |

### Verify

```bash
cargo check
```

---

## LLM provider API keys

Flowgentra supports multiple LLM providers. You'll need an API key for whichever you use.

| Provider | Environment variable |
|----------|----------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| HuggingFace | `HF_API_KEY` |
| Azure | `AZURE_OPENAI_API_KEY` |
| Ollama | No key needed (local) |

You can pass the key directly in code or read it from the environment. For local development, create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

=== "Python"

    ```python
    import os
    from flowgentra_ai.llm import LLMConfig

    config = LLMConfig("openai", "gpt-4", api_key=os.environ["OPENAI_API_KEY"])
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::llm::LLMConfig;

    let config = LLMConfig::openai("gpt-4", &std::env::var("OPENAI_API_KEY").unwrap());
    ```

---

## Next step

[Quick Start →](quickstart.md)
