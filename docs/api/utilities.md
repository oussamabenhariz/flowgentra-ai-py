# Utilities API Reference

## model_pricing

Returns cost information for an LLM model.

```python
from flowgentra_ai.llm import model_pricing

pricing = model_pricing("gpt-4")
if pricing:
    input_price, output_price = pricing
    print(f"Input:  ${input_price}/M tokens")
    print(f"Output: ${output_price}/M tokens")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model name (e.g., `"gpt-4"`, `"claude-3-opus-20240229"`) |

Returns `(float, float) | None` — `(input_price_per_M, output_price_per_M)`, or `None` if the model is not in the pricing database.

---

## Text Splitters

```python
from flowgentra_ai.rag import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    HTMLTextSplitter,
    TokenTextSplitter,
    CodeTextSplitter,
)
```

All splitters have the same interface:

| Method | Returns | Description |
|--------|---------|-------------|
| `split(text)` | `list[str]` | Split text into chunks |

### RecursiveCharacterTextSplitter

Best for plain text. Recursively tries to split on paragraphs, then sentences, then words.

```python
RecursiveCharacterTextSplitter(chunk_size: int, overlap: int = 0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | `int` | required | Max characters per chunk |
| `overlap` | `int` | `0` | Characters to repeat between adjacent chunks |

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)
chunks = splitter.split(long_text)
```

### MarkdownTextSplitter

Splits on Markdown headings (`#`, `##`, etc.) to preserve section boundaries.

```python
MarkdownTextSplitter(chunk_size: int, overlap: int = 0)
```

### HTMLTextSplitter

Strips HTML tags, then splits while respecting tag boundaries.

```python
HTMLTextSplitter(chunk_size: int, overlap: int = 0)
```

### TokenTextSplitter

Splits by token count rather than characters. More accurate for LLM context window management.

```python
TokenTextSplitter(max_tokens: int, overlap_tokens: int = 0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `int` | required | Max tokens per chunk |
| `overlap_tokens` | `int` | `0` | Tokens to repeat between chunks |

```python
splitter = TokenTextSplitter(max_tokens=200, overlap_tokens=20)
chunks = splitter.split(text)
```

### CodeTextSplitter

Splits code files while respecting function and class boundaries.

```python
CodeTextSplitter(chunk_size: int, overlap: int = 0)
```

---

## Prompt Templates

### PromptTemplate

Format a string template with named variables.

```python
from flowgentra_ai import PromptTemplate
```

### Constructor

```python
PromptTemplate(template: str)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `template` | `str` | Template string with `{variable_name}` placeholders |

### Methods

#### `format(**kwargs)` → `str`

Fill in all variables and return the formatted string.

```python
template = PromptTemplate(
    "You are a {role}.\n\nQuestion: {question}\n\nContext: {context}"
)

prompt = template.format(
    role="financial analyst",
    question="What was Apple's revenue in 2023?",
    context="Apple's FY2023 revenue was $383 billion.",
)
```

---

## Output Parsers

### JsonOutputParser

Parse JSON from LLM responses. Handles common formatting issues like code fences and extra prose.

```python
from flowgentra_ai import JsonOutputParser
```

### Constructor

```python
JsonOutputParser()
```

### Methods

#### `parse(text)` → `Any`

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | LLM response text |

```python
parser = JsonOutputParser()
data = parser.parse("""
Here is the extracted data:
```json
{"name": "Alice", "age": 30, "city": "Paris"}
```
""")
# {"name": "Alice", "age": 30, "city": "Paris"}
```

---

### ListOutputParser

Parse a list from LLM output. Handles bullet points, numbered lists, and newline-separated items.

```python
from flowgentra_ai import ListOutputParser
```

### Constructor

```python
ListOutputParser()
```

### Methods

#### `parse(text)` → `list[str]`

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | LLM response text |

```python
parser = ListOutputParser()
items = parser.parse("- Rust\n- Python\n- Go")
# ["Rust", "Python", "Go"]

items = parser.parse("1. First\n2. Second\n3. Third")
# ["First", "Second", "Third"]
```

---

## Routing Conditions

Build declarative routing conditions without writing if-statements.

```python
from flowgentra_ai.types import ComparisonOp, Condition, ConditionBuilder
```

### ConditionBuilder

```python
condition = (
    ConditionBuilder()
    .field("score")
    .op(ComparisonOp.GreaterThan)
    .value(0.8)
    .build()
)
```

### ComparisonOp

| Value | Operator |
|-------|----------|
| `ComparisonOp.Equals` | `==` |
| `ComparisonOp.NotEquals` | `!=` |
| `ComparisonOp.GreaterThan` | `>` |
| `ComparisonOp.LessThan` | `<` |
| `ComparisonOp.GreaterThanOrEqual` | `>=` |
| `ComparisonOp.LessThanOrEqual` | `<=` |
| `ComparisonOp.Contains` | String or list containment |

---

## Evaluation

```python
from flowgentra_ai.evaluation import (
    EvaluationNodeConfig,
    EvaluationResult,
    EvaluationConfig,
    ScoringConfig,
    GradingConfig,
    EvaluationReport,
    NodeResult,
    evaluate_output_score,
)
```

### EvaluationNodeConfig

Configuration for `builder.add_evaluation_node()`.

```python
EvaluationNodeConfig(
    name: str,
    field_state: str,
    min_confidence: float = 0.8,
    max_retries: int = 3,
    rubric: str = "",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Node name |
| `field_state` | `str` | required | State key containing the output to evaluate |
| `min_confidence` | `float` | `0.8` | Minimum score to stop refining (0.0–1.0) |
| `max_retries` | `int` | `3` | Max refinement attempts before giving up |
| `rubric` | `str` | `""` | Plain-English criteria for evaluating the output |

---

## Rerankers

```python
from flowgentra_ai.rerankers import NoopReranker, RRFReranker, CrossEncoderReranker, LLMReranker
```

| Class | Speed | Quality | Description |
|-------|-------|---------|-------------|
| `NoopReranker` | instant | baseline | Pass-through, no reranking |
| `RRFReranker` | fast | good | Reciprocal Rank Fusion — combines rankings |
| `CrossEncoderReranker` | slow | high | Neural cross-encoder model |
| `LLMReranker` | slow | very high | Uses your LLM to score relevance |

```python
reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rerank(results, query="my query")
```

---

## Advanced Node Configs

```python
from flowgentra_ai.nodes import (
    LoopNodeConfig,
    ParallelNodeConfig,
    SubgraphNodeConfig,
    JoinNodeConfig,
    JoinType,
    MergeStrategy,
    BranchConfig,
)
```

### LoopNodeConfig

Repeat a node until a condition in state is truthy.

```python
LoopNodeConfig(
    max_iterations: int = 10,
    break_condition: str = "done",  # state key; loop breaks when truthy
)
```

### ParallelNodeConfig

Run multiple branches concurrently and merge results.

```python
ParallelNodeConfig(
    branches: list[str],             # node names to run in parallel
    merge_strategy: MergeStrategy,   # how to merge their states
)
```

### JoinNodeConfig

Wait for multiple branches before proceeding.

```python
JoinNodeConfig(
    join_type: JoinType,   # And, Or, SelectFirst
)
```

**JoinType values:**
- `JoinType.And` — wait for ALL branches
- `JoinType.Or` — proceed when ANY branch finishes
- `JoinType.SelectFirst` — proceed with the first branch to complete

### SubgraphNodeConfig

Embed a compiled graph as a single node.

```python
SubgraphNodeConfig(graph: StateGraph)
```

### MergeStrategy

| Method | Description |
|--------|-------------|
| `MergeStrategy.overwrite()` | Last branch's value wins |
| `MergeStrategy.deep_merge()` | Deep-merge all branches' states |
| `MergeStrategy.append()` | Append list values from all branches |

---

## MCP Config

```python
from flowgentra_ai.types import MCPConfig
```

### Constructor

```python
MCPConfig(
    transport: str,          # "sse", "stdio", or "docker"
    url: str = "",           # For SSE transport
    command: str = "",       # For Stdio transport
    args: list[str] = [],    # For Stdio transport
)
```

---

## Visualization Config

```python
from flowgentra_ai.observability import VisualizationConfig

config = VisualizationConfig()
```

Used with `visualize_graph(graph, config)`.

---

## RAG Configuration Classes

Used when loading a RAG pipeline from a YAML config file.

```python
from flowgentra_ai.rag import (
    VectorStoreType,
    RAGConfig,
    VectorStoreConfig,
    EmbeddingsConfig,
    RetrievalSettings,
    PdfSettings,
    RAGGraphConfig,
)
```
