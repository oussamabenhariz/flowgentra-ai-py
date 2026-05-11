# Validation System

**Ensure data integrity** throughout your agent workflows with Flowgentra's validation system. Validate inputs, state transitions, and outputs to catch errors early and maintain reliable execution.

## What is Validation?

Validation ensures data meets expected criteria:
- **Input validation**: Check user inputs before processing
- **State validation**: Verify state integrity between nodes
- **Output validation**: Confirm results meet requirements
- **Schema validation**: Enforce data structure contracts

## When to Use Validation

Use validation to:
- Prevent invalid data from breaking your agents
- Enforce business rules and constraints
- Provide clear error messages to users
- Maintain data consistency across workflows
- Debug issues by catching problems early

## Validation Types

### Schema Validation

Validate data structure against JSON schemas:

```python
from flowgentra_ai.validation import SchemaValidator
from pydantic import BaseModel

class UserQuery(BaseModel):
    question: str
    max_length: int = 1000
    category: Optional[str] = None

validator = SchemaValidator(UserQuery)

# Validate input
try:
    valid_data = validator.validate({"question": "What is AI?", "max_length": 500})
    print("Valid:", valid_data.question)
except ValidationError as e:
    print("Invalid:", e)
```

### Custom Validators

Create domain-specific validation logic:

```python
from flowgentra_ai.validation import Validator

class QuestionValidator(Validator):
    def validate(self, data):
        if not isinstance(data, dict):
            raise ValidationError("Input must be a dictionary")

        question = data.get("question", "")
        if not question.strip():
            raise ValidationError("Question cannot be empty")

        if len(question) > 2000:
            raise ValidationError("Question too long (max 2000 chars)")

        # Business rule: must contain question words
        question_words = ["what", "how", "why", "when", "where", "who"]
        if not any(word in question.lower() for word in question_words):
            raise ValidationError("Question must contain question words")

        return data
```

### State Validators

Validate state transitions between nodes:

```python
from flowgentra_ai.validation import StateValidator

class ConversationStateValidator(StateValidator):
    def validate_transition(self, old_state, new_state, node_name):
        # Ensure conversation flows logically
        if node_name == "answer" and not old_state.get("question"):
            raise ValidationError("Cannot answer without a question")

        # Check message limits
        messages = new_state.get("messages", [])
        if len(messages) > 100:
            raise ValidationError("Too many messages in conversation")

        return new_state
```

## Configuration

### In Python

> **Note:** The Python validation API (`ValidationChain`, `QuestionValidator`, etc.) is not yet
> available in the current release. Configure validation via the YAML config block below.

```yaml
# non-executable conceptual example — Python validation API not yet implemented
# from flowgentra_ai.validation import ValidationChain
# validators = ValidationChain()
# validators.add_input_validator(QuestionValidator())
```

### In Configuration

```yaml
validation:
  input_validators:
    - type: schema
      schema: UserQuery
      strict: true

    - type: custom
      class: "myapp.validators.QuestionValidator"

  state_validators:
    - type: custom
      class: "myapp.validators.ConversationStateValidator"

  output_validators:
    - type: schema
      schema: AgentResponse
```

## Built-in Validators

### Type Validators

```python
from flowgentra_ai.validation import TypeValidator

# Validate data types
string_validator = TypeValidator(str)
int_validator = TypeValidator(int, min_value=0, max_value=100)
list_validator = TypeValidator(list, max_length=10)
```

### Range Validators

```python
from flowgentra_ai.validation import RangeValidator

# Numeric ranges
score_validator = RangeValidator(0.0, 1.0)  # 0.0 to 1.0
count_validator = RangeValidator(1, 100)    # 1 to 100

# String lengths
name_validator = RangeValidator(min_length=1, max_length=50)
```

### Enum Validators

```python
from flowgentra_ai.validation import EnumValidator

category_validator = EnumValidator(["tech", "business", "science", "general"])
priority_validator = EnumValidator(["low", "medium", "high"], case_sensitive=False)
```

### Regex Validators

```python
from flowgentra_ai.validation import RegexValidator

email_validator = RegexValidator(r'^[^@]+@[^@]+\.[^@]+$')
phone_validator = RegexValidator(r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$')
```

## Advanced Validation Patterns

### Conditional Validation

```python
class ConditionalValidator(Validator):
    def validate(self, data):
        # Different rules based on input type
        if data.get("type") == "user_query":
            return self._validate_user_query(data)
        elif data.get("type") == "system_command":
            return self._validate_system_command(data)
        else:
            raise ValidationError("Unknown input type")

    def _validate_user_query(self, data):
        # User queries need questions
        if "question" not in data:
            raise ValidationError("User queries must have questions")
        return data

    def _validate_system_command(self, data):
        # System commands need admin privileges
        if not data.get("is_admin", False):
            raise ValidationError("System commands require admin privileges")
        return data
```

### Cross-Field Validation

```python
class CrossFieldValidator(Validator):
    def validate(self, data):
        start_date = data.get("start_date")
        end_date = data.get("end_date")

        if start_date and end_date:
            if start_date > end_date:
                raise ValidationError("Start date must be before end date")

        budget = data.get("budget", 0)
        items = data.get("items", [])

        total_cost = sum(item.get("cost", 0) for item in items)
        if total_cost > budget:
            raise ValidationError(f"Total cost ${total_cost} exceeds budget ${budget}")

        return data
```

### Async Validation

```python
class AsyncValidator(Validator):
    async def validate(self, data):
        # Check against external service
        user_id = data.get("user_id")
        if user_id:
            is_valid = await self._check_user_exists(user_id)
            if not is_valid:
                raise ValidationError(f"User {user_id} does not exist")

        return data

    async def _check_user_exists(self, user_id):
        # Call user service API
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.example.com/users/{user_id}") as resp:
                return resp.status == 200
```

## Validation in Agent Workflows

### Input Validation

```python
from flowgentra_ai.graph import StateGraph
from flowgentra_ai.validation import InputValidator

def validate_input_node(state):
    validator = InputValidator(QuestionValidator())
    try:
        validated_input = validator.validate(state["user_input"])
        state["validated_input"] = validated_input
        return state
    except ValidationError as e:
        state["error"] = str(e)
        return state

graph = StateGraph()
graph.add_node("validate", validate_input_node)
graph.set_entry_point("validate")
```

### State Validation Between Nodes

```python
from flowgentra_ai.validation import StateTransitionValidator

# Automatically validate state between all nodes
graph = StateGraph()
graph.add_state_validator(ConversationStateValidator())

# Or validate specific transitions
graph.add_transition_validator("ask_question", "provide_answer", AnswerValidator())
```

### Output Validation

```python
from flowgentra_ai.validation import OutputValidator

def validate_output_node(state):
    validator = OutputValidator(AnswerSchema())
    try:
        validated_answer = validator.validate(state["generated_answer"])
        state["final_answer"] = validated_answer
        return state
    except ValidationError as e:
        # Fallback to default answer
        state["final_answer"] = {"text": "I couldn't generate a valid answer", "confidence": 0.0}
        return state
```

## Error Handling

Validation errors provide detailed feedback:

```python
try:
    validator.validate(data)
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Field: {e.field}")
    print(f"Value: {e.value}")
    print(f"Expected: {e.expected}")

    # Structured error details
    for detail in e.details:
        print(f"- {detail.field}: {detail.message}")
```

## Performance Considerations

- **Cache validation results**: Avoid re-validating unchanged data
- **Validate early**: Catch errors before expensive operations
- **Use async validation**: For external API calls
- **Profile validators**: Monitor performance impact

## Testing Validation

```python
import pytest
from flowgentra_ai.validation import ValidationError

def test_question_validator():
    validator = QuestionValidator()

    # Valid input
    valid_data = {"question": "What is the capital of France?"}
    result = validator.validate(valid_data)
    assert result == valid_data

    # Invalid: empty question
    with pytest.raises(ValidationError, match="cannot be empty"):
        validator.validate({"question": ""})

    # Invalid: too long
    with pytest.raises(ValidationError, match="too long"):
        validator.validate({"question": "x" * 2001})

    # Invalid: no question words
    with pytest.raises(ValidationError, match="question words"):
        validator.validate({"question": "Paris is a city"})
```

## Best Practices

### Design Principles
- **Fail Fast**: Validate early, provide clear error messages
- **Single Responsibility**: One validator, one concern
- **Composable**: Combine simple validators into complex ones
- **Testable**: Easy to unit test validation logic

### Production Considerations
- **Graceful Degradation**: Continue with defaults on validation failure
- **Monitoring**: Track validation success/failure rates
- **Logging**: Log validation errors with context
- **User-Friendly**: Convert technical errors to user messages

### Common Patterns
- **Progressive Validation**: Basic checks first, detailed checks later
- **Conditional Validation**: Different rules based on context
- **Validation Chains**: Series of validators with early exit
- **Validation Reports**: Detailed feedback for debugging