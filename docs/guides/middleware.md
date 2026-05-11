# Middleware System

**Add cross-cutting concerns** to your agents with middleware. Middleware provides hooks into the request/response lifecycle for logging, metrics, validation, caching, and more.

## What is Middleware?

Middleware intercepts agent execution at key points:
- Before/after agent execution
- Before/after node execution
- On errors and retries
- State transitions

Unlike plugins (which extend functionality), middleware modifies existing behavior.

## When to Use Middleware

Use middleware for:
- **Logging**: Request/response logging, audit trails
- **Metrics**: Performance monitoring, error tracking
- **Validation**: Input validation, schema checking
- **Caching**: Response caching, deduplication
- **Security**: Authentication, authorization, rate limiting
- **Debugging**: Tracing, profiling, diagnostics

## Middleware Types

### Request Interceptors

Intercept and modify requests before processing:

```python
from flowgentra_ai.middleware import RequestInterceptor

class LoggingMiddleware(RequestInterceptor):
    async def intercept_request(self, request, context):
        print(f"Processing request: {request}")
        # Modify request if needed
        request.metadata["start_time"] = time.time()
        return request
```

### Response Interceptors

Modify responses after processing:

```python
from flowgentra_ai.middleware import ResponseInterceptor

class CachingMiddleware(ResponseInterceptor):
    def __init__(self):
        self.cache = {}

    async def intercept_response(self, response, context):
        # Cache successful responses
        if response.success:
            key = self._get_cache_key(context.request)
            self.cache[key] = response

        return response
```

### Error Interceptors

Handle and potentially recover from errors:

```python
from flowgentra_ai.middleware import ErrorInterceptor

class RetryMiddleware(ErrorInterceptor):
    async def intercept_error(self, error, context):
        if isinstance(error, TemporaryError) and context.retry_count < 3:
            # Retry with exponential backoff
            await asyncio.sleep(2 ** context.retry_count)
            return RetryAction.RETRY

        return RetryAction.FAIL
```

## Middleware Chain

Middleware executes in order:

```
Request → [Middleware 1] → [Middleware 2] → [Handler] → [Middleware 2] → [Middleware 1] → Response
```

## Configuration

### In Python

> **Note:** The Python middleware API (`MiddlewareChain`, `LoggingMiddleware`, etc.) is not yet
> available in the current release. Configure middleware via the YAML config block below.

```yaml
# non-executable conceptual example — Python middleware API not yet implemented
# from flowgentra_ai.middleware import MiddlewareChain
# middleware = MiddlewareChain()
# middleware.add(LoggingMiddleware())
# agent = ZeroShotReAct(name="...", llm=..., ...)  # no .with_middleware() yet
```

### In Configuration

```yaml
middleware:
  - name: logging
    type: request_interceptor
    config:
      level: info
      format: json

  - name: rate_limiting
    type: request_interceptor
    config:
      requests_per_minute: 60

  - name: caching
    type: response_interceptor
    config:
      ttl_seconds: 300
```

## Built-in Middleware

### Logging Middleware

```python
from flowgentra_ai.middleware import LoggingMiddleware

middleware = LoggingMiddleware(
    level="INFO",
    include_state=True,  # Log state contents
    include_timing=True  # Log execution times
)
```

### Metrics Middleware

```python
from flowgentra_ai.middleware import MetricsMiddleware

middleware = MetricsMiddleware(
    collector="prometheus",  # or "statsd", "datadog"
    endpoint="http://localhost:9090",
    include_node_metrics=True,
    include_error_metrics=True
)
```

### Validation Middleware

```python
from flowgentra_ai.middleware import ValidationMiddleware

middleware = ValidationMiddleware(
    schemas={
        "user_query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1000
        }
    }
)
```

### Caching Middleware

```python
from flowgentra_ai.middleware import CachingMiddleware

middleware = CachingMiddleware(
    backend="redis",  # or "memory", "disk"
    ttl_seconds=300,
    key_function=lambda request: hash(request.input)
)
```

## Custom Middleware Examples

### Authentication Middleware

```python
class AuthMiddleware(RequestInterceptor):
    def __init__(self, api_key):
        self.api_key = api_key

    async def intercept_request(self, request, context):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not self._validate_token(auth_header):
            raise AuthenticationError("Invalid API key")

        # Add user info to context
        request.user = self._get_user_from_token(auth_header)
        return request

    def _validate_token(self, token):
        return token == f"Bearer {self.api_key}"
```

### Rate Limiting Middleware

```python
from collections import defaultdict
import time

class RateLimitMiddleware(RequestInterceptor):
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    async def intercept_request(self, request, context):
        user_id = request.user.id
        now = time.time()

        # Clean old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if now - req_time < 60
        ]

        if len(self.requests[user_id]) >= self.requests_per_minute:
            raise RateLimitError("Too many requests")

        self.requests[user_id].append(now)
        return request
```

### Tracing Middleware

```python
import uuid

class TracingMiddleware(RequestInterceptor, ResponseInterceptor):
    async def intercept_request(self, request, context):
        request.trace_id = str(uuid.uuid4())
        print(f"Starting trace {request.trace_id}")
        return request

    async def intercept_response(self, response, context):
        trace_id = context.request.trace_id
        duration = time.time() - context.start_time
        print(f"Completed trace {trace_id} in {duration:.2f}s")
        return response
```

## Middleware Order Matters

Order middleware by dependency:

```python
# Correct order
middleware.add(AuthenticationMiddleware())  # Must come first
middleware.add(RateLimitMiddleware())       # Needs user context
middleware.add(LoggingMiddleware())         # Logs everything
middleware.add(CachingMiddleware())         # Caches final responses
```

## Error Handling in Middleware

Middleware can:
- **Catch errors**: Handle and potentially recover
- **Transform errors**: Convert to user-friendly messages
- **Retry operations**: Implement custom retry logic
- **Log errors**: Add context for debugging

```python
class ErrorHandlingMiddleware(ErrorInterceptor):
    async def intercept_error(self, error, context):
        # Log error with context
        logger.error(f"Error in {context.node_name}: {error}", extra={
            "trace_id": context.trace_id,
            "user_id": context.user_id,
            "input": context.input
        })

        # Convert technical errors to user-friendly messages
        if isinstance(error, DatabaseError):
            return UserFriendlyError("Service temporarily unavailable")

        # Don't retry certain errors
        if isinstance(error, AuthenticationError):
            return ErrorAction.FAIL

        return ErrorAction.RETRY
```

## Performance Considerations

- **Keep middleware fast**: Avoid expensive operations
- **Use async operations**: Don't block the event loop
- **Cache expensive lookups**: Database queries, API calls
- **Profile regularly**: Monitor middleware performance impact

## Testing Middleware

```python
import pytest
from unittest.mock import Mock

def test_auth_middleware():
    middleware = AuthMiddleware("test-key")
    request = Mock()
    request.headers = {"Authorization": "Bearer test-key"}

    result = await middleware.intercept_request(request, Mock())

    assert result.user is not None

def test_rate_limit_middleware():
    middleware = RateLimitMiddleware(requests_per_minute=2)

    # First request should succeed
    request1 = Mock(user=Mock(id="user1"))
    result1 = await middleware.intercept_request(request1, Mock())
    assert result1 is request1

    # Third request should fail
    with pytest.raises(RateLimitError):
        await middleware.intercept_request(request1, Mock())
```

## Best Practices

### Design Principles
- **Single Responsibility**: One middleware, one concern
- **Composability**: Middleware should work together
- **Testability**: Easy to unit test in isolation
- **Configurability**: External configuration over hardcoding

### Production Considerations
- **Observability**: Monitor middleware performance
- **Graceful Degradation**: Continue working if middleware fails
- **Security**: Validate all inputs and outputs
- **Documentation**: Document middleware behavior and configuration

### Common Patterns
- **Circuit Breaker**: Stop calling failing services
- **Timeout**: Prevent hanging operations
- **Fallback**: Provide default responses on failure
- **Monitoring**: Track success rates and latencies