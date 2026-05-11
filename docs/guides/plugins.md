# Plugins System

**Extend Flowgentra with custom functionality** through the plugin architecture. Plugins allow you to hook into the agent execution lifecycle, add custom tools, modify behavior, and integrate with external systems.

## What are Plugins?

Plugins are modular extensions that can:
- Hook into agent lifecycle events (start, end, errors)
- Register custom tools and handlers
- Modify state during execution
- Add custom logging and monitoring
- Integrate with external services

Unlike built-in features, plugins are loaded at runtime and don't require code changes to the core library.

## When to Use Plugins

Use plugins when you need to:
- Add custom monitoring or logging
- Integrate with proprietary systems
- Create reusable extensions across projects
- Modify agent behavior without forking the codebase
- Add domain-specific functionality

## Plugin Architecture

### Plugin Trait

All plugins implement the `Plugin` trait:

```rust
use flowgentra_ai::plugins::Plugin;
use async_trait::async_trait;

struct MyPlugin;

#[async_trait]
impl Plugin for MyPlugin {
    fn name(&self) -> &str {
        "my-custom-plugin"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    async fn initialize(&self, context: &mut PluginContext) -> Result<()> {
        println!("Plugin initialized!");
        Ok(())
    }

    async fn on_handler_start(
        &self,
        _context: &PluginContext,
        handler_name: &str,
    ) -> Result<()> {
        println!("Handler starting: {}", handler_name);
        Ok(())
    }
}
```

### Plugin Context

Plugins receive a `PluginContext` that provides access to:
- Current agent state
- Execution context
- Configuration
- Tool registry

## Plugin Lifecycle

Plugins can hook into these lifecycle events:

- `initialize()` - Called when plugin is loaded
- `on_agent_start()` - Called before agent execution
- `on_handler_start()` - Called before each node execution
- `on_handler_end()` - Called after each node execution
- `on_agent_end()` - Called after agent execution
- `on_error()` - Called when errors occur

## Registering Plugins

### In Rust

```rust
use flowgentra_ai::plugins::PluginRegistry;

let mut registry = PluginRegistry::new();
registry.register(Box::new(MyPlugin))?;
```

### In Configuration

```yaml
plugins:
  - name: "logging-plugin"
    path: "./plugins/logging.so"
    config:
      log_level: "debug"

  - name: "metrics-plugin"
    path: "./plugins/metrics.so"
    config:
      endpoint: "http://metrics.example.com"
```

## Example: Logging Plugin

```rust
use flowgentra_ai::plugins::{Plugin, PluginContext};
use async_trait::async_trait;
use std::time::Instant;

pub struct LoggingPlugin {
    start_time: Option<Instant>,
}

impl LoggingPlugin {
    pub fn new() -> Self {
        LoggingPlugin { start_time: None }
    }
}

#[async_trait]
impl Plugin for LoggingPlugin {
    fn name(&self) -> &str { "logging" }
    fn version(&self) -> &str { "1.0.0" }

    async fn on_agent_start(&self, context: &PluginContext) -> Result<()> {
        self.start_time = Some(Instant::now());
        println!("Agent execution started");
        Ok(())
    }

    async fn on_handler_end(
        &self,
        context: &PluginContext,
        handler_name: &str,
        duration: Duration,
    ) -> Result<()> {
        println!("Handler '{}' completed in {:?}", handler_name, duration);
        Ok(())
    }

    async fn on_agent_end(&self, context: &PluginContext) -> Result<()> {
        if let Some(start) = self.start_time {
            let total_duration = start.elapsed();
            println!("Agent execution completed in {:?}", total_duration);
        }
        Ok(())
    }
}
```

## Example: Custom Tool Plugin

```rust
use flowgentra_ai::plugins::{Plugin, PluginContext};
use flowgentra_ai::tools::{Tool, ToolRegistry};
use async_trait::async_trait;

pub struct WeatherPlugin;

#[async_trait]
impl Plugin for WeatherPlugin {
    fn name(&self) -> &str { "weather-tools" }
    fn version(&self) -> &str { "1.0.0" }

    async fn initialize(&self, context: &mut PluginContext) -> Result<()> {
        // Register custom weather tool
        context.tool_registry.register(
            "get_weather",
            WeatherTool::new(),
            "Get current weather for a location",
        )?;
        Ok(())
    }
}

struct WeatherTool;

impl Tool for WeatherTool {
    fn name(&self) -> &str { "get_weather" }

    fn description(&self) -> &str {
        "Get current weather information for a city"
    }

    fn schema(&self) -> ToolSpec {
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value> {
        let city = input["city"].as_str().unwrap();
        // Call weather API...
        Ok(json!({"temperature": 72, "condition": "sunny"}))
    }
}
```

## Plugin Development Best Practices

### Error Handling
- Plugins should not crash the agent execution
- Use proper error types and logging
- Handle network timeouts gracefully

### Performance
- Keep plugin operations fast
- Avoid blocking operations in lifecycle hooks
- Use async operations for I/O

### Security
- Validate all inputs
- Don't store sensitive data in plugin state
- Use secure communication channels

### Testing
- Test plugins in isolation
- Mock external dependencies
- Test error conditions

## Built-in Plugins

Flowgentra includes several built-in plugins:

- **Metrics Plugin**: Collects execution metrics
- **Tracing Plugin**: Distributed tracing integration
- **Audit Plugin**: Security and compliance logging
- **Cache Plugin**: Response caching

## Loading Plugins

### At Runtime

```rust
// Load plugin from file
let plugin = PluginRegistry::load_from_file("./plugins/my-plugin.so")?;

// Or create and register directly
registry.register(Box::new(MyPlugin::new()))?;
```

### Configuration-Based Loading

```yaml
plugins:
  enabled:
    - logging
    - metrics

  config:
    logging:
      level: "info"
      format: "json"

    metrics:
      endpoint: "http://localhost:9090"
```

## Troubleshooting

**Plugin fails to load**
- Check file permissions
- Verify plugin implements required trait methods
- Check for missing dependencies

**Plugin hooks not called**
- Ensure plugin is properly registered
- Check agent configuration includes plugins
- Verify hook method signatures

**Performance impact**
- Profile plugin execution time
- Move expensive operations to background tasks
- Consider lazy initialization