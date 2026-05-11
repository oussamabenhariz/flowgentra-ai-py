//! Channel types — the backbone of LangGraph-style state management.
//!
//! Each state field maps to a Channel, which holds the current value and
//! knows how to merge incoming updates via its reducer strategy.
//!
//! # Channel Types
//! - `LastValue` — replaces the current value (default, like a regular variable)
//! - `Topic`     — appends incoming items to a list (like `operator.add` on lists)
//! - `BinaryOperator` — merges using a custom Rust closure

use serde_json::Value;
use std::sync::Arc;

// ── ChannelType ───────────────────────────────────────────────────────────────

/// Reducer strategy for a channel field.
#[derive(Clone)]
pub enum ChannelType {
    /// Overwrite: the latest value wins. (default)
    LastValue,

    /// Accumulate: append new items to the existing array.
    Topic,

    /// Custom: merge via an arbitrary binary function `f(current, new) -> merged`.
    BinaryOperator(Arc<dyn Fn(Value, Value) -> Value + Send + Sync>),
}

impl std::fmt::Debug for ChannelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelType::LastValue => write!(f, "LastValue"),
            ChannelType::Topic => write!(f, "Topic"),
            ChannelType::BinaryOperator(_) => write!(f, "BinaryOperator(<fn>)"),
        }
    }
}

// ── FieldSchema ───────────────────────────────────────────────────────────────

/// Descriptor for a single field in a state schema.
///
/// Describes the field's name, how it merges updates (channel type), and
/// the value to use when the field has not been set yet.
#[derive(Clone, Debug)]
pub struct FieldSchema {
    /// Field name (must match the key used in state dicts).
    pub name: String,
    /// Reducer strategy.
    pub channel_type: ChannelType,
    /// Initial / default value when the field is absent.
    pub default: Value,
}

impl FieldSchema {
    /// A replace-on-write field (the default).
    pub fn last_value(name: impl Into<String>) -> Self {
        FieldSchema {
            name: name.into(),
            channel_type: ChannelType::LastValue,
            default: Value::Null,
        }
    }

    /// An accumulating list field.
    pub fn topic(name: impl Into<String>) -> Self {
        FieldSchema {
            name: name.into(),
            channel_type: ChannelType::Topic,
            default: Value::Array(vec![]),
        }
    }

    /// A field merged by a custom binary function.
    pub fn binary_operator(
        name: impl Into<String>,
        f: impl Fn(Value, Value) -> Value + Send + Sync + 'static,
    ) -> Self {
        FieldSchema {
            name: name.into(),
            channel_type: ChannelType::BinaryOperator(Arc::new(f)),
            default: Value::Null,
        }
    }

    /// Override the default value.
    pub fn with_default(mut self, default: Value) -> Self {
        self.default = default;
        self
    }
}

// ── Channel ───────────────────────────────────────────────────────────────────

/// A runtime channel — holds one field's current value and applies its reducer.
#[derive(Clone)]
pub struct Channel {
    /// The current stored value.
    pub value: Value,
    /// How incoming updates are merged.
    pub channel_type: ChannelType,
}

impl std::fmt::Debug for Channel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Channel")
            .field("value", &self.value)
            .field("channel_type", &self.channel_type)
            .finish()
    }
}

impl Channel {
    /// Construct a channel from its schema descriptor.
    pub fn from_schema(schema: &FieldSchema) -> Self {
        Channel {
            value: schema.default.clone(),
            channel_type: schema.channel_type.clone(),
        }
    }

    /// Construct a plain `LastValue` channel with an initial value.
    pub fn last_value(value: Value) -> Self {
        Channel {
            value,
            channel_type: ChannelType::LastValue,
        }
    }

    /// Apply an incoming update value using this channel's reducer strategy.
    /// Mutates `self.value` in place.
    pub fn apply(&mut self, new_val: Value) {
        self.value = apply_channel_reducer(
            std::mem::replace(&mut self.value, Value::Null),
            new_val,
            &self.channel_type,
        );
    }
}

// ── Standalone reducer function ───────────────────────────────────────────────

/// Apply a reducer strategy to merge `current` with `new_val`.
///
/// Used both by `Channel::apply` and directly in the graph merge step.
pub fn apply_channel_reducer(current: Value, new_val: Value, channel_type: &ChannelType) -> Value {
    match channel_type {
        ChannelType::LastValue => new_val,

        ChannelType::Topic => {
            // Both sides are coerced to arrays before merging.
            let mut base = match current {
                Value::Array(arr) => arr,
                Value::Null => vec![],
                other => vec![other],
            };
            match new_val {
                Value::Array(inc) => base.extend(inc),
                Value::Null => {} // nothing to add
                other => base.push(other),
            }
            Value::Array(base)
        }

        ChannelType::BinaryOperator(f) => f(current, new_val),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn last_value_replaces() {
        let mut ch = Channel::last_value(json!(1));
        ch.apply(json!(42));
        assert_eq!(ch.value, json!(42));
    }

    #[test]
    fn topic_appends_arrays() {
        let mut ch = Channel::from_schema(&FieldSchema::topic("msgs"));
        ch.apply(json!(["a", "b"]));
        ch.apply(json!(["c"]));
        assert_eq!(ch.value, json!(["a", "b", "c"]));
    }

    #[test]
    fn topic_coerces_scalar_to_array() {
        let mut ch = Channel::from_schema(&FieldSchema::topic("items"));
        ch.apply(json!("hello"));
        ch.apply(json!("world"));
        assert_eq!(ch.value, json!(["hello", "world"]));
    }

    #[test]
    fn binary_operator_sums() {
        let mut ch = Channel::from_schema(&FieldSchema::binary_operator("count", |a, b| {
            json!(a.as_i64().unwrap_or(0) + b.as_i64().unwrap_or(0))
        }));
        ch.value = json!(10);
        ch.apply(json!(5));
        assert_eq!(ch.value, json!(15));
    }
}
