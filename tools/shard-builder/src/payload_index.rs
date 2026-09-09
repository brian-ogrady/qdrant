//! Payload field indexes, built offline instead of at load.
//!
//! # Why build them here
//!
//! The alternative is to declare the indexes on the collection and let the shards build them when
//! they load. That works, but it happens on the serving filesystem, it blocks startup, and it costs
//! a full pass over every point in the shard. Measured at roughly 1 second per million points for a
//! low-cardinality keyword field on local NVMe — and that is the cheap case: a high-cardinality
//! float or datetime field costs more, and Lustre is slower than NVMe. At a billion points per
//! shard that is the difference between a cluster that is ready when it comes up and one that
//! spends an hour of every restart rebuilding indexes it could have shipped with.
//!
//! Building them here costs almost nothing by comparison, because the points are already in memory
//! on the way through.
//!
//! # How
//!
//! `FieldIndexOperations::CreateIndex` is an ordinary [`CollectionUpdateOperations`] variant, so it
//! goes through the same `EdgeShard::update` path as the points. `create_field_index`
//! (`lib/shard/src/update.rs`) applies it with `apply_segments`, which covers every segment in
//! the holder rather than just the appendable one.
//!
//! Indexes are created **after** `optimize()`, so they are built on the finished indexed segment
//! rather than on a staging segment the optimizer is about to replace.
//!
//! # Residency
//!
//! The placement (`memory`, or the deprecated `on_disk` it resolves against) is *optional*,
//! following the config gate's rule: required means "cannot be changed after the build", and a
//! payload field index can — it is dropped and recreated on a live collection without touching
//! the segments' point data. Left unstated it takes Qdrant's default, `pinned`
//! (`PayloadSchemaParams::memory_placement`), i.e. held in RAM — a real memory cost on a
//! billion-point shard, but a recoverable one.
//!
//! The schema is parsed as the `payload_index` section of the input document
//! ([`crate::document`]), so `build` and `assemble` cannot be handed different ones.
//!
//! [`CollectionUpdateOperations`]: shard::operations::CollectionUpdateOperations

use std::collections::BTreeMap;

use segment::json_path::JsonPath;
use segment::types::PayloadFieldSchema;
use serde::{Deserialize, Serialize};

/// The payload index schema for a collection: field name -> index definition.
///
/// Deliberately the same shape Qdrant's own `payload_index.json` uses, so `assemble` can write it
/// through unchanged and a reader can compare the two directly.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PayloadIndexSchema {
    pub schema: BTreeMap<JsonPath, PayloadFieldSchema>,
}

impl PayloadIndexSchema {
    pub fn len(&self) -> usize {
        self.schema.len()
    }

    /// Build from an already-parsed field map.
    pub fn from_fields(schema: BTreeMap<JsonPath, PayloadFieldSchema>) -> Self {
        Self { schema }
    }

    /// The indexed field names.
    pub fn fields(&self) -> impl Iterator<Item = &JsonPath> {
        self.schema.keys()
    }

    /// The schema with every placement spelling resolved, for hashing and diffing.
    ///
    /// `on_disk: true` and `memory: "cold"` (and `on_disk: false` / `memory: "pinned"` /
    /// leaving both unset) are different spellings of the same field-index placement, and
    /// the server's own `schema_transition::classify` treats them as `Identical` — no
    /// rebuild, no swap. So the fingerprint and `verify-config` must not distinguish them
    /// either: each definition is expanded (`FieldType` -> full params), the deprecated
    /// `on_disk` is dropped, and `memory` is set to the resolved placement. Everything
    /// else in the params stays raw — a tokenizer or `is_tenant` difference is a real
    /// rebuild and must keep moving the hash.
    pub fn canonical_fields(&self) -> anyhow::Result<BTreeMap<String, serde_json::Value>> {
        use anyhow::Context as _;

        self.schema
            .iter()
            .map(|(field, definition)| {
                let mut value = serde_json::to_value(definition.expand().as_ref())
                    .context("failed to serialize payload index definition")?;
                let object = value
                    .as_object_mut()
                    .context("expanded payload index params must be an object")?;
                object.remove("on_disk");
                object.insert(
                    "memory".to_string(),
                    serde_json::to_value(definition.memory_placement())
                        .context("failed to serialize resolved placement")?,
                );
                Ok((field.to_string(), value))
            })
            .collect()
    }

    /// The operations that create these indexes, in a stable order.
    ///
    /// Consumed by the build phase (stage 3 of the port), which applies them through
    /// `EdgeShard::update` after `optimize()`.
    #[allow(dead_code)]
    pub fn create_operations(&self) -> Vec<shard::operations::CollectionUpdateOperations> {
        use shard::operations::{CollectionUpdateOperations, CreateIndex, FieldIndexOperations};

        self.schema
            .iter()
            .map(|(field, definition)| {
                CollectionUpdateOperations::FieldIndexOperation(FieldIndexOperations::CreateIndex(
                    CreateIndex {
                        field_name: field.clone(),
                        field_schema: Some(definition.clone()),
                    },
                ))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a `payload_index` section the way the document does.
    fn parse(body: &str) -> serde_json::Result<PayloadIndexSchema> {
        let fields: BTreeMap<JsonPath, PayloadFieldSchema> = serde_json::from_str(body)?;
        Ok(PayloadIndexSchema::from_fields(fields))
    }

    #[test]
    fn parses_the_three_fineweb_indexes() {
        let schema = parse(
            r#"{
                "dump":           {"type": "keyword",  "memory": "cold"},
                "date":           {"type": "datetime", "memory": "cold"},
                "language_score": {"type": "float",    "memory": "cold"}
            }"#,
        )
        .unwrap();
        assert_eq!(schema.len(), 3);

        let operations = schema.create_operations();
        assert_eq!(operations.len(), 3, "one create per field");
    }

    /// Residency is optional — an index may leave it to Qdrant's default (`pinned`), because
    /// a field index can be dropped and recreated after the build. Both the parameter form
    /// without a placement and the bare-type shorthand are accepted.
    #[test]
    fn residency_is_optional() {
        let schema = parse(r#"{"dump": {"type": "keyword"}}"#).unwrap();
        assert_eq!(schema.len(), 1);

        let shorthand = parse(r#"{"dump": "keyword"}"#).unwrap();
        assert_eq!(shorthand.len(), 1);
    }

    /// ...and both placement spellings remain specifiable.
    #[test]
    fn accepts_both_placement_spellings() {
        let modern = parse(r#"{"dump": {"type": "keyword", "memory": "cold"}}"#).unwrap();
        assert_eq!(modern.len(), 1);

        let legacy = parse(r#"{"dump": {"type": "keyword", "on_disk": true}}"#).unwrap();
        assert_eq!(legacy.len(), 1);
    }

    /// Serialises as the shape Qdrant writes, so `assemble` can hand it straight over.
    #[test]
    fn round_trips_through_qdrants_own_format() {
        let schema = parse(r#"{"dump": {"type": "keyword", "memory": "cold"}}"#).unwrap();

        let written = serde_json::to_string(&schema).unwrap();
        assert!(
            written.starts_with("{\"schema\""),
            "wrapped as Qdrant writes it: {written}"
        );
        let read: PayloadIndexSchema = serde_json::from_str(&written).unwrap();
        assert_eq!(read, schema);
    }
}
