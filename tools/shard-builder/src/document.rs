//! The single input document every command reads.
//!
//! One file with three sections, rather than three files passed as three flags:
//!
//! ```json
//! {
//!   "collection":     { "params": ..., "hnsw_config": ..., "optimizer_config": ... },
//!   "mapping":        { "id_column": "id", "dense_vectors": ... },
//!   "payload_index":  { "dump": { "type": "keyword", "memory": "cold" } }
//! }
//! ```
//!
//! # Why one file
//!
//! The three parts are not independent. The mapping names the vectors the collection declares, and
//! the payload index names fields the mapping carries into the payload — so a mismatch between any
//! two of them is a configuration error, and keeping them together makes that checkable at load
//! rather than discoverable at build time. It also removes the way they could previously go wrong:
//! `build` and `assemble` each took `--payload-index` separately, so they could be given different
//! schemas hours apart.
//!
//! # Sections
//!
//! * **`collection`** is exactly Qdrant's own `config.json` — the same shape `result.config` from
//!   `GET /collections/{name}` returns, so it can be lifted from a live collection unchanged. This
//!   is what `verify-config` compares and what the fingerprint is computed over.
//! * **`mapping`** is required for Parquet input and unused for JSONL, whose records already match
//!   the point shape.
//! * **`payload_index`** is optional. Field name to index definition, flattened: the wrapping
//!   `{"schema": ...}` that Qdrant's on-disk file uses is added by `assemble`, not written here.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use segment::json_path::JsonPath;
use segment::types::PayloadFieldSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::config::LoadedConfig;
use crate::parquet_source::ParquetMapping;
use crate::payload_index::PayloadIndexSchema;

/// The parsed input document.
#[derive(Debug)]
pub struct Document {
    pub config: LoadedConfig,
    pub mapping: Option<ParquetMapping>,
    pub payload_index: Option<PayloadIndexSchema>,
}

/// Raw shape, before the sections are validated individually.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDocument {
    /// Left as `Value` so the collection section can be checked for explicitly-stated fields on the
    /// raw JSON, before serde fills in any defaults.
    collection: Value,
    #[serde(default)]
    mapping: Option<ParquetMapping>,
    #[serde(default)]
    payload_index: Option<BTreeMap<JsonPath, PayloadFieldSchema>>,
}

/// Read and validate the input document.
pub fn load(path: &Path) -> Result<Document> {
    let text =
        fs_err::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    from_str(&text).with_context(|| format!("invalid input document {}", path.display()))
}

pub fn from_str(text: &str) -> Result<Document> {
    let raw: RawDocument = serde_json::from_str(text).map_err(|err| {
        // The most likely mistake is passing a bare collection config, which is what this tool
        // used to take. Recognise it and say what to do rather than reporting a missing field.
        if let Ok(value) = serde_json::from_str::<Value>(text)
            && value.get("collection").is_none()
            && value.get("params").is_some()
        {
            return anyhow::anyhow!(
                "this looks like a bare collection config. The input is now a single document \
                 with the collection nested under a \"collection\" key, alongside \"mapping\" and \
                 an optional \"payload_index\":\n\n\
                 {{\n  \"collection\": {{ ... this file ... }},\n  \"mapping\": {{ ... }},\n  \
                 \"payload_index\": {{ ... }}\n}}",
            );
        }
        anyhow::anyhow!("{err}")
    })?;

    let mut config =
        crate::config::from_value(raw.collection).context("in the \"collection\" section")?;

    if let Some(mapping) = &raw.mapping {
        mapping.validate().context("in the \"mapping\" section")?;
    }

    let payload_index = match raw.payload_index {
        Some(schema) if schema.is_empty() => bail!(
            "the \"payload_index\" section is empty; omit it entirely rather than declaring no \
             fields",
        ),
        Some(schema) => Some(PayloadIndexSchema::from_fields(schema)),
        None => None,
    };

    // Both fingerprints cover fields that live outside the `collection` section, so they can
    // only be finalised here: the part fingerprint covers the mapping's `id_column` and
    // `id_format` (point identity, and therefore routing), and the full fingerprint covers
    // the payload-index schema (baked into every built segment). `config::from_value`
    // computed both without those sections; recompute now that all three are in hand.
    //
    // This is the invariant every consumer relies on: a `LoadedConfig` reached through a
    // `Document` always carries the fingerprints of the whole document, not just its config
    // half.
    config.part_fingerprint =
        crate::config::part_fingerprint(&config.config, raw.mapping.as_ref())?;
    config.fingerprint = crate::config::fingerprint(&config.config, payload_index.as_ref())?;

    let document = Document {
        config,
        mapping: raw.mapping,
        payload_index,
    };
    document.check_sections_agree()?;
    Ok(document)
}

impl Document {
    /// Cross-checks between sections, which is the reason for keeping them in one file.
    fn check_sections_agree(&self) -> Result<()> {
        let Some(mapping) = &self.mapping else {
            return Ok(());
        };

        // Every vector the collection declares must be produced by the mapping, and vice versa.
        // A name present on one side only means a shard that silently lacks a vector, or a column
        // read for nothing.
        let declared_dense: Vec<String> = self
            .config
            .config
            .params
            .vectors
            .params_iter()
            .map(|(name, _)| name.to_string())
            .collect();

        for name in &declared_dense {
            if !mapping.dense_vectors.contains_key(name) {
                bail!(
                    "the collection declares dense vector '{name}' but the mapping has no column \
                     for it; points would be built without it",
                );
            }
        }
        for name in mapping.dense_vectors.keys() {
            if !declared_dense.contains(name) {
                bail!(
                    "the mapping produces dense vector '{name}' but the collection does not \
                     declare it",
                );
            }
        }

        let declared_sparse: Vec<String> = self
            .config
            .config
            .params
            .sparse_vectors
            .iter()
            .flat_map(|map| map.keys().cloned())
            .collect();

        for name in &declared_sparse {
            if !mapping.sparse_vectors.contains_key(name) {
                bail!(
                    "the collection declares sparse vector '{name}' but the mapping has no column \
                     for it",
                );
            }
        }
        for name in mapping.sparse_vectors.keys() {
            if !declared_sparse.contains(name) {
                bail!(
                    "the mapping produces sparse vector '{name}' but the collection does not \
                     declare it",
                );
            }
        }

        // An indexed field that is never carried into the payload can never match anything.
        if let Some(schema) = &self.payload_index {
            for field in schema.fields() {
                let field = field.to_string();
                if !mapping.payload_columns.contains(&field) {
                    bail!(
                        "the payload index declares field '{field}' but the mapping does not carry \
                         it into the payload; the index would be built over nothing",
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A document mirroring the shipped example, small enough to mutate per test.
    fn document_json() -> Value {
        serde_json::json!({
            "collection": crate::config::tests::valid_config_json(),
            "mapping": {
                "id_column": "id",
                "id_format": "urn_uuid",
                "dense_vectors": { "dense": "dense_embedding" },
                "sparse_vectors": { "sparse": { "column": "sparse_embedding" } },
                "payload_columns": ["dump", "date", "language_score"],
            },
            "payload_index": {
                "dump": { "type": "keyword", "memory": "cold" },
                "date": { "type": "datetime", "memory": "cold" },
                "language_score": { "type": "float", "memory": "cold" },
            },
        })
    }

    #[test]
    fn parses_all_three_sections() {
        let document = from_str(&document_json().to_string()).unwrap();
        assert!(document.mapping.is_some());
        assert_eq!(document.payload_index.as_ref().unwrap().len(), 3);
        assert!(!document.config.fingerprint.is_empty());
    }

    /// The mapping and payload index are both optional; the collection is not.
    #[test]
    fn collection_alone_is_a_valid_document() {
        let value = serde_json::json!({ "collection": crate::config::tests::valid_config_json() });
        let document = from_str(&value.to_string()).unwrap();
        assert!(document.mapping.is_none());
        assert!(document.payload_index.is_none());
    }

    #[test]
    fn requires_the_collection_section() {
        let err = from_str("{}").unwrap_err();
        assert!(format!("{err:#}").contains("collection"), "{err:#}");
    }

    /// The old input was a bare collection config; that mistake must be named, not reported as a
    /// missing field.
    #[test]
    fn recognises_a_bare_collection_config() {
        let err = from_str(&crate::config::tests::valid_config_json().to_string()).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("bare collection config"), "{text}");
        assert!(text.contains("\"collection\""), "{text}");
    }

    #[test]
    fn rejects_unknown_sections() {
        let mut value = document_json();
        value["mappings"] = serde_json::json!({});
        let err = from_str(&value.to_string()).unwrap_err();
        assert!(format!("{err:#}").contains("mappings"), "{err:#}");
    }

    #[test]
    fn rejects_an_empty_payload_index_section() {
        let mut value = document_json();
        value["payload_index"] = serde_json::json!({});
        let err = from_str(&value.to_string()).unwrap_err();
        assert!(format!("{err:#}").contains("omit it"), "{err:#}");
    }

    /// The cross-checks are the point of one file: catch a mismatch here, not at build time.
    #[test]
    fn rejects_a_vector_the_mapping_does_not_produce() {
        let mut value = document_json();
        value["mapping"]["dense_vectors"] = serde_json::json!({ "wrong_name": "dense_embedding" });
        let err = from_str(&value.to_string()).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("dense vector 'dense'"), "{text}");
    }

    #[test]
    fn rejects_a_mapping_vector_the_collection_does_not_declare() {
        let mut value = document_json();
        value["mapping"]["dense_vectors"]["extra"] = serde_json::json!("other_column");
        let err = from_str(&value.to_string()).unwrap_err();
        assert!(format!("{err:#}").contains("'extra'"), "{err:#}");
    }

    #[test]
    fn rejects_a_sparse_vector_the_mapping_does_not_produce() {
        let mut value = document_json();
        value["mapping"]["sparse_vectors"] = serde_json::json!({});
        let err = from_str(&value.to_string()).unwrap_err();
        assert!(
            format!("{err:#}").contains("sparse vector 'sparse'"),
            "{err:#}"
        );
    }

    /// An index over a field the mapping never stores can never match anything.
    #[test]
    fn rejects_an_index_on_a_field_that_is_not_in_the_payload() {
        let mut value = document_json();
        value["payload_index"]["url"] = serde_json::json!({ "type": "keyword", "memory": "cold" });
        let err = from_str(&value.to_string()).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("'url'"), "{text}");
        assert!(text.contains("index would be built over nothing"), "{text}");
    }

    /// The payload index is frozen at plan, not at scatter: editing it changes the full
    /// fingerprint (invalidating plans), but never the part fingerprint (part files stay
    /// usable — the schema does not influence a byte of them).
    #[test]
    fn payload_index_moves_the_fingerprint_but_not_the_part_fingerprint() {
        let base = from_str(&document_json().to_string()).unwrap().config;

        let mut changed = document_json();
        changed["payload_index"]["dump"] =
            serde_json::json!({ "type": "keyword", "memory": "pinned" });
        let changed = from_str(&changed.to_string()).unwrap().config;

        assert_ne!(
            base.fingerprint, changed.fingerprint,
            "an edited payload index must invalidate existing plans",
        );
        assert_eq!(
            base.part_fingerprint, changed.part_fingerprint,
            "an edited payload index must NOT invalidate a scatter",
        );

        let mut dropped = document_json();
        dropped
            .as_object_mut()
            .unwrap()
            .shift_remove("payload_index");
        let dropped = from_str(&dropped.to_string()).unwrap().config;
        assert_ne!(
            base.fingerprint, dropped.fingerprint,
            "declaring no indexes is a different artifact than declaring three",
        );
    }

    /// Residency is optional (a field index can be dropped and recreated after the build),
    /// so an index definition without a placement — even the bare-type shorthand — is fine.
    #[test]
    fn payload_index_residency_is_optional() {
        let mut value = document_json();
        value["payload_index"]["dump"] = serde_json::json!({ "type": "keyword" });
        from_str(&value.to_string()).expect("placement-less index definition must be accepted");

        let mut value = document_json();
        value["payload_index"]["dump"] = serde_json::json!("keyword");
        from_str(&value.to_string()).expect("shorthand index definition must be accepted");
    }
}
