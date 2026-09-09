//! Compare a local config document against a live Qdrant collection.
//!
//! This closes a real gap: a restore of segments built against the wrong config is *accepted*
//! by the restore path (the fork's `check_snapshot_hash_ring_compatible` covers only the ring
//! scale) — and you find out later, when the config-mismatch optimizer rewrites every segment.
//! On a network-filesystem-backed 50 TB collection that is the difference between a
//! minutes-long restore and a multi-day one.
//!
//! Run this before copying artifacts anywhere.
//!
//! The comparison covers the same surface [`config::fingerprint`] hashes — including the
//! payload-index schema, which lives in `result.payload_schema` on the live side rather than
//! inside `result.config`, so both are fetched from the one `GET /collections/{name}` call.

use anyhow::{Context as _, Result, bail};
use collection::config::CollectionConfigInternal;
use serde_json::Value;

use crate::config::{self, LoadedConfig};
use crate::payload_index::PayloadIndexSchema;

/// Outcome of comparing a local config document with a live collection.
#[derive(Debug)]
pub struct VerifyReport {
    pub collection: String,
    pub local_fingerprint: String,
    pub remote_fingerprint: String,
    /// Human-readable field-level differences over the rebuild-triggering surface.
    pub differences: Vec<String>,
}

impl VerifyReport {
    pub fn matches(&self) -> bool {
        self.local_fingerprint == self.remote_fingerprint
    }
}

/// Fetch a collection's resolved config and payload schema, and compare them with the document.
pub async fn verify(
    url: &str,
    collection: &str,
    local: &LoadedConfig,
    local_payload_index: Option<&PayloadIndexSchema>,
) -> Result<VerifyReport> {
    let (remote_config, remote_payload_index) = fetch_collection(url, collection).await?;
    let remote_fingerprint = config::fingerprint(&remote_config, remote_payload_index.as_ref())?;

    let mut differences = diff_rebuild_surface(&local.config, &remote_config);
    differences.extend(diff_payload_index(
        local_payload_index,
        remote_payload_index.as_ref(),
    ));

    Ok(VerifyReport {
        collection: collection.to_string(),
        local_fingerprint: local.fingerprint.clone(),
        remote_fingerprint,
        differences,
    })
}

/// `GET /collections/{name}` and pull `result.config` and `result.payload_schema` out of the
/// envelope.
async fn fetch_collection(
    url: &str,
    collection: &str,
) -> Result<(CollectionConfigInternal, Option<PayloadIndexSchema>)> {
    let endpoint = format!("{}/collections/{collection}", url.trim_end_matches('/'));

    let client = reqwest::Client::builder()
        .build()
        .context("failed to build HTTP client")?;

    let response = client
        .get(&endpoint)
        .send()
        .await
        .with_context(|| format!("request to {endpoint} failed"))?;

    let status = response.status();
    let body: Value = response
        .json()
        .await
        .with_context(|| format!("{endpoint} did not return JSON"))?;

    if !status.is_success() {
        bail!("{endpoint} returned {status}: {body}");
    }

    let config = body
        .pointer("/result/config")
        .with_context(|| format!("{endpoint} response has no result.config field: {body}"))?;
    let config = serde_json::from_value(config.clone())
        .context("collection config from server does not parse as CollectionConfigInternal")?;

    let payload_index = body
        .pointer("/result/payload_schema")
        .map(payload_schema_from_info)
        .transpose()
        .context("payload_schema from server does not parse")?
        .flatten();

    Ok((config, payload_index))
}

/// Convert the API's `payload_schema` map back into the schema shape the fingerprint hashes.
///
/// The live side reports each index as a `PayloadIndexInfo` — `{data_type, params?, points}` —
/// which is exactly `PayloadFieldSchema` unfolded (`PayloadIndexInfo::new` is the other
/// direction): `params` present means the parameter form, absent means the bare-type shorthand.
/// `points` is runtime state, not schema, and is dropped.
///
/// An empty map becomes `None`, matching the local side where an absent `payload_index` section
/// means "no indexes declared" — the two must fingerprint identically.
fn payload_schema_from_info(schema: &Value) -> Result<Option<PayloadIndexSchema>> {
    use segment::types::PayloadFieldSchema;

    let map = schema
        .as_object()
        .context("payload_schema is not an object")?;
    if map.is_empty() {
        return Ok(None);
    }

    let mut fields = std::collections::BTreeMap::new();
    for (name, info) in map {
        let field: segment::json_path::JsonPath = name
            .parse()
            .map_err(|_| anyhow::anyhow!("payload field '{name}' is not a valid path"))?;

        let schema: PayloadFieldSchema = match info.get("params") {
            Some(params) if !params.is_null() => serde_json::from_value(params.clone())
                .with_context(|| format!("field '{name}': cannot parse index params"))?,
            _ => {
                let data_type = info
                    .get("data_type")
                    .with_context(|| format!("field '{name}' has no data_type"))?;
                serde_json::from_value(data_type.clone())
                    .with_context(|| format!("field '{name}': cannot parse data_type"))?
            }
        };
        fields.insert(field, schema);
    }

    Ok(Some(PayloadIndexSchema::from_fields(fields)))
}

/// Field-set diff over the declared payload indexes.
fn diff_payload_index(
    local: Option<&PayloadIndexSchema>,
    remote: Option<&PayloadIndexSchema>,
) -> Vec<String> {
    let mut out = Vec::new();

    // Canonical form: placement spellings resolved, everything else raw — matching both
    // the fingerprint and the server's own `schema_transition::classify`, which treats
    // `on_disk: true` and `memory: "cold"` as one identical placement. Comparing the raw
    // params here used to fail `verify-config` over pure notation.
    let names =
        |schema: Option<&PayloadIndexSchema>| -> std::collections::BTreeMap<String, String> {
            schema
                .and_then(|schema| schema.canonical_fields().ok())
                .map(|fields| {
                    fields
                        .into_iter()
                        .map(|(field, definition)| (field, definition.to_string()))
                        .collect()
                })
                .unwrap_or_default()
        };

    let local = names(local);
    let remote = names(remote);

    for field in local
        .keys()
        .chain(remote.keys())
        .collect::<std::collections::BTreeSet<_>>()
    {
        match (local.get(field), remote.get(field)) {
            (Some(l), Some(r)) if l != r => {
                out.push(format!("payload_index.{field}: local={l} remote={r}"));
            }
            (Some(_), None) => {
                out.push(format!(
                    "payload_index.{field}: local=present remote=missing"
                ));
            }
            (None, Some(_)) => {
                out.push(format!(
                    "payload_index.{field}: local=missing remote=present"
                ));
            }
            _ => {}
        }
    }

    out
}

/// The resolved placement of a dense vector's storage, `memory` over the deprecated `on_disk`.
fn vector_placement(
    params: &collection::operations::types::VectorParams,
) -> Option<segment::types::Memory> {
    // Exhaustive over the two placement spellings; `on_disk` is deprecated but still a valid
    // input on a live collection created before 1.19.
    #[allow(deprecated)]
    segment::types::Memory::resolve(
        params.memory,
        params.on_disk.map(segment::types::Memory::from_on_disk),
    )
}

/// Field-level diff over exactly what forces a segment rebuild, plus topology and budget.
///
/// The fingerprint comparison alone tells you *that* something differs; this tells you
/// which field, which is what an operator needs at 3am. Placements are compared *resolved*,
/// the way the fingerprint hashes them and the mismatch optimizer reads them, so a legacy
/// `on_disk: true` on the live side matches a local `memory: "cold"`.
fn diff_rebuild_surface(
    local: &CollectionConfigInternal,
    remote: &CollectionConfigInternal,
) -> Vec<String> {
    let mut out = Vec::new();

    let mut note = |field: &str, local: String, remote: String| {
        if local != remote {
            out.push(format!("{field}: local={local} remote={remote}"));
        }
    };

    note(
        "params.shard_number",
        local.params.shard_number.to_string(),
        remote.params.shard_number.to_string(),
    );
    note(
        "params.sharding_method",
        format!("{:?}", local.params.sharding_method.unwrap_or_default()),
        format!("{:?}", remote.params.sharding_method.unwrap_or_default()),
    );
    note(
        "params.hash_ring_shard_scale",
        local.params.hash_ring_shard_scale.to_string(),
        remote.params.hash_ring_shard_scale.to_string(),
    );
    note(
        "params.payload placement (resolved)",
        format!("{:?}", local.params.payload_storage_type()),
        format!("{:?}", remote.params.payload_storage_type()),
    );
    note(
        "optimizer_config.default_segment_number",
        local.optimizer_config.default_segment_number.to_string(),
        remote.optimizer_config.default_segment_number.to_string(),
    );
    note(
        "optimizer_config.max_segment_size",
        format!("{:?}", local.optimizer_config.max_segment_size),
        format!("{:?}", remote.optimizer_config.max_segment_size),
    );

    // Every field on HnswConfig::mismatch_requires_rebuild, individually, so the report
    // names the exact arm that will trigger the rebuild. max_indexing_threads is exempt
    // and deliberately absent; the placement is compared resolved, exactly as the mismatch
    // compares it.
    let (lh, rh) = (&local.hnsw_config, &remote.hnsw_config);
    note("hnsw_config.m", lh.m.to_string(), rh.m.to_string());
    note(
        "hnsw_config.ef_construct",
        lh.ef_construct.to_string(),
        rh.ef_construct.to_string(),
    );
    note(
        "hnsw_config.full_scan_threshold",
        lh.full_scan_threshold.to_string(),
        rh.full_scan_threshold.to_string(),
    );
    note(
        "hnsw_config.payload_m",
        format!("{:?}", lh.payload_m),
        format!("{:?}", rh.payload_m),
    );
    note(
        "hnsw_config placement (resolved)",
        format!("{:?}", lh.memory_placement()),
        format!("{:?}", rh.memory_placement()),
    );
    note(
        "hnsw_config.inline_storage",
        format!("{:?}", lh.inline_storage),
        format!("{:?}", rh.inline_storage),
    );

    // QuantizationConfig::mismatch_requires_rebuild is `self != other`, so compare whole.
    note(
        "quantization_config",
        format!("{:?}", local.quantization_config),
        format!("{:?}", remote.quantization_config),
    );

    // Per-vector params. Reported by name so a multi-vector collection is diagnosable.
    let local_vectors: std::collections::BTreeMap<_, _> =
        local.params.vectors.params_iter().collect();
    let remote_vectors: std::collections::BTreeMap<_, _> =
        remote.params.vectors.params_iter().collect();

    for name in local_vectors
        .keys()
        .chain(remote_vectors.keys())
        .collect::<std::collections::BTreeSet<_>>()
    {
        match (local_vectors.get(*name), remote_vectors.get(*name)) {
            (Some(l), Some(r)) => {
                note(
                    &format!("params.vectors.{name}.size"),
                    l.size.to_string(),
                    r.size.to_string(),
                );
                note(
                    &format!("params.vectors.{name}.distance"),
                    format!("{:?}", l.distance),
                    format!("{:?}", r.distance),
                );
                note(
                    &format!("params.vectors.{name} placement (resolved)"),
                    format!("{:?}", vector_placement(l)),
                    format!("{:?}", vector_placement(r)),
                );
                note(
                    &format!("params.vectors.{name}.datatype"),
                    format!("{:?}", l.datatype),
                    format!("{:?}", r.datatype),
                );
                note(
                    &format!("params.vectors.{name}.multivector_config"),
                    format!("{:?}", l.multivector_config),
                    format!("{:?}", r.multivector_config),
                );
                note(
                    &format!("params.vectors.{name}.hnsw_config"),
                    format!("{:?}", l.hnsw_config),
                    format!("{:?}", r.hnsw_config),
                );
                note(
                    &format!("params.vectors.{name}.quantization_config"),
                    format!("{:?}", l.quantization_config),
                    format!("{:?}", r.quantization_config),
                );
            }
            // Routed through `note` as well, so every line shares one format.
            (Some(_), None) => note(
                &format!("params.vectors.{name}"),
                "present".to_string(),
                "missing".to_string(),
            ),
            (None, Some(_)) => note(
                &format!("params.vectors.{name}"),
                "missing".to_string(),
                "present".to_string(),
            ),
            (None, None) => unreachable!("name came from one of the two maps"),
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::config;

    fn load(value: &serde_json::Value) -> LoadedConfig {
        config::from_str(&value.to_string()).unwrap()
    }

    #[test]
    fn identical_configs_have_no_differences() {
        let a = load(&config::tests::valid_config_json());
        let b = load(&config::tests::valid_config_json());

        assert_eq!(a.fingerprint, b.fingerprint);
        assert!(
            diff_rebuild_surface(&a.config, &b.config).is_empty(),
            "identical configs must produce no diff",
        );
    }

    /// The diff must name each rebuild-triggering field, not just report "something differs".
    #[test]
    fn diff_names_each_perturbed_field() {
        let base = load(&config::tests::valid_config_json());

        let cases: &[(&str, &str, &str, serde_json::Value)] = &[
            ("hnsw_config", "m", "hnsw_config.m", json!(32)),
            (
                "hnsw_config",
                "ef_construct",
                "hnsw_config.ef_construct",
                json!(256),
            ),
            (
                "hnsw_config",
                "full_scan_threshold",
                "hnsw_config.full_scan_threshold",
                json!(20000),
            ),
            (
                "hnsw_config",
                "payload_m",
                "hnsw_config.payload_m",
                json!(16),
            ),
            (
                "hnsw_config",
                "memory",
                "hnsw_config placement",
                json!("cold"),
            ),
            (
                "hnsw_config",
                "inline_storage",
                "hnsw_config.inline_storage",
                json!(true),
            ),
        ];

        for (section, field, expected, new_value) in cases {
            let mut value = config::tests::valid_config_json();
            value[*section][*field] = new_value.clone();
            let changed = load(&value);

            let differences = diff_rebuild_surface(&base.config, &changed.config);
            assert!(
                differences.iter().any(|d| d.starts_with(expected)),
                "diff should name {expected}, got: {differences:?}",
            );
        }
    }

    /// Two spellings of one placement must not diff — the live side may still speak `on_disk`.
    #[test]
    fn equivalent_placement_spellings_do_not_diff() {
        let modern = load(&config::tests::valid_config_json());

        let mut legacy = config::tests::valid_config_json();
        legacy["params"]["vectors"]["dense"]
            .as_object_mut()
            .unwrap()
            .shift_remove("memory");
        legacy["params"]["vectors"]["dense"]["on_disk"] = json!(true);
        let legacy = load(&legacy);

        let differences = diff_rebuild_surface(&modern.config, &legacy.config);
        assert!(
            differences.is_empty(),
            "resolved-equal placements must not diff: {differences:?}",
        );
    }

    #[test]
    fn diff_catches_topology_and_quantization() {
        let base = load(&config::tests::valid_config_json());

        let mut shards = config::tests::valid_config_json();
        shards["params"]["shard_number"] = json!(8);
        let differences = diff_rebuild_surface(&base.config, &load(&shards).config);
        assert!(
            differences
                .iter()
                .any(|d| d.starts_with("params.shard_number")),
            "got: {differences:?}",
        );

        let mut scale = config::tests::valid_config_json();
        scale["params"]["hash_ring_shard_scale"] = json!(500);
        let differences = diff_rebuild_surface(&base.config, &load(&scale).config);
        assert!(
            differences
                .iter()
                .any(|d| d.starts_with("params.hash_ring_shard_scale")),
            "got: {differences:?}",
        );

        let mut bits = config::tests::valid_config_json();
        bits["quantization_config"]["turbo"]["bits"] = json!("bits2");
        let differences = diff_rebuild_surface(&base.config, &load(&bits).config);
        assert!(
            differences
                .iter()
                .any(|d| d.starts_with("quantization_config")),
            "got: {differences:?}",
        );
    }

    #[test]
    fn diff_reports_vector_present_on_only_one_side() {
        let base = load(&config::tests::valid_config_json());

        let mut extra = config::tests::valid_config_json();
        extra["params"]["vectors"]["second"] = json!({
            "size": 128,
            "distance": "Dot",
            "memory": "cold",
            "datatype": null,
            "multivector_config": null,
        });
        let extra = load(&extra);

        let differences = diff_rebuild_surface(&base.config, &extra.config);
        assert!(
            differences
                .iter()
                .any(|d| d.contains("second") && d.contains("local=missing")),
            "got: {differences:?}",
        );
    }

    /// The API reports indexes as `PayloadIndexInfo`; the conversion back must reproduce the
    /// schema shape the fingerprint hashes, and an empty map must mean "no indexes".
    #[test]
    fn payload_schema_round_trips_from_the_api_shape() {
        let schema = payload_schema_from_info(&json!({
            "group": { "data_type": "keyword", "params": { "type": "keyword", "memory": "cold" }, "points": 42 },
            "bare": { "data_type": "integer", "points": 7 },
        }))
        .unwrap()
        .expect("two indexes");

        assert_eq!(schema.len(), 2);

        // The parameter form keeps its params; the bare form becomes the shorthand.
        let as_json = serde_json::to_value(&schema).unwrap();
        assert_eq!(as_json["schema"]["group"]["memory"], "cold");
        assert_eq!(as_json["schema"]["bare"], "integer");

        assert!(
            payload_schema_from_info(&json!({})).unwrap().is_none(),
            "an empty schema map must mean no indexes, matching an absent local section",
        );
    }

    /// The remote fingerprint must fold the payload schema in, or a live collection whose
    /// indexes differ from the document would still verify clean.
    #[test]
    fn remote_fingerprint_covers_the_payload_schema() {
        let base = load(&config::tests::valid_config_json());

        let remote_schema = payload_schema_from_info(&json!({
            "group": { "data_type": "keyword", "params": { "type": "keyword", "memory": "cold" }, "points": 0 },
        }))
        .unwrap();

        let bare = config::fingerprint(&base.config, None).unwrap();
        let with_schema = config::fingerprint(&base.config, remote_schema.as_ref()).unwrap();
        assert_ne!(
            bare, with_schema,
            "a remote index must move the remote fingerprint",
        );
    }

    #[test]
    fn diff_reports_payload_index_differences() {
        let schema = |body: serde_json::Value| -> PayloadIndexSchema {
            PayloadIndexSchema::from_fields(serde_json::from_value(body).unwrap())
        };

        let local = schema(json!({ "group": { "type": "keyword", "memory": "cold" } }));
        let remote = schema(json!({ "other": { "type": "integer", "memory": "cold" } }));

        let differences = diff_payload_index(Some(&local), Some(&remote));
        assert!(
            differences
                .iter()
                .any(|d| d.contains("group") && d.contains("remote=missing")),
            "got: {differences:?}",
        );
        assert!(
            differences
                .iter()
                .any(|d| d.contains("other") && d.contains("local=missing")),
            "got: {differences:?}",
        );

        assert!(diff_payload_index(Some(&local), Some(&local)).is_empty());
        assert!(diff_payload_index(None, None).is_empty());
    }

    /// Placement spellings compare resolved — a live collection whose indexes were created
    /// with the deprecated `on_disk` flag must verify clean against a document using
    /// `memory`, because the server itself treats the two as identical (no rebuild).
    #[test]
    fn diff_treats_equivalent_placement_spellings_as_equal() {
        let schema = |body: serde_json::Value| -> PayloadIndexSchema {
            PayloadIndexSchema::from_fields(serde_json::from_value(body).unwrap())
        };

        let memory = schema(json!({ "group": { "type": "keyword", "memory": "cold" } }));
        let legacy = schema(json!({ "group": { "type": "keyword", "on_disk": true } }));
        assert!(
            diff_payload_index(Some(&memory), Some(&legacy)).is_empty(),
            "on_disk: true and memory: cold are one placement in two spellings",
        );

        let bare = schema(json!({ "group": "keyword" }));
        let pinned = schema(json!({ "group": { "type": "keyword", "memory": "pinned" } }));
        assert!(diff_payload_index(Some(&bare), Some(&pinned)).is_empty());

        // A real placement change still diffs, and so does a structural change.
        assert!(!diff_payload_index(Some(&memory), Some(&pinned)).is_empty());
        let tenant =
            schema(json!({ "group": { "type": "keyword", "memory": "cold", "is_tenant": true } }));
        assert!(!diff_payload_index(Some(&memory), Some(&tenant)).is_empty());
    }
}
