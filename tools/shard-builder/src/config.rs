//! Config contract for the offline shard builder.
//!
//! The builder takes a single input document: a resolved [`CollectionConfigInternal`]
//! (the same shape Qdrant persists as `config.json`, and the same shape returned by
//! `GET /collections/{name}`). Working from the *resolved* config rather than a
//! create-collection request is deliberate — a create request is full of `Option`s and
//! diffs whose defaults are applied server-side, which is exactly the ambiguity we need
//! to eliminate.
//!
//! Two guarantees this module provides:
//!
//! 1. **Nothing that cannot be changed after the build is defaulted.** The rule for what
//!    [`require_explicit_fields`] demands: a field is required exactly when it is frozen into
//!    the artifact — routing, vector shape, index structure — or when its default is derived
//!    from the build host's core count, so omitting it means one thing on the build box and
//!    another on each serving node. Everything that can be adjusted on the collection *after*
//!    the build (placements, runtime optimizer knobs) may be omitted and takes Qdrant's own
//!    default — though every field remains specifiable. The gate inspects the *raw* JSON,
//!    because after `serde` has applied defaults an absent field is indistinguishable from an
//!    explicitly-set one.
//!
//! 2. **Two fingerprints, over two different surfaces.** They exist because a config change
//!    invalidates two very differently-priced things.
//!
//!    [`fingerprint`] hashes the fields that `ConfigMismatchOptimizer::has_config_mismatch`
//!    compares (`lib/shard/src/optimizers/config_mismatch_optimizer.rs:68`) plus shard topology,
//!    the segment-size budget, and the document's payload-index schema (which the build phase
//!    bakes into every segment). It is recorded in every plan and compared by `build`, which
//!    freezes the config from `plan` onward: ten nodes building one collection all check against
//!    the same plan, so a config edited mid-build fails loudly instead of producing a shard of
//!    mismatched segments.
//!
//!    [`part_fingerprint`] hashes **routing alone** — the ring that chose the shard
//!    (`shard_number`, `sharding_method`, `hash_ring_shard_scale`) and the point ids fed into it
//!    (`id_column`, `id_format`). Nothing else can make a part's records wrong rather than merely
//!    a superset of what a build wants. It is stamped into part files and the scatter state.
//!
//!    Everything else a part carries — which dense and sparse vectors, at what width and
//!    dimensionality, from which column, and which payload columns — lives in
//!    `crate::partfile::PartManifest` (stage 2) and is checked as a *subset* at build time. That
//!    is what lets a build drop a vector or a payload column, and retune `hnsw_config`,
//!    `quantization_config` or `max_segment_size`, without re-scattering tens of terabytes.
//!
//!    The rule the two encode: *everything is frozen once a plan is generated, but a scatter can
//!    be reused as long as routing is unchanged and the build asks for no more than was captured.*
//!
//! # Fork adaptations
//!
//! This port targets a Qdrant fork with three additions the reference tool did not know about,
//! and each one lands here:
//!
//! * **`params.hash_ring_shard_scale`** — virtual nodes per shard on the hash ring. Routing now
//!   depends on it exactly as it depends on `shard_number`, so it is required explicitly and
//!   hashed into *both* fingerprints.
//! * **`memory` placement parameters** — every `on_disk` flag is deprecated in favour of a
//!   three-state `memory` placement (`cold`/`cached`/`pinned`), and the mismatch optimizer
//!   compares the *resolved* placement (`memory` overriding `on_disk`). Most placements are
//!   *optional* under the required-only-if-frozen rule: the HNSW graph, dense vector storage
//!   and payload storage placements are load-time metadata over identical bytes, adjustable
//!   after the build. The one placement that stays required is the sparse index's, because
//!   pinned builds an `ImmutableRam` index and cold/cached build an `Mmap` index — different
//!   structures, not relabelable metadata. The fingerprint hashes *resolved* placements
//!   (stated or defaulted), because `verify-config` fingerprints remote configs fetched from
//!   live collections, and those may carry the legacy spellings.
//! * **`wand_pruning`** on sparse indexes — baked into the built segment's sparse index config
//!   (`CollectionParams::to_sparse_vector_data`), so it must be stated rather than inherited.
//!
//! # Layout
//!
//! Three sections, in pipeline order: **loading** (the entry points and the [`LoadedConfig`]
//! they produce), **the explicit-fields gate** (what a document must state, and the machinery
//! that checks the raw JSON), and **the fingerprints** (the two hashes, each with its struct,
//! version and producing function together).

use std::collections::BTreeMap;

use anyhow::{Context as _, Result, bail};
use collection::config::{CollectionConfigInternal, ShardingMethod};
use collection::operations::types::Datatype;
use collection::operations::validation::label_errors;
use collection::optimizers_builder::build_segment_optimizer_config;
use common::types::PointOffsetType;
use segment::types::HnswConfig;
use serde::{Deserialize as _, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use validator::Validate as _;

use crate::payload_index::PayloadIndexSchema;

// ============================================================================
// Loading
// ============================================================================
// The entry points every command goes through, and the validated result they
// produce. `from_value` runs the three gates in order: explicit fields,
// Qdrant's own `#[validate]` rules, then the structural hard limits.

/// A config document that has passed [`require_explicit_fields`].
#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub config: CollectionConfigInternal,
    /// Hex sha256 over the rebuild-triggering surface. See [`fingerprint`].
    ///
    /// Recorded in every plan and compared by `build`, so the config is frozen from `plan` onward.
    pub fingerprint: String,
    /// Hex sha256 over just the surface a part file's contents depend on. See [`part_fingerprint`].
    ///
    /// Stamped into every part file and the scatter state, so an existing scatter survives any
    /// config change that does not alter routing. Composition — which vectors and payload columns
    /// a part carries — is recorded in `crate::partfile::PartManifest` (stage 2) and checked as a
    /// subset.
    pub part_fingerprint: String,
}

impl LoadedConfig {
    pub fn shard_number(&self) -> u32 {
        self.config.params.shard_number.get()
    }

    pub fn sharding_method(&self) -> ShardingMethod {
        self.config.params.sharding_method.unwrap_or_default()
    }

    /// Virtual nodes per shard on the hash ring.
    ///
    /// Decides routing exactly as `shard_number` does: `ShardHolder::new_ring` constructs the
    /// collection's ring at this scale, so building with a different value silently misroutes
    /// every point whose ring segment moved.
    pub fn hash_ring_shard_scale(&self) -> u32 {
        self.config.params.hash_ring_shard_scale
    }

    /// `max_segment_size`, in bytes. The optimizer's target ceiling for one segment.
    pub fn max_segment_size_bytes(&self) -> Option<u64> {
        Some(self.config.optimizer_config.max_segment_size? as u64 * 1024)
    }

    /// Smallest segment size that keeps the merge optimizer permanently idle.
    ///
    /// `MergeOptimizer::plan_optimizations` (`lib/shard/src/optimizers/merge_optimizer.rs`)
    /// sorts segments ascending by size, then accumulates them while the running total stays
    /// under `max_segment_size`:
    ///
    /// ```text
    /// .scan(0, |size_sum, &(segment_id, size)| {
    ///     *size_sum += size;
    ///     (*size_sum < threshold).then_some(segment_id)
    /// })
    /// ...
    /// if batch.len() < 2 { return; }
    /// ```
    ///
    /// Two segments of size `S` therefore pair up only when `2S < max_segment_size`. If the
    /// two *smallest* segments cannot both fit under the threshold, the batch has fewer than
    /// two members and the optimizer returns having planned nothing — no matter how many
    /// segments the shard holds.
    ///
    /// So the operative constraint is a per-segment floor, not a per-shard budget: keep every
    /// segment at or above `max_segment_size / 2` and total shard size is unbounded. The
    /// `default_segment_number` count only bounds the outer loop, which never runs when
    /// segments are this large.
    pub fn merge_safe_min_segment_bytes(&self) -> Option<u64> {
        Some(self.max_segment_size_bytes()?.div_ceil(2))
    }

    /// The size band to aim for when planning segments.
    ///
    /// Below the floor, the merge optimizer starts combining segments on load. Above the
    /// ceiling we exceed the configured target, which costs longer per-segment HNSW builds
    /// and coarser resume granularity, but provokes no optimizer.
    pub fn target_segment_band_bytes(&self) -> Option<(u64, u64)> {
        Some((
            self.merge_safe_min_segment_bytes()?,
            self.max_segment_size_bytes()?,
        ))
    }
}

/// Load and validate a bare collection config from a JSON string.
///
/// Only the tests use this directly; production parses the `collection` section of the input
/// document and calls [`from_value`].
#[cfg(test)]
pub fn from_str(text: &str) -> Result<LoadedConfig> {
    let raw: Value = serde_json::from_str(text).context("config is not valid JSON")?;
    from_value(raw)
}

/// Load and validate a collection section already parsed out of the input document.
///
/// Four gates, in order. Each rejects a config the next one could not diagnose:
///
/// 1. [`require_explicit_fields`] — nothing that shapes the artifact may be defaulted or
///    left to an "auto" sentinel. Runs against the raw JSON, before serde hides omissions.
/// 2. `Validate::validate` — Qdrant's *own* declared constraints (`ef_construct >= 4`,
///    `0.0 <= deleted_threshold <= 1.0`, `1 <= vector size <= 65536`, and so on). Delegating
///    here rather than restating the ranges means the tool tracks Qdrant automatically.
/// 3. [`check_hard_limits`] — structural caps that are implied by Qdrant's types rather than
///    declared as attributes, so `validate` cannot see them.
/// 4. [`check_no_unknown_fields`] — keys the parser silently dropped, which are almost
///    always typos. Runs last because it needs the parsed config to round-trip.
pub fn from_value(raw: Value) -> Result<LoadedConfig> {
    require_explicit_fields(&raw)?;

    // Deserialized from a borrow so `raw` stays available for the round-trip check below.
    let config = CollectionConfigInternal::deserialize(&raw)
        .context("config does not match Qdrant's collection config")?;

    // Qdrant only *warns* on these (`CollectionConfigInternal::validate_and_warn`). For a
    // build that produces 50 TB of artifacts, a warning is not good enough.
    if let Err(errors) = config.validate() {
        bail!(
            "config violates Qdrant's own validation rules:\n{}",
            label_errors("config", &errors),
        );
    }

    check_hard_limits(&config)?;
    check_no_unknown_fields(&raw, &config)?;

    // Neither the mapping nor the payload index is in scope here — both are siblings of the
    // `collection` section in the input document, so `document::from_str` recomputes each
    // fingerprint once it has all three sections. A bare config with no mapping and no
    // declared indexes (JSONL input) is already correct as-is.
    let fingerprint = fingerprint(&config, None)?;
    let part_fingerprint = part_fingerprint(&config, None)?;

    Ok(LoadedConfig {
        config,
        fingerprint,
        part_fingerprint,
    })
}

/// Reject configs that are structurally impossible, as opposed to merely unwise.
///
/// These limits come from Qdrant's types, not from `#[validate]` attributes, so
/// `Validate::validate` cannot catch them.
fn check_hard_limits(config: &CollectionConfigInternal) -> Result<()> {
    let mut problems: Vec<String> = Vec::new();

    // A segment addresses points with `PointOffsetType`, which is `u32`
    // (`lib/common/common/src/types.rs`). No segment can hold more points than that,
    // whatever `max_segment_size` says, so a `max_segment_size` implying more is unbuildable.
    if let Some(max_segment_size_kb) = config.optimizer_config.max_segment_size {
        let max_segment_bytes = (max_segment_size_kb as u128) * 1024;

        for (name, params) in config.params.vectors.params_iter() {
            let element_bytes: u128 = match params.datatype {
                Some(Datatype::Float16) => 2,
                Some(Datatype::Uint8) | Some(Datatype::Turbo4) => 1,
                Some(Datatype::Float32) | None => 4,
            };
            let vector_bytes = element_bytes * u128::from(params.size.get());
            let cap_bytes = vector_bytes * u128::from(PointOffsetType::MAX);

            if max_segment_bytes > cap_bytes {
                problems.push(format!(
                    "  optimizer_config.max_segment_size ({max_segment_size_kb} KB) exceeds what \
                     one segment can address for vector '{name}': {} points max at \
                     {vector_bytes} bytes/vector = {} KB",
                    PointOffsetType::MAX,
                    cap_bytes / 1024,
                ));
            }
        }
    }

    // `SegmentBuilder::update` refuses more than `U24::MAX` source segments. We build one
    // segment per batch so this is not reachable today, but assert the shard-level analogue:
    // a shard's segment count is bounded by what the builder can later merge.
    const U24_MAX: u32 = (1 << 24) - 1;
    if config.params.shard_number.get() > U24_MAX {
        problems.push(format!(
            "  params.shard_number ({}) exceeds U24::MAX ({U24_MAX})",
            config.params.shard_number,
        ));
    }

    if !problems.is_empty() {
        bail!(
            "config is structurally impossible, not just unwise:\n{}",
            problems.join("\n"),
        );
    }

    Ok(())
}

/// Reject keys that Qdrant's parser silently dropped — almost always typos.
///
/// `CollectionConfigInternal` and its nested types do not set `deny_unknown_fields` (the
/// live server is lenient for rolling-upgrade reasons), so a misspelled key vanishes during
/// parsing. A typo on a *required* field surfaces as "missing" at gate 1, but a typo on an
/// *optional* field silently becomes "use the default". This gate closes that hole without
/// maintaining a field list: it re-serializes the parsed config and demands that every key
/// stated in the raw document survived the round trip — so it tracks the fork's types
/// automatically, the same way delegating to `Validate::validate` does.
///
/// Null-valued keys are exempt, deliberately and soundly: many optional fields carry
/// `skip_serializing_if` and legitimately vanish from the round trip when null — and a
/// *typo'd* key holding null produces exactly the config the correctly-spelled key holding
/// null would have (both mean "unset"), so there is nothing to catch.
///
/// The residual blind spot is semantic: a typo that happens to spell a *different real
/// field* is invisible to any syntax check — that is what gate 2 and the fingerprint
/// comparison exist for.
fn check_no_unknown_fields(raw: &Value, config: &CollectionConfigInternal) -> Result<()> {
    let round_tripped =
        serde_json::to_value(config).context("failed to re-serialize the parsed config")?;

    let mut unknown = Vec::new();
    collect_unknown_keys(raw, &round_tripped, "", &mut unknown);

    if !unknown.is_empty() {
        bail!(
            "config contains fields Qdrant's parser does not recognise (typos?):\n{}\n\n\
             Each listed key was silently dropped during parsing, so the build would run \
             with that field at its default. Fix the spelling, or remove the key.",
            unknown
                .iter()
                .map(|path| format!("  {path}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }

    Ok(())
}

/// Walk `raw`, recording every non-null key with no counterpart in the round-tripped config.
fn collect_unknown_keys(raw: &Value, parsed: &Value, path: &str, unknown: &mut Vec<String>) {
    let (Some(raw_map), Some(parsed_map)) = (raw.as_object(), parsed.as_object()) else {
        // Scalars and arrays are leaves: if the key itself was known, its value was parsed
        // (or rejected by the typed parse long before this gate runs).
        return;
    };

    for (key, value) in raw_map {
        // The null exemption — see [`check_no_unknown_fields`].
        if value.is_null() {
            continue;
        }

        let here = if path.is_empty() {
            key.clone()
        } else {
            format!("{path}.{key}")
        };

        match parsed_map.get(canonical_key(key)) {
            Some(parsed_value) => collect_unknown_keys(value, parsed_value, &here, unknown),
            None => unknown.push(here),
        }
    }
}

/// Serde aliases re-serialize under their canonical name, so the round-trip lookup has to
/// translate them. These are the collection config's declared `#[serde(alias)]`s
/// (`lib/collection/src/operations/config_diff.rs:153-176`, `lib/segment/src/types.rs:826`)
/// — the one hand-maintained piece of this gate, and the same knowledge the [`REQUIRED`]
/// entries carry in their `aliases` lists.
fn canonical_key(key: &str) -> &str {
    match key {
        "max_segment_size_kb" => "max_segment_size",
        "memmap_threshold_kb" => "memmap_threshold",
        "indexing_threshold_kb" => "indexing_threshold",
        "full_scan_threshold_kb" => "full_scan_threshold",
        other => other,
    }
}

// ============================================================================
// The explicit-fields gate
// ============================================================================
// What a document must state, and why — followed by the machinery that checks
// it against the raw JSON: `require_explicit_fields` walks [`REQUIRED`], the
// wildcard expanders turn `*` patterns into concrete paths, and `lookup`
// resolves a path with its alias spellings.

/// A field that must be stated explicitly, and why.
struct Required {
    /// Dotted path into the config document. `*` matches every dense vector.
    path: &'static str,
    /// Accepted alternative sibling spellings: serde aliases such as `max_segment_size_kb`,
    /// or the deprecated `on_disk` spelling of the sparse index's `memory` placement.
    aliases: &'static [&'static str],
    /// What counts as a stated value, beyond mere presence.
    value: ValueRule,
    reason: &'static str,
}

/// How much a field has to say before we accept it as "stated".
///
/// Presence alone is not always enough. Some Qdrant fields use a sentinel value to mean
/// "derive this from the host": `optimizer_config.max_segment_size: null` resolves to
/// `num_indexing_threads * DEFAULT_MAX_SEGMENT_PER_CPU_KB`, and
/// `optimizer_config.default_segment_number: 0` resolves to `default_segment_number()`
/// (`lib/shard/src/optimizers/config.rs`). A config carrying those sentinels is
/// exactly as host-dependent as one that omits the field, so it must be rejected too.
///
/// This distinction is not cosmetic: `GET /collections/{name}` on a real cluster returns
/// `"default_segment_number": 0` and `"max_segment_size": null`, so the most obvious way to
/// produce a config document hands you those sentinels.
#[derive(Clone, Copy, PartialEq)]
enum ValueRule {
    /// The key must exist. An explicit `null` is a real, fixed value for this field —
    /// `payload_m`, `inline_storage`, and the placement fields are compared as `Option`s (or
    /// resolved to a definite default) by the mismatch optimizer, so `null` is a legitimate
    /// choice rather than a request to derive something.
    Present,
    /// The key must exist and not be `null`.
    NonNull,
    /// The key must exist and not be `0`, which is Qdrant's "auto" sentinel here.
    NonZero,
}

impl ValueRule {
    fn satisfied_by(self, value: &Value) -> bool {
        match self {
            ValueRule::Present => true,
            ValueRule::NonNull => !value.is_null(),
            ValueRule::NonZero => value.as_u64() != Some(0),
        }
    }

    fn complaint(self) -> &'static str {
        match self {
            ValueRule::Present => "is missing",
            ValueRule::NonNull => "is missing or null (null defers to a default)",
            ValueRule::NonZero => "is missing or 0 (0 means 'derive from this host')",
        }
    }
}

/// Fields the builder refuses to infer.
///
/// The membership rule: a field is required exactly when **it cannot be changed after the
/// build** — it is frozen into the artifact's structure (routing, vector shape, HNSW graph
/// parameters, quantization, the sparse index's structure class, the deferred-id threshold)
/// — or when its default is **derived from the build host** (core-count sentinels), so that
/// omitting it couples the artifact to the machine that built it.
///
/// Everything else is deliberately optional and takes Qdrant's own default: placements that
/// are load-time metadata (the HNSW graph's, dense vector storage's and payload storage's
/// `memory`), and runtime optimizer knobs (`prevent_unoptimized`, the deprecated
/// `memmap_threshold`, `deleted_threshold`, `default_segment_number`, ...). Those can be
/// adjusted on the collection after the build — or, eventually, retargeted on the artifacts
/// themselves (see the placement-retargeting note above [`Fingerprint`]). Optional never
/// means invisible: the resolved values, stated or defaulted, are still hashed by
/// [`fingerprint`] and frozen at plan.
///
/// A third category is **not the document's business at all**: the build box's own resource
/// usage. `hnsw_config.max_indexing_threads` (and `max_optimization_threads`,
/// `flush_interval_sec`, ...) are per-host runtime knobs — Qdrant resolves them on whichever
/// node is running, and `mismatch_requires_rebuild` ignores `max_indexing_threads` entirely,
/// which is also why the fingerprint does. The build phase must take its thread budget from
/// CLI flags (`--concurrency`, `--indexing-threads`; see the NOTE(stage 3) in `main.rs`) and
/// never from these fields, so that "how many cores this build box burns" and "what the
/// collection declares" are stated in different places by different people.
const REQUIRED: &[Required] = &[
    // -- Topology. Determines the hash ring; a mismatch silently misroutes points, and the
    // -- failure mode is partial: search still finds them, retrieve-by-id does not.
    Required {
        path: "params.shard_number",
        aliases: &[],
        value: ValueRule::NonZero,
        reason: "determines the hash ring; defaults to 1 and cannot be changed after creation",
    },
    Required {
        path: "params.sharding_method",
        aliases: &[],
        value: ValueRule::NonNull,
        reason: "determines whether routing is by point id or by shard key",
    },
    Required {
        path: "params.hash_ring_shard_scale",
        aliases: &[],
        value: ValueRule::NonZero,
        reason: "virtual nodes per shard; decides routing exactly as shard_number does, and is \
                 read-only after creation",
    },
    // -- Segment planning. `max_segment_size`'s default is derived from the host's core
    // -- count (lib/shard/src/optimizers/config.rs), and it is what the plan sizes every
    // -- segment against, so it may not be inferred.
    // -- `default_segment_number` is deliberately optional: the builder only reads it for
    // -- plan review warnings, nothing at serve time compares it, and it can be adjusted on
    // -- the collection after the build. Its 0 sentinel resolves host-dependently only on
    // -- the serving node, at runtime — never into the artifact. Still hashed (stated or
    // -- defaulted), like the placements.
    Required {
        path: "optimizer_config.max_segment_size",
        aliases: &["max_segment_size_kb"],
        value: ValueRule::NonNull,
        reason: "defaults to num_indexing_threads * 256000 KB; bounds each built segment",
    },
    // -- HNSW. Every field on HnswConfig::mismatch_requires_rebuild
    // -- (lib/segment/src/types.rs:863) except max_indexing_threads, which is exempt.
    Required {
        path: "hnsw_config.m",
        aliases: &[],
        value: ValueRule::NonNull,
        reason: "triggers a full segment rebuild on mismatch",
    },
    Required {
        path: "hnsw_config.ef_construct",
        aliases: &[],
        value: ValueRule::NonNull,
        reason: "triggers a full segment rebuild on mismatch",
    },
    Required {
        path: "hnsw_config.full_scan_threshold",
        aliases: &["full_scan_threshold_kb"],
        value: ValueRule::NonNull,
        reason: "triggers a full segment rebuild on mismatch",
    },
    Required {
        path: "hnsw_config.payload_m",
        aliases: &[],
        value: ValueRule::Present,
        reason: "triggers a full segment rebuild on mismatch; state null to mean unset",
    },
    Required {
        path: "hnsw_config.inline_storage",
        aliases: &[],
        value: ValueRule::Present,
        reason: "triggers a full segment rebuild on mismatch; state null to mean unset",
    },
    // -- Quantization. QuantizationConfig::mismatch_requires_rebuild is plain `self != other`
    // -- (lib/segment/src/types.rs:962), so any field difference forces a rebuild. Must be
    // -- present, but may be explicitly null to mean "no quantization".
    Required {
        path: "quantization_config",
        aliases: &[],
        value: ValueRule::Present,
        reason: "any difference forces a rebuild; state null to mean no quantization",
    },
    // -- Thresholds that get baked into the segments we build.
    //
    // -- `indexing_threshold` and `prevent_unoptimized` feed
    // -- `get_deferred_points_threshold_bytes` -> `CollectionParams::get_deferred_point_id`,
    // -- whose result is passed to `build_segment` and `load_segment` as
    // -- `deferred_internal_id`. Different values produce different segments, so
    // -- `indexing_threshold` may not be inferred. `prevent_unoptimized` is deliberately
    // -- *optional*: its default is a constant (disabled), not host-derived, so omitting it
    // -- is deterministic — the same artifact on every build box.
    Required {
        path: "optimizer_config.indexing_threshold",
        aliases: &["indexing_threshold_kb"],
        value: ValueRule::NonNull,
        reason: "feeds deferred_internal_id, which is baked into every segment",
    },
    // -- Sparse index placement selects the built index's *structure*: pinned builds
    // -- an `ImmutableRam` index, cold/cached build an `Mmap` index. Different structures,
    // -- so it cannot be changed after the build
    Required {
        path: "params.sparse_vectors.*.index",
        aliases: &[],
        value: ValueRule::NonNull,
        reason: "sparse index config; state at least {\"memory\": ...} rather than inheriting",
    },
    Required {
        path: "params.sparse_vectors.*.index.memory",
        aliases: &["on_disk"],
        value: ValueRule::NonNull,
        reason: "defaults to pinned (RAM); an inverted index over many non-zeros per point is \
                 large, and the resolved placement is compared by the config-mismatch optimizer",
    },
    // -- WAND pruning shapes the built index, not just its config: with pruning disabled the
    // -- index skips maintaining `max_next_weight` (`inverted_index_ram.rs`,
    // -- `maintain_max_next_weight` / `max_next_weight_reliable`), so pruning cannot simply
    // -- be switched on later. Cannot change after the build => required. Explicit null
    // -- means Qdrant's default (enabled).
    Required {
        path: "params.sparse_vectors.*.index.wand_pruning",
        aliases: &[],
        value: ValueRule::Present,
        reason: "baked into the built segment's sparse index config; state null to mean \
                 Qdrant's default (enabled)",
    },
    // -- Per-vector shape and placement.
    Required {
        path: "params.vectors.*.size",
        aliases: &[],
        value: ValueRule::NonNull,
        reason: "vector dimensionality",
    },
    // -- `datatype` selects the storage element width and `multivector_config` changes the
    // -- storage layout; both are recorded in the segment's VectorDataConfig.
    Required {
        path: "params.vectors.*.datatype",
        aliases: &[],
        value: ValueRule::Present,
        reason: "storage element width; state null to mean float32",
    },
    Required {
        path: "params.vectors.*.multivector_config",
        aliases: &[],
        value: ValueRule::Present,
        reason: "changes vector storage layout; state null to mean single-vector",
    },
    Required {
        path: "params.vectors.*.distance",
        aliases: &[],
        value: ValueRule::NonNull,
        reason: "distance function",
    },
    // Per-vector storage *placement* (`params.vectors.*.memory`, legacy `on_disk`) is
    // deliberately NOT required: Mmap and InRamMmap are the same bytes loaded differently,
    // so it is adjustable after the build. The resolved on-disk-ness is still hashed as
    // `storage_on_disk` in the fingerprint.
];

/// Reject config documents that leave any [`REQUIRED`] field to a host-derived default.
pub fn require_explicit_fields(raw: &Value) -> Result<()> {
    let mut missing: Vec<String> = Vec::new();

    for req in REQUIRED {
        for path in expand_vector_wildcard(raw, req.path) {
            let stated =
                lookup(raw, &path, req.aliases).is_some_and(|value| req.value.satisfied_by(value));

            if !stated {
                missing.push(format!(
                    "  {path} {}\n      -> {}",
                    req.value.complaint(),
                    req.reason,
                ));
            }
        }
    }

    if !missing.is_empty() {
        bail!(
            "config must state these fields explicitly; \
             leaving them to a default or an \"auto\" sentinel couples the build to this \
             machine:\n{}\n\n\
             Note: `GET /collections/{{name}}` returns `default_segment_number: 0` and \
             `max_segment_size: null` when they were never set — both mean \"derive from \
             this host\", so a config exported from a live collection usually needs those \
             two pinned to real values before it can be built from.",
            missing.join("\n"),
        );
    }

    Ok(())
}

/// Expand a `params.vectors.*.field` pattern into one path per configured dense vector.
///
/// `VectorsConfig` is an untagged enum: `Single(VectorParams)` serializes as the vector params
/// inline, `Multi(map)` as a map of name to params. A `size` key at the top level distinguishes
/// the single form.
fn expand_vector_wildcard(raw: &Value, path: &'static str) -> Vec<String> {
    if path.starts_with("params.sparse_vectors.*.") {
        return expand_sparse_wildcard(raw, path);
    }

    let Some(suffix) = path.strip_prefix("params.vectors.*.") else {
        return vec![path.to_string()];
    };

    let Some(vectors) = raw.pointer("/params/vectors") else {
        // Absent `vectors` is reported by the typed parse, not here.
        return Vec::new();
    };

    if vectors.get("size").is_some() {
        // Single unnamed vector.
        return vec![format!("params.vectors.{suffix}")];
    }

    match vectors.as_object() {
        Some(map) => map
            .keys()
            .map(|name| format!("params.vectors.{name}.{suffix}"))
            .collect(),
        None => Vec::new(),
    }
}

/// Expand a `params.sparse_vectors.*.field` pattern into one path per sparse vector.
///
/// Unlike dense vectors, sparse vectors are optional, so an absent `sparse_vectors` map yields
/// no paths rather than an error.
fn expand_sparse_wildcard(raw: &Value, path: &'static str) -> Vec<String> {
    let Some(suffix) = path.strip_prefix("params.sparse_vectors.*.") else {
        return Vec::new();
    };

    let Some(map) = raw
        .pointer("/params/sparse_vectors")
        .and_then(Value::as_object)
    else {
        return Vec::new();
    };

    map.keys()
        .map(|name| format!("params.sparse_vectors.{name}.{suffix}"))
        .collect()
}

/// Resolve a dotted path to its value, accepting `aliases` for the leaf name.
///
/// Returns `None` only when the key is absent. An explicit `null` resolves to
/// `Some(Value::Null)`, so [`ValueRule`] decides whether that counts as stated. An alias is
/// consulted only when the primary leaf is absent.
fn lookup<'a>(raw: &'a Value, path: &str, aliases: &[&str]) -> Option<&'a Value> {
    let mut parts: Vec<&str> = path.split('.').collect();
    let leaf = parts.pop().expect("path is never empty");

    let mut node = raw;
    for part in parts {
        node = node.get(part)?;
    }

    node.get(leaf)
        .or_else(|| aliases.iter().find_map(|alias| node.get(alias)))
}

// ============================================================================
// Fingerprints
// ============================================================================
// Two hashes over two surfaces: the full fingerprint (frozen at plan) and the
// part fingerprint (frozen at scatter). Each struct sits next to its version
// constant and the function that produces it.
//
// NOTE(retarget): placement changes after the build. Most resolved placements hashed below —
// dense storage, the HNSW graph, payload storage, sparse cold-vs-cached — are load-time
// metadata over identical bytes; `HnswConfig::mismatch_requires_rebuild`
// (lib/segment/src/types.rs:881) says so itself ("Data on disk is the same ... just to flip
// this flag"). They are frozen at plan anyway because the serving cluster's
// ConfigMismatchOptimizer rebuilds on a placement mismatch *today*, and this hash must
// predict what the cluster will do.
//
// The `retarget` step (`crate::retarget`) is the sanctioned way past that freeze: it rewrites
// each *built* segment's recorded config to a placement-edited document, refusing anything
// structural by name, so a placement change costs a metadata rewrite instead of a rebuild.
// Its correctness claim — the cluster loads a rewritten segment and queues zero
// optimizations — is proven differentially in `build_e2e_tests`
// (`retargeting_placements_matches_a_fresh_build`): retargeted must equal freshly built,
// and the bulk route resolves configs with the serving optimizer's own code.
//
// An earlier draft of this note proposed splitting the hash instead — a structural
// fingerprint frozen at plan, placements checked only at launch. That was deliberately NOT
// done: with placements outside the freeze, a document edited mid-build (or mid-resume)
// would stamp part of a shard one way and the rest another, silently — the exact hazard the
// plan freeze exists to prevent. So placements stay in this hash, `build` still refuses any
// drift, and retarget runs over finished artifacts only.
//
// Never retargetable, and correctly structural: sparse pinned-vs-mmap (ImmutableRam and
// Mmap are different index structures), hnsw_config.inline_storage (changes the graph
// file's bytes), and quantization wholesale — faithful to the cluster, since
// QuantizationConfig::mismatch_requires_rebuild is plain equality, i.e. even a quantization
// placement flip is a rebuild there.

// ----- The full fingerprint: frozen at plan -----

/// The rebuild-triggering surface, in a form that hashes deterministically.
#[derive(Debug, Serialize)]
struct Fingerprint {
    /// Bumped when the set of hashed fields changes, so old artifacts fail loudly rather
    /// than comparing equal by accident.
    version: u32,
    shard_number: u32,
    sharding_method: String,
    hash_ring_shard_scale: u32,
    default_segment_number: usize,
    max_segment_size_kb: Option<usize>,
    /// Resolved optimizer thresholds that shape the *built bytes*, hashed exactly as `build`
    /// resolves them (and thread-independent — `max_segment_size_kb`'s thread-derived form is
    /// deliberately not hashed; the raw `max_segment_size` above is):
    ///
    /// * `indexing_threshold_kb` decides Plain-vs-HNSW and, with quantization, whether a
    ///   segment is indexed at all (`optimized_segment_config`).
    /// * `memmap_threshold_kb` decides on-disk vs in-RAM storage type.
    /// * `deferred_points_threshold_bytes` resolves `prevent_unoptimized` together with the
    ///   indexing threshold into the deferred-id boundary baked into every segment.
    ///
    /// Without these, editing `indexing_threshold`/`memmap_threshold`/`prevent_unoptimized`
    /// between `plan` and `build` left the fingerprint unchanged while the segments' shape
    /// changed — the "mismatched segments in one shard" hazard the freeze exists to prevent.
    indexing_threshold_kb: usize,
    memmap_threshold_kb: usize,
    deferred_points_threshold_bytes: Option<usize>,
    /// The resolved `payload_storage_type()`'s on-disk-ness — exactly what
    /// `has_config_mismatch` compares, so equivalent spellings (`payload.memory` vs the
    /// deprecated `on_disk_payload`) hash identically.
    payload_storage_on_disk: bool,
    /// The document's payload-index schema, if any. Not compared by `has_config_mismatch` —
    /// a field index can be rebuilt without rewriting segments — but the build phase bakes
    /// the declared indexes into every segment, so a mid-build schema edit must fail the
    /// plan gate rather than produce a shard whose segments carry different indexes (the
    /// ones missing an index would push an index build onto the serving filesystem at load,
    /// the exact cost this tool exists to avoid).
    ///
    /// Hashed in canonical form ([`PayloadIndexSchema::canonical_fields`]): placement
    /// spellings resolved, everything else raw — the same equivalence the server's
    /// `schema_transition::classify` applies when deciding whether a field index needs
    /// rebuilding.
    ///
    /// [`PayloadIndexSchema::canonical_fields`]: crate::payload_index::PayloadIndexSchema::canonical_fields
    payload_index: Option<BTreeMap<String, Value>>,
    /// BTreeMap so key order is deterministic regardless of the source document's order.
    dense_vectors: BTreeMap<String, Value>,
    sparse_vectors: BTreeMap<String, Value>,
}

// v2 (fork adaptation): added `hash_ring_shard_scale`, and placements are hashed *resolved*
// (`memory` over `on_disk`) so equivalent spellings hash identically — see `hashable_hnsw`
// and the dense/sparse entries in `fingerprint`.
// v3: the payload-index schema joined the surface — it was previously hashed nowhere, so a
// mid-build edit of that document section slipped past the plan gate entirely.
// v4: payload-index placements hashed *resolved*, like every other placement — `on_disk:
// true` and `memory: "cold"` are one placement in two spellings, the server's
// `schema_transition::classify` treats them as identical (no rebuild), and hashing them
// raw made `verify-config` fail over pure notation.
// v5: the resolved optimizer thresholds (`indexing_threshold_kb`, `memmap_threshold_kb`,
// `deferred_points_threshold_bytes`) joined the surface — `build` reads them into the built
// segments' shape (Plain-vs-HNSW, storage type, deferred-id boundary), but they were hashed
// nowhere, so editing `indexing_threshold`/`memmap_threshold`/`prevent_unoptimized` between
// plan and build slipped past the freeze.
const FINGERPRINT_VERSION: u32 = 5;

/// Hash the config fields that decide whether the serving cluster rebuilds our segments,
/// plus the payload-index schema the build phase bakes into them.
///
/// Resolution is delegated to [`build_segment_optimizer_config`] — the same function
/// `build_optimizers` uses — so per-vector HNSW merging (`global.update_opt(per_vector)`)
/// and quantization fallback (`per_vector.or(global)`) are computed by the production code
/// path rather than reimplemented here. Placements are hashed *resolved* (`memory` over the
/// deprecated `on_disk`), because that is the surface `has_config_mismatch` compares — two
/// spellings of the same placement must hash identically.
///
/// The payload-index schema lives in the *document*, not the collection config, so
/// `config::from_value` computes this hash without one and `document::from_str` recomputes
/// it once both sections are in hand — the same two-step dance `part_fingerprint` does for
/// the mapping's id fields. (`verify-config`, stage 4, must likewise fetch the live
/// collection's payload schema to compare like with like.)
pub fn fingerprint(
    config: &CollectionConfigInternal,
    payload_index: Option<&PayloadIndexSchema>,
) -> Result<String> {
    let optimizer_config = build_segment_optimizer_config(
        &config.params,
        &config.hnsw_config,
        &config.quantization_config,
    );

    let mut dense_vectors = BTreeMap::new();
    for (name, plain) in &optimizer_config.plain_dense_vector_config {
        let optimized = optimizer_config.dense_vector.get(name);

        dense_vectors.insert(
            name.clone(),
            json!({
                // Shape and storage of the plain (pre-optimization) segment.
                "plain": plain,
                // What the optimizer would impose on an indexed segment. This is the surface
                // has_config_mismatch actually compares: the resolved placement's on-disk-ness
                // against the segment's storage type...
                "storage_on_disk": optimized
                    .and_then(|cfg| cfg.memory_placement())
                    .map(|memory| memory.is_on_disk()),
                // ...and the full HNSW config, including its resolved placement.
                "hnsw": optimized.map(|cfg| hashable_hnsw(&cfg.hnsw_config)),
                "quantization": optimized.and_then(|cfg| cfg.quantization_config.as_ref()),
            }),
        );
    }

    let mut sparse_vectors = BTreeMap::new();
    for (name, plain) in &optimizer_config.plain_sparse_vector_config {
        let optimized = optimizer_config.sparse_vector.get(name);

        // `plain` embeds the raw `memory` parameter verbatim, so two spellings of one
        // placement (`on_disk: true` vs `memory: "cold"`) would hash differently through
        // it. The plain segment's sparse index is MutableRam, whose placement the mismatch
        // optimizer never compares — the placement that governs rebuilds is the resolved
        // one hashed below — so the raw field is dropped from the hash.
        let mut plain =
            serde_json::to_value(plain).context("failed to serialize sparse vector config")?;
        if let Some(index) = plain.get_mut("index").and_then(Value::as_object_mut) {
            index.remove("memory");
        }

        sparse_vectors.insert(
            name.clone(),
            json!({
                // Still includes the rest of the index config — and with it `wand_pruning`,
                // which `to_sparse_vector_data` carries into every built segment.
                "plain": plain,
                // The sparse arm of has_config_mismatch compares the resolved placement in
                // full (pinned vs cached vs cold), not just its on-disk-ness.
                "memory_placement": optimized.and_then(|cfg| cfg.memory_placement()),
            }),
        );
    }

    // Thread count only affects `max_segment_size_kb` (not hashed here), so `1` is fine.
    let optimizer_thresholds = config.optimizer_config.optimizer_thresholds(1, None);

    let fingerprint = Fingerprint {
        version: FINGERPRINT_VERSION,
        shard_number: config.params.shard_number.get(),
        sharding_method: format!("{:?}", config.params.sharding_method.unwrap_or_default()),
        hash_ring_shard_scale: config.params.hash_ring_shard_scale,
        default_segment_number: config.optimizer_config.default_segment_number,
        max_segment_size_kb: config.optimizer_config.max_segment_size,
        // Resolved exactly as `build` does. The thread count only feeds `max_segment_size_kb`
        // (hashed raw above, deliberately not here), so any value gives the same
        // `indexing_threshold_kb`/`memmap_threshold_kb`; `1` is a safe stand-in.
        indexing_threshold_kb: optimizer_thresholds.indexing_threshold_kb,
        memmap_threshold_kb: optimizer_thresholds.memmap_threshold_kb,
        deferred_points_threshold_bytes: config
            .optimizer_config
            .get_deferred_points_threshold_bytes()
            .map(|threshold| threshold.get()),
        payload_storage_on_disk: config.params.payload_storage_type().is_on_disk(),
        payload_index: payload_index
            .map(crate::payload_index::PayloadIndexSchema::canonical_fields)
            .transpose()
            .context("failed to serialize payload index schema")?,
        dense_vectors,
        sparse_vectors,
    };

    let canonical =
        serde_json::to_vec(&fingerprint).context("failed to serialize config fingerprint")?;

    Ok(hex(&Sha256::digest(&canonical)))
}

// ----- The part fingerprint: frozen at scatter -----

/// The surface that decides whether an existing scatter can still be consumed.
///
/// Deliberately much narrower than [`Fingerprint`]. See [`part_fingerprint`].
#[derive(Debug, Serialize)]
struct PartFingerprint {
    version: u32,
    shard_number: u32,
    sharding_method: String,
    hash_ring_shard_scale: u32,
    /// Which column supplies the point id, and how it is parsed. Both decide point identity, and
    /// therefore routing. `None` for input formats that carry ids directly, such as JSONL.
    id_column: Option<String>,
    id_format: Option<String>,
}

// v2: narrowed from (routing + vector names + dims + datatypes) to routing alone. Vector and
// payload composition moved into `PartManifest`, where it is checked as a subset instead — which
// is what lets a build drop a vector or a payload column without re-scattering.
// v3 (fork adaptation): routing now also depends on `hash_ring_shard_scale`.
const PART_FINGERPRINT_VERSION: u32 = 3;

/// Hash only the config fields that a part file's contents actually depend on.
///
/// This is the guard stamped into every part file, the scatter state and every plan. It covers
/// exactly one thing: **routing** — `shard_number`, `sharding_method` and
/// `hash_ring_shard_scale` decide which shard a point lands in, so consuming parts under a
/// different ring would put points in the wrong shard.
///
/// Everything else is deliberately excluded, and the exclusions are the point. `hnsw_config`,
/// `quantization_config`, `max_segment_size` and the placement flags do not influence a single
/// byte of a part file, so hashing them here would mean **re-scattering tens of terabytes to
/// change `ef_construct`**.
///
/// Those fields are not unguarded — they are frozen one stage later. A plan records the *full*
/// [`fingerprint`], and `build` refuses a plan whose fingerprint differs from the config it was
/// given. So the contract is:
///
/// * before `plan`: any config field may change, and an existing scatter is still usable as long
///   as routing is untouched;
/// * after `plan`: the config is frozen, and changing HNSW means re-running `plan` (minutes) —
///   never re-scattering (hours, tens of terabytes).
///
/// That also keeps the guard that matters during a distributed build: ten nodes all compare
/// against the same plan, so a config edited mid-build is a hard error rather than a shard of
/// silently mismatched segments.
///
/// # Why *only* routing
///
/// Everything else a part depends on is recorded in its own header, in
/// `crate::partfile::PartManifest` (stage 2), and checked at build time as a *subset*: which
/// dense and sparse vectors it carries, at what element width and dimensionality, from which
/// source column, and which payload columns were captured. A build may ask for fewer of those
/// than the scatter captured — that is how a vector or a payload column gets dropped without
/// re-scattering — and asking for one that is absent is refused by name.
///
/// Only two things cannot be resolved that way, because they make every record in the file wrong
/// rather than merely a superset: the ring that chose the shard, and the point ids fed into it.
/// Hence `shard_number`, `sharding_method`, `hash_ring_shard_scale`, `id_column`, `id_format`,
/// and nothing more.
///
/// `id_column` and `id_format` come from the *mapping*, which was previously hashed nowhere at
/// all. Changing `id_format` and resuming a scatter used to leave earlier files routed by one
/// scheme and later ones by another, in one work directory, silently.
pub fn part_fingerprint(
    config: &CollectionConfigInternal,
    mapping: Option<&crate::parquet_source::ParquetMapping>,
) -> Result<String> {
    let fingerprint = PartFingerprint {
        version: PART_FINGERPRINT_VERSION,
        shard_number: config.params.shard_number.get(),
        sharding_method: format!("{:?}", config.params.sharding_method.unwrap_or_default()),
        hash_ring_shard_scale: config.params.hash_ring_shard_scale,
        id_column: mapping.map(|m| m.id_column.clone()),
        id_format: mapping.map(|m| format!("{:?}", m.id_format)),
    };

    let canonical =
        serde_json::to_vec(&fingerprint).context("failed to serialize part fingerprint")?;

    Ok(hex(&Sha256::digest(&canonical)))
}

// ----- Hashing helpers -----

/// Strip the one HNSW field that does *not* force a rebuild, and resolve the placement.
///
/// `HnswConfig::mismatch_requires_rebuild` (`lib/segment/src/types.rs:863`) destructures
/// `max_indexing_threads` into `_` and ignores it. Including it here would make an
/// otherwise-identical config hash differently on a machine with a different core count,
/// producing false mismatches on exactly the setups this tool exists to protect.
///
/// The raw `on_disk` and `memory` fields are likewise ignored in favour of
/// `memory_placement()`, mirroring the same function: it compares the resolved placement so
/// that expressing one placement through either parameter does not trigger a rebuild — and
/// must not move this hash either.
fn hashable_hnsw(hnsw: &HnswConfig) -> Value {
    // The exhaustive destructure is the point: a new HnswConfig field fails compilation here,
    // forcing a decision about whether it belongs in the hash. `on_disk` is deprecated
    // upstream, but naming it (as `_`) is required for exhaustiveness.
    #[allow(deprecated)]
    let HnswConfig {
        m,
        ef_construct,
        full_scan_threshold,
        max_indexing_threads: _,
        payload_m,
        on_disk: _,
        memory: _,
        inline_storage,
    } = *hnsw;

    json!({
        "m": m,
        "ef_construct": ef_construct,
        "full_scan_threshold": full_scan_threshold,
        "payload_m": payload_m,
        "memory_placement": hnsw.memory_placement(),
        "inline_storage": inline_storage,
    })
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut acc, byte| {
        let _ = write!(acc, "{byte:02x}");
        acc
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    // ----- Fixture and helpers -----

    /// A config document with every required field stated, in the fork's preferred spellings
    /// (`memory` placements, `hash_ring_shard_scale`, `wand_pruning`). Tests perturb copies.
    pub(crate) fn valid_config_json() -> Value {
        json!({
            "params": {
                "vectors": {
                    "dense": {
                        "size": 768,
                        "distance": "Cosine",
                        "memory": "cold",
                        "datatype": null,
                        "multivector_config": null,
                    },
                },
                "sparse_vectors": {
                    "sparse": { "index": { "memory": "cold", "wand_pruning": null } }
                },
                "shard_number": 4,
                "sharding_method": "auto",
                "hash_ring_shard_scale": 100,
                "replication_factor": 1,
                "write_consistency_factor": 1,
                "payload": { "memory": "cold" },
            },
            "hnsw_config": {
                "m": 16,
                "ef_construct": 128,
                "full_scan_threshold": 10000,
                "max_indexing_threads": 8,
                "payload_m": null,
                "memory": null,
                "inline_storage": null,
            },
            // `memmap_threshold` and `prevent_unoptimized` are deliberately absent: both are
            // optional (adjustable after the build), and the fixture proves omission is fine.
            "optimizer_config": {
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": 8,
                "max_segment_size": 5_000_000,
                "indexing_threshold": 10000,
                "flush_interval_sec": 5,
            },
            "wal_config": {
                "wal_capacity_mb": 32,
                "wal_segments_ahead": 0,
            },
            "quantization_config": {
                "turbo": {
                    "memory": "pinned",
                    "bits": "bits4",
                },
            },
        })
    }

    fn load_value(value: &Value) -> Result<LoadedConfig> {
        from_str(&value.to_string())
    }

    /// Remove a dotted path from a JSON document, for the "missing field" tests.
    fn remove_path(value: &mut Value, path: &str) {
        let mut parts: Vec<&str> = path.split('.').collect();
        let leaf = parts.pop().expect("path is never empty");

        let mut node = value;
        for part in parts {
            match node.get_mut(part) {
                Some(next) => node = next,
                None => return,
            }
        }

        if let Some(map) = node.as_object_mut() {
            map.shift_remove(leaf);
        }
    }

    // ----- The explicit-fields gate -----

    #[test]
    fn accepts_fully_specified_config() {
        let loaded = load_value(&valid_config_json()).expect("valid config should load");
        assert_eq!(loaded.shard_number(), 4);
        assert_eq!(loaded.sharding_method(), ShardingMethod::Auto);
        assert_eq!(loaded.hash_ring_shard_scale(), 100);
        assert_eq!(loaded.fingerprint.len(), 64, "sha256 hex is 64 chars");
    }

    /// Every required field, removed one at a time, must be reported.
    ///
    /// Parameterised over `REQUIRED` itself so adding an entry there without a test is
    /// impossible. This relies on the fixture using each entry's *primary* spelling — a
    /// fixture that satisfied a rule through an alias would make the removal a no-op.
    #[test]
    fn rejects_each_missing_required_field() {
        for req in REQUIRED {
            let paths = expand_vector_wildcard(&valid_config_json(), req.path);
            assert!(
                !paths.is_empty(),
                "fixture does not exercise {}; add it to valid_config_json()",
                req.path,
            );

            for path in &paths {
                let mut value = valid_config_json();
                remove_path(&mut value, path);

                let err = load_value(&value).unwrap_err();
                let message = format!("{err:#}");
                assert!(
                    message.contains(path),
                    "error for missing {path} should name the field, got: {message}",
                );
            }
        }
    }

    #[test]
    fn accepts_serde_aliases_for_required_fields() {
        // The `_kb` spellings are serde aliases; a document using the alias form is explicit
        // and must be accepted — by the gate (via `Required::aliases`) and by the round-trip
        // typo check (via `canonical_key`), which this exercises for all four aliases.
        let mut value = valid_config_json();
        remove_path(&mut value, "optimizer_config.max_segment_size");
        value["optimizer_config"]["max_segment_size_kb"] = json!(5_000_000);
        remove_path(&mut value, "hnsw_config.full_scan_threshold");
        value["hnsw_config"]["full_scan_threshold_kb"] = json!(10000);
        remove_path(&mut value, "optimizer_config.indexing_threshold");
        value["optimizer_config"]["indexing_threshold_kb"] = json!(10000);
        value["optimizer_config"]["memmap_threshold_kb"] = json!(20000);

        load_value(&value).expect("alias spellings should be accepted");
    }

    /// Qdrant's types do not deny unknown fields, so a typo would otherwise silently become
    /// "use the default". Gate 4 catches it by round-tripping the parsed config and
    /// demanding every stated key survived.
    #[test]
    fn rejects_typoed_field_names_by_path() {
        // A typo on an optional field is the dangerous case: every other gate passes.
        let mut value = valid_config_json();
        value["hnsw_config"]["inline_storag"] = json!(true);
        let err = load_value(&value).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("hnsw_config.inline_storag"), "{text}");
        assert!(text.contains("typo"), "{text}");

        // ...including nested inside a named vector.
        let mut value = valid_config_json();
        value["params"]["vectors"]["dense"]["memmory"] = json!("cold");
        let err = load_value(&value).unwrap_err();
        assert!(
            format!("{err:#}").contains("params.vectors.dense.memmory"),
            "got: {err:#}",
        );
    }

    /// A null-valued typo is exempt: it produces exactly the config the correctly-spelled
    /// key holding null would have (both mean "unset"), and the `skip_serializing_if`
    /// fields make explicit nulls indistinguishable from omissions after the round trip.
    #[test]
    fn null_valued_unknown_keys_are_tolerated() {
        let mut value = valid_config_json();
        value["hnsw_config"]["inline_storag"] = json!(null);
        load_value(&value).expect("a null-valued unknown key changes nothing and is tolerated");
    }

    /// Fields Qdrant knows but this module never mentions must pass through untouched —
    /// including free-form ones like `metadata`, whose keys are user data, not field names.
    #[test]
    fn unmentioned_qdrant_fields_pass_through() {
        let mut value = valid_config_json();
        value["metadata"] = json!({ "team": "research", "corpus": "fineweb" });
        value["params"]["replication_factor"] = json!(2);
        load_value(&value).expect("every real Qdrant field must be accepted");
    }

    /// The required-only-if-frozen rule: fields adjustable after the build may be omitted
    /// entirely and take Qdrant's own defaults.
    #[test]
    fn post_build_adjustable_fields_may_be_omitted() {
        let mut value = valid_config_json();
        remove_path(&mut value, "hnsw_config.memory");
        remove_path(&mut value, "params.vectors.dense.memory");
        remove_path(&mut value, "params.payload");
        // memmap_threshold and prevent_unoptimized are already absent from the fixture.
        load_value(&value).expect("adjustable-after-build fields must be optional");
    }

    /// The deprecated `on_disk` spellings still parse — and for the sparse index placement,
    /// the one placement that stays required, the legacy spelling counts as stating it.
    #[test]
    fn accepts_the_deprecated_on_disk_spelling_of_placements() {
        let mut value = valid_config_json();

        remove_path(&mut value, "hnsw_config.memory");
        value["hnsw_config"]["on_disk"] = json!(false);

        remove_path(&mut value, "params.vectors.dense.memory");
        value["params"]["vectors"]["dense"]["on_disk"] = json!(true);

        remove_path(&mut value, "params.sparse_vectors.sparse.index.memory");
        value["params"]["sparse_vectors"]["sparse"]["index"]["on_disk"] = json!(true);

        // The payload placement's deprecated spelling is not even a sibling key — it lives at
        // `params.on_disk_payload` while its replacement is `params.payload.memory`.
        remove_path(&mut value, "params.payload");
        value["params"]["on_disk_payload"] = json!(true);

        load_value(&value).expect("legacy on_disk spellings should be accepted");
    }

    /// A config exported verbatim from an older live collection must be rejected.
    ///
    /// This is the shape `GET /collections/{name}` returned before the fork's parameters
    /// existed. It is the most obvious way to produce a config document, and it carries
    /// three host- or version-coupling hazards: `sharding_method` omitted,
    /// `max_segment_size: null`, and no `hash_ring_shard_scale`. The `max_segment_size` key
    /// is *present*, so a presence-only check would wave it through and the build would
    /// silently inherit this machine's core count.
    #[test]
    fn rejects_config_exported_from_a_live_collection() {
        let exported = json!({
            "params": {
                "vectors": { "dense": { "size": 16, "distance": "Cosine" } },
                "shard_number": 1,
                "replication_factor": 1,
                "write_consistency_factor": 1,
                "on_disk_payload": true,
            },
            "hnsw_config": {
                "m": 48,
                "ef_construct": 256,
                "full_scan_threshold": 12000,
                "max_indexing_threads": 2,
                "on_disk": true,
                "payload_m": 16,
            },
            "optimizer_config": {
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": 0,
                "max_segment_size": null,
                "memmap_threshold": null,
                "indexing_threshold": 20000,
                "flush_interval_sec": 5,
                "max_optimization_threads": null,
                "prevent_unoptimized": null,
            },
            "wal_config": {
                "wal_capacity_mb": 32,
                "wal_segments_ahead": 0,
                "wal_retain_closed": 1,
            },
            "quantization_config": null,
        });

        let err = load_value(&exported).unwrap_err();
        let message = format!("{err:#}");

        for expected in [
            "params.sharding_method",
            "params.hash_ring_shard_scale",
            "optimizer_config.max_segment_size",
            "params.vectors.dense.datatype",
            "hnsw_config.inline_storage",
        ] {
            assert!(
                message.contains(expected),
                "a live-exported config must be rejected for {expected}, got: {message}",
            );
        }
    }

    /// `0` and `null` are Qdrant's "derive from host" sentinels, not stated values.
    #[test]
    fn rejects_auto_sentinels_for_host_derived_fields() {
        // default_segment_number's 0 sentinel is the exception: it resolves host-dependently
        // only on the serving node at runtime, never into the artifact, so the field left
        // the gate when it was ruled adjustable-after-build. The sentinel is accepted.
        let mut zero_segments = valid_config_json();
        zero_segments["optimizer_config"]["default_segment_number"] = json!(0);
        load_value(&zero_segments).expect("default_segment_number is optional; 0 is accepted");

        let mut null_size = valid_config_json();
        null_size["optimizer_config"]["max_segment_size"] = Value::Null;
        let err = load_value(&null_size).unwrap_err();
        assert!(
            format!("{err:#}").contains("max_segment_size is missing or null"),
            "got: {err:#}",
        );

        // shard_number: 0 is not a legal NonZeroU32 anyway, but reject it at the gate so the
        // error explains the topology consequence instead of surfacing as a parse failure.
        let mut zero_shards = valid_config_json();
        zero_shards["params"]["shard_number"] = json!(0);
        let err = load_value(&zero_shards).unwrap_err();
        assert!(
            format!("{err:#}").contains("shard_number is missing or 0"),
            "got: {err:#}",
        );

        // hash_ring_shard_scale: 0 is likewise outside the validator's 1..=100_000 range, but
        // the gate names the routing consequence rather than a range violation.
        let mut zero_scale = valid_config_json();
        zero_scale["params"]["hash_ring_shard_scale"] = json!(0);
        let err = load_value(&zero_scale).unwrap_err();
        assert!(
            format!("{err:#}").contains("hash_ring_shard_scale is missing or 0"),
            "got: {err:#}",
        );

        // Explicit null sharding_method is not a choice between auto and custom.
        let mut null_sharding = valid_config_json();
        null_sharding["params"]["sharding_method"] = Value::Null;
        let err = load_value(&null_sharding).unwrap_err();
        assert!(
            format!("{err:#}").contains("sharding_method is missing or null"),
            "got: {err:#}",
        );
    }

    #[test]
    fn explicit_null_counts_as_stated() {
        // `payload_m: null` and `quantization_config: null` are deliberate "unset", not
        // omissions. The point of the gate is to force a decision, not a non-null value.
        let mut value = valid_config_json();
        value["quantization_config"] = Value::Null;
        load_value(&value).expect("explicit null quantization should be accepted");
    }

    #[test]
    fn handles_single_unnamed_vector_form() {
        // VectorsConfig is untagged: the single form inlines the params.
        let mut value = valid_config_json();
        value["params"]["vectors"] = json!({
            "size": 768,
            "distance": "Cosine",
            "memory": "cold",
            "datatype": null,
            "multivector_config": null,
        });
        load_value(&value).expect("single-vector form should be accepted");

        // ...and the wildcard must still catch a missing structural field in that form.
        let mut value = valid_config_json();
        value["params"]["vectors"] = json!({ "size": 768, "distance": "Cosine" });
        let err = load_value(&value).expect_err("missing datatype must be rejected");
        assert!(format!("{err:#}").contains("params.vectors.datatype"));
    }

    /// Sparse index placement must be stated when sparse vectors exist.
    #[test]
    fn requires_sparse_index_placement() {
        // No sparse vectors: the rule must not fire at all.
        let mut dense_only = valid_config_json();
        dense_only["params"]
            .as_object_mut()
            .unwrap()
            .shift_remove("sparse_vectors");
        load_value(&dense_only).expect("a dense-only config is fine");

        // Sparse present but index unset: rejected.
        let mut value = valid_config_json();
        value["params"]["sparse_vectors"] = json!({ "sparse": {} });
        let err = load_value(&value).unwrap_err();
        assert!(
            format!("{err:#}").contains("params.sparse_vectors.sparse.index"),
            "got: {err:#}",
        );

        // Index present but placement unset: still rejected, because the default is RAM.
        let mut value = valid_config_json();
        value["params"]["sparse_vectors"] = json!({ "sparse": { "index": {} } });
        let err = load_value(&value).unwrap_err();
        assert!(
            format!("{err:#}").contains("params.sparse_vectors.sparse.index.memory"),
            "got: {err:#}",
        );

        // Placement stated but wand_pruning omitted: rejected — it is baked into segments.
        let mut value = valid_config_json();
        value["params"]["sparse_vectors"] = json!({ "sparse": { "index": { "memory": "cold" } } });
        let err = load_value(&value).unwrap_err();
        assert!(
            format!("{err:#}").contains("params.sparse_vectors.sparse.index.wand_pruning"),
            "got: {err:#}",
        );

        // Fully stated: accepted.
        let mut value = valid_config_json();
        value["params"]["sparse_vectors"] = json!({
            "sparse": { "index": { "memory": "cold", "wand_pruning": null } }
        });
        load_value(&value).expect("an explicit sparse index config must be accepted");
    }

    #[test]
    fn requires_every_named_vector_to_be_explicit() {
        let mut value = valid_config_json();
        value["params"]["vectors"]["sparse_ish"] = json!({
            "size": 256,
            "distance": "Dot",
            "multivector_config": null,
            // datatype omitted — structural (storage element width), so required
        });

        let err = load_value(&value).expect_err("second vector missing datatype must be rejected");
        assert!(
            format!("{err:#}").contains("params.vectors.sparse_ish.datatype"),
            "got: {err:#}",
        );
    }

    // ----- The full fingerprint -----

    #[test]
    fn fingerprint_is_stable_across_key_order() {
        let loaded = load_value(&valid_config_json()).unwrap();

        // Re-serialise with a different key order. serde_json is built with
        // `preserve_order`, so this genuinely changes the document's byte order.
        let reordered = json!({
            "quantization_config": valid_config_json()["quantization_config"],
            "wal_config": valid_config_json()["wal_config"],
            "optimizer_config": valid_config_json()["optimizer_config"],
            "hnsw_config": valid_config_json()["hnsw_config"],
            "params": valid_config_json()["params"],
        });
        let reordered = load_value(&reordered).unwrap();

        assert_eq!(
            loaded.fingerprint, reordered.fingerprint,
            "fingerprint must not depend on document key order",
        );
    }

    #[test]
    fn fingerprint_ignores_max_indexing_threads() {
        // Exempt from mismatch_requires_rebuild, and host-dependent. Including it would
        // cause false mismatches between build box and serving nodes.
        let base = load_value(&valid_config_json()).unwrap();

        let mut value = valid_config_json();
        value["hnsw_config"]["max_indexing_threads"] = json!(64);
        let changed = load_value(&value).unwrap();

        assert_eq!(
            base.fingerprint, changed.fingerprint,
            "max_indexing_threads must not affect the fingerprint",
        );
    }

    /// Two spellings of one placement must hash identically.
    ///
    /// `mismatch_requires_rebuild` compares the *resolved* placement precisely so that
    /// migrating a config from `on_disk` to `memory` does not rewrite every segment. If the
    /// fingerprint hashed the raw fields instead, `verify-config` would report a rebuild
    /// where the optimizer plans none — a false alarm on every legacy config.
    #[test]
    fn equivalent_placement_spellings_hash_identically() {
        let modern = load_value(&valid_config_json()).unwrap();

        // on_disk: true resolves to `cold`; hnsw on_disk: false resolves to `cached`, the
        // same as the fixture's explicit-null default.
        let mut legacy = valid_config_json();
        remove_path(&mut legacy, "params.vectors.dense.memory");
        legacy["params"]["vectors"]["dense"]["on_disk"] = json!(true);
        remove_path(&mut legacy, "hnsw_config.memory");
        legacy["hnsw_config"]["on_disk"] = json!(false);
        remove_path(&mut legacy, "params.sparse_vectors.sparse.index.memory");
        legacy["params"]["sparse_vectors"]["sparse"]["index"]["on_disk"] = json!(true);
        // on_disk_payload: true resolves to `cold`, matching the fixture's payload placement.
        remove_path(&mut legacy, "params.payload");
        legacy["params"]["on_disk_payload"] = json!(true);
        let legacy = load_value(&legacy).unwrap();

        assert_eq!(
            modern.fingerprint, legacy.fingerprint,
            "the same resolved placement must hash identically in both spellings",
        );
        assert_eq!(modern.part_fingerprint, legacy.part_fingerprint);
    }

    /// Each arm of `HnswConfig::mismatch_requires_rebuild` must move the fingerprint.
    ///
    /// If one of these ever stops changing the hash, `verify-config` would pass on a
    /// config that provokes a full segment rebuild — the exact failure this tool exists
    /// to prevent.
    #[test]
    fn fingerprint_changes_for_every_rebuild_triggering_hnsw_field() {
        let base = load_value(&valid_config_json()).unwrap().fingerprint;

        let perturbations: &[(&str, Value)] = &[
            ("m", json!(32)),
            ("ef_construct", json!(256)),
            ("full_scan_threshold", json!(20000)),
            ("payload_m", json!(16)),
            // The fixture's `memory: null` resolves to `cached`; `cold` is a different
            // resolved placement, which the mismatch compares directly.
            ("memory", json!("cold")),
            ("inline_storage", json!(true)),
        ];

        for (field, new_value) in perturbations {
            let mut value = valid_config_json();
            value["hnsw_config"][*field] = new_value.clone();
            let changed = load_value(&value)
                .unwrap_or_else(|err| panic!("perturbing {field} produced invalid config: {err:#}"))
                .fingerprint;

            assert_ne!(
                base, changed,
                "changing hnsw_config.{field} must change the fingerprint",
            );
        }
    }

    #[test]
    fn fingerprint_changes_for_quantization_and_placement() {
        let base = load_value(&valid_config_json()).unwrap().fingerprint;

        let mut bits = valid_config_json();
        bits["quantization_config"]["turbo"]["bits"] = json!("bits2");
        assert_ne!(
            base,
            load_value(&bits).unwrap().fingerprint,
            "quantization bit size must change the fingerprint",
        );

        let mut quantization_memory = valid_config_json();
        quantization_memory["quantization_config"]["turbo"]["memory"] = json!("cached");
        assert_ne!(
            base,
            load_value(&quantization_memory).unwrap().fingerprint,
            "quantization placement must change the fingerprint",
        );

        // The fixture pins the dense vector to `cold` (on disk); dropping the placement to
        // its unstated form changes what the optimizer would compare, and must move the hash.
        let mut vector_placement = valid_config_json();
        vector_placement["params"]["vectors"]["dense"]["memory"] = json!(null);
        assert_ne!(
            base,
            load_value(&vector_placement).unwrap().fingerprint,
            "per-vector placement must change the fingerprint",
        );

        // `cached` resolves to the in-RAM-mmap storage type, flipping the on-disk-ness the
        // mismatch optimizer compares.
        let mut payload_placement = valid_config_json();
        payload_placement["params"]["payload"]["memory"] = json!("cached");
        assert_ne!(
            base,
            load_value(&payload_placement).unwrap().fingerprint,
            "payload placement must change the fingerprint",
        );

        // Sparse placement is compared as the full resolved placement, and wand_pruning is
        // baked into every built segment; both must move the hash.
        let mut sparse_placement = valid_config_json();
        sparse_placement["params"]["sparse_vectors"]["sparse"]["index"]["memory"] = json!("cached");
        assert_ne!(
            base,
            load_value(&sparse_placement).unwrap().fingerprint,
            "sparse index placement must change the fingerprint",
        );

        let mut wand = valid_config_json();
        wand["params"]["sparse_vectors"]["sparse"]["index"]["wand_pruning"] = json!(false);
        assert_ne!(
            base,
            load_value(&wand).unwrap().fingerprint,
            "wand_pruning is baked into built segments and must change the fingerprint",
        );
    }

    #[test]
    fn fingerprint_changes_for_topology_and_budget() {
        let base = load_value(&valid_config_json()).unwrap().fingerprint;

        let mut shards = valid_config_json();
        shards["params"]["shard_number"] = json!(8);
        assert_ne!(
            base,
            load_value(&shards).unwrap().fingerprint,
            "shard_number must change the fingerprint",
        );

        let mut ring_scale = valid_config_json();
        ring_scale["params"]["hash_ring_shard_scale"] = json!(500);
        assert_ne!(
            base,
            load_value(&ring_scale).unwrap().fingerprint,
            "hash_ring_shard_scale must change the fingerprint",
        );

        // Optional in the gate, but still hashed (stated or defaulted), like the placements.
        let mut segments = valid_config_json();
        segments["optimizer_config"]["default_segment_number"] = json!(16);
        assert_ne!(
            base,
            load_value(&segments).unwrap().fingerprint,
            "default_segment_number must change the fingerprint",
        );

        let mut size = valid_config_json();
        size["optimizer_config"]["max_segment_size"] = json!(1_000_000);
        assert_ne!(
            base,
            load_value(&size).unwrap().fingerprint,
            "max_segment_size must change the fingerprint",
        );
    }

    /// The resolved optimizer thresholds shape the built bytes (Plain-vs-HNSW, storage type,
    /// deferred-id boundary), so editing them between plan and build must move the
    /// fingerprint — otherwise a mid-build edit produces mismatched segments the freeze is
    /// supposed to catch.
    #[test]
    fn fingerprint_changes_for_optimizer_thresholds() {
        let base = load_value(&valid_config_json()).unwrap().fingerprint;

        let mut indexing = valid_config_json();
        indexing["optimizer_config"]["indexing_threshold"] = json!(0); // disabled -> usize::MAX
        assert_ne!(
            base,
            load_value(&indexing).unwrap().fingerprint,
            "indexing_threshold must change the fingerprint (Plain-vs-HNSW + deferred id)",
        );

        let mut memmap = valid_config_json();
        memmap["optimizer_config"]["memmap_threshold"] = json!(1);
        assert_ne!(
            base,
            load_value(&memmap).unwrap().fingerprint,
            "memmap_threshold must change the fingerprint (storage type)",
        );

        let mut prevent = valid_config_json();
        prevent["optimizer_config"]["prevent_unoptimized"] = json!(true);
        assert_ne!(
            base,
            load_value(&prevent).unwrap().fingerprint,
            "prevent_unoptimized must change the fingerprint (deferred-id boundary)",
        );

        // Equivalence, not just sensitivity: `memmap_threshold` unset and `0` both resolve to
        // "disabled" (usize::MAX), so they must hash identically — no false rebuild.
        let mut unset = valid_config_json();
        unset["optimizer_config"]["memmap_threshold"] = serde_json::Value::Null;
        let mut zero = valid_config_json();
        zero["optimizer_config"]["memmap_threshold"] = json!(0);
        assert_eq!(
            load_value(&unset).unwrap().fingerprint,
            load_value(&zero).unwrap().fingerprint,
            "unset and 0 memmap_threshold both mean disabled and must hash identically",
        );
    }

    /// The payload-index schema is part of the plan-frozen surface.
    ///
    /// Not because the cluster rebuilds segments on a schema change — a field index can be
    /// rebuilt without rewriting segments — but because `build` bakes the declared indexes
    /// into every segment: two build nodes disagreeing on the schema would produce one shard
    /// with inconsistent segments, and the ones missing an index push an index build onto
    /// the serving filesystem at load, the exact cost this tool exists to avoid.
    #[test]
    fn fingerprint_covers_the_payload_index_schema() {
        let config: CollectionConfigInternal = serde_json::from_value(valid_config_json()).unwrap();

        let schema = |body: &str| -> crate::payload_index::PayloadIndexSchema {
            crate::payload_index::PayloadIndexSchema::from_fields(
                serde_json::from_str(body).unwrap(),
            )
        };

        let none = fingerprint(&config, None).unwrap();
        let cold = schema(r#"{"dump": {"type": "keyword", "memory": "cold"}}"#);
        let pinned = schema(r#"{"dump": {"type": "keyword", "memory": "pinned"}}"#);

        assert_ne!(
            none,
            fingerprint(&config, Some(&cold)).unwrap(),
            "declaring an index must change the fingerprint",
        );
        assert_ne!(
            fingerprint(&config, Some(&cold)).unwrap(),
            fingerprint(&config, Some(&pinned)).unwrap(),
            "changing an index definition must change the fingerprint",
        );
    }

    /// Placement spellings in the payload-index schema hash resolved, like every other
    /// placement: `on_disk: true` and `memory: "cold"` are one placement, and the server's
    /// `schema_transition::classify` treats them as identical — no rebuild — so the
    /// fingerprint (and with it `verify-config`) must not tell them apart either.
    #[test]
    fn equivalent_payload_index_spellings_hash_identically() {
        let config: CollectionConfigInternal = serde_json::from_value(valid_config_json()).unwrap();

        let schema = |body: &str| -> crate::payload_index::PayloadIndexSchema {
            crate::payload_index::PayloadIndexSchema::from_fields(
                serde_json::from_str(body).unwrap(),
            )
        };

        let memory = schema(r#"{"dump": {"type": "keyword", "memory": "cold"}}"#);
        let legacy = schema(r#"{"dump": {"type": "keyword", "on_disk": true}}"#);
        assert_eq!(
            fingerprint(&config, Some(&memory)).unwrap(),
            fingerprint(&config, Some(&legacy)).unwrap(),
            "on_disk: true and memory: cold are one placement in two spellings",
        );

        // The in-RAM equivalence too: absent == on_disk: false == memory: pinned.
        let bare = schema(r#"{"dump": "keyword"}"#);
        let off = schema(r#"{"dump": {"type": "keyword", "on_disk": false}}"#);
        let pinned = schema(r#"{"dump": {"type": "keyword", "memory": "pinned"}}"#);
        assert_eq!(
            fingerprint(&config, Some(&bare)).unwrap(),
            fingerprint(&config, Some(&off)).unwrap(),
        );
        assert_eq!(
            fingerprint(&config, Some(&off)).unwrap(),
            fingerprint(&config, Some(&pinned)).unwrap(),
        );

        // But only the placement is resolved — a structural difference still moves the hash.
        let tenant =
            schema(r#"{"dump": {"type": "keyword", "memory": "cold", "is_tenant": true}}"#);
        assert_ne!(
            fingerprint(&config, Some(&memory)).unwrap(),
            fingerprint(&config, Some(&tenant)).unwrap(),
            "is_tenant is not a placement and must stay strict",
        );
    }

    // ----- The part fingerprint -----

    #[test]
    fn the_two_fingerprints_are_distinct_and_well_formed() {
        let loaded = load_value(&valid_config_json()).unwrap();

        assert_eq!(loaded.part_fingerprint.len(), 64, "sha256 hex is 64 chars");
        assert_ne!(
            loaded.fingerprint, loaded.part_fingerprint,
            "the two surfaces must not hash to the same value, or one is being computed twice",
        );
    }

    #[test]
    fn part_fingerprint_is_stable_across_key_order() {
        let loaded = load_value(&valid_config_json()).unwrap();

        let reordered = json!({
            "quantization_config": valid_config_json()["quantization_config"],
            "wal_config": valid_config_json()["wal_config"],
            "optimizer_config": valid_config_json()["optimizer_config"],
            "hnsw_config": valid_config_json()["hnsw_config"],
            "params": valid_config_json()["params"],
        });
        let reordered = load_value(&reordered).unwrap();

        assert_eq!(
            loaded.part_fingerprint, reordered.part_fingerprint,
            "part fingerprint must not depend on document key order",
        );
    }

    /// The whole point of the narrow surface: retuning the index must not invalidate a scatter.
    ///
    /// Each of these fields costs hours and tens of terabytes to re-scatter for, and none of them
    /// changes a single byte of a part file. If one ever starts moving this hash, tuning
    /// `ef_construct` goes back to being a full re-scatter — so this test is the guard on that.
    #[test]
    fn part_fingerprint_ignores_everything_a_part_does_not_depend_on() {
        let base = load_value(&valid_config_json()).unwrap().part_fingerprint;

        let perturbations: &[(&str, &str, Value)] = &[
            ("hnsw_config", "m", json!(48)),
            ("hnsw_config", "ef_construct", json!(256)),
            ("hnsw_config", "full_scan_threshold", json!(20000)),
            ("hnsw_config", "payload_m", json!(0)),
            ("hnsw_config", "memory", json!("cold")),
            ("hnsw_config", "inline_storage", json!(true)),
            ("hnsw_config", "max_indexing_threads", json!(32)),
            ("optimizer_config", "max_segment_size", json!(9_000_000)),
            ("optimizer_config", "default_segment_number", json!(16)),
            ("optimizer_config", "indexing_threshold", json!(50000)),
            ("optimizer_config", "memmap_threshold", json!(20000)),
            ("wal_config", "wal_capacity_mb", json!(64)),
        ];

        for (section, field, new_value) in perturbations {
            let mut value = valid_config_json();
            value[*section][*field] = new_value.clone();
            let changed = load_value(&value)
                .unwrap_or_else(|err| {
                    panic!("perturbing {section}.{field} produced invalid config: {err:#}")
                })
                .part_fingerprint;

            assert_eq!(
                base, changed,
                "{section}.{field} must NOT affect the part fingerprint — changing it would \
                 force a re-scatter for no reason",
            );
        }

        // Storage placement and payload residency shape the built segment, not the parts.
        let mut payload_placement = valid_config_json();
        payload_placement["params"]["payload"]["memory"] = json!("cached");
        assert_eq!(
            base,
            load_value(&payload_placement).unwrap().part_fingerprint,
            "payload placement must not affect the part fingerprint",
        );

        let mut vector_placement = valid_config_json();
        vector_placement["params"]["vectors"]["dense"]["memory"] = json!(null);
        assert_eq!(
            base,
            load_value(&vector_placement).unwrap().part_fingerprint,
            "a vector's placement must not affect the part fingerprint",
        );

        // Distance is applied at build time (`NamedVectors::preprocess`), not scatter time — a
        // part holds the raw vector either way, so an existing scatter is still usable.
        let mut distance = valid_config_json();
        distance["params"]["vectors"]["dense"]["distance"] = json!("Dot");
        assert_eq!(
            base,
            load_value(&distance).unwrap().part_fingerprint,
            "distance must not affect the part fingerprint",
        );

        let mut quantization = valid_config_json();
        quantization["quantization_config"]["turbo"]["bits"] = json!("bits2");
        assert_eq!(
            base,
            load_value(&quantization).unwrap().part_fingerprint,
            "quantization must not affect the part fingerprint",
        );
    }

    /// The other direction: anything that makes a part's *records* wrong must move the hash.
    ///
    /// Only routing qualifies. Everything else about a part's composition is recorded in its
    /// manifest and checked as a subset at build time, which is what lets a build drop a vector or
    /// a payload column without re-scattering — see `PartProjection::resolve` (stage 2).
    #[test]
    fn part_fingerprint_changes_only_for_routing() {
        let base = load_value(&valid_config_json()).unwrap().part_fingerprint;

        let mut shard_number = valid_config_json();
        shard_number["params"]["shard_number"] = json!(8);
        assert_ne!(
            base,
            load_value(&shard_number).unwrap().part_fingerprint,
            "shard_number decides routing and must change the part fingerprint",
        );

        let mut sharding_method = valid_config_json();
        sharding_method["params"]["sharding_method"] = json!("custom");
        assert_ne!(
            base,
            load_value(&sharding_method).unwrap().part_fingerprint,
            "sharding_method decides routing and must change the part fingerprint",
        );

        let mut ring_scale = valid_config_json();
        ring_scale["params"]["hash_ring_shard_scale"] = json!(200);
        assert_ne!(
            base,
            load_value(&ring_scale).unwrap().part_fingerprint,
            "hash_ring_shard_scale decides routing and must change the part fingerprint",
        );

        // Composition is a subset question, resolved against the manifest, so none of these may
        // invalidate a scatter. This is the capability, not an oversight.
        let mut size = valid_config_json();
        size["params"]["vectors"]["dense"]["size"] = json!(512);
        assert_eq!(
            base,
            load_value(&size).unwrap().part_fingerprint,
            "a size change is caught by the manifest (dim), not by the fingerprint",
        );

        let mut datatype = valid_config_json();
        datatype["params"]["vectors"]["dense"]["datatype"] = json!("float16");
        assert_eq!(
            base,
            load_value(&datatype).unwrap().part_fingerprint,
            "a datatype change is safe: parts decode at their own recorded width",
        );

        let mut dropped_sparse = valid_config_json();
        dropped_sparse["params"]["sparse_vectors"] = json!(null);
        assert_eq!(
            base,
            load_value(&dropped_sparse).unwrap().part_fingerprint,
            "dropping a sparse vector must NOT force a re-scatter — it is projected away",
        );

        let mut extra_dense = valid_config_json();
        extra_dense["params"]["vectors"]["second"] = json!({
            "size": 256,
            "distance": "Dot",
            "memory": "cold",
            "datatype": null,
            "multivector_config": null,
        });
        assert_eq!(
            base,
            load_value(&extra_dense).unwrap().part_fingerprint,
            "asking for a vector the parts lack is refused by the manifest, by name",
        );
    }

    /// `id_column` and `id_format` live in the mapping, and both decide point identity.
    ///
    /// They were hashed nowhere before this: changing `id_format` and resuming a scatter left
    /// earlier files routed by one scheme and later ones by another, silently.
    #[test]
    fn part_fingerprint_covers_the_mapping_fields_that_decide_routing() {
        let config: CollectionConfigInternal = serde_json::from_value(valid_config_json()).unwrap();

        let mapping = |id_column: &str, id_format: &str| {
            serde_json::from_value::<crate::parquet_source::ParquetMapping>(json!({
                "id_column": id_column,
                "id_format": id_format,
                "dense_vectors": { "dense": "dense_embedding" },
                "payload_columns": ["url"],
            }))
            .unwrap()
        };

        let base = mapping("id", "urn_uuid");
        let baseline = part_fingerprint(&config, Some(&base)).unwrap();

        let renamed = mapping("doc_id", "urn_uuid");
        assert_ne!(
            baseline,
            part_fingerprint(&config, Some(&renamed)).unwrap(),
            "id_column decides point identity and must change the part fingerprint",
        );

        let reformatted = mapping("id", "integer");
        assert_ne!(
            baseline,
            part_fingerprint(&config, Some(&reformatted)).unwrap(),
            "id_format decides point identity and must change the part fingerprint",
        );

        // Payload columns and vector mappings are manifest concerns, not routing.
        let recolumned = serde_json::from_value::<crate::parquet_source::ParquetMapping>(json!({
            "id_column": "id",
            "id_format": "urn_uuid",
            "dense_vectors": { "dense": "dense_embedding" },
            "payload_columns": ["url", "dump", "text"],
        }))
        .unwrap();
        assert_eq!(
            baseline,
            part_fingerprint(&config, Some(&recolumned)).unwrap(),
            "changing payload_columns must NOT invalidate a scatter",
        );

        // And no mapping at all (JSONL input) is a distinct, stable value.
        assert_ne!(baseline, part_fingerprint(&config, None).unwrap());
    }

    /// A name may be both a dense and a sparse vector, and the provenance check must not confuse
    /// them.
    ///
    /// Qdrant validates nothing here: `params.vectors` and `params.sparse_vectors` are separate
    /// maps, so `dense` can appear in both. An earlier version of `wanted_fields` merged their
    /// source columns into one map keyed by name, so the sparse entry overwrote the dense one and
    /// the dense vector was checked against the sparse column — a spurious failure at best, and a
    /// missed remap at worst, in the very check that exists to catch silent corruption.
    #[test]
    fn a_name_used_for_both_dense_and_sparse_keeps_its_own_source_column() {
        use crate::partfile::{DenseSpec, PartHeader, PartManifest, PartProjection, SparseSpec};

        let mut value = valid_config_json();
        value["params"]["sparse_vectors"] =
            json!({ "dense": { "index": { "memory": "cold", "wand_pruning": null } } });
        let loaded = load_value(&value).expect("a shared name is accepted by the config");

        let mapping: crate::parquet_source::ParquetMapping = serde_json::from_value(json!({
            "id_column": "id",
            "id_format": "urn_uuid",
            "dense_vectors": { "dense": "dense_column" },
            "sparse_vectors": { "dense": { "column": "sparse_column" } },
            "payload_columns": [],
        }))
        .unwrap();

        let wanted = crate::build::wanted_fields_for_test(&loaded, Some(&mapping));
        assert_eq!(
            wanted.dense_sources.get("dense").map(String::as_str),
            Some("dense_column"),
        );
        assert_eq!(
            wanted.sparse_sources.get("dense").map(String::as_str),
            Some("sparse_column"),
        );

        // A part that carries both, each from its own column, must resolve without complaint.
        let head = PartHeader {
            part_fingerprint: loaded.part_fingerprint.clone(),
            file_id: "f".to_string(),
            source_path: "x.parquet".to_string(),
            shard_id: 0,
            manifest: PartManifest {
                dense: std::collections::BTreeMap::from([(
                    "dense".to_string(),
                    DenseSpec {
                        encoding: crate::dense_codec::DenseEncoding::F32,
                        dim: 768,
                        source: Some("dense_column".to_string()),
                    },
                )]),
                sparse: std::collections::BTreeMap::from([(
                    "dense".to_string(),
                    SparseSpec {
                        source: Some("sparse_column".to_string()),
                    },
                )]),
                payload: Some(vec![]),
            },
        };

        PartProjection::resolve(&head, &wanted, std::path::Path::new("p"))
            .expect("a name shared between dense and sparse must not trip the provenance check");
    }

    // ----- Hard limits and Qdrant's own validation -----

    /// Qdrant's own `#[validate]` rules must be enforced, not merely warned about.
    ///
    /// Delegating to `Validate::validate` means the tool inherits new constraints whenever
    /// Qdrant adds them, instead of drifting behind a hand-maintained copy.
    #[test]
    fn enforces_qdrants_declared_validation_rules() {
        // ef_construct has `#[validate(range(min = 4))]`.
        let mut value = valid_config_json();
        value["hnsw_config"]["ef_construct"] = json!(1);
        let err = load_value(&value).unwrap_err();
        assert!(format!("{err:#}").contains("ef_construct"), "got: {err:#}",);

        // deleted_threshold is constrained to 0.0..=1.0.
        let mut value = valid_config_json();
        value["optimizer_config"]["deleted_threshold"] = json!(5.0);
        let err = load_value(&value).unwrap_err();
        assert!(
            format!("{err:#}").contains("deleted_threshold"),
            "got: {err:#}",
        );

        // vacuum_min_vector_number has `#[validate(range(min = 100))]`.
        let mut value = valid_config_json();
        value["optimizer_config"]["vacuum_min_vector_number"] = json!(1);
        let err = load_value(&value).unwrap_err();
        assert!(
            format!("{err:#}").contains("vacuum_min_vector_number"),
            "got: {err:#}",
        );

        // Vector size is capped at 65536.
        let mut value = valid_config_json();
        value["params"]["vectors"]["dense"]["size"] = json!(70_000);
        let err = load_value(&value).unwrap_err();
        assert!(format!("{err:#}").contains("size"), "got: {err:#}");

        // hash_ring_shard_scale is capped at 100_000 (`MAX_HASH_RING_SHARD_SCALE`).
        let mut value = valid_config_json();
        value["params"]["hash_ring_shard_scale"] = json!(200_000);
        let err = load_value(&value).unwrap_err();
        assert!(
            format!("{err:#}").contains("hash_ring_shard_scale"),
            "got: {err:#}",
        );

        // `pinned` placement is rejected for dense vector storage by Qdrant's own validator.
        let mut value = valid_config_json();
        value["params"]["vectors"]["dense"]["memory"] = json!("pinned");
        let err = load_value(&value).unwrap_err();
        assert!(format!("{err:#}").contains("memory"), "got: {err:#}");
    }

    /// A `max_segment_size` no segment could ever address must be rejected.
    ///
    /// Points are addressed by `PointOffsetType` (`u32`), so one segment holds at most
    /// `u32::MAX` vectors. `max_segment_size` is only range-validated as `min = 1`, so
    /// nothing in Qdrant stops a wildly oversized value — it just produces a target the
    /// builder can never hit.
    #[test]
    fn rejects_max_segment_size_beyond_what_a_segment_can_address() {
        // 768 dims * 4 bytes * u32::MAX = ~12.6 PiB. Ask for more than that.
        let addressable_kb = 768u128 * 4 * u128::from(u32::MAX) / 1024;

        let mut value = valid_config_json();
        value["optimizer_config"]["max_segment_size"] = json!(addressable_kb as u64 + 1);
        let err = load_value(&value).unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("structurally impossible") && message.contains("max_segment_size"),
            "got: {message}",
        );

        // Exactly at the limit is allowed: absurd, but not impossible.
        let mut value = valid_config_json();
        value["optimizer_config"]["max_segment_size"] = json!(addressable_kb as u64);
        load_value(&value).expect("exactly at the addressable limit must be accepted");
    }

    /// A narrower vector lowers the addressable ceiling, so the check is per-vector.
    #[test]
    fn addressable_limit_accounts_for_vector_width() {
        // uint8 at 16 dims: 16 bytes/vector * u32::MAX = 64 GiB addressable.
        let mut value = valid_config_json();
        value["params"]["vectors"]["dense"]["size"] = json!(16);
        value["params"]["vectors"]["dense"]["datatype"] = json!("uint8");

        // 16 dims at 1 byte per element (uint8) = 16 bytes per vector.
        let addressable_kb = 16u128 * u128::from(u32::MAX) / 1024;
        value["optimizer_config"]["max_segment_size"] = json!(addressable_kb as u64 + 1);

        let err = load_value(&value).unwrap_err();
        assert!(
            format!("{err:#}").contains("structurally impossible"),
            "a narrow uint8 vector must lower the ceiling, got: {err:#}",
        );
    }

    // ----- The merge-optimizer model behind the segment band -----

    #[test]
    fn merge_safe_floor_is_half_of_max_segment_size() {
        let loaded = load_value(&valid_config_json()).unwrap();

        let max = 5_000_000u64 * 1024;
        assert_eq!(loaded.max_segment_size_bytes(), Some(max));
        assert_eq!(loaded.merge_safe_min_segment_bytes(), Some(max / 2));
        assert_eq!(loaded.target_segment_band_bytes(), Some((max / 2, max)));
    }

    /// Replicates `MergeOptimizer::plan_optimizations`' batching rule to prove the floor.
    ///
    /// This is a model of the merge optimizer's batching, not a call into it, so it is only as
    /// good as the reading it encodes. It exists to pin the *reasoning* behind
    /// `merge_safe_min_segment_bytes`: that a merge needs two segments whose combined size is
    /// under the threshold, so equal segments at or above half the threshold can never pair.
    /// The empirical check is the tier-1 assertion that a restored shard reports zero queued
    /// optimizations.
    fn merge_would_be_planned(segment_sizes: &[u64], threshold: u64) -> bool {
        let mut sorted = segment_sizes.to_vec();
        sorted.sort_unstable();

        let batch_len = sorted
            .iter()
            .scan(0u64, |sum, &size| {
                *sum += size;
                (*sum < threshold).then_some(())
            })
            .count();

        batch_len >= 2
    }

    #[test]
    fn segments_at_or_above_the_floor_never_pair() {
        let threshold = 1_000u64;
        let floor = threshold.div_ceil(2);

        // Exactly at the floor: two of them sum to >= threshold, so no batch forms.
        assert!(!merge_would_be_planned(&[floor; 2], threshold));
        assert!(!merge_would_be_planned(&[floor; 50], threshold));

        // Above the floor: still no pair, however many segments exist.
        assert!(!merge_would_be_planned(&[threshold - 1; 200], threshold));

        // Just below the floor: a pair fits, so a merge is planned. This is the failure the
        // floor prevents, and it is independent of segment count.
        assert!(merge_would_be_planned(&[floor - 1; 2], threshold));
    }

    /// Segment *count* alone does not provoke a merge — only pairable sizes do.
    ///
    /// This is the misreading the floor replaced: `default_segment_number` bounds the outer
    /// loop, but the loop returns immediately when the two smallest segments cannot fit
    /// together, so a shard may hold far more than `default_segment_number` large segments.
    #[test]
    fn many_large_segments_are_left_alone() {
        let threshold = 20 * 1024 * 1024 * 1024u64; // 20 GiB
        let segment = threshold; // build at the ceiling

        // 160 segments -> 3.1 TiB in one shard, far beyond any per-shard "budget", yet the
        // merge optimizer cannot pair any two of them.
        assert!(!merge_would_be_planned(&vec![segment; 160], threshold));
    }
}
