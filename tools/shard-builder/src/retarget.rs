//! Retarget: rewrite built segments' recorded configs to new memory placements.
//!
//! The config gate divides parameters into two classes: those that decide the *bytes* of a
//! built segment (HNSW `m`, quantization, dims, sparse index structure) and those that only
//! decide where the same bytes sit at load time (cold/cached placements). The first class is
//! frozen from `plan` onward. The second class is frozen too — but only because the serving
//! cluster's `ConfigMismatchOptimizer` compares placements and would rebuild every imported
//! segment whose *recorded* config disagrees with the collection config it serves under.
//!
//! This step is the escape hatch that makes the second class cheap to change after the fact:
//! given a document that differs from what the artifact was built under **only in placement**,
//! it rewrites each segment's `segment.json` so the recorded config is byte-for-byte what a
//! fresh build under the new document would have recorded. No data file is touched — every
//! rewritable flip maps within a family of storage types that share one on-disk format:
//!
//! * dense storage: `Mmap` ↔ `InRamMmap` (one mmap file; the latter is pre-fetched on load),
//!   and on the plain appendable segment `ChunkedMmap` ↔ `InRamChunkedMmap`;
//! * the HNSW graph: `hnsw_config.memory` / deprecated `on_disk` — Qdrant's own
//!   `mismatch_requires_rebuild` documents that "data on disk is the same";
//! * payload storage: `Mmap` ↔ `InRamMmap`, same single format;
//! * the sparse index: cold ↔ cached within `SparseIndexType::Mmap`, carried by the raw
//!   `memory` field exactly as the serving optimizer stamps it.
//!
//! Everything else is refused by name. The loud refusals are the safety property: a document
//! that changes `m`, quantization (whose mismatch check is plain equality — the cluster treats
//! even a quantization *placement* flip as a rebuild), a sparse pinned flip
//! (`ImmutableRam` and `Mmap` are different index structures), `inline_storage`, or the
//! vector set itself, cannot be retargeted onto existing bytes.
//!
//! # Why the plan freeze does not move
//!
//! An earlier design note considered splitting the plan-frozen fingerprint so placements were
//! never frozen at all. That would reintroduce the exact hazard the freeze exists to prevent:
//! a placement edited mid-build (or mid-resume) would leave some segments recorded one way and
//! the rest another, silently. So `build` still refuses any config drift, placements included,
//! and this step is the *sanctioned* placement change: it runs over finished artifacts,
//! touches every segment of every shard it is pointed at, and is idempotent.
//!
//! Run it after `build` (or after `assemble` — the empty appendable segment is rewritten too,
//! so `assemble-verify` still passes). The correctness claim — a serving cluster loads the
//! rewritten segments and queues zero optimizations — is proven differentially in
//! `build_e2e_tests`: retarget(built under A → B) must equal a fresh build under B, and the
//! bulk route resolves configs with the serving optimizer's own code.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use collection::optimizers_builder::build_segment_optimizer_config;
use segment::index::sparse_index::sparse_index_config::{
    SPARSE_INDEX_CONFIG_FILE, SparseIndexConfig, SparseIndexType,
};
use segment::segment::Segment;
use segment::types::{
    HnswConfig, Indexes, Memory, PayloadStorageType, QuantizationConfig, SegmentConfig,
    SparseVectorDataConfig, VectorDataConfig, VectorStorageType,
};

use crate::config::LoadedConfig;

/// How a retarget run is tuned, as opposed to what it operates on.
pub struct RetargetOptions {
    /// Report what would change without writing anything.
    pub dry_run: bool,
}

/// What a retarget run did (or, under `--dry-run`, would do).
#[derive(Debug, Default)]
pub struct RetargetReport {
    pub shards: usize,
    pub segments_examined: usize,
    pub segments_rewritten: usize,
    /// One human-readable line per changed field, prefixed with `shard/segment`.
    pub changes: Vec<String>,
}

/// Rewrite every built segment under `out` to the placements `config` declares.
///
/// Fails on the first segment whose recorded config differs from the document in anything
/// *other* than placement — and having failed, has written either nothing (the failing
/// segment is checked before any write) or a prefix of segments that are each individually
/// valid and re-runnable, because the rewrite is idempotent.
pub fn run(config: &LoadedConfig, out: &Path, options: &RetargetOptions) -> Result<RetargetReport> {
    let targets = PlacementTargets::resolve(config);
    let mut report = RetargetReport::default();

    let shards = crate::assemble::discover_shards(out)?;
    for (shard_id, shard_dir) in shards {
        report.shards += 1;
        let segments_dir = shard_dir.join(shard::files::SEGMENTS_PATH);
        if !segments_dir.is_dir() {
            bail!(
                "{} has no segments directory; retarget runs on built shards",
                shard_dir.display(),
            );
        }

        // Refuse a document that changes *routing*. retarget rewrites only placement, which the
        // fields inside `segment.json` capture; routing (shard_number, sharding_method,
        // hash_ring_shard_scale, id column/format) lives in neither `segment.json` nor the
        // placement targets, so without this check a routing edit would be silently ignored —
        // retarget would report success while the artifact still routes by the *built* ring, and
        // an operator creating the collection under the new scale would misroute every point. The
        // build stamped both fingerprints; `part` is routing alone, so an unchanged `part` proves
        // the edit is placement-only. (Legacy artifacts predating `part` leave it empty; routing
        // then cannot be verified and the check is skipped.)
        let fingerprint_path = shard_dir.join(crate::build::BUILD_FINGERPRINT_FILE);
        let built_fingerprint = match fs_err::read_to_string(&fingerprint_path) {
            Ok(body) => crate::build::BuildFingerprint::parse(&body),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => bail!(
                "{} has no build fingerprint ({}); retarget runs on shards produced by this \
                 tool's build phase. Re-run build, or (for a hand-staged artifact) retarget is \
                 not supported.",
                shard_dir.display(),
                crate::build::BUILD_FINGERPRINT_FILE,
            ),
            Err(err) => return Err(err.into()),
        };
        if built_fingerprint.part.is_empty() {
            // A legacy artifact (built before routing was fingerprinted) or a torn fingerprint
            // write. Retarget cannot verify the routing is unchanged, and it *refreshes* the full
            // fingerprint below — which folds in routing — so proceeding would let a document with
            // different routing assemble cleanly onto data routed the old way, silently
            // misrouting every point. Refuse rather than launder an unverifiable routing claim.
            bail!(
                "{} has no routing fingerprint (built by an older tool version, or its build \
                 fingerprint is corrupt); retarget cannot verify the document does not change \
                 routing. Rebuild the shard with the current tool to retarget it, or assemble it \
                 directly without retargeting.",
                shard_dir.display(),
            );
        }
        if built_fingerprint.part != config.part_fingerprint {
            bail!(
                "{} was built with different routing than the document given to retarget:\n  \
                 built part fingerprint:    {}\n  retarget part fingerprint: {}\n\nRetarget only \
                 rewrites placement (cold/cached); it cannot change routing (shard_number, \
                 sharding_method, hash_ring_shard_scale, id column/format), which decides which \
                 point lands in which shard. A routing change needs a fresh scatter + plan + \
                 build. Re-run retarget with the build's document, changing only placement.",
                shard_dir.display(),
                built_fingerprint.part,
                config.part_fingerprint,
            );
        }

        for dir in crate::assemble::segment_dirs(&segments_dir)? {
            report.segments_examined += 1;
            let mut state = Segment::load_state(&dir)
                .with_context(|| format!("cannot read segment state in {}", dir.display()))?;

            let label = format!(
                "shard_{shard_id}/{}",
                dir.file_name().unwrap_or_default().to_string_lossy(),
            );
            let mut changes = retarget_config(&targets, &mut state.config)
                .with_context(|| format!("segment {}", dir.display()))?;

            // A persisted sparse index keeps its own `sparse_index_config.json` in the index
            // directory, and the loader *prefers* that file over the segment config for
            // Mmap/ImmutableRam indexes. Rewriting only `segment.json` would leave the two
            // disagreeing, so each persisted index's file is synced to the (now-updated) recorded
            // `memory`. Planned here without writing so the writes below can be ordered safely.
            let sparse_syncs = plan_sparse_index_syncs(&dir, &state.config)
                .with_context(|| format!("segment {}", dir.display()))?;
            changes.extend(sparse_syncs.iter().map(|sync| sync.change.clone()));

            if changes.is_empty() {
                continue;
            }
            if !options.dry_run {
                // Order matters for crash-safety. The serving cluster's `ConfigMismatchOptimizer`
                // reads `segment.json`; the loader reads the sparse index file. Writing
                // `segment.json` *first* means a crash between the two writes leaves the optimizer
                // already seeing the retargeted config — so it queues no rebuild — while only the
                // index file's placement lags, which the loader honours harmlessly and a re-run
                // fixes idempotently. The reverse order would leave `segment.json` at the old
                // placement, and the optimizer would rebuild the (possibly huge) segment on the
                // serving filesystem — the exact cost retarget exists to avoid.
                Segment::save_state(&state, &dir)
                    .with_context(|| format!("cannot write segment state in {}", dir.display()))?;
                for sync in &sparse_syncs {
                    sync.apply()?;
                }
            }
            report.segments_rewritten += 1;
            report.changes.extend(
                changes
                    .into_iter()
                    .map(|change| format!("{label}: {change}")),
            );
        }

        // The full fingerprint hashes placements, so a placement rewrite moves it. Refresh the
        // stamped `full` to the retargeted document's, keeping the (verified-equal) `part`, so a
        // later `assemble` under this same document is accepted rather than refused for a
        // fingerprint that still describes the pre-retarget placements. Written after every
        // segment in the shard succeeded; a mid-shard failure leaves the old fingerprint, which
        // correctly still matches the un-retargeted remainder for a fix-and-rerun.
        if !options.dry_run {
            // `part` is guaranteed non-empty and equal to `config.part_fingerprint` here (verified
            // above), so refreshing to the retarget document's fingerprints keeps routing honest
            // while updating `full` for the new placements. Written atomically: a torn write would
            // leave a body that `parse` reads as legacy (empty `part`), which the next retarget
            // would then refuse.
            let refreshed = crate::build::BuildFingerprint {
                full: config.fingerprint.clone(),
                part: config.part_fingerprint.clone(),
            };
            common::fs::atomic_save_json(&fingerprint_path, &refreshed)
                .with_context(|| format!("cannot refresh {}", fingerprint_path.display()))?;
        }
    }

    Ok(report)
}

/// The document's placement decisions, resolved once through the same code path the serving
/// optimizer (and the bulk build route) uses.
struct PlacementTargets {
    payload_storage: PayloadStorageType,
    dense: BTreeMap<String, DenseTarget>,
    sparse: BTreeMap<String, SparseTarget>,
}

struct DenseTarget {
    /// The plain (pre-optimization) config: the structural reference for size, distance,
    /// datatype and multivector. Its `storage_type` is placement-derived and used as the
    /// appendable-segment target.
    plain: VectorDataConfig,
    /// Resolved storage placement (`memory` over the deprecated `on_disk`). `None` means the
    /// document leaves storage to the optimizer's thresholds — which the cluster's mismatch
    /// check then never compares, so the recorded value is left alone.
    placement: Option<Memory>,
    /// The full per-vector HNSW config a fresh build would stamp, placement included.
    hnsw: HnswConfig,
    quantization: Option<QuantizationConfig>,
}

struct SparseTarget {
    /// Structural reference: modifier, datatype, `full_scan_threshold`, `wand_pruning`.
    plain: SparseVectorDataConfig,
    /// The raw explicit `memory` — exactly what `optimized_segment_config` persists into the
    /// built index config (the resolved decision lives in `index_type`; only this field
    /// carries cold-vs-cached).
    raw_memory: Option<Memory>,
    /// Resolved placement, deciding the index-type *class* (pinned → `ImmutableRam`).
    placement: Option<Memory>,
}

impl PlacementTargets {
    fn resolve(config: &LoadedConfig) -> Self {
        let optimizer = build_segment_optimizer_config(
            &config.config.params,
            &config.config.hnsw_config,
            &config.config.quantization_config,
        );

        let dense = optimizer
            .plain_dense_vector_config
            .iter()
            .map(|(name, plain)| {
                let cfg = optimizer
                    .dense_vector
                    .get(name)
                    .expect("optimizer config lists every dense vector in both maps");
                (
                    name.clone(),
                    DenseTarget {
                        plain: plain.clone(),
                        placement: cfg.memory_placement(),
                        hnsw: cfg.hnsw_config,
                        quantization: cfg.quantization_config.clone(),
                    },
                )
            })
            .collect();

        let sparse = optimizer
            .plain_sparse_vector_config
            .iter()
            .map(|(name, plain)| {
                let cfg = optimizer
                    .sparse_vector
                    .get(name)
                    .expect("optimizer config lists every sparse vector in both maps");
                (
                    name.clone(),
                    SparseTarget {
                        plain: *plain,
                        raw_memory: cfg.memory,
                        placement: cfg.memory_placement(),
                    },
                )
            })
            .collect();

        PlacementTargets {
            payload_storage: optimizer.payload_storage_type,
            dense,
            sparse,
        }
    }
}

/// Rewrite one recorded config in place, returning one line per changed field.
///
/// Errors on any difference that is not a placement — those are the structural fields whose
/// change would require rebuilding the segment's bytes, or would make the recorded config
/// diverge from what a fresh build under the document produces.
fn retarget_config(targets: &PlacementTargets, config: &mut SegmentConfig) -> Result<Vec<String>> {
    let mut changes = Vec::new();

    // The vector sets are structural: a vector present in the segment but not the document
    // (or vice versa) cannot be reconciled by rewriting metadata.
    let recorded_dense: Vec<&str> = config.vector_data.keys().map(AsRef::as_ref).collect();
    for name in &recorded_dense {
        if !targets.dense.contains_key(*name) {
            bail!("segment carries dense vector `{name}` that the document does not declare");
        }
    }
    for name in targets.dense.keys() {
        if !config.vector_data.contains_key(name.as_str()) {
            bail!("document declares dense vector `{name}` that the segment does not carry");
        }
    }
    for name in config.sparse_vector_data.keys() {
        if !targets.sparse.contains_key(name.as_str()) {
            bail!("segment carries sparse vector `{name}` that the document does not declare");
        }
    }
    for name in targets.sparse.keys() {
        if !config.sparse_vector_data.contains_key(name.as_str()) {
            bail!("document declares sparse vector `{name}` that the segment does not carry");
        }
    }

    // Payload storage: `Mmap` and `InRamMmap` share one format (on disk either way; the
    // latter is populated into RAM on load), so this flip is always safe.
    let payload_target = targets.payload_storage;
    if config.payload_storage_type != payload_target {
        changes.push(format!(
            "payload storage {:?} -> {payload_target:?}",
            config.payload_storage_type,
        ));
        config.payload_storage_type = payload_target;
    }

    for (name, recorded) in &mut config.vector_data {
        let target = &targets.dense[name.as_str()];
        retarget_dense(name, recorded, target, &mut changes)?;
    }

    for (name, recorded) in &mut config.sparse_vector_data {
        let target = &targets.sparse[name.as_str()];
        retarget_sparse(name, recorded, target, &mut changes)?;
    }

    Ok(changes)
}

/// One persisted sparse index's `sparse_index_config.json` that needs its `memory` rewritten to
/// match the retargeted segment config, computed without writing so the caller controls order.
struct SparseIndexSync {
    path: std::path::PathBuf,
    /// The index config with `memory` already updated, ready to save.
    config: SparseIndexConfig,
    /// Human-readable change line for the report.
    change: String,
}

impl SparseIndexSync {
    fn apply(&self) -> Result<()> {
        self.config
            .save(&self.path)
            .with_context(|| format!("cannot write {}", self.path.display()))
    }
}

/// Compute the sparse index-file rewrites needed so each persisted index's own
/// `sparse_index_config.json` agrees with the (already-retargeted) segment config — **without
/// writing**.
///
/// The loader prefers this file over the segment config for `Mmap`/`ImmutableRam` sparse indexes,
/// so retarget must keep it in step. Only the raw `memory` field is touched — the structural
/// fields (`index_type`, datatype, `full_scan_threshold`) are guaranteed equal by
/// `retarget_sparse`'s refusals, so a fresh build would write the same file except for placement.
/// Idempotent: a file already matching yields no sync. Files absent (`MutableRam`/appendable, or
/// no index built) are skipped.
fn plan_sparse_index_syncs(
    segment_dir: &Path,
    config: &SegmentConfig,
) -> Result<Vec<SparseIndexSync>> {
    let mut syncs = Vec::new();
    for (name, recorded) in &config.sparse_vector_data {
        // Use the server's own path helper rather than reconstructing the directory name —
        // it emits bare `vector_index` for an empty vector name, which a hand-rolled
        // `vector_index-{name}` would miss, silently skipping the sync.
        let config_path = segment::segment_constructor::get_vector_index_path(segment_dir, name)
            .join(SPARSE_INDEX_CONFIG_FILE);
        if !config_path.exists() {
            continue;
        }
        let mut index_config = SparseIndexConfig::load(&config_path)
            .with_context(|| format!("cannot read {}", config_path.display()))?;
        if index_config.memory == recorded.index.memory {
            continue;
        }
        let change = format!(
            "sparse `{name}` index-file memory {:?} -> {:?}",
            index_config.memory, recorded.index.memory,
        );
        index_config.memory = recorded.index.memory;
        syncs.push(SparseIndexSync {
            path: config_path,
            config: index_config,
            change,
        });
    }
    Ok(syncs)
}

fn retarget_dense(
    name: &str,
    recorded: &mut VectorDataConfig,
    target: &DenseTarget,
    changes: &mut Vec<String>,
) -> Result<()> {
    // Structural identity first: these decide the bytes in the vector storage files.
    if recorded.size != target.plain.size {
        bail!(
            "dense vector `{name}`: size {} in the segment vs {} in the document; \
             dimensionality is not retargetable",
            recorded.size,
            target.plain.size,
        );
    }
    if recorded.distance != target.plain.distance {
        bail!(
            "dense vector `{name}`: distance {:?} in the segment vs {:?} in the document; \
             preprocessing differs, so the stored vectors would be wrong",
            recorded.distance,
            target.plain.distance,
        );
    }
    if recorded.datatype != target.plain.datatype {
        bail!(
            "dense vector `{name}`: datatype {:?} in the segment vs {:?} in the document; \
             element width is not retargetable",
            recorded.datatype,
            target.plain.datatype,
        );
    }
    if recorded.multivector_config != target.plain.multivector_config {
        bail!("dense vector `{name}`: multivector config differs; not retargetable",);
    }

    match &recorded.index {
        // A plain (unindexed or appendable) segment has no HNSW config to rewrite; its
        // quantization is the appendable subset of the document's.
        Indexes::Plain {} => {
            let expected = QuantizationConfig::for_appendable_segment(target.quantization.as_ref());
            if recorded.quantization_config != expected {
                bail!(
                    "dense vector `{name}`: quantization differs from the document's; \
                     `QuantizationConfig::mismatch_requires_rebuild` is plain equality, so \
                     even a quantization placement flip means a rebuild, not a retarget",
                );
            }
        }
        Indexes::Hnsw(recorded_hnsw) => {
            require_hnsw_structurally_equal(name, recorded_hnsw, &target.hnsw)?;
            if *recorded_hnsw != target.hnsw {
                changes.push(format!(
                    "dense `{name}` hnsw placement {:?} -> {:?}",
                    recorded_hnsw.memory_placement(),
                    target.hnsw.memory_placement(),
                ));
                recorded.index = Indexes::Hnsw(target.hnsw);
            }

            if recorded.quantization_config != target.quantization {
                bail!(
                    "dense vector `{name}`: quantization differs from the document's; \
                     `QuantizationConfig::mismatch_requires_rebuild` is plain equality, so \
                     even a quantization placement flip means a rebuild, not a retarget",
                );
            }
        }
    }

    if let Some(storage) = dense_storage_target(name, recorded.storage_type, target)?
        && storage != recorded.storage_type
    {
        changes.push(format!(
            "dense `{name}` storage {:?} -> {storage:?}",
            recorded.storage_type,
        ));
        recorded.storage_type = storage;
    }

    Ok(())
}

/// The storage type a fresh build under the document would record, given what is on disk now.
///
/// The recorded storage type names the file format, so the mapping is per family:
/// single-file (`Mmap`/`InRamMmap`, what indexed segments get) and chunked
/// (`ChunkedMmap`/`InRamChunkedMmap`, what the plain appendable segment gets). Within a
/// family the bytes are identical and only load behaviour differs. `None` means leave it.
fn dense_storage_target(
    name: &str,
    recorded: VectorStorageType,
    target: &DenseTarget,
) -> Result<Option<VectorStorageType>> {
    match recorded {
        VectorStorageType::Mmap | VectorStorageType::InRamMmap => match target.placement {
            Some(Memory::Cold) => Ok(Some(VectorStorageType::Mmap)),
            // Mirrors `optimized_segment_config`: the in-RAM single-file form exists only
            // behind this flag; without it a fresh build would fall back to a chunked
            // (different-format) storage, which cannot be reached by rewriting metadata.
            Some(Memory::Cached | Memory::Pinned) => {
                if common::flags::feature_flags().single_file_mmap_vector_storage {
                    Ok(Some(VectorStorageType::InRamMmap))
                } else {
                    bail!(
                        "dense vector `{name}`: an in-RAM placement needs the \
                         `single_file_mmap_vector_storage` feature flag to stay in the \
                         single-file format this segment was built with",
                    );
                }
            }
            // No explicit placement: a fresh build would decide by thresholds, and the
            // cluster's mismatch check skips storage entirely when nothing is requested.
            None => Ok(None),
        },
        // The appendable family: recompute exactly as the plain config does.
        VectorStorageType::ChunkedMmap | VectorStorageType::InRamChunkedMmap => {
            let placement = target.placement.unwrap_or(Memory::Cached);
            Ok(Some(VectorStorageType::appendable_from_memory(placement)))
        }
        // Legacy RAM storage and data-less placeholders carry nothing to retarget.
        VectorStorageType::Memory | VectorStorageType::Empty => Ok(None),
    }
}

/// The HNSW fields whose change means different graph bytes — everything
/// `HnswConfig::mismatch_requires_rebuild` compares except the placement.
fn require_hnsw_structurally_equal(
    name: &str,
    recorded: &HnswConfig,
    target: &HnswConfig,
) -> Result<()> {
    let fields = [
        ("m", recorded.m == target.m),
        ("ef_construct", recorded.ef_construct == target.ef_construct),
        (
            "full_scan_threshold",
            recorded.full_scan_threshold == target.full_scan_threshold,
        ),
        ("payload_m", recorded.payload_m == target.payload_m),
        (
            "inline_storage",
            recorded.inline_storage == target.inline_storage,
        ),
    ];
    for (field, equal) in fields {
        if !equal {
            bail!(
                "dense vector `{name}`: hnsw_config.{field} differs from the document; \
                 the graph would have to be rebuilt, which retarget never does — \
                 change it with a re-plan and re-build instead",
            );
        }
    }
    Ok(())
}

fn retarget_sparse(
    name: &str,
    recorded: &mut SparseVectorDataConfig,
    target: &SparseTarget,
    changes: &mut Vec<String>,
) -> Result<()> {
    if recorded.storage_type != target.plain.storage_type {
        bail!(
            "sparse vector `{name}`: storage type {:?} vs {:?}; not retargetable",
            recorded.storage_type,
            target.plain.storage_type,
        );
    }
    if recorded.modifier != target.plain.modifier {
        bail!(
            "sparse vector `{name}`: modifier differs from the document; the stored weights \
             would be wrong, so this is not retargetable",
        );
    }
    if recorded.index.datatype != target.plain.index.datatype {
        bail!("sparse vector `{name}`: index datatype differs; not retargetable");
    }
    if recorded.index.full_scan_threshold != target.plain.index.full_scan_threshold {
        bail!(
            "sparse vector `{name}`: full_scan_threshold differs from the document; \
             a fresh build would record the new value, so re-plan and re-build instead",
        );
    }

    // `wand_pruning` is read only by the mutable RAM index; built (immutable) indexes have it
    // cleared on every route. On a mutable index a recorded `None` is also accepted unchanged:
    // the appendable segment `assemble` creates goes through the edge config, which does not
    // expose `wand_pruning`, so `None` there means "not stamped", not "disabled".
    let wand_pruning_ok = match recorded.index.index_type {
        SparseIndexType::MutableRam => {
            recorded.index.wand_pruning.is_none()
                || recorded.index.wand_pruning == target.plain.index.wand_pruning
        }
        SparseIndexType::ImmutableRam | SparseIndexType::Mmap => {
            recorded.index.wand_pruning.is_none()
        }
    };
    if !wand_pruning_ok {
        bail!(
            "sparse vector `{name}`: wand_pruning differs from the document; it shapes the \
             mutable index's search behaviour and is frozen at plan — re-plan instead",
        );
    }

    // The index-type *class* is structural: `ImmutableRam` (pinned) and `Mmap` are different
    // index structures on disk. Cold vs cached both live in `Mmap` and differ only in the
    // raw `memory` field rewritten below.
    if recorded.index.index_type != SparseIndexType::MutableRam {
        let class = if target.placement == Some(Memory::Pinned) {
            SparseIndexType::ImmutableRam
        } else {
            SparseIndexType::Mmap
        };
        if recorded.index.index_type != class {
            bail!(
                "sparse vector `{name}`: the requested placement needs index type {class:?} \
                 but the segment was built as {:?}; pinned flips change the index structure, \
                 so this needs a re-plan and re-build",
                recorded.index.index_type,
            );
        }
    }

    // The raw explicitly-requested `memory`, exactly as the serving optimizer stamps it.
    if recorded.index.memory != target.raw_memory {
        changes.push(format!(
            "sparse `{name}` memory {:?} -> {:?}",
            recorded.index.memory, target.raw_memory,
        ));
        recorded.index.memory = target.raw_memory;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use segment::types::SegmentState;
    use tempfile::TempDir;

    use super::*;
    use crate::config;

    fn loaded(mutate: impl FnOnce(&mut serde_json::Value)) -> LoadedConfig {
        let mut value = config::tests::valid_config_json();
        mutate(&mut value);
        config::from_str(&value.to_string()).expect("test config must load")
    }

    /// The recorded config a bulk build under the fixture document produces (cold
    /// placements everywhere), built by resolving the fixture and applying the optimized
    /// shapes — mirroring what the e2e suite observes from real segments.
    fn built_config(config: &LoadedConfig) -> SegmentConfig {
        let optimizer = build_segment_optimizer_config(
            &config.config.params,
            &config.config.hnsw_config,
            &config.config.quantization_config,
        );

        let mut segment_config = SegmentConfig {
            vector_data: optimizer.plain_dense_vector_config.clone(),
            sparse_vector_data: optimizer.plain_sparse_vector_config.clone(),
            payload_storage_type: optimizer.payload_storage_type,
        };
        for (name, data) in &mut segment_config.vector_data {
            let cfg = optimizer.dense_vector.get(name).unwrap();
            data.index = Indexes::Hnsw(cfg.hnsw_config);
            data.quantization_config = cfg.quantization_config.clone();
            data.storage_type = match cfg.memory_placement() {
                Some(Memory::Cold) | None => VectorStorageType::Mmap,
                Some(Memory::Cached | Memory::Pinned) => VectorStorageType::InRamMmap,
            };
        }
        for (name, data) in &mut segment_config.sparse_vector_data {
            let cfg = optimizer.sparse_vector.get(name).unwrap();
            data.index.index_type = if cfg.memory_placement() == Some(Memory::Pinned) {
                SparseIndexType::ImmutableRam
            } else {
                SparseIndexType::Mmap
            };
            data.index.memory = cfg.memory;
            data.index.wand_pruning = None;
        }
        segment_config
    }

    fn cached_everywhere(value: &mut serde_json::Value) {
        value["params"]["vectors"]["dense"]["memory"] = serde_json::json!("cached");
        value["params"]["sparse_vectors"]["sparse"]["index"]["memory"] =
            serde_json::json!("cached");
        value["params"]["payload"]["memory"] = serde_json::json!("cached");
        value["hnsw_config"]["memory"] = serde_json::json!("cached");
    }

    #[test]
    fn placement_flips_rewrite_every_surface() {
        let built = built_config(&loaded(|_| {}));
        let retuned = loaded(cached_everywhere);
        let targets = PlacementTargets::resolve(&retuned);

        let mut config = built.clone();
        let changes = retarget_config(&targets, &mut config).unwrap();

        assert_eq!(
            config.payload_storage_type,
            PayloadStorageType::InRamMmap,
            "payload cached must become the in-RAM mmap form",
        );
        let dense = &config.vector_data["dense"];
        assert_eq!(dense.storage_type, VectorStorageType::InRamMmap);
        match &dense.index {
            Indexes::Hnsw(hnsw) => assert_eq!(hnsw.memory, Some(Memory::Cached)),
            other @ Indexes::Plain { .. } => panic!("index must stay HNSW, got {other:?}"),
        }
        let sparse = &config.sparse_vector_data["sparse"];
        assert_eq!(sparse.index.memory, Some(Memory::Cached));
        assert_eq!(
            sparse.index.index_type,
            SparseIndexType::Mmap,
            "cold -> cached must not change the sparse index structure",
        );

        // Four surfaces, four change lines; and the result is exactly a fresh build's config.
        assert_eq!(changes.len(), 4, "unexpected changes: {changes:?}");
        assert_eq!(config, built_config(&retuned));
    }

    #[test]
    fn retargeting_is_idempotent() {
        let retuned = loaded(cached_everywhere);
        let targets = PlacementTargets::resolve(&retuned);

        let mut config = built_config(&loaded(|_| {}));
        retarget_config(&targets, &mut config).unwrap();
        let again = retarget_config(&targets, &mut config).unwrap();
        assert!(again.is_empty(), "second pass changed {again:?}");
    }

    #[test]
    fn an_unchanged_document_changes_nothing() {
        let same = loaded(|_| {});
        let targets = PlacementTargets::resolve(&same);
        let mut config = built_config(&same);
        let before = config.clone();
        let changes = retarget_config(&targets, &mut config).unwrap();
        assert!(changes.is_empty(), "no-op retarget changed {changes:?}");
        assert_eq!(config, before);
    }

    #[test]
    fn structural_hnsw_changes_are_refused_by_name() {
        let retuned = loaded(|value| {
            value["hnsw_config"]["m"] = serde_json::json!(64);
        });
        let targets = PlacementTargets::resolve(&retuned);

        let mut config = built_config(&loaded(|_| {}));
        let err = retarget_config(&targets, &mut config).unwrap_err();
        assert!(
            err.to_string().contains("hnsw_config.m"),
            "the refusal must name the field, got: {err:#}",
        );
    }

    #[test]
    fn a_sparse_pinned_flip_is_refused() {
        let retuned = loaded(|value| {
            value["params"]["sparse_vectors"]["sparse"]["index"]["memory"] =
                serde_json::json!("pinned");
        });
        let targets = PlacementTargets::resolve(&retuned);

        let mut config = built_config(&loaded(|_| {}));
        let err = retarget_config(&targets, &mut config).unwrap_err();
        assert!(
            format!("{err:#}").contains("index structure"),
            "the refusal must explain the structural change, got: {err:#}",
        );
    }

    #[test]
    fn a_quantization_change_is_refused() {
        let retuned = loaded(|value| {
            value["quantization_config"]["turbo"]["memory"] = serde_json::json!("cached");
        });
        let targets = PlacementTargets::resolve(&retuned);

        let mut config = built_config(&loaded(|_| {}));
        let err = retarget_config(&targets, &mut config).unwrap_err();
        assert!(format!("{err:#}").contains("quantization"), "got: {err:#}",);
    }

    #[test]
    fn a_missing_or_extra_vector_is_refused() {
        // Document without the sparse vector: the segment still carries it.
        let dense_only = loaded(|value| {
            value["params"]["sparse_vectors"] = serde_json::Value::Null;
        });
        let targets = PlacementTargets::resolve(&dense_only);
        let mut config = built_config(&loaded(|_| {}));
        let err = retarget_config(&targets, &mut config).unwrap_err();
        assert!(
            format!("{err:#}").contains("sparse vector `sparse`"),
            "got: {err:#}",
        );
    }

    #[test]
    fn omitting_a_placement_leaves_the_recorded_storage_alone() {
        // A document with the dense placement removed entirely: the cluster's mismatch check
        // skips storage when nothing is requested, so retarget must not guess.
        let unplaced = loaded(|value| {
            value["params"]["vectors"]["dense"]["memory"] = serde_json::Value::Null;
        });
        let targets = PlacementTargets::resolve(&unplaced);

        let mut config = built_config(&loaded(|_| {}));
        let before = config.vector_data["dense"].storage_type;
        retarget_config(&targets, &mut config).unwrap();
        assert_eq!(config.vector_data["dense"].storage_type, before);
    }

    #[test]
    fn the_appendable_segment_family_is_rewritten_in_place() {
        // A plain appendable segment, as `assemble` creates it under the cold fixture.
        let cold = loaded(|_| {});
        let optimizer = build_segment_optimizer_config(
            &cold.config.params,
            &cold.config.hnsw_config,
            &cold.config.quantization_config,
        );
        let mut config = SegmentConfig {
            vector_data: optimizer.plain_dense_vector_config.clone(),
            sparse_vector_data: optimizer.plain_sparse_vector_config.clone(),
            payload_storage_type: optimizer.payload_storage_type,
        };
        assert_eq!(
            config.vector_data["dense"].storage_type,
            VectorStorageType::ChunkedMmap,
            "the cold fixture's appendable storage should be the chunked on-disk form",
        );

        let retuned = loaded(cached_everywhere);
        let targets = PlacementTargets::resolve(&retuned);
        retarget_config(&targets, &mut config).unwrap();
        assert_eq!(
            config.vector_data["dense"].storage_type,
            VectorStorageType::InRamChunkedMmap,
            "the flip must stay within the chunked family",
        );
        match &config.vector_data["dense"].index {
            Indexes::Plain {} => {}
            other @ Indexes::Hnsw(_) => {
                panic!("an appendable segment must stay plain, got {other:?}")
            }
        }
    }

    /// Fabricate a one-segment shard under `out`, built under `built_doc`: a segment state plus
    /// the build fingerprint that `run` now requires. Returns the segment directory.
    fn fabricate_shard(out: &Path, built_doc: &LoadedConfig) -> std::path::PathBuf {
        let shard_dir = out.join("shard_0");
        let segment_dir = shard_dir
            .join(shard::files::SEGMENTS_PATH)
            .join("0aa5ab5f-0000-4000-8000-000000000001");
        fs_err::create_dir_all(&segment_dir).unwrap();

        let state = SegmentState {
            initial_version: None,
            version: Some(1),
            config: built_config(built_doc),
        };
        Segment::save_state(&state, &segment_dir).unwrap();

        let fingerprint = crate::build::BuildFingerprint {
            full: built_doc.fingerprint.clone(),
            part: built_doc.part_fingerprint.clone(),
        };
        fs_err::write(
            shard_dir.join(crate::build::BUILD_FINGERPRINT_FILE),
            fingerprint.to_json(),
        )
        .unwrap();
        segment_dir
    }

    /// `run` end to end over a fabricated artifact: writes on change, honours dry-run,
    /// and reports what it did.
    #[test]
    fn run_rewrites_segment_state_files_and_dry_run_does_not() {
        let out = TempDir::with_prefix("retarget").unwrap();
        let built_doc = loaded(|_| {});
        let segment_dir = fabricate_shard(out.path(), &built_doc);

        let retuned = loaded(cached_everywhere);

        let dry = run(&retuned, out.path(), &RetargetOptions { dry_run: true }).unwrap();
        assert_eq!(dry.segments_rewritten, 1);
        assert_eq!(
            Segment::load_state(&segment_dir).unwrap().config,
            built_config(&built_doc),
            "dry run must not write",
        );

        let wet = run(&retuned, out.path(), &RetargetOptions { dry_run: false }).unwrap();
        assert_eq!(wet.segments_examined, 1);
        assert_eq!(wet.segments_rewritten, 1);
        assert!(!wet.changes.is_empty());

        let rewritten = Segment::load_state(&segment_dir).unwrap();
        assert_eq!(rewritten.config, built_config(&retuned));
        assert_eq!(
            rewritten.version,
            Some(1),
            "retarget must touch only the config, never the version",
        );

        let noop = run(&retuned, out.path(), &RetargetOptions { dry_run: false }).unwrap();
        assert_eq!(noop.segments_rewritten, 0, "retarget must be idempotent");
    }

    /// #5: after a placement rewrite the stamped `full` fingerprint is refreshed to the retargeted
    /// document's, so a later `assemble` under that document is accepted (its `part` is preserved).
    #[test]
    fn run_refreshes_the_full_fingerprint_and_preserves_part() {
        let out = TempDir::with_prefix("retarget").unwrap();
        let built_doc = loaded(|_| {});
        fabricate_shard(out.path(), &built_doc);
        let retuned = loaded(cached_everywhere);

        // Sanity: placement-only edit moves `full` but not `part`.
        assert_ne!(built_doc.fingerprint, retuned.fingerprint);
        assert_eq!(built_doc.part_fingerprint, retuned.part_fingerprint);

        run(&retuned, out.path(), &RetargetOptions { dry_run: false }).unwrap();

        let body = fs_err::read_to_string(
            out.path()
                .join("shard_0")
                .join(crate::build::BUILD_FINGERPRINT_FILE),
        )
        .unwrap();
        let stamped = crate::build::BuildFingerprint::parse(&body);
        assert_eq!(
            stamped.full, retuned.fingerprint,
            "full must be refreshed to the new document"
        );
        assert_eq!(
            stamped.part, retuned.part_fingerprint,
            "part (routing) must be preserved"
        );
    }

    /// #5: a dry run must not refresh the fingerprint file.
    #[test]
    fn dry_run_does_not_refresh_the_fingerprint() {
        let out = TempDir::with_prefix("retarget").unwrap();
        let built_doc = loaded(|_| {});
        fabricate_shard(out.path(), &built_doc);
        let retuned = loaded(cached_everywhere);

        run(&retuned, out.path(), &RetargetOptions { dry_run: true }).unwrap();

        let body = fs_err::read_to_string(
            out.path()
                .join("shard_0")
                .join(crate::build::BUILD_FINGERPRINT_FILE),
        )
        .unwrap();
        assert_eq!(
            crate::build::BuildFingerprint::parse(&body).full,
            built_doc.fingerprint,
            "dry run must leave the stamped fingerprint untouched",
        );
    }

    /// #4: a document that changes routing (here `hash_ring_shard_scale`) is refused by name,
    /// before any segment is touched — retarget cannot re-route data.
    #[test]
    fn a_routing_change_is_refused() {
        let out = TempDir::with_prefix("retarget").unwrap();
        let built_doc = loaded(|_| {});
        let segment_dir = fabricate_shard(out.path(), &built_doc);
        let before = Segment::load_state(&segment_dir).unwrap().config;

        // Change the ring scale: routing differs, so `part` will not match.
        let rerouted = loaded(|value| {
            value["params"]["hash_ring_shard_scale"] = serde_json::json!(37);
        });
        assert_ne!(
            built_doc.part_fingerprint, rerouted.part_fingerprint,
            "a ring-scale change must move the routing fingerprint",
        );

        let err = run(&rerouted, out.path(), &RetargetOptions { dry_run: false }).unwrap_err();
        assert!(
            format!("{err:#}").contains("different routing"),
            "the refusal must name routing, got: {err:#}",
        );
        assert_eq!(
            Segment::load_state(&segment_dir).unwrap().config,
            before,
            "a refused routing change must not touch any segment",
        );
    }

    /// #1 (review): a legacy fingerprint (plain string, empty `part`) carries no routing to
    /// verify against, and retarget refreshes `full` (which folds in routing) — so proceeding
    /// could launder a different-routing document onto the data. It must be refused, not skipped.
    #[test]
    fn a_legacy_fingerprint_without_routing_is_refused() {
        let out = TempDir::with_prefix("retarget").unwrap();
        let built_doc = loaded(|_| {});
        let segment_dir = fabricate_shard(out.path(), &built_doc);
        // Overwrite with a legacy plain-string body (full only, no `part`).
        fs_err::write(
            out.path()
                .join("shard_0")
                .join(crate::build::BUILD_FINGERPRINT_FILE),
            &built_doc.fingerprint,
        )
        .unwrap();

        let retuned = loaded(cached_everywhere);
        let err = run(&retuned, out.path(), &RetargetOptions { dry_run: false }).unwrap_err();
        assert!(
            format!("{err:#}").contains("no routing fingerprint"),
            "got: {err:#}",
        );
        // Untouched.
        assert_eq!(
            Segment::load_state(&segment_dir).unwrap().config,
            built_config(&built_doc),
        );
    }

    /// #4: retarget refuses an artifact with no build fingerprint rather than running unverified.
    #[test]
    fn a_missing_build_fingerprint_is_refused() {
        let out = TempDir::with_prefix("retarget").unwrap();
        let built_doc = loaded(|_| {});
        let segment_dir = fabricate_shard(out.path(), &built_doc);
        fs_err::remove_file(
            out.path()
                .join("shard_0")
                .join(crate::build::BUILD_FINGERPRINT_FILE),
        )
        .unwrap();

        let retuned = loaded(cached_everywhere);
        let err = run(&retuned, out.path(), &RetargetOptions { dry_run: false }).unwrap_err();
        assert!(
            format!("{err:#}").contains("no build fingerprint"),
            "got: {err:#}",
        );
        // The segment is untouched.
        assert_eq!(
            Segment::load_state(&segment_dir).unwrap().config,
            built_config(&built_doc),
        );
    }
}
