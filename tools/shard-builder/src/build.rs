//! Phase 3: build one indexed segment per plan entry.
//!
//! # Two routes
//!
//! [`BuildRoute::Bulk`] is the default. It fills a segment's storages directly through
//! `SegmentBuilder::update_from_points` and then calls `SegmentBuilder::build` once, which is the
//! same call `SegmentOptimizer` makes. See [`build_segment_bulk`].
//!
//! [`BuildRoute::Edge`] is what this module did originally: upsert into a staging [`EdgeShard`] and
//! let [`EdgeShard::optimize`] (`lib/edge/src/optimize.rs:26`) produce the segment. It is retained
//! as the reference — see below — and selected with `--route edge`.
//!
//! # Why the default changed
//!
//! The edge route creates an appendable segment and a WAL purely to give `EdgeShard::update`
//! somewhere to write, and then throws both away. For sparse vectors that discarded work dominated
//! the whole build: an appendable segment holds a `MutableRam` sparse index, so every point went
//! through `PostingList::upsert` and its `propagate_max_next_weight_to_the_left` walk, and the
//! optimizer then rebuilt the index from scratch with `InvertedIndexBuilder`'s one-pass form.
//!
//! Measured on 100,000 real FineWeb points, one segment, `m=48`/`ef_construct=256`, TurboQuant
//! 4-bit, text payload:
//!
//! | route | ingest | index | total | peak RSS |
//! |---|---|---|---|---|
//! | edge | 160.4 s | 38.3 s | 200 s | 2.7 GiB |
//! | bulk | 1.7 s | 35.7 s | 38 s | 1.1 GiB |
//!
//! Without sparse vectors the same comparison is 42 s against 36 s, so essentially the entire
//! saving is the sparse index that was being built and discarded.
//!
//! # What the edge route is still for
//!
//! Both routes resolve the segment config through
//! `SegmentOptimizerConfig::optimized_segment_config`, the same function
//! `SegmentOptimizer::optimized_segment_builder` calls, so HNSW, quantization, storage placement
//! and the sparse index type are Qdrant's decisions on either route.
//!
//! What only the edge route gives is a *self-check*: `optimize()` returning with nothing left to
//! plan means the segment satisfies the very optimizers the serving cluster runs on load. The bulk
//! route resolves the config rather than discovering it, so build a sample both ways and compare
//! when the config changes. `build_e2e_tests` does exactly that on every test run.
//!
//! # Memory
//!
//! One task's working set is **one segment**, never a shard. Parts are streamed through in
//! batches of `batch_points`, so a segment's points are never all resident either — only the
//! batch plus whatever the segment's own storages hold.
//!
//! # Parallelism
//!
//! Within a segment, threading is left entirely to Qdrant: `hnsw_config.max_indexing_threads`
//! drives the HNSW build. Across segments, this module runs several builds concurrently, gated
//! by [`ResourceBudget`] using the same accounting the server's optimization worker uses
//! (`lib/collection/src/update_workers/optimization_worker.rs:316-335`), which charges IO rather
//! than CPU for indexing.
//!
//! # Resume
//!
//! A segment is done when its directory exists at the final path. Builds happen in a staging
//! directory and are renamed in, so a crash with several builds in flight leaves only staging
//! debris. The rename must be the last step: `normalize_segment_dir` **deletes** any segment
//! directory lacking `version.info` (`segment_constructor_base.rs:655`), so a partially-written
//! directory under the final name would be silently destroyed at load.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result, anyhow, bail};
use collection::optimizers_builder::build_segment_optimizer_config;
use common::budget::ResourceBudget;
use edge::config::optimizers::EdgeOptimizersConfig;
use edge::{EdgeConfig, EdgeShard, EdgeSparseVectorParams, EdgeVectorParams};
use shard::operations::CollectionUpdateOperations;
use shard::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted,
};
use shard::optimizers::segment_optimizer::max_num_indexing_threads;

use crate::config::LoadedConfig;
use crate::partfile::{PartProjection, PartReader, PartRecord};
use crate::plan::{SegmentPlan, ShardPlan};
use crate::scatter::ScatterLayout;

/// Where built artifacts and staging directories live.
pub struct BuildLayout {
    /// Final artifact root: `<out>/shard_{id}/segments/<uuid>`.
    out: PathBuf,
    /// Staging root, where a segment is built before being published to [`Self::out`].
    ///
    /// May be on a different filesystem: [`publish_segment`] renames when it can and copies when it
    /// cannot. Pointing this at tmpfs is the case worth supporting — the index phase's random reads
    /// stay in RAM and the shared filesystem sees one sequential write per segment.
    staging: PathBuf,
}

/// Hidden file in each `shard_N/` recording the config fingerprints the segments were built
/// under. Kept (not consumed), so re-runs of `assemble`/`retarget` stay gated. `assemble` refuses
/// a document whose `full` fingerprint disagrees; `retarget` refuses one whose `part` (routing)
/// fingerprint disagrees, and refreshes `full` after rewriting placements — closing the gaps in
/// the fingerprint chain.
pub const BUILD_FINGERPRINT_FILE: &str = ".sb-build-fingerprint";

/// The two fingerprints stamped into [`BUILD_FINGERPRINT_FILE`].
///
/// `full` is the whole rebuild surface — placements included — which is what `assemble` checks.
/// `part` is routing alone (the ring that decides which point lands in which shard). They are
/// stored together because `retarget` legitimately changes placement (moving `full`) but must
/// never change routing: it verifies `part` is unchanged and then rewrites `full` to match the
/// retargeted segments, so a later `assemble` under the retargeted document is accepted.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BuildFingerprint {
    pub full: String,
    pub part: String,
}

impl BuildFingerprint {
    /// Parse a fingerprint-file body, accepting the legacy plain-string form (a bare `full`
    /// fingerprint, written before `part` was recorded) so pre-existing artifacts still load.
    /// A legacy body leaves `part` empty, which callers read as "routing cannot be re-verified".
    pub fn parse(body: &str) -> Self {
        serde_json::from_str(body).unwrap_or_else(|_| BuildFingerprint {
            full: body.trim().to_string(),
            part: String::new(),
        })
    }

    /// Serialize to the on-disk body. Production writes go through `atomic_save_json`; this is a
    /// convenience for tests that fabricate a fingerprint file.
    #[cfg(test)]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("BuildFingerprint serializes")
    }
}

impl BuildLayout {
    pub fn new(out: impl Into<PathBuf>, staging: impl Into<PathBuf>) -> Self {
        Self {
            out: out.into(),
            staging: staging.into(),
        }
    }

    pub fn shard_dir(&self, shard_id: collection::shards::shard::ShardId) -> PathBuf {
        self.out.join(format!("shard_{shard_id}"))
    }

    pub fn shard_segments_dir(&self, shard_id: collection::shards::shard::ShardId) -> PathBuf {
        self.out.join(format!("shard_{shard_id}")).join("segments")
    }

    pub fn final_segment_dir(
        &self,
        shard_id: collection::shards::shard::ShardId,
        segment: &SegmentPlan,
    ) -> PathBuf {
        self.shard_segments_dir(shard_id)
            .join(segment.uuid.to_string())
    }

    fn staging_dir(
        &self,
        shard_id: collection::shards::shard::ShardId,
        segment: &SegmentPlan,
    ) -> PathBuf {
        self.staging
            .join(format!("shard_{shard_id}__segment_{}", segment.uuid))
    }
}

/// Everything a segment build needs that is the same for every segment in a run.
///
/// Grouped because it is threaded from `run_all` through the worker closures into
/// `build_segment`, and each new option was another parameter on three signatures. Only the shard
/// plan and the segment vary per task; these do not.
struct SegmentBuildContext<'a> {
    edge_config: &'a EdgeConfig,
    config: &'a LoadedConfig,
    scatter: &'a ScatterLayout,
    layout: &'a BuildLayout,
    batch_points: usize,
    payload_index: Option<&'a crate::payload_index::PayloadIndexSchema>,
    via_edge: bool,
    /// What every part is required to supply, resolved once per run.
    wanted: crate::partfile::WantedFields,
}

/// What this build requires every part file to supply.
///
/// The collection config names the vectors that must reach the segment; the mapping names the
/// payload columns and the source column each vector should have come from. A part may carry more
/// than this — that surplus is what gets projected away — but never less.
fn wanted_fields(
    config: &LoadedConfig,
    mapping: Option<&crate::parquet_source::ParquetMapping>,
) -> crate::partfile::WantedFields {
    let mut wanted = crate::partfile::WantedFields::default();

    for (name, params) in config.config.params.vectors.params_iter() {
        wanted
            .dense
            .insert(name.to_string(), params.size.get() as usize);
    }
    for name in config
        .config
        .params
        .sparse_vectors
        .iter()
        .flat_map(|map| map.keys())
    {
        wanted.sparse.insert(name.clone());
    }

    if let Some(mapping) = mapping {
        // `payload_columns` means "capture these" at scatter time and "keep these" at build time.
        // One field, two stages, superset relation between them — so trimming it here is exactly
        // how a payload column gets dropped without touching the scatter.
        wanted.payload = Some(mapping.payload_columns.iter().cloned().collect());

        for (name, column) in &mapping.dense_vectors {
            wanted.dense_sources.insert(name.clone(), column.clone());
        }
        for (name, sparse) in &mapping.sparse_vectors {
            wanted
                .sparse_sources
                .insert(name.clone(), sparse.column.clone());
        }
    }

    wanted
}

/// Test-only accessor for [`wanted_fields`], so `config`'s tests can assert on what a document
/// resolves to without duplicating the derivation.
#[cfg(test)]
pub fn wanted_fields_for_test(
    config: &LoadedConfig,
    mapping: Option<&crate::parquet_source::ParquetMapping>,
) -> crate::partfile::WantedFields {
    wanted_fields(config, mapping)
}

/// Which route a segment is built by.
///
/// Both produce a segment the serving cluster accepts; they differ in what they do on the way
/// there. See [`build_segment_bulk`] for why the default is no longer the edge route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildRoute {
    /// Feed the storages directly, then build indexes once. The default.
    Bulk,
    /// Upsert into a staging `EdgeShard` and let its optimizers produce the segment.
    ///
    /// Kept because it is the reference: it is the route whose output is known to satisfy the
    /// optimizers the serving cluster runs, so it is what a differential check compares against.
    Edge,
}

/// How a build run is tuned, as opposed to what it operates on.
pub struct BuildOptions<'a> {
    /// Concurrent segment builds.
    pub concurrency: usize,
    /// Points per update batch while streaming parts in.
    pub batch_points: usize,
    /// Field indexes to create inside each segment, if any.
    pub payload_index: Option<&'a crate::payload_index::PayloadIndexSchema>,
    /// This machine's share of the segments, as `(index, total)`.
    pub slice: Option<(usize, usize)>,
    /// Which route to build by.
    pub route: BuildRoute,
    /// Threads one segment's index build may use. Overrides the document's
    /// `hnsw_config.max_indexing_threads` for the build call only.
    ///
    /// This is the build box's resource decision, not the collection's: the document's value
    /// (usually absent, meaning auto-select on the serving node) is still what gets stamped
    /// into the artifact, and `mismatch_requires_rebuild` ignores that field either way — so
    /// no combination of this flag and the document can trigger a rebuild. See the
    /// NOTE(stage 3) in `main.rs`.
    pub indexing_threads: Option<usize>,
    /// The input document's mapping, when it has one.
    ///
    /// Read for two things, neither of which touches the input files: `payload_columns` selects
    /// which captured payload keys reach the segment, and the per-vector source columns are checked
    /// against what the scatter recorded. Absent for JSONL input, where nothing is projected away.
    pub mapping: Option<&'a crate::parquet_source::ParquetMapping>,
}

/// Outcome of a build run.
#[derive(Debug, Default, PartialEq)]
pub struct BuildStats {
    pub segments_built: u64,
    pub segments_skipped: u64,
    pub points: u64,
}

/// Translate a collection config into the equivalent [`EdgeConfig`].
///
/// Per-vector HNSW and quantization are taken from [`build_segment_optimizer_config`] — the
/// same function `build_optimizers` uses — so the *effective* merged values are baked into the
/// segment rather than a second interpretation of the diffs. `EdgeVectorParams::hnsw_config`
/// takes a full `HnswConfig` (not a diff), which is exactly what that resolution produces.
pub fn edge_config_for(config: &LoadedConfig) -> Result<EdgeConfig> {
    let resolved = build_segment_optimizer_config(
        &config.config.params,
        &config.config.hnsw_config,
        &config.config.quantization_config,
    );

    let mut vectors = HashMap::new();
    for (name, dense) in &resolved.dense_vector {
        let plain = resolved
            .plain_dense_vector_config
            .get(name)
            .ok_or_else(|| anyhow!("vector '{name}' has no plain config"))?;

        vectors.insert(
            name.clone(),
            EdgeVectorParams {
                size: plain.size,
                distance: plain.distance,
                // Edge speaks the deprecated boolean, so hand it the *resolved* placement's
                // on-disk-ness (`memory` over `on_disk`). The bool cannot express cold vs
                // cached, so on this route a `cached` request degrades to `cold`; the
                // mismatch optimizer compares dense storage by on-disk-ness only, so the
                // cluster does not rebuild for it. HNSW placement is unaffected — the full
                // `HnswConfig`, `memory` included, is passed below.
                on_disk: dense.memory_placement().map(|memory| memory.is_on_disk()),
                multivector_config: plain.multivector_config,
                datatype: plain.datatype,
                quantization_config: dense.quantization_config.clone(),
                hnsw_config: Some(dense.hnsw_config),
            },
        );
    }

    if vectors.is_empty() {
        bail!("collection configures no dense vectors; nothing to build");
    }

    let mut sparse_vectors = HashMap::new();
    if let Some(configured) = &config.config.params.sparse_vectors {
        for (name, params) in configured {
            let index = params.index.as_ref();
            // Unlike the dense case above, both raw placement spellings pass through
            // untouched: the sparse index config *persists* the explicitly requested
            // `memory`, so the staging shard must see exactly what the serving optimizer
            // would, or the two routes stamp different segment configs. One gap remains:
            // edge does not expose `wand_pruning` — fine for *built* (immutable) segments,
            // whose recorded index config clears it on every route.
            let placement = resolved.sparse_vector.get(name);
            sparse_vectors.insert(
                name.clone(),
                EdgeSparseVectorParams {
                    full_scan_threshold: index.and_then(|index| index.full_scan_threshold),
                    on_disk: placement.and_then(|cfg| cfg.on_disk),
                    memory: placement.and_then(|cfg| cfg.memory),
                    modifier: params.modifier,
                    datatype: index.and_then(|index| index.datatype).map(Into::into),
                },
            );
        }
    }

    let optimizer = &config.config.optimizer_config;

    Ok(EdgeConfig {
        // Resolved (`payload.memory` over the deprecated `on_disk_payload`), so the staging
        // shard's payload placement matches what the fingerprint hashed.
        on_disk_payload: Some(config.config.params.payload_storage_type().is_on_disk()),
        vectors,
        sparse_vectors,
        hnsw_config: Some(config.config.hnsw_config),
        quantization_config: config.config.quantization_config.clone(),
        optimizers: Some(EdgeOptimizersConfig {
            deleted_threshold: Some(optimizer.deleted_threshold),
            vacuum_min_vector_number: Some(optimizer.vacuum_min_vector_number),
            default_segment_number: Some(optimizer.default_segment_number),
            max_segment_size: optimizer.max_segment_size,
            indexing_threshold: optimizer.indexing_threshold,
            prevent_unoptimized: optimizer.prevent_unoptimized,
        }),
        // The staging shard's WAL is written and then discarded — only its segments are kept, and
        // the assembled artifact gets a fresh empty WAL. So the only thing that matters here is how
        // cheaply it can be written.
        //
        // Segment capacity is the knob that matters, and small is the wrong choice. Every ingest
        // batch is serialised into the WAL (`lib/edge/src/update.rs:15`), and exceeding a segment's
        // capacity retires it: a new file created and `fallocate`d, a directory `fsync`, and a wait
        // on the previous segment's `msync` (`lib/wal/src/lib.rs:263`). At 1 MiB — which this used to
        // be, to save transient disk — a single 1024-point batch of FineWeb rows overflows six
        // times, so a production-sized segment paid tens of thousands of directory fsyncs for a WAL
        // that is thrown away.
        //
        // 256 MiB turns that into a handful of retirements per segment. The space is `fallocate`d
        // rather than written, and `segment_queue_len: 1` creates the next one in the background so
        // the create does not sit in the ingest path.
        wal_options: Some(wal::WalOptions {
            segment_capacity: 256 * 1024 * 1024,
            segment_queue_len: 1,
            retain_closed: std::num::NonZeroUsize::new(1).expect("1 is non-zero"),
        }),
        // Everything else (search thread pool sizing and friends) is serving-time tuning the
        // staging shard never exercises; the fork's defaults are fine.
        ..EdgeConfig::default()
    })
}

/// Bytes one element of a configured dense vector occupies in storage.
///
/// Must track `size_of::<T>()` for the storage's element type, because that is what
/// `size_of_available_vectors_in_bytes` multiplies by
/// (`vector_storage_base.rs:215`) and therefore what the optimizer's thresholds see.
fn dense_element_size(datatype: Option<segment::types::VectorStorageDatatype>) -> Result<usize> {
    use segment::types::VectorStorageDatatype;

    Ok(match datatype {
        None | Some(VectorStorageDatatype::Float32) => size_of::<f32>(),
        Some(VectorStorageDatatype::Float16) => size_of::<half::f16>(),
        Some(VectorStorageDatatype::Uint8) => size_of::<u8>(),
        Some(VectorStorageDatatype::Turbo4) => {
            bail!("the turbo4 storage datatype is not wired up in Qdrant yet")
        }
    })
}

/// The `SegmentConfig` the indexing optimizer would give a segment holding `points` points.
///
/// Resolved by [`SegmentOptimizerConfig::optimized_segment_config`] — the same function
/// `SegmentOptimizer::optimized_segment_builder` calls — rather than by interpreting the thresholds
/// here a second time. That matters beyond tidiness: a segment whose recorded config differs from
/// what the optimizers would choose is exactly what `ConfigMismatchOptimizer` looks for, so a
/// second interpretation that drifted would produce segments the serving cluster silently rebuilds.
///
/// # The size the thresholds are compared against
///
/// `maximal_vector_store_size_bytes` is the largest single vector storage the segment will hold.
/// The optimizer reads it off real source segments; here it is computed from the planned point
/// count, which is exact for dense vectors — `points * dim * element_size` is precisely what
/// `size_of_available_vectors_in_bytes` returns.
///
/// Sparse vectors are **not** included, because their footprint depends on how many terms each
/// point carries and that is not known until the parts are read. [`check_sparse_size_assumption`]
/// re-checks the decision against the real sparse size once ingest is done, so a segment small
/// enough for the difference to matter fails loudly rather than being built under the wrong config.
fn target_segment_config(
    config: &LoadedConfig,
    points: u64,
) -> Result<segment::types::SegmentConfig> {
    let resolved = build_segment_optimizer_config(
        &config.config.params,
        &config.config.hnsw_config,
        &config.config.quantization_config,
    );

    let thresholds = config
        .config
        .optimizer_config
        .optimizer_thresholds(max_num_indexing_threads(&resolved).max(1), None);

    let maximal = dense_storage_bytes(config, points)?;

    Ok(resolved.optimized_segment_config(&thresholds, maximal, false))
}

/// The largest single dense vector storage a segment of this size holds.
///
/// The **maximum** over vector names, not the sum, because that is what the optimizer compares its
/// thresholds against (`segment_optimizer.rs:195`). `crate::plan::bytes_per_point` sums instead,
/// which is right for sizing a segment but would over-state the figure the thresholds see and could
/// flip a decision the optimizer would not have flipped.
fn dense_storage_bytes(config: &LoadedConfig, points: u64) -> Result<usize> {
    let resolved = build_segment_optimizer_config(
        &config.config.params,
        &config.config.hnsw_config,
        &config.config.quantization_config,
    );

    let mut maximal = 0usize;
    for (name, dense) in &resolved.plain_dense_vector_config {
        let element =
            dense_element_size(dense.datatype).with_context(|| format!("dense vector '{name}'"))?;
        let bytes = usize::try_from(points)
            .ok()
            .and_then(|points| points.checked_mul(dense.size))
            .and_then(|value| value.checked_mul(element))
            .ok_or_else(|| anyhow!("vector '{name}' storage size overflows usize"))?;
        maximal = maximal.max(bytes);
    }

    if maximal == 0 {
        bail!("collection configures no dense vectors; nothing to size a segment from");
    }

    Ok(maximal)
}

/// Refuse a segment whose sparse vectors would have changed the config decision.
///
/// [`target_segment_config`] sizes the segment from its dense vectors only. That is safe whenever
/// the dense footprint already puts the segment on the same side of both thresholds as the true
/// maximum would — which at any production segment size it does, by orders of magnitude. It is not
/// safe for a segment small enough that sparse data alone crosses a threshold the dense data does
/// not, so this checks that case explicitly rather than leaving it to chance.
///
/// `sparse_bytes` must be measured the way `InvertedIndexRam::total_sparse_size` measures it:
/// summed `nnz * size_of::<PostingElementEx>()`.
fn check_sparse_size_assumption(
    config: &LoadedConfig,
    points: u64,
    dense_bytes: usize,
    sparse_bytes: usize,
) -> Result<()> {
    use segment::common::BYTES_IN_KB;

    if sparse_bytes <= dense_bytes {
        return Ok(());
    }

    let resolved = build_segment_optimizer_config(
        &config.config.params,
        &config.config.hnsw_config,
        &config.config.quantization_config,
    );
    let thresholds = config
        .config
        .optimizer_config
        .optimizer_thresholds(max_num_indexing_threads(&resolved).max(1), None);

    for (name, threshold_kb) in [
        ("indexing_threshold", thresholds.indexing_threshold_kb),
        ("memmap_threshold", thresholds.memmap_threshold_kb),
    ] {
        let threshold = threshold_kb.saturating_mul(BYTES_IN_KB);
        if dense_bytes < threshold && sparse_bytes >= threshold {
            bail!(
                "this segment's sparse vectors ({sparse_bytes} bytes over {points} points) cross \
                 the {name} of {threshold} bytes but its dense vectors ({dense_bytes} bytes) do \
                 not, so the segment config was resolved from the wrong size.\n\n\
                 The bulk build route sizes a segment from its dense vectors, which is exact at \
                 production segment sizes but not for a segment this small. Either plan larger \
                 segments, or build this run with --route edge.",
            );
        }
    }

    Ok(())
}

/// Cores a single segment build actually keeps busy, on average.
///
/// Measured, not derived: one shard of three ~30k-point segments at `--concurrency 1` averaged
/// **218% CPU**, and a 33-segment run at `--concurrency 8` averaged 1400% on a 24-core box —
/// both consistent with ~2.2 cores per build.
///
/// It is well below `max_indexing_threads` because a segment build is not all HNSW. Ingest runs
/// through `EdgeShard::update`, which is single-threaded by construction
/// (`CollectionUpdater::update` takes `update_operation_lock.blocking_write()`), so each build is
/// roughly serial-ingest then parallel-HNSW and the average lands between the two.
///
/// Raising `--batch-points` does **not** change this: 1024 -> 65536 moved wall time by under 1.5%
/// while growing RSS, so the serial cost is per-point work in the ingest path rather than
/// per-batch lock acquisition.
///
/// This is a floor rather than a fixed constant. It was measured on ~30k-point segments; HNSW
/// thread efficiency should improve with graph size, so production-sized segments may keep more
/// cores busy and want *lower* concurrency. Measure per deployment with
/// `/usr/bin/time -f %P` on a single-shard build.
const OBSERVED_CORES_PER_BUILD: usize = 2;

/// Default concurrency: enough concurrent builds to keep the machine busy.
///
/// Divides by [`OBSERVED_CORES_PER_BUILD`] rather than by `max_indexing_threads`. Dividing by the
/// latter under-subscribes badly — it assumes each build saturates its whole thread pool, which
/// measurement shows it does not. On a 208-core node it would give 26 concurrent builds using
/// ~57 cores; this gives 104.
///
/// Memory, not cores, is the likely binding constraint at production segment sizes: each build
/// holds a staging segment plus HNSW construction state, so verify the product fits RAM before
/// trusting this default on large segments.
pub fn default_concurrency(config: &LoadedConfig) -> usize {
    // Read for the warning below; the value no longer sets concurrency directly.
    let resolved = build_segment_optimizer_config(
        &config.config.params,
        &config.config.hnsw_config,
        &config.config.quantization_config,
    );
    let per_segment_threads = max_num_indexing_threads(&resolved).max(1);

    let cores = common::cpu::get_num_cpus();
    let concurrency = (cores / OBSERVED_CORES_PER_BUILD).max(1);

    log::debug!(
        "default concurrency {concurrency} from {cores} cores at ~{OBSERVED_CORES_PER_BUILD}          cores/build (max_indexing_threads is {per_segment_threads})",
    );

    concurrency
}

/// Take one machine's share of the segment work.
///
/// Striding over the flat segment list rather than splitting by shard, because shards are not the
/// same size: the hash ring leaves roughly a 14% spread at 10 shards, so a shard-per-node split
/// leaves the smallest node idle while the largest is still going. Striding at segment granularity
/// also means the node count is independent of the shard count.
///
/// The list is a pure function of the plans, so every machine derives the same ordering and picks a
/// disjoint stride from it, and one machine's share can be re-run alone after a failure.
fn take_slice<T>(work: Vec<T>, index: usize, total: usize) -> Result<Vec<T>> {
    if total == 0 || index >= total {
        bail!("--slice must be I/N with N >= 1 and I < N; got {index}/{total}");
    }

    let mine: Vec<T> = work
        .into_iter()
        .enumerate()
        .filter(|(position, _)| position % total == index)
        .map(|(_, item)| item)
        .collect();

    if mine.is_empty() {
        bail!(
            "slice {index}/{total} covers no segments; there are fewer segments than slices, so \
             use a smaller N",
        );
    }

    Ok(mine)
}

/// Build every planned segment across every shard, in one pool.
///
/// Flattened deliberately. An earlier version looped shards sequentially and parallelised only
/// *within* a shard, so a plan with one segment per shard ran entirely serially no matter what
/// `--concurrency` said — the concurrency existed at the wrong level. One queue over all
/// `(shard, segment)` pairs has no such blind spot, and it also keeps every worker fed when
/// shards differ in size.
pub fn run_all(
    config: &LoadedConfig,
    plans: &[ShardPlan],
    scatter: &ScatterLayout,
    layout: &BuildLayout,
    options: &BuildOptions<'_>,
) -> Result<BuildStats> {
    let &BuildOptions {
        concurrency,
        batch_points,
        payload_index,
        slice,
        route,
        indexing_threads,
        mapping,
    } = options;

    if batch_points == 0 {
        bail!("--batch-points must be at least 1");
    }

    let wanted = wanted_fields(config, mapping);

    for plan in plans {
        // This is the check that freezes the config. A plan records the *full* config
        // fingerprint, so once planning is done every node building this collection is pinned to
        // one config — a document edited mid-build fails here instead of silently producing a
        // shard whose segments disagree with each other.
        //
        // It is also the reason the part files carry only the narrow `part_fingerprint`: tuning
        // `hnsw_config` or `max_segment_size` has to be cheap, and re-running `plan` is minutes
        // against a scatter that took hours.
        if plan.config_fingerprint != config.fingerprint {
            bail!(
                "plan for shard {} was made under a different collection config\n  \
                 plan:    {}\n  current: {}\n\n\
                 The config is frozen at `plan` time. If you changed it deliberately — tuning \
                 `hnsw_config`, `quantization_config` or `max_segment_size`, say — re-run `plan` \
                 and build again; the scatter output stays valid and is not re-read.\n\n\
                 If you did not change it, one node is reading a different document than the \
                 others, and building on would leave this shard with mismatched segments.",
                plan.shard_id,
                plan.config_fingerprint,
                config.fingerprint,
            );
        }
        fs_err::create_dir_all(layout.shard_segments_dir(plan.shard_id))?;

        // Refuse to build into an `--out` that already holds segment directories from a
        // *different* plan. Segment directories are content-addressed (see `plan::segment_uuid`),
        // so a legitimate resume of the same plan finds exactly its own segments and any extra
        // directory is a leftover from an earlier plan (inputs re-scattered, or the config
        // retuned) — assembling it alongside this plan's segments would ship a shard mixing two
        // plans. Every node holds the whole plan, so the expected UUID set is identical
        // cluster-wide and this stays correct under `--slice`.
        let expected: std::collections::HashSet<String> = plan
            .segments
            .iter()
            .map(|segment| segment.uuid.to_string())
            .collect();
        let segments_dir = layout.shard_segments_dir(plan.shard_id);
        let mut orphans = Vec::new();
        for entry in fs_err::read_dir(&segments_dir)? {
            let path = entry?.path();
            if !path.is_dir() {
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            // Hidden scratch (`.<uuid>...incomplete` publish temps) is swept elsewhere.
            if name.starts_with('.') {
                continue;
            }
            // A non-UUID directory is not ours; leave it be. An unexpected UUID is a stale
            // segment from a previous plan.
            if uuid::Uuid::parse_str(&name).is_ok() && !expected.contains(name.as_ref()) {
                orphans.push(name.into_owned());
            }
        }
        if !orphans.is_empty() {
            orphans.sort();
            bail!(
                "shard {} output directory {} holds {} segment director{} from a different plan \
                 ({}{}). Inputs were re-scattered or the config was retuned since these were \
                 built; building on would ship a shard mixing two plans. Build into a fresh \
                 --out, or remove the stale segment directories.",
                plan.shard_id,
                segments_dir.display(),
                orphans.len(),
                if orphans.len() == 1 { "y" } else { "ies" },
                orphans
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
                if orphans.len() > 3 { ", ..." } else { "" },
            );
        }

        // Stamp this build's config fingerprints into the shard directory so `assemble` can
        // refuse a document that differs from the one the segments were built under (the one
        // phase otherwise ungated against config drift, e.g. a ring-scale edit silently
        // restamped into `shard_config.json`), and `retarget` can refuse a routing change.
        // Idempotent: every node/slice writes the same value.
        let build_fingerprint = BuildFingerprint {
            full: config.fingerprint.clone(),
            part: config.part_fingerprint.clone(),
        };
        // Atomic: a torn write would be read back as a legacy (empty-`part`) body, which
        // `retarget` then refuses and `assemble` reports as a fingerprint mismatch.
        common::fs::atomic_save_json(
            &layout.shard_dir(plan.shard_id).join(BUILD_FINGERPRINT_FILE),
            &build_fingerprint,
        )?;
    }
    fs_err::create_dir_all(&layout.staging)?;
    if !same_filesystem(&layout.staging, &layout.out)? {
        // Worth stating: it changes the cost model of the last step from free to one sequential
        // write of every segment, and it is usually a deliberate choice rather than an accident.
        log::info!(
            "--staging ({}) and --out ({}) are on different filesystems; each finished segment              will be copied rather than renamed into place",
            layout.staging.display(),
            layout.out.display(),
        );
    }

    let mut edge_config = edge_config_for(config)?;
    // Build resources come from the CLI, never from the document: `--indexing-threads`
    // overrides the resolved `max_indexing_threads` for this run's permits (and, on the edge
    // route, for the staging shard's own optimizer budget). The artifact still records the
    // document's declared value.
    let per_segment_threads = match indexing_threads {
        Some(0) => bail!("--indexing-threads must be at least 1"),
        Some(threads) => threads,
        None => {
            let resolved = build_segment_optimizer_config(
                &config.config.params,
                &config.config.hnsw_config,
                &config.config.quantization_config,
            );
            max_num_indexing_threads(&resolved).max(1)
        }
    };
    if let Some(threads) = indexing_threads {
        // The edge route's `EdgeShard::optimize` sizes its own thread budget from the edge
        // config's HNSW settings, so the override has to reach them too. Harmless to the
        // artifact: `max_indexing_threads` is outside the mismatch surface.
        let threads =
            u32::try_from(threads).context("--indexing-threads is implausibly large")? as usize;
        if let Some(hnsw) = edge_config.hnsw_config.as_mut() {
            hnsw.max_indexing_threads = threads;
        }
        for params in edge_config.vectors.values_mut() {
            if let Some(hnsw) = params.hnsw_config.as_mut() {
                hnsw.max_indexing_threads = threads;
            }
        }
    }

    let budget = ResourceBudget::new(
        concurrency * per_segment_threads,
        concurrency * per_segment_threads,
    );

    // Every segment of every shard, as one flat work list.
    //
    // Ordering is deterministic — plans are read in shard-id order and segments are in `seq`
    // order — which is what lets several machines take disjoint strides of it without
    // coordinating.
    let all_work: Vec<(&ShardPlan, &SegmentPlan)> = plans
        .iter()
        .flat_map(|plan| plan.segments.iter().map(move |segment| (plan, segment)))
        .collect();

    let work = match slice {
        Some((index, total)) => take_slice(all_work, index, total)?,
        None => all_work,
    };

    let stopped = std::sync::atomic::AtomicBool::new(false);
    let built = AtomicU64::new(0);
    let skipped = AtomicU64::new(0);
    let points = AtomicU64::new(0);
    // Total includes segments that will be skipped as already built, so a resumed run shows how
    // much of the whole job is done rather than restarting the count from zero.
    let progress = crate::progress::Progress::new("build", "segment", work.len() as u64);
    let ctx = SegmentBuildContext {
        edge_config: &edge_config,
        config,
        scatter,
        layout,
        batch_points,
        payload_index,
        via_edge: route == BuildRoute::Edge,
        wanted,
    };
    let queue = Mutex::new(work.into_iter());

    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(concurrency);

        for worker in 0..concurrency {
            let (queue, built, skipped, points, budget, ctx, stopped, progress) = (
                &queue, &built, &skipped, &points, &budget, &ctx, &stopped, &progress,
            );

            handles.push(
                std::thread::Builder::new()
                    .name(format!("build-{worker}"))
                    .spawn_scoped(scope, move || -> Result<()> {
                        loop {
                            let next = queue.lock().expect("build queue poisoned").next();
                            let Some((plan, segment)) = next else {
                                return Ok(());
                            };

                            let final_dir = layout.final_segment_dir(plan.shard_id, segment);
                            if final_dir.exists() {
                                // Sweep staging this segment may have left. Both routes already
                                // clear stale staging for a segment they *rebuild*, but a skipped
                                // one is never rebuilt, so this is the only place that can clean it.
                                //
                                // The window is narrow — the process has to die between the publish
                                // and the removal — but on tmpfs the leftover is RAM that outlives
                                // the run and counts against whatever lands on this node next.
                                //
                                // Safe from here: `--slice` gives each node a disjoint set of
                                // segments, and within a run each segment is claimed by one worker,
                                // so this never touches staging another build is using.
                                let stale = layout.staging_dir(plan.shard_id, segment);
                                if stale.exists() {
                                    log::info!(
                                        "shard {} segment {}: clearing staging left by an earlier \
                                         run ({})",
                                        plan.shard_id,
                                        segment.seq,
                                        stale.display(),
                                    );
                                    // Warn rather than fail: this segment is already built, so
                                    // being unable to reclaim leftover bytes is no reason to abandon
                                    // a run that has nothing left to do for it.
                                    if let Err(err) = fs_err::remove_dir_all(&stale) {
                                        log::warn!(
                                            "cannot clear stale staging {}: {err}",
                                            stale.display(),
                                        );
                                    }
                                }
                                sweep_incomplete_publishes(
                                    &layout.shard_segments_dir(plan.shard_id),
                                    &segment.uuid,
                                );
                                skipped.fetch_add(1, Ordering::Relaxed);
                                progress.advance(0);
                                continue;
                            }

                            let permit =
                                budget.acquire(0, per_segment_threads, stopped).ok_or_else(
                                    || anyhow!("resource budget closed while acquiring a permit"),
                                )?;

                            let built_points =
                                build_segment(ctx, plan, segment, permit, budget, stopped)
                                    .with_context(|| {
                                        format!(
                                            "failed building shard {} segment {} ({})",
                                            plan.shard_id, segment.seq, segment.uuid,
                                        )
                                    })?;

                            built.fetch_add(1, Ordering::Relaxed);
                            points.fetch_add(built_points, Ordering::Relaxed);
                            progress.advance(built_points);
                        }
                    })
                    .context("cannot spawn build worker")?,
            );
        }

        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow!("build worker panicked"))??;
        }

        Ok(())
    })?;

    progress.finish();

    Ok(BuildStats {
        segments_built: built.load(Ordering::Relaxed),
        segments_skipped: skipped.load(Ordering::Relaxed),
        points: points.load(Ordering::Relaxed),
    })
}

/// Whether `staging` and `out` sit on the same filesystem.
///
/// Decides how a finished segment is published, and nothing else. Same filesystem means a rename —
/// atomic and free. Different filesystems mean a copy, which is supported deliberately: on a
/// cluster whose only fast local storage is RAM, staging in `/dev/shm` while the artifacts go to a
/// shared filesystem is the *right* configuration, not a mistake. It keeps the index phase's random
/// reads off the network and pays one sequential write instead.
///
/// Probed up front so the choice can be reported once, rather than inferred per segment.
fn same_filesystem(staging: &Path, out: &Path) -> Result<bool> {
    use std::os::unix::fs::MetadataExt as _;

    // `out` may not exist yet on a first run; the shard directories under it were just created, so
    // walk up to the nearest existing ancestor.
    let mut probe = out.to_path_buf();
    while !probe.exists() {
        match probe.parent() {
            Some(parent) => probe = parent.to_path_buf(),
            // Nothing to compare against; assume the cheap path and let the publish fall back.
            None => return Ok(true),
        }
    }

    let staging_dev = fs_err::metadata(staging)?.dev();
    let out_dev = fs_err::metadata(&probe)?.dev();
    Ok(staging_dev == out_dev)
}

/// Move a finished segment from staging to its final path, atomically either way.
///
/// A rename when the two share a filesystem. Otherwise a copy into a **hidden** sibling of the
/// destination, fsynced, then renamed into place.
///
/// The hidden name and the rename-last discipline are both load-bearing. Qdrant *deletes* any
/// segment directory that lacks `version.info`, taking it for wreckage from a crash mid-write
/// (`segment_constructor_base.rs`), so a half-copied directory must never be visible under a name
/// the server would consider. A leading dot means both Qdrant and this tool's own `assemble` skip
/// it, and the final path does not exist until every byte is there — so an interrupted publish
/// leaves the segment counting as unbuilt, and a resumed run redoes it.
fn publish_segment(produced: &Path, final_dir: &Path) -> Result<()> {
    match fs_err::rename(produced, final_dir) {
        Ok(()) => {}
        // EXDEV, matched on `kind` rather than the raw errno: `fs_err` rebuilds the error to attach
        // paths, which drops `raw_os_error()` but preserves the kind. A test covers this exact
        // branch, because getting it wrong means the cross-filesystem case fails only after a
        // segment has already cost minutes to build.
        //
        // Detected here rather than pre-decided from the up-front probe, so a wrong probe degrades
        // to the slow path instead of failing the build.
        Err(err) if err.kind() == std::io::ErrorKind::CrossesDevices => {
            let temp = incomplete_temp_path(final_dir)?;

            let staged = (|| -> Result<()> {
                copy_dir_all(produced, &temp).with_context(|| {
                    format!("cannot copy {} to {}", produced.display(), temp.display())
                })?;
                // Durable before it is visible: a rename that outlives its own data would publish
                // a segment Qdrant then deletes for missing files.
                common::fs::bulk_sync_dir(&temp)
                    .with_context(|| format!("cannot fsync {}", temp.display()))?;
                fs_err::rename(&temp, final_dir).with_context(|| {
                    format!(
                        "cannot rename {} to {}",
                        temp.display(),
                        final_dir.display()
                    )
                })
            })();

            if let Err(err) = staged {
                // Ours, so ours to remove. Leaving it would accumulate hidden directories that
                // nothing else sweeps; a failure to clean up must not mask the real error.
                if let Err(cleanup) = fs_err::remove_dir_all(&temp) {
                    log::warn!(
                        "could not remove {} after a failed publish: {cleanup}",
                        temp.display()
                    );
                }
                return Err(err);
            }

            fs_err::remove_dir_all(produced)
                .with_context(|| format!("cannot remove {}", produced.display()))?;
        }
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "cannot move {} to {}",
                    produced.display(),
                    final_dir.display(),
                )
            });
        }
    }

    common::fs::sync_parent_dir(final_dir)
        .with_context(|| format!("cannot fsync parent of {}", final_dir.display()))?;
    Ok(())
}

/// Distinguishes concurrent publishers within one process, so their staging directories never
/// collide. Paired with the pid, which distinguishes processes.
static PUBLISH_ATTEMPT: AtomicU64 = AtomicU64::new(0);

/// Where a cross-filesystem publish assembles the segment before renaming it into place.
///
/// Two properties, both load-bearing:
///
/// * **Hidden.** Qdrant deletes any segment directory lacking `version.info`, taking it for crash
///   wreckage, and this directory lacks everything until the copy finishes. A leading dot means both
///   Qdrant and this tool's `assemble` skip it, so a half-copied tree is never a candidate.
/// * **Unique per publisher.** A shared name would let two builds publishing the same segment id
///   into one `--out` interleave into the same directory and rename a partial tree into place. That
///   configuration is a documented mistake — two nodes given the same `--slice` — but it became
///   plausible once staging could be node-local, and the rename-only path used to fail it cleanly.
///   With unique names the loser's rename fails on an existing destination, which is loud and
///   harmless.
fn incomplete_temp_path(final_dir: &Path) -> Result<PathBuf> {
    let parent = final_dir
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent", final_dir.display()))?;
    let file_name = final_dir
        .file_name()
        .ok_or_else(|| anyhow!("{} has no file name", final_dir.display()))?
        .to_string_lossy()
        .to_string();

    Ok(parent.join(format!(
        ".{file_name}.{}.{}.incomplete",
        std::process::id(),
        PUBLISH_ATTEMPT.fetch_add(1, Ordering::Relaxed),
    )))
}

/// Remove abandoned cross-filesystem publish directories for one segment.
///
/// [`publish_segment`] cleans up after a publish that *fails*, but not after one that is *killed* —
/// and because the temp name is unique per attempt, a retry never reuses the orphan. At production
/// segment size that is tens of gigabytes of hidden space on the output filesystem per interrupted
/// build, invisible to `ls` and reclaimed by nothing.
///
/// Safe whenever this run owns the segment, which is the only place it is called from: a temp for a
/// given segment id is only ever written by a publisher of that id, and `--slice` gives each node a
/// disjoint set of segments.
///
/// Failure to reclaim is a warning, not an error. Being unable to delete stale bytes is no reason to
/// abandon a build that is otherwise fine.
fn sweep_incomplete_publishes(segments_dir: &Path, uuid: &uuid::Uuid) {
    let prefix = format!(".{uuid}.");

    let entries = match fs_err::read_dir(segments_dir) {
        Ok(entries) => entries,
        // Nothing built for this shard yet, so nothing to sweep.
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return,
        Err(err) => {
            log::warn!(
                "cannot scan {} for abandoned publishes: {err}",
                segments_dir.display()
            );
            return;
        }
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with(&prefix) || !name.ends_with(".incomplete") {
            continue;
        }
        log::info!(
            "removing abandoned publish directory left by an interrupted run: {}",
            entry.path().display(),
        );
        if let Err(err) = fs_err::remove_dir_all(entry.path()) {
            log::warn!("cannot remove {}: {err}", entry.path().display());
        }
    }
}

/// Recursively copy `src` into a fresh `dst`.
fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs_err::create_dir_all(dst)?;
    for entry in fs_err::read_dir(src)? {
        let entry = entry?;
        let target = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else {
            fs_err::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// Build one segment by whichever route the run selected.
fn build_segment(
    ctx: &SegmentBuildContext<'_>,
    plan: &ShardPlan,
    segment: &SegmentPlan,
    permit: common::budget::ResourcePermit,
    budget: &ResourceBudget,
    stopped: &std::sync::atomic::AtomicBool,
) -> Result<u64> {
    // This segment is not published yet, so any publish directory bearing its id is wreckage from an
    // interrupted run. Swept before building rather than after, so the space is available for the
    // build that is about to need it.
    sweep_incomplete_publishes(&ctx.layout.shard_segments_dir(plan.shard_id), &segment.uuid);

    if ctx.via_edge {
        // Held for the whole build rather than dropped: this permit is the only thing bounding
        // how many builds run at once, so releasing it early would let every worker start
        // immediately and ignore `--concurrency`. The edge route needs no CPU permit of its own
        // because `EdgeShard::optimize` allocates its own budget internally.
        let _permit = permit;
        build_segment_via_edge(ctx, plan, segment)
    } else {
        build_segment_bulk(ctx, plan, segment, permit, budget, stopped)
    }
}

/// Build one segment straight into its storages, then build its indexes once.
///
/// # What this does not do
///
/// It does not create an appendable segment, a WAL, or a shard. Those exist on the edge route
/// only to give `EdgeShard::update` somewhere to write, and everything they cost is discarded:
/// the WAL is thrown away, and the appendable segment's indexes are rebuilt from scratch by the
/// optimizer that replaces it.
///
/// For sparse vectors that discarded work dominated the whole build. An appendable segment holds a
/// `MutableRam` sparse index, so each point went through `PostingList::upsert`, which calls
/// `propagate_max_next_weight_to_the_left` on every insertion and walks left through a posting
/// list averaging ~888 entries for the FineWeb corpus. Measured at 100,000 points that was ~160 s
/// of a 206 s build — and then thrown away, because the built segment records `sparse index=Mmap`,
/// produced by `InvertedIndexBuilder`'s one-pass form. This route only ever runs the one-pass form.
///
/// # What still decides the shape of the segment
///
/// [`target_segment_config`] resolves the config through the optimizer's own
/// `optimized_segment_config`, and `SegmentBuilder::build` is the same call
/// `SegmentOptimizer` makes, so HNSW, quantization, storage placement and the sparse index type
/// are still Qdrant's decisions rather than ours. What is lost relative to the edge route is the
/// self-check: `optimize()` returning with nothing left to plan proved the segment satisfied the
/// optimizers the serving cluster runs. Use `--route edge` on a sample to re-establish that.
///
/// # Atomicity
///
/// `SegmentBuilder::build` writes into its own temporary directory and renames the finished
/// segment to `<segments>/<uuid>` as its last step, so pointing it at the final segments directory
/// gives the same all-or-nothing visibility the edge route got from staging plus a rename, with
/// one rename instead of two. A crash leaves only a `segment_builder_*` directory in staging.
fn build_segment_bulk(
    ctx: &SegmentBuildContext<'_>,
    plan: &ShardPlan,
    segment: &SegmentPlan,
    permit: common::budget::ResourcePermit,
    budget: &ResourceBudget,
    stopped: &std::sync::atomic::AtomicBool,
) -> Result<u64> {
    use common::counter::hardware_counter::HardwareCounterCell;
    use common::progress_tracker::new_progress_tracker;
    use segment::segment_constructor::segment_builder::SegmentBuilder;
    use segment::types::HnswGlobalConfig;

    let SegmentBuildContext {
        config,
        scatter,
        layout,
        batch_points,
        payload_index,
        ..
    } = *ctx;

    let staging = layout.staging_dir(plan.shard_id, segment);
    if staging.exists() {
        fs_err::remove_dir_all(&staging)
            .with_context(|| format!("cannot clear stale staging {}", staging.display()))?;
    }
    fs_err::create_dir_all(&staging)?;

    let target_config = target_segment_config(config, segment.points)?;
    let hw_counter = HardwareCounterCell::disposable();

    let mut builder = SegmentBuilder::new(&staging, &target_config, &HnswGlobalConfig::default())
        .map_err(|err| anyhow!("cannot create segment builder: {err}"))?;

    // Declared before ingest so `build` constructs them on the finished segment, in the same pass
    // as the vector index. The edge route had to apply them as operations after `optimize`,
    // which is why they landed on the empty appendable segment too.
    if let Some(schema) = payload_index {
        for (field, definition) in &schema.schema {
            builder.add_indexed_field(field.clone(), definition.clone());
        }
    }

    let mut tally = IngestTally::default();

    let ingest_started = std::time::Instant::now();
    let stored = {
        // The tally is filled as the builder pulls, so it is only complete after this call.
        let tally = &mut tally;
        let part_planned_end = part_planned_ends(plan);
        let records = PlannedRecords::new(
            config,
            &ctx.wanted,
            scatter.shard_dir(plan.shard_id),
            segment,
            &part_planned_end,
        )
        .map(|record| {
            let (record, projection) = record.map_err(|err| {
                segment::common::operation_error::OperationError::service_error(format!("{err:#}"))
            })?;

            tally.sparse_bytes += record
                .sparse
                .values()
                .map(|vector| {
                    vector.indices.len()
                        * size_of::<sparse::index::posting_list_common::PostingElementEx>()
                })
                .sum::<usize>();

            // One version per batch, 1-based, matching the edge route's WAL op numbering: a
            // batch is one upsert operation there. Within a batch a repeated id ties, and the
            // later copy wins either way — which is also what an `AHashMap` built from the
            // batch does on the edge route.
            let version = tally.records / batch_points as u64 + 1;
            tally.records += 1;

            record.into_point_to_insert(&projection, version)
        });

        builder
            .update_from_points(records, batch_points, stopped, &hw_counter)
            .map_err(|err| anyhow!("cannot insert points into the segment builder: {err}"))?
            as u64
    };

    let ingest_elapsed = ingest_started.elapsed();

    let ingested = tally.records;
    if ingested != segment.points {
        bail!(
            "planned {} points but ingested {ingested}; the scatter output changed since \
             planning. Re-run the plan phase.",
            segment.points,
        );
    }

    // See `build_segment_via_edge` for why this is a warning rather than an error.
    if stored != ingested {
        log::warn!(
            "shard {} segment {} ({}): fed {ingested} records but stored {stored} points; \
             {} duplicate point id(s) collapsed on upsert, so this segment holds fewer points \
             than the plan projects",
            plan.shard_id,
            segment.seq,
            segment.uuid,
            ingested.saturating_sub(stored),
        );
    }

    let dense_bytes = dense_storage_bytes(config, segment.points)?;
    check_sparse_size_assumption(config, segment.points, dense_bytes, tally.sparse_bytes)?;

    // The index build is CPU work, so trade the IO permit the worker acquired for a CPU one, the
    // same swap `lib/shard/src/optimize.rs:330` makes before calling `build`.
    let desired_cpus = permit.num_io as usize;
    let indexing_permit = budget
        .replace_with(permit, desired_cpus, 0, stopped)
        .map_err(|_| anyhow!("build cancelled while waiting for a CPU permit"))?;

    let (_handle, progress) = new_progress_tracker();
    let mut rng = rand::rng();
    let index_started = std::time::Instant::now();

    // Built into staging rather than straight into `--out`: `SegmentBuilder::build` renames its
    // temp directory into whatever path it is given, and a rename cannot cross filesystems. Landing
    // it in staging first keeps that internal rename local, and publishing is then this module's
    // decision — a rename or a copy, whichever the two filesystems allow.
    let staging_segments = staging.join("segments");
    fs_err::create_dir_all(&staging_segments)?;
    let built = builder
        .build(
            &staging_segments,
            segment.uuid,
            None,
            indexing_permit,
            stopped,
            &mut rng,
            &hw_counter,
            progress,
        )
        .map_err(|err| anyhow!("index build failed: {err}"))?;

    // Reported per segment so a run shows where its time went without needing a profiler: the
    // whole point of this route is moving work out of ingest, and that is only visible split.
    log::info!(
        "shard {} segment {} ({stored} points): ingest {:.1?}, index {:.1?}",
        plan.shard_id,
        segment.seq,
        ingest_elapsed,
        index_started.elapsed(),
    );

    // Dropped before publishing, not after: `build` returns the segment loaded from staging, and
    // holding it keeps every file mmapped. That matters most when staging is tmpfs, where a mapped
    // page is RAM that cannot be reclaimed until the mapping goes.
    drop(built);

    let final_dir = layout.final_segment_dir(plan.shard_id, segment);
    publish_segment(&staging_segments.join(segment.uuid.to_string()), &final_dir)?;

    fs_err::remove_dir_all(&staging)
        .with_context(|| format!("cannot remove staging {}", staging.display()))?;

    Ok(stored)
}

/// Streams a segment's planned records out of its part files, one at a time.
///
/// Lazy on purpose: the builder pulls points in batches, so at most one batch plus one open part
/// reader is resident regardless of how large the segment is. Written as a struct rather than a
/// chain of adaptors because each slice needs its own reader and its own skip/take state, and a
/// read failure has to surface as an item rather than unwinding.
/// The planned end offset (`skip + take` of the terminal slice) of every part named anywhere in
/// the shard plan.
///
/// A part's slices tile it contiguously from 0, so its terminal slice's end is the exact record
/// count the plan expects the part to hold. Both build routes read each part sequentially and, on
/// reaching a part's terminal slice, check the reader is then exhausted: a part carrying *more*
/// records than this grew since planning — an input re-scattered without a re-`plan` — and those
/// extra records belong to no segment, so a build that just read `take` and moved on would drop
/// them silently. This catches the grow direction; the per-slice short-read checks catch shrink.
fn part_planned_ends(plan: &ShardPlan) -> std::collections::HashMap<&str, u64> {
    let mut ends = std::collections::HashMap::new();
    for segment in &plan.segments {
        for slice in &segment.parts {
            let end = slice.skip + slice.take;
            let entry = ends.entry(slice.part.as_str()).or_insert(0u64);
            *entry = (*entry).max(end);
        }
    }
    ends
}

/// The slice [`PlannedRecords`] is currently reading: its path, open reader, remaining `take`, its
/// resolved projection, and — if this is the part's terminal slice — the part's planned end, so
/// the reader can be checked for leftover (grown) records once `take` is exhausted (`None` for a
/// non-terminal slice).
///
/// The projection is per part rather than per segment because parts are only guaranteed to agree
/// on the *routing* fingerprint — two scatter runs under different `payload_columns` produce
/// compatible parts with different manifests, and both may feed one segment.
type OpenSlice = (PathBuf, PartReader, u64, Arc<PartProjection>, Option<u64>);

struct PlannedRecords<'a> {
    part_fingerprint: &'a str,
    /// What each part must supply, and what to drop from it.
    wanted: &'a crate::partfile::WantedFields,
    shard_dir: PathBuf,
    slices: std::slice::Iter<'a, crate::plan::PartSlice>,
    /// Terminal end offset of every part in the shard plan, from [`part_planned_ends`].
    part_planned_end: &'a std::collections::HashMap<&'a str, u64>,
    open: Option<OpenSlice>,
    /// Reported once, so a downselect shows up in the log rather than having to be inferred.
    reported_projection: bool,
    /// Set once anything fails, so the iterator ends rather than reporting the same error forever.
    done: bool,
}

impl<'a> PlannedRecords<'a> {
    fn new(
        config: &'a LoadedConfig,
        wanted: &'a crate::partfile::WantedFields,
        shard_dir: PathBuf,
        segment: &'a SegmentPlan,
        part_planned_end: &'a std::collections::HashMap<&'a str, u64>,
    ) -> Self {
        Self {
            part_fingerprint: &config.part_fingerprint,
            wanted,
            shard_dir,
            slices: segment.parts.iter(),
            part_planned_end,
            open: None,
            reported_projection: false,
            done: false,
        }
    }

    /// Open the next slice, skipping the records that belong to an earlier segment.
    ///
    /// The skip is sequential rather than a seek because records are variable-length, so there is
    /// no offset table to jump with.
    fn open_next_slice(&mut self) -> Result<bool> {
        let Some(slice) = self.slices.next() else {
            return Ok(false);
        };

        let path = self.shard_dir.join(&slice.part);
        let mut reader = PartReader::open(&path, self.part_fingerprint)?;

        // Every compatibility check happens here, before a single record is read: a missing
        // vector, a dimensionality change, a remapped source column. Past this point conversion
        // cannot fail on a config mismatch.
        let projection = PartProjection::resolve(reader.header(), self.wanted, &path)?;
        if !self.reported_projection {
            self.reported_projection = true;
            if !projection.is_identity() {
                log::info!(
                    "dropping {} (present in parts, not in the config)",
                    projection.summary()
                );
            }
        }

        for skipped in 0..slice.skip {
            if reader.next_record()?.is_none() {
                bail!(
                    "{}: ended after skipping {skipped} of {} records of the plan's slice; the \
                     scatter output changed since planning. Re-run the plan phase.",
                    path.display(),
                    slice.skip,
                );
            }
        }

        // If this is the part's terminal slice, remember its planned end so `next_planned` can
        // check the reader is exhausted once `take` records are read — a leftover record means the
        // part grew since planning.
        let terminal_end = {
            let end = slice.skip + slice.take;
            (end == self.part_planned_end[slice.part.as_str()]).then_some(end)
        };

        self.open = Some((path, reader, slice.take, Arc::new(projection), terminal_end));
        Ok(true)
    }

    /// The next record, or `None` once the plan is exhausted.
    fn next_planned(&mut self) -> Result<Option<(PartRecord, Arc<PartProjection>)>> {
        loop {
            let Some((path, reader, remaining, projection, terminal_end)) = &mut self.open else {
                if self.open_next_slice()? {
                    continue;
                }
                return Ok(None);
            };

            if *remaining == 0 {
                // Terminal slice fully read: the part must now be exhausted. A surviving record
                // means it grew since planning (records past the plan that belong to no segment).
                if let Some(end) = *terminal_end
                    && reader.next_record()?.is_some()
                {
                    bail!(
                        "{}: has more than the {end} record(s) planned across all segments; the \
                         scatter output grew since planning — an input was likely re-scattered \
                         without re-running `plan`, so the extra records belong to no segment. \
                         Re-run the plan phase.",
                        path.display(),
                    );
                }
                self.open = None;
                continue;
            }

            match reader.next_record()? {
                Some(record) => {
                    *remaining -= 1;
                    return Ok(Some((record, Arc::clone(projection))));
                }
                None => bail!(
                    "{}: ended with {remaining} planned record(s) unread; the scatter output \
                     changed since planning. Re-run the plan phase.",
                    path.display(),
                ),
            }
        }
    }
}

impl Iterator for PlannedRecords<'_> {
    type Item = Result<(PartRecord, Arc<PartProjection>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        match self.next_planned() {
            Ok(Some(record)) => Some(Ok(record)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(err) => {
                self.done = true;
                Some(Err(err))
            }
        }
    }
}

/// What a pass over a segment's parts observed, beyond the points themselves.
#[derive(Debug, Default)]
struct IngestTally {
    /// Records fed in, which is not the same as points stored — see `build_segment_bulk`.
    records: u64,
    /// Sparse footprint, measured the way `InvertedIndexRam::total_sparse_size` measures it.
    sparse_bytes: usize,
}

/// Build one segment through a staging `EdgeShard`: stream its parts in, optimize, rename in.
fn build_segment_via_edge(
    ctx: &SegmentBuildContext<'_>,
    plan: &ShardPlan,
    segment: &SegmentPlan,
) -> Result<u64> {
    let SegmentBuildContext {
        edge_config,
        config,
        scatter,
        layout,
        batch_points,
        payload_index,
        ..
    } = *ctx;
    let staging = layout.staging_dir(plan.shard_id, segment);

    // Any leftover staging from an interrupted run is unusable: we cannot tell how much of the
    // plan it already contains, and a partial segment would silently lose points.
    if staging.exists() {
        fs_err::remove_dir_all(&staging)
            .with_context(|| format!("cannot clear stale staging {}", staging.display()))?;
    }
    fs_err::create_dir_all(&staging)?;

    let (ingested, stored) = {
        let shard = EdgeShard::new(&staging, edge_config.clone())
            .map_err(|err| anyhow!("cannot create staging shard: {err}"))?;

        let ingest_started = std::time::Instant::now();
        let ingested = ingest_parts(
            &shard,
            config,
            plan,
            segment,
            scatter,
            batch_points,
            &ctx.wanted,
        )?;
        let ingest_elapsed = ingest_started.elapsed();

        // The real index build: Qdrant's own optimizers, looping until nothing is planned.
        let optimize_started = std::time::Instant::now();
        shard
            .optimize()
            .map_err(|err| anyhow!("index build failed: {err}"))?;
        log::info!(
            "shard {} segment {} ({ingested} records): ingest {ingest_elapsed:.1?}, optimize {:.1?}",
            plan.shard_id,
            segment.seq,
            optimize_started.elapsed(),
        );

        // Payload field indexes after optimize, so they are built on the finished indexed segment
        // rather than on a staging segment the optimizer is about to replace. `create_field_index`
        // applies to every segment in the holder (`lib/shard/src/update.rs:921`), so the empty
        // appendable segment gets them too and stays consistent with the collection's schema.
        if let Some(schema) = payload_index {
            for operation in schema.create_operations() {
                shard.update(operation).map_err(|err| {
                    anyhow!("cannot create payload field index on the staging shard: {err}")
                })?;
            }
            log::debug!("created {} payload field index(es)", schema.len());
        }

        // Points actually held, which is not the same as records fed in — see below.
        //
        // `ShardInfo::points_count` is documented as approximate, but that caveat is about
        // summing across segments: it comes from `id_tracker.available_point_count()`
        // (`segment/read_view/info.rs:84`), which is exact per segment, and a shard holding the
        // same id in two segments would be counted twice. The staging shard cannot be in that
        // state — it ends with one data segment plus an empty appendable one, and an upsert of an
        // id already present updates it in place rather than adding it elsewhere. So this is an
        // exact count here, which is what makes comparing it to `ingested` meaningful.
        let stored = shard
            .info()
            .map_err(|err| anyhow!("cannot read staging shard info: {err}"))?
            .points_count as u64;

        (ingested, stored)
    };

    if ingested != segment.points {
        bail!(
            "planned {} points but ingested {ingested}; the scatter output changed since \
             planning. Re-run the plan phase.",
            segment.points,
        );
    }

    // `ingested` counts records read out of the part files; `stored` counts points the segment
    // holds. They differ when one point id appears more than once, because an upsert of an
    // existing id replaces it rather than adding to it. Without this check that is invisible:
    // the record count still matches the plan exactly, so the build reports success while the
    // collection quietly ends up smaller than projected.
    //
    // A warning rather than an error, because duplicate ids in the source are the source's
    // business and failing hours into a large run would be worse than reporting it. The
    // returned count is the stored one, so the run summary totals what the cluster will hold.
    if stored != ingested {
        log::warn!(
            "shard {} segment {} ({}): fed {ingested} records but stored {stored} points; \
             {} duplicate point id(s) collapsed on upsert, so this segment holds fewer points \
             than the plan projects",
            plan.shard_id,
            segment.seq,
            segment.uuid,
            ingested.saturating_sub(stored),
        );
    }

    let produced = built_segment_dir(&staging)?;

    // Published last. Until this succeeds the final path does not exist, so the segment counts as
    // unbuilt and a resumed run redoes it.
    let final_dir = layout.final_segment_dir(plan.shard_id, segment);
    publish_segment(&produced, &final_dir)?;

    fs_err::remove_dir_all(&staging)
        .with_context(|| format!("cannot remove staging {}", staging.display()))?;

    Ok(stored)
}

/// Stream this segment's parts into the staging shard in bounded batches.
fn ingest_parts(
    shard: &EdgeShard,
    config: &LoadedConfig,
    plan: &ShardPlan,
    segment: &SegmentPlan,
    scatter: &ScatterLayout,
    batch_points: usize,
    wanted: &crate::partfile::WantedFields,
) -> Result<u64> {
    let mut ingested = 0u64;
    let mut batch: Vec<PointStructPersisted> = Vec::with_capacity(batch_points);

    // Terminal end offset of each part, so the last slice of a part can be checked for leftover
    // (grown) records once its planned records are read. See [`part_planned_ends`].
    let part_planned_end = part_planned_ends(plan);

    for slice in &segment.parts {
        let path = scatter.shard_dir(plan.shard_id).join(&slice.part);
        let mut reader = PartReader::open(&path, &config.part_fingerprint)?;
        // Same projection as the bulk route, so the two stay differentially comparable: whatever
        // one drops, the other drops.
        let projection = PartProjection::resolve(reader.header(), wanted, &path)?;

        // Skip records belonging to an earlier segment. Sequential rather than a seek because
        // records are variable-length, so there is no offset table to jump with.
        for _ in 0..slice.skip {
            if reader.next_record()?.is_none() {
                bail!(
                    "{}: ended while skipping to record {} of the plan's slice; the scatter \
                     output changed since planning. Re-run the plan phase.",
                    path.display(),
                    slice.skip,
                );
            }
        }

        for index in 0..slice.take {
            let Some(record) = reader.next_record()? else {
                bail!(
                    "{}: ended after {index} of {} planned records; the scatter output changed \
                     since planning. Re-run the plan phase.",
                    path.display(),
                    slice.take,
                );
            };
            // The one widening: Qdrant's ingest type is f32-only, so f16 parts are expanded
            // here, at the boundary, rather than being stored expanded on disk.
            batch.push(record.to_point(&projection)?);
            if batch.len() >= batch_points {
                ingested += flush_batch(shard, &mut batch)?;
            }
        }

        // Terminal slice of this part: nothing was planned past `skip + take`, so the reader must
        // be exhausted. A surviving record means the part grew since planning.
        let planned_end = slice.skip + slice.take;
        if planned_end == part_planned_end[slice.part.as_str()] && reader.next_record()?.is_some() {
            bail!(
                "{}: has more than the {planned_end} record(s) planned across all segments; the \
                 scatter output grew since planning — an input was likely re-scattered without \
                 re-running `plan`, so the extra records belong to no segment. Re-run the plan \
                 phase.",
                path.display(),
            );
        }
    }

    ingested += flush_batch(shard, &mut batch)?;
    Ok(ingested)
}

fn flush_batch(shard: &EdgeShard, batch: &mut Vec<PointStructPersisted>) -> Result<u64> {
    if batch.is_empty() {
        return Ok(0);
    }

    let count = batch.len() as u64;
    let operation = CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
        PointInsertOperationsInternal::PointsList(std::mem::take(batch)),
    ));

    shard
        .update(operation)
        .map_err(|err| anyhow!("cannot apply points to staging shard: {err}"))?;

    batch.reserve(count as usize);
    Ok(count)
}

/// The one segment directory holding this build's points.
///
/// A finished staging shard usually contains **two** segments: the indexed one the optimizer
/// produced, and a fresh empty appendable segment, because a shard must always have somewhere to
/// accept writes. The empty one is distinguished by `version: null` in `segment.json` —
/// `SegmentState::version` is `Option<SeqNumberType>` and stays `None` until an operation is
/// applied, so this is an exact test rather than a size heuristic.
///
/// When the data falls below `indexing_threshold` no optimization runs at all, and the single
/// appendable segment *is* the result. Both shapes therefore reduce to "exactly one segment with
/// a version".
///
/// Erroring on more than one is deliberate: silently picking one would drop points, and the
/// pinned-UUID contract assumes one segment per plan entry.
fn built_segment_dir(staging: &Path) -> Result<PathBuf> {
    let segments = staging.join(shard::files::SEGMENTS_PATH);

    let mut dirs = Vec::new();
    for entry in fs_err::read_dir(&segments)
        .with_context(|| format!("cannot read {}", segments.display()))?
    {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        // Temporary optimizer scratch, not a finished segment.
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if name.starts_with('.') || name == shard::optimizers::config::TEMP_SEGMENTS_PATH {
            continue;
        }
        dirs.push(path);
    }

    // Classify by appendability, not by version.
    //
    // The optimizer writes an immutable, mmapped segment and leaves the shard a fresh *appendable*
    // one to accept writes. Version presence used to stand in for "holds the points", and it worked
    // as long as only point operations were applied. It stopped working once payload field indexes
    // were added: `create_field_index` applies to every segment in the holder
    // (`lib/shard/src/update.rs:921`) and `apply_field_index` stamps the op number on each via
    // `handle_segment_version_and_failure`, so the empty segment gets a version too and two
    // segments look like they hold points.
    //
    // Appendability is the property actually being asked about and no operation changes it.
    let mut built = Vec::new();
    let mut appendable = Vec::new();
    for path in dirs {
        if segment_is_appendable(&path)? {
            appendable.push(path);
        } else {
            built.push(path);
        }
    }

    log::debug!(
        "staging produced {} optimized segment(s) and {} appendable",
        built.len(),
        appendable.len(),
    );

    // Below `indexing_threshold` the optimizers do nothing and the points stay in the appendable
    // segment, which is then the only segment and the correct answer.
    let mut with_points = if built.is_empty() {
        let mut versioned = Vec::new();
        for path in appendable {
            if segment_has_version(&path)? {
                versioned.push(path);
            }
        }
        versioned
    } else {
        built
    };

    match with_points.len() {
        1 => Ok(with_points.pop().expect("checked length")),
        0 => bail!(
            "staging shard produced no segment holding points under {}. The parts may have been \
             empty.",
            segments.display(),
        ),
        n => bail!(
            "staging shard produced {n} segments with points under {}, expected 1: {:?}\n\n\
             A plan entry must map to exactly one segment. If the segment exceeded \
             max_segment_size the optimizers may have split it; lower the points per segment \
             and re-run the plan phase.",
            segments.display(),
            with_points,
        ),
    }
}

/// Whether a segment has ever had an operation applied.
///
/// Reads only `segment.json` rather than loading the segment: at 8 GiB a load would mmap the
/// whole thing just to answer a metadata question.
/// Whether a segment still accepts writes.
///
/// Read through `SegmentConfig::is_appendable()` rather than by inspecting storage-type strings, so
/// this follows Qdrant's own definition. In practice the optimizer's output is `Mmap` and the fresh
/// write target is `ChunkedMmap`.
fn segment_is_appendable(dir: &Path) -> Result<bool> {
    #[derive(serde::Deserialize)]
    struct ConfigOnly {
        config: segment::types::SegmentConfig,
    }

    let path = dir.join("segment.json");
    let bytes = fs_err::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
    let state: ConfigOnly = serde_json::from_slice(&bytes)
        .with_context(|| format!("cannot parse {}", path.display()))?;

    Ok(state.config.is_appendable())
}

fn segment_has_version(dir: &Path) -> Result<bool> {
    #[derive(serde::Deserialize)]
    struct VersionOnly {
        version: Option<u64>,
    }

    let path = dir.join("segment.json");
    let bytes = fs_err::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
    let state: VersionOnly = serde_json::from_slice(&bytes)
        .with_context(|| format!("cannot parse {}", path.display()))?;

    Ok(state.version.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;

    #[test]
    fn build_fingerprint_round_trips_and_falls_back_to_legacy() {
        // New format round-trips.
        let fp = BuildFingerprint {
            full: "a".repeat(64),
            part: "b".repeat(64),
        };
        assert_eq!(BuildFingerprint::parse(&fp.to_json()), fp);

        // Legacy plain-string bodies (written before `part` existed) parse as `full` only, with
        // an empty `part` that callers read as "routing unverifiable". Covers a hex body with
        // letters, an all-digit body (which is valid JSON as a *number*, not an object, so it
        // must still fall back), and surrounding whitespace.
        for legacy in [
            "deadbeef".repeat(8),   // 64 hex chars with letters
            "1".repeat(64),         // 64 all-digit chars — valid JSON number
            "  9f2c\n".to_string(), // whitespace around a short body
        ] {
            let parsed = BuildFingerprint::parse(&legacy);
            assert_eq!(parsed.full, legacy.trim(), "legacy body {legacy:?} -> full");
            assert!(
                parsed.part.is_empty(),
                "legacy body {legacy:?} -> empty part"
            );
        }

        // An empty file is treated as an (empty) legacy body, not a panic.
        assert_eq!(BuildFingerprint::parse("").full, "");
        assert!(BuildFingerprint::parse("").part.is_empty());

        // A JSON object missing our fields is not our format: fall back rather than accept a
        // partial struct.
        let stray = r#"{"other":1}"#;
        assert_eq!(BuildFingerprint::parse(stray).full, stray);
    }

    #[test]
    fn edge_config_carries_resolved_placement_and_quantization() {
        let mut value = config::tests::valid_config_json();
        value["params"]["vectors"]["dense"]["datatype"] = serde_json::json!("float16");
        let loaded = config::from_str(&value.to_string()).unwrap();

        let edge = edge_config_for(&loaded).unwrap();

        assert_eq!(
            edge.on_disk_payload,
            Some(true),
            "payload placement must carry over, resolved from `payload.memory`",
        );

        let dense = edge.vectors.get("dense").expect("dense vector");
        assert_eq!(dense.size, 768);
        assert_eq!(dense.on_disk, Some(true), "original vectors stay on disk");
        assert_eq!(
            dense.datatype,
            Some(segment::types::VectorStorageDatatype::Float16),
        );

        // Quantization must be the resolved per-vector value, so the segment records the same
        // config `has_config_mismatch` will compare against.
        let quantization = dense
            .quantization_config
            .as_ref()
            .expect("turbo quantization is configured");
        assert_eq!(
            quantization.memory_placement(),
            Some(segment::types::Memory::Pinned),
            "the pinned placement must survive translation, or the 4-bit vectors land on disk",
        );

        // Per-vector HNSW is a full config here, not a diff.
        let hnsw = dense.hnsw_config.expect("effective hnsw config");
        assert_eq!(hnsw.m, 16);
        assert_eq!(hnsw.ef_construct, 128);
        assert!(
            !hnsw.memory_placement().is_on_disk(),
            "graph stays in RAM (the fixture's `memory: null` resolves to cached)",
        );
    }

    #[test]
    fn edge_config_carries_sparse_placement() {
        let loaded = config::from_str(&config::tests::valid_config_json().to_string()).unwrap();
        let edge = edge_config_for(&loaded).unwrap();

        let sparse = edge.sparse_vectors.get("sparse").expect("sparse vector");
        // Raw pass-through, not a collapsed bool: the built index config persists the
        // explicitly requested `memory`, so edge must see exactly what the document said.
        // The fixture declares `memory: "cold"` and no deprecated `on_disk` flag.
        assert_eq!(
            sparse.memory,
            Some(segment::types::Memory::Cold),
            "sparse index placement must carry over; its default is RAM",
        );
        assert_eq!(
            sparse.on_disk, None,
            "the deprecated flag was not in the document, so it must not be invented",
        );
    }

    #[test]
    fn edge_config_carries_segment_thresholds() {
        let loaded = config::from_str(&config::tests::valid_config_json().to_string()).unwrap();
        let edge = edge_config_for(&loaded).unwrap();

        // These decide whether the optimizers index the segment and whether they would split
        // or merge it, so they must not fall back to edge's own defaults.
        let optimizers = edge.optimizers.as_ref().expect("optimizers must be set");
        assert_eq!(optimizers.max_segment_size, Some(5_000_000));
        assert_eq!(optimizers.default_segment_number, Some(8));
        assert_eq!(optimizers.indexing_threshold, Some(10000));
    }

    /// The size the thresholds see is the largest single vector storage, not the sum of all of them.
    ///
    /// `crate::plan::bytes_per_point` sums, which is correct for sizing a segment but would
    /// over-state this figure and could flip a threshold the optimizer would not have flipped.
    #[test]
    fn dense_storage_bytes_takes_the_largest_vector_not_the_total() {
        let mut value = config::tests::valid_config_json();
        value["params"]["vectors"] = serde_json::json!({
            "small": { "size": 16, "distance": "Cosine", "on_disk": true, "datatype": null,
                       "multivector_config": null },
            "large": { "size": 64, "distance": "Cosine", "on_disk": true, "datatype": null,
                       "multivector_config": null },
        });
        let loaded = config::from_str(&value.to_string()).unwrap();

        // 1000 * 64 * 4, not 1000 * (16 + 64) * 4.
        assert_eq!(dense_storage_bytes(&loaded, 1000).unwrap(), 1000 * 64 * 4);
    }

    /// The configured datatype must be reflected, since it halves or quarters the figure.
    #[test]
    fn dense_storage_bytes_follows_the_configured_datatype() {
        let mut value = config::tests::valid_config_json();
        value["params"]["vectors"]["dense"]["datatype"] = serde_json::json!("float16");
        let loaded = config::from_str(&value.to_string()).unwrap();

        assert_eq!(dense_storage_bytes(&loaded, 100).unwrap(), 100 * 768 * 2);
    }

    /// A segment above the indexing threshold must get HNSW, quantization and mmap storage.
    ///
    /// This is the shape a production segment ships in, and the one the bulk route has to resolve
    /// for itself rather than discovering by running the optimizers.
    #[test]
    fn target_config_of_a_large_segment_is_indexed_and_on_disk() {
        use segment::index::sparse_index::sparse_index_config::SparseIndexType;
        use segment::types::{Indexes, VectorStorageType};

        let loaded = config::from_str(&config::tests::valid_config_json().to_string()).unwrap();

        // 100k * 768 * 4 bytes is far above the 10,000 KB indexing threshold.
        let target = target_segment_config(&loaded, 100_000).unwrap();

        let dense = target.vector_data.get("dense").expect("dense");
        assert!(
            matches!(dense.index, Indexes::Hnsw(_)),
            "expected HNSW, got {:?}",
            dense.index,
        );
        assert!(
            dense.quantization_config.is_some(),
            "the collection configures turbo quantization; it must reach the segment",
        );
        assert_eq!(dense.storage_type, VectorStorageType::Mmap);

        let sparse = target.sparse_vector_data.get("sparse").expect("sparse");
        assert_eq!(
            sparse.index.index_type,
            SparseIndexType::Mmap,
            "an on-disk sparse vector in a big segment gets the mmap index, not MutableRam",
        );
    }

    /// Below the indexing threshold nothing is indexed, matching what the optimizers would leave.
    #[test]
    fn target_config_of_a_tiny_segment_is_plain() {
        use segment::types::Indexes;

        let loaded = config::from_str(&config::tests::valid_config_json().to_string()).unwrap();

        // 1 * 768 * 4 bytes is nowhere near the threshold.
        let target = target_segment_config(&loaded, 1).unwrap();
        let dense = target.vector_data.get("dense").expect("dense");
        assert!(
            matches!(dense.index, Indexes::Plain {}),
            "expected a plain index, got {:?}",
            dense.index,
        );
    }

    /// Sparse data that crosses a threshold the dense data does not must be refused, not guessed.
    #[test]
    fn a_segment_whose_sparse_data_dominates_is_refused() {
        let loaded = config::from_str(&config::tests::valid_config_json().to_string()).unwrap();

        // indexing_threshold is 10,000 KB = 10,240,000 bytes. Put dense below it and sparse above.
        let err = check_sparse_size_assumption(&loaded, 1_000, 1_000_000, 50_000_000)
            .expect_err("a threshold-crossing sparse footprint must be refused");
        let message = format!("{err:#}");
        assert!(message.contains("indexing_threshold"), "{message}");
        assert!(message.contains("--route edge"), "{message}");
    }

    /// The common case: dense dominates, so the decision made from it stands.
    #[test]
    fn a_segment_whose_dense_data_dominates_is_accepted() {
        let loaded = config::from_str(&config::tests::valid_config_json().to_string()).unwrap();
        check_sparse_size_assumption(&loaded, 100_000, 300_000_000, 40_000_000).unwrap();
    }

    /// Sparse larger than dense is fine as long as both land on the same side of the thresholds.
    #[test]
    fn a_larger_sparse_footprint_on_the_same_side_of_the_threshold_is_accepted() {
        let loaded = config::from_str(&config::tests::valid_config_json().to_string()).unwrap();
        // Both far above the 10,240,000-byte indexing threshold; memmap_threshold is disabled
        // (usize::MAX) in this fixture, so neither crosses it.
        check_sparse_size_assumption(&loaded, 100_000, 300_000_000, 900_000_000).unwrap();
    }

    /// Concurrency must follow measured cores-per-build, not `max_indexing_threads`.
    ///
    /// Dividing by `max_indexing_threads` was the earlier behaviour and it under-subscribed by
    /// ~4x, because a build averages ~2.2 cores rather than the 8 its thread pool allows.
    #[test]
    fn default_concurrency_follows_measured_cores_per_build() {
        let loaded = config::from_str(&config::tests::valid_config_json().to_string()).unwrap();
        let concurrency = default_concurrency(&loaded);
        let cores = common::cpu::get_num_cpus();

        assert!(concurrency >= 1, "must always allow one build");
        assert_eq!(concurrency, (cores / OBSERVED_CORES_PER_BUILD).max(1));

        // The fixture sets max_indexing_threads to 8; the default must no longer use it.
        if cores >= 8 {
            assert!(
                concurrency > cores / 8,
                "must not under-subscribe the way dividing by max_indexing_threads did",
            );
        }
    }

    /// Write a minimal `segment.json` with or without a version.
    /// A segment directory whose `segment.json` matches the shape Qdrant writes.
    ///
    /// `appendable` picks the storage type that decides `SegmentConfig::is_appendable()`:
    /// `ChunkedMmap` for the shard's write target, `Mmap` for the optimizer's immutable output.
    /// Taken from real artifacts rather than invented, since `built_segment_dir` now classifies on
    /// exactly this.
    fn fake_segment_with(segments: &Path, name: &str, version: Option<u64>, appendable: bool) {
        let dir = segments.join(name);
        fs_err::create_dir_all(&dir).unwrap();

        let (storage, index) = if appendable {
            (
                "ChunkedMmap",
                serde_json::json!({"type": "plain", "options": {}}),
            )
        } else {
            (
                "Mmap",
                serde_json::json!({"type": "hnsw", "options": {
                    "m": 16, "ef_construct": 128, "full_scan_threshold": 10000,
                    "max_indexing_threads": 0, "on_disk": false
                }}),
            )
        };

        let body = serde_json::json!({
            "version": version,
            "config": {
                "vector_data": {
                    "dense": {
                        "size": 4,
                        "distance": "Cosine",
                        "storage_type": storage,
                        "index": index,
                        "datatype": "float32"
                    }
                },
                "payload_storage_type": {"type": "mmap"}
            }
        });
        fs_err::write(dir.join("segment.json"), body.to_string()).unwrap();
    }

    /// The common case: an optimized segment holding the points.
    fn fake_segment(segments: &Path, name: &str, version: Option<u64>) {
        // Version present means "the optimizer produced this"; absent means the fresh appendable
        // write target. Preserves what the existing tests intend.
        fake_segment_with(segments, name, version, version.is_none());
    }

    /// The real shape after an indexing build: one indexed segment plus one empty appendable.
    #[test]
    fn picks_the_segment_with_points_over_the_empty_appendable_one() {
        let dir = tempfile::TempDir::with_prefix("build").unwrap();
        let segments = dir.path().join(shard::files::SEGMENTS_PATH);
        fake_segment(&segments, "indexed", Some(2));
        fake_segment(&segments, "fresh-appendable", None);

        let found = built_segment_dir(dir.path()).unwrap();
        assert!(found.ends_with("indexed"), "{found:?}");
    }

    /// Below `indexing_threshold` nothing is optimized and the appendable segment is the result.
    #[test]
    fn accepts_a_single_unoptimized_segment() {
        let dir = tempfile::TempDir::with_prefix("build").unwrap();
        let segments = dir.path().join(shard::files::SEGMENTS_PATH);
        fake_segment(&segments, "only", Some(1));

        let found = built_segment_dir(dir.path()).unwrap();
        assert!(found.ends_with("only"), "{found:?}");
    }

    #[test]
    fn reports_a_staging_shard_with_no_points() {
        let dir = tempfile::TempDir::with_prefix("build").unwrap();
        let segments = dir.path().join(shard::files::SEGMENTS_PATH);
        fake_segment(&segments, "empty", None);

        let err = built_segment_dir(dir.path()).unwrap_err();
        assert!(
            format!("{err:#}").contains("no segment holding points"),
            "{err:#}",
        );
    }

    #[test]
    fn refuses_multiple_segments_with_points() {
        let dir = tempfile::TempDir::with_prefix("build").unwrap();
        let segments = dir.path().join(shard::files::SEGMENTS_PATH);
        fake_segment(&segments, "aaaa", Some(1));
        fake_segment(&segments, "bbbb", Some(2));

        let err = built_segment_dir(dir.path()).unwrap_err();
        assert!(format!("{err:#}").contains("expected 1"), "{err:#}");
    }

    /// Slices must be disjoint, cover everything, and be stable.
    #[test]
    fn segment_slices_partition_the_work_exactly_once() {
        let work: Vec<usize> = (0..37).collect();

        let mut seen = Vec::new();
        for index in 0..4 {
            let mine = take_slice(work.clone(), index, 4).unwrap();
            assert_eq!(
                mine,
                take_slice(work.clone(), index, 4).unwrap(),
                "a slice must be deterministic",
            );
            seen.extend(mine);
        }

        seen.sort_unstable();
        assert_eq!(
            seen, work,
            "the slices must reassemble the whole list exactly"
        );
    }

    /// Striding, not chunking: consecutive segments go to different machines, so a shard that is
    /// larger than the others does not land entirely on one node.
    #[test]
    fn slices_stride_rather_than_chunk() {
        let work: Vec<usize> = (0..12).collect();
        assert_eq!(take_slice(work.clone(), 0, 3).unwrap(), vec![0, 3, 6, 9]);
        assert_eq!(take_slice(work, 1, 3).unwrap(), vec![1, 4, 7, 10]);
    }

    fn segment_with_parts(seq: usize, parts: Vec<crate::plan::PartSlice>) -> SegmentPlan {
        SegmentPlan {
            seq,
            uuid: uuid::Uuid::nil(),
            parts,
            points: 0,
            vector_bytes: 0,
        }
    }

    fn slice(part: &str, skip: u64, take: u64) -> crate::plan::PartSlice {
        crate::plan::PartSlice {
            part: part.to_string(),
            skip,
            take,
        }
    }

    /// The terminal end of each part is the max `skip + take` across every segment — including a
    /// large part split across consecutive segments, where the terminal slice carries a non-zero
    /// `skip`. This is the offset build checks the reader against to catch a grown part.
    #[test]
    fn part_planned_ends_takes_the_max_end_across_split_segments() {
        let plan = ShardPlan {
            shard_id: 0,
            config_fingerprint: "fp".to_string(),
            max_segment_size_bytes: 0,
            merge_safe_min_bytes: 0,
            bytes_per_point: 0,
            points: 0,
            segments: vec![
                // `big` is split across two segments (0..100, then 100..250); `small` lives
                // wholly in the first segment.
                segment_with_parts(0, vec![slice("big", 0, 100), slice("small", 0, 40)]),
                segment_with_parts(1, vec![slice("big", 100, 150)]),
            ],
        };

        let ends = part_planned_ends(&plan);
        assert_eq!(
            ends.get("big"),
            Some(&250),
            "terminal slice is the 100..250 one"
        );
        assert_eq!(ends.get("small"), Some(&40));
        assert_eq!(ends.len(), 2);
    }

    #[test]
    fn rejects_a_segment_slice_outside_the_total() {
        let err = take_slice((0..10).collect::<Vec<_>>(), 4, 4).unwrap_err();
        assert!(format!("{err:#}").contains("--slice must be"), "{err:#}");
    }

    /// More machines than segments must be rejected rather than silently doing nothing.
    #[test]
    fn rejects_a_segment_slice_that_covers_nothing() {
        let err = take_slice((0..3).collect::<Vec<_>>(), 5, 8).unwrap_err();
        assert!(format!("{err:#}").contains("covers no segments"), "{err:#}");
    }

    /// The regression this function was rewritten for.
    ///
    /// A payload-index operation applies to *every* segment in the holder, so the empty appendable
    /// segment ends up carrying a version too. Selecting on version presence then found two
    /// "segments with points" and the whole build failed with `expected 1`. Classifying on
    /// appendability is immune to it.
    #[test]
    fn picks_the_built_segment_even_when_the_empty_one_has_a_version() {
        let dir = tempfile::TempDir::with_prefix("build").unwrap();
        let segments = dir.path().join(shard::files::SEGMENTS_PATH);

        // Both stamped with the same op number, which is what creating a field index does.
        fake_segment_with(&segments, "indexed", Some(7), false);
        fake_segment_with(&segments, "fresh-appendable", Some(7), true);

        let found = built_segment_dir(dir.path()).unwrap();
        assert!(found.ends_with("indexed"), "{found:?}");
    }

    /// Two optimized segments is still an error: a plan entry must map to one segment.
    #[test]
    fn still_refuses_two_optimized_segments() {
        let dir = tempfile::TempDir::with_prefix("build").unwrap();
        let segments = dir.path().join(shard::files::SEGMENTS_PATH);
        fake_segment_with(&segments, "aaaa", Some(1), false);
        fake_segment_with(&segments, "bbbb", Some(2), false);

        let err = built_segment_dir(dir.path()).unwrap_err();
        assert!(format!("{err:#}").contains("expected 1"), "{err:#}");
    }

    #[test]
    fn ignores_optimizer_scratch_directories() {
        let dir = tempfile::TempDir::with_prefix("build").unwrap();
        let segments = dir.path().join(shard::files::SEGMENTS_PATH);
        fake_segment(&segments, "aaaa", Some(1));
        fs_err::create_dir_all(segments.join(shard::optimizers::config::TEMP_SEGMENTS_PATH))
            .unwrap();
        fs_err::create_dir_all(segments.join(".hidden")).unwrap();

        let found = built_segment_dir(dir.path()).unwrap();
        assert!(found.ends_with("aaaa"), "{found:?}");
    }
}

#[cfg(test)]
mod publish_tests {
    use std::path::Path;

    use super::*;

    /// A complete, plausible segment directory: nested dirs, an empty dir, and the file whose
    /// absence makes Qdrant delete a segment.
    fn fake_segment(root: &Path, marker: &str) {
        fs_err::create_dir_all(root.join("vector_storage-dense")).unwrap();
        fs_err::create_dir_all(root.join("payload_index")).unwrap();
        // Deliberately empty: `copy_dir_all` must preserve it, since a segment's directory shape is
        // part of what `load_segment` expects.
        fs_err::create_dir_all(root.join("empty_dir")).unwrap();
        fs_err::write(
            root.join("segment.json"),
            format!("{{\"marker\":\"{marker}\"}}"),
        )
        .unwrap();
        fs_err::write(root.join("version.info"), b"0.6.0").unwrap();
        fs_err::write(
            root.join("vector_storage-dense").join("matrix.dat"),
            vec![7u8; 4096],
        )
        .unwrap();
        fs_err::write(root.join("payload_index").join("config.json"), b"{}").unwrap();
        // Bulk, so a copy takes long enough for two publishers to actually overlap. Without this the
        // copy finishes inside a scheduling quantum and the race almost never manifests, which would
        // make this test pass against the very bug it exists to catch.
        for i in 0..24 {
            fs_err::write(
                root.join("vector_storage-dense")
                    .join(format!("chunk-{i}.dat")),
                vec![(i % 251) as u8; 64 * 1024],
            )
            .unwrap();
        }
    }

    fn assert_complete(segment: &Path, expected_marker: &str) {
        for relative in [
            "segment.json",
            "version.info",
            "vector_storage-dense/matrix.dat",
            "payload_index/config.json",
        ] {
            assert!(
                segment.join(relative).exists(),
                "published segment is missing {relative}",
            );
        }
        assert!(
            segment.join("empty_dir").is_dir(),
            "an empty directory was not preserved by the copy",
        );
        assert_eq!(
            fs_err::metadata(segment.join("vector_storage-dense").join("matrix.dat"))
                .unwrap()
                .len(),
            4096,
        );
        let body = fs_err::read_to_string(segment.join("segment.json")).unwrap();
        assert!(
            body.contains(expected_marker),
            "published segment mixes sources: {body}",
        );
    }

    /// The cross-filesystem copy must reproduce the directory faithfully, empty dirs included.
    #[test]
    fn a_copied_segment_is_complete() {
        let src = tempfile::Builder::new()
            .prefix("pub_src")
            .tempdir_in("/dev/shm");
        let Ok(src) = src else {
            eprintln!("skipping: /dev/shm unavailable");
            return;
        };
        let dst = tempfile::Builder::new()
            .prefix("pub_dst")
            .tempdir()
            .unwrap();

        let produced = src.path().join("seg");
        fake_segment(&produced, "only");
        let final_dir = dst.path().join("segments").join("uuid-1");
        fs_err::create_dir_all(final_dir.parent().unwrap()).unwrap();

        publish_segment(&produced, &final_dir).unwrap();

        assert_complete(&final_dir, "only");
        assert!(!produced.exists(), "staging copy was not removed");
        let hidden: Vec<String> = fs_err::read_dir(final_dir.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with('.'))
            .collect();
        assert!(hidden.is_empty(), "temp directory left behind: {hidden:?}");
    }

    /// Two publishers racing for the same destination must never produce a mixed or partial segment.
    ///
    /// With a shared temp name plus a "clear it first" step, they interleaved into one directory and
    /// a partial tree could be renamed into place. Whatever lands must come from exactly one of them.
    #[test]
    fn racing_publishers_never_publish_a_partial_segment() {
        let Ok(shm) = tempfile::Builder::new()
            .prefix("pub_race")
            .tempdir_in("/dev/shm")
        else {
            eprintln!("skipping: /dev/shm unavailable");
            return;
        };
        let dst = tempfile::Builder::new()
            .prefix("pub_race_dst")
            .tempdir()
            .unwrap();
        let segments = dst.path().join("segments");
        fs_err::create_dir_all(&segments).unwrap();

        // Probabilistic: it depends on two threads overlapping inside the copy, and measured against
        // the shared-temp-name version it catches the corruption roughly 3 runs in 5. Kept as a smoke
        // test rather than the real guard -- `the_publish_temp_path_is_unique_per_publisher` pins the
        // invariant deterministically. A pass here is not evidence on its own.
        for round in 0..40 {
            let final_dir = segments.join(format!("uuid-{round}"));

            let a = shm.path().join(format!("a-{round}"));
            let b = shm.path().join(format!("b-{round}"));
            fake_segment(&a, "aaa");
            fake_segment(&b, "bbb");

            let (final_a, final_b) = (final_dir.clone(), final_dir.clone());
            let results = std::thread::scope(|scope| {
                let ha = scope.spawn(move || publish_segment(&a, &final_a));
                let hb = scope.spawn(move || publish_segment(&b, &final_b));
                (ha.join().unwrap(), hb.join().unwrap())
            });

            // At least one must succeed; both succeeding is also acceptable only if the result is
            // still one coherent segment, which the marker check below enforces.
            assert!(
                results.0.is_ok() || results.1.is_ok(),
                "both publishers failed: {:?} / {:?}",
                results.0.as_ref().err().map(|e| format!("{e:#}")),
                results.1.as_ref().err().map(|e| format!("{e:#}")),
            );

            let body = fs_err::read_to_string(final_dir.join("segment.json")).unwrap();
            let marker = if body.contains("aaa") { "aaa" } else { "bbb" };
            assert_complete(&final_dir, marker);
        }
    }

    /// The deterministic guard behind `racing_publishers_never_publish_a_partial_segment`.
    ///
    /// Two publishers must never share a staging directory. Testing that directly beats testing the
    /// race, which only manifests some of the time.
    #[test]
    fn the_publish_temp_path_is_unique_per_publisher() {
        let final_dir = Path::new("/out/shard_0/segments/3ed0eebe-1ff3-534b-92d1-4dd3036c9840");

        let first = incomplete_temp_path(final_dir).unwrap();
        let second = incomplete_temp_path(final_dir).unwrap();

        assert_ne!(
            first, second,
            "two publishers of the same segment would share a staging directory",
        );

        for path in [&first, &second] {
            let name = path.file_name().unwrap().to_string_lossy();
            assert!(
                name.starts_with('.'),
                "{name} is not hidden, so a half-copied tree would be a load candidate",
            );
            assert!(
                name.contains("3ed0eebe-1ff3-534b-92d1-4dd3036c9840"),
                "{name} does not identify its segment",
            );
            assert_eq!(path.parent(), final_dir.parent(), "must be a sibling");
        }
    }

    /// Same filesystem still takes the rename, so the fast path has not been lost.
    #[test]
    fn same_filesystem_publish_is_a_rename() {
        let dir = tempfile::Builder::new()
            .prefix("pub_same")
            .tempdir()
            .unwrap();
        let produced = dir.path().join("seg");
        fake_segment(&produced, "same");
        let final_dir = dir.path().join("segments").join("uuid-2");
        fs_err::create_dir_all(final_dir.parent().unwrap()).unwrap();

        // Inode preserved by a rename; a copy would allocate a new one.
        use std::os::unix::fs::MetadataExt as _;
        let before = fs_err::metadata(produced.join("version.info"))
            .unwrap()
            .ino();

        publish_segment(&produced, &final_dir).unwrap();

        let after = fs_err::metadata(final_dir.join("version.info"))
            .unwrap()
            .ino();
        assert_eq!(before, after, "same-filesystem publish should not copy");
        assert_complete(&final_dir, "same");
    }
}
