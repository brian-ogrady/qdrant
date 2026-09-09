//! Phase 2 planning: group scattered parts into segments.
//!
//! This step exists so phase 3 can be **streaming and resumable**. It decides, up front and
//! deterministically, which part files make up which segment. Phase 3 then builds one segment
//! at a time by streaming its parts through, and never has to hold a shard — or even know how
//! big a shard is.
//!
//! Three properties come out of that:
//!
//! * **Bounded memory.** A build task's working set is one segment, not one shard. With a 42 TiB
//!   corpus and 20 GiB segments, that is the difference between feasible and not.
//! * **Resumable at segment granularity.** A segment either exists at its final path or it does
//!   not. A crash costs at most one segment's work.
//! * **Distributable.** The plan is a file, so segments can be handed to different machines and
//!   the results collected.
//!
//! Segment sizing follows the merge-safe floor from
//! [`LoadedConfig::merge_safe_min_segment_bytes`]: the merge optimizer can only combine
//! segments whose *pair* fits under `max_segment_size`, so every segment at or above half that
//! is left alone permanently. Batches are filled towards the ceiling, which maximises distance
//! from the floor.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use collection::shards::shard::ShardId;
use segment::types::VectorStorageDatatype;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::LoadedConfig;
use crate::partfile;
use crate::ring::ShardRouter;
use crate::scatter::ScatterLayout;

/// A contiguous run of records within one part file.
///
/// Parts are *sliced* rather than assigned whole. Assigning whole parts made segment size
/// bounded below by part size, so a single large input file produced segments far above
/// `max_segment_size` with nothing to warn about it. Slicing decouples the two: segment sizes
/// follow the config, whatever the input file sizes happen to be.
///
/// Resume is unaffected. The unit of work is still a whole segment — either its directory exists
/// or it does not — and a slice is a deterministic `(skip, take)` derived from the plan, not a
/// checkpoint that has to survive a crash.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PartSlice {
    /// Part file name, relative to the shard's scatter directory.
    pub part: String,
    /// Records to skip before reading.
    pub skip: u64,
    /// Records to read.
    pub take: u64,
}

/// One segment's worth of work.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SegmentPlan {
    /// Position within the shard. Stable across replans of the same inputs.
    pub seq: usize,
    /// Pinned so a rebuild produces the same directory name, making phase 3 idempotent.
    pub uuid: Uuid,
    /// Slices of part files making up this segment, in order.
    pub parts: Vec<PartSlice>,
    pub points: u64,
    /// Estimated vector-storage bytes. This, not the part file size, is what the optimizer
    /// thresholds are measured against.
    pub vector_bytes: u64,
}

/// The plan for one shard.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShardPlan {
    pub shard_id: ShardId,
    /// Guards against building from parts scattered under a different collection config.
    pub config_fingerprint: String,
    pub max_segment_size_bytes: u64,
    pub merge_safe_min_bytes: u64,
    pub bytes_per_point: u64,
    pub points: u64,
    pub segments: Vec<SegmentPlan>,
}

impl ShardPlan {
    pub fn path_in(work: &Path, shard_id: ShardId) -> PathBuf {
        work.join("plan").join(format!("shard_{shard_id}.json"))
    }
}

/// Bytes one point occupies in vector storage, summed over every configured vector.
///
/// This mirrors how `CollectionParams::get_deferred_point_id` sizes vectors
/// (`lib/collection/src/config.rs:194-210`): element width comes from the configured
/// `datatype`, defaulting to 4 bytes for float32.
///
/// Sparse vectors are deliberately excluded. Their size depends on per-point non-zero counts,
/// which the plan cannot know without reading the data, and the optimizer thresholds are
/// measured against `max_available_vectors_size_in_bytes`, which covers dense storage. A corpus
/// with large sparse vectors will therefore produce segments somewhat larger than planned —
/// which is the safe direction, since oversized segments are never merged.
pub fn bytes_per_point(config: &LoadedConfig) -> Result<u64> {
    let mut total = 0u64;

    for (name, params) in config.config.params.vectors.params_iter() {
        let element_bytes: u64 = match params.datatype.map(VectorStorageDatatype::from) {
            Some(VectorStorageDatatype::Float16) => 2,
            Some(VectorStorageDatatype::Uint8) => 1,
            // TurboQuant 4-bit is ~0.5 bytes/dim plus a per-row scale. Qdrant's own sizing uses
            // 1 byte as a placeholder here (`lib/collection/src/config.rs:203-206`), so mirror
            // that rather than inventing a second estimate: over-estimating segment size is the
            // safe direction, because oversized segments are never merged.
            Some(VectorStorageDatatype::Turbo4) => 1,
            Some(VectorStorageDatatype::Float32) | None => 4,
        };

        let dim = params.size.get();
        let per_vector = element_bytes
            .checked_mul(dim)
            .with_context(|| format!("vector '{name}' has an implausible size {dim}"))?;

        total = total
            .checked_add(per_vector)
            .context("total per-point vector size overflows")?;
    }

    if total == 0 {
        bail!("collection configures no dense vectors, so segment sizes cannot be planned");
    }

    Ok(total)
}

/// Build the plan for every configured shard.
/// How a plan run is tuned.
pub struct PlanOptions {
    /// Threads reading part metadata.
    pub workers: usize,
    /// This machine's share of the shards, as `(index, total)`.
    pub slice: Option<(usize, usize)>,
    /// Re-plan shards that already have a plan.
    pub replan: bool,
}

/// Outcome of a plan run.
#[derive(Debug)]
pub struct PlanReport {
    pub plans: Vec<ShardPlan>,
    /// Shards skipped because a plan already existed.
    pub shards_skipped: usize,
}

/// Group each shard's parts into segments and write one plan per shard.
///
/// # Why this is parallel and resumable
///
/// The work is one small read per part file to get its record count, and at production size that is
/// a hundred thousand of them — the kind of metadata-heavy pass a network filesystem is worst at.
/// Sequential and silent, it looks indistinguishable from a hang.
///
/// Shards are independent: each produces its own plan file, grouped only from its own parts. So the
/// pass is parallel over the metadata reads, splittable across machines by shard, and resumable at
/// shard granularity — a shard whose plan already exists is skipped without reading any of its
/// parts, which is what makes a re-run cheap rather than a repeat.
pub fn build(
    config: &LoadedConfig,
    router: &ShardRouter,
    layout: &ScatterLayout,
    work: &Path,
    options: &PlanOptions,
) -> Result<PlanReport> {
    let &PlanOptions {
        workers,
        slice,
        replan,
    } = options;

    if workers == 0 {
        bail!("--workers must be at least 1");
    }

    // The parts' own headers carry this, but planning reads only the `.meta` sidecars, which do
    // not — so without this a changed ring would go unnoticed until the data was already wrong.
    crate::scatter::check_part_fingerprint(config, layout)?;

    let (min_bytes, max_bytes) = config
        .target_segment_band_bytes()
        .context("max_segment_size must be set to plan segments")?;
    let bytes_per_point = bytes_per_point(config)?;

    // Points that fill a segment to the ceiling. At least one, so an absurdly wide vector
    // cannot produce a zero-capacity segment.
    let points_per_segment = (max_bytes / bytes_per_point).max(1);

    let all_shards: Vec<ShardId> = (0..router.shard_count() as ShardId).collect();
    let mine = match slice {
        Some((index, total)) => crate::scatter::take_slice(all_shards, index, total)?,
        None => all_shards,
    };

    // A shard's plan is written atomically, so its presence means the whole shard is planned.
    let (pending, skipped): (Vec<_>, Vec<_>) = if replan {
        (mine, Vec::new())
    } else {
        mine.into_iter()
            .partition(|shard_id| !ShardPlan::path_in(work, *shard_id).exists())
    };

    // The unit of progress is a part file, not a shard: there are ten shards and a hundred thousand
    // parts, so shard granularity would report almost nothing for almost all of the run.
    let part_names = discover_part_names(layout, &pending)?;

    // An adoption artifact is a shard, not merely a bag of non-empty segments. Omitting an empty
    // shard here used to let scatter -> plan -> build -> assemble report success for only the
    // populated subset of `shard_number`; the missing shard then had no artifact to install.
    //
    // Fail before writing *any* plans for this invocation. That makes the condition obvious and
    // prevents a partial plan directory from looking like a usable collection. A distributed
    // `--slice` checks its own assigned shards, so every planner reaches the same conclusion for
    // the shard it owns.
    let empty_shards: Vec<ShardId> = pending
        .iter()
        .copied()
        .filter(|shard_id| part_names.get(shard_id).is_none_or(Vec::is_empty))
        .collect();
    if !empty_shards.is_empty() {
        bail!(
            "cannot build a complete shard set: configured shard(s) {empty_shards:?} contain no \
             scattered points. The shard builder requires every shard to be non-empty; reduce \
             params.shard_number, supply more data, or use Qdrant's normal empty-shard creation \
             path instead of offline adoption."
        );
    }

    let plan_dir = work.join("plan");
    fs_err::create_dir_all(&plan_dir)
        .with_context(|| format!("cannot create {}", plan_dir.display()))?;

    let total_parts: usize = part_names.values().map(Vec::len).sum();
    let progress = crate::progress::Progress::new("plan", "part", total_parts as u64);

    let mut plans = Vec::new();
    for shard_id in &pending {
        // Checked non-empty above, before this invocation wrote any plans.
        let names = part_names
            .get(shard_id)
            .expect("empty shards rejected before planning")
            .clone();

        let parts = read_part_metas(config, layout, *shard_id, names, workers, &progress)?;
        let segments = group_into_segments(
            &parts,
            points_per_segment,
            bytes_per_point,
            *shard_id,
            &config.fingerprint,
        );
        let points = parts.iter().map(|(_, meta)| meta.records).sum();

        let plan = ShardPlan {
            shard_id: *shard_id,
            config_fingerprint: config.fingerprint.clone(),
            max_segment_size_bytes: max_bytes,
            merge_safe_min_bytes: min_bytes,
            bytes_per_point,
            points,
            segments,
        };

        // Written per shard, atomically, so phase 3 can be distributed one shard at a time and a
        // resumed plan can tell finished shards from unfinished ones.
        let path = ShardPlan::path_in(work, plan.shard_id);
        common::fs::atomic_save_json(&path, &plan)
            .with_context(|| format!("cannot write {}", path.display()))?;
        plans.push(plan);
    }

    progress.finish();

    if plans.is_empty() && skipped.is_empty() {
        bail!(
            "no scattered parts found under {}; run the scatter phase first",
            layout.state_path().display(),
        );
    }

    Ok(PlanReport {
        plans,
        shards_skipped: skipped.len(),
    })
}

/// Part file names per shard, without reading any of them.
///
/// Separated from reading the metadata so the total is known before work starts, which is what lets
/// progress be reported against a real denominator.
fn discover_part_names(
    layout: &ScatterLayout,
    shards: &[ShardId],
) -> Result<BTreeMap<ShardId, Vec<String>>> {
    let mut found = BTreeMap::new();

    for shard_id in shards {
        let dir = layout.shard_dir(*shard_id);
        let entries = match fs_err::read_dir(&dir) {
            Ok(entries) => entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err.into()),
        };

        let mut names = Vec::new();
        for entry in entries {
            let path = entry?.path();
            match path.extension().and_then(|ext| ext.to_str()) {
                Some("meta") => continue,
                Some("tmp") => bail!(
                    "{} is an uncommitted part; re-run the scatter phase before planning",
                    path.display(),
                ),
                _ => {}
            }
            names.push(
                path.file_name()
                    .and_then(|name| name.to_str())
                    .context("part file has a non-UTF-8 name")?
                    .to_string(),
            );
        }

        // Sorted so the same inputs always produce the same grouping, which is what makes a
        // replan idempotent and pinned segment UUIDs meaningful.
        names.sort();
        found.insert(*shard_id, names);
    }

    Ok(found)
}

/// Read one shard's part metadata in parallel, preserving the sorted order.
fn read_part_metas(
    config: &LoadedConfig,
    layout: &ScatterLayout,
    shard_id: ShardId,
    names: Vec<String>,
    workers: usize,
    progress: &crate::progress::Progress,
) -> Result<Vec<(String, partfile::PartMeta)>> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Indexed rather than pushed, so the result order is the sorted order regardless of which
    // worker finishes first — the grouping, and therefore every segment UUID, depends on it.
    let slots: Vec<Mutex<Option<partfile::PartMeta>>> =
        (0..names.len()).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);

    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::new();
        for worker in 0..workers.min(names.len().max(1)) {
            let (names, slots, next) = (&names, &slots, &next);
            handles.push(
                std::thread::Builder::new()
                    .name(format!("plan-{worker}"))
                    .spawn_scoped(scope, move || -> Result<()> {
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            if index >= names.len() {
                                return Ok(());
                            }
                            let path = layout.shard_dir(shard_id).join(&names[index]);
                            let meta = partfile::read_meta(&path, &config.part_fingerprint)?;
                            let records = meta.records;
                            *slots[index].lock().expect("plan slot poisoned") = Some(meta);
                            progress.advance(records);
                        }
                    })
                    .context("cannot spawn plan worker")?,
            );
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("plan worker panicked"))??;
        }
        Ok(())
    })?;

    let mut parts = Vec::with_capacity(names.len());
    for (name, slot) in names.into_iter().zip(slots) {
        let meta = slot
            .into_inner()
            .unwrap_or_else(|err| err.into_inner())
            .context("a part's metadata was not read")?;
        parts.push((name, meta));
    }
    Ok(parts)
}

/// Split a shard's parts into segments that all land inside the configured size band.
///
/// Distributes the shard's points evenly over `ceil(total / ceiling)` segments rather than
/// greedily filling to the ceiling and leaving a remainder. That choice is what makes the band
/// guarantee hold:
///
/// * **Never oversized.** With `n = ceil(total / ceiling)` segments, `total / n <= ceiling`.
/// * **Never undersized**, given enough data. `total / n >= ceiling / 2` whenever
///   `total >= ceiling`, because `n <= total/ceiling + 1`.
/// * A shard holding less than one full segment becomes a single segment, which may be below the
///   floor. That is safe and unavoidable: a merge needs *two* segments whose combined size fits
///   under the threshold, so a lone segment can never be paired.
///
/// Greedy filling could not offer this — it produced a short tail segment, and folding that tail
/// into its neighbour just moved the problem to the oversized end.
fn group_into_segments(
    parts: &[(String, partfile::PartMeta)],
    points_per_segment: u64,
    bytes_per_point: u64,
    shard_id: ShardId,
    config_fingerprint: &str,
) -> Vec<SegmentPlan> {
    let total: u64 = parts.iter().map(|(_, meta)| meta.records).sum();
    if total == 0 {
        return Vec::new();
    }

    let segment_count = total.div_ceil(points_per_segment).max(1);
    // Spread the remainder one point at a time over the leading segments, so sizes differ by at
    // most one and every segment stays inside the band.
    let base = total / segment_count;
    let remainder = total % segment_count;

    let mut segments = Vec::with_capacity(segment_count as usize);
    let mut part_index = 0usize;
    let mut consumed_in_part = 0u64;

    for seq in 0..segment_count {
        let mut wanted = base + u64::from(seq < remainder);
        let mut slices = Vec::new();
        let points = wanted;

        while wanted > 0 {
            let (name, meta) = &parts[part_index];
            let available = meta.records - consumed_in_part;
            let take = available.min(wanted);

            if take > 0 {
                slices.push(PartSlice {
                    part: name.clone(),
                    skip: consumed_in_part,
                    take,
                });
                consumed_in_part += take;
                wanted -= take;
            }

            if consumed_in_part == meta.records {
                part_index += 1;
                consumed_in_part = 0;
            }
        }

        segments.push(finish_segment(
            seq as usize,
            slices,
            points,
            bytes_per_point,
            shard_id,
            config_fingerprint,
        ));
    }

    segments
}

fn finish_segment(
    seq: usize,
    parts: Vec<PartSlice>,
    points: u64,
    bytes_per_point: u64,
    shard_id: ShardId,
    config_fingerprint: &str,
) -> SegmentPlan {
    SegmentPlan {
        // Named by its *content*, not just its position — see `segment_uuid`.
        uuid: segment_uuid(shard_id, seq, config_fingerprint, &parts),
        seq,
        parts,
        points,
        vector_bytes: points.saturating_mul(bytes_per_point),
    }
}

/// Deterministic, **content-addressed** segment UUID.
///
/// Derived from the shard, the position, the plan-frozen config fingerprint, and the exact
/// ordered slices of part files this segment covers. Determinism across a replan of identical
/// inputs is preserved (same inputs + same config → same slices → same UUID → phase 3 skips
/// finished work). The reason it is content-addressed rather than `(shard, seq)` alone: a
/// replan after the inputs or config changed regroups the parts, and a position-only name
/// would collide with a segment on disk that holds a *different* grouping — build would then
/// skip it as "already done" and silently ship an artifact mixing two plans. Folding the
/// slices and fingerprint in means a changed grouping produces a new directory name, so the
/// stale one is never mistaken for current (it surfaces as an orphan, refused by
/// `run_all`'s stale-output check).
fn segment_uuid(
    shard_id: ShardId,
    seq: usize,
    config_fingerprint: &str,
    parts: &[PartSlice],
) -> Uuid {
    let mut name = format!(
        "qdrant-shard-builder/shard_{shard_id}/segment_{seq}/fp_{config_fingerprint}/parts["
    );
    for slice in parts {
        // `part` names are hashes of store-relative paths (no delimiters), so this framing is
        // unambiguous.
        name.push_str(&format!("{}:{}:{};", slice.part, slice.skip, slice.take));
    }
    name.push(']');
    Uuid::new_v5(&Uuid::NAMESPACE_OID, name.as_bytes())
}

/// Warnings about a plan that would provoke the serving cluster's optimizers.
pub fn review(plan: &ShardPlan) -> Vec<String> {
    let mut notes = Vec::new();

    // Only the *last* segment can fall short, since earlier ones are filled to the ceiling.
    // A single short segment is harmless: with nothing to pair against under the threshold, the
    // merge optimizer still cannot act. Two or more are a problem.
    let short: Vec<&SegmentPlan> = plan
        .segments
        .iter()
        .filter(|segment| segment.vector_bytes < plan.merge_safe_min_bytes)
        .collect();

    if short.len() >= 2 {
        notes.push(format!(
            "{} segments are below the {} merge-safe floor; the cluster would merge them on \
             load. Raise optimizer_config.max_segment_size or reduce shard_number.",
            short.len(),
            crate::human_bytes(plan.merge_safe_min_bytes),
        ));
    }

    if plan.segments.len() == 1 && plan.segments[0].vector_bytes < plan.merge_safe_min_bytes {
        notes.push(format!(
            "shard {} holds only {}, below the {} floor. A single segment is safe (nothing to \
             pair with), but the shard is far smaller than the config implies.",
            plan.shard_id,
            crate::human_bytes(plan.segments[0].vector_bytes),
            crate::human_bytes(plan.merge_safe_min_bytes),
        ));
    }

    notes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;

    fn meta(records: u64) -> partfile::PartMeta {
        partfile::PartMeta {
            records,
            bytes: records * 64,
        }
    }

    /// A work directory holding `parts_per_shard` parts in each of `shards` shards.
    ///
    /// Only the `.meta` sidecars need to be real: `read_meta` reads them and never opens the part
    /// itself when one is present, which is the path production takes. The part file exists so
    /// discovery finds it.
    fn work_dir(shards: u32, parts_per_shard: usize, records: u64) -> tempfile::TempDir {
        let dir = tempfile::TempDir::with_prefix("plan").unwrap();
        for shard_id in 0..shards {
            let shard_dir = dir.path().join(format!("shard_{shard_id}"));
            fs_err::create_dir_all(&shard_dir).unwrap();
            for index in 0..parts_per_shard {
                let part = shard_dir.join(format!("part_{index:04}"));
                fs_err::write(&part, b"placeholder").unwrap();
                fs_err::write(
                    partfile::meta_path(&part),
                    serde_json::to_vec(&meta(records)).unwrap(),
                )
                .unwrap();
            }
        }
        dir
    }

    fn planning_config(shards: u32) -> config::LoadedConfig {
        let mut value = config::tests::valid_config_json();
        value["params"]["shard_number"] = serde_json::json!(shards);
        config::from_str(&value.to_string()).unwrap()
    }

    fn options(workers: usize, slice: Option<(usize, usize)>, replan: bool) -> PlanOptions {
        PlanOptions {
            workers,
            slice,
            replan,
        }
    }

    /// Tuning HNSW between scatter and plan is supported: part files do not depend on it.
    #[test]
    fn planning_accepts_a_changed_hnsw_config() {
        let mut value = config::tests::valid_config_json();
        value["params"]["shard_number"] = serde_json::json!(4);
        let scattered = config::from_str(&value.to_string()).unwrap();

        let dir = work_dir(4, 2, 50);
        let layout = crate::scatter::ScatterLayout::new(dir.path());
        crate::scatter::write_state_for_test(&scattered, &layout).unwrap();

        // Only the HNSW parameters differ.
        value["hnsw_config"]["m"] = serde_json::json!(64);
        value["hnsw_config"]["ef_construct"] = serde_json::json!(512);
        let retuned = config::from_str(&value.to_string()).unwrap();
        let router = ShardRouter::new(&retuned).unwrap();

        let report = build(
            &retuned,
            &router,
            &layout,
            dir.path(),
            &options(2, None, false),
        )
        .unwrap();
        assert_eq!(
            report.plans.len(),
            4,
            "a retuned config still plans the same parts"
        );
    }

    /// But a changed ring must be refused, not silently planned against the wrong shards.
    #[test]
    fn planning_refuses_a_changed_shard_count() {
        let mut value = config::tests::valid_config_json();
        value["params"]["shard_number"] = serde_json::json!(4);
        let scattered = config::from_str(&value.to_string()).unwrap();

        let dir = work_dir(4, 2, 50);
        let layout = crate::scatter::ScatterLayout::new(dir.path());
        crate::scatter::write_state_for_test(&scattered, &layout).unwrap();

        value["params"]["shard_number"] = serde_json::json!(2);
        let fewer = config::from_str(&value.to_string()).unwrap();
        let router = ShardRouter::new(&fewer).unwrap();

        let err = build(
            &fewer,
            &router,
            &layout,
            dir.path(),
            &options(2, None, false),
        )
        .unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("incompatible collection config"), "{text}");
        assert!(
            text.contains("hnsw_config"),
            "and says what may be tuned freely: {text}"
        );
    }

    /// A shard already planned is skipped without reading its parts.
    #[test]
    fn plan_is_resumable_per_shard() {
        let config = planning_config(4);
        let router = ShardRouter::new(&config).unwrap();
        let dir = work_dir(4, 3, 100);
        let layout = crate::scatter::ScatterLayout::new(dir.path());

        let first = build(
            &config,
            &router,
            &layout,
            dir.path(),
            &options(4, None, false),
        )
        .unwrap();
        assert_eq!(first.plans.len(), 4, "all four shards planned");
        assert_eq!(first.shards_skipped, 0);

        let second = build(
            &config,
            &router,
            &layout,
            dir.path(),
            &options(4, None, false),
        )
        .unwrap();
        assert!(second.plans.is_empty(), "nothing replanned");
        assert_eq!(second.shards_skipped, 4);

        let forced = build(
            &config,
            &router,
            &layout,
            dir.path(),
            &options(4, None, true),
        )
        .unwrap();
        assert_eq!(forced.plans.len(), 4, "--replan redoes them");
        assert_eq!(forced.shards_skipped, 0);
    }

    /// Offline adoption needs an artifact for every configured shard. A shard with no parts
    /// cannot produce one, so fail before leaving a partial set of plans behind.
    #[test]
    fn planning_refuses_an_empty_shard_before_writing_plans() {
        let config = planning_config(4);
        let router = ShardRouter::new(&config).unwrap();
        let dir = work_dir(4, 2, 50);
        fs_err::remove_dir_all(dir.path().join("shard_2")).unwrap();
        let layout = crate::scatter::ScatterLayout::new(dir.path());

        let err = build(
            &config,
            &router,
            &layout,
            dir.path(),
            &options(2, None, false),
        )
        .unwrap_err();
        let text = format!("{err:#}");
        assert!(
            text.contains("shard(s) [2] contain no scattered points"),
            "{text}"
        );
        assert!(
            !dir.path().join("plan").exists(),
            "an empty shard must fail before any plans are written"
        );
    }

    /// Slices divide the shards, so several machines can plan one work directory.
    #[test]
    fn plan_slices_cover_every_shard_exactly_once() {
        let config = planning_config(4);
        let router = ShardRouter::new(&config).unwrap();
        let dir = work_dir(4, 2, 50);
        let layout = crate::scatter::ScatterLayout::new(dir.path());

        let mut planned = Vec::new();
        for index in 0..2 {
            let report = build(
                &config,
                &router,
                &layout,
                dir.path(),
                &options(2, Some((index, 2)), false),
            )
            .unwrap();
            planned.extend(report.plans.iter().map(|plan| plan.shard_id));
        }
        planned.sort_unstable();
        assert_eq!(planned, vec![0, 1, 2, 3], "between them, every shard");

        let after = build(
            &config,
            &router,
            &layout,
            dir.path(),
            &options(2, None, false),
        )
        .unwrap();
        assert_eq!(after.shards_skipped, 4, "nothing left to plan");
    }

    /// Worker count must not change the plan — segment UUIDs are keyed on the grouping order.
    #[test]
    fn plan_worker_count_does_not_change_the_result() {
        let config = planning_config(2);
        let router = ShardRouter::new(&config).unwrap();

        let one_dir = work_dir(2, 9, 40);
        let one = build(
            &config,
            &router,
            &crate::scatter::ScatterLayout::new(one_dir.path()),
            one_dir.path(),
            &options(1, None, false),
        )
        .unwrap();

        let many_dir = work_dir(2, 9, 40);
        let many = build(
            &config,
            &router,
            &crate::scatter::ScatterLayout::new(many_dir.path()),
            many_dir.path(),
            &options(8, None, false),
        )
        .unwrap();

        assert_eq!(one.plans.len(), many.plans.len());
        for (a, b) in one.plans.iter().zip(&many.plans) {
            assert_eq!(a.points, b.points);
            assert_eq!(
                a.segments, b.segments,
                "identical segments, including pinned uuids and part order",
            );
        }
    }

    fn parts(counts: &[u64]) -> Vec<(String, partfile::PartMeta)> {
        counts
            .iter()
            .enumerate()
            .map(|(i, &n)| (format!("part_{i:04}"), meta(n)))
            .collect()
    }

    #[test]
    fn bytes_per_point_follows_the_configured_datatype() {
        // Fixture is 768 dims, datatype null => float32 => 4 bytes.
        let loaded = config::from_str(&config::tests::valid_config_json().to_string()).unwrap();
        assert_eq!(bytes_per_point(&loaded).unwrap(), 768 * 4);

        let mut value = config::tests::valid_config_json();
        value["params"]["vectors"]["dense"]["datatype"] = serde_json::json!("float16");
        let loaded = config::from_str(&value.to_string()).unwrap();
        assert_eq!(bytes_per_point(&loaded).unwrap(), 768 * 2);
    }

    #[test]
    fn bytes_per_point_sums_over_named_vectors() {
        let mut value = config::tests::valid_config_json();
        value["params"]["vectors"]["second"] = serde_json::json!({
            "size": 256,
            "distance": "Dot",
            "on_disk": true,
            "datatype": "uint8",
            "multivector_config": null,
        });
        let loaded = config::from_str(&value.to_string()).unwrap();
        assert_eq!(bytes_per_point(&loaded).unwrap(), 768 * 4 + 256);
    }

    /// Total points sliced into a segment, for checking nothing is lost or duplicated.
    fn slice_total(segment: &SegmentPlan) -> u64 {
        segment.parts.iter().map(|slice| slice.take).sum()
    }

    /// The guarantee: every segment inside the band, whenever there is at least a segment's worth.
    #[test]
    fn every_segment_lands_inside_the_size_band() {
        let ceiling = 100u64;
        let bytes = 4u64;

        // A spread of shapes: one big part, many small, awkward remainders, exact multiples.
        for counts in [
            vec![1000u64],
            vec![250, 250, 250, 250],
            vec![7, 3, 11, 1, 5, 9, 400],
            vec![100],
            vec![101],
            vec![199],
            vec![201],
        ] {
            let total: u64 = counts.iter().sum();
            let segments = group_into_segments(&parts(&counts), ceiling, bytes, 0, "fp");

            assert_eq!(
                segments.iter().map(|s| s.points).sum::<u64>(),
                total,
                "points lost for {counts:?}",
            );

            for segment in &segments {
                assert!(
                    segment.points <= ceiling,
                    "oversized segment {} for {counts:?}: {} > {ceiling}",
                    segment.seq,
                    segment.points,
                );
                if segments.len() > 1 {
                    assert!(
                        segment.points * 2 >= ceiling,
                        "undersized segment {} for {counts:?}: {} < {}",
                        segment.seq,
                        segment.points,
                        ceiling / 2,
                    );
                }
                assert_eq!(slice_total(segment), segment.points, "slice sum mismatch");
            }
        }
    }

    /// A single large part must be split, not turned into one oversized segment.
    ///
    /// This is the case the previous whole-part grouping got wrong: one 5 GiB input file produced
    /// one 160 MB segment against a 55 MiB ceiling, silently.
    #[test]
    fn a_single_large_part_is_split_across_segments() {
        let segments = group_into_segments(&parts(&[1000]), 100, 4, 0, "fp");

        assert_eq!(segments.len(), 10);
        for segment in &segments {
            assert_eq!(segment.points, 100);
            assert_eq!(segment.parts.len(), 1, "each slice comes from the one part");
        }
        // Slices must tile the part exactly, with no gap or overlap.
        let mut expected_skip = 0;
        for segment in &segments {
            assert_eq!(segment.parts[0].skip, expected_skip);
            expected_skip += segment.parts[0].take;
        }
        assert_eq!(expected_skip, 1000);
    }

    /// Slices must tile the whole input in order, across part boundaries too.
    #[test]
    fn slices_tile_every_part_exactly_once() {
        let counts = [30u64, 45, 25, 60];
        let segments = group_into_segments(&parts(&counts), 40, 4, 0, "fp");

        // Reconstruct (part, record) coverage and confirm it is a partition.
        let mut covered: Vec<(String, u64)> = Vec::new();
        for segment in &segments {
            for slice in &segment.parts {
                for offset in slice.skip..slice.skip + slice.take {
                    covered.push((slice.part.clone(), offset));
                }
            }
        }
        let unique: std::collections::HashSet<_> = covered.iter().cloned().collect();
        assert_eq!(covered.len(), unique.len(), "a record was read twice");
        assert_eq!(covered.len() as u64, counts.iter().sum::<u64>());
    }

    /// Below one segment's worth there is nothing to balance; a lone small segment is correct.
    #[test]
    fn a_shard_smaller_than_one_segment_becomes_one_segment() {
        let segments = group_into_segments(&parts(&[10]), 1000, 4, 0, "fp");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].points, 10);
        // Below the floor, but safe: a merge needs two segments that fit together.
        assert!(segments[0].points * 2 < 1000);
    }

    #[test]
    fn grouping_is_deterministic_and_uuids_are_pinned() {
        let first = group_into_segments(&parts(&[3, 3, 3, 3]), 10, 4, 7, "fp");
        let second = group_into_segments(&parts(&[3, 3, 3, 3]), 10, 4, 7, "fp");
        assert_eq!(first, second, "replanning must produce the same segments");

        // Different shards must not collide on segment UUIDs.
        let other_shard = group_into_segments(&parts(&[3, 3, 3, 3]), 10, 4, 8, "fp");
        assert_ne!(first[0].uuid, other_shard[0].uuid);

        // ...and the UUID is content-addressed: same inputs + same config → same name.
        assert_eq!(first[0].uuid, segment_uuid(7, 0, "fp", &first[0].parts));

        // A different config fingerprint moves the UUID, even for identical grouping — this is
        // what stops an in-place retune from silently skipping the stale segment.
        let retuned = group_into_segments(&parts(&[3, 3, 3, 3]), 10, 4, 7, "fp2");
        assert_ne!(
            first[0].uuid, retuned[0].uuid,
            "a changed config fingerprint must change the segment directory name",
        );
        assert_eq!(
            first[0].seq, retuned[0].seq,
            "but the position is unchanged"
        );

        // A different grouping (same config) also moves the UUID, so re-scattered inputs
        // cannot collide with a stale segment holding a different slice of records.
        let regrouped = group_into_segments(&parts(&[6, 6]), 10, 4, 7, "fp");
        assert_ne!(
            first[0].uuid, regrouped[0].uuid,
            "a changed part grouping must change the segment directory name",
        );
    }

    #[test]
    fn every_point_appears_in_exactly_one_segment() {
        let counts = [7u64, 3, 11, 1, 5, 9];
        let segments = group_into_segments(&parts(&counts), 10, 4, 0, "fp");

        let planned: u64 = segments.iter().map(|s| s.points).sum();
        assert_eq!(planned, counts.iter().sum::<u64>());

        let mut seen: Vec<&str> = segments
            .iter()
            .flat_map(|s| s.parts.iter().map(|slice| slice.part.as_str()))
            .collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), counts.len(), "every part contributes");
    }

    #[test]
    fn review_flags_multiple_short_segments() {
        let plan = ShardPlan {
            shard_id: 0,
            config_fingerprint: "fp".to_string(),
            max_segment_size_bytes: 1000,
            merge_safe_min_bytes: 500,
            bytes_per_point: 1,
            points: 300,
            segments: vec![
                SegmentPlan {
                    seq: 0,
                    uuid: Uuid::nil(),
                    parts: vec![PartSlice {
                        part: "a".into(),
                        skip: 0,
                        take: 1,
                    }],
                    points: 100,
                    vector_bytes: 100,
                },
                SegmentPlan {
                    seq: 1,
                    uuid: Uuid::nil(),
                    parts: vec![PartSlice {
                        part: "b".into(),
                        skip: 0,
                        take: 1,
                    }],
                    points: 200,
                    vector_bytes: 200,
                },
            ],
        };

        let notes = review(&plan);
        assert!(
            notes.iter().any(|note| note.contains("merge-safe floor")),
            "got: {notes:?}",
        );
    }

    #[test]
    fn review_accepts_a_single_short_segment() {
        let plan = ShardPlan {
            shard_id: 3,
            config_fingerprint: "fp".to_string(),
            max_segment_size_bytes: 1000,
            merge_safe_min_bytes: 500,
            bytes_per_point: 1,
            points: 100,
            segments: vec![SegmentPlan {
                seq: 0,
                uuid: Uuid::nil(),
                parts: vec![PartSlice {
                    part: "a".into(),
                    skip: 0,
                    take: 1,
                }],
                points: 100,
                vector_bytes: 100,
            }],
        };

        let notes = review(&plan);
        // Informational, but must not claim a merge would happen.
        assert!(
            !notes.iter().any(|note| note.contains("would merge")),
            "a lone short segment cannot be merged: {notes:?}",
        );
    }

    #[test]
    fn review_is_silent_for_a_well_sized_plan() {
        let plan = ShardPlan {
            shard_id: 0,
            config_fingerprint: "fp".to_string(),
            max_segment_size_bytes: 1000,
            merge_safe_min_bytes: 500,
            bytes_per_point: 1,
            points: 1800,
            segments: vec![
                SegmentPlan {
                    seq: 0,
                    uuid: Uuid::nil(),
                    parts: vec![PartSlice {
                        part: "a".into(),
                        skip: 0,
                        take: 1,
                    }],
                    points: 1000,
                    vector_bytes: 1000,
                },
                SegmentPlan {
                    seq: 1,
                    uuid: Uuid::nil(),
                    parts: vec![PartSlice {
                        part: "b".into(),
                        skip: 0,
                        take: 1,
                    }],
                    points: 800,
                    vector_bytes: 800,
                },
            ],
        };
        assert!(review(&plan).is_empty(), "{:?}", review(&plan));
    }
}
