//! Phase 4: turn built segments into a restorable shard directory.
//!
//! `out/shard_N/` already holds `segments/`. This adds the shard-level files a Qdrant shard
//! snapshot carries, matching what `ShardReplicaSet::create_snapshot` produces
//! (`lib/collection/src/shards/replica_set/snapshots.rs:22`):
//!
//! ```text
//! shard_N/
//!   segments/<uuid>/     built in phase 3
//!   wal/                 empty, based above every segment version
//!   applied_seq.json
//!   payload_index.json
//!   replica_state.json
//!   shard_config.json
//! ```
//!
//! One fork-specific addition: this fork's `shard_config.json` carries the hash-ring scale the
//! shard's points were routed at (`ShardConfig::new_replica_set_with_scale`), and
//! `check_snapshot_hash_ring_compatible` refuses a restore into a collection with a different
//! scale. So `assemble` stamps the config's `hash_ring_shard_scale` — omitting it would make the
//! restore path assume the pre-field default (100) and refuse artifacts built at any other scale.
//!
//! Clock files are deliberately omitted: `ClockMap::load_or_default` treats them as absent-means-
//! default (`local_shard/clock_map.rs:23`), and a freshly built shard has no clock history to
//! record. Writing empty ones would imply knowledge we do not have.
//!
//! # The WAL base index is the correctness-critical part
//!
//! Every segment was built by its own staging `EdgeShard`, each with a WAL starting from zero, so
//! segment versions are small per-segment counters rather than shard-global op numbers. The
//! assembled shard's WAL must therefore start **above the highest version any of its segments
//! carries**.
//!
//! If it did not, the first normal write after restore would get an op number at or below an
//! existing segment version, and `Segment::handle_segment_version`
//! (`lib/segment/src/segment/segment_ops.rs:404`) would **silently skip it** — returning
//! `Ok(false)`, no error, no log. Writes would vanish. This is the one place in the whole
//! pipeline where getting it wrong is invisible.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use collection::shards::shard::ShardId;
use serde::{Deserialize, Serialize};
use wal::{Wal, WalOptions};

use crate::config::LoadedConfig;

/// Files added alongside `segments/`.
const APPLIED_SEQ_FILE: &str = "applied_seq.json";
const PAYLOAD_INDEX_FILE: &str = "payload_index.json";
const REPLICA_STATE_FILE: &str = "replica_state.json";
const SHARD_CONFIG_FILE: &str = "shard_config.json";
const WAL_DIR: &str = "wal";

/// Result of assembling one shard.
#[derive(Debug, Clone, PartialEq)]
pub struct AssembleReport {
    pub shard_id: ShardId,
    pub segments: usize,
    /// Highest version across the shard's segments.
    pub max_segment_version: u64,
    /// Index the assembled WAL's next append will return.
    pub wal_base: u64,
}

/// Minimal `replica_state.json`.
///
/// Mirrors `ReplicaSetState` (`lib/collection/src/shards/replica_set/replica_set_state.rs:17`).
/// `this_peer_id` is a placeholder: `ShardReplicaSet::restore_snapshot` rewrites it via
/// `switch_peer_id` (`snapshots.rs:96`) and `ShardReplicaSet::load` rewrites it again on every
/// startup (`replica_set/mod.rs:257-271`), so a built shard adopts whichever node loads it.
/// `is_local` must be true or the restore path will not install the local data.
#[derive(Debug, Serialize, Deserialize)]
struct ReplicaStateFile {
    is_local: bool,
    this_peer_id: u64,
    /// Must contain an `Active` entry for `this_peer_id`.
    ///
    /// `switch_peer_id` *moves* the entry from the old id to the new one
    /// (`replica_set_state.rs:80`), so an empty map has nothing to move and the loaded replica
    /// ends up with no state at all. `peer_state()` then returns `None`, the replica is not
    /// considered Active, and every read reports "0 of 0 read operations failed" — the
    /// collection looks present but serves nothing.
    peers: std::collections::HashMap<u64, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AppliedSeqFile {
    op_num: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct PayloadIndexFile {
    schema: std::collections::HashMap<String, serde_json::Value>,
}

/// Assemble one shard directory in place.
pub fn assemble_shard(
    config: &LoadedConfig,
    shard_dir: &Path,
    shard_id: ShardId,
    payload_index: Option<&crate::payload_index::PayloadIndexSchema>,
) -> Result<AssembleReport> {
    let segments_dir = shard_dir.join(shard::files::SEGMENTS_PATH);
    if !segments_dir.is_dir() {
        bail!(
            "{} has no segments directory; run the build phase first",
            shard_dir.display(),
        );
    }

    // Close the fingerprint chain: `build` stamped its config fingerprint here, so assembling
    // against a *different* document (e.g. an edited `hash_ring_shard_scale`, silently
    // restamped into `shard_config.json`, or a changed payload-index schema) is refused
    // rather than producing a shard that misroutes or declares indexes its segments lack.
    //
    // The file is *kept*, not consumed: assemble is a supported re-run (e.g. rebasing the WAL),
    // and a partial failure below must stay recoverable — removing it first would make a
    // re-run bail "no build fingerprint" and force a full rebuild. Verifying on every assemble
    // also catches a drifted document on a re-run. It rides along inert: it lives at the shard
    // root (not in `segments/`, which is the only directory `LocalShard::load` enumerates), and
    // the adopt path never even moves it into the live shard.
    let fingerprint_path = shard_dir.join(crate::build::BUILD_FINGERPRINT_FILE);
    match fs_err::read_to_string(&fingerprint_path) {
        Ok(body) => {
            let built = crate::build::BuildFingerprint::parse(&body);
            if built.full != config.fingerprint {
                bail!(
                    "{} was built under a different document than the one given to assemble:\n  \
                     built:    {}\n  assemble: {}\n\nThe build and assemble \
                     phases must use the same document — assemble writes `shard_config.json` \
                     (hash-ring scale) and `payload_index.json` from the document it is given, \
                     so a mismatch would ship a shard that misroutes or rebuilds indexes at \
                     load. Re-run assemble with the build's document.",
                    shard_dir.display(),
                    built.full,
                    config.fingerprint,
                );
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            bail!(
                "{} has no build fingerprint ({}); it was not produced by this tool's build \
                 phase, or is from an incompatible version. Refusing to assemble an \
                 unverifiable shard.",
                shard_dir.display(),
                crate::build::BUILD_FINGERPRINT_FILE,
            );
        }
        Err(err) => return Err(err.into()),
    }

    let versions = segment_versions(&segments_dir)?;
    if versions.is_empty() {
        bail!(
            "{} contains no segments; a shard with no segments cannot be loaded",
            segments_dir.display(),
        );
    }

    let max_segment_version = versions.iter().copied().max().expect("checked non-empty");
    let wal_base = max_segment_version
        .checked_add(1)
        .context("segment version is at u64::MAX")?;

    // Verify before adding the appendable segment, so only the segments `build` produced are
    // checked — the fresh one below has no indexes yet by definition.
    if let Some(schema) = payload_index {
        verify_segment_indexes(&segments_dir, schema)?;
    }

    ensure_appendable_segment(config, &segments_dir, payload_index, max_segment_version)?;
    write_empty_wal(config, shard_dir, max_segment_version)?;

    // `applied_seq` records the last applied op number. Setting it to the highest segment
    // version means startup considers everything already applied, so `load_from_wal` replays
    // nothing. `AppliedSeqHandler::load_or_init` asserts `applied_seq <= wal.last_index()`
    // (`applied_seq.rs:97`), and `last_index()` of the empty WAL is `wal_base`, so this holds.
    write_json(
        &shard_dir.join(APPLIED_SEQ_FILE),
        &AppliedSeqFile {
            op_num: max_segment_version,
        },
    )?;

    // Records which field indexes this shard's segments actually carry.
    //
    // It has to match reality in both directions. Claiming an index the segments lack makes
    // `LocalShard::load` build it on the serving filesystem, over every point in the shard
    // (`segment_ops.rs:601`) -- the cost this builder exists to avoid. Omitting one the segments do
    // carry leaves the collection unaware of it.
    //
    // With no `payload_index` section the schema is empty, which is correct: `build` made none.
    match payload_index {
        Some(schema) => write_json(&shard_dir.join(PAYLOAD_INDEX_FILE), schema)?,
        None => write_json(
            &shard_dir.join(PAYLOAD_INDEX_FILE),
            &PayloadIndexFile {
                schema: Default::default(),
            },
        )?,
    }

    const PLACEHOLDER_PEER: u64 = 0;
    write_json(
        &shard_dir.join(REPLICA_STATE_FILE),
        &ReplicaStateFile {
            is_local: true,
            this_peer_id: PLACEHOLDER_PEER,
            peers: std::collections::HashMap::from([(PLACEHOLDER_PEER, "Active".to_string())]),
        },
    )?;

    // The fork's own type, not a lookalike: `restore_local_replica_from` checks the recorded
    // ring scale against the collection's (`check_snapshot_hash_ring_compatible`), and an
    // absent scale is read as the pre-field default of 100 — wrong for any other scale.
    write_json(
        &shard_dir.join(SHARD_CONFIG_FILE),
        &collection::shards::shard_config::ShardConfig::new_replica_set_with_scale(
            config.hash_ring_shard_scale(),
        ),
    )?;

    // Counted from the directory rather than from `versions`, so the number is the same on a
    // re-run: the appendable segment gains a version once it is created and indexed, which would
    // otherwise make an unchanged shard report a different count the second time.
    let segments = segment_dirs(&segments_dir)?.len();

    // Record the collection-level shape this shard was built for, so the server refuses adopting
    // it into a collection with different routing or vector shapes (which would install a
    // silently-wrong shard). Complements the ring-scale check `shard_config.json` already carries:
    // this also covers shard_number, sharding_method, the shard's own id, and vector shapes.
    collection::shards::adopt_manifest::AdoptManifest::for_shard(shard_id, &config.config.params)
        // Record the per-vector effective index config too: if the collection is later created
        // with a different HNSW/quantization config (per-vector overrides included), adoption
        // refuses rather than silently kicking off a full segment rebuild.
        .with_index_configs(
            &config.config.params,
            &config.config.hnsw_config,
            config.config.quantization_config.as_ref(),
        )
        .save(shard_dir)
        .with_context(|| format!("cannot write adopt manifest in {}", shard_dir.display()))?;

    Ok(AssembleReport {
        shard_id,
        segments,
        max_segment_version,
        wal_base,
    })
}

/// Add an empty appendable segment if the shard has none.
///
/// Every segment phase 3 produces is non-appendable: the indexing optimizer writes immutable,
/// mmapped, HNSW-indexed segments. But a shard must always have somewhere to accept writes.
/// `LocalShard::load` handles the absence by creating one — behind
/// `debug_assert!(false, "Shard has no appendable segments, this should never happen")`
/// (`local_shard/mod.rs:495`), so a debug-build server would panic rather than recover.
///
/// Providing one here also means the shard is writable the instant it loads, instead of after a
/// recovery path that logs a warning.
fn ensure_appendable_segment(
    config: &LoadedConfig,
    segments_dir: &Path,
    payload_index: Option<&crate::payload_index::PayloadIndexSchema>,
    op_num: u64,
) -> Result<()> {
    let existing = appendable_segment_count(segments_dir)?;
    if existing > 0 {
        log::debug!(
            "{} already has {existing} appendable segment(s)",
            segments_dir.display()
        );
        return Ok(());
    }

    let edge_config = crate::build::edge_config_for(config)?;
    let segment_config = edge_config.plain_segment_config();

    // `build_segment` with `ready = true` writes `version.info`, so the directory survives
    // `normalize_segment_dir`, which deletes segment directories that lack it.
    let (mut segment, _token) =
        segment::segment_constructor::build_segment(segments_dir, &segment_config, None, true)
            .map_err(|err| anyhow::anyhow!("cannot create appendable segment: {err}"))?;

    // The new segment must carry the same field indexes as the built ones. Without this, the
    // collection declares indexes that one of its segments lacks, and `LocalShard::load` builds
    // them — cheap here because the segment is empty, but it logs a rebuild for every field on
    // every shard at every startup, which is exactly the signal an operator needs to stay clean.
    //
    // Stamped with `op_num` = the highest built-segment version, so it stays below the WAL base and
    // a second `assemble` run computes the same maximum.
    if let Some(schema) = payload_index {
        use segment::entry::entry_point::{
            NonAppendableSegmentEntry as _, StorageSegmentEntry as _,
        };

        let counter = common::counter::hardware_counter::HardwareCounterCell::disposable();
        for (field, definition) in &schema.schema {
            segment
                .create_field_index(op_num, field, Some(definition), &counter)
                .map_err(|err| {
                    anyhow::anyhow!("cannot index '{field}' on the appendable segment: {err}")
                })?;
        }
        segment.flush(true).map_err(|err| {
            anyhow::anyhow!("cannot flush the appendable segment's indexes: {err}")
        })?;
    }

    log::debug!(
        "created empty appendable segment at {}",
        segment.segment_path.display(),
    );
    drop(segment);

    Ok(())
}

/// Check every built segment actually carries the declared field indexes.
///
/// `build` and `assemble` are separate commands taking the schema separately, and a build can be
/// resumed with a different one, so the two can disagree. The consequence is invisible at assemble
/// time and expensive later: a shard claiming an index its segments lack makes `LocalShard::load`
/// build it over every point, on the serving filesystem. Reading what the segments really contain
/// is the only way to catch it before the artifacts ship.
///
/// Each segment records its own schema in `payload_index/config.json` under `indexed_fields`, which
/// is what Qdrant compares against at load.
fn verify_segment_indexes(
    segments_dir: &Path,
    declared: &crate::payload_index::PayloadIndexSchema,
) -> Result<()> {
    #[derive(Deserialize)]
    struct SegmentIndexConfig {
        #[serde(default)]
        indexed_fields: std::collections::BTreeMap<String, segment::types::PayloadFieldSchema>,
    }

    // Declared field -> its schema, so definitions are compared, not just names.
    let expected: std::collections::BTreeMap<String, &segment::types::PayloadFieldSchema> =
        declared
            .schema
            .iter()
            .map(|(field, schema)| (field.to_string(), schema))
            .collect();

    for dir in segment_dirs(segments_dir)? {
        let path = dir.join("payload_index").join("config.json");
        let found: std::collections::BTreeMap<String, segment::types::PayloadFieldSchema> =
            match fs_err::read(&path) {
                Ok(bytes) => {
                    let config: SegmentIndexConfig = serde_json::from_slice(&bytes)
                        .with_context(|| format!("cannot parse {}", path.display()))?;
                    config.indexed_fields
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => Default::default(),
                Err(err) => return Err(err.into()),
            };

        let segment_name = dir.file_name().unwrap_or_default().to_string_lossy();

        // A declared field must be present *and its definition must match* — not just its
        // name. A same-name-different-definition index (keyword vs integer, a tokenizer or
        // `is_tenant` change) would otherwise pass here, and `payload_index.json` would then
        // declare an index the segment does not actually carry, forcing a full index build on
        // the serving filesystem at load. Placement spelling differences (`on_disk` vs
        // `memory`) are *not* a mismatch — resolved-equivalence via `no_change_needed`, the
        // same equivalence the server uses to decide a rebuild is unnecessary.
        for (field, want) in &expected {
            match found.get(field) {
                Some(have)
                    if segment::index::field_index::schema_transition::no_change_needed(
                        have, want,
                    ) => {}
                Some(have) => bail!(
                    "segment {segment_name} carries payload index `{field}` with a different \
                     definition than the document declares:\n  segment:  {have:?}\n  \
                     document: {want:?}\n\nThe document's `payload_index` section must be the \
                     one the segments were built against; rebuild if it was edited.",
                ),
                None => bail!(
                    "segment {segment_name} is missing the declared payload index `{field}`. \
                     If a build was resumed against an edited document, the segments are \
                     inconsistent and must be rebuilt.",
                ),
            }
        }

        // A field indexed on the segment but not declared would push a stale index into the
        // shipped `payload_index.json`.
        for field in found.keys() {
            if !expected.contains_key(field) {
                bail!(
                    "segment {segment_name} carries payload index `{field}` that the document \
                     does not declare; rebuild against the correct document.",
                );
            }
        }
    }

    Ok(())
}

/// How many segments in the directory accept writes.
fn appendable_segment_count(segments_dir: &Path) -> Result<usize> {
    #[derive(Deserialize)]
    struct ConfigOnly {
        config: segment::types::SegmentConfig,
    }

    let mut appendable = 0;
    for path in segment_dirs(segments_dir)? {
        let state_path = path.join("segment.json");
        let bytes = fs_err::read(&state_path)
            .with_context(|| format!("cannot read {}", state_path.display()))?;
        let state: ConfigOnly = serde_json::from_slice(&bytes)
            .with_context(|| format!("cannot parse {}", state_path.display()))?;
        if state.config.is_appendable() {
            appendable += 1;
        }
    }

    Ok(appendable)
}

/// Segment directories, skipping scratch and hidden entries.
pub(crate) fn segment_dirs(segments_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();

    for entry in fs_err::read_dir(segments_dir)
        .with_context(|| format!("cannot read {}", segments_dir.display()))?
    {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if name.starts_with('.') || name == shard::optimizers::config::TEMP_SEGMENTS_PATH {
            continue;
        }
        dirs.push(path);
    }

    dirs.sort();
    Ok(dirs)
}

/// Versions of every segment in the directory.
///
/// A segment with no version has never had an operation applied, and is **skipped** rather than
/// counted — `ensure_appendable_segment` deliberately creates exactly such a segment, so its
/// absence of a version is expected and says nothing about data loss. Skipping also makes
/// `assemble` idempotent: a second run sees the appendable segment from the first, ignores it,
/// and computes the same WAL base.
///
/// The cost of that choice is that a *built* segment which somehow ingested no points would be
/// skipped here too. Phase 3 catches that case instead: `built_segment_dir` selects the segment
/// with `version: Some(_)`, so a segment that applied no operations fails the build rather than
/// reaching this function.
fn segment_versions(segments_dir: &Path) -> Result<Vec<u64>> {
    #[derive(Deserialize)]
    struct VersionOnly {
        version: Option<u64>,
    }

    let mut versions = Vec::new();

    for path in segment_dirs(segments_dir)? {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // `normalize_segment_dir` takes a segment's UUID from its directory name and *deletes*
        // any directory whose name is not a UUID... after renaming it. Catch a bad name here so
        // the artifact is not quietly renamed at load, which would break the pinned-UUID
        // contract the plan relies on.
        if uuid::Uuid::try_parse(&name).is_err() {
            bail!(
                "segment directory name '{name}' is not a UUID; Qdrant would rename it at load \
                 and the plan's pinned UUID would no longer match",
            );
        }

        let state_path = path.join("segment.json");
        let bytes = fs_err::read(&state_path)
            .with_context(|| format!("cannot read {}", state_path.display()))?;
        let state: VersionOnly = serde_json::from_slice(&bytes)
            .with_context(|| format!("cannot parse {}", state_path.display()))?;

        // The empty appendable segment added by `ensure_appendable_segment` has no version, by
        // definition. Skip it rather than counting it: it contributes no points and its absence
        // of a version says nothing about whether data was lost.
        if let Some(version) = state.version {
            versions.push(version);
        } else {
            log::debug!("{} has no version (empty segment)", path.display());
        }
    }

    Ok(versions)
}

/// Create an empty WAL whose next append lands above every segment version.
///
/// Uses `Wal::generate_empty_wal_starting_at_index` (`lib/wal/src/lib.rs:106`), the same helper
/// `snapshot_empty_wal` uses when producing a WAL-less shard snapshot
/// (`local_shard/snapshot.rs:157`). Passing `index` yields a WAL with
/// `first_index() == last_index() == index + 1` and a next append of `index + 1`.
fn write_empty_wal(config: &LoadedConfig, shard_dir: &Path, index: u64) -> Result<()> {
    let wal_dir = shard_dir.join(WAL_DIR);

    // Regenerate rather than reuse: a stale WAL could carry a base below the current segment
    // versions, which is exactly the silent-skip hazard this function exists to avoid.
    if wal_dir.exists() {
        fs_err::remove_dir_all(&wal_dir)
            .with_context(|| format!("cannot clear {}", wal_dir.display()))?;
    }
    fs_err::create_dir_all(&wal_dir)
        .with_context(|| format!("cannot create {}", wal_dir.display()))?;

    let wal_config = &config.config.wal_config;
    let options = WalOptions {
        segment_capacity: wal_config.wal_capacity_mb * 1024 * 1024,
        segment_queue_len: wal_config.wal_segments_ahead,
        retain_closed: NonZeroUsize::new(wal_config.wal_retain_closed.max(1))
            .expect("max(1) is non-zero"),
    };

    Wal::generate_empty_wal_starting_at_index(&wal_dir, &options, index)
        .map_err(|err| anyhow::anyhow!("cannot create empty WAL: {err}"))?;

    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    common::fs::atomic_save_json(path, value)
        .with_context(|| format!("cannot write {}", path.display()))
}

/// Check an assembled shard looks loadable, without starting Qdrant.
///
/// Mirrors the conditions `LocalShard::check_data` and the restore path apply
/// (`lib/shard/src/files/mod.rs:59`): both `wal/` and `segments/` must exist, plus the
/// shard-level files the replica-set restore reads.
pub fn verify_assembled(shard_dir: &Path) -> Result<()> {
    let mut missing = Vec::new();

    for required in [
        WAL_DIR,
        shard::files::SEGMENTS_PATH,
        APPLIED_SEQ_FILE,
        PAYLOAD_INDEX_FILE,
        REPLICA_STATE_FILE,
        SHARD_CONFIG_FILE,
    ] {
        if !shard_dir.join(required).exists() {
            missing.push(required);
        }
    }

    if !missing.is_empty() {
        bail!(
            "{} is not a complete shard: missing {}",
            shard_dir.display(),
            missing.join(", "),
        );
    }

    // The invariant that matters: the WAL must not hand out op numbers a segment would skip.
    let versions = segment_versions(&shard_dir.join(shard::files::SEGMENTS_PATH))?;
    let max_version = versions.iter().copied().max().unwrap_or(0);

    let wal = Wal::open(shard_dir.join(WAL_DIR))
        .map_err(|err| anyhow::anyhow!("cannot open assembled WAL: {err}"))?;
    let next_append = wal.last_index();

    if next_append <= max_version {
        bail!(
            "assembled WAL would hand out op number {next_append}, but a segment already has \
             version {max_version}. The first write after restore would be silently skipped by \
             `handle_segment_version`. Re-run the assemble phase.",
        );
    }

    let applied: AppliedSeqFile =
        serde_json::from_slice(&fs_err::read(shard_dir.join(APPLIED_SEQ_FILE))?)?;
    if applied.op_num > next_append {
        bail!(
            "applied_seq {} exceeds the WAL's last index {next_append}; startup asserts the \
             opposite",
            applied.op_num,
        );
    }

    Ok(())
}

/// Every `shard_N` directory under an output root, sorted by id.
pub fn discover_shards(out: &Path) -> Result<Vec<(ShardId, PathBuf)>> {
    let mut shards = Vec::new();

    for entry in fs_err::read_dir(out).with_context(|| format!("cannot read {}", out.display()))? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if let Some(id) = name.strip_prefix("shard_")
            && let Ok(id) = id.parse::<ShardId>()
        {
            shards.push((id, path));
        }
    }

    shards.sort_by_key(|(id, _)| *id);

    if shards.is_empty() {
        bail!("no shard_N directories found under {}", out.display());
    }

    Ok(shards)
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::config;

    fn loaded_config() -> LoadedConfig {
        config::from_str(&config::tests::valid_config_json().to_string()).unwrap()
    }

    /// Build a fake shard directory with segments at the given versions.
    fn fake_shard(dir: &Path, versions: &[Option<u64>]) {
        // The build fingerprint `assemble` now requires; matches `loaded_config()`.
        fs_err::create_dir_all(dir).unwrap();
        fs_err::write(
            dir.join(crate::build::BUILD_FINGERPRINT_FILE),
            &loaded_config().fingerprint,
        )
        .unwrap();
        let segments = dir.join(shard::files::SEGMENTS_PATH);
        for (index, version) in versions.iter().enumerate() {
            // Deterministic valid UUIDs, since a non-UUID name is rejected.
            let uuid = uuid::Uuid::new_v5(
                &uuid::Uuid::NAMESPACE_OID,
                format!("test-segment-{index}").as_bytes(),
            );
            let segment = segments.join(uuid.to_string());
            fs_err::create_dir_all(&segment).unwrap();
            // A real `SegmentConfig`: `appendable_segment_count` parses this, so an empty
            // object would fail. `Plain` index + `InRamChunkedMmap` storage makes it appendable,
            // which is what a fresh staging segment looks like.
            let body = serde_json::json!({
                "version": version,
                "config": {
                    "vector_data": {},
                    "sparse_vector_data": {},
                    "payload_storage_type": { "type": "in_ram_mmap" },
                },
            });
            fs_err::write(segment.join("segment.json"), body.to_string()).unwrap();
        }
    }

    #[test]
    fn assembles_a_complete_shard() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fake_shard(dir.path(), &[Some(3), Some(7), Some(5)]);

        let report = assemble_shard(&loaded_config(), dir.path(), 0, None).unwrap();
        assert_eq!(report.segments, 3);
        assert_eq!(report.max_segment_version, 7);
        assert_eq!(report.wal_base, 8);

        verify_assembled(dir.path()).expect("assembled shard must verify");

        // `LocalShard::check_data` requires exactly these two.
        assert!(dir.path().join("wal").is_dir());
        assert!(dir.path().join("segments").is_dir());
    }

    /// The invariant the whole phase exists for.
    #[test]
    fn wal_base_is_above_every_segment_version() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fake_shard(dir.path(), &[Some(1), Some(42), Some(9)]);

        let report = assemble_shard(&loaded_config(), dir.path(), 0, None).unwrap();

        let wal = Wal::open(dir.path().join("wal")).unwrap();
        assert_eq!(wal.last_index(), report.wal_base);
        assert!(
            wal.last_index() > 42,
            "the WAL must not hand out an op number a segment would skip",
        );
    }

    #[test]
    fn applied_seq_does_not_exceed_the_wal_last_index() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fake_shard(dir.path(), &[Some(11)]);
        assemble_shard(&loaded_config(), dir.path(), 0, None).unwrap();

        let applied: AppliedSeqFile =
            serde_json::from_slice(&fs_err::read(dir.path().join(APPLIED_SEQ_FILE)).unwrap())
                .unwrap();
        let wal = Wal::open(dir.path().join("wal")).unwrap();

        // Startup asserts `applied_seq <= wal.last_index()` in debug builds.
        assert!(applied.op_num <= wal.last_index());
        assert_eq!(applied.op_num, 11);
    }

    /// The fork's restore path refuses a shard whose recorded ring scale differs from the
    /// collection's — and reads an *absent* scale as the pre-field default of 100. So the
    /// stamp must be present and carry the document's value.
    #[test]
    fn shard_config_records_the_ring_scale() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fake_shard(dir.path(), &[Some(1)]);

        let mut value = config::tests::valid_config_json();
        value["params"]["hash_ring_shard_scale"] = serde_json::json!(500);
        let loaded = config::from_str(&value.to_string()).unwrap();
        // This shard is built under the scale-500 document, so its build fingerprint must
        // match it (the gate would otherwise correctly refuse the scale drift).
        fs_err::write(
            dir.path().join(crate::build::BUILD_FINGERPRINT_FILE),
            &loaded.fingerprint,
        )
        .unwrap();
        assemble_shard(&loaded, dir.path(), 0, None).unwrap();

        let written: collection::shards::shard_config::ShardConfig =
            serde_json::from_slice(&fs_err::read(dir.path().join(SHARD_CONFIG_FILE)).unwrap())
                .unwrap();
        assert_eq!(written.hash_ring_shard_scale, Some(500));
        assert_eq!(written.snapshot_hash_ring_shard_scale(), 500);
    }

    /// A document that differs from the one the segments were built under is refused, so the
    /// ring scale (and payload schema) cannot be silently restamped at assemble.
    #[test]
    fn rejects_a_document_that_differs_from_the_build() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fake_shard(dir.path(), &[Some(1)]); // fingerprint = valid_config_json (scale 100)

        let mut value = config::tests::valid_config_json();
        value["params"]["hash_ring_shard_scale"] = serde_json::json!(500);
        let drifted = config::from_str(&value.to_string()).unwrap();

        let err = assemble_shard(&drifted, dir.path(), 0, None).unwrap_err();
        assert!(
            format!("{err:#}").contains("different document"),
            "the mismatch must be refused, got: {err:#}",
        );
    }

    /// A shard with no build fingerprint (not produced by this tool's build phase) is refused
    /// rather than assembled unverifiably.
    #[test]
    fn rejects_a_shard_without_a_build_fingerprint() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fake_shard(dir.path(), &[Some(1)]);
        fs_err::remove_file(dir.path().join(crate::build::BUILD_FINGERPRINT_FILE)).unwrap();

        let err = assemble_shard(&loaded_config(), dir.path(), 0, None).unwrap_err();
        assert!(
            format!("{err:#}").contains("no build fingerprint"),
            "an unstamped shard must be refused, got: {err:#}",
        );
    }

    #[test]
    fn replica_state_marks_the_shard_local() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fake_shard(dir.path(), &[Some(1)]);
        assemble_shard(&loaded_config(), dir.path(), 0, None).unwrap();

        let state: ReplicaStateFile =
            serde_json::from_slice(&fs_err::read(dir.path().join(REPLICA_STATE_FILE)).unwrap())
                .unwrap();
        assert!(
            state.is_local,
            "restore will not install data for a non-local replica",
        );
    }

    /// A versionless segment is legitimate: the empty appendable one has no version by
    /// definition, so it must be tolerated alongside real segments rather than rejected.
    #[test]
    fn tolerates_a_versionless_segment_alongside_real_ones() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fake_shard(dir.path(), &[Some(2), None, Some(6)]);

        let report = assemble_shard(&loaded_config(), dir.path(), 0, None).unwrap();
        assert_eq!(report.max_segment_version, 6);
        assert_eq!(report.wal_base, 7);
        // Every segment directory is counted, versioned or not. Counting only versioned ones
        // made an unchanged shard report a different number on a re-run, because the appendable
        // segment gains a version once it is created and indexed.
        assert_eq!(report.segments, 3);
        verify_assembled(dir.path()).unwrap();
    }

    /// A shard where nothing has a version holds no points at all.
    #[test]
    fn rejects_a_shard_where_no_segment_has_points() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fake_shard(dir.path(), &[None, None]);

        let err = assemble_shard(&loaded_config(), dir.path(), 0, None).unwrap_err();
        assert!(
            format!("{err:#}").contains("contains no segments"),
            "{err:#}",
        );
    }

    #[test]
    fn rejects_a_non_uuid_segment_directory() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fs_err::write(
            dir.path().join(crate::build::BUILD_FINGERPRINT_FILE),
            &loaded_config().fingerprint,
        )
        .unwrap();
        let segments = dir.path().join(shard::files::SEGMENTS_PATH);
        fs_err::create_dir_all(segments.join("not-a-uuid")).unwrap();
        fs_err::write(
            segments.join("not-a-uuid").join("segment.json"),
            serde_json::json!({
                "version": 1,
                "config": {
                    "vector_data": {},
                    "sparse_vector_data": {},
                    "payload_storage_type": { "type": "in_ram_mmap" },
                },
            })
            .to_string(),
        )
        .unwrap();

        let err = assemble_shard(&loaded_config(), dir.path(), 0, None).unwrap_err();
        assert!(format!("{err:#}").contains("is not a UUID"), "{err:#}");
    }

    #[test]
    fn rejects_a_shard_with_no_segments() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fs_err::create_dir_all(dir.path().join(shard::files::SEGMENTS_PATH)).unwrap();
        fs_err::write(
            dir.path().join(crate::build::BUILD_FINGERPRINT_FILE),
            &loaded_config().fingerprint,
        )
        .unwrap();

        let err = assemble_shard(&loaded_config(), dir.path(), 0, None).unwrap_err();
        assert!(
            format!("{err:#}").contains("contains no segments"),
            "{err:#}"
        );
    }

    /// Re-assembling must regenerate the WAL, not trust a stale one.
    #[test]
    fn reassembly_rebases_the_wal() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fake_shard(dir.path(), &[Some(3)]);
        assemble_shard(&loaded_config(), dir.path(), 0, None).unwrap();
        assert_eq!(Wal::open(dir.path().join("wal")).unwrap().last_index(), 4);

        // A later build produced a higher-versioned segment.
        fake_shard(dir.path(), &[Some(3), Some(99)]);
        let report = assemble_shard(&loaded_config(), dir.path(), 0, None).unwrap();

        assert_eq!(report.wal_base, 100);
        assert_eq!(Wal::open(dir.path().join("wal")).unwrap().last_index(), 100);
        verify_assembled(dir.path()).unwrap();
    }

    /// `verify_assembled` must actually catch a WAL that is too low.
    #[test]
    fn verify_catches_a_wal_below_a_segment_version() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fake_shard(dir.path(), &[Some(5)]);
        assemble_shard(&loaded_config(), dir.path(), 0, None).unwrap();
        verify_assembled(dir.path()).unwrap();

        // Add a segment with a higher version without re-assembling: the WAL is now stale.
        fake_shard(dir.path(), &[Some(5), Some(500)]);

        let err = verify_assembled(dir.path()).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("silently skipped"), "{message}");
    }

    #[test]
    fn verify_reports_missing_files() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        fake_shard(dir.path(), &[Some(1)]);

        let err = verify_assembled(dir.path()).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("not a complete shard"), "{message}");
        assert!(message.contains("wal"), "{message}");
    }

    #[test]
    fn discover_shards_sorts_by_id() {
        let dir = TempDir::with_prefix("assemble").unwrap();
        for id in [7u32, 0, 3] {
            fs_err::create_dir_all(dir.path().join(format!("shard_{id}"))).unwrap();
        }
        fs_err::create_dir_all(dir.path().join("not-a-shard")).unwrap();

        let shards = discover_shards(dir.path()).unwrap();
        assert_eq!(
            shards.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![0, 3, 7],
        );
    }
}
