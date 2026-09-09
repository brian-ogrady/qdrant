//! Phase 1: partition input files by shard.
//!
//! Reads every input file once, routes each point through the hash ring, and appends it to a
//! part file for its shard. No sorting, no shuffle, no global view — routing is a pure
//! function of the point id, so this is a streaming scatter.
//!
//! Inputs are read through [`InputStore`] (see `store.rs`): a local tree today, an S3 prefix
//! later, with this phase unchanged. The *work* directory is always a local (or cluster)
//! filesystem — parts are written, fsynced and renamed here, which is not an object-store
//! access pattern.
//!
//! # Resumability
//!
//! Four rules, and the first three are what make a resumed run indistinguishable from a clean
//! one rather than merely "usually fine":
//!
//! 1. **Output names are a pure function of (config, input file).** A part is
//!    `scatter/shard_{s}/part_{file_id}` where `file_id` is derived from the input's
//!    store-relative path — never from a worker index, timestamp, or iteration order.
//!    Reprocessing an input file therefore *overwrites* its parts deterministically instead of
//!    appending duplicates, which makes the whole phase idempotent by construction. Hashing the
//!    store-relative path (rather than whatever path was typed) also means every process must
//!    point at the same `--input` root — and that two mounts of the same corpus agree on ids.
//! 2. **Temp-then-rename.** A part is only visible under its final name after fsync + rename
//!    ([`PartWriter::commit`]), so a crash leaves `.tmp` debris rather than a
//!    committed-looking file with a truncated tail.
//! 3. **Completion markers are an optimization, not a correctness dependency.** Deleting all
//!    of `done/` forces a full re-scatter and must produce byte-identical output. A test
//!    asserts exactly that.
//! 4. **The part fingerprint is checked before resuming.** Resuming into a work directory
//!    scattered under a different routing ring or dense element width would mix incompatible
//!    records, so it is refused outright. Note the surface is deliberately narrow: retuning
//!    `hnsw_config` or `max_segment_size` leaves a scatter reusable, because neither changes
//!    what was written here. See [`crate::config::part_fingerprint`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context as _, Result, bail};
use collection::shards::shard::ShardId;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::config::LoadedConfig;
use crate::partfile::{PartHeader, PartWriter};
use crate::ring::ShardRouter;
use crate::source;
use crate::store::InputStore;

/// Conservative ceiling on concurrently open output files.
///
/// A worker holds one writer per shard it has seen while processing a file, so open
/// descriptors reach `workers * shard_count`. Typical `ulimit -n` is 1024, and exhausting it
/// mid-run produces a confusing failure deep in a write, so bound it up front with an error
/// that says which knob to turn.
const MAX_OPEN_PART_FILES: usize = 512;

/// Layout of the scatter working directory.
pub struct ScatterLayout {
    root: PathBuf,
}

impl ScatterLayout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn state_path(&self) -> PathBuf {
        self.root.join("scatter_state.json")
    }

    /// Where the file_id -> path mapping for one process's inputs is recorded.
    ///
    /// Per slice, because `manifest.jsonl` is written whole rather than appended: with several
    /// processes sharing a work directory, one filename would leave only the last writer's
    /// mapping and the rest of the corpus would be untraceable.
    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.jsonl")
    }

    pub fn slice_manifest_path(&self, slice: Option<(usize, usize)>) -> PathBuf {
        match slice {
            Some((index, total)) => self.root.join(format!("manifest.{index}-of-{total}.jsonl")),
            None => self.manifest_path(),
        }
    }

    pub fn done_dir(&self) -> PathBuf {
        self.root.join("done")
    }

    pub fn done_marker(&self, file_id: &str) -> PathBuf {
        self.done_dir().join(file_id)
    }

    pub fn verified_dir(&self) -> PathBuf {
        self.root.join("verified")
    }

    /// Marker recording that one part file has been checked.
    ///
    /// Keyed by shard and part, because a part exists once per shard it has points for.
    pub fn verified_marker(&self, shard_id: ShardId, part: &str) -> PathBuf {
        self.verified_dir().join(format!("{shard_id}_{part}"))
    }

    pub fn shard_dir(&self, shard_id: ShardId) -> PathBuf {
        self.root.join(format!("shard_{shard_id}"))
    }

    pub fn part_path(&self, shard_id: ShardId, file_id: &str) -> PathBuf {
        self.shard_dir(shard_id).join(format!("part_{file_id}"))
    }
}

/// Persisted marker tying a work directory to one routing ring and dense encoding.
///
/// Holds the *part* fingerprint, not the full config one: a work directory stays resumable
/// across an `hnsw_config` or `max_segment_size` change, since neither affects what was written.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScatterState {
    part_fingerprint: String,
    shard_number: u32,
}

/// One input file, with the stable id its outputs are named after.
///
/// `path` is store-relative — resolved through the [`InputStore`] the run was given, and the
/// string [`file_id_for`] hashes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct InputFile {
    pub path: PathBuf,
    pub file_id: String,
}

/// Derive a stable id for an input file.
///
/// Hashes the store-relative path rather than using a position in a sorted list, so adding or
/// removing input files does not renumber everything else and invalidate a partially-finished
/// run. Truncated to 16 hex chars: 64 bits of collision resistance over a set of filenames, with
/// the full path recorded in the manifest so an id can always be traced back.
pub fn file_id_for(path: &Path) -> String {
    let digest = Sha256::digest(path.to_string_lossy().as_bytes());
    digest[..8].iter().fold(String::new(), |mut acc, byte| {
        use std::fmt::Write as _;
        let _ = write!(acc, "{byte:02x}");
        acc
    })
}

/// Outcome of a scatter run.
#[derive(Debug, Default, PartialEq)]
pub struct ScatterStats {
    pub files_processed: u64,
    pub files_skipped: u64,
    pub points: u64,
    pub parts_written: u64,
    pub temp_files_swept: u64,
    /// Files that could not be read. They are reported, not fatal — see [`run`].
    pub files_failed: Vec<FailedFile>,
}

/// An input file that could not be scattered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailedFile {
    pub path: String,
    pub file_id: String,
    pub error: String,
}

/// How a scatter run is tuned, as opposed to what it operates on.
pub struct ScatterOptions<'a> {
    /// Reader threads. Bounded by `workers * shards` open output files.
    pub workers: usize,
    /// Column mapping, required for Parquet input.
    pub mapping: Option<&'a crate::parquet_source::ParquetMapping>,
    /// This machine's share of the input, as `(index, total)`.
    pub slice: Option<(usize, usize)>,
    /// Stop after this many unreadable files.
    pub max_failures: usize,
}

/// Run (or resume) the scatter phase.
///
/// An input file that cannot be read is **skipped and reported**, not fatal: one bad file among
/// thousands should not discard the work already finished, and the file is left without a `done`
/// marker so a later run retries it. `max_failures` bounds that tolerance, because a mapping that
/// does not match the corpus fails every file and should be caught immediately rather than after
/// reading everything.
pub fn run(
    config: &LoadedConfig,
    router: &ShardRouter,
    store: &dyn InputStore,
    inputs: &[InputFile],
    layout: &ScatterLayout,
    options: &ScatterOptions<'_>,
) -> Result<ScatterStats> {
    let &ScatterOptions {
        workers,
        mapping,
        slice,
        max_failures,
    } = options;

    if workers == 0 {
        bail!("--workers must be at least 1");
    }

    let open_files = workers.saturating_mul(router.shard_count() as usize);
    if open_files > MAX_OPEN_PART_FILES {
        bail!(
            "workers ({workers}) x shards ({}) = {open_files} concurrently open part files, \
             above the {MAX_OPEN_PART_FILES} limit.\n\n\
             Each worker holds one output file per shard while processing an input file. \
             Lower --workers to at most {} for this shard count.",
            router.shard_count(),
            (MAX_OPEN_PART_FILES / router.shard_count().max(1) as usize).max(1),
        );
    }

    fs_err::create_dir_all(&layout.root)
        .with_context(|| format!("cannot create {}", layout.root.display()))?;
    fs_err::create_dir_all(layout.done_dir())?;

    reconcile_state(config, router, layout)?;
    write_manifest(inputs, layout, slice)?;

    let temp_files_swept = sweep_temp_files(layout, inputs)?;
    if temp_files_swept > 0 {
        log::info!("swept {temp_files_swept} incomplete .tmp part files from a previous run");
    }

    // Skip files already finished. The marker is only an optimization: with `done/` removed,
    // every file is reprocessed and the output is identical.
    let (pending, skipped): (Vec<_>, Vec<_>) = inputs
        .iter()
        .cloned()
        .partition(|file| !layout.done_marker(&file.file_id).exists());

    if !skipped.is_empty() {
        log::info!(
            "resuming: {} of {} input files already scattered",
            skipped.len(),
            inputs.len(),
        );
    }

    let points = AtomicU64::new(0);
    let parts_written = AtomicU64::new(0);
    let files_processed = AtomicU64::new(0);
    // Counts only the files this run will actually read: already-scattered files are skipped, so
    // including them would make a resumed run appear to stall at the start.
    let progress = crate::progress::Progress::new("scatter", "file", pending.len() as u64);
    let failed: Mutex<Vec<FailedFile>> = Mutex::new(Vec::new());
    let queue = Mutex::new(pending.into_iter());

    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(workers);

        for worker in 0..workers {
            let queue = &queue;
            let points = &points;
            let parts_written = &parts_written;
            let files_processed = &files_processed;
            let progress = &progress;
            let failed = &failed;

            handles.push(
                std::thread::Builder::new()
                    .name(format!("scatter-{worker}"))
                    .spawn_scoped(scope, move || -> Result<()> {
                        loop {
                            // Held only long enough to pop; never across the file's IO.
                            let next = queue.lock().expect("scatter queue poisoned").next();
                            let Some(file) = next else { return Ok(()) };

                            let outcome =
                                match scatter_file(config, router, store, layout, &file, mapping) {
                                    Ok(outcome) => outcome,
                                    Err(err) => {
                                        // One unreadable file among thousands must not discard the
                                        // work already done. It is recorded, left without a `done`
                                        // marker so a later run retries it, and its uncommitted
                                        // `.tmp` parts are swept then too — so nothing partial is
                                        // ever visible under a final name.
                                        log::warn!("skipping {}: {err:#}", file.path.display(),);
                                        failed.lock().expect("failure list poisoned").push(
                                            FailedFile {
                                                path: file.path.display().to_string(),
                                                file_id: file.file_id.clone(),
                                                error: format!("{err:#}"),
                                            },
                                        );
                                        progress.advance(0);

                                        // A systematically wrong mapping fails every file. Stop
                                        // rather than reading the whole corpus to prove it.
                                        let count =
                                            failed.lock().expect("failure list poisoned").len();
                                        if count > max_failures {
                                            bail!(
                                                "{count} input files failed, above the \
                                                 --max-failures limit of {max_failures}. This \
                                                 usually means the mapping does not match the \
                                                 corpus rather than that individual files are bad; \
                                                 the first was: {}",
                                                failed.lock().expect("poisoned")[0].error,
                                            );
                                        }
                                        continue;
                                    }
                                };

                            points.fetch_add(outcome.points, Ordering::Relaxed);
                            parts_written.fetch_add(outcome.parts, Ordering::Relaxed);
                            files_processed.fetch_add(1, Ordering::Relaxed);
                            progress.advance(outcome.points);
                        }
                    })
                    .context("cannot spawn scatter worker")?,
            );
        }

        // Propagate the first worker error; a panic surfaces as a join error.
        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("scatter worker panicked"))??;
        }

        Ok(())
    })?;

    progress.finish();

    let failed = failed.into_inner().unwrap_or_else(|err| err.into_inner());
    if !failed.is_empty() {
        write_failures(&failed, layout, slice)?;
    }

    Ok(ScatterStats {
        files_failed: failed,
        files_processed: files_processed.load(Ordering::Relaxed),
        files_skipped: skipped.len() as u64,
        points: points.load(Ordering::Relaxed),
        parts_written: parts_written.load(Ordering::Relaxed),
        temp_files_swept,
    })
}

struct FileOutcome {
    points: u64,
    parts: u64,
}

/// Record what the parts about to be written will contain.
///
/// Vector names and dimensionality come from the collection config, since that is what the scatter
/// captures under. Source columns come from the mapping when there is one — JSONL input has none,
/// and a `None` there means "unknown", which the build treats as unverifiable rather than absent.
fn manifest_for(
    config: &LoadedConfig,
    mapping: Option<&crate::parquet_source::ParquetMapping>,
) -> crate::partfile::PartManifest {
    use segment::types::VectorStorageDatatype;

    use crate::dense_codec::DenseEncoding;
    use crate::partfile::{DenseSpec, PartManifest, SparseSpec};

    let mut dense = BTreeMap::new();
    for (name, params) in config.config.params.vectors.params_iter() {
        dense.insert(
            name.to_string(),
            DenseSpec {
                encoding: DenseEncoding::for_datatype(
                    params.datatype.map(VectorStorageDatatype::from),
                ),
                dim: params.size.get() as usize,
                source: mapping.and_then(|m| m.dense_vectors.get(name).cloned()),
            },
        );
    }

    let mut sparse = BTreeMap::new();
    for name in config
        .config
        .params
        .sparse_vectors
        .iter()
        .flat_map(|map| map.keys())
    {
        let name: String = name.clone();
        let source = mapping.and_then(|m| m.sparse_vectors.get(&name).map(|s| s.column.clone()));
        sparse.insert(name, SparseSpec { source });
    }

    PartManifest {
        dense,
        sparse,
        // Only the mapping enumerates columns. JSONL payloads are whatever each line held, so
        // there is no set to record and no subset check to make later.
        payload: mapping.map(|m| m.payload_columns.clone()),
    }
}

/// Scatter one input file into per-shard part files, then mark it done.
fn scatter_file(
    config: &LoadedConfig,
    router: &ShardRouter,
    store: &dyn InputStore,
    layout: &ScatterLayout,
    file: &InputFile,
    mapping: Option<&crate::parquet_source::ParquetMapping>,
) -> Result<FileOutcome> {
    let format = source::detect(&file.path)?;
    let mut reader = format.open(store, &file.path, mapping)?;

    // Writers are created lazily, so a file whose points miss a shard entirely leaves no
    // empty part there. Phase 2 must therefore treat a missing part as "no points", not as an
    // error.
    let encodings = crate::dense_codec::encodings_for(config);
    // Recorded once per file and copied into every part header, so each part can be interpreted
    // and subset-checked without reference to the config that produced it.
    let manifest = manifest_for(config, mapping);
    let mut writers: BTreeMap<ShardId, PartWriter> = BTreeMap::new();
    let mut points = 0u64;

    while let Some(point) = reader.next_point()? {
        let shard_id = router.shard_of(point.id)?;

        let writer = match writers.get_mut(&shard_id) {
            Some(writer) => writer,
            None => {
                let header = PartHeader {
                    part_fingerprint: config.part_fingerprint.clone(),
                    file_id: file.file_id.clone(),
                    source_path: file.path.display().to_string(),
                    shard_id,
                    manifest: manifest.clone(),
                };
                let path = layout.part_path(shard_id, &file.file_id);
                writers.insert(shard_id, PartWriter::create(path, &header)?);
                writers.get_mut(&shard_id).expect("just inserted")
            }
        };

        // Packed here rather than in the reader, so the source stays a plain `PointSource`.
        // Narrowing is lossless for f16-configured vectors: the values arrived as f16.
        let record = crate::partfile::PartRecord::from_point(point, &encodings)?;
        writer.append(&record)?;
        points += 1;
    }

    // Commit every part before the done marker. If we crash between commits, the marker is
    // absent, so the next run redoes the whole file and overwrites all of its parts.
    let mut parts = 0u64;
    for (_shard_id, writer) in writers {
        writer.commit()?;
        parts += 1;
    }

    mark_done(layout, &file.file_id)?;

    Ok(FileOutcome { points, parts })
}

/// Record that an input file is fully scattered.
fn mark_done(layout: &ScatterLayout, file_id: &str) -> Result<()> {
    let path = layout.done_marker(file_id);
    let file =
        fs_err::File::create(&path).with_context(|| format!("cannot create {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("cannot fsync {}", path.display()))?;
    // fsync `done/` so the marker's directory entry survives a crash. Not a safety
    // requirement — a lost marker only causes an idempotent reprocess — but it makes resume
    // skip finished files as intended. The safety-critical ordering (part durable before
    // marker) is upheld by `PartWriter::commit` fsyncing the shard dir before this runs.
    common::fs::sync_parent_dir(&path)
        .with_context(|| format!("cannot fsync parent dir of {}", path.display()))?;
    Ok(())
}

/// Refuse a work directory that was scattered under an incompatible routing config.
///
/// For phases that read the scatter output without opening the parts themselves. `plan` is the case
/// that matters: it reads each part's `.meta` sidecar, which carries no fingerprint, so nothing else
/// would notice that the ring has changed. A `shard_number` reduced between scatter and plan would
/// otherwise plan only the shards that still exist and silently drop the rest.
///
/// Deliberately the *part* fingerprint, not the config fingerprint: tuning `hnsw_config` between
/// scatter and plan is supported and must stay that way.
pub fn check_part_fingerprint(config: &LoadedConfig, layout: &ScatterLayout) -> Result<()> {
    let path = layout.state_path();
    let Ok(bytes) = fs_err::read(&path) else {
        // No state file: an older work directory, or one that was never scattered. The parts
        // themselves still carry the fingerprint, so this is not the only line of defence.
        return Ok(());
    };

    let existing: ScatterState = serde_json::from_slice(&bytes)
        .with_context(|| format!("cannot parse {}", path.display()))?;

    if existing.part_fingerprint != config.part_fingerprint {
        bail!(
            "work directory {} was scattered under an incompatible collection config\n  \
             scattered fingerprint: {}\n  \
             current fingerprint:   {}\n\n\
             One of `shard_number`, `sharding_method`, `hash_ring_shard_scale`, or the mapping's \
             `id_column`/`id_format` changed, so the parts were routed by a different ring than \
             this config describes. Planning them would silently mis-place or drop points.\n\n\
             Changes to `hnsw_config`, `quantization_config` or `max_segment_size` do not reach \
             this check — those may be tuned freely between scatter and plan.",
            layout.root.display(),
            existing.part_fingerprint,
            config.part_fingerprint,
        );
    }

    Ok(())
}

/// Write the state file a work directory would have after scattering. Tests only.
#[cfg(test)]
pub fn write_state_for_test(config: &LoadedConfig, layout: &ScatterLayout) -> Result<()> {
    fs_err::create_dir_all(&layout.root)?;
    let state = ScatterState {
        part_fingerprint: config.part_fingerprint.clone(),
        shard_number: config.shard_number(),
    };
    common::fs::atomic_save_json(&layout.state_path(), &state)?;
    Ok(())
}

fn reconcile_state(
    config: &LoadedConfig,
    router: &ShardRouter,
    layout: &ScatterLayout,
) -> Result<()> {
    let state = ScatterState {
        part_fingerprint: config.part_fingerprint.clone(),
        shard_number: router.shard_count(),
    };
    let path = layout.state_path();

    if path.exists() {
        let existing: ScatterState = serde_json::from_slice(&fs_err::read(&path)?)
            .with_context(|| format!("cannot parse {}", path.display()))?;

        if existing.part_fingerprint != state.part_fingerprint {
            bail!(
                "work directory {} was scattered under an incompatible collection config\n  \
                 existing fingerprint: {}\n  \
                 current fingerprint:  {}\n\n\
                 One of `shard_number`, `sharding_method`, `hash_ring_shard_scale`, or the \
                 mapping's `id_column`/`id_format` changed. Resuming would mix records routed by \
                 different hash rings. Use a clean work directory, or re-run with the original \
                 config.\n\n\
                 Changes to `hnsw_config`, `quantization_config` or `max_segment_size` do not \
                 reach this check — they need only a re-run of `plan`.",
                layout.root.display(),
                existing.part_fingerprint,
                state.part_fingerprint,
            );
        }
        return Ok(());
    }

    common::fs::atomic_save_json(&path, &state)
        .with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

/// Record the files that could not be read, so they can be inspected or re-fed later.
///
/// Per slice for the same reason the manifest is: written whole, so one filename would leave only
/// the last writer's list when several processes share a work directory.
fn write_failures(
    failed: &[FailedFile],
    layout: &ScatterLayout,
    slice: Option<(usize, usize)>,
) -> Result<()> {
    let name = match slice {
        Some((index, total)) => format!("failed.{index}-of-{total}.jsonl"),
        None => "failed.jsonl".to_string(),
    };
    let path = layout.root.join(name);

    let mut body = String::new();
    for entry in failed {
        use std::fmt::Write as _;
        let line = serde_json::to_string(entry)?;
        let _ = writeln!(body, "{line}");
    }

    fs_err::write(&path, body).with_context(|| format!("cannot write {}", path.display()))?;
    log::warn!(
        "{} input file(s) could not be read; details in {}",
        failed.len(),
        path.display(),
    );
    Ok(())
}

/// Record file_id -> store-relative path, so an opaque id can be traced back to its input.
fn write_manifest(
    inputs: &[InputFile],
    layout: &ScatterLayout,
    slice: Option<(usize, usize)>,
) -> Result<()> {
    #[derive(Serialize)]
    struct Entry<'a> {
        file_id: &'a str,
        path: &'a str,
    }

    let mut body = String::new();
    for file in inputs {
        let path = file.path.display().to_string();
        let entry = Entry {
            file_id: &file.file_id,
            path: &path,
        };
        body.push_str(&serde_json::to_string(&entry)?);
        body.push('\n');
    }

    let path = layout.slice_manifest_path(slice);
    fs_err::write(&path, body).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

/// Delete `.tmp` parts left by an interrupted run — but only for this run's own input files.
///
/// Scoped to `inputs` rather than sweeping every `.tmp` in the tree, because several processes may
/// share one work directory when the input is split with `--slice`. A blanket sweep would unlink a
/// part another process was still writing: that process still holds the descriptor, so it would
/// keep writing to an unlinked inode and then fail its rename, losing the work with a confusing
/// error. Temps are named `part_{file_id}.{pid}.{n}.tmp`, so a *disjoint* slice on another process
/// has different file_ids and is never touched; only an overlapping (misconfigured) assignment of
/// the same file_id could hit a live sibling temp, and there the sibling merely fails its rename.
fn sweep_temp_files(layout: &ScatterLayout, inputs: &[InputFile]) -> Result<u64> {
    let mine: Vec<String> = inputs
        .iter()
        .map(|file| format!("part_{}", file.file_id))
        .collect();

    let mut swept = 0;

    let entries = match fs_err::read_dir(&layout.root) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(err) => return Err(err.into()),
    };

    for entry in entries {
        let entry = entry?;
        let dir = entry.path();

        // Only the shard directories hold parts. Restricting to them rather than walking whatever
        // is in the root matters when processes share a work directory: `atomic_save_json` writes
        // `scatter_state.json` through a transient `.atomicwrite*` entry in the same directory, so
        // a sweep that tried to descend into everything would race with another process and fail
        // on an entry that no longer exists.
        let is_shard_dir = dir
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("shard_"));
        if !is_shard_dir || !dir.is_dir() {
            continue;
        }

        let parts = match fs_err::read_dir(&dir) {
            Ok(parts) => parts,
            // Another process may be creating this shard directory right now.
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err.into()),
        };

        for part in parts {
            let part = part?;
            let path = part.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if mine
                .iter()
                .any(|final_name| crate::partfile::is_temp_for(&name, final_name))
            {
                match fs_err::remove_file(&path) {
                    Ok(()) => swept += 1,
                    // Only this run owns these names, but a retry of the same slice elsewhere
                    // could have removed it first; either way it is gone, which is the goal.
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                    Err(err) => {
                        return Err(anyhow::Error::from(err)
                            .context(format!("cannot remove {}", path.display())));
                    }
                }
            }
        }
    }

    Ok(swept)
}

/// Per-shard tally from re-reading scatter output.
#[derive(Debug, Default, PartialEq)]
pub struct VerifyReport {
    pub parts: u64,
    /// Parts already marked verified by an earlier run.
    pub parts_skipped: u64,
    pub points: u64,
    /// Points per shard, for spotting a badly skewed partition before phase 2.
    pub per_shard: BTreeMap<ShardId, u64>,
}

/// How a verify run is tuned.
pub struct VerifyOptions {
    /// Reader threads.
    pub workers: usize,
    /// This machine's share of the parts, as `(index, total)`.
    pub slice: Option<(usize, usize)>,
    /// Re-check parts already marked verified.
    pub recheck: bool,
    /// Read every record instead of checking each part's header and length.
    pub deep: bool,
}

/// One part file to check.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PartToVerify {
    shard_id: ShardId,
    /// File name, e.g. `part_453e65d8c956aa7b`.
    name: String,
    path: PathBuf,
    /// Size at discovery, recorded in the marker so a rewritten part is re-checked.
    len: u64,
}

/// Check the scatter output is intact and internally consistent.
///
/// # What is worth checking, and what is not
///
/// Re-deriving every point's shard is close to worthless as a *routing* check: it is the same pure
/// function of the id, in the same binary, so it cannot disagree. If the ring were wrong it would be
/// wrong identically on the second pass. Paying a full read of tens of terabytes for that is the
/// wrong trade.
///
/// What can actually go wrong is the file: a part truncated, a part written under a different
/// config, or a part sitting in the wrong shard's directory. All three are catchable without reading
/// the records:
///
/// * **Truncation or a short write** — `PartWriter::commit` records the part's byte length in its
///   sidecar after the rename, so comparing that to the file's current length catches it with one
///   `stat`.
/// * **A foreign config** — the part header carries the part fingerprint, which `PartReader::open`
///   checks. That is one small read at the front of the file.
/// * **A misplaced part** — the header also carries the shard id, compared against the directory it
///   was found in.
///
/// So the default is a metadata pass: two small reads per part rather than the whole thing. On the
/// production corpus that is minutes instead of hours.
///
/// `deep` restores the full read — every record decoded, counted against the sidecar, and re-routed.
/// It is worth running once over a corpus, or when a filesystem is under suspicion, because it is
/// the only thing that would catch damage in the middle of a part that left its length unchanged.
///
/// # Why this is parallel and resumable
///
/// Even the metadata pass is a hundred thousand small reads, which is the access pattern a network
/// filesystem is slowest at. So it is parallel over parts, splittable across machines with `--slice`,
/// and resumable through a marker per part. The marker records how the part was checked, so
/// switching to `deep` re-checks parts that were only checked cheaply.
pub fn verify(
    config: &LoadedConfig,
    router: &ShardRouter,
    layout: &ScatterLayout,
    options: &VerifyOptions,
) -> Result<VerifyReport> {
    let &VerifyOptions {
        workers,
        slice,
        recheck,
        deep,
    } = options;

    if workers == 0 {
        bail!("--workers must be at least 1");
    }

    let all = discover_parts(router, layout)?;
    if all.is_empty() {
        return Ok(VerifyReport::default());
    }

    let mine = match slice {
        Some((index, total)) => take_slice(all, index, total)?,
        None => all,
    };

    fs_err::create_dir_all(layout.verified_dir())?;

    // Skipping is an optimization only: with `verified/` removed every part is re-read.
    let (pending, skipped): (Vec<_>, Vec<_>) = if recheck {
        (mine, Vec::new())
    } else {
        mine.into_iter()
            .partition(|part| !marker_matches(layout, part, deep))
    };

    let progress = crate::progress::Progress::new("scatter-verify", "part", pending.len() as u64);
    let points = AtomicU64::new(0);
    let checked = AtomicU64::new(0);
    let per_shard: Mutex<BTreeMap<ShardId, u64>> = Mutex::new(BTreeMap::new());
    let queue = Mutex::new(pending.into_iter());

    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(workers);

        for worker in 0..workers {
            let (queue, points, checked, per_shard, progress) =
                (&queue, &points, &checked, &per_shard, &progress);

            handles.push(
                std::thread::Builder::new()
                    .name(format!("verify-{worker}"))
                    .spawn_scoped(scope, move || -> Result<()> {
                        loop {
                            let next = queue.lock().expect("verify queue poisoned").next();
                            let Some(part) = next else { return Ok(()) };

                            let found = if deep {
                                verify_part_deep(config, router, &part)?
                            } else {
                                verify_part_metadata(config, &part)?
                            };

                            points.fetch_add(found, Ordering::Relaxed);
                            checked.fetch_add(1, Ordering::Relaxed);
                            *per_shard
                                .lock()
                                .expect("per-shard tally poisoned")
                                .entry(part.shard_id)
                                .or_default() += found;

                            // Marked only after the whole part read cleanly, so an interrupted
                            // run never records a part it did not finish.
                            write_verified_marker(layout, &part, deep)?;
                            progress.advance(found);
                        }
                    })
                    .context("cannot spawn verify worker")?,
            );
        }

        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("verify worker panicked"))??;
        }
        Ok(())
    })?;

    progress.finish();

    Ok(VerifyReport {
        parts: checked.load(Ordering::Relaxed),
        parts_skipped: skipped.len() as u64,
        points: points.load(Ordering::Relaxed),
        per_shard: per_shard
            .into_inner()
            .unwrap_or_else(|err| err.into_inner()),
    })
}

/// Every committed part file, in a deterministic order so `--slice` is stable.
fn discover_parts(router: &ShardRouter, layout: &ScatterLayout) -> Result<Vec<PartToVerify>> {
    let mut parts = Vec::new();

    for shard_id in 0..router.shard_count() as ShardId {
        let dir = layout.shard_dir(shard_id);
        let entries = match fs_err::read_dir(&dir) {
            Ok(entries) => entries,
            // A shard with no parts is legitimate: no input point routed there.
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err.into()),
        };

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            match path.extension().and_then(|ext| ext.to_str()) {
                Some("tmp") => bail!(
                    "{} is an uncommitted part file; re-run the scatter phase to sweep it",
                    path.display(),
                ),
                // Sidecars are metadata, not records.
                Some("meta") => continue,
                _ => {}
            }

            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            parts.push(PartToVerify {
                shard_id,
                len: entry.metadata()?.len(),
                name,
                path,
            });
        }
    }

    parts.sort();
    Ok(parts)
}

/// Check a part without reading its records.
///
/// Two reads: the header, which `PartReader::open` validates the part fingerprint of, and the
/// sidecar's recorded length against the file's current length. Returns the sidecar's record count.
fn verify_part_metadata(config: &LoadedConfig, part: &PartToVerify) -> Result<u64> {
    let reader = crate::partfile::PartReader::open(&part.path, &config.part_fingerprint)?;
    if reader.header().shard_id != part.shard_id {
        bail!(
            "{} claims shard {} but sits in shard {}'s directory",
            part.path.display(),
            reader.header().shard_id,
            part.shard_id,
        );
    }
    drop(reader);

    let meta = crate::partfile::read_meta(&part.path, &config.part_fingerprint)?;
    if meta.bytes != part.len {
        bail!(
            "{} is {} bytes but its sidecar recorded {} at commit; the part has been truncated or \
             rewritten. Re-run the scatter phase for this input file.",
            part.path.display(),
            part.len,
            meta.bytes,
        );
    }

    Ok(meta.records)
}

/// Read one part in full and check every record routes to the shard it sits in.
fn verify_part_deep(
    config: &LoadedConfig,
    router: &ShardRouter,
    part: &PartToVerify,
) -> Result<u64> {
    let mut reader = crate::partfile::PartReader::open(&part.path, &config.part_fingerprint)?;
    if reader.header().shard_id != part.shard_id {
        bail!(
            "{} claims shard {} but sits in shard {}'s directory",
            part.path.display(),
            reader.header().shard_id,
            part.shard_id,
        );
    }

    let mut points = 0;
    while let Some(record) = reader.next_record()? {
        let expected = router.shard_of(record.id)?;
        if expected != part.shard_id {
            bail!(
                "{}: point {:?} routes to shard {expected} but was scattered into shard {}",
                part.path.display(),
                record.id,
                part.shard_id,
            );
        }
        points += 1;
    }

    // The sidecar is what `plan` trusts to size segments, so a disagreement here would silently
    // mis-plan. Only a full read can find it.
    let meta = crate::partfile::read_meta(&part.path, &config.part_fingerprint)?;
    if meta.records != points {
        bail!(
            "{} holds {points} records but its sidecar claims {}; the plan phase would size \
             segments from the wrong count",
            part.path.display(),
            meta.records,
        );
    }

    Ok(points)
}

/// Whether a part has already been checked, at its current length and at least as thoroughly.
///
/// A part checked cheaply does not satisfy a `deep` run, so switching to `deep` re-reads it; a part
/// checked deeply satisfies a cheap run.
fn marker_matches(layout: &ScatterLayout, part: &PartToVerify, deep: bool) -> bool {
    let path = layout.verified_marker(part.shard_id, &part.name);
    let Ok(text) = fs_err::read_to_string(&path) else {
        return false;
    };

    let mut fields = text.split_whitespace();
    let recorded_len = fields.next().and_then(|len| len.parse::<u64>().ok());
    let was_deep = fields.next() == Some("deep");

    recorded_len == Some(part.len) && (!deep || was_deep)
}

fn write_verified_marker(layout: &ScatterLayout, part: &PartToVerify, deep: bool) -> Result<()> {
    let path = layout.verified_marker(part.shard_id, &part.name);
    let body = if deep {
        format!("{} deep", part.len)
    } else {
        part.len.to_string()
    };
    fs_err::write(&path, body).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

/// Discover inputs in the store, optionally taking only one format. Sorted (by the store's
/// listing contract) for reproducible logs and stable `--slice` striding.
///
/// A corpus directory commonly holds more than the data: `.jsonl` sidecars, manifests, notes.
/// Every extension this tool recognises is a *candidate*, so those get picked up and then fail on
/// read. Naming the format excludes them at discovery instead, which is both faster and clearer
/// than letting each one fail. Files of unrecognised extensions are ignored either way.
pub fn discover_inputs(
    store: &dyn InputStore,
    only: Option<source::InputFormat>,
) -> Result<Vec<InputFile>> {
    let mut files: Vec<InputFile> = store
        .list()?
        .into_iter()
        .filter(|object| {
            if source::detect(&object.path).is_ok() {
                true
            } else {
                log::debug!("ignoring non-input file {}", object.path.display());
                false
            }
        })
        .map(|object| InputFile {
            file_id: file_id_for(&object.path),
            path: object.path,
        })
        .collect();
    files.sort();

    if let Some(only) = only {
        let before = files.len();
        files.retain(|file| source::detect(&file.path).is_ok_and(|format| format == only));
        let excluded = before - files.len();
        if excluded > 0 {
            log::info!("ignoring {excluded} file(s) that are not {only:?}");
        }
    }

    if files.is_empty() {
        bail!(
            "no {}input files found",
            match only {
                Some(format) => format!("{format:?} "),
                None => String::new(),
            },
        );
    }

    Ok(files)
}

/// Take one process's share of the discovered inputs.
///
/// `discover_inputs` sorts, so the split is a pure function of the input set: every process sees
/// the same ordering and picks a disjoint stride from it. That is what lets ten nodes scatter the
/// same corpus concurrently without coordinating, and lets a failed node's slice be re-run on its
/// own without touching the others.
///
/// Striding rather than contiguous chunking, so an unequal file-size distribution does not land
/// all the large files on one node.
pub fn take_slice<T>(inputs: Vec<T>, index: usize, total: usize) -> Result<Vec<T>> {
    if total == 0 || index >= total {
        bail!("--slice must be I/N with N >= 1 and I < N; got {index}/{total}");
    }

    let mine: Vec<_> = inputs
        .into_iter()
        .enumerate()
        .filter(|(position, _)| position % total == index)
        .map(|(_, file)| file)
        .collect();

    if mine.is_empty() {
        bail!(
            "slice {index}/{total} covers no input files; there are fewer files than slices, so \
             use a smaller N",
        );
    }

    Ok(mine)
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::config;
    use crate::store::LocalStore;

    struct Fixture {
        _dir: TempDir,
        input_root: PathBuf,
        work_root: PathBuf,
        config: LoadedConfig,
        router: ShardRouter,
    }

    fn fixture(shard_number: u32, files: usize, points_per_file: u64) -> Fixture {
        let dir = TempDir::with_prefix("scatter").unwrap();
        let input_root = dir.path().join("input");
        let work_root = dir.path().join("work");
        fs_err::create_dir_all(&input_root).unwrap();

        let mut id = 0u64;
        for file in 0..files {
            let mut body = String::new();
            for _ in 0..points_per_file {
                body.push_str(&format!("{{\"id\": {id}, \"vector\": [0.1, 0.2]}}\n"));
                id += 1;
            }
            fs_err::write(input_root.join(format!("data-{file:04}.jsonl")), body).unwrap();
        }

        let mut value = config::tests::valid_config_json();
        value["params"]["shard_number"] = serde_json::json!(shard_number);
        let config = config::from_str(&value.to_string()).unwrap();
        let router = ShardRouter::new(&config).unwrap();

        Fixture {
            _dir: dir,
            input_root,
            work_root,
            config,
            router,
        }
    }

    impl Fixture {
        fn layout(&self) -> ScatterLayout {
            ScatterLayout::new(&self.work_root)
        }

        fn store(&self) -> LocalStore {
            LocalStore::new(&self.input_root)
        }

        fn inputs(&self) -> Vec<InputFile> {
            discover_inputs(&self.store(), None).unwrap()
        }

        fn run(&self, workers: usize) -> Result<ScatterStats> {
            self.run_with(workers, usize::MAX)
        }

        /// `max_failures` exposed so a test can assert both the tolerant and the abort path.
        fn run_with(&self, workers: usize, max_failures: usize) -> Result<ScatterStats> {
            run(
                &self.config,
                &self.router,
                &self.store(),
                &self.inputs(),
                &self.layout(),
                &ScatterOptions {
                    workers,
                    mapping: None,
                    slice: None,
                    max_failures,
                },
            )
        }

        /// Shard id -> sorted point ids landed in it.
        ///
        /// Path-independent, unlike [`Self::part_bytes`]: part filenames and the header's
        /// `source_path` both derive from the store-relative input path, so two fixtures
        /// scattering identical file names compare equal through this — and through
        /// `part_bytes` too, now that paths are store-relative rather than absolute.
        fn shard_contents(&self) -> BTreeMap<ShardId, Vec<segment::types::PointIdType>> {
            let mut out: BTreeMap<ShardId, Vec<_>> = BTreeMap::new();

            for key in self.part_bytes().into_keys() {
                let shard_id: ShardId = key
                    .split('/')
                    .next()
                    .unwrap()
                    .trim_start_matches("shard_")
                    .parse()
                    .unwrap();

                let path = self.work_root.join(&key);
                let mut reader =
                    crate::partfile::PartReader::open(&path, &self.config.part_fingerprint)
                        .unwrap();
                let entry = out.entry(shard_id).or_default();
                while let Some(record) = reader.next_record().unwrap() {
                    entry.push(record.id);
                }
            }

            for ids in out.values_mut() {
                ids.sort();
            }
            out
        }

        /// Every part file's bytes, keyed by path relative to the work root.
        fn part_bytes(&self) -> BTreeMap<String, Vec<u8>> {
            let mut out = BTreeMap::new();
            let Ok(entries) = fs_err::read_dir(&self.work_root) else {
                return out;
            };
            for entry in entries {
                let entry = entry.unwrap();
                if !entry.path().is_dir()
                    || !entry.file_name().to_string_lossy().starts_with("shard_")
                {
                    continue;
                }
                for part in fs_err::read_dir(entry.path()).unwrap() {
                    let part = part.unwrap();
                    // Sidecars are metadata, not part content.
                    if part.path().extension().and_then(|e| e.to_str()) == Some("meta") {
                        continue;
                    }
                    let key = format!(
                        "{}/{}",
                        entry.file_name().to_string_lossy(),
                        part.file_name().to_string_lossy(),
                    );
                    out.insert(key, fs_err::read(part.path()).unwrap());
                }
            }
            out
        }
    }

    #[test]
    fn scatters_every_point_exactly_once() {
        let fixture = fixture(8, 4, 250);
        let stats = fixture.run(2).unwrap();

        assert_eq!(stats.files_processed, 4);
        assert_eq!(stats.files_skipped, 0);
        assert_eq!(stats.points, 1_000);

        // Read everything back and confirm the full id set survived, with no duplicates.
        let mut ids = Vec::new();
        for (key, _) in fixture.part_bytes() {
            let path = fixture.work_root.join(&key);
            let mut reader =
                crate::partfile::PartReader::open(&path, &fixture.config.part_fingerprint).unwrap();
            while let Some(record) = reader.next_record().unwrap() {
                ids.push(record.id);
            }
        }

        assert_eq!(ids.len(), 1_000, "no points lost or duplicated");
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 1_000);
    }

    /// `verify` must be resumable, parallel and slice-able — it is a second full pass over the
    /// intermediate, which at production size is tens of terabytes.
    #[test]
    fn verify_is_resumable_and_reports_what_it_skipped() {
        let fixture = fixture(4, 3, 50);
        fixture.run(1).unwrap();

        let options = |recheck| VerifyOptions {
            workers: 4,
            slice: None,
            recheck,
            deep: true,
        };

        let first = verify(
            &fixture.config,
            &fixture.router,
            &fixture.layout(),
            &options(false),
        )
        .unwrap();
        assert_eq!(first.points, 150, "every point checked on the first pass");
        assert!(first.parts > 0);
        assert_eq!(first.parts_skipped, 0);

        // Second pass: everything already verified, so nothing is re-read.
        let second = verify(
            &fixture.config,
            &fixture.router,
            &fixture.layout(),
            &options(false),
        )
        .unwrap();
        assert_eq!(second.parts, 0, "no part re-read");
        assert_eq!(second.parts_skipped, first.parts, "all of them skipped");
        assert_eq!(second.points, 0);

        // --recheck ignores the markers.
        let forced = verify(
            &fixture.config,
            &fixture.router,
            &fixture.layout(),
            &options(true),
        )
        .unwrap();
        assert_eq!(forced.points, 150, "--recheck re-reads everything");
        assert_eq!(forced.parts_skipped, 0);
    }

    /// The default pass catches a truncated part from its recorded length, without reading records.
    #[test]
    fn metadata_verify_catches_truncation() {
        let fixture = fixture(4, 2, 50);
        fixture.run(1).unwrap();

        let cheap = VerifyOptions {
            workers: 2,
            slice: None,
            recheck: true,
            deep: false,
        };

        let clean = verify(&fixture.config, &fixture.router, &fixture.layout(), &cheap).unwrap();
        assert_eq!(
            clean.points, 100,
            "the point total comes from the sidecars, so it is still exact",
        );

        // Truncate a part. Its sidecar still records the length it had at commit.
        let victim = fixture
            .work_root
            .join(fixture.part_bytes().keys().next().unwrap());
        let bytes = fs_err::read(&victim).unwrap();
        fs_err::write(&victim, &bytes[..bytes.len() / 2]).unwrap();

        let err = verify(&fixture.config, &fixture.router, &fixture.layout(), &cheap).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("truncated or rewritten"), "{text}");
    }

    /// A part in the wrong shard's directory is caught from its header alone.
    #[test]
    fn metadata_verify_catches_a_misplaced_part() {
        let fixture = fixture(4, 2, 50);
        fixture.run(1).unwrap();

        // Move a part from shard 0 into shard 1's directory, sidecar and all.
        let from = fixture.work_root.join("shard_0");
        let to = fixture.work_root.join("shard_1");
        let part = fs_err::read_dir(&from)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.extension().is_none())
            .expect("a part in shard 0");
        let name = part.file_name().unwrap().to_owned();
        fs_err::rename(&part, to.join(&name)).unwrap();
        fs_err::rename(
            crate::partfile::meta_path(&part),
            crate::partfile::meta_path(&to.join(&name)),
        )
        .unwrap();

        let err = verify(
            &fixture.config,
            &fixture.router,
            &fixture.layout(),
            &VerifyOptions {
                workers: 2,
                slice: None,
                recheck: true,
                deep: false,
            },
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("claims shard 0"), "{err:#}");
    }

    /// Switching to `deep` must not trust a part that was only checked cheaply.
    #[test]
    fn a_cheaply_verified_part_does_not_satisfy_a_deep_run() {
        let fixture = fixture(4, 2, 50);
        fixture.run(1).unwrap();
        let layout = fixture.layout();

        let cheap = verify(
            &fixture.config,
            &fixture.router,
            &layout,
            &VerifyOptions {
                workers: 2,
                slice: None,
                recheck: false,
                deep: false,
            },
        )
        .unwrap();
        assert!(cheap.parts > 0);

        let deep = verify(
            &fixture.config,
            &fixture.router,
            &layout,
            &VerifyOptions {
                workers: 2,
                slice: None,
                recheck: false,
                deep: true,
            },
        )
        .unwrap();
        assert_eq!(
            deep.parts, cheap.parts,
            "deep re-reads what was only checked cheaply"
        );
        assert_eq!(deep.parts_skipped, 0);

        // And the reverse: a deeply-checked part satisfies a later cheap run.
        let after = verify(
            &fixture.config,
            &fixture.router,
            &layout,
            &VerifyOptions {
                workers: 2,
                slice: None,
                recheck: false,
                deep: false,
            },
        )
        .unwrap();
        assert_eq!(after.parts, 0);
        assert_eq!(after.parts_skipped, deep.parts);
    }

    /// A part rewritten since it was verified must be checked again, not trusted.
    #[test]
    fn verify_rechecks_a_part_whose_length_changed() {
        let fixture = fixture(4, 2, 50);
        fixture.run(1).unwrap();

        let options = VerifyOptions {
            workers: 2,
            slice: None,
            recheck: false,
            deep: true,
        };
        let first = verify(
            &fixture.config,
            &fixture.router,
            &fixture.layout(),
            &options,
        )
        .unwrap();
        assert!(first.parts > 0);

        // Truncate one part: same name, different length.
        let victim = fixture
            .work_root
            .join(fixture.part_bytes().keys().next().unwrap());
        let bytes = fs_err::read(&victim).unwrap();
        fs_err::write(&victim, &bytes[..bytes.len() / 2]).unwrap();

        // It is re-read, and being truncated it now fails — which is the point: a changed part is
        // not silently trusted because an earlier run passed.
        let result = verify(
            &fixture.config,
            &fixture.router,
            &fixture.layout(),
            &options,
        );
        assert!(
            result.is_err(),
            "a truncated part must be caught on re-verify"
        );
    }

    /// Slices partition the parts, so several machines can verify one work directory.
    #[test]
    fn verify_slices_cover_every_part_exactly_once() {
        let fixture = fixture(4, 4, 50);
        fixture.run(1).unwrap();

        let mut parts = 0;
        let mut points = 0;
        for index in 0..3 {
            let report = verify(
                &fixture.config,
                &fixture.router,
                &fixture.layout(),
                &VerifyOptions {
                    workers: 2,
                    slice: Some((index, 3)),
                    recheck: false,
                    deep: true,
                },
            )
            .unwrap();
            parts += report.parts;
            points += report.points;
        }

        assert_eq!(points, 200, "the slices together check every point once");

        // And a full pass now finds nothing left to do.
        let after = verify(
            &fixture.config,
            &fixture.router,
            &fixture.layout(),
            &VerifyOptions {
                workers: 2,
                slice: None,
                recheck: false,
                deep: true,
            },
        )
        .unwrap();
        assert_eq!(after.parts, 0, "slices between them verified everything");
        assert_eq!(after.parts_skipped, parts);
    }

    /// Worker count must not change the result.
    #[test]
    fn verify_worker_count_does_not_change_the_result() {
        let fixture = fixture(4, 4, 50);
        fixture.run(1).unwrap();

        let one = verify(
            &fixture.config,
            &fixture.router,
            &fixture.layout(),
            &VerifyOptions {
                workers: 1,
                slice: None,
                recheck: true,
                deep: true,
            },
        )
        .unwrap();
        let many = verify(
            &fixture.config,
            &fixture.router,
            &fixture.layout(),
            &VerifyOptions {
                workers: 8,
                slice: None,
                recheck: true,
                deep: true,
            },
        )
        .unwrap();

        assert_eq!(one.points, many.points);
        assert_eq!(one.parts, many.parts);
        assert_eq!(one.per_shard, many.per_shard);
    }

    #[test]
    fn every_point_lands_in_the_shard_the_ring_chose() {
        let fixture = fixture(8, 2, 200);
        fixture.run(1).unwrap();

        for (key, _) in fixture.part_bytes() {
            let shard_id: ShardId = key
                .split('/')
                .next()
                .unwrap()
                .trim_start_matches("shard_")
                .parse()
                .unwrap();

            let path = fixture.work_root.join(&key);
            let mut reader =
                crate::partfile::PartReader::open(&path, &fixture.config.part_fingerprint).unwrap();
            assert_eq!(reader.header().shard_id, shard_id);

            while let Some(record) = reader.next_record().unwrap() {
                assert_eq!(
                    fixture.router.shard_of(record.id).unwrap(),
                    shard_id,
                    "point {:?} is in the wrong shard directory",
                    record.id,
                );
            }
        }
    }

    /// Re-running a completed scatter must do nothing and change nothing.
    #[test]
    fn rerun_is_a_no_op() {
        let fixture = fixture(4, 3, 100);
        fixture.run(2).unwrap();
        let first = fixture.part_bytes();

        let stats = fixture.run(2).unwrap();
        assert_eq!(stats.files_processed, 0, "nothing should be reprocessed");
        assert_eq!(stats.files_skipped, 3);
        assert_eq!(stats.points, 0);

        assert_eq!(first, fixture.part_bytes(), "output must be unchanged");
    }

    /// Deleting all completion markers must produce byte-identical output.
    ///
    /// This is the load-bearing resumability property: markers are an optimization, and
    /// correctness comes from output names being a pure function of (config, input file).
    #[test]
    fn markers_are_only_an_optimization() {
        let fixture = fixture(4, 3, 100);
        fixture.run(2).unwrap();
        let with_markers = fixture.part_bytes();

        fs_err::remove_dir_all(fixture.layout().done_dir()).unwrap();

        let stats = fixture.run(2).unwrap();
        assert_eq!(stats.files_processed, 3, "all files reprocessed");
        assert_eq!(stats.points, 300);

        assert_eq!(
            with_markers,
            fixture.part_bytes(),
            "a full re-scatter must be byte-identical, not merely equivalent",
        );
    }

    /// A resumed run must be byte-identical to scattering the same inputs in one pass.
    ///
    /// Both runs happen in the same work directory so the comparison can be byte-level: part
    /// names and the header's `source_path` derive from the store-relative input paths, which
    /// are fixed here.
    #[test]
    fn resume_after_partial_run_matches_a_clean_run() {
        let fixture = fixture(4, 4, 100);
        let inputs = fixture.inputs();

        // Interrupted run: only the first two files get scattered.
        let stats = run(
            &fixture.config,
            &fixture.router,
            &fixture.store(),
            &inputs[..2],
            &fixture.layout(),
            &ScatterOptions {
                workers: 1,
                mapping: None,
                slice: None,
                max_failures: usize::MAX,
            },
        )
        .unwrap();
        assert_eq!(stats.files_processed, 2);

        // Resume with the full input set.
        let stats = fixture.run(1).unwrap();
        assert_eq!(stats.files_skipped, 2, "the first two must be skipped");
        assert_eq!(stats.files_processed, 2);
        let resumed = fixture.part_bytes();

        // Now force a complete re-scatter in the same directory and compare.
        fs_err::remove_dir_all(fixture.layout().done_dir()).unwrap();
        let stats = fixture.run(1).unwrap();
        assert_eq!(stats.files_processed, 4);

        assert_eq!(
            resumed,
            fixture.part_bytes(),
            "resumed output must be byte-identical to a single-pass scatter",
        );
    }

    /// A crash leaves `.tmp` files for this run's own inputs; those are swept and not consumed.
    #[test]
    fn sweeps_temp_files_left_by_its_own_interrupted_run() {
        let fixture = fixture(4, 2, 50);
        fixture.run(1).unwrap();

        // Simulate a crash mid-write: a process-unique `.tmp` for an input this run owns
        // (`part_{file_id}.{pid}.{n}.tmp`, as a real interrupted run would leave).
        let owned = &fixture.inputs()[0].file_id;
        let debris = fixture
            .work_root
            .join("shard_0")
            .join(format!("part_{owned}.99999.0.tmp"));
        fs_err::create_dir_all(debris.parent().unwrap()).unwrap();
        fs_err::write(&debris, b"partial garbage").unwrap();

        let stats = fixture.run(1).unwrap();
        assert_eq!(stats.temp_files_swept, 1);
        assert!(!debris.exists(), "debris must be removed");
    }

    /// A `.tmp` for an input this run does *not* own belongs to another process — leave it.
    ///
    /// This is what makes `--slice` safe to run concurrently into one work directory. Sweeping
    /// every `.tmp` would unlink a part another node was mid-write on: it holds the descriptor, so
    /// it would go on writing to an unlinked inode and then fail its rename.
    #[test]
    fn leaves_temp_files_belonging_to_another_process() {
        let fixture = fixture(4, 2, 50);

        let foreign = fixture.work_root.join("shard_0").join("part_deadbeef.tmp");
        fs_err::create_dir_all(foreign.parent().unwrap()).unwrap();
        fs_err::write(&foreign, b"another node is writing this").unwrap();

        let stats = fixture.run(1).unwrap();
        assert_eq!(
            stats.temp_files_swept, 0,
            "must not touch a foreign temp file"
        );
        assert!(
            foreign.exists(),
            "another process's in-flight part must survive"
        );
    }

    /// Slices are disjoint, cover everything, and are stable.
    #[test]
    fn slices_partition_the_input_exactly_once() {
        let fixture = fixture(4, 9, 10);
        let all = fixture.inputs();

        let mut seen = Vec::new();
        for index in 0..3 {
            let mine = take_slice(all.clone(), index, 3).unwrap();
            assert_eq!(
                mine,
                take_slice(all.clone(), index, 3).unwrap(),
                "a slice must be deterministic",
            );
            seen.extend(mine);
        }

        seen.sort();
        assert_eq!(
            seen, all,
            "the slices must reassemble the whole input exactly"
        );
    }

    #[test]
    fn rejects_a_slice_index_outside_the_total() {
        let fixture = fixture(4, 2, 10);
        let err = take_slice(fixture.inputs(), 3, 3).unwrap_err();
        assert!(format!("{err:#}").contains("--slice must be"), "{err:#}");
    }

    /// More slices than files must be rejected, not silently produce an empty run.
    #[test]
    fn rejects_a_slice_that_covers_no_files() {
        let fixture = fixture(4, 2, 10);
        let err = take_slice(fixture.inputs(), 5, 8).unwrap_err();
        assert!(
            format!("{err:#}").contains("covers no input files"),
            "{err:#}"
        );
    }

    /// Resuming with a different config must be refused, not silently mixed.
    #[test]
    fn refuses_to_resume_under_a_different_config() {
        let fixture = fixture(4, 2, 50);
        fixture.run(1).unwrap();

        // Same work directory, different shard count => different ring, different routing.
        let mut value = config::tests::valid_config_json();
        value["params"]["shard_number"] = serde_json::json!(8);
        let other_config = config::from_str(&value.to_string()).unwrap();
        let other_router = ShardRouter::new(&other_config).unwrap();

        let err = run(
            &other_config,
            &other_router,
            &fixture.store(),
            &fixture.inputs(),
            &fixture.layout(),
            &ScatterOptions {
                workers: 1,
                mapping: None,
                slice: None,
                max_failures: usize::MAX,
            },
        )
        .unwrap_err();

        let message = format!("{err:#}");
        assert!(
            message.contains("incompatible collection config"),
            "{message}"
        );
        assert!(message.contains("hash rings"), "{message}");
    }

    /// The fork's per-collection ring scale is routing, so it must also gate a resume.
    #[test]
    fn refuses_to_resume_under_a_different_ring_scale() {
        let fixture = fixture(4, 2, 50);
        fixture.run(1).unwrap();

        let mut value = config::tests::valid_config_json();
        value["params"]["hash_ring_shard_scale"] = serde_json::json!(500);
        let rescaled = config::from_str(&value.to_string()).unwrap();
        let router = ShardRouter::new(&rescaled).unwrap();

        let err = run(
            &rescaled,
            &router,
            &fixture.store(),
            &fixture.inputs(),
            &fixture.layout(),
            &ScatterOptions {
                workers: 1,
                mapping: None,
                slice: None,
                max_failures: usize::MAX,
            },
        )
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("incompatible collection config"),
            "{err:#}"
        );
    }

    /// The capability the two-fingerprint split exists to provide.
    ///
    /// Retuning the index must leave a finished scatter reusable. If this ever regresses, changing
    /// `ef_construct` means re-reading and re-writing tens of terabytes, which is the entire
    /// reason the part fingerprint is narrower than the config one.
    #[test]
    fn resumes_after_an_index_config_change() {
        let fixture = fixture(4, 2, 50);
        let first = fixture.run(1).unwrap();
        assert_eq!(first.files_processed, 2);

        // Exactly the edit an operator makes to cut HNSW build time: a different index shape,
        // same corpus, same ring.
        let mut value = config::tests::valid_config_json();
        value["hnsw_config"]["m"] = serde_json::json!(48);
        value["hnsw_config"]["ef_construct"] = serde_json::json!(256);
        value["hnsw_config"]["payload_m"] = serde_json::json!(0);
        value["optimizer_config"]["max_segment_size"] = serde_json::json!(9_000_000);
        let retuned = config::from_str(&value.to_string()).unwrap();
        let router = ShardRouter::new(&retuned).unwrap();

        assert_ne!(
            retuned.fingerprint, fixture.config.fingerprint,
            "the perturbation must move the full fingerprint, or this proves nothing",
        );
        assert_eq!(
            retuned.part_fingerprint, fixture.config.part_fingerprint,
            "an index-only change must not move the part fingerprint",
        );

        // Resuming is accepted, and every input is skipped as already done rather than redone.
        let second = run(
            &retuned,
            &router,
            &fixture.store(),
            &fixture.inputs(),
            &fixture.layout(),
            &ScatterOptions {
                workers: 1,
                mapping: None,
                slice: None,
                max_failures: usize::MAX,
            },
        )
        .expect("an index-only config change must not invalidate the scatter");

        assert_eq!(
            second.files_skipped, 2,
            "finished work must not be repeated"
        );
        assert_eq!(second.files_processed, 0);

        // And the parts are still readable under the retuned config.
        let path = fixture.layout().part_path(0, &fixture.inputs()[0].file_id);
        if path.exists() {
            crate::partfile::PartReader::open(&path, &retuned.part_fingerprint)
                .expect("parts must still open under the retuned config");
        }
    }

    /// Worker count and scheduling must not affect which shard a point lands in.
    ///
    /// Compared via [`Fixture::shard_contents`] rather than raw bytes out of habit from the
    /// reference; with store-relative paths the two fixtures would compare byte-equal too.
    #[test]
    fn worker_count_does_not_change_output() {
        let one = fixture(8, 4, 100);
        one.run(1).unwrap();

        let many = fixture(8, 4, 100);
        many.run(4).unwrap();

        assert_eq!(
            one.shard_contents(),
            many.shard_contents(),
            "shard placement must not depend on worker count or scheduling",
        );

        // Guard against the comparison being vacuous.
        let total: usize = one.shard_contents().values().map(Vec::len).sum();
        assert_eq!(total, 400);
        assert!(
            one.shard_contents().len() > 1,
            "must exercise several shards"
        );
    }

    #[test]
    fn rejects_a_workers_times_shards_product_that_would_exhaust_descriptors() {
        let fixture = fixture(64, 1, 10);
        let err = run(
            &fixture.config,
            &fixture.router,
            &fixture.store(),
            &fixture.inputs(),
            &fixture.layout(),
            &ScatterOptions {
                workers: 64, // 64 * 64 = 4096 open files
                mapping: None,
                slice: None,
                max_failures: usize::MAX,
            },
        )
        .unwrap_err();

        let message = format!("{err:#}");
        assert!(
            message.contains("concurrently open part files"),
            "{message}"
        );
        assert!(message.contains("Lower --workers"), "{message}");
    }

    #[test]
    fn file_ids_are_stable_and_path_derived() {
        let a = file_id_for(Path::new("input/data-0001.jsonl"));
        let b = file_id_for(Path::new("input/data-0001.jsonl"));
        let c = file_id_for(Path::new("input/data-0002.jsonl"));

        assert_eq!(a, b, "same path must give the same id");
        assert_ne!(a, c, "different paths must give different ids");
        assert_eq!(a.len(), 16);
    }

    /// Two mounts of the same corpus agree on file ids, because the id hashes the
    /// store-relative path. Under the reference's absolute-path hashing these differed,
    /// which is why its docs warned against per-node symlink trees.
    #[test]
    fn file_ids_are_root_independent() {
        let a = fixture(4, 2, 10);
        let b = fixture(4, 2, 10);
        assert_eq!(
            a.inputs(),
            b.inputs(),
            "identical corpora under different roots must discover identically",
        );
    }

    /// `--input-format` excludes other formats at discovery, before anything is opened.
    ///
    /// This is the right answer when the corpus directory legitimately holds other files: they are
    /// never read, so they cannot fail.
    #[test]
    fn input_format_restricts_discovery() {
        let fixture = fixture(4, 2, 10);
        fs_err::write(fixture.input_root.join("notes.jsonl"), "garbage\n").unwrap();

        let all = discover_inputs(&fixture.store(), None).unwrap();
        let jsonl_only =
            discover_inputs(&fixture.store(), Some(source::InputFormat::Jsonl)).unwrap();
        assert_eq!(all.len(), jsonl_only.len(), "the fixture writes jsonl");

        // Asking for Parquet in a directory of jsonl must say so rather than scatter nothing.
        let err =
            discover_inputs(&fixture.store(), Some(source::InputFormat::Parquet)).unwrap_err();
        assert!(format!("{err:#}").contains("Parquet"), "{err:#}");
    }

    /// A file that cannot be read is skipped and reported, not fatal.
    ///
    /// One unreadable file among thousands must not discard the work already finished — that was
    /// the original behaviour and it is wrong at corpus scale. The file gets no `done` marker, so a
    /// later run retries it, and its points are absent from the output until it does.
    #[test]
    fn a_malformed_input_file_is_skipped_and_reported() {
        let fixture = fixture(4, 3, 10);
        fs_err::write(
            fixture.input_root.join("bad.jsonl"),
            "{\"id\": 1, \"vector\": \"nope\"}\n",
        )
        .unwrap();

        let stats = fixture.run(1).unwrap();

        assert_eq!(stats.files_failed.len(), 1, "the bad file is reported");
        assert!(
            stats.files_failed[0].path.contains("bad.jsonl"),
            "{:?}",
            stats.files_failed[0],
        );
        assert_eq!(
            stats.files_processed, 3,
            "the good files still went through"
        );
        assert_eq!(stats.points, 30, "and their points are all present");

        // The id hashes the store-relative path.
        assert!(
            !fixture
                .layout()
                .done_marker(&file_id_for(Path::new("bad.jsonl")))
                .exists(),
            "a failed file must not be marked done, or a re-run would skip it",
        );

        // Recorded on disk so the operator can inspect or re-feed them.
        let report = fixture.work_root.join("failed.jsonl");
        assert!(report.exists(), "failures must be written to {report:?}");
        assert!(
            fs_err::read_to_string(&report)
                .unwrap()
                .contains("bad.jsonl")
        );
    }

    /// Past `max_failures` it does abort: that shape means the mapping is wrong, not the files.
    #[test]
    fn too_many_failures_aborts_rather_than_reading_the_whole_corpus() {
        let fixture = fixture(4, 2, 10);
        for name in ["bad1.jsonl", "bad2.jsonl", "bad3.jsonl"] {
            fs_err::write(fixture.input_root.join(name), "not json at all\n").unwrap();
        }

        let err = fixture.run_with(1, 1).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("--max-failures"), "{text}");
        assert!(
            text.contains("mapping"),
            "should point at the likely cause: {text}"
        );
    }

    /// A resumed run retries only the file that failed.
    #[test]
    fn a_failed_file_is_retried_on_the_next_run() {
        let fixture = fixture(4, 3, 10);
        let bad = fixture.input_root.join("bad.jsonl");
        fs_err::write(&bad, "garbage\n").unwrap();

        let first = fixture.run(1).unwrap();
        assert_eq!(first.files_failed.len(), 1);

        // Repair it: one valid point, matching what the fixture writes.
        fs_err::write(
            &bad,
            "{\"id\": 999999, \"vector\": {\"dense\": [0.5, 0.5, 0.5, 0.5]}}\n",
        )
        .unwrap();

        let second = fixture.run(1).unwrap();
        assert!(second.files_failed.is_empty(), "{:?}", second.files_failed);
        assert_eq!(
            second.files_processed, 1,
            "only the repaired file is re-read"
        );
        assert_eq!(second.files_skipped, 3, "the rest are already done");
        assert_eq!(second.points, 1);
    }

    #[test]
    fn discover_inputs_ignores_unreadable_extensions_and_errors_when_empty() {
        let dir = TempDir::with_prefix("discover").unwrap();
        fs_err::write(dir.path().join("notes.txt"), "ignore me").unwrap();

        let store = LocalStore::new(dir.path());
        let err = discover_inputs(&store, None).unwrap_err();
        assert!(
            format!("{err:#}").contains("no input files found"),
            "{err:#}"
        );

        fs_err::write(dir.path().join("data.jsonl"), "").unwrap();
        let files = discover_inputs(&store, None).unwrap();
        assert_eq!(files.len(), 1, "only the .jsonl file should be picked up");
    }
}
