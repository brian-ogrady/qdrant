//! On-disk format for scatter output.
//!
//! Phase 1 writes one part file per (input file, shard) pair. Phase 2 reads them back and
//! feeds them into segment builds. The format is deliberately dull: a magic number, a JSON
//! header, then length-prefixed CBOR records.
//!
//! Two format decisions worth recording:
//!
//! * **CBOR, not bincode.** [`PointStructPersisted::vector`] is `VectorStructPersisted`,
//!   which is `#[serde(untagged)]` (`lib/shard/src/operations/point_ops.rs:490`), and
//!   `Payload` wraps `serde_json::Value`. Both need `deserialize_any`, which bincode does not
//!   provide. Qdrant's own WAL uses CBOR for exactly this reason
//!   (`lib/shard/src/wal.rs:42`).
//!
//! * **A part describes itself, and the build asks for a subset of it.** Two mechanisms, and the
//!   split between them is what decides whether a config change costs a re-scatter.
//!
//!   [`PartHeader::part_fingerprint`] covers **routing only** — the ring that chose the shard and
//!   the ids fed into it. Those make every record in the file wrong rather than merely a superset,
//!   so they demand equality. A mismatch is refused outright.
//!
//!   [`PartManifest`] records everything else the part carries: which dense and sparse vectors, at
//!   what element width and dimensionality, from which source column, and which payload columns
//!   were captured. [`PartProjection::resolve`] checks a build's requirements against it as a
//!   *subset* and drops the surplus. That is how a vector or a payload column gets removed, and how
//!   `hnsw_config`, `quantization_config` and `max_segment_size` get retuned, without re-reading
//!   tens of terabytes. Asking for something the manifest does not list is refused by name — a
//!   projection can drop, never invent.
//!
//!   Decoding reads the manifest, never the current config. That is the load-bearing part: it is
//!   why narrowing the fingerprint is safe, since a part's bytes are interpreted by the part's own
//!   record of how they were written.

use std::io::{BufReader, BufWriter, Read as _, Write as _};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use collection::shards::shard::ShardId;
use fs_err::File;
use serde::{Deserialize, Serialize};
use shard::operations::point_ops::{PointStructPersisted, VectorPersisted, VectorStructPersisted};

use crate::dense_codec::{DenseBytes, DenseEncoding, DenseEncodings};

/// Magic prefix. The trailing byte is the container version; bump it on any framing change.
// \x02: dense vectors moved from CBOR f32 arrays to packed byte strings at the
// config-declared element width. Bumped so \x01 parts cannot be silently misread.
// \x03: the header's fingerprint narrowed from the full config surface to just the surface a
// part's contents depend on (`config::part_fingerprint`). Same field width, different meaning,
// so the version has to move — otherwise a \x02 part would be compared against a hash it was
// never written with and report "scattered under a different config", which is misleading.
// \x04: `dense_encodings` became `manifest`, recording vectors, dimensionality, source columns
// and payload columns, and the fingerprint narrowed again to routing alone. Older parts record
// neither, so their composition cannot be subset-checked and their fingerprint covers a wider
// surface than the current one computes.
const MAGIC: &[u8; 8] = b"QSBPART\x04";

/// Largest record we will read. Guards against a corrupt length prefix causing a huge
/// allocation; a single point far larger than this is a bug, not a legitimate input.
const MAX_RECORD_BYTES: u32 = 256 * 1024 * 1024;

/// How one dense vector in this part was written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseSpec {
    /// Element width its bytes are packed at. This is what [`PartRecord`] is decoded with — the
    /// part's own record of how it was written, never the current config's opinion.
    pub encoding: DenseEncoding,
    /// Element count per vector, so a dimensionality mismatch is caught at open rather than on
    /// the first record.
    pub dim: usize,
    /// Source column, when the input format has named columns. `None` for JSONL input.
    ///
    /// Checked at build time so remapping a vector name to a different column cannot silently
    /// reuse the old column's values under the new mapping.
    #[serde(default)]
    pub source: Option<String>,
}

/// How one sparse vector in this part was written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseSpec {
    /// Source column, when the input format has named columns. `None` for JSONL input.
    #[serde(default)]
    pub source: Option<String>,
}

/// Exactly what a part carries, recorded so it can be interpreted without the config.
///
/// This is what lets the config change between `scatter` and `build`. A build asks for a *subset*
/// of the manifest rather than requiring the config to still match: drop a vector, drop a payload
/// column, retune anything, and the existing scatter stays usable. Asking for something the
/// manifest does not list is refused by name — the one thing a projection cannot do is invent data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartManifest {
    /// Dense vectors present, by Qdrant vector name.
    pub dense: std::collections::BTreeMap<String, DenseSpec>,
    /// Sparse vectors present, by Qdrant vector name.
    #[serde(default)]
    pub sparse: std::collections::BTreeMap<String, SparseSpec>,
    /// Payload keys captured. `None` when the input format does not declare a column set, which
    /// is JSONL — there the payload is whatever each line contained and cannot be enumerated up
    /// front, so no subset check is possible and nothing is projected away by default.
    #[serde(default)]
    pub payload: Option<Vec<String>>,
}

/// Metadata at the head of every part file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PartHeader {
    /// Fingerprint of the *part-relevant* config surface this part was scattered under.
    ///
    /// Narrow on purpose — routing only. See [`crate::config::part_fingerprint`]. Everything else
    /// that used to live here is now in [`Self::manifest`] and checked as a subset instead, which
    /// is what makes a scatter reusable across a config change.
    pub part_fingerprint: String,
    /// Stable id of the input file these records came from.
    pub file_id: String,
    /// Input path as given, for debugging. Not load-bearing.
    pub source_path: String,
    /// Shard these records route to.
    pub shard_id: ShardId,
    /// What this part contains, and how.
    pub manifest: PartManifest,
}

/// What a build wants out of a part, derived from the collection config and the input mapping.
///
/// Built once per run and checked against every part's [`PartManifest`].
#[derive(Debug, Clone, Default)]
pub struct WantedFields {
    /// Dense vector name -> the dimensionality the collection config declares.
    pub dense: std::collections::BTreeMap<String, usize>,
    /// Sparse vector names the collection config declares.
    pub sparse: std::collections::BTreeSet<String>,
    /// Payload keys to keep. `None` keeps whatever the part carries.
    pub payload: Option<std::collections::BTreeSet<String>>,
    /// Expected source column per dense vector name, when the mapping names one.
    ///
    /// Kept separate from [`Self::sparse_sources`] rather than merged into one map: dense and
    /// sparse names live in different namespaces and a collection may legitimately use the same
    /// name for both, in which case one entry would silently overwrite the other and the
    /// provenance check would compare a vector against the wrong column.
    pub dense_sources: std::collections::BTreeMap<String, String>,
    /// Expected source column per sparse vector name.
    pub sparse_sources: std::collections::BTreeMap<String, String>,
}

/// A resolved plan for turning one part's records into points.
///
/// Produced by [`Self::resolve`], which is where every compatibility check happens. Once a
/// projection exists, converting records cannot fail on a config mismatch.
#[derive(Debug, Clone)]
pub struct PartProjection {
    /// Dense vectors to keep, with the encoding to decode each one at.
    dense: std::collections::BTreeMap<String, DenseEncoding>,
    /// Sparse vectors to keep.
    sparse: std::collections::BTreeSet<String>,
    /// Payload keys to keep, or `None` to keep all.
    payload: Option<std::collections::BTreeSet<String>>,
    /// Counts for reporting, so a downselect is visible in a log rather than inferred.
    pub dropped_dense: Vec<String>,
    pub dropped_sparse: Vec<String>,
    pub dropped_payload: Vec<String>,
}

impl PartProjection {
    /// Check that `part` can supply `wanted`, and work out what to drop.
    ///
    /// Every error here names the specific field at fault, because the alternative — a segment
    /// quietly missing a vector or a payload key — is invisible until something queries it.
    pub fn resolve(part: &PartHeader, wanted: &WantedFields, path: &Path) -> Result<Self> {
        let manifest = &part.manifest;
        let mut dense = std::collections::BTreeMap::new();

        for (name, want_dim) in &wanted.dense {
            let spec = manifest.dense.get(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "{} does not carry a dense vector named '{name}'.\n\n\
                     It carries: {:?}. A build can drop vectors the scatter captured, but it \
                     cannot invent one — re-scatter with '{name}' in the mapping's \
                     `dense_vectors`.",
                    path.display(),
                    manifest.dense.keys().collect::<Vec<_>>(),
                )
            })?;

            if spec.dim != *want_dim {
                bail!(
                    "{}: dense vector '{name}' was scattered with {} elements, but the config \
                     declares size {want_dim}.\n\n\
                     Dimensionality is a property of the source data, not something a build can \
                     change. Re-scatter, or set the config back to {}.",
                    path.display(),
                    spec.dim,
                    spec.dim,
                );
            }

            check_source(
                path,
                name,
                spec.source.as_deref(),
                wanted.dense_sources.get(name),
            )?;
            dense.insert(name.clone(), spec.encoding);
        }

        let mut sparse = std::collections::BTreeSet::new();
        for name in &wanted.sparse {
            let spec = manifest.sparse.get(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "{} does not carry a sparse vector named '{name}'.\n\n\
                     It carries: {:?}. Re-scatter with '{name}' in the mapping's \
                     `sparse_vectors`.",
                    path.display(),
                    manifest.sparse.keys().collect::<Vec<_>>(),
                )
            })?;
            check_source(
                path,
                name,
                spec.source.as_deref(),
                wanted.sparse_sources.get(name),
            )?;
            sparse.insert(name.clone());
        }

        // Only checkable when the part enumerated its columns. JSONL parts cannot, so a payload
        // key that was never captured surfaces as an absent field rather than an error here.
        if let (Some(captured), Some(requested)) = (&manifest.payload, &wanted.payload) {
            let captured: std::collections::BTreeSet<&String> = captured.iter().collect();
            let missing: Vec<&String> = requested
                .iter()
                .filter(|key| !captured.contains(key))
                .collect();
            if !missing.is_empty() {
                bail!(
                    "{} does not carry payload column(s) {missing:?}.\n\n\
                     It carries: {captured:?}. `payload_columns` at build time selects from what \
                     the scatter captured; it cannot add columns that were never read.",
                    path.display(),
                );
            }
        }

        let dropped_dense = manifest
            .dense
            .keys()
            .filter(|name| !dense.contains_key(*name))
            .cloned()
            .collect();
        let dropped_sparse = manifest
            .sparse
            .keys()
            .filter(|name| !sparse.contains(*name))
            .cloned()
            .collect();
        // Sorted, unlike the manifest's declaration order, so the log line a downselect produces
        // is stable across configs that list the same columns differently.
        let mut dropped_payload: Vec<String> = match (&manifest.payload, &wanted.payload) {
            (Some(captured), Some(requested)) => captured
                .iter()
                .filter(|key| !requested.contains(*key))
                .cloned()
                .collect(),
            _ => Vec::new(),
        };
        dropped_payload.sort();

        Ok(Self {
            dense,
            sparse,
            payload: wanted.payload.clone(),
            dropped_dense,
            dropped_sparse,
            dropped_payload,
        })
    }

    /// True when nothing is being dropped, so a log line can be skipped.
    pub fn is_identity(&self) -> bool {
        self.dropped_dense.is_empty()
            && self.dropped_sparse.is_empty()
            && self.dropped_payload.is_empty()
    }

    /// One-line summary of what is being dropped.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.dropped_dense.is_empty() {
            parts.push(format!("dense {:?}", self.dropped_dense));
        }
        if !self.dropped_sparse.is_empty() {
            parts.push(format!("sparse {:?}", self.dropped_sparse));
        }
        if !self.dropped_payload.is_empty() {
            parts.push(format!("payload {:?}", self.dropped_payload));
        }
        parts.join(", ")
    }
}

/// Refuse a vector whose source column changed since the scatter.
///
/// Without this, remapping `dense` from one column to another and reusing the scatter would build
/// the *old* column's values under the new mapping — every count correct, every vector wrong.
fn check_source(
    path: &Path,
    name: &str,
    scattered: Option<&str>,
    wanted: Option<&String>,
) -> Result<()> {
    let (Some(scattered), Some(wanted)) = (scattered, wanted) else {
        return Ok(());
    };
    if scattered != wanted {
        bail!(
            "{}: vector '{name}' was scattered from column '{scattered}', but the mapping now \
             names '{wanted}'.\n\n\
             Reusing the scatter would store the old column's values under the new mapping. \
             Re-scatter, or point '{name}' back at '{scattered}'.",
            path.display(),
        );
    }
    Ok(())
}

/// One point as stored in a part file.
///
/// Distinct from [`PointStructPersisted`] purely so dense vectors can be held at their
/// configured width. Qdrant's ingest type is f32-only, so [`to_point`] widens on the way out —
/// once, at the moment points are handed to `EdgeShard::update`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartRecord {
    pub id: segment::types::PointIdType,
    /// Vector name -> packed little-endian elements.
    pub dense: std::collections::BTreeMap<String, DenseBytes>,
    /// Sparse vectors keep their structured form: indices and values are already compact and
    /// their count varies per point, so a packed layout would save little.
    pub sparse: std::collections::BTreeMap<String, sparse::common::sparse_vector::SparseVector>,
    pub payload: Option<segment::types::Payload>,
}

impl PartRecord {
    /// Build a record from a point, packing each dense vector at its configured width.
    pub fn from_point(point: PointStructPersisted, encodings: &DenseEncodings) -> Result<Self> {
        let mut dense = std::collections::BTreeMap::new();
        let mut sparse = std::collections::BTreeMap::new();

        let named = match point.vector {
            VectorStructPersisted::Named(named) => named,
            // The unnamed forms only arise from hand-written JSONL input; map them to the
            // default vector name so the rest of the pipeline sees one shape.
            VectorStructPersisted::Single(values) => std::collections::HashMap::from([(
                segment::data_types::vectors::DEFAULT_VECTOR_NAME.to_string(),
                VectorPersisted::Dense(values),
            )]),
            VectorStructPersisted::MultiDense(_) => {
                bail!("multi-dense vectors are not supported by the part format yet")
            }
        };

        for (name, vector) in named {
            match vector {
                VectorPersisted::Dense(values) => {
                    let encoding = encodings.get(&name).copied().unwrap_or(DenseEncoding::F32);
                    dense.insert(name, encoding.encode(&values));
                }
                VectorPersisted::Sparse(vector) => {
                    sparse.insert(name, vector);
                }
                VectorPersisted::MultiDense(_) => {
                    bail!("multi-dense vector '{name}' is not supported by the part format yet")
                }
            }
        }

        Ok(Self {
            id: point.id,
            dense,
            sparse,
            payload: point.payload,
        })
    }

    /// Convert into the type the bulk segment builder accepts, consuming the record.
    ///
    /// Takes `self` by value rather than by reference so the payload and the sparse vectors are
    /// *moved* into the point. [`Self::to_point`] has to clone them, which for a corpus whose
    /// payload is a document body is a deep clone of a JSON tree per point — pure waste, since the
    /// record is dropped immediately afterwards either way.
    ///
    /// Dense vectors are still widened to `f32` here: that is the element type
    /// [`segment::data_types::named_vectors::CowVector`] carries, and the storage narrows back to
    /// the configured width on write.
    /// Iterates the record's own fields and skips what the projection excludes, so a dropped
    /// dense vector is never decoded and a dropped payload key is never cloned — the work saved is
    /// proportional to what is dropped, which is the point.
    pub fn into_point_to_insert(
        self,
        projection: &PartProjection,
        version: segment::types::SeqNumberType,
    ) -> segment::common::operation_error::OperationResult<
        segment::segment_constructor::segment_builder::PointToInsert<'static>,
    > {
        use segment::common::operation_error::OperationError;
        use segment::data_types::named_vectors::NamedVectors;
        use segment::data_types::vectors::VectorInternal;

        let Self {
            id,
            dense,
            sparse,
            mut payload,
        } = self;

        let mut vectors = NamedVectors::default();

        for (name, bytes) in dense {
            // Not in the projection means the config no longer declares it. Skipping before the
            // decode is what makes dropping a dense vector cheaper than keeping it.
            let Some(encoding) = projection.dense.get(&name).copied() else {
                continue;
            };
            let values = encoding.decode(&bytes).map_err(|err| {
                OperationError::service_error(format!(
                    "dense vector '{name}' of point {id:?}: {err:#}"
                ))
            })?;
            vectors.insert(name, VectorInternal::Dense(values));
        }

        for (name, vector) in sparse {
            if !projection.sparse.contains(&name) {
                continue;
            }
            vectors.insert(name, VectorInternal::Sparse(vector));
        }

        if let (Some(keep), Some(payload)) = (&projection.payload, payload.as_mut()) {
            payload.0.retain(|key, _| keep.contains(key));
        }

        Ok(
            segment::segment_constructor::segment_builder::PointToInsert {
                external_id: id,
                version,
                vectors,
                payload,
            },
        )
    }

    /// Widen back into the type `EdgeShard::update` accepts, applying the same projection.
    pub fn to_point(&self, projection: &PartProjection) -> Result<PointStructPersisted> {
        let mut vectors = std::collections::HashMap::new();

        for (name, bytes) in &self.dense {
            let Some(encoding) = projection.dense.get(name).copied() else {
                continue;
            };
            let values = encoding
                .decode(bytes)
                .with_context(|| format!("dense vector '{name}' of point {:?}", self.id))?;
            vectors.insert(name.clone(), VectorPersisted::Dense(values));
        }

        for (name, vector) in &self.sparse {
            if !projection.sparse.contains(name) {
                continue;
            }
            vectors.insert(name.clone(), VectorPersisted::Sparse(vector.clone()));
        }

        Ok(PointStructPersisted {
            id: self.id,
            vector: VectorStructPersisted::Named(vectors),
            payload: self.payload.clone(),
        })
    }
}

/// Appends records to a part file, writing to a temporary path until [`Self::commit`].
///
/// The temporary-then-rename discipline is what makes the scatter phase resumable: a
/// half-written part is never visible under its final name, so a crashed run leaves only
/// `.tmp` files for the next run to sweep.
pub struct PartWriter {
    writer: BufWriter<File>,
    temp_path: PathBuf,
    final_path: PathBuf,
    records: u64,
    committed: bool,
}

impl PartWriter {
    /// Create a part file at `final_path`, writing through `final_path` + `.tmp`.
    pub fn create(final_path: PathBuf, header: &PartHeader) -> Result<Self> {
        if let Some(parent) = final_path.parent() {
            fs_err::create_dir_all(parent)
                .with_context(|| format!("cannot create {}", parent.display()))?;
        }

        let temp_path = unique_tmp_path(&final_path);
        let file = File::create(&temp_path)
            .with_context(|| format!("cannot create {}", temp_path.display()))?;
        let mut writer = BufWriter::new(file);

        let header_bytes = serde_json::to_vec(header).context("cannot serialize part header")?;
        let header_len =
            u32::try_from(header_bytes.len()).context("part header is implausibly large")?;

        writer.write_all(MAGIC)?;
        writer.write_all(&header_len.to_le_bytes())?;
        writer.write_all(&header_bytes)?;

        Ok(Self {
            writer,
            temp_path,
            final_path,
            records: 0,
            committed: false,
        })
    }

    pub fn append(&mut self, record: &PartRecord) -> Result<()> {
        let record = serde_cbor::to_vec(record).context("cannot serialize point")?;
        let record_len = u32::try_from(record.len()).context("point is implausibly large")?;

        if record_len > MAX_RECORD_BYTES {
            bail!("point serializes to {record_len} bytes, above the {MAX_RECORD_BYTES} limit");
        }

        self.writer.write_all(&record_len.to_le_bytes())?;
        self.writer.write_all(&record)?;
        self.records += 1;
        Ok(())
    }

    /// Flush, fsync, and rename into place.
    ///
    /// The fsync matters: without it the rename can land while the contents are still only in
    /// the page cache, so a crash would leave a *committed-looking* part file with a
    /// truncated tail — which is worse than an obviously-incomplete `.tmp`.
    pub fn commit(mut self) -> Result<u64> {
        self.writer
            .flush()
            .with_context(|| format!("cannot flush {}", self.temp_path.display()))?;
        self.writer
            .get_ref()
            .sync_all()
            .with_context(|| format!("cannot fsync {}", self.temp_path.display()))?;

        fs_err::rename(&self.temp_path, &self.final_path).with_context(|| {
            format!(
                "cannot rename {} to {}",
                self.temp_path.display(),
                self.final_path.display(),
            )
        })?;

        // fsync the shard directory so the rename itself is durable, not just the file
        // contents. Without this a crash can lose the rename while the `done/` marker (fsync'd
        // in `scatter::mark_done`) survives — and cross-directory metadata ordering is not
        // guaranteed — so resume would skip the file and its points would silently vanish.
        // The tool targets network filesystems where assuming local-fs ordering is untenable.
        common::fs::sync_parent_dir(&self.final_path)
            .with_context(|| format!("cannot fsync parent dir of {}", self.final_path.display()))?;

        // Sidecar after the rename: the part is the source of truth, and a sidecar without a
        // part would be misleading. `read_meta` recovers a missing sidecar by counting.
        let bytes = fs_err::metadata(&self.final_path)
            .with_context(|| format!("cannot stat {}", self.final_path.display()))?
            .len();
        let meta = PartMeta {
            records: self.records,
            bytes,
        };
        common::fs::atomic_save_json(&meta_path(&self.final_path), &meta)
            .with_context(|| format!("cannot write sidecar for {}", self.final_path.display()))?;

        self.committed = true;
        Ok(self.records)
    }
}

impl Drop for PartWriter {
    fn drop(&mut self) {
        // An uncommitted writer means the run failed or was interrupted. Remove the temp file
        // so the work directory does not accumulate debris; a failure to remove it is not
        // itself fatal, since the next run sweeps `.tmp` files anyway.
        if !self.committed && self.temp_path.exists() {
            let _ = fs_err::remove_file(&self.temp_path);
        }
    }
}

/// Reads records from a part file, verifying the header first.
#[derive(Debug)]
pub struct PartReader {
    reader: BufReader<File>,
    header: PartHeader,
    path: PathBuf,
}

impl PartReader {
    /// Open a part file and check it was written under `expected_fingerprint`.
    pub fn open(path: &Path, expected_fingerprint: &str) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .with_context(|| format!("{} is too short to be a part file", path.display()))?;
        if &magic != MAGIC {
            // Distinguish "wrong file" from "part file from an older format": the second is a
            // recoverable situation with a specific remedy, and reporting it as bad magic sends
            // the reader looking for corruption that is not there.
            if magic.starts_with(b"QSBPART") {
                bail!(
                    "{} is a part file in format version {}, but this build reads version {}.\n\n\
                     Version \\x03 narrowed the header fingerprint so that tuning `hnsw_config`, \
                     `quantization_config` or `max_segment_size` no longer invalidates a scatter. \
                     Older parts record the full config surface in the same field, so they cannot \
                     be compared against the new one. Re-run the scatter phase into a clean work \
                     directory; from then on those fields are frozen at `plan` time instead.",
                    path.display(),
                    magic[7],
                    MAGIC[7],
                );
            }
            bail!(
                "{} is not a part file (bad magic); expected {:?}",
                path.display(),
                String::from_utf8_lossy(MAGIC),
            );
        }

        let mut header_len = [0u8; 4];
        reader.read_exact(&mut header_len)?;
        let header_len = u32::from_le_bytes(header_len);
        if header_len > MAX_RECORD_BYTES {
            bail!(
                "{}: header length {header_len} is implausible",
                path.display()
            );
        }

        let mut header_bytes = vec![0u8; header_len as usize];
        reader.read_exact(&mut header_bytes)?;
        let header: PartHeader = serde_json::from_slice(&header_bytes)
            .with_context(|| format!("{}: cannot parse part header", path.display()))?;

        if header.part_fingerprint != expected_fingerprint {
            bail!(
                "{} was scattered under an incompatible collection config\n  \
                 part fingerprint:     {}\n  \
                 expected fingerprint: {expected_fingerprint}\n\n\
                 This fingerprint covers only what a part's contents depend on: `shard_number`, \
                 `sharding_method`, and each dense vector's `size` and `datatype`. One of those \
                 changed, so records in this file may belong to a different shard or decode as \
                 garbage. Re-run the scatter phase into a clean work directory.\n\n\
                 Note that `hnsw_config`, `quantization_config` and `max_segment_size` are *not* \
                 part of this check — changing them needs only a re-run of `plan`, not a scatter.",
                path.display(),
                header.part_fingerprint,
            );
        }

        Ok(Self {
            reader,
            header,
            path: path.to_path_buf(),
        })
    }

    pub fn header(&self) -> &PartHeader {
        &self.header
    }

    /// Read the next record, or `None` at a clean end of file.
    ///
    /// A truncated record is an error rather than a silent stop: part files are only visible
    /// under their final name after an fsync and rename, so a short tail means real
    /// corruption and must not be mistaken for "no more points".
    pub fn next_record(&mut self) -> Result<Option<PartRecord>> {
        // Distinguish a clean end of file (0 bytes left at a record boundary) from a truncated
        // length prefix (1-3 bytes). Reading all 4 with `read_exact` would report both as
        // `UnexpectedEof`, so a part chopped 1-3 bytes past a record boundary would read as a
        // clean stop and its tail records would silently vanish. Read the first byte with
        // `read` — `Ok(0)` is the only true EOF — then require the remaining three.
        let mut first = [0u8; 1];
        match self.reader.read(&mut first) {
            Ok(0) => return Ok(None),
            Ok(_) => {}
            Err(err) => {
                return Err(err).with_context(|| format!("{}: cannot read", self.path.display()));
            }
        }
        let mut rest = [0u8; 3];
        self.reader.read_exact(&mut rest).with_context(|| {
            format!(
                "{}: truncated record-length prefix (got 1 of 4 bytes); file is corrupt",
                self.path.display(),
            )
        })?;
        let len_bytes = [first[0], rest[0], rest[1], rest[2]];

        let record_len = u32::from_le_bytes(len_bytes);
        if record_len > MAX_RECORD_BYTES {
            bail!(
                "{}: record length {record_len} exceeds the {MAX_RECORD_BYTES} limit; \
                 file is corrupt",
                self.path.display(),
            );
        }

        let mut record = vec![0u8; record_len as usize];
        self.reader.read_exact(&mut record).with_context(|| {
            format!(
                "{}: record truncated (expected {record_len} bytes)",
                self.path.display(),
            )
        })?;

        let record = serde_cbor::from_slice(&record)
            .with_context(|| format!("{}: cannot deserialize point", self.path.display()))?;

        Ok(Some(record))
    }
}

/// Sidecar recorded next to each committed part.
///
/// The plan step needs a record count per part to group parts into segments. Counting by
/// re-reading every part would mean a full extra pass over the whole corpus, so the count is
/// written once, at commit, by the worker that already has it.
///
/// A sidecar is used rather than a footer inside the part so the part format stays a plain
/// sequence of frames, and so a missing sidecar can be recovered by counting just that one
/// part instead of invalidating the file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PartMeta {
    pub records: u64,
    pub bytes: u64,
}

/// Path of the sidecar belonging to `part_path`.
pub fn meta_path(part_path: &Path) -> PathBuf {
    let mut name = part_path.file_name().unwrap_or_default().to_os_string();
    name.push(".meta");
    part_path.with_file_name(name)
}

/// Read a part's sidecar, counting the part itself if the sidecar is absent.
pub fn read_meta(part_path: &Path, expected_fingerprint: &str) -> Result<PartMeta> {
    let meta = meta_path(part_path);

    if meta.exists() {
        let bytes =
            fs_err::read(&meta).with_context(|| format!("cannot read {}", meta.display()))?;
        return serde_json::from_slice(&bytes)
            .with_context(|| format!("{}: cannot parse part sidecar", meta.display()));
    }

    log::warn!(
        "{} has no sidecar; counting records directly (slower)",
        part_path.display(),
    );

    let mut reader = PartReader::open(part_path, expected_fingerprint)?;
    let mut records = 0;
    while reader.next_record()?.is_some() {
        records += 1;
    }

    Ok(PartMeta {
        records,
        bytes: fs_err::metadata(part_path)?.len(),
    })
}

/// A process-unique temporary path for `final_path`: `<name>.<pid>.<n>.tmp`.
///
/// Unique per (process, attempt) so two processes assigned the *same* input file — an
/// overlapping `--slice` assignment — never open and interleave bytes into one temp, which a
/// single deterministic `<name>.tmp` allowed (the interleaved mess then committed under the
/// final name with a matching sidecar: a corrupt-but-committed part). With distinct temps each
/// writer renames its own *complete* file to the same final path; the last rename wins and both
/// candidates are valid (same input + config produce identical parts). Mirrors build's
/// `incomplete_temp_path` for publishes.
pub fn unique_tmp_path(final_path: &Path) -> PathBuf {
    let mut name = final_path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        TMP_ATTEMPT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ));
    final_path.with_file_name(name)
}

static TMP_ATTEMPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether `temp_name` is a temporary file for the final part named `final_name`, i.e.
/// `<final_name>.<pid>.<n>.tmp`. Used by the scatter sweep to find this run's own temps.
/// Unambiguous because part names are `part_<fixed-length hex file_id>`, so one final name is
/// never a prefix of another before the trailing `.`.
pub fn is_temp_for(temp_name: &str, final_name: &str) -> bool {
    temp_name
        .strip_prefix(final_name)
        .and_then(|rest| rest.strip_prefix('.'))
        .is_some_and(|rest| rest.ends_with(".tmp"))
}

#[cfg(test)]
mod tests {
    use segment::types::ExtendedPointId;
    use shard::operations::point_ops::VectorStructPersisted;
    use tempfile::TempDir;

    use super::*;

    fn header(fingerprint: &str) -> PartHeader {
        PartHeader {
            part_fingerprint: fingerprint.to_string(),
            file_id: "abc123".to_string(),
            source_path: "input/data-0001.jsonl".to_string(),
            shard_id: 3,
            manifest: manifest(DenseEncoding::F16),
        }
    }

    /// A manifest for one dense vector at `encoding`, plus the sparse vector the fixtures carry.
    fn manifest(encoding: DenseEncoding) -> PartManifest {
        PartManifest {
            dense: std::collections::BTreeMap::from([(
                "dense".to_string(),
                DenseSpec {
                    encoding,
                    dim: 768,
                    source: None,
                },
            )]),
            sparse: std::collections::BTreeMap::from([(
                "sparse".to_string(),
                SparseSpec { source: None },
            )]),
            payload: None,
        }
    }

    /// An identity projection over `manifest` — keeps everything it carries.
    fn keep_all(manifest: &PartManifest) -> PartProjection {
        let wanted = WantedFields {
            dense: manifest
                .dense
                .iter()
                .map(|(name, spec)| (name.clone(), spec.dim))
                .collect(),
            sparse: manifest.sparse.keys().cloned().collect(),
            payload: None,
            dense_sources: Default::default(),
            sparse_sources: Default::default(),
        };
        let mut head = PartHeader {
            part_fingerprint: "fp".to_string(),
            file_id: "abc123".to_string(),
            source_path: "x".to_string(),
            shard_id: 3,
            manifest: manifest.clone(),
        };
        head.manifest = manifest.clone();
        PartProjection::resolve(&head, &wanted, Path::new("test")).unwrap()
    }

    /// One f16 vector, matching a `datatype: float16` collection.
    fn encodings() -> DenseEncodings {
        DenseEncodings::from([("dense".to_string(), DenseEncoding::F16)])
    }

    fn record(id: u64) -> PartRecord {
        PartRecord::from_point(point(id), &encodings()).unwrap()
    }

    /// Named "dense" so it matches `encodings()`; the real corpus is named too.
    fn point(id: u64) -> PointStructPersisted {
        PointStructPersisted {
            id: ExtendedPointId::NumId(id),
            vector: VectorStructPersisted::Named(std::collections::HashMap::from([(
                "dense".to_string(),
                VectorPersisted::Dense(vec![0.1, 0.2, 0.3]),
            )])),
            payload: None,
        }
    }

    #[test]
    fn round_trips_points() {
        let dir = TempDir::with_prefix("partfile").unwrap();
        let path = dir.path().join("part_abc123");

        let mut writer = PartWriter::create(path.clone(), &header("fp")).unwrap();
        for id in 0..100 {
            writer.append(&record(id)).unwrap();
        }
        assert_eq!(writer.commit().unwrap(), 100);

        let mut reader = PartReader::open(&path, "fp").unwrap();
        assert_eq!(reader.header(), &header("fp"));

        let mut seen = Vec::new();
        while let Some(r) = reader.next_record().unwrap() {
            seen.push(r);
        }
        assert_eq!(seen.len(), 100);
        // Round-trips through the packed f16 encoding back to the ingest type. Exact, because
        // 0.1/0.2/0.3 are stored as the nearest f16 and widened back to that same value.
        for (index, expected) in [(0usize, 0u64), (99, 99)] {
            let got = seen[index]
                .to_point(&keep_all(&manifest(DenseEncoding::F16)))
                .unwrap();
            let VectorStructPersisted::Named(vectors) = &got.vector else {
                panic!("expected named vectors");
            };
            let VectorPersisted::Dense(values) = &vectors["dense"] else {
                panic!("expected a dense vector");
            };
            assert_eq!(got.id, ExtendedPointId::NumId(expected));
            assert_eq!(values.len(), 3);
            let want: Vec<f32> = [0.1f32, 0.2, 0.3]
                .iter()
                .map(|v| half::f16::from_f32(*v).to_f32())
                .collect();
            assert_eq!(values, &want);
        }
    }

    #[test]
    fn round_trips_payloads_and_named_vectors() {
        // Exercises the CBOR-over-bincode decision: both `Payload` (serde_json::Value) and
        // the untagged `VectorStructPersisted::Named` need `deserialize_any`.
        let dir = TempDir::with_prefix("partfile").unwrap();
        let path = dir.path().join("part_named");

        let mut named = std::collections::HashMap::new();
        named.insert(
            "dense".to_string(),
            shard::operations::point_ops::VectorPersisted::Dense(vec![1.0, 2.0]),
        );

        let original = PointStructPersisted {
            id: ExtendedPointId::Uuid("550e8400-e29b-41d4-a716-446655440000".parse().unwrap()),
            vector: VectorStructPersisted::Named(named),
            payload: Some(segment::payload_json! { "city": "Berlin", "count": 42 }),
        };

        let mut writer = PartWriter::create(path.clone(), &header("fp")).unwrap();
        writer
            .append(&PartRecord::from_point(original.clone(), &encodings()).unwrap())
            .unwrap();
        writer.commit().unwrap();

        let mut reader = PartReader::open(&path, "fp").unwrap();
        let read_back = reader
            .next_record()
            .unwrap()
            .unwrap()
            .to_point(&keep_all(&manifest(DenseEncoding::F16)))
            .unwrap();
        // Named-vector shape, payload, and UUID id all survive; f16 narrowing is exact for
        // values that fit, which these do.
        assert_eq!(read_back.id, original.id);
        assert_eq!(read_back.payload, original.payload);
        assert!(reader.next_record().unwrap().is_none());
    }

    #[test]
    fn refuses_a_part_written_under_a_different_config() {
        let dir = TempDir::with_prefix("partfile").unwrap();
        let path = dir.path().join("part_abc123");

        let mut writer = PartWriter::create(path.clone(), &header("old-fingerprint")).unwrap();
        writer.append(&record(1)).unwrap();
        writer.commit().unwrap();

        let err = PartReader::open(&path, "new-fingerprint").unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("incompatible collection config"),
            "{message}"
        );
        assert!(message.contains("old-fingerprint"), "{message}");
        assert!(message.contains("new-fingerprint"), "{message}");
    }

    #[test]
    fn uncommitted_writer_leaves_no_final_file() {
        let dir = TempDir::with_prefix("partfile").unwrap();
        let path = dir.path().join("part_abc123");

        {
            let mut writer = PartWriter::create(path.clone(), &header("fp")).unwrap();
            writer.append(&record(1)).unwrap();
            // Dropped without commit, simulating a crash mid-file.
        }

        assert!(!path.exists(), "final path must not exist without a commit");
        // The temp name is process-unique, so scan for any leftover `.tmp` in the directory.
        let leftover_tmp = fs_err::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| e.file_name().to_string_lossy().ends_with(".tmp"));
        assert!(!leftover_tmp, "Drop should clean up the temp file");
    }

    #[test]
    fn rejects_a_non_part_file() {
        let dir = TempDir::with_prefix("partfile").unwrap();
        let path = dir.path().join("not_a_part");
        fs_err::write(&path, b"just some bytes that are long enough").unwrap();

        let err = PartReader::open(&path, "fp").unwrap_err();
        assert!(format!("{err:#}").contains("bad magic"), "{err:#}");
    }

    /// A truncated tail must error, not read as a clean end of file.
    #[test]
    fn detects_a_truncated_record() {
        let dir = TempDir::with_prefix("partfile").unwrap();
        let path = dir.path().join("part_trunc");

        let mut writer = PartWriter::create(path.clone(), &header("fp")).unwrap();
        for id in 0..10 {
            writer.append(&record(id)).unwrap();
        }
        writer.commit().unwrap();

        // Chop the last few bytes, leaving a length prefix promising more than remains.
        let bytes = fs_err::read(&path).unwrap();
        fs_err::write(&path, &bytes[..bytes.len() - 5]).unwrap();

        let mut reader = PartReader::open(&path, "fp").unwrap();
        let err = loop {
            match reader.next_record() {
                Ok(Some(_)) => {}
                Ok(None) => panic!("truncated file must not read as a clean EOF"),
                Err(err) => break err,
            }
        };
        assert!(format!("{err:#}").contains("truncated"), "{err:#}");
    }

    /// The specific hole behind the sidecar: a part chopped 1-3 bytes into a record-length
    /// prefix (past a complete-records boundary) must be an error, not a clean EOF.
    #[test]
    fn detects_a_truncated_length_prefix() {
        let dir = TempDir::with_prefix("partfile").unwrap();

        // A one-record file measures `header + frame(record 0)`, i.e. the byte offset of the
        // second record's length prefix in a two-record file.
        let one_path = dir.path().join("part_one");
        let mut w = PartWriter::create(one_path.clone(), &header("fp")).unwrap();
        w.append(&record(0)).unwrap();
        w.commit().unwrap();
        let boundary = fs_err::metadata(&one_path).unwrap().len() as usize;

        let two_path = dir.path().join("part_two");
        let mut w = PartWriter::create(two_path.clone(), &header("fp")).unwrap();
        w.append(&record(0)).unwrap();
        w.append(&record(1)).unwrap();
        w.commit().unwrap();

        // Leave exactly 2 of the 4 bytes of the second record's length prefix.
        let bytes = fs_err::read(&two_path).unwrap();
        fs_err::write(&two_path, &bytes[..boundary + 2]).unwrap();

        let mut reader = PartReader::open(&two_path, "fp").unwrap();
        assert!(
            reader.next_record().unwrap().is_some(),
            "first record intact"
        );
        let err = reader
            .next_record()
            .expect_err("a truncated length prefix must not read as a clean EOF");
        assert!(format!("{err:#}").contains("truncated"), "{err:#}");
    }

    /// Two writers to the same final path get distinct temps and each produces a complete,
    /// valid part — no interleaving. Simulates overlapping `--slice` assignment of one input.
    #[test]
    fn concurrent_writers_do_not_interleave() {
        let dir = TempDir::with_prefix("partfile").unwrap();
        let final_path = dir.path().join("part_abc0000000000000");

        let mut a = PartWriter::create(final_path.clone(), &header("fp")).unwrap();
        let mut b = PartWriter::create(final_path.clone(), &header("fp")).unwrap();
        let a_temp = a.temp_path.clone();
        let b_temp = b.temp_path.clone();
        assert_ne!(a_temp, b_temp, "temps must be process/attempt-unique");

        // Interleave appends across the two open writers, then commit both.
        for id in 0..5 {
            a.append(&record(id)).unwrap();
            b.append(&record(id)).unwrap();
        }
        a.commit().unwrap();
        b.commit().unwrap(); // last rename wins; both were complete

        // The committed final part is a complete, readable 5-record file — not interleaved garbage.
        let mut reader = PartReader::open(&final_path, "fp").unwrap();
        let mut count = 0;
        while reader.next_record().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 5);
        // is_temp_for recognizes both temps for this final name, and not a different one.
        for temp in [&a_temp, &b_temp] {
            let name = temp.file_name().unwrap().to_string_lossy();
            assert!(is_temp_for(&name, "part_abc0000000000000"));
            assert!(!is_temp_for(&name, "part_def0000000000000"));
        }
    }

    #[test]
    fn empty_part_file_reads_as_zero_points() {
        let dir = TempDir::with_prefix("partfile").unwrap();
        let path = dir.path().join("part_empty");

        let writer = PartWriter::create(path.clone(), &header("fp")).unwrap();
        assert_eq!(writer.commit().unwrap(), 0);

        let mut reader = PartReader::open(&path, "fp").unwrap();
        assert!(reader.next_record().unwrap().is_none());
    }

    // ---------------------------------------------------------------- projection

    /// Build a manifest resembling a real parquet scatter: two dense, one sparse, three columns.
    fn rich_manifest() -> PartManifest {
        PartManifest {
            dense: std::collections::BTreeMap::from([
                (
                    "dense".to_string(),
                    DenseSpec {
                        encoding: DenseEncoding::F16,
                        dim: 768,
                        source: Some("dense_embedding".to_string()),
                    },
                ),
                (
                    "extra".to_string(),
                    DenseSpec {
                        encoding: DenseEncoding::F32,
                        dim: 4,
                        source: Some("extra_embedding".to_string()),
                    },
                ),
            ]),
            sparse: std::collections::BTreeMap::from([(
                "sparse".to_string(),
                SparseSpec {
                    source: Some("sparse_embedding".to_string()),
                },
            )]),
            payload: Some(vec![
                "text".to_string(),
                "url".to_string(),
                "dump".to_string(),
            ]),
        }
    }

    fn head_with(manifest: PartManifest) -> PartHeader {
        PartHeader {
            part_fingerprint: "fp".to_string(),
            file_id: "abc123".to_string(),
            source_path: "input/data.parquet".to_string(),
            shard_id: 3,
            manifest,
        }
    }

    fn wants(dense: &[(&str, usize)], sparse: &[&str], payload: Option<&[&str]>) -> WantedFields {
        WantedFields {
            dense: dense
                .iter()
                .map(|(name, dim)| (name.to_string(), *dim))
                .collect(),
            sparse: sparse.iter().map(|name| name.to_string()).collect(),
            payload: payload.map(|keys| keys.iter().map(|key| key.to_string()).collect()),
            dense_sources: Default::default(),
            sparse_sources: Default::default(),
        }
    }

    /// The capability: a build may ask for less than the scatter captured.
    #[test]
    fn a_projection_can_drop_vectors_and_payload_columns() {
        let head = head_with(rich_manifest());
        let wanted = wants(&[("dense", 768)], &[], Some(&["url"]));

        let projection = PartProjection::resolve(&head, &wanted, Path::new("p")).unwrap();

        assert_eq!(projection.dropped_dense, vec!["extra".to_string()]);
        assert_eq!(projection.dropped_sparse, vec!["sparse".to_string()]);
        assert_eq!(
            projection.dropped_payload,
            vec!["dump".to_string(), "text".to_string()],
            "dropped payload keys are reported so a downselect is visible",
        );
        assert!(!projection.is_identity());
        assert!(projection.summary().contains("extra"));
    }

    /// Wanting exactly what is there drops nothing, and says so.
    #[test]
    fn a_projection_that_keeps_everything_is_the_identity() {
        let manifest = rich_manifest();
        let head = head_with(manifest.clone());
        let wanted = wants(
            &[("dense", 768), ("extra", 4)],
            &["sparse"],
            Some(&["text", "url", "dump"]),
        );

        let projection = PartProjection::resolve(&head, &wanted, Path::new("p")).unwrap();
        assert!(projection.is_identity(), "{:?}", projection.summary());
    }

    /// The projection must actually shape the point, not just report intent.
    #[test]
    fn projecting_a_record_omits_the_dropped_fields() {
        use segment::data_types::vectors::VectorRef;

        let manifest = PartManifest {
            dense: std::collections::BTreeMap::from([(
                "dense".to_string(),
                DenseSpec {
                    encoding: DenseEncoding::F16,
                    dim: 2,
                    source: None,
                },
            )]),
            sparse: std::collections::BTreeMap::from([(
                "sparse".to_string(),
                SparseSpec { source: None },
            )]),
            payload: Some(vec!["keep".to_string(), "drop".to_string()]),
        };
        let head = head_with(manifest);

        let record = PartRecord {
            id: ExtendedPointId::from(7u64),
            dense: std::collections::BTreeMap::from([(
                "dense".to_string(),
                DenseEncoding::F16.encode(&[1.0, 2.0]),
            )]),
            sparse: std::collections::BTreeMap::from([(
                "sparse".to_string(),
                sparse::common::sparse_vector::SparseVector::new(vec![1], vec![1.0]).unwrap(),
            )]),
            payload: Some(segment::types::Payload(
                serde_json::json!({"keep": 1, "drop": 2})
                    .as_object()
                    .unwrap()
                    .clone(),
            )),
        };

        // Keep the dense vector and one payload key; drop the sparse vector and the other key.
        let wanted = wants(&[("dense", 2)], &[], Some(&["keep"]));
        let projection = PartProjection::resolve(&head, &wanted, Path::new("p")).unwrap();

        let point = record.into_point_to_insert(&projection, 1).unwrap();

        let names: Vec<&str> = point.vectors.keys().collect();
        assert_eq!(names, vec!["dense"], "sparse must not reach the point");
        assert!(matches!(
            point.vectors.get("dense").unwrap(),
            VectorRef::Dense(_)
        ));

        let payload = point.payload.expect("payload kept");
        assert!(payload.0.contains_key("keep"));
        assert!(
            !payload.0.contains_key("drop"),
            "dropped payload key reached the point: {payload:?}",
        );
    }

    /// A projection can drop, never invent — and it must say what is missing.
    #[test]
    fn asking_for_something_the_part_lacks_is_refused_by_name() {
        let head = head_with(rich_manifest());

        let missing_dense = wants(&[("dense", 768), ("nope", 16)], &[], None);
        let err = PartProjection::resolve(&head, &missing_dense, Path::new("p")).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("'nope'"), "{message}");
        assert!(message.contains("re-scatter"), "{message}");

        let missing_sparse = wants(&[("dense", 768)], &["absent"], None);
        let err = PartProjection::resolve(&head, &missing_sparse, Path::new("p")).unwrap_err();
        assert!(format!("{err:#}").contains("'absent'"));

        let missing_payload = wants(&[("dense", 768)], &[], Some(&["url", "language"]));
        let err = PartProjection::resolve(&head, &missing_payload, Path::new("p")).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("language"), "{message}");
        assert!(message.contains("cannot add columns"), "{message}");
    }

    /// Dimensionality is a property of the data, so a size change cannot be projected.
    #[test]
    fn a_dimensionality_change_is_refused() {
        let head = head_with(rich_manifest());
        let wanted = wants(&[("dense", 512)], &[], None);

        let err = PartProjection::resolve(&head, &wanted, Path::new("p")).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("768"), "{message}");
        assert!(message.contains("512"), "{message}");
    }

    /// The silent-corruption case: same vector name, different source column.
    ///
    /// Without this check, remapping `dense` to another column and reusing the scatter would store
    /// the old column's values — every count right, every vector wrong.
    #[test]
    fn remapping_a_vector_to_a_different_column_is_refused() {
        let head = head_with(rich_manifest());

        let mut wanted = wants(&[("dense", 768)], &[], None);
        wanted
            .dense_sources
            .insert("dense".to_string(), "other_embedding".to_string());

        let err = PartProjection::resolve(&head, &wanted, Path::new("p")).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("dense_embedding"), "{message}");
        assert!(message.contains("other_embedding"), "{message}");

        // The same column is of course fine.
        let mut same = wants(&[("dense", 768)], &[], None);
        same.dense_sources
            .insert("dense".to_string(), "dense_embedding".to_string());
        PartProjection::resolve(&head, &same, Path::new("p")).unwrap();
    }

    /// JSONL parts cannot enumerate their payload, so no subset check is possible there.
    #[test]
    fn an_unenumerated_payload_is_not_subset_checked() {
        let mut manifest = rich_manifest();
        manifest.payload = None;
        let head = head_with(manifest);

        // Requesting a key the part may or may not have must not fail: there is nothing to check.
        let wanted = wants(&[("dense", 768)], &[], Some(&["anything"]));
        let projection = PartProjection::resolve(&head, &wanted, Path::new("p")).unwrap();
        assert!(projection.dropped_payload.is_empty());
    }

    /// The point of the format change: an f16 part must be materially smaller.
    #[test]
    fn f16_parts_are_much_smaller_than_f32_parts() {
        let dir = TempDir::with_prefix("partfile").unwrap();

        let wide: DenseEncodings =
            DenseEncodings::from([("dense".to_string(), DenseEncoding::F32)]);
        let narrow = encodings();

        let mut sizes = Vec::new();
        for (name, enc) in [("f32", &wide), ("f16", &narrow)] {
            let path = dir.path().join(format!("part_{name}"));
            let mut head = header("fp");
            head.manifest = manifest(*enc.get("dense").unwrap());
            let mut writer = PartWriter::create(path.clone(), &head).unwrap();
            for id in 0..200 {
                let mut p = point(id);
                p.vector = VectorStructPersisted::Named(std::collections::HashMap::from([(
                    "dense".to_string(),
                    VectorPersisted::Dense(vec![0.5; 768]),
                )]));
                writer
                    .append(&PartRecord::from_point(p, enc).unwrap())
                    .unwrap();
            }
            writer.commit().unwrap();
            sizes.push(fs_err::metadata(&path).unwrap().len());
        }

        let (f32_size, f16_size) = (sizes[0], sizes[1]);
        assert!(
            f16_size * 3 / 2 < f32_size,
            "f16 part ({f16_size}) should be far below f32 ({f32_size})",
        );
    }
}
