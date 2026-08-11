//! Shared helpers for the Puffin-HNSW PoC test phases (§5 of the spec).
//! Phase 1 (`test_puffin_serialization`) exercises the writer; Phase 2
//! (`test_puffin_deserialization`) exercises the reader path — including the
//! new `GraphLinks::load_from_ranged_mmap` production API added in §6.3.
//!
//! Both phases share:
//!   - the on-disk container layout (§3.1)
//!   - the row-pointer wire format (§3.2 v3.11)
//!   - the deterministic build fixture that produces a well-formed `.puffin`
//!
//! Kept `pub(super)` so both sibling test modules can reach items via
//! `super::puffin_shared::*` without exposing anything past the `tests` module.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::types::PointOffsetType;
use fs_err as fs;
use memmap2::{Mmap, MmapOptions};
use quantization::encoded_storage::TestEncodedStorageBuilder;
use quantization::encoded_vectors_binary::{
    EncodedVectorsBin, Encoding, QueryEncoding, get_quantized_vector_size_from_params,
};
use quantization::{DistanceType, VectorParameters};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::json;
use tempfile::TempDir;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::fixtures::index_fixtures::{TestRawScorerProducer, random_vector};
use crate::index::hnsw_index::HnswM;
use crate::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
use crate::index::hnsw_index::graph_links::GraphLinksFormatParam;
use crate::types::Distance;

// ---------- Container-format constants -------------------------------------

pub(super) const MAGIC: &[u8; 4] = b"PFA1";
pub(super) const BLOB_ALIGN: usize = 64;
/// Trailing footer trailer: `[footer_size u32 LE | flags u32 LE | trailing magic 4B]`.
pub(super) const TRAILER_LEN: usize = 4 + 4 + 4;

// ---------- Fixture parameters (deterministic across phases) --------------

pub(super) const NUM_VECTORS: usize = 10_000;
pub(super) const DIM: usize = 768;
pub(super) const M: usize = 16;
pub(super) const EF_CONSTRUCT: usize = 100;
pub(super) const ENTRY_POINTS_NUM: usize = 10;
/// Seed = "PFA1" bytes packed as u64. Same seed both phases so the fixture is
/// bit-identical between Phase 1's writer test and Phase 2's reader test.
pub(super) const FIXTURE_SEED: u64 = 0x5046_4131;

// ---------- Helpers -------------------------------------------------------

#[inline]
pub(super) fn align_up(n: usize, to: usize) -> usize {
    debug_assert!(to.is_power_of_two());
    (n + to - 1) & !(to - 1)
}

pub(super) struct BlobSpec<'a> {
    pub blob_type: &'static str,
    pub bytes: &'a [u8],
    pub properties: serde_json::Value,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BlobRange {
    pub offset: usize,
    pub length: usize,
}

pub(super) struct PuffinLayout {
    // Only `footer_len` is currently consumed downstream (Phase 2 uses it to
    // locate a manually-corrupted footer in the negative test). If future
    // callers need ranges or mid_magic_offset, `write_puffin` records them
    // internally and can return them without changing its API contract.
    pub footer_len: usize,
}

/// Serialize a `.puffin` container per §3.1 of the spec. Returns the recorded blob
/// ranges plus the mid-magic offset and footer JSON length so callers can construct
/// an exact file-size formula.
pub(super) fn write_puffin(
    path: &Path,
    blobs: &[BlobSpec],
) -> std::io::Result<PuffinLayout> {
    let mut buf: Vec<u8> = Vec::new();

    // Leading magic.
    buf.extend_from_slice(MAGIC);

    // Blob region. Pad before each blob to align its start to 64 B. We track
    // (offset, length) per blob so we can emit the footer JSON; the concrete
    // records are consumed by the footer builder below rather than surfaced.
    let mut ranges: Vec<BlobRange> = Vec::with_capacity(blobs.len());
    for blob in blobs {
        let aligned = align_up(buf.len(), BLOB_ALIGN);
        buf.resize(aligned, 0);
        let offset = buf.len();
        buf.extend_from_slice(blob.bytes);
        ranges.push(BlobRange {
            offset,
            length: blob.bytes.len(),
        });
    }

    // Pad after last blob so mid-magic sits on a 64 B boundary (§3.1 v3.11
    // writer invariant; readers MUST NOT rely on this alignment).
    let mid_magic_offset = align_up(buf.len(), BLOB_ALIGN);
    buf.resize(mid_magic_offset, 0);
    buf.extend_from_slice(MAGIC);

    // Footer JSON. Per §3.3, `fields` MUST be a single-element array of the indexed
    // vector column's Iceberg field ID (all `ann-hnsw-*` blobs reference the same
    // column). `[1]` here only because the PoC has no Iceberg catalog to resolve
    // against; a production writer passes through the real field ID unchanged.
    let footer = json!({
        "blobs": blobs.iter().zip(&ranges).map(|(b, r)| json!({
            "type": b.blob_type,
            "fields": [1],
            "offset": r.offset,
            "length": r.length,
            "compression": "none",
            "properties": b.properties,
        })).collect::<Vec<_>>(),
        "properties": {
            "created-by": "qdrant-edge-builder-v1",
        },
    });
    let footer_bytes = serde_json::to_vec(&footer).expect("footer JSON serialisation");
    buf.extend_from_slice(&footer_bytes);

    // Trailer: footer_size (u32 LE), flags (u32 LE = 0), trailing magic.
    buf.extend_from_slice(&(footer_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(MAGIC);

    fs::write(path, &buf)?;
    // `mid_magic_offset` and the per-blob ranges were consumed by the footer
    // builder above; the caller only needs `footer_len` to locate the footer
    // in downstream verification.
    let _ = mid_magic_offset;
    Ok(PuffinLayout {
        footer_len: footer_bytes.len(),
    })
}

/// Build the `ann-hnsw-row-pointers-v1` blob body per §3.2 v3.11:
///   [u8 version=1] [u32 LE entry_count]
///   [u32 LE path_count] [{u32 LE len, UTF-8 bytes} × path_count]
///   [{u32 vec_id, u32 file_path_idx, u32 row_group, u32 row_offset} × entry_count]
/// All multi-byte integers little-endian. Entries pre-sorted ascending by vec_id.
pub(super) fn build_row_pointer_blob(
    file_paths: &[&str],
    entries: &[(u32, u32, u32, u32)], // (vec_id, file_path_idx, row_group, row_offset)
) -> Vec<u8> {
    // Entries MUST be strictly ascending by vec_id — the reader binary-searches on
    // vec_id, and strict-less-than also enforces uniqueness. `assert!` (not
    // `debug_assert!`) so the invariant still runs under `--release`.
    assert!(
        entries.windows(2).all(|w| w[0].0 < w[1].0),
        "row-pointer entries must be strictly ascending by vec_id",
    );

    // Spec §3.2 writer invariant: fail, do not wrap or truncate. `as u32` would
    // silently wrap on overflow, so use fallible conversion at every u32-width
    // boundary. This helper is the wire-format reference implementation.
    let entry_count = u32::try_from(entries.len())
        .expect("row-pointer entry count exceeds u32::MAX (§3.2 writer invariant)");
    let path_count = u32::try_from(file_paths.len())
        .expect("row-pointer path count exceeds u32::MAX");

    let mut out = Vec::with_capacity(
        1 + 4 + 4
            + file_paths.iter().map(|p| 4 + p.len()).sum::<usize>()
            + entries.len() * 16,
    );

    out.push(1u8);
    out.extend_from_slice(&entry_count.to_le_bytes());
    out.extend_from_slice(&path_count.to_le_bytes());
    for p in file_paths {
        let bytes = p.as_bytes();
        let path_len = u32::try_from(bytes.len())
            .expect("row-pointer path byte-length exceeds u32::MAX");
        out.extend_from_slice(&path_len.to_le_bytes());
        out.extend_from_slice(bytes);
    }
    for &(v, f, g, r) in entries {
        out.extend_from_slice(&v.to_le_bytes());
        out.extend_from_slice(&f.to_le_bytes());
        out.extend_from_slice(&g.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
    }
    out
}

// ---------- Fixture builder -----------------------------------------------

/// Fixture output — a fully-formed `.puffin` on disk plus the expected values
/// downstream tests need for their assertions. Keeps the `TempDir` open for the
/// lifetime of the fixture so paths remain valid.
pub(super) struct TestFixture {
    pub puffin_path: PathBuf,
    pub num_vectors: usize,
    pub dim: usize,
    pub m: usize,
    pub ef_construct: usize,
    pub quantized_vec_size: usize,
    pub file_paths: Vec<String>,
    // Kept alive so `puffin_path` remains valid until the fixture is dropped.
    pub _tmp: TempDir,
}

/// Build a deterministic `.puffin` container and return it as a fixture.
/// Called by both Phase 1 (which asserts writer output structure) and Phase 2
/// (which reads it back). Because the seed and parameters are fixed, both
/// phases see identical bytes.
pub(super) fn build_test_puffin_fixture() -> TestFixture {
    let tmp = TempDir::new().unwrap();
    let puffin_path = tmp.path().join("test_index.puffin");
    let graph_dir = tmp.path().join("graph");
    fs::create_dir_all(&graph_dir).unwrap();

    // ---- Build the HNSW graph -----------------------------------------------
    // Standard test scaffold for the scorer, but driving `link_new_point`
    // ourselves so we can set ef_construct=100 as §5 Phase 1 requires (the
    // shared fixture hardcodes ef=16, which is fine for speed tests but not
    // this spec).
    let mut rng = StdRng::seed_from_u64(FIXTURE_SEED);
    let vector_holder = TestRawScorerProducer::new(
        DIM,
        Distance::Dot,
        NUM_VECTORS,
        /* use_quantization */ false,
        &mut rng,
    );
    let mut builder = GraphLayersBuilder::new(
        NUM_VECTORS,
        HnswM::new2(M),
        EF_CONSTRUCT,
        ENTRY_POINTS_NUM,
        /* use_heuristic */ true,
    );
    for idx in 0..NUM_VECTORS as PointOffsetType {
        let level = builder.get_random_layer(&mut rng);
        builder.set_levels(idx, level);
        builder.link_new_point(idx, vector_holder.internal_scorer(idx));
    }

    // ---- Save graph to disk (yields graph.bin + links_compressed.bin) -------
    builder
        .into_graph_layers(
            &graph_dir,
            GraphLinksFormatParam::Compressed,
            /* on_disk */ false,
        )
        .unwrap();
    let graph_meta_bytes = fs::read(graph_dir.join("graph.bin")).unwrap();
    let graph_links_bytes = fs::read(graph_dir.join("links_compressed.bin")).unwrap();
    assert!(!graph_meta_bytes.is_empty(), "graph.bin empty");
    assert!(!graph_links_bytes.is_empty(), "links_compressed.bin empty");

    // ---- Quantize the SAME vector set the graph was trained against ---------
    // §3.4 (v3.12) invariant: all blobs in a container must derive from the
    // same vector set. `TestRawScorerProducer::new` iterates the passed RNG
    // via `random_vector(rng, dim)` × num_vectors (no extra draws for
    // Distance::Dot, whose preprocess is a no-op). A freshly-seeded RNG that
    // runs the same loop reproduces the identical vector sequence bit-for-bit.
    // Earlier drafts used FIXTURE_SEED.wrapping_add(1) here, which produced
    // an internally inconsistent container that no Phase 1/2 structural check
    // could detect — only the Phase 3 containment check catches it.
    let mut rng2 = StdRng::seed_from_u64(FIXTURE_SEED);
    let vectors_for_quant: Vec<Vec<f32>> = (0..NUM_VECTORS)
        .map(|_| random_vector(&mut rng2, DIM))
        .collect();

    let vector_parameters = VectorParameters {
        dim: DIM,
        distance_type: DistanceType::Dot,
        invert: false,
        deprecated_count: None,
    };
    let quantized_vec_size = get_quantized_vector_size_from_params::<u128>(DIM, Encoding::OneBit);
    let quant_data_path = tmp.path().join("quant.bin");
    let quant_meta_path = tmp.path().join("quant.meta.json");
    let storage_builder =
        TestEncodedStorageBuilder::new(Some(&quant_data_path), quantized_vec_size);
    let _encoded = EncodedVectorsBin::<u128, _>::encode(
        vectors_for_quant.iter().map(|v| v.as_slice()),
        storage_builder,
        &vector_parameters,
        Encoding::OneBit,
        QueryEncoding::SameAsStorage,
        Some(&quant_meta_path),
        &AtomicBool::new(false),
    )
    .expect("EncodedVectorsBin::encode");
    let quant_bytes = fs::read(&quant_data_path).unwrap();
    let quant_meta_bytes = fs::read(&quant_meta_path).unwrap();
    assert_eq!(
        quant_bytes.len(),
        NUM_VECTORS * quantized_vec_size,
        "quantized data length mismatch",
    );

    // ---- Row-pointer blob ----------------------------------------------------
    let file_paths = vec!["mock://parquet/dataset/file.parquet".to_string()];
    let path_refs: Vec<&str> = file_paths.iter().map(|s| s.as_str()).collect();
    let entries: Vec<(u32, u32, u32, u32)> =
        (0..NUM_VECTORS as u32).map(|i| (i, 0u32, 0u32, i)).collect();
    let row_ptr_bytes = build_row_pointer_blob(&path_refs, &entries);

    // ---- Assemble the .puffin container -------------------------------------
    let blobs = [
        BlobSpec {
            blob_type: "ann-hnsw-quantized-vectors-v1",
            bytes: &quant_bytes,
            properties: json!({
                "quantization_variant": "EncodedVectorsBin_u128",
                "quantization_family": "binary",
                "dimensions": DIM.to_string(),
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
        BlobSpec {
            blob_type: "ann-hnsw-quantized-meta-v1",
            bytes: &quant_meta_bytes,
            properties: json!({
                "quantization_family": "binary",
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
        BlobSpec {
            blob_type: "ann-hnsw-graph-meta-v1",
            bytes: &graph_meta_bytes,
            properties: json!({
                "m": M.to_string(),
                "ef_construct": EF_CONSTRUCT.to_string(),
                "vector_count": NUM_VECTORS.to_string(),
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
        BlobSpec {
            blob_type: "ann-hnsw-graph-links-v1",
            bytes: &graph_links_bytes,
            properties: json!({
                "format": "compressed",
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
        BlobSpec {
            blob_type: "ann-hnsw-row-pointers-v1",
            bytes: &row_ptr_bytes,
            properties: json!({
                "entry_count": NUM_VECTORS.to_string(),
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
    ];
    write_puffin(&puffin_path, &blobs).expect("write_puffin");

    TestFixture {
        puffin_path,
        num_vectors: NUM_VECTORS,
        dim: DIM,
        m: M,
        ef_construct: EF_CONSTRUCT,
        quantized_vec_size,
        file_paths,
        _tmp: tmp,
    }
}

// ---------- Reader helpers (§3.3 footer + §6.1 validation) ----------------

/// Parsed representation of the footer's blob descriptor array. Order preserved.
#[derive(Debug)]
pub(super) struct ParsedFooter {
    pub blobs: Vec<ParsedBlobDescriptor>,
    pub footer_size: usize,
}

#[derive(Debug, Clone)]
pub(super) struct ParsedBlobDescriptor {
    pub blob_type: String,
    pub offset: usize,
    pub length: usize,
    pub compression: String,
    pub properties: serde_json::Value,
}

/// Locate + validate a `.puffin` footer per §3.3 and §6.1.
///
/// - §3.3: reads the trailing 12 bytes (footer_size + flags + magic), then reads
///   exactly `footer_size` bytes.
/// - §6.1 corruption invariant: rejects footers whose blob byte ranges overlap
///   or exceed the file length.
///
/// This validator is **ann-hnsw-specific** (stricter than generic Puffin): §3
/// requires every blob start to be 64-byte aligned, and this validator rejects
/// misalignment with a hard error. A generic Puffin reader would accept
/// unaligned blobs; conforming ann-hnsw writers/readers must not.
///
/// This is the seed of the eventual `PuffinReader` struct (§6.1) — kept as a
/// function so Phase 2 can exercise the corruption path without depending on
/// a struct that hasn't been designed yet.
pub(super) fn read_and_validate_footer(file_bytes: &[u8]) -> OperationResult<ParsedFooter> {
    let file_size = file_bytes.len();
    if file_size < TRAILER_LEN + 4 {
        return Err(OperationError::service_error(format!(
            "file too small to contain a Puffin trailer: {file_size} bytes",
        )));
    }
    // Trailing magic.
    if &file_bytes[file_size - 4..file_size] != MAGIC {
        return Err(OperationError::service_error(
            "missing trailing PFA1 magic".to_string(),
        ));
    }
    // Flags word at (file_size - 8 .. file_size - 4). Real Puffin uses bit 0
    // for LZ4 footer compression; we do not support flagged/compressed footers
    // in v1. Reject non-zero here rather than let a corrupt/compressed footer
    // fall through to serde_json::from_slice and fail with a misleading JSON
    // parse error.
    let flags_bytes: [u8; 4] = file_bytes[file_size - 8..file_size - 4]
        .try_into()
        .expect("4-byte slice");
    let flags = u32::from_le_bytes(flags_bytes);
    if flags != 0 {
        return Err(OperationError::service_error(format!(
            "flagged/compressed footers not supported (flags={flags:#010x})",
        )));
    }
    // footer_size u32 LE at offset (file_size - 12).
    let footer_size_bytes: [u8; 4] = file_bytes[file_size - 12..file_size - 8]
        .try_into()
        .expect("4-byte slice");
    let footer_size = u32::from_le_bytes(footer_size_bytes) as usize;
    if footer_size + TRAILER_LEN > file_size {
        return Err(OperationError::service_error(format!(
            "declared footer_size ({footer_size}) exceeds file bounds ({file_size})",
        )));
    }
    let footer_start = file_size - TRAILER_LEN - footer_size;
    let footer_bytes = &file_bytes[footer_start..file_size - TRAILER_LEN];

    let footer: serde_json::Value = serde_json::from_slice(footer_bytes).map_err(|e| {
        OperationError::service_error(format!("footer JSON parse failed: {e}"))
    })?;
    let blob_arr = footer["blobs"].as_array().ok_or_else(|| {
        OperationError::service_error("footer missing `blobs` array".to_string())
    })?;

    let mut blobs: Vec<ParsedBlobDescriptor> = Vec::with_capacity(blob_arr.len());
    for (i, b) in blob_arr.iter().enumerate() {
        let blob_type = b["type"].as_str().ok_or_else(|| {
            OperationError::service_error(format!("blob {i} missing `type`"))
        })?.to_string();
        let offset = b["offset"].as_u64().ok_or_else(|| {
            OperationError::service_error(format!("blob {i} missing `offset`"))
        })? as usize;
        let length = b["length"].as_u64().ok_or_else(|| {
            OperationError::service_error(format!("blob {i} missing `length`"))
        })? as usize;
        // §3.2 makes `compression` load-bearing: absence is malformed, and only
        // the {"none","zstd"} set is defined. Reject anything else — a
        // conforming reader must not silently accept `lz4` or any unknown
        // codec that would leave bytes undecodable at load time.
        let compression = b["compression"]
            .as_str()
            .ok_or_else(|| {
                OperationError::service_error(format!(
                    "blob {i} ({blob_type}) missing `compression` field",
                ))
            })?
            .to_string();
        if compression != "none" && compression != "zstd" {
            return Err(OperationError::service_error(format!(
                "blob {i} ({blob_type}) has unsupported compression {compression:?} (expected one of \"none\", \"zstd\")",
            )));
        }

        // §6.1 reject-on-corruption: bounds + arithmetic overflow check.
        let end = offset.checked_add(length).ok_or_else(|| {
            OperationError::service_error(format!(
                "blob {i} ({blob_type}) range overflows usize: offset={offset}, length={length}",
            ))
        })?;
        if end > file_size {
            return Err(OperationError::service_error(format!(
                "blob {i} ({blob_type}) range [{offset}..{end}] exceeds file bounds ({file_size})",
            )));
        }
        // 64-byte alignment (§3 hard invariant, ann-hnsw-specific).
        if !offset.is_multiple_of(BLOB_ALIGN) {
            return Err(OperationError::service_error(format!(
                "blob {i} ({blob_type}) offset {offset} is not 64-byte aligned",
            )));
        }

        blobs.push(ParsedBlobDescriptor {
            blob_type,
            offset,
            length,
            compression,
            properties: b["properties"].clone(),
        });
    }

    // Non-overlap check: sort by offset then verify each blob ends at-or-before
    // the next one's start. Blobs in the footer are already ordered by writer
    // convention, but do not assume that when validating an untrusted footer.
    let mut ordered: Vec<(usize, usize)> =
        blobs.iter().map(|b| (b.offset, b.length)).collect();
    ordered.sort_by_key(|(o, _)| *o);
    for pair in ordered.windows(2) {
        if pair[0].0 + pair[0].1 > pair[1].0 {
            return Err(OperationError::service_error(format!(
                "overlapping blob ranges: [{}..{}] and [{}..{}]",
                pair[0].0,
                pair[0].0 + pair[0].1,
                pair[1].0,
                pair[1].0 + pair[1].1,
            )));
        }
    }

    Ok(ParsedFooter { blobs, footer_size })
}

impl ParsedFooter {
    pub fn by_type(&self, ty: &str) -> Option<&ParsedBlobDescriptor> {
        self.blobs.iter().find(|b| b.blob_type == ty)
    }
}

/// Memory-map an entire test file. Wraps the standard-mmap-over-owned-tempfile
/// pattern both Phase 2 and Phase 3 use so the unsafe live in one place.
pub(super) fn mmap_whole_file(path: &Path) -> Arc<Mmap> {
    let file = fs::File::open(path).unwrap();
    // SAFETY: standard mmap over a test-owned temp file that is not
    // concurrently truncated or written for the duration of the test.
    let mmap = unsafe { MmapOptions::new().map(file.file()) }.unwrap();
    Arc::new(mmap)
}

// ---------- Real-data fixture (§5 Phase 3, v3.12) --------------------------
//
// The random-vector fixture above stays as-is for Phase 1 & Phase 2 structural
// tests. Phase 3 pulls a real-data body + held-out queries out of
// `data/fixtures/gte_100k_vectors.{bin,json}` when present; the .bin is git-
// excluded, so environments without it fall back to random data (report-only).

use crate::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;
use crate::vector_storage::VectorStorage;
use crate::vector_storage::VectorStorageEnum;
use crate::data_types::vectors::{VectorElementType, VectorRef};
use common::bitvec::BitVec;
use common::counter::hardware_counter::HardwareCounterCell;
use crate::index::hnsw_index::point_scorer::FilteredScorer;
use crate::data_types::vectors::QueryVector;

/// Path (repo-relative) to the sampler-produced fixture body.
pub(super) const REAL_BODY_BIN: &str = "data/fixtures/gte_100k_vectors.bin";
pub(super) const REAL_BODY_JSON: &str = "data/fixtures/gte_100k_vectors.json";
pub(super) const REAL_QUERIES_BIN: &str = "data/fixtures/gte_20_queries.bin";
pub(super) const REAL_QUERIES_JSON: &str = "data/fixtures/gte_20_queries.json";

/// v3.13 Phase-4c full-scale (~1M) fixture paths. The 100k fixture stays
/// the default; loading the 1M fixture is a one-shot opt-in.
pub(super) const REAL_1M_BODY_BIN: &str = "data/fixtures/gte_1m_vectors.bin";
pub(super) const REAL_1M_BODY_JSON: &str = "data/fixtures/gte_1m_vectors.json";
pub(super) const REAL_1M_QUERIES_BIN: &str = "data/fixtures/gte_1m_queries.bin";
pub(super) const REAL_1M_QUERIES_JSON: &str = "data/fixtures/gte_1m_queries.json";

pub(super) fn repo_root() -> PathBuf {
    // segment/Cargo.toml is at lib/segment; CARGO_MANIFEST_DIR points there.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join("..").join("..")
}

/// Read a .bin of raw LE f32 into `count` rows of `dim` values.
/// Returns None if the file is missing.
fn read_bin_vectors(path: &Path, expected_count: usize, dim: usize) -> Option<Vec<Vec<f32>>> {
    if !path.exists() {
        return None;
    }
    let bytes = fs::read(path).expect("read bin");
    let expected_bytes = expected_count * dim * 4;
    if bytes.len() < expected_bytes {
        panic!(
            "bin at {} is {} bytes; expected at least {} (count={}, dim={})",
            path.display(),
            bytes.len(),
            expected_bytes,
            expected_count,
            dim,
        );
    }
    let mut out = Vec::with_capacity(expected_count);
    for i in 0..expected_count {
        let mut v = Vec::with_capacity(dim);
        for j in 0..dim {
            let off = (i * dim + j) * 4;
            let arr: [u8; 4] = bytes[off..off + 4].try_into().unwrap();
            v.push(f32::from_le_bytes(arr));
        }
        out.push(v);
    }
    Some(out)
}

/// Sidecar contents the loader validates before returning vectors.
/// `created` and `notes` are read for logging/diagnostics; the compiler can't
/// see the print sites through serde so the fields warn as dead — allow.
#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
pub(super) struct RealDataSidecar {
    pub count: usize,
    pub dim: usize,
    pub source_file: String,
    pub sample_seed: u64,
    pub normalized: bool,
    pub created: String,
    #[serde(default)]
    pub notes: String,
    /// Parquet row-group boundaries. Present in v3.13+ body sidecars; absent
    /// on the query sidecar (queries aren't row-mapped). Sum equals the
    /// source parquet's total row count.
    #[serde(default)]
    pub row_group_sizes: Vec<u64>,
    /// Repo-relative path to the raw LE-u64 file mapping body-index → source
    /// parquet row. Present in v3.13+ body sidecars.
    #[serde(default)]
    pub source_indices_file: Option<String>,
}

fn read_sidecar(path: &Path) -> Option<RealDataSidecar> {
    if !path.exists() {
        return None;
    }
    let bytes = fs::read(path).expect("read sidecar");
    Some(serde_json::from_slice(&bytes).expect("parse sidecar json"))
}

/// Loaded real-data pair, ready for a Phase 3 test run. v3.13 adds
/// `body_source_indices` — for each `body[i]`, the parquet row it came from.
#[allow(dead_code)] // `query_sidecar` reserved for future use / diagnostics
pub(super) struct RealDataFixture {
    pub body: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub body_sidecar: RealDataSidecar,
    pub query_sidecar: RealDataSidecar,
    /// Source parquet row indices for `body[i]`, one u64 per body vector.
    /// `None` when the sidecar predates v3.13 or the indices file is absent.
    pub body_source_indices: Option<Vec<u64>>,
}

/// Logical file-path identifier we bake into row-pointer blobs. Neutral —
/// tests resolve it against env-configured bucket/endpoint at runtime.
pub(super) const LOGICAL_PARQUET_NAME: &str = "gte_product_embeddings.parquet";

/// Convert a source parquet row → `(row_group_index, row_offset_within_group)`
/// via prefix-sum over `row_group_sizes`. `row_group_sizes` must be the same
/// list the sampler emitted (i.e. sum equals total row count in the source
/// parquet).
pub(super) fn source_row_to_row_group(
    row_group_sizes: &[u64],
    source_row: u64,
) -> (u32, u32) {
    let mut cum: u64 = 0;
    for (i, size) in row_group_sizes.iter().enumerate() {
        if source_row < cum + size {
            let offset = source_row - cum;
            return (i as u32, offset as u32);
        }
        cum += size;
    }
    panic!(
        "source row {source_row} out of range (total {} across {} row groups)",
        cum,
        row_group_sizes.len(),
    );
}

/// Named-file group so callers can target different fixture sizes without
/// duplicating the loader body.
pub(super) struct RealDataFiles {
    pub body_bin: &'static str,
    pub body_json: &'static str,
    pub queries_bin: &'static str,
    pub queries_json: &'static str,
}

pub(super) const FIXTURE_100K: RealDataFiles = RealDataFiles {
    body_bin: REAL_BODY_BIN,
    body_json: REAL_BODY_JSON,
    queries_bin: REAL_QUERIES_BIN,
    queries_json: REAL_QUERIES_JSON,
};

pub(super) const FIXTURE_1M: RealDataFiles = RealDataFiles {
    body_bin: REAL_1M_BODY_BIN,
    body_json: REAL_1M_BODY_JSON,
    queries_bin: REAL_1M_QUERIES_BIN,
    queries_json: REAL_1M_QUERIES_JSON,
};

/// Attempt to load the real-data body + queries from the default 100k
/// fixture. Convenience wrapper for [`try_load_real_data_from`] preserved for
/// existing call sites; new callers should reach for
/// [`try_load_real_data_from`] with an explicit [`RealDataFiles`].
pub(super) fn try_load_real_data(num_body: usize) -> Option<RealDataFixture> {
    try_load_real_data_from(&FIXTURE_100K, num_body)
}

/// Attempt to load the real-data body + queries. Returns None if the .bin is
/// absent (so environments without the fixture stay green). Fails loudly on
/// dim/count mismatches from the sidecar — a silent shape drift would mask a
/// container-consistency bug (see §3.4).
pub(super) fn try_load_real_data_from(
    files: &RealDataFiles,
    num_body: usize,
) -> Option<RealDataFixture> {
    let root = repo_root();
    let body_json = read_sidecar(&root.join(files.body_json))?;
    let body_bin_path = root.join(files.body_bin);
    let query_json = read_sidecar(&root.join(files.queries_json))?;
    let query_bin_path = root.join(files.queries_bin);

    assert_eq!(
        body_json.dim, DIM,
        "sidecar dim {} != fixture DIM {}",
        body_json.dim, DIM,
    );
    assert_eq!(
        query_json.dim, DIM,
        "query sidecar dim {} != fixture DIM {}",
        query_json.dim, DIM,
    );
    assert!(
        body_json.normalized,
        "sidecar reports body vectors NOT L2-normalized — decision required before choosing distance",
    );
    assert!(
        query_json.normalized,
        "sidecar reports query vectors NOT L2-normalized",
    );
    assert!(
        body_json.count >= num_body,
        "requested {num_body} body vectors but sidecar reports only {}",
        body_json.count,
    );

    let body = read_bin_vectors(&body_bin_path, num_body, DIM)?;
    let queries = read_bin_vectors(&query_bin_path, query_json.count, DIM)?;

    // v3.13: source-indices are optional (older sidecars won't have them);
    // when present, load and truncate to num_body since the body is a strict
    // prefix of the full 100k sample by contract.
    let body_source_indices = body_json.source_indices_file.as_ref().and_then(|rel| {
        let path = root.join(rel);
        if !path.exists() {
            return None;
        }
        let bytes = fs::read(&path).expect("read source_indices");
        // File is 100k × u64 = 800_000 bytes; may be longer than num_body needs.
        let full_count = bytes.len() / 8;
        if full_count < num_body {
            panic!(
                "source_indices file has {full_count} entries; need at least {num_body}",
            );
        }
        let mut out = Vec::with_capacity(num_body);
        for i in 0..num_body {
            let off = i * 8;
            out.push(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
        }
        Some(out)
    });

    Some(RealDataFixture {
        body,
        queries,
        body_sidecar: body_json,
        query_sidecar: query_json,
        body_source_indices,
    })
}

/// Scaffold that owns a `VectorStorageEnum` over externally-supplied vectors
/// and exposes the same `scorer`/`internal_scorer` surface as
/// `TestRawScorerProducer` (which insists on generating its own random vectors,
/// so cannot be used with real data).
pub(super) struct RealDataScaffold {
    storage: VectorStorageEnum,
    deleted: BitVec,
}

impl RealDataScaffold {
    pub fn new(distance: Distance, vectors: &[Vec<f32>]) -> Self {
        let mut storage = new_volatile_dense_vector_storage(DIM, distance);
        let hw = HardwareCounterCell::new();
        for (i, v) in vectors.iter().enumerate() {
            let v = distance.preprocess_vector::<VectorElementType>(v.clone());
            storage
                .insert_vector(i as PointOffsetType, VectorRef::from(&v), &hw)
                .expect("insert_vector");
        }
        let deleted = BitVec::repeat(false, vectors.len());
        Self { storage, deleted }
    }

    pub fn scorer(&self, query: impl Into<QueryVector>) -> FilteredScorer<'_> {
        FilteredScorer::new(
            query.into(),
            &self.storage,
            None::<&crate::vector_storage::quantized::quantized_vectors::QuantizedVectors>,
            None,
            &self.deleted,
            HardwareCounterCell::new(),
        )
        .expect("FilteredScorer::new")
    }

    pub fn internal_scorer(&self, point_id: PointOffsetType) -> FilteredScorer<'_> {
        FilteredScorer::new_internal(
            point_id,
            &self.storage,
            None::<&crate::vector_storage::quantized::quantized_vectors::QuantizedVectors>,
            None,
            &self.deleted,
            HardwareCounterCell::new(),
        )
        .expect("FilteredScorer::new_internal")
    }
}

// ---------- Phase 4d: quantized-scored scaffold (production-shaped hot path) --
//
// Sidestep pattern (matches Phase 3 recall test's `RealDataScaffold`):
// FilteredScorer::new (point_scorer.rs:118) demands a &VectorStorageEnum even
// when the caller wants quantized-only scoring. §6.2 tracks the disaggregated
// FilteredScorer story as production work; here we build both a
// VectorStorageEnum and a QuantizedVectors from the supplied vectors so a
// production-shaped encoder (variant selected via `PUFFIN_QUANT`, see
// [`QuantVariant`]; default 1-bit binary) drives the scoring RawScorer. Graph traversal uses the QuantizedVectors' raw_scorer
// (point_scorer.rs:127), i.e. the production hot path in disaggregated
// intent. Full-precision storage stays alongside only to satisfy the
// FilteredScorer signature — it is NOT touched during scoring.

use crate::types::{
    BinaryQuantizationConfig, BinaryQuantizationEncoding, Memory, QuantizationConfig,
    TurboQuantBitSize, TurboQuantQuantizationConfig, TurboQuantization,
};
use crate::vector_storage::quantized::quantized_vectors::{
    QuantizedVectors, QuantizedVectorsStorageType,
};

/// Quantization variant driving the quantized-scored scaffold, selected via
/// the `PUFFIN_QUANT` env var (same opt-in pattern as `PUFFIN_FIXTURE_N`):
/// `bq` (default — preserves all previously-committed numbers), `tq4`,
/// `tq2`, `tq1_5`, `tq1`. Both families go through the production
/// `QuantizedVectors::create` path, so a run compares codecs, not harnesses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QuantVariant {
    Bq,
    Tq(TurboQuantBitSize),
}

impl QuantVariant {
    pub fn from_env() -> Self {
        match std::env::var("PUFFIN_QUANT").as_deref() {
            Err(_) | Ok("") => Self::Bq,
            Ok(code) => Self::from_code(code).unwrap_or_else(|| {
                panic!("PUFFIN_QUANT={code:?} not recognised (expected: bq, tq4, tq2, tq1_5, tq1)")
            }),
        }
    }

    /// Parse the short variant code — the token `code()` emits and Phase-5
    /// containers store under the `quantization_variant` blob property, making
    /// a container self-describing to a reader that has no env context.
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "bq" => Some(Self::Bq),
            "tq4" => Some(Self::Tq(TurboQuantBitSize::Bits4)),
            "tq2" => Some(Self::Tq(TurboQuantBitSize::Bits2)),
            "tq1_5" => Some(Self::Tq(TurboQuantBitSize::Bits1_5)),
            "tq1" => Some(Self::Tq(TurboQuantBitSize::Bits1)),
            _ => None,
        }
    }

    /// Short machine-readable code; always the first token of `label()`.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Bq => "bq",
            Self::Tq(TurboQuantBitSize::Bits4) => "tq4",
            Self::Tq(TurboQuantBitSize::Bits2) => "tq2",
            Self::Tq(TurboQuantBitSize::Bits1_5) => "tq1_5",
            Self::Tq(TurboQuantBitSize::Bits1) => "tq1",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Bq => "bq (binary 1-bit, EncodedVectorsBin<u128>)",
            Self::Tq(TurboQuantBitSize::Bits4) => "tq4 (TurboQuant 4-bit)",
            Self::Tq(TurboQuantBitSize::Bits2) => "tq2 (TurboQuant 2-bit)",
            Self::Tq(TurboQuantBitSize::Bits1_5) => "tq1_5 (TurboQuant 1.5-bit)",
            Self::Tq(TurboQuantBitSize::Bits1) => "tq1 (TurboQuant 1-bit)",
        }
    }

    // `always_ram` is deprecated in favor of `memory` (1.19.0) but the config
    // structs derive no `Default`, so the field must still be named. Upstream's
    // own tests take a file-wide `allow(deprecated)`; scope it to this fn.
    #[expect(deprecated)]
    pub(super) fn config(&self) -> QuantizationConfig {
        match self {
            // `memory: Pinned` is what the deprecated `always_ram: true` now
            // resolves to (see `legacy_always_ram_placement`): quantized
            // vectors resident in RAM and never evicted, which is what these
            // measurements assume.
            Self::Bq => BinaryQuantizationConfig {
                always_ram: None,
                memory: Some(Memory::Pinned),
                encoding: Some(BinaryQuantizationEncoding::OneBit),
                query_encoding: None,
            }
            .into(),
            Self::Tq(bits) => QuantizationConfig::Turbo(TurboQuantization {
                turbo: TurboQuantQuantizationConfig {
                    always_ram: None,
                    memory: Some(Memory::Pinned),
                    bits: Some(*bits),
                },
            }),
        }
    }
}

pub(super) struct QuantizedRealDataScaffold {
    storage: VectorStorageEnum,
    quantized: QuantizedVectors,
    deleted: BitVec,
    _tmp: TempDir,
}

impl QuantizedRealDataScaffold {
    pub fn new(vectors: &[Vec<f32>], variant: QuantVariant) -> Self {
        let mut storage = new_volatile_dense_vector_storage(DIM, Distance::Dot);
        let hw = HardwareCounterCell::new();
        for (i, v) in vectors.iter().enumerate() {
            let v = Distance::Dot.preprocess_vector::<VectorElementType>(v.clone());
            storage
                .insert_vector(i as PointOffsetType, VectorRef::from(&v), &hw)
                .expect("insert_vector");
        }
        let tmp = TempDir::new().unwrap();
        let config = variant.config();
        let quantized = QuantizedVectors::create(
            &storage,
            &config,
            QuantizedVectorsStorageType::Immutable,
            tmp.path(),
            /* max_threads */ 1,
            &AtomicBool::new(false),
        )
        .unwrap_or_else(|e| panic!("QuantizedVectors::create ({}): {e}", variant.label()));
        let deleted = BitVec::repeat(false, vectors.len());
        Self {
            storage,
            quantized,
            deleted,
            _tmp: tmp,
        }
    }

    pub fn scorer(&self, query: impl Into<QueryVector>) -> FilteredScorer<'_> {
        FilteredScorer::new(
            query.into(),
            &self.storage,
            Some(&self.quantized), // ← quantized RawScorer path
            None,
            &self.deleted,
            HardwareCounterCell::new(),
        )
        .expect("FilteredScorer::new (quantized)")
    }
}

/// Optional source-row mapping for the row-pointer blob. When supplied, the
/// container's row-pointer entries reference real parquet rows so a rerank
/// step can locate the right bytes; when absent, row-pointer entries fall
/// back to the mocked `(i, 0, 0, i)` shape Phase 1/2 relied on.
pub(super) struct SourceRowMapping<'a> {
    /// Per-vector source parquet row.
    pub source_rows: &'a [u64],
    /// Parquet row-group sizes for prefix-sum → (row_group, row_offset).
    pub row_group_sizes: &'a [u64],
    /// Neutral logical file-path stored in the row-pointer path table.
    pub logical_file_name: &'a str,
}

/// Build a `.puffin` container over an externally-supplied vector body. Used
/// by Phase 3 to build a container over 10k (or 100k) real vectors — same
/// blob layout, quantization, and row-pointer shape as the random fixture.
///
/// Determinism: the graph's level assignments are drawn from `rng`. Pass a
/// seeded RNG to make repeated runs identical.
pub(super) fn build_test_puffin_fixture_from_vectors(
    body: &[Vec<f32>],
    scaffold: &RealDataScaffold,
    rng: &mut StdRng,
) -> TestFixture {
    build_test_puffin_fixture_from_vectors_with_mapping(body, scaffold, rng, None)
}

/// Variant that takes an optional [`SourceRowMapping`] and emits
/// row-pointer entries pointing at real parquet rows. Callers that don't
/// need row-mapping fidelity use [`build_test_puffin_fixture_from_vectors`].
pub(super) fn build_test_puffin_fixture_from_vectors_with_mapping(
    body: &[Vec<f32>],
    scaffold: &RealDataScaffold,
    rng: &mut StdRng,
    mapping: Option<SourceRowMapping<'_>>,
) -> TestFixture {
    let num_vectors = body.len();
    let tmp = TempDir::new().unwrap();
    let puffin_path = tmp.path().join("test_index.puffin");
    let graph_dir = tmp.path().join("graph");
    fs::create_dir_all(&graph_dir).unwrap();

    // Build HNSW.
    let mut builder = GraphLayersBuilder::new(
        num_vectors,
        HnswM::new2(M),
        EF_CONSTRUCT,
        ENTRY_POINTS_NUM,
        /* use_heuristic */ true,
    );
    for idx in 0..num_vectors as PointOffsetType {
        let level = builder.get_random_layer(rng);
        builder.set_levels(idx, level);
        builder.link_new_point(idx, scaffold.internal_scorer(idx));
    }
    builder
        .into_graph_layers(
            &graph_dir,
            GraphLinksFormatParam::Compressed,
            /* on_disk */ false,
        )
        .unwrap();
    let graph_meta_bytes = fs::read(graph_dir.join("graph.bin")).unwrap();
    let graph_links_bytes = fs::read(graph_dir.join("links_compressed.bin")).unwrap();

    // Quantize the SAME body — §3.4 (v3.12) container-consistency invariant.
    let vector_parameters = VectorParameters {
        dim: DIM,
        distance_type: DistanceType::Dot,
        invert: false,
        deprecated_count: None,
    };
    let quantized_vec_size = get_quantized_vector_size_from_params::<u128>(DIM, Encoding::OneBit);
    let quant_data_path = tmp.path().join("quant.bin");
    let quant_meta_path = tmp.path().join("quant.meta.json");
    let storage_builder =
        TestEncodedStorageBuilder::new(Some(&quant_data_path), quantized_vec_size);
    let _encoded = EncodedVectorsBin::<u128, _>::encode(
        body.iter().map(|v| v.as_slice()),
        storage_builder,
        &vector_parameters,
        Encoding::OneBit,
        QueryEncoding::SameAsStorage,
        Some(&quant_meta_path),
        &AtomicBool::new(false),
    )
    .expect("EncodedVectorsBin::encode");
    let quant_bytes = fs::read(&quant_data_path).unwrap();
    let quant_meta_bytes = fs::read(&quant_meta_path).unwrap();

    // Row-pointers. When a SourceRowMapping is supplied, entries reference
    // real parquet rows via prefix-sum over row_group_sizes; otherwise fall
    // back to the (i, 0, 0, i) mock shape that Phase 1/2 tests rely on for
    // structural checks.
    type RpEntry = (u32, u32, u32, u32);
    let (file_paths, entries): (Vec<String>, Vec<RpEntry>) = match &mapping {
        Some(m) => {
            assert_eq!(
                m.source_rows.len(),
                num_vectors,
                "SourceRowMapping.source_rows length {} != body {num_vectors}",
                m.source_rows.len(),
            );
            let entries: Vec<(u32, u32, u32, u32)> = m
                .source_rows
                .iter()
                .enumerate()
                .map(|(i, &src_row)| {
                    let (rg, off) = source_row_to_row_group(m.row_group_sizes, src_row);
                    (i as u32, 0u32, rg, off)
                })
                .collect();
            (vec![m.logical_file_name.to_string()], entries)
        }
        None => (
            vec!["mock://parquet/dataset/gte.parquet".to_string()],
            (0..num_vectors as u32).map(|i| (i, 0, 0, i)).collect(),
        ),
    };
    let path_refs: Vec<&str> = file_paths.iter().map(|s| s.as_str()).collect();
    let row_ptr_bytes = build_row_pointer_blob(&path_refs, &entries);

    let blobs = [
        BlobSpec {
            blob_type: "ann-hnsw-quantized-vectors-v1",
            bytes: &quant_bytes,
            properties: json!({
                "quantization_variant": "EncodedVectorsBin_u128",
                "quantization_family": "binary",
                "dimensions": DIM.to_string(),
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
        BlobSpec {
            blob_type: "ann-hnsw-quantized-meta-v1",
            bytes: &quant_meta_bytes,
            properties: json!({
                "quantization_family": "binary",
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
        BlobSpec {
            blob_type: "ann-hnsw-graph-meta-v1",
            bytes: &graph_meta_bytes,
            properties: json!({
                "m": M.to_string(),
                "ef_construct": EF_CONSTRUCT.to_string(),
                "vector_count": num_vectors.to_string(),
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
        BlobSpec {
            blob_type: "ann-hnsw-graph-links-v1",
            bytes: &graph_links_bytes,
            properties: json!({
                "format": "compressed",
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
        BlobSpec {
            blob_type: "ann-hnsw-row-pointers-v1",
            bytes: &row_ptr_bytes,
            properties: json!({
                "entry_count": num_vectors.to_string(),
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
    ];
    write_puffin(&puffin_path, &blobs).expect("write_puffin");

    TestFixture {
        puffin_path,
        num_vectors,
        dim: DIM,
        m: M,
        ef_construct: EF_CONSTRUCT,
        quantized_vec_size,
        file_paths,
        _tmp: tmp,
    }
}

// ---------- Phase 4 (v3.13): object-store fetch + snapshot-keyed cache -----
//
// Test-scope seed of the §6.1 `PuffinReader` surface. All items live here so
// production `graph_layers` / `graph_links` stay untouched. When PuffinReader
// becomes a real production abstraction, this fetcher lifts into `io_bridge_
// object_store` on top of `AsyncRead::read_range`; for the test we cut
// through `object_store` directly to keep the surface small.

use std::sync::Mutex;
use std::time::Instant;

use object_store::{GetOptions, GetRange, ObjectStore, ObjectStoreExt};

/// Book-keeping the tests use to verify hit/miss behaviour without depending
/// on the store implementation's internals. Only the count of `store.get`
/// and `store.head` calls is exposed; the fetcher increments the counters
/// itself (not through an ObjectStore-wrapping proxy) so semantics don't
/// depend on a particular wrapper.
#[derive(Clone, Debug, Default)]
pub(super) struct FetcherStats {
    pub gets: usize,
    pub heads: usize,
    pub post_cache_verifies: usize,
}

/// Test-scope PuffinFetcher: fetches an object from an `ObjectStore`, validates
/// its footer via `read_and_validate_footer` BEFORE committing the cache
/// entry, then atomically renames a `.tmp` into place. Directory is keyed by
/// snapshot_id per §2.4.
pub(super) struct PuffinFetcher {
    store: Arc<dyn ObjectStore>,
    cache_dir: PathBuf,
    runtime: Arc<tokio::runtime::Runtime>,
    stats: Arc<Mutex<FetcherStats>>,
}

impl PuffinFetcher {
    pub fn new(store: Arc<dyn ObjectStore>, cache_dir: PathBuf) -> OperationResult<Self> {
        fs::create_dir_all(&cache_dir).map_err(|e| {
            OperationError::service_error(format!("create cache_dir: {e}"))
        })?;
        let runtime = tokio::runtime::Runtime::new().map_err(|e| {
            OperationError::service_error(format!("start tokio runtime: {e}"))
        })?;
        Ok(Self {
            store,
            cache_dir,
            runtime: Arc::new(runtime),
            stats: Arc::new(Mutex::new(FetcherStats::default())),
        })
    }

    pub fn stats(&self) -> FetcherStats {
        self.stats.lock().unwrap().clone()
    }

    /// Path where this (path, snapshot_id) pair caches.
    pub fn cached_path(&self, remote_path: &str, snapshot_id: u64) -> PathBuf {
        let basename = std::path::Path::new(remote_path)
            .file_name()
            .map(|s| s.to_os_string())
            .unwrap_or_else(|| std::ffi::OsString::from("index.puffin"));
        self.cache_dir.join(snapshot_id.to_string()).join(basename)
    }

    /// Fetch `remote_path` from the store into a local cache file keyed by
    /// `snapshot_id`. On a hit, returns the local path with zero store access.
    /// On a miss, downloads, validates, and post-verifies before returning.
    pub fn fetch(
        &self,
        remote_path: &str,
        snapshot_id: u64,
    ) -> OperationResult<PathBuf> {
        let final_path = self.cached_path(remote_path, snapshot_id);
        if final_path.exists() {
            return Ok(final_path);
        }
        let snapshot_dir = final_path.parent().expect("cached_path has parent");
        fs::create_dir_all(snapshot_dir).map_err(|e| {
            OperationError::service_error(format!("create snapshot dir: {e}"))
        })?;
        let tmp_path = final_path.with_extension("tmp");
        // Ensure any leftover .tmp from a prior aborted run is gone.
        let _ = fs::remove_file(&tmp_path);

        // The download/validate happens in a helper so the ? early-return
        // still cleans up the .tmp if any step fails.
        let outcome = self.miss_path(remote_path, &tmp_path);
        match outcome {
            Ok(()) => {
                fs::rename(&tmp_path, &final_path).map_err(|e| {
                    OperationError::service_error(format!("atomic rename: {e}"))
                })?;
                // Post-cache verify (v3.13): validate what we'll mmap, not
                // just what we streamed.
                let bytes = fs::read(&final_path).map_err(|e| {
                    OperationError::service_error(format!("post-verify read: {e}"))
                })?;
                read_and_validate_footer(&bytes).map_err(|e| {
                    // Nuke the poisoned entry — a subsequent fetch retries.
                    let _ = fs::remove_file(&final_path);
                    OperationError::service_error(format!(
                        "post-cache validation failed (cache entry removed): {e}"
                    ))
                })?;
                self.stats.lock().unwrap().post_cache_verifies += 1;
                Ok(final_path)
            }
            Err(err) => {
                // Corrupted download / tiny object / missing key: leave no
                // .tmp behind so the cache directory contains only committed
                // entries. Best-effort remove.
                let _ = fs::remove_file(&tmp_path);
                Err(err)
            }
        }
    }

    /// Download + validate + write to .tmp. Does NOT rename. All error paths
    /// leave the .tmp in whatever state; the caller wipes it on Err.
    fn miss_path(&self, remote_path: &str, tmp_path: &Path) -> OperationResult<()> {
        let store = Arc::clone(&self.store);
        let stats = Arc::clone(&self.stats);
        let runtime = Arc::clone(&self.runtime);
        let key = object_store::path::Path::from(remote_path);

        // Step (a): trailer via Suffix(12). Tiny-object case must produce
        // "not a puffin container", not a byte-slicing panic.
        let trailer_bytes = runtime.block_on(async {
            stats.lock().unwrap().gets += 1;
            let opts = GetOptions {
                range: Some(GetRange::Suffix(TRAILER_LEN as u64)),
                ..Default::default()
            };
            let res = store.get_opts(&key, opts).await.map_err(|e| {
                OperationError::service_error(format!("store.get_opts(trailer): {e}"))
            })?;
            let bytes = res.bytes().await.map_err(|e| {
                OperationError::service_error(format!("read trailer bytes: {e}"))
            })?;
            Ok::<_, OperationError>(bytes)
        })?;
        if trailer_bytes.len() < TRAILER_LEN {
            return Err(OperationError::service_error(format!(
                "not a puffin container: object smaller than the {TRAILER_LEN}-byte trailer (got {} bytes)",
                trailer_bytes.len(),
            )));
        }
        // Trailer layout: [footer_size u32 LE | flags u32 LE | trailing magic]
        let footer_size = u32::from_le_bytes(
            trailer_bytes[0..4].try_into().unwrap(),
        ) as usize;
        // Step (c): footer + trailer in one shot — Suffix(12 + footer_size).
        let footer_and_trailer = runtime.block_on(async {
            stats.lock().unwrap().gets += 1;
            let opts = GetOptions {
                range: Some(GetRange::Suffix((TRAILER_LEN + footer_size) as u64)),
                ..Default::default()
            };
            let res = store.get_opts(&key, opts).await.map_err(|e| {
                OperationError::service_error(format!("store.get_opts(footer): {e}"))
            })?;
            let bytes = res.bytes().await.map_err(|e| {
                OperationError::service_error(format!("read footer bytes: {e}"))
            })?;
            Ok::<_, OperationError>(bytes)
        })?;
        if footer_and_trailer.len() < TRAILER_LEN + footer_size {
            return Err(OperationError::service_error(format!(
                "short footer read: got {} bytes, expected at least {}",
                footer_and_trailer.len(),
                TRAILER_LEN + footer_size,
            )));
        }

        // Step (d): validate the footer BEFORE any writes. The validator
        // parses trailer bytes at end-of-buffer, so we synthesise a
        // minimal-yet-realistic prefix.
        //
        // Sizing: leading magic + one 64-aligned body byte + zero-pad + magic
        // up to the footer's declared blob offsets. But `read_and_validate_
        // footer` derives everything from the trailer + JSON: `file_size` is
        // the buffer length. The declared blob offsets must fit within that
        // length. Simplest: fetch `head.size` first so we can size a buffer
        // of exactly `size` bytes with the tail bytes at the right offset.
        let size = runtime.block_on(async {
            stats.lock().unwrap().heads += 1;
            store.head(&key).await.map_err(|e| {
                OperationError::service_error(format!("store.head: {e}"))
            })
        })?.size as usize;
        if size < TRAILER_LEN + footer_size {
            return Err(OperationError::service_error(format!(
                "object size {size} smaller than footer+trailer ({})",
                TRAILER_LEN + footer_size,
            )));
        }
        let mut probe_buf = vec![0u8; size];
        let tail_start = size - (TRAILER_LEN + footer_size);
        probe_buf[tail_start..].copy_from_slice(&footer_and_trailer);
        // Leading magic so the validator's file-size checks feel realistic;
        // strictly speaking the validator only looks at the tail.
        probe_buf[0..4].copy_from_slice(MAGIC);
        read_and_validate_footer(&probe_buf).map_err(|e| {
            OperationError::service_error(format!("pre-cache footer validation: {e}"))
        })?;

        // Step (f): fetch the body only — Bounded(0..size - (12 + footer_size)).
        // We already have the tail; refetching it would risk torn-object reads.
        let body_end = size - (TRAILER_LEN + footer_size);
        let body_bytes = runtime.block_on(async {
            stats.lock().unwrap().gets += 1;
            let opts = GetOptions {
                range: Some(GetRange::Bounded(0..body_end as u64)),
                ..Default::default()
            };
            let res = store.get_opts(&key, opts).await.map_err(|e| {
                OperationError::service_error(format!("store.get_opts(body): {e}"))
            })?;
            let bytes = res.bytes().await.map_err(|e| {
                OperationError::service_error(format!("read body bytes: {e}"))
            })?;
            Ok::<_, OperationError>(bytes)
        })?;
        if body_bytes.len() != body_end {
            return Err(OperationError::service_error(format!(
                "short body read: got {}, expected {body_end}",
                body_bytes.len(),
            )));
        }

        // Step (g): write body + already-validated tail to .tmp. Single copy
        // of the tail bytes across the whole miss path.
        use std::io::Write;
        let mut out = fs::File::create(tmp_path).map_err(|e| {
            OperationError::service_error(format!("create .tmp: {e}"))
        })?;
        out.write_all(&body_bytes).map_err(|e| {
            OperationError::service_error(format!("write body: {e}"))
        })?;
        out.write_all(&footer_and_trailer).map_err(|e| {
            OperationError::service_error(format!("write tail: {e}"))
        })?;
        out.sync_all().map_err(|e| {
            OperationError::service_error(format!("sync .tmp: {e}"))
        })?;
        Ok(())
    }
}

/// Tiny stub for the Iceberg REST-catalog surface Phase 4 needs. Returns
/// the pair `(snapshot_id, statistics_file_uri)` — the actual response
/// shape a production reader consumes. Not a REST client.
#[derive(Clone, Debug)]
pub(super) struct MockCatalogStub {
    pub snapshot_id: u64,
    pub statistics_file_uri: String,
}

impl MockCatalogStub {
    pub fn new(snapshot_id: u64, uri: impl Into<String>) -> Self {
        Self { snapshot_id, statistics_file_uri: uri.into() }
    }

    pub fn snapshot_summary(&self) -> (u64, &str) {
        (self.snapshot_id, self.statistics_file_uri.as_str())
    }
}

/// Convenience: put a byte slice into an `ObjectStore` at `key` via a single
/// PUT. Only use for small objects — real S3-compatible endpoints reject
/// single-PUT payloads above ~5 GB, and object_store's client-side path may
/// abort even earlier. For anything large (≳ a few hundred MB) use
/// [`put_bytes_multipart`].
pub(super) fn put_bytes(
    runtime: &tokio::runtime::Runtime,
    store: &dyn ObjectStore,
    key: &str,
    bytes: Vec<u8>,
) -> OperationResult<()> {
    runtime.block_on(async {
        let path = object_store::path::Path::from(key);
        store.put(&path, bytes.into()).await.map_err(|e| {
            OperationError::service_error(format!("store.put({key}): {e}"))
        })?;
        Ok(())
    })
}

/// Upload `bytes` at `key` via multipart. Sized parts (default 16 MiB) satisfy
/// S3's 5 MiB minimum. Used for the 3.3 GB parquet in the Phase-4 rerank test,
/// where a single PUT is client-side-rejected.
pub(super) fn put_bytes_multipart(
    runtime: &tokio::runtime::Runtime,
    store: &dyn ObjectStore,
    key: &str,
    bytes: Vec<u8>,
) -> OperationResult<()> {
    const PART_SIZE: usize = 16 * 1024 * 1024; // 16 MiB
    runtime.block_on(async {
        let path = object_store::path::Path::from(key);
        let mut upload = store.put_multipart(&path).await.map_err(|e| {
            OperationError::service_error(format!("store.put_multipart({key}): {e}"))
        })?;
        for chunk in bytes.chunks(PART_SIZE) {
            upload
                .put_part(chunk.to_vec().into())
                .await
                .map_err(|e| {
                    OperationError::service_error(format!("multipart.put_part({key}): {e}"))
                })?;
        }
        upload.complete().await.map_err(|e| {
            OperationError::service_error(format!("multipart.complete({key}): {e}"))
        })?;
        Ok(())
    })
}

/// Measure a closure's elapsed wall-clock time.
pub(super) fn time_it<F: FnOnce() -> T, T>(f: F) -> (T, std::time::Duration) {
    let start = Instant::now();
    let out = f();
    (out, start.elapsed())
}
