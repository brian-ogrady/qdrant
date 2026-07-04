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
use std::sync::atomic::AtomicBool;

use common::types::PointOffsetType;
use fs_err as fs;
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

    // ---- Quantize a matching-shape vector set (u128 binary) -----------------
    // Phase 1 tests file-format correctness, not graph-vs-quant scoring
    // consistency, so an independently seeded set of vectors is sufficient.
    let mut rng2 = StdRng::seed_from_u64(FIXTURE_SEED.wrapping_add(1));
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
