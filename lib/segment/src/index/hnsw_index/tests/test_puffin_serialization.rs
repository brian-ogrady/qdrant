//! Phase 1 of the Puffin-HNSW PoC: write a valid `.puffin` container end-to-end and
//! assert its structural invariants (§3 layout, §5 Phase 1 of `docs/puffin_ann_spec.md`
//! v3.11). This test does NOT verify search correctness — that is Phase 2/3 territory.
//!
//! Blob order matches §3.1 v3.11:
//!   0. ann-hnsw-quantized-vectors-v1
//!   1. ann-hnsw-quantized-meta-v1
//!   2. ann-hnsw-graph-meta-v1        (raw graph.bin bytes)
//!   3. ann-hnsw-graph-links-v1       (raw links_compressed.bin bytes)
//!   4. ann-hnsw-row-pointers-v1
//!
//! All blobs are written `compression: none` for Phase 1 (zstd on graph-meta is deferred
//! to a reader-capable phase). Every blob starts at a 64-byte-aligned file offset. The
//! mid-magic is also aligned so the file-size formula is exact.

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

use crate::fixtures::index_fixtures::{TestRawScorerProducer, random_vector};
use crate::index::hnsw_index::HnswM;
use crate::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
use crate::index::hnsw_index::graph_links::GraphLinksFormatParam;
use crate::types::Distance;

const MAGIC: &[u8; 4] = b"PFA1";
const BLOB_ALIGN: usize = 64;
// The trailing footer trailer is [footer_size u32 LE | flags u32 LE | trailing magic 4B].
const TRAILER_LEN: usize = 4 + 4 + 4;

#[inline]
fn align_up(n: usize, to: usize) -> usize {
    debug_assert!(to.is_power_of_two());
    (n + to - 1) & !(to - 1)
}

struct BlobSpec<'a> {
    blob_type: &'static str,
    bytes: &'a [u8],
    properties: serde_json::Value,
}

#[derive(Clone, Copy, Debug)]
struct BlobRange {
    offset: usize,
    length: usize,
}

/// Serialize a `.puffin` container per §3.1 of the spec. Returns the recorded blob
/// ranges plus the mid-magic offset and footer JSON length so callers can construct
/// an exact file-size formula.
struct PuffinLayout {
    ranges: Vec<BlobRange>,
    mid_magic_offset: usize,
    footer_len: usize,
}

fn write_puffin(path: &std::path::Path, blobs: &[BlobSpec]) -> std::io::Result<PuffinLayout> {
    let mut buf: Vec<u8> = Vec::new();

    // Leading magic.
    buf.extend_from_slice(MAGIC);

    // Blob region. Pad before each blob to align its start to 64 B.
    let mut ranges = Vec::with_capacity(blobs.len());
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

    // Pad after last blob so mid-magic sits on a 64 B boundary. Spec §3.1 does not
    // strictly require this, but it keeps the file-size formula symmetrical and gives
    // the mid-magic a predictable alignment for any range-reader implementation.
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
    Ok(PuffinLayout {
        ranges,
        mid_magic_offset,
        footer_len: footer_bytes.len(),
    })
}

/// Build the `ann-hnsw-row-pointers-v1` blob body per §3.2 v3.11:
///   [u8 version=1] [u32 LE entry_count]
///   [u32 LE path_count] [{u32 LE len, UTF-8 bytes} × path_count]
///   [{u32 vec_id, u32 file_path_idx, u32 row_group, u32 row_offset} × entry_count]
/// All multi-byte integers little-endian. Entries pre-sorted ascending by vec_id.
fn build_row_pointer_blob(
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

#[test]
fn test_puffin_writer_writes_valid_container() {
    const NUM_VECTORS: usize = 10_000;
    const DIM: usize = 768;
    const M: usize = 16;
    const EF_CONSTRUCT: usize = 100;
    const ENTRY_POINTS_NUM: usize = 10;
    // Deterministic seed: "PFA1" bytes as a u64.
    const SEED: u64 = 0x5046_4131;

    let tmp = TempDir::new().unwrap();
    let puffin_path = tmp.path().join("test_index.puffin");
    let graph_dir = tmp.path().join("graph");
    fs::create_dir_all(&graph_dir).unwrap();

    // ---- 1. Build the HNSW graph ------------------------------------------------
    // Use the standard test scaffold for the scorer, but drive `link_new_point`
    // ourselves so we can set ef_construct=100 as §5 Phase 1 requires (the shared
    // fixture hardcodes ef=16, which is fine for speed tests but not the spec).
    let mut rng = StdRng::seed_from_u64(SEED);
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

    // ---- 2. Save graph to disk (yields graph.bin + links_compressed.bin) --------
    // on_disk=false keeps the build in RAM but still atomic_save's both files.
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

    // ---- 3. Quantize a matching-shape vector set (u128 binary) -----------------
    // Phase 1 tests file-format correctness, not graph-vs-quant scoring consistency,
    // so an independently seeded set of vectors is sufficient. Any 10k×768 f32
    // vectors would produce a structurally identical quantized blob.
    let mut rng2 = StdRng::seed_from_u64(SEED.wrapping_add(1));
    let vectors_for_quant: Vec<Vec<f32>> = (0..NUM_VECTORS)
        .map(|_| random_vector(&mut rng2, DIM))
        .collect();

    let vector_parameters = VectorParameters {
        dim: DIM,
        distance_type: DistanceType::Dot,
        invert: false,
        deprecated_count: None,
    };
    let quantized_vec_size =
        get_quantized_vector_size_from_params::<u128>(DIM, Encoding::OneBit);
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

    // ---- 4. Row-pointer blob ---------------------------------------------------
    let paths = ["mock://parquet/dataset/file.parquet"];
    let entries: Vec<(u32, u32, u32, u32)> =
        (0..NUM_VECTORS as u32).map(|i| (i, 0u32, 0u32, i)).collect();
    let row_ptr_bytes = build_row_pointer_blob(&paths, &entries);

    // ---- 5. Assemble the .puffin container ------------------------------------
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
    let layout = write_puffin(&puffin_path, &blobs).unwrap();

    // ---- 6. Structural assertions ---------------------------------------------
    let actual_size = fs::metadata(&puffin_path).unwrap().len() as usize;
    let file_bytes = fs::read(&puffin_path).unwrap();

    // (a) Leading magic.
    assert_eq!(&file_bytes[..4], MAGIC, "missing leading PFA1 magic");

    // (b) Every blob offset is 64-byte aligned, non-zero length, and within file bounds.
    for (i, r) in layout.ranges.iter().enumerate() {
        assert_eq!(
            r.offset % BLOB_ALIGN,
            0,
            "blob {i} ({}) offset {} not 64-aligned",
            blobs[i].blob_type,
            r.offset,
        );
        assert!(r.length > 0, "blob {i} ({}) is empty", blobs[i].blob_type);
        assert!(
            r.offset + r.length <= actual_size,
            "blob {i} ({}) exceeds file bounds",
            blobs[i].blob_type,
        );
    }

    // (c) Blob ranges are non-overlapping and in-order (writer invariant).
    for pair in layout.ranges.windows(2) {
        assert!(
            pair[0].offset + pair[0].length <= pair[1].offset,
            "blob ranges overlap: {pair:?}",
        );
    }

    // (d) Row-pointer blob has correct leading version byte.
    let rp_range = *layout.ranges.last().unwrap();
    assert_eq!(
        file_bytes[rp_range.offset],
        1u8,
        "row-pointer version byte must be 1",
    );

    // (e) Row-pointer entry_count matches.
    let rp_entry_count = u32::from_le_bytes(
        file_bytes[rp_range.offset + 1..rp_range.offset + 5]
            .try_into()
            .unwrap(),
    );
    assert_eq!(rp_entry_count as usize, NUM_VECTORS);

    // (f) Mid-magic present and 64-byte aligned.
    assert_eq!(layout.mid_magic_offset % BLOB_ALIGN, 0);
    assert_eq!(
        &file_bytes[layout.mid_magic_offset..layout.mid_magic_offset + 4],
        MAGIC,
        "missing mid PFA1 magic",
    );

    // (g) Trailing magic.
    assert_eq!(
        &file_bytes[actual_size - 4..actual_size],
        MAGIC,
        "missing trailing PFA1 magic",
    );

    // (h) Footer size at [actual - 12 .. actual - 8] and matches recorded value.
    let footer_size_from_trailer = u32::from_le_bytes(
        file_bytes[actual_size - 12..actual_size - 8]
            .try_into()
            .unwrap(),
    ) as usize;
    assert_eq!(
        footer_size_from_trailer, layout.footer_len,
        "footer_size trailer field disagrees with writer's recorded footer length",
    );

    // (i) Footer parses as JSON with the exact expected 5 blobs in order.
    let footer_start = actual_size - TRAILER_LEN - footer_size_from_trailer;
    let footer_bytes = &file_bytes[footer_start..actual_size - TRAILER_LEN];
    let footer: serde_json::Value = serde_json::from_slice(footer_bytes).unwrap();
    let arr = footer["blobs"].as_array().unwrap();
    let expected_types = [
        "ann-hnsw-quantized-vectors-v1",
        "ann-hnsw-quantized-meta-v1",
        "ann-hnsw-graph-meta-v1",
        "ann-hnsw-graph-links-v1",
        "ann-hnsw-row-pointers-v1",
    ];
    assert_eq!(arr.len(), expected_types.len(), "footer blob count");
    for (i, ty) in expected_types.iter().enumerate() {
        assert_eq!(arr[i]["type"].as_str().unwrap(), *ty);
        assert_eq!(arr[i]["compression"].as_str().unwrap(), "none");
        assert_eq!(
            arr[i]["offset"].as_u64().unwrap() as usize,
            layout.ranges[i].offset,
        );
        assert_eq!(
            arr[i]["length"].as_u64().unwrap() as usize,
            layout.ranges[i].length,
        );
    }

    // (j) Exact file-size formula:
    //     mid_magic_offset + 4 (mid magic) + footer_len + 4 (size) + 4 (flags) + 4 (trailing magic)
    let expected_size = layout.mid_magic_offset + 4 + layout.footer_len + TRAILER_LEN;
    assert_eq!(
        actual_size, expected_size,
        "file-size formula mismatch: expected {expected_size}, got {actual_size}",
    );
}
