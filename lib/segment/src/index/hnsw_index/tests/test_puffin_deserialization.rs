//! Phase 2 of the Puffin-HNSW PoC: read a `.puffin` container end-to-end using
//! the production reader path — including the new
//! `GraphLinks::load_from_ranged_mmap` API added for §6.3.
//!
//! Fixtures come from `puffin_shared::build_test_puffin_fixture()` (deterministic
//! seed shared with Phase 1). Reader-side validation goes through
//! `puffin_shared::read_and_validate_footer`, which is the seed of the eventual
//! `PuffinReader` struct (§6.1).
//!
//! Coverage:
//!   - Positive: end-to-end round-trip (footer parse → mmap → graph meta →
//!     ranged links → variant-dispatched quantized vectors → row pointers).
//!   - Negative 1: reader rejects a footer whose blob range exceeds file bounds
//!     (§6.1 corruption invariant).
//!   - Negative 2: `load_from_ranged_mmap` rejects a range whose arithmetic
//!     overflows `usize` (the strictest new production guard).

use std::sync::Arc;

use common::universal_io::MmapFs;
use fs_err as fs;
use quantization::encoded_storage::TestEncodedStorage;
use quantization::encoded_vectors_binary::{
    EncodedVectorsBin, Encoding, get_quantized_vector_size_from_params,
};
use quantization::{EncodedVectors, VectorParameters};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use tempfile::TempDir;

use crate::index::hnsw_index::graph_layers::GraphLayerData;
use crate::index::hnsw_index::graph_links::{GraphLinks, GraphLinksFormat};

use super::puffin_shared::{
    BlobSpec, MAGIC, TRAILER_LEN, build_test_puffin_fixture, mmap_whole_file,
    read_and_validate_footer, write_puffin,
};

// ---------- Helpers -------------------------------------------------------

/// Parse the row-pointer blob bytes into a decoded form the test can spot-check.
#[derive(Debug)]
struct DecodedRowPointers {
    version: u8,
    entry_count: u32,
    paths: Vec<String>,
    entries: Vec<(u32, u32, u32, u32)>, // (vec_id, file_path_idx, row_group, row_offset)
}

fn parse_row_pointer_blob(bytes: &[u8]) -> DecodedRowPointers {
    let version = bytes[0];
    let entry_count = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
    let path_count = u32::from_le_bytes(bytes[5..9].try_into().unwrap()) as usize;

    let mut cursor = 9usize;
    let mut paths = Vec::with_capacity(path_count);
    for _ in 0..path_count {
        let len =
            u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        let s = std::str::from_utf8(&bytes[cursor..cursor + len]).unwrap().to_string();
        cursor += len;
        paths.push(s);
    }

    let mut entries = Vec::with_capacity(entry_count as usize);
    for _ in 0..entry_count {
        let v = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        let f = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap());
        let g = u32::from_le_bytes(bytes[cursor + 8..cursor + 12].try_into().unwrap());
        let r = u32::from_le_bytes(bytes[cursor + 12..cursor + 16].try_into().unwrap());
        entries.push((v, f, g, r));
        cursor += 16;
    }

    DecodedRowPointers {
        version,
        entry_count,
        paths,
        entries,
    }
}

// ---------- Positive round-trip ------------------------------------------

#[test]
fn test_puffin_reader_full_roundtrip() {
    // ---- 1. Build fixture and locate/validate footer -----------------------
    let fixture = build_test_puffin_fixture();
    let file_bytes = fs::read(&fixture.puffin_path).unwrap();
    let footer = read_and_validate_footer(&file_bytes)
        .expect("well-formed fixture must pass footer validation");

    let expected_types = [
        "ann-hnsw-quantized-vectors-v1",
        "ann-hnsw-quantized-meta-v1",
        "ann-hnsw-graph-meta-v1",
        "ann-hnsw-graph-links-v1",
        "ann-hnsw-row-pointers-v1",
    ];
    assert_eq!(footer.blobs.len(), 5, "expected 5 blobs, got {}", footer.blobs.len());
    for (i, ty) in expected_types.iter().enumerate() {
        assert_eq!(footer.blobs[i].blob_type, *ty);
    }

    // ---- 2. mmap the whole file ONCE at offset 0 --------------------------
    let shared_mmap = mmap_whole_file(&fixture.puffin_path);
    assert_eq!(
        shared_mmap.len(),
        file_bytes.len(),
        "mmap length disagrees with fs::read",
    );

    // ---- 3. Graph meta — deserialize GraphLayerData owned from its slice ---
    let meta_range = footer.by_type("ann-hnsw-graph-meta-v1").unwrap();
    let meta_slice = &shared_mmap[meta_range.offset..meta_range.offset + meta_range.length];
    // `atomic_save_bin` (graph_layers_builder.rs:243) uses bincode; use the
    // symmetric `bincode::deserialize` here. `GraphLayerData<'de>` borrows via
    // `Cow<'de, EntryPoints>`; `DeserializeOwned` is not implemented so we use
    // the borrowing form and then extract owned copies of the fields we care
    // about before the slice's borrow ends.
    let graph_data: GraphLayerData<'_> =
        bincode::deserialize(meta_slice).expect("bincode deserialize GraphLayerData");
    assert_eq!(graph_data.m, fixture.m, "m mismatch");
    assert_eq!(
        graph_data.ef_construct, fixture.ef_construct,
        "ef_construct mismatch",
    );
    // Entry-point validity: at least one entry point present and its point_id
    // is inside [0, num_vectors). `EntryPoints`'s internal Vec isn't public,
    // so we probe via `get_entry_point` with an always-true checker.
    let ep = graph_data
        .entry_points
        .get_entry_point(|_| true)
        .expect("at least one entry point must be serialized");
    assert!(
        (ep.point_id as usize) < fixture.num_vectors,
        "entry point {} outside [0, {})",
        ep.point_id,
        fixture.num_vectors,
    );

    // ---- 4. Links — production path: load_from_ranged_mmap ----------------
    let links_range = footer.by_type("ann-hnsw-graph-links-v1").unwrap();
    let links = GraphLinks::load_from_ranged_mmap(
        Arc::clone(&shared_mmap),
        links_range.offset as u64,
        links_range.length as u64,
        GraphLinksFormat::Compressed,
    )
    .expect("load_from_ranged_mmap");
    assert_eq!(
        links.format(),
        GraphLinksFormat::Compressed,
        "format() disagrees with footer property",
    );
    assert_eq!(
        links.num_points(),
        fixture.num_vectors,
        "num_points() disagrees with fixture",
    );

    // ---- 5. Smoke: exercise populate/clear_cache on the ranged variant ----
    // These are the two new MmapRanged match arms in `graph_links.rs`. They
    // return `OperationResult<()>` but madvise is advisory: errors are
    // swallowed by design (log-and-continue, matching the whole-map
    // Madviseable pattern). This smoke check verifies NO-PANIC only — a real
    // page-residency check would require Linux `mincore(2)` and isn't
    // portable enough for a unit test.
    links.populate().expect("populate on MmapRanged");
    links.clear_cache().expect("clear_cache on MmapRanged");

    // ---- 6. Quantized vectors — variant-dispatched, fail loudly on unknown -
    let quant_range = footer.by_type("ann-hnsw-quantized-vectors-v1").unwrap();
    let quant_meta_range = footer.by_type("ann-hnsw-quantized-meta-v1").unwrap();
    let variant = quant_range.properties["quantization_variant"]
        .as_str()
        .expect("quantization_variant must be present per §3.2");

    match variant {
        "EncodedVectorsBin_u128" => {
            // PoC decision (§3.2 v3.11): extract-to-tempfile since
            // EncodedVectorsBin::load takes a `&Path` for the metadata. A
            // `load_from_bytes(storage, meta_bytes)` upstream would eliminate
            // this hop — documented as v-next in the spec.
            let extract_tmp = TempDir::new().unwrap();
            let data_path = extract_tmp.path().join("recovered.bin");
            let meta_path = extract_tmp.path().join("recovered.meta.json");
            fs::write(
                &data_path,
                &shared_mmap[quant_range.offset..quant_range.offset + quant_range.length],
            )
            .unwrap();
            fs::write(
                &meta_path,
                &shared_mmap
                    [quant_meta_range.offset..quant_meta_range.offset + quant_meta_range.length],
            )
            .unwrap();

            let quantized_vec_size =
                get_quantized_vector_size_from_params::<u128>(fixture.dim, Encoding::OneBit);
            assert_eq!(
                quantized_vec_size, fixture.quantized_vec_size,
                "recomputed quantized_vec_size disagrees with fixture",
            );
            let storage =
                TestEncodedStorage::from_file(&data_path, quantized_vec_size).unwrap();
            let encoded =
                EncodedVectorsBin::<u128, TestEncodedStorage>::load(&MmapFs, storage, &meta_path)
                    .expect("EncodedVectorsBin::<u128, _>::load");

            let params: &VectorParameters = encoded.get_vector_parameters();
            assert_eq!(params.dim, fixture.dim);
            assert_eq!(
                <EncodedVectorsBin<u128, _> as EncodedVectors>::vectors_count(&encoded),
                fixture.num_vectors,
            );
        }
        other => panic!(
            "unknown quantization_variant {other:?} — per §3.2 readers MUST fail loudly",
        ),
    }

    // ---- 7. Row pointers — parse per §3.2, spot-check ---------------------
    let rp_range = footer.by_type("ann-hnsw-row-pointers-v1").unwrap();
    let rp_bytes = &shared_mmap[rp_range.offset..rp_range.offset + rp_range.length];
    let parsed = parse_row_pointer_blob(rp_bytes);
    assert_eq!(parsed.version, 1, "row-pointer version byte");
    assert_eq!(parsed.entry_count as usize, fixture.num_vectors);
    assert_eq!(parsed.paths, fixture.file_paths, "path table content");

    // Spot-check 10 seeded-random entries. RNG seeded independently of the
    // fixture so this pattern is stable across runs.
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    for _ in 0..10 {
        let i = rng.random_range(0..fixture.num_vectors as u32);
        let entry = parsed.entries[i as usize];
        assert_eq!(entry, (i, 0u32, 0u32, i), "spot-check entry {i}");
    }
}

// ---------- Negative-test helper -----------------------------------------

/// Read `path`, parse its footer JSON, apply `mutate`, and rewrite the file
/// with the (possibly resized) footer + refreshed trailer. Returns the new
/// bytes without touching disk again.
fn corrupt_footer(
    path: &std::path::Path,
    original_footer_len: usize,
    mutate: impl FnOnce(&mut serde_json::Value),
) -> Vec<u8> {
    let file_bytes = fs::read(path).unwrap();
    let file_size = file_bytes.len();
    let footer_start = file_size - TRAILER_LEN - original_footer_len;
    let mut footer: serde_json::Value =
        serde_json::from_slice(&file_bytes[footer_start..file_size - TRAILER_LEN]).unwrap();
    mutate(&mut footer);
    let new_footer = serde_json::to_vec(&footer).unwrap();

    let mut out = file_bytes[..footer_start].to_vec();
    out.extend_from_slice(&new_footer);
    out.extend_from_slice(&(new_footer.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(MAGIC);
    out
}

// ---------- Negative: out-of-bounds footer --------------------------------

#[test]
fn test_puffin_reader_rejects_out_of_bounds_blob() {
    // Synthesize a minimal file that would pass trailer parsing but declares a
    // blob range beyond the file end. `read_and_validate_footer` MUST reject
    // this per §6.1's "reject on corruption" invariant.
    let tmp = TempDir::new().unwrap();
    let bad_path = tmp.path().join("bad.puffin");

    let junk_bytes: Vec<u8> = (0..16).collect();
    let blobs = [BlobSpec {
        blob_type: "ann-hnsw-quantized-vectors-v1",
        bytes: &junk_bytes,
        properties: serde_json::json!({
            "quantization_variant": "EncodedVectorsBin_u128",
            "quantization_family": "binary",
            "dimensions": "8",
            "created-by": "test-negative",
        }),
    }];
    let layout = write_puffin(&bad_path, &blobs).unwrap();

    // Inflate the blob's length field well past file end.
    let file_bytes = corrupt_footer(&bad_path, layout.footer_len, |footer| {
        footer["blobs"][0]["length"] = serde_json::json!(u64::from(u32::MAX));
    });

    let err = read_and_validate_footer(&file_bytes)
        .expect_err("out-of-bounds blob length must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("exceeds file bounds") || msg.contains("overflows usize"),
        "unexpected error message: {msg}",
    );
}

// ---------- Negative: missing / unsupported compression ------------------

#[test]
fn test_puffin_reader_rejects_missing_or_unknown_compression() {
    // §3.2 makes `compression` load-bearing; absent OR outside {"none","zstd"}
    // must fail-fast rather than fall through to a later decode error.
    let tmp = TempDir::new().unwrap();

    // Case 1: field present but with an unsupported codec value.
    let path = tmp.path().join("lz4.puffin");
    let junk: Vec<u8> = (0..16).collect();
    let blobs = [BlobSpec {
        blob_type: "ann-hnsw-quantized-vectors-v1",
        bytes: &junk,
        properties: serde_json::json!({
            "quantization_variant": "EncodedVectorsBin_u128",
            "quantization_family": "binary",
            "dimensions": "8",
            "created-by": "test-negative",
        }),
    }];
    let layout = write_puffin(&path, &blobs).unwrap();
    let file_bytes = corrupt_footer(&path, layout.footer_len, |footer| {
        footer["blobs"][0]["compression"] = serde_json::json!("lz4");
    });
    let err = read_and_validate_footer(&file_bytes)
        .expect_err("unsupported compression must be rejected");
    assert!(
        err.to_string().contains("unsupported compression"),
        "unexpected error message: {err}",
    );

    // Case 2: `compression` field removed entirely.
    let file_bytes = corrupt_footer(&path, layout.footer_len, |footer| {
        footer["blobs"][0]
            .as_object_mut()
            .unwrap()
            .remove("compression");
    });
    let err = read_and_validate_footer(&file_bytes)
        .expect_err("missing compression must be rejected");
    assert!(
        err.to_string().contains("missing `compression`"),
        "unexpected error message: {err}",
    );
}

// ---------- Negative: non-zero flags -------------------------------------

#[test]
fn test_puffin_reader_rejects_nonzero_flags() {
    // Real Puffin uses bit 0 of the flags word for LZ4 footer compression.
    // We do not support flagged/compressed footers in v1 — reject with an
    // explicit message rather than let a corrupt footer trigger a misleading
    // JSON parse error.
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("flagged.puffin");
    let junk: Vec<u8> = (0..16).collect();
    let blobs = [BlobSpec {
        blob_type: "ann-hnsw-quantized-vectors-v1",
        bytes: &junk,
        properties: serde_json::json!({
            "quantization_variant": "EncodedVectorsBin_u128",
            "quantization_family": "binary",
            "dimensions": "8",
            "created-by": "test-negative",
        }),
    }];
    write_puffin(&path, &blobs).unwrap();

    let mut file_bytes = fs::read(&path).unwrap();
    let file_size = file_bytes.len();
    // Overwrite the flags word (u32 LE at [file_size-8 .. file_size-4]) with
    // bit 0 set — the LZ4-footer flag in real Puffin. Trailing magic stays
    // intact; footer_size stays intact.
    file_bytes[file_size - 8..file_size - 4].copy_from_slice(&1u32.to_le_bytes());

    let err = read_and_validate_footer(&file_bytes)
        .expect_err("non-zero flags must be rejected");
    assert!(
        err.to_string().contains("flagged/compressed footers not supported"),
        "unexpected error message: {err}",
    );
}

// ---------- Negative: load_from_ranged_mmap arithmetic-overflow guard -----

#[test]
fn test_load_from_ranged_mmap_rejects_overflow() {
    // Create a small real mmap (its exact contents don't matter — we never
    // touch them; construction fails at the bounds/overflow check before
    // reaching GraphLinksView::load).
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("tiny.bin");
    fs::write(&path, [0u8; 4096]).unwrap();
    let shared_mmap = mmap_whole_file(&path);

    // (a) Arithmetic-overflow case — offset + length wraps.
    //     `checked_add(u64::MAX - 10, 100)` overflows; the constructor must
    //     reject rather than pass a garbage slice to GraphLinksView::load.
    let err = GraphLinks::load_from_ranged_mmap(
        Arc::clone(&shared_mmap),
        u64::MAX - 10,
        100,
        GraphLinksFormat::Compressed,
    )
    .expect_err("overflow (offset=u64::MAX-10, length=100) must be rejected");
    let msg = err.to_string();
    // On a 64-bit host the offset fits in usize but offset+length overflows.
    // On a 32-bit host offset itself won't fit in usize. Accept either error.
    assert!(
        msg.contains("overflows usize") || msg.contains("does not fit in usize"),
        "unexpected error message: {msg}",
    );

    // (b) In-bounds arithmetic but past mmap end — a saner corruption pattern.
    let err = GraphLinks::load_from_ranged_mmap(
        Arc::clone(&shared_mmap),
        0,
        (shared_mmap.len() + 1) as u64,
        GraphLinksFormat::Compressed,
    )
    .expect_err("length past mmap end must be rejected");
    assert!(
        err.to_string().contains("exceeds file capacity"),
        "unexpected error message: {err}",
    );
}
