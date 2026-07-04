//! Phase 1 of the Puffin-HNSW PoC: write a valid `.puffin` container end-to-end
//! and assert its structural invariants (§3 layout, §5 Phase 1 of
//! `docs/puffin_ann_spec.md` v3.11). This test does NOT verify search
//! correctness — that is Phase 2/3 territory.
//!
//! Blob order matches §3.1 v3.11:
//!   0. ann-hnsw-quantized-vectors-v1
//!   1. ann-hnsw-quantized-meta-v1
//!   2. ann-hnsw-graph-meta-v1        (raw graph.bin bytes)
//!   3. ann-hnsw-graph-links-v1       (raw links_compressed.bin bytes)
//!   4. ann-hnsw-row-pointers-v1
//!
//! Shared build fixture + wire-format helpers live in `puffin_shared.rs`; this
//! test focuses on the structural assertions that verify the writer produced a
//! spec-conformant container.

use fs_err as fs;

use super::puffin_shared::{
    BLOB_ALIGN, MAGIC, TRAILER_LEN, build_test_puffin_fixture, read_and_validate_footer,
};

#[test]
fn test_puffin_writer_writes_valid_container() {
    let fixture = build_test_puffin_fixture();
    let file_bytes = fs::read(&fixture.puffin_path).unwrap();
    let actual_size = file_bytes.len();

    // (a) Leading magic.
    assert_eq!(&file_bytes[..4], MAGIC, "missing leading PFA1 magic");

    // (b) Trailing magic.
    assert_eq!(
        &file_bytes[actual_size - 4..actual_size],
        MAGIC,
        "missing trailing PFA1 magic",
    );

    // (c) Footer parses cleanly (§3.3 trailer + JSON) and passes §6.1 range
    // validation via the shared reader helper — the same code path Phase 2
    // exercises. Failure here means the writer produced a container the
    // reader would reject.
    let footer = read_and_validate_footer(&file_bytes)
        .expect("writer output must pass read_and_validate_footer");

    // (d) Exact blob type list in exact order.
    let expected_types = [
        "ann-hnsw-quantized-vectors-v1",
        "ann-hnsw-quantized-meta-v1",
        "ann-hnsw-graph-meta-v1",
        "ann-hnsw-graph-links-v1",
        "ann-hnsw-row-pointers-v1",
    ];
    assert_eq!(footer.blobs.len(), expected_types.len(), "blob count");
    for (i, ty) in expected_types.iter().enumerate() {
        assert_eq!(footer.blobs[i].blob_type, *ty);
        assert_eq!(footer.blobs[i].compression, "none");
        assert_eq!(footer.blobs[i].offset % BLOB_ALIGN, 0, "blob {i} not aligned");
        assert!(footer.blobs[i].length > 0, "blob {i} empty");
    }

    // (e) Row-pointer blob has correct leading version byte and entry count.
    let rp = footer.by_type("ann-hnsw-row-pointers-v1").unwrap();
    assert_eq!(file_bytes[rp.offset], 1u8, "row-pointer version byte must be 1");
    let rp_entry_count = u32::from_le_bytes(
        file_bytes[rp.offset + 1..rp.offset + 5].try_into().unwrap(),
    );
    assert_eq!(rp_entry_count as usize, fixture.num_vectors);

    // (f) Mid-magic present and 64-byte aligned (§3.1 v3.11 writer invariant).
    // Reader-visible location: it lives immediately before the footer JSON,
    // so we can locate it from the trailer without trusting the writer's
    // internal recording.
    let footer_start = actual_size - TRAILER_LEN - footer.footer_size;
    let mid_magic_offset = footer_start - 4;
    assert_eq!(
        mid_magic_offset % BLOB_ALIGN,
        0,
        "mid magic offset {mid_magic_offset} not 64-aligned (writer invariant)",
    );
    assert_eq!(
        &file_bytes[mid_magic_offset..mid_magic_offset + 4],
        MAGIC,
        "missing mid PFA1 magic",
    );

    // (g) Exact file-size formula:
    //     mid_magic_offset + 4 (mid magic) + footer_size + TRAILER_LEN
    let expected_size = mid_magic_offset + 4 + footer.footer_size + TRAILER_LEN;
    assert_eq!(
        actual_size, expected_size,
        "file-size formula mismatch: expected {expected_size}, got {actual_size}",
    );
}
