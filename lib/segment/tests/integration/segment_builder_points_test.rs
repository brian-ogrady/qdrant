//! Tests for [`SegmentBuilder::update_from_points`], the bulk ingest path.
//!
//! The load-bearing test here is [`bulk_ingest_matches_upsert_then_merge`]: it builds the same
//! points twice, once through an appendable segment and `update` (what an offline loader had to do
//! before) and once through `update_from_points`, and compares the two finished segments. Anything
//! the new path forgets to do that the write path does — preprocessing above all — shows up there
//! rather than as a plausible-looking wrong answer in production.

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use segment::data_types::named_vectors::NamedVectors;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, VectorInternal, VectorRef};
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use segment::index::sparse_index::sparse_index_config::{SparseIndexConfig, SparseIndexType};
use segment::segment::Segment;
use segment::segment_constructor::build_segment;
use segment::segment_constructor::segment_builder::{PointToInsert, SegmentBuilder};
use segment::types::{
    Distance, HnswGlobalConfig, Indexes, Payload, PayloadStorageType, PointIdType, SegmentConfig,
    SparseVectorDataConfig, SparseVectorStorageType, VectorDataConfig, VectorStorageType,
};
use sparse::common::sparse_vector::SparseVector;
use tempfile::Builder;

const DENSE: &str = DEFAULT_VECTOR_NAME;
const SPARSE: &str = "sparse";
const DIM: usize = 4;

/// One dense and one sparse vector, plus on-disk payload.
///
/// `appendable` picks between the two shapes that actually occur: the write target a shard upserts
/// into, and the immutable segment the indexing optimizer produces. The bulk path is aimed at the
/// second, so that is what the differential test compares — but the *source* for the old route has
/// to be appendable, because that is the only thing `upsert_point` accepts.
///
/// Cosine is the interesting distance here: it is the one whose preprocessing step
/// (normalisation) is observable in the stored vectors, so a path that skips preprocessing cannot
/// pass a comparison against one that does not.
fn config(distance: Distance, appendable: bool) -> SegmentConfig {
    let (storage_type, sparse_index_type) = if appendable {
        (VectorStorageType::ChunkedMmap, SparseIndexType::MutableRam)
    } else {
        (VectorStorageType::Mmap, SparseIndexType::Mmap)
    };

    SegmentConfig {
        vector_data: HashMap::from([(
            DENSE.to_owned(),
            VectorDataConfig {
                size: DIM,
                distance,
                storage_type,
                index: Indexes::Plain {},
                quantization_config: None,
                multivector_config: None,
                datatype: None,
            },
        )]),
        sparse_vector_data: HashMap::from([(
            SPARSE.to_owned(),
            SparseVectorDataConfig {
                index: SparseIndexConfig::new(None, sparse_index_type, None, None),
                storage_type: SparseVectorStorageType::default(),
                modifier: None,
            },
        )]),
        payload_storage_type: PayloadStorageType::Mmap,
    }
}

/// The immutable target every route builds into.
fn target_config(distance: Distance) -> SegmentConfig {
    config(distance, false)
}

/// Deterministic, deliberately un-normalised input so Cosine preprocessing has work to do.
fn sample_points(count: u64) -> Vec<(PointIdType, VectorInternal, SparseVector, Payload)> {
    (0..count)
        .map(|i| {
            let f = i as f32;
            let dense = VectorInternal::from(vec![f + 1.0, f + 2.0, f + 3.0, f + 4.0]);
            // Varying term counts and dimensions, so posting lists differ in length.
            let indices: Vec<u32> = (0..=(i as u32 % 5)).map(|t| t * 3 + 1).collect();
            let values: Vec<f32> = indices.iter().map(|t| 1.0 + *t as f32).collect();
            let sparse = SparseVector::new(indices, values).unwrap();
            let payload: Payload = serde_json::from_value(serde_json::json!({
                "idx": i,
                "group": if i % 2 == 0 { "even" } else { "odd" },
                "text": format!("document number {i}"),
            }))
            .unwrap();
            (PointIdType::NumId(i), dense, sparse, payload)
        })
        .collect()
}

fn named(dense: &VectorInternal, sparse: &SparseVector) -> NamedVectors<'static> {
    let mut vectors = NamedVectors::default();
    vectors.insert(DENSE.to_owned(), dense.clone());
    vectors.insert(SPARSE.to_owned(), VectorInternal::Sparse(sparse.clone()));
    vectors
}

/// Build via the old route: upsert into an appendable segment, then merge it.
fn build_via_upsert(
    dir: &std::path::Path,
    temp: &std::path::Path,
    distance: Distance,
    points: &[(PointIdType, VectorInternal, SparseVector, Payload)],
) -> Segment {
    let hw_counter = HardwareCounterCell::new();
    let stopped = AtomicBool::new(false);

    let source_dir = dir.join("source");
    fs_err::create_dir_all(&source_dir).unwrap();

    let (mut appendable, _token) =
        build_segment(&source_dir, &config(distance, true), None, true).unwrap();
    for (op_num, (id, dense, sparse, payload)) in points.iter().enumerate() {
        let op_num = op_num as u64 + 1;
        appendable
            .upsert_point(op_num, *id, named(dense, sparse), &hw_counter)
            .unwrap();
        appendable
            .set_full_payload(op_num, *id, payload, &hw_counter)
            .unwrap();
    }

    let mut builder =
        SegmentBuilder::new(temp, &target_config(distance), &HnswGlobalConfig::default()).unwrap();
    builder
        .update(&[&appendable], &stopped, &hw_counter)
        .unwrap();
    builder.build_for_test(dir)
}

/// Build via the new route: straight into the builder's storages.
fn build_via_points(
    dir: &std::path::Path,
    temp: &std::path::Path,
    cfg: &SegmentConfig,
    points: &[(PointIdType, VectorInternal, SparseVector, Payload)],
    batch_points: usize,
) -> (Segment, usize) {
    let hw_counter = HardwareCounterCell::new();
    let stopped = AtomicBool::new(false);

    let mut builder = SegmentBuilder::new(temp, cfg, &HnswGlobalConfig::default()).unwrap();
    let inserted = builder
        .update_from_points(
            points
                .iter()
                .enumerate()
                .map(|(op_num, (id, dense, sparse, payload))| {
                    Ok(PointToInsert {
                        external_id: *id,
                        version: op_num as u64 + 1,
                        vectors: named(dense, sparse),
                        payload: Some(payload.clone()),
                    })
                }),
            batch_points,
            &stopped,
            &hw_counter,
        )
        .unwrap();

    (builder.build_for_test(dir), inserted)
}

/// Every externally observable property of the two segments, for comparison.
fn snapshot(segment: &Segment) -> Vec<(PointIdType, Vec<f32>, Option<SparseVector>, Payload)> {
    let hw_counter = HardwareCounterCell::new();

    let mut rows: Vec<_> = segment
        .iter_points()
        .map(|external_id| {
            let vectors = segment.all_vectors(external_id, &hw_counter).unwrap();

            let dense = match vectors.get(DENSE) {
                Some(VectorRef::Dense(values)) => values.to_vec(),
                other => panic!("expected a dense vector for {external_id}, got {other:?}"),
            };

            let sparse = match vectors.get(SPARSE) {
                Some(VectorRef::Sparse(vector)) => Some(vector.clone()),
                None => None,
                other => panic!("unexpected sparse vector for {external_id}: {other:?}"),
            };

            let payload = segment.payload(external_id, &hw_counter).unwrap();

            (external_id, dense, sparse, payload)
        })
        .collect();

    rows.sort_by_key(|(id, ..)| *id);
    rows
}

/// The differential test. Both routes must produce the same segment contents.
#[test]
fn bulk_ingest_matches_upsert_then_merge() {
    let old_dir = Builder::new().prefix("old").tempdir().unwrap();
    let old_temp = Builder::new().prefix("old_temp").tempdir().unwrap();
    let new_dir = Builder::new().prefix("new").tempdir().unwrap();
    let new_temp = Builder::new().prefix("new_temp").tempdir().unwrap();

    let cfg = target_config(Distance::Cosine);
    let points = sample_points(64);

    let old = build_via_upsert(old_dir.path(), old_temp.path(), Distance::Cosine, &points);
    let (new, inserted) = build_via_points(new_dir.path(), new_temp.path(), &cfg, &points, 8);

    assert_eq!(inserted, points.len(), "every point should be new");
    assert_eq!(
        new.available_point_count(),
        old.available_point_count(),
        "point counts must agree",
    );

    let old_rows = snapshot(&old);
    let new_rows = snapshot(&new);

    assert_eq!(old_rows.len(), points.len());
    assert_eq!(
        new_rows, old_rows,
        "the bulk path must store the same ids, vectors, sparse vectors and payloads",
    );
}

/// Cosine vectors must be normalised on the way in.
///
/// Called out separately from the differential test because this is the failure that would be
/// invisible otherwise: `VectorStorage::update_from` documents that it does not preprocess, since
/// on the merge path its inputs were preprocessed when they were first written. A bulk path that
/// inherits that assumption stores raw vectors, and every Cosine search then returns subtly wrong
/// neighbours while every count and every id still checks out.
#[test]
fn cosine_vectors_are_normalised_by_the_bulk_path() {
    let dir = Builder::new().prefix("norm").tempdir().unwrap();
    let temp = Builder::new().prefix("norm_temp").tempdir().unwrap();

    let cfg = target_config(Distance::Cosine);
    let points = sample_points(4);
    let (segment, _) = build_via_points(dir.path(), temp.path(), &cfg, &points, 2);

    for (_, dense, ..) in snapshot(&segment) {
        let norm: f32 = dense.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "stored Cosine vector should be unit length, got norm {norm} for {dense:?}",
        );
    }
}

/// Dot distance does not normalise, so the stored vector must be byte-identical to the input.
///
/// The mirror of the test above: it catches a bulk path that "fixes" preprocessing by always
/// normalising rather than by asking the config.
#[test]
fn dot_vectors_are_stored_unchanged_by_the_bulk_path() {
    let dir = Builder::new().prefix("dot").tempdir().unwrap();
    let temp = Builder::new().prefix("dot_temp").tempdir().unwrap();

    let cfg = target_config(Distance::Dot);
    let points = sample_points(4);
    let (segment, _) = build_via_points(dir.path(), temp.path(), &cfg, &points, 2);

    for ((_, expected, ..), (_, stored, ..)) in points.iter().zip(snapshot(&segment)) {
        let VectorInternal::Dense(expected) = expected else {
            panic!("fixture is dense");
        };
        assert_eq!(&stored, expected, "Dot vectors must not be rewritten");
    }
}

/// Batch size must not be observable in the result.
#[test]
fn batch_size_does_not_change_the_result() {
    let cfg = target_config(Distance::Cosine);
    let points = sample_points(37);

    let mut snapshots = Vec::new();
    for batch_points in [1, 2, 37, 1024] {
        let dir = Builder::new().prefix("batch").tempdir().unwrap();
        let temp = Builder::new().prefix("batch_temp").tempdir().unwrap();
        let (segment, inserted) =
            build_via_points(dir.path(), temp.path(), &cfg, &points, batch_points);
        assert_eq!(inserted, points.len(), "batch_points={batch_points}");
        snapshots.push((batch_points, snapshot(&segment)));
    }

    let (_, first) = &snapshots[0];
    for (batch_points, rows) in &snapshots[1..] {
        assert_eq!(
            rows, first,
            "batch_points={batch_points} produced a different segment",
        );
    }
}

/// A repeated external id must collapse to one point, keeping the later version.
#[test]
fn repeated_external_ids_collapse_to_the_later_version() {
    let dir = Builder::new().prefix("dup").tempdir().unwrap();
    let temp = Builder::new().prefix("dup_temp").tempdir().unwrap();
    let hw_counter = HardwareCounterCell::new();
    let stopped = AtomicBool::new(false);

    let cfg = target_config(Distance::Dot);
    let sparse = SparseVector::new(vec![1], vec![1.0]).unwrap();

    let first = VectorInternal::from(vec![1.0, 0.0, 0.0, 0.0]);
    let second = VectorInternal::from(vec![0.0, 2.0, 0.0, 0.0]);

    let mut builder = SegmentBuilder::new(temp.path(), &cfg, &HnswGlobalConfig::default()).unwrap();
    let inserted = builder
        .update_from_points(
            vec![
                Ok(PointToInsert {
                    external_id: PointIdType::NumId(7),
                    version: 1,
                    vectors: named(&first, &sparse),
                    payload: Some(serde_json::from_value(serde_json::json!({"v": 1})).unwrap()),
                }),
                Ok(PointToInsert {
                    external_id: PointIdType::NumId(7),
                    version: 2,
                    vectors: named(&second, &sparse),
                    payload: Some(serde_json::from_value(serde_json::json!({"v": 2})).unwrap()),
                }),
            ],
            8,
            &stopped,
            &hw_counter,
        )
        .unwrap();

    assert_eq!(inserted, 1, "a repeated id is one point, not two");

    let segment = builder.build_for_test(dir.path());
    assert_eq!(segment.available_point_count(), 1);

    let rows = snapshot(&segment);
    assert_eq!(rows.len(), 1);
    let (id, dense, _, payload) = &rows[0];
    assert_eq!(*id, PointIdType::NumId(7));
    assert_eq!(dense, &vec![0.0, 2.0, 0.0, 0.0], "the later version wins");
    assert_eq!(
        payload.0.get("v").and_then(|v| v.as_i64()),
        Some(2),
        "the later payload wins",
    );
}

/// An earlier-versioned repeat must not overwrite a later one already stored.
#[test]
fn an_older_repeat_does_not_overwrite_a_newer_point() {
    let dir = Builder::new().prefix("old_dup").tempdir().unwrap();
    let temp = Builder::new().prefix("old_dup_temp").tempdir().unwrap();
    let hw_counter = HardwareCounterCell::new();
    let stopped = AtomicBool::new(false);

    let cfg = target_config(Distance::Dot);
    let sparse = SparseVector::new(vec![1], vec![1.0]).unwrap();
    let newer = VectorInternal::from(vec![9.0, 0.0, 0.0, 0.0]);
    let older = VectorInternal::from(vec![1.0, 0.0, 0.0, 0.0]);

    let mut builder = SegmentBuilder::new(temp.path(), &cfg, &HnswGlobalConfig::default()).unwrap();
    builder
        .update_from_points(
            vec![
                Ok(PointToInsert {
                    external_id: PointIdType::NumId(1),
                    version: 10,
                    vectors: named(&newer, &sparse),
                    payload: Some(serde_json::from_value(serde_json::json!({"v": 10})).unwrap()),
                }),
                Ok(PointToInsert {
                    external_id: PointIdType::NumId(1),
                    version: 2,
                    vectors: named(&older, &sparse),
                    payload: Some(serde_json::from_value(serde_json::json!({"v": 2})).unwrap()),
                }),
            ],
            8,
            &stopped,
            &hw_counter,
        )
        .unwrap();

    let segment = builder.build_for_test(dir.path());
    assert_eq!(segment.available_point_count(), 1);

    let rows = snapshot(&segment);
    let (_, dense, _, payload) = &rows[0];
    assert_eq!(dense, &vec![9.0, 0.0, 0.0, 0.0], "the newer version stands");
    assert_eq!(payload.0.get("v").and_then(|v| v.as_i64()), Some(10));
}

/// Duplicates interleaved among normal points must not shift the other points' vectors.
///
/// This is the riskiest path in the bulk builder. `SegmentBuilder::update` has the same
/// duplicate-resolution branch but marks it `debug_assert!(false, "should not be reachable")`,
/// because its sources are pre-deduplicated — so the bulk path exercises code upstream never
/// does. The failure it guards against is not the duplicate itself but its neighbours: the losing
/// row still occupies an internal id, and if that were mishandled every later point would be
/// paired with the wrong vector while all the counts still looked right.
#[test]
fn duplicates_among_normal_points_do_not_shift_their_neighbours() {
    let dir = Builder::new().prefix("interleaved").tempdir().unwrap();
    let temp = Builder::new().prefix("interleaved_temp").tempdir().unwrap();
    let hw_counter = HardwareCounterCell::new();
    let stopped = AtomicBool::new(false);

    let cfg = target_config(Distance::Dot);
    let sparse = SparseVector::new(vec![1, 4], vec![1.0, 2.0]).unwrap();

    // ids 0..20 once each, plus a second copy of every fourth id carrying a marker vector. The
    // duplicates land in different batches as well as within one, since batch_points is 4.
    let mut feed: Vec<(u64, f32, u64)> = Vec::new();
    let mut version = 0u64;
    for id in 0..20u64 {
        version += 1;
        feed.push((id, id as f32, version));
        if id % 4 == 0 {
            version += 1;
            feed.push((id, 1000.0 + id as f32, version));
        }
    }

    let mut builder = SegmentBuilder::new(temp.path(), &cfg, &HnswGlobalConfig::default()).unwrap();
    let inserted = builder
        .update_from_points(
            feed.iter().map(|(id, marker, version)| {
                Ok(PointToInsert {
                    external_id: PointIdType::NumId(*id),
                    version: *version,
                    vectors: named(&VectorInternal::from(vec![*marker, 0.0, 0.0, 0.0]), &sparse),
                    payload: Some(
                        serde_json::from_value(serde_json::json!({"marker": marker})).unwrap(),
                    ),
                })
            }),
            4,
            &stopped,
            &hw_counter,
        )
        .unwrap();

    assert_eq!(inserted, 20, "20 distinct ids among 25 records");

    let segment = builder.build_for_test(dir.path());
    assert_eq!(segment.available_point_count(), 20);

    let rows = snapshot(&segment);
    assert_eq!(rows.len(), 20);

    for (index, (id, dense, sparse_vector, payload)) in rows.iter().enumerate() {
        let expected_id = index as u64;
        assert_eq!(
            *id,
            PointIdType::NumId(expected_id),
            "ids must be dense 0..20"
        );

        // Every fourth id was written twice, so it must show the second (marker) vector; every
        // other id must show its own, unshifted.
        let expected_marker = if expected_id.is_multiple_of(4) {
            1000.0 + expected_id as f32
        } else {
            expected_id as f32
        };
        assert_eq!(
            dense,
            &vec![expected_marker, 0.0, 0.0, 0.0],
            "point {expected_id} has the wrong vector; the id sequence shifted",
        );
        assert_eq!(
            payload.0.get("marker").and_then(|v| v.as_f64()),
            Some(f64::from(expected_marker)),
            "point {expected_id} has the wrong payload",
        );
        assert!(
            sparse_vector.is_some(),
            "point {expected_id} lost its sparse vector",
        );
    }
}

/// A point missing a configured vector name keeps the internal id sequence dense.
///
/// The storages share one internal id, so a skipped row in one of them would silently shift every
/// later point's vectors against its id. The appendable path stores a default and flags it
/// deleted; this must do the same.
#[test]
fn a_missing_vector_name_keeps_the_id_sequence_aligned() {
    let dir = Builder::new().prefix("missing").tempdir().unwrap();
    let temp = Builder::new().prefix("missing_temp").tempdir().unwrap();
    let hw_counter = HardwareCounterCell::new();
    let stopped = AtomicBool::new(false);

    let cfg = target_config(Distance::Dot);
    let sparse = SparseVector::new(vec![1, 4], vec![1.0, 2.0]).unwrap();

    // The middle point carries no sparse vector at all.
    let mut middle = NamedVectors::default();
    middle.insert(
        DENSE.to_owned(),
        VectorInternal::from(vec![5.0, 5.0, 5.0, 5.0]),
    );

    let mut builder = SegmentBuilder::new(temp.path(), &cfg, &HnswGlobalConfig::default()).unwrap();
    builder
        .update_from_points(
            vec![
                Ok(PointToInsert {
                    external_id: PointIdType::NumId(1),
                    version: 1,
                    vectors: named(&VectorInternal::from(vec![1.0, 0.0, 0.0, 0.0]), &sparse),
                    payload: None,
                }),
                Ok(PointToInsert {
                    external_id: PointIdType::NumId(2),
                    version: 2,
                    vectors: middle,
                    payload: None,
                }),
                Ok(PointToInsert {
                    external_id: PointIdType::NumId(3),
                    version: 3,
                    vectors: named(&VectorInternal::from(vec![0.0, 0.0, 0.0, 3.0]), &sparse),
                    payload: None,
                }),
            ],
            8,
            &stopped,
            &hw_counter,
        )
        .unwrap();

    let segment = builder.build_for_test(dir.path());
    assert_eq!(segment.available_point_count(), 3);

    let rows = snapshot(&segment);
    assert_eq!(rows.len(), 3);

    // The dense vectors must still line up with their own ids, which is what a shifted sequence
    // would break.
    assert_eq!(rows[0].1, vec![1.0, 0.0, 0.0, 0.0]);
    assert_eq!(rows[1].1, vec![5.0, 5.0, 5.0, 5.0]);
    assert_eq!(rows[2].1, vec![0.0, 0.0, 0.0, 3.0]);

    // And the point with no sparse vector must report none, while its neighbours keep theirs.
    assert!(rows[0].2.is_some(), "point 1 keeps its sparse vector");
    assert!(rows[1].2.is_none(), "point 2 never had one");
    assert!(rows[2].2.is_some(), "point 3 keeps its sparse vector");
}

/// An empty input is not an error, and produces an empty segment.
#[test]
fn no_points_builds_an_empty_segment() {
    let dir = Builder::new().prefix("empty").tempdir().unwrap();
    let temp = Builder::new().prefix("empty_temp").tempdir().unwrap();
    let hw_counter = HardwareCounterCell::new();
    let stopped = AtomicBool::new(false);

    let cfg = target_config(Distance::Cosine);
    let mut builder = SegmentBuilder::new(temp.path(), &cfg, &HnswGlobalConfig::default()).unwrap();

    let inserted = builder
        .update_from_points(
            Vec::<segment::common::operation_error::OperationResult<PointToInsert>>::new(),
            16,
            &stopped,
            &hw_counter,
        )
        .unwrap();

    assert_eq!(inserted, 0);
    let segment = builder.build_for_test(dir.path());
    assert_eq!(segment.available_point_count(), 0);
}

/// An error from the input iterator must abort rather than be silently skipped.
#[test]
fn an_input_error_aborts_the_build() {
    let temp = Builder::new().prefix("err_temp").tempdir().unwrap();
    let hw_counter = HardwareCounterCell::new();
    let stopped = AtomicBool::new(false);

    let cfg = target_config(Distance::Dot);
    let sparse = SparseVector::new(vec![1], vec![1.0]).unwrap();
    let mut builder = SegmentBuilder::new(temp.path(), &cfg, &HnswGlobalConfig::default()).unwrap();

    let result = builder.update_from_points(
        vec![
            Ok(PointToInsert {
                external_id: PointIdType::NumId(1),
                version: 1,
                vectors: named(&VectorInternal::from(vec![1.0, 0.0, 0.0, 0.0]), &sparse),
                payload: None,
            }),
            Err(
                segment::common::operation_error::OperationError::service_error(
                    "part file is truncated",
                ),
            ),
        ],
        16,
        &stopped,
        &hw_counter,
    );

    let err = result.expect_err("a failed read must not be swallowed");
    assert!(
        format!("{err}").contains("truncated"),
        "the original error should survive: {err}",
    );
}

/// A batch size of zero would loop forever; it must be rejected up front.
#[test]
fn zero_batch_size_is_rejected() {
    let temp = Builder::new().prefix("zero_temp").tempdir().unwrap();
    let hw_counter = HardwareCounterCell::new();
    let stopped = AtomicBool::new(false);

    let cfg = target_config(Distance::Dot);
    let mut builder = SegmentBuilder::new(temp.path(), &cfg, &HnswGlobalConfig::default()).unwrap();

    let err = builder
        .update_from_points(
            Vec::<segment::common::operation_error::OperationResult<PointToInsert>>::new(),
            0,
            &stopped,
            &hw_counter,
        )
        .expect_err("batch_points of 0 must be refused");
    assert!(format!("{err}").contains("batch_points"), "{err}");
}

/// Unknown vector names must be refused, not silently dropped.
#[test]
fn an_unknown_vector_name_is_refused() {
    let temp = Builder::new().prefix("unknown_temp").tempdir().unwrap();
    let hw_counter = HardwareCounterCell::new();
    let stopped = AtomicBool::new(false);

    let cfg = target_config(Distance::Dot);
    let mut vectors = NamedVectors::default();
    vectors.insert(
        "not_in_the_config".to_owned(),
        VectorInternal::from(vec![1.0, 2.0, 3.0, 4.0]),
    );

    let mut builder = SegmentBuilder::new(temp.path(), &cfg, &HnswGlobalConfig::default()).unwrap();
    let err = builder
        .update_from_points(
            vec![Ok(PointToInsert {
                external_id: PointIdType::NumId(1),
                version: 1,
                vectors,
                payload: None,
            })],
            16,
            &stopped,
            &hw_counter,
        )
        .expect_err("an unconfigured vector name must be refused");
    assert!(
        format!("{err}")
            .to_lowercase()
            .contains("not_in_the_config")
            || format!("{err}").to_lowercase().contains("not exist"),
        "{err}",
    );
}

/// Sparse search must work on a segment built by the bulk path.
///
/// This is the end of the chain that motivates the whole change: the bulk path never builds a
/// mutable sparse index, so the only inverted index the segment ever gets is the one `build`
/// constructs. If that were not wired up, counts and vectors would all still be right and only
/// retrieval would be broken.
#[test]
fn sparse_search_works_on_a_bulk_built_segment() {
    let dir = Builder::new().prefix("sparse_search").tempdir().unwrap();
    let temp = Builder::new()
        .prefix("sparse_search_temp")
        .tempdir()
        .unwrap();

    let cfg = target_config(Distance::Dot);
    let points = sample_points(32);
    let (segment, _) = build_via_points(dir.path(), temp.path(), &cfg, &points, 4);

    // Query on a dimension every fixture point carries (index 1).
    let query = SparseVector::new(vec![1], vec![1.0]).unwrap();

    let results = segment
        .search(
            SPARSE,
            &VectorInternal::Sparse(query).into(),
            &Default::default(),
            &Default::default(),
            None,
            10,
            None,
        )
        .unwrap();

    assert!(
        !results.is_empty(),
        "the bulk-built segment must be searchable on its sparse vector",
    );
}
