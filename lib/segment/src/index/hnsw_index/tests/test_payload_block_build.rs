use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize};

use atomic_refcell::AtomicRefCell;
use common::budget::ResourcePermit;
use common::counter::hardware_counter::HardwareCounterCell;
use common::flags::FeatureFlags;
use common::progress_tracker::ProgressTracker;
use common::types::PointOffsetType;
use parking_lot::Mutex;
use rand::SeedableRng;
use rand::prelude::StdRng;
use rstest::rstest;
use tempfile::Builder;

use crate::data_types::index::KeywordIndexParams;
use crate::data_types::vectors::{DEFAULT_VECTOR_NAME, only_default_vector};
use crate::entry::entry_point::SegmentEntry;
use crate::fixtures::index_fixtures::random_vector;
use crate::id_tracker::IdTracker;
use crate::index::PayloadIndex;
use crate::index::hnsw_index::get_num_indexing_threads;
use crate::index::hnsw_index::hnsw::{HNSWIndex, HnswBuildDebugOptions, HnswIndexOpenArgs};
use crate::index::hnsw_index::point_scorer::MAX_BLOCK_GATHER_BYTES;
use crate::json_path::JsonPath;
use crate::payload_json;
use crate::segment::Segment;
use crate::segment_constructor::VectorIndexBuildArgs;
use crate::segment_constructor::simple_segment_constructor::build_simple_segment;
use crate::types::{
    Distance, HnswConfig, HnswGlobalConfig, Memory, PayloadFieldSchema, PayloadSchemaParams,
    PayloadSchemaType, QuantizationConfig, SeqNumberType, TurboQuantBitSize,
    TurboQuantQuantizationConfig, TurboQuantization,
};
use crate::vector_storage::quantized::quantized_vectors::{
    QuantizedVectors, QuantizedVectorsStorageType,
};

const DIM: usize = 8;
const NUM_VECTORS: u64 = 2_000;
/// Distinct payload values. Every value covers 250 points of the fixture,
/// enough for the block it generates to clear `full_scan_threshold`.
const NUM_VALUES: u64 = 8;
const SEED: u64 = 42;

const KEYWORD_KEY: &str = "tenant";
const KEYWORD_TWIN_KEY: &str = "tenant_twin";
const TENANT_KEY: &str = "tenant_flagged";
const TENANT_TWIN_KEY: &str = "tenant_flagged_twin";
const INT_KEY: &str = "bucket";

#[derive(Clone, Copy, Debug)]
pub(super) enum FixtureField {
    Keyword,
    Integer,
    /// Two keyword fields holding the same values.
    TwinKeyword,
    /// Two keyword fields holding the same values, over vectors that cluster
    /// tightly per value. Each value's block is then already well connected in
    /// the main graph, which is what makes the connectivity shortcut skip it.
    ClusteredTwinKeyword,
    /// Two keyword fields carrying `is_tenant`, alongside two plain ones, over
    /// the same clustered vectors. Every field holds identical values, so the
    /// connectivity shortcut would skip any of their blocks - which makes the
    /// tenant guard the only thing keeping the tenant fields' blocks.
    ClusteredTenantAndPlainKeyword,
    /// One keyword field whose values span a wide range of cardinalities, so the
    /// blocks take very different times to build. That is the shape the drain is
    /// sorted for: the largest block starts first and is still going long after
    /// the rest of the queue is claimed, leaving the pool converged on it.
    SkewedKeyword,
    /// One keyword field, with some points dropped from the ID tracker while
    /// their vectors stay live in vector storage. Those points leave both the
    /// main graph and the payload blocks, so blocks end up smaller than their
    /// payload cardinality.
    KeywordWithStalePoints,
}

/// How a [`FixtureField`] lays out its payload and vectors.
struct FixtureSpec {
    keys: &'static [(&'static str, PayloadSchemaType)],
    /// Points per value. Sums to `NUM_VECTORS`, and every entry has to stay
    /// above the block-generation threshold or it produces no block at all.
    value_sizes: Vec<u64>,
    /// Draw each value's points around a per-value centroid.
    clustered: bool,
    /// Drop every Nth point from the ID tracker after indexing.
    drop_every: Option<u64>,
    /// Keys to index with `is_tenant` set, which exempts them from the
    /// connectivity shortcut.
    tenant_keys: &'static [&'static str],
}

impl FixtureField {
    fn spec(self) -> FixtureSpec {
        const KEYWORD: PayloadSchemaType = PayloadSchemaType::Keyword;
        let uniform = vec![NUM_VECTORS / NUM_VALUES; NUM_VALUES as usize];

        let (keys, value_sizes, clustered, drop_every): (&[_], _, _, _) = match self {
            // Keyword values produce one block each.
            FixtureField::Keyword => (&[(KEYWORD_KEY, KEYWORD)], uniform, false, None),
            // Integer values produce overlapping range blocks, as they do for
            // the numeric fields that dominate this stage in production.
            FixtureField::Integer => (
                &[(INT_KEY, PayloadSchemaType::Integer)],
                uniform,
                false,
                None,
            ),
            FixtureField::TwinKeyword => (
                &[(KEYWORD_KEY, KEYWORD), (KEYWORD_TWIN_KEY, KEYWORD)],
                uniform,
                false,
                None,
            ),
            FixtureField::ClusteredTwinKeyword => (
                &[(KEYWORD_KEY, KEYWORD), (KEYWORD_TWIN_KEY, KEYWORD)],
                uniform,
                true,
                None,
            ),
            // Two of each kind, because `indexed_fields` hands back a `HashMap`
            // and the field at `index_pos == 0` has the shortcut off whoever it
            // is. With two of each, at most one can be first, so a field of each
            // kind always sits where the shortcut is live - no matter the order.
            FixtureField::ClusteredTenantAndPlainKeyword => (
                &[
                    (KEYWORD_KEY, KEYWORD),
                    (KEYWORD_TWIN_KEY, KEYWORD),
                    (TENANT_KEY, KEYWORD),
                    (TENANT_TWIN_KEY, KEYWORD),
                ],
                uniform,
                true,
                None,
            ),
            // Sums to NUM_VECTORS, spans a 4.5x range of block sizes, and stays
            // under the percolation ceiling of `total / avg_links * 4`.
            FixtureField::SkewedKeyword => (
                &[(KEYWORD_KEY, KEYWORD)],
                vec![450, 400, 350, 300, 250, 150, 100],
                false,
                None,
            ),
            FixtureField::KeywordWithStalePoints => {
                (&[(KEYWORD_KEY, KEYWORD)], uniform, false, Some(9))
            }
        };

        // Which of `keys` are indexed with `is_tenant` set.
        // Listed rather than wildcarded, so a new fixture has to decide whether its
        // fields are tenant fields instead of silently defaulting to "no".
        let tenant_keys: &[&str] = match self {
            FixtureField::ClusteredTenantAndPlainKeyword => &[TENANT_KEY, TENANT_TWIN_KEY],
            FixtureField::Keyword
            | FixtureField::Integer
            | FixtureField::TwinKeyword
            | FixtureField::ClusteredTwinKeyword
            | FixtureField::SkewedKeyword
            | FixtureField::KeywordWithStalePoints => &[],
        };

        debug_assert_eq!(value_sizes.iter().sum::<u64>(), NUM_VECTORS);
        FixtureSpec {
            keys,
            value_sizes,
            clustered,
            drop_every,
            tenant_keys,
        }
    }

    fn is_keyword(self) -> bool {
        !matches!(self, FixtureField::Integer)
    }
}

/// Value assigned to each point, indexed by internal offset.
///
/// Points are upserted in id order into a fresh segment, so external ids and
/// internal offsets coincide and this doubles as the block membership map.
fn value_assignment(spec: &FixtureSpec) -> Vec<u64> {
    let mut values = Vec::with_capacity(NUM_VECTORS as usize);
    for (value, &size) in spec.value_sizes.iter().enumerate() {
        values.extend(std::iter::repeat_n(value as u64, size as usize));
    }
    values
}

/// The blocks a keyword fixture generates, as internal point offsets.
fn value_blocks(field: FixtureField) -> Vec<Vec<PointOffsetType>> {
    assert!(field.is_keyword(), "range blocks are not one per value");
    let spec = field.spec();
    let values = value_assignment(&spec);

    (0..spec.value_sizes.len() as u64)
        .map(|value| {
            values
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v == value)
                .map(|(point, _)| point as PointOffsetType)
                .collect()
        })
        .collect()
}

/// Every link container of the final graph, per point and per level, plus the
/// entry point list. Compared verbatim, so any reordering shows up.
type GraphSnapshot = (Vec<Vec<Vec<PointOffsetType>>>, String);

/// A fixture segment with one or two indexed payload fields.
///
/// Only the single-field cases are comparable across builds. `indexed_fields`
/// hands back a `HashMap`, so which field is indexed first - and therefore the
/// order in which their links are merged - varies from build to build. Giving
/// two fields identical values does not rescue it either: each field's index
/// carries its own value map, so the two fields generate their blocks in
/// different orders and are not interchangeable.
///
/// The two-field cases are therefore only used by tests that look at a single
/// build, where they buy coverage of a multi-field segment and of the
/// connectivity shortcut, which only runs from the second field on.
fn build_fixture_segment(dir: &Path, field: FixtureField) -> Segment {
    let mut rng = StdRng::seed_from_u64(SEED);
    let hw_counter = HardwareCounterCell::new();

    let spec = field.spec();
    let values = value_assignment(&spec);

    // Tight enough that a value's points are each other's nearest neighbours,
    // so the main graph links them to one another and barely to anything else.
    const CLUSTER_NOISE: f32 = 0.02;
    let centroids: Vec<Vec<f32>> = (0..spec.value_sizes.len())
        .map(|_| random_vector(&mut rng, DIM))
        .collect();

    let mut segment = build_simple_segment(dir, DIM, Distance::Cosine).unwrap();
    for n in 0..NUM_VECTORS {
        let value = values[n as usize];
        let vector = if spec.clustered {
            let noise = random_vector(&mut rng, DIM);
            centroids[value as usize]
                .iter()
                .zip(&noise)
                .map(|(centroid, noise)| centroid + CLUSTER_NOISE * noise)
                .collect()
        } else {
            random_vector(&mut rng, DIM)
        };

        let payload = if field.is_keyword() {
            let mut payload = payload_json! {};
            for &(key, _) in spec.keys {
                payload
                    .0
                    .insert(key.to_owned(), format!("tenant-{value}").into());
            }
            payload
        } else {
            payload_json! {INT_KEY: value as i64}
        };

        segment
            .upsert_point(
                n as SeqNumberType,
                n.into(),
                only_default_vector(&vector),
                &hw_counter,
            )
            .unwrap();
        segment
            .set_full_payload(n as SeqNumberType, n.into(), &payload, &hw_counter)
            .unwrap();
    }

    for &(key, schema) in spec.keys {
        // A bare `PayloadSchemaType` can never be a tenant field, so the tenant
        // keys have to go in as full params.
        let schema: PayloadFieldSchema = if spec.tenant_keys.contains(&key) {
            debug_assert_eq!(
                schema,
                PayloadSchemaType::Keyword,
                "only keyword fields carry `is_tenant`",
            );
            PayloadFieldSchema::FieldParams(PayloadSchemaParams::Keyword(KeywordIndexParams {
                is_tenant: Some(true),
                ..Default::default()
            }))
        } else {
            schema.into()
        };
        segment
            .payload_index
            .borrow_mut()
            .set_indexed(&JsonPath::new(key), schema, &hw_counter)
            .unwrap();
    }

    if let Some(drop_every) = spec.drop_every {
        // Drops the ID tracker mapping and nothing else: the vector stays live
        // in vector storage and the payload index keeps the point, so the point
        // still turns up as a block member while counting as deleted. That is
        // exactly the state `block_deleted_flags` exists to describe.
        //
        // `Segment::delete_point` would not do: it also clears the payload row,
        // which would take the point out of its block entirely.
        let mut id_tracker = segment.id_tracker.borrow_mut();
        for n in (0..NUM_VECTORS).step_by(drop_every as usize) {
            id_tracker.drop(n.into()).unwrap();
        }
    }

    segment
}

fn hnsw_config(threads: usize) -> HnswConfig {
    HnswConfig {
        memory: Some(Memory::Cached),
        m: 8,
        ef_construct: 16,
        // In KB. Vectors are 8 `f32`s, so this lands far below the per-value
        // cardinality of the fixture and every value produces a block.
        full_scan_threshold: 1,
        max_indexing_threads: threads,
        on_disk: None,
        payload_m: None,
        inline_storage: None,
    }
}

/// What the payload-block stage did during one build.
#[derive(Debug, Default)]
struct BlockStats {
    built: usize,
    skipped_by_connectivity: usize,
    /// Blocks whose vectors were copied into a block-local scoring buffer.
    gathered: usize,
    /// Block index each block was filtered under, in call order.
    indices: Vec<usize>,
    /// Field of every block the connectivity shortcut skipped.
    skipped_fields: Vec<JsonPath>,
    /// Blocks that could have gathered but lost the allowance, or bailed part way.
    gather_declined: usize,
    /// Blocks whose storage cannot be gathered at all.
    gather_unavailable: usize,
}

/// Non-default knobs for one build. `None` leaves the production default.
#[derive(Clone, Copy, Default)]
struct BuildOverrides {
    force_legacy_payload_blocks: bool,
    /// Build over TurboQuant 4-bit quantized vectors instead of the raw dense
    /// storage - the production configuration of the block gather.
    quantize: bool,
    /// Stage gather allowance. `None` takes production's, sized from the pool;
    /// a small value forces the contention path.
    gather_budget_bytes: Option<usize>,
    /// Per-block gather cap. `None` takes production's; a small value forces the
    /// oversized path without a fixture of tens of MiB.
    max_block_gather_bytes: Option<usize>,
}

/// Build an HNSW index over `segment` and snapshot the graph before it is
/// serialized.
fn build_and_snapshot(
    segment: &Segment,
    force_legacy_payload_blocks: bool,
    threads: usize,
) -> (GraphSnapshot, BlockStats) {
    build_and_snapshot_opts(
        segment,
        threads,
        BuildOverrides {
            force_legacy_payload_blocks,
            ..Default::default()
        },
    )
}

fn build_and_snapshot_opts(
    segment: &Segment,
    threads: usize,
    overrides: BuildOverrides,
) -> (GraphSnapshot, BlockStats) {
    let BuildOverrides {
        force_legacy_payload_blocks,
        quantize,
        gather_budget_bytes,
        max_block_gather_bytes,
    } = overrides;

    let stopped = AtomicBool::new(false);
    let hnsw_dir = Builder::new().prefix("hnsw_dir").tempdir().unwrap();
    let mut rng = StdRng::seed_from_u64(SEED);

    let quantized_vectors = if quantize {
        let config = QuantizationConfig::Turbo(TurboQuantization {
            turbo: TurboQuantQuantizationConfig {
                always_ram: None,
                bits: Some(TurboQuantBitSize::Bits4),
                memory: Some(Memory::Pinned),
            },
        });
        let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow();
        let quantized = QuantizedVectors::create(
            &storage,
            &config,
            QuantizedVectorsStorageType::Immutable,
            hnsw_dir.path(),
            1,
            &stopped,
        )
        .unwrap();
        Arc::new(AtomicRefCell::new(Some(quantized)))
    } else {
        Default::default()
    };

    let snapshot = Mutex::new(None);
    let blocks_built = AtomicUsize::new(0);
    let blocks_skipped = AtomicUsize::new(0);
    let blocks_gathered = AtomicUsize::new(0);
    let block_indices = Mutex::new(Vec::new());
    let skipped_fields = Mutex::new(Vec::new());
    let gather_declined = AtomicUsize::new(0);
    let gather_unavailable = AtomicUsize::new(0);

    let hnsw_config = hnsw_config(threads);
    let permit_cpu_count = get_num_indexing_threads(hnsw_config.max_indexing_threads);
    let permit = Arc::new(ResourcePermit::dummy(permit_cpu_count as u32));

    HNSWIndex::build_with_debug_options(
        HnswIndexOpenArgs {
            path: hnsw_dir.path(),
            id_tracker: segment.id_tracker.clone(),
            vector_storage: segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_storage
                .clone(),
            quantized_vectors,
            payload_index: segment.payload_index.clone(),
            hnsw_config,
        },
        VectorIndexBuildArgs {
            permit,
            old_indices: &[],
            gpu_device: None,
            rng: &mut rng,
            stopped: &stopped,
            hnsw_global_config: &HnswGlobalConfig::default(),
            feature_flags: FeatureFlags::default(),
            progress: ProgressTracker::new_for_test(),
        },
        HnswBuildDebugOptions {
            force_legacy_payload_blocks,
            blocks_built: Some(&blocks_built),
            blocks_skipped_by_connectivity: Some(&blocks_skipped),
            blocks_gathered: Some(&blocks_gathered),
            block_indices: Some(&block_indices),
            skipped_fields: Some(&skipped_fields),
            gather_budget_bytes,
            max_block_gather_bytes,
            gather_declined: Some(&gather_declined),
            gather_unavailable: Some(&gather_unavailable),
            inspect_builder: Some(&|builder| {
                *snapshot.lock() = Some((
                    builder.links_snapshot(),
                    format!("{:?}", *builder.get_entry_points()),
                ));
            }),
        },
    )
    .unwrap();

    let stats = BlockStats {
        built: blocks_built.into_inner(),
        skipped_by_connectivity: blocks_skipped.into_inner(),
        gathered: blocks_gathered.into_inner(),
        indices: block_indices.into_inner(),
        skipped_fields: skipped_fields.into_inner(),
        gather_declined: gather_declined.into_inner(),
        gather_unavailable: gather_unavailable.into_inner(),
    };
    (
        snapshot.into_inner().expect("builder was never inspected"),
        stats,
    )
}

/// The compact per-block builder must produce exactly the graph the legacy
/// segment-sized builder produces, link for link and in the same order.
///
/// Single-field fixtures only: see [`build_fixture_segment`] for why anything
/// with two indexed fields cannot be compared across builds.
/// The `stale_points` case covers a segment carrying ID-tracker-deleted points
/// whose vectors are still live. It does *not* reach `block_deleted_flags`:
/// `iter_filtered_points` drops those points before they can become block
/// members, so the flags come out all-false either way. See that function for
/// why they are still derived.
#[rstest]
#[case::keyword(FixtureField::Keyword)]
#[case::integer(FixtureField::Integer)]
#[case::stale_points(FixtureField::KeywordWithStalePoints)]
fn test_compact_payload_blocks_match_legacy(#[case] field: FixtureField) {
    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let segment = build_fixture_segment(dir.path(), field);

    let (legacy, legacy_stats) = build_and_snapshot(&segment, true, 1);
    let (compact, compact_stats) = build_and_snapshot(&segment, false, 1);

    assert!(
        legacy_stats.built >= 5,
        "fixture is vacuous, only {} payload blocks were built",
        legacy_stats.built,
    );
    assert_eq!(legacy_stats.built, compact_stats.built);

    // Without these, the block-vector copy and the cross-field queue can both
    // silently stop engaging while every other assertion here still passes - the
    // optimisation becomes a no-op that looks identical from the outside.
    // Every vector in the fixture together is orders of magnitude under one
    // block's cap, so the stage allowance cannot be what binds here. Without this
    // the equality below would quietly become a statement about the fixture size
    // rather than about the copy still engaging.
    const FIXTURE_BYTES: usize = NUM_VECTORS as usize * DIM * size_of::<f32>();
    const _: () = assert!(FIXTURE_BYTES < MAX_BLOCK_GATHER_BYTES);

    assert_eq!(
        compact_stats.gathered, compact_stats.built,
        "every built block should have gathered its vectors: {compact_stats:?}",
    );
    assert_eq!(
        legacy_stats.gathered, 0,
        "the legacy path has no block-local buffer to gather into: {legacy_stats:?}",
    );

    // Both paths must number a field's blocks the same way, or they seed the
    // connectivity shortcut differently and can skip different blocks. Single
    // field here, so the numbering is simply generation order.
    let expected_indices: Vec<usize> = (0..legacy_stats.indices.len()).collect();
    assert_eq!(legacy_stats.indices, expected_indices);
    assert_eq!(compact_stats.indices, expected_indices);

    let (legacy_links, legacy_entry_points) = legacy;
    let (compact_links, compact_entry_points) = compact;

    assert_eq!(legacy_links.len(), compact_links.len());
    for (point_id, (legacy, compact)) in legacy_links.iter().zip(&compact_links).enumerate() {
        assert_eq!(
            legacy, compact,
            "links of point {point_id} differ between the legacy and compact block builders",
        );
    }
    assert_eq!(legacy_entry_points, compact_entry_points);
}

/// The compact path must leave the graph usable: no duplicate links, no links
/// to points outside the segment, no self-links.
#[rstest]
#[case::keyword(FixtureField::Keyword)]
#[case::integer(FixtureField::Integer)]
#[case::twin_keyword(FixtureField::TwinKeyword)]
#[case::stale_points(FixtureField::KeywordWithStalePoints)]
fn test_compact_payload_blocks_structural_invariants(#[case] field: FixtureField) {
    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let segment = build_fixture_segment(dir.path(), field);

    let ((links, _), stats) = build_and_snapshot(&segment, false, 1);
    assert!(stats.built >= 5);

    assert_links_are_sane(&links);
}

/// The connectivity shortcut has to actually skip blocks somewhere in the
/// suite, otherwise nothing pins down its predicate.
///
/// Clustered vectors are what make that happen: each value's points are one
/// another's nearest neighbours, so the main graph already connects them and a
/// block's own connectivity estimate lands above the whole-graph estimate the
/// threshold is drawn from. The count is totalled across fields because which
/// of the twin fields is visited first is not deterministic.
#[test]
fn test_connectivity_shortcut_skips_well_connected_blocks() {
    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let segment = build_fixture_segment(dir.path(), FixtureField::ClusteredTwinKeyword);

    let ((links, _), stats) = build_and_snapshot(&segment, false, 1);

    assert!(
        stats.skipped_by_connectivity >= 4,
        "connectivity shortcut skipped only {} blocks, so its predicate is untested: {stats:?}",
        stats.skipped_by_connectivity,
    );
    assert!(
        stats.built >= 5,
        "fixture skipped everything, so nothing was built: {stats:?}",
    );
    assert_links_are_sane(&links);
}

/// Fraction of `points` that fall in the largest connected component of the
/// level-0 graph restricted to `points`.
fn largest_component_fraction(
    links: &[Vec<Vec<PointOffsetType>>],
    points: &[PointOffsetType],
) -> f64 {
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    let local: HashMap<PointOffsetType, usize> = points.iter().copied().zip(0..).collect();
    let mut parent: Vec<usize> = (0..points.len()).collect();

    for (&global, &from) in &local {
        for &link in &links[global as usize][0] {
            if let Some(&to) = local.get(&link) {
                let (from, to) = (find(&mut parent, from), find(&mut parent, to));
                if from != to {
                    parent[from] = to;
                }
            }
        }
    }

    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for point in 0..points.len() {
        *sizes.entry(find(&mut parent, point)).or_default() += 1;
    }
    sizes.values().copied().max().unwrap_or(0) as f64 / points.len() as f64
}

/// Building payload blocks concurrently must leave every block as connected as
/// the sequential build leaves it, and must not corrupt any link container.
///
/// The two builds are not compared link for link: the main graph is itself
/// built in parallel, so a multi-threaded build differs from a single-threaded
/// one before the payload stage even starts. `merge_block` is pinned down
/// exactly by `test_merge_block_concurrent_matches_sequential` instead.
///
/// Both fixture shapes are covered because they exercise different drains: the
/// uniform field yields same-sized blocks, while the skewed one yields a spread
/// of cardinalities, so one block's insertions run long after the rest of the
/// queue is claimed and the pool converges on it.
///
/// `more_threads_than_blocks` is the case the drain is most exposed to: with
/// nothing left to claim, every idle thread ends up inside a block that is
/// already running, so a block's points are inserted by many threads at once.
/// That is the configuration where concurrent insertion could plausibly leave a
/// subgraph fragmented, which is exactly what this asserts it does not.
#[rstest]
#[case::uniform(FixtureField::Keyword, 4)]
#[case::mixed_block_sizes(FixtureField::SkewedKeyword, 4)]
#[case::more_threads_than_blocks(FixtureField::SkewedKeyword, 16)]
fn test_parallel_payload_blocks_stay_connected(
    #[case] field: FixtureField,
    #[case] threads: usize,
) {
    const ITERATIONS: usize = 10;

    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let segment = build_fixture_segment(dir.path(), field);
    let blocks = value_blocks(field);

    let ((sequential, _), sequential_stats) = build_and_snapshot(&segment, false, 1);
    assert!(
        sequential_stats.built >= 5,
        "fixture is vacuous, only {} payload blocks were built",
        sequential_stats.built,
    );

    let sequential_fractions: Vec<f64> = blocks
        .iter()
        .map(|points| largest_component_fraction(&sequential, points))
        .collect();
    assert!(
        sequential_fractions.iter().all(|&fraction| fraction > 0.9),
        "sequential build left blocks fragmented: {sequential_fractions:?}",
    );

    for iteration in 0..ITERATIONS {
        let ((parallel, _), parallel_stats) = build_and_snapshot(&segment, false, threads);
        assert_eq!(
            parallel_stats.built, sequential_stats.built,
            "iteration {iteration}: a different set of blocks was built",
        );
        // The concurrent reserve path: with `threads` blocks live at once every one
        // of them still gets its buffer, so the shared allowance is not silently
        // turning the copy off exactly where the drain is widest.
        assert_eq!(
            parallel_stats.gathered, parallel_stats.built,
            "iteration {iteration}: the block-vector copy stopped engaging under \
             {threads} threads",
        );

        assert_links_are_sane(&parallel);

        for (value, (points, sequential)) in blocks.iter().zip(&sequential_fractions).enumerate() {
            let fraction = largest_component_fraction(&parallel, points);
            assert!(
                fraction >= sequential - 0.01,
                "iteration {iteration}, block {value}: connectivity {fraction} below the \
                 sequential build's {sequential}",
            );
        }
    }
}

/// Draining every field's blocks through one parallel scope must leave the graph
/// intact and must still filter each block exactly once, under its own field's
/// numbering.
///
/// Unlike the serial case this is not a census-equality check. The connectivity
/// shortcut reads the graph as it stands when a block is filtered, and
/// interleaving the fields changes how much of an earlier field has merged by
/// then - so a block can be skipped by one schedule and built by another. That
/// was already true between blocks of one field on the parallel path; the
/// unified queue extends it across fields. What is pinned down here is what does
/// not move: every block is still filtered once, under the same index.
#[rstest]
#[case::twin_keyword(FixtureField::TwinKeyword)]
#[case::clustered_twin_keyword(FixtureField::ClusteredTwinKeyword)]
fn test_unified_block_queue_parallel_stays_sane(#[case] field: FixtureField) {
    const ITERATIONS: usize = 6;

    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let segment = build_fixture_segment(dir.path(), field);

    let (_, serial_stats) = build_and_snapshot_opts(&segment, 1, BuildOverrides::default());
    assert!(
        serial_stats.built >= 5,
        "fixture is vacuous: {serial_stats:?}"
    );

    let mut expected_indices = serial_stats.indices.clone();
    expected_indices.sort_unstable();

    for iteration in 0..ITERATIONS {
        let ((links, _), stats) = build_and_snapshot_opts(&segment, 4, BuildOverrides::default());

        assert_links_are_sane(&links);

        let mut indices = stats.indices.clone();
        indices.sort_unstable();
        assert_eq!(
            indices, expected_indices,
            "iteration {iteration}: blocks were filtered a different number of times, \
             or under different indices: {stats:?}",
        );
        assert!(
            stats.built > 0,
            "iteration {iteration}: nothing was built: {stats:?}",
        );
    }
}

pub(super) fn assert_links_are_sane(links: &[Vec<Vec<PointOffsetType>>]) {
    for (point_id, levels) in links.iter().enumerate() {
        for (level, container) in levels.iter().enumerate() {
            let mut seen = container.clone();
            seen.sort_unstable();
            let before = seen.len();
            seen.dedup();
            assert_eq!(
                before,
                seen.len(),
                "point {point_id} level {level} has duplicate links",
            );
            assert!(
                container.iter().all(|&link| (link as usize) < links.len()),
                "point {point_id} level {level} links outside the segment",
            );
            assert!(
                container.iter().all(|&link| link as usize != point_id),
                "point {point_id} level {level} links to itself",
            );
        }
    }
}

/// Number of payload values a fixture's fields are split across.
fn spec_value_count(field: FixtureField) -> usize {
    field.spec().value_sizes.len()
}

/// The connectivity shortcut must never apply to a tenant field.
///
/// Payload-block links exist mostly so that a tenant's own points stay reachable
/// under its filter. Skipping a tenant's block because the *main* graph already
/// connects those points trades away exactly the recall the field was indexed for,
/// and for a multi-tenant collection that is the highest blast radius in the
/// stage. The whole guard is one boolean - `!is_tenant` in `check_connectivity` -
/// and until this fixture existed no test marked any field as a tenant field, so
/// deleting the guard left every test green.
///
/// Written to be order-independent on purpose. `indexed_fields` hands back a
/// `HashMap`, so which field lands at `index_pos == 0` - where the shortcut is off
/// for tenant and plain fields alike - is not fixed from run to run. The fixture
/// carries two fields of each kind, and only one field can be first, so whatever
/// the order at least one tenant field and at least one plain field sit at
/// `index_pos > 0` where the shortcut is live.
#[test]
fn test_connectivity_shortcut_never_skips_a_tenant_field() {
    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let segment = build_fixture_segment(dir.path(), FixtureField::ClusteredTenantAndPlainKeyword);

    let (_, stats) = build_and_snapshot(&segment, false, 1);

    let tenant_keys = [JsonPath::new(TENANT_KEY), JsonPath::new(TENANT_TWIN_KEY)];
    let (tenant_skips, plain_skips): (Vec<_>, Vec<_>) = stats
        .skipped_fields
        .iter()
        .partition(|field| tenant_keys.contains(field));

    // Without this the assertion below passes on a build where the shortcut never
    // fired at all, which would prove nothing about the guard.
    assert!(
        !plain_skips.is_empty(),
        "the shortcut skipped nothing, so the tenant half of this test is vacuous: \
         {stats:?}",
    );
    assert!(
        tenant_skips.is_empty(),
        "the connectivity shortcut skipped {} block(s) of a tenant field: {:?}",
        tenant_skips.len(),
        tenant_skips,
    );
    // Tenant blocks are not merely un-skipped, they are built: a field whose
    // blocks all fell out earlier would satisfy the assertion above for the wrong
    // reason.
    assert!(
        stats.built >= spec_value_count(FixtureField::ClusteredTenantAndPlainKeyword),
        "expected at least one block built per value of the tenant fields, got {}: {stats:?}",
        stats.built,
    );
}

/// What happens when the stage allowance runs out.
///
/// Declining a copy is supposed to cost nothing but time: the scorer falls back to
/// reading the storage by global id, and the existing scorer-level tests pin those
/// scores as bitwise identical. This asserts the consequence of that at *build*
/// level, which is much stronger than comparing scores - a fully declined build
/// must produce the same graph, link for link, as one that copied every block.
///
/// Production sizes the allowance from the pool, so reaching contention honestly
/// would need a fixture of tens of MiB. The allowance is overridden instead, which
/// is the only part of the path that has to be simulated: that an exhausted
/// `GatherBudget` declines is covered by its own unit tests.
///
/// Also pins the two counters apart. Declining because the allowance was spent and
/// declining because the storage has no byte layout are different facts with
/// different fixes, and the build reports them on separate lines.
#[rstest]
#[case::dense(false)]
#[case::quantized(true)]
fn test_gather_declines_when_the_allowance_is_spent(#[case] quantize: bool) {
    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let segment = build_fixture_segment(dir.path(), FixtureField::Keyword);

    // One byte cannot hold any block's vectors, so every block is declined - and
    // deterministically so, whatever order the drain reaches them in.
    let starved = BuildOverrides {
        quantize,
        gather_budget_bytes: Some(1),
        ..Default::default()
    };
    let fed = BuildOverrides {
        quantize,
        ..Default::default()
    };

    let ((declined_links, declined_entries), declined_stats) =
        build_and_snapshot_opts(&segment, 1, starved);
    let ((gathered_links, gathered_entries), gathered_stats) =
        build_and_snapshot_opts(&segment, 1, fed);

    assert!(
        gathered_stats.built >= 5,
        "fixture is vacuous, only {} blocks built",
        gathered_stats.built,
    );
    assert_eq!(
        gathered_stats.built, declined_stats.built,
        "starving the allowance changed which blocks were built",
    );

    // The starved build must reach the decline path, and must attribute it to the
    // allowance rather than to the storage - this storage gathers perfectly well.
    assert_eq!(
        declined_stats.gathered, 0,
        "a one-byte allowance still gathered: {declined_stats:?}",
    );
    assert_eq!(
        declined_stats.gather_declined, declined_stats.built,
        "every built block should have been declined for want of allowance: \
         {declined_stats:?}",
    );
    assert_eq!(
        declined_stats.gather_unavailable, 0,
        "contention was misreported as an unusable storage: {declined_stats:?}",
    );

    // And the control: with the production allowance nothing is declined at all.
    assert_eq!(
        gathered_stats.gathered, gathered_stats.built,
        "the fed build failed to gather: {gathered_stats:?}",
    );
    assert_eq!(gathered_stats.gather_declined, 0, "{gathered_stats:?}");
    assert_eq!(gathered_stats.gather_unavailable, 0, "{gathered_stats:?}");

    // The payoff: scoring from the copy and scoring from the storage are the same
    // computation, so the graphs are identical and not merely similar.
    assert_eq!(
        declined_links, gathered_links,
        "declining the copy changed the graph",
    );
    assert_eq!(declined_entries, gathered_entries);
}

/// A block too big for the per-block cap is a fact about that block, not about
/// the storage: this storage gathers every other block fine. Reporting it as an
/// ungatherable storage would send a reader looking at the wrong thing, and the
/// two causes are counted and logged separately precisely so they can be told
/// apart.
#[rstest]
#[case::dense(false)]
#[case::quantized(true)]
fn test_oversized_blocks_are_declined_not_reported_unavailable(#[case] quantize: bool) {
    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let segment = build_fixture_segment(dir.path(), FixtureField::Keyword);

    // One byte is under any block's vectors, so every block is over the cap.
    let capped = BuildOverrides {
        quantize,
        max_block_gather_bytes: Some(1),
        ..Default::default()
    };
    let ((capped_links, capped_entries), capped_stats) =
        build_and_snapshot_opts(&segment, 1, capped);
    let ((gathered_links, gathered_entries), gathered_stats) = build_and_snapshot_opts(
        &segment,
        1,
        BuildOverrides {
            quantize,
            ..Default::default()
        },
    );

    assert!(
        gathered_stats.built >= 5,
        "fixture is vacuous, only {} blocks built",
        gathered_stats.built,
    );
    assert_eq!(capped_stats.gathered, 0, "{capped_stats:?}");
    assert_eq!(
        capped_stats.gather_declined, capped_stats.built,
        "an oversized block should be declined: {capped_stats:?}",
    );
    // The point of the test: this storage gathers perfectly well, and said so on
    // the control build below. Blaming it would be the wrong diagnosis.
    assert_eq!(
        capped_stats.gather_unavailable, 0,
        "an oversized block was reported as an ungatherable storage: {capped_stats:?}",
    );
    assert_eq!(gathered_stats.gathered, gathered_stats.built);
    assert_eq!(gathered_stats.gather_unavailable, 0);

    // And declining still has to leave the graph alone.
    assert_eq!(
        capped_links, gathered_links,
        "declining the copy changed the graph",
    );
    assert_eq!(capped_entries, gathered_entries);
}
