//! Tests for the `wand_pruning` switch, surfaced as `InvertedIndexRam::maintain_max_next_weight`.
//!
//! The switch is a pure performance trade, so the property under test is that it never changes an
//! answer. It does that by governing both halves of the mechanism at once:
//!
//! - writes skip [`PostingList::propagate_max_next_weight_to_the_left`], which is what makes bulk
//!   sparse ingest quadratic in posting list length, and
//! - searches report the resulting bounds unusable, so pruning is never attempted over them.
//!
//! Splitting those two apart is the dangerous failure: skipping propagation leaves bounds that are
//! stale — too *low* on an append, and too *high* after a delete. The too-low direction is the
//! dangerous one: it makes `prune_longest_posting_list` skip candidates that should have entered
//! the top-k, a silent recall loss with no error anywhere. The tests below pin the coupling, pin
//! both directions of staleness, and deliberately assert that the bounds really do go stale, so
//! that the equivalence test cannot pass vacuously if propagation were left switched on.

use std::sync::atomic::AtomicBool;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::types::{PointOffsetType, ScoredPointOffset};
use rand::SeedableRng;

use crate::SearchScratch;
use crate::common::sparse_vector::RemappedSparseVector;
use crate::index::inverted_index::InvertedIndex;
use crate::index::inverted_index::inverted_index_ram::InvertedIndexRam;
use crate::index::inverted_index::inverted_index_ram_builder::InvertedIndexBuilder;
use crate::index::posting_list_common::DEFAULT_MAX_NEXT_WEIGHT;
use crate::index::search_context::SearchContext;
use crate::index::tests::common::{match_all, random_sparse_vector};

/// Must exceed `ADVANCE_BATCH_SIZE` (10_000, `search_context.rs:25`), or this whole file is
/// vacuous with respect to pruning.
///
/// `SearchContext::search` walks `ADVANCE_BATCH_SIZE` contiguous ids per iteration and only reaches
/// the pruning branch *after* a batch. With contiguous ids and a corpus that fits in one batch, the
/// first `advance_batch` exhausts every posting list and the loop breaks at `search_context.rs:322`
/// before pruning is ever considered — so the two arms run byte-identical code and every
/// "on == off" assertion below passes no matter what the read-side gate does. Measured at
/// COUNT=5000: 200 searches, 0 prune calls, all 3300 scores bitwise identical, and deleting the
/// `max_next_weight_reliable()` conjunct from the gate did not fail a single test.
///
/// `pruning_actually_engages_in_this_fixture` fails if this is ever lowered back under the batch
/// size.
const COUNT: usize = 25_000;
const DENSITY: usize = 8;
const VOCAB1: usize = 200;
const VOCAB2: usize = 400;

/// Relative tolerance for comparing scores between the two arms.
///
/// Scores must NOT be compared exactly. Pruning calls
/// `promote_longest_posting_lists_to_the_front`, which the non-pruning arm never reaches, so the
/// two arms accumulate the same products in a different order and the `f32` sum can differ in its
/// last bit. Reproduced with this fixture: the same point scored 4.002059 against 4.0020585, one
/// ulp apart. An exact `assert_eq!` on the score reports "results diverged when wand pruning was
/// disabled", which reads as a recall bug and is not one — it happens to pass at a few hundred
/// points, where the sums are short, and starts failing as the fixture grows.
///
/// Ids and their order are still compared exactly. An over-prune DROPS a point from the result,
/// which changes the id sequence and is caught outright.
const SCORE_TOL: f32 = 1e-5;

fn assert_same_results(expected: &[ScoredPointOffset], actual: &[ScoredPointOffset], top: usize) {
    let ids = |v: &[ScoredPointOffset]| v.iter().map(|s| s.idx).collect::<Vec<_>>();
    assert_eq!(
        ids(expected),
        ids(actual),
        "results diverged with top={top} when wand pruning was disabled",
    );
    for (a, b) in expected.iter().zip(actual.iter()) {
        let scale = a.score.abs().max(b.score.abs()).max(f32::MIN_POSITIVE);
        assert!(
            (a.score - b.score).abs() <= SCORE_TOL * scale,
            "score for point {} differed by more than f32 noise with top={top}: {} vs {}",
            a.idx,
            a.score,
            b.score,
        );
    }
}

fn corpus(seed: u64, count: usize) -> Vec<(PointOffsetType, RemappedSparseVector)> {
    let mut rnd = rand::rngs::StdRng::seed_from_u64(seed);
    (0..count)
        .map(|i| {
            (
                i as PointOffsetType,
                random_sparse_vector(&mut rnd, DENSITY, VOCAB1, VOCAB2),
            )
        })
        .collect()
}

/// Build through the mutable `upsert` path, which is the only path that maintains the bound
/// incrementally — and therefore the only one the switch affects. The one-pass builder always
/// computes exact bounds regardless.
fn build_via_upsert(
    corpus: &[(PointOffsetType, RemappedSparseVector)],
    maintain: bool,
) -> InvertedIndexRam {
    let mut index = InvertedIndexRam::empty();
    index.set_maintain_max_next_weight(maintain);
    for (id, vector) in corpus {
        index.upsert(*id, vector.clone(), None);
    }
    index
}

fn search(
    index: &InvertedIndexRam,
    query: RemappedSparseVector,
    top: usize,
) -> Vec<ScoredPointOffset> {
    let is_stopped = AtomicBool::new(false);
    let accumulator = HwMeasurementAcc::new();
    let hardware_counter = accumulator.get_counter_cell();
    let mut scratch = SearchScratch::new_for_test();
    let mut search_context = SearchContext::new(
        query,
        top,
        index,
        &mut scratch,
        &is_stopped,
        &hardware_counter,
    )
    .unwrap();
    search_context.search(&match_all)
}

/// The flag reaches both the write path and the read gate, and does so consistently.
#[test]
fn wand_pruning_flag_gates_reads_and_writes_together() {
    let corpus = corpus(42, COUNT);

    let maintained = build_via_upsert(&corpus, true);
    assert!(maintained.maintain_max_next_weight());
    assert!(
        maintained.max_next_weight_reliable(),
        "an index that maintains the bound must let searches prune with it",
    );

    let skipped = build_via_upsert(&corpus, false);
    assert!(!skipped.maintain_max_next_weight());
    assert!(
        !skipped.max_next_weight_reliable(),
        "an index that skips propagation must never report its bounds usable for pruning; \
         they are stale in an unpredictable direction, and pruning over a too-low one silently \
         drops valid results",
    );
}

/// Guards against the equivalence test below becoming vacuous: skipping propagation has to
/// actually leave the bounds stale, otherwise it is not saving any work.
#[test]
fn skipping_propagation_really_leaves_bounds_stale() {
    let corpus = corpus(42, COUNT);
    let maintained = build_via_upsert(&corpus, true);
    let skipped = build_via_upsert(&corpus, false);

    // Same records in the same order, so any difference is the bound and nothing else.
    assert_eq!(maintained.postings.len(), skipped.postings.len());
    for (a, b) in maintained.postings.iter().zip(skipped.postings.iter()) {
        let ids_a: Vec<_> = a.elements.iter().map(|e| e.record_id).collect();
        let ids_b: Vec<_> = b.elements.iter().map(|e| e.record_id).collect();
        assert_eq!(ids_a, ids_b);
    }

    let stale = maintained
        .postings
        .iter()
        .zip(skipped.postings.iter())
        .flat_map(|(a, b)| a.elements.iter().zip(b.elements.iter()))
        .filter(|(a, b)| a.max_next_weight != b.max_next_weight)
        .count();
    assert!(
        stale > 0,
        "expected skipped propagation to leave stale bounds, found none — the switch is not \
         actually skipping the propagation walk",
    );

    // For THIS fixture, which only ever upserts increasing ids, staleness is one-directional: too
    // low, because the skipped index never accounts for weights written after an element. That is
    // the dangerous direction, and the reason pruning must stay off.
    //
    // It is not the general rule — see `skipping_propagation_can_also_leave_bounds_too_high` — so
    // this assertion is scoped to the append-only shape deliberately. Adding a delete here would
    // make it fail on a state that is not a bug.
    for (a, b) in maintained
        .postings
        .iter()
        .zip(skipped.postings.iter())
        .flat_map(|(a, b)| a.elements.iter().zip(b.elements.iter()))
    {
        assert!(
            b.max_next_weight <= a.max_next_weight,
            "skipped bound {} exceeded the exact bound {} on an append-only fixture",
            b.max_next_weight,
            a.max_next_weight,
        );
    }
}

/// The counterexample to "stale bounds are only ever too low".
///
/// Production disables the switch on an index whose bounds the one-pass build has already made
/// exact, so a subsequent delete strands a bound that was derived from the removed weight: nothing
/// walks it back down. Too-high bounds only ever *under*-prune, so this is the safe direction — but
/// it means the safety argument for the switch cannot be "the bounds are too low", it has to be
/// "the bounds are unusable in either direction", and this test is what stops that wording from
/// silently regressing.
#[test]
fn skipping_propagation_can_also_leave_bounds_too_high() {
    // One dimension, three records, weights chosen so the middle one is the suffix maximum.
    let mut index = InvertedIndexBuilder::new();
    index.add(1, [(7, 1.0)].into());
    index.add(2, [(7, 5.0)].into());
    index.add(3, [(7, 2.0)].into());
    let mut index = index.build();

    // Exact after the one-pass build: record 1's bound is max(5.0, 2.0) = 5.0.
    assert_eq!(index.postings[7].elements[0].max_next_weight, 5.0);

    // Disable maintenance, then remove the record that bound was derived from.
    index.set_maintain_max_next_weight(false);
    index.remove(2, [(7, 5.0)].into());

    // The exact bound for record 1 is now 2.0, but the stored bound is still 5.0 — too HIGH.
    let stored = index.postings[7].elements[0].max_next_weight;
    assert_eq!(
        stored, 5.0,
        "expected the stale bound to survive the delete untouched",
    );
    assert!(
        stored > 2.0,
        "a delete with propagation skipped must be able to leave a bound ABOVE the exact suffix \
         maximum; if this no longer holds, the one-directional wording may be correct again",
    );
    assert!(
        !index.max_next_weight_reliable(),
        "whichever direction the bound drifted, the index must report it unusable",
    );
}

/// The property that makes this switch safe to expose: results do not depend on it.
#[test]
fn disabling_wand_pruning_does_not_change_results() {
    let corpus = corpus(42, COUNT);
    let maintained = build_via_upsert(&corpus, true);
    let skipped = build_via_upsert(&corpus, false);

    let mut rnd = rand::rngs::StdRng::seed_from_u64(7);
    for _ in 0..50 {
        let query = random_sparse_vector(&mut rnd, DENSITY, VOCAB1, VOCAB2);
        // Several `top` values: pruning only engages once the result heap is full, so a `top`
        // larger than the match count would never exercise it.
        for top in [1, 5, 10, 50] {
            let expected = search(&maintained, query.clone(), top);
            let actual = search(&skipped, query.clone(), top);
            assert_same_results(&expected, &actual, top);
        }
    }
}

/// The one-pass builder computes exact bounds whatever the runtime policy is, which is what makes
/// re-enabling the collection parameter safe: a restart rebuilds the mutable index this way.
#[test]
fn one_pass_build_always_produces_usable_bounds() {
    let corpus = corpus(42, COUNT);
    let mut builder = InvertedIndexBuilder::new();
    for (id, vector) in &corpus {
        builder.add(*id, vector.clone());
    }
    let built = builder.build();

    assert!(built.max_next_weight_reliable());

    let upserted = build_via_upsert(&corpus, true);
    for (a, b) in built.postings.iter().zip(upserted.postings.iter()) {
        let bounds_a: Vec<_> = a.elements.iter().map(|e| e.max_next_weight).collect();
        let bounds_b: Vec<_> = b.elements.iter().map(|e| e.max_next_weight).collect();
        assert_eq!(
            bounds_a, bounds_b,
            "one-pass build and incremental maintenance must agree on the bound",
        );
    }
}

/// Non-vacuity guard for every "on == off" assertion in this file.
///
/// Pruning calls `promote_longest_posting_lists_to_the_front`, which reorders the posting-list
/// iterators; the non-pruning arm never reaches it. So when pruning genuinely engages, the two arms
/// accumulate the same products in a different order and at least some `f32` sums differ in their
/// last bit. That bitwise difference is therefore *evidence the pruning path ran* — and its absence
/// is evidence it did not.
///
/// This is the test that fails if `COUNT` drops below `ADVANCE_BATCH_SIZE`, if the batch size grows,
/// or if anything else quietly makes the pruning branch unreachable, at which point the equivalence
/// tests would still pass while proving nothing.
#[test]
fn pruning_actually_engages_in_this_fixture() {
    let corpus = corpus(42, COUNT);
    let maintained = build_via_upsert(&corpus, true);
    let skipped = build_via_upsert(&corpus, false);

    let mut rnd = rand::rngs::StdRng::seed_from_u64(7);
    let mut compared = 0usize;
    let mut bitwise_diffs = 0usize;
    for _ in 0..50 {
        let query = random_sparse_vector(&mut rnd, DENSITY, VOCAB1, VOCAB2);
        let a = search(&maintained, query.clone(), 10);
        let b = search(&skipped, query, 10);
        for (x, y) in a.iter().zip(b.iter()) {
            compared += 1;
            if x.score.to_bits() != y.score.to_bits() {
                bitwise_diffs += 1;
            }
        }
    }
    assert!(compared > 0, "no scores compared; fixture produced no hits");
    assert!(
        bitwise_diffs > 0,
        "the pruning and non-pruning arms produced bitwise-identical scores across {compared} \
         comparisons, which means the pruning branch never ran: the corpus fits in one \
         ADVANCE_BATCH_SIZE batch, so `search` breaks before reaching it. Every equivalence \
         assertion in this file is vacuous in that state. Raise COUNT above the batch size.",
    );
}

/// Pruning must honour the bound, not just be consistent with a non-pruning arm.
///
/// Both arms agreeing proves nothing if a bug makes them agree on the *wrong* answer. This builds
/// the geometry where pruning takes its largest possible skip and buries a high-weight sentinel in
/// the middle of the long posting list: the exact suffix maximum stays above the top-k threshold
/// until the scan passes the sentinel, so a correct implementation may not skip over it. If pruning
/// ever reads a bound that is too low — the failure this whole switch is arranged to prevent — the
/// sentinel is skipped and vanishes from the answer, and the assertion below names it.
///
/// Ported from the live suite `tests/claude/wand_prune_fires.py`, which was the only test anywhere
/// in this change sensitive to that failure.
#[test]
fn pruning_honours_the_bound_and_keeps_a_buried_sentinel() {
    const N: PointOffsetType = 30_000;
    const SENTINEL: PointOffsetType = N / 2;
    const LONG_DIM: u32 = 0;
    const END_DIM_A: u32 = 1;
    const END_DIM_B: u32 = 2;
    const CLUSTER: PointOffsetType = 100;

    // dim 0 in every record at 0.01 (one posting list as long as the corpus, the prune target);
    // dims 1 and 2 only at the two ends, so between the clusters the other iterators sit ~N ids
    // ahead of the long list's cursor — the gap the prune needs, and far wider than one batch.
    let mut corpus: Vec<(PointOffsetType, RemappedSparseVector)> = Vec::with_capacity(N as usize);
    for id in 0..N {
        let mut indices = vec![LONG_DIM];
        let mut values = vec![if id == SENTINEL { 5.0 } else { 0.01 }];
        if !(CLUSTER..N - CLUSTER).contains(&id) {
            // Distinct weights so the top-k has no ties to reorder.
            let rank = if id < CLUSTER {
                id
            } else {
                CLUSTER + (N - 1 - id)
            } as f32;
            indices.push(END_DIM_A);
            values.push(1.0 + rank * 0.001);
            indices.push(END_DIM_B);
            values.push(1.0 + rank * 0.0007);
        }
        corpus.push((id, RemappedSparseVector::new(indices, values).unwrap()));
    }

    let query =
        RemappedSparseVector::new(vec![LONG_DIM, END_DIM_A, END_DIM_B], vec![1.0, 1.0, 1.0])
            .unwrap();

    // Exhaustive ground truth, computed here rather than taken from the other arm.
    let mut expected: Vec<(f32, PointOffsetType)> = corpus
        .iter()
        .map(|(id, v)| {
            let score: f32 = v
                .indices
                .iter()
                .zip(v.values.iter())
                .filter(|(d, _)| query.indices.contains(d))
                .map(|(_, w)| w)
                .sum();
            (score, *id)
        })
        .collect();
    expected.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    let expected_ids: Vec<_> = expected.iter().take(10).map(|(_, id)| *id).collect();
    assert_eq!(
        expected_ids[0], SENTINEL,
        "fixture is set up wrong: the sentinel must be the top hit",
    );

    for maintain in [true, false] {
        let index = build_via_upsert(&corpus, maintain);
        let got: Vec<_> = search(&index, query.clone(), 10)
            .into_iter()
            .map(|s| s.idx)
            .collect();
        assert_eq!(
            got, expected_ids,
            "wand_pruning={maintain}: top-10 did not match the exhaustive ground truth",
        );
        assert!(
            got.contains(&SENTINEL),
            "wand_pruning={maintain}: the buried high-weight sentinel was dropped, which means a \
             prune used a bound below the true suffix maximum",
        );
    }
}

/// Deletes and overwrites must consult the flag too, not just fresh inserts.
///
/// Found by mutation testing: hard-coding `propagate = false` in `InvertedIndexRam::remove`, or in
/// the delete branch of the inherent `upsert` that clears dimensions the new vector dropped, left
/// the entire sparse suite green. Every other test here only ever upserts fresh ids, so the
/// remove path had no coverage at all.
#[test]
fn deletes_and_overwrites_respect_the_flag() {
    // The record being removed must hold the UNIQUE maximum weight, or the test cannot see the
    // mutant: with monotonically increasing weights the suffix maximum always sits at the last
    // element, so removing a middle record changes no bound and skipping propagation is invisible.
    // Weight 99.0 at id 4 is what makes the prefix bounds move when it goes away.
    const WEIGHTS: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 99.0, 5.0, 6.0, 7.0];
    let base: Vec<(PointOffsetType, RemappedSparseVector)> = (0..8)
        .map(|i| {
            (
                i,
                RemappedSparseVector::new(vec![3, 4], vec![WEIGHTS[i as usize], 0.5]).unwrap(),
            )
        })
        .collect();

    // With maintenance ON, a remove must leave the bounds exactly where a fresh one-pass build over
    // the surviving records would put them. A `propagate = false` slipped into `remove` breaks this.
    let mut maintained = build_via_upsert(&base, true);
    maintained.remove(
        4,
        RemappedSparseVector::new(vec![3, 4], vec![99.0, 0.5]).unwrap(),
    );
    // Every bound left of id 4 was 99.0; the exact value after the removal is 7.0.
    assert!(
        maintained.postings[3].elements[0].max_next_weight < 99.0,
        "removing the maximum-weight record with maintenance ON must walk the prefix bounds \
         back down; leaving 99.0 means propagation was skipped on the delete path",
    );
    let mut rebuilt = InvertedIndexBuilder::new();
    for (id, v) in base.iter().filter(|(id, _)| *id != 4) {
        rebuilt.add(*id, v.clone());
    }
    let rebuilt = rebuilt.build();
    for dim in [3usize, 4usize] {
        let after: Vec<_> = maintained.postings[dim]
            .elements
            .iter()
            .map(|e| (e.record_id, e.max_next_weight))
            .collect();
        let want: Vec<_> = rebuilt.postings[dim]
            .elements
            .iter()
            .map(|e| (e.record_id, e.max_next_weight))
            .collect();
        assert_eq!(
            after, want,
            "dim {dim}: removing a record with maintenance ON must leave exact bounds",
        );
    }

    // An overwrite that RAISES a weight must raise the bounds of every record to its LEFT, or
    // pruning over them over-prunes. This is the dangerous direction, and it is specific to the
    // overwrite path.
    //
    // The record overwritten has to be the LAST one: raising the first record's weight changes no
    // bound at all, because `max_next_weight` only ever looks rightward. An earlier version of this
    // test overwrote id 0 and asserted `record_id == 0 || .. || record_id > 0`, which is a tautology
    // for a `u32` and could not fail — clippy's `!Range::contains` lint is what caught it.
    const LAST: PointOffsetType = 7;
    let mut maintained = build_via_upsert(&base, true);
    let old = RemappedSparseVector::new(vec![3, 4], vec![WEIGHTS[LAST as usize], 0.5]).unwrap();
    let new = RemappedSparseVector::new(vec![3, 4], vec![999.0, 0.5]).unwrap();
    maintained.upsert(LAST, new, Some(old));
    assert_eq!(
        maintained.postings[3].elements[LAST as usize].weight, 999.0,
        "the overwrite itself must land",
    );
    for element in &maintained.postings[3].elements {
        if element.record_id == LAST {
            continue;
        }
        assert_eq!(
            element.max_next_weight, 999.0,
            "record {} still bounds its suffix at {} after the last record was raised to 999.0; \
             a bound below the true suffix maximum is what makes pruning drop valid results",
            element.record_id, element.max_next_weight,
        );
    }

    // The THIRD mutable path: an overwrite whose new vector DROPS a dimension the old one had.
    // That routes through `elements_to_delete` -> `PostingList::delete_with` inside the inherent
    // `upsert`, which is a different call site from `InvertedIndexRam::remove` above. Covering it
    // needs old and new vectors with *different* dimension sets — an earlier version of this test
    // used `{3, 4}` for both, so `elements_to_delete` was always empty and a hardcoded
    // `propagate = false` at that call site stayed invisible to the whole suite.
    let mut maintained = build_via_upsert(&base, true);
    let old = RemappedSparseVector::new(vec![3, 4], vec![WEIGHTS[4], 0.5]).unwrap();
    let new = RemappedSparseVector::new(vec![4], vec![0.5]).unwrap();
    maintained.upsert(4, new, Some(old));
    assert!(
        !maintained.postings[3]
            .elements
            .iter()
            .any(|e| e.record_id == 4),
        "dropping dimension 3 from the vector must remove its posting entry",
    );
    for element in &maintained.postings[3].elements {
        assert!(
            element.max_next_weight < 99.0,
            "record {} still bounds its suffix at 99.0 after the record holding that weight \
             dropped dimension 3; propagation was skipped on the dim-drop delete path",
            element.record_id,
        );
    }

    // With maintenance OFF, remove must skip propagation: the surviving prefix keeps the bound it
    // had, which is the stale-high state `skipping_propagation_can_also_leave_bounds_too_high`
    // documents.
    let mut skipped = build_via_upsert(&base, false);
    skipped.remove(
        4,
        RemappedSparseVector::new(vec![3, 4], vec![99.0, 0.5]).unwrap(),
    );
    assert!(
        !skipped.max_next_weight_reliable(),
        "an index that skipped propagation on delete must still report its bounds unusable",
    );
    // Note which shape this arm is: `build_via_upsert(.., false)` disables maintenance on an EMPTY
    // index, so no bound is ever written and every one stays at `DEFAULT_MAX_NEXT_WEIGHT`. That is
    // not the production shape — `plan()` disables the flag on an index whose bounds the one-pass
    // build already made exact, which is what produces the stale-HIGH bounds that
    // `skipping_propagation_can_also_leave_bounds_too_high` covers. Asserting 99.0 here would be
    // wrong for exactly that reason.
    assert!(
        skipped.postings[3]
            .elements
            .iter()
            .all(|e| e.max_next_weight == DEFAULT_MAX_NEXT_WEIGHT),
        "an index built entirely with maintenance off should never have written a bound",
    );
}
