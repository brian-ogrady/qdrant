//! Realistic id-tracker lookup costs. Public API only, so this file compiles
//! unchanged with and without the Bloom pre-screen.
//!
//! Two things the first cut of this bench got wrong, corrected here:
//!  - a small probe set kept the filter's working set L2-resident, flattering
//!    the miss numbers; PROBES is now large enough to defeat that.
//!  - only numeric ids were exercised, so UUID-keyed collections — where the
//!    map holds 16-byte keys — went unmeasured.

use std::hint::black_box;

use common::types::DeferredBehavior;
use criterion::{Criterion, criterion_group, criterion_main};
use segment::id_tracker::mutable_id_tracker::MutableIdTracker;
use segment::id_tracker::{IdTracker, IdTrackerRead};
use segment::types::PointIdType;
use tempfile::TempDir;
use uuid::Uuid;

/// Points per tracker. The 10B deployment targets ~5.4M per segment; the
/// default is kept lower so the bench is cheap to run, and the ranking of the
/// variants does not change with it.
const POINTS: u64 = 1_000_000;
/// Large enough that the probe stream cannot sit in L2.
const PROBES: usize = 1 << 17;

fn spread(i: u64) -> u64 {
    i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Only even slots are populated, so the odd ids used as absent keys interleave
/// with live ones. Absent keys drawn from beyond the maximum key would all
/// descend the same rightmost spine, which stays cached and makes a `BTreeMap`
/// miss look far cheaper than a real one.
fn key(i: u64, as_uuid: bool) -> PointIdType {
    if as_uuid {
        PointIdType::Uuid(Uuid::from_u128(u128::from(spread(i))))
    } else {
        PointIdType::NumId(spread(i))
    }
}

fn build(points: u64, as_uuid: bool) -> (TempDir, MutableIdTracker) {
    let dir = TempDir::new().unwrap();
    let mut tracker = MutableIdTracker::open(dir.path(), None).unwrap();
    for i in 0..points {
        tracker.set_link(key(i * 2, as_uuid), i as u32).unwrap();
    }
    (dir, tracker)
}

fn probe(c: &mut Criterion, name: &str, tracker: &MutableIdTracker, keys: Vec<PointIdType>) {
    c.bench_function(name, |b| {
        let mut next = 0usize;
        b.iter(|| {
            let id = keys[next & (PROBES - 1)];
            next += 1;
            black_box(
                tracker.internal_id_with_behavior(black_box(id), DeferredBehavior::WithDeferred),
            )
        })
    });
}

fn realistic(c: &mut Criterion) {
    // Which mode this run measured. Set QDRANT_ID_TRACKER_BLOOM_FILTER=0 to
    // get the unfiltered baseline without switching branches.
    println!(
        "\n=== external id pre-screen: {} ===",
        if segment::id_tracker::external_id_filter::is_enabled() {
            "ENABLED"
        } else {
            "DISABLED"
        },
    );

    // ---- numeric-keyed collection ----
    let (dir, tracker) = build(POINTS, false);
    println!("\nnumeric tracker RAM: {} bytes", tracker.ram_usage_bytes());
    probe(
        c,
        "num/miss",
        &tracker,
        (0..PROBES)
            .map(|i| key((i as u64 % POINTS) * 2 + 1, false))
            .collect(),
    );
    probe(
        c,
        "num/hit",
        &tracker,
        (0..PROBES)
            .map(|i| key((i as u64 % POINTS) * 2, false))
            .collect(),
    );
    drop(tracker);
    drop(dir);

    // ---- uuid-keyed collection ----
    let (dir, tracker) = build(POINTS, true);
    println!("uuid tracker RAM: {} bytes\n", tracker.ram_usage_bytes());
    probe(
        c,
        "uuid/miss",
        &tracker,
        (0..PROBES)
            .map(|i| key((i as u64 % POINTS) * 2 + 1, true))
            .collect(),
    );
    probe(
        c,
        "uuid/hit",
        &tracker,
        (0..PROBES)
            .map(|i| key((i as u64 % POINTS) * 2, true))
            .collect(),
    );
    drop(tracker);
    drop(dir);
}

fn id_tracker_load(c: &mut Criterion) {
    // Build once, flush, then measure repeated cold reopens. On a filtered
    // branch this includes seeding the pre-screen for every replayed link.
    let (dir, tracker) = build(200_000, false);
    let flush = tracker.mapping_flusher();
    flush().unwrap();
    drop(tracker);

    c.bench_function("load/reopen_200k", |b| {
        b.iter(|| {
            let t = MutableIdTracker::open(dir.path(), None).unwrap();
            black_box(t.total_point_count())
        })
    });

    drop(dir);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).measurement_time(std::time::Duration::from_secs(4));
    targets = realistic, id_tracker_load
}

criterion_main!(benches);
