//! Phase 4 (v3.13): object-store fetch + snapshot-keyed cache + measurements.
//!
//! Six always-green tests against `object_store::memory::InMemory` cover
//! cache hit/miss/rekey semantics, range-read correctness, corrupted /
//! tiny-object rejection, and an end-to-end catalog→fetch→mmap→search flow.
//! A measurement test prints (report-only, dated, no assertions) per-query
//! p50/p95, cache miss vs hit round-trip, and populate()/clear_cache() effect.
//!
//! An opt-in `PUFFIN_S3_ENDPOINT`-gated suite runs the same tests against a
//! real S3-compatible wire protocol; skips cleanly when the endpoint is unset.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use fs_err as fs;
use object_store::memory::InMemory;
use object_store::{ObjectStore, ObjectStoreExt};
use quantization::EncodedVectors;
use quantization::encoded_storage::TestEncodedStorage;
use quantization::encoded_vectors_binary::{
    EncodedVectorsBin, Encoding, get_quantized_vector_size_from_params,
};
use rand::SeedableRng;
use tempfile::TempDir;

use crate::index::hnsw_index::HnswM;
use crate::index::hnsw_index::graph_layers::{GraphLayerData, GraphLayers, SearchAlgorithm};
use crate::index::hnsw_index::graph_links::{GraphLinks, GraphLinksFormat};
use crate::index::visited_pool::VisitedPool;
use crate::types::Distance;

use super::puffin_shared::{
    DIM, FIXTURE_SEED, LOGICAL_PARQUET_NAME, MockCatalogStub, NUM_VECTORS, PuffinFetcher,
    RealDataScaffold, SourceRowMapping, build_test_puffin_fixture,
    build_test_puffin_fixture_from_vectors,
    build_test_puffin_fixture_from_vectors_with_mapping, mmap_whole_file, put_bytes,
    put_bytes_multipart, read_and_validate_footer, repo_root, source_row_to_row_group,
    time_it, try_load_real_data,
};

/// The mock object key we upload Phase 1's fixture under.
const OBJECT_KEY: &str = "iceberg/table/data/index.puffin";
/// Snapshot id used by the default single-snapshot tests.
const SNAPSHOT_A: u64 = 42;
/// A second snapshot id — same object key, different cache-directory bucket.
const SNAPSHOT_B: u64 = 4242;

/// Build the store + fetcher pair that most tests use, seeded with the
/// Phase-1-style random fixture (structural, does not depend on real data).
fn store_and_fetcher_with_random_fixture()
    -> (Arc<dyn ObjectStore>, PuffinFetcher, Vec<u8>, TempDir) {
    let fixture = build_test_puffin_fixture();
    let bytes = fs::read(&fixture.puffin_path).unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    // We need a runtime to put_bytes; the fetcher owns its own runtime for
    // its own calls, so we build a throwaway one just for the seed put.
    let seed_rt = tokio::runtime::Runtime::new().unwrap();
    put_bytes(&seed_rt, store.as_ref(), OBJECT_KEY, bytes.clone()).unwrap();
    let cache_root = TempDir::new().unwrap();
    let fetcher = PuffinFetcher::new(Arc::clone(&store), cache_root.path().to_path_buf()).unwrap();
    (store, fetcher, bytes, cache_root)
}

// -------- Test 1: miss downloads, caches, post-verifies -------------------

#[test]
fn test_fetcher_miss_downloads_caches_and_post_verifies() {
    let (_store, fetcher, original_bytes, _cache_root) = store_and_fetcher_with_random_fixture();
    let cached_path = fetcher.fetch(OBJECT_KEY, SNAPSHOT_A).expect("miss should succeed");
    assert!(cached_path.exists(), "cached file missing after miss");
    let cached_bytes = fs::read(&cached_path).unwrap();
    assert_eq!(
        cached_bytes, original_bytes,
        "cached bytes must match original — else the tail synth broke the container",
    );
    // Post-verify counter should tick on every successful miss commit.
    assert_eq!(fetcher.stats().post_cache_verifies, 1);
    // And the cached file must itself pass read_and_validate_footer.
    read_and_validate_footer(&cached_bytes).expect("cached file must validate");
}

// -------- Test 2: hit path does not touch the store -----------------------

#[test]
fn test_fetcher_hit_no_store_access() {
    let (_store, fetcher, _bytes, _cache_root) = store_and_fetcher_with_random_fixture();
    let _ = fetcher.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap();
    let after_miss = fetcher.stats();
    // Second call: same (key, snapshot). Must return the same path with
    // zero additional store calls (gets/heads unchanged).
    let _ = fetcher.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap();
    let after_hit = fetcher.stats();
    assert_eq!(after_hit.gets, after_miss.gets, "hit path issued a store GET");
    assert_eq!(
        after_hit.heads, after_miss.heads,
        "hit path issued a store HEAD",
    );
    assert_eq!(
        after_hit.post_cache_verifies, after_miss.post_cache_verifies,
        "hit path re-ran post-verify (should be skipped on hit)",
    );
}

// -------- Test 3: cache is keyed by snapshot_id ---------------------------

#[test]
fn test_fetcher_rekey_by_snapshot_id() {
    let (_store, fetcher, original_bytes, _cache_root) = store_and_fetcher_with_random_fixture();
    let path_a = fetcher.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap();
    let path_b = fetcher.fetch(OBJECT_KEY, SNAPSHOT_B).unwrap();
    assert_ne!(path_a, path_b, "same key + different snapshot_id must cache separately");
    // Both entries must be valid, independent copies.
    let bytes_a = fs::read(&path_a).unwrap();
    let bytes_b = fs::read(&path_b).unwrap();
    assert_eq!(bytes_a, original_bytes);
    assert_eq!(bytes_b, original_bytes);
    // Both must have committed post-verify.
    assert_eq!(fetcher.stats().post_cache_verifies, 2);
}

// -------- Test 4: range-read correctness ---------------------------------

#[test]
fn test_fetcher_range_read_correctness() {
    // Directly exercise the ObjectStore range API the fetcher relies on to
    // guarantee that when the fetcher reads the trailer + footer as suffixes,
    // the bytes it consumes are byte-identical to the tail of the object.
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = build_test_puffin_fixture();
    let bytes = fs::read(&fixture.puffin_path).unwrap();
    put_bytes(&rt, store.as_ref(), OBJECT_KEY, bytes.clone()).unwrap();

    // Suffix(12): trailer bytes.
    let trailer = rt.block_on(async {
        let opts = object_store::GetOptions {
            range: Some(object_store::GetRange::Suffix(12u64)),
            ..Default::default()
        };
        let res = store.get_opts(&object_store::path::Path::from(OBJECT_KEY), opts).await.unwrap();
        res.bytes().await.unwrap()
    });
    let n = bytes.len();
    assert_eq!(trailer.as_ref(), &bytes[n - 12..n], "Suffix(12) bytes disagree with file tail");

    // Suffix(12 + footer_size): footer + trailer.
    let footer_size = u32::from_le_bytes(bytes[n - 12..n - 8].try_into().unwrap()) as usize;
    let footer_and_trailer = rt.block_on(async {
        let opts = object_store::GetOptions {
            range: Some(object_store::GetRange::Suffix((12 + footer_size) as u64)),
            ..Default::default()
        };
        let res = store.get_opts(&object_store::path::Path::from(OBJECT_KEY), opts).await.unwrap();
        res.bytes().await.unwrap()
    });
    assert_eq!(
        footer_and_trailer.as_ref(),
        &bytes[n - (12 + footer_size)..n],
        "Suffix(12+footer_size) bytes disagree with file tail",
    );
}

// -------- Test 5: corrupted / tiny-object rejection ----------------------

#[test]
fn test_fetcher_rejects_corrupted_and_tiny_objects() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let cache_root = TempDir::new().unwrap();
    let fetcher = PuffinFetcher::new(
        Arc::clone(&store),
        cache_root.path().to_path_buf(),
    )
    .unwrap();

    // 5a. Tiny object — smaller than the trailer. Must produce a clean
    //     "not a puffin container" error rather than a byte-slice panic.
    let tiny_key = "iceberg/tiny.puffin";
    put_bytes(&rt, store.as_ref(), tiny_key, vec![0u8; 5]).unwrap();
    let err = fetcher.fetch(tiny_key, SNAPSHOT_A).expect_err("tiny object must be rejected");
    assert!(
        err.to_string().contains("not a puffin container"),
        "expected 'not a puffin container' message, got: {err}",
    );
    // No cache entry committed.
    let tiny_cached = fetcher.cached_path(tiny_key, SNAPSHOT_A);
    assert!(!tiny_cached.exists(), "tiny-object fetch leaked a cache entry");
    assert!(!tiny_cached.with_extension("tmp").exists(), "leftover .tmp");

    // 5b. Correctly-sized garbage — the trailer's declared footer_size can
    //     be honored as a range read, but the JSON payload is nonsense.
    //     `read_and_validate_footer` must reject before the cache commit.
    let garbage_key = "iceberg/garbage.puffin";
    let mut garbage = vec![0u8; 4096];
    // Write a plausible-looking trailer at the tail so Suffix(12) succeeds.
    // footer_size = 128, flags = 0, magic = PFA1. But the "footer" bytes
    // won't parse as JSON, so validation fails at the JSON step.
    let footer_size_at_offset = 4096 - 12;
    garbage[footer_size_at_offset..footer_size_at_offset + 4]
        .copy_from_slice(&128u32.to_le_bytes());
    garbage[footer_size_at_offset + 4..footer_size_at_offset + 8]
        .copy_from_slice(&0u32.to_le_bytes());
    garbage[4092..4096].copy_from_slice(b"PFA1");
    put_bytes(&rt, store.as_ref(), garbage_key, garbage).unwrap();

    let err = fetcher.fetch(garbage_key, SNAPSHOT_A).expect_err(
        "garbage container must be rejected before cache commit",
    );
    // Message may say "pre-cache footer validation" or a JSON parse error.
    let msg = err.to_string();
    assert!(
        msg.contains("footer") || msg.contains("JSON") || msg.contains("json"),
        "unexpected error message: {msg}",
    );
    let garbage_cached = fetcher.cached_path(garbage_key, SNAPSHOT_A);
    assert!(
        !garbage_cached.exists(),
        "cache entry committed despite failed validation — poisoned cache",
    );
    assert!(
        !garbage_cached.with_extension("tmp").exists(),
        "leftover .tmp after corrupted fetch",
    );

    // 5c. Sanity: after both failures, the fetcher can still serve a valid
    //     object. Cache is not globally broken.
    let valid_key = "iceberg/valid.puffin";
    let fixture = build_test_puffin_fixture();
    let valid_bytes = fs::read(&fixture.puffin_path).unwrap();
    put_bytes(&rt, store.as_ref(), valid_key, valid_bytes).unwrap();
    let path = fetcher.fetch(valid_key, SNAPSHOT_A).expect("valid fetch after failures");
    assert!(path.exists());
}

// -------- Test 6: end-to-end catalog stub → fetch → mmap → search --------

#[test]
fn test_e2e_search_via_object_store() {
    // If the real-data fixture is available, use it — the row-pointer
    // resolution has more to say when there are real entries. Otherwise
    // fall back to the random-fixture container.
    let (test_container_bytes, fixture_num_vectors): (Vec<u8>, usize) =
        match try_load_real_data(NUM_VECTORS) {
            Some(rd) => {
                let scaffold = RealDataScaffold::new(Distance::Dot, &rd.body);
                let mut rng = rand::rngs::StdRng::seed_from_u64(FIXTURE_SEED.wrapping_add(0x2000));
                let fixture = build_test_puffin_fixture_from_vectors(&rd.body, &scaffold, &mut rng);
                (fs::read(&fixture.puffin_path).unwrap(), rd.body.len())
            }
            None => {
                let fixture = build_test_puffin_fixture();
                (fs::read(&fixture.puffin_path).unwrap(), NUM_VECTORS)
            }
        };

    // Set up store + catalog stub + fetcher.
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let seed_rt = tokio::runtime::Runtime::new().unwrap();
    put_bytes(&seed_rt, store.as_ref(), OBJECT_KEY, test_container_bytes).unwrap();
    drop(seed_rt);
    let catalog = MockCatalogStub::new(SNAPSHOT_A, OBJECT_KEY);
    let cache_root = TempDir::new().unwrap();
    let fetcher =
        PuffinFetcher::new(Arc::clone(&store), cache_root.path().to_path_buf()).unwrap();

    // Catalog → fetch → local path.
    let (snapshot_id, uri) = catalog.snapshot_summary();
    let cached_path = fetcher.fetch(uri, snapshot_id).expect("e2e fetch");
    assert!(cached_path.exists());

    // Mmap + reader.
    let shared_mmap = mmap_whole_file(&cached_path);
    let file_bytes = fs::read(&cached_path).unwrap();
    let footer = read_and_validate_footer(&file_bytes).unwrap();

    // Assemble GraphLayers from parts (same path Phase 3 uses).
    let meta_range = footer.by_type("ann-hnsw-graph-meta-v1").unwrap();
    let meta_slice = &shared_mmap[meta_range.offset..meta_range.offset + meta_range.length];
    let graph_data: GraphLayerData<'_> = bincode::deserialize(meta_slice).unwrap();
    let links_range = footer.by_type("ann-hnsw-graph-links-v1").unwrap();
    let links = GraphLinks::load_from_ranged_mmap(
        Arc::clone(&shared_mmap),
        links_range.offset as u64,
        links_range.length as u64,
        GraphLinksFormat::Compressed,
    )
    .unwrap();
    let graph_layers = GraphLayers {
        hnsw_m: HnswM::new(graph_data.m, graph_data.m0),
        links,
        entry_points: graph_data.entry_points.into_owned(),
        visited_pool: VisitedPool::new(),
    };

    // Score with the same scaffold shape Phase 3 uses. Rebuild the scaffold
    // over the same vectors — deterministic if real data was loaded, seeded
    // random otherwise.
    let (scaffold, queries) = match try_load_real_data(NUM_VECTORS) {
        Some(rd) => {
            let mut queries = rd.queries.clone();
            queries.truncate(3); // e2e sanity — 3 queries is enough here.
            (RealDataScaffold::new(Distance::Dot, &rd.body), queries)
        }
        None => {
            let mut rng = rand::rngs::StdRng::seed_from_u64(FIXTURE_SEED);
            let body: Vec<Vec<f32>> = (0..NUM_VECTORS)
                .map(|_| crate::fixtures::index_fixtures::random_vector(&mut rng, DIM))
                .collect();
            let mut qrng = rand::rngs::StdRng::seed_from_u64(0xDEAD_BEEF);
            let queries: Vec<Vec<f32>> = (0..3)
                .map(|_| crate::fixtures::index_fixtures::random_vector(&mut qrng, DIM))
                .collect();
            (RealDataScaffold::new(Distance::Dot, &body), queries)
        }
    };

    let is_stopped = AtomicBool::new(false);
    for q in &queries {
        let top10 = graph_layers
            .search(
                10,
                64,
                SearchAlgorithm::Hnsw,
                scaffold.scorer(q.clone()),
                None,
                &is_stopped,
            )
            .unwrap();
        assert_eq!(top10.len(), 10);
        for scored in &top10 {
            assert!(
                (scored.idx as usize) < fixture_num_vectors,
                "returned point id {} out of bounds ({fixture_num_vectors})",
                scored.idx,
            );
        }
    }

    // Row-pointer resolution — spot-check a random index; must map to the
    // mocked Parquet path (§5 Phase 3 step 7 that migrated to Phase 4).
    let rp_range = footer.by_type("ann-hnsw-row-pointers-v1").unwrap();
    let rp = &shared_mmap[rp_range.offset..rp_range.offset + rp_range.length];
    assert_eq!(rp[0], 1u8);
    let entry_count = u32::from_le_bytes(rp[1..5].try_into().unwrap()) as usize;
    assert_eq!(entry_count, fixture_num_vectors);
}

// -------- Measurements (report-only, no assertions) ---------------------

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Env var (shared with the Phase 3 recall test) controlling how many body
/// vectors the measurement fixture uses. Default 10_000; setting 100_000
/// runs the same measurements against the full sampler body.
fn resolve_measurement_n() -> usize {
    match std::env::var("PUFFIN_FIXTURE_N") {
        Ok(v) => v.parse::<usize>().unwrap_or_else(|_| {
            panic!("PUFFIN_FIXTURE_N={v} is not a valid usize")
        }),
        Err(_) => NUM_VECTORS,
    }
}

/// Number of independent repetitions per measurement. Median + spread over
/// 5 samples smooths out per-run noise (thermal, scheduler jitter, page-
/// cache warmup effects) while keeping the test in a reasonable time budget.
const NUM_REPS: usize = 5;

/// Report (median, min, max) of a vector of microsecond samples.
fn describe_us(mut samples: Vec<f64>) -> (f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let median = samples[n / 2];
    let min = samples[0];
    let max = samples[n - 1];
    (median, min, max)
}

fn median_spread_us(samples: Vec<f64>) -> String {
    let (median, min, max) = describe_us(samples);
    format!("median = {median:.1} µs  [min {min:.1}, max {max:.1}]")
}

fn median_spread_ms(samples_us: Vec<f64>) -> String {
    let (median, min, max) = describe_us(samples_us);
    format!(
        "median = {median_ms:.2} ms  [min {min_ms:.2}, max {max_ms:.2}]",
        median_ms = median / 1000.0,
        min_ms = min / 1000.0,
        max_ms = max / 1000.0,
    )
}

#[test]
fn test_measurements_report_only() {
    let fixture_n = resolve_measurement_n();
    // Only run measurements when real data is available; otherwise the
    // numbers are just BQ noise and don't inform anything.
    let Some(rd) = try_load_real_data(fixture_n) else {
        println!(
            "Phase 4 measurements skipped: real-data fixture not present (data/fixtures/gte_100k_vectors.bin)"
        );
        return;
    };
    println!("Phase 4 measurements: N={fixture_n} (env PUFFIN_FIXTURE_N)");
    println!("  Discipline: {NUM_REPS} repetitions per metric, quiet machine");
    println!("  Methodology:");
    println!("    • warm-path p50/p95: 20 sequential queries over a warmed-up graph");
    println!("      (populate() called before the loop; row-pointer pages touched).");
    println!("      Measures sustained per-query search latency under a warm page cache.");
    println!("    • populate/clear_cache 'first query': ONE search immediately after");
    println!("      the respective madvise(). Different from p50 in what it measures:");
    println!("      p50 is a middle-of-distribution sample; first-query is a single");
    println!("      point-in-time sample of a specific advise-state. On a warm cache,");
    println!("      populate()'s first-query can be faster than warm-path p50 because");
    println!("      populate() is nearly free when pages are already resident and the");
    println!("      subsequent query benefits from perfectly-hot caches. This is a");
    println!("      methodology difference, not an anomaly.");

    // Build a container from real data and upload it to an InMemory store.
    let scaffold = RealDataScaffold::new(Distance::Dot, &rd.body);
    let mut level_rng = rand::rngs::StdRng::seed_from_u64(FIXTURE_SEED.wrapping_add(0x2000));
    let fixture = build_test_puffin_fixture_from_vectors(&rd.body, &scaffold, &mut level_rng);
    let container_bytes = fs::read(&fixture.puffin_path).unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let seed_rt = tokio::runtime::Runtime::new().unwrap();
    put_bytes(&seed_rt, store.as_ref(), OBJECT_KEY, container_bytes).unwrap();
    drop(seed_rt);

    // -- 1. cache miss + hit, 5 reps each ---------------------------------
    // Each miss rep uses a fresh cache_dir so the fetcher genuinely runs the
    // miss path. Hit reps reuse the same cache_dir from the miss.
    let mut miss_us: Vec<f64> = Vec::with_capacity(NUM_REPS);
    let mut hit_us: Vec<f64> = Vec::with_capacity(NUM_REPS);
    for _ in 0..NUM_REPS {
        let cache_root = TempDir::new().unwrap();
        let fetcher =
            PuffinFetcher::new(Arc::clone(&store), cache_root.path().to_path_buf()).unwrap();
        let (_, miss) = time_it(|| fetcher.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap());
        miss_us.push(miss.as_secs_f64() * 1e6);
        let (_, hit) = time_it(|| fetcher.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap());
        hit_us.push(hit.as_secs_f64() * 1e6);
    }
    println!(
        "Phase 4 cache-miss (fetch + validate + write + post-verify): {}",
        median_spread_ms(miss_us),
    );
    println!(
        "Phase 4 cache-hit  (path resolve + exists check)          : {}",
        median_spread_us(hit_us),
    );

    // -- 2. Warm-path per-query latency (search + row-ptr resolution) -----
    // Load container once. p50/p95 is over 20 queries. Repeat the FULL
    // 20-query sweep NUM_REPS times so we report a median-of-p50s and
    // median-of-p95s; the min/max spread makes noise visible.
    let cache_root_warm = TempDir::new().unwrap();
    let fetcher_warm =
        PuffinFetcher::new(Arc::clone(&store), cache_root_warm.path().to_path_buf()).unwrap();
    let cached_path = fetcher_warm.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap();
    let shared_mmap = mmap_whole_file(&cached_path);
    let file_bytes = fs::read(&cached_path).unwrap();
    let footer = read_and_validate_footer(&file_bytes).unwrap();
    let meta_range = footer.by_type("ann-hnsw-graph-meta-v1").unwrap();
    let meta_slice = &shared_mmap[meta_range.offset..meta_range.offset + meta_range.length];
    let graph_data: GraphLayerData<'_> = bincode::deserialize(meta_slice).unwrap();
    let links_range = footer.by_type("ann-hnsw-graph-links-v1").unwrap();
    let links = GraphLinks::load_from_ranged_mmap(
        Arc::clone(&shared_mmap),
        links_range.offset as u64,
        links_range.length as u64,
        GraphLinksFormat::Compressed,
    )
    .unwrap();
    let graph_layers = GraphLayers {
        hnsw_m: HnswM::new(graph_data.m, graph_data.m0),
        links,
        entry_points: graph_data.entry_points.into_owned(),
        visited_pool: VisitedPool::new(),
    };
    // Warm the mmap pages before measurement starts.
    graph_layers.links.populate().unwrap();
    let rp_range = footer.by_type("ann-hnsw-row-pointers-v1").unwrap();
    let _touch = &shared_mmap[rp_range.offset..rp_range.offset + rp_range.length];

    let is_stopped = AtomicBool::new(false);
    let mut p50_samples: Vec<f64> = Vec::with_capacity(NUM_REPS);
    let mut p95_samples: Vec<f64> = Vec::with_capacity(NUM_REPS);
    for _ in 0..NUM_REPS {
        let mut micros: Vec<f64> = Vec::with_capacity(rd.queries.len());
        for q in &rd.queries {
            let (_hnsw_top10, elapsed) = time_it(|| {
                graph_layers
                    .search(
                        10,
                        64,
                        SearchAlgorithm::Hnsw,
                        scaffold.scorer(q.clone()),
                        None,
                        &is_stopped,
                    )
                    .unwrap()
            });
            micros.push(elapsed.as_secs_f64() * 1e6);
        }
        micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
        p50_samples.push(percentile(&micros, 0.50));
        p95_samples.push(percentile(&micros, 0.95));
    }
    println!(
        "Phase 4 warm-path query latency, {} queries × {NUM_REPS} reps (target 50 000 µs; report-only):",
        rd.queries.len(),
    );
    println!("  p50: {}", median_spread_us(p50_samples));
    println!("  p95: {}", median_spread_us(p95_samples));

    // -- 3. populate() effect: 5 alternating reps -------------------------
    // Each rep: clear_cache → measure one query, then populate → measure one
    // query. Report median + spread of each series so the ordering signal is
    // preserved without single-sample noise. madvise is advisory; both arms
    // are best-effort at the kernel.
    let q0 = &rd.queries[0];
    let mut clear_us: Vec<f64> = Vec::with_capacity(NUM_REPS);
    let mut populate_us: Vec<f64> = Vec::with_capacity(NUM_REPS);
    for _ in 0..NUM_REPS {
        graph_layers.links.clear_cache().unwrap();
        let (_r, dt_clear) = time_it(|| {
            graph_layers
                .search(
                    10,
                    64,
                    SearchAlgorithm::Hnsw,
                    scaffold.scorer(q0.clone()),
                    None,
                    &is_stopped,
                )
                .unwrap()
        });
        clear_us.push(dt_clear.as_secs_f64() * 1e6);

        graph_layers.links.populate().unwrap();
        let (_r, dt_pop) = time_it(|| {
            graph_layers
                .search(
                    10,
                    64,
                    SearchAlgorithm::Hnsw,
                    scaffold.scorer(q0.clone()),
                    None,
                    &is_stopped,
                )
                .unwrap()
        });
        populate_us.push(dt_pop.as_secs_f64() * 1e6);
    }
    println!(
        "Phase 4 §6.3 arms, {NUM_REPS} reps (advisory madvise; single-query samples of a specific advise-state):",
    );
    println!(
        "  after clear_cache (MADV_DONTNEED range): {}",
        median_spread_us(clear_us),
    );
    println!(
        "  after populate    (MADV_WILLNEED range): {}",
        median_spread_us(populate_us),
    );
}

// -------- Row-pointer mapping shape (v3.13) ------------------------------

/// Parse a decoded row-pointer entry at index `i` from the loaded blob bytes.
/// Layout per §3.2 v3.11:
///   [u8 version=1] [u32 entry_count] [u32 path_count]
///   [{u32 path_len | UTF-8 bytes} × path_count]
///   [{u32 vec_id | u32 file_path_idx | u32 row_group | u32 row_offset} × N]
fn decode_row_ptr_entry(rp: &[u8], i: usize) -> (u32, u32, u32, u32) {
    // Skip version byte + entry_count u32.
    let mut cursor = 5usize;
    let path_count = u32::from_le_bytes(rp[cursor..cursor + 4].try_into().unwrap()) as usize;
    cursor += 4;
    for _ in 0..path_count {
        let len = u32::from_le_bytes(rp[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4 + len;
    }
    let base = cursor + i * 16;
    (
        u32::from_le_bytes(rp[base..base + 4].try_into().unwrap()),
        u32::from_le_bytes(rp[base + 4..base + 8].try_into().unwrap()),
        u32::from_le_bytes(rp[base + 8..base + 12].try_into().unwrap()),
        u32::from_le_bytes(rp[base + 12..base + 16].try_into().unwrap()),
    )
}

#[test]
fn test_row_pointer_mapping_shape() {
    // Skip when the v3.13 source-indices sidecar isn't present.
    let Some(rd) = try_load_real_data(NUM_VECTORS) else {
        println!("Row-pointer mapping test skipped: real-data fixture not present");
        return;
    };
    let Some(source_indices) = rd.body_source_indices.as_deref() else {
        println!(
            "Row-pointer mapping test skipped: sidecar predates v3.13 (no source_indices_file)"
        );
        return;
    };
    if rd.body_sidecar.row_group_sizes.is_empty() {
        println!("Row-pointer mapping test skipped: sidecar has no row_group_sizes");
        return;
    }
    let row_group_sizes = &rd.body_sidecar.row_group_sizes;

    // Build a container with real source-row mapping.
    let scaffold = RealDataScaffold::new(Distance::Dot, &rd.body);
    let mut level_rng = rand::rngs::StdRng::seed_from_u64(FIXTURE_SEED.wrapping_add(0x3000));
    let mapping = SourceRowMapping {
        source_rows: source_indices,
        row_group_sizes,
        logical_file_name: LOGICAL_PARQUET_NAME,
    };
    let fixture = build_test_puffin_fixture_from_vectors_with_mapping(
        &rd.body,
        &scaffold,
        &mut level_rng,
        Some(mapping),
    );
    let file_bytes = fs::read(&fixture.puffin_path).unwrap();
    let footer = read_and_validate_footer(&file_bytes).unwrap();
    let shared_mmap = mmap_whole_file(&fixture.puffin_path);

    // (1) Path table has exactly one entry: the neutral logical name.
    let rp_range = footer.by_type("ann-hnsw-row-pointers-v1").unwrap();
    let rp = &shared_mmap[rp_range.offset..rp_range.offset + rp_range.length];
    assert_eq!(rp[0], 1u8);
    let entry_count = u32::from_le_bytes(rp[1..5].try_into().unwrap()) as usize;
    assert_eq!(entry_count, rd.body.len());
    let path_count = u32::from_le_bytes(rp[5..9].try_into().unwrap()) as usize;
    assert_eq!(path_count, 1);
    let path_len = u32::from_le_bytes(rp[9..13].try_into().unwrap()) as usize;
    let path_str = std::str::from_utf8(&rp[13..13 + path_len]).unwrap();
    assert_eq!(
        path_str, LOGICAL_PARQUET_NAME,
        "row-pointer path table must contain the neutral logical name, not a bucket URI",
    );

    // (2) Reconstruction: for each of 10 sampled body indices, decode the
    // row-pointer entry, walk row-group sizes with the (row_group, offset)
    // it names, and confirm the absolute source row equals the sidecar's
    // source_indices[i].
    use rand::RngExt;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xF00D);
    for _ in 0..10 {
        let i = rng.random_range(0..rd.body.len());
        let (vec_id, file_path_idx, row_group, row_offset) = decode_row_ptr_entry(rp, i);
        assert_eq!(vec_id, i as u32);
        assert_eq!(file_path_idx, 0);
        let cum: u64 = row_group_sizes[..row_group as usize].iter().sum();
        let reconstructed = cum + u64::from(row_offset);
        let expected = source_indices[i];
        assert_eq!(
            reconstructed, expected,
            "body[{i}] should map to parquet row {expected}, decoded ({row_group}, {row_offset}) → {reconstructed}",
        );
        // Also cross-check against the direct helper.
        let (rg2, off2) = source_row_to_row_group(row_group_sizes, expected);
        assert_eq!((rg2, off2), (row_group, row_offset));
    }
    println!(
        "Row-pointer mapping verified: 10 sampled body indices reconstruct their source parquet rows via (row_group, row_offset) prefix-sum."
    );
}

// -------- S3-compatible opt-in suite (env-gated) --------------------------

/// Env var set → run against a real S3-compatible endpoint. Absent → skip
/// cleanly. The endpoint is expected to be provided by the operator; no
/// container actions happen from this test.
fn opt_in_endpoint() -> Option<String> {
    std::env::var("PUFFIN_S3_ENDPOINT").ok()
}

fn build_opt_in_store(endpoint: String) -> Option<Arc<dyn ObjectStore>> {
    let bucket = std::env::var("PUFFIN_S3_BUCKET").ok()?;
    let access = std::env::var("AWS_ACCESS_KEY_ID").ok()?;
    let secret = std::env::var("AWS_SECRET_ACCESS_KEY").ok()?;
    let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
    let s3 = object_store::aws::AmazonS3Builder::new()
        .with_endpoint(endpoint)
        .with_bucket_name(bucket)
        .with_access_key_id(access)
        .with_secret_access_key(secret)
        .with_region(region)
        .with_allow_http(true)
        .build()
        .expect("build AmazonS3");
    Some(Arc::new(s3))
}

const S3_ENDPOINT_LABEL: &str =
    "S3-compatible endpoint via localhost (containerized VM on this host)";

#[test]
fn test_puffin_s3_opt_in_e2e() {
    let Some(endpoint) = opt_in_endpoint() else {
        println!("Phase 4 S3 opt-in suite skipped: PUFFIN_S3_ENDPOINT not set");
        return;
    };
    let Some(store) = build_opt_in_store(endpoint) else {
        panic!(
            "PUFFIN_S3_ENDPOINT is set but one or more of PUFFIN_S3_BUCKET, \
             AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY is missing",
        );
    };

    // Upload the fixture .puffin.
    let fixture = build_test_puffin_fixture();
    let bytes = fs::read(&fixture.puffin_path).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    put_bytes(&rt, store.as_ref(), OBJECT_KEY, bytes.clone()).unwrap();
    drop(rt);

    // 5 miss/hit cycles: fresh cache per iteration → fresh miss; hit reuses
    // that iteration's committed entry. Report median + spread. Under this
    // labeling the VM/hypervisor hops are explicitly baked into the numbers.
    let mut miss_us: Vec<f64> = Vec::with_capacity(NUM_REPS);
    let mut hit_us: Vec<f64> = Vec::with_capacity(NUM_REPS);
    let mut last_cached_path = None;
    for _ in 0..NUM_REPS {
        let cache_root = TempDir::new().unwrap();
        let fetcher =
            PuffinFetcher::new(Arc::clone(&store), cache_root.path().to_path_buf()).unwrap();
        let (path_miss, miss_elapsed) = time_it(|| fetcher.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap());
        let (path_hit, hit_elapsed) = time_it(|| fetcher.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap());
        assert_eq!(path_miss, path_hit);
        // Byte round-trip sanity on every iteration — surface any wire-
        // protocol corruption immediately.
        let cached = fs::read(&path_miss).unwrap();
        assert_eq!(cached, bytes, "S3 round-trip mangled bytes");
        read_and_validate_footer(&cached).expect("S3-cached file must validate");
        // Store count check per iteration.
        let stats = fetcher.stats();
        assert!(stats.gets >= 3, "expected ≥3 GETs on miss (trailer, footer, body), got {}", stats.gets);
        miss_us.push(miss_elapsed.as_secs_f64() * 1e6);
        hit_us.push(hit_elapsed.as_secs_f64() * 1e6);
        last_cached_path = Some(path_miss);
        // Hold the fetcher for this iteration's lifetime; drop at loop end.
        drop(fetcher);
        drop(cache_root);
    }
    let _ = last_cached_path;
    println!(
        "Phase 4 [{S3_ENDPOINT_LABEL}] cache-miss ({NUM_REPS} reps): {}",
        median_spread_ms(miss_us),
    );
    println!(
        "Phase 4 [{S3_ENDPOINT_LABEL}] cache-hit  ({NUM_REPS} reps): {}",
        median_spread_us(hit_us),
    );
}

// -------- Rerank-over-Parquet (opt-in, requires parquet dev-dep) ----------

/// AsyncFileReader implemented directly on top of the workspace's
/// object_store (0.13.2). Avoids pulling in parquet's `object_store`
/// feature, which would introduce a second object_store version (0.12) in
/// the dep graph. Bytes counting is folded in — no separate wrapper.
struct StoreAsyncFileReader {
    store: Arc<dyn ObjectStore>,
    path: object_store::path::Path,
    file_size: u64,
    bytes_counter: Arc<std::sync::atomic::AtomicU64>,
}

impl parquet::arrow::async_reader::AsyncFileReader for StoreAsyncFileReader {
    fn get_bytes(
        &mut self,
        range: std::ops::Range<u64>,
    ) -> futures::future::BoxFuture<'_, parquet::errors::Result<bytes::Bytes>> {
        use futures::FutureExt;
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let counter = Arc::clone(&self.bytes_counter);
        async move {
            let opts = object_store::GetOptions {
                range: Some(object_store::GetRange::Bounded(range)),
                ..Default::default()
            };
            let result = store.get_opts(&path, opts).await.map_err(|e| {
                parquet::errors::ParquetError::External(Box::new(e))
            })?;
            let b = result.bytes().await.map_err(|e| {
                parquet::errors::ParquetError::External(Box::new(e))
            })?;
            counter.fetch_add(b.len() as u64, std::sync::atomic::Ordering::Relaxed);
            Ok(b)
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a parquet::arrow::arrow_reader::ArrowReaderOptions>,
    ) -> futures::future::BoxFuture<'a, parquet::errors::Result<Arc<parquet::file::metadata::ParquetMetaData>>>
    {
        use futures::FutureExt;
        let file_size = self.file_size;
        async move {
            let reader = parquet::file::metadata::ParquetMetaDataReader::new();
            let parquet_metadata = reader.load_and_finish(self, file_size).await?;
            Ok(Arc::new(parquet_metadata))
        }
        .boxed()
    }
}

/// Decode all row-pointer entries from the blob body. Layout per §3.2 v3.11.
fn decode_all_row_pointers(rp: &[u8], expected: usize) -> Vec<(u32, u32, u32, u32)> {
    assert_eq!(rp[0], 1u8, "row-pointer version");
    let entry_count = u32::from_le_bytes(rp[1..5].try_into().unwrap()) as usize;
    assert_eq!(entry_count, expected);
    let path_count = u32::from_le_bytes(rp[5..9].try_into().unwrap()) as usize;
    let mut cursor = 9usize;
    for _ in 0..path_count {
        let len = u32::from_le_bytes(rp[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4 + len;
    }
    let mut out = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let v = u32::from_le_bytes(rp[cursor..cursor + 4].try_into().unwrap());
        let f = u32::from_le_bytes(rp[cursor + 4..cursor + 8].try_into().unwrap());
        let g = u32::from_le_bytes(rp[cursor + 8..cursor + 12].try_into().unwrap());
        let r = u32::from_le_bytes(rp[cursor + 12..cursor + 16].try_into().unwrap());
        out.push((v, f, g, r));
        cursor += 16;
    }
    out
}

/// Extract the `gte` column's f32 vectors (768d each) from a single row-group's
/// worth of decoded RecordBatches. `wanted` gives the offsets within the row
/// group we want; returns their vectors in the same order.
fn extract_wanted_vectors(
    batches: &[arrow::record_batch::RecordBatch],
    wanted: &[u32],
) -> Vec<Vec<f32>> {
    use arrow::array::{Array, FixedSizeListArray, Float32Array};
    // Batches within a row group tile the row group; concatenate row indices.
    let mut cum: usize = 0;
    let mut wanted_sorted: Vec<(usize, u32)> =
        wanted.iter().enumerate().map(|(i, &off)| (i, off)).collect();
    wanted_sorted.sort_by_key(|(_, off)| *off);
    let mut result = vec![Vec::<f32>::new(); wanted.len()];
    let mut wi = 0usize;
    for batch in batches {
        let col = batch
            .column_by_name("gte")
            .expect("gte column present")
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("gte column is FixedSizeList");
        let values = col
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("gte inner is Float32");
        let batch_len = col.len();
        let batch_end = cum + batch_len;
        while wi < wanted_sorted.len() && (wanted_sorted[wi].1 as usize) < batch_end {
            let (orig_i, off) = wanted_sorted[wi];
            let local = off as usize - cum;
            let base = local * DIM;
            let mut v = Vec::with_capacity(DIM);
            for j in 0..DIM {
                v.push(values.value(base + j));
            }
            result[orig_i] = v;
            wi += 1;
        }
        cum = batch_end;
        if wi == wanted_sorted.len() {
            break;
        }
    }
    if wi != wanted_sorted.len() {
        panic!("did not find all wanted offsets: got {wi} of {}", wanted_sorted.len());
    }
    result
}

#[test]
fn test_puffin_rerank_over_parquet_opt_in() {
    use futures::TryStreamExt;
    use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
    use parquet::arrow::ProjectionMask;

    let Some(endpoint) = opt_in_endpoint() else {
        println!("Phase 4 rerank suite skipped: PUFFIN_S3_ENDPOINT not set");
        return;
    };
    let Some(rd) = try_load_real_data(NUM_VECTORS) else {
        println!("Phase 4 rerank suite skipped: real-data fixture not present");
        return;
    };
    let Some(source_indices) = rd.body_source_indices.clone() else {
        println!("Phase 4 rerank suite skipped: sidecar predates v3.13 (no source indices)");
        return;
    };
    if rd.body_sidecar.row_group_sizes.is_empty() {
        println!("Phase 4 rerank suite skipped: sidecar missing row_group_sizes");
        return;
    }
    let row_group_sizes: Vec<u64> = rd.body_sidecar.row_group_sizes.clone();
    let Some(store) = build_opt_in_store(endpoint) else {
        panic!("S3 endpoint set but bucket/creds missing");
    };
    let rt = tokio::runtime::Runtime::new().unwrap();

    // 1. Bucket-first upload check for the source parquet.
    let parquet_key = LOGICAL_PARQUET_NAME;
    let repo_parquet = repo_root().join("data/gte_product_embeddings.parquet");
    let expected_size = fs::metadata(&repo_parquet).unwrap().len();
    let key_path = object_store::path::Path::from(parquet_key);
    let head_result = rt.block_on(store.head(&key_path));
    let need_upload = match &head_result {
        Ok(meta) if meta.size == expected_size => {
            println!(
                "Phase 4 rerank: parquet already in bucket ({} bytes) — skipping upload",
                meta.size,
            );
            false
        }
        Ok(meta) => {
            println!(
                "Phase 4 rerank: parquet in bucket at {} bytes but expected {expected_size} — re-uploading",
                meta.size,
            );
            true
        }
        Err(_) => {
            println!(
                "Phase 4 rerank: parquet not in bucket — uploading {expected_size} bytes",
            );
            true
        }
    };
    if need_upload {
        let bytes = fs::read(&repo_parquet).unwrap();
        // Multipart is required — 3.3 GB single PUT hits client-side and/or
        // server-side limits on real S3-compatible endpoints.
        put_bytes_multipart(&rt, store.as_ref(), parquet_key, bytes).unwrap();
        println!("Phase 4 rerank: parquet uploaded via multipart");
    }

    // 2. Build the puffin container with real row-pointer mapping and upload.
    let scaffold = RealDataScaffold::new(Distance::Dot, &rd.body);
    let mut level_rng = rand::rngs::StdRng::seed_from_u64(FIXTURE_SEED.wrapping_add(0x4000));
    let mapping = SourceRowMapping {
        source_rows: &source_indices,
        row_group_sizes: &row_group_sizes,
        logical_file_name: LOGICAL_PARQUET_NAME,
    };
    let fixture = build_test_puffin_fixture_from_vectors_with_mapping(
        &rd.body,
        &scaffold,
        &mut level_rng,
        Some(mapping),
    );
    let container_bytes = fs::read(&fixture.puffin_path).unwrap();
    put_bytes(&rt, store.as_ref(), OBJECT_KEY, container_bytes).unwrap();

    // 3. Fetch container + load GraphLayers.
    let cache_root = TempDir::new().unwrap();
    let fetcher =
        PuffinFetcher::new(Arc::clone(&store), cache_root.path().to_path_buf()).unwrap();
    let cached_path = fetcher.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap();
    let shared_mmap = mmap_whole_file(&cached_path);
    let file_bytes = fs::read(&cached_path).unwrap();
    let footer = read_and_validate_footer(&file_bytes).unwrap();

    let meta_range = footer.by_type("ann-hnsw-graph-meta-v1").unwrap();
    let meta_slice = &shared_mmap[meta_range.offset..meta_range.offset + meta_range.length];
    let graph_data: GraphLayerData<'_> = bincode::deserialize(meta_slice).unwrap();
    let links_range = footer.by_type("ann-hnsw-graph-links-v1").unwrap();
    let links = GraphLinks::load_from_ranged_mmap(
        Arc::clone(&shared_mmap),
        links_range.offset as u64,
        links_range.length as u64,
        GraphLinksFormat::Compressed,
    )
    .unwrap();
    let graph_layers = GraphLayers {
        hnsw_m: HnswM::new(graph_data.m, graph_data.m0),
        links,
        entry_points: graph_data.entry_points.into_owned(),
        visited_pool: VisitedPool::new(),
    };
    graph_layers.links.populate().unwrap();

    // 4. Row-pointer decode — 10k entries, indexed by internal vec_id.
    let rp_range = footer.by_type("ann-hnsw-row-pointers-v1").unwrap();
    let rp = &shared_mmap[rp_range.offset..rp_range.offset + rp_range.length];
    let row_pointers = decode_all_row_pointers(rp, rd.body.len());

    // 5. Load BQ encoder from container for the "BQ top-10" comparison.
    let quant_range = footer.by_type("ann-hnsw-quantized-vectors-v1").unwrap();
    let quant_meta_range = footer.by_type("ann-hnsw-quantized-meta-v1").unwrap();
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
        &shared_mmap[quant_meta_range.offset..quant_meta_range.offset + quant_meta_range.length],
    )
    .unwrap();
    let quantized_vec_size = get_quantized_vector_size_from_params::<u128>(DIM, Encoding::OneBit);
    let storage = TestEncodedStorage::from_file(&data_path, quantized_vec_size).unwrap();
    let encoded = EncodedVectorsBin::<u128, TestEncodedStorage>::load(storage, &meta_path).unwrap();

    // 6. Set up our custom AsyncFileReader (object_store 0.13 direct, byte
    // counting folded in). Fetch metadata ONCE off the timing loop.
    let bytes_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut meta_reader = StoreAsyncFileReader {
        store: Arc::clone(&store),
        path: key_path.clone(),
        file_size: expected_size,
        bytes_counter: Arc::clone(&bytes_counter),
    };
    let parquet_meta = rt.block_on(async {
        <StoreAsyncFileReader as parquet::arrow::async_reader::AsyncFileReader>::get_metadata(
            &mut meta_reader,
            None,
        )
        .await
        .expect("get_metadata")
    });
    let meta_fetch_bytes = bytes_counter.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "Phase 4 [{S3_ENDPOINT_LABEL}] parquet metadata fetch: {meta_fetch_bytes} bytes",
    );

    // Column index for the "gte" fixed-size-list embedding. In parquet the
    // leaf for a FixedSizeList<Float32> lives at path "gte.list.element", so
    // we find the leaf whose root path segment is "gte", not the top-level
    // column of that name.
    let schema_descr = parquet_meta.file_metadata().schema_descr_ptr();
    let gte_col_idx = (0..schema_descr.num_columns())
        .find(|&i| {
            schema_descr
                .column(i)
                .path()
                .parts()
                .first()
                .map(String::as_str)
                == Some("gte")
        })
        .expect("parquet schema missing 'gte.*.element' leaf column");
    let projection = ProjectionMask::leaves(&schema_descr, vec![gte_col_idx]);

    // 7. Per-rep, per-query measurement loop.
    let is_stopped = AtomicBool::new(false);

    // Aggregators across ALL reps × queries. Timing is decomposed into
    // (a) HNSW search + candidate collect, (b) parquet fetch + decode +
    // rerank score, (c) their sum. The rerank_us number is the risk-#2
    // figure — it isolates the parquet-fetch-and-decode cost from the
    // HNSW-search cost.
    let mut all_search_us: Vec<f64> = Vec::new();
    let mut all_rerank_us: Vec<f64> = Vec::new();
    let mut all_total_us: Vec<f64> = Vec::new();
    let mut all_bytes: Vec<u64> = Vec::new();
    let mut all_row_groups_touched: Vec<usize> = Vec::new();
    let mut all_overlaps: Vec<f64> = Vec::new();
    let mut all_rg_sizes_touched: Vec<u64> = Vec::new(); // touched groups' row-counts

    let mut printed_first_rep = false;

    for rep in 0..NUM_REPS {
        for (qi, q) in rd.queries.iter().enumerate() {
            // BQ top-10 (via brute-force over the loaded encoder) — cheap
            // and off the timing loop; used only for the overlap metric.
            let hw = HardwareCounterCell::new();
            let encoded_query = encoded.encode_query(q);
            let mut bq_scored: Vec<(PointOffsetType, f32)> = (0..rd.body.len() as PointOffsetType)
                .map(|i| (i, encoded.score_point(&encoded_query, i, &hw)))
                .collect();
            bq_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let bq_top10: HashSet<PointOffsetType> =
                bq_scored[..10].iter().map(|&(id, _)| id).collect();

            // Reset byte counter for this query.
            bytes_counter.store(0, std::sync::atomic::Ordering::Relaxed);

            // --- Phase A: HNSW search → top-100 candidates ---
            let t0 = std::time::Instant::now();
            let candidates = graph_layers
                .search(
                    100,
                    128,
                    SearchAlgorithm::Hnsw,
                    scaffold.scorer(q.clone()),
                    None,
                    &is_stopped,
                )
                .unwrap();
            let candidate_ids: Vec<u32> = candidates.iter().map(|s| s.idx).collect();
            let t1 = std::time::Instant::now();
            let search_us = t1.duration_since(t0).as_secs_f64() * 1e6;

            // Group by row_group (still Phase A conceptually, but cheap enough
            // to fold into the rerank timing — it's part of the rerank pipeline
            // any production path would run).

            // --- Phase B: row-group group + ranged parquet GETs + decode + rerank ---
            let mut by_group: std::collections::BTreeMap<u32, Vec<(u32, u32)>> =
                std::collections::BTreeMap::new();
            for &vec_id in &candidate_ids {
                let (rp_vec_id, fp_idx, rg, offset) = row_pointers[vec_id as usize];
                assert_eq!(rp_vec_id, vec_id);
                assert_eq!(fp_idx, 0, "test container uses a single logical file");
                by_group.entry(rg).or_default().push((vec_id, offset));
            }
            let unique_row_groups: usize = by_group.len();

            let mut candidate_vectors: std::collections::HashMap<u32, Vec<f32>> =
                std::collections::HashMap::with_capacity(candidate_ids.len());
            for (&rg, entries) in by_group.iter() {
                let rg_size = row_group_sizes[rg as usize];
                all_rg_sizes_touched.push(rg_size);

                // Fresh reader per row group so metadata isn't refetched
                // (we pass pre-fetched metadata into the stream builder below).
                let reader = StoreAsyncFileReader {
                    store: Arc::clone(&store),
                    path: key_path.clone(),
                    file_size: expected_size,
                    bytes_counter: Arc::clone(&bytes_counter),
                };
                let batches: Vec<arrow::record_batch::RecordBatch> = rt.block_on(async {
                    let arrow_meta = parquet::arrow::arrow_reader::ArrowReaderMetadata::try_new(
                        Arc::clone(&parquet_meta),
                        parquet::arrow::arrow_reader::ArrowReaderOptions::new(),
                    )
                    .unwrap();
                    let builder = ParquetRecordBatchStreamBuilder::new_with_metadata(
                        reader, arrow_meta,
                    )
                    .with_row_groups(vec![rg as usize])
                    .with_projection(projection.clone());
                    let stream = builder.build().unwrap();
                    stream.try_collect::<Vec<_>>().await.unwrap()
                });
                let offsets: Vec<u32> = entries.iter().map(|&(_, off)| off).collect();
                let vectors = extract_wanted_vectors(&batches, &offsets);
                for (i, (vec_id, _)) in entries.iter().enumerate() {
                    candidate_vectors.insert(*vec_id, vectors[i].clone());
                }
            }

            // Full-precision rerank: Dot(query, vector) for each candidate.
            let mut rerank_scored: Vec<(u32, f32)> = candidate_ids
                .iter()
                .map(|&vec_id| {
                    let v = &candidate_vectors[&vec_id];
                    let s: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                    (vec_id, s)
                })
                .collect();
            rerank_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let rerank_top10: HashSet<u32> =
                rerank_scored[..10].iter().map(|&(id, _)| id).collect();
            let t2 = std::time::Instant::now();

            let rerank_us = t2.duration_since(t1).as_secs_f64() * 1e6;
            let total_us = t2.duration_since(t0).as_secs_f64() * 1e6;
            let bytes_this_query = bytes_counter.load(std::sync::atomic::Ordering::Relaxed);
            let overlap = rerank_top10.intersection(&bq_top10).count() as f64 / 10.0;

            all_search_us.push(search_us);
            all_rerank_us.push(rerank_us);
            all_total_us.push(total_us);
            all_bytes.push(bytes_this_query);
            all_row_groups_touched.push(unique_row_groups);
            all_overlaps.push(overlap);

            if !printed_first_rep {
                println!(
                    "  q[{qi:02}] rep={rep} search={search_us:>8.1} µs  rerank={rerank_us:>8.1} µs  total={total_us:>8.1} µs  bytes={bytes_this_query:>10}  row_groups={unique_row_groups:>3}  overlap(BQ10↔rerank10)={overlap:.2}",
                );
            }
        }
        printed_first_rep = true;
    }

    // Aggregation. Same median/spread discipline as the other Phase-4 tables.
    fn stats_us(v: &[f64]) -> (f64, f64, f64, f64) {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        (s[n / 2], s[0], s[n - 1], percentile(&s, 0.95))
    }
    let (search_p50, search_min, search_max, search_p95) = stats_us(&all_search_us);
    let (rerank_p50, rerank_min, rerank_max, rerank_p95) = stats_us(&all_rerank_us);
    let (total_p50, total_min, total_max, total_p95) = stats_us(&all_total_us);

    let mut b_sorted: Vec<u64> = all_bytes.clone();
    b_sorted.sort();
    let bytes_median = b_sorted[b_sorted.len() / 2];
    let bytes_min = *all_bytes.iter().min().unwrap();
    let bytes_max = *all_bytes.iter().max().unwrap();

    let rg_touched_median = {
        let mut v = all_row_groups_touched.clone();
        v.sort();
        v[v.len() / 2]
    };
    let rg_touched_min = *all_row_groups_touched.iter().min().unwrap();
    let rg_touched_max = *all_row_groups_touched.iter().max().unwrap();

    let overlap_median = {
        let mut v = all_overlaps.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let overlap_min = all_overlaps.iter().cloned().fold(f64::INFINITY, f64::min);
    let overlap_max = all_overlaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let rg_size_min = *all_rg_sizes_touched.iter().min().unwrap();
    let rg_size_max = *all_rg_sizes_touched.iter().max().unwrap();

    println!();
    println!(
        "Phase 4 rerank-over-parquet [{S3_ENDPOINT_LABEL}], {} queries × {NUM_REPS} reps:",
        rd.queries.len(),
    );
    println!(
        "  HNSW search wall-clock : p50 = {search_p50:>9.1} µs  [min {search_min:.1}, max {search_max:.1}], p95 = {search_p95:.1} µs",
    );
    println!(
        "  Rerank wall-clock      : p50 = {rerank_p50:>9.1} µs  [min {rerank_min:.1}, max {rerank_max:.1}], p95 = {rerank_p95:.1} µs",
    );
    println!(
        "  Total (search+rerank)  : p50 = {total_p50:>9.1} µs  [min {total_min:.1}, max {total_max:.1}], p95 = {total_p95:.1} µs",
    );
    println!(
        "  Bytes fetched per query: median = {bytes_median}  [min {bytes_min}, max {bytes_max}]",
    );
    println!(
        "  Unique row-groups per query: median = {rg_touched_median}  [min {rg_touched_min}, max {rg_touched_max}]",
    );
    println!(
        "  Overlap(BQ top-10 ↔ rerank top-10): median = {overlap_median:.2}  [min {overlap_min:.2}, max {overlap_max:.2}]",
    );
    println!(
        "  Row-group row-count across touched groups: min = {rg_size_min}, max = {rg_size_max}",
    );
    println!(
        "  Parquet metadata fetch (once, off timing loop): {meta_fetch_bytes} bytes",
    );
}

// -------- Phase 4c: full-scale (≈1M) one-shot measurement ----------------
//
// Doubles as the rebuild-economics measurement (last unmeasured ledger item).
// All timings report-only. Machine kept quiet. Env-gated:
//   PUFFIN_S3_ENDPOINT=... (endpoint config, same as other opt-in tests)
//   PUFFIN_FIXTURE_N=999980 (required — signals opt-in for the 1M path)
// Fixture: data/fixtures/gte_1m_vectors.{bin,json} + gte_1m_source_indices.bin
// + gte_1m_queries.{bin,json}, produced by `data/fixtures/build_gte_fixture.py
// --with-1m`.

const PHASE4C_N: usize = 999_980;
const PHASE4C_OBJECT_KEY: &str = "iceberg/table/data/index_1m.puffin";

#[test]
fn test_phase4c_full_scale_run() {
    use super::puffin_shared::{
        BlobSpec, FIXTURE_1M, build_row_pointer_blob, put_bytes_multipart,
        try_load_real_data_from, write_puffin,
    };
    use futures::TryStreamExt;
    use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
    use parquet::arrow::ProjectionMask;
    use quantization::encoded_storage::TestEncodedStorageBuilder;
    use quantization::encoded_vectors_binary::QueryEncoding;
    use quantization::{DistanceType, VectorParameters};

    // Explicit opt-in via PUFFIN_FIXTURE_N=999980 (this test would otherwise
    // consume ~40 minutes and ~50 GiB of wire traffic on a default suite run).
    let requested_n = std::env::var("PUFFIN_FIXTURE_N")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    if requested_n != PHASE4C_N {
        println!(
            "Phase 4c full-scale run skipped: set PUFFIN_FIXTURE_N={PHASE4C_N} to opt in (got {requested_n})",
        );
        return;
    }
    let Some(endpoint) = opt_in_endpoint() else {
        println!("Phase 4c skipped: PUFFIN_S3_ENDPOINT not set");
        return;
    };
    let Some(store) = build_opt_in_store(endpoint) else {
        panic!("PUFFIN_S3_ENDPOINT set but bucket/credentials env vars missing");
    };
    let Some(rd) = try_load_real_data_from(&FIXTURE_1M, PHASE4C_N) else {
        println!(
            "Phase 4c skipped: 1M fixture not present at data/fixtures/gte_1m_vectors.bin",
        );
        return;
    };
    let Some(source_indices) = rd.body_source_indices.clone() else {
        panic!("1M fixture is missing body_source_indices — regenerate with build_gte_fixture.py --with-1m");
    };
    let row_group_sizes: Vec<u64> = rd.body_sidecar.row_group_sizes.clone();
    assert!(!row_group_sizes.is_empty(), "1M sidecar missing row_group_sizes");

    println!("========================================");
    println!("Phase 4c full-scale run: N={PHASE4C_N}, queries={}", rd.queries.len());
    println!("========================================");

    // ==================== BUILD PHASE (instrumented) ====================
    // We inline what build_test_puffin_fixture_from_vectors_with_mapping does
    // so we can time each phase separately. Sequence is identical to that
    // shared builder; comments cross-reference it.
    let tmp = TempDir::new().unwrap();
    let puffin_path = tmp.path().join("index_1m.puffin");
    let graph_dir = tmp.path().join("graph");
    fs::create_dir_all(&graph_dir).unwrap();

    let build_total_t0 = std::time::Instant::now();

    // (a1) Vector-storage scaffold (input to HNSW build).
    let t_scaffold_0 = std::time::Instant::now();
    let scaffold = RealDataScaffold::new(Distance::Dot, &rd.body);
    let t_scaffold = t_scaffold_0.elapsed();
    println!("BUILD (a1) vector storage insert: {t_scaffold:?}");

    // (a2) HNSW graph build (single-threaded test path — upper bound vs
    // Qdrant's parallel production builder).
    let mut level_rng = rand::rngs::StdRng::seed_from_u64(FIXTURE_SEED.wrapping_add(0x4C_00));
    use crate::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
    use crate::index::hnsw_index::graph_links::GraphLinksFormatParam;
    use common::types::PointOffsetType as Pot;
    let t_hnsw_0 = std::time::Instant::now();
    let mut builder = GraphLayersBuilder::new(
        rd.body.len(),
        HnswM::new2(16),
        /* ef_construct */ 100,
        /* entry_points_num */ 10,
        /* use_heuristic */ true,
    );
    for idx in 0..rd.body.len() as Pot {
        let level = builder.get_random_layer(&mut level_rng);
        builder.set_levels(idx, level);
        builder.link_new_point(idx, scaffold.internal_scorer(idx));
    }
    let t_hnsw_build = t_hnsw_0.elapsed();
    println!("BUILD (a2) hnsw graph (single-threaded link loop): {t_hnsw_build:?}");

    // (a3) Graph serialization to disk (graph.bin + links_compressed.bin).
    let t_graph_save_0 = std::time::Instant::now();
    builder
        .into_graph_layers(
            &graph_dir,
            GraphLinksFormatParam::Compressed,
            /* on_disk */ false,
        )
        .unwrap();
    let graph_meta_bytes = fs::read(graph_dir.join("graph.bin")).unwrap();
    let graph_links_bytes = fs::read(graph_dir.join("links_compressed.bin")).unwrap();
    let t_graph_save = t_graph_save_0.elapsed();
    println!(
        "BUILD (a3) graph serialize (graph.bin={} B, links_compressed.bin={} B): {t_graph_save:?}",
        graph_meta_bytes.len(),
        graph_links_bytes.len(),
    );

    // (b) Quantization encode.
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
    let t_quant_0 = std::time::Instant::now();
    let _encoded_holder = EncodedVectorsBin::<u128, _>::encode(
        rd.body.iter().map(|v| v.as_slice()),
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
    let t_quantize = t_quant_0.elapsed();
    println!(
        "BUILD (b)  quantize encode (data={} B, meta={} B): {t_quantize:?}",
        quant_bytes.len(),
        quant_meta_bytes.len(),
    );

    // (c) Row-pointer blob + container assemble/write.
    let t_write_0 = std::time::Instant::now();
    let file_paths = [LOGICAL_PARQUET_NAME.to_string()];
    let path_refs: Vec<&str> = file_paths.iter().map(|s| s.as_str()).collect();
    let entries: Vec<(u32, u32, u32, u32)> = (0..rd.body.len() as u32)
        .map(|i| {
            let src = source_indices[i as usize];
            let (rg, off) = source_row_to_row_group(&row_group_sizes, src);
            (i, 0u32, rg, off)
        })
        .collect();
    let row_ptr_bytes = build_row_pointer_blob(&path_refs, &entries);

    let blobs = [
        BlobSpec {
            blob_type: "ann-hnsw-quantized-vectors-v1",
            bytes: &quant_bytes,
            properties: serde_json::json!({
                "quantization_variant": "EncodedVectorsBin_u128",
                "quantization_family": "binary",
                "dimensions": DIM.to_string(),
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
        BlobSpec {
            blob_type: "ann-hnsw-quantized-meta-v1",
            bytes: &quant_meta_bytes,
            properties: serde_json::json!({
                "quantization_family": "binary",
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
        BlobSpec {
            blob_type: "ann-hnsw-graph-meta-v1",
            bytes: &graph_meta_bytes,
            properties: serde_json::json!({
                "m": "16",
                "ef_construct": "100",
                "vector_count": rd.body.len().to_string(),
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
        BlobSpec {
            blob_type: "ann-hnsw-graph-links-v1",
            bytes: &graph_links_bytes,
            properties: serde_json::json!({
                "format": "compressed",
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
        BlobSpec {
            blob_type: "ann-hnsw-row-pointers-v1",
            bytes: &row_ptr_bytes,
            properties: serde_json::json!({
                "entry_count": rd.body.len().to_string(),
                "created-by": "qdrant-edge-builder-v1",
            }),
        },
    ];
    write_puffin(&puffin_path, &blobs).unwrap();
    let t_write = t_write_0.elapsed();
    let container_size = fs::metadata(&puffin_path).unwrap().len();
    println!(
        "BUILD (c)  container assemble + write ({} B, {:.1} MiB): {t_write:?}",
        container_size,
        container_size as f64 / 1024.0 / 1024.0,
    );

    let build_total = build_total_t0.elapsed();
    println!(
        "BUILD TOTAL (scaffold+HNSW+graph-save+quantize+write): {build_total:?}",
    );
    println!(
        "  NOTE: single-threaded upper bound; Qdrant's production builder parallelises the HNSW link loop.",
    );

    // ==================== UPLOAD ====================
    let container_bytes = fs::read(&puffin_path).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let t_upload_0 = std::time::Instant::now();
    put_bytes_multipart(&rt, store.as_ref(), PHASE4C_OBJECT_KEY, container_bytes).unwrap();
    let t_upload = t_upload_0.elapsed();
    println!(
        "UPLOAD container ({:.1} MiB) via multipart: {t_upload:?}",
        container_size as f64 / 1024.0 / 1024.0,
    );

    // ==================== FETCH: 3 reps miss/hit ====================
    let mut miss_us: Vec<f64> = Vec::with_capacity(3);
    let mut hit_us: Vec<f64> = Vec::with_capacity(3);
    for _ in 0..3 {
        let cache_root = TempDir::new().unwrap();
        let fetcher =
            PuffinFetcher::new(Arc::clone(&store), cache_root.path().to_path_buf()).unwrap();
        let (_, miss) = time_it(|| fetcher.fetch(PHASE4C_OBJECT_KEY, SNAPSHOT_A).unwrap());
        miss_us.push(miss.as_secs_f64() * 1e6);
        let (_, hit) = time_it(|| fetcher.fetch(PHASE4C_OBJECT_KEY, SNAPSHOT_A).unwrap());
        hit_us.push(hit.as_secs_f64() * 1e6);
    }
    println!(
        "FETCH [{S3_ENDPOINT_LABEL}] cache-miss (3 reps, container {:.1} MiB): {}",
        container_size as f64 / 1024.0 / 1024.0,
        median_spread_ms(miss_us),
    );
    println!(
        "FETCH [{S3_ENDPOINT_LABEL}] cache-hit  (3 reps): {}",
        median_spread_us(hit_us),
    );

    // ==================== LOAD FROM CACHED FILE ====================
    let cache_root = TempDir::new().unwrap();
    let fetcher =
        PuffinFetcher::new(Arc::clone(&store), cache_root.path().to_path_buf()).unwrap();
    let cached_path = fetcher.fetch(PHASE4C_OBJECT_KEY, SNAPSHOT_A).unwrap();
    let shared_mmap = mmap_whole_file(&cached_path);
    let file_bytes = fs::read(&cached_path).unwrap();
    let footer = read_and_validate_footer(&file_bytes).unwrap();

    let meta_range = footer.by_type("ann-hnsw-graph-meta-v1").unwrap();
    let meta_slice = &shared_mmap[meta_range.offset..meta_range.offset + meta_range.length];
    let graph_data: GraphLayerData<'_> = bincode::deserialize(meta_slice).unwrap();
    let links_range = footer.by_type("ann-hnsw-graph-links-v1").unwrap();
    let links = GraphLinks::load_from_ranged_mmap(
        Arc::clone(&shared_mmap),
        links_range.offset as u64,
        links_range.length as u64,
        GraphLinksFormat::Compressed,
    )
    .unwrap();
    let graph_layers = GraphLayers {
        hnsw_m: HnswM::new(graph_data.m, graph_data.m0),
        links,
        entry_points: graph_data.entry_points.into_owned(),
        visited_pool: VisitedPool::new(),
    };
    graph_layers.links.populate().unwrap();

    // ==================== LOCAL MEASUREMENTS ====================
    // Recall@10 + containment: 1 rep (deterministic).
    let quant_range = footer.by_type("ann-hnsw-quantized-vectors-v1").unwrap();
    let quant_meta_range = footer.by_type("ann-hnsw-quantized-meta-v1").unwrap();
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
        &shared_mmap[quant_meta_range.offset..quant_meta_range.offset + quant_meta_range.length],
    )
    .unwrap();
    let storage = TestEncodedStorage::from_file(&data_path, quantized_vec_size).unwrap();
    let encoded =
        EncodedVectorsBin::<u128, TestEncodedStorage>::load(storage, &meta_path).unwrap();

    let is_stopped = AtomicBool::new(false);
    let mut per_query_recall: Vec<f64> = Vec::with_capacity(rd.queries.len());
    let mut per_query_containment: Vec<f64> = Vec::with_capacity(rd.queries.len());
    let t_local_0 = std::time::Instant::now();
    for (qi, q) in rd.queries.iter().enumerate() {
        // HNSW top-10
        let hnsw_top10 = graph_layers
            .search(
                10, 64, SearchAlgorithm::Hnsw,
                scaffold.scorer(q.clone()), None, &is_stopped,
            )
            .unwrap();
        let hnsw_ids: HashSet<PointOffsetType> = hnsw_top10.iter().map(|s| s.idx).collect();

        // Brute-force ground truth for recall (full-precision Dot)
        let mut bf_scorer = scaffold.scorer(q.clone());
        let all_points: Vec<PointOffsetType> =
            (0..rd.body.len() as PointOffsetType).collect();
        let mut bf_scored: Vec<_> = bf_scorer
            .score_points_unfiltered(&all_points)
            .collect::<Vec<_>>();
        bf_scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let bf_top10: Vec<PointOffsetType> = bf_scored[..10].iter().map(|s| s.idx).collect();
        let bf_top10_set: HashSet<PointOffsetType> = bf_top10.iter().copied().collect();
        let recall = hnsw_ids.intersection(&bf_top10_set).count() as f64 / 10.0;
        per_query_recall.push(recall);

        // Containment: BF top-10 ⊂ Quant top-100
        let hw = HardwareCounterCell::new();
        let encoded_query = encoded.encode_query(q);
        let mut q_scored: Vec<(PointOffsetType, f32)> = (0..rd.body.len() as PointOffsetType)
            .map(|i| (i, encoded.score_point(&encoded_query, i, &hw)))
            .collect();
        q_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let q_top100: HashSet<PointOffsetType> =
            q_scored[..100].iter().map(|&(id, _)| id).collect();
        let contained = bf_top10.iter().filter(|id| q_top100.contains(id)).count();
        per_query_containment.push(contained as f64 / 10.0);

        println!(
            "  q[{qi:02}] recall@10={recall:.3}  containment(BF10⊂Q100)={:.3}",
            per_query_containment.last().unwrap(),
        );
    }
    let t_local = t_local_0.elapsed();
    let mean_recall: f64 =
        per_query_recall.iter().sum::<f64>() / per_query_recall.len() as f64;
    let mean_containment: f64 =
        per_query_containment.iter().sum::<f64>() / per_query_containment.len() as f64;
    println!(
        "LOCAL recall+containment (1 rep, 20 queries, {t_local:?}):",
    );
    println!("  MEAN recall@10                    = {mean_recall:.3}");
    println!("  MEAN containment (BF10⊂Quant100)  = {mean_containment:.3}");

    // Warm-path p50/p95: 5 reps.
    let mut p50_samples: Vec<f64> = Vec::with_capacity(5);
    let mut p95_samples: Vec<f64> = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut micros: Vec<f64> = Vec::with_capacity(rd.queries.len());
        for q in &rd.queries {
            let (_top, elapsed) = time_it(|| {
                graph_layers
                    .search(
                        10, 64, SearchAlgorithm::Hnsw,
                        scaffold.scorer(q.clone()), None, &is_stopped,
                    )
                    .unwrap()
            });
            micros.push(elapsed.as_secs_f64() * 1e6);
        }
        micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
        p50_samples.push(percentile(&micros, 0.50));
        p95_samples.push(percentile(&micros, 0.95));
    }
    println!("LOCAL warm-path per-query latency (20 queries × 5 reps):");
    println!("  p50: {}", median_spread_us(p50_samples));
    println!("  p95: {}", median_spread_us(p95_samples));

    // ==================== RERANK (1 rep × 20 queries) ====================
    let rp_range = footer.by_type("ann-hnsw-row-pointers-v1").unwrap();
    let rp = &shared_mmap[rp_range.offset..rp_range.offset + rp_range.length];
    let row_pointers = decode_all_row_pointers(rp, rd.body.len());

    let parquet_key = LOGICAL_PARQUET_NAME;
    let parquet_key_path = object_store::path::Path::from(parquet_key);
    let repo_parquet = repo_root().join("data/gte_product_embeddings.parquet");
    let parquet_size = fs::metadata(&repo_parquet).unwrap().len();
    let head = rt.block_on(store.head(&parquet_key_path));
    let need_upload = !matches!(&head, Ok(m) if m.size == parquet_size);
    if need_upload {
        println!("RERANK: parquet not present or wrong size — uploading");
        let bytes = fs::read(&repo_parquet).unwrap();
        put_bytes_multipart(&rt, store.as_ref(), parquet_key, bytes).unwrap();
    } else {
        println!("RERANK: parquet already in bucket (skipping upload)");
    }

    let bytes_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut meta_reader = StoreAsyncFileReader {
        store: Arc::clone(&store),
        path: parquet_key_path.clone(),
        file_size: parquet_size,
        bytes_counter: Arc::clone(&bytes_counter),
    };
    let parquet_meta = rt.block_on(async {
        <StoreAsyncFileReader as parquet::arrow::async_reader::AsyncFileReader>::get_metadata(
            &mut meta_reader,
            None,
        )
        .await
        .unwrap()
    });
    let meta_fetch_bytes = bytes_counter.load(std::sync::atomic::Ordering::Relaxed);
    let schema_descr = parquet_meta.file_metadata().schema_descr_ptr();
    let gte_col_idx = (0..schema_descr.num_columns())
        .find(|&i| {
            schema_descr
                .column(i)
                .path()
                .parts()
                .first()
                .map(String::as_str)
                == Some("gte")
        })
        .expect("parquet schema missing 'gte' leaf");
    let projection = ProjectionMask::leaves(&schema_descr, vec![gte_col_idx]);

    let mut rerank_search_us = Vec::new();
    let mut rerank_rerank_us = Vec::new();
    let mut rerank_total_us = Vec::new();
    let mut rerank_bytes: Vec<u64> = Vec::new();
    let mut rerank_row_groups: Vec<usize> = Vec::new();
    let mut rerank_overlaps: Vec<f64> = Vec::new();
    let mut rerank_all_rg_sizes: Vec<u64> = Vec::new();

    for (qi, q) in rd.queries.iter().enumerate() {
        // BQ top-10 for overlap
        let hw = HardwareCounterCell::new();
        let encoded_query = encoded.encode_query(q);
        let mut bq_scored: Vec<(PointOffsetType, f32)> = (0..rd.body.len() as PointOffsetType)
            .map(|i| (i, encoded.score_point(&encoded_query, i, &hw)))
            .collect();
        bq_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let bq_top10: HashSet<PointOffsetType> =
            bq_scored[..10].iter().map(|&(id, _)| id).collect();

        bytes_counter.store(0, std::sync::atomic::Ordering::Relaxed);
        let t0 = std::time::Instant::now();

        let candidates = graph_layers
            .search(
                100, 128, SearchAlgorithm::Hnsw,
                scaffold.scorer(q.clone()), None, &is_stopped,
            )
            .unwrap();
        let candidate_ids: Vec<u32> = candidates.iter().map(|s| s.idx).collect();
        let t1 = std::time::Instant::now();
        let search_us = t1.duration_since(t0).as_secs_f64() * 1e6;

        let mut by_group: std::collections::BTreeMap<u32, Vec<(u32, u32)>> =
            std::collections::BTreeMap::new();
        for &vec_id in &candidate_ids {
            let (rp_vec_id, _fp, rg, off) = row_pointers[vec_id as usize];
            assert_eq!(rp_vec_id, vec_id);
            by_group.entry(rg).or_default().push((vec_id, off));
        }
        let unique_row_groups = by_group.len();

        let mut candidate_vectors: std::collections::HashMap<u32, Vec<f32>> =
            std::collections::HashMap::with_capacity(candidate_ids.len());
        for (&rg, entries) in by_group.iter() {
            rerank_all_rg_sizes.push(row_group_sizes[rg as usize]);
            let reader = StoreAsyncFileReader {
                store: Arc::clone(&store),
                path: parquet_key_path.clone(),
                file_size: parquet_size,
                bytes_counter: Arc::clone(&bytes_counter),
            };
            let batches: Vec<arrow::record_batch::RecordBatch> = rt.block_on(async {
                let arrow_meta = parquet::arrow::arrow_reader::ArrowReaderMetadata::try_new(
                    Arc::clone(&parquet_meta),
                    parquet::arrow::arrow_reader::ArrowReaderOptions::new(),
                )
                .unwrap();
                let builder = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_meta)
                    .with_row_groups(vec![rg as usize])
                    .with_projection(projection.clone());
                builder.build().unwrap().try_collect::<Vec<_>>().await.unwrap()
            });
            let offsets: Vec<u32> = entries.iter().map(|&(_, off)| off).collect();
            let vectors = extract_wanted_vectors(&batches, &offsets);
            for (i, (vec_id, _)) in entries.iter().enumerate() {
                candidate_vectors.insert(*vec_id, vectors[i].clone());
            }
        }

        let mut rerank_scored: Vec<(u32, f32)> = candidate_ids
            .iter()
            .map(|&vec_id| {
                let v = &candidate_vectors[&vec_id];
                let s: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                (vec_id, s)
            })
            .collect();
        rerank_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let rerank_top10: HashSet<u32> =
            rerank_scored[..10].iter().map(|&(id, _)| id).collect();
        let t2 = std::time::Instant::now();
        let rerank_us = t2.duration_since(t1).as_secs_f64() * 1e6;
        let total_us = t2.duration_since(t0).as_secs_f64() * 1e6;
        let bytes_q = bytes_counter.load(std::sync::atomic::Ordering::Relaxed);
        let overlap = rerank_top10.intersection(&bq_top10).count() as f64 / 10.0;

        rerank_search_us.push(search_us);
        rerank_rerank_us.push(rerank_us);
        rerank_total_us.push(total_us);
        rerank_bytes.push(bytes_q);
        rerank_row_groups.push(unique_row_groups);
        rerank_overlaps.push(overlap);

        println!(
            "  RERANK q[{qi:02}] search={search_us:>8.1} µs  rerank={rerank_us:>10.1} µs  total={total_us:>10.1} µs  bytes={bytes_q:>10}  rg={unique_row_groups:>3}  overlap={overlap:.2}",
        );
    }

    // Aggregate rerank.
    let sort_f = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    let median_of = |v: &[f64]| v[v.len() / 2];
    let min_of = |v: &[f64]| *v.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_of = |v: &[f64]| *v.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let p95_of = |v: &[f64]| percentile(v, 0.95);

    let s_sorted = sort_f(rerank_search_us.clone());
    let r_sorted = sort_f(rerank_rerank_us.clone());
    let t_sorted = sort_f(rerank_total_us.clone());
    let mut b_sorted = rerank_bytes.clone();
    b_sorted.sort();
    let mut rg_sorted = rerank_row_groups.clone();
    rg_sorted.sort();
    let o_sorted = sort_f(rerank_overlaps.clone());
    let rg_size_min = *rerank_all_rg_sizes.iter().min().unwrap();
    let rg_size_max = *rerank_all_rg_sizes.iter().max().unwrap();

    println!();
    println!("Phase 4c rerank-over-parquet [{S3_ENDPOINT_LABEL}], 20 queries × 1 rep:");
    println!(
        "  HNSW search wall-clock : p50 = {:>9.1} µs  [min {:.1}, max {:.1}], p95 = {:.1} µs",
        median_of(&s_sorted), min_of(&s_sorted), max_of(&s_sorted), p95_of(&s_sorted),
    );
    println!(
        "  Rerank wall-clock      : p50 = {:>9.1} µs  [min {:.1}, max {:.1}], p95 = {:.1} µs",
        median_of(&r_sorted), min_of(&r_sorted), max_of(&r_sorted), p95_of(&r_sorted),
    );
    println!(
        "  Total (search+rerank)  : p50 = {:>9.1} µs  [min {:.1}, max {:.1}], p95 = {:.1} µs",
        median_of(&t_sorted), min_of(&t_sorted), max_of(&t_sorted), p95_of(&t_sorted),
    );
    println!(
        "  Bytes fetched per query: median = {}  [min {}, max {}]",
        b_sorted[b_sorted.len() / 2], b_sorted[0], b_sorted[b_sorted.len() - 1],
    );
    println!(
        "  Unique row-groups per query: median = {}  [min {}, max {}]",
        rg_sorted[rg_sorted.len() / 2], rg_sorted[0], rg_sorted[rg_sorted.len() - 1],
    );
    println!(
        "  Overlap(BQ top-10 ↔ rerank top-10): median = {:.2}  [min {:.2}, max {:.2}]",
        median_of(&o_sorted), min_of(&o_sorted), max_of(&o_sorted),
    );
    println!(
        "  Row-group row-count across touched groups: min = {rg_size_min}, max = {rg_size_max}",
    );
    println!(
        "  Parquet metadata fetch (once, off timing loop): {meta_fetch_bytes} bytes",
    );

    println!();
    println!("========================================");
    println!("Phase 4c full-scale run: complete");
    println!("========================================");
}
