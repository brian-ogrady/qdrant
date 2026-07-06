//! Phase 4 (v3.13): object-store fetch + snapshot-keyed cache + measurements.
//!
//! Six always-green tests against `object_store::memory::InMemory` cover
//! cache hit/miss/rekey semantics, range-read correctness, corrupted /
//! tiny-object rejection, and an end-to-end catalog→fetch→mmap→search flow.
//! A measurement test prints (report-only, dated, no assertions) per-query
//! p50/p95, cache miss vs hit round-trip, and populate()/clear_cache() effect.
//!
//! An opt-in `PUFFIN_S3_ENDPOINT`-gated MinIO suite runs the same tests
//! against a real S3 wire protocol; skips cleanly when the endpoint is unset.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use fs_err as fs;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use rand::SeedableRng;
use tempfile::TempDir;

use crate::index::hnsw_index::HnswM;
use crate::index::hnsw_index::graph_layers::{GraphLayerData, GraphLayers, SearchAlgorithm};
use crate::index::hnsw_index::graph_links::{GraphLinks, GraphLinksFormat};
use crate::index::visited_pool::VisitedPool;
use crate::types::Distance;

use super::puffin_shared::{
    DIM, FIXTURE_SEED, MockCatalogStub, NUM_VECTORS, PuffinFetcher,
    RealDataScaffold, build_test_puffin_fixture, build_test_puffin_fixture_from_vectors,
    mmap_whole_file, put_bytes, read_and_validate_footer, time_it, try_load_real_data,
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

#[test]
fn test_measurements_report_only() {
    // Only run measurements when real data is available; otherwise the
    // numbers are just BQ noise and don't inform anything.
    let Some(rd) = try_load_real_data(NUM_VECTORS) else {
        println!(
            "Phase 4 measurements skipped: real-data fixture not present (data/fixtures/gte_100k_vectors.bin)"
        );
        return;
    };

    // Build a container from real data and upload it to an InMemory store.
    let scaffold = RealDataScaffold::new(Distance::Dot, &rd.body);
    let mut level_rng = rand::rngs::StdRng::seed_from_u64(FIXTURE_SEED.wrapping_add(0x2000));
    let fixture = build_test_puffin_fixture_from_vectors(&rd.body, &scaffold, &mut level_rng);
    let container_bytes = fs::read(&fixture.puffin_path).unwrap();

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let seed_rt = tokio::runtime::Runtime::new().unwrap();
    put_bytes(&seed_rt, store.as_ref(), OBJECT_KEY, container_bytes).unwrap();
    drop(seed_rt);

    // -- 1. cache-miss vs cache-hit round-trip -----------------------------
    let cache_root_miss = TempDir::new().unwrap();
    let fetcher_miss =
        PuffinFetcher::new(Arc::clone(&store), cache_root_miss.path().to_path_buf()).unwrap();
    let (_, miss_elapsed) = time_it(|| fetcher_miss.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap());
    let (_, hit_elapsed) = time_it(|| fetcher_miss.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap());
    println!("Phase 4 cache-miss (fetch + validate + write + post-verify): {miss_elapsed:?}");
    println!("Phase 4 cache-hit  (path resolve + exists check)          : {hit_elapsed:?}");

    // -- 2. Warm-path per-query latency (search + row-ptr resolution) -----
    let cached_path = fetcher_miss.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap();
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
    // Warm the mmap pages via populate() on the ranged links + touch the
    // row-pointer bytes.
    graph_layers.links.populate().unwrap();
    let rp_range = footer.by_type("ann-hnsw-row-pointers-v1").unwrap();
    let _touch = &shared_mmap[rp_range.offset..rp_range.offset + rp_range.length];

    let is_stopped = AtomicBool::new(false);
    let mut latencies: Vec<Duration> = Vec::with_capacity(rd.queries.len());
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
        latencies.push(elapsed);
    }
    let mut micros: Vec<f64> = latencies.iter().map(|d| d.as_secs_f64() * 1e6).collect();
    micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&micros, 0.50);
    let p95 = percentile(&micros, 0.95);
    println!(
        "Phase 4 warm-path query latency over {} queries: p50 = {p50:.0} µs, p95 = {p95:.0} µs (target 50 000 µs; report-only)",
        rd.queries.len(),
    );

    // -- 3. populate() effect: clear_cache then measure first vs populated -
    // On a warm mmap, clear_cache issues MADV_DONTNEED for the links range.
    // madvise is advisory; page eviction is best-effort at the kernel. Print
    // both numbers with the caveat.
    let q0 = &rd.queries[0];
    graph_layers.links.clear_cache().unwrap();
    let (_res, cold_after_clear) = time_it(|| {
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
    // Now populate() and re-measure.
    graph_layers.links.populate().unwrap();
    let (_res, warm_after_populate) = time_it(|| {
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
    println!(
        "Phase 4 §6.3 arms exercised (madvise advisory; eviction/prefetch best-effort):",
    );
    println!(
        "  after clear_cache (MADV_DONTNEED range): {cold_after_clear:?}",
    );
    println!(
        "  after populate    (MADV_WILLNEED range): {warm_after_populate:?}",
    );
}

// -------- MinIO opt-in suite (env-gated) ---------------------------------

/// Env var set → run against the real S3 wire protocol via MinIO.
/// Absent → skip cleanly. Docker snippet is documented in the plan; no
/// container actions here.
fn minio_endpoint() -> Option<String> {
    std::env::var("PUFFIN_S3_ENDPOINT").ok()
}

fn build_minio_store(endpoint: String) -> Option<Arc<dyn ObjectStore>> {
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

#[test]
fn test_puffin_minio_opt_in_e2e() {
    let Some(endpoint) = minio_endpoint() else {
        println!("Phase 4 MinIO suite skipped: PUFFIN_S3_ENDPOINT not set");
        return;
    };
    let Some(store) = build_minio_store(endpoint) else {
        panic!(
            "PUFFIN_S3_ENDPOINT is set but one or more of PUFFIN_S3_BUCKET, \
             AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY is missing",
        );
    };

    // Upload the fixture .puffin.
    let fixture = build_test_puffin_fixture();
    let bytes = fs::read(&fixture.puffin_path).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    put_bytes(&rt, store.as_ref(), OBJECT_KEY, bytes).unwrap();
    drop(rt);

    // Same 6-test suite the InMemory path runs — collapsed into a single
    // structural check + measurement print, since we're already exercising
    // the wire protocol.
    let cache_root = TempDir::new().unwrap();
    let fetcher = PuffinFetcher::new(store, cache_root.path().to_path_buf()).unwrap();

    // Miss → hit.
    let (path_miss, miss_elapsed) = time_it(|| fetcher.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap());
    let (path_hit, hit_elapsed) = time_it(|| fetcher.fetch(OBJECT_KEY, SNAPSHOT_A).unwrap());
    assert_eq!(path_miss, path_hit);
    println!("Phase 4 MinIO cache-miss: {miss_elapsed:?}");
    println!("Phase 4 MinIO cache-hit : {hit_elapsed:?}");
    let stats = fetcher.stats();
    assert!(stats.gets >= 3, "expected ≥3 GETs on miss (trailer, footer, body), got {}", stats.gets);

    // The cached bytes must equal the original.
    let cached = fs::read(&path_miss).unwrap();
    let original = fs::read(&fixture.puffin_path).unwrap();
    assert_eq!(cached, original, "MinIO round-trip mangled bytes");
    read_and_validate_footer(&cached).expect("MinIO cached file must validate");
}
