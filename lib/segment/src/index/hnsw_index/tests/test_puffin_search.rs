//! Phase 3 of the Puffin-HNSW PoC: end-to-end search over a `.puffin` container.
//! Assembles `GraphLayers` from the graph-meta blob + the ranged-mmap links load
//! and drives a query through `GraphLayers::search` (the real production API).
//! Then validates the quantized blob's bytes against the same query without
//! going through `FilteredScorer` — see §6.2 v3.12 for why the disaggregated
//! path can't yet reach `FilteredScorer::new` and how this test sidesteps it.
//!
//! v3.12 relocation: this test lives in `hnsw_index/tests/` because the
//! edge-side location the original spec proposed needs three prereqs not yet
//! in the tree (public `puffin_shared`, segment's testing feature exposed to
//! edge, `GraphLayers::from_parts`). All three are deferred to §6.2
//! `PuffinSegment` work.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use fs_err as fs;
use quantization::encoded_storage::TestEncodedStorage;
use quantization::encoded_vectors_binary::{
    EncodedVectorsBin, Encoding, get_quantized_vector_size_from_params,
};
use quantization::EncodedVectors;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tempfile::TempDir;

use crate::fixtures::index_fixtures::{TestRawScorerProducer, random_vector};
use crate::index::hnsw_index::HnswM;
use crate::index::hnsw_index::graph_layers::{GraphLayerData, GraphLayers, SearchAlgorithm};
use crate::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
use crate::index::hnsw_index::graph_links::{GraphLinks, GraphLinksFormat, GraphLinksFormatParam};
use crate::index::visited_pool::VisitedPool;
use crate::types::Distance;

use super::puffin_shared::{
    DIM, EF_CONSTRUCT, ENTRY_POINTS_NUM, FIXTURE_SEED, M, NUM_VECTORS,
    build_test_puffin_fixture, mmap_whole_file, read_and_validate_footer,
};

// ---------- Step 1 (v3.12 diagnostic): Puffin load-path equality ---------
//
// Deterministic search + same graph must return identical top-10 IDs. If the
// in-RAM-built graph and the Puffin-round-tripped graph diverge on the same
// query, the Puffin load path is buggy — full match fully exonerates it.
// Always-on regardless of real-data fixture availability; runs on the seeded
// random fixture.

/// Build a fresh graph in-RAM (never touches disk, never touches Puffin) with
/// the exact same seed/params as the shared fixture. Returns the graph and the
/// vector holder needed to score against it.
fn build_reference_ram_graph()
    -> (TestRawScorerProducer, GraphLayers) {
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
    let graph = builder.into_graph_layers_ram(GraphLinksFormatParam::Compressed);
    (vector_holder, graph)
}

#[test]
fn test_puffin_graph_load_matches_ram_bit_for_bit() {
    // --- Build the graph twice with the same seed ---
    // Once through the Puffin round-trip:
    let fixture = build_test_puffin_fixture();
    let file_bytes = fs::read(&fixture.puffin_path).unwrap();
    let footer = read_and_validate_footer(&file_bytes).unwrap();
    let shared_mmap = mmap_whole_file(&fixture.puffin_path);

    let meta_range = footer.by_type("ann-hnsw-graph-meta-v1").unwrap();
    let meta_slice = &shared_mmap[meta_range.offset..meta_range.offset + meta_range.length];
    let graph_data: GraphLayerData<'_> =
        bincode::deserialize(meta_slice).expect("bincode GraphLayerData");
    let links_range = footer.by_type("ann-hnsw-graph-links-v1").unwrap();
    let links = GraphLinks::load_from_ranged_mmap(
        Arc::clone(&shared_mmap),
        links_range.offset as u64,
        links_range.length as u64,
        GraphLinksFormat::Compressed,
    )
    .unwrap();
    let graph_from_puffin = GraphLayers {
        hnsw_m: HnswM::new(graph_data.m, graph_data.m0),
        links,
        entry_points: graph_data.entry_points.into_owned(),
        visited_pool: VisitedPool::new(),
    };

    // Once purely in RAM (never serialized):
    let (vector_holder, graph_from_ram) = build_reference_ram_graph();

    // --- Same query, same scorer, same search params ---
    let mut query_rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    let query_vec = random_vector(&mut query_rng, DIM);
    let is_stopped = AtomicBool::new(false);

    let top10_puffin = graph_from_puffin
        .search(
            10,
            64,
            SearchAlgorithm::Hnsw,
            vector_holder.scorer(query_vec.clone()),
            None,
            &is_stopped,
        )
        .unwrap();
    let top10_ram = graph_from_ram
        .search(
            10,
            64,
            SearchAlgorithm::Hnsw,
            vector_holder.scorer(query_vec.clone()),
            None,
            &is_stopped,
        )
        .unwrap();

    let ids_puffin: Vec<PointOffsetType> = top10_puffin.iter().map(|s| s.idx).collect();
    let ids_ram: Vec<PointOffsetType> = top10_ram.iter().map(|s| s.idx).collect();
    let scores_puffin: Vec<f32> = top10_puffin.iter().map(|s| s.score).collect();
    let scores_ram: Vec<f32> = top10_ram.iter().map(|s| s.score).collect();

    assert_eq!(
        ids_puffin, ids_ram,
        "Puffin-loaded graph diverges from RAM-built graph on same query — indicates Puffin load bug.\n  puffin: {ids_puffin:?}\n  ram:    {ids_ram:?}",
    );
    // Belt-and-braces: scores also must match bit-for-bit (Dot with a fixed
    // scorer is deterministic; a divergence here would signal wrong metric
    // wiring even if IDs happened to match).
    assert_eq!(
        scores_puffin, scores_ram,
        "Puffin-loaded graph diverges from RAM-built graph on scores.\n  puffin: {scores_puffin:?}\n  ram:    {scores_ram:?}",
    );

    println!(
        "Phase 3 Step-1 equality verdict: MATCH ({} top-10 IDs identical, scores identical)",
        ids_puffin.len(),
    );
}

/// Recall floor for HNSW search on real data. Applied to the MEAN across 20
/// held-out queries — the intent is to catch construction/load regressions,
/// not to profile HNSW's Pareto frontier. On the random-fallback path (no
/// real-data fixture present) recall is reported but NOT asserted; running
/// with PUFFIN_FIXTURE_N=100000 is also report-only per the Phase 3 plan.
const RECALL_FLOOR: f64 = 0.7;

/// Containment floor for the "brute-force top-10 vs. quantized top-100"
/// check on the real-data path. One-bit binary quantization at 768d on
/// structured (real) embeddings recovers ~1.0 containment at 10× oversample
/// — 0.7 is far above chance and catches a §3.4 invariant break decisively.
/// On the random-fallback path, containment measures ~0.6 (a property of
/// random-noise BQ, not consistency), so this floor only applies when the
/// real-data fixture is present.
const CONTAINMENT_FLOOR: f64 = 0.7;

/// Oversample factor for the containment check.
const QUANT_TOP_K: usize = 100;

/// Number of held-out queries for the recall measurement. Matches the
/// sampler-produced fixture; the random-fallback path fabricates the same
/// count.
const NUM_QUERIES: usize = 20;

/// Env var controlling the fixture body size for the recall test.
/// Default 10_000. Setting to 100_000 (manual, slow) produces the report-only
/// recall number for the spec.
const FIXTURE_SIZE_ENV: &str = "PUFFIN_FIXTURE_N";
const DEFAULT_FIXTURE_N: usize = 10_000;
const REPORT_ONLY_FIXTURE_N: usize = 100_000;

fn resolve_fixture_n() -> usize {
    match std::env::var(FIXTURE_SIZE_ENV) {
        Ok(v) => v.parse::<usize>().unwrap_or_else(|_| {
            panic!("{FIXTURE_SIZE_ENV}={v} is not a valid usize")
        }),
        Err(_) => DEFAULT_FIXTURE_N,
    }
}

/// Compute recall of `hnsw` top-K against `bf` top-K, in [0.0, 1.0].
fn recall(hnsw: &[PointOffsetType], bf: &[PointOffsetType]) -> f64 {
    let bf_set: HashSet<_> = bf.iter().copied().collect();
    hnsw.iter().filter(|id| bf_set.contains(id)).count() as f64 / bf.len() as f64
}

/// Brute-force top-K over the full body using `scaffold`'s full-precision
/// scorer. Returns the top-K point IDs sorted by descending score.
fn brute_force_top_k(
    scaffold: &super::puffin_shared::RealDataScaffold,
    query: &[f32],
    n: usize,
    k: usize,
) -> Vec<PointOffsetType> {
    let mut scorer = scaffold.scorer(query.to_vec());
    let all_points: Vec<PointOffsetType> = (0..n as PointOffsetType).collect();
    let mut scored: Vec<_> = scorer
        .score_points_unfiltered(&all_points)
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .expect("no NaN in scored data")
    });
    scored[..k].iter().map(|s| s.idx).collect()
}

#[test]
fn test_puffin_search_full_roundtrip_with_recall_and_quantized_containment() {
    use super::puffin_shared::{
        build_test_puffin_fixture_from_vectors, try_load_real_data, RealDataScaffold,
    };

    let fixture_n = resolve_fixture_n();
    let is_report_only_size = fixture_n == REPORT_ONLY_FIXTURE_N;
    let real_data = try_load_real_data(fixture_n);

    // Assemble the source-of-truth vectors + queries and describe what we're doing.
    // `enforce_floors` gates BOTH the recall floor and the containment floor —
    // random-fallback / 100k report-only paths measure and print, don't assert
    // (Step 4: "environments without the fixture stay green"). The
    // containment floor is meaningful only against real, structured data;
    // on random 768-dim uniform-in-[-1,1] noise, 1-bit BQ at 10× oversample
    // does not clear 0.7 (measured ~0.60) — that's a property of the metric,
    // not a container-consistency signal.
    let (body_vectors, queries, source_label, enforce_floors): (
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        String,
        bool,
    ) = match real_data {
        Some(mut r) => {
            assert_eq!(r.body.len(), fixture_n);
            let label = format!(
                "real data (source={}, N={}, seed={:#x}, normalized={})",
                r.body_sidecar.source_file, fixture_n, r.body_sidecar.sample_seed,
                r.body_sidecar.normalized,
            );
            // Real data: enforce floor at default N; report-only at 100k.
            let enforce = !is_report_only_size;
            r.queries.truncate(NUM_QUERIES);
            (r.body, r.queries, label, enforce)
        }
        None => {
            // Random fallback: same shape, report-only. Uses a fresh seed
            // that doesn't collide with the Phase 1/2 fixture seed.
            let mut rng = StdRng::seed_from_u64(FIXTURE_SEED.wrapping_add(0x1000));
            let body: Vec<Vec<f32>> = (0..fixture_n)
                .map(|_| random_vector(&mut rng, DIM))
                .collect();
            let queries: Vec<Vec<f32>> = (0..NUM_QUERIES)
                .map(|_| random_vector(&mut rng, DIM))
                .collect();
            let label = format!(
                "random fallback (N={fixture_n}, fixture .bin absent — search test is report-only)",
            );
            (body, queries, label, false)
        }
    };

    println!("Phase 3 fixture: {source_label}");

    // --- Build the container over `body_vectors` ---
    let scaffold = RealDataScaffold::new(Distance::Dot, &body_vectors);
    // Deterministic RNG for level assignment; separate from vector generation
    // so the graph is reproducible regardless of which body path was taken.
    let mut level_rng = StdRng::seed_from_u64(FIXTURE_SEED.wrapping_add(0x2000));
    let fixture =
        build_test_puffin_fixture_from_vectors(&body_vectors, &scaffold, &mut level_rng);

    // --- Load the container back ---
    let file_bytes = fs::read(&fixture.puffin_path).unwrap();
    let footer = read_and_validate_footer(&file_bytes)
        .expect("well-formed fixture must pass footer validation");
    let shared_mmap = mmap_whole_file(&fixture.puffin_path);

    let meta_range = footer.by_type("ann-hnsw-graph-meta-v1").unwrap();
    let meta_slice = &shared_mmap[meta_range.offset..meta_range.offset + meta_range.length];
    let graph_data: GraphLayerData<'_> =
        bincode::deserialize(meta_slice).expect("bincode deserialize GraphLayerData");
    let links_range = footer.by_type("ann-hnsw-graph-links-v1").unwrap();
    let links = GraphLinks::load_from_ranged_mmap(
        Arc::clone(&shared_mmap),
        links_range.offset as u64,
        links_range.length as u64,
        GraphLinksFormat::Compressed,
    )
    .expect("load_from_ranged_mmap");
    let graph_layers = GraphLayers {
        hnsw_m: HnswM::new(graph_data.m, graph_data.m0),
        links,
        entry_points: graph_data.entry_points.into_owned(),
        visited_pool: VisitedPool::new(),
    };

    // --- Mean recall over 20 queries ---
    let is_stopped = AtomicBool::new(false);
    let mut per_query_recall: Vec<f64> = Vec::with_capacity(NUM_QUERIES);
    for (qi, q) in queries.iter().enumerate() {
        let hnsw_top10 = graph_layers
            .search(
                10,
                64,
                SearchAlgorithm::Hnsw,
                scaffold.scorer(q.clone()),
                None,
                &is_stopped,
            )
            .expect("GraphLayers::search");
        assert_eq!(hnsw_top10.len(), 10);
        let hnsw_ids: Vec<PointOffsetType> = hnsw_top10.iter().map(|s| s.idx).collect();
        let bf_top10 = brute_force_top_k(&scaffold, q, fixture_n, 10);
        let r = recall(&hnsw_ids, &bf_top10);
        println!("  Phase 3 recall@10 query[{qi:02}] = {r:.3}");
        per_query_recall.push(r);
    }
    let mean_recall: f64 =
        per_query_recall.iter().sum::<f64>() / per_query_recall.len() as f64;
    println!("Phase 3 MEAN recall@10 over {NUM_QUERIES} queries: {mean_recall:.3}");

    if enforce_floors {
        assert!(
            mean_recall >= RECALL_FLOOR,
            "mean recall@10 {mean_recall:.3} below floor {RECALL_FLOOR}",
        );
    } else {
        println!(
            "  (recall floor NOT enforced — {})",
            if is_report_only_size {
                "PUFFIN_FIXTURE_N=100000 is report-only per Phase 3 plan"
            } else {
                "random fallback fixture — real-data .bin absent"
            },
        );
    }

    // --- Quantized-blob containment (mean over all 20 held-out queries) ---
    // Loads the encoder from the LOADED-FROM-BLOB bytes; validates that the
    // quantized bytes correspond to the graph's vector set (§3.4). Trait
    // methods used: `EncodedVectors::encode_query` (encoded_vectors.rs:47)
    // and `EncodedVectors::score_point` (encoded_vectors.rs:64).
    //
    // At N=10k the whole containment pass takes < 1s on a modest laptop
    // (20 × 10k score_point calls); at N=100k it's report-only and the
    // full 20-query sweep is 20× slower — measurable but tolerable given
    // 100k is a manual, off-critical-path run.
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
    assert_eq!(quantized_vec_size, fixture.quantized_vec_size);
    let storage = TestEncodedStorage::from_file(&data_path, quantized_vec_size).unwrap();
    let encoded = EncodedVectorsBin::<u128, TestEncodedStorage>::load(storage, &meta_path)
        .expect("EncodedVectorsBin::<u128, _>::load from Puffin bytes");

    let hw = HardwareCounterCell::new();
    let mut per_query_containment: Vec<f64> = Vec::with_capacity(queries.len());
    for (qi, q) in queries.iter().enumerate() {
        let bf_top10 = brute_force_top_k(&scaffold, q, fixture_n, 10);
        let encoded_query = encoded.encode_query(q);
        let mut q_scored: Vec<(PointOffsetType, f32)> = (0..fixture_n as PointOffsetType)
            .map(|i| (i, encoded.score_point(&encoded_query, i, &hw)))
            .collect();
        q_scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("no NaN in seeded quantized scores")
        });
        let q_top: HashSet<PointOffsetType> = q_scored[..QUANT_TOP_K]
            .iter()
            .map(|&(idx, _)| idx)
            .collect();
        let contained = bf_top10.iter().filter(|id| q_top.contains(id)).count();
        let containment = contained as f64 / 10.0;
        println!("  Phase 3 containment query[{qi:02}] = {containment:.3}");
        per_query_containment.push(containment);
    }
    let mean_containment: f64 = per_query_containment.iter().sum::<f64>()
        / per_query_containment.len() as f64;
    println!(
        "Phase 3 MEAN quantized containment (BF top-10 ⊂ Quant top-{QUANT_TOP_K}) over {} queries: {mean_containment:.3}",
        queries.len(),
    );
    if enforce_floors {
        assert!(
            mean_containment >= CONTAINMENT_FLOOR,
            "mean containment {mean_containment:.3} below floor {CONTAINMENT_FLOOR} — \
             indicates the quantized blob's bytes do not derive from the graph's \
             vector set (see §3.4 v3.12 container-consistency invariant)",
        );
    } else {
        println!(
            "  (containment floor NOT enforced on the report-only / random-fallback path)",
        );
    }
}
