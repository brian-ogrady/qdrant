//! `CollectionParams::hash_ring_shard_scale` is an input to the point-to-shard routing function, so
//! it has to be read back from the persisted config rather than re-derived. If a reload ever built
//! the rings at a different scale, points would stay on disk but become unreachable by id — silently,
//! with no error anywhere.
//!
//! These tests pin that: routing must survive a reload at a *non-default* scale, and the scale must
//! actually reach the ring (otherwise the first test would pass trivially even if the value were
//! ignored end to end).

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

use collection::collection::Collection;
use collection::config::{CollectionConfigInternal, CollectionParams, ShardingMethod, WalConfig};
use collection::operations::CollectionUpdateOperations;
use collection::operations::point_ops::{
    BatchPersisted, BatchVectorStructPersisted, PointInsertOperationsInternal, PointOperations,
};
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::{CollectionResult, PointRequestInternal};
use collection::operations::vector_params_builder::VectorParamsBuilder;
use collection::shards::channel_service::ChannelService;
use collection::shards::collection_shard_distribution::CollectionShardDistribution;
use collection::shards::replica_set::replica_set_state::ReplicaState;
use collection::shards::shard::PeerId;
use common::budget::ResourceBudget;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use segment::types::{Distance, ShardKey, WithPayloadInterface};
use tempfile::Builder;
use tonic::transport::Uri;

use crate::common::{
    REST_PORT, TEST_OPTIMIZERS_CONFIG, dummy_abort_shard_transfer, dummy_on_replica_failure,
    dummy_request_shard_transfer, load_local_collection, new_local_collection,
};

const COLLECTION_ID: &str = "test_hash_ring_scale";
const PEER_ID: PeerId = 0;
const SHARD_COUNT: u32 = 4;
const POINT_COUNT: u64 = 200;

/// Deliberately not 100: at the default this test would pass even if the persisted value were
/// ignored and the ring always built at the hardcoded default.
const NON_DEFAULT_SCALE: u32 = 7;

fn config(hash_ring_shard_scale: u32) -> CollectionConfigInternal {
    let params = CollectionParams {
        vectors: VectorParamsBuilder::new(4, Distance::Dot).build().into(),
        shard_number: NonZeroU32::new(SHARD_COUNT).unwrap(),
        hash_ring_shard_scale,
        ..CollectionParams::empty()
    };

    CollectionConfigInternal {
        params,
        optimizer_config: TEST_OPTIMIZERS_CONFIG.clone(),
        wal_config: WalConfig {
            wal_capacity_mb: 1,
            wal_segments_ahead: 0,
            wal_retain_closed: 1,
        },
        hnsw_config: Default::default(),
        quantization_config: Default::default(),
        strict_mode_config: Default::default(),
        uuid: None,
        metadata: None,
    }
}

/// `new_local_collection` also transitions every local shard to `Active`, without which updates are
/// rejected.
async fn new_collection(path: &Path, scale: u32) -> Collection {
    new_local_collection(
        COLLECTION_ID.to_string(),
        path,
        &path.join("snapshots"),
        &config(scale),
    )
    .await
    .unwrap()
}

async fn load_collection(path: &Path) -> Collection {
    load_local_collection(COLLECTION_ID.to_string(), path, &path.join("snapshots")).await
}

/// Custom sharding starts out with no shards at all, so the distribution is empty and shards arrive
/// later via `create_shard_key`. That is the whole point of these tests: the ring is built lazily.
async fn new_custom_collection(path: &Path, config: &CollectionConfigInternal) -> Collection {
    Collection::new(
        COLLECTION_ID.to_string(),
        PEER_ID,
        path,
        &path.join("snapshots"),
        config,
        Arc::default(),
        CollectionShardDistribution::all_local(Some(0), PEER_ID),
        None,
        channel_service(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::default(),
        None,
        None,
    )
    .await
    .unwrap()
}

/// `create_shard_key` validates the placement against known peers, so the local peer has to be
/// resolvable through the channel service.
fn channel_service() -> ChannelService {
    let channel_service = ChannelService::new(REST_PORT, false, None, None);
    channel_service
        .id_to_address
        .write()
        .insert(PEER_ID, Uri::from_static("http://127.0.0.1:6333"));
    channel_service
}

async fn insert_points(collection: &Collection) {
    insert_points_with_key(collection, None).await
}

/// As [`insert_points`], but surfaces the error instead of unwrapping it.
async fn try_insert_points(collection: &Collection) -> CollectionResult<()> {
    let ids = (1..=POINT_COUNT).map(Into::into).collect::<Vec<_>>();
    let vectors = (1..=POINT_COUNT)
        .map(|i| {
            let f = i as f32;
            vec![f, f + 1.0, f + 2.0, f + 3.0]
        })
        .collect::<Vec<_>>();

    collection
        .update_from_client(
            CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
                PointInsertOperationsInternal::from(BatchPersisted {
                    ids,
                    vectors: BatchVectorStructPersisted::Single(vectors),
                    payloads: None,
                }),
            )),
            true.into(),
            None,
            Default::default(),
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .map(|_| ())
}

async fn insert_points_with_key(collection: &Collection, shard_key: Option<ShardKey>) {
    let ids = (1..=POINT_COUNT).map(Into::into).collect::<Vec<_>>();
    let vectors = (1..=POINT_COUNT)
        .map(|i| {
            let f = i as f32;
            vec![f, f + 1.0, f + 2.0, f + 3.0]
        })
        .collect::<Vec<_>>();

    let batch = BatchPersisted {
        ids,
        vectors: BatchVectorStructPersisted::Single(vectors),
        payloads: None,
    };

    collection
        .update_from_client(
            CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
                PointInsertOperationsInternal::from(batch),
            )),
            true.into(),
            None,
            Default::default(),
            shard_key,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
}

/// How many of `1..=POINT_COUNT` the collection can find by id.
///
/// Note this is a liveness sanity check, *not* a routing check: `ShardSelectorInternal::All` fans a
/// retrieve out to every shard instead of routing by hash ring, so it finds the points whatever the
/// scale is. Write routing is what depends on the ring — see [`shard_distribution`].
async fn retrievable_count(collection: &Collection) -> usize {
    let request = PointRequestInternal {
        ids: (1..=POINT_COUNT).map(Into::into).collect(),
        with_payload: Some(WithPayloadInterface::Bool(false)),
        with_vector: false.into(),
    };

    collection
        .retrieve(
            request,
            None,
            None,
            &ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap()
        .len()
}

/// Per-shard point counts, in shard-id order — a fingerprint of where the ring placed each point.
///
/// Unsorted on purpose: sorting would hide a permutation, i.e. the same spread landing on different
/// shards, which is exactly the corruption a scale change causes.
async fn shard_distribution(collection: &Collection) -> Vec<usize> {
    let info = collection.cluster_info(PEER_ID).await.unwrap();
    let mut shards = info.local_shards;
    shards.sort_by_key(|shard| shard.shard_id);
    shards.iter().map(|shard| shard.points_count).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn routing_survives_reload_at_non_default_scale() {
    let dir = Builder::new().prefix("storage").tempdir().unwrap();

    let before = {
        let collection = new_collection(dir.path(), NON_DEFAULT_SCALE).await;
        insert_points(&collection).await;

        assert_eq!(
            retrievable_count(&collection).await,
            POINT_COUNT as usize,
            "all points should be retrievable before reload",
        );

        let distribution = shard_distribution(&collection).await;
        collection.stop_gracefully().await;
        distribution
    };

    let collection = load_collection(dir.path()).await;

    let reported_scale = collection
        .info(&ShardSelectorInternal::All)
        .await
        .unwrap()
        .config
        .params
        .hash_ring_shard_scale;
    assert_eq!(
        reported_scale, NON_DEFAULT_SCALE,
        "reload must read the scale back from the persisted config",
    );

    assert_eq!(
        retrievable_count(&collection).await,
        POINT_COUNT as usize,
        "sanity: the reloaded collection still holds every point",
    );

    // The real check. Re-upserting the identical ids exercises *write* routing, which is the thing
    // that consults the hash ring. If the reloaded rings were built at a different scale, a portion
    // of the ids would now route to a different shard, creating a second copy there and leaving the
    // original behind — so per-shard counts would grow and shift. Rebuilt at the persisted scale,
    // every id lands back on the shard that already holds it and the distribution is untouched.
    insert_points(&collection).await;

    assert_eq!(
        shard_distribution(&collection).await,
        before,
        "re-upserting the same ids after reload must not move or duplicate any point; a changed \
         distribution means the rings were rebuilt at a different scale than the data was written \
         with, which silently orphans points",
    );

    assert_eq!(
        retrievable_count(&collection).await,
        POINT_COUNT as usize,
        "the collection must still hold exactly the original points, with no duplicates",
    );

    collection.stop_gracefully().await;
}

/// Guards against the scale being persisted but never actually reaching `HashRing::fair`, which
/// would make the test above vacuous.
#[tokio::test(flavor = "multi_thread")]
async fn scale_changes_the_shard_distribution() {
    let default_dir = Builder::new().prefix("storage").tempdir().unwrap();
    let scaled_dir = Builder::new().prefix("storage").tempdir().unwrap();

    let default_collection = new_collection(
        default_dir.path(),
        collection::config::default_hash_ring_shard_scale(),
    )
    .await;
    insert_points(&default_collection).await;
    let default_distribution = shard_distribution(&default_collection).await;
    default_collection.stop_gracefully().await;

    let scaled_collection = new_collection(scaled_dir.path(), NON_DEFAULT_SCALE).await;
    insert_points(&scaled_collection).await;
    let scaled_distribution = shard_distribution(&scaled_collection).await;
    scaled_collection.stop_gracefully().await;

    assert_eq!(
        default_distribution.iter().sum::<usize>(),
        POINT_COUNT as usize,
        "sanity: every point landed on some shard at the default scale",
    );
    assert_eq!(
        scaled_distribution.iter().sum::<usize>(),
        POINT_COUNT as usize,
        "sanity: every point landed on some shard at the non-default scale",
    );

    // Few virtual nodes per shard means a visibly lumpier spread than many.
    assert_ne!(
        default_distribution, scaled_distribution,
        "scale {NON_DEFAULT_SCALE} must distribute points differently than the default; identical \
         distributions mean the configured value never reached the hash ring",
    );
}

/// Custom sharding builds NO rings at construction time — one is created lazily per shard key, in
/// `ShardHolder::add_shards`. That closure has to use the collection's scale; nothing else in the
/// suite covers it, so hardcoding a default there would otherwise keep every test green.
///
/// A shard key added *after* a reload is the case that matters: the create request is long gone, so
/// the value can only come from the persisted config.
#[tokio::test(flavor = "multi_thread")]
async fn custom_sharding_lazily_created_rings_use_the_collection_scale() {
    let dir = Builder::new().prefix("storage").tempdir().unwrap();
    let key = ShardKey::Keyword("alpha".into());

    let mut config = config(NON_DEFAULT_SCALE);
    config.params.sharding_method = Some(ShardingMethod::Custom);
    config.params.shard_number = NonZeroU32::new(1).unwrap();

    let collection = new_custom_collection(dir.path(), &config).await;

    collection
        .create_shard_key(
            key.clone(),
            vec![vec![PEER_ID]; SHARD_COUNT as usize],
            ReplicaState::Active,
        )
        .await
        .unwrap();

    insert_points_with_key(&collection, Some(key.clone())).await;
    let before = shard_distribution(&collection).await;
    assert_eq!(
        before.iter().sum::<usize>(),
        POINT_COUNT as usize,
        "sanity: every point landed on a shard of the key",
    );
    collection.stop_gracefully().await;

    // The ring for this key is rebuilt from the persisted config on load.
    let collection = load_collection(dir.path()).await;
    assert_eq!(
        collection.hash_ring_shard_scale().await,
        NON_DEFAULT_SCALE,
        "reload must read the scale back from the persisted config",
    );

    insert_points_with_key(&collection, Some(key)).await;
    assert_eq!(
        shard_distribution(&collection).await,
        before,
        "re-upserting the same ids into a custom shard key after reload must not move or duplicate \
         any point; a changed distribution means the lazily rebuilt ring used a different scale",
    );

    collection.stop_gracefully().await;
}

/// Builds a custom-sharded collection at `scale`, creates one shard key with `SHARD_COUNT` shards,
/// inserts the point set, and returns the resulting per-shard distribution.
async fn custom_sharded_distribution(path: &Path, scale: u32) -> Vec<usize> {
    let key = ShardKey::Keyword("alpha".into());

    let mut config = config(scale);
    config.params.sharding_method = Some(ShardingMethod::Custom);
    config.params.shard_number = NonZeroU32::new(1).unwrap();

    let collection = new_custom_collection(path, &config).await;
    collection
        .create_shard_key(
            key.clone(),
            vec![vec![PEER_ID]; SHARD_COUNT as usize],
            ReplicaState::Active,
        )
        .await
        .unwrap();
    insert_points_with_key(&collection, Some(key)).await;

    let distribution = shard_distribution(&collection).await;
    collection.stop_gracefully().await;
    distribution
}

/// The companion to the test above, and the one that actually pins the *value*.
///
/// The reload test only proves the scale is used *consistently* — it passes even if every custom
/// sharding ring is built at some other scale, as long as creation and reload agree. Comparing two
/// collections built at different scales is what detects a hardcoded value in the lazy
/// ring-construction site.
#[tokio::test(flavor = "multi_thread")]
async fn custom_sharding_scale_changes_the_shard_distribution() {
    let default_dir = Builder::new().prefix("storage").tempdir().unwrap();
    let scaled_dir = Builder::new().prefix("storage").tempdir().unwrap();

    let default_distribution = custom_sharded_distribution(
        default_dir.path(),
        collection::config::default_hash_ring_shard_scale(),
    )
    .await;
    let scaled_distribution =
        custom_sharded_distribution(scaled_dir.path(), NON_DEFAULT_SCALE).await;

    assert_eq!(
        default_distribution.iter().sum::<usize>(),
        POINT_COUNT as usize,
        "sanity: every point landed on a shard at the default scale",
    );
    assert_eq!(
        scaled_distribution.iter().sum::<usize>(),
        POINT_COUNT as usize,
        "sanity: every point landed on a shard at the non-default scale",
    );

    assert_ne!(
        default_distribution, scaled_distribution,
        "a custom-sharded collection at scale {NON_DEFAULT_SCALE} must distribute points \
         differently than one at the default; identical distributions mean the configured value \
         never reached the lazily created per-shard-key ring",
    );
}

/// Applying a Raft snapshot whose scale differs from ours must be refused.
///
/// This is the guard that keeps `config.json` from drifting away from the rings already built in
/// memory. Without it, `apply_config` would write the incoming scale to disk while the live rings keep
/// routing at the old one, and the next restart would rebuild at the new value and orphan everything
/// written in between — which the shard-cleanup endpoint then hard-deletes.
///
/// It is deliberately NOT implemented by adding the field to `CollectionParams::check_compatible`,
/// because a failure there makes `apply_collections_snapshot` drop and recreate the collection,
/// discarding all local data. So this test also pins the *choice* of enforcement point.
#[tokio::test(flavor = "multi_thread")]
async fn applying_a_raft_snapshot_with_a_different_scale_is_refused() {
    let dir = Builder::new().prefix("storage").tempdir().unwrap();
    let collection = new_collection(dir.path(), NON_DEFAULT_SCALE).await;
    insert_points(&collection).await;
    let before = shard_distribution(&collection).await;

    // A state that is identical except for the scale — exactly what an incompatible peer would send.
    let mut state = collection.state().await;
    assert_eq!(state.config.params.hash_ring_shard_scale, NON_DEFAULT_SCALE);
    state.config.params.hash_ring_shard_scale = NON_DEFAULT_SCALE + 1;

    let err = collection
        .apply_state(state, PEER_ID, |_transfer| {})
        .await
        .expect_err("applying a state with a different hash ring scale must be refused");

    let message = err.to_string();
    for expected in ["hash ring shard scale", "cannot change in place"] {
        assert!(
            message.contains(expected),
            "the refusal should explain what happened; missing {expected:?} in {message:?}",
        );
    }

    // The refusal must not have half-applied anything.
    assert_eq!(
        collection
            .info(&ShardSelectorInternal::All)
            .await
            .unwrap()
            .config
            .params
            .hash_ring_shard_scale,
        NON_DEFAULT_SCALE,
        "a refused state apply must leave the scale untouched",
    );
    assert_eq!(
        shard_distribution(&collection).await,
        before,
        "a refused state apply must not move any point",
    );

    // An otherwise-identical state that agrees on the scale must still apply, or the assertion above
    // would pass simply because `apply_state` rejects everything.
    let agreeing = collection.state().await;
    collection
        .apply_state(agreeing, PEER_ID, |_transfer| {})
        .await
        .expect("a state that agrees on the scale must apply");

    collection.stop_gracefully().await;
}

/// A shard whose recorded scale disagrees with the collection's must refuse to serve.
///
/// This is the self-describing-data half: each shard records the scale its points were placed under, so
/// a `config.json` that later says something else is detectable rather than silently misrouting. Left
/// undetected, every id in the shard hashes elsewhere, and the shard-cleanup endpoint hard-deletes what
/// no longer maps locally — so refusing to serve is what makes the divergence recoverable.
#[tokio::test(flavor = "multi_thread")]
async fn a_shard_placed_under_a_different_scale_refuses_to_serve() {
    let dir = Builder::new().prefix("storage").tempdir().unwrap();

    {
        let collection = new_collection(dir.path(), NON_DEFAULT_SCALE).await;
        insert_points(&collection).await;
        collection.stop_gracefully().await;
    }

    // Rewrite only the collection config, leaving the data and its recorded scale alone — what a
    // hand-edit or a partial restore produces.
    let config_path = dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&fs_err::read_to_string(&config_path).unwrap()).unwrap();
    config["params"]["hash_ring_shard_scale"] = serde_json::json!(NON_DEFAULT_SCALE + 1);
    fs_err::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

    let diverged = load_collection(dir.path()).await;
    let err = try_insert_points(&diverged)
        .await
        .expect_err("a shard placed under a different scale must not accept writes");
    // Assert on something discriminating, not merely that *something* failed: an "is not empty" check
    // on the message would be satisfied by any unrelated write rejection and would stop testing this
    // mechanism entirely.
    //
    // What the client sees is a replica-set-level failure, because the diverged replica is disabled at
    // load and the set is left with no active replica — the dummy is never even asked. The actionable
    // detail (which scale the data was placed under, and what to restore) goes to the server log only.
    let reason = format!("{err}");
    assert!(
        reason.contains("active replica"),
        "the write must be refused by the replica set, got: {reason}",
    );

    // ...so pin the *cause* directly, which is what makes the refusal above attributable to the scale
    // rather than to anything else that can leave a replica set without an active replica.
    {
        let shards = diverged.shards_holder();
        let shards = shards.read().await;
        assert_eq!(
            shards
                .get_shard(0)
                .expect("shard 0 must exist")
                .diverged_hash_ring_shard_scale()
                .await
                .unwrap(),
            Some(NON_DEFAULT_SCALE),
            "the shard must report the scale its points were actually placed under",
        );
    }
    diverged.stop_gracefully().await;

    // Restoring the recorded scale must bring the shard back with its points intact — refusing to serve
    // is a recoverable stop, not a deletion.
    config["params"]["hash_ring_shard_scale"] = serde_json::json!(NON_DEFAULT_SCALE);
    fs_err::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

    let recovered = load_collection(dir.path()).await;
    assert_eq!(
        retrievable_count(&recovered).await,
        POINT_COUNT as usize,
        "every point must still be there once the scale agrees again",
    );
    try_insert_points(&recovered)
        .await
        .expect("writes must work again once the scale agrees");
    recovered.stop_gracefully().await;
}

/// Refusing to serve a diverged shard must not turn into deleting it.
///
/// This is the trap the mechanism sets for itself. A diverged shard loads as a `DummyShard`, and every
/// *other* cause of a dummy shard in this codebase means "safe to overwrite": the replica gets marked
/// dead, an automatic transfer is proposed, and the transfer clears the directory before refilling it
/// from another replica. Verified against a real two-peer cluster before this guard existed, the
/// diverged segments were deleted within seconds of the first write — and then the refill was rejected
/// by the snapshot-scale check, leaving the peer empty and looping. Both clear paths must refuse.
#[tokio::test(flavor = "multi_thread")]
async fn a_diverged_shard_refuses_to_be_cleared() {
    let dir = Builder::new().prefix("storage").tempdir().unwrap();

    {
        let collection = new_collection(dir.path(), NON_DEFAULT_SCALE).await;
        insert_points(&collection).await;
        collection.stop_gracefully().await;
    }

    let config_path = dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&fs_err::read_to_string(&config_path).unwrap()).unwrap();
    config["params"]["hash_ring_shard_scale"] = serde_json::json!(NON_DEFAULT_SCALE + 1);
    fs_err::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

    let diverged = load_collection(dir.path()).await;

    // What is on disk before anything tries to recover the shard.
    let segments_path = dir.path().join("0").join("segments");
    let segments_before = segment_dir_names(&segments_path);
    assert!(
        !segments_before.is_empty(),
        "the shard must have segments on disk for this test to mean anything",
    );

    {
        let shards = diverged.shards_holder();
        let shards = shards.read().await;
        let replica_set = shards.get_shard(0).expect("shard 0 must exist");

        // The path automatic recovery takes: clear the directory, then refill from a healthy replica.
        let err = replica_set
            .init_empty_local_shard()
            .await
            .expect_err("clearing a diverged shard must be refused");
        let reason = format!("{err}");
        assert!(
            reason.contains("hash ring shard scale"),
            "the refusal must name the scale mismatch, got: {reason}",
        );

        // And the same for the snapshot-transfer path, which clears *before* downloading the
        // replacement, so a rejection afterwards would arrive too late to matter. Marked dead first
        // because clearing a source-of-truth replica is refused for an unrelated reason, which would
        // let this pass without exercising the scale guard at all.
        replica_set
            .set_replica_state(PEER_ID, ReplicaState::Dead)
            .await
            .unwrap();
        let err = replica_set
            .clear_local_for_snapshot_recovery(dir.path())
            .await
            .expect_err("clearing a diverged shard for snapshot recovery must be refused");
        let reason = format!("{err}");
        assert!(
            reason.contains("hash ring shard scale"),
            "the refusal must name the scale mismatch, got: {reason}",
        );

        // The in-memory answer the non-destructive callers use — the transfer path's refusal and the
        // consensus tick's skip both take this, and neither is otherwise exercised by any test, so
        // without this they are unverified redundancy that could be deleted with everything still green.
        let reason = replica_set
            .local_dummy_reason_forbidding_discard()
            .await
            .expect("a diverged shard's data must report as not discardable");
        assert!(
            !reason.may_be_discarded(),
            "the returned reason must be the protected kind, got {reason:?}",
        );
        assert!(
            format!("{reason}").contains("hash ring shard scale"),
            "the reason must explain itself for the callers that log it, got {reason}",
        );

        // Undo this test's own perturbation. `Dead` is persisted to `replica_state.json`, so leaving it
        // set would make the recovery assertion below fail for a reason that has nothing to do with the
        // scale — the replica set would simply have no active replica any more.
        replica_set
            .set_replica_state(PEER_ID, ReplicaState::Active)
            .await
            .unwrap();
    }

    assert_eq!(
        segment_dir_names(&segments_path),
        segments_before,
        "the diverged data must still be on disk, byte-for-byte the same segments",
    );
    diverged.stop_gracefully().await;

    // The whole point of keeping it: restoring the scale must bring every point back.
    config["params"]["hash_ring_shard_scale"] = serde_json::json!(NON_DEFAULT_SCALE);
    fs_err::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

    let recovered = load_collection(dir.path()).await;
    assert_eq!(
        retrievable_count(&recovered).await,
        POINT_COUNT as usize,
        "refusing to clear is what makes the divergence recoverable",
    );
    recovered.stop_gracefully().await;
}

/// A shard that agrees with the collection must still be clearable, or the guard above would be
/// indistinguishable from breaking recovery outright.
#[tokio::test(flavor = "multi_thread")]
async fn an_agreeing_shard_can_still_be_cleared() {
    let dir = Builder::new().prefix("storage").tempdir().unwrap();

    let collection = new_collection(dir.path(), NON_DEFAULT_SCALE).await;
    insert_points(&collection).await;

    {
        let shards = collection.shards_holder();
        let shards = shards.read().await;
        let replica_set = shards.get_shard(0).expect("shard 0 must exist");

        // The other direction of the same accessor: a healthy shard must not report as protected, or the
        // transfer path would refuse every legitimate recovery.
        assert!(
            replica_set
                .local_dummy_reason_forbidding_discard()
                .await
                .is_none(),
            "a healthy shard's data must not report as protected from discarding",
        );

        replica_set
            .init_empty_local_shard()
            .await
            .expect("a shard whose recorded scale agrees must still be clearable");

        // And it must re-describe itself afterwards: `LocalShard::clear` keeps `shard_config.json`, so
        // a stamp left behind from before would either read as divergence later or leave the rebuilt
        // data undescribed.
        assert_eq!(
            replica_set.diverged_hash_ring_shard_scale().await.unwrap(),
            None,
            "a shard rebuilt under the collection's scale must not read as diverged",
        );
    }

    collection.stop_gracefully().await;
}

fn segment_dir_names(segments_path: &Path) -> Vec<String> {
    let Ok(entries) = fs_err::read_dir(segments_path) else {
        return Vec::new();
    };
    let mut names: Vec<_> = entries
        .filter_map(|entry| Some(entry.ok()?.file_name().to_string_lossy().into_owned()))
        .collect();
    names.sort();
    names
}

/// A `config.json` written before this field existed must load as the historical default, since
/// those collections were routed at that scale.
#[test]
fn missing_scale_in_persisted_config_defaults_to_historical_value() {
    let params: CollectionParams = serde_json::from_str(
        r#"{
            "vectors": {"size": 4, "distance": "Dot"},
            "shard_number": 1,
            "replication_factor": 1,
            "write_consistency_factor": 1
        }"#,
    )
    .unwrap();

    assert_eq!(
        params.hash_ring_shard_scale,
        collection::config::default_hash_ring_shard_scale(),
    );
    assert_eq!(params.hash_ring_shard_scale, 100);
}
