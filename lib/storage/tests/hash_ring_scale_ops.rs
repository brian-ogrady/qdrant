//! The hash ring scale is resolved once, on the node receiving a create request, and is immutable
//! afterwards. Both halves live in `Dispatcher::submit_collection_meta_op`, and both are easy to
//! regress into something that still looks like it works:
//!
//! - Resolving the service default anywhere further down (e.g. while *applying* the operation) would
//!   read each peer's own `storage_config`, letting peers derive different point-to-shard mappings
//!   for the same collection.
//! - Dropping the update rejection would restore the previous behaviour, where a request asking to
//!   change the scale returned `200 OK` and silently changed nothing.
//!
//! Everything runs in a single `#[test]` against one `TableOfContent`, because constructing one
//! installs a process-global quota manager that panics if initialised twice.

#![allow(deprecated)]

use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;

use collection::hash_ring::{MAX_HASH_RING_SHARD_SCALE, MAX_HASH_RING_VIRTUAL_NODES};
use collection::operations::config_diff::CollectionParamsDiff;
use collection::operations::vector_params_builder::VectorParamsBuilder;
use collection::optimizers_builder::OptimizersConfig;
use collection::shards::channel_service::ChannelService;
use common::budget::ResourceBudget;
use common::load_concurrency::LoadConcurrencyConfig;
use common::mmap;
use segment::data_types::collection_defaults::CollectionConfigDefaults;
use segment::types::Distance;
use storage::content_manager::collection_meta_ops::{
    CollectionMetaOperations, CreateCollection, CreateCollectionOperation,
    UpdateCollectionOperation,
};
use storage::content_manager::consensus::operation_sender::OperationSender;
use storage::content_manager::errors::StorageError;
use storage::content_manager::toc::TableOfContent;
use storage::dispatcher::Dispatcher;
use storage::rbac::{Access, Auth};
use storage::types::{PerformanceConfig, StorageConfig};
use tempfile::Builder;

const FULL_ACCESS: Auth = Auth::new_internal(Access::full("For test"));

/// Service default used throughout. Deliberately not 100, so "the configured default was applied" is
/// distinguishable from "the hardcoded fallback was used".
const SERVICE_DEFAULT_SCALE: u32 = 7;

fn try_create_op(
    name: &str,
    hash_ring_shard_scale: Option<u32>,
) -> Result<CollectionMetaOperations, StorageError> {
    Ok(CollectionMetaOperations::CreateCollection(
        CreateCollectionOperation::new(
            name.to_string(),
            CreateCollection {
                vectors: VectorParamsBuilder::new(4, Distance::Cosine).build().into(),
                sparse_vectors: None,
                hnsw_config: None,
                wal_config: None,
                optimizers_config: None,
                shard_number: Some(2),
                on_disk_payload: None,
                payload: None,
                replication_factor: None,
                write_consistency_factor: None,
                quantization_config: None,
                sharding_method: None,
                strict_mode_config: None,
                uuid: None,
                metadata: None,
                hash_ring_shard_scale,
                shard_placement: None,
                adopt_shards_from: None,
                adopt_allow_config_rebuild: None,
            },
        )?,
    ))
}

fn create_op(name: &str, hash_ring_shard_scale: Option<u32>) -> CollectionMetaOperations {
    try_create_op(name, hash_ring_shard_scale).expect("create operation should be valid")
}

fn params_diff(hash_ring_shard_scale: Option<u32>) -> CollectionParamsDiff {
    CollectionParamsDiff {
        replication_factor: None,
        write_consistency_factor: None,
        read_fan_out_factor: None,
        read_fan_out_delay_ms: None,
        on_disk_payload: None,
        payload: None,
        hash_ring_shard_scale,
    }
}

fn update_op(name: &str, params: CollectionParamsDiff) -> CollectionMetaOperations {
    let mut op = UpdateCollectionOperation::new_empty(name.to_string());
    op.update_collection.params = Some(params);
    CollectionMetaOperations::UpdateCollection(op)
}

#[test]
fn hash_ring_shard_scale_create_and_immutability() {
    let storage_dir = Builder::new().prefix("storage").tempdir().unwrap();

    let config = StorageConfig {
        storage_path: storage_dir.path().to_path_buf(),
        snapshots_path: storage_dir.path().join("snapshots"),
        snapshots_config: Default::default(),
        temp_path: None,
        shard_adoption_path: None,
        on_disk_payload: false,
        payload: None,
        optimizers: OptimizersConfig {
            deleted_threshold: 0.5,
            vacuum_min_vector_number: 100,
            default_segment_number: 2,
            max_segment_size: None,
            #[expect(deprecated)]
            memmap_threshold: Some(100),
            indexing_threshold: Some(100),
            flush_interval_sec: 2,
            max_optimization_threads: Some(2),
            prevent_unoptimized: None,
        },
        optimizers_overwrite: None,
        // Deliberately tiny. The virtual-node bound below is exercised by asking for more shards than
        // the limit allows, and if that bound is ever removed this test would materialise every one of
        // them — at the default 32 MiB WAL that is gigabytes of preallocation per run.
        wal: collection::config::WalConfig {
            wal_capacity_mb: 1,
            wal_segments_ahead: 0,
            wal_retain_closed: 1,
        },
        performance: PerformanceConfig {
            max_search_threads: 1,
            max_optimization_runtime_threads: 1,
            optimizer_cpu_budget: 0,
            optimizer_io_budget: 0,
            update_rate_limit: None,
            search_timeout_sec: None,
            incoming_shard_transfers_limit: Some(1),
            outgoing_shard_transfers_limit: Some(1),
            async_scorer: None,
            io_uring: None,
            load_concurrency: LoadConcurrencyConfig::default(),
        },
        hnsw_index: Default::default(),
        hnsw_global_config: Default::default(),
        mmap_advice: mmap::Advice::Random,
        low_memory_mode: Default::default(),
        node_type: Default::default(),
        update_queue_size: Default::default(),
        handle_collection_load_errors: false,
        recovery_mode: None,
        update_concurrency: Some(NonZeroUsize::new(2).unwrap()),
        shard_transfer_method: None,
        collection: Some(CollectionConfigDefaults {
            vectors: None,
            quantization: None,
            shard_number: None,
            shard_number_per_node: None,
            replication_factor: None,
            write_consistency_factor: None,
            hash_ring_shard_scale: Some(SERVICE_DEFAULT_SCALE),
            strict_mode: None,
        }),
        max_collections: None,
        quotas: Default::default(),
    };

    let (propose_sender, _propose_receiver) = std::sync::mpsc::channel();
    let toc = Arc::new(
        TableOfContent::new(
            &config,
            ResourceBudget::default(),
            ChannelService::new(6333, false, None, None),
            0,
            Some(OperationSender::new(propose_sender)),
        )
        .unwrap(),
    );
    let handle = toc.general_runtime_handle().clone();
    let dispatcher = Dispatcher::new(toc);

    let submit = |op| handle.block_on(dispatcher.submit_collection_meta_op(op, FULL_ACCESS, None));

    // Read the scale straight off disk rather than through `Collection::info`, which needs an active
    // replica. This also asserts the value was actually *persisted*, which is the property that has
    // to survive a restart.
    let persisted_scale = |name: &str| -> u32 {
        let path = storage_dir
            .path()
            .join("collections")
            .join(name)
            .join("config.json");
        let raw = fs_err::read_to_string(&path).expect("collection config should exist");
        serde_json::from_str::<serde_json::Value>(&raw).unwrap()["params"]["hash_ring_shard_scale"]
            .as_u64()
            .expect("hash_ring_shard_scale should be persisted") as u32
    };

    // The configured service default is applied when the request does not specify one.
    submit(create_op("from_default", None)).unwrap();
    assert_eq!(persisted_scale("from_default"), SERVICE_DEFAULT_SCALE);

    // An explicit request value overrides the service default.
    submit(create_op("explicit", Some(33))).unwrap();
    assert_eq!(persisted_scale("explicit"), 33);

    // Out of range is refused by `validate()` when the operation is built, i.e. before it can be
    // proposed to consensus. That ordering is the point: the value would otherwise be durable in the
    // Raft log, and every peer would re-attempt building a ring for it on every restart.
    let err = try_create_op("too_big", Some(u32::MAX))
        .expect_err("an out-of-range scale must be rejected before the operation is proposed");
    assert!(
        err.to_string().contains("hash_ring_shard_scale"),
        "the rejection should name the offending field, got {err:?}",
    );
    assert!(
        !storage_dir.path().join("collections/too_big").exists(),
        "a rejected create must not leave a collection directory behind",
    );

    // Bounding the scale alone is not enough: a ring holds `scale * shard_number` nodes and
    // `shard_number` has no maximum of its own, so an in-range scale with many shards is still an
    // unbounded ring. `validate()` cannot catch it, because it sees each field separately.
    let mut many_nodes = create_op("too_many_nodes", Some(MAX_HASH_RING_SHARD_SCALE));
    if let CollectionMetaOperations::CreateCollection(op) = &mut many_nodes {
        // Smallest count that exceeds the limit, so a run with the bound removed stays survivable.
        op.create_collection.shard_number =
            Some((MAX_HASH_RING_VIRTUAL_NODES / u64::from(MAX_HASH_RING_SHARD_SCALE) + 1) as u32);
    }
    let err = submit(many_nodes)
        .expect_err("scale * shard_number above the virtual-node limit must be refused");
    let message = err.to_string();
    for expected in ["virtual nodes", "hash_ring_shard_scale"] {
        assert!(
            message.contains(expected),
            "the refusal should name the product and the field; missing {expected:?} in {message:?}",
        );
    }

    // A wide collection at the default scale stays inside the limit, so the bound does not penalise
    // ordinary large collections.
    let mut wide = create_op("wide", None);
    if let CollectionMetaOperations::CreateCollection(op) = &mut wide {
        op.create_collection.shard_number = Some(64);
    }
    submit(wide).expect("a wide collection at the default scale must still be allowed");
    assert_eq!(persisted_scale("wide"), SERVICE_DEFAULT_SCALE);

    // The regression this guards: before the rejection existed, `CollectionParamsDiff` had no field
    // for the scale, so serde dropped the key and the request reported success while doing nothing.
    let err = submit(update_op("explicit", params_diff(Some(500))))
        .expect_err("changing the scale must be rejected, not silently ignored");
    assert!(
        matches!(err, StorageError::BadInput { .. }),
        "expected a bad-input error, got {err:?}",
    );
    let message = err.to_string();
    for expected in [
        "hash_ring_shard_scale",
        "cannot be changed after a collection is created",
        "explicit",
    ] {
        assert!(
            message.contains(expected),
            "the rejection must explain what happened and why; \
             missing {expected:?} in {message:?}",
        );
    }
    assert_eq!(
        persisted_scale("explicit"),
        33,
        "a rejected update must leave the scale untouched",
    );

    // Echoing the *current* value back must succeed. `CollectionParams` serializes this field, so it
    // is present in `GET /collections/{name}`, and a read-modify-write client that PATCHes the whole
    // params object back would be broken by rejecting a no-op.
    submit(update_op("explicit", params_diff(Some(33))))
        .expect("re-stating the current scale is not a change and must be accepted");
    assert_eq!(persisted_scale("explicit"), 33);

    // An update that does not mention the scale is unaffected.
    submit(update_op(
        "explicit",
        CollectionParamsDiff {
            replication_factor: Some(NonZeroU32::new(1).unwrap()),
            ..params_diff(None)
        },
    ))
    .expect("an update that does not touch the scale must succeed");
    assert_eq!(persisted_scale("explicit"), 33);
}
