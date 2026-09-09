// Deprecated storage placement params (`on_disk`, `always_ram`, `on_disk_payload`) are still
// handled here for backward compatibility with the new `memory` parameter
#![allow(deprecated)]

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

use collection::collection::Collection;
use collection::config::{
    self, CollectionConfigInternal, CollectionParams, PayloadStorageParams, ShardingMethod,
};
use collection::hash_ring::{MAX_HASH_RING_SHARD_SCALE, MAX_HASH_RING_VIRTUAL_NODES};
use collection::operations::config_diff::DiffConfig as _;
use collection::operations::types::{CollectionResult, VectorParams, VectorsConfig};
use collection::shards::collection_shard_distribution::CollectionShardDistribution;
use collection::shards::replica_set::replica_set_state::ReplicaState;
use collection::shards::shard::{PeerId, ShardId};
use segment::types::VectorsConfigDefaults;

use super::{COLLECTION_DELETE_SPIN_INTERVAL, COLLECTION_DELETE_WAIT_TIMEOUT, TableOfContent};
use crate::common::utils::try_unwrap_with_timeout_async;
use crate::content_manager::collection_meta_ops::*;
use crate::content_manager::collections_ops::{Checker as _, Collections};
use crate::content_manager::consensus_ops::ConsensusOperations;
use crate::content_manager::errors::StorageError;

/// Fill exactly one placement level, by precedence: request `memory`, request legacy `on_disk`,
/// default `memory`, default `on_disk`. Filling a lower level alongside a higher one would cause
/// spurious `memory`-vs-legacy mismatch warnings at resolution time.
fn apply_vector_placement_defaults(params: &mut VectorParams, defaults: &VectorsConfigDefaults) {
    let VectorsConfigDefaults { on_disk, memory } = defaults;
    if params.memory.is_some() || params.on_disk.is_some() {
        return;
    }
    if memory.is_some() {
        params.memory = *memory;
    } else {
        params.on_disk = *on_disk;
    }
}

/// Service-level `payload.memory` default applies unless the request specifies `payload.memory`
/// or the legacy `on_disk_payload` flag.
fn apply_payload_placement_defaults(
    payload: Option<PayloadStorageParams>,
    on_disk_payload: Option<bool>,
    defaults: Option<PayloadStorageParams>,
) -> Option<PayloadStorageParams> {
    if on_disk_payload.is_some() {
        return payload;
    }
    match (defaults, payload) {
        (Some(defaults), Some(payload)) => Some(defaults.update(&payload)),
        (defaults, payload) => payload.or(defaults),
    }
}

/// Whether `collection` is STILL the registered instance for `collection_name`, compared by `Arc`
/// identity. A straggler Phase B task from a deleted (and possibly recreated) collection must NOT
/// touch adoption markers: they are keyed by the collection/shard path, which a same-name recreate
/// reuses, so writing/clearing them would corrupt the new collection. Returns false if the name is
/// absent or now maps to a different `Collection`.
async fn is_current_adopted_collection(
    collections: &tokio::sync::RwLock<Collections>,
    collection_name: &str,
    collection: &Arc<Collection>,
) -> bool {
    collections
        .read()
        .await
        .get(collection_name)
        .is_some_and(|current| Arc::ptr_eq(current, collection))
}

/// Record that an adopted shard's Phase B load failed. Writes the durable `.adopt_failed` marker
/// (with `reason`); in a cluster `sync_local_state` then drives the shard to `Dead` (leader-guarded,
/// retried, crash-surviving) so the failure is queryable rather than sitting indistinguishable from
/// a healthy in-progress load. Single-node has no reconciler, so set `Dead` directly here — allowed
/// even though it is the sole replica, because a `ManualRecovery` replica is not a "source of truth"
/// so the last-active-replica guard does not fire.
///
/// No-op if `collection` is no longer the registered instance for `collection_name` (deleted or
/// recreated): the markers would land in a path a same-name recreate now owns.
async fn mark_adopted_shard_failed(
    collections: &tokio::sync::RwLock<Collections>,
    collection: &Arc<Collection>,
    collection_name: &str,
    shard_id: ShardId,
    reason: &str,
    is_distributed: bool,
    this_peer_id: PeerId,
) {
    if !is_current_adopted_collection(collections, collection_name, collection).await {
        return;
    }
    let marked = collection
        .mark_shard_adopt_activation_failed(shard_id, reason)
        .await;
    // Single-node has no reconciler, so set `Dead` directly — but ONLY if the failure marker was
    // durably written. That marker is also the reload-panic suppressor (`is_adopted`), so a `Dead`,
    // marker-less shard could crash-loop the node on a later failed load. If the marker write
    // failed, leave the shard `ManualRecovery`: the in-progress marker still suppresses the panic
    // and drives a restart-time retry, mirroring the load-time resolution's `failed_written &&`
    // gate. (In a cluster the reconciler drives `Dead` from the marker, so this branch is skipped.)
    if marked
        && !is_distributed
        && let Err(err) = collection
            .set_shard_replica_state(
                shard_id,
                this_peer_id,
                ReplicaState::Dead,
                Some(ReplicaState::ManualRecovery),
            )
            .await
    {
        log::warn!("adopted shard {shard_id}: could not mark failed replica Dead: {err}");
    }
}

impl TableOfContent {
    pub(super) async fn create_collection(
        &self,
        collection_name: &str,
        operation: CreateCollection,
        collection_shard_distribution: CollectionShardDistribution,
    ) -> Result<bool, StorageError> {
        // Collection operations require multiple file operations,
        // before collection can actually be registered in the service.
        // To prevent parallel writing of the files, we use this lock.
        let collection_create_guard = self.collection_create_lock.lock().await;

        let CreateCollection {
            mut vectors,
            shard_number,
            sharding_method,
            on_disk_payload,
            payload,
            hnsw_config: hnsw_config_diff,
            wal_config: wal_config_diff,
            optimizers_config: optimizers_config_diff,
            replication_factor,
            write_consistency_factor,
            quantization_config,
            sparse_vectors,
            strict_mode_config,
            uuid,
            metadata,
            hash_ring_shard_scale,
            shard_placement: _,
            adopt_shards_from,
            adopt_allow_config_rebuild,
        } = operation;

        // Normally populated by `submit_collection_meta_op` before this operation was proposed or
        // applied. `None` means the request did not come through that path, so fall back to the
        // historical scale rather than this peer's environment.
        let hash_ring_shard_scale =
            hash_ring_shard_scale.unwrap_or_else(config::default_hash_ring_shard_scale);

        // Checked before any directory is created, so a rejected value cannot leave a half-built
        // collection behind.
        if !(1..=MAX_HASH_RING_SHARD_SCALE).contains(&hash_ring_shard_scale) {
            return Err(StorageError::bad_input(format!(
                "`hash_ring_shard_scale` must be in 1..={MAX_HASH_RING_SHARD_SCALE}, \
                 got {hash_ring_shard_scale}",
            )));
        }

        {
            let collections = self.collections.read().await;
            collections.validate_collection_not_exists(collection_name)?;

            if let Some(max_collections) = self.storage_config.max_collections
                && collections.len() >= max_collections
            {
                return Err(StorageError::bad_request(format!(
                    "Can't create collection with name {collection_name}. Max collections limit reached: {max_collections}",
                )));
            }
        }

        if self
            .alias_persistence
            .read()
            .await
            .check_alias_exists(collection_name)
        {
            return Err(StorageError::bad_input(format!(
                "Can't create collection with name {collection_name}. Alias with the same name already exists",
            )));
        }

        let collection_path = self.create_collection_path(collection_name).await?;
        // derive the snapshots path for the collection to be used across collection operation, the directories for the snapshot
        // is created only when a create snapshot api is invoked.
        let snapshots_path = self.snapshots_path_for_collection(collection_name);

        let collection_defaults_config = self.storage_config.collection.as_ref();

        let default_shard_number = collection_defaults_config
            .and_then(|x| x.shard_number)
            .unwrap_or_else(|| config::default_shard_number().get());

        let shard_number = match sharding_method.unwrap_or_default() {
            ShardingMethod::Auto => {
                if let Some(shard_number) = shard_number {
                    debug_assert_eq!(
                        shard_number as usize,
                        collection_shard_distribution.shard_count(),
                        "If shard number was supplied then this exact number should be used in a distribution",
                    );
                    shard_number
                } else {
                    collection_shard_distribution.shard_count() as u32
                }
            }
            ShardingMethod::Custom => {
                if let Some(shard_number) = shard_number {
                    shard_number
                } else {
                    default_shard_number
                }
            }
        };

        let virtual_nodes = u64::from(hash_ring_shard_scale) * u64::from(shard_number);
        if virtual_nodes > MAX_HASH_RING_VIRTUAL_NODES {
            return Err(StorageError::bad_input(format!(
                "`hash_ring_shard_scale` of {hash_ring_shard_scale} with {shard_number} shards needs \
                 {virtual_nodes} virtual nodes on the hash ring, above the limit of \
                 {MAX_HASH_RING_VIRTUAL_NODES}. Virtual nodes exist to even out distribution when \
                 there are few shards, so a collection with this many shards does not need a high \
                 scale — lower `hash_ring_shard_scale` or reduce `shard_number`",
            )));
        }

        let replication_factor = replication_factor
            .or_else(|| collection_defaults_config.and_then(|i| i.replication_factor))
            .unwrap_or_else(|| config::default_replication_factor().get());

        let write_consistency_factor = write_consistency_factor
            .or_else(|| collection_defaults_config.and_then(|i| i.write_consistency_factor))
            .unwrap_or_else(|| config::default_write_consistency_factor().get());

        // Apply default vector config values if not set.
        let vectors_defaults = collection_defaults_config.and_then(|i| i.vectors.as_ref());
        if let Some(vectors_defaults) = vectors_defaults {
            match &mut vectors {
                VectorsConfig::Single(s) => {
                    apply_vector_placement_defaults(s, vectors_defaults);
                }
                VectorsConfig::Multi(m) => {
                    for vec_params in m.values_mut() {
                        apply_vector_placement_defaults(vec_params, vectors_defaults);
                    }
                }
            };
        }

        let payload =
            apply_payload_placement_defaults(payload, on_disk_payload, self.storage_config.payload);

        let collection_params = CollectionParams {
            vectors,
            sparse_vectors,
            shard_number: NonZeroU32::new(shard_number)
                .ok_or_else(|| StorageError::bad_input("`shard_number` cannot be 0"))?,
            sharding_method,
            on_disk_payload: Some(on_disk_payload.unwrap_or(self.storage_config.on_disk_payload)),
            payload,
            replication_factor: NonZeroU32::new(replication_factor).ok_or_else(|| {
                StorageError::bad_input("`replication_factor` cannot be 0".to_string())
            })?,
            write_consistency_factor: NonZeroU32::new(write_consistency_factor).ok_or_else(
                || StorageError::bad_input("`write_consistency_factor` cannot be 0".to_string()),
            )?,
            read_fan_out_factor: None,
            read_fan_out_delay_ms: None,
            hash_ring_shard_scale,
        };
        let wal_config = self.storage_config.wal.update_opt(wal_config_diff.as_ref());

        let optimizer_config = self
            .storage_config
            .optimizers
            .update_opt(optimizers_config_diff.as_ref());

        let hnsw_config = self
            .storage_config
            .hnsw_index
            .update_opt(hnsw_config_diff.as_ref());

        let quantization_config = match quantization_config {
            None => self
                .storage_config
                .collection
                .as_ref()
                .and_then(|i| i.quantization.clone()),
            Some(diff) => Some(diff),
        };

        let strict_mode_config = match strict_mode_config {
            Some(diff) => {
                let default_config = self
                    .storage_config
                    .collection
                    .as_ref()
                    .and_then(|i| i.strict_mode.clone())
                    .unwrap_or_default();
                Some(default_config.update(&diff))
            }
            None => self
                .storage_config
                .collection
                .as_ref()
                .and_then(|i| i.strict_mode.as_ref())
                .cloned(),
        };

        let storage_config = self
            .storage_config
            .to_shared_storage_config(self.is_distributed())
            .into();

        let collection_config = CollectionConfigInternal {
            wal_config,
            params: collection_params,
            optimizer_config,
            hnsw_config,
            quantization_config,
            strict_mode_config,
            uuid,
            metadata,
        };

        // No shard key mapping on creation, shard keys are set up after creating the collection
        let shard_key_mapping = None;

        let collection = Collection::new(
            collection_name.to_string(),
            self.this_peer_id,
            &collection_path,
            &snapshots_path,
            &collection_config,
            storage_config,
            collection_shard_distribution,
            shard_key_mapping,
            self.channel_service.clone(),
            Self::change_peer_from_state_callback(
                self.consensus_proposal_sender.clone(),
                collection_name.to_string(),
                ReplicaState::Dead,
            ),
            Self::request_shard_transfer_callback(
                self.consensus_proposal_sender.clone(),
                collection_name.to_string(),
            ),
            Self::abort_shard_transfer_callback(
                self.consensus_proposal_sender.clone(),
                collection_name.to_string(),
            ),
            Some(self.adaptive_search_handle.clone()),
            Some(self.update_runtime.handle().clone()),
            self.optimizer_resource_budget.clone(),
            self.storage_config.optimizers_overwrite.clone(),
            // When shards will be populated by adoption, build them parked in
            // `ManualRecovery`
            adopt_shards_from
                .as_ref()
                .map(|_| ReplicaState::ManualRecovery),
        )
        .await?;

        collection.print_warnings().await;

        let local_shards = collection.get_local_shards().await;

        // Arm the crash-recovery markers for every local adopted shard BEFORE the collection is
        // inserted into the live map (and thus before any subsequent apply-thread work), so a crash
        // — or an early error return inside `adopt_local_shards_on_create` below, before its own
        // arming — can never leave a persisted `ManualRecovery` shard with NO marker. A marker-less
        // adopted shard is the one state both the load-time resolver (`ShardReplicaSet::load`, gated
        // on the in-progress marker) and the reconciler (`sync_local_state`, needs a marker or a
        // `Dead` state) silently skip, stranding it invisibly. `Collection::new` has already
        // persisted the shards, so a tiny residual window remains inside the constructor itself;
        // fully closing that would require arming within `Collection::new`.
        //
        // Ordering per shard: the `.initializing` dirty flag FIRST, the in-progress marker ONLY if
        // the flag was durably written — an in-progress marker without the dirty flag is exactly the
        // state the salvage check would mistake for a completed load and activate empty (silent data
        // loss). A flag write failure falls back to the safe pre-H1 behavior (a crash may leave the
        // shard parked, surfaced by `report_parked_adopted_shards`).
        if adopt_shards_from.is_some() {
            for &shard_id in &local_shards {
                if collection.mark_shard_initializing(shard_id) {
                    collection.mark_shard_adopt_in_progress(shard_id);
                }
            }
        }

        {
            let mut write_collections = self.collections.write().await;
            write_collections.validate_collection_not_exists(collection_name)?;
            let existing_collection =
                write_collections.insert(collection_name.to_string(), Arc::new(collection));
            if let Some(existing_collection) = existing_collection {
                debug_assert!(
                    false,
                    "Collection `{collection_name}` was not expected to exist"
                );

                existing_collection.stop_gracefully().await;

                let removed_collection_res = try_unwrap_with_timeout_async(
                    existing_collection,
                    COLLECTION_DELETE_SPIN_INTERVAL,
                    COLLECTION_DELETE_WAIT_TIMEOUT,
                )
                .await;

                match removed_collection_res {
                    Ok(collection) => drop(collection),
                    Err(busy_collection) => {
                        debug_assert!(false, "Collection `{collection_name}` is busy");
                        log::error!(
                            "Collection `{collection_name}` is busy and cannot be removed in time."
                        );
                        drop(busy_collection);
                    }
                };
            }

            self.telemetry.init_snapshot_telemetry(collection_name);
        }

        drop(collection_create_guard);

        // Adopt staged artifacts into the just-created local shards.
        match &adopt_shards_from {
            // Only peers actually assigned a shard adopt; a peer with no local shards must not
            // even require `shard_adoption_path` to be set (it has nothing to adopt).
            Some(staging_subdir) if !local_shards.is_empty() => {
                // Adopt in two phases (see `adopt_local_shards_on_create`): Phase A resolves and
                // topology-checks each artifact inline on this apply thread; Phase B runs the
                // heavy `LocalShard::load` OFF the apply thread.
                self.adopt_local_shards_on_create(
                    collection_name,
                    staging_subdir,
                    &local_shards,
                    adopt_allow_config_rebuild.unwrap_or(false),
                )
                .await?;
            }
            _ => {
                // No adoption: shards were built `Initializing`; activate them the usual way.
                for shard_id in local_shards {
                    self.on_peer_created(collection_name.to_string(), self.this_peer_id, shard_id)
                        .await?;
                }
            }
        }

        Ok(true)
    }

    /// Adopt one staged artifact per local shard, by rename, at creation time.
    /// Split into two phases so the heavy load never blocks the consensus apply thread:
    ///
    /// * **Phase A — inline, on the apply thread.** Resolve and topology-check each staged
    ///   artifact. A missing artifact or a manifest mismatch fails the create *here*,
    ///   synchronously, with a precise error, before any load is scheduled.
    /// * **Phase B — off the apply thread, bounded by `adopt_load_semaphore`.** Run the
    ///   restore (`rename` + `LocalShard::load`, itself spawned on the update runtime) and
    ///   then flip the parked shard to `Active`. Detached: the apply returns as soon as
    ///   Phase A is done.
    async fn adopt_local_shards_on_create(
        &self,
        collection_name: &str,
        staging_subdir: &str,
        local_shards: &[ShardId],
        allow_config_rebuild: bool,
    ) -> Result<(), StorageError> {
        use shard::snapshots::snapshot_data::SnapshotData;
        use shard::snapshots::snapshot_manifest::RecoveryType;

        use crate::content_manager::snapshots::adopt;

        // Deterministic across peers and already checked at op construction; a defensive
        // re-check here, and the only path that fails the whole apply (all peers alike).
        adopt::validate_staging_subdir(staging_subdir)?;

        // These run inline on the consensus apply thread; a `service_error` here would HALT
        // consensus (and, if IO-dependent, potentially on one peer only → divergence). Adoption's
        // expected failures are all `bad_request` (non-halting, logged; the shard stays parked and
        // is surfaced by `report_parked_adopted_shards`), so keep these that way too.
        let collection = self
            .collections
            .read()
            .await
            .get(collection_name)
            .cloned()
            .ok_or_else(|| {
                StorageError::bad_request(format!(
                    "collection `{collection_name}` disappeared during adoption",
                ))
            })?;

        let temp_dir = self.optional_temp_or_storage_temp_path().map_err(|err| {
            StorageError::bad_request(format!(
                "cannot determine a temp directory for adopting into `{collection_name}`: {err}",
            ))
        })?;
        let collections_dir = self.collections_dir_path();
        let root = self.shard_adoption_path().map(Path::to_path_buf);

        // The crash-recovery markers (`.initializing` dirty flag + `.adopt_in_progress`) were
        // already armed for every local adopted shard by the caller, BEFORE the collection was
        // inserted into the live map — so they are guaranteed present here (an early error return
        // below cannot leave a marker-less strand). See the arming loop in `create_collection`.

        // ---- Phase A (inline, serialized with delete on the apply thread): resolve + verify.
        // Rejects a missing artifact or a topology/schema mismatch synchronously, with a
        // precise error, before any load is scheduled.
        let mut staged: Vec<(ShardId, std::path::PathBuf)> = Vec::new();
        let mut failed = Vec::new();
        for &shard_id in local_shards {
            match self
                .stage_one_adopted_shard(
                    &collection,
                    root.as_deref(),
                    staging_subdir,
                    shard_id,
                    &temp_dir,
                    &collections_dir,
                    allow_config_rebuild,
                )
                .await
            {
                Ok(source) => {
                    // Crash-recovery markers were already armed for every shard up front, before
                    // this staging loop (see above), so nothing to arm here.
                    staged.push((shard_id, source));
                }
                Err(err) => {
                    log::error!(
                        "shard {shard_id} of `{collection_name}` could not be staged for \
                         adoption (left parked): {err}",
                    );
                    failed.push(format!("shard {shard_id}: {err}"));
                }
            }
        }

        // ---- Phase B (off the apply thread): restore + activate, one heavy load at a time
        // per the load-concurrency governor.
        let mut load_tasks: Vec<(ShardId, tokio::task::JoinHandle<Result<(), String>>)> =
            Vec::with_capacity(staged.len());
        for (shard_id, source) in staged {
            let collection = Arc::clone(&collection);
            let collections = Arc::clone(&self.collections);
            let this_peer_id = self.this_peer_id;
            let is_distributed = self.is_distributed();
            let semaphore = Arc::clone(&self.adopt_load_semaphore);
            let temp_dir = temp_dir.clone();
            let collection_name = collection_name.to_string();

            // The crash-recovery markers were armed in Phase A, as soon as this shard was staged.
            let collection_super = Arc::clone(&collection);
            let collection_name_super = collection_name.clone();
            let collections_super = Arc::clone(&self.collections);

            // A child of the collection's adoption-cancellation token. `stop_gracefully` (collection
            // delete / drop) fires the parent, so a delete does not stall behind this load: a not-
            // yet-started load bails here, and the restore honors it at its cancel checkpoints.
            let adopt_cancel = collection.adopt_cancel_token();

            // Returns `Ok(())` on success (or a deliberate skip), `Err(reason)` when the shard is
            // left parked. The result is surfaced by `report_parked_adopted_shards` (cluster,
            // via the supervisor below) or collected by the single-node loop.
            let handle = self.general_runtime.spawn(async move {
                // Bound concurrent heavy loads exactly like startup shard loading. The permit is
                // held for the whole restore (released on drop, including on panic). The acquire is
                // itself cancellable: the load semaphore is TOC-global, so while THIS shard waits
                // for a permit (behind another collection's loads) it must still abort promptly on
                // delete — otherwise it keeps the collection `Arc` alive and stalls the delete's
                // `try_unwrap`/write-lock on the apply thread for the full timeout. `biased` prefers
                // the cancel branch. Aborting while queued holds no shard-holder guard and drops the
                // `Arc`; the in-progress marker is left for restart-time resolution (or the delete
                // removes the whole collection dir).
                let _permit = tokio::select! {
                    biased;
                    () = adopt_cancel.cancelled() => return Ok(()),
                    permit = semaphore.acquire_owned() => match permit {
                        Ok(permit) => permit,
                        // The semaphore lives for the whole process, so a closed semaphore is
                        // effectively unreachable; if it ever fired, report the shard parked (which
                        // leaves the in-progress marker) rather than a silent success.
                        Err(_) => {
                            return Err(format!(
                                "shard {shard_id}: adoption load governor unavailable"
                            ));
                        }
                    },
                };

                // Also re-check after acquiring: the collection may have been deleted in the window
                // between winning the permit and here. Bail before touching disk (the restore would
                // only take the shard-holder read guard the delete is trying to acquire, and could
                // rename the artifact into a now-unregistered collection's dir). Gate on IDENTITY,
                // not just the token: `delete_collection` removes the collection from the map as its
                // very first step, but only fires `adopt_cancel` later (after its resharding-abort
                // awaits) — so the map removal is the earlier, tighter signal that this collection
                // is going away.
                if adopt_cancel.is_cancelled()
                    || !is_current_adopted_collection(&collections, &collection_name, &collection)
                        .await
                {
                    return Ok(());
                }

                // Symlink walk deferred from Phase A (kept off the apply thread): reject any
                // symlink inside the staged tree BEFORE the install rename, so a malicious/broken
                // artifact cannot be moved in and followed out of the staging root on load.
                if let Err(err) =
                    crate::content_manager::snapshots::adopt::reject_symlinks_within(&source)
                {
                    let reason = format!("staged artifact rejected: {err}");
                    log::error!(
                        "adopted shard {shard_id} of `{collection_name}` {reason} (marking failed)",
                    );
                    mark_adopted_shard_failed(&collections, &collection, &collection_name, shard_id, &reason, is_distributed, this_peer_id)
                        .await;
                    return Err(format!("shard {shard_id} ({reason})"));
                }

                let restore = match collection
                    .restore_shard_snapshot(
                        shard_id,
                        SnapshotData::Adopted(source),
                        RecoveryType::Full,
                        this_peer_id,
                        is_distributed,
                        &temp_dir,
                        None,
                        adopt_cancel,
                    )
                    .await
                {
                    Ok(restore) => restore,
                    Err(err) => {
                        let reason = format!("could not start restore: {err}");
                        log::error!(
                            "adopted shard {shard_id} of `{collection_name}` {reason} (marking failed)",
                        );
                        mark_adopted_shard_failed(
                            &collections,
                            &collection,
                            &collection_name,
                            shard_id,
                            &reason,
                            is_distributed,
                            this_peer_id,
                        )
                        .await;
                        return Err(format!("shard {shard_id} (restore failed): {err}"));
                    }
                };
                if let Err(err) = restore.await {
                    let reason = format!("failed to load: {err}");
                    log::error!(
                        "adopted shard {shard_id} of `{collection_name}` {reason} (marking failed)",
                    );
                    mark_adopted_shard_failed(
                        &collections,
                        &collection,
                        &collection_name,
                        shard_id,
                        &reason,
                        is_distributed,
                        this_peer_id,
                    )
                    .await;
                    return Err(format!("shard {shard_id} (load failed): {err}"));
                }

                // A concurrent `delete_collection` may have removed the collection while we
                // loaded (delete's `try_unwrap` only *delays* up to a timeout, then removes
                // the directory regardless). Skip activation unless the registered collection is
                // STILL THIS instance (compared by `Arc` identity, not name): if it was deleted, or
                // deleted-and-recreated under the same name, the entry is a different `Collection`
                // and our loaded data + markers belong to the old, orphaned one. Activating (or
                // writing markers) against a recreated same-name collection would corrupt it —
                // stamping `.adopt_activate_pending`/`.adopt_failed` onto its shard dir (same path
                // string) could activate an empty shard or wrongly drive a healthy one `Dead`.
                let still_this_collection =
                    is_current_adopted_collection(&collections, &collection_name, &collection).await;
                if !still_this_collection {
                    log::warn!(
                        "collection `{collection_name}` was deleted (or recreated) while adopting \
                         shard {shard_id}; skipping activation",
                    );
                    return Ok(());
                }

                // Hand off to activation. The data is now installed; flip the shard to `Active`.
                if is_distributed {
                    // Cluster: do NOT propose `Active` from here. A one-shot proposal from this
                    // detached task is silently dropped if this peer is not leader when it fires,
                    // with no retry, and a crash here would strand a loaded shard parked forever
                    // (nothing re-proposes `ManualRecovery`). Instead write the durable activation
                    // marker; `sync_local_state` then proposes `Active` through the leader-guarded
                    // consensus path, every reconcile tick, until it commits — surviving a lost
                    // proposal or a restart. `report_parked_adopted_shards` waits for that.
                    match collection.mark_shard_adopt_activation_pending(shard_id).await {
                        Ok(()) => {
                            log::info!(
                                "adopted shard {shard_id} of `{collection_name}` loaded; \
                                 activation pending",
                            );
                            Ok(())
                        }
                        Err(err) => {
                            log::error!(
                                "adopted shard {shard_id} of `{collection_name}` loaded but could \
                                 not be marked for activation (left parked): {err}",
                            );
                            Err(format!("shard {shard_id} (activation-marker failed): {err}"))
                        }
                    }
                } else {
                    // Single node: there is no consensus (so no leader to lose a proposal to) and
                    // `sync_local_state` does not run — activate directly. This task is awaited by
                    // the create below, so the shard is `Active` before the create returns.
                    match collection
                        .set_shard_replica_state(
                            shard_id,
                            this_peer_id,
                            ReplicaState::Active,
                            Some(ReplicaState::ManualRecovery),
                        )
                        .await
                    {
                        Ok(()) => {
                            log::info!(
                                "adopted and activated shard {shard_id} of `{collection_name}`"
                            );
                            Ok(())
                        }
                        Err(err) => {
                            log::error!(
                                "adopted shard {shard_id} of `{collection_name}` loaded but could \
                                 not be activated (left parked): {err}",
                            );
                            Err(format!("shard {shard_id} (activation failed): {err}"))
                        }
                    }
                }
            });

            if is_distributed {
                // Cluster: supervise the detached task. Clear the in-progress marker ONLY on
                // success — then the durable activation marker is in place. On any failure leave it:
                // a restart's `ShardReplicaSet::load` then resolves the shard (deferring to whatever
                // terminal marker exists), instead of the marker being dropped while a terminal
                // marker may have failed to write, which would strand the shard. On a panic/abort
                // (which the task's own error handling cannot catch) also record the failure so
                // `sync_local_state` drives the shard to `Dead` meanwhile.
                self.general_runtime.spawn(async move {
                    match handle.await {
                        Ok(Ok(())) => {
                            // Clear the in-progress marker only if this is still the registered
                            // instance — a delete+recreate under the same name during the load would
                            // otherwise make us clear the NEW collection's marker (path-keyed).
                            if is_current_adopted_collection(
                                &collections_super,
                                &collection_name_super,
                                &collection_super,
                            )
                            .await
                            {
                                collection_super.clear_shard_adopt_in_progress(shard_id);
                            }
                        }
                        Ok(Err(_reason)) => {
                            // Handled failure: the task wrote `.adopt_failed` (best-effort). Leave
                            // the in-progress marker for restart-time resolution.
                        }
                        Err(join_err) => {
                            let reason = format!("adoption load task aborted: {join_err}");
                            log::error!(
                                "adopted shard {shard_id} of `{collection_name_super}` {reason} \
                                 (marking failed)",
                            );
                            mark_adopted_shard_failed(
                                &collections_super,
                                &collection_super,
                                &collection_name_super,
                                shard_id,
                                &reason,
                                true,
                                this_peer_id,
                            )
                            .await;
                        }
                    }
                });
            } else {
                load_tasks.push((shard_id, handle));
            }
        }

        // Block here until the background loads finish, preserving the "create returns
        //  with shards active" contract in a single node context.
        if !self.is_distributed() {
            for (shard_id, task) in load_tasks {
                match task.await {
                    Ok(Ok(())) => {
                        // Success: the shard is active; drop the in-progress marker — but only if
                        // this is still the registered instance (a delete+recreate under the same
                        // name during the load would otherwise clear the NEW collection's marker).
                        if is_current_adopted_collection(
                            &self.collections,
                            collection_name,
                            &collection,
                        )
                        .await
                        {
                            collection.clear_shard_adopt_in_progress(shard_id);
                        }
                    }
                    Ok(Err(reason)) => {
                        // Handled failure: the task wrote `.adopt_failed` (and set `Dead` on
                        // single-node). Leave the in-progress marker so a restart's H1 resolves the
                        // shard even if the terminal write did not land.
                        failed.push(reason);
                    }
                    Err(join_err) => {
                        // The task panicked/aborted before writing a terminal marker: record the
                        // failure (also sets `Dead` directly on single-node) so the shard is not
                        // stranded in `ManualRecovery`. Leave the in-progress marker for restart.
                        let reason = format!("adoption load task aborted: {join_err}");
                        mark_adopted_shard_failed(
                            &self.collections,
                            &collection,
                            collection_name,
                            shard_id,
                            &reason,
                            false,
                            self.this_peer_id,
                        )
                        .await;
                        failed.push(format!("shard {shard_id} ({reason})"));
                    }
                }
            }
        }

        if failed.is_empty() {
            Ok(())
        } else {
            Err(StorageError::bad_request(format!(
                "collection `{collection_name}` was created, but {} of its local shard(s) on \
                 this peer could not be adopted and are left parked (not serving — the \
                 collection is degraded, not silently empty): {}. Fix the staging, then \
                 recover each parked shard via `adopt://` snapshot recovery, or drop the \
                 collection and recreate it from a re-staged source.",
                failed.len(),
                failed.join("; "),
            )))
        }
    }

    /// Phase A of adoption: resolve one staged shard directory and verify it is compatible
    /// with the collection being created. Returns the resolved source directory to install;
    /// does **not** load it — that is Phase B, off the apply thread (see
    /// [`Self::adopt_local_shards_on_create`]).
    ///
    /// A missing artifact, or one built for a different topology / vector shape, fails here
    /// synchronously with a precise `bad_request`, before any heavy load is scheduled.
    #[allow(clippy::too_many_arguments)]
    async fn stage_one_adopted_shard(
        &self,
        collection: &Collection,
        root: Option<&Path>,
        staging_subdir: &str,
        shard_id: ShardId,
        temp_dir: &Path,
        collections_dir: &Path,
        allow_config_rebuild: bool,
    ) -> Result<std::path::PathBuf, StorageError> {
        use crate::content_manager::snapshots::adopt;

        let Some(root) = root else {
            return Err(StorageError::bad_request(
                "`adopt_shards_from` requires `storage.shard_adoption_path` to be configured \
                 on every peer that is assigned a shard, and this peer has none",
            ));
        };

        let staged = root.join(staging_subdir).join(format!("shard_{shard_id}"));
        // Phase A runs on the consensus apply thread, so skip the O(files) symlink walk here — it
        // is done off-thread in Phase B before the install rename (see `adopt_local_shards_on_create`).
        let source = adopt::resolve_staged_dir(root, &staged, &[temp_dir, collections_dir], false)
            .map_err(|err| {
                StorageError::bad_request(format!(
                    "its artifact could not be adopted from {}: {err}",
                    staged.display(),
                ))
            })?;

        // Verify the artifact matches this collection (the same check `adopt://` recovery runs).
        // A missing manifest or a routing / vector-shape mismatch is refused — installing either
        // would serve a wrong or misrouted subset. An index-config difference (which would make
        // the optimizer rebuild every segment — expensive at scale, usually accidental) is refused
        // by default, but the caller may opt in via `adopt_allow_config_rebuild`.
        if let Some(consequence) = collection
            .check_adopted_shard_compatible(shard_id, &source)
            .await?
        {
            if allow_config_rebuild {
                log::warn!(
                    "adopting shard {shard_id} of `{}` despite a config mismatch \
                     (`adopt_allow_config_rebuild` is set): {consequence}",
                    collection.name(),
                );
            } else {
                return Err(StorageError::bad_request(format!(
                    "its artifact was built with a different index config than the collection \
                     being created: {consequence} If this reconfiguration is intended, set \
                     `adopt_allow_config_rebuild: true` to proceed anyway.",
                )));
            }
        }

        Ok(source)
    }

    async fn on_peer_created(
        &self,
        collection_name: String,
        peer_id: PeerId,
        shard_id: ShardId,
    ) -> CollectionResult<()> {
        if let Some(proposal_sender) = &self.consensus_proposal_sender {
            let operation =
                ConsensusOperations::initialize_replica(collection_name.clone(), shard_id, peer_id);
            if let Err(send_error) = proposal_sender.send(operation) {
                log::error!(
                    "Can't send proposal to deactivate replica on peer {peer_id} of shard {shard_id} of collection {collection_name}. Error: {send_error}",
                );
            }
        } else {
            // Just activate the shard
            let collections = self.collections.read().await;
            if let Some(collection) = collections.get(&collection_name) {
                collection
                    .set_shard_replica_state(
                        shard_id,
                        peer_id,
                        ReplicaState::Active,
                        Some(ReplicaState::Initializing),
                    )
                    .await?;
            }
        }
        Ok(())
    }
}
