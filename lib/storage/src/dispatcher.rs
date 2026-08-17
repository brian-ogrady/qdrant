use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use api::rest::models::HardwareUsage;
use collection::common::fetch_vectors::CollectionName;
use collection::config::ShardingMethod;
use collection::hash_ring::MAX_HASH_RING_SHARD_SCALE;
use collection::operations::verification::VerificationPass;
use collection::shards::replica_set::replica_set_state::ReplicaState;
use common::counter::hardware_accumulator::HwSharedDrain;
use common::defaults::CONSENSUS_META_OP_WAIT;
use futures::StreamExt as _;
use futures::stream::FuturesUnordered;
use segment::types::ShardKey;

use crate::content_manager::collection_meta_ops::AliasOperations;
use crate::content_manager::shard_distribution::ShardDistributionProposal;
use crate::rbac::{Auth, CollectionMultipass};
use crate::{
    ClusterStatus, CollectionMetaOperations, ConsensusOperations, ConsensusStateRef, StorageError,
    TableOfContent,
};

#[derive(Clone)]
pub struct Dispatcher {
    toc: Arc<TableOfContent>,
    consensus_state: Option<ConsensusStateRef>,
    resharding_enabled: bool,
}

impl Dispatcher {
    pub fn new(toc: Arc<TableOfContent>) -> Self {
        Self {
            toc,
            consensus_state: None,
            resharding_enabled: false,
        }
    }

    pub fn with_consensus(self, state_ref: ConsensusStateRef, resharding_enabled: bool) -> Self {
        Self {
            consensus_state: Some(state_ref),
            resharding_enabled,
            ..self
        }
    }

    /// Get the table of content.
    /// The `_auth` and `_verification_pass` parameter are not used, but it's required to verify caller's possession
    /// of both objects.
    pub fn toc(&self, _auth: &Auth, _verification_pass: &VerificationPass) -> &Arc<TableOfContent> {
        &self.toc
    }

    pub fn consensus_state(&self) -> Option<&ConsensusStateRef> {
        self.consensus_state.as_ref()
    }

    pub fn is_resharding_enabled(&self) -> bool {
        self.resharding_enabled
    }

    /// Whether a non-default `hash_ring_shard_scale` can safely be put into consensus.
    /// Raft entries are CBOR maps with no `deny_unknown_fields`, so a peer that predates the field
    /// silently drops it and builds its ring at the default — one collection, two point-to-shard
    /// mappings, and nothing reconciles them afterwards because the scale is deliberately excluded
    /// from `CollectionParams::check_compatible`. Always true single-node: nobody to disagree.
    pub fn hash_ring_shard_scale_supported(&self) -> bool {
        self.consensus_state.is_none()
            || self
                .toc
                .get_channel_service()
                .all_known_peers_at_version(&collection::hash_ring::HASH_RING_SHARD_SCALE_VERSION)
    }

    /// Reject a *client-requested* non-default scale that consensus cannot carry safely yet.
    pub fn check_hash_ring_shard_scale_requestable(
        &self,
        collection_name: &str,
        requested: Option<u32>,
    ) -> Result<(), StorageError> {
        check_hash_ring_shard_scale_requestable(
            collection_name,
            requested,
            self.hash_ring_shard_scale_supported(),
        )
    }

    /// If `wait_timeout` is not supplied - then default duration will be used.
    ///
    /// This function needs to be called from a runtime with timers enabled.
    ///
    /// ## Cancel safety
    ///
    /// This function is cancel safe.
    ///
    /// On deployments without consensus - a submitted operation is always run to completion.
    pub async fn submit_collection_meta_op(
        &self,
        mut operation: CollectionMetaOperations,
        auth: Auth,
        wait_timeout: Option<Duration>,
    ) -> Result<bool, StorageError> {
        auth.check_collection_meta_operation(&operation)?;

        // Refuse an attempt to *change* the scale of an existing collection rather than dropping it
        // on the floor (immutability itself: see `CollectionParams::hash_ring_shard_scale`).
        if let CollectionMetaOperations::UpdateCollection(op) = &operation
            && let Some(params) = &op.update_collection.params
            && let Some(requested) = params.hash_ring_shard_scale
            && let Some(collection) = self
                .toc
                .get_collection_opt(op.collection_name.clone())
                .await
            && collection.hash_ring_shard_scale().await != requested
        {
            let current = collection.hash_ring_shard_scale().await;
            return Err(StorageError::bad_input(format!(
                "`hash_ring_shard_scale` cannot be changed after a collection is created \
                 (collection {} is {current}, requested {requested}). It determines which shard each \
                 point belongs to, so changing it would move a portion of the keyspace and leave \
                 existing points unreachable by id. To use a different scale, create a new collection \
                 with `hash_ring_shard_scale` set and migrate the points into it",
                op.collection_name,
            )));
        }

        if let CollectionMetaOperations::CreateCollection(op) = &mut operation
            && op.create_collection.hash_ring_shard_scale.is_none()
        {
            let configured = self
                .toc
                .storage_config
                .collection
                .as_ref()
                .and_then(|defaults| defaults.hash_ring_shard_scale);

            // `Settings::validate_and_warn` already reports an out-of-range value at startup, but it
            // only warns, so the bad value is still here. Fall back to default value rather than
            // propagating it.
            let mut scale = match configured {
                Some(scale) if (1..=MAX_HASH_RING_SHARD_SCALE).contains(&scale) => scale,
                Some(invalid) => {
                    log::warn!(
                        "Ignoring storage.collection.hash_ring_shard_scale={invalid}: \
                         must be in 1..={MAX_HASH_RING_SHARD_SCALE}. \
                         Creating collection {} with {} instead",
                        op.collection_name,
                        collection::config::default_hash_ring_shard_scale(),
                    );
                    collection::config::default_hash_ring_shard_scale()
                }
                None => collection::config::default_hash_ring_shard_scale(),
            };

            // A non-default default cannot go into consensus while any peer predates the field (see
            // `hash_ring_shard_scale_supported`), so fall back rather than refuse.
            if scale != collection::config::default_hash_ring_shard_scale()
                && !self.hash_ring_shard_scale_supported()
            {
                log::warn!(
                    "Creating collection {} with hash ring shard scale {} instead of the configured \
                     {scale}: not all peers run at least {}, and older peers would build a different \
                     point-to-shard mapping for it",
                    op.collection_name,
                    collection::config::default_hash_ring_shard_scale(),
                    *collection::hash_ring::HASH_RING_SHARD_SCALE_VERSION,
                );
                scale = collection::config::default_hash_ring_shard_scale();
            }

            op.create_collection.hash_ring_shard_scale = Some(scale);
        }

        // if distributed deployment is enabled
        if let Some(state) = self.consensus_state.as_ref() {
            let start = Instant::now();

            // List of operations to await for collection to be operational
            let mut expect_operations: Vec<ConsensusOperations> = vec![];

            let op = match operation {
                CollectionMetaOperations::CreateCollection(mut op) => {
                    if !op.is_distribution_set() {
                        match op.create_collection.sharding_method.unwrap_or_default() {
                            ShardingMethod::Auto => {
                                // Suggest even distribution of shards across nodes
                                let number_of_peers = state.0.peer_count();

                                let collection_defaults =
                                    self.toc.storage_config.collection.as_ref();

                                let shard_distribution = self.toc.suggest_shard_distribution(
                                    &op,
                                    collection_defaults,
                                    number_of_peers,
                                );

                                // Expect all replicas to become active eventually
                                for (shard_id, peer_ids) in &shard_distribution.distribution {
                                    for peer_id in peer_ids {
                                        expect_operations.push(
                                            ConsensusOperations::initialize_replica(
                                                op.collection_name.clone(),
                                                *shard_id,
                                                *peer_id,
                                            ),
                                        );
                                    }
                                }

                                op.set_distribution(shard_distribution);
                            }
                            ShardingMethod::Custom => {
                                // If custom sharding is used - we don't create any shards in advance
                                let empty_distribution = ShardDistributionProposal::empty();
                                op.set_distribution(empty_distribution);
                            }
                        }
                    }

                    if let Some(uuid) = &op.create_collection.uuid {
                        log::warn!(
                            "Collection UUID {uuid} explicitly specified, \
                             when proposing create collection {} operation, \
                             new random UUID will be generated instead",
                            op.collection_name,
                        );
                    }

                    op.create_collection.uuid = Some(uuid::Uuid::new_v4());

                    CollectionMetaOperations::CreateCollection(op)
                }
                CollectionMetaOperations::CreateShardKey(op) => {
                    CollectionMetaOperations::CreateShardKey(op)
                }

                CollectionMetaOperations::UpdateCollection(_)
                | CollectionMetaOperations::DeleteCollection(_)
                | CollectionMetaOperations::ChangeAliases(_)
                | CollectionMetaOperations::Resharding(_, _)
                | CollectionMetaOperations::TransferShard(_, _)
                | CollectionMetaOperations::SetShardReplicaState(_)
                | CollectionMetaOperations::DropShardKey(_)
                | CollectionMetaOperations::CreatePayloadIndex(_)
                | CollectionMetaOperations::DropPayloadIndex(_)
                | CollectionMetaOperations::CreateNamedVector(_)
                | CollectionMetaOperations::DeleteNamedVector(_)
                | CollectionMetaOperations::Nop { .. } => operation,
                #[cfg(feature = "staging")]
                CollectionMetaOperations::TestSlowDown(_)
                | CollectionMetaOperations::TestTransientError(_) => operation,
            };

            let operation_awaiter =
                // If explicit timeout is set - then we need to wait for all expected operations.
                // E.g. in case of `CreateCollection` we will explicitly wait for all replicas to be activated.
                // We need to register receivers(by calling the function) before submitting the operation.
                if !expect_operations.is_empty() {
                    Some(state.await_for_multiple_operations(expect_operations, wait_timeout))
                } else {
                    None
                };

            let do_sync_nodes = match &op {
                // Sync nodes after collection or shard key creation, or after
                // adding/removing a named vector — in all of these cases callers
                // expect the updated collection config to be visible on every
                // peer on return.
                CollectionMetaOperations::CreateCollection(_)
                | CollectionMetaOperations::CreateShardKey(_)
                | CollectionMetaOperations::CreateNamedVector(_)
                | CollectionMetaOperations::DeleteNamedVector(_) => true,

                // Sync nodes when creating or renaming collection aliases
                CollectionMetaOperations::ChangeAliases(changes) => {
                    changes.actions.iter().any(|change| match change {
                        AliasOperations::CreateAlias(_) | AliasOperations::RenameAlias(_) => true,
                        AliasOperations::DeleteAlias(_) => false,
                    })
                }

                // TODO(resharding): Do we need/want to synchronize `Resharding` operations?
                CollectionMetaOperations::Resharding(_, _) => false,

                // No need to sync nodes for other operations
                CollectionMetaOperations::UpdateCollection(_)
                | CollectionMetaOperations::DeleteCollection(_)
                | CollectionMetaOperations::TransferShard(_, _)
                | CollectionMetaOperations::SetShardReplicaState(_)
                | CollectionMetaOperations::DropShardKey(_)
                | CollectionMetaOperations::CreatePayloadIndex(_)
                | CollectionMetaOperations::DropPayloadIndex(_)
                | CollectionMetaOperations::Nop { .. } => false,

                #[cfg(feature = "staging")]
                CollectionMetaOperations::TestSlowDown(_)
                | CollectionMetaOperations::TestTransientError(_) => false,
            };

            // During creation of a shard key, we must ensure that all replicas are ready to accept
            // write requests, so the client-side script can rely on the fact that the
            // shard creation request is complete.
            //
            // For this we explicitly wait for validation this, we do following checks:
            //
            // 1. Wait for consensus to accept shard create operation on current machine.
            //    ( here newly created shards should start to report state change from `Inactive` to `Active` )
            // 2. Wait for all local shards to become active.
            //    ( At this stage we are sure, that all consensus operations are created, but might not be applied everywhere )
            // 3. Wait for all remote peers to have at least the same state as the current peer.
            //    ( So we are sure, that all remote peers have also switched to `Active` state )
            #[expect(clippy::wildcard_enum_match_arm, reason = "too many enum variants")]
            let create_shard_key = match &op {
                CollectionMetaOperations::CreateShardKey(op) => {
                    let collection_name: CollectionName = op.collection_name.clone();
                    let shard_key = op.shard_key.clone();
                    let initial_state = op.initial_state;
                    Some((collection_name, shard_key, initial_state))
                }
                _ => None,
            };

            // Send operation to consensus and wait for it to be applied locally
            let res = state
                .propose_consensus_op_with_await(
                    ConsensusOperations::CollectionMeta(Box::new(op)),
                    wait_timeout,
                )
                .await?;

            if let Some(operation_awaiter) = operation_awaiter {
                // Actually await for expected operations to complete on the consensus
                match operation_awaiter.await {
                    Ok(Ok(())) => {} // all good
                    Ok(Err(err)) => {
                        log::warn!("Not all expected operations were completed: {err}")
                    }
                    Err(err) => log::warn!("Awaiting for expected operations timed out: {err}"),
                }
            }

            // Wait for shards activation
            if let Some((collection_name, shard_key, initial_state)) = create_shard_key
                && initial_state.is_none()
            {
                // Only do if initial state is not set because we only wanted to wait for Active since introducing
                // the Initial state which needs a transition to Active.
                let remaining_timeout =
                    wait_timeout.map(|timeout| timeout.saturating_sub(start.elapsed()));
                self.wait_for_shard_key_activation(collection_name, shard_key, remaining_timeout)
                    .await?;
            };

            // On some operations, synchronize all nodes to ensure all are ready for point operations
            if do_sync_nodes {
                let remaining_timeout =
                    wait_timeout.map(|timeout| timeout.saturating_sub(start.elapsed()));
                if let Err(err) = self.await_consensus_sync(remaining_timeout).await {
                    log::warn!(
                        "Failed to synchronize all nodes after collection operation in time, some nodes may not be ready: {err}",
                    );
                }
            }

            Ok(res)
        } else {
            let toc = self.toc.clone();
            tokio::task::spawn(async move { toc.perform_collection_meta_op(operation).await })
                .await?
        }
    }

    pub fn cluster_status(&self) -> ClusterStatus {
        match self.consensus_state.as_ref() {
            Some(state) => state.cluster_status(),
            None => ClusterStatus::Disabled,
        }
    }

    pub async fn await_consensus_sync(
        &self,
        timeout: Option<Duration>,
    ) -> Result<(), StorageError> {
        let timeout = timeout.unwrap_or(CONSENSUS_META_OP_WAIT);

        let Some(state) = self.consensus_state.as_ref() else {
            return Ok(());
        };

        let state = state.hard_state();
        let term = state.term;
        let commit = state.commit;
        let channel_service = self.toc.get_channel_service();
        let this_peer_id = self.toc.this_peer_id;

        channel_service
            .await_commit_on_all_peers(this_peer_id, commit, term, timeout)
            .await?;

        log::debug!("Consensus is synchronized with term: {term}, commit: {commit}");

        Ok(())
    }

    /// Waits for all shards of a specific shard key to become active.
    pub async fn wait_for_shard_key_activation(
        &self,
        collection_name: CollectionName,
        shard_key: ShardKey,
        timeout: Option<Duration>,
    ) -> Result<(), StorageError> {
        let timeout = timeout.unwrap_or(CONSENSUS_META_OP_WAIT);

        let mut wait_for_active = FuturesUnordered::new();

        {
            let shard_holder = self
                .toc
                .get_collection(&CollectionMultipass.issue_pass(&collection_name))
                .await?
                .shards_holder()
                .read_owned()
                .await;

            for replica_set in shard_holder.all_shards() {
                if replica_set.shard_key().as_ref() != Some(&shard_key) {
                    continue;
                }

                for (peer_id, replica_state) in replica_set.peers() {
                    if replica_state == ReplicaState::Active {
                        continue;
                    }

                    wait_for_active.push(replica_set.wait_for_state(
                        peer_id,
                        ReplicaState::Active,
                        timeout,
                    ));
                }
            }
        }

        while let Some(result) = wait_for_active.next().await {
            result?;
        }

        Ok(())
    }

    pub fn all_hw_metrics(&self) -> HashMap<String, HardwareUsage> {
        self.toc.all_hw_metrics()
    }

    #[must_use]
    pub fn get_collection_hw_metrics(&self, collection: String) -> Arc<HwSharedDrain> {
        self.toc.get_collection_hw_metrics(collection)
    }
}

/// The decision behind [`Dispatcher::check_hash_ring_shard_scale_requestable`], split out so the
/// matrix is testable: constructing a `Dispatcher` whose `hash_ring_shard_scale_supported()` is false
/// requires a live `ConsensusStateRef`, which a unit test cannot build.
fn check_hash_ring_shard_scale_requestable(
    collection_name: &str,
    requested: Option<u32>,
    supported: bool,
) -> Result<(), StorageError> {
    let Some(requested) = requested else {
        return Ok(());
    };

    // The default is always safe to propose: it is exactly what a peer that ignores the field picks.
    if requested == collection::config::default_hash_ring_shard_scale() || supported {
        return Ok(());
    }

    Err(StorageError::bad_request(format!(
        "`hash_ring_shard_scale` of {requested} cannot be used for collection {collection_name} \
         until every peer runs at least {}: older peers ignore this setting and would build a \
         different point-to-shard mapping for the same collection. Finish the rolling upgrade, or \
         omit `hash_ring_shard_scale` to use the default of {}",
        *collection::hash_ring::HASH_RING_SHARD_SCALE_VERSION,
        collection::config::default_hash_ring_shard_scale(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A create request must be refused only when it asks for a scale that consensus cannot carry
    /// safely yet. Getting any cell of this wrong is either a silent split-brain (accepting when it
    /// should refuse) or an outage (refusing a default-scale create, which is every normal request).
    #[test]
    fn hash_ring_shard_scale_is_refused_only_when_requested_and_unsupported() {
        let default = collection::config::default_hash_ring_shard_scale();
        let non_default = default + 1;

        for (requested, supported, should_pass, case) in [
            (None, false, true, "no request, mixed cluster"),
            (None, true, true, "no request, upgraded cluster"),
            (
                Some(default),
                false,
                true,
                "default requested, mixed cluster",
            ),
            (
                Some(default),
                true,
                true,
                "default requested, upgraded cluster",
            ),
            (
                Some(non_default),
                true,
                true,
                "non-default, upgraded cluster",
            ),
            (
                Some(non_default),
                false,
                false,
                "non-default, mixed cluster",
            ),
        ] {
            let result = check_hash_ring_shard_scale_requestable("c", requested, supported);
            assert_eq!(
                result.is_ok(),
                should_pass,
                "{case}: expected pass={should_pass}, got {result:?}",
            );
        }

        // The refusal has to say what to do about it, not just that it failed.
        let err = check_hash_ring_shard_scale_requestable("c", Some(non_default), false)
            .expect_err("a non-default scale on a mixed cluster must be refused");
        let message = err.to_string();
        for expected in ["hash_ring_shard_scale", "every peer runs at least", "c"] {
            assert!(
                message.contains(expected),
                "refusal should be actionable; missing {expected:?} in {message:?}",
            );
        }
    }
}
