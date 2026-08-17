use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::types::DeferredBehavior;
use segment::data_types::facets::{FacetParams, FacetResponse};
use segment::index::field_index::CardinalityEstimation;
use segment::types::{
    ExtendedPointId, Filter, ScoredPoint, SizeStats, StrictModeConfig, WithPayload,
    WithPayloadInterface, WithVector,
};
use shard::count::CountRequestInternal;
use shard::operations::CollectionUpdateOperations;
use shard::retrieve::record_internal::RecordInternal;
use shard::scroll::ScrollRequestInternal;
use shard::search::CoreSearchRequestBatch;
use shard::snapshots::snapshot_manifest::SnapshotManifest;

use crate::common::adaptive_handle::AdaptiveSearchHandle;
use crate::operations::OperationWithClockTag;
use crate::operations::types::{
    CollectionError, CollectionInfo, CollectionResult, CountResult, OptimizersStatus,
    PointRequestInternal, ShardStatus, UpdateResult, UpdateStatus,
};
use crate::operations::universal_query::shard_query::{ShardQueryRequest, ShardQueryResponse};
use crate::shards::shard_trait::{ShardOperation, WaitUntil};
use crate::shards::telemetry::LocalShardTelemetry;

/// Why a local shard is standing in as a [`DummyShard`] instead of serving.
///
/// This exists to answer one question that consumers cannot otherwise ask: **may the data in this
/// shard's directory be discarded?** Most causes of a dummy shard mean "yes, and please refill me" —
/// the recovery machinery relies on that, clearing the directory and pulling a fresh copy from another
/// replica. One cause means the exact opposite, and reading it as discardable destroys the only copy of
/// the data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DummyShardReason {
    /// Qdrant was started in recovery mode, so no local shard was loaded at all.
    RecoveryMode(String),

    /// The shard is not fully initialized — the "dirty shard" flag is set, so a previous write or
    /// recovery was interrupted partway. Recovery is expected to replace it.
    DirtyShard,

    /// The shard's points were placed by a hash ring at a different scale than the collection now
    /// routes with, so serving them would misroute every id in the shard.
    ///
    /// The one reason that must **not** be discarded: this directory holds the only copy of data placed
    /// under `placed_under`, and restoring the collection to that scale is supposed to bring it back.
    HashRingShardScaleMismatch { placed_under: u32, expected: u32 },

    /// Placeholder held while the local shard is deliberately cleared ahead of a snapshot recovery.
    ClearingForSnapshotRecovery,

    /// Loading the shard from disk failed.
    LoadFailed(String),

    /// Restoring the shard from a snapshot failed, and its data was cleared.
    RestoreFailed(String),

    /// Building a fresh empty local shard failed.
    InitializationFailed(String),
}

impl DummyShardReason {
    /// Whether the data in this shard's directory may be discarded to make room for a replacement.
    ///
    /// `false` means the bytes on disk are the only copy and destroying them loses data. Callers that
    /// clear or remove shard data must not do so when this is `false` without explicit operator intent.
    pub fn may_be_discarded(&self) -> bool {
        match self {
            // All of these mean the directory holds nothing worth keeping, or is expected to be
            // replaced by a healthy copy.
            Self::RecoveryMode(_)
            | Self::DirtyShard
            | Self::ClearingForSnapshotRecovery
            | Self::LoadFailed(_)
            | Self::RestoreFailed(_)
            | Self::InitializationFailed(_) => true,

            // The only copy of data placed under another scale. See the variant's docs.
            Self::HashRingShardScaleMismatch { .. } => false,
        }
    }
}

impl std::fmt::Display for DummyShardReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RecoveryMode(reason) => write!(f, "{reason}"),
            Self::DirtyShard => write!(f, "Dirty shard - shard is not fully initialized"),
            Self::HashRingShardScaleMismatch {
                placed_under,
                expected,
            } => write!(
                f,
                "Shard was placed under hash ring shard scale {placed_under}, \
                 but the collection is configured for {expected}",
            ),
            Self::ClearingForSnapshotRecovery => {
                write!(f, "Local shard is being cleared for snapshot recovery")
            }
            Self::LoadFailed(err) => write!(f, "{err}"),
            Self::RestoreFailed(err) => write!(f, "{err}"),
            Self::InitializationFailed(err) => write!(f, "{err}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DummyShard {
    reason: DummyShardReason,
    message: String,
}

impl DummyShard {
    pub fn new(reason: DummyShardReason) -> Self {
        Self {
            message: reason.to_string(),
            reason,
        }
    }

    /// Why this shard is not serving, in a form consumers can branch on.
    pub fn reason(&self) -> &DummyShardReason {
        &self.reason
    }

    pub fn snapshot_manifest(&self) -> CollectionResult<SnapshotManifest> {
        Ok(SnapshotManifest::default())
    }

    pub fn on_optimizer_config_update(&self) -> CollectionResult<()> {
        let error = self.dummy_error("Update optimizer config");
        log::error!("{error}");
        // We can't fail this operation because this operation is part of consensus loop
        Ok(())
    }

    pub fn on_strict_mode_config_update(&mut self, _new_strict_mode: &StrictModeConfig) {}

    pub fn get_telemetry_data(&self) -> LocalShardTelemetry {
        LocalShardTelemetry {
            variant_name: Some("dummy shard".into()),
            status: Some(ShardStatus::Green),
            total_optimized_points: 0,
            vectors_size_bytes: None,
            payloads_size_bytes: None,
            num_points: None,
            num_vectors: None,
            num_vectors_by_name: None,
            segments: None,
            optimizations: Default::default(),
            async_scorer: None,
            indexed_only_excluded_vectors: None,
            update_queue: None,
        }
    }

    pub fn get_optimization_status(&self) -> OptimizersStatus {
        OptimizersStatus::Ok
    }

    pub fn get_size_stats(&self) -> SizeStats {
        SizeStats::default()
    }

    pub fn estimate_cardinality(
        &self,
        _: Option<&Filter>,
    ) -> CollectionResult<CardinalityEstimation> {
        self.dummy("estimate_cardinality")
    }

    pub fn dummy_error(&self, action: &str) -> CollectionError {
        CollectionError::service_error(format!("Failed to {action},  {}", self.message))
    }

    fn dummy<T>(&self, action: &str) -> CollectionResult<T> {
        Err(self.dummy_error(action))
    }
}

#[async_trait]
impl ShardOperation for DummyShard {
    async fn update(
        &self,
        op: OperationWithClockTag,
        _: WaitUntil,
        _: Option<Duration>,
        _: HwMeasurementAcc,
    ) -> CollectionResult<UpdateResult> {
        match &op.operation {
            CollectionUpdateOperations::PointOperation(_) => self.dummy("Update Points"),
            CollectionUpdateOperations::VectorOperation(_) => self.dummy("Update Vectors"),
            CollectionUpdateOperations::PayloadOperation(_) => self.dummy("Update Payloads"),

            // Allow (and ignore) field index and vector name operations.
            // These schemas are stored in collection config and will be recreated when recovered.
            CollectionUpdateOperations::FieldIndexOperation(_)
            | CollectionUpdateOperations::VectorNameOperation(_) => Ok(UpdateResult {
                operation_id: None,
                status: UpdateStatus::Acknowledged,
                clock_tag: None,
            }),
            // Allow (and ignore) staging operations on dummy shards
            #[cfg(feature = "staging")]
            CollectionUpdateOperations::StagingOperation(_) => Ok(UpdateResult {
                operation_id: None,
                status: UpdateStatus::Acknowledged,
                clock_tag: None,
            }),
        }
    }

    /// Forward read-only `scroll_by` to `wrapped_shard`
    async fn scroll_by(
        &self,
        _: Arc<ScrollRequestInternal>,
        _: &AdaptiveSearchHandle,
        _: Option<Duration>,
        _: HwMeasurementAcc,
    ) -> CollectionResult<Vec<RecordInternal>> {
        self.dummy("Scroll")
    }

    async fn local_scroll_by_id(
        &self,
        _: Option<ExtendedPointId>,
        _: usize,
        _: &WithPayloadInterface,
        _: &WithVector,
        _: Option<&Filter>,
        _: &AdaptiveSearchHandle,
        _: Option<Duration>,
        _: HwMeasurementAcc,
        _: DeferredBehavior,
    ) -> CollectionResult<Vec<RecordInternal>> {
        self.dummy("Scroll by ID")
    }

    async fn info(&self) -> CollectionResult<CollectionInfo> {
        self.dummy("Get Info")
    }

    async fn core_search(
        &self,
        _: Arc<CoreSearchRequestBatch>,
        _: &AdaptiveSearchHandle,
        _: Option<Duration>,
        _: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        self.dummy("search")
    }

    async fn count(
        &self,
        _: Arc<CountRequestInternal>,
        _: &AdaptiveSearchHandle,
        _: Option<Duration>,
        _: HwMeasurementAcc,
        _: DeferredBehavior,
    ) -> CollectionResult<CountResult> {
        self.dummy("count")
    }

    async fn retrieve(
        &self,
        _: Arc<PointRequestInternal>,
        _: &WithPayload,
        _: &WithVector,
        _: &AdaptiveSearchHandle,
        _: Option<Duration>,
        _: HwMeasurementAcc,
        _: DeferredBehavior,
    ) -> CollectionResult<Vec<RecordInternal>> {
        self.dummy("retrieve")
    }

    async fn query_batch(
        &self,
        _requests: Arc<Vec<ShardQueryRequest>>,
        _search_runtime_handle: &AdaptiveSearchHandle,
        _timeout: Option<Duration>,
        _: HwMeasurementAcc,
    ) -> CollectionResult<Vec<ShardQueryResponse>> {
        self.dummy("query")
    }

    async fn facet(
        &self,
        _: Arc<FacetParams>,
        _search_runtime_handle: &AdaptiveSearchHandle,
        _: Option<Duration>,
        _: HwMeasurementAcc,
    ) -> CollectionResult<FacetResponse> {
        self.dummy("facet")
    }

    async fn stop_gracefully(self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The distinction the type exists for. Every cause that means "this directory is expendable" must
    /// read as discardable, and the one that means "this is the only copy" must not — that inversion is
    /// what previously destroyed a diverged shard and then could not refill it.
    #[test]
    fn only_a_scale_mismatch_forbids_discarding_the_data() {
        let discardable = [
            DummyShardReason::RecoveryMode("recovering".into()),
            DummyShardReason::DirtyShard,
            DummyShardReason::ClearingForSnapshotRecovery,
            DummyShardReason::LoadFailed("boom".into()),
            DummyShardReason::RestoreFailed("boom".into()),
            DummyShardReason::InitializationFailed("boom".into()),
        ];
        for reason in &discardable {
            assert!(
                reason.may_be_discarded(),
                "{reason:?} should be discardable - recovery relies on refilling these",
            );
        }

        let protected = DummyShardReason::HashRingShardScaleMismatch {
            placed_under: 7,
            expected: 100,
        };
        assert!(
            !protected.may_be_discarded(),
            "a scale mismatch means the directory holds the only copy of that data",
        );

        // Guards against the whole thing collapsing to one answer, which would make the loop above
        // pass while protecting nothing.
        assert_ne!(
            discardable[0].may_be_discarded(),
            protected.may_be_discarded(),
            "the two kinds must not answer the same way",
        );
    }

    /// The reason has to survive into the message a caller sees, because that message is the only thing
    /// an operator gets when a write is refused.
    #[test]
    fn the_reason_reaches_the_error_a_caller_sees() {
        let shard = DummyShard::new(DummyShardReason::HashRingShardScaleMismatch {
            placed_under: 7,
            expected: 100,
        });

        let message = shard.dummy_error("update").to_string();
        for expected in ["hash ring shard scale", "7", "100"] {
            assert!(
                message.contains(expected),
                "the error should carry {expected:?}, got {message:?}",
            );
        }

        assert_eq!(
            shard.reason(),
            &DummyShardReason::HashRingShardScaleMismatch {
                placed_under: 7,
                expected: 100,
            },
            "the reason must be readable back for consumers that branch on it",
        );
    }

    /// A dirty shard is the case recovery depends on, so its message must stay recognisable.
    #[test]
    fn a_dirty_shard_still_reports_itself_as_before() {
        let shard = DummyShard::new(DummyShardReason::DirtyShard);
        assert!(
            shard
                .dummy_error("update")
                .to_string()
                .contains("Dirty shard")
        );
        assert!(shard.reason().may_be_discarded());
    }
}
