use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use common::bitvec::BitSlice;
use common::condition_checker::{CheckItem, ConditionChecker, Rest, Select};
use common::counter::hardware_counter::HardwareCounterCell;
use common::fixed_length_priority_queue::FixedLengthPriorityQueue;
use common::generic_consts::Random;
use common::types::{PointOffsetType, ScoreType, ScoredPointOffset};
use smallvec::SmallVec;

use crate::common::operation_error::{OperationError, OperationResult, check_process_stopped};
use crate::data_types::vectors::QueryVector;
use crate::index::query_optimization::optimized_filter::OptimizedFilter;
use crate::vector_storage::common::VECTOR_READ_BATCH_SIZE;
#[cfg(feature = "testing")]
use crate::vector_storage::new_raw_scorer;
use crate::vector_storage::quantized::quantized_query_scorer::InternalScorerUnsupported;
use crate::vector_storage::quantized::quantized_vectors::{QuantizedVectors, QuantizedVectorsRead};
use crate::vector_storage::query_scorer::QueryScorerBytes;
use crate::vector_storage::{
    NotDeletedChecker, RawScorer, RawScorerBuilder, VectorStorageEnum, VectorStorageRead,
};

/// The encoded vectors of one payload block, copied into a single buffer.
pub struct BlockVectors {
    /// One encoded vector per block point, in local id order, spaced
    /// `padded_stride` bytes apart.
    data: Vec<u64>,
    stride: usize,
    padded_stride: usize,
    len: usize,
}

/// Largest buffer a single block may hold.
pub const MAX_BLOCK_GATHER_BYTES: usize = 16 * 1024 * 1024;

/// Stage allowance per indexing thread the build was granted.
pub const GATHER_BYTES_PER_THREAD: usize = 8 * 1024 * 1024;

/// A stage-wide allowance for block-vector copies, reserved before each copy and
/// released when it is dropped.
pub struct GatherBudget {
    remaining: AtomicUsize,
}

/// Holds a slice of [`GatherBudget`] until dropped. Keep it alive for exactly as
/// long as the buffer it accounts for.
#[must_use = "dropping the reservation immediately releases the allowance it holds"]
pub struct GatherReservation<'a> {
    budget: &'a GatherBudget,
    bytes: usize,
}

impl GatherBudget {
    pub fn new(total_bytes: usize) -> Self {
        GatherBudget {
            remaining: AtomicUsize::new(total_bytes),
        }
    }

    /// Reserve `bytes`, or `None` if that would exceed what is left.
    pub fn try_reserve(&self, bytes: usize) -> Option<GatherReservation<'_>> {
        self.remaining
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |left| {
                left.checked_sub(bytes)
            })
            .ok()
            .map(|_| GatherReservation {
                budget: self,
                bytes,
            })
    }
}

impl Drop for GatherReservation<'_> {
    fn drop(&mut self) {
        self.budget
            .remaining
            .fetch_add(self.bytes, Ordering::AcqRel);
    }
}

/// Alignment the gather buffer can guarantee, from its `u64` backing store.
const GATHER_ALIGN: usize = align_of::<u64>();

impl BlockVectors {
    pub fn gather_size(
        points: usize,
        vectors: &VectorStorageEnum,
        quantized_vectors: Option<&QuantizedVectors>,
    ) -> Option<usize> {
        let layout = match quantized_vectors {
            Some(quantized) => quantized.get_quantized_vector_layout().ok()?,
            None => vectors.get_vector_layout().ok()?,
        };
        if points == 0 || layout.size() == 0 || layout.align() > GATHER_ALIGN {
            return None;
        }
        Self::backing_bytes(layout.size().next_multiple_of(layout.align()), points)
    }

    /// Bytes the backing store occupies for `points` vectors spaced
    /// `padded_stride` apart.
    fn backing_bytes(padded_stride: usize, points: usize) -> Option<usize> {
        padded_stride
            .checked_mul(points)?
            .div_ceil(size_of::<u64>())
            .checked_mul(size_of::<u64>())
    }

    /// Copy the encoded vectors of `points` into one contiguous buffer.
    ///
    /// Returns `None` - meaning "score against the global storage instead" -
    /// when this storage exposes no fixed-layout byte view (sparse and
    /// multivector storages), when the layout needs more alignment than the
    /// buffer provides, or when the copy would exceed `max_bytes`.
    pub fn try_gather(
        points: &[PointOffsetType],
        vectors: &VectorStorageEnum,
        quantized_vectors: Option<&QuantizedVectors>,
        max_bytes: usize,
    ) -> Option<Self> {
        let layout = match quantized_vectors {
            Some(quantized) => quantized.get_quantized_vector_layout().ok()?,
            None => vectors.get_vector_layout().ok()?,
        };
        let stride = layout.size();
        if stride == 0 || layout.align() > GATHER_ALIGN {
            return None;
        }

        let padded_stride = stride.next_multiple_of(layout.align());

        // Bound the allocation, not just the vectors in it, so a caller that
        // reserved `gather_size` bytes cannot be overrun by the word rounding.
        let backing = Self::backing_bytes(padded_stride, points.len())?;
        if backing == 0 || backing > max_bytes {
            return None;
        }

        let mut data = vec![0u64; backing / size_of::<u64>()];
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);

        for (local, &global) in points.iter().enumerate() {
            let dst = &mut bytes[local * padded_stride..local * padded_stride + stride];
            let copied = match quantized_vectors {
                Some(quantized) => {
                    let src = quantized.get_quantized_vector(global);
                    (src.len() == stride).then(|| dst.copy_from_slice(&src))
                }
                // 1.19 returns the byte view through a `Result`. A read error and a
                // missing vector mean the same thing here - decline the copy and let
                // the scorer read this storage by id, which surfaces any real error
                // on its own.
                None => match vectors.with_vector_bytes_opt::<Random, _>(global, |src| {
                    (src.len() == stride).then(|| dst.copy_from_slice(src))
                }) {
                    Ok(copied) => copied.flatten(),
                    Err(_) => None,
                },
            };
            // A short or missing vector means the layout does not describe this
            // storage after all. Score against the storage rather than guess.
            copied?;
        }

        Some(BlockVectors {
            data,
            stride,
            padded_stride,
            len: points.len(),
        })
    }

    /// What [`Self::try_gather`] requires of this storage, phrased for the one
    /// line a build logs when it declined to copy some of its blocks.
    ///
    /// A build where the copy silently never engaged should be visible without
    /// turning on per-block debug logging, because from the outside it looks
    /// exactly like a build where it did.
    pub fn gather_constraints(
        vectors: &VectorStorageEnum,
        quantized_vectors: Option<&QuantizedVectors>,
        max_bytes: usize,
    ) -> String {
        let layout = match quantized_vectors {
            Some(quantized) => quantized.get_quantized_vector_layout(),
            None => vectors.get_vector_layout(),
        };
        match layout {
            Err(err) => format!("storage exposes no fixed-layout byte view ({err})"),
            Ok(layout) if layout.align() > GATHER_ALIGN => format!(
                "vector alignment {} exceeds the buffer's {GATHER_ALIGN} bytes",
                layout.align(),
            ),
            Ok(layout) => {
                let padded_stride = layout.size().next_multiple_of(layout.align().max(1));
                format!(
                    "stride {} B against a {} MiB per-block cap, so blocks over ~{} points",
                    layout.size(),
                    max_bytes / (1024 * 1024),
                    max_bytes / padded_stride.max(1),
                )
            }
        }
    }

    /// Number of block points held.
    fn len(&self) -> usize {
        self.len
    }

    /// Encoded bytes of the block point with local id `local_id`.
    #[inline]
    fn get(&self, local_id: PointOffsetType) -> &[u8] {
        // The `u64` backing store rounds the buffer up to a multiple of 8, so an
        // out-of-range id can land in the padding and hand back zeroed bytes
        // rather than tripping the slice bounds check. Fail loudly instead.
        debug_assert!(
            (local_id as usize) < self.len,
            "block-local id {local_id} is past the {} vectors in this buffer",
            self.len,
        );
        let start = self.padded_stride * local_id as usize;
        &bytemuck::cast_slice::<u64, u8>(&self.data)[start..start + self.stride]
    }
}

/// Scorers composition:
///
/// ```plaintext
///                                                               Metric
///                                                              ┌─────────────┐
///                                                              │ - Cosine    │
///  FilteredScorer      RawScorer          QueryScorer          │ - Dot       │
/// ┌─────────────────┐ ┌───────────────┐   ┌────────────────┐ ┌─┤ - Euclidean │
/// │ RawScorer ◄─────┼─┤ QueryScorer ◄─┼───│ Metric ◄───────┼─┘ └─────────────┘
/// │                 │ └───────────────┘   │                │    - Vector Distance
/// │ ConditionChecker│  - Access patterns  │ Query  ◄───────┼─┐
/// │                 │                     │                │ │  Query
/// │ deleted_points  │                     │ TVectorStorage │ │ ┌──────────────────┐
/// │ deleted_vectors │                     └────────────────┘ └─┤ - RecoQuery      │
/// └─────────────────┘                                          │ - DiscoverQuery  │
///                                                              │ - ContextQuery   │
///                                                              └──────────────────┘
///                                                              - Scoring logic
///                                                              - Complex queries
/// ```
///
/// The `BatchFilteredSearcher` contains an array of `RawScorer`s, a common filter and certain parameters.
///
/// ```plaintext
/// BatchFilteredSearcher  RawScorer
///  ┌─────────────────┐  ┌───────────────┐
///  │ [RawScorer] ◄───┼──┤ QueryScorer ◄─┼── (ditto)
///  │                 │  └───────────────┘
///  │ ConditionChecker│
///  └─────────────────┘
/// ```
pub struct FilteredScorer<'a> {
    raw_scorer: Box<dyn RawScorer + 'a>,
    filters: ScorerFilters<'a>,
    /// Temporary buffer for scores.
    scores_buffer: Vec<ScoreType>,
}

pub struct ScorerFilters<'a> {
    filter_context: Option<OptimizedFilter<'a>>,
    deleted: NotDeletedChecker<'a>,
}

impl<'a> ScorerFilters<'a> {
    pub fn new(
        filter_context: Option<OptimizedFilter<'a>>,
        deleted: NotDeletedChecker<'a>,
    ) -> Self {
        ScorerFilters {
            filter_context,
            deleted,
        }
    }

    /// Return true if vector satisfies current search context for given point:
    /// exists, not deleted, and satisfies filter context.
    pub fn check_vector(&self, point_id: PointOffsetType) -> bool {
        self.deleted.check_infallible(point_id)
            && self
                .filter_context
                .as_ref()
                .is_none_or(|f| f.check_infallible(point_id))
    }
}

impl ConditionChecker for ScorerFilters<'_> {
    type Error = OperationError;

    fn check(&self, point_id: PointOffsetType) -> OperationResult<bool> {
        Ok(self.deleted.check(point_id)?
            && match &self.filter_context {
                Some(f) => f.check(point_id)?,
                None => true,
            })
    }

    fn check_infallible(&self, point_id: PointOffsetType) -> bool {
        self.check_vector(point_id)
    }

    #[inline]
    fn check_batched<K: CheckItem>(
        &self,
        ids: &mut [K],
        select: Select,
        rest: Rest,
    ) -> OperationResult<usize> {
        let Self {
            filter_context,
            deleted,
        } = self;
        match select {
            Select::Matches => {
                let n = deleted.check_batched(ids, Select::Matches, rest)?;
                match filter_context {
                    Some(f) => f.check_batched(&mut ids[..n], Select::Matches, rest),
                    None => Ok(n),
                }
            }
            Select::NonMatches => {
                let deleted_rest = rest.keep_if(filter_context.is_some());
                let mut f = deleted.check_batched(ids, Select::NonMatches, deleted_rest)?;
                if let Some(filter) = filter_context {
                    f += filter.check_batched(&mut ids[f..], Select::NonMatches, rest)?;
                }
                Ok(f)
            }
        }
    }
}

pub struct FilteredBytesScorer<'a> {
    scorer_bytes: &'a dyn QueryScorerBytes,
    filters: &'a ScorerFilters<'a>,
}

impl<'a> FilteredBytesScorer<'a> {
    pub fn score_points(
        &self,
        points: &mut Vec<(PointOffsetType, &[u8])>,
        limit: usize,
    ) -> impl Iterator<Item = ScoredPointOffset> {
        points.retain(|(point_id, _)| self.filters.check_vector(*point_id));
        if limit != 0 {
            points.truncate(limit);
        }

        points.iter().map(|&(idx, bytes)| ScoredPointOffset {
            idx,
            score: self.scorer_bytes.score_bytes(bytes),
        })
    }
}

impl<'a> FilteredScorer<'a> {
    /// Create a new filtered scorer.
    ///
    /// If present, `quantized_vectors` will be used for scoring, otherwise `vectors` will be used.
    pub fn new<V, Q>(
        query: QueryVector,
        vectors: &'a V,
        quantized_vectors: Option<&'a Q>,
        filter_context: Option<OptimizedFilter<'a>>,
        point_deleted: &'a BitSlice,
        hardware_counter: HardwareCounterCell,
    ) -> OperationResult<Self>
    where
        V: VectorStorageRead + RawScorerBuilder,
        Q: QuantizedVectorsRead,
    {
        let raw_scorer = match quantized_vectors {
            Some(quantized_vectors) => quantized_vectors.raw_scorer(query, hardware_counter)?,
            None => vectors.build_raw_scorer(query, hardware_counter)?,
        };
        Ok(FilteredScorer {
            raw_scorer,
            filters: ScorerFilters::new(filter_context, vectors.not_deleted_checker(point_deleted)),
            scores_buffer: Vec::new(),
        })
    }

    pub fn new_internal<V, Q>(
        point_id: PointOffsetType,
        vectors: &'a V,
        quantized_vectors: Option<&'a Q>,
        filter_context: Option<OptimizedFilter<'a>>,
        point_deleted: &'a BitSlice,
        hardware_counter: HardwareCounterCell,
    ) -> OperationResult<Self>
    where
        V: VectorStorageRead + RawScorerBuilder,
        Q: QuantizedVectorsRead,
    {
        let raw_scorer =
            Self::internal_raw_scorer(point_id, vectors, quantized_vectors, hardware_counter)?;
        Ok(FilteredScorer {
            raw_scorer,
            filters: ScorerFilters::new(filter_context, vectors.not_deleted_checker(point_deleted)),
            scores_buffer: Vec::new(),
        })
    }

    /// Score a payload-block subgraph whose points are numbered `0..to_global.len()`
    #[allow(clippy::too_many_arguments)]
    pub fn new_block_scorer(
        global_point_id: PointOffsetType,
        to_global: &'a [PointOffsetType],
        vectors: &'a VectorStorageEnum,
        quantized_vectors: Option<&'a QuantizedVectors>,
        block_vectors: Option<&'a BlockVectors>,
        internal_from_block: bool,
        block_deleted: &'a BitSlice,
        hardware_counter: HardwareCounterCell,
    ) -> OperationResult<Self> {
        let inner = Self::internal_raw_scorer(
            global_point_id,
            vectors,
            quantized_vectors,
            hardware_counter,
        )?;

        // Falling back keeps scoring correct either way, so this stays a debug
        // assert - but a copy sized to a different number of points than the
        // block is a caller bug, not a storage that cannot support the copy,
        // and should not quietly look like one.
        debug_assert!(
            block_vectors.is_none_or(|block| block.len() == to_global.len()),
            "gathered block holds {:?} vectors for {} block points",
            block_vectors.map(BlockVectors::len),
            to_global.len(),
        );

        // Without a byte-level entry point there is nothing to score the copy
        // with, and a copy covering a different number of points than the block
        // is not this block's.
        debug_assert_eq!(
            block_deleted.len(),
            to_global.len(),
            "block deleted-flags must have one bit per block point: a short slice \
             reads as deleted and silently costs that point its links",
        );
        let usable = block_vectors
            .filter(|block| block.len() == to_global.len())
            .filter(|_| inner.scorer_bytes().is_some());

        let raw_scorer: Box<dyn RawScorer + 'a> = match usable {
            Some(block) => Box::new(GatherRawScorer {
                inner,
                to_global,
                block,
                internal_from_block,
            }),
            None => Box::new(RemappedRawScorer { inner, to_global }),
        };

        Ok(FilteredScorer {
            raw_scorer,
            filters: ScorerFilters::new(
                None,
                NotDeletedChecker {
                    point_deleted: block_deleted,
                    vec_deleted: block_deleted,
                },
            ),
            scores_buffer: Vec::new(),
        })
    }

    /// Raw scorer that scores against the vector stored under `point_id`.
    fn internal_raw_scorer<V, Q>(
        point_id: PointOffsetType,
        vectors: &'a V,
        quantized_vectors: Option<&'a Q>,
        hardware_counter: HardwareCounterCell,
    ) -> OperationResult<Box<dyn RawScorer + 'a>>
    where
        V: VectorStorageRead + RawScorerBuilder,
        Q: QuantizedVectorsRead,
    {
        // This is a fallback function, which is used if quantized vector storage
        // is not capable of reconstructing the query vector.
        let original_query_fn = || {
            let query = vectors.get_vector::<Random>(point_id);
            let query: QueryVector = query.as_vec_ref().into();
            query
        };
        match quantized_vectors {
            Some(quantized_vectors) => quantized_vectors
                .raw_internal_scorer(point_id, hardware_counter)
                .or_else(|InternalScorerUnsupported(hardware_counter)| {
                    quantized_vectors.raw_scorer(original_query_fn(), hardware_counter)
                }),
            None => {
                let query = original_query_fn();
                vectors.build_raw_scorer(query, hardware_counter)
            }
        }
    }

    /// Create a new filtered scorer for testing purposes.
    ///
    /// # Panics
    ///
    /// Panics if [`new_raw_scorer`] fails.
    #[cfg(feature = "testing")]
    pub fn new_for_test(
        vector: QueryVector,
        vector_storage: &'a VectorStorageEnum,
        point_deleted: &'a BitSlice,
    ) -> Self {
        FilteredScorer {
            raw_scorer: new_raw_scorer(vector, vector_storage, HardwareCounterCell::new()).unwrap(),
            filters: ScorerFilters::new(None, vector_storage.not_deleted_checker(point_deleted)),
            scores_buffer: Vec::new(),
        }
    }

    pub fn raw_scorer(&self) -> &dyn RawScorer {
        self.raw_scorer.as_ref()
    }

    pub fn filters(&self) -> &ScorerFilters<'a> {
        &self.filters
    }

    /// Return [`FilteredBytesScorer`] if the underlying scorer supports it.
    pub fn scorer_bytes(&self) -> Option<FilteredBytesScorer<'_>> {
        Some(FilteredBytesScorer {
            scorer_bytes: self.raw_scorer.scorer_bytes()?,
            filters: &self.filters,
        })
    }

    /// Filters and calculates scores for the given slice of points IDs.
    ///
    /// For performance reasons this method mutates `point_ids`.
    ///
    /// # Arguments
    ///
    /// * `point_ids` - list of points to score.
    ///   **Warning**: This input will be wrecked during the execution.
    /// * `limit` - limits the number of points to process after filtering.
    ///   `0` means no limit.
    #[inline(always)]
    pub fn score_points(
        &mut self,
        point_ids: &mut Vec<PointOffsetType>,
        limit: usize,
    ) -> impl Iterator<Item = ScoredPointOffset> {
        let mut n = self
            .filters
            .check_batched(point_ids, Select::Matches, Rest::Discard)
            .unwrap_or(0 /* TODO(uio): propagate error */);
        if limit != 0 {
            n = n.min(limit);
        }
        point_ids.truncate(n);

        self.score_points_unfiltered(point_ids)
    }

    pub fn score_points_unfiltered(
        &mut self,
        point_ids: &[PointOffsetType],
    ) -> impl Iterator<Item = ScoredPointOffset> {
        if self.scores_buffer.len() < point_ids.len() {
            self.scores_buffer.resize(point_ids.len(), 0.0);
        }

        self.raw_scorer
            .score_points(point_ids, &mut self.scores_buffer[..point_ids.len()]);

        std::iter::zip(point_ids, &self.scores_buffer)
            .map(|(&idx, &score)| ScoredPointOffset { idx, score })
    }

    pub fn score_point(&self, point_id: PointOffsetType) -> ScoreType {
        self.raw_scorer.score_point(point_id)
    }

    pub fn score_internal(&self, point_a: PointOffsetType, point_b: PointOffsetType) -> ScoreType {
        self.raw_scorer.score_internal(point_a, point_b)
    }
}

/// Drives a scorer built over segment-wide ids with the local ids of a
/// payload-block subgraph.
struct RemappedRawScorer<'a> {
    inner: Box<dyn RawScorer + 'a>,
    to_global: &'a [PointOffsetType],
}

/// Score block points through a scorer built over segment-wide ids, translating
/// local ids to global ones as it goes.
fn remapped_score_points(
    inner: &dyn RawScorer,
    to_global: &[PointOffsetType],
    points: &[PointOffsetType],
    scores: &mut [ScoreType],
) {
    // The `zip` below would silently score a prefix and leave the rest of
    // `scores` untouched. `RawScorerImpl::score_points` enforces this in
    // release, and a wrapper around it is not the place to weaken the contract.
    assert_eq!(points.len(), scores.len());
    // The trait takes `&self`, so there is nowhere to keep a growable
    // scratch buffer. Translate through the stack instead, in the same
    // batches the storage reads in anyway.
    let mut translated = [0; VECTOR_READ_BATCH_SIZE];
    for (points, scores) in points
        .chunks(VECTOR_READ_BATCH_SIZE)
        .zip(scores.chunks_mut(VECTOR_READ_BATCH_SIZE))
    {
        let translated = &mut translated[..points.len()];
        for (global, &local) in translated.iter_mut().zip(points) {
            *global = to_global[local as usize];
        }
        inner.score_points(translated, scores);
    }
}

impl RawScorer for RemappedRawScorer<'_> {
    fn score_points(&self, points: &[PointOffsetType], scores: &mut [ScoreType]) {
        remapped_score_points(self.inner.as_ref(), self.to_global, points, scores);
    }

    fn score_point(&self, point: PointOffsetType) -> ScoreType {
        self.inner.score_point(self.to_global[point as usize])
    }

    fn score_internal(&self, point_a: PointOffsetType, point_b: PointOffsetType) -> ScoreType {
        self.inner.score_internal(
            self.to_global[point_a as usize],
            self.to_global[point_b as usize],
        )
    }

    fn scorer_bytes(&self) -> Option<&dyn QueryScorerBytes> {
        None
    }
}

/// Like [`RemappedRawScorer`], but scores candidates against a copy of the
/// block's vectors instead of the segment-wide storage.
struct GatherRawScorer<'a> {
    inner: Box<dyn RawScorer + 'a>,
    to_global: &'a [PointOffsetType],
    block: &'a BlockVectors,
    /// Serve the link-selection heuristic's stored-vs-stored scoring from the
    /// copy too, where the underlying scorer offers a byte-level symmetric
    /// kernel. Off falls back to scoring by global id.
    internal_from_block: bool,
}

impl RawScorer for GatherRawScorer<'_> {
    fn score_points(&self, points: &[PointOffsetType], scores: &mut [ScoreType]) {
        // Same contract as `RawScorerImpl::score_points`, enforced the same way:
        // the `zip` below would otherwise score a prefix and say nothing.
        assert_eq!(points.len(), scores.len());
        let Some(bytes_scorer) = self.inner.scorer_bytes() else {
            // Ruled out when this scorer was built. Score against the storage
            // rather than report something else.
            debug_assert!(false, "gather scorer built over a scorer without bytes");
            return remapped_score_points(self.inner.as_ref(), self.to_global, points, scores);
        };
        for (&local, score) in points.iter().zip(scores.iter_mut()) {
            *score = bytes_scorer.score_bytes(self.block.get(local));
        }
    }

    fn score_point(&self, point: PointOffsetType) -> ScoreType {
        match self.inner.scorer_bytes() {
            Some(bytes_scorer) => bytes_scorer.score_bytes(self.block.get(point)),
            None => self.inner.score_point(self.to_global[point as usize]),
        }
    }

    fn score_internal(&self, point_a: PointOffsetType, point_b: PointOffsetType) -> ScoreType {
        if self.internal_from_block
            && let Some(bytes_scorer) = self.inner.scorer_bytes()
            && let Some(score) =
                bytes_scorer.score_internal_bytes(self.block.get(point_a), self.block.get(point_b))
        {
            return score;
        }
        self.inner.score_internal(
            self.to_global[point_a as usize],
            self.to_global[point_b as usize],
        )
    }

    fn scorer_bytes(&self) -> Option<&dyn QueryScorerBytes> {
        // Same reason as `RemappedRawScorer`: the ids handed out would be
        // segment-wide, and the build-time search path never asks.
        None
    }
}

// We keep each scorer with its queue to reduce allocations and improve data locality.
struct BatchSearch<'a> {
    raw_scorer: Box<dyn RawScorer + 'a>,
    pq: FixedLengthPriorityQueue<ScoredPointOffset>,
}

pub struct BatchFilteredSearcher<'a> {
    scorer_batch: SmallVec<[BatchSearch<'a>; 1]>,
    filters: ScorerFilters<'a>,
}

impl<'a> BatchFilteredSearcher<'a> {
    /// Create a new batch filtered searcher.
    ///
    /// If present, `quantized_vectors` will be used for scoring, otherwise `vectors` will be used.
    pub fn new<V, Q>(
        queries: &[&QueryVector],
        vectors: &'a V,
        quantized_vectors: Option<&'a Q>,
        filter_context: Option<OptimizedFilter<'a>>,
        top: usize,
        point_deleted: &'a BitSlice,
        hardware_counter: HardwareCounterCell,
    ) -> OperationResult<Self>
    where
        V: VectorStorageRead + RawScorerBuilder,
        Q: QuantizedVectorsRead,
    {
        let scorer_batch = queries
            .iter()
            .map(|&query| {
                let query = query.to_owned();
                let hardware_counter = hardware_counter.fork();
                let raw_scorer = match quantized_vectors {
                    Some(quantized_vectors) => {
                        quantized_vectors.raw_scorer(query, hardware_counter)
                    }
                    None => vectors.build_raw_scorer(query, hardware_counter),
                };
                let pq = FixedLengthPriorityQueue::new(top);
                raw_scorer.map(|raw_scorer| BatchSearch { raw_scorer, pq })
            })
            .collect::<Result<_, _>>()?;
        let filters =
            ScorerFilters::new(filter_context, vectors.not_deleted_checker(point_deleted));
        Ok(Self {
            scorer_batch,
            filters,
        })
    }

    /// Create a new batched filtered searcher for testing purposes.
    ///
    /// # Panics
    ///
    /// Panics if [`new_raw_scorer`] fails.
    #[cfg(feature = "testing")]
    pub fn new_for_test(
        vectors: &[QueryVector],
        vector_storage: &'a VectorStorageEnum,
        point_deleted: &'a BitSlice,
        top: usize,
    ) -> Self {
        let scorer_batch = vectors
            .iter()
            .map(|vector| {
                let raw_scorer = new_raw_scorer(
                    vector.to_owned(),
                    vector_storage,
                    HardwareCounterCell::new(),
                )
                .unwrap();
                BatchSearch {
                    raw_scorer,
                    pq: FixedLengthPriorityQueue::new(top),
                }
            })
            .collect();
        Self {
            scorer_batch,
            filters: ScorerFilters::new(None, vector_storage.not_deleted_checker(point_deleted)),
        }
    }

    /// Iterator over every internal point ID that isn't soft-deleted in this
    /// searcher's `point_deleted` bitslice.
    ///
    /// Does not apply deferred-point filtering — wrap with
    /// `PointMappingsRefEnum::filter_deferred_and_deleted` (or compose otherwise) before
    /// passing to [`Self::peek_top_iter`] when deferred awareness is needed.
    ///
    /// The returned iterator borrows the underlying bitslice (lifetime `'a`),
    /// independent of `&self`, so it can be composed and then passed into
    /// `peek_top_iter(self, ...)` which consumes the searcher.
    pub fn iter_not_deleted(&self) -> impl Iterator<Item = PointOffsetType> + 'a {
        self.filters
            .deleted
            .point_deleted
            .iter_zeros()
            .map(|p| p as PointOffsetType)
    }

    /// Score every non-deleted point without deferred filtering.
    ///
    /// Production paths compose `iter_not_deleted` with
    /// `PointMappingsRefEnum::filter_deferred_and_deleted` and call
    /// [`Self::peek_top_iter`] directly.
    #[cfg(feature = "testing")]
    pub fn peek_top_all(
        self,
        is_stopped: &AtomicBool,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        let iter = self.iter_not_deleted();
        self.peek_top_iter(iter, is_stopped)
    }

    /// This function expects deferred points to be already filtered from the iterator.
    pub fn peek_top_iter(
        mut self,
        mut points: impl Iterator<Item = PointOffsetType>,
        is_stopped: &AtomicBool,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        // Reuse the same buffer for all chunks, to avoid reallocation
        let mut chunk = [0; VECTOR_READ_BATCH_SIZE];
        let mut scores_buffer = [0.0; VECTOR_READ_BATCH_SIZE];

        loop {
            check_process_stopped(is_stopped)?;

            let mut chunk_size = 0;
            for point_id in &mut points {
                check_process_stopped(is_stopped)?;

                if !self.filters.check_vector(point_id) {
                    continue;
                }
                chunk[chunk_size] = point_id;
                chunk_size += 1;
                if chunk_size == VECTOR_READ_BATCH_SIZE {
                    break;
                }
            }

            if chunk_size == 0 {
                break;
            }

            // Switching the loops improves batching performance, but slightly degrades single-query performance.
            for BatchSearch { raw_scorer, pq } in &mut self.scorer_batch {
                raw_scorer.score_points(&chunk[..chunk_size], &mut scores_buffer[..chunk_size]);

                for i in 0..chunk_size {
                    pq.push(ScoredPointOffset {
                        idx: chunk[i],
                        score: scores_buffer[i],
                    });
                }
            }
        }

        let results = self
            .scorer_batch
            .into_iter()
            .map(|BatchSearch { pq, .. }| pq.into_sorted_vec())
            .collect();
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use common::bitvec::BitVec;
    use rand::SeedableRng as _;
    use rand::rngs::StdRng;

    use super::*;
    use crate::data_types::vectors::{VectorElementType, VectorRef};
    use crate::fixtures::index_fixtures::random_vector;
    use crate::types::{
        Distance, Memory, MultiVectorConfig, QuantizationConfig, TurboQuantBitSize,
        TurboQuantQuantizationConfig, TurboQuantization,
    };
    use crate::vector_storage::VectorStorage;
    use crate::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;
    use crate::vector_storage::multi_dense::volatile_multi_dense_vector_storage::new_volatile_multi_dense_vector_storage;
    use crate::vector_storage::quantized::quantized_vectors::QuantizedVectorsStorageType;
    use crate::vector_storage::sparse::volatile_sparse_vector_storage::new_volatile_sparse_vector_storage;

    /// A dense storage with TurboQuant 4-bit quantization over it. At `dim`
    /// values not covered by [`TurboQuantizer::padded_dim`]'s packing, the
    /// quantized stride is not a whole number of alignment units, which is
    /// exactly the case the gather buffer's padded stride exists for.
    fn tq_fixture(
        dim: usize,
        num_vectors: usize,
    ) -> (VectorStorageEnum, QuantizedVectors, tempfile::TempDir) {
        let mut rng = StdRng::seed_from_u64(42);
        let mut storage = new_volatile_dense_vector_storage(dim, Distance::Dot);
        let hw_counter = HardwareCounterCell::new();
        for offset in 0..num_vectors as PointOffsetType {
            let vector =
                Distance::Dot.preprocess_vector::<VectorElementType>(random_vector(&mut rng, dim));
            storage
                .insert_vector(offset, VectorRef::from(&vector), &hw_counter)
                .unwrap();
        }

        let config = QuantizationConfig::Turbo(TurboQuantization {
            turbo: TurboQuantQuantizationConfig {
                always_ram: None,
                bits: Some(TurboQuantBitSize::Bits4),
                memory: Some(Memory::Pinned),
            },
        });
        let dir = tempfile::tempdir().unwrap();
        let quantized = QuantizedVectors::create(
            &storage,
            &config,
            QuantizedVectorsStorageType::Immutable,
            dir.path(),
            1,
            &AtomicBool::new(false),
        )
        .unwrap();
        (storage, quantized, dir)
    }

    #[test]
    fn gather_stride_padding_holds_its_invariant() {
        // The padded stride exists because a layout may report a size that is not
        // a whole number of alignment units, which would leave every second vector
        // in the buffer misaligned - and the unquantized byte scorer casts these
        // bytes back to its element type.
        //
        // No layout in this version can trigger it: dense storages build their
        // layout with `Layout::array`, whose size is always a multiple of its
        // align, and TurboQuant reports `align == 1` for every dim. It was live in
        // 1.18, where TurboQuant reported a 13-byte size against 4-byte alignment.
        // So this asserts the invariant rather than the padded case, and the
        // padding stays as the guard for a layout that reports one again.
        let (storage, quantized, _dir) = tq_fixture(10, 64);
        let layout = quantized.get_quantized_vector_layout().unwrap();

        // Every other point, so local and global ids differ.
        let points: Vec<PointOffsetType> = (0..64).step_by(2).collect();
        let block =
            BlockVectors::try_gather(&points, &storage, Some(&quantized), MAX_BLOCK_GATHER_BYTES)
                .expect("a self-aligned stride must gather");

        assert_eq!(
            block.padded_stride,
            layout.size().next_multiple_of(layout.align()),
        );
        assert!(block.padded_stride >= block.stride);
        for (local, &global) in points.iter().enumerate() {
            let bytes = block.get(local as PointOffsetType);
            assert_eq!(
                bytes,
                &*quantized.get_quantized_vector(global),
                "gathered bytes of local {local} differ from storage",
            );
            assert_eq!(bytes.as_ptr().addr() % layout.align(), 0);
        }
    }

    #[test]
    fn gather_padded_scores_match_storage() {
        let (storage, quantized, _dir) = tq_fixture(10, 64);

        // The gather only engages when the inner scorer scores raw bytes; if
        // TurboQuant ever stops offering that, the equality below would just
        // compare the fallback path against itself. Same for the symmetric
        // kernel and the internal-scoring comparison.
        let inner = FilteredScorer::internal_raw_scorer(
            0,
            &storage,
            Some(&quantized),
            HardwareCounterCell::new(),
        )
        .unwrap();
        let inner_bytes = inner
            .scorer_bytes()
            .expect("TQ scorer lost its byte entry point; this test no longer covers the gather");

        let points: Vec<PointOffsetType> = (1..64).step_by(3).collect();
        let block =
            BlockVectors::try_gather(&points, &storage, Some(&quantized), MAX_BLOCK_GATHER_BYTES)
                .unwrap();

        assert!(
            inner_bytes
                .score_internal_bytes(block.get(0), block.get(1))
                .is_some(),
            "TQ scorer lost its symmetric byte kernel; the internal-scoring \
             comparison below no longer covers the gathered path",
        );

        let block_deleted = BitVec::repeat(false, points.len());
        let scorer = |block_vectors, internal_from_block| {
            FilteredScorer::new_block_scorer(
                points[0],
                &points,
                &storage,
                Some(&quantized),
                block_vectors,
                internal_from_block,
                &block_deleted,
                HardwareCounterCell::new(),
            )
            .unwrap()
        };
        let gathered = scorer(Some(&block), true);
        let gathered_no_internal = scorer(Some(&block), false);
        let ungathered = scorer(None, false);

        for local in 0..points.len() as PointOffsetType {
            assert_eq!(
                gathered.score_point(local).to_bits(),
                ungathered.score_point(local).to_bits(),
                "scores diverge at local {local}",
            );
        }

        for a in 0..points.len() as PointOffsetType {
            for b in 0..points.len() as PointOffsetType {
                let from_block = gathered.score_internal(a, b);
                assert_eq!(
                    from_block.to_bits(),
                    ungathered.score_internal(a, b).to_bits(),
                    "internal scores diverge from storage at locals {a}, {b}",
                );
                assert_eq!(
                    from_block.to_bits(),
                    gathered_no_internal.score_internal(a, b).to_bits(),
                    "internal scores diverge from the id fallback at locals {a}, {b}",
                );
            }
        }
    }

    #[test]
    fn gather_aligned_stride_stays_unpadded() {
        // f32 dense: stride dim * 4 with 4-byte alignment - already aligned,
        // so padding must not change the layout the equality tests pinned.
        let mut rng = StdRng::seed_from_u64(42);
        let dim = 8;
        let mut storage = new_volatile_dense_vector_storage(dim, Distance::Dot);
        let hw_counter = HardwareCounterCell::new();
        for offset in 0..32 {
            let vector = random_vector(&mut rng, dim);
            storage
                .insert_vector(offset, VectorRef::from(&vector), &hw_counter)
                .unwrap();
        }

        let points: Vec<PointOffsetType> = (0..32).step_by(2).collect();
        let block = BlockVectors::try_gather(&points, &storage, None, MAX_BLOCK_GATHER_BYTES)
            .expect("aligned dense storage must gather");

        assert_eq!(block.padded_stride, block.stride);
        for (local, &global) in points.iter().enumerate() {
            let matches = storage
                .with_vector_bytes_opt::<Random, _>(global, |src| {
                    src == block.get(local as PointOffsetType)
                })
                .unwrap()
                .expect("storage must expose bytes for a gathered point");
            assert!(
                matches,
                "gathered bytes of local {local} differ from storage"
            );
        }
    }
    /// A dense storage with no quantization, for the size arithmetic.
    fn dense_fixture(dim: usize, num_vectors: PointOffsetType) -> VectorStorageEnum {
        let mut rng = StdRng::seed_from_u64(42);
        let mut storage = new_volatile_dense_vector_storage(dim, Distance::Dot);
        let hw_counter = HardwareCounterCell::new();
        for offset in 0..num_vectors {
            let vector = random_vector(&mut rng, dim);
            storage
                .insert_vector(offset, VectorRef::from(&vector), &hw_counter)
                .unwrap();
        }
        storage
    }

    #[test]
    fn gather_budget_declines_past_its_allowance() {
        let budget = GatherBudget::new(3 * 1024);
        let _a = budget.try_reserve(1024).expect("first fits");
        let _b = budget.try_reserve(1024).expect("second fits");
        let _c = budget.try_reserve(1024).expect("third fits exactly");
        assert!(
            budget.try_reserve(1).is_none(),
            "a spent allowance must decline rather than overdraw",
        );
        assert_eq!(budget.remaining.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn gather_budget_declines_a_request_larger_than_the_whole_allowance() {
        let budget = GatherBudget::new(1024);
        assert!(budget.try_reserve(1025).is_none());
        // A declined ask must consume nothing, or one oversized block would poison
        // the allowance for every block behind it.
        assert_eq!(budget.remaining.load(Ordering::Relaxed), 1024);
        let _held = budget
            .try_reserve(1024)
            .expect("the allowance is untouched");
    }

    #[test]
    fn gather_budget_releases_exactly_what_it_reserved_on_drop() {
        let budget = GatherBudget::new(1024);
        let held = budget.try_reserve(768).expect("fits");
        assert_eq!(budget.remaining.load(Ordering::Relaxed), 256);
        assert!(budget.try_reserve(768).is_none());
        drop(held);
        assert_eq!(
            budget.remaining.load(Ordering::Relaxed),
            1024,
            "drop must hand back the reserved amount, no more and no less",
        );
        let _reused = budget
            .try_reserve(1024)
            .expect("the allowance came back whole");
    }

    /// The property the stage's memory bound rests on: however many threads race,
    /// the number of live buffers never exceeds what the allowance pays for, and
    /// the allowance is whole again once they finish.
    ///
    /// This is the concurrent reserve path, which the segment-level block tests
    /// cannot reach - their fixture is small enough that the budget never binds.
    #[test]
    fn gather_budget_never_oversubscribes_under_contention() {
        const CHUNK: usize = 4 * 1024;
        const SLOTS: usize = 4;
        const THREADS: usize = 16;
        const ROUNDS: usize = 1_000;

        let budget = GatherBudget::new(SLOTS * CHUNK);
        let live = AtomicUsize::new(0);
        let granted = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    for _ in 0..ROUNDS {
                        let Some(reservation) = budget.try_reserve(CHUNK) else {
                            continue;
                        };
                        let held = live.fetch_add(1, Ordering::AcqRel) + 1;
                        assert!(
                            held <= SLOTS,
                            "{held} buffers live against an allowance for {SLOTS}",
                        );
                        granted.fetch_add(1, Ordering::Relaxed);
                        // Decremented before the reservation is released, so this
                        // can only ever undercount what is live - it cannot fail
                        // spuriously on another thread's release.
                        live.fetch_sub(1, Ordering::AcqRel);
                        drop(reservation);
                    }
                });
            }
        });

        assert!(
            granted.load(Ordering::Relaxed) > 0,
            "no thread was ever granted an allowance, so nothing was exercised",
        );
        assert_eq!(
            budget.remaining.load(Ordering::Relaxed),
            SLOTS * CHUNK,
            "the allowance leaked: reservations released less than they took",
        );
    }

    /// A caller reserves off `gather_size` and then copies with `try_gather`, so
    /// the two must agree to the byte or the budget accounts for the wrong thing.
    #[test]
    fn gather_size_matches_what_try_gather_allocates() {
        let (storage, quantized, _dir) = tq_fixture(48, 64);
        let points: Vec<PointOffsetType> = (0..64).collect();

        let wanted = BlockVectors::gather_size(points.len(), &storage, Some(&quantized))
            .expect("quantized storage exposes a fixed layout");
        let block = BlockVectors::try_gather(&points, &storage, Some(&quantized), wanted)
            .expect("a copy that fits exactly what gather_size asked for");
        // The buffer itself, not just the vectors in it: the `u64` store rounds
        // up, and the reservation has to cover the rounding too.
        assert_eq!(block.data.len() * size_of::<u64>(), wanted);
        assert!(block.padded_stride * block.len() <= wanted);

        // One byte short must decline, not allocate past what was reserved.
        assert!(
            BlockVectors::try_gather(&points, &storage, Some(&quantized), wanted - 1).is_none(),
            "try_gather allocated more than the caller reserved",
        );
    }

    #[test]
    fn gather_size_declines_an_empty_block() {
        let dim = 8;
        let storage = dense_fixture(dim, 32);
        assert_eq!(
            BlockVectors::gather_size(4, &storage, None),
            Some(4 * dim * size_of::<VectorElementType>()),
        );
        // `try_reserve(0)` would succeed, so an empty block must be declined here
        // or it would hold a reservation for a copy that then refuses to happen.
        assert!(
            BlockVectors::gather_size(0, &storage, None).is_none(),
            "an empty block must not reserve an allowance the copy would refuse",
        );
        assert!(BlockVectors::try_gather(&[], &storage, None, MAX_BLOCK_GATHER_BYTES).is_none());
    }
    /// `remapped_score_points` is the batch path of the fallback scorer, and
    /// `score_points` is the only way in - `score_point` translates its one id
    /// inline and never reaches it. Nothing exercised it, so replacing
    /// `to_global[local]` with `to_global[0]` kept the whole suite green.
    ///
    /// This is the path taken whenever the copy is declined, which the per-block
    /// cap now makes reachable in production for a large enough block.
    ///
    /// The batch deliberately spans more than one `VECTOR_READ_BATCH_SIZE` chunk
    /// and is not a multiple of it: the translation reuses one stack buffer across
    /// chunks and reslices it per chunk, so the final short chunk is where an
    /// off-by-one lives.
    #[test]
    fn remapped_batch_scores_match_scoring_one_at_a_time() {
        const BLOCK_POINTS: usize = 150;
        const _: () = assert!(BLOCK_POINTS > 2 * VECTOR_READ_BATCH_SIZE);
        const _: () = assert!(!BLOCK_POINTS.is_multiple_of(VECTOR_READ_BATCH_SIZE));

        let (storage, quantized, _dir) = tq_fixture(16, BLOCK_POINTS * 2);
        // Every other vector, so `to_global` is not the identity: a translation
        // that ignored the local id would still land on a valid vector and score
        // something plausible.
        let points: Vec<PointOffsetType> = (0..BLOCK_POINTS as PointOffsetType * 2)
            .step_by(2)
            .collect();
        assert_eq!(points.len(), BLOCK_POINTS);

        let block_deleted = BitVec::repeat(false, points.len());
        let block =
            BlockVectors::try_gather(&points, &storage, Some(&quantized), MAX_BLOCK_GATHER_BYTES)
                .expect("fixture must gather, or the gathered half compares nothing");
        let make = |block_vectors| {
            FilteredScorer::new_block_scorer(
                points[0],
                &points,
                &storage,
                Some(&quantized),
                block_vectors,
                true,
                &block_deleted,
                HardwareCounterCell::new(),
            )
            .unwrap()
        };

        let locals: Vec<PointOffsetType> = (0..BLOCK_POINTS as PointOffsetType).collect();
        for (label, mut scorer) in [("remapped", make(None)), ("gathered", make(Some(&block)))] {
            let batched: Vec<ScoreType> = scorer
                .score_points_unfiltered(&locals)
                .map(|scored| scored.score)
                .collect();
            assert_eq!(batched.len(), BLOCK_POINTS, "{label}");
            // Without distinct scores, a translation that returns the same vector
            // for every local id would be indistinguishable from a correct one.
            assert!(
                batched
                    .iter()
                    .any(|score| score.to_bits() != batched[0].to_bits()),
                "{label}: every score identical, so this test proves nothing",
            );
            for (&local, &score) in locals.iter().zip(&batched) {
                assert_eq!(
                    score.to_bits(),
                    scorer.score_point(local).to_bits(),
                    "{label}: batch and single-point scores diverge at local {local}",
                );
            }
        }
    }

    /// The point ceiling this reports has to be the cutoff the gather actually
    /// applies. It is the only thing an operator sees when a build silently stopped
    /// copying, so a diagnostic quoting a threshold the code does not use is worse
    /// than no diagnostic.
    #[test]
    fn gather_constraints_report_the_real_point_ceiling() {
        let dim = 8;
        let storage = dense_fixture(dim, 32);
        let stride = dim * size_of::<VectorElementType>();

        let text = BlockVectors::gather_constraints(&storage, None, MAX_BLOCK_GATHER_BYTES);
        let ceiling = MAX_BLOCK_GATHER_BYTES / stride;
        assert!(
            text.contains(&format!("~{ceiling} points")),
            "expected a ceiling of {ceiling} points in: {text}",
        );
        assert!(
            text.contains(&format!("stride {stride} B")),
            "expected a stride of {stride} B in: {text}",
        );

        // And that number is where the gather really turns over.
        assert!(
            BlockVectors::gather_size(ceiling, &storage, None).unwrap() <= MAX_BLOCK_GATHER_BYTES,
        );
        assert!(
            BlockVectors::gather_size(ceiling + 1, &storage, None).unwrap()
                > MAX_BLOCK_GATHER_BYTES,
        );
    }

    /// Storages with no fixed-layout byte view are the permanent, explainable
    /// reason a build never gathers, and the reason `gather_size` declines them is
    /// the same `Err` this message is built from.
    #[test]
    fn gather_constraints_name_storages_with_no_byte_layout() {
        let multi =
            new_volatile_multi_dense_vector_storage(4, Distance::Dot, MultiVectorConfig::default());
        for (label, storage) in [
            ("sparse", new_volatile_sparse_vector_storage()),
            ("multivector", multi),
        ] {
            let text = BlockVectors::gather_constraints(&storage, None, MAX_BLOCK_GATHER_BYTES);
            assert!(
                text.contains("no fixed-layout byte view"),
                "{label}: {text}",
            );
            // The witness counter's honesty depends on these two agreeing with the
            // message: neither may reserve an allowance nor produce a buffer.
            assert!(
                BlockVectors::gather_size(16, &storage, None).is_none(),
                "{label} reported a gather size",
            );
            assert!(
                BlockVectors::try_gather(&[0], &storage, None, MAX_BLOCK_GATHER_BYTES).is_none(),
                "{label} produced a buffer",
            );
        }
    }

    /// The alignment arm of `gather_constraints` - and the stride padding it
    /// explains - cannot be reached by any storage in the tree: every layout that
    /// exposes bytes reports an alignment the `u64` buffer already satisfies.
    ///
    /// Pinned rather than covered. If this ever fails, the padding path and that
    /// arm of the message both come alive and want tests of their own.
    #[test]
    fn gathered_storages_never_out_align_the_buffer() {
        let dense = dense_fixture(8, 8);
        let dense_layout = dense.get_vector_layout().expect("dense exposes bytes");
        assert!(dense_layout.align() <= GATHER_ALIGN, "{dense_layout:?}");

        let (storage, quantized, _dir) = tq_fixture(48, 8);
        let _ = storage;
        let quantized_layout = quantized
            .get_quantized_vector_layout()
            .expect("TurboQuant exposes bytes");
        assert!(
            quantized_layout.align() <= GATHER_ALIGN,
            "{quantized_layout:?}"
        );
    }
}
