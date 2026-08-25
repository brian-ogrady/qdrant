use std::collections::{HashMap, HashSet};
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use segment::common::BYTES_IN_KB;
use segment::data_types::modifier::Modifier;
use segment::index::sparse_index::sparse_index_config::{SparseIndexConfig, SparseIndexType};
use segment::types::{
    Distance, HnswConfig, Indexes, Memory, MultiVectorConfig, PayloadStorageType,
    QuantizationConfig, SegmentConfig, SparseVectorDataConfig, SparseVectorStorageType,
    VectorDataConfig, VectorNameBuf, VectorStorageDatatype, VectorStorageType,
};

pub const TEMP_SEGMENTS_PATH: &str = "temp_segments";
pub const DEFAULT_MAX_SEGMENT_PER_CPU_KB: usize = 256_000;
pub const DEFAULT_INDEXING_THRESHOLD_KB: usize = 10_000;
pub const DEFAULT_DELETED_THRESHOLD: f64 = 0.2;
pub const DEFAULT_VACUUM_MIN_VECTOR_NUMBER: usize = 1000;

/// Extra configuration for dense vectors, applied on top of the plain config during optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseVectorOptimizerConfig {
    pub on_disk: Option<bool>,
    pub memory: Option<Memory>,
    pub hnsw_config: HnswConfig,
    pub quantization_config: Option<QuantizationConfig>,
}

impl DenseVectorOptimizerConfig {
    /// Requested memory placement of the original vector storage, resolving the new `memory`
    /// parameter against the deprecated `on_disk` flag. `None` if neither is configured.
    pub fn memory_placement(&self) -> Option<Memory> {
        Memory::resolve(self.memory, self.on_disk.map(Memory::from_on_disk))
    }
}

/// Extra configuration for sparse vectors, applied on top of the plain config during optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseVectorOptimizerConfig {
    pub on_disk: Option<bool>,
    pub memory: Option<Memory>,
}

impl SparseVectorOptimizerConfig {
    /// Requested memory placement of the sparse index, resolving the new `memory` parameter
    /// against the deprecated `on_disk` flag. `None` if neither is configured.
    pub fn memory_placement(&self) -> Option<Memory> {
        Memory::resolve(self.memory, self.on_disk.map(Memory::from_on_disk_heap))
    }
}

/// Live read of the vector names currently present in the collection schema.
///
/// Unlike the rest of [`SegmentOptimizerConfig`], which is a frozen snapshot taken when the
/// optimizer was built, this reads the *current* schema each time it is called. Optimization needs
/// the live view to tell a vector name that was deleted from the collection (and should be pruned
/// when rebuilding old segments) from one that was just created but is not yet in this optimizer's
/// frozen target (the CreateVectorName race, which must cancel instead). Wrapped in a newtype so
/// `SegmentOptimizerConfig` can keep deriving `Debug`.
#[derive(Clone)]
pub struct LiveVectorNamesProvider(Arc<dyn Fn() -> HashSet<VectorNameBuf> + Send + Sync>);

impl LiveVectorNamesProvider {
    pub fn new(read: impl Fn() -> HashSet<VectorNameBuf> + Send + Sync + 'static) -> Self {
        Self(Arc::new(read))
    }

    pub fn get(&self) -> HashSet<VectorNameBuf> {
        self.0()
    }
}

impl fmt::Debug for LiveVectorNamesProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiveVectorNamesProvider")
            .finish_non_exhaustive()
    }
}

/// This configuration contains all necessary information to build an optimized segment.
#[derive(Debug, Clone)]
pub struct SegmentOptimizerConfig {
    pub payload_storage_type: PayloadStorageType,
    /// Configuration of dense vectors, as it should be for a plain segment (without any optimization).
    pub plain_dense_vector_config: HashMap<VectorNameBuf, VectorDataConfig>,
    /// Configuration of sparse vectors, as it should be for a plain segment (without any optimization).
    pub plain_sparse_vector_config: HashMap<VectorNameBuf, SparseVectorDataConfig>,
    /// Extra configuration for dense vectors, which _might_ be applied during optimization,
    /// depending on the segment state.
    pub dense_vector: HashMap<VectorNameBuf, DenseVectorOptimizerConfig>,
    /// Extra configuration for sparse vectors, which _might_ be applied during optimization,
    /// depending on the segment state.
    pub sparse_vector: HashMap<VectorNameBuf, SparseVectorOptimizerConfig>,
    /// Live read of the collection's vector names, when wired in via
    /// [`SegmentOptimizerConfig::with_live_vector_names`]. `None` if no live source is available.
    pub live_vector_names: Option<LiveVectorNamesProvider>,
}

impl SegmentOptimizerConfig {
    pub fn plain_segment_config(&self) -> SegmentConfig {
        SegmentConfig {
            vector_data: self.plain_dense_vector_config.clone(),
            sparse_vector_data: self.plain_sparse_vector_config.clone(),
            payload_storage_type: self.payload_storage_type,
        }
    }

    /// The config an optimized segment of this size gets.
    ///
    /// Starts from the plain config and applies what the segment's size earns it: an HNSW index
    /// and quantization once the indexing threshold is crossed, mmap storage once the mmap
    /// threshold is or an on-disk vector is indexed, and the matching sparse index type.
    ///
    /// `maximal_vector_store_size_bytes` is the largest per-vector-name storage footprint the
    /// segment will hold, which is what both thresholds are compared against — not the segment's
    /// total size across all names.
    pub fn optimized_segment_config(
        &self,
        thresholds: &crate::operations::optimization::OptimizerThresholds,
        maximal_vector_store_size_bytes: usize,
        any_has_deferred: bool,
    ) -> SegmentConfig {
        let threshold_is_indexed = maximal_vector_store_size_bytes
            >= thresholds.indexing_threshold_kb.saturating_mul(BYTES_IN_KB);

        let threshold_is_on_disk = maximal_vector_store_size_bytes
            >= thresholds.memmap_threshold_kb.saturating_mul(BYTES_IN_KB);

        let mut vector_data = self.plain_dense_vector_config.clone();
        let mut sparse_vector_data = self.plain_sparse_vector_config.clone();

        // If indexing, change to HNSW index and quantization
        // We must always create an HNSW index if we have deferred points to be able to promote them
        if threshold_is_indexed || any_has_deferred {
            if !threshold_is_indexed {
                log::info!(
                    "Segment has deferred points, but doesn't exceed indexing threshold. It will be optimized with HNSW index and quantization."
                );
            }
            vector_data.iter_mut().for_each(|(vector_name, config)| {
                if let Some(vector_cfg) = self.dense_vector.get(vector_name) {
                    // Assign HNSW index
                    config.index = Indexes::Hnsw(vector_cfg.hnsw_config);
                    // Assign quantization config
                    config.quantization_config = vector_cfg.quantization_config.clone();
                }
            });
        }

        // We want to use single-file mmap in the following cases:
        // - It is explicitly configured by `mmap_threshold` -> threshold_is_on_disk=true
        // - The segment is indexed and configured on disk
        //   -> threshold_is_indexed=true && requested placement is cold
        if threshold_is_on_disk || threshold_is_indexed {
            vector_data.iter_mut().for_each(|(vector_name, config)| {
                // Requested memory placement: explicit `memory`, or the deprecated `on_disk`
                let config_memory = self.dense_vector.get(vector_name).and_then(|cfg| {
                    Memory::resolve_or_warn(
                        cfg.memory,
                        cfg.on_disk.map(Memory::from_on_disk),
                        &format_args!("dense vector `{vector_name}`"),
                    )
                });

                match config_memory {
                    // Both agree, but prefer mmap storage type
                    Some(Memory::Cold) => config.storage_type = VectorStorageType::Mmap,
                    // `pinned` is not supported for dense vector storage (rejected by API
                    // validation); defensively treated as the closest supported placement
                    Some(Memory::Cached) | Some(Memory::Pinned) => {
                        if common::flags::feature_flags().single_file_mmap_vector_storage {
                            config.storage_type = VectorStorageType::InRamMmap;
                        }
                        // requested in-RAM placement wins, do nothing
                    }
                    None => {
                        if threshold_is_on_disk {
                            // Mmap threshold wins
                            config.storage_type = VectorStorageType::Mmap
                        } else if common::flags::feature_flags().single_file_mmap_vector_storage {
                            config.storage_type = VectorStorageType::InRamMmap;
                        }
                    }
                }

                // If we explicitly configure the placement, but the segment storage type uses
                // something that doesn't match, warn about it
                if let Some(config_memory) = config_memory
                    && config_memory.is_on_disk() != config.storage_type.is_on_disk()
                {
                    log::warn!(
                        "Collection config for vector {vector_name} has memory placement {config_memory:?} configured, but storage type for segment doesn't match it"
                    );
                }
            });
        }

        sparse_vector_data
            .iter_mut()
            .for_each(|(vector_name, config)| {
                // Requested memory placement: explicit `memory`, or the deprecated `on_disk`
                let config_memory = self.sparse_vector.get(vector_name).and_then(|cfg| {
                    Memory::resolve_or_warn(
                        cfg.memory,
                        cfg.on_disk.map(Memory::from_on_disk_heap),
                        &format_args!("sparse vector `{vector_name}`"),
                    )
                });

                let requested_memory = config_memory
                    .unwrap_or_else(|| Memory::from_on_disk_heap(threshold_is_on_disk));

                // If mmap OR index is exceeded
                let is_big = threshold_is_on_disk || threshold_is_indexed;

                let index_type = if is_big {
                    match requested_memory {
                        // Both cold and cached are backed by the mmap index; the requested
                        // placement is kept in the index config to drive cache population
                        Memory::Cold | Memory::Cached => SparseIndexType::Mmap,
                        Memory::Pinned => SparseIndexType::ImmutableRam,
                    }
                } else {
                    SparseIndexType::MutableRam
                };

                config.index.index_type = index_type;
                // Persist only the explicitly requested `memory` parameter: the structural
                // decision is carried by `index_type`, and only the cold/cached distinction
                // (reachable solely through the explicit parameter) needs the extra field.
                // Legacy-only configurations thus keep a byte-identical index config,
                // which older Qdrant versions can load without any unknown fields.
                config.index.memory = self
                    .sparse_vector
                    .get(vector_name)
                    .and_then(|cfg| cfg.memory);
                // Same reasoning as `memory` just above, applied in the other direction:
                // `wand_pruning` is read only by the mutable RAM index, so persisting it into a
                // compressed one records a setting that index will never consult and contradicts
                // the "only MutableRam is affected" invariant its own doc states. Clearing it also
                // keeps a legacy-only compressed index config byte-identical for older versions.
                if index_type != SparseIndexType::MutableRam {
                    config.index.wand_pruning = None;
                }
            });

        SegmentConfig {
            vector_data,
            sparse_vector_data,
            payload_storage_type: self.payload_storage_type,
        }
    }

    pub fn new(
        payload_storage_type: PayloadStorageType,
        dense_vectors: HashMap<VectorNameBuf, DenseVectorOptimizerInput>,
        sparse_vectors: HashMap<VectorNameBuf, SparseVectorOptimizerInput>,
    ) -> SegmentOptimizerConfig {
        let (mut plain_dense_vector_config, mut dense_vector) = (HashMap::new(), HashMap::new());
        for (name, input) in dense_vectors {
            let DenseVectorOptimizerInput {
                size,
                distance,
                on_disk,
                memory,
                hnsw_config,
                quantization_config,
                multivector_config,
                datatype,
            } = input;
            let plain_memory = Memory::resolve(
                memory,
                Some(Memory::from_on_disk(on_disk.unwrap_or_default())),
            )
            .unwrap_or(Memory::Cached);
            plain_dense_vector_config.insert(
                name.clone(),
                VectorDataConfig {
                    size,
                    distance,
                    index: Indexes::Plain {},
                    storage_type: VectorStorageType::appendable_from_memory(plain_memory),
                    quantization_config: QuantizationConfig::for_appendable_segment(
                        quantization_config.as_ref(),
                    ),
                    multivector_config,
                    datatype,
                },
            );
            dense_vector.insert(
                name,
                DenseVectorOptimizerConfig {
                    on_disk,
                    memory,
                    hnsw_config,
                    quantization_config,
                },
            );
        }

        let (mut plain_sparse_vector_config, mut sparse_vector) = (HashMap::new(), HashMap::new());
        for (name, input) in sparse_vectors {
            let SparseVectorOptimizerInput {
                on_disk,
                memory,
                full_scan_threshold,
                index_datatype,
                storage_type,
                modifier,
                wand_pruning,
            } = input;
            plain_sparse_vector_config.insert(
                name.clone(),
                SparseVectorDataConfig {
                    index: SparseIndexConfig {
                        full_scan_threshold,
                        index_type: SparseIndexType::MutableRam,
                        datatype: index_datatype,
                        memory,
                        wand_pruning,
                    },
                    storage_type,
                    modifier,
                },
            );
            sparse_vector.insert(name, SparseVectorOptimizerConfig { on_disk, memory });
        }

        SegmentOptimizerConfig {
            payload_storage_type,
            plain_dense_vector_config,
            plain_sparse_vector_config,
            dense_vector,
            sparse_vector,
            live_vector_names: None,
        }
    }

    /// Attach a live read of the collection's vector names (see [`LiveVectorNamesProvider`]).
    #[must_use]
    pub fn with_live_vector_names(mut self, provider: LiveVectorNamesProvider) -> Self {
        self.live_vector_names = Some(provider);
        self
    }

    /// The collection's current vector names, if a live source was wired in.
    pub fn live_vector_names(&self) -> Option<HashSet<VectorNameBuf>> {
        self.live_vector_names
            .as_ref()
            .map(LiveVectorNamesProvider::get)
    }
}

/// Per-dense-vector input for the optimizer builder.
#[derive(Debug, Clone)]
pub struct DenseVectorOptimizerInput {
    pub size: usize,
    pub distance: Distance,
    pub on_disk: Option<bool>,
    pub memory: Option<Memory>,
    pub hnsw_config: HnswConfig,
    pub quantization_config: Option<QuantizationConfig>,
    pub multivector_config: Option<MultiVectorConfig>,
    pub datatype: Option<VectorStorageDatatype>,
}

/// Per-sparse-vector input for the optimizer builder.
#[derive(Debug, Clone)]
pub struct SparseVectorOptimizerInput {
    pub on_disk: Option<bool>,
    pub memory: Option<Memory>,
    pub full_scan_threshold: Option<usize>,
    pub index_datatype: Option<VectorStorageDatatype>,
    pub storage_type: SparseVectorStorageType,
    pub modifier: Option<Modifier>,
    pub wand_pruning: Option<bool>,
}

/// Target segment count for the merge optimizer.
pub fn default_segment_number() -> usize {
    // Configure 1 segment per 2 CPUs, as a middle ground between
    // latency and RPS.
    let expected_segments = common::cpu::get_num_cpus() / 2;
    // Do not configure less than 2 and more than 8 segments
    // until it is not explicitly requested
    expected_segments.clamp(2, 8)
}

// --- Shared optimizer threshold helpers (used by collection and edge) ---

/// Resolve number of segments: if `default_segment_number` is 0, use CPU-based default.
pub fn get_number_segments(requested_segment_number: usize) -> usize {
    if requested_segment_number == 0 {
        default_segment_number()
    } else {
        requested_segment_number
    }
}

/// Resolve indexing threshold in KB: `None` => default, `Some(0)` => disable (usize::MAX).
pub fn get_indexing_threshold_kb(indexing_threshold: Option<usize>) -> usize {
    match indexing_threshold {
        None => DEFAULT_INDEXING_THRESHOLD_KB,
        Some(0) => usize::MAX,
        Some(custom) => custom,
    }
}

/// Resolve max segment size in KB: custom value or per-thread default.
pub fn get_max_segment_size_kb(
    max_segment_size: Option<usize>,
    num_indexing_threads: usize,
) -> usize {
    if let Some(max) = max_segment_size {
        max
    } else {
        num_indexing_threads.saturating_mul(DEFAULT_MAX_SEGMENT_PER_CPU_KB)
    }
}

/// Build deferred points threshold in bytes when `prevent_unoptimized` is true.
pub fn get_deferred_points_threshold_bytes(
    prevent_unoptimized: Option<bool>,
    indexing_threshold_kb: usize,
) -> Option<NonZeroUsize> {
    (prevent_unoptimized == Some(true))
        .then(|| indexing_threshold_kb.saturating_mul(BYTES_IN_KB))
        .and_then(NonZeroUsize::new)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[test]
    fn live_vector_names_provider_reads_current_state() {
        // The provider must re-read the live source on every call rather than snapshot it once,
        // otherwise a vector deleted (or created) after optimizer construction would be missed and
        // the merge would make the wrong cancel/prune decision.
        let source = Arc::new(Mutex::new(HashSet::from(["a".to_owned(), "b".to_owned()])));
        let provider = {
            let source = source.clone();
            LiveVectorNamesProvider::new(move || source.lock().unwrap().clone())
        };

        assert_eq!(
            provider.get(),
            HashSet::from(["a".to_owned(), "b".to_owned()])
        );

        // Delete "b" from the live source: the provider must reflect it on the next read.
        source.lock().unwrap().remove("b");
        assert_eq!(provider.get(), HashSet::from(["a".to_owned()]));
    }
}
