//! End-to-end check that the `wand_pruning` sparse index parameter reaches the index it governs.
//!
//! The unit tests in `lib/sparse` cover what the switch *does*. This one covers the wiring: a
//! `SparseIndexConfig` carrying `wand_pruning: Some(false)` has to arrive at the `InvertedIndexRam`
//! that `SparseVectorIndex::open` builds, and be applied before that index takes any writes. If the
//! plumbing through `plan()` were dropped the collection parameter would silently do nothing, and
//! nothing else in the test suite would notice.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use atomic_refcell::AtomicRefCell;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use common::universal_io::MmapFs;
use rand::SeedableRng;
use segment::common::operation_error::OperationResult;
use segment::fixtures::payload_context_fixture::create_id_tracker_fixture;
use segment::index::sparse_index::sparse_index_config::{SparseIndexConfig, SparseIndexType};
use segment::index::sparse_index::sparse_vector_index::{
    SparseVectorIndex, SparseVectorIndexOpenArgs,
};
use segment::index::struct_payload_index::{IndexLoadMode, StorageType, StructPayloadIndex};
use segment::payload_storage::in_memory_payload_storage::InMemoryPayloadStorage;
use segment::vector_storage::sparse::mmap_sparse_vector_storage::MmapSparseVectorStorage;
use segment::vector_storage::{VectorStorage, VectorStorageEnum};
use sparse::common::sparse_vector_fixture::random_sparse_vector;
use sparse::index::inverted_index::InvertedIndex;
use sparse::index::inverted_index::inverted_index_ram::InvertedIndexRam;
use tempfile::Builder;

const NUM_VECTORS: usize = 200;
const MAX_DIM: usize = 32;

/// Open a mutable-RAM sparse index over a small random corpus with the given `wand_pruning`.
fn open_index(
    data_dir: &std::path::Path,
    wand_pruning: Option<bool>,
) -> OperationResult<SparseVectorIndex<InvertedIndexRam>> {
    let stopped = AtomicBool::new(false);
    let index_dir = &data_dir.join("index");
    let payload_dir = &data_dir.join("payload");
    let storage_dir = &data_dir.join("storage");

    let id_tracker = Arc::new(AtomicRefCell::new(create_id_tracker_fixture(NUM_VECTORS)));
    let payload_storage = InMemoryPayloadStorage::default();
    let wrapped_payload_storage = Arc::new(AtomicRefCell::new(payload_storage.into()));
    let payload_index = StructPayloadIndex::open(
        wrapped_payload_storage,
        id_tracker.clone(),
        std::collections::HashMap::new(),
        payload_dir,
        StorageType::Appendable,
        IndexLoadMode::CreateIfMissing,
    )?;
    let wrapped_payload_index = Arc::new(AtomicRefCell::new(payload_index));

    let vector_storage = Arc::new(AtomicRefCell::new(VectorStorageEnum::SparseMmap(
        MmapSparseVectorStorage::open_or_create(storage_dir)?,
    )));
    {
        let mut borrowed_storage = vector_storage.borrow_mut();
        let hw_counter = HardwareCounterCell::new();
        let mut rnd = rand::rngs::StdRng::seed_from_u64(42);
        for idx in 0..NUM_VECTORS {
            let vec = random_sparse_vector(&mut rnd, MAX_DIM);
            borrowed_storage.insert_vector(idx as PointOffsetType, (&vec).into(), &hw_counter)?;
        }
    }

    let config = SparseIndexConfig {
        wand_pruning,
        ..SparseIndexConfig::new(Some(1), SparseIndexType::MutableRam, None, None)
    };

    SparseVectorIndex::open(SparseVectorIndexOpenArgs {
        fs: &MmapFs,
        config,
        id_tracker,
        vector_storage,
        payload_index: wrapped_payload_index,
        path: index_dir,
        stopped: &stopped,
        num_threads: 1,
        tick_progress: || (),
    })
}

#[test]
fn wand_pruning_parameter_reaches_the_mutable_ram_index() {
    // Unset: the default, pruning stays on. This is what every pre-existing collection gets.
    let dir = Builder::new().prefix("wand_default").tempdir().unwrap();
    let index = open_index(dir.path(), None).unwrap();
    assert!(
        index.inverted_index().max_next_weight_reliable(),
        "wand_pruning defaults to enabled",
    );

    // Explicitly enabled behaves the same as unset.
    let dir = Builder::new().prefix("wand_on").tempdir().unwrap();
    let index = open_index(dir.path(), Some(true)).unwrap();
    assert!(index.inverted_index().max_next_weight_reliable());

    // Disabled: the index must both skip bound maintenance and refuse to prune. Asserting the
    // read-side gate is enough, because a single field drives both.
    let dir = Builder::new().prefix("wand_off").tempdir().unwrap();
    let index = open_index(dir.path(), Some(false)).unwrap();
    assert!(
        !index.inverted_index().max_next_weight_reliable(),
        "wand_pruning=false must reach the inverted index and disable pruning; if this fails the \
         collection parameter is silently doing nothing",
    );
    assert!(!index.inverted_index().maintain_max_next_weight());
}
