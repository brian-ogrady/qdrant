use std::cmp::max;

use common::types::PointOffsetType;
use log::debug;
use rayon::prelude::*;

use crate::common::sparse_vector::RemappedSparseVector;
use crate::index::inverted_index::inverted_index_ram::InvertedIndexRam;
use crate::index::posting_list::PostingBuilder;
use crate::index::posting_list_common::PostingElementEx;

/// Builder for InvertedIndexRam
pub struct InvertedIndexBuilder {
    pub posting_builders: Vec<PostingBuilder>,
    pub vector_count: usize,
    pub total_sparse_size: usize,
}

impl Default for InvertedIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl InvertedIndexBuilder {
    pub fn new() -> InvertedIndexBuilder {
        InvertedIndexBuilder {
            posting_builders: Vec::new(),
            vector_count: 0,
            total_sparse_size: 0,
        }
    }

    /// Add a vector to the inverted index builder
    pub fn add(&mut self, id: PointOffsetType, vector: RemappedSparseVector) {
        let sparse_size = vector.len() * size_of::<PostingElementEx>();
        for (dim_id, weight) in vector.indices.into_iter().zip(vector.values) {
            let dim_id = dim_id as usize;
            self.posting_builders.resize_with(
                max(dim_id + 1, self.posting_builders.len()),
                PostingBuilder::new,
            );
            self.posting_builders[dim_id].add(id, weight);
        }
        self.vector_count += 1;
        self.total_sparse_size = self.total_sparse_size.saturating_add(sparse_size);
    }

    /// Consumes the builder and returns an InvertedIndexRam
    pub fn build(self) -> InvertedIndexRam {
        self.build_with_threads(1)
    }

    /// Finalize independent posting lists using at most `num_threads` workers.
    ///
    /// Indexed Rayon collection retains the dimension order, so this produces the same inverted
    /// index as [`Self::build`] while parallelizing the per-posting sort and WAND-bound pass.
    pub fn build_with_threads(self, num_threads: usize) -> InvertedIndexRam {
        if self.posting_builders.is_empty() {
            return InvertedIndexRam {
                postings: vec![],
                total_sparse_size: self.total_sparse_size,
                vector_count: self.vector_count,
                // The one-pass build always computes exact bounds; whether they are then
                // maintained across later writes is the caller's policy, set afterwards via
                // `set_maintain_max_next_weight`.
                maintain_max_next_weight: true,
            };
        }

        debug!(
            "building inverted index with {} sparse vectors in {} posting lists",
            self.vector_count,
            self.posting_builders.len(),
        );

        let threads = num_threads.clamp(1, self.posting_builders.len());
        let postings = if threads == 1 {
            self.posting_builders
                .into_iter()
                .map(PostingBuilder::build)
                .collect()
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a positive sparse-index thread count must create a Rayon pool")
                .install(|| {
                    self.posting_builders
                        .into_par_iter()
                        .map(PostingBuilder::build)
                        .collect()
                })
        };

        let vector_count = self.vector_count;
        let total_sparse_size = self.total_sparse_size;
        InvertedIndexRam {
            postings,
            vector_count,
            total_sparse_size,
            maintain_max_next_weight: true,
        }
    }

    /// Creates an [InvertedIndexRam] from an iterator of (id, vector) pairs.
    pub fn build_from_iterator(
        iter: impl Iterator<Item = (PointOffsetType, RemappedSparseVector)>,
    ) -> InvertedIndexRam {
        let mut builder = InvertedIndexBuilder::new();
        for (id, vector) in iter {
            builder.add(id, vector);
        }
        builder.build()
    }
}
