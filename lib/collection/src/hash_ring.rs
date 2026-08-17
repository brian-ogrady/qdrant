use std::collections::HashSet;
use std::fmt;
use std::hash::{BuildHasherDefault, Hash};
use std::sync::LazyLock;

use bytemuck::TransparentWrapper as _;
use common::stable_hash::{StableHash, StableHashed};
use itertools::Itertools as _;
use segment::index::field_index::CardinalityEstimation;
use segment::types::{CustomIdCheckerCondition, PointIdType};
use semver::Version;
use smallvec::SmallVec;

use crate::operations::cluster_ops::ReshardingDirection;
use crate::shards::shard::ShardId;

/// Default number of virtual nodes per shard on the hash ring.
///
/// Must never change: it is the serde default for `CollectionParams::hash_ring_shard_scale`, so every
/// collection written before that field existed is routed at this value. A different number here
/// silently remaps all of them.
pub const DEFAULT_HASH_RING_SHARD_SCALE: u32 = 100;

/// Upper bound on the hash ring scale.
pub const MAX_HASH_RING_SHARD_SCALE: u32 = 100_000;

/// Upper bound on `hash_ring_shard_scale * shard_number`, i.e. the total virtual nodes in one ring.
pub const MAX_HASH_RING_VIRTUAL_NODES: u64 = 1_000_000;

/// The bound is repeated as a literal in four `validate(range(max = ...))` attributes —
/// `CollectionParams`, `CollectionParamsDiff`, the `CreateCollection` op, and
/// `CollectionConfigDefaults` — plus in prose in `config/config.yaml`. Only the last of those is forced
/// to duplicate it (`segment` sits below `collection` in the dependency graph and cannot name the
/// constant); the other three could reference it and do not.
///
/// This pins the constant itself, so editing it without the attributes fails the build. It does not
/// catch the opposite edit — relaxing one attribute in isolation — so if you change the bound, grep
/// for `100_000`.
const _: () = assert!(MAX_HASH_RING_SHARD_SCALE == 100_000);

/// Minimum Qdrant version required for `hash_ring_shard_scale`.
///
/// Prevents mixed-version clusters from using inconsistent hash-ring mappings
/// when older peers do not understand this collection parameter.
pub static HASH_RING_SHARD_SCALE_VERSION: LazyLock<Version> =
    LazyLock::new(|| Version::parse("1.19.1-dev").expect("valid version string"));

#[derive(Clone, Debug, PartialEq)]
pub enum HashRingRouter<T: Eq + StableHash + Hash = ShardId> {
    /// Single hashring
    Single(HashRing<T>),

    /// Two hashrings when transitioning during resharding
    /// Depending on the current resharding state, points may be in either or both shards.
    Resharding { old: HashRing<T>, new: HashRing<T> },
}

impl<T: Copy + Eq + StableHash + Hash> HashRingRouter<T> {
    /// Create a new single hashring with a fair distribution of points at the given `scale`.
    ///
    /// `scale` is a routing input, so it must come from the owning collection's persisted
    /// [`CollectionParams::hash_ring_shard_scale`](crate::config::CollectionParams::hash_ring_shard_scale)
    /// rather than from the environment — otherwise the mapping could shift across restarts or
    /// differ between peers. It is a required argument precisely so no caller can silently fall
    /// back to a default.
    pub fn single(scale: u32) -> Self {
        Self::Single(HashRing::fair(scale))
    }

    pub fn add(&mut self, shard: T) -> bool {
        match self {
            Self::Single(ring) => ring.add(shard),
            Self::Resharding { old, new } => {
                // When resharding is in progress:
                // - either `new` hashring contains a shard, that is not in `old` (when resharding *up*)
                // - or `old` contains a shard, that is not in `new` (when resharding *down*)
                //
                // This check ensures, that we don't accidentally break this invariant when adding
                // nodes to `Resharding` hashring.

                if !old.contains(&shard) && !new.contains(&shard) {
                    old.add(shard);
                    new.add(shard);
                    true
                } else {
                    false
                }
            }
        }
    }

    pub fn start_resharding(&mut self, shard: T, direction: ReshardingDirection) {
        if let Self::Single(ring) = self {
            let (old, new) = (ring.clone(), ring.clone());
            *self = Self::Resharding { old, new };
        }

        let Self::Resharding { old, new } = self else {
            unreachable!();
        };

        match direction {
            ReshardingDirection::Up => {
                old.remove(&shard);
                new.add(shard);
            }

            ReshardingDirection::Down => {
                assert!(new.len() > 1, "cannot remove last shard from hash ring");

                old.add(shard);
                new.remove(&shard);
            }
        }
    }

    pub fn commit_resharding(&mut self) -> bool {
        let Self::Resharding { new, .. } = self else {
            log::warn!("committing resharding hashring, but hashring is not in resharding mode");
            return false;
        };

        *self = Self::Single(new.clone());
        true
    }

    pub fn abort_resharding(&mut self, shard: T, direction: ReshardingDirection) -> bool
    where
        T: fmt::Display,
    {
        let context = match direction {
            ReshardingDirection::Up => "reverting scale-up hashring into single mode",
            ReshardingDirection::Down => "reverting scale-down hashring into single mode",
        };

        let Self::Resharding { old, new } = self else {
            log::warn!("{context}, but hashring is not in resharding mode");
            return false;
        };

        let mut old = old.clone();
        let mut new = new.clone();

        let (expected_in_old, expected_in_new) = match direction {
            ReshardingDirection::Up => (old.remove(&shard), new.remove(&shard)),
            ReshardingDirection::Down => (old.add(shard), new.add(shard)),
        };

        match (expected_in_old, expected_in_new) {
            (false, true) => (),

            (true, false) => {
                log::error!("{context}, but expected state of hashrings is reversed");
            }

            (true, true) => {
                log::error!("{context}, but {shard} is not a target shard");
            }

            (false, false) => {
                log::warn!("{context}, but shard {shard} does not exist in the hashring");
            }
        };

        if old == new {
            log::debug!("{context}, because the rerouting for resharding is done");
            *self = Self::Single(old.clone());
            true
        } else {
            log::warn!("{context}, but rerouting for resharding is not done yet");
            false
        }
    }

    pub fn get<U: StableHash>(&self, key: &U) -> ShardIds<T> {
        match self {
            Self::Single(ring) => ring.get(key).into_iter().copied().collect(),
            Self::Resharding { old, new } => old
                .get(key)
                .into_iter()
                .chain(new.get(key))
                .copied()
                .dedup() // Both hash rings may return the same shard ID, take it once
                .collect(),
        }
    }

    /// Check whether the given point is in the given shard
    ///
    /// In case of resharding, the new hashring is checked.
    pub fn is_in_shard<U: StableHash>(&self, key: &U, shard: T) -> bool {
        let ring = match self {
            Self::Resharding { new, .. } => new,
            Self::Single(ring) => ring,
        };

        ring.get(key) == Some(&shard)
    }
}

impl<T: Eq + StableHash + Hash> HashRingRouter<T> {
    pub fn is_resharding(&self) -> bool {
        matches!(self, Self::Resharding { .. })
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Single(ring) => ring.is_empty(),
            Self::Resharding { old, new } => old.is_empty() && new.is_empty(),
        }
    }

    /// Get unique nodes from the hashring
    pub fn nodes(&self) -> &HashSet<T> {
        match self {
            HashRingRouter::Single(ring) => ring.nodes(),
            HashRingRouter::Resharding { new, .. } => new.nodes(),
        }
    }
}

type StableHashBuilder = BuildHasherDefault<siphasher::sip::SipHasher24>;

/// List type for shard IDs
///
/// Uses a `SmallVec` putting two IDs on the stack. That's the maximum number of shards we expect
/// with the current resharding implementation.
pub type ShardIds<T = ShardId> = SmallVec<[T; 2]>;

#[derive(Clone, Debug, PartialEq)]
pub enum HashRing<T: Eq + StableHash + Hash> {
    Raw {
        nodes: HashSet<T>,
        ring: hashring::HashRing<StableHashed<T>, StableHashBuilder>,
    },

    Fair {
        nodes: HashSet<T>,
        ring: hashring::HashRing<StableHashed<(T, u32)>, StableHashBuilder>,
        scale: u32,
    },
}

impl<T: Copy + Eq + StableHash + Hash> HashRing<T> {
    pub fn raw() -> Self {
        Self::Raw {
            nodes: HashSet::new(),
            ring: hashring::HashRing::with_hasher(StableHashBuilder::new()),
        }
    }

    /// Constructs a HashRing that tries to give all shards equal space on the ring.
    /// The higher the `scale` - the more equal the distribution of points on the shards will be,
    /// but shard search might be slower.
    pub fn fair(scale: u32) -> Self {
        Self::Fair {
            nodes: HashSet::new(),
            ring: hashring::HashRing::with_hasher(StableHashBuilder::new()),
            scale,
        }
    }

    pub fn add(&mut self, shard: T) -> bool {
        if !self.nodes_mut().insert(shard) {
            return false;
        }

        match self {
            HashRing::Raw { ring, .. } => {
                ring.add(StableHashed(shard));
            }

            HashRing::Fair { ring, scale, .. } => {
                ring.batch_add((0..*scale).map(|idx| StableHashed((shard, idx))).collect());
            }
        }

        true
    }

    pub fn remove(&mut self, shard: &T) -> bool {
        if !self.nodes_mut().remove(shard) {
            return false;
        }

        match self {
            HashRing::Raw { ring, .. } => {
                ring.remove(&StableHashed(*shard));
            }

            HashRing::Fair { nodes, ring, scale } => {
                // Rebuilt from the remaining shards rather than deleted node by node. `hashring::remove`
                // is a binary search plus a `Vec::remove` per virtual node, so dropping one shard costs
                // O(scale × nodes) — measured in release at 9.76s for 1M nodes, versus 39ms for this
                // rebuild. Producing the same nodes in the same order is what keeps routing identical;
                // `removing_a_shard_leaves_the_ring_the_remaining_shards_would_have_built` pins that.
                let mut rebuilt = hashring::HashRing::with_hasher(StableHashBuilder::new());
                rebuilt.batch_add(
                    nodes
                        .iter()
                        .flat_map(|&remaining| {
                            (0..*scale).map(move |idx| StableHashed((remaining, idx)))
                        })
                        .collect(),
                );
                *ring = rebuilt;
            }
        }

        true
    }
}

impl<T: Eq + StableHash + Hash> HashRing<T> {
    pub fn get<U: StableHash>(&self, key: &U) -> Option<&T> {
        let key = StableHashed::wrap_ref(key);
        match self {
            HashRing::Raw { ring, .. } => ring.get(key).map(|StableHashed(shard)| shard),
            HashRing::Fair { ring, .. } => ring.get(key).map(|StableHashed((shard, _))| shard),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes().is_empty()
    }

    pub fn len(&self) -> usize {
        self.nodes().len()
    }

    pub fn contains(&self, shard: &T) -> bool {
        self.nodes().contains(shard)
    }

    pub fn nodes(&self) -> &HashSet<T> {
        match self {
            HashRing::Raw { nodes, .. } => nodes,
            HashRing::Fair { nodes, .. } => nodes,
        }
    }

    fn nodes_mut(&mut self) -> &mut HashSet<T> {
        match self {
            HashRing::Raw { nodes, .. } => nodes,
            HashRing::Fair { nodes, .. } => nodes,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HashRingFilter {
    ring: HashRing<ShardId>,
    expected_shard_id: ShardId,
}

impl HashRingFilter {
    pub fn new(ring: HashRing<ShardId>, expected_shard_id: ShardId) -> Self {
        Self {
            ring,
            expected_shard_id,
        }
    }
}

impl CustomIdCheckerCondition for HashRingFilter {
    fn estimate_cardinality(&self, points: usize) -> CardinalityEstimation {
        CardinalityEstimation {
            primary_clauses: vec![],
            min: 0,
            exp: points / self.ring.len(),
            max: points,
        }
    }

    fn check(&self, point_id: PointIdType) -> bool {
        self.ring.get(&point_id) == Some(&self.expected_shard_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_seq_keys() {
        let mut ring = HashRing::fair(100);
        ring.add(5);
        ring.add(7);
        ring.add(8);
        ring.add(20);

        for i in 0..20 {
            match ring.get(&i) {
                None => panic!("Key {i} has no shard"),
                Some(x) => assert!([5, 7, 8, 20].contains(x)),
            }
        }
    }

    #[test]
    fn test_repartition() {
        let mut ring = HashRing::fair(100);

        ring.add(1);
        ring.add(2);
        ring.add(3);

        let mut pre_split = Vec::new();
        let mut post_split = Vec::new();

        for i in 0..100 {
            match ring.get(&i) {
                None => panic!("Key {i} has no shard"),
                Some(x) => pre_split.push(*x),
            }
        }

        ring.add(4);

        for i in 0..100 {
            match ring.get(&i) {
                None => panic!("Key {i} has no shard"),
                Some(x) => post_split.push(*x),
            }
        }

        assert_ne!(pre_split, post_split);

        for (x, y) in pre_split.iter().zip(post_split.iter()) {
            if x != y {
                assert_eq!(*y, 4);
            }
        }
    }

    /// Removing a shard must leave exactly the ring the remaining shards would have built.
    /// Compares the surviving mapping key by key rather than just the node count, since two rings can
    /// hold the same nodes and still order them differently.
    #[test]
    fn removing_a_shard_leaves_the_ring_the_remaining_shards_would_have_built() {
        const SCALE: u32 = 7;
        const SHARDS: ShardId = 6;
        const REMOVED: ShardId = 2;
        const KEYS: u64 = 2_000;

        let mut removed_from = HashRing::fair(SCALE);
        for shard in 0..SHARDS {
            removed_from.add(shard);
        }
        assert!(removed_from.remove(&REMOVED));

        let mut built_without = HashRing::fair(SCALE);
        for shard in (0..SHARDS).filter(|&shard| shard != REMOVED) {
            built_without.add(shard);
        }

        let mapping =
            |ring: &HashRing<ShardId>| (0..KEYS).map(|key| *ring.get(&key).unwrap()).collect_vec();
        let after_removal = mapping(&removed_from);
        assert_eq!(
            after_removal,
            mapping(&built_without),
            "a ring with a shard removed must route exactly like one built without that shard",
        );

        // The removed shard must own nothing, and the rest must still own something — otherwise the
        // comparison above could hold for a degenerate ring.
        assert!(
            !after_removal.contains(&REMOVED),
            "the removed shard must not own any part of the keyspace",
        );
        assert_eq!(
            after_removal.iter().unique().count(),
            SHARDS as usize - 1,
            "every remaining shard should still own part of the keyspace",
        );

        // Removing a shard that was never there must not disturb the ring.
        let mut untouched = removed_from.clone();
        assert!(!untouched.remove(&REMOVED));
        assert_eq!(mapping(&untouched), after_removal);
    }

    #[test]
    fn batching_virtual_nodes_does_not_change_routing() {
        const SCALE: u32 = 7;
        const SHARDS: ShardId = 5;
        const KEYS: u64 = 500;

        // What the code does now: one batch per shard.
        let batched = {
            let mut ring = HashRing::fair(SCALE);
            for shard in 0..SHARDS {
                ring.add(shard);
            }
            (0..KEYS).map(|key| *ring.get(&key).unwrap()).collect_vec()
        };

        let per_node = {
            let mut ring = hashring::HashRing::with_hasher(StableHashBuilder::new());
            for shard in 0..SHARDS {
                for idx in 0..SCALE {
                    ring.add(StableHashed((shard, idx)));
                }
            }
            (0..KEYS)
                .map(|key| {
                    let StableHashed((shard, _idx)) =
                        ring.get(StableHashed::wrap_ref(&key)).unwrap();
                    *shard
                })
                .collect_vec()
        };

        assert_eq!(
            batched, per_node,
            "batching virtual nodes must not change the point-to-shard mapping",
        );

        // Guards against a degenerate ring making the comparison above vacuous.
        let distinct = batched.iter().unique().count();
        assert_eq!(
            distinct, SHARDS as usize,
            "expected all {SHARDS} shards to own part of the keyspace, saw {distinct}",
        );
    }
}
