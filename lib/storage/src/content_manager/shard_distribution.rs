use std::cmp::{self, Reverse};
use std::collections::BinaryHeap;
use std::iter::repeat_with;
use std::num::NonZeroU32;

use collection::shards::collection_shard_distribution::CollectionShardDistribution;
use collection::shards::shard::{PeerId, ShardId};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq)]
struct PeerShardCount {
    shard_count: usize,
    /// Randomized bias value, to prevent having a consistent order of peers across multiple
    /// generated distributions. This roughly balances nodes across all nodes, if the number of
    /// shards is less than the number of nodes.
    bias: usize,
    peer_id: PeerId,
}

impl PeerShardCount {
    fn new(peer_id: PeerId) -> Self {
        Self {
            shard_count: 0,
            bias: rand::random::<u32>() as usize,
            peer_id,
        }
    }

    fn get_and_inc_shard_count(&mut self) -> PeerId {
        self.shard_count += 1;
        self.peer_id
    }
}

impl PartialOrd for PeerShardCount {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Explicitly implement ordering to make sure we don't accidentally break this.
///
/// Ordering:
/// - shard_count: lowest number of shards first
/// - bias: randomize order of peers with same number of shards
/// - peer_id
impl Ord for PeerShardCount {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.shard_count
            .cmp(&other.shard_count)
            .then(self.bias.cmp(&other.bias))
            // It is very unlikely that we need this, so `then_with` is a bit faster
            .then_with(|| self.peer_id.cmp(&other.peer_id))
    }
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq, Hash, Clone)]
pub struct ShardDistributionProposal {
    /// A shard can be located on several peers if it has replicas
    pub distribution: Vec<(ShardId, Vec<PeerId>)>,
}

impl ShardDistributionProposal {
    /// Suggest an empty shard distribution placement
    /// This is useful when a collection is configured for custom sharding and
    /// we don't want to create any shards in advance.
    pub fn empty() -> Self {
        Self {
            distribution: Vec::new(),
        }
    }

    /// Builds a proposal for the distribution of shards.
    /// It will propose to allocate shards so that all peers have the same number of shards of this collection  at the end.
    pub fn new(
        shard_number: NonZeroU32,
        replication_factor: NonZeroU32,
        known_peers: &[PeerId],
    ) -> Self {
        // Min-heap: peer with lowest number of shards is on top
        let mut min_heap: BinaryHeap<_> = known_peers
            .iter()
            .map(|peer| Reverse(PeerShardCount::new(*peer)))
            .collect();

        // There should not be more than 1 replica per peer
        let replica_number = cmp::min(replication_factor.get() as usize, known_peers.len());

        // Get fair distribution of shards on peers
        let distribution = (0..shard_number.get())
            .map(|shard_id| {
                let replicas =
                    repeat_with(|| min_heap.peek_mut().unwrap().0.get_and_inc_shard_count())
                        .take(replica_number)
                        .collect();
                (shard_id, replicas)
            })
            .collect();

        Self { distribution }
    }
}

impl From<ShardDistributionProposal> for CollectionShardDistribution {
    fn from(proposal: ShardDistributionProposal) -> Self {
        let ShardDistributionProposal { distribution } = proposal;
        CollectionShardDistribution {
            shards: distribution
                .into_iter()
                .map(|(shard_id, peers)| (shard_id, peers.into_iter().collect()))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_distribution() {
        let known_peers = vec![1, 2, 3, 4];
        let distribution = ShardDistributionProposal::new(
            NonZeroU32::new(6).unwrap(),
            NonZeroU32::new(1).unwrap(),
            &known_peers,
        );

        // Check it distribution is as even as possible
        let mut shard_counts: Vec<usize> = vec![0; known_peers.len()];
        for (_shard_id, peers) in &distribution.distribution {
            for peer_id in peers {
                let peer_offset = known_peers
                    .iter()
                    .enumerate()
                    .find(|(_, x)| *x == peer_id)
                    .unwrap()
                    .0;
                shard_counts[peer_offset] += 1;
            }
        }

        assert_eq!(shard_counts.iter().sum::<usize>(), 6);
        assert_eq!(shard_counts.iter().min(), Some(&1));
        assert_eq!(shard_counts.iter().max(), Some(&2));
    }

    #[test]
    fn test_distribution_is_spread() {
        let known_peers = vec![1, 2, 3, 4];
        let shard_numbers = 1..=3;
        let replication_factors = 1..=4;
        let tries = 100;

        // With 4 peers, for various shard number and replication factor ranges, always generate
        // distributions that inhabit all peers across 100 retries.
        for shard_number in shard_numbers {
            for replication_factor in replication_factors.clone() {
                let inhabited_peers = (0..tries)
                    // Generate distribution
                    .map(|_| {
                        ShardDistributionProposal::new(
                            NonZeroU32::new(shard_number).unwrap(),
                            NonZeroU32::new(replication_factor).unwrap(),
                            &known_peers,
                        )
                    })
                    // Take just the inhabited peer IDs
                    .flat_map(|proposal| {
                        proposal
                            .distribution
                            .into_iter()
                            .flat_map(|(_, peers)| peers)
                    })
                    .collect::<HashSet<_>>();

                assert_eq!(
                    inhabited_peers.len(),
                    known_peers.len(),
                    "must inhabit all {} peers across {tries} distributions",
                    known_peers.len(),
                );
            }
        }
    }
}

/// A peer named in an explicit `shard_placement`: by numeric id, or by the p2p URI the
/// peer was started with (`--uri`, as listed by the `/cluster` endpoint).
#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq, Hash, Clone)]
#[serde(untagged)]
pub enum PeerRef {
    Id(PeerId),
    Uri(String),
}

/// Resolve and validate an explicit shard placement into a distribution proposal.
pub fn resolve_explicit_placement(
    placement: &std::collections::BTreeMap<ShardId, Vec<PeerRef>>,
    shard_number: Option<u32>,
    replication_factor: Option<u32>,
    sharding_method: collection::config::ShardingMethod,
    peer_addresses: &crate::types::PeerAddressById,
) -> Result<ShardDistributionProposal, crate::content_manager::errors::StorageError> {
    use crate::content_manager::errors::StorageError;

    if sharding_method != collection::config::ShardingMethod::Auto {
        return Err(StorageError::bad_request(
            "`shard_placement` is only supported with `sharding_method: auto`",
        ));
    }

    let Some(shard_number) = shard_number else {
        return Err(StorageError::bad_request(
            "`shard_placement` requires an explicit `shard_number` matching the placement",
        ));
    };
    let Some(replication_factor) = replication_factor else {
        return Err(StorageError::bad_request(
            "`shard_placement` requires an explicit `replication_factor` matching the \
             placement",
        ));
    };

    let expected: std::collections::BTreeSet<ShardId> = (0..shard_number).collect();
    let given: std::collections::BTreeSet<ShardId> = placement.keys().copied().collect();
    if given != expected {
        return Err(StorageError::bad_request(format!(
            "`shard_placement` must name every shard id 0..{shard_number} exactly once, \
             got {given:?}",
        )));
    }

    // URIs are compared without their trailing slash: the stored form carries one
    // (`http://host:port/`), the form operators write usually does not.
    let normalize = |uri: &str| uri.trim_end_matches('/').to_string();
    let mut uri_to_id: std::collections::HashMap<String, Option<PeerId>> =
        std::collections::HashMap::new();
    for (peer_id, uri) in peer_addresses {
        // `None` marks an ambiguous URI (two peers advertising the same address);
        // resolving through it must fail rather than pick one.
        uri_to_id
            .entry(normalize(&uri.to_string()))
            .and_modify(|entry| *entry = None)
            .or_insert(Some(*peer_id));
    }

    let known_uris = || {
        let mut uris: Vec<&String> = uri_to_id.keys().collect();
        uris.sort();
        format!("{uris:?}")
    };

    let mut distribution = Vec::with_capacity(placement.len());
    for (shard_id, replicas) in placement {
        if replicas.len() != replication_factor as usize {
            return Err(StorageError::bad_request(format!(
                "`shard_placement` for shard {shard_id} names {} peer(s), but \
                 `replication_factor` is {replication_factor}",
                replicas.len(),
            )));
        }

        let mut resolved = Vec::with_capacity(replicas.len());
        for peer in replicas {
            let peer_id = match peer {
                PeerRef::Id(peer_id) => {
                    if !peer_addresses.contains_key(peer_id) {
                        return Err(StorageError::bad_request(format!(
                            "`shard_placement` for shard {shard_id} names unknown peer \
                             {peer_id}; known peers: {:?} (see the /cluster endpoint)",
                            peer_addresses
                                .keys()
                                .collect::<std::collections::BTreeSet<_>>(),
                        )));
                    }
                    *peer_id
                }
                PeerRef::Uri(uri) => match uri_to_id.get(&normalize(uri)) {
                    Some(Some(peer_id)) => *peer_id,
                    Some(None) => {
                        return Err(StorageError::bad_request(format!(
                            "`shard_placement` for shard {shard_id}: URI {uri} is \
                             advertised by more than one peer; use numeric peer ids \
                             (see the /cluster endpoint)",
                        )));
                    }
                    None => {
                        return Err(StorageError::bad_request(format!(
                            "`shard_placement` for shard {shard_id} names unknown peer \
                             URI {uri}; known peer URIs: {}",
                            known_uris(),
                        )));
                    }
                },
            };
            resolved.push(peer_id);
        }

        let unique: std::collections::HashSet<PeerId> = resolved.iter().copied().collect();
        if unique.len() != resolved.len() {
            return Err(StorageError::bad_request(format!(
                "`shard_placement` for shard {shard_id} lists a peer more than once; \
                 replicas of one shard must live on distinct peers",
            )));
        }

        distribution.push((*shard_id, resolved));
    }

    Ok(ShardDistributionProposal { distribution })
}

#[cfg(test)]
mod placement_tests {
    use std::collections::BTreeMap;

    use collection::config::ShardingMethod;

    use super::*;
    use crate::types::PeerAddressById;

    fn peers() -> PeerAddressById {
        PeerAddressById::from([
            (11, "http://nodeA:6335".parse().unwrap()),
            (22, "http://nodeB:6335".parse().unwrap()),
            (33, "http://nodeC:6335".parse().unwrap()),
        ])
    }

    fn resolve(
        placement: BTreeMap<ShardId, Vec<PeerRef>>,
        shard_number: u32,
        replication_factor: u32,
    ) -> Result<ShardDistributionProposal, crate::content_manager::errors::StorageError> {
        resolve_explicit_placement(
            &placement,
            Some(shard_number),
            Some(replication_factor),
            ShardingMethod::Auto,
            &peers(),
        )
    }

    #[test]
    fn uris_ids_and_mixed_forms_all_resolve() {
        let placement = BTreeMap::from([
            (0, vec![PeerRef::Uri("http://nodeA:6335".to_string())]),
            // Trailing slash, as the /cluster endpoint prints it.
            (1, vec![PeerRef::Uri("http://nodeB:6335/".to_string())]),
            (2, vec![PeerRef::Id(33)]),
        ]);
        let proposal = resolve(placement, 3, 1).unwrap();
        assert_eq!(
            proposal.distribution,
            vec![(0, vec![11]), (1, vec![22]), (2, vec![33])],
        );
    }

    #[test]
    fn an_unknown_uri_lists_the_known_ones() {
        let placement = BTreeMap::from([(0, vec![PeerRef::Uri("http://nodeX:6335".to_string())])]);
        let err = resolve(placement, 1, 1).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("nodeX"), "got: {message}");
        assert!(message.contains("http://nodeA:6335"), "got: {message}");
    }

    #[test]
    fn an_ambiguous_uri_is_refused() {
        let mut peers = peers();
        peers.insert(44, "http://nodeA:6335".parse().unwrap());
        let placement = BTreeMap::from([(0, vec![PeerRef::Uri("http://nodeA:6335".to_string())])]);
        let err =
            resolve_explicit_placement(&placement, Some(1), Some(1), ShardingMethod::Auto, &peers)
                .unwrap_err();
        assert!(err.to_string().contains("more than one peer"), "got: {err}",);
    }

    #[test]
    fn duplicates_are_caught_after_resolution() {
        // The same peer once by id and once by URI: only detectable after resolving.
        let placement = BTreeMap::from([(
            0,
            vec![
                PeerRef::Id(11),
                PeerRef::Uri("http://nodeA:6335".to_string()),
            ],
        )]);
        let err = resolve(placement, 1, 2).unwrap_err();
        assert!(err.to_string().contains("more than once"), "got: {err}");
    }

    #[test]
    fn coverage_and_factor_mismatches_are_refused() {
        let placement = BTreeMap::from([(0, vec![PeerRef::Id(11)])]);
        let err = resolve(placement.clone(), 2, 1).unwrap_err();
        assert!(err.to_string().contains("every shard id"), "got: {err}");

        let err = resolve(placement.clone(), 1, 2).unwrap_err();
        assert!(err.to_string().contains("replication_factor"), "got: {err}");

        let err =
            resolve_explicit_placement(&placement, None, Some(1), ShardingMethod::Auto, &peers())
                .unwrap_err();
        assert!(err.to_string().contains("shard_number"), "got: {err}");
    }

    #[test]
    fn unknown_ids_are_still_refused() {
        let placement = BTreeMap::from([(0, vec![PeerRef::Id(99)])]);
        let err = resolve(placement, 1, 1).unwrap_err();
        assert!(err.to_string().contains("unknown peer 99"), "got: {err}");
    }
}
