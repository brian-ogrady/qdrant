use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use api::grpc::qdrant::WaitOnConsensusCommitRequest;
use api::grpc::qdrant::qdrant_internal_client::QdrantInternalClient;
use api::grpc::transport_channel_pool::{PoolInterceptor, TransportChannelPool};
use futures::Future;
use futures::future::try_join_all;
use semver::Version;
use tonic::codegen::InterceptedService;
use tonic::transport::{Channel, Uri};
use tonic::{Request, Status};
use url::Url;

use crate::operations::types::{CollectionError, CollectionResult, PeerMetadata};
use crate::shards::shard::PeerId;

#[derive(Clone)]
pub struct ChannelService {
    // Shared with consensus_state
    pub id_to_address: Arc<parking_lot::RwLock<HashMap<PeerId, Uri>>>,
    // Shared with consensus_state
    pub id_to_metadata: Arc<parking_lot::RwLock<HashMap<PeerId, PeerMetadata>>>,
    pub channel_pool: Arc<TransportChannelPool>,
    /// Port at which the public REST API is exposed for the current peer.
    pub current_rest_port: u16,
    /// Indicates whether the TLS is enabled for the public REST API.
    pub rest_tls_enabled: bool,

    /// Instance wide API key if configured, must be used with care.
    pub api_key: Option<String>,

    /// Alternative API key, works the same as `api_key`. Intended for rolling key updates.
    pub alt_api_key: Option<String>,
}

impl ChannelService {
    /// Construct a new channel service with the given REST port.
    pub fn new(
        current_rest_port: u16,
        rest_tls_enabled: bool,
        api_key: Option<String>,
        alt_api_key: Option<String>,
    ) -> Self {
        Self {
            id_to_address: Default::default(),
            id_to_metadata: Default::default(),
            channel_pool: Default::default(),
            current_rest_port,
            rest_tls_enabled,
            api_key,
            alt_api_key,
        }
    }

    pub async fn remove_peer(&self, peer_id: PeerId) {
        let removed = self.id_to_address.write().remove(&peer_id);
        if let Some(uri) = removed {
            self.channel_pool.drop_pool(&uri).await;
        }
    }

    /// Wait until all other known peers reach the given commit
    ///
    /// # Errors
    ///
    /// This errors if:
    /// - any of the peers is not on the same term
    /// - waiting takes longer than the specified timeout
    /// - any of the peers cannot be reached
    ///
    /// # Cancel safety
    ///
    /// This method is cancel safe.
    pub async fn await_commit_on_all_peers(
        &self,
        this_peer_id: PeerId,
        commit: u64,
        term: u64,
        timeout: Duration,
    ) -> CollectionResult<()> {
        let requests = self
            .id_to_address
            .read()
            .keys()
            .filter(|id| **id != this_peer_id)
            // The collective timeout at the bottom of this function handles actually timing out.
            // Since an explicit timeout must be given here as well, it is multiplied by two to
            // give the collective timeout some space.
            .map(|peer_id| self.await_commit_on_peer(*peer_id, commit, term, timeout * 2))
            .collect::<Vec<_>>();
        let responses = try_join_all(requests);

        // Handle requests with timeout
        tokio::time::timeout(timeout, responses)
            .await
            // Timeout error
            .map_err(|_elapsed| CollectionError::Timeout {
                description: "Failed to wait for consensus commit on all peers, timed out.".into(),
            })?
            // Await consensus error
            .map_err(|err| {
                CollectionError::service_error(format!(
                    "Failed to wait for consensus commit on peer: {err}"
                ))
            })?;
        Ok(())
    }

    /// Wait until the given peer reaches the given commit
    ///
    /// # Errors
    ///
    /// This errors if the given peer is on a different term. Also errors if the peer cannot be reached.
    ///
    /// # Cancel safety
    ///
    /// This method is cancel safe.
    async fn await_commit_on_peer(
        &self,
        peer_id: PeerId,
        commit: u64,
        term: u64,
        timeout: Duration,
    ) -> CollectionResult<()> {
        let response = self
            .with_qdrant_client(peer_id, |mut client| async move {
                let request = WaitOnConsensusCommitRequest {
                    commit: commit as i64,
                    term: term as i64,
                    timeout: timeout.as_secs() as i64,
                };
                client.wait_on_consensus_commit(Request::new(request)).await
            })
            .await
            .map_err(|err| {
                CollectionError::service_error(format!(
                    "Failed to wait for consensus commit on peer {peer_id}: {err}"
                ))
            })?
            .into_inner();

        // Create error if wait request failed
        if !response.ok {
            return Err(CollectionError::service_error(format!(
                "Failed to wait for consensus commit on peer {peer_id}, has diverged commit/term or timed out."
            )));
        }
        Ok(())
    }

    pub async fn with_qdrant_client<T, O: Future<Output = Result<T, Status>>>(
        &self,
        peer_id: PeerId,
        f: impl Fn(QdrantInternalClient<InterceptedService<Channel, PoolInterceptor>>) -> O,
    ) -> CollectionResult<T> {
        let address = self
            .id_to_address
            .read()
            .get(&peer_id)
            .ok_or_else(|| CollectionError::service_error("Address for peer ID is not found."))?
            .clone();
        self.channel_pool
            .with_channel(&address, |channel| {
                let client = QdrantInternalClient::new(channel);
                let client = client.max_decoding_message_size(usize::MAX);
                f(client)
            })
            .await
            .map_err(Into::into)
    }

    /// Check whether all peers are running at least the given version
    ///
    /// If the version is not known for any peer, this returns `false`.
    /// Peer versions are known since 1.9 and up.
    pub fn all_peers_at_version(&self, version: &Version) -> bool {
        let id_to_address = self.id_to_address.read();
        let id_to_metadata = self.id_to_metadata.read();

        // Ensure there aren't more peer addresses than metadata
        if id_to_address.len() > id_to_metadata.len() {
            let peers_without_metadata: HashMap<_, _> = id_to_address
                .iter()
                .filter(|(id, _uri)| !id_to_metadata.contains_key(id))
                .collect();
            log::info!(
                "Not all peers at version:{version} because there are peers without metadata:{peers_without_metadata:?}"
            );
            return false;
        }

        let all = id_to_metadata
            .values()
            .all(|metadata| &metadata.version >= version);

        if !all {
            log::info!("Not all peers at version:{version} peers:{id_to_metadata:?}");
        }

        all
    }

    /// Check whether every peer whose version we *know* is running at least the given version.
    ///
    /// Unlike [`Self::all_peers_at_version`], a peer that has not published its version yet does not
    /// count against this. The two answer different questions: that one asks "can I prove every peer
    /// is new enough", this one asks "do I know of any peer that is too old".
    pub fn all_known_peers_at_version(&self, version: &Version) -> bool {
        let id_to_metadata = self.id_to_metadata.read();

        let all = id_to_metadata
            .values()
            .all(|metadata| &metadata.version >= version);

        if !all {
            log::info!("Not all known peers at version:{version} peers:{id_to_metadata:?}");
        }

        all
    }

    /// Check whether the specified peer is running at least the given version
    ///
    /// If the version is not known for the peer, this returns `false`.
    /// Peer versions are known since 1.9 and up.
    pub fn peer_is_at_version(&self, peer_id: PeerId, version: &Version) -> bool {
        self.id_to_metadata
            .read()
            .get(&peer_id)
            .is_some_and(|metadata| &metadata.version >= version)
    }

    /// Get the REST address for the current peer.
    pub fn current_rest_address(&self, this_peer_id: PeerId) -> CollectionResult<Url> {
        // Get local peer URI
        let local_peer_uri = self
            .id_to_address
            .read()
            .get(&this_peer_id)
            .cloned()
            .ok_or_else(|| {
                CollectionError::service_error(format!(
                    "Cannot determine REST address, this peer not found in cluster by ID {this_peer_id} ",
                ))
            })?;

        // Construct REST URL from URI
        let mut url = Url::parse(&local_peer_uri.to_string()).expect("Malformed URL");
        url.set_port(Some(self.current_rest_port))
            .map_err(|()| {
                CollectionError::service_error(format!(
                    "Cannot determine REST address, cannot specify port on address {url} for peer ID {this_peer_id}",
                ))
            })?;
        let scheme = if self.rest_tls_enabled {
            "https"
        } else {
            "http"
        };
        url.set_scheme(scheme).map_err(|()| {
            CollectionError::service_error(format!(
                "Cannot determine REST address, cannot set {scheme} scheme on address {url} for peer ID {this_peer_id}",
            ))
        })?;

        Ok(url)
    }

    pub fn other_peers(&self, this_peer_id: PeerId) -> Vec<PeerId> {
        self.id_to_address
            .read()
            .keys()
            .filter(|id| **id != this_peer_id)
            .copied()
            .collect()
    }

    pub fn request_timeout(&self) -> Duration {
        self.channel_pool.request_timeout()
    }
}

#[cfg(any(test, feature = "testing"))]
impl Default for ChannelService {
    fn default() -> Self {
        Self {
            id_to_address: Default::default(),
            id_to_metadata: Default::default(),
            channel_pool: Default::default(),
            current_rest_port: 6333,
            rest_tls_enabled: false,
            api_key: None,
            alt_api_key: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATE: &str = "1.19.1";

    fn gate() -> Version {
        Version::parse(GATE).unwrap()
    }

    fn with_peers(peers: &[(PeerId, Option<&str>)]) -> ChannelService {
        let service = ChannelService::new(6333, false, None, None);
        {
            let mut addresses = service.id_to_address.write();
            let mut metadata = service.id_to_metadata.write();
            for &(peer_id, version) in peers {
                // Every peer has an address. Only a peer that has published has metadata — that
                // asymmetry is the whole subject of these tests.
                addresses.insert(
                    peer_id,
                    format!("http://127.0.0.1:{peer_id}").parse().unwrap(),
                );
                if let Some(version) = version {
                    metadata.insert(
                        peer_id,
                        PeerMetadata {
                            version: Version::parse(version).unwrap(),
                        },
                    );
                }
            }
        }
        service
    }

    /// The reason this function exists. A peer publishes its version from its own consensus tick, so a
    /// brand-new peer is silent for the whole of its join — and `all_peers_at_version` reports a fully
    /// upgraded cluster as mixed for that entire time. Measured against a real cluster before this
    /// existed, 29 of 30 collections created during one peer join silently took the built-in hash ring
    /// scale instead of the configured one, permanently, because that answer is persisted.
    ///
    /// Asserting both functions on the same input on purpose: if they ever agree here, the distinction
    /// has been lost and the join window is back.
    #[test]
    fn a_peer_that_has_not_published_its_version_does_not_block() {
        // Peer 2 has just joined: address known, nothing published yet.
        let service = with_peers(&[(1, Some(GATE)), (2, None)]);

        assert!(
            service.all_known_peers_at_version(&gate()),
            "a silent peer must not be treated as too old",
        );
        assert!(
            !service.all_peers_at_version(&gate()),
            "the strict check is expected to still refuse here — if it does not, these two functions \
             no longer differ and this test has stopped testing anything",
        );
    }

    /// What the gate is actually for. A peer that has been running has published its version, so a
    /// rolling upgrade in progress is visible through metadata that is *present* — which is why
    /// ignoring absent metadata above does not weaken this.
    #[test]
    fn a_peer_known_to_be_older_still_blocks() {
        let service = with_peers(&[(1, Some(GATE)), (2, Some("1.18.0"))]);

        assert!(
            !service.all_known_peers_at_version(&gate()),
            "a peer whose published version is too old must block",
        );

        // ...and it keeps blocking even with a silent peer alongside it, so the leniency above cannot
        // be used to smuggle a genuinely old cluster past the gate.
        let service = with_peers(&[(1, Some(GATE)), (2, Some("1.18.0")), (3, None)]);
        assert!(
            !service.all_known_peers_at_version(&gate()),
            "a known-old peer must block regardless of any silent peer",
        );
    }

    /// A newer peer satisfies a gate, and an exactly-equal version does too.
    #[test]
    fn known_peers_at_or_above_the_gate_pass() {
        assert!(
            with_peers(&[(1, Some(GATE)), (2, Some("1.20.0"))]).all_known_peers_at_version(&gate())
        );
        assert!(
            with_peers(&[(1, Some(GATE)), (2, Some(GATE))]).all_known_peers_at_version(&gate())
        );
    }

    /// Documented consequence rather than an endorsement: with nothing published at all the check is
    /// vacuously true, so during the bootstrap of a fresh cluster the gate passes without having
    /// verified any version. `all_peers_at_version` behaves the same way once both maps are empty.
    #[test]
    fn no_published_versions_at_all_is_vacuously_true() {
        assert!(with_peers(&[]).all_known_peers_at_version(&gate()));
        assert!(with_peers(&[(1, None), (2, None)]).all_known_peers_at_version(&gate()));
    }
}
