//! Verify that each shard's artifact is serving as that shard, and plan where artifacts go.
//!
//! # Why this exists
//!
//! Installing shard N's artifact onto the peer that consensus assigned shard M is a **silent**
//! failure. Reads do not catch it: `Collection::retrieve` calls `select_shards` with no shard
//! key, which fans out to every shard (`lib/collection/src/collection/point_ops.rs:463`), so a
//! misplaced point is still found. Search is the same.
//!
//! Writes are what route through the hash ring. An upsert of an id that already exists in the
//! wrong shard sends the new copy to the ring-chosen shard and leaves the stale one behind, so
//! the collection ends up holding two points with one id — permanently, because
//! `deduplicate_points` operates within a `SegmentHolder` and never across shards.
//!
//! With one shard per node and machines that may be replaced between runs, that mapping is
//! re-established by hand every time. This module makes it checkable.
//!
//! # How the check works
//!
//! `POST /collections/{c}/shards/{shard}/points` reads from **one specific shard** instead of
//! fanning out (`src/actix/api/local_shard_api.rs:36`, registered on the public API at
//! `src/actix/mod.rs:165`). So for ids taken from shard N's own scatter output we can ask shard N
//! directly and assert they are there — a positive test of placement, which no fan-out read can
//! provide.
//!
//! It also asks a *different* shard for the same ids and asserts they are absent. Without that,
//! the positive half would pass even if the endpoint silently fanned out too.
//!
//! # Why this takes a URL per peer
//!
//! That endpoint is **local-only**. `ShardSelectorInternal::ShardId` makes `retrieve` pass
//! `local_only = true` (`collection/point_ops.rs:484`), and `execute_and_resolve_read_operation`
//! then short-circuits to `execute_local_read_operation`
//! (`replica_set/execute_read_operation.rs:52`) — it never forwards to the peer that owns the
//! shard. Asking a peer about a shard it does not hold is an error, not a fan-out.
//!
//! So a single URL can only verify the shards that one peer happens to own. In the topology this
//! tool targets — one shard per node — that is exactly one shard out of ten, and the other nine
//! would fail for a reason that has nothing to do with whether placement is correct.
//!
//! `verify` therefore takes every peer's REST URL, asks each which shards it holds, and sends each
//! shard's query to its owner. Ownership is discovered rather than assumed: the peer URIs in
//! `/cluster` are internal gRPC addresses, which are frequently not the addresses an operator can
//! reach, so they cannot be reused as REST endpoints.

use std::collections::BTreeMap;

use anyhow::{Context as _, Result, anyhow, bail};
use collection::shards::shard::ShardId;
use segment::types::PointIdType;
use serde_json::{Value, json};

use crate::config::LoadedConfig;
use crate::partfile::PartReader;
use crate::ring::ShardRouter;
use crate::scatter::ScatterLayout;

/// Outcome of a placement check.
#[derive(Debug, Default)]
pub struct PlacementReport {
    /// Distinct peers reached, which is not `urls.len()`: several URLs can name one peer.
    pub peers_queried: usize,
    pub shards_checked: usize,
    pub ids_checked: usize,
    /// Ids missing from the shard the ring assigned them to.
    pub misplaced: Vec<(ShardId, PointIdType)>,
    /// Ids found in a shard the ring did *not* assign them to.
    pub leaked: Vec<(ShardId, PointIdType)>,
}

impl PlacementReport {
    pub fn is_correct(&self) -> bool {
        self.misplaced.is_empty() && self.leaked.is_empty()
    }
}

/// Which peer URL to send each shard's local-only query to.
///
/// Built by asking every supplied peer which shards it reports as local. A shard that no peer
/// claims is an error rather than something to skip: silently not checking a shard is the failure
/// this whole module exists to prevent.
fn resolve_shard_urls(
    urls: &[String],
    collection: &str,
    shard_count: ShardId,
    replication_factor: u32,
) -> Result<BTreeMap<ShardId, String>> {
    let mut owners: BTreeMap<ShardId, String> = BTreeMap::new();
    // Keyed by the peer id each URL reports, not by the URL text. One peer is reachable under
    // many spellings — `localhost` and `127.0.0.1`, a hostname and its alias, or simply the same
    // URL listed twice in a generated ten-node command line. Comparing URL strings would read
    // those as distinct peers claiming the same shard and report a duplicate install that does
    // not exist.
    let mut peer_urls: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    let mut shard_peers: BTreeMap<ShardId, Vec<u64>> = BTreeMap::new();

    for url in urls {
        let base = url.trim_end_matches('/').to_string();
        let cluster: Value = http_get(&format!("{base}/collections/{collection}/cluster"))
            .with_context(|| format!("cannot read cluster info for {collection} from {base}"))?;

        let peer_id = cluster
            .pointer("/result/peer_id")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow!("{base} reported no peer_id for '{collection}'"))?;

        if let Some(first) = peer_urls.get(&peer_id) {
            log::debug!("{base} is peer {peer_id}, already reached via {first}; skipping");
            continue;
        }
        peer_urls.insert(peer_id, base.clone());

        let local = cluster
            .pointer("/result/local_shards")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("{base} reported no local_shards for '{collection}'"))?;

        for shard in local {
            let Some(shard_id) = shard.get("shard_id").and_then(Value::as_u64) else {
                continue;
            };
            let shard_id = shard_id as ShardId;
            let holders = shard_peers.entry(shard_id).or_default();
            holders.push(peer_id);
            // Up to `replication_factor` distinct peers legitimately hold a shard — those are
            // its replicas. One *more* than that is the shared-filesystem mistake: the same
            // artifact installed on an extra node, where writes reach only one copy and the
            // peers diverge. Report it rather than picking one.
            if holders.len() > replication_factor as usize {
                let others: Vec<String> = holders
                    .iter()
                    .map(|peer| {
                        let url = peer_urls.get(peer).map(String::as_str).unwrap_or("?");
                        format!("{peer} ({url})")
                    })
                    .collect();
                bail!(
                    "shard {shard_id} is reported local by {} peers ({}), but the document's \
                     replication_factor is {replication_factor}; the artifact appears to be \
                     installed on more peers than the collection has replicas, so writes will \
                     reach only some copies and they will diverge",
                    holders.len(),
                    others.join(", "),
                );
            }
            // Any replica can answer the shard-scoped reads below; keep the first one seen.
            owners.entry(shard_id).or_insert_with(|| base.clone());
        }
    }

    let missing: Vec<_> = (0..shard_count)
        .filter(|id| !owners.contains_key(id))
        .collect();
    if !missing.is_empty() {
        bail!(
            "no supplied peer holds shard(s) {missing:?}. The shard-scoped read endpoint is \
             local-only, so every peer's REST URL must be passed with --url; got {urls:?}",
        );
    }

    Ok(owners)
}

/// Sample ids from each shard's scatter output and confirm the live cluster agrees.
///
/// `urls` must cover every peer holding a shard of the collection — see the module docs for why a
/// single URL is not enough outside a one-peer cluster.
pub fn verify(
    config: &LoadedConfig,
    router: &ShardRouter,
    layout: &ScatterLayout,
    urls: &[String],
    collection: &str,
    per_shard: usize,
) -> Result<PlacementReport> {
    let mut report = PlacementReport::default();
    let shard_count = router.shard_count() as ShardId;
    let replication_factor = config.config.params.replication_factor.get();
    let owners = resolve_shard_urls(urls, collection, shard_count, replication_factor)?;
    report.peers_queried = owners
        .values()
        .collect::<std::collections::HashSet<_>>()
        .len();

    for shard_id in 0..shard_count {
        let ids = sample_ids(config, layout, shard_id, per_shard)?;
        if ids.is_empty() {
            log::debug!("shard {shard_id} has no scattered points, skipping");
            continue;
        }
        report.shards_checked += 1;
        report.ids_checked += ids.len();

        // Positive: the ring says these belong to `shard_id`, so that shard must hold them.
        let url = &owners[&shard_id];
        let present = ids_present_in_shard(url, collection, shard_id, &ids)?;
        for id in &ids {
            if !present.contains(id) {
                report.misplaced.push((shard_id, *id));
            }
        }

        // Negative: a different shard must not hold them. This is what proves the endpoint is
        // really shard-scoped rather than fanning out, without which the positive half is
        // vacuous. Asked of *that* shard's owner, which may be a different peer.
        if shard_count > 1 {
            let other = (shard_id + 1) % shard_count;
            let leaked = ids_present_in_shard(&owners[&other], collection, other, &ids)?;
            for id in &ids {
                if leaked.contains(id) {
                    report.leaked.push((other, *id));
                }
            }
        }
    }

    if report.shards_checked == 0 {
        bail!(
            "no scattered points found under {}; placement cannot be verified without the \
             scatter output the artifacts were built from",
            layout.state_path().display(),
        );
    }

    Ok(report)
}

/// Read up to `limit` point ids out of one shard's part files.
///
/// Sourced from the *artifacts* rather than from the collection, because the question is
/// whether shard N's artifact is serving as shard N. Ids read back from the collection would
/// already have lost that association.
fn sample_ids(
    config: &LoadedConfig,
    layout: &ScatterLayout,
    shard_id: ShardId,
    limit: usize,
) -> Result<Vec<PointIdType>> {
    let dir = layout.shard_dir(shard_id);
    let entries = match fs_err::read_dir(&dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };

    let mut ids = Vec::new();
    for entry in entries {
        if ids.len() >= limit {
            break;
        }
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()).is_some() {
            // Skips `.meta` and any `.tmp` debris.
            continue;
        }

        let mut reader = PartReader::open(&path, &config.part_fingerprint)?;
        while ids.len() < limit {
            match reader.next_record()? {
                Some(record) => ids.push(record.id),
                None => break,
            }
        }
    }

    Ok(ids)
}

/// Which of `ids` a specific shard holds.
fn ids_present_in_shard(
    url: &str,
    collection: &str,
    shard_id: ShardId,
    ids: &[PointIdType],
) -> Result<std::collections::HashSet<PointIdType>> {
    let endpoint = format!(
        "{}/collections/{collection}/shards/{shard_id}/points",
        url.trim_end_matches('/'),
    );

    let body = json!({
        "ids": ids,
        "with_payload": false,
        "with_vector": false,
    });

    let response: Value = http_post(&endpoint, &body)?;
    let records = response
        .pointer("/result")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("{endpoint} returned no result array: {response}"))?;

    records
        .iter()
        .map(|record| {
            let id = record
                .get("id")
                .ok_or_else(|| anyhow!("record has no id: {record}"))?;
            serde_json::from_value(id.clone())
                .with_context(|| format!("cannot parse point id {id}"))
        })
        .collect()
}

/// Where each shard's artifact should be installed.
#[derive(Debug)]
pub struct InstallPlan {
    /// This peer, as reported by the collection it was asked about.
    pub queried_peer: u64,
    /// shard id -> (peer id, peer uri)
    pub placement: BTreeMap<ShardId, (u64, String)>,
}

/// Read the cluster's shard-to-peer assignment.
///
/// Consensus decides which peer owns which shard, so the mapping has to be read from the
/// cluster rather than assumed from directory names. With one shard per node and machines that
/// change between runs, reconstructing this by hand is exactly where a silent misplacement
/// comes from.
pub fn install_plan(url: &str, collection: &str) -> Result<InstallPlan> {
    let base = url.trim_end_matches('/');

    let cluster: Value = http_get(&format!("{base}/collections/{collection}/cluster"))?;
    let result = cluster
        .pointer("/result")
        .ok_or_else(|| anyhow!("collection cluster info has no result: {cluster}"))?;

    let queried_peer = result
        .get("peer_id")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("collection cluster info has no peer_id"))?;

    // peer id -> uri, from the top-level cluster endpoint.
    let peers: Value = http_get(&format!("{base}/cluster"))?;
    let mut uris: BTreeMap<u64, String> = BTreeMap::new();
    if let Some(map) = peers.pointer("/result/peers").and_then(Value::as_object) {
        for (peer_id, info) in map {
            if let (Ok(peer_id), Some(uri)) = (
                peer_id.parse::<u64>(),
                info.get("uri").and_then(Value::as_str),
            ) {
                uris.insert(peer_id, uri.to_string());
            }
        }
    }

    let mut placement = BTreeMap::new();

    for local in result
        .get("local_shards")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        if let Some(shard_id) = local.get("shard_id").and_then(Value::as_u64) {
            let uri = uris
                .get(&queried_peer)
                .cloned()
                .unwrap_or_else(|| format!("<this peer {queried_peer}>"));
            placement.insert(shard_id as ShardId, (queried_peer, uri));
        }
    }

    for remote in result
        .get("remote_shards")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        let (Some(shard_id), Some(peer_id)) = (
            remote.get("shard_id").and_then(Value::as_u64),
            remote.get("peer_id").and_then(Value::as_u64),
        ) else {
            continue;
        };
        let uri = uris
            .get(&peer_id)
            .cloned()
            .unwrap_or_else(|| format!("<unknown uri for peer {peer_id}>"));
        placement.insert(shard_id as ShardId, (peer_id, uri));
    }

    if placement.is_empty() {
        bail!("cluster reports no shards for collection '{collection}'");
    }

    Ok(InstallPlan {
        queried_peer,
        placement,
    })
}

fn http_post(endpoint: &str, body: &Value) -> Result<Value> {
    let response = reqwest::blocking::Client::new()
        .post(endpoint)
        .json(body)
        .send()
        .with_context(|| format!("request to {endpoint} failed"))?;

    let status = response.status();
    let value: Value = response
        .json()
        .with_context(|| format!("{endpoint} did not return JSON"))?;

    if !status.is_success() {
        bail!("{endpoint} returned {status}: {value}");
    }
    Ok(value)
}

fn http_get(endpoint: &str) -> Result<Value> {
    let response = reqwest::blocking::Client::new()
        .get(endpoint)
        .send()
        .with_context(|| format!("request to {endpoint} failed"))?;

    let status = response.status();
    let value: Value = response
        .json()
        .with_context(|| format!("{endpoint} did not return JSON"))?;

    if !status.is_success() {
        bail!("{endpoint} returned {status}: {value}");
    }
    Ok(value)
}
