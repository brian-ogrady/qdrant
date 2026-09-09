pub mod adopt_manifest;
pub mod channel_service;
pub mod collection_shard_distribution;
mod conversions;
pub mod dummy_shard;
pub mod forward_proxy_shard;
pub mod local_shard;
pub mod proxy_shard;
pub mod queue_proxy_shard;
pub mod remote_shard;
pub mod replica_set;
pub mod resharding;
pub mod resolve;
pub mod shard;
pub mod shard_config;
pub mod shard_holder;
pub mod shard_trait;
pub mod telemetry;
pub mod transfer;
pub mod update_tracker;

#[cfg(test)]
mod test;

use std::path::{Path, PathBuf};

use channel_service::ChannelService;
use common::defaults;
use fs_err::tokio as tokio_fs;
use shard::ShardId;
use tokio::time::{sleep_until, timeout_at};
use transfer::ShardTransferConsensus;

use crate::operations::types::{CollectionError, CollectionResult};
use crate::shards::shard_config::ShardConfig;

pub type CollectionId = String;

/// Path to a shard directory
pub fn shard_path(collection_path: &Path, shard_id: ShardId) -> PathBuf {
    collection_path.join(shard_id.to_string())
}

/// Path to a shard directory
pub fn shard_initializing_flag_path(collection_path: &Path, shard_id: ShardId) -> PathBuf {
    collection_path.join(format!("shard_{shard_id}.initializing"))
}

/// Durably create the shard's `.initializing` (dirty) flag. Adoption writes this in Phase A (as soon
/// as a shard is staged), *before* the install begins, so a crash before the install completes —
/// including while the load task is still
/// queued behind the load governor, when no data has moved yet — leaves the shard marked dirty. The
/// salvage check in [`replica_set::ShardReplicaSet::load`] then treats such a shard as incomplete
/// (drives it `Dead`) rather than activating an empty, never-installed shard. The restore path
/// removes this flag on full success; it is also removed when the shard is dropped
/// (`drop_and_remove_shard`) or rebuilt by transfer. The load-bearing direction — presence ⇒ "data
/// not yet complete" — holds at every one of those removal sites. Idempotent: `File::create`
/// truncates an existing flag, matching restore's own re-create.
pub fn write_shard_initializing_flag(
    collection_path: &Path,
    shard_id: ShardId,
) -> std::io::Result<()> {
    let path = shard_initializing_flag_path(collection_path, shard_id);
    let file = fs_err::File::create(&path)?;
    file.sync_all()?;
    // Propagate the directory fsync: adoption gates arming the in-progress marker on this returning
    // `Ok`, and that gate is only meaningful if the flag's *directory entry* is durable — not just
    // its inode. Without it, a crash could leave the in-progress marker's dirent persisted while the
    // flag's dirent is lost, reopening the empty-salvage hole this flag exists to close.
    fs_err::File::open(collection_path)?.sync_all()?;
    Ok(())
}

/// Marker file placed inside an adopted shard directory once its data is fully installed but the
/// shard is still parked in `ManualRecovery` awaiting activation.
///
/// It is the durable, crash-surviving signal that drives the *single activation path* for adopted
/// shards: `Collection::sync_local_state` (re)proposes `Active` for any `ManualRecovery` shard
/// carrying this marker, through the leader-guarded consensus path, every reconcile tick — so a
/// lost activation proposal (this peer was not leader when it fired) or a restart that lands after
/// the load but before activation is healed automatically. It is written only *after* the data is
/// installed, so it can never cause a shard whose data is absent (a failed or mid-crash adoption)
/// to be activated empty; and it is adoption-specific, so it never disturbs a shard another flow
/// (e.g. snapshot recovery) parks in `ManualRecovery` transiently. Removed once the shard reaches
/// `Active`.
pub const ADOPT_ACTIVATE_MARKER_FILE: &str = ".adopt_activate_pending";

/// Path to the [`ADOPT_ACTIVATE_MARKER_FILE`] inside `shard_path`.
pub fn adopt_activate_marker_path(shard_path: &Path) -> PathBuf {
    shard_path.join(ADOPT_ACTIVATE_MARKER_FILE)
}

/// Durably record that an adopted shard's data is installed and it is awaiting activation. fsyncs
/// the file (and best-effort the parent directory) so the marker survives a crash.
pub fn write_adopt_activate_marker(shard_path: &Path) -> std::io::Result<()> {
    let path = adopt_activate_marker_path(shard_path);
    let file = fs_err::File::create(&path)?;
    file.sync_all()?;
    // Best-effort directory fsync so the new entry itself is durable.
    if let Ok(dir) = fs_err::File::open(shard_path) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Whether `shard_path` carries the adoption activation marker.
pub fn has_adopt_activate_marker(shard_path: &Path) -> bool {
    adopt_activate_marker_path(shard_path).exists()
}

/// Marker placed in an adopted shard directory when its Phase B load **failed** definitively (a
/// corrupt/unloadable artifact that passed Phase A's shape check but errored during segment load).
///
/// It is the mirror of [`ADOPT_ACTIVATE_MARKER_FILE`]: the reconciler drives a `ManualRecovery`
/// shard carrying this marker to `Dead`, so a failed adoption becomes a distinct, queryable
/// unhealthy state instead of sitting in `ManualRecovery` — where it is indistinguishable from a
/// healthy in-progress load. The file's contents are the failure reason, kept for forensics (it is
/// left in place on the resulting `Dead` shard; cleared only when the shard is re-adopted, i.e. its
/// data is cleared, or when it successfully activates). RF=1 (adoption forces it) means the sole
/// replica can be set `Dead` — the "last source of truth" guard does not fire for a non-active
/// (`ManualRecovery`) replica.
pub const ADOPT_FAILED_MARKER_FILE: &str = ".adopt_failed";

/// Path to the [`ADOPT_FAILED_MARKER_FILE`] inside `shard_path`.
pub fn adopt_failed_marker_path(shard_path: &Path) -> PathBuf {
    shard_path.join(ADOPT_FAILED_MARKER_FILE)
}

/// Durably record that an adopted shard's load failed, with `reason` as the file contents (fsynced,
/// plus a best-effort parent-directory fsync, so it survives a crash and the reconciler still
/// drives the shard to `Dead` on restart).
pub fn write_adopt_failed_marker(shard_path: &Path, reason: &str) -> std::io::Result<()> {
    let path = adopt_failed_marker_path(shard_path);
    let mut file = fs_err::File::create(&path)?;
    std::io::Write::write_all(&mut file, reason.as_bytes())?;
    file.sync_all()?;
    if let Ok(dir) = fs_err::File::open(shard_path) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Whether `shard_path` carries the adoption-failure marker.
pub fn has_adopt_failed_marker(shard_path: &Path) -> bool {
    adopt_failed_marker_path(shard_path).exists()
}

/// The recorded failure reason, if the marker is present and readable.
pub fn read_adopt_failed_marker(shard_path: &Path) -> Option<String> {
    fs_err::read_to_string(adopt_failed_marker_path(shard_path)).ok()
}

/// Remove the adoption-failure marker (durable, like [`remove_adopt_activate_marker`]).
pub fn remove_adopt_failed_marker(shard_path: &Path) {
    let path = adopt_failed_marker_path(shard_path);
    match fs_err::remove_file(&path) {
        Ok(()) => {
            if let Ok(dir) = fs_err::File::open(shard_path) {
                let _ = dir.sync_all();
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => log::warn!(
            "failed to remove adopt failure marker {}: {err}",
            path.display(),
        ),
    }
}

/// Remove the marker once the shard is active. Durable: the unlink is followed by a best-effort
/// parent-directory fsync, mirroring [`write_adopt_activate_marker`]. Without that fsync a crash
/// could resurrect a removed marker on an already-`Active` shard, and a stale marker is NOT inert
/// — snapshot recovery re-parks shards in `ManualRecovery`, so a leftover marker could make the
/// reconciler activate one mid-recovery. (Its `LocalShard` gate in
/// `ShardReplicaSet::is_adopt_activation_pending` is the primary defense; durable removal keeps a
/// stale marker from lingering in the first place.)
pub fn remove_adopt_activate_marker(shard_path: &Path) {
    let path = adopt_activate_marker_path(shard_path);
    match fs_err::remove_file(&path) {
        Ok(()) => {
            // Persist the removal so a crash cannot bring the entry back.
            if let Ok(dir) = fs_err::File::open(shard_path) {
                let _ = dir.sync_all();
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => log::warn!(
            "failed to remove adopt activation marker {}: {err}",
            path.display(),
        ),
    }
}

/// Marker recording that an adopted shard's Phase B install/load is **in progress** — written
/// durably *before* the data is moved into place and replaced by a terminal marker
/// ([`ADOPT_ACTIVATE_MARKER_FILE`] on success, [`ADOPT_FAILED_MARKER_FILE`] on failure).
///
/// Unlike the two terminal markers, this one lives in the **collection** directory (keyed by shard
/// id), not the shard directory: the install renames/moves data into the shard dir and the failure
/// handler clears it, so a marker there could be clobbered mid-install — exactly the window this
/// marker must survive. It is the missing "adoption underway" signal: if the process crashes (or the
/// Phase B task panics) before a terminal marker is written, the shard would otherwise sit in
/// `ManualRecovery` forever, indistinguishable from a healthy transient re-park. On the next load
/// `ShardReplicaSet::load` finds this marker with no terminal marker and resolves the interrupted
/// adoption (salvages a completed-but-unmarked load, or drives an incomplete one to `Dead`).
pub fn adopt_in_progress_marker_path(collection_path: &Path, shard_id: ShardId) -> PathBuf {
    collection_path.join(format!(".adopt_in_progress_{shard_id}"))
}

/// Durably record that an adopted shard's Phase B install is underway (fsyncs the file and,
/// best-effort, the collection directory, so the marker survives a crash mid-install).
pub fn write_adopt_in_progress_marker(
    collection_path: &Path,
    shard_id: ShardId,
) -> std::io::Result<()> {
    let path = adopt_in_progress_marker_path(collection_path, shard_id);
    let file = fs_err::File::create(&path)?;
    file.sync_all()?;
    if let Ok(dir) = fs_err::File::open(collection_path) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Whether the collection directory carries the in-progress marker for `shard_id`.
pub fn has_adopt_in_progress_marker(collection_path: &Path, shard_id: ShardId) -> bool {
    adopt_in_progress_marker_path(collection_path, shard_id).exists()
}

/// Remove the in-progress marker (durable: unlink followed by a best-effort collection-directory
/// fsync, so a crash cannot resurrect it). No-op if absent.
pub fn remove_adopt_in_progress_marker(collection_path: &Path, shard_id: ShardId) {
    let path = adopt_in_progress_marker_path(collection_path, shard_id);
    match fs_err::remove_file(&path) {
        Ok(()) => {
            if let Ok(dir) = fs_err::File::open(collection_path) {
                let _ = dir.sync_all();
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => log::warn!(
            "failed to remove adopt in-progress marker {}: {err}",
            path.display(),
        ),
    }
}

/// Verify that a shard exists by loading its configuration.
/// Returns the path to the shard if it exists.
pub async fn check_shard_path(
    collection_path: &Path,
    shard_id: ShardId,
) -> CollectionResult<PathBuf> {
    let path = shard_path(collection_path, shard_id);
    let shard_config_opt = ShardConfig::load(&path)?;
    if shard_config_opt.is_some() {
        Ok(path)
    } else {
        Err(CollectionError::service_error(format!(
            "No shard found: {shard_id} at {collection_path}",
            shard_id = shard_id,
            collection_path = collection_path.display()
        )))
    }
}

pub async fn create_shard_dir(
    collection_path: &Path,
    shard_id: ShardId,
) -> CollectionResult<PathBuf> {
    let shard_path = shard_path(collection_path, shard_id);
    match tokio_fs::create_dir(&shard_path).await {
        Ok(_) => Ok(shard_path),
        // If the directory already exists, remove it and create it again
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            log::warn!("Shard path already exists, removing and creating again: {shard_path:?}");
            tokio_fs::remove_dir_all(&shard_path)
                .await
                .map_err(CollectionError::from)?;
            tokio_fs::create_dir(&shard_path)
                .await
                .map_err(CollectionError::from)?;
            Ok(shard_path)
        }
        Err(e) => Err(CollectionError::from(e)),
    }
}

/// Await for consensus to synchronize across all peers
///
/// This will take the current consensus state of this node. It then explicitly waits on all other
/// nodes to reach the same (or later) consensus.
///
/// If awaiting on other nodes fails for any reason, this simply continues after the consensus
/// timeout.
///
/// # Cancel safety
///
/// This function is cancel safe.
async fn await_consensus_sync(
    consensus: &dyn ShardTransferConsensus,
    channel_service: &ChannelService,
) {
    let wait_until = tokio::time::Instant::now() + defaults::CONSENSUS_META_OP_WAIT;
    let sync_consensus =
        timeout_at(wait_until, consensus.await_consensus_sync(channel_service)).await;

    match sync_consensus {
        Ok(Ok(_)) => log::trace!("All peers reached consensus"),
        // Failed to sync explicitly, waiting until timeout to assume synchronization
        Ok(Err(err)) => {
            log::warn!("All peers failed to synchronize consensus, waiting until timeout: {err}");
            sleep_until(wait_until).await;
        }
        // Reached timeout, assume consensus is synchronized
        Err(err) => {
            log::warn!(
                "All peers failed to synchronize consensus, continuing after timeout: {err}"
            );
        }
    }
}
