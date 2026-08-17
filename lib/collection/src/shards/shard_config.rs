use std::path::{Path, PathBuf};

use common::fs::{atomic_save_json, read_json};
use common::tar_ext;
use serde::{Deserialize, Serialize};

use crate::operations::types::CollectionResult;
use crate::shards::shard::PeerId;

pub const SHARD_CONFIG_FILE: &str = "shard_config.json";

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub enum ShardType {
    Local,                      // Deprecated
    Remote { peer_id: PeerId }, // Deprecated
    Temporary,                  // Deprecated
    ReplicaSet,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct ShardConfig {
    pub r#type: ShardType,

    /// Hash ring scale the points in this shard directory were placed under.
    /// `None` means the shard predates this field, and is treated as *unknown* on the live path — never
    /// as a mismatch — so existing shards are never flagged. In a *snapshot* the same absence means
    /// something stronger, see [`Self::snapshot_hash_ring_shard_scale`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash_ring_shard_scale: Option<u32>,
}

impl ShardConfig {
    pub fn get_config_path(shard_path: &Path) -> PathBuf {
        shard_path.join(SHARD_CONFIG_FILE)
    }

    pub fn new_replica_set() -> Self {
        Self {
            r#type: ShardType::ReplicaSet,
            hash_ring_shard_scale: None,
        }
    }

    /// As [`Self::new_replica_set`], but records the hash ring scale the shard's points were placed
    /// under.
    pub fn new_replica_set_with_scale(hash_ring_shard_scale: u32) -> Self {
        Self {
            r#type: ShardType::ReplicaSet,
            hash_ring_shard_scale: Some(hash_ring_shard_scale),
        }
    }

    /// Hash ring scale the points in this snapshot were routed at.
    ///
    /// An absent value is not "unknown": it means the snapshot predates the field, and back then the
    /// scale was a compile-time constant, so
    /// [`DEFAULT_HASH_RING_SHARD_SCALE`](crate::hash_ring::DEFAULT_HASH_RING_SHARD_SCALE) is the value
    /// it was actually routed at. Treating it as unknown and skipping the check would silently allow
    /// the corruption this exists to prevent.
    pub fn snapshot_hash_ring_shard_scale(&self) -> u32 {
        self.hash_ring_shard_scale
            .unwrap_or(crate::hash_ring::DEFAULT_HASH_RING_SHARD_SCALE)
    }

    /// Whether this shard's points were placed by a ring at a *different* scale than `expected`.
    pub fn diverges_from_hash_ring_shard_scale(&self, expected: u32) -> bool {
        self.hash_ring_shard_scale
            .is_some_and(|placed_under| placed_under != expected)
    }

    pub fn load(shard_path: &Path) -> CollectionResult<Option<Self>> {
        let config_path = Self::get_config_path(shard_path);
        if !config_path.exists() {
            log::info!("Detected missing shard config file in {shard_path:?}");
            return Ok(None);
        }
        Ok(Some(read_json(&config_path)?))
    }

    pub fn save(&self, shard_path: &Path) -> CollectionResult<()> {
        let config_path = Self::get_config_path(shard_path);
        Ok(atomic_save_json(&config_path, self)?)
    }

    pub async fn save_to_tar(&self, tar: &tar_ext::BuilderExt) -> CollectionResult<()> {
        let bytes = serde_json::to_vec(self)?;
        tar.append_data(bytes, Path::new(SHARD_CONFIG_FILE)).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash_ring::DEFAULT_HASH_RING_SHARD_SCALE;

    /// A shard config written before the field existed came from a collection whose scale was the
    /// hardcoded constant, so that — not "unknown" — is what its points were routed at. If this ever
    /// reported something else, restoring an older shard snapshot would either be refused for no
    /// reason or accepted into a collection that routes its points elsewhere.
    #[test]
    fn absent_scale_means_the_historical_default() {
        let legacy: ShardConfig = serde_json::from_str(r#"{"type":"ReplicaSet"}"#).unwrap();
        assert_eq!(legacy.hash_ring_shard_scale, None);
        assert_eq!(
            legacy.snapshot_hash_ring_shard_scale(),
            DEFAULT_HASH_RING_SHARD_SCALE,
        );
    }

    #[test]
    fn snapshot_config_records_the_scale_it_was_given() {
        let config = ShardConfig::new_replica_set_with_scale(7);
        assert_eq!(config.hash_ring_shard_scale, Some(7));
        assert_eq!(config.snapshot_hash_ring_shard_scale(), 7);

        // Must survive the round trip through the snapshot tar.
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ShardConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, config);
        assert_eq!(parsed.snapshot_hash_ring_shard_scale(), 7);
    }

    /// A config with no recorded scale must read as *unknown* on the live path, never as a mismatch.
    /// Shards created before this field existed have no record, and refusing to serve them would turn
    /// an upgrade into an outage.
    #[test]
    fn an_unrecorded_scale_never_counts_as_divergence() {
        let legacy = ShardConfig::new_replica_set();
        assert_eq!(legacy.hash_ring_shard_scale, None);
        assert!(!legacy.diverges_from_hash_ring_shard_scale(7));
        assert!(!legacy.diverges_from_hash_ring_shard_scale(DEFAULT_HASH_RING_SHARD_SCALE));

        // Serialization is unchanged for such a config, so an older binary reading it sees exactly
        // what it saw before.
        assert_eq!(
            serde_json::to_string(&legacy).unwrap(),
            r#"{"type":"ReplicaSet"}"#,
        );
    }

    /// The whole point of recording it: a shard placed under one scale must be recognisable as not
    /// belonging to a collection now routing at another.
    #[test]
    fn a_recorded_scale_diverges_only_from_a_different_one() {
        let placed_at_7 = ShardConfig::new_replica_set_with_scale(7);

        assert!(!placed_at_7.diverges_from_hash_ring_shard_scale(7));
        assert!(placed_at_7.diverges_from_hash_ring_shard_scale(8));
        assert!(placed_at_7.diverges_from_hash_ring_shard_scale(DEFAULT_HASH_RING_SHARD_SCALE));

        // ...and it survives the round trip through disk, which is where it actually lives.
        let reloaded: ShardConfig =
            serde_json::from_str(&serde_json::to_string(&placed_at_7).unwrap()).unwrap();
        assert!(reloaded.diverges_from_hash_ring_shard_scale(8));
        assert!(!reloaded.diverges_from_hash_ring_shard_scale(7));
    }
}
