//! Manifest describing the collection-level shape a shard artifact was built for.
//!
//! The offline shard-builder stamps one of these into each assembled `shard_{id}` directory. At
//! adoption time (`adopt_shards_from`, or `adopt://` recovery) the server compares it against the
//! live collection's config and refuses a mismatch — otherwise an artifact built for a different
//! topology (say a 3-shard build's `shard_0` dropped into a 1-shard collection) would install
//! cleanly and silently serve a wrong, misrouted subset of the data.
//!
//! Only the fields that decide whether the artifact's data is *correct* for the collection are
//! recorded: routing (`shard_number`, `hash_ring_shard_scale`, `sharding_method`), the shard's own
//! id, and the dense/sparse vector shapes. Placement, optimizer thresholds and replication factor
//! are deliberately excluded — they are adjustable after the build (see `retarget`) and do not
//! change which point belongs where or how a stored vector is interpreted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use segment::data_types::modifier::Modifier;
use segment::types::{Distance, HnswConfig, Memory, MultiVectorConfig, QuantizationConfig};
use serde::{Deserialize, Serialize};

use crate::config::{CollectionParams, ShardingMethod};
use crate::operations::config_diff::DiffConfig;
use crate::operations::types::Datatype;
use crate::shards::shard::ShardId;

/// File name inside each `shard_{id}` directory.
pub const ADOPT_MANIFEST_FILE: &str = "adopt_manifest.json";

/// Bumped if the comparison surface changes; a manifest from a newer tool is not trusted by an
/// older server (it might omit a field this server would have compared).
///
/// v2 added `datatype` / `multivector_config` to [`DenseShape`] (both change how a stored vector is
/// interpreted, so a mismatch serves wrong data — part of the compatibility `diff`). v3 replaced the
/// sparse-vector name list with [`SparseShape`] carrying each sparse vector's `datatype` and
/// `modifier` — a mismatch there does not corrupt stored bytes but silently skews scoring and never
/// self-heals (the optimizer never rebuilds sparse segments on those fields), so it is refused too.
/// An older-version artifact is refused by [`AdoptManifest::check_compatible`]'s version gate and
/// must be rebuilt.
const ADOPT_MANIFEST_VERSION: u32 = 3;

/// The shape of one dense vector, as far as adoption compatibility is concerned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseShape {
    pub size: u64,
    pub distance: Distance,
    /// Storage datatype (float32/float16/uint8/turbo4). Changes how the stored bytes are decoded,
    /// so an artifact built for one datatype adopted into a collection declaring another would
    /// serve misinterpreted vectors. `None` == the collection default (float32).
    #[serde(default)]
    pub datatype: Option<Datatype>,
    /// Multivector configuration. A mismatch changes how a stored vector is interpreted / compared.
    #[serde(default)]
    pub multivector_config: Option<MultiVectorConfig>,
}

/// The shape of one sparse vector, as far as adoption compatibility is concerned. Only the fields
/// that decide whether the artifact's *data* is correct for the collection are recorded — the
/// `datatype` and `modifier` (placement/index tuning are adjustable after build and are not checked,
/// the same policy as dense storage placement).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseShape {
    /// Sparse index/weight datatype. A mismatch does not corrupt stored values (each segment decodes
    /// by its own persisted datatype), but the adopted segments would serve their built precision
    /// forever alongside new segments at the collection's precision — `ConfigMismatchOptimizer`
    /// never rebuilds on datatype, so it never self-heals — silently skewing scores. Part of the
    /// compatibility `diff`. `None` == the collection default (`Float32`).
    #[serde(default)]
    pub datatype: Option<Datatype>,
    /// Sparse value modifier (e.g. `Idf`). It is persisted per segment and is the one field
    /// [`segment::types::SparseVectorDataConfig::check_compatible`] treats as a cross-segment
    /// incompatibility: adopting an artifact built under a different modifier flips scoring
    /// semantics and leaves the adopted segments permanently disagreeing with newly created ones
    /// (which nothing reconciles). Compared here so such a mismatch is refused. `None` == the
    /// collection default (`Modifier::None`, i.e. no modifier).
    #[serde(default)]
    pub modifier: Option<Modifier>,
}

/// See module docs.
// No `Eq`: `QuantizationConfig` carries floats. `PartialEq` is enough (tests use `assert_eq!`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdoptManifest {
    pub version: u32,
    /// The shard id this artifact was built as; it must be installed as this same shard.
    pub shard_id: ShardId,
    pub shard_number: u32,
    pub hash_ring_shard_scale: u32,
    pub sharding_method: ShardingMethod,
    /// Dense vector name -> shape.
    pub dense_vectors: BTreeMap<String, DenseShape>,
    /// Sparse vector name -> shape.
    pub sparse_vectors: BTreeMap<String, SparseShape>,
    /// Per dense-vector *effective* index config the artifact was built with — HNSW resolved
    /// against the collection's global config plus any per-vector override, and quantization
    /// per-vector-else-global — i.e. exactly the per-vector values `ConfigMismatchOptimizer`
    /// compares. **NOT** part of the compatibility `diff` (a config mismatch does not serve wrong
    /// data); used by [`Self::config_rebuild_warning`] to detect, at adoption, whether the target
    /// collection's config would make the optimizer rebuild segments. `None` for artifacts built
    /// before this was recorded (the check is then skipped — it cannot be verified).
    #[serde(default)]
    pub dense_index_configs: Option<BTreeMap<String, DenseIndexConfig>>,
    /// Whether the collection's payload storage is on-disk, as the artifact was built. A change
    /// here makes `ConfigMismatchOptimizer` rebuild EVERY segment (`has_config_mismatch`'s first
    /// check). `None` for artifacts built before this was recorded.
    #[serde(default)]
    pub payload_on_disk: Option<bool>,
}

/// The effective index config of one dense vector, as `ConfigMismatchOptimizer` sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseIndexConfig {
    pub hnsw: HnswConfig,
    pub quantization: Option<QuantizationConfig>,
    /// The *explicitly requested* vector-storage placement (`memory` / deprecated `on_disk`) the
    /// artifact was built with, as `is_on_disk`; `None` when the build left placement unset.
    ///
    /// Compared against the target only when BOTH sides set an explicit placement and they differ
    /// (see [`dense_storage_requires_rebuild`]). A dense vector's *actual* per-segment placement is
    /// decided by size thresholds (`mmap_threshold`), so it cannot be predicted from
    /// `CollectionParams` in general — but for the large segments this feature targets, an explicit
    /// build placement is honored (the segment gets exactly that), so an explicit target placement
    /// that differs deterministically forces `ConfigMismatchOptimizer` to rebuild. When either side
    /// left placement unset the actual placement is threshold-driven and unpredictable (the same
    /// reason sparse-index placement is not checked), so it is not compared — recording it and
    /// comparing it unconditionally produced both false refusals and false confidence.
    #[serde(default)]
    pub storage_on_disk: Option<bool>,
}

/// Compute the per-vector effective index config the same way the collection resolves it for its
/// segments: HNSW = `global_hnsw` updated by the per-vector diff (`config.rs`'s `update_opt`),
/// quantization = per-vector if set, else the collection default. This is what the optimizer
/// compares a segment against, so recording it lets adoption predict an HNSW/quant rebuild.
#[allow(deprecated)] // reads the deprecated `on_disk` to resolve the requested placement, exactly
// as `optimizers_builder` / `ConfigMismatchOptimizer` do.
fn effective_dense_index_configs(
    params: &CollectionParams,
    global_hnsw: &HnswConfig,
    global_quantization: Option<&QuantizationConfig>,
) -> BTreeMap<String, DenseIndexConfig> {
    params
        .vectors
        .params_iter()
        .map(|(name, vector_params)| {
            (
                name.to_string(),
                DenseIndexConfig {
                    hnsw: global_hnsw.update_opt(vector_params.hnsw_config.as_ref()),
                    quantization: vector_params
                        .quantization_config
                        .clone()
                        .or_else(|| global_quantization.cloned()),
                    // Resolve the *explicitly requested* placement the same way the optimizer's
                    // `memory_placement()` does: explicit `memory` wins, else the deprecated
                    // `on_disk` flag; `None` when neither is set.
                    storage_on_disk: Memory::resolve(
                        vector_params.memory,
                        vector_params.on_disk.map(Memory::from_on_disk),
                    )
                    .map(Memory::is_on_disk),
                },
            )
        })
        .collect()
}

/// Whether a change in *explicit* dense vector-storage placement requires a segment rebuild. Only
/// the both-explicit-and-differing case is reliably predictable (see [`DenseIndexConfig::storage_on_disk`]);
/// when either side left placement unset the actual per-segment placement is threshold-driven, so
/// we cannot predict a rebuild and conservatively report none.
fn dense_storage_requires_rebuild(built: Option<bool>, target: Option<bool>) -> bool {
    matches!((built, target), (Some(built), Some(target)) if built != target)
}

/// Whether a change from `built` to `target` quantization requires a segment rebuild.
fn quantization_requires_rebuild(
    built: &Option<QuantizationConfig>,
    target: &Option<QuantizationConfig>,
) -> bool {
    match (built, target) {
        (Some(built), Some(target)) => built.mismatch_requires_rebuild(target),
        // Enabling or disabling quantization entirely forces a rebuild.
        (Some(_), None) | (None, Some(_)) => true,
        (None, None) => false,
    }
}

impl AdoptManifest {
    /// The manifest a shard built for `shard_id` under `params` should carry.
    pub fn for_shard(shard_id: ShardId, params: &CollectionParams) -> Self {
        let dense_vectors = params
            .vectors
            .params_iter()
            .map(|(name, vector_params)| {
                (
                    name.to_string(),
                    DenseShape {
                        size: vector_params.size.get(),
                        distance: vector_params.distance,
                        datatype: vector_params.datatype,
                        multivector_config: vector_params.multivector_config,
                    },
                )
            })
            .collect();

        let sparse_vectors: BTreeMap<String, SparseShape> = params
            .sparse_vectors
            .as_ref()
            .map(|sparse| {
                sparse
                    .iter()
                    .map(|(name, params)| {
                        (
                            name.clone(),
                            SparseShape {
                                datatype: params.index.as_ref().and_then(|index| index.datatype),
                                modifier: params.modifier,
                            },
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();

        AdoptManifest {
            version: ADOPT_MANIFEST_VERSION,
            shard_id,
            shard_number: params.shard_number.get(),
            hash_ring_shard_scale: params.hash_ring_shard_scale,
            sharding_method: params.sharding_method.unwrap_or_default(),
            dense_vectors,
            sparse_vectors,
            // Populated separately via `with_index_configs` — `for_shard` only needs
            // `CollectionParams`, while the (advisory) index config also needs the global HNSW /
            // quantization config to resolve per-vector effective values.
            dense_index_configs: None,
            payload_on_disk: None,
        }
    }

    /// Record the per-vector effective index config the artifact was built with, so adoption can
    /// detect whether the target collection's config would trigger a segment rebuild.
    pub fn with_index_configs(
        mut self,
        params: &CollectionParams,
        global_hnsw: &HnswConfig,
        global_quantization: Option<&QuantizationConfig>,
    ) -> Self {
        self.dense_index_configs = Some(effective_dense_index_configs(
            params,
            global_hnsw,
            global_quantization,
        ));
        self.payload_on_disk = Some(params.payload_storage_type().is_on_disk());
        self
    }

    /// If the artifact recorded its build index config, return a human-readable warning when the
    /// target collection's *effective per-vector* HNSW or quantization config differs in a way
    /// that would make the optimizer rebuild the affected adopted segments
    /// (`ConfigMismatchOptimizer`) — expensive at scale, and usually an accidental config change.
    /// `None` when nothing would rebuild, or when the artifact predates index-config recording
    /// (cannot check).
    ///
    /// Compares the resolved *per-vector* config (per-vector override merged with the global), so
    /// it catches per-vector overrides a collection-level comparison would miss. Advisory: unlike a
    /// topology/vector-shape mismatch (refused by [`Self::check_compatible`]), a config mismatch is
    /// a cost, not a correctness problem.
    pub fn config_rebuild_warning(
        &self,
        params: &CollectionParams,
        global_hnsw: &HnswConfig,
        global_quantization: Option<&QuantizationConfig>,
    ) -> Option<String> {
        // An artifact that recorded no index config at all (a build from before this was recorded,
        // or one hand-staged without the tool) cannot be checked: we cannot rule out a full,
        // expensive segment rebuild. Refuse rather than silently eat it — the caller downgrades
        // this to a warning, or lets it through under `adopt_allow_config_rebuild`. (A collection
        // with only sparse vectors still records `Some(empty map)`, so this fires only for a truly
        // unrecorded manifest.)
        if self.dense_index_configs.is_none() {
            return Some(
                "this artifact did not record the index config it was built with, so whether the \
                 collection's config would force an expensive segment rebuild cannot be verified. \
                 Rebuild the artifact with a current builder, or proceed anyway."
                    .to_string(),
            );
        }

        let mut reasons: Vec<String> = Vec::new();

        // Collection-level payload storage placement: a change rebuilds EVERY segment. Checked
        // independently of the dense index config, so a payload change is caught even when the
        // artifact recorded payload placement but no dense index config.
        if let Some(built_payload) = self.payload_on_disk
            && built_payload != params.payload_storage_type().is_on_disk()
        {
            reasons.push("payload storage placement".to_string());
        }

        // Per-vector HNSW / quantization / explicit storage placement.
        if let Some(built) = self.dense_index_configs.as_ref() {
            let target = effective_dense_index_configs(params, global_hnsw, global_quantization);
            let mut dense_rebuilt: Vec<String> = target
                .iter()
                .filter_map(|(name, target_cfg)| {
                    // A vector present only in `target` is a shape/topology mismatch already
                    // refused by `check_compatible`, so only compare vectors the artifact carries.
                    let built_cfg = built.get(name)?;
                    let rebuild = built_cfg.hnsw.mismatch_requires_rebuild(&target_cfg.hnsw)
                        || quantization_requires_rebuild(
                            &built_cfg.quantization,
                            &target_cfg.quantization,
                        )
                        || dense_storage_requires_rebuild(
                            built_cfg.storage_on_disk,
                            target_cfg.storage_on_disk,
                        );
                    rebuild.then(|| name.clone())
                })
                .collect();
            dense_rebuilt.sort();
            if !dense_rebuilt.is_empty() {
                reasons.push(format!(
                    "index config for dense vector(s) [{}]",
                    dense_rebuilt.join(", "),
                ));
            }
        }

        if reasons.is_empty() {
            None
        } else {
            Some(format!(
                "the collection's {} differ(s) from what this artifact was built with, so the \
                 optimizer will rebuild the affected adopted segments to match (expensive at \
                 scale). Create the collection with the config the artifact was built for and \
                 change it afterwards if needed, or rebuild the artifact for this collection's \
                 config. (Note: sparse-index storage placement, and dense vector-storage placement \
                 when the build or target left it unset, are not checked — they depend on \
                 per-segment size/index state, which cannot be predicted before load.)",
                reasons.join(" and "),
            ))
        }
    }

    pub fn path_in(shard_dir: &Path) -> PathBuf {
        shard_dir.join(ADOPT_MANIFEST_FILE)
    }

    /// Write atomically into `shard_dir`.
    pub fn save(&self, shard_dir: &Path) -> std::io::Result<()> {
        common::fs::atomic_save_json(&Self::path_in(shard_dir), self)
    }

    /// A manifest is small metadata (topology + per-vector config); a real one is well under a KB.
    /// Cap the read so a staged multi-GB file cannot OOM or stall the consensus apply thread, which
    /// reads the manifest inline in Phase A.
    const MAX_SIZE_BYTES: u64 = 4 * 1024 * 1024;

    /// Read from a file path, if present. Returns `Ok(None)` when the file does not exist (an
    /// artifact built before manifests, or one hand-staged without the tool).
    pub fn load_opt(path: &Path) -> std::io::Result<Option<Self>> {
        // Reject a non-regular or oversized manifest before opening it. `fs_err::metadata` (a stat,
        // which does not block on a fifo — only open(2) would) lets us screen the file first: a fifo,
        // socket, or device staged as the manifest would otherwise block `fs_err::read` forever in
        // open(2) on the consensus apply thread, and a multi-GB regular file would OOM/stall it.
        match fs_err::metadata(path) {
            Ok(meta) if !meta.is_file() => {
                // No path in the message: callers prefix it (like the version error below).
                return Err(std::io::Error::other(
                    "is not a regular file; a manifest must be a plain file",
                ));
            }
            Ok(meta) if meta.len() > Self::MAX_SIZE_BYTES => {
                return Err(std::io::Error::other(format!(
                    "is {} bytes, exceeding the {}-byte limit",
                    meta.len(),
                    Self::MAX_SIZE_BYTES,
                )));
            }
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err),
        }
        let bytes = match fs_err::read(path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err),
        };

        // Check the version BEFORE full deserialization. A manifest from a different tool version
        // can have an incompatible *shape* (a field's type changed across a version, not just a
        // field added/removed), which would fail `from_slice` into the current struct with a cryptic
        // serde error ("invalid type: sequence, expected a map") instead of the clean "rebuild the
        // artifact" message. Parsing just the `version` first lets the version gate speak clearly.
        #[derive(Deserialize)]
        struct VersionProbe {
            version: u32,
        }
        let probe: VersionProbe = serde_json::from_slice(&bytes).map_err(std::io::Error::other)?;
        if probe.version != ADOPT_MANIFEST_VERSION {
            // No path in the message: callers prefix it (e.g. "adopt manifest {path} …").
            return Err(std::io::Error::other(format!(
                "was built for manifest version {} but this server expects \
                 {ADOPT_MANIFEST_VERSION}; rebuild the artifact with a matching tool version",
                probe.version,
            )));
        }

        Ok(Some(
            serde_json::from_slice(&bytes).map_err(std::io::Error::other)?,
        ))
    }

    /// Refuse if this artifact does not match `params` as installed under `target_shard_id`.
    ///
    /// Returns a human-readable, single-string list of every field that differs (empty `Ok(())`
    /// when compatible). The caller wraps it in whatever error type its API uses.
    pub fn check_compatible(
        &self,
        target_shard_id: ShardId,
        params: &CollectionParams,
    ) -> Result<(), String> {
        // A manifest from a future tool version may compare fewer fields than that tool intended;
        // refuse rather than pass it on an incomplete check.
        if self.version != ADOPT_MANIFEST_VERSION {
            return Err(format!(
                "artifact adopt manifest version {} is not understood by this server (expects \
                 {ADOPT_MANIFEST_VERSION}); rebuild the artifact with a matching tool version",
                self.version,
            ));
        }

        let expected = AdoptManifest::for_shard(target_shard_id, params);
        let diffs = self.diff(&expected);
        if diffs.is_empty() {
            Ok(())
        } else {
            Err(diffs.join("; "))
        }
    }

    /// Field-by-field differences from the manifest a correct artifact would carry. Empty = match.
    fn diff(&self, expected: &AdoptManifest) -> Vec<String> {
        let mut diffs = Vec::new();
        if self.shard_id != expected.shard_id {
            diffs.push(format!(
                "artifact was built as shard {} but is being installed as shard {}",
                self.shard_id, expected.shard_id,
            ));
        }
        if self.shard_number != expected.shard_number {
            diffs.push(format!(
                "shard_number {} (artifact) vs {} (collection)",
                self.shard_number, expected.shard_number,
            ));
        }
        if self.hash_ring_shard_scale != expected.hash_ring_shard_scale {
            diffs.push(format!(
                "hash_ring_shard_scale {} (artifact) vs {} (collection)",
                self.hash_ring_shard_scale, expected.hash_ring_shard_scale,
            ));
        }
        if self.sharding_method != expected.sharding_method {
            diffs.push(format!(
                "sharding_method {:?} (artifact) vs {:?} (collection)",
                self.sharding_method, expected.sharding_method,
            ));
        }
        // Normalize `datatype` before comparing: `None` means the collection default (`Float32`),
        // so `None` and `Some(Float32)` are the same shape and must not be reported as a mismatch.
        // (`multivector_config` is compared as-is — `None` vs `Some(_)` genuinely differ.)
        let normalize = |shapes: &BTreeMap<String, DenseShape>| -> BTreeMap<String, DenseShape> {
            shapes
                .iter()
                .map(|(name, shape)| {
                    let mut shape = shape.clone();
                    shape.datatype = Some(shape.datatype.unwrap_or_default());
                    (name.clone(), shape)
                })
                .collect()
        };
        if normalize(&self.dense_vectors) != normalize(&expected.dense_vectors) {
            diffs.push(format!(
                "dense vectors {:?} (artifact) vs {:?} (collection)",
                self.dense_vectors, expected.dense_vectors,
            ));
        }
        // Normalize `datatype` (`None` == default `Float32`) and `modifier` (`None` == default
        // `Modifier::None`, i.e. no modifier) before comparing, so an unset field and its explicit
        // default are the same shape. The set of names must match too — a normalized-map inequality
        // also catches an added/removed sparse vector.
        let normalize_sparse =
            |shapes: &BTreeMap<String, SparseShape>| -> BTreeMap<String, SparseShape> {
                shapes
                    .iter()
                    .map(|(name, shape)| {
                        (
                            name.clone(),
                            SparseShape {
                                datatype: Some(shape.datatype.unwrap_or_default()),
                                modifier: Some(shape.modifier.unwrap_or_default()),
                            },
                        )
                    })
                    .collect()
            };
        if normalize_sparse(&self.sparse_vectors) != normalize_sparse(&expected.sparse_vectors) {
            diffs.push(format!(
                "sparse vectors {:?} (artifact) vs {:?} (collection)",
                self.sparse_vectors, expected.sparse_vectors,
            ));
        }
        diffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::config_diff::HnswConfigDiff;

    fn manifest() -> AdoptManifest {
        AdoptManifest {
            version: ADOPT_MANIFEST_VERSION,
            shard_id: 0,
            shard_number: 3,
            hash_ring_shard_scale: 200,
            sharding_method: ShardingMethod::Auto,
            dense_vectors: BTreeMap::from([(
                "dense".to_string(),
                DenseShape {
                    size: 8,
                    distance: Distance::Cosine,
                    datatype: None,
                    multivector_config: None,
                },
            )]),
            sparse_vectors: BTreeMap::from([(
                "sparse".to_string(),
                SparseShape {
                    datatype: None,
                    modifier: None,
                },
            )]),
            dense_index_configs: None,
            payload_on_disk: None,
        }
    }

    fn params_with(hnsw_diff: Option<HnswConfigDiff>) -> CollectionParams {
        use crate::operations::types::VectorsConfig;
        use crate::operations::vector_params_builder::VectorParamsBuilder;
        let mut builder = VectorParamsBuilder::new(8, Distance::Cosine);
        if let Some(diff) = hnsw_diff {
            builder = builder.with_hnsw_config(diff);
        }
        let mut params = CollectionParams::empty();
        params.vectors = VectorsConfig::Single(builder.build());
        params
    }

    fn hnsw(m: usize) -> HnswConfig {
        HnswConfig {
            m,
            ef_construct: 100,
            full_scan_threshold: 10_000,
            ..Default::default()
        }
    }

    #[test]
    fn identical_manifests_are_compatible() {
        assert!(manifest().diff(&manifest()).is_empty());
    }

    #[test]
    fn a_shard_number_difference_is_caught() {
        // The exact silent-misroute case: a 3-shard build installed into a 1-shard collection.
        let mut expected = manifest();
        expected.shard_number = 1;
        let diffs = manifest().diff(&expected);
        assert!(
            diffs.iter().any(|d| d.contains("shard_number")),
            "{diffs:?}"
        );
    }

    #[test]
    fn installing_as_the_wrong_shard_is_caught() {
        let mut expected = manifest();
        expected.shard_id = 2;
        let diffs = manifest().diff(&expected);
        assert!(
            diffs.iter().any(|d| d.contains("built as shard 0")),
            "{diffs:?}"
        );
    }

    #[test]
    fn config_rebuild_warning_fires_on_hnsw_mismatch_and_refuses_when_unrecorded() {
        let params = params_with(None);
        let global = hnsw(16);

        // No index config recorded (legacy / hand-staged artifact) => cannot verify => refuse
        // (warn), so an unverifiable artifact does not silently eat a rebuild.
        let unrecorded = manifest()
            .config_rebuild_warning(&params, &global, None)
            .expect("unrecorded index config should warn");
        assert!(unrecorded.contains("did not record"), "{unrecorded}");

        // Recorded and matching => no warning.
        let m = manifest().with_index_configs(&params, &global, None);
        assert!(m.config_rebuild_warning(&params, &global, None).is_none());

        // Target HNSW `m` differs (rebuild-requiring) => warning.
        let warn = m
            .config_rebuild_warning(&params, &hnsw(32), None)
            .expect("hnsw mismatch should warn");
        assert!(warn.contains("index config"), "{warn}");

        // `max_indexing_threads` differs but does not require a rebuild => no warning.
        let mut threads_only = global;
        threads_only.max_indexing_threads = 4;
        assert!(
            m.config_rebuild_warning(&params, &threads_only, None)
                .is_none()
        );
    }

    #[test]
    fn config_rebuild_warning_uses_per_vector_effective_config() {
        // Built with global m=16 but the vector overrides m=8 => effective m=8.
        let override_m8 = HnswConfigDiff {
            m: Some(8),
            ..Default::default()
        };
        let built = manifest().with_index_configs(&params_with(Some(override_m8)), &hnsw(16), None);

        // Target: global m=8, no override => effective m=8. Collection-LEVEL m (16 vs 8) differs,
        // but the effective per-vector config matches, so NO rebuild. (A collection-level-only
        // check would have wrongly refused this.)
        assert!(
            built
                .config_rebuild_warning(&params_with(None), &hnsw(8), None)
                .is_none(),
            "per-vector override should cancel the collection-level difference",
        );

        // Target: global m=16, vector override m=32 => effective m=32, differs from built m=8 =>
        // rebuild. (A collection-level check (16 == 16) would have MISSED this.)
        let override_m32 = HnswConfigDiff {
            m: Some(32),
            ..Default::default()
        };
        assert!(
            built
                .config_rebuild_warning(&params_with(Some(override_m32)), &hnsw(16), None)
                .is_some(),
            "per-vector override mismatch should be caught",
        );
    }

    #[test]
    fn config_rebuild_warning_dense_storage_only_when_both_placements_explicit() {
        use crate::operations::types::VectorsConfig;
        use crate::operations::vector_params_builder::VectorParamsBuilder;
        let global = hnsw(16);
        let params_placement = |on_disk: Option<bool>| {
            let mut builder = VectorParamsBuilder::new(8, Distance::Cosine);
            if let Some(on_disk) = on_disk {
                builder = builder.with_on_disk(on_disk);
            }
            let mut params = CollectionParams::empty();
            params.vectors = VectorsConfig::Single(builder.build());
            params
        };

        // Built with an explicit in-RAM placement.
        let built = manifest().with_index_configs(&params_placement(Some(false)), &global, None);

        // Both sides explicit and differing => the large-segment rebuild is predictable => warn.
        let warn = built
            .config_rebuild_warning(&params_placement(Some(true)), &global, None)
            .expect("explicit placement flip should warn");
        assert!(warn.contains("index config"), "{warn}");

        // Target leaves placement unset => threshold-driven, unpredictable => no warning.
        assert!(
            built
                .config_rebuild_warning(&params_placement(None), &global, None)
                .is_none(),
            "unset target placement must not warn",
        );

        // Both explicit and identical => no warning.
        assert!(
            built
                .config_rebuild_warning(&params_placement(Some(false)), &global, None)
                .is_none(),
        );
    }

    #[test]
    fn datatype_and_multivector_differences_are_caught() {
        // A datatype change (float32 artifact -> uint8 collection) reinterprets stored bytes:
        // a correctness mismatch that must be refused, not a mere rebuild.
        let mut expected = manifest();
        expected.dense_vectors.get_mut("dense").unwrap().datatype = Some(Datatype::Uint8);
        let diffs = manifest().diff(&expected);
        assert!(
            diffs.iter().any(|d| d.contains("dense vectors")),
            "{diffs:?}"
        );

        // A multivector-config difference likewise changes interpretation.
        let mut expected = manifest();
        expected
            .dense_vectors
            .get_mut("dense")
            .unwrap()
            .multivector_config = Some(MultiVectorConfig::default());
        let diffs = manifest().diff(&expected);
        assert!(
            diffs.iter().any(|d| d.contains("dense vectors")),
            "{diffs:?}"
        );
    }

    #[test]
    fn sparse_datatype_difference_is_caught_but_default_is_not() {
        // A sparse datatype change reinterprets stored values — refuse it.
        let mut expected = manifest();
        expected.sparse_vectors.get_mut("sparse").unwrap().datatype = Some(Datatype::Uint8);
        let diffs = manifest().diff(&expected);
        assert!(
            diffs.iter().any(|d| d.contains("sparse vectors")),
            "{diffs:?}"
        );

        // `None` vs an explicit default `Float32` is the same shape — must NOT be flagged.
        let mut expected = manifest();
        expected.sparse_vectors.get_mut("sparse").unwrap().datatype = Some(Datatype::Float32);
        let diffs = manifest().diff(&expected);
        assert!(
            !diffs.iter().any(|d| d.contains("sparse vectors")),
            "None vs Some(Float32) must not be a sparse mismatch: {diffs:?}",
        );
    }

    #[test]
    fn sparse_modifier_difference_is_caught_but_default_is_not() {
        // Modifier flip (none artifact -> idf collection) changes scoring semantics — refuse it.
        let mut expected = manifest();
        expected.sparse_vectors.get_mut("sparse").unwrap().modifier = Some(Modifier::Idf);
        let diffs = manifest().diff(&expected);
        assert!(
            diffs.iter().any(|d| d.contains("sparse vectors")),
            "{diffs:?}"
        );

        // `None` vs explicit `Modifier::None` is the same (no modifier) — must NOT be flagged.
        let mut expected = manifest();
        expected.sparse_vectors.get_mut("sparse").unwrap().modifier = Some(Modifier::None);
        let diffs = manifest().diff(&expected);
        assert!(
            !diffs.iter().any(|d| d.contains("sparse vectors")),
            "None vs Some(Modifier::None) must not be a sparse mismatch: {diffs:?}",
        );
    }

    #[test]
    fn load_opt_refuses_a_wrong_version_manifest_before_parsing() {
        // A manifest whose `version` differs is refused with a clean message — even when its SHAPE
        // is incompatible (the pre-parse version probe), not a cryptic serde error.
        let dir = tempfile::Builder::new()
            .prefix("adopt-ver")
            .tempdir()
            .unwrap();
        let path = dir.path().join(ADOPT_MANIFEST_FILE);
        // A v2-shaped manifest: `sparse_vectors` as an array (v2), which would fail to deserialize
        // into v3's map — the probe must intercept it on the version first.
        fs_err::write(
            &path,
            br#"{"version":2,"shard_id":0,"shard_number":3,"hash_ring_shard_scale":200,"sharding_method":"auto","dense_vectors":{},"sparse_vectors":["sparse"]}"#,
        )
        .unwrap();
        let err = AdoptManifest::load_opt(&path).expect_err("a wrong-version manifest must error");
        let msg = err.to_string();
        assert!(msg.contains("version 2"), "{msg}");
        assert!(
            msg.contains(&format!("expects {ADOPT_MANIFEST_VERSION}")),
            "{msg}",
        );
        assert!(
            !msg.contains("invalid type"),
            "should not be a raw serde error: {msg}"
        );

        // A matching-version manifest still loads.
        AdoptManifest::for_shard(0, &params_with(None))
            .save(dir.path())
            .unwrap();
        assert!(AdoptManifest::load_opt(&path).unwrap().is_some());

        // A missing file is `Ok(None)`.
        assert!(
            AdoptManifest::load_opt(&dir.path().join("nope.json"))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn an_old_manifest_version_is_refused() {
        let mut old = manifest();
        old.version = ADOPT_MANIFEST_VERSION - 1;
        let err = old
            .check_compatible(0, &params_with(None))
            .expect_err("an older manifest version must be refused");
        assert!(err.contains("is not understood"), "{err}");
    }

    #[test]
    fn scale_sharding_and_vector_shape_differences_are_caught() {
        let mut expected = manifest();
        expected.hash_ring_shard_scale = 100;
        expected.sharding_method = ShardingMethod::Custom;
        expected.dense_vectors.get_mut("dense").unwrap().size = 16;
        expected.dense_vectors.get_mut("dense").unwrap().distance = Distance::Dot;
        expected.sparse_vectors = BTreeMap::new();
        let diffs = manifest().diff(&expected);
        for field in [
            "hash_ring_shard_scale",
            "sharding_method",
            "dense vectors",
            "sparse vectors",
        ] {
            assert!(
                diffs.iter().any(|d| d.contains(field)),
                "missing {field} in {diffs:?}"
            );
        }
    }
}
