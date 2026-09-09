use common::tempfile_ext::MaybeTempPath;
use tempfile::TempDir;

pub enum SnapshotData {
    /// Tar file containing the snapshot, needs to be unpacked
    Packed(MaybeTempPath),
    /// Directory containing the unpacked snapshot
    Unpacked(TempDir),
    /// Local unpacked shard directory to be **consumed by rename** — no copy.
    ///
    /// The adopt recovery path (`adopt://` snapshot locations): the directory is renamed
    /// into the recovery area atomically, so it must live on the same filesystem as the
    /// storage. On failure the recovery path renames it back rather than deleting it —
    /// unlike the variants above, this data is the operator's artifact, not our download.
    Adopted(std::path::PathBuf),
}

impl SnapshotData {
    pub fn new_packed_persistent<P: AsRef<std::path::Path>>(path: P) -> Self {
        SnapshotData::Packed(MaybeTempPath::Persistent(path.as_ref().to_path_buf()))
    }

    /// Get path to the downloaded data
    pub fn path(&self) -> &std::path::Path {
        match self {
            SnapshotData::Packed(maybe_path) => maybe_path,
            SnapshotData::Unpacked(temp_dir) => temp_dir.path(),
            SnapshotData::Adopted(path) => path,
        }
    }
}
