//! Consume-by-rename primitives for adopted shard directories.
//!
//! Adoption (`SnapshotData::Adopted`) installs a fully built shard directory with **no
//! copy**: one atomic `rename(2)` moves it into the recovery temp area, from where the
//! ordinary restore path takes over. Because the rename consumes the operator's artifact —
//! there is no pristine source left to retry from, unlike a downloaded snapshot — the two
//! invariants here are:
//!
//! * **All-or-nothing entry.** A whole-directory rename is atomic: after a crash the
//!   artifact is either untouched at its staging path or fully inside the recovery area,
//!   never half-moved. (This is why adoption must not go through `move_all`, which moves
//!   entry by entry.)
//! * **Never deleted on failure.** If any later restore step fails, [`rescue`] renames the
//!   artifact back to its staging path — and if even that fails, it *keeps* the recovery
//!   temp directory (defusing the `TempDir` cleanup that is correct for downloaded data)
//!   and names the surviving location in the error.

use std::path::{Path, PathBuf};

use tempfile::TempDir;

use crate::operations::types::{CollectionError, CollectionResult};

/// Rename `source` onto `target`, an existing **empty** directory (as freshly created by
/// `TempDir`) — POSIX `rename(2)` replaces an empty directory atomically.
///
/// A cross-device source cannot be renamed; that is a hard error rather than a fallback to
/// copying, because no-copy is the entire point of adoption.
pub(super) fn rename_into(source: &Path, target: &Path) -> CollectionResult<()> {
    fs_err::rename(source, target).map_err(|err| {
        if err.kind() == std::io::ErrorKind::CrossesDevices {
            CollectionError::bad_input(format!(
                "cannot adopt {}: it is on a different filesystem than the storage \
                 ({}). Adoption installs by rename, never by copy — stage the artifact \
                 on the same filesystem as `storage.storage_path`, and leave \
                 `storage.temp_path` unset (or on that same filesystem).",
                source.display(),
                target.display(),
            ))
        } else {
            CollectionError::service_error(format!("cannot adopt {}: {err}", source.display()))
        }
    })
}

/// A restore step failed: return the operator's artifact to `source` when it still exists to
/// return, and — this is the part that must not lie — report accurately when it does not.
///
/// After [`rename_into`] the artifact's data lives in the recovery temp dir, and `source` is
/// gone. The restore path then moves that data into the shard directory (`move_data`) and, on
/// its *own* failure, deletes it there (`LocalShard::clear` in the restore error arm). So by
/// the time a failure reaches here the data is in exactly one of three states, told apart by
/// where the shard data (`wal/` + `segments/`) actually is:
///
/// * **`rename_into` itself failed** — `source` still exists, temp is empty. Nothing was
///   consumed; surface the original error unchanged.
/// * **failure before `move_data`** (bad manifest, incompatible ring, cancel) — the full data
///   is still in the temp dir. Rename it back to `source`, and confirm it actually landed
///   before claiming so.
/// * **failure after `move_data`** — the data was moved into the shard dir and cleared there
///   by the restore path's failure handler. There is nothing left to return. Say that plainly;
///   preserve whatever partial remains sit in the temp dir rather than deleting them.
///
/// The invariant this upholds is not "the artifact always survives" — a corrupt adopted shard
/// is genuinely destroyed on load failure — but "the report is always true about where the
/// operator's data is (or is not)".
pub(super) fn rescue(
    snapshot_temp_dir: TempDir,
    source: &Path,
    err: CollectionError,
) -> CollectionError {
    let temp = snapshot_temp_dir.path().to_path_buf();

    // `rename_into` was refused: `source` is intact, nothing to do.
    if source.exists() {
        return err;
    }

    // Data never left the temp dir: return it wholesale, and verify it arrived.
    if shard::files::check_data(&temp) {
        match fs_err::rename(&temp, source) {
            Ok(()) if shard::files::check_data(source) => {
                return CollectionError::service_error(format!(
                    "{err}; the adopted artifact was returned to {}",
                    source.display(),
                ));
            }
            // Rename reported success but the data is not there, or the rename failed
            // outright: keep the temp dir (defusing delete-on-drop) and name it.
            outcome => {
                let detail = match outcome {
                    Err(rescue_err) => {
                        format!("returning it to {} failed ({rescue_err})", source.display())
                    }
                    Ok(()) => format!("it did not arrive at {}", source.display()),
                };
                let kept: PathBuf = snapshot_temp_dir.keep();
                return CollectionError::service_error(format!(
                    "{err}; recovering the adopted artifact did not complete ({detail}); \
                     it is preserved at {}",
                    kept.display(),
                ));
            }
        }
    }

    // The destructive window: `move_data` moved the data into the shard directory and the
    // restore path's failure handler cleared it there; `source` was already consumed. The
    // staged artifact cannot be recovered. Never claim otherwise — preserve any partial
    // remains for forensics and report the truth.
    let kept: PathBuf = snapshot_temp_dir.keep();
    let remains = if fs_err::read_dir(&kept)
        .map(|mut d| d.next().is_some())
        .unwrap_or(false)
    {
        format!(" partial remains are preserved at {}", kept.display())
    } else {
        let _ = fs_err::remove_dir_all(&kept);
        String::new()
    };
    CollectionError::service_error(format!(
        "{err}; the adopted artifact from {} was consumed by the failed restore and could \
         not be recovered — a corrupt adopted shard is destroyed when it fails to load, so \
         rebuild it from your source of truth rather than relying on the staged copy.{remains}",
        source.display(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A structurally valid shard directory: `check_data` requires both `wal/` and
    /// `segments/`, so a fixture missing either would look "dataless" to `rescue`.
    fn artifact(root: &Path) -> PathBuf {
        let dir = root.join("staging").join("shard_0");
        fs_err::create_dir_all(dir.join("segments")).unwrap();
        fs_err::create_dir_all(dir.join("wal")).unwrap();
        fs_err::write(dir.join("segments").join("data.bin"), b"artifact bytes").unwrap();
        dir
    }

    #[test]
    fn rename_into_moves_the_whole_directory_atomically() {
        let root = TempDir::new().unwrap();
        let source = artifact(root.path());
        let target = TempDir::new_in(root.path()).unwrap();

        rename_into(&source, target.path()).unwrap();

        assert!(!source.exists(), "the source must be consumed");
        assert_eq!(
            fs_err::read(target.path().join("segments").join("data.bin")).unwrap(),
            b"artifact bytes",
        );
    }

    #[test]
    fn a_missing_source_is_a_clean_error() {
        let root = TempDir::new().unwrap();
        let target = TempDir::new_in(root.path()).unwrap();
        let err = rename_into(&root.path().join("nope"), target.path()).unwrap_err();
        assert!(err.to_string().contains("cannot adopt"), "got: {err}");
    }

    #[test]
    fn rescue_returns_the_artifact_to_its_staging_path() {
        let root = TempDir::new().unwrap();
        let source = artifact(root.path());
        let temp = TempDir::new_in(root.path()).unwrap();
        rename_into(&source, temp.path()).unwrap();

        let err = rescue(
            temp,
            &source,
            CollectionError::bad_request("something later failed"),
        );

        assert!(
            source.join("segments").join("data.bin").exists(),
            "the artifact must be back at its staging path",
        );
        assert!(
            err.to_string().contains("was returned to"),
            "the error must say the artifact was rescued, got: {err}",
        );
    }

    #[test]
    fn rescue_is_a_no_op_when_the_source_never_left() {
        let root = TempDir::new().unwrap();
        let source = artifact(root.path());
        let temp = TempDir::new_in(root.path()).unwrap();

        // The failure struck before the rename: source intact, temp empty.
        let err = rescue(temp, &source, CollectionError::bad_request("early failure"));
        assert!(source.exists());
        assert!(
            !err.to_string().contains("returned to"),
            "nothing was rescued, so the error must not claim it was: {err}",
        );
    }

    #[test]
    fn rescue_keeps_the_temp_dir_when_the_rename_back_fails() {
        let root = TempDir::new().unwrap();
        let source = artifact(root.path());
        let temp = TempDir::new_in(root.path()).unwrap();
        rename_into(&source, temp.path()).unwrap();
        let temp_path = temp.path().to_path_buf();

        // Make the rename-back impossible: the staging parent is gone.
        fs_err::remove_dir_all(root.path().join("staging")).unwrap();

        let err = rescue(temp, &source, CollectionError::bad_request("later failure"));

        assert!(
            temp_path.join("segments").join("data.bin").exists(),
            "the artifact must survive at the kept temp path",
        );
        assert!(
            err.to_string().contains("preserved at"),
            "the error must name the surviving location, got: {err}",
        );
        fs_err::remove_dir_all(&temp_path).ok();
    }

    /// The destructive window: `move_data` moved the data out of the temp dir into the shard
    /// dir, where the restore path's failure handler then deleted it, and `source` was already
    /// consumed. `rescue` must NOT rename the now-empty temp dir back and claim success — it
    /// must report that the artifact was consumed and could not be recovered.
    #[test]
    fn rescue_does_not_claim_success_when_the_data_was_already_consumed() {
        let root = TempDir::new().unwrap();
        let source = artifact(root.path());
        let temp = TempDir::new_in(root.path()).unwrap();
        rename_into(&source, temp.path()).unwrap();

        // Simulate `move_data` + the restore error arm's `clear`: the shard data is gone from
        // the temp dir (an empty temp dir remains, exactly as after a full move-then-clear).
        fs_err::remove_dir_all(temp.path().join("segments")).unwrap();
        fs_err::remove_dir_all(temp.path().join("wal")).unwrap();

        let err = rescue(temp, &source, CollectionError::bad_request("load failed"));

        let message = err.to_string();
        assert!(
            !message.contains("was returned to"),
            "rescue must not falsely claim the artifact was returned: {message}",
        );
        assert!(
            message.contains("consumed by the failed restore"),
            "rescue must report the data was consumed: {message}",
        );
        assert!(
            !source.exists() || !shard::files::check_data(&source),
            "the empty temp dir must not have been passed off as the restored artifact",
        );
    }

    /// Partial remains (e.g. `move_data` moved `wal/` then failed on `segments/`) must be
    /// preserved and named, not silently dropped.
    #[test]
    fn rescue_preserves_partial_remains() {
        let root = TempDir::new().unwrap();
        let source = artifact(root.path());
        let temp = TempDir::new_in(root.path()).unwrap();
        rename_into(&source, temp.path()).unwrap();

        // Only `wal/` was moved out; `segments/` remains — so `check_data` is false but the
        // temp dir is non-empty.
        fs_err::remove_dir_all(temp.path().join("wal")).unwrap();
        let temp_path = temp.path().to_path_buf();

        let err = rescue(temp, &source, CollectionError::bad_request("load failed"));
        let message = err.to_string();
        assert!(
            message.contains("partial remains are preserved at"),
            "got: {message}"
        );
        assert!(
            temp_path.join("segments").join("data.bin").exists(),
            "partial data must survive at the kept path",
        );
        fs_err::remove_dir_all(&temp_path).ok();
    }
}
