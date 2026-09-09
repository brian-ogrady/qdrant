//! Validation for `adopt://` shard snapshot locations.
//!
//! Adoption tells the server "this local directory *is* the shard — take ownership of it
//! by rename." That is a sharper capability than the ordinary `file://` snapshot path
//! (which is caged inside the snapshots directory), so it is doubly gated:
//!
//! * **Off by default.** A node only adopts when `storage.shard_adoption_path` names a
//!   staging root, and only paths under that root are accepted — a REST caller can never
//!   make the server consume an arbitrary directory.
//! * **Same filesystem, checked up front.** Adoption installs by `rename(2)`, never by
//!   copy. The device check runs here, before anything is touched, so a mis-staged
//!   artifact produces a clear error instead of a half-started recovery.

use std::path::{Path, PathBuf};

use url::Url;

use crate::content_manager::errors::StorageError;

/// Resolve and validate the staging directory named by an `adopt://` URL.
///
/// `must_share_device_with` are storage-side directories the rename chain passes through
/// (the recovery temp dir and the collections dir); every one must be on the staging
/// directory's filesystem or the renames would degrade into copies.
pub fn resolve_adoption_source(
    configured_root: Option<&Path>,
    url: &Url,
    must_share_device_with: &[&Path],
) -> Result<PathBuf, StorageError> {
    let Some(root) = configured_root else {
        return Err(StorageError::bad_request(
            "shard adoption is disabled on this node; set `storage.shard_adoption_path` \
             to the staging root to enable `adopt://` recovery",
        ));
    };

    // `adopt://host/path` would silently swallow the first path segment as a host name.
    if let Some(host) = url.host_str()
        && !host.is_empty()
    {
        return Err(StorageError::bad_request(format!(
            "invalid adopt URL {url}: use three slashes and an absolute path \
             (adopt:///data/staging/shard_0)",
        )));
    }

    // `Url` keeps the path percent-encoded and offers no decoding for non-file schemes;
    // rather than guess, refuse the ambiguity.
    let raw_path = url.path();
    if raw_path.contains('%') {
        return Err(StorageError::bad_request(format!(
            "invalid adopt URL {url}: percent-encoded paths are not supported; \
             stage the artifact at a plain ASCII path",
        )));
    }
    if raw_path.is_empty() || raw_path == "/" {
        return Err(StorageError::bad_request(format!(
            "invalid adopt URL {url}: no path given",
        )));
    }

    // `adopt://` recovery already runs off the apply thread, so do the full symlink walk inline.
    resolve_staged_dir(root, Path::new(raw_path), must_share_device_with, true)
}

/// Validate a staged shard directory against the configured staging root: containment
/// (canonicalized, so `..` and symlinks cannot escape), directory-ness, and the
/// same-filesystem requirement. Shared by `adopt://` recovery and `adopt_shards_from`
/// creation.
/// Resolve and validate a staged shard directory. `check_symlinks` controls the recursive symlink
/// walk ([`reject_symlinks_within`]): it is O(files) of blocking `lstat`, so the `adopt_shards_from`
/// CREATE path passes `false` here — that path runs inline on the consensus apply thread (Phase A),
/// where a 10B-scale artifact's walk would stall consensus — and instead runs the walk off-thread
/// in Phase B, before the install rename. `adopt://` recovery, which already runs off the apply
/// thread, passes `true` and does the walk inline. The cheap checks (canonicalize, containment,
/// is-dir, same-device) always run.
pub fn resolve_staged_dir(
    root: &Path,
    source: &Path,
    must_share_device_with: &[&Path],
    check_symlinks: bool,
) -> Result<PathBuf, StorageError> {
    let root = fs_err::canonicalize(root).map_err(|err| {
        StorageError::bad_request(format!(
            "shard adoption staging root {} is not accessible: {err}",
            root.display(),
        ))
    })?;

    // Canonicalize before the containment check so `..` segments and symlinks cannot
    // escape the staging root.
    let source = fs_err::canonicalize(source).map_err(|err| {
        StorageError::bad_request(format!(
            "adopt path {} is not accessible: {err}",
            source.display(),
        ))
    })?;

    if !source.starts_with(&root) {
        return Err(StorageError::bad_request(format!(
            "adopt path {} is outside the configured staging root {}",
            source.display(),
            root.display(),
        )));
    }

    if !source.is_dir() {
        return Err(StorageError::bad_request(format!(
            "adopt path {} is not a directory; adoption takes an unpacked shard directory, \
             not a snapshot archive",
            source.display(),
        )));
    }

    // Reject any symlink *inside* the staged directory. Canonicalizing `source` above only
    // proves the top-level directory is under the root; adoption then moves the tree with a
    // plain `rename(2)`, which does not rewrite interior symlinks. Without this a staged
    // `shard/segments` symlinked to `/etc` (or another collection's data) would be installed
    // as-is and loaded through — an arbitrary-read escape from the staging root the whole
    // feature promises to stay within. A genuine built shard contains only regular files and
    // directories, so this rejects nothing legitimate.
    // Deferred to Phase B (before the rename) on the create path — see `check_symlinks`.
    if check_symlinks {
        reject_symlinks_within(&source)?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;

        let source_device = fs_err::metadata(&source)?.dev();
        for target in must_share_device_with {
            let target_device = fs_err::metadata(target)?.dev();
            if target_device != source_device {
                return Err(StorageError::bad_request(format!(
                    "adopt path {} is on a different filesystem than {}; adoption installs \
                     by rename, never by copy — stage the artifact on the same filesystem \
                     as `storage.storage_path`, and leave `storage.temp_path` unset (or on \
                     that same filesystem)",
                    source.display(),
                    target.display(),
                )));
            }
        }
    }

    Ok(source)
}

/// Walk `root` and fail if any entry (at any depth) is a symlink.
///
/// Uses `symlink_metadata` (lstat), so a symlink is detected rather than followed. Directory
/// symlinks are rejected up front, so the recursion never traverses through one.
///
/// KNOWN LIMITATION: this does not catch **hardlinks** — `lstat` cannot distinguish a hardlink from
/// an ordinary regular file. A hardlink inside the staged tree pointing at a sensitive inode
/// *outside* the staging root would pass this walk, be moved in by `rename(2)`, and be readable
/// after load. Adoption's same-filesystem requirement is exactly the condition hardlinks need, so
/// this is a real (if narrow) gap; it is bounded by the staging root being operator-controlled
/// (creating the hardlink already requires write access inside the trust boundary).
pub(crate) fn reject_symlinks_within(root: &Path) -> Result<(), StorageError> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs_err::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let meta = fs_err::symlink_metadata(&path)?;
            if meta.file_type().is_symlink() {
                return Err(StorageError::bad_request(format!(
                    "adopt path contains a symlink at {}; adoption installs the directory by \
                     rename and would follow it out of the staging root — stage a directory of \
                     regular files only",
                    path.display(),
                )));
            }
            if meta.is_dir() {
                stack.push(path);
            }
        }
    }
    Ok(())
}

/// Validate the `adopt_shards_from` value: a path *relative to* each peer's staging root,
/// with plain components only. Empty means the root itself. The containment check in
/// [`resolve_staged_dir`] is the backstop; this exists to reject nonsense by name before
/// any peer resolves it.
pub fn validate_staging_subdir(subdir: &str) -> Result<(), StorageError> {
    use std::path::Component;

    let ok = Path::new(subdir)
        .components()
        .all(|component| matches!(component, Component::Normal(_)));
    if !ok {
        return Err(StorageError::bad_request(format!(
            "invalid `adopt_shards_from` value {subdir:?}: must be a plain path relative \
             to each peer's `storage.shard_adoption_path` (no leading `/`, no `..`)",
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn adopt_url(path: &Path) -> Url {
        Url::parse(&format!("adopt://{}", path.display())).unwrap()
    }

    #[test]
    fn a_staged_directory_resolves() {
        let root = TempDir::new().unwrap();
        let shard = root.path().join("shard_0");
        fs_err::create_dir_all(&shard).unwrap();

        let resolved =
            resolve_adoption_source(Some(root.path()), &adopt_url(&shard), &[root.path()]).unwrap();
        assert_eq!(resolved, fs_err::canonicalize(&shard).unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_inside_the_staged_dir_is_refused() {
        let root = TempDir::new().unwrap();
        let shard = root.path().join("shard_0");
        fs_err::create_dir_all(shard.join("wal")).unwrap();
        // A `segments` entry that is a symlink escaping the staging root.
        let outside = root.path().join("outside");
        fs_err::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, shard.join("segments")).unwrap();

        let err = resolve_adoption_source(Some(root.path()), &adopt_url(&shard), &[]).unwrap_err();
        assert!(
            err.to_string().contains("symlink"),
            "an interior symlink must be refused, got: {err}",
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_nested_symlink_deep_in_the_tree_is_refused() {
        let root = TempDir::new().unwrap();
        let shard = root.path().join("shard_0");
        fs_err::create_dir_all(shard.join("segments").join("seg-uuid")).unwrap();
        std::os::unix::fs::symlink(
            "/etc/passwd",
            shard.join("segments").join("seg-uuid").join("data.bin"),
        )
        .unwrap();

        let err = resolve_adoption_source(Some(root.path()), &adopt_url(&shard), &[]).unwrap_err();
        assert!(err.to_string().contains("symlink"), "got: {err}");
    }

    #[test]
    fn disabled_without_configuration() {
        let root = TempDir::new().unwrap();
        let err = resolve_adoption_source(None, &adopt_url(root.path()), &[]).unwrap_err();
        assert!(
            err.to_string().contains("shard_adoption_path"),
            "the error must name the config key: {err}",
        );
    }

    #[test]
    fn paths_outside_the_root_are_refused() {
        let root = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let shard = outside.path().join("shard_0");
        fs_err::create_dir_all(&shard).unwrap();

        let err = resolve_adoption_source(Some(root.path()), &adopt_url(&shard), &[]).unwrap_err();
        assert!(err.to_string().contains("outside"), "got: {err}");
    }

    #[test]
    fn dotdot_traversal_cannot_escape_the_root() {
        let parent = TempDir::new().unwrap();
        let root = parent.path().join("staging");
        let secret = parent.path().join("secret");
        fs_err::create_dir_all(&root).unwrap();
        fs_err::create_dir_all(&secret).unwrap();

        let sneaky = root.join("..").join("secret");
        let err = resolve_adoption_source(Some(&root), &adopt_url(&sneaky), &[]).unwrap_err();
        assert!(err.to_string().contains("outside"), "got: {err}");
    }

    #[test]
    fn a_missing_path_is_a_clean_error() {
        let root = TempDir::new().unwrap();
        let err = resolve_adoption_source(
            Some(root.path()),
            &adopt_url(&root.path().join("nope")),
            &[],
        )
        .unwrap_err();
        assert!(err.to_string().contains("not accessible"), "got: {err}");
    }

    #[test]
    fn a_file_is_refused_with_guidance() {
        let root = TempDir::new().unwrap();
        let file = root.path().join("shard_0.snapshot");
        fs_err::write(&file, b"tar bytes").unwrap();

        let err = resolve_adoption_source(Some(root.path()), &adopt_url(&file), &[]).unwrap_err();
        assert!(err.to_string().contains("not a directory"), "got: {err}");
    }

    #[test]
    fn host_style_urls_are_refused() {
        let root = TempDir::new().unwrap();
        // Two slashes: the first segment parses as a host and would vanish from the path.
        let url = Url::parse("adopt://data/staging/shard_0").unwrap();
        let err = resolve_adoption_source(Some(root.path()), &url, &[]).unwrap_err();
        assert!(err.to_string().contains("three slashes"), "got: {err}");
    }

    #[test]
    fn percent_encoded_paths_are_refused() {
        let root = TempDir::new().unwrap();
        let url = Url::parse("adopt:///data/with%20space/shard_0").unwrap();
        let err = resolve_adoption_source(Some(root.path()), &url, &[]).unwrap_err();
        assert!(err.to_string().contains("percent-encoded"), "got: {err}");
    }

    #[test]
    fn staging_subdirs_are_plain_relative_paths() {
        validate_staging_subdir("web-corpus").unwrap();
        validate_staging_subdir("builds/2026-08-20").unwrap();
        validate_staging_subdir("").unwrap(); // the root itself

        for bad in ["/abs/path", "../escape", "a/../../b", "a/./b/.."] {
            let err = validate_staging_subdir(bad).unwrap_err();
            assert!(
                err.to_string().contains("adopt_shards_from"),
                "{bad:?} must be refused by name, got: {err}",
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_cross_device_target_is_refused_up_front() {
        let root = TempDir::new().unwrap();
        let shard = root.path().join("shard_0");
        fs_err::create_dir_all(&shard).unwrap();

        let shm = Path::new("/dev/shm");
        if !shm.is_dir() {
            eprintln!("skipping: /dev/shm unavailable");
            return;
        }
        use std::os::unix::fs::MetadataExt as _;
        if fs_err::metadata(shm).unwrap().dev() == fs_err::metadata(root.path()).unwrap().dev() {
            eprintln!("skipping: /dev/shm shares a device with the test dir");
            return;
        }

        let err =
            resolve_adoption_source(Some(root.path()), &adopt_url(&shard), &[shm]).unwrap_err();
        assert!(
            err.to_string().contains("different filesystem"),
            "got: {err}",
        );
    }
}
