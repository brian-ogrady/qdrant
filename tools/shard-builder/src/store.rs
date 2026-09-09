//! Where input bytes live: the storage half of the input layer.
//!
//! The input layer is two orthogonal traits. [`crate::source::PointSource`] answers *how bytes
//! become points* (JSONL, Parquet, later NumPy); [`InputStore`] answers *where the bytes are*
//! (a local tree now, an S3 bucket later). Formats declare their access pattern by which store
//! methods they call:
//!
//! * **`read`** — a sequential stream. Enough for JSONL and `.npy`.
//! * **`chunk_reader`** — random access. Parquet reads its footer from the end of the file and
//!   then per-row-group column chunk ranges, so a stream cannot serve it; the same goes for a
//!   `.npz`'s zip central directory. The return type implements parquet's own [`ChunkReader`],
//!   so the format layer plugs it straight into `ParquetRecordBatchReaderBuilder`.
//! * **`list`** — discovery. Scatter (stage 3) walks the store instead of the filesystem, so
//!   slicing across machines and part naming work identically over any backend. Sorted, because
//!   `--slice` striding and resumability depend on every process seeing the same order.
//!
//! # The S3 backend (post-stage-3)
//!
//! Lands as a second implementation over the `object_store` crate — ranged reads, retries and
//! request signing for free, and native parquet integration — with one shared tokio runtime
//! bridged *inside* the store impl, so this trait and the whole scatter pipeline stay
//! synchronous. `--input` picks the store by scheme (`/data/corpus` vs `s3://bucket/prefix`).
//! Part file names hash the store-relative path, so the reference's resumability rule ("point
//! every process at the same paths") carries over verbatim: same URL everywhere, or the same
//! file scatters twice under two names.
//!
//! [`ChunkReader`]: parquet::file::reader::ChunkReader

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use bytes::Bytes;
use parquet::file::reader::{ChunkReader, Length};

/// One input object as the store lists it.
#[derive(Debug, Clone)]
pub struct InputObject {
    /// Store-relative path — what part naming hashes and error messages print.
    pub path: PathBuf,
    /// Size in bytes, from the listing. Saves a stat/HEAD per file downstream; unread until
    /// a consumer needs it (the S3 backend's ranged reads will).
    #[allow(dead_code)]
    pub size: u64,
}

/// Where input bytes live. Local filesystem now; S3 later (see the module doc).
pub trait InputStore: Send + Sync {
    /// Every regular file under the root, recursively, sorted by path.
    ///
    /// No format filtering here — which extensions are inputs is the format layer's
    /// question (`source::detect`, and scatter's `--input-format`).
    fn list(&self) -> Result<Vec<InputObject>>;

    /// Open a sequential stream over one object. For formats that read front to back.
    fn read(&self, path: &Path) -> Result<Box<dyn Read + Send>>;

    /// Open one object for random access. For formats that seek: parquet, npz.
    fn chunk_reader(&self, path: &Path) -> Result<StoreChunkReader>;
}

/// A random-access handle over one stored object, usable wherever parquet wants a
/// [`ChunkReader`].
///
/// An enum rather than a boxed trait object because `ChunkReader` has an associated reader
/// type and is not object-safe. The S3 backend adds a variant; every format keeps working
/// unchanged because dispatch happens here, not in the formats.
pub enum StoreChunkReader {
    // `std::fs::File` rather than `fs_err::File` because parquet implements `ChunkReader`
    // for the former; opens still go through `fs_err` so failures name the path.
    #[allow(clippy::disallowed_types)]
    Local(std::fs::File),
}

impl Length for StoreChunkReader {
    fn len(&self) -> u64 {
        match self {
            StoreChunkReader::Local(file) => file.len(),
        }
    }
}

impl ChunkReader for StoreChunkReader {
    type T = StoreChunkRead;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        match self {
            StoreChunkReader::Local(file) => Ok(StoreChunkRead::Local(file.get_read(start)?)),
        }
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        match self {
            StoreChunkReader::Local(file) => file.get_bytes(start, length),
        }
    }
}

/// The sequential reader [`StoreChunkReader::get_read`] hands out.
pub enum StoreChunkRead {
    // See `StoreChunkReader::Local` for why this names `std::fs::File`.
    #[allow(clippy::disallowed_types)]
    Local(<std::fs::File as ChunkReader>::T),
}

impl Read for StoreChunkRead {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            StoreChunkRead::Local(reader) => reader.read(buf),
        }
    }
}

/// The local-filesystem store: a recursive walk rooted at one directory (or a single file).
pub struct LocalStore {
    root: PathBuf,
}

impl LocalStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Resolve a store-relative path back to the filesystem.
    fn absolute(&self, path: &Path) -> PathBuf {
        // A root that is itself a file lists as one object with its file name; joining a
        // relative path onto a file root would be nonsense, so pass the root through.
        if self.root.is_file() {
            self.root.clone()
        } else {
            self.root.join(path)
        }
    }
}

impl InputStore for LocalStore {
    fn list(&self) -> Result<Vec<InputObject>> {
        fn walk(
            dir: &Path,
            root: &Path,
            out: &mut Vec<InputObject>,
            skipped_symlinks: &mut Vec<PathBuf>,
        ) -> Result<()> {
            for entry in fs_err::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                let file_type = entry.file_type()?;
                if file_type.is_dir() {
                    walk(&path, root, out, skipped_symlinks)?;
                } else if file_type.is_file() {
                    let size = entry.metadata()?.len();
                    let relative = path
                        .strip_prefix(root)
                        .expect("walk never leaves the root")
                        .to_path_buf();
                    out.push(InputObject {
                        path: relative,
                        size,
                    });
                } else if file_type.is_symlink() {
                    // Symlinks (to files *or* directories) are skipped, not followed: a
                    // per-node symlink tree reached under two names would scatter the same
                    // file twice (see the module doc on path identity). But skipping silently
                    // means a mixed corpus — some shards symlinked into a blob store — would
                    // scatter only a subset with no sign; collect them so the caller can warn.
                    skipped_symlinks.push(path);
                }
            }
            Ok(())
        }

        let mut objects = Vec::new();
        if self.root.is_file() {
            let size = fs_err::metadata(&self.root)?.len();
            let name = self
                .root
                .file_name()
                .context("input file has no file name")?;
            objects.push(InputObject {
                path: PathBuf::from(name),
                size,
            });
        } else {
            let mut skipped_symlinks = Vec::new();
            walk(&self.root, &self.root, &mut objects, &mut skipped_symlinks)
                .with_context(|| format!("cannot list inputs under {}", self.root.display()))?;
            if !skipped_symlinks.is_empty() {
                skipped_symlinks.sort();
                let examples: Vec<String> = skipped_symlinks
                    .iter()
                    .take(3)
                    .map(|p| p.display().to_string())
                    .collect();
                log::warn!(
                    "skipped {} symlink(s) under {} — symlinked inputs are NOT scattered \
                     (they are not followed, to avoid scattering the same file twice under \
                     two names). If these are data files, copy them in as regular files. \
                     Examples: {}{}",
                    skipped_symlinks.len(),
                    self.root.display(),
                    examples.join(", "),
                    if skipped_symlinks.len() > 3 {
                        ", ..."
                    } else {
                        ""
                    },
                );
            }
        }

        // Sorted so every process — and every resumed run — sees the same order.
        objects.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(objects)
    }

    fn read(&self, path: &Path) -> Result<Box<dyn Read + Send>> {
        let absolute = self.absolute(path);
        let file = fs_err::File::open(&absolute)
            .with_context(|| format!("cannot open {}", absolute.display()))?;
        Ok(Box::new(file))
    }

    fn chunk_reader(&self, path: &Path) -> Result<StoreChunkReader> {
        let absolute = self.absolute(path);
        // Opened through `fs_err` so a failure names the path, then converted: parquet's
        // ChunkReader is implemented for `std::fs::File`, which is why the annotation is here.
        #[allow(clippy::disallowed_types)]
        let file: std::fs::File = fs_err::File::open(&absolute)
            .with_context(|| format!("cannot open {}", absolute.display()))?
            .into();
        Ok(StoreChunkReader::Local(file))
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn lists_recursively_sorted_and_relative() {
        let dir = TempDir::with_prefix("store").unwrap();
        fs_err::create_dir_all(dir.path().join("b/nested")).unwrap();
        fs_err::write(dir.path().join("b/nested/two.jsonl"), "x").unwrap();
        fs_err::write(dir.path().join("a.jsonl"), "yy").unwrap();
        fs_err::write(dir.path().join("b/one.jsonl"), "zzz").unwrap();

        let store = LocalStore::new(dir.path());
        let objects = store.list().unwrap();

        let paths: Vec<_> = objects
            .iter()
            .map(|o| o.path.display().to_string())
            .collect();
        assert_eq!(paths, ["a.jsonl", "b/nested/two.jsonl", "b/one.jsonl"]);
        assert_eq!(objects[0].size, 2, "sizes come from the listing");
    }

    /// Symlinked inputs are skipped (not followed), and the listing still succeeds with the
    /// regular files. The warning path is exercised; regular files are unaffected.
    #[cfg(unix)]
    #[test]
    fn symlinked_inputs_are_skipped_not_followed() {
        let dir = TempDir::with_prefix("store").unwrap();
        fs_err::write(dir.path().join("real.jsonl"), "yy").unwrap();
        let target = dir.path().join("elsewhere.jsonl");
        fs_err::write(&target, "zzz").unwrap();
        std::os::unix::fs::symlink(&target, dir.path().join("linked.jsonl")).unwrap();
        // A symlinked directory must not be traversed either.
        fs_err::create_dir_all(dir.path().join("realdir")).unwrap();
        fs_err::write(dir.path().join("realdir/inside.jsonl"), "w").unwrap();
        std::os::unix::fs::symlink(dir.path().join("realdir"), dir.path().join("linkeddir"))
            .unwrap();

        let objects = LocalStore::new(dir.path()).list().unwrap();
        let paths: Vec<_> = objects
            .iter()
            .map(|o| o.path.display().to_string())
            .collect();
        // `linked.jsonl` and anything under `linkeddir/` are absent; `elsewhere.jsonl` and the
        // real tree are present (each real file exactly once).
        assert_eq!(
            paths,
            ["elsewhere.jsonl", "real.jsonl", "realdir/inside.jsonl"],
            "symlinks skipped, real files listed once",
        );
    }

    #[test]
    fn a_single_file_root_lists_as_one_object() {
        let dir = TempDir::with_prefix("store").unwrap();
        let file = dir.path().join("corpus.jsonl");
        fs_err::write(&file, "{}").unwrap();

        let store = LocalStore::new(&file);
        let objects = store.list().unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].path, PathBuf::from("corpus.jsonl"));

        let mut contents = String::new();
        store
            .read(&objects[0].path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert_eq!(contents, "{}");
    }

    #[test]
    fn reads_are_relative_to_the_root() {
        let dir = TempDir::with_prefix("store").unwrap();
        fs_err::create_dir_all(dir.path().join("sub")).unwrap();
        fs_err::write(dir.path().join("sub/data.jsonl"), "hello").unwrap();

        let store = LocalStore::new(dir.path());
        let mut contents = String::new();
        store
            .read(Path::new("sub/data.jsonl"))
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert_eq!(contents, "hello");
    }

    /// The chunk reader must satisfy parquet's random-access contract.
    #[test]
    fn chunk_reader_serves_ranged_reads() {
        let dir = TempDir::with_prefix("store").unwrap();
        fs_err::write(dir.path().join("data.bin"), b"0123456789").unwrap();

        let store = LocalStore::new(dir.path());
        let reader = store.chunk_reader(Path::new("data.bin")).unwrap();

        assert_eq!(reader.len(), 10);
        assert_eq!(reader.get_bytes(3, 4).unwrap().as_ref(), b"3456");

        let mut tail = String::new();
        reader
            .get_read(7)
            .unwrap()
            .read_to_string(&mut tail)
            .unwrap();
        assert_eq!(tail, "789");
    }
}
