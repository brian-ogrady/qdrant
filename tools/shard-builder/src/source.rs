//! Reading points from input files: the format half of the input layer.
//!
//! [`PointSource`] answers *how bytes become points*; where the bytes live is
//! [`crate::store::InputStore`]'s question, and every format opens through it — JSONL pulls a
//! sequential stream, Parquet pulls a random-access [`crate::store::StoreChunkReader`]. Adding
//! a format (NumPy is the expected next one) means implementing [`PointSource`] and extending
//! [`detect`]; adding a storage backend (S3) means implementing `InputStore`; nothing in the
//! scatter phase needs to change for either.
//!
//! Records deserialize straight into [`PointStructPersisted`], the same type
//! `PointInsertOperationsInternal::PointsList` carries, so phase 2 can feed them to
//! `EdgeShard::update` without a conversion step.

use std::io::{BufRead as _, BufReader, Read};
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use shard::operations::point_ops::PointStructPersisted;

use crate::parquet_source::{ParquetMapping, ParquetSource};
use crate::store::InputStore;

/// A stream of points from one input file.
pub trait PointSource {
    /// The next point, or `None` at end of input.
    ///
    /// Returns an error for a malformed record rather than skipping it: silently dropping
    /// points during a 50 TB build would surface much later as an unexplained count mismatch.
    fn next_point(&mut self) -> Result<Option<PointStructPersisted>>;
}

/// Input file formats the builder can read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[value(rename_all = "lower")]
pub enum InputFormat {
    /// One JSON object per line, each deserializing to `PointStructPersisted`.
    ///
    /// Useful for tests and small inputs. Carries no column mapping because the JSON already
    /// matches `PointStructPersisted`.
    Jsonl,
    /// Columnar; requires an explicit column mapping.
    Parquet,
}

impl InputFormat {
    /// Open a source over one stored object, using `mapping` when the format needs one.
    pub fn open(
        self,
        store: &dyn InputStore,
        path: &Path,
        mapping: Option<&ParquetMapping>,
    ) -> Result<Box<dyn PointSource>> {
        match self {
            InputFormat::Jsonl => Ok(Box::new(JsonlSource::open(store, path)?)),
            InputFormat::Parquet => {
                let mapping = mapping.ok_or_else(|| {
                    anyhow::anyhow!(
                        "{} is Parquet, which needs a column mapping. Add a \"mapping\" section \
                         to the input document, naming the id, vector and payload columns; \
                         `inspect-parquet` prints a file's schema to build one from.",
                        path.display(),
                    )
                })?;
                Ok(Box::new(ParquetSource::open(store, path, mapping)?))
            }
        }
    }
}

/// Guess the format from a file extension.
pub fn detect(path: &Path) -> Result<InputFormat> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("jsonl") | Some("ndjson") => Ok(InputFormat::Jsonl),
        Some("parquet") => Ok(InputFormat::Parquet),
        Some(other) => bail!(
            "unsupported input extension '.{other}' for {}; expected .parquet, .jsonl or \
             .ndjson",
            path.display(),
        ),
        None => bail!(
            "cannot determine input format for {} (no extension)",
            path.display(),
        ),
    }
}

struct JsonlSource {
    reader: BufReader<Box<dyn Read + Send>>,
    path: String,
    line_number: u64,
    buffer: String,
}

impl JsonlSource {
    fn open(store: &dyn InputStore, path: &Path) -> Result<Self> {
        let stream = store
            .read(path)
            .with_context(|| format!("cannot open {}", path.display()))?;
        Ok(Self {
            reader: BufReader::new(stream),
            path: path.display().to_string(),
            line_number: 0,
            buffer: String::new(),
        })
    }
}

impl PointSource for JsonlSource {
    fn next_point(&mut self) -> Result<Option<PointStructPersisted>> {
        loop {
            self.buffer.clear();
            let read = self
                .reader
                .read_line(&mut self.buffer)
                .with_context(|| format!("{}: cannot read", self.path))?;

            if read == 0 {
                return Ok(None);
            }
            self.line_number += 1;

            // Blank lines are tolerated; they carry no data and appear in hand-edited files.
            let line = self.buffer.trim();
            if line.is_empty() {
                continue;
            }

            let point: PointStructPersisted = serde_json::from_str(line).with_context(|| {
                format!("{}:{}: not a valid point", self.path, self.line_number)
            })?;

            return Ok(Some(point));
        }
    }
}

#[cfg(test)]
mod tests {
    use segment::types::ExtendedPointId;
    use tempfile::TempDir;

    use super::*;
    use crate::store::LocalStore;

    fn write(dir: &TempDir, name: &str, contents: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        fs_err::write(&path, contents).unwrap();
        std::path::PathBuf::from(name)
    }

    fn store(dir: &TempDir) -> LocalStore {
        LocalStore::new(dir.path())
    }

    #[test]
    fn reads_unnamed_and_named_vectors() {
        let dir = TempDir::with_prefix("source").unwrap();
        let path = write(
            &dir,
            "data.jsonl",
            r#"{"id": 1, "vector": [0.1, 0.2]}
{"id": 2, "vector": {"dense": [0.3, 0.4]}, "payload": {"city": "Berlin"}}
{"id": "550e8400-e29b-41d4-a716-446655440000", "vector": [0.5, 0.6]}
"#,
        );

        let mut source = detect(&path)
            .unwrap()
            .open(&store(&dir), &path, None)
            .unwrap();

        let first = source.next_point().unwrap().unwrap();
        assert_eq!(first.id, ExtendedPointId::NumId(1));
        assert!(first.payload.is_none());

        let second = source.next_point().unwrap().unwrap();
        assert_eq!(second.id, ExtendedPointId::NumId(2));
        assert!(second.payload.is_some());

        let third = source.next_point().unwrap().unwrap();
        assert!(matches!(third.id, ExtendedPointId::Uuid(_)));

        assert!(source.next_point().unwrap().is_none());
    }

    #[test]
    fn tolerates_blank_lines() {
        let dir = TempDir::with_prefix("source").unwrap();
        let path = write(
            &dir,
            "data.jsonl",
            "{\"id\": 1, \"vector\": [0.1]}\n\n\n{\"id\": 2, \"vector\": [0.2]}\n",
        );

        let mut source = detect(&path)
            .unwrap()
            .open(&store(&dir), &path, None)
            .unwrap();
        assert!(source.next_point().unwrap().is_some());
        assert!(source.next_point().unwrap().is_some());
        assert!(source.next_point().unwrap().is_none());
    }

    /// A malformed record must stop the run, naming the line.
    ///
    /// Skipping it would turn a data problem into a silent count mismatch discovered days
    /// later, after the artifacts are already on the target filesystem.
    #[test]
    fn malformed_record_errors_with_line_number() {
        let dir = TempDir::with_prefix("source").unwrap();
        let path = write(
            &dir,
            "data.jsonl",
            "{\"id\": 1, \"vector\": [0.1]}\n{\"id\": 2, \"vector\": \"not-a-vector\"}\n",
        );

        let mut source = detect(&path)
            .unwrap()
            .open(&store(&dir), &path, None)
            .unwrap();
        source.next_point().unwrap().unwrap();

        let err = source.next_point().unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("data.jsonl:2"), "got: {message}");
    }

    #[test]
    fn rejects_unknown_extensions() {
        let dir = TempDir::with_prefix("source").unwrap();
        // .parquet is recognised, but needs a mapping.
        let path = write(&dir, "data.parquet", "");
        assert_eq!(detect(&path).unwrap(), InputFormat::Parquet);
        let Err(err) = detect(&path).unwrap().open(&store(&dir), &path, None) else {
            panic!("Parquet without a mapping must be rejected");
        };
        assert!(
            format!("{err:#}").contains("needs a column mapping"),
            "{err:#}"
        );

        let path = write(&dir, "data.avro", "");
        let err = detect(&path).unwrap_err();
        assert!(
            format!("{err:#}").contains("unsupported input extension"),
            "{err:#}"
        );

        let path = write(&dir, "noext", "");
        let err = detect(&path).unwrap_err();
        assert!(
            format!("{err:#}").contains("cannot determine input format"),
            "{err:#}"
        );
    }

    #[test]
    fn empty_file_yields_no_points() {
        let dir = TempDir::with_prefix("source").unwrap();
        let path = write(&dir, "data.jsonl", "");
        let mut source = detect(&path)
            .unwrap()
            .open(&store(&dir), &path, None)
            .unwrap();
        assert!(source.next_point().unwrap().is_none());
    }
}
