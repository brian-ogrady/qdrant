//! Streaming Parquet reader.
//!
//! # Memory
//!
//! At 40+ TiB uncompressed, nothing may materialize a whole file — or a whole column. Two
//! mechanisms bound memory here:
//!
//! * **Column projection.** Only the columns named in the mapping are read at all. On a corpus
//!   whose payload includes a large text field, projecting it away avoids decompressing it
//!   entirely, which is the single largest saving available.
//! * **Batched decode.** [`ParquetRecordBatchReader`] yields `RecordBatch`es of at most
//!   `batch_size` rows, and points are drained from one batch before the next is decoded.
//!
//! Peak resident memory is therefore roughly *one row group's projected column chunks* plus
//! one decoded batch — not one file. Row-group size is a property of the input, so
//! [`ParquetSource::open`] logs it and warns when a single row group is large enough to
//! dominate the footprint.
//!
//! # Storage
//!
//! Parquet is the format that forces the input layer's random-access path: the footer lives at
//! the end of the file and column chunks are read by range. So the reader opens through
//! [`InputStore::chunk_reader`] — parquet's own [`ChunkReader`] contract over whatever backend
//! the store provides — rather than a bare path. Local files today, `object_store`-backed S3
//! later, with this module unchanged (see the NOTE(stage 2) in `main.rs`).
//!
//! # Mapping
//!
//! Column mapping is explicit and required. Guessing which column holds the vectors would be
//! a guess about 40 TiB of someone else's data, and a wrong guess produces a shard full of
//! plausible nonsense rather than an error.
//!
//! `id_column` and `id_format` also feed [`crate::config::part_fingerprint`]: they decide point
//! identity, and therefore routing, so changing them invalidates a scatter.
//!
//! [`ChunkReader`]: parquet::file::reader::ChunkReader

// This module dispatches on `arrow::datatypes::DataType`, which has ~38 variants of which a
// handful can hold ids or vectors. Enumerating the rest to name them in a `bail!` would be pages
// of noise that says nothing, and would need editing on every arrow release. Each wildcard arm
// here reports the unsupported type it saw, so a new variant surfaces as a clear runtime error
// rather than silently misreading data.
#![allow(clippy::wildcard_enum_match_arm)]

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context as _, Result, anyhow, bail};
use arrow::array::{
    Array, ArrayRef, AsArray as _, Float16Array, Float32Array, Float64Array, LargeStringArray,
    StringArray, StringViewArray, StructArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Float32Type, Float64Type};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use segment::types::{ExtendedPointId, Payload};
use serde::{Deserialize, Serialize};
use shard::operations::point_ops::{PointStructPersisted, VectorPersisted, VectorStructPersisted};
use sparse::common::sparse_vector::SparseVector;

use crate::source::PointSource;
use crate::store::InputStore;

/// How a Parquet file's columns map onto Qdrant points.
///
/// `deny_unknown_fields` so a typo in a column key is an error rather than a silently ignored
/// mapping that produces points without that vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParquetMapping {
    /// Column holding the point id.
    pub id_column: String,

    /// How to interpret the id column's contents.
    ///
    /// Required, with no default. Qdrant ids are either a `u64` or a UUID, so a string column
    /// always needs interpreting, and the interpretation decides both the point's identity and
    /// — through the hash ring — which shard it lands in. Guessing would be guessing at
    /// routing.
    pub id_format: IdFormat,

    /// Qdrant dense vector name -> Parquet column.
    ///
    /// Accepts `fixed_size_list` or `list` of float16/float32/float64. Float16 is read
    /// natively: the FineWeb GTE corpus stores `fixed_size_list<halffloat>[768]`.
    #[serde(default)]
    pub dense_vectors: HashMap<String, String>,

    /// Qdrant sparse vector name -> struct column holding indices and values.
    #[serde(default)]
    pub sparse_vectors: HashMap<String, SparseMapping>,

    /// Columns to carry into the point payload. Anything not listed is never read.
    #[serde(default)]
    pub payload_columns: Vec<String>,

    /// Rows decoded per batch. Bounds working memory together with row-group size.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

/// How to read a point id out of its column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdFormat {
    /// A bare UUID string, e.g. `62ce2668-49d0-4376-95b8-89fee313cd3b`.
    Uuid,
    /// A UUID in URN form, e.g. `<urn:uuid:62ce2668-49d0-4376-95b8-89fee313cd3b>`.
    ///
    /// This is how the FineWeb corpus stores ids. The wrapper is stripped and the UUID inside
    /// becomes the Qdrant point id, so a client retrieving by id must use the bare UUID.
    UrnUuid,
    /// An unsigned integer, either as a numeric column or as decimal text.
    Integer,
}

impl IdFormat {
    /// Parse one id cell.
    fn parse(self, raw: &str) -> Result<ExtendedPointId> {
        match self {
            IdFormat::Uuid => raw
                .parse()
                .map(ExtendedPointId::Uuid)
                .map_err(|err| anyhow!("id '{raw}' is not a UUID: {err}")),
            IdFormat::UrnUuid => {
                let inner = raw
                    .trim_start_matches('<')
                    .trim_end_matches('>')
                    .strip_prefix("urn:uuid:")
                    .ok_or_else(|| {
                        anyhow!(
                            "id '{raw}' is not a URN UUID; expected the form \
                             <urn:uuid:XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX>"
                        )
                    })?;
                inner
                    .parse()
                    .map(ExtendedPointId::Uuid)
                    .map_err(|err| anyhow!("id '{raw}' has an invalid UUID body: {err}"))
            }
            IdFormat::Integer => raw
                .parse::<u64>()
                .map(ExtendedPointId::NumId)
                .map_err(|err| anyhow!("id '{raw}' is not an unsigned integer: {err}")),
        }
    }
}

/// A sparse vector stored as a struct of two parallel lists.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMapping {
    pub column: String,
    #[serde(default = "default_indices_field")]
    pub indices_field: String,
    #[serde(default = "default_values_field")]
    pub values_field: String,
}

fn default_batch_size() -> usize {
    1024
}

fn default_indices_field() -> String {
    "indices".to_string()
}

fn default_values_field() -> String {
    "values".to_string()
}

impl ParquetMapping {
    pub fn validate(&self) -> Result<()> {
        if self.dense_vectors.is_empty() && self.sparse_vectors.is_empty() {
            bail!("mapping defines no vectors; at least one dense or sparse vector is required");
        }
        if self.batch_size == 0 {
            bail!("batch_size must be at least 1");
        }
        Ok(())
    }

    /// Every Parquet column this mapping touches.
    fn referenced_columns(&self) -> Vec<&str> {
        let mut columns = vec![self.id_column.as_str()];
        columns.extend(self.dense_vectors.values().map(String::as_str));
        columns.extend(self.sparse_vectors.values().map(|s| s.column.as_str()));
        columns.extend(self.payload_columns.iter().map(String::as_str));
        columns
    }
}

/// Streams points out of one Parquet file.
pub struct ParquetSource {
    reader: ParquetRecordBatchReader,
    mapping: ParquetMapping,
    path: String,
    /// Current decoded batch, drained row by row.
    batch: Option<RecordBatch>,
    row: usize,
    rows_read: u64,
}

impl std::fmt::Debug for ParquetSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetSource")
            .field("path", &self.path)
            .field("rows_read", &self.rows_read)
            .finish_non_exhaustive()
    }
}

impl ParquetSource {
    pub fn open(store: &dyn InputStore, path: &Path, mapping: &ParquetMapping) -> Result<Self> {
        mapping.validate()?;

        let chunk_reader = store.chunk_reader(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(chunk_reader)
            .with_context(|| format!("{} is not a readable Parquet file", path.display()))?;

        let metadata = builder.metadata().clone();
        let schema = builder.parquet_schema();

        // Resolve every mapped column to a leaf index, erroring on anything missing so a
        // mapping mistake surfaces on the first file rather than as absent vectors later.
        let arrow_schema = builder.schema().clone();
        let mut leaf_indices = Vec::new();
        for name in mapping.referenced_columns() {
            let field_index = arrow_schema.index_of(name).map_err(|_| {
                anyhow!(
                    "{}: column '{name}' is not in the file. Available: {}",
                    path.display(),
                    arrow_schema
                        .fields()
                        .iter()
                        .map(|f| f.name().as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            })?;
            leaf_indices.push(field_index);
        }

        // Projection is the main memory lever: unlisted columns are never decompressed.
        let mask = ProjectionMask::roots(schema, leaf_indices);

        // Report the row-group footprint, since that -- not batch_size -- sets peak memory.
        if let Some(largest) = metadata
            .row_groups()
            .iter()
            .map(|group| group.compressed_size())
            .max()
        {
            let largest_mb = largest as f64 / 1e6;
            log::debug!(
                "{}: {} row groups, largest {largest_mb:.0} MB compressed",
                path.display(),
                metadata.num_row_groups(),
            );
            if largest > 2 * 1024 * 1024 * 1024 {
                log::warn!(
                    "{}: largest row group is {largest_mb:.0} MB compressed; peak memory is \
                     bounded by one row group's projected columns, not by batch_size",
                    path.display(),
                );
            }
        }

        let reader = builder
            .with_projection(mask)
            .with_batch_size(mapping.batch_size)
            .build()
            .with_context(|| format!("cannot build reader for {}", path.display()))?;

        Ok(Self {
            reader,
            mapping: mapping.clone(),
            path: path.display().to_string(),
            batch: None,
            row: 0,
            rows_read: 0,
        })
    }

    /// Decode the next batch, or return false at end of file.
    fn advance_batch(&mut self) -> Result<bool> {
        match self.reader.next() {
            None => Ok(false),
            Some(Ok(batch)) => {
                self.batch = Some(batch);
                self.row = 0;
                Ok(true)
            }
            Some(Err(err)) => {
                Err(anyhow!(err)).with_context(|| format!("{}: cannot decode batch", self.path))
            }
        }
    }
}

impl PointSource for ParquetSource {
    fn next_point(&mut self) -> Result<Option<PointStructPersisted>> {
        loop {
            let Some(batch) = &self.batch else {
                if !self.advance_batch()? {
                    return Ok(None);
                }
                continue;
            };

            if self.row >= batch.num_rows() {
                self.batch = None;
                continue;
            }

            let row = self.row;
            self.row += 1;
            self.rows_read += 1;

            let point = read_row(batch, row, &self.mapping).with_context(|| {
                format!("{}: row {} (of the whole file)", self.path, self.rows_read)
            })?;

            return Ok(Some(point));
        }
    }
}

/// Build one point from one row of a decoded batch.
fn read_row(
    batch: &RecordBatch,
    row: usize,
    mapping: &ParquetMapping,
) -> Result<PointStructPersisted> {
    let id = read_id(column(batch, &mapping.id_column)?, row, mapping.id_format)
        .with_context(|| format!("column '{}'", mapping.id_column))?;

    let mut vectors: HashMap<String, VectorPersisted> = HashMap::new();

    for (vector_name, column_name) in &mapping.dense_vectors {
        let values = read_dense(column(batch, column_name)?, row)
            .with_context(|| format!("column '{column_name}' (vector '{vector_name}')"))?;
        vectors.insert(vector_name.clone(), VectorPersisted::Dense(values));
    }

    for (vector_name, sparse) in &mapping.sparse_vectors {
        let vector = read_sparse(column(batch, &sparse.column)?, row, sparse)
            .with_context(|| format!("column '{}' (vector '{vector_name}')", sparse.column))?;
        vectors.insert(vector_name.clone(), VectorPersisted::Sparse(vector));
    }

    let payload = read_payload(batch, row, &mapping.payload_columns)?;

    Ok(PointStructPersisted {
        id,
        vector: VectorStructPersisted::Named(vectors),
        payload,
    })
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("column '{name}' missing from decoded batch"))
}

/// Read a point id from a string (including UUID text) or unsigned integer column.
fn read_id(array: &ArrayRef, row: usize, format: IdFormat) -> Result<ExtendedPointId> {
    if array.is_null(row) {
        bail!("id is null");
    }

    match array.data_type() {
        // Utf8View appears in files written by newer Arrow versions; handle all three so a
        // writer upgrade upstream does not break ingest.
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("checked data type");
            format.parse(array.value(row))
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("checked data type");
            format.parse(array.value(row))
        }
        DataType::Utf8View => {
            let array = array
                .as_any()
                .downcast_ref::<StringViewArray>()
                .expect("checked data type");
            format.parse(array.value(row))
        }
        DataType::UInt64 => {
            require_integer_format(format)?;
            let array = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("checked data type");
            Ok(ExtendedPointId::NumId(array.value(row)))
        }
        DataType::UInt32 => {
            require_integer_format(format)?;
            let array = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .expect("checked data type");
            Ok(ExtendedPointId::NumId(u64::from(array.value(row))))
        }
        DataType::Int64 => {
            require_integer_format(format)?;
            let value = array
                .as_primitive::<arrow::datatypes::Int64Type>()
                .value(row);
            let value = u64::try_from(value)
                .map_err(|_| anyhow!("id {value} is negative; Qdrant ids are unsigned"))?;
            Ok(ExtendedPointId::NumId(value))
        }
        other => bail!(
            "unsupported id column type {other:?}; expected a string/UUID or unsigned integer"
        ),
    }
}

/// A numeric id column can only be an integer id; a UUID format would be a mapping mistake.
fn require_integer_format(format: IdFormat) -> Result<()> {
    if format != IdFormat::Integer {
        bail!("id column is numeric but id_format is {format:?}; set id_format to \"integer\"");
    }
    Ok(())
}

/// Read a dense vector from a `fixed_size_list` or `list` of floats.
///
/// Float16 is converted to f32, which is what `DenseVector` holds. That is not a precision
/// loss: f32 represents every f16 value exactly. Storing them as `Datatype::Float16` in the
/// segment is a separate, config-driven decision.
fn read_dense(array: &ArrayRef, row: usize) -> Result<Vec<f32>> {
    if array.is_null(row) {
        bail!("vector is null");
    }

    let values: ArrayRef = match array.data_type() {
        DataType::FixedSizeList(_, _) => array.as_fixed_size_list().value(row),
        DataType::List(_) => array.as_list::<i32>().value(row),
        DataType::LargeList(_) => array.as_list::<i64>().value(row),
        other => bail!("unsupported dense vector column type {other:?}; expected a list of floats"),
    };

    match values.data_type() {
        DataType::Float16 => {
            let values = values
                .as_any()
                .downcast_ref::<Float16Array>()
                .expect("checked data type");
            Ok(values
                .iter()
                .map(|v| v.unwrap_or_default().to_f32())
                .collect())
        }
        DataType::Float32 => {
            let values = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("checked data type");
            Ok(values.iter().map(|v| v.unwrap_or_default()).collect())
        }
        DataType::Float64 => {
            let values = values
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("checked data type");
            Ok(values
                .iter()
                .map(|v| v.unwrap_or_default() as f32)
                .collect())
        }
        other => bail!("unsupported dense element type {other:?}; expected float16/32/64"),
    }
}

/// Read a sparse vector from a struct of parallel `indices` and `values` lists.
fn read_sparse(array: &ArrayRef, row: usize, mapping: &SparseMapping) -> Result<SparseVector> {
    if array.is_null(row) {
        // An absent sparse vector is legitimate; represent it as empty rather than failing.
        return SparseVector::new(Vec::new(), Vec::new())
            .map_err(|err| anyhow!("cannot build empty sparse vector: {err}"));
    }

    let structure = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            anyhow!(
                "sparse column has type {:?}; expected a struct of indices and values",
                array.data_type(),
            )
        })?;

    let indices = structure
        .column_by_name(&mapping.indices_field)
        .ok_or_else(|| anyhow!("sparse struct has no '{}' field", mapping.indices_field))?;
    let values = structure
        .column_by_name(&mapping.values_field)
        .ok_or_else(|| anyhow!("sparse struct has no '{}' field", mapping.values_field))?;

    let indices = list_values(indices, row)?;
    let values = list_values(values, row)?;

    let indices: Vec<u32> = match indices.data_type() {
        DataType::UInt32 => indices
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("checked data type")
            .iter()
            .map(|v| v.unwrap_or_default())
            .collect(),
        DataType::UInt64 => indices
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("checked data type")
            .iter()
            .map(|v| u32::try_from(v.unwrap_or_default()).unwrap_or(u32::MAX))
            .collect(),
        DataType::Int32 => indices
            .as_primitive::<arrow::datatypes::Int32Type>()
            .iter()
            .map(|v| u32::try_from(v.unwrap_or_default()).unwrap_or_default())
            .collect(),
        other => bail!("unsupported sparse index type {other:?}; expected uint32"),
    };

    let values: Vec<f32> = match values.data_type() {
        DataType::Float32 => values
            .as_primitive::<Float32Type>()
            .iter()
            .map(|v| v.unwrap_or_default())
            .collect(),
        DataType::Float64 => values
            .as_primitive::<Float64Type>()
            .iter()
            .map(|v| v.unwrap_or_default() as f32)
            .collect(),
        other => bail!("unsupported sparse value type {other:?}; expected float32"),
    };

    if indices.len() != values.len() {
        bail!(
            "sparse vector has {} indices but {} values",
            indices.len(),
            values.len(),
        );
    }

    // `SparseVector::new` sorts and validates; a duplicate index is rejected there rather than
    // silently producing a vector Qdrant would reject much later.
    SparseVector::new(indices, values).map_err(|err| anyhow!("invalid sparse vector: {err}"))
}

fn list_values(array: &ArrayRef, row: usize) -> Result<ArrayRef> {
    match array.data_type() {
        DataType::List(_) => Ok(array.as_list::<i32>().value(row)),
        DataType::LargeList(_) => Ok(array.as_list::<i64>().value(row)),
        DataType::FixedSizeList(_, _) => Ok(array.as_fixed_size_list().value(row)),
        other => bail!("expected a list, found {other:?}"),
    }
}

/// Build a payload from the mapped columns.
///
/// Uses a per-type conversion so every Arrow type maps to JSON the same way everywhere,
/// rather than through per-call ad-hoc handling that would drift between payload columns.
fn read_payload(batch: &RecordBatch, row: usize, columns: &[String]) -> Result<Option<Payload>> {
    if columns.is_empty() {
        return Ok(None);
    }

    let mut map = serde_json::Map::with_capacity(columns.len());

    for name in columns {
        let array = column(batch, name)?;
        if array.is_null(row) {
            continue;
        }
        map.insert(name.clone(), arrow_value_to_json(array, row)?);
    }

    if map.is_empty() {
        return Ok(None);
    }

    Ok(Some(Payload(map)))
}

/// Convert one Arrow cell to JSON.
fn arrow_value_to_json(array: &ArrayRef, row: usize) -> Result<serde_json::Value> {
    use serde_json::Value;

    Ok(match array.data_type() {
        DataType::Utf8 => Value::String(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("checked")
                .value(row)
                .to_string(),
        ),
        DataType::LargeUtf8 => Value::String(
            array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("checked")
                .value(row)
                .to_string(),
        ),
        DataType::Utf8View => Value::String(
            array
                .as_any()
                .downcast_ref::<StringViewArray>()
                .expect("checked")
                .value(row)
                .to_string(),
        ),
        DataType::Boolean => Value::Bool(array.as_boolean().value(row)),
        DataType::Int32 => Value::from(
            array
                .as_primitive::<arrow::datatypes::Int32Type>()
                .value(row),
        ),
        DataType::Int64 => Value::from(
            array
                .as_primitive::<arrow::datatypes::Int64Type>()
                .value(row),
        ),
        DataType::UInt32 => Value::from(
            array
                .as_primitive::<arrow::datatypes::UInt32Type>()
                .value(row),
        ),
        DataType::UInt64 => Value::from(
            array
                .as_primitive::<arrow::datatypes::UInt64Type>()
                .value(row),
        ),
        DataType::Float32 => json_number(f64::from(array.as_primitive::<Float32Type>().value(row))),
        DataType::Float64 => json_number(array.as_primitive::<Float64Type>().value(row)),
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => {
            let values = list_values(array, row)?;
            let mut out = Vec::with_capacity(values.len());
            for index in 0..values.len() {
                out.push(if values.is_null(index) {
                    Value::Null
                } else {
                    arrow_value_to_json(&values, index)?
                });
            }
            Value::Array(out)
        }
        other => bail!(
            "unsupported payload column type {other:?}; convert it upstream or drop it from \
             payload_columns"
        ),
    })
}

/// JSON has no NaN or infinity, so a non-finite float becomes null rather than failing.
fn json_number(value: f64) -> serde_json::Value {
    serde_json::Number::from_f64(value)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

/// Human-readable schema of a Parquet file, for building a mapping.
pub fn describe(store: &dyn InputStore, path: &Path) -> Result<String> {
    use std::fmt::Write as _;

    let chunk_reader = store.chunk_reader(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(chunk_reader)
        .with_context(|| format!("{} is not a readable Parquet file", path.display()))?;

    let metadata = builder.metadata();
    let schema = builder.schema();

    let mut out = String::new();
    writeln!(out, "file: {}", path.display())?;
    writeln!(out, "rows: {}", metadata.file_metadata().num_rows())?;
    writeln!(out, "row groups: {}", metadata.num_row_groups())?;

    let total: i64 = metadata
        .row_groups()
        .iter()
        .map(|group| group.compressed_size())
        .sum();
    let largest = metadata
        .row_groups()
        .iter()
        .map(|group| group.compressed_size())
        .max()
        .unwrap_or(0);
    writeln!(out, "compressed: {:.1} MB", total as f64 / 1e6)?;
    writeln!(
        out,
        "largest row group: {:.1} MB  <- bounds peak memory per reader",
        largest as f64 / 1e6,
    )?;
    writeln!(out, "\ncolumns:")?;

    for field in schema.fields() {
        writeln!(out, "  {}: {:?}", field.name(), field.data_type())?;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, FixedSizeListArray, Float16Array, Float32Array, Int64Array, ListArray,
        StringArray, StructArray, UInt32Array,
    };
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{DataType, Field, Fields, Schema};
    use half::f16;
    use tempfile::TempDir;

    use super::*;
    use crate::store::LocalStore;

    fn write_parquet(dir: &TempDir, name: &str, batch: RecordBatch) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let file = fs_err::File::create(&path).unwrap();
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        std::path::PathBuf::from(name)
    }

    /// Mirrors the real corpus: UUID ids, `fixed_size_list<halffloat>[N]` dense,
    /// `struct<indices: list<uint32>, values: list<float>>` sparse.
    fn corpus_batch(rows: usize, dim: usize) -> RecordBatch {
        let ids: Vec<String> = (0..rows)
            .map(|i| format!("550e8400-e29b-41d4-a716-4466554400{i:02}"))
            .collect();
        let id_array = Arc::new(StringArray::from(ids)) as ArrayRef;

        let dense_values: Vec<Option<f16>> = (0..rows * dim)
            .map(|i| Some(f16::from_f32(i as f32 / 100.0)))
            .collect();
        let dense_field = Arc::new(Field::new("element", DataType::Float16, true));
        let dense = Arc::new(
            FixedSizeListArray::try_new(
                dense_field,
                dim as i32,
                Arc::new(Float16Array::from(dense_values)),
                None,
            )
            .unwrap(),
        ) as ArrayRef;

        // Two non-zero entries per row.
        let indices_field = Arc::new(Field::new("element", DataType::UInt32, true));
        let indices = Arc::new(
            ListArray::try_new(
                indices_field,
                OffsetBuffer::from_lengths(std::iter::repeat_n(2usize, rows)),
                Arc::new(UInt32Array::from(
                    (0..rows)
                        .flat_map(|i| [i as u32 * 2, i as u32 * 2 + 1])
                        .collect::<Vec<_>>(),
                )),
                None,
            )
            .unwrap(),
        ) as ArrayRef;

        let values_field = Arc::new(Field::new("element", DataType::Float32, true));
        let values = Arc::new(
            ListArray::try_new(
                values_field,
                OffsetBuffer::from_lengths(std::iter::repeat_n(2usize, rows)),
                Arc::new(Float32Array::from(
                    (0..rows).flat_map(|_| [0.5f32, 0.25]).collect::<Vec<_>>(),
                )),
                None,
            )
            .unwrap(),
        ) as ArrayRef;

        let sparse = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new(
                    "indices",
                    DataType::List(Arc::new(Field::new("element", DataType::UInt32, true))),
                    true,
                )),
                indices,
            ),
            (
                Arc::new(Field::new(
                    "values",
                    DataType::List(Arc::new(Field::new("element", DataType::Float32, true))),
                    true,
                )),
                values,
            ),
        ])) as ArrayRef;

        let url = Arc::new(StringArray::from(
            (0..rows)
                .map(|i| format!("https://example.test/{i}"))
                .collect::<Vec<_>>(),
        )) as ArrayRef;
        let tokens = Arc::new(Int64Array::from((0..rows as i64).collect::<Vec<_>>())) as ArrayRef;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "dense_embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("element", DataType::Float16, true)),
                    dim as i32,
                ),
                true,
            ),
            Field::new(
                "sparse_embedding",
                DataType::Struct(Fields::from(vec![
                    Field::new(
                        "indices",
                        DataType::List(Arc::new(Field::new("element", DataType::UInt32, true))),
                        true,
                    ),
                    Field::new(
                        "values",
                        DataType::List(Arc::new(Field::new("element", DataType::Float32, true))),
                        true,
                    ),
                ])),
                true,
            ),
            Field::new("url", DataType::Utf8, true),
            Field::new("tokens", DataType::Int64, true),
        ]));

        RecordBatch::try_new(schema, vec![id_array, dense, sparse, url, tokens]).unwrap()
    }

    fn corpus_mapping() -> ParquetMapping {
        ParquetMapping {
            id_column: "id".to_string(),
            id_format: IdFormat::Uuid,
            dense_vectors: HashMap::from([("dense".to_string(), "dense_embedding".to_string())]),
            sparse_vectors: HashMap::from([(
                "sparse".to_string(),
                SparseMapping {
                    column: "sparse_embedding".to_string(),
                    indices_field: "indices".to_string(),
                    values_field: "values".to_string(),
                },
            )]),
            payload_columns: vec!["url".to_string(), "tokens".to_string()],
            batch_size: 8,
        }
    }

    fn collect(dir: &TempDir, path: &Path, mapping: &ParquetMapping) -> Vec<PointStructPersisted> {
        let store = LocalStore::new(dir.path());
        let mut source = ParquetSource::open(&store, path, mapping).unwrap();
        let mut out = Vec::new();
        while let Some(point) = source.next_point().unwrap() {
            out.push(point);
        }
        out
    }

    #[test]
    fn reads_the_real_corpus_shape() {
        let dir = TempDir::with_prefix("parquet").unwrap();
        let path = write_parquet(&dir, "corpus.parquet", corpus_batch(20, 4));

        let points = collect(&dir, &path, &corpus_mapping());
        assert_eq!(points.len(), 20);

        let first = &points[0];
        assert!(matches!(first.id, ExtendedPointId::Uuid(_)), "UUID ids");

        let VectorStructPersisted::Named(vectors) = &first.vector else {
            panic!("expected named vectors");
        };
        match vectors.get("dense").unwrap() {
            VectorPersisted::Dense(values) => assert_eq!(values.len(), 4),
            other @ (VectorPersisted::Sparse(_) | VectorPersisted::MultiDense(_)) => {
                panic!("expected dense, got {other:?}")
            }
        }
        match vectors.get("sparse").unwrap() {
            VectorPersisted::Sparse(sparse) => {
                assert_eq!(sparse.indices, vec![0, 1]);
                assert_eq!(sparse.values, vec![0.5, 0.25]);
            }
            other @ (VectorPersisted::Dense(_) | VectorPersisted::MultiDense(_)) => {
                panic!("expected sparse, got {other:?}")
            }
        }

        let payload = first.payload.as_ref().expect("payload columns were mapped");
        assert_eq!(payload.0.get("url").unwrap(), "https://example.test/0");
        assert_eq!(payload.0.get("tokens").unwrap(), 0);
    }

    /// Streaming: the reader must cross batch boundaries without losing or repeating rows.
    #[test]
    fn streams_across_batch_boundaries() {
        let dir = TempDir::with_prefix("parquet").unwrap();
        let path = write_parquet(&dir, "corpus.parquet", corpus_batch(50, 2));

        // batch_size 8 over 50 rows => 7 batches, last one partial.
        let mut mapping = corpus_mapping();
        mapping.batch_size = 8;
        let points = collect(&dir, &path, &mapping);
        assert_eq!(points.len(), 50);

        // A tiny batch size must give the same result as one covering the whole file.
        mapping.batch_size = 1;
        let single = collect(&dir, &path, &mapping);
        mapping.batch_size = 4096;
        let whole = collect(&dir, &path, &mapping);

        assert_eq!(single.len(), 50);
        assert_eq!(single, whole, "batch size must not affect the points read");
        assert_eq!(points, whole);
    }

    #[test]
    fn float16_is_converted_without_loss() {
        let dir = TempDir::with_prefix("parquet").unwrap();
        let path = write_parquet(&dir, "corpus.parquet", corpus_batch(3, 4));
        let points = collect(&dir, &path, &corpus_mapping());

        let VectorStructPersisted::Named(vectors) = &points[0].vector else {
            panic!()
        };
        let VectorPersisted::Dense(values) = vectors.get("dense").unwrap() else {
            panic!()
        };

        // Row 0 holds f16(0/100), f16(1/100), f16(2/100), f16(3/100). f32 represents every
        // f16 exactly, so equality against the f16 round-trip must be exact.
        for (index, actual) in values.iter().enumerate() {
            let expected = f16::from_f32(index as f32 / 100.0).to_f32();
            assert_eq!(*actual, expected, "element {index}");
        }
    }

    #[test]
    fn unmapped_columns_are_not_required_or_read() {
        let dir = TempDir::with_prefix("parquet").unwrap();
        let path = write_parquet(&dir, "corpus.parquet", corpus_batch(5, 2));

        // Drop payload columns from the mapping: they are projected away entirely.
        let mut mapping = corpus_mapping();
        mapping.payload_columns.clear();

        let points = collect(&dir, &path, &mapping);
        assert_eq!(points.len(), 5);
        assert!(
            points[0].payload.is_none(),
            "unmapped columns must not appear in the payload",
        );
    }

    #[test]
    fn missing_column_names_the_available_ones() {
        let dir = TempDir::with_prefix("parquet").unwrap();
        let path = write_parquet(&dir, "corpus.parquet", corpus_batch(2, 2));

        let mut mapping = corpus_mapping();
        mapping.dense_vectors = HashMap::from([("dense".to_string(), "embedding".to_string())]);

        let store = LocalStore::new(dir.path());
        let err = ParquetSource::open(&store, &path, &mapping).unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("'embedding' is not in the file"),
            "{message}"
        );
        assert!(
            message.contains("dense_embedding"),
            "should list available columns: {message}"
        );
    }

    #[test]
    fn mapping_must_define_at_least_one_vector() {
        let mapping = ParquetMapping {
            id_column: "id".to_string(),
            id_format: IdFormat::Uuid,
            dense_vectors: HashMap::new(),
            sparse_vectors: HashMap::new(),
            payload_columns: vec![],
            batch_size: 8,
        };
        let err = mapping.validate().unwrap_err();
        assert!(format!("{err:#}").contains("no vectors"), "{err:#}");
    }

    #[test]
    fn rejects_unknown_mapping_fields() {
        // A typo in the mapping file must fail loudly, not silently skip a vector.
        let err = serde_json::from_str::<ParquetMapping>(
            r#"{"id_column": "id", "id_format": "uuid", "dense_vector": {"dense": "e"}}"#,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("unknown field"), "got: {err}",);
    }

    /// The id interpretation decides routing, so each format's parse is pinned here.
    #[test]
    fn id_formats_parse_and_reject_as_documented() {
        let uuid = "62ce2668-49d0-4376-95b8-89fee313cd3b";

        assert!(matches!(
            IdFormat::Uuid.parse(uuid).unwrap(),
            ExtendedPointId::Uuid(_),
        ));
        assert!(IdFormat::Uuid.parse("not-a-uuid").is_err());

        // The URN wrapper is stripped; the bare UUID inside is the point id.
        let urn = format!("<urn:uuid:{uuid}>");
        assert_eq!(
            IdFormat::UrnUuid.parse(&urn).unwrap(),
            IdFormat::Uuid.parse(uuid).unwrap(),
            "URN form must yield the same point id as the bare UUID",
        );
        assert!(
            IdFormat::UrnUuid.parse(uuid).is_err(),
            "a bare UUID is not a URN; the formats must not silently overlap",
        );

        assert_eq!(
            IdFormat::Integer.parse("42").unwrap(),
            ExtendedPointId::NumId(42),
        );
        assert!(IdFormat::Integer.parse("-1").is_err());
    }

    #[test]
    fn integer_ids_are_supported() {
        let dir = TempDir::with_prefix("parquet").unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "vec",
                DataType::List(Arc::new(Field::new("element", DataType::Float32, true))),
                true,
            ),
        ]));
        let vectors = Arc::new(
            ListArray::try_new(
                Arc::new(Field::new("element", DataType::Float32, true)),
                OffsetBuffer::from_lengths(std::iter::repeat_n(2usize, 3)),
                Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])),
                None,
            )
            .unwrap(),
        ) as ArrayRef;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![7i64, 8, 9])) as ArrayRef,
                vectors,
            ],
        )
        .unwrap();
        let path = write_parquet(&dir, "ints.parquet", batch);

        let mapping = ParquetMapping {
            id_column: "id".to_string(),
            id_format: IdFormat::Integer,
            dense_vectors: HashMap::from([("dense".to_string(), "vec".to_string())]),
            sparse_vectors: HashMap::new(),
            payload_columns: vec![],
            batch_size: 2,
        };

        let points = collect(&dir, &path, &mapping);
        assert_eq!(points.len(), 3);
        assert_eq!(points[0].id, ExtendedPointId::NumId(7));
        assert_eq!(points[2].id, ExtendedPointId::NumId(9));
    }

    #[test]
    fn describe_reports_row_group_footprint() {
        let dir = TempDir::with_prefix("parquet").unwrap();
        let path = write_parquet(&dir, "corpus.parquet", corpus_batch(10, 2));

        let store = LocalStore::new(dir.path());
        let described = describe(&store, &path).unwrap();
        assert!(described.contains("rows: 10"), "{described}");
        assert!(described.contains("largest row group"), "{described}");
        assert!(described.contains("dense_embedding"), "{described}");
    }
}
