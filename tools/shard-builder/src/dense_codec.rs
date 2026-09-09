//! Compact on-disk encoding for dense vectors in part files.
//!
//! # Why
//!
//! Qdrant's *storage* is f16-native (`DenseMemmapHalf`, `vector_storage_base.rs:265`), but its
//! *ingest operation* is not: `VectorInternal::Dense(DenseVector)` where
//! `DenseVector = Vec<f32>` (`data_types/vectors.rs:253`). There is no f16 variant on the update
//! path, so points must be widened to f32 before reaching `EdgeShard::update`.
//!
//! That widening is unavoidable at the ingest boundary. It is *not* required in the intermediate.
//! Storing part files as f32 cost 3,072 bytes per 768-dim point instead of 1,536, plus CBOR's
//! per-value type tag on every element — measured at 4,211 bytes/point overall, projecting to
//! ~39 TiB of scratch for a 10 B point corpus.
//!
//! This module stores dense vectors as a raw little-endian byte string, at the element width the
//! *collection config* declares. Two bytes per element for `float16`, with no per-value tagging.
//!
//! # Precision
//!
//! Narrowing only happens where the config says the vectors are f16, in which case the values
//! originated as f16 (the parquet reader widens them on read) and the round trip is exact — f32
//! represents every f16 value. A collection configured for f32 keeps 4-byte elements, so no
//! precision is ever silently discarded.
//!
//! Serialising as a CBOR *byte string* rather than an array is deliberate: `serde_cbor` encodes
//! `Vec<u8>` through `serialize_seq`, tagging every element. `DenseBytes` implements
//! `serialize_bytes` directly, which avoids pulling in `serde_bytes` for one newtype.

use std::collections::BTreeMap;
use std::fmt;

use anyhow::{Result, bail};
use half::f16;
use segment::types::VectorStorageDatatype;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Element width used for one vector's on-disk representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseEncoding {
    /// 2 bytes per element. Chosen when the collection declares `datatype: float16`.
    F16,
    /// 4 bytes per element. The default, and what a `float32` or unset datatype gets.
    F32,
}

impl DenseEncoding {
    /// Pick an encoding from a vector's configured datatype.
    ///
    /// Anything that is not explicitly `float16` gets f32. In particular `uint8` and `turbo4`
    /// are *quantization* targets applied inside the segment, not a description of the values
    /// arriving from the source, so narrowing the intermediate for them would lose data.
    pub fn for_datatype(datatype: Option<VectorStorageDatatype>) -> Self {
        match datatype {
            Some(VectorStorageDatatype::Float16) => DenseEncoding::F16,
            _ => DenseEncoding::F32,
        }
    }

    pub fn bytes_per_element(self) -> usize {
        match self {
            DenseEncoding::F16 => 2,
            DenseEncoding::F32 => 4,
        }
    }

    /// Pack f32 values into little-endian bytes at this width.
    pub fn encode(self, values: &[f32]) -> DenseBytes {
        let mut out = Vec::with_capacity(values.len() * self.bytes_per_element());
        match self {
            DenseEncoding::F16 => {
                for value in values {
                    out.extend_from_slice(&f16::from_f32(*value).to_le_bytes());
                }
            }
            DenseEncoding::F32 => {
                for value in values {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        DenseBytes(out)
    }

    /// Unpack bytes back into the f32 vector the ingest path requires.
    pub fn decode(self, bytes: &DenseBytes) -> Result<Vec<f32>> {
        let width = self.bytes_per_element();
        if !bytes.0.len().is_multiple_of(width) {
            bail!(
                "dense vector is {} bytes, not a multiple of the {width}-byte element width; \
                 the part file may have been written under a different encoding",
                bytes.0.len(),
            );
        }

        let mut out = Vec::with_capacity(bytes.0.len() / width);
        match self {
            DenseEncoding::F16 => {
                for chunk in bytes.0.chunks_exact(2) {
                    out.push(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
                }
            }
            DenseEncoding::F32 => {
                for chunk in bytes.0.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
        }
        Ok(out)
    }
}

/// A dense vector as packed bytes, serialised as a CBOR byte string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseBytes(pub Vec<u8>);

impl Serialize for DenseBytes {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for DenseBytes {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct BytesVisitor;

        impl<'de> Visitor<'de> for BytesVisitor {
            type Value = DenseBytes;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a byte string holding packed dense vector elements")
            }

            fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                Ok(DenseBytes(v.to_vec()))
            }

            fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                Ok(DenseBytes(v))
            }

            /// Accept an array of integers too, so a byte string written by a different CBOR
            /// encoder still reads back rather than failing obscurely.
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(byte) = seq.next_element::<u8>()? {
                    out.push(byte);
                }
                Ok(DenseBytes(out))
            }
        }

        deserializer.deserialize_bytes(BytesVisitor)
    }
}

/// Per-vector encodings for a collection, derived from its config.
pub type DenseEncodings = BTreeMap<String, DenseEncoding>;

/// Build the encoding map from a loaded collection config.
pub fn encodings_for(config: &crate::config::LoadedConfig) -> DenseEncodings {
    config
        .config
        .params
        .vectors
        .params_iter()
        .map(|(name, params)| {
            (
                name.to_string(),
                DenseEncoding::for_datatype(params.datatype.map(VectorStorageDatatype::from)),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips_exactly_for_values_that_came_from_f16() {
        // The only case where narrowing is applied: values already representable in f16, because
        // the parquet reader widened them from f16 on the way in.
        let original: Vec<f32> = (0..64)
            .map(|i| f16::from_f32(i as f32 / 100.0).to_f32())
            .collect();

        let encoded = DenseEncoding::F16.encode(&original);
        assert_eq!(encoded.0.len(), 64 * 2, "two bytes per element");

        let decoded = DenseEncoding::F16.decode(&encoded).unwrap();
        assert_eq!(decoded, original, "f16 round trip must be exact");
    }

    #[test]
    fn f32_round_trips_exactly() {
        let original = vec![0.1f32, -2.5, 3.75, f32::MIN_POSITIVE, 1e30];
        let encoded = DenseEncoding::F32.encode(&original);
        assert_eq!(encoded.0.len(), original.len() * 4);
        assert_eq!(DenseEncoding::F32.decode(&encoded).unwrap(), original);
    }

    #[test]
    fn f16_halves_the_bytes() {
        let values = vec![1.0f32; 768];
        assert_eq!(DenseEncoding::F32.encode(&values).0.len(), 3072);
        assert_eq!(DenseEncoding::F16.encode(&values).0.len(), 1536);
    }

    /// A CBOR byte string must not carry per-element tagging.
    #[test]
    fn cbor_encodes_as_a_byte_string_not_an_array() {
        let values = vec![1.0f32; 768];
        let bytes = DenseEncoding::F16.encode(&values);
        let cbor = serde_cbor::to_vec(&bytes).unwrap();

        // 1536 bytes of payload plus a short header. An array of 1536 tagged integers would be
        // far larger, and an array of 768 tagged f32s larger still.
        assert!(
            cbor.len() < 1536 + 16,
            "expected a compact byte string, got {} bytes",
            cbor.len(),
        );

        let decoded: DenseBytes = serde_cbor::from_slice(&cbor).unwrap();
        assert_eq!(decoded, bytes);
    }

    /// Compare against what storing f32 through plain CBOR costs, which is what we replaced.
    #[test]
    fn is_much_smaller_than_a_cbor_f32_array() {
        let values = vec![0.123f32; 768];

        let old = serde_cbor::to_vec(&values).unwrap().len();
        let new = serde_cbor::to_vec(&DenseEncoding::F16.encode(&values))
            .unwrap()
            .len();

        assert!(
            new * 2 < old,
            "f16 byte string ({new}) should be well under half a CBOR f32 array ({old})",
        );
    }

    #[test]
    fn only_float16_selects_the_narrow_encoding() {
        assert_eq!(
            DenseEncoding::for_datatype(Some(VectorStorageDatatype::Float16)),
            DenseEncoding::F16,
        );
        // Not float16: keep full width. `uint8` and `turbo4` are quantization targets applied
        // inside the segment, not a claim about the source values.
        for datatype in [
            None,
            Some(VectorStorageDatatype::Float32),
            Some(VectorStorageDatatype::Uint8),
            Some(VectorStorageDatatype::Turbo4),
        ] {
            assert_eq!(
                DenseEncoding::for_datatype(datatype),
                DenseEncoding::F32,
                "{datatype:?}"
            );
        }
    }

    #[test]
    fn rejects_a_length_that_does_not_match_the_encoding() {
        // Three bytes cannot be whole f16 elements; reading f32 from an f16 part would land here.
        let err = DenseEncoding::F16
            .decode(&DenseBytes(vec![0, 1, 2]))
            .unwrap_err();
        assert!(format!("{err:#}").contains("not a multiple"), "{err:#}");

        let err = DenseEncoding::F32
            .decode(&DenseBytes(vec![0, 1, 2]))
            .unwrap_err();
        assert!(format!("{err:#}").contains("not a multiple"), "{err:#}");
    }

    #[test]
    fn encodings_come_from_the_collection_config() {
        let mut value = crate::config::tests::valid_config_json();
        value["params"]["vectors"]["dense"]["datatype"] = serde_json::json!("float16");
        let loaded = crate::config::from_str(&value.to_string()).unwrap();
        assert_eq!(
            encodings_for(&loaded).get("dense"),
            Some(&DenseEncoding::F16)
        );

        let loaded =
            crate::config::from_str(&crate::config::tests::valid_config_json().to_string())
                .unwrap();
        assert_eq!(
            encodings_for(&loaded).get("dense"),
            Some(&DenseEncoding::F32),
            "an unset datatype must keep full width",
        );
    }
}
