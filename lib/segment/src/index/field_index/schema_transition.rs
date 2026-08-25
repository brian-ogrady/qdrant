//! Classify a payload index schema transition between the previously stored
//! schema and the newly requested one.
//!
//! Used by `StructPayloadIndex::set_indexed` to detect the case where the
//! only difference is the memory placement. For non-appendable segments this
//! lets us swap the in-memory wrapper variant in place instead of dropping
//! and rebuilding the entire field index from payload storage.
//!
//! Placement is compared *resolved* (`memory_placement()`, which reconciles the
//! new `memory` parameter with the deprecated `on_disk` flag), never field by
//! field. Two spellings of one placement — `on_disk: true` vs `memory: "cold"`,
//! `on_disk: false` vs `memory: "pinned"`, or an absent flag vs its explicit
//! default — are `Identical`: the index bytes and wrapper are the same either
//! way, and classifying them as anything else would rebuild an index (via the
//! appendable drop-and-rebuild path) over a pure change of notation.
//!
//! Each per-kind arm blanks both placement fields on clones and compares the
//! rest via the derived `PartialEq`, so a newly added field is accounted for
//! automatically: any difference outside the placement yields `Incompatible`.

// Deprecated storage placement params (`on_disk`, `always_ram`, `on_disk_payload`) are still
// handled here for backward compatibility with the new `memory` parameter
#![allow(deprecated)]

use crate::types::{PayloadFieldSchema, PayloadSchemaParams};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaTransition {
    /// The two schemas are functionally identical: everything outside the
    /// placement matches, and the *resolved* placements are equal — even when
    /// the raw `on_disk`/`memory` spellings differ (modulo `FieldType` vs
    /// fully-expanded `FieldParams`).
    Identical,
    /// The two schemas differ only in their resolved memory placement.
    OnlyOnDiskFlipped { new_on_disk: bool },
    /// The two schemas differ in a way that requires the legacy
    /// drop-and-rebuild path.
    Incompatible,
}

/// Returns `true` if the existing index already matches the requested schema.
///
/// Equivalent placement settings, such as `on_disk: true` and `memory: "cold"`, are treated as
/// a match so callers do not rebuild an unchanged index.
pub fn no_change_needed(current: &PayloadFieldSchema, requested: &PayloadFieldSchema) -> bool {
    matches!(classify(current, requested), SchemaTransition::Identical)
}

pub fn classify(old: &PayloadFieldSchema, new: &PayloadFieldSchema) -> SchemaTransition {
    let old = old.expand();
    let new = new.expand();
    let old = &*old;
    let new = &*new;

    if !equal_outside_placement(old, new) {
        return SchemaTransition::Incompatible;
    }

    if old.memory_placement() == new.memory_placement() {
        return SchemaTransition::Identical;
    }

    SchemaTransition::OnlyOnDiskFlipped {
        new_on_disk: new.is_on_disk(),
    }
}

fn equal_outside_placement(old: &PayloadSchemaParams, new: &PayloadSchemaParams) -> bool {
    use PayloadSchemaParams as P;

    // Blank both placement spellings on clones of both sides and compare the
    // rest through the derived `PartialEq` — a newly added field is therefore
    // accounted for automatically (any difference makes this `false`, i.e.
    // `Incompatible`, the safe default). The placement itself is compared
    // separately, resolved, in `classify`.
    macro_rules! rest_equal {
        ($a:expr, $b:expr) => {{
            let mut a = $a.clone();
            let mut b = $b.clone();
            a.on_disk = None;
            a.memory = None;
            b.on_disk = None;
            b.memory = None;
            a == b
        }};
    }

    match (old, new) {
        (P::Keyword(a), P::Keyword(b)) => rest_equal!(a, b),
        (P::Integer(a), P::Integer(b)) => rest_equal!(a, b),
        (P::Float(a), P::Float(b)) => rest_equal!(a, b),
        (P::Geo(a), P::Geo(b)) => rest_equal!(a, b),
        (P::Text(a), P::Text(b)) => rest_equal!(a, b),
        (P::Bool(a), P::Bool(b)) => rest_equal!(a, b),
        (P::Datetime(a), P::Datetime(b)) => rest_equal!(a, b),
        (P::Uuid(a), P::Uuid(b)) => rest_equal!(a, b),
        // Cross-kind pairs are never compatible. Listed exhaustively (rather
        // than `_ =>`) so a new `PayloadSchemaParams` variant triggers a
        // compile error here.
        (P::Keyword(_), _)
        | (P::Integer(_), _)
        | (P::Float(_), _)
        | (P::Geo(_), _)
        | (P::Text(_), _)
        | (P::Bool(_), _)
        | (P::Datetime(_), _)
        | (P::Uuid(_), _) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::index::{
        BoolIndexParams, BoolIndexType, DatetimeIndexParams, DatetimeIndexType, FloatIndexParams,
        FloatIndexType, GeoIndexParams, GeoIndexType, IntegerIndexParams, IntegerIndexType,
        KeywordIndexParams, KeywordIndexType, TextIndexParams, TextIndexType, TokenizerType,
        UuidIndexParams, UuidIndexType,
    };
    use crate::types::{Memory, PayloadSchemaType};

    fn wrap(p: PayloadSchemaParams) -> PayloadFieldSchema {
        PayloadFieldSchema::FieldParams(p)
    }

    fn keyword(on_disk: Option<bool>, is_tenant: Option<bool>) -> PayloadSchemaParams {
        PayloadSchemaParams::Keyword(KeywordIndexParams {
            memory: None,
            r#type: KeywordIndexType::Keyword,
            is_tenant,
            on_disk,
            enable_hnsw: None,
            prefix: None,
        })
    }

    fn integer(on_disk: Option<bool>, lookup: Option<bool>) -> PayloadSchemaParams {
        PayloadSchemaParams::Integer(IntegerIndexParams {
            memory: None,
            r#type: IntegerIndexType::Integer,
            lookup,
            range: Some(true),
            is_principal: None,
            on_disk,
            enable_hnsw: None,
        })
    }

    fn float(on_disk: Option<bool>) -> PayloadSchemaParams {
        PayloadSchemaParams::Float(FloatIndexParams {
            memory: None,
            r#type: FloatIndexType::Float,
            is_principal: None,
            on_disk,
            enable_hnsw: None,
        })
    }

    fn geo(on_disk: Option<bool>) -> PayloadSchemaParams {
        PayloadSchemaParams::Geo(GeoIndexParams {
            memory: None,
            r#type: GeoIndexType::Geo,
            on_disk,
            enable_hnsw: None,
        })
    }

    fn text(on_disk: Option<bool>, tokenizer: TokenizerType) -> PayloadSchemaParams {
        PayloadSchemaParams::Text(TextIndexParams {
            memory: None,
            r#type: TextIndexType::Text,
            tokenizer,
            min_token_len: None,
            max_token_len: None,
            lowercase: None,
            ascii_folding: None,
            phrase_matching: None,
            stopwords: None,
            on_disk,
            stemmer: None,
            enable_hnsw: None,
        })
    }

    fn bool_p(on_disk: Option<bool>) -> PayloadSchemaParams {
        PayloadSchemaParams::Bool(BoolIndexParams {
            memory: None,
            r#type: BoolIndexType::Bool,
            on_disk,
            enable_hnsw: None,
        })
    }

    fn datetime(on_disk: Option<bool>, is_principal: Option<bool>) -> PayloadSchemaParams {
        PayloadSchemaParams::Datetime(DatetimeIndexParams {
            memory: None,
            r#type: DatetimeIndexType::Datetime,
            is_principal,
            on_disk,
            enable_hnsw: None,
        })
    }

    fn uuid(on_disk: Option<bool>, is_tenant: Option<bool>) -> PayloadSchemaParams {
        PayloadSchemaParams::Uuid(UuidIndexParams {
            memory: None,
            r#type: UuidIndexType::Uuid,
            is_tenant,
            on_disk,
            enable_hnsw: None,
        })
    }

    #[test]
    fn identical_returns_identical() {
        let s = wrap(keyword(Some(false), None));
        assert_eq!(classify(&s, &s.clone()), SchemaTransition::Identical);
    }

    #[test]
    fn fieldtype_vs_fieldparams_identical() {
        // FieldType expands to default params; FieldParams with default values must compare equal.
        let by_type = PayloadFieldSchema::FieldType(PayloadSchemaType::Keyword);
        let by_params = wrap(PayloadSchemaParams::Keyword(KeywordIndexParams::default()));
        assert_eq!(classify(&by_type, &by_params), SchemaTransition::Identical);
        assert_eq!(classify(&by_params, &by_type), SchemaTransition::Identical);
    }

    #[test]
    fn keyword_on_disk_flip_only() {
        let off = wrap(keyword(Some(false), None));
        let on = wrap(keyword(Some(true), None));
        assert_eq!(
            classify(&off, &on),
            SchemaTransition::OnlyOnDiskFlipped { new_on_disk: true },
        );
        assert_eq!(
            classify(&on, &off),
            SchemaTransition::OnlyOnDiskFlipped { new_on_disk: false },
        );
    }

    #[test]
    fn keyword_prefix_change_is_incompatible() {
        // Enabling or disabling prefix matching requires building or dropping
        // the sorted key dictionary — a full rebuild, never an in-place swap.
        let plain = wrap(keyword(Some(false), None));
        let with_prefix = wrap(PayloadSchemaParams::Keyword(KeywordIndexParams {
            memory: None,
            r#type: KeywordIndexType::Keyword,
            is_tenant: None,
            on_disk: Some(false),
            enable_hnsw: None,
            prefix: Some(true),
        }));
        assert_eq!(
            classify(&plain, &with_prefix),
            SchemaTransition::Incompatible
        );
        assert_eq!(
            classify(&with_prefix, &plain),
            SchemaTransition::Incompatible
        );
    }

    #[test]
    fn keyword_other_field_differs_is_incompatible() {
        // Same on_disk, but is_tenant differs.
        let a = wrap(keyword(Some(false), Some(false)));
        let b = wrap(keyword(Some(false), Some(true)));
        assert_eq!(classify(&a, &b), SchemaTransition::Incompatible);
        // Both on_disk AND another field differ — also Incompatible (swap
        // can't paper over the other change).
        let c = wrap(keyword(Some(true), Some(true)));
        assert_eq!(classify(&a, &c), SchemaTransition::Incompatible);
    }

    #[test]
    fn integer_on_disk_flip_only() {
        let off = wrap(integer(Some(false), Some(true)));
        let on = wrap(integer(Some(true), Some(true)));
        assert_eq!(
            classify(&off, &on),
            SchemaTransition::OnlyOnDiskFlipped { new_on_disk: true },
        );
    }

    #[test]
    fn integer_lookup_change_is_incompatible() {
        let a = wrap(integer(Some(false), Some(true)));
        let b = wrap(integer(Some(false), Some(false)));
        assert_eq!(classify(&a, &b), SchemaTransition::Incompatible);
    }

    #[test]
    fn float_on_disk_flip_only() {
        assert_eq!(
            classify(&wrap(float(Some(false))), &wrap(float(Some(true)))),
            SchemaTransition::OnlyOnDiskFlipped { new_on_disk: true },
        );
    }

    #[test]
    fn geo_on_disk_flip_only() {
        assert_eq!(
            classify(&wrap(geo(Some(false))), &wrap(geo(Some(true)))),
            SchemaTransition::OnlyOnDiskFlipped { new_on_disk: true },
        );
    }

    #[test]
    fn text_on_disk_flip_only() {
        assert_eq!(
            classify(
                &wrap(text(Some(false), TokenizerType::Word)),
                &wrap(text(Some(true), TokenizerType::Word)),
            ),
            SchemaTransition::OnlyOnDiskFlipped { new_on_disk: true },
        );
    }

    #[test]
    fn text_tokenizer_change_is_incompatible() {
        assert_eq!(
            classify(
                &wrap(text(Some(false), TokenizerType::Word)),
                &wrap(text(Some(false), TokenizerType::Whitespace)),
            ),
            SchemaTransition::Incompatible,
        );
    }

    #[test]
    fn bool_on_disk_flip_only() {
        assert_eq!(
            classify(&wrap(bool_p(Some(false))), &wrap(bool_p(Some(true)))),
            SchemaTransition::OnlyOnDiskFlipped { new_on_disk: true },
        );
    }

    #[test]
    fn datetime_on_disk_flip_only() {
        assert_eq!(
            classify(
                &wrap(datetime(Some(false), None)),
                &wrap(datetime(Some(true), None))
            ),
            SchemaTransition::OnlyOnDiskFlipped { new_on_disk: true },
        );
    }

    #[test]
    fn uuid_on_disk_flip_only() {
        assert_eq!(
            classify(
                &wrap(uuid(Some(false), None)),
                &wrap(uuid(Some(true), None))
            ),
            SchemaTransition::OnlyOnDiskFlipped { new_on_disk: true },
        );
    }

    #[test]
    fn cross_kind_is_incompatible() {
        assert_eq!(
            classify(
                &wrap(keyword(Some(false), None)),
                &wrap(integer(Some(false), Some(true)))
            ),
            SchemaTransition::Incompatible,
        );
        assert_eq!(
            classify(&wrap(geo(Some(false))), &wrap(float(Some(false)))),
            SchemaTransition::Incompatible,
        );
    }

    #[test]
    fn on_disk_none_treated_as_default_false() {
        // Both `None` => Identical (both expand to default).
        let none_a = wrap(keyword(None, None));
        let none_b = wrap(keyword(None, None));
        assert_eq!(classify(&none_a, &none_b), SchemaTransition::Identical);

        // None vs Some(true) is a flip from default-false to explicit-true.
        let none = wrap(keyword(None, None));
        let on = wrap(keyword(Some(true), None));
        assert_eq!(
            classify(&none, &on),
            SchemaTransition::OnlyOnDiskFlipped { new_on_disk: true },
        );

        // None vs Some(false) — both resolve to the same placement, so they are
        // Identical. Classifying this as a flip used to send *appendable* segments
        // through the drop-and-rebuild path over a pure change of notation
        // (`drop_index_if_incompatible` drops on any flip for Gridstore); the
        // persisted spelling staying stale is the strictly cheaper outcome.
        let none = wrap(keyword(None, None));
        let off = wrap(keyword(Some(false), None));
        assert_eq!(classify(&none, &off), SchemaTransition::Identical);
    }

    fn keyword_memory(memory: Option<Memory>) -> PayloadSchemaParams {
        PayloadSchemaParams::Keyword(KeywordIndexParams {
            memory,
            r#type: KeywordIndexType::Keyword,
            is_tenant: None,
            on_disk: None,
            enable_hnsw: None,
            prefix: None,
        })
    }

    /// The 1.19 `memory` parameter and the deprecated `on_disk` flag are two
    /// spellings of one placement; classify must compare them resolved, or a
    /// collection config restated in the new spelling rebuilds every field
    /// index over nothing.
    #[test]
    fn equivalent_placement_spellings_are_identical() {
        // on_disk: true == memory: cold
        assert_eq!(
            classify(
                &wrap(keyword(Some(true), None)),
                &wrap(keyword_memory(Some(Memory::Cold))),
            ),
            SchemaTransition::Identical,
        );
        // on_disk: false == memory: pinned (the in-RAM field index is a heap structure)
        assert_eq!(
            classify(
                &wrap(keyword(Some(false), None)),
                &wrap(keyword_memory(Some(Memory::Pinned))),
            ),
            SchemaTransition::Identical,
        );
        // absent == the explicit default
        assert_eq!(
            classify(
                &wrap(keyword(None, None)),
                &wrap(keyword_memory(Some(Memory::Pinned))),
            ),
            SchemaTransition::Identical,
        );
    }

    #[test]
    fn a_real_placement_change_via_memory_is_a_flip() {
        assert_eq!(
            classify(
                &wrap(keyword_memory(Some(Memory::Pinned))),
                &wrap(keyword_memory(Some(Memory::Cold))),
            ),
            SchemaTransition::OnlyOnDiskFlipped { new_on_disk: true },
        );
        // Mixed spellings, real change: on_disk: true -> memory: pinned.
        assert_eq!(
            classify(
                &wrap(keyword(Some(true), None)),
                &wrap(keyword_memory(Some(Memory::Pinned))),
            ),
            SchemaTransition::OnlyOnDiskFlipped { new_on_disk: false },
        );
    }

    #[test]
    fn memory_spelling_with_another_change_is_incompatible() {
        // Placement expressed via `memory`, but is_tenant differs too.
        let a = wrap(keyword(Some(true), Some(true)));
        let mut b_params = keyword_memory(Some(Memory::Cold));
        if let PayloadSchemaParams::Keyword(k) = &mut b_params {
            k.is_tenant = Some(false);
        }
        assert_eq!(
            classify(&a, &wrap(b_params)),
            SchemaTransition::Incompatible
        );
    }
}
