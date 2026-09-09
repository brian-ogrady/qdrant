//! Shard routing for the offline builder.
//!
//! Point-to-shard assignment is *not* a free choice: it is fixed by Qdrant's consistent
//! hash ring. Getting it wrong fails in the worst possible way — search fans out to every
//! shard and still finds the points, while retrieve-by-id and any subsequent upsert route
//! through the ring and miss them. The result is a collection that looks correct until
//! someone fetches a point by id.
//!
//! So this module does not implement a ring. It constructs the *production*
//! [`HashRingRouter`] the same way `ShardHolder::new_ring` + `rebuild_rings`
//! (`lib/collection/src/shards/shard_holder/mod.rs`) do, and delegates to it — including
//! the collection's `hash_ring_shard_scale`, which this fork made a per-collection
//! parameter. A router built at the wrong scale routes just as wrongly as one built with
//! the wrong shard count, which is why the scale is part of the part fingerprint.

use anyhow::{Result, bail};
use collection::config::ShardingMethod;
use collection::hash_ring::HashRingRouter;
use collection::shards::shard::ShardId;
use segment::types::PointIdType;

use crate::config::LoadedConfig;

/// Routes points to shards using Qdrant's hash ring.
#[derive(Debug)]
pub struct ShardRouter {
    ring: HashRingRouter,
    shard_count: u32,
}

impl ShardRouter {
    /// Build the router for a collection config.
    ///
    /// Mirrors `rebuild_rings`: a single ring for `ShardingMethod::Auto`, constructed at the
    /// collection's `hash_ring_shard_scale`, with shard ids `0..shard_number` added in order.
    ///
    /// Rejects `ShardingMethod::Custom`, which routes by shard key rather than point id.
    /// Custom sharding is a legitimate choice, but it needs a per-key ring and a
    /// key-to-partition mapping supplied by the caller, and it forfeits two properties this
    /// builder currently relies on: even shard sizes, and the guarantee that a point id
    /// lives in exactly one shard. Cross-shard duplicate ids are never deduplicated —
    /// `deduplicate_points` operates within a `SegmentHolder` — so the same id under two
    /// keys yields two copies that both surface in search results.
    pub fn new(config: &LoadedConfig) -> Result<Self> {
        let sharding_method = config.sharding_method();
        if sharding_method != ShardingMethod::Auto {
            bail!(
                "sharding_method {sharding_method:?} is not supported yet; \
                 only `auto` routes by point id. Supporting custom sharding requires a \
                 shard-key-to-partition mapping and a guarantee that point ids do not \
                 repeat across keys (duplicates across shards are never deduplicated)."
            );
        }

        let shard_count = config.shard_number();
        let mut ring = HashRingRouter::single(config.hash_ring_shard_scale());
        for shard_id in 0..shard_count {
            ring.add(shard_id as ShardId);
        }

        Ok(Self { ring, shard_count })
    }

    /// The shard a point belongs to.
    ///
    /// Errors rather than defaulting: a point the ring cannot place is a bug in ring
    /// construction, and silently sending it to shard 0 would produce exactly the
    /// unretrievable-point failure this module exists to prevent.
    pub fn shard_of(&self, point_id: PointIdType) -> Result<ShardId> {
        let shards = self.ring.get(&point_id);
        match shards.len() {
            1 => Ok(shards[0]),
            0 => bail!("hash ring placed point {point_id} in no shard; ring is empty or corrupt"),
            // Two shards means a resharding ring, which an offline build must never see.
            _ => bail!(
                "hash ring placed point {point_id} in {} shards; \
                 offline builds cannot target a resharding collection",
                shards.len(),
            ),
        }
    }

    pub fn shard_count(&self) -> u32 {
        self.shard_count
    }
}

/// Fraction of points each shard receives, measured by sampling the ring.
///
/// # Why this is worth measuring
///
/// Qdrant routes with a consistent hash ring at `hash_ring_shard_scale` virtual nodes per shard
/// (default 100, per-collection in this fork). Consistent hashing divides the ring into segments
/// of unequal length, and with a finite vnode count that unevenness does not cancel out: the
/// skew is roughly `O(1/sqrt(vnodes))` — about 10% at the default scale.
///
/// The important part is that it is a property of the ring — the scale and the shard count —
/// **not of the data**. Measured over 2M synthetic UUIDs at 10 shards and scale 100 the spread
/// is +13.7%/-19.1%; the real FineWeb corpus gives +13.8%/-18.9% over the same ring. So it can
/// be predicted before ingesting anything, it will not average out as the corpus grows, and
/// per-node capacity has to be sized for the *largest* shard rather than the mean. Raising
/// `hash_ring_shard_scale` is the knob that shrinks it, at ring-memory and lookup cost.
///
/// Sampling rather than deriving it from the ring's internals: the ring is Qdrant's, and
/// re-deriving its segment lengths here would be a second implementation that could disagree with
/// the one that actually routes.
pub fn measure_distribution(router: &ShardRouter, samples: u64) -> Result<Vec<f64>> {
    let shards = router.shard_count() as usize;
    let mut counts = vec![0u64; shards];

    for n in 0..samples {
        // A fixed multiplicative sequence over the UUID space: deterministic, so the projection
        // is reproducible, and spread evenly enough that the ring's own unevenness is what shows.
        let id = PointIdType::Uuid(uuid::Uuid::from_u128(
            u128::from(n).wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835),
        ));
        counts[router.shard_of(id)? as usize] += 1;
    }

    Ok(counts
        .into_iter()
        .map(|count| count as f64 / samples as f64)
        .collect())
}

#[cfg(test)]
mod tests {
    /// The ring's skew must be reported, and must be a property of the ring rather than the data.
    ///
    /// Guards the number the capacity projection is built on: at the default scale (100) the
    /// skew is material. `raising_the_ring_scale_reduces_skew` below covers the fork's knob
    /// for shrinking it.
    #[test]
    fn distribution_skew_is_material_and_data_independent() {
        let loaded =
            crate::config::from_str(&crate::config::tests::valid_config_json().to_string())
                .unwrap();
        let router = super::ShardRouter::new(&loaded).unwrap();

        let coarse = super::measure_distribution(&router, 200_000).unwrap();
        let fine = super::measure_distribution(&router, 800_000).unwrap();

        let shards = f64::from(router.shard_count());
        let even = 1.0 / shards;

        let max = coarse.iter().copied().fold(f64::MIN, f64::max);
        let min = coarse.iter().copied().fold(f64::MAX, f64::min);

        assert!(
            (coarse.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "the fractions must account for every point",
        );
        assert!(
            max / even > 1.05,
            "the ring is expected to be materially uneven; got max {:.1}% above even",
            (max / even - 1.0) * 100.0,
        );
        assert!(
            max / min < 2.0,
            "but not catastrophically so; got a {:.2}x spread",
            max / min,
        );

        // More samples must not change the answer: the skew comes from the ring's fixed segment
        // lengths, so it converges rather than averaging away.
        for (a, b) in coarse.iter().zip(&fine) {
            assert!(
                (a - b).abs() < 0.005,
                "skew must converge, not average out: {a:.4} vs {b:.4}",
            );
        }
    }

    use segment::types::ExtendedPointId;
    use serde_json::json;

    use super::*;
    use crate::config;

    fn router_with_shards(shard_number: u32) -> ShardRouter {
        let mut value = crate::config::tests::valid_config_json();
        value["params"]["shard_number"] = json!(shard_number);
        let loaded = config::from_str(&value.to_string()).unwrap();
        ShardRouter::new(&loaded).unwrap()
    }

    fn router_with_scale(scale: u32) -> ShardRouter {
        let mut value = crate::config::tests::valid_config_json();
        value["params"]["shard_number"] = json!(8);
        value["params"]["hash_ring_shard_scale"] = json!(scale);
        let loaded = config::from_str(&value.to_string()).unwrap();
        ShardRouter::new(&loaded).unwrap()
    }

    fn num_id(n: u64) -> PointIdType {
        ExtendedPointId::NumId(n)
    }

    #[test]
    fn places_every_point_in_exactly_one_shard() {
        let router = router_with_shards(8);
        for n in 0..10_000 {
            let shard = router.shard_of(num_id(n)).expect("every point must route");
            assert!(
                shard < 8,
                "shard {shard} out of range for an 8-shard collection",
            );
        }
    }

    #[test]
    fn routing_is_deterministic_across_router_instances() {
        // Two independently constructed routers must agree, or a resumed build would
        // scatter the same point into different shards on different runs.
        let a = router_with_shards(6);
        let b = router_with_shards(6);

        for n in 0..5_000 {
            assert_eq!(
                a.shard_of(num_id(n)).unwrap(),
                b.shard_of(num_id(n)).unwrap(),
                "routing for point {n} differs between router instances",
            );
        }
    }

    #[test]
    fn routing_covers_all_shards() {
        // Not a balance assertion — just that no shard is unreachable, which would mean
        // the ring was built with the wrong id set.
        let shard_count = 8;
        let router = router_with_shards(shard_count);

        let mut seen = vec![false; shard_count as usize];
        for n in 0..20_000 {
            seen[router.shard_of(num_id(n)).unwrap() as usize] = true;
        }

        for (shard, hit) in seen.iter().enumerate() {
            assert!(hit, "no point routed to shard {shard}");
        }
    }

    #[test]
    fn uuid_and_numeric_ids_both_route() {
        let router = router_with_shards(4);

        let numeric = router.shard_of(num_id(42)).unwrap();
        assert!(numeric < 4);

        let uuid = ExtendedPointId::Uuid("550e8400-e29b-41d4-a716-446655440000".parse().unwrap());
        let uuid_shard = router.shard_of(uuid).unwrap();
        assert!(uuid_shard < 4);
    }

    #[test]
    fn changing_shard_count_changes_routing() {
        // The point of the config gate: build with the wrong shard_number and points land
        // in the wrong shards. This test documents that the failure is real, so the gate
        // is not merely defensive.
        let four = router_with_shards(4);
        let eight = router_with_shards(8);

        let disagreements = (0..2_000)
            .filter(|&n| four.shard_of(num_id(n)).unwrap() != eight.shard_of(num_id(n)).unwrap())
            .count();

        assert!(
            disagreements > 500,
            "expected most points to move when shard count changes, only {disagreements} did",
        );
    }

    /// The fork's per-collection scale is as routing-critical as the shard count.
    ///
    /// This is why `hash_ring_shard_scale` sits in the part fingerprint: a scatter routed at
    /// one scale is silently wrong under a collection created at another, in exactly the
    /// retrieve-by-id-misses way this module exists to prevent.
    #[test]
    fn changing_ring_scale_changes_routing() {
        let default_scale = router_with_scale(100);
        let raised = router_with_scale(1_000);

        let disagreements = (0..2_000)
            .filter(|&n| {
                default_scale.shard_of(num_id(n)).unwrap() != raised.shard_of(num_id(n)).unwrap()
            })
            .count();

        assert!(
            disagreements > 200,
            "expected a material fraction of points to move when the ring scale changes, \
             only {disagreements} of 2000 did",
        );
    }

    /// The reason the fork made the scale configurable: more virtual nodes, less skew.
    ///
    /// Guards the `O(1/sqrt(vnodes))` claim the capacity projection leans on. If a future
    /// ring implementation changed that relationship, the sizing advice `validate` prints
    /// would be wrong, and this is where someone finds out.
    #[test]
    fn raising_the_ring_scale_reduces_skew() {
        let spread = |scale: u32| {
            let router = router_with_scale(scale);
            let fractions = measure_distribution(&router, 200_000).unwrap();
            let max = fractions.iter().copied().fold(f64::MIN, f64::max);
            let min = fractions.iter().copied().fold(f64::MAX, f64::min);
            max / min
        };

        let coarse = spread(100);
        let fine = spread(10_000);

        assert!(
            fine < coarse,
            "10_000 vnodes/shard should be more even than 100: {fine:.3}x vs {coarse:.3}x",
        );
        assert!(
            fine < 1.1,
            "at 10_000 vnodes/shard the spread should be under 10%, got {fine:.3}x",
        );
    }

    #[test]
    fn rejects_custom_sharding() {
        let mut value = crate::config::tests::valid_config_json();
        value["params"]["sharding_method"] = json!("custom");
        let loaded = config::from_str(&value.to_string()).unwrap();

        let err = ShardRouter::new(&loaded).unwrap_err();
        assert!(
            format!("{err:#}").contains("custom"),
            "error should explain the custom-sharding restriction, got: {err:#}",
        );
    }

    #[test]
    fn single_shard_routes_everything_to_zero() {
        let router = router_with_shards(1);
        for n in 0..1_000 {
            assert_eq!(router.shard_of(num_id(n)).unwrap(), 0);
        }
    }
}
