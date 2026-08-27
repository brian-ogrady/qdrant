//! Blocked Bloom Filter pre-screen for external → internal id lookups.
//!
//! To turn off the bloom filter, set `QDRANT_ID_TRACKER_BLOOM_FILTER=0`
//! (or `false`/`off`/`no`)
//!
//! # Why this needs no delete handling and no persistence
//!
//! The filter tracks *ever inserted*, which is a superset of *currently live*.
//! That makes both of the usual Bloom-filter complications disappear:
//!
//! - **Deletes need no hook.** Dropping a point leaves its bits set. The stale
//!   bit costs a false positive, which falls through to the `BTreeMap` and gets
//!   the correct answer — exactly today's behaviour. The one invariant is that
//!   a bit is *never cleared*, so there are no false negatives.
//! - **Restarts need no format.** The filter is never written to disk.
//!   `MutableIdTracker::open` always starts from an empty `PointMappings` and
//!   replays the persisted change log through `set_link`, so the same insert
//!   hook that maintains the filter at runtime rebuilds it during recovery.
//!
//! Stale bits accumulate over a segment's life, but appendable segments are
//! rolled into fresh ones by the optimizer, and each new segment builds a fresh
//! filter — so the false-positive rate is bounded by the segment lifecycle
//! rather than by any compaction logic here.
//!

use std::fmt;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::types::PointIdType;

/// Lanes per block. Each key sets exactly one bit in every lane, so a lookup
/// is eight independent lane tests with no dependency chain between them —
/// the shape LLVM turns into vector code.
const LANES: usize = 8;

/// A block is [`LANES`] x 32 bits = 32 bytes. Aligned to its own size, so it
/// never straddles a cache line: one probe, one cache miss.
const BITS_PER_BLOCK: usize = LANES * u32::BITS as usize;

/// ~1.3% false-positive rate for a split-block filter at this sizing.
const BITS_PER_KEY: usize = 10;

/// Odd multipliers, one per lane, from the Impala/Parquet split-block filter.
/// Multiplying the key by a distinct odd constant and keeping the top 5 bits
/// gives each lane an independent bit index in `0..32`.
const SALT: [u32; LANES] = [
    0x47b6_137b,
    0x4497_4d91,
    0x8824_ad5b,
    0xa2b7_289d,
    0x7054_95c7,
    0x2df1_424b,
    0x9efc_4947,
    0x5c6b_fb31,
];

/// Smallest filter worth allocating: 5 KiB
const MIN_CAPACITY: usize = 4096;

/// Separates the numeric and UUID key spaces so a UUID whose low bits happen to
/// equal a numeric id does not systematically collide with it.
const UUID_DOMAIN: u64 = 0xD1B5_4A32_D192_ED03;

/// Folds a UUID's high half into its low half before mixing. Odd, so the
/// multiply is invertible and cannot collapse distinct high halves.
const UUID_FOLD: u64 = 0x9E37_79B9_7F4A_7C15;

/// Environment variable that turns the pre-screen off.
///
/// Set to `0`, `false`, `off`, or `no` to disable. Anything else, or unset,
/// leaves it on.
pub const ENABLE_ENV_VAR: &str = "QDRANT_ID_TRACKER_BLOOM_FILTER";

/// Whether filters created from here on are enabled. Read from the environment
/// once, then overridable via [`set_enabled`].
static ENABLED: LazyLock<AtomicBool> = LazyLock::new(|| {
    let enabled = enabled_from_env();
    // Logged once per process, on the first filter built. Without it a running
    // node gives no way to tell which mode it is in, which matters most while
    // rolling the pre-screen out or bisecting a regression against it.
    if enabled {
        log::debug!("External id pre-screen enabled (disable with {ENABLE_ENV_VAR}=0)");
    } else {
        log::info!("External id pre-screen disabled via {ENABLE_ENV_VAR}");
    }
    AtomicBool::new(enabled)
});

/// Whether new filters are enabled.
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Turn the pre-screen on or off for filters created from here on.
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// Parse the toggle from a raw value. Absent, empty, or anything not spelled
/// like "off" leaves the pre-screen on.
fn enabled_from_value(value: Option<&str>) -> bool {
    match value {
        Some(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        None => true,
    }
}

fn enabled_from_env() -> bool {
    enabled_from_value(std::env::var(ENABLE_ENV_VAR).ok().as_deref())
}

/// One block of the filter: [`LANES`] 32-bit lanes, aligned to its own size so
/// it cannot straddle a cache line. All bits for a key live in one block, so a
/// negative lookup is a single cache miss instead of one per `BTreeMap` level.
#[repr(align(32))]
#[derive(Clone, Copy, Default)]
struct Block([u32; LANES]);

/// Set-membership pre-screen over the external ids held by a `PointMappings`.
///
/// A `false` from [`maybe_contains`](Self::maybe_contains) is authoritative:
/// the id is definitely absent. A `true` means "maybe" and must be confirmed
/// against the map.
#[derive(Clone)]
pub struct ExternalIdFilter {
    /// Empty when the filter is disabled, in which case every lookup answers
    /// "maybe" and the caller always falls through to the map.
    blocks: Vec<Block>,
    /// Key count this filter was sized for.
    capacity: usize,
    /// Keys recorded since it was built. Counts insertions, not live keys —
    /// re-inserting an existing key still moves it, which is what makes it a
    /// conservative trigger for a rebuild.
    inserted: usize,
}

impl ExternalIdFilter {
    /// Dummy filter that allocates nothing and screens nothing out. Used when the
    /// toggle is off.
    pub fn disabled() -> Self {
        Self {
            blocks: Vec::new(),
            capacity: 0,
            inserted: 0,
        }
    }

    /// If the toggle is on, allocate a filter sized for at least `capacity` keys
    /// If the toggle is off, return the disabled filter.
    pub fn for_new_mapping(capacity: usize) -> Self {
        if is_enabled() {
            Self::with_capacity(capacity)
        } else {
            Self::disabled()
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(MIN_CAPACITY);
        // Saturating on purpose: a wrapping multiply could turn an absurd
        // capacity into a tiny block count, silently producing a filter that
        // is over-saturated from birth. Saturating fails loudly at allocation
        // instead.
        let block_count = capacity
            .saturating_mul(BITS_PER_KEY)
            .div_ceil(BITS_PER_BLOCK)
            .max(1);
        Self {
            blocks: vec![Block::default(); block_count],
            capacity,
            inserted: 0,
        }
    }

    /// Whether the filter has taken more keys than it was sized for and should
    /// be rebuilt against the live key set.
    pub fn is_saturated(&self) -> bool {
        !self.blocks.is_empty() && self.inserted > self.capacity
    }

    /// Headroom to size a rebuild for, given the current live key count.
    /// The doubling is what keeps rebuilds geometric, so inserts amortise to
    /// O(1) even though each rebuild is O(live keys).
    pub fn rebuild_capacity(live_keys: usize) -> usize {
        live_keys.saturating_mul(2).max(MIN_CAPACITY)
    }

    /// Record `external_id` as present. Never clears bits, so the filter stays
    /// a superset of the live key set.
    #[inline]
    pub fn insert(&mut self, external_id: &PointIdType) {
        if self.blocks.is_empty() {
            return;
        }
        self.inserted += 1;
        let hash = hash_point_id(external_id);
        let block_index = fastrange(hash, self.blocks.len());
        let block = &mut self.blocks[block_index];
        for (lane, mask) in block.0.iter_mut().zip(lane_masks(hash)) {
            *lane |= mask;
        }
    }

    /// `false` means `external_id` is definitely absent; `true` means it may be
    /// present and the caller must consult the map.
    #[inline]
    pub fn maybe_contains(&self, external_id: &PointIdType) -> bool {
        if self.blocks.is_empty() {
            // Disabled filter: no information, so never screen anything out.
            return true;
        }
        let hash = hash_point_id(external_id);
        let block_index = fastrange(hash, self.blocks.len());
        let block = &self.blocks[block_index];
        // Accumulate the missing bits rather than short-circuiting: branchless
        // and vectorizable, and a miss has to test every lane anyway.
        let mut missing = 0u32;
        for (lane, mask) in block.0.iter().zip(lane_masks(hash)) {
            missing |= mask & !lane;
        }
        missing == 0
    }

    /// Approximate RAM usage in bytes.
    pub fn ram_usage_bytes(&self) -> usize {
        self.blocks.capacity() * std::mem::size_of::<Block>()
    }
}

impl Default for ExternalIdFilter {
    fn default() -> Self {
        Self::for_new_mapping(MIN_CAPACITY)
    }
}

impl fmt::Debug for ExternalIdFilter {
    /// Compact debug output that omits the backing bitset.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalIdFilter")
            .field("blocks", &self.blocks.len())
            .field("bytes", &self.ram_usage_bytes())
            .finish()
    }
}

/// The bit each lane holds for `hash`: lane `i` takes the top 5 bits of
/// `key * SALT[i]`, giving an index in `0..32`.
///
/// The eight products are independent, so this compiles to a handful of vector
/// instructions rather than the serial chain a double-hashing probe loop needs.
/// The low half of the hash feeds the lanes while [`fastrange`] selects the
/// block from the high half, keeping the two independent.
#[inline]
fn lane_masks(hash: u64) -> [u32; LANES] {
    let key = hash as u32;
    let mut masks = [0u32; LANES];
    for (mask, salt) in masks.iter_mut().zip(SALT) {
        *mask = 1u32 << (key.wrapping_mul(salt) >> 27);
    }
    masks
}

#[inline]
fn hash_point_id(external_id: &PointIdType) -> u64 {
    match external_id {
        PointIdType::NumId(num) => mix64(*num),
        PointIdType::Uuid(uuid) => {
            // Fold to one word, then mix once. Chaining two mixes doubles the
            // dependency chain — measurably slower, with no better spread.
            let bits = uuid.as_u128();
            let folded = (bits as u64) ^ ((bits >> 64) as u64).wrapping_mul(UUID_FOLD);
            mix64(folded ^ UUID_DOMAIN)
        }
    }
}

/// SplitMix64 finalizer: cheap, dependency-free, and diffuses the sequential
/// numeric ids that dominate real workloads.
#[inline]
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Map `hash` onto `0..len` by taking the high bits of a widening multiply.
/// Avoids both the modulo bias of `%` and the power-of-two rounding a bitmask
/// would force on the allocation.
#[inline]
fn fastrange(hash: u64, len: usize) -> usize {
    ((u128::from(hash) * len as u128) >> 64) as usize
}

/// Serialises tests that flip the process-global toggle
#[cfg(test)]
static TOGGLE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Pins the toggle to a chosen value for the life of the guard, serialising
/// against every other holder and restoring the previous value on drop —
/// including on panic, so one failing test cannot leave the toggle flipped for
/// the rest of the run.
///
/// Tests that assert a filter actually screens must hold one of these rather
/// than trusting the ambient default, which the environment can flip.
#[cfg(test)]
pub(crate) struct ToggleGuard {
    previous: bool,
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl ToggleGuard {
    pub(crate) fn set(enabled: bool) -> Self {
        let lock = TOGGLE_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let previous = is_enabled();
        set_enabled(enabled);
        Self {
            previous,
            _lock: lock,
        }
    }
}

#[cfg(test)]
impl Drop for ToggleGuard {
    fn drop(&mut self) {
        set_enabled(self.previous);
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    fn num(n: u64) -> PointIdType {
        PointIdType::NumId(n)
    }

    fn uuid(n: u128) -> PointIdType {
        PointIdType::Uuid(uuid::Uuid::from_u128(n))
    }

    /// The invariant the whole design rests on: anything inserted must always
    /// report as present. A false negative would silently lose a point.
    #[test]
    fn never_reports_a_false_negative() {
        let mut rng = StdRng::seed_from_u64(0xF11E);
        let mut filter = ExternalIdFilter::with_capacity(100_000);

        let keys: Vec<_> = (0..100_000)
            .map(|i| {
                if i % 3 == 0 {
                    uuid(rng.random())
                } else {
                    num(rng.random())
                }
            })
            .collect();

        for key in &keys {
            filter.insert(key);
        }
        for key in &keys {
            assert!(filter.maybe_contains(key), "false negative for {key}");
        }
    }

    /// Sequential ids are the common real-world case and must not degenerate
    /// into a hot block: the mixer has to spread them across the whole filter.
    #[test]
    fn sequential_ids_have_no_false_negatives() {
        let mut filter = ExternalIdFilter::with_capacity(50_000);
        for i in 0..50_000u64 {
            filter.insert(&num(i));
        }
        for i in 0..50_000u64 {
            assert!(filter.maybe_contains(&num(i)), "false negative for {i}");
        }
    }

    /// The pre-screen only pays off if misses are actually rejected.
    #[test]
    fn rejects_the_vast_majority_of_absent_keys() {
        let mut filter = ExternalIdFilter::with_capacity(100_000);
        for i in 0..100_000u64 {
            filter.insert(&num(i));
        }

        let probes = 100_000u64;
        let false_positives = (1_000_000..1_000_000 + probes)
            .filter(|i| filter.maybe_contains(&num(*i)))
            .count();

        let rate = false_positives as f64 / probes as f64;
        assert!(rate < 0.05, "false positive rate too high: {rate}");
    }

    /// A filter loaded far past its sizing degrades to more false positives,
    /// never to a false negative. This is what makes an un-rebuilt filter safe
    /// between growth points.
    #[test]
    fn oversubscription_degrades_without_false_negatives() {
        let mut filter = ExternalIdFilter::with_capacity(1);
        let keys: Vec<_> = (0..MIN_CAPACITY as u64 * 64).map(num).collect();
        for key in &keys {
            filter.insert(key);
        }
        for key in &keys {
            assert!(filter.maybe_contains(key), "false negative under load");
        }
    }

    /// Growth is driven by insert count crossing the sizing, and a fresh
    /// filter must not immediately ask to be rebuilt.
    #[test]
    fn saturation_tracks_capacity() {
        let mut filter = ExternalIdFilter::with_capacity(MIN_CAPACITY);
        assert!(!filter.is_saturated());
        for i in 0..MIN_CAPACITY as u64 {
            filter.insert(&num(i));
        }
        assert!(!filter.is_saturated(), "at capacity is not yet over it");
        filter.insert(&num(MIN_CAPACITY as u64));
        assert!(filter.is_saturated());
    }

    /// Toggling must be honoured by production construction, and a disabled
    /// filter must cost nothing and screen nothing.
    #[test]
    fn toggle_controls_new_filters() {
        let _guard = ToggleGuard::set(false);
        let off = ExternalIdFilter::for_new_mapping(100_000);
        assert_eq!(
            off.ram_usage_bytes(),
            0,
            "disabled filter must allocate nothing"
        );
        assert!(
            off.maybe_contains(&num(1)),
            "disabled filter must screen nothing out"
        );
        assert_eq!(ExternalIdFilter::default().ram_usage_bytes(), 0);

        set_enabled(true);
        assert!(ExternalIdFilter::for_new_mapping(100_000).ram_usage_bytes() > 0);
        assert!(ExternalIdFilter::default().ram_usage_bytes() > 0);
    }

    /// Parsed as a pure function of the raw value: mutating the process
    /// environment would race every other test that builds a filter.
    #[test]
    fn env_var_parsing() {
        for value in ["0", "false", "FALSE", "off", "no", " off "] {
            assert!(
                !enabled_from_value(Some(value)),
                "{value:?} should disable the filter",
            );
        }
        for value in ["1", "true", "yes", "on", "", "anything"] {
            assert!(
                enabled_from_value(Some(value)),
                "{value:?} should leave the filter on",
            );
        }
        assert!(enabled_from_value(None), "unset should leave the filter on");
    }

    /// A disabled filter screens nothing out, so growing it would spend memory
    /// for no benefit. Reachable only by constructing one explicitly — the
    /// `Default` impl is enabled on purpose.
    #[test]
    fn disabled_filter_never_asks_to_grow() {
        let mut filter = ExternalIdFilter::disabled();
        for i in 0..10_000u64 {
            filter.insert(&num(i));
        }
        assert!(!filter.is_saturated());
        assert_eq!(filter.ram_usage_bytes(), 0);
        assert!(
            filter.maybe_contains(&num(1)),
            "a disabled filter must screen nothing out"
        );
    }

    /// `PointMappings::default()` backs `InMemoryIdTracker`, which backs
    /// `segment_builder`. If that filter were disabled it would never grow, so
    /// every segment build would run unscreened.
    #[test]
    fn default_filter_is_enabled_and_grows() {
        let _guard = ToggleGuard::set(true);
        let mut filter = ExternalIdFilter::default();
        assert!(
            filter.ram_usage_bytes() > 0,
            "default filter must be enabled"
        );
        for i in 0..=MIN_CAPACITY as u64 {
            filter.insert(&num(i));
        }
        assert!(
            filter.is_saturated(),
            "default filter must be able to saturate"
        );
    }

    /// Rebuild sizing must leave headroom, or every subsequent insert would
    /// trigger another O(n) rebuild.
    #[test]
    fn rebuild_capacity_leaves_headroom() {
        assert!(ExternalIdFilter::rebuild_capacity(1_000_000) >= 2_000_000);
        assert_eq!(ExternalIdFilter::rebuild_capacity(0), MIN_CAPACITY);
        // Must not overflow on an absurd input.
        assert!(ExternalIdFilter::rebuild_capacity(usize::MAX) > 0);
    }

    /// Numeric and UUID keys share one filter; inserting one must not be
    /// required to make the other findable, and neither may go missing.
    #[test]
    fn numeric_and_uuid_keys_coexist() {
        let mut filter = ExternalIdFilter::with_capacity(1000);
        for i in 0..1000u64 {
            filter.insert(&num(i));
            filter.insert(&uuid(u128::from(i)));
        }
        for i in 0..1000u64 {
            assert!(filter.maybe_contains(&num(i)));
            assert!(filter.maybe_contains(&uuid(u128::from(i))));
        }
    }

    /// UUID keys must be screened out about as effectively as numeric ones;
    /// a weak fold would show up here as a much higher rate.
    #[test]
    fn rejects_absent_uuid_keys() {
        let mut filter = ExternalIdFilter::with_capacity(100_000);
        for i in 0..100_000u128 {
            filter.insert(&uuid(i * 6364136223846793005));
        }

        let probes = 100_000u128;
        let false_positives = (0..probes)
            .filter(|i| filter.maybe_contains(&uuid(i * 6364136223846793005 + 1)))
            .count();

        let rate = false_positives as f64 / probes as f64;
        assert!(rate < 0.05, "uuid false positive rate too high: {rate}");
    }

    /// UUIDs differing only in their high half must not collide, which is what
    /// the fold multiply protects against.
    #[test]
    fn uuid_high_half_affects_the_hash() {
        let mut filter = ExternalIdFilter::with_capacity(100_000);
        filter.insert(&uuid(1));

        let differing_high: Vec<u128> = (1..2000u128).map(|i| (i << 64) | 1).collect();
        let collisions = differing_high
            .iter()
            .filter(|bits| filter.maybe_contains(&uuid(**bits)))
            .count();
        let rate = collisions as f64 / differing_high.len() as f64;
        assert!(rate < 0.05, "uuid high half is being ignored: {rate}");
    }

    /// A block must be aligned to its own size, so a probe can never straddle
    /// two cache lines. That single-cache-miss property is the whole premise.
    #[test]
    fn block_never_straddles_a_cache_line() {
        assert_eq!(std::mem::size_of::<Block>(), 32);
        assert_eq!(std::mem::align_of::<Block>(), 32);
        assert_eq!(BITS_PER_BLOCK, 256);

        // Alignment is what guarantees it in practice: check real allocations.
        let filter = ExternalIdFilter::with_capacity(100_000);
        for block in &filter.blocks {
            let start = std::ptr::from_ref(block) as usize;
            assert_eq!(
                start / 64,
                (start + std::mem::size_of::<Block>() - 1) / 64,
                "block straddles a cache line",
            );
        }
    }

    /// Each lane must hold exactly one bit, or the false-positive maths and the
    /// branchless lane test below both break.
    #[test]
    fn every_lane_gets_exactly_one_bit() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..10_000 {
            let masks = lane_masks(rng.random());
            assert_eq!(masks.len(), LANES);
            for mask in masks {
                assert_eq!(mask.count_ones(), 1, "lane mask must be a single bit");
            }
        }
    }

    /// The lanes must not move in lockstep: if every salt produced the same bit
    /// index, the filter would behave like a single-hash filter.
    #[test]
    fn lanes_are_independent() {
        let mut rng = StdRng::seed_from_u64(13);
        let mut all_lanes_equal = 0;
        let trials = 10_000;
        for _ in 0..trials {
            let masks = lane_masks(rng.random());
            if masks.iter().all(|m| *m == masks[0]) {
                all_lanes_equal += 1;
            }
        }
        assert!(
            all_lanes_equal < trials / 100,
            "lanes are correlated: {all_lanes_equal}/{trials} had every lane on the same bit",
        );
    }

    #[test]
    fn fastrange_stays_in_bounds() {
        let mut rng = StdRng::seed_from_u64(11);
        for len in [1usize, 3, 20_480, 65_536] {
            for _ in 0..1000 {
                assert!(fastrange(rng.random(), len) < len);
            }
        }
    }
}
