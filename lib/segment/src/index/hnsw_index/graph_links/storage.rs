//! In-memory vs. universal-IO storage backing for serialized graph links.
//!
//! [`GraphLinksEnum`] is the owned half of [`GraphLinks`](super::GraphLinks):
//! it holds the serialized bytes, either resident in RAM or behind a live
//! universal-IO file handle. The handle is type-erased through
//! [`GraphLinksStorage`] so that [`GraphLinks`](super::GraphLinks) stays
//! non-generic.

use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use common::mmap::advice::dontneed_range;
use common::universal_io::UniversalRead;
use memmap2::Mmap;

use crate::common::operation_error::{OperationError, OperationResult};

/// Type-erased universal-IO storage backing a [`GraphLinksEnum::Universal`].
///
/// [`UniversalRead`] is not object-safe (it is `Sized` and has generic
/// methods), so the storage handle is kept behind this minimal object-safe
/// trait. It is blanket-implemented for every [`UniversalRead`], which lets
/// the links keep an arbitrary universal-IO file handle alive (mirroring the
/// former mmap-backed variant) without making [`GraphLinks`](super::GraphLinks)
/// generic.
pub(super) trait GraphLinksStorage: Debug + Send + Sync {
    /// Borrow the whole serialized links blob.
    ///
    /// The backing storage must be borrowable (i.e. mmap-backed): backends
    /// that materialize the whole file into an owned buffer on read are not
    /// supported here.
    fn bytes(&self) -> OperationResult<&[u8]>;

    /// Populate the OS page cache for the backing file, if applicable.
    fn populate(&self) -> OperationResult<()>;

    /// Hint to the OS that the backing pages can be reclaimed, if applicable.
    fn clear_cache(&self) -> OperationResult<()>;
}

impl<S: UniversalRead> GraphLinksStorage for S {
    fn bytes(&self) -> OperationResult<&[u8]> {
        match self.read_whole::<u8>()? {
            Cow::Borrowed(bytes) => Ok(bytes),
            Cow::Owned(_) => Err(OperationError::service_error(
                "Universal graph links storage must be borrowable (mmap-backed)",
            )),
        }
    }

    fn populate(&self) -> OperationResult<()> {
        UniversalRead::populate(self)?;
        Ok(())
    }

    fn clear_cache(&self) -> OperationResult<()> {
        self.clear_ram_cache()?;
        Ok(())
    }
}

/// Platform-independent advice for the ranged madvise helper. Only the two
/// variants the [`GraphLinksEnum::MmapRanged`] arms need — kept local to this
/// module so we don't enlarge the surface of `common::mmap::advice::Advice` for
/// one caller. Mirrors that module's `#[cfg(unix)] impl From<...> for
/// memmap2::Advice` pattern so `memmap2::Advice` and `memmap2::UncheckedAdvice`
/// never appear outside a `cfg(unix)` gate.
#[derive(Copy, Clone, Debug)]
enum RangeAdvice {
    WillNeed,
    DontNeed,
}

/// Issue a range-scoped `madvise` on the given mmap. Log-and-swallow on error
/// to match the convention of the full-map `populate` / `clear_cache`
/// implementations (advisory calls; failure is non-fatal). On non-Unix both
/// branches are no-ops (matching `Madviseable::advise_impl` and
/// `dontneed_range`'s cfg fallback).
///
/// This function contains no `unsafe`. `WillNeed` uses memmap2's safe
/// `Mmap::advise_range`; `DontNeed` delegates to
/// `common::mmap::advice::dontneed_range`, whose type-level `&Mmap` (read-only,
/// file-backed) precondition eliminates the data-loss path that motivates
/// memmap2's `UncheckedAdvice` gating.
fn madvise_range(mmap: &Mmap, offset: usize, length: usize, advice: RangeAdvice) {
    let res: std::io::Result<()> = match advice {
        RangeAdvice::WillNeed => {
            #[cfg(unix)]
            {
                mmap.advise_range(memmap2::Advice::WillNeed, offset, length)
            }
            #[cfg(not(unix))]
            {
                let _ = (mmap, offset, length);
                Ok(())
            }
        }
        RangeAdvice::DontNeed => dontneed_range(mmap, offset, length),
    };
    if let Err(err) = res {
        log::warn!(
            "madvise({advice:?}) on links range [{offset}..{}] failed: {err}",
            offset + length,
        );
    }
}

#[derive(Debug)]
pub(super) enum GraphLinksEnum {
    /// Links built in memory (e.g. freshly serialized from edges).
    Ram(Vec<u8>),
    /// Links backed by a (type-erased) universal-IO storage handle.
    Universal(Box<dyn GraphLinksStorage>),
    /// A sub-range of a mmap shared with other blobs in a single container file
    /// (see spec §6.3). The `Arc<Mmap>` is owned by the self_cell exactly as a
    /// borrowable [`GraphLinksEnum::Universal`] handle is; `offset` + `length`
    /// bound the links slice inside it.
    ///
    /// Kept as its own variant rather than a [`GraphLinksStorage`] impl because
    /// that trait is blanket-implemented for every [`UniversalRead`], so a
    /// concrete impl would overlap; and a ranged view over an already-open mmap
    /// has no file handle of its own to model as a `UniversalRead` backend.
    MmapRanged {
        mmap: Arc<Mmap>,
        offset: usize,
        length: usize,
    },
}

impl GraphLinksEnum {
    /// Build the backing for serialized links from a universal-IO file handle.
    ///
    /// Backends whose data is resident in RAM or mapped into the address space
    /// (`UniversalKind::is_in_ram_or_mmap`) yield borrowable reads, so their
    /// handle is kept live as [`GraphLinksEnum::Universal`]. Any other backend
    /// (io_uring, remote object stores, …) is not borrowable, so its contents
    /// are materialized into RAM as [`GraphLinksEnum::Ram`]. This is what
    /// upholds the borrowability invariant relied on by [`GraphLinksStorage::bytes`],
    /// so that error path is unreachable in practice.
    pub(super) fn from_storage<S: UniversalRead + 'static>(storage: S) -> OperationResult<Self> {
        if S::kind().is_in_ram_or_mmap() {
            Ok(GraphLinksEnum::Universal(Box::new(storage)))
        } else {
            Self::pinned_from_storage(storage)
        }
    }

    /// Materialize the whole links blob into an anonymous heap allocation
    /// ([`GraphLinksEnum::Ram`]), regardless of the backend.
    pub(super) fn pinned_from_storage<S: UniversalRead>(storage: S) -> OperationResult<Self> {
        let bytes = storage.read_whole::<u8>()?.into_owned();
        // The heap copy is authoritative from here on: evict whatever the read
        // left in the OS page cache or backend caches, so the links are not
        // resident twice.
        storage.clear_ram_cache()?;
        Ok(GraphLinksEnum::Ram(bytes))
    }

    pub(super) fn as_bytes(&self) -> OperationResult<&[u8]> {
        match self {
            GraphLinksEnum::Ram(data) => Ok(data.as_slice()),
            GraphLinksEnum::Universal(storage) => storage.bytes(),
            GraphLinksEnum::MmapRanged {
                mmap,
                offset,
                length,
            } => Ok(&mmap[*offset..*offset + *length]),
        }
    }

    /// Heap RAM held by the links themselves, in bytes.
    ///
    /// Non-zero only for [`GraphLinksEnum::Ram`], i.e. freshly built links or
    /// links materialized from a non-borrowable universal-IO backend. Storage
    /// kept behind a live handle ([`GraphLinksEnum::Universal`]) is backed by
    /// the OS page cache and reported via file residency instead.
    pub(super) fn heap_size_bytes(&self) -> usize {
        match self {
            GraphLinksEnum::Ram(data) => data.len(),
            // Backed by the shared container mmap, i.e. the OS page cache —
            // reported via file residency, not as heap. Also avoids
            // double-counting the mmap across every blob that shares it.
            GraphLinksEnum::Universal(_) | GraphLinksEnum::MmapRanged { .. } => 0,
        }
    }

    /// Populate the OS page cache for the backing storage, if applicable.
    pub(super) fn populate(&self) -> OperationResult<()> {
        match self {
            GraphLinksEnum::Universal(storage) => storage.populate(),
            GraphLinksEnum::MmapRanged {
                mmap,
                offset,
                length,
            } => {
                madvise_range(mmap, *offset, *length, RangeAdvice::WillNeed);
                Ok(())
            }
            GraphLinksEnum::Ram(_) => Ok(()),
        }
    }

    /// Hint to the OS that the backing pages can be reclaimed, if applicable.
    pub(super) fn clear_cache(&self) -> OperationResult<()> {
        match self {
            GraphLinksEnum::Universal(storage) => storage.clear_cache(),
            GraphLinksEnum::MmapRanged {
                mmap,
                offset,
                length,
            } => {
                madvise_range(mmap, *offset, *length, RangeAdvice::DontNeed);
                Ok(())
            }
            GraphLinksEnum::Ram(_) => Ok(()),
        }
    }
}
