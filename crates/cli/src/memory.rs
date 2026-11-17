//! The allocator: the system's, except that blocks of a megabyte or more
//! are mapped from the operating system, and unmapped when freed unless a
//! few of them are kept to reuse.
//!
//! macOS's malloc keeps freed large blocks to reuse, without much limit,
//! and they count against the process until it does. The GNFS square root
//! allocates and frees transforms of tens of megabytes on every thread, so
//! at RSA-120 its 1.3 GB of live memory had a 3.0 GB footprint. Here at
//! most [`KEPT`] freed blocks, [`KEPT_BYTES`] in all, wait to be reused,
//! which is what block Lanczos needs: it frees and takes vectors of the
//! same few sizes at every step, and without them it was 10% slower.

use std::alloc::{GlobalAlloc, Layout, System};

/// Blocks from this size up are mapped.
#[cfg(unix)]
const LARGE: usize = 1 << 20;

/// The most freed mappings kept, and the most bytes in them.
#[cfg(unix)]
const KEPT: usize = 16;
#[cfg(unix)]
const KEPT_BYTES: usize = 256 << 20;

pub struct Allocator;

#[cfg(unix)]
mod os {
    use core::ffi::c_void;
    use core::sync::atomic::{AtomicBool, Ordering};

    use super::{KEPT, KEPT_BYTES};

    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_PRIVATE: i32 = 2;
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
    const MAP_ANON: i32 = 0x1000;
    #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "freebsd")))]
    const MAP_ANON: i32 = 0x20;
    const MAP_FAILED: *mut c_void = !0 as *mut c_void;

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
    }

    /// Freed mappings waiting to be reused, as `(address, size)`, oldest
    /// first, behind a spin lock: a lock that might allocate can't be
    /// taken inside the allocator, and blocks this large come and go
    /// rarely enough that it is never held for long.
    struct Kept {
        blocks: [(usize, usize); KEPT],
        len: usize,
        bytes: usize,
    }

    static LOCK: AtomicBool = AtomicBool::new(false);
    static mut KEPT_BLOCKS: Kept = Kept {
        blocks: [(0, 0); KEPT],
        len: 0,
        bytes: 0,
    };

    /// Runs `f` on the kept blocks, holding the lock.
    fn with_kept<T>(f: impl FnOnce(&mut Kept) -> T) -> T {
        while LOCK
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        // SAFETY: the lock makes this the only reference.
        let t = f(unsafe { &mut *core::ptr::addr_of_mut!(KEPT_BLOCKS) });
        LOCK.store(false, Ordering::Release);
        t
    }

    /// `size` bytes, page-aligned: a kept block of that size if there is
    /// one, whose contents are left over, or else a fresh mapping, which is
    /// zero. Returns the block and whether it is known to be zero; null if
    /// mapping fails.
    pub fn take(size: usize) -> (*mut u8, bool) {
        let kept = with_kept(|k| {
            let i = k.blocks[..k.len].iter().rposition(|&(_, s)| s == size)?;
            let (p, _) = k.blocks[i];
            k.blocks.copy_within(i + 1..k.len, i);
            k.len -= 1;
            k.bytes -= size;
            Some(p as *mut u8)
        });
        if let Some(p) = kept {
            return (p, false);
        }
        // SAFETY: an anonymous private mapping touches no existing memory.
        let p = unsafe {
            mmap(
                core::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if p == MAP_FAILED {
            (core::ptr::null_mut(), true)
        } else {
            (p.cast(), true)
        }
    }

    /// Takes back a block from [`take`] of `size` bytes: kept, if it fits,
    /// letting go of the oldest kept to make room, or else unmapped.
    ///
    /// # Safety
    ///
    /// `p` must be such a block, not used after.
    pub unsafe fn give(p: *mut u8, size: usize) {
        let mut evicted = [(0usize, 0usize); KEPT];
        let mut n = 0;
        let kept = size <= KEPT_BYTES / 4
            && with_kept(|k| {
                while k.len > 0 && (k.len == KEPT || k.bytes + size > KEPT_BYTES) {
                    evicted[n] = k.blocks[0];
                    n += 1;
                    k.bytes -= k.blocks[0].1;
                    k.blocks.copy_within(1..k.len, 0);
                    k.len -= 1;
                }
                k.blocks[k.len] = (p as usize, size);
                k.len += 1;
                k.bytes += size;
                true
            });
        if !kept {
            evicted[n] = (p as usize, size);
            n += 1;
        }
        for &(q, s) in &evicted[..n] {
            // SAFETY: each is a mapping of its size, no longer kept or used.
            unsafe { munmap(q as *mut c_void, s) };
        }
    }
}

impl Allocator {
    /// Whether a block of `layout` is mapped: large, and aligned no more
    /// than a page, which every mapping is.
    #[cfg(unix)]
    fn mapped(layout: Layout) -> bool {
        layout.size() >= LARGE && layout.align() <= 4096
    }

    #[cfg(not(unix))]
    fn mapped(_: Layout) -> bool {
        false
    }
}

// SAFETY: every block is either the system allocator's, with the same
// layout, or a mapping of its exact size, told apart by the layout, which
// the caller passes back unchanged.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        #[cfg(unix)]
        if Self::mapped(layout) {
            return os::take(layout.size()).0;
        }
        // SAFETY: as the caller promises.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        #[cfg(unix)]
        if Self::mapped(layout) {
            let (p, zero) = os::take(layout.size());
            if !zero {
                // SAFETY: a kept block of exactly this size.
                unsafe { core::ptr::write_bytes(p, 0, layout.size()) };
            }
            return p;
        }
        // SAFETY: as the caller promises.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        #[cfg(unix)]
        if Self::mapped(layout) {
            // SAFETY: a large block of this layout came from `os::take`.
            unsafe { os::give(p, layout.size()) };
            return;
        }
        // SAFETY: as the caller promises.
        unsafe { System.dealloc(p, layout) }
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: the caller promises a nonzero size that, rounded up to
        // the alignment, doesn't overflow.
        let new = unsafe { Layout::from_size_align_unchecked(size, layout.align()) };
        if !Self::mapped(layout) && !Self::mapped(new) {
            // SAFETY: as the caller promises.
            return unsafe { System.realloc(p, layout, size) };
        }
        // SAFETY: `new` is a valid layout, as above.
        let q = unsafe { self.alloc(new) };
        if !q.is_null() {
            // SAFETY: both blocks hold at least the bytes copied, and are
            // distinct; the old one goes once copied.
            unsafe {
                core::ptr::copy_nonoverlapping(p, q, layout.size().min(size));
                self.dealloc(p, layout);
            }
        }
        q
    }
}
