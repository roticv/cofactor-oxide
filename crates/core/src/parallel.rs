//! Splitting work across threads, which `core` can't start itself: the
//! caller supplies them through [`Parallel`].

use alloc::vec;
use alloc::vec::Vec;

/// Runs pieces of work, perhaps at once on several threads.
pub trait Parallel: Sync {
    /// How many pieces work is split into.
    fn threads(&self) -> usize;

    /// `f(i)` for each `i` in `0..threads()`, in order, returning once all
    /// are done.
    fn map<T: Send>(&self, f: impl Fn(usize) -> T + Sync) -> Vec<T>;
}

/// Everything on the calling thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct Serial;

impl Parallel for Serial {
    fn threads(&self) -> usize {
        1
    }

    fn map<T: Send>(&self, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
        vec![f(0)]
    }
}

/// `pieces + 1` bounds splitting `0..weights.len()` into runs of about
/// equal total weight.
pub(crate) fn split(weights: impl Iterator<Item = usize> + Clone, pieces: usize) -> Vec<usize> {
    let total: usize = weights.clone().sum();
    let mut bounds = vec![0];
    let (mut sum, mut len) = (0, 0);
    for w in weights {
        sum += w;
        len += 1;
        if bounds.len() < pieces && sum * pieces >= total * bounds.len() {
            bounds.push(len);
        }
    }
    while bounds.len() <= pieces {
        bounds.push(len);
    }
    bounds
}
