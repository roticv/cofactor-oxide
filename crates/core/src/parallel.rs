//! Splitting work across threads, which `core` can't start itself: the
//! caller supplies them through [`Parallel`].

use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

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

/// `f(i)` for each `i` in `0..n`, in order, shared out among `par`'s
/// threads as they become free, so that faster cores do more.
pub(crate) fn map_dynamic<T: Send>(
    par: &impl Parallel,
    n: usize,
    f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let mut done: Vec<(usize, T)> = par
        .map(|_| {
            let mut done = Vec::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    return done;
                }
                done.push((i, f(i)));
            }
        })
        .into_iter()
        .flatten()
        .collect();
    done.sort_unstable_by_key(|&(i, _)| i);
    done.into_iter().map(|(_, t)| t).collect()
}
