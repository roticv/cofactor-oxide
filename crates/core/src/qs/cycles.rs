//! Combining relations with large primes through cycles.
//!
//! A partial relation is an edge between its two large primes, with the
//! vertex 1 standing in for a missing second one: `[1, L]` joins 1 and
//! `L`, and `[L₁, L₂]` joins `L₁` and `L₂`. The relations along a cycle
//! have every large prime an even number of times (each vertex on a cycle
//! has two of its edges), so their product has a square cofactor and makes
//! a row of the matrix. Two relations sharing one large prime are the
//! smallest cycle: 1, `L`, 1.
//!
//! The graph is kept as a spanning forest, with each tree rooted somewhere
//! and every other vertex pointing to its parent. An edge between two
//! trees joins them; an edge within a tree closes exactly one new cycle:
//! the edge itself and the tree path between its ends.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// The spanning forest of the large-prime graph.
#[derive(Clone, Debug, Default)]
pub struct Cycles {
    /// Vertex numbers for the primes seen so far.
    vertices: BTreeMap<u64, usize>,
    /// For each vertex, its parent and the relation on the edge to it, or
    /// `None` for a root.
    parent: Vec<Option<(usize, usize)>>,
    /// For each root, the size of its tree.
    size: Vec<usize>,
    /// Scratch marks for finding where two paths to the root meet, with a
    /// fresh stamp each time.
    mark: Vec<u32>,
    stamp: u32,
}

impl Cycles {
    fn vertex(&mut self, prime: u64) -> usize {
        let next = self.parent.len();
        let v = *self.vertices.entry(prime).or_insert(next);
        if v == next {
            self.parent.push(None);
            self.size.push(1);
            self.mark.push(0);
        }
        v
    }

    fn root(&self, mut v: usize) -> usize {
        while let Some((p, _)) = self.parent[v] {
            v = p;
        }
        v
    }

    /// Makes `v` the root of its tree, reversing the edges on its path to
    /// the old root.
    fn reroot(&mut self, v: usize) {
        let old = self.root(v);
        let size = self.size[old];
        let mut prev = None;
        let mut cur = v;
        loop {
            let next = self.parent[cur];
            self.parent[cur] = prev;
            match next {
                Some((p, relation)) => {
                    prev = Some((cur, relation));
                    cur = p;
                }
                None => break,
            }
        }
        self.size[v] = size;
    }

    /// Adds the edge for relation `relation`, with large primes `a` and
    /// `b` (1 for none). If it closes a cycle, returns the relations on it,
    /// this one included.
    pub fn add(&mut self, a: u64, b: u64, relation: usize) -> Option<Vec<usize>> {
        let (u, v) = (self.vertex(a), self.vertex(b));
        let (ru, rv) = (self.root(u), self.root(v));
        if ru != rv {
            // Hang the smaller tree from the other one, by its endpoint.
            let (small, big, from, to) = if self.size[ru] < self.size[rv] {
                (ru, rv, u, v)
            } else {
                (rv, ru, v, u)
            };
            let moved = self.size[small];
            self.reroot(from);
            self.parent[from] = Some((to, relation));
            self.size[big] += moved;
            return None;
        }
        // The tree path from u to v: up from u to where it meets the path
        // up from v.
        self.stamp += 1;
        let mut x = u;
        self.mark[x] = self.stamp;
        while let Some((p, _)) = self.parent[x] {
            x = p;
            self.mark[x] = self.stamp;
        }
        let mut cycle = alloc::vec![relation];
        let mut meet = v;
        while self.mark[meet] != self.stamp {
            let (p, r) = self.parent[meet].expect("u and v share a root");
            cycle.push(r);
            meet = p;
        }
        let mut x = u;
        while x != meet {
            let (p, r) = self.parent[x].expect("meet is above u");
            cycle.push(r);
            x = p;
        }
        Some(cycle)
    }
}
