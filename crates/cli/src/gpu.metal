// Bucket fill for the lattice siever, as cofox_core::nfs::BucketPlan
// describes it, for one side and one special-q:
//
//   transform  each ideal's ρ and walk steps, or none where α = 0;
//   count      each group's hits in each slab;
//   scan       per slab, where each group's hits go among the slab's;
//   starts     where each slab's hits start;
//   marks      where each slice's hits start within each slab's;
//   place      each group walks again and writes its hits there.
//
// Groups lie within one slice, so each slice's hits are together in each
// slab, in the order of the groups; within a group, in any order.

#include <metal_stdlib>
using namespace metal;

struct Geo {
    uint n;
    uint groups;
    uint slabs;
    uint slices;
    uint log_i;
    uint log_end;
    uint slab_bits;
    uint cap;
};

struct Basis {
    long ua, ub, va, vb;
};

inline ulong reduce_mod(long x, uint p) {
    long r = x % long(p);
    return ulong(r < 0 ? r + long(p) : r);
}

// a − r·b modulo p, or b for the root at infinity.
inline ulong at(long a, long b, uint p, uint r) {
    ulong bb = reduce_mod(b, p);
    if (r == p) return bb;
    return (reduce_mod(a, p) + (ulong(p) - bb) * ulong(r)) % ulong(p);
}

// The steps (a, ja, b, jb), ja and jb capped at 2^31 − 1; all 0 where α = 0.
kernel void transform(device const uint *P [[buffer(0)]],
                      device const uint *R [[buffer(1)]],
                      device uint4 *ST [[buffer(2)]],
                      constant Basis &uv [[buffer(3)]],
                      constant Geo &geo [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= geo.n) return;
    uint p = P[gid], r = R[gid];
    uint w = 1u << geo.log_i;
    uint x = uint(at(uv.ua, uv.ub, p, r));
    if (x == 0) {
        ST[gid] = uint4(0);
        return;
    }
    ulong beta = at(uv.va, uv.vb, p, r);
    // x⁻¹ mod p by Euclid's algorithm, the cofactors' signs alternating.
    uint r0 = p, r1 = x, t0 = 0, t1 = 1;
    bool odd = false;
    while (r1 != 0) {
        uint k = r0 / r1;
        uint nr = r0 - k * r1;
        r0 = r1;
        r1 = nr;
        uint nt = t0 + k * t1;
        t0 = t1;
        t1 = nt;
        odd = !odd;
    }
    uint inv = (odd || t0 == 0) ? t0 : p - t0;
    uint rho = uint((ulong(p) - beta) * ulong(inv) % ulong(p));
    uint a = p, ja = 0, b = rho, jb = 1;
    if (b == 0) a = w;
    while (a >= w && b >= w) {
        if (a >= b) {
            uint k = a / b;
            a -= k * b;
            ja += k * jb;
        } else {
            uint k = b / a;
            b -= k * a;
            jb += k * ja;
        }
    }
    if (rho != 0) {
        if (a < w) {
            uint k = (b + a - w) / a;
            b -= k * a;
            jb += k * ja;
        } else {
            uint k = (a + b - w) / b;
            a -= k * b;
            ja += k * jb;
        }
    }
    ST[gid] = uint4(a, min(ja, 0x7FFFFFFFu), b, min(jb, 0x7FFFFFFFu));
}

// The hit after pos = j·I + x.
inline ulong step(ulong pos, uint4 s, uint log_i) {
    ulong w = 1ul << log_i;
    ulong x = pos & (w - 1);
    ulong a = s.x, b = s.z;
    bool first = x >= a || x + b >= w;
    bool second = x < a;
    ulong da = first ? (ulong(s.y) << log_i) - a : 0;
    ulong db = second ? (ulong(s.w) << log_i) + b : 0;
    return pos + da + db;
}

// TGC[k · groups + g]: group g's hits in slab k.
kernel void count(device const uint4 *ST [[buffer(0)]],
                  device const uint4 *G [[buffer(1)]],
                  device uint *TGC [[buffer(2)]],
                  constant Geo &geo [[buffer(3)]],
                  threadgroup atomic_uint *hist [[threadgroup(0)]],
                  uint g [[threadgroup_position_in_grid]],
                  uint t [[thread_index_in_threadgroup]],
                  uint nt [[threads_per_threadgroup]]) {
    for (uint k = t; k < geo.slabs; k += nt)
        atomic_store_explicit(&hist[k], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint4 grp = G[g];
    uint i = grp.x + t;
    if (i < grp.y) {
        uint4 s = ST[i];
        if (s.w != 0) {
            ulong end = 1ul << geo.log_end;
            for (ulong pos = step(1ul << (geo.log_i - 1), s, geo.log_i); pos < end;
                 pos = step(pos, s, geo.log_i))
                atomic_fetch_add_explicit(&hist[uint(pos >> geo.slab_bits)], 1,
                                          memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = t; k < geo.slabs; k += nt)
        TGC[ulong(k) * geo.groups + g] = atomic_load_explicit(&hist[k], memory_order_relaxed);
}

// One threadgroup a slab: TGC's counts, in place, become where each group's
// hits start among the slab's, and TOTAL[k] the slab's hits.
kernel void scan(device uint *TGC [[buffer(0)]],
                 device uint *TOTAL [[buffer(1)]],
                 constant Geo &geo [[buffer(2)]],
                 threadgroup uint *part [[threadgroup(0)]],
                 uint k [[threadgroup_position_in_grid]],
                 uint t [[thread_index_in_threadgroup]],
                 uint nt [[threads_per_threadgroup]]) {
    device uint *row = TGC + ulong(k) * geo.groups;
    uint chunk = (geo.groups + nt - 1) / nt;
    uint lo = min(t * chunk, geo.groups), hi = min(lo + chunk, geo.groups);
    uint sum = 0;
    for (uint g = lo; g < hi; g++) sum += row[g];
    part[t] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 1; d < nt; d <<= 1) {
        uint v = t >= d ? part[t - d] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        part[t] += v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint run = part[t] - sum;
    for (uint g = lo; g < hi; g++) {
        uint c = row[g];
        row[g] = run;
        run += c;
    }
    if (t == nt - 1) TOTAL[k] = part[t];
}

kernel void starts(device const uint *TOTAL [[buffer(0)]],
                   device uint *STARTS [[buffer(1)]],
                   constant Geo &geo [[buffer(2)]]) {
    uint n = 0;
    for (uint k = 0; k < geo.slabs; k++) {
        STARTS[k] = n;
        n += TOTAL[k];
    }
    STARTS[geo.slabs] = n;
}

// MARKS[s · slabs + k]: where slice s's hits start among slab k's, from
// FIRST[s], the slice's first group.
kernel void marks(device const uint *TGC [[buffer(0)]],
                  device const uint *FIRST [[buffer(1)]],
                  device uint *MARKS [[buffer(2)]],
                  constant Geo &geo [[buffer(3)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid >= geo.slices * geo.slabs) return;
    uint s = gid / geo.slabs, k = gid % geo.slabs;
    MARKS[gid] = TGC[ulong(k) * geo.groups + FIRST[s]];
}

kernel void place(device const uint4 *ST [[buffer(0)]],
                  device const uint4 *G [[buffer(1)]],
                  device const uint *TGC [[buffer(2)]],
                  device const uint *STARTS [[buffer(3)]],
                  device const uint *TAG [[buffer(4)]],
                  device uint *DATA [[buffer(5)]],
                  constant Geo &geo [[buffer(6)]],
                  threadgroup uint *base [[threadgroup(0)]],
                  threadgroup atomic_uint *next [[threadgroup(1)]],
                  uint g [[threadgroup_position_in_grid]],
                  uint t [[thread_index_in_threadgroup]],
                  uint nt [[threads_per_threadgroup]]) {
    for (uint k = t; k < geo.slabs; k += nt) {
        base[k] = STARTS[k] + TGC[ulong(k) * geo.groups + g];
        atomic_store_explicit(&next[k], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint4 grp = G[g];
    uint i = grp.x + t;
    if (i >= grp.y) return;
    uint4 s = ST[i];
    if (s.w == 0) return;
    uint tag = TAG[i];
    uint mask = (1u << geo.slab_bits) - 1;
    ulong end = 1ul << geo.log_end;
    for (ulong pos = step(1ul << (geo.log_i - 1), s, geo.log_i); pos < end;
         pos = step(pos, s, geo.log_i)) {
        uint k = uint(pos >> geo.slab_bits);
        uint idx = base[k] + atomic_fetch_add_explicit(&next[k], 1, memory_order_relaxed);
        if (idx < geo.cap) DATA[idx] = (uint(pos) & mask) | tag;
    }
}
