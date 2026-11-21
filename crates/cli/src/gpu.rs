//! The siever's bucket fill on the GPU, with Metal: the kernels of
//! `gpu.metal`, compiled when the GPU is first used, so that no Metal
//! compiler is needed to build.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use cofox_core::nfs::{BucketFill, BucketPlan, GROUP};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};

const SOURCE: &str = include_str!("gpu.metal");

/// The kernels, in the order a fill runs them.
const KERNELS: [&str; 6] = ["transform", "count", "scan", "starts", "marks", "place"];

/// Threads a threadgroup for the kernels that aren't by group.
const THREADS: usize = 256;

/// Threadgroup memory a slab takes in `place`, the most of the kernels.
const SLAB_BYTES: usize = 8;

/// Threadgroup memory there is at least, on any Metal GPU.
const THREADGROUP_MEMORY: usize = 16384;

#[repr(C)]
#[derive(Clone, Copy)]
struct Geo {
    n: u32,
    groups: u32,
    slabs: u32,
    slices: u32,
    log_i: u32,
    log_end: u32,
    slab_bits: u32,
    cap: u32,
}

#[repr(C)]
struct Basis {
    ua: i64,
    ub: i64,
    va: i64,
    vb: i64,
}

/// A buffer in memory the CPU and GPU share.
struct Buffer(Retained<ProtocolObject<dyn MTLBuffer>>);

// SAFETY: Metal's buffers may be used from any thread; the ones shared
// between threads are only read once filled, and each `Filler`'s own are
// used by one thread at a time.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    fn new(device: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Option<Self> {
        device
            .newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
            .map(Self)
    }

    fn with(device: &ProtocolObject<dyn MTLDevice>, words: &[u32]) -> Option<Self> {
        let buffer = Self::new(device, size_of_val(words))?;
        buffer.words(words.len()).copy_from_slice(words);
        Some(buffer)
    }

    /// The first `len` words.
    #[allow(clippy::mut_from_ref)]
    fn words(&self, len: usize) -> &mut [u32] {
        assert!(len * 4 <= self.0.length());
        // SAFETY: shared storage, `len` words long, and not in use by the
        // GPU while the CPU reads or writes it.
        unsafe { std::slice::from_raw_parts_mut(self.0.contents().as_ptr().cast(), len) }
    }
}

/// One side's plan on the GPU.
struct Side {
    geo: Geo,
    p: Buffer,
    r: Buffer,
    tag: Buffer,
    groups: Buffer,
    /// Each slice's first group.
    first: Buffer,
}

/// The GPU, its kernels and the sides' plans, shared by the threads'
/// [`Filler`]s.
pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    kernels: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    sides: [Option<Side>; 2],
}

impl Gpu {
    /// The system's GPU set up for `plans`, or `None` if there's none or
    /// it can't be used, saying why.
    pub fn new(plans: &[BucketPlan; 2]) -> Option<Arc<Self>> {
        let device = MTLCreateSystemDefaultDevice()?;
        let library =
            match device.newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None) {
                Ok(library) => library,
                Err(e) => {
                    eprintln!("cofox: can't compile the GPU's kernels, sieving on the CPU: {e}");
                    return None;
                }
            };
        let mut kernels = Vec::new();
        for name in KERNELS {
            let function = library.newFunctionWithName(&NSString::from_str(name))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .ok()?;
            if pipeline.maxTotalThreadsPerThreadgroup() < GROUP.max(THREADS) {
                return None;
            }
            kernels.push(pipeline);
        }
        let mut sides = [None, None];
        for (side, plan) in sides.iter_mut().zip(plans) {
            if plan.p.is_empty() {
                continue;
            }
            if plan.slabs() * SLAB_BYTES > THREADGROUP_MEMORY {
                return None;
            }
            let groups: Vec<u32> = plan
                .groups
                .iter()
                .flat_map(|&[first, end, slice]| [first, end, slice, 0])
                .collect();
            let mut first = vec![u32::MAX; plan.slices as usize];
            for (g, &[_, _, slice]) in plan.groups.iter().enumerate() {
                let f = &mut first[slice as usize];
                *f = (*f).min(g as u32);
            }
            *side = Some(Side {
                geo: Geo {
                    n: plan.p.len() as u32,
                    groups: plan.groups.len() as u32,
                    slabs: plan.slabs() as u32,
                    slices: plan.slices,
                    log_i: plan.log_i,
                    log_end: plan.log_i + plan.log_j,
                    slab_bits: plan.slab_bits,
                    cap: 0,
                },
                p: Buffer::with(&device, &plan.p)?,
                r: Buffer::with(&device, &plan.r)?,
                tag: Buffer::with(&device, &plan.tag)?,
                groups: Buffer::with(&device, &groups)?,
                first: Buffer::with(&device, &first)?,
            });
        }
        let queue = device.newCommandQueue()?;
        Some(Arc::new(Self {
            device,
            queue,
            kernels,
            sides,
        }))
    }

    /// The GPU's name.
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// A filler for one thread's sieve.
    pub fn filler(self: &Arc<Self>) -> Box<dyn BucketFill> {
        let geos = self.sides.iter().flatten().map(|s| s.geo);
        let most = |f: fn(&Geo) -> usize| geos.clone().map(|g| f(&g)).max().unwrap_or(0);
        let device = &self.device;
        let buffer = |words: usize| Buffer::new(device, words * 4).expect("GPU memory");
        Box::new(Filler {
            gpu: Arc::clone(self),
            steps: buffer(most(|g| g.n as usize) * 4),
            counts: buffer(most(|g| g.groups as usize * g.slabs as usize)),
            totals: buffer(most(|g| g.slabs as usize)),
            starts: buffer(most(|g| g.slabs as usize + 1)),
            marks: buffer(most(|g| g.slices as usize * g.slabs as usize)),
            data: [None, None],
        })
    }
}

/// One thread's buffers for filling its sieve's buckets.
struct Filler {
    gpu: Arc<Gpu>,
    steps: Buffer,
    counts: Buffer,
    totals: Buffer,
    starts: Buffer,
    marks: Buffer,
    /// Each side's `data`, where it is, as a buffer over it.
    data: [Option<(usize, usize, Buffer)>; 2],
}

impl Filler {
    /// A buffer over `data`, which a sieve keeps for as long as it has
    /// this filler.
    fn data(&mut self, side: usize, data: &mut [u32]) -> Option<&Buffer> {
        let at = (data.as_mut_ptr() as usize, data.len());
        if self.data[side].as_ref().is_none_or(|d| (d.0, d.1) != at) {
            let pointer = NonNull::new(data.as_mut_ptr().cast::<c_void>())?;
            // SAFETY: `data` is page-aligned and a whole number of pages,
            // as `BucketFill` promises, and outlives the buffer's use.
            let buffer = unsafe {
                self.gpu
                    .device
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        pointer,
                        size_of_val(data),
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
            }?;
            self.data[side] = Some((at.0, at.1, Buffer(buffer)));
        }
        self.data[side].as_ref().map(|d| &d.2)
    }
}

impl BucketFill for Filler {
    fn fill(
        &mut self,
        side: usize,
        u: (i64, i64),
        v: (i64, i64),
        data: &mut [u32],
        starts: &mut [u32],
        marks: &mut [u32],
    ) -> bool {
        let gpu = Arc::clone(&self.gpu);
        let Some(plan) = &gpu.sides[side] else {
            return false;
        };
        let geo = Geo {
            cap: data.len().min(u32::MAX as usize) as u32,
            ..plan.geo
        };
        let slabs = geo.slabs as usize;
        if starts.len() != slabs + 1 || marks.len() != geo.slices as usize * slabs {
            return false;
        }
        let Some(out) = self.data(side, data).map(|b| Buffer(b.0.clone())) else {
            return false;
        };
        let basis = Basis {
            ua: u.0,
            ub: u.1,
            va: v.0,
            vb: v.1,
        };
        let Some(commands) = gpu.queue.commandBuffer() else {
            return false;
        };
        let Some(encoder) = commands.computeCommandEncoder() else {
            return false;
        };
        let size = |n: usize| MTLSize {
            width: n,
            height: 1,
            depth: 1,
        };
        let k = &gpu.kernels;
        // SAFETY: each kernel's buffers and bytes are as `gpu.metal`
        // declares them, and large enough for `geo`.
        unsafe {
            let bytes = |value: *const c_void, len: usize, index: usize| {
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(value.cast_mut()),
                    len,
                    index,
                );
            };
            let geo_at = |index| bytes((&raw const geo).cast(), size_of::<Geo>(), index);
            let buffers = |list: &[&Buffer]| {
                for (i, b) in list.iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(&b.0), 0, i);
                }
            };
            // Transform.
            encoder.setComputePipelineState(&k[0]);
            buffers(&[&plan.p, &plan.r, &self.steps]);
            bytes((&raw const basis).cast(), size_of::<Basis>(), 3);
            geo_at(4);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                size((geo.n as usize).div_ceil(THREADS)),
                size(THREADS),
            );
            // Count.
            encoder.setComputePipelineState(&k[1]);
            buffers(&[&self.steps, &plan.groups, &self.counts]);
            geo_at(3);
            encoder.setThreadgroupMemoryLength_atIndex(slabs * 4, 0);
            encoder
                .dispatchThreadgroups_threadsPerThreadgroup(size(geo.groups as usize), size(GROUP));
            // Scan.
            encoder.setComputePipelineState(&k[2]);
            buffers(&[&self.counts, &self.totals]);
            geo_at(2);
            encoder.setThreadgroupMemoryLength_atIndex(THREADS * 4, 0);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(slabs), size(THREADS));
            // Starts.
            encoder.setComputePipelineState(&k[3]);
            buffers(&[&self.totals, &self.starts]);
            geo_at(2);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(1), size(1));
            // Marks.
            encoder.setComputePipelineState(&k[4]);
            buffers(&[&self.counts, &plan.first, &self.marks]);
            geo_at(3);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                size((geo.slices as usize * slabs).div_ceil(THREADS)),
                size(THREADS),
            );
            // Place.
            encoder.setComputePipelineState(&k[5]);
            buffers(&[
                &self.steps,
                &plan.groups,
                &self.counts,
                &self.starts,
                &plan.tag,
                &out,
            ]);
            geo_at(6);
            encoder.setThreadgroupMemoryLength_atIndex(slabs * 4, 0);
            encoder.setThreadgroupMemoryLength_atIndex(slabs * 4, 1);
            encoder
                .dispatchThreadgroups_threadsPerThreadgroup(size(geo.groups as usize), size(GROUP));
        }
        encoder.endEncoding();
        commands.commit();
        commands.waitUntilCompleted();
        if commands.status() != MTLCommandBufferStatus::Completed {
            return false;
        }
        starts.copy_from_slice(self.starts.words(slabs + 1));
        marks.copy_from_slice(self.marks.words(marks.len()));
        starts[slabs] as usize <= data.len()
    }
}
