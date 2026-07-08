// foldit:allow-long-file — consolidated GPU module: all xtal #[cube] kernels
// (splat/gather/mixed-radix FFT/deblur/extract/scatter) + their resident host
// wrappers live together; splitting fragments the device pipeline.
//! GPU real-space density splat via CubeCL on the wgpu backend.
//!
//! Ports the ASU-only Gaussian splat (`density::splat_density`) to a
//! per-atom compute kernel. Each thread walks one atom's bounding box,
//! evaluates the five-Gaussian kernel at every covered voxel, and accumulates
//! the value into a shared grid.
//!
//! WebGPU has no floating-point atomics, so contributions are quantized to
//! unsigned fixed-point (`round(value * SCALE)`) and summed with `u32` atomic
//! adds; the grid is rescaled by `1 / SCALE` on readback. The bounding box,
//! squared-distance, periodic-wrap, and row-major flat-index arithmetic mirror
//! `density::for_each_covered_voxel` exactly; only the per-voxel kernel value
//! is computed from the exact five-Gaussian sum rather than the CPU radial LUT.
//!
//! The adjoint gather ([`gpu_gather_gradient`]) walks the same per-copy boxes
//! but reads a real-space gradient map and writes one partial per atom copy.
//! It needs no atomics: each copy owns its output slot, so the reduction over
//! a copy's voxels stays in a thread-local register.

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

use super::density::{
    build_orth_n, cutoff_radius, estimate_grid_radius, kernel_coeffs,
    SplatParams,
};
use super::types::{has_small_factorization, DensityGrid};

/// Fixed-point scale for the atomic accumulator: `2^26`. Contributions are
/// non-negative, so an unsigned quantization `round(value * SCALE)` inverts to
/// `u32 / SCALE` on readback with ~`1/SCALE` resolution.
const SCALE: f32 = 67_108_864.0;

/// Threads per work group along the atom axis.
const WORKGROUP: u32 = 64;

/// Backend cap on cube groups per dispatch dimension (Metal/Vulkan: 65 535).
const MAX_CUBE_DIM: u32 = 65_535;

/// Lay `groups` 1-D cubes out as a cube count that respects [`MAX_CUBE_DIM`].
///
/// Large cells (e.g. a ~47 Å short axis sampled onto a power-of-two grid) push
/// a per-window or per-voxel dispatch past the 65 535 groups-per-dimension cap,
/// so a flat `(groups, 1, 1)` launch is a validation error. Every kernel here
/// indexes off `ABSOLUTE_POS` / `CUBE_POS`, both of which flatten across cube
/// dimensions, so folding the excess into a second dimension leaves indexing
/// untouched.
///
/// The returned `Static(a, b, 1)` count is an *exact* cover: `a * b == groups`
/// with both `a` and `b` `<= MAX_CUBE_DIM`, so no cube lands past `groups` and
/// no dispatched thread indexes a window beyond its bound. Panics if `groups`
/// has no such in-range factorization; this is unreachable for the 5-smooth
/// grid-window counts callers dispatch (each grid axis `<= MAX_FFT_LEN`), and a
/// clear panic is preferable to overshooting into unguarded window reads.
// The panic is the intended safe terminus for an unfactorable count, so the
// restriction lint is suppressed here.
#[allow(clippy::panic)]
fn cube_count_1d(groups: u32) -> CubeCount {
    if groups <= MAX_CUBE_DIM {
        return CubeCount::Static(groups, 1, 1);
    }
    // Search downward from floor(sqrt(groups)) for the first exact divisor
    // whose cofactor also fits the cap, giving a reasonably balanced 2-D
    // split.
    let mut a = groups.isqrt().min(MAX_CUBE_DIM);
    while a >= 1 {
        if groups.is_multiple_of(a) && groups / a <= MAX_CUBE_DIM {
            return CubeCount::Static(a, groups / a, 1);
        }
        a -= 1;
    }
    panic!(
        "cube_count_1d: {groups} groups has no factorization a*b with both \
         dims <= MAX_CUBE_DIM ({MAX_CUBE_DIM})"
    );
}

/// Largest window the shared-memory FFT accepts. One thread per sample caps the
/// window at the per-cube thread limit, and the four `f32` ping-pong scratch
/// arrays cost `4 * MAX_FFT_LEN * 4` bytes of shared memory; 1024 stays inside
/// both the Metal threadgroup and shared-memory budgets.
const MAX_FFT_LEN: usize = 1024;

/// Transform direction for [`gpu_cfft_3d`]. `Inverse` conjugates the twiddles
/// and scales by `1/n` per axis, so a `Forward` then `Inverse` pass recovers
/// the input (the combined `1/(nu*nv*nw)` matches the CPU convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftMode {
    /// Unnormalized forward transform, `exp(-2*pi*i*k*n/N)`.
    Forward,
    /// Inverse transform with `1/n`-per-axis normalization.
    Inverse,
}

/// One thread per atom: walk the atom's bounding box, evaluate the exact
/// five-Gaussian kernel at each covered voxel, quantize, and atomic-add into
/// the periodic grid.
///
/// All per-atom inputs are flat arrays indexed by atom: `u0`, `frac_offset`,
/// and `radius` hold three entries each (grid axes u, v, w);
/// `a_coeffs`/`b_coeffs` hold five each. `orth_n` is the shared 3x3
/// grid-step-to-Cartesian matrix in row-major order.
#[cube(launch)]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::excessive_nesting,
    clippy::cast_sign_loss,
    clippy::needless_pass_by_ref_mut
)]
fn splat_kernel(
    grid: &mut Array<Atomic<u32>>,
    u0: &Array<i32>,
    frac_offset: &Array<f32>,
    radius: &Array<i32>,
    a_coeffs: &Array<f32>,
    b_coeffs: &Array<f32>,
    occupancy: &Array<f32>,
    cutoff_sq: &Array<f32>,
    orth_n: &Array<f32>,
    n_atoms: u32,
    nu: i32,
    nv: i32,
    nw: i32,
) {
    let atom = ABSOLUTE_POS;
    if atom < n_atoms as usize {
        let base3 = atom * 3;
        let u0u = u0[base3];
        let u0v = u0[base3 + 1];
        let u0w = u0[base3 + 2];
        let off_u = frac_offset[base3];
        let off_v = frac_offset[base3 + 1];
        let off_w = frac_offset[base3 + 2];
        let r_u = radius[base3];
        let r_v = radius[base3 + 1];
        let r_w = radius[base3 + 2];

        let o0 = orth_n[0];
        let o1 = orth_n[1];
        let o2 = orth_n[2];
        let o3 = orth_n[3];
        let o4 = orth_n[4];
        let o5 = orth_n[5];
        let o6 = orth_n[6];
        let o7 = orth_n[7];
        let o8 = orth_n[8];

        let csq = cutoff_sq[atom];
        let occ = occupancy[atom];
        let base5 = atom * 5;
        let a0 = a_coeffs[base5];
        let a1 = a_coeffs[base5 + 1];
        let a2 = a_coeffs[base5 + 2];
        let a3 = a_coeffs[base5 + 3];
        let a4 = a_coeffs[base5 + 4];
        let b0 = b_coeffs[base5];
        let b1 = b_coeffs[base5 + 1];
        let b2 = b_coeffs[base5 + 2];
        let b3 = b_coeffs[base5 + 3];
        let b4 = b_coeffs[base5 + 4];

        let span_u = 2 * r_u + 1;
        let span_v = 2 * r_v + 1;
        let span_w = 2 * r_w + 1;

        for iu in 0..span_u {
            let du = iu - r_u;
            for iv in 0..span_v {
                let dv = iv - r_v;
                for iw in 0..span_w {
                    let dw = iw - r_w;
                    let dx = f32::cast_from(du) - off_u;
                    let dy = f32::cast_from(dv) - off_v;
                    let dz = f32::cast_from(dw) - off_w;
                    let px = o0 * dx + o1 * dy + o2 * dz;
                    let py = o3 * dx + o4 * dy + o5 * dz;
                    let pz = o6 * dx + o7 * dy + o8 * dz;
                    let r_sq = px * px + py * py + pz * pz;
                    if r_sq <= csq {
                        let mut val = a0 * (b0 * r_sq).exp();
                        val += a1 * (b1 * r_sq).exp();
                        val += a2 * (b2 * r_sq).exp();
                        val += a3 * (b3 * r_sq).exp();
                        val += a4 * (b4 * r_sq).exp();
                        val *= occ;
                        let q = u32::cast_from((val * SCALE + 0.5).floor());
                        let gu = (((u0u + du) % nu) + nu) % nu;
                        let gv = (((u0v + dv) % nv) + nv) % nv;
                        let gw = (((u0w + dw) % nw) + nw) % nw;
                        let idx = (gu * nv + gv) * nw + gw;
                        grid[idx as usize].fetch_add(q);
                    }
                }
            }
        }
    }
}

/// One thread per atom copy: walk the copy's bounding box, evaluate the exact
/// five-Gaussian kernel at each covered voxel, and accumulate
/// `g_map[voxel] * kernel(voxel) * occupancy` into a thread-local register.
/// Each copy writes its scalar partial to `partials[copy]`; the host reduces
/// partials per atom over the symmetry orbit.
///
/// The input layout mirrors [`splat_kernel`]: per-copy flat arrays index by
/// copy (`u0`, `frac_offset`, `radius` three each; `a_coeffs`/`b_coeffs` five
/// each), and `orth_n` is the shared 3x3 grid-step-to-Cartesian matrix.
#[cube(launch)]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::excessive_nesting,
    clippy::cast_sign_loss,
    clippy::needless_pass_by_ref_mut
)]
fn gather_kernel(
    g_map: &Array<f32>,
    partials: &mut Array<f32>,
    u0: &Array<i32>,
    frac_offset: &Array<f32>,
    radius: &Array<i32>,
    a_coeffs: &Array<f32>,
    b_coeffs: &Array<f32>,
    occupancy: &Array<f32>,
    cutoff_sq: &Array<f32>,
    orth_n: &Array<f32>,
    n_copies: u32,
    nu: i32,
    nv: i32,
    nw: i32,
) {
    let copy = ABSOLUTE_POS;
    if copy < n_copies as usize {
        let base3 = copy * 3;
        let u0u = u0[base3];
        let u0v = u0[base3 + 1];
        let u0w = u0[base3 + 2];
        let off_u = frac_offset[base3];
        let off_v = frac_offset[base3 + 1];
        let off_w = frac_offset[base3 + 2];
        let r_u = radius[base3];
        let r_v = radius[base3 + 1];
        let r_w = radius[base3 + 2];

        let o0 = orth_n[0];
        let o1 = orth_n[1];
        let o2 = orth_n[2];
        let o3 = orth_n[3];
        let o4 = orth_n[4];
        let o5 = orth_n[5];
        let o6 = orth_n[6];
        let o7 = orth_n[7];
        let o8 = orth_n[8];

        let csq = cutoff_sq[copy];
        let occ = occupancy[copy];
        let base5 = copy * 5;
        let a0 = a_coeffs[base5];
        let a1 = a_coeffs[base5 + 1];
        let a2 = a_coeffs[base5 + 2];
        let a3 = a_coeffs[base5 + 3];
        let a4 = a_coeffs[base5 + 4];
        let b0 = b_coeffs[base5];
        let b1 = b_coeffs[base5 + 1];
        let b2 = b_coeffs[base5 + 2];
        let b3 = b_coeffs[base5 + 3];
        let b4 = b_coeffs[base5 + 4];

        let span_u = 2 * r_u + 1;
        let span_v = 2 * r_v + 1;
        let span_w = 2 * r_w + 1;

        let mut acc = 0.0f32;
        for iu in 0..span_u {
            let du = iu - r_u;
            for iv in 0..span_v {
                let dv = iv - r_v;
                for iw in 0..span_w {
                    let dw = iw - r_w;
                    let dx = f32::cast_from(du) - off_u;
                    let dy = f32::cast_from(dv) - off_v;
                    let dz = f32::cast_from(dw) - off_w;
                    let px = o0 * dx + o1 * dy + o2 * dz;
                    let py = o3 * dx + o4 * dy + o5 * dz;
                    let pz = o6 * dx + o7 * dy + o8 * dz;
                    let r_sq = px * px + py * py + pz * pz;
                    if r_sq <= csq {
                        let mut val = a0 * (b0 * r_sq).exp();
                        val += a1 * (b1 * r_sq).exp();
                        val += a2 * (b2 * r_sq).exp();
                        val += a3 * (b3 * r_sq).exp();
                        val += a4 * (b4 * r_sq).exp();
                        val *= occ;
                        let gu = (((u0u + du) % nu) + nu) % nu;
                        let gv = (((u0v + dv) % nv) + nv) % nv;
                        let gw = (((u0w + dw) % nw) + nw) % nw;
                        let idx = (gu * nv + gv) * nw + gw;
                        acc += g_map[idx as usize] * val;
                    }
                }
            }
        }
        partials[copy] = acc;
    }
}

/// Batched mixed-radix (2/3/5) shared-memory 1-D complex FFT along one grid
/// axis, via Stockham autosort.
///
/// Each cube owns one length-`n` window; `UNIT_POS` is the sample within it and
/// `CUBE_POS` selects the window. The window is loaded natural-order from the
/// flat `(u,v,w)` buffer with `base + t*stride`, so the same kernel serves all
/// three axes. `n` factors into the prime radices in `radices` (their product
/// is `n`); each pass reads the current buffer, applies the radix-`r`
/// butterfly, and writes the ping-pong buffer, after which a
/// `sync_cube()`-guarded copy restores the result into the primary buffer so
/// the next pass reads natural order. Stockham needs no bit/digit-reversal
/// permutation.
///
/// For a pass with radix `r` and accumulated span `p`, thread `t < n/r` gathers
/// `r` samples spaced `n/r` apart and scatters `r` outputs spaced `p` apart.
/// The twist `W_{r*p}^{ridx*k}` (`k = t mod p`) and the radix-`r` DFT root
/// `W_r^{q*ridx}` are a single lookup into the host twiddle table
/// `tw[j] = exp(-2*pi*i*j/n)` (full length `n`) at index
/// `ridx*(k*(n/(r*p)) + q*(n/r)) mod n`. `tw_sign` (`-1`) conjugates it for the
/// inverse; `scale` folds the per-axis `1/n` normalization applied on
/// writeback.
#[cube(launch)]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::needless_pass_by_ref_mut
)]
fn fft_axis(
    re: &mut Array<f32>,
    im: &mut Array<f32>,
    radices: &Array<u32>,
    tw_re: &Array<f32>,
    tw_im: &Array<f32>,
    num_stages: u32,
    n: u32,
    stride: u32,
    tw_sign: f32,
    scale: f32,
) {
    let mut ar = SharedMemory::<f32>::new(MAX_FFT_LEN);
    let mut ai = SharedMemory::<f32>::new(MAX_FFT_LEN);
    let mut br = SharedMemory::<f32>::new(MAX_FFT_LEN);
    let mut bi = SharedMemory::<f32>::new(MAX_FFT_LEN);

    let tid = UNIT_POS as usize;
    let n = n as usize;
    let stride = stride as usize;
    let num_stages = num_stages as usize;

    // Origin of this cube's window in the flat buffer: contiguous runs when
    // stride == 1, interleaved blocks otherwise. This is the on-device mirror
    // of `fft_cpu::batch_start`, which is canonical; the two must stay
    // bit-identical or the batched passes address different samples. The
    // `cfft_tests` GPU-vs-CPU comparison guards that agreement.
    let window = CUBE_POS;
    let base = if stride == 1 {
        window * n
    } else {
        let block = stride * n;
        (window / stride) * block + (window % stride)
    };

    let src = base + tid * stride;
    ar[tid] = re[src];
    ai[tid] = im[src];
    sync_cube();

    let mut p = 1usize;
    let mut stage = 0usize;
    while stage < num_stages {
        let r = radices[stage] as usize;
        let nr = n / r;
        let m = n / (r * p);
        if tid < nr {
            let k = tid % p;
            let out_block = (tid / p) * (r * p) + k;
            let twist = k * m;
            let mut q = 0usize;
            while q < r {
                let step = twist + q * nr;
                let mut acc_re = 0.0f32;
                let mut acc_im = 0.0f32;
                let mut ridx = 0usize;
                while ridx < r {
                    let s = tid + ridx * nr;
                    let vr = ar[s];
                    let vi = ai[s];
                    let ang = (ridx * step) % n;
                    let wr = tw_re[ang];
                    let wi = tw_sign * tw_im[ang];
                    acc_re += vr * wr - vi * wi;
                    acc_im += vr * wi + vi * wr;
                    ridx += 1usize;
                }
                let dst = out_block + q * p;
                br[dst] = acc_re;
                bi[dst] = acc_im;
                q += 1usize;
            }
        }
        sync_cube();
        ar[tid] = br[tid];
        ai[tid] = bi[tid];
        sync_cube();
        p *= r;
        stage += 1usize;
    }

    let dst = base + tid * stride;
    re[dst] = ar[tid] * scale;
    im[dst] = ai[tid] * scale;
}

/// One thread per grid point: convert the `u32` fixed-point splat accumulator
/// to `f32` electron density (`src / SCALE`), writing the real channel the FFT
/// consumes. Mirrors the `q as f32 / SCALE` readback rescale in
/// [`gpu_splat_density`], but on device so the density never leaves the GPU.
#[cube(launch)]
fn scale_kernel(
    src: &Array<u32>,
    dst: &mut Array<f32>,
    inv_scale: f32,
    total: u32,
) {
    let gp = ABSOLUTE_POS;
    if gp < total as usize {
        dst[gp] = f32::cast_from(src[gp]) * inv_scale;
    }
}

/// One thread per grid point: in-place reciprocal-space deblur of the complex
/// spectrum. Reconstructs the signed Miller index from the row-major flat
/// position (the `u <= n/2` fold matching `density::deblur_fc`), forms
/// `d*^2` from the reciprocal-cell constants, and scales the complex value by
/// `exp(min(blur * d*^2 / 4, 80))`. The `80` cap matches the CPU clamp that
/// keeps the `f32` exponential from overflowing.
#[cube(launch)]
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn deblur_kernel(
    re: &mut Array<f32>,
    im: &mut Array<f32>,
    ar: f32,
    br: f32,
    cr: f32,
    cos_ar: f32,
    cos_br: f32,
    cos_cr: f32,
    blur: f32,
    nu: i32,
    nv: i32,
    nw: i32,
    total: u32,
) {
    let gp = ABSOLUTE_POS;
    if gp < total as usize {
        let idx = i32::cast_from(gp);
        let w = idx % nw;
        let uv = idx / nw;
        let v = uv % nv;
        let u = uv / nv;

        let h = if u <= nu / 2 { u } else { u - nu };
        let k = if v <= nv / 2 { v } else { v - nv };
        let l = if w <= nw / 2 { w } else { w - nw };

        let hf = f32::cast_from(h);
        let kf = f32::cast_from(k);
        let lf = f32::cast_from(l);

        let ha = hf * ar;
        let kb = kf * br;
        let lc = lf * cr;
        let diag = ha * ha + kb * kb + lc * lc;
        let cross = 2.0 * kf * lf * br * cr * cos_ar
            + 2.0 * hf * lf * ar * cr * cos_br
            + 2.0 * hf * kf * ar * br * cos_cr;
        let d_star_sq = diag + cross;

        let raw = blur * d_star_sq / 4.0;
        let mut exponent = raw;
        if exponent > 80.0 {
            exponent = 80.0;
        }
        let s = exponent.exp();
        re[gp] = re[gp] * s;
        im[gp] = im[gp] * s;
    }
}

/// One thread per working reflection: gather the complex spectrum value at the
/// reflection's precomputed wrapped-Miller flat index, scale by the voxel
/// volume `V/N` (the physical structure-factor scale
/// `extract_reflection_values` applies), and write the `[re, im]` pair to the
/// output vector. This small vector is the only buffer read back to host.
#[cube(launch)]
fn extract_kernel(
    re: &Array<f32>,
    im: &Array<f32>,
    gidx: &Array<u32>,
    out: &mut Array<f32>,
    scale: f32,
    n_refl: u32,
) {
    let r = ABSOLUTE_POS;
    if r < n_refl as usize {
        let gi = gidx[r] as usize;
        out[2 * r] = re[gi] * scale;
        out[2 * r + 1] = im[gi] * scale;
    }
}

/// One thread per working reflection: scatter the gradient coefficient into a
/// zeroed complex grid at the reflection's wrapped Miller bin (`idx`) and its
/// Friedel conjugate at the mirror bin (`idxn`), so the inverse FFT yields a
/// purely real map. Mirrors the host pack in `b_factor_gradients_from_fc`:
/// `dre + i*dim` at `idx` and `dre - i*dim` at `idxn`.
///
/// Absent Miller aliasing every reflection owns distinct forward and mirror
/// bins, so the writes never collide across threads. The one exception is a
/// self-Friedel bin (`idx == idxn`, e.g. `2h ≡ 0 (mod nu)`), where the two
/// contributions land in the same slot; that thread combines them locally into
/// `2*dre + i*0`, matching the host accumulation exactly.
#[cube(launch)]
fn scatter_coeff_kernel(
    re: &mut Array<f32>,
    im: &mut Array<f32>,
    idx: &Array<u32>,
    idxn: &Array<u32>,
    dre: &Array<f32>,
    dim: &Array<f32>,
    n_refl: u32,
) {
    let r = ABSOLUTE_POS;
    if r < n_refl as usize {
        let i = idx[r] as usize;
        let ic = idxn[r] as usize;
        let vr = dre[r];
        let vi = dim[r];
        if i == ic {
            re[i] = 2.0 * vr;
            im[i] = 0.0;
        } else {
            re[i] = vr;
            im[i] = vi;
            re[ic] = vr;
            im[ic] = -vi;
        }
    }
}

/// One thread per unique bin: plain-ASSIGN the map coefficient into a zeroed
/// complex grid at that bin (`bin`), so the inverse FFT yields a real density.
/// The bin list from `build_coefficient_assignments` is collision-free by
/// construction (its host-side last-write dedup already resolved every
/// forward/Friedel overlap, including self-Friedel bins), so each thread owns a
/// distinct slot and the parallel scatter is race-free.
///
/// This ASSIGNS; it does NOT double the self-Friedel bin the way
/// [`scatter_coeff_kernel`] does. The two must stay distinct: the gradient
/// scatter's `2*vr` doubling is correct for the adjoint but wrong for the
/// density map, where each transformed index carries a symmetry-consistent
/// coefficient assigned once.
#[cube(launch)]
fn map_scatter_kernel(
    re: &mut Array<f32>,
    im: &mut Array<f32>,
    bin: &Array<u32>,
    vre: &Array<f32>,
    vim: &Array<f32>,
    n_entry: u32,
) {
    let r = ABSOLUTE_POS;
    if r < n_entry as usize {
        let b = bin[r] as usize;
        re[b] = vre[r];
        im[b] = vim[r];
    }
}

/// Per-atom-copy kernel inputs, flattened for upload. Shared by the forward
/// splat and the adjoint gather: both traverse the identical covered-voxel box,
/// so both derive bounding boxes, offsets, cutoffs, and five-Gaussian
/// coefficients the same way from a [`SplatParams`].
struct FlatInputs {
    u0: Vec<i32>,
    frac_offset: Vec<f32>,
    radius: Vec<i32>,
    a_coeffs: Vec<f32>,
    b_coeffs: Vec<f32>,
    occupancy: Vec<f32>,
    cutoff_sq: Vec<f32>,
    orth_flat: Vec<f32>,
}

/// Build the flattened per-copy kernel inputs from `params`. One entry set per
/// `params.positions[i]`; the coefficients come from [`kernel_coeffs`] and the
/// geometry from [`build_orth_n`]/[`estimate_grid_radius`], matching the CPU
/// traversal exactly.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
fn flatten_inputs(params: &SplatParams<'_>, dims: [usize; 3]) -> FlatInputs {
    let [nu, nv, nw] = dims;
    let n = params.positions.len();
    let dims_f = [nu as f64, nv as f64, nw as f64];
    let orth_n = build_orth_n(params.unit_cell, [nu, nv, nw]);

    let mut u0 = Vec::with_capacity(n * 3);
    let mut frac_offset = Vec::with_capacity(n * 3);
    let mut radius = Vec::with_capacity(n * 3);
    let mut a_coeffs = Vec::with_capacity(n * 5);
    let mut b_coeffs = Vec::with_capacity(n * 5);
    let mut occupancy = Vec::with_capacity(n);
    let mut cutoff_sq = Vec::with_capacity(n);

    for atom in 0..n {
        let ff = params.form_factors[atom];
        let b_atom = params.b_factors[atom];
        let frac = params.positions[atom];

        let (a, b) = kernel_coeffs(ff, b_atom, params.blur);
        for i in 0..5 {
            a_coeffs.push(a[i] as f32);
            b_coeffs.push(b[i] as f32);
        }

        let cutoff = cutoff_radius(b_atom + params.blur);
        cutoff_sq.push((cutoff * cutoff) as f32);
        occupancy.push(params.occupancies[atom] as f32);

        for axis in 0..3 {
            let pos_grid = frac[axis] * dims_f[axis];
            let nearest = pos_grid.round();
            u0.push(nearest as i32);
            frac_offset.push((pos_grid - nearest) as f32);
            radius.push(estimate_grid_radius(&orth_n, cutoff, axis));
        }
    }

    let orth_flat = orth_n
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as f32))
        .collect();

    FlatInputs {
        u0,
        frac_offset,
        radius,
        a_coeffs,
        b_coeffs,
        occupancy,
        cutoff_sq,
        orth_flat,
    }
}

/// Splat atomic electron density onto a grid on the GPU.
///
/// Builds per-atom inputs from `params` (five-Gaussian coefficients from
/// [`kernel_coeffs`], bounding boxes, nearest-grid-point offsets, all downcast
/// to `f32`), launches [`splat_kernel`] over the atoms on the default wgpu
/// device, and rescales the `u32` fixed-point accumulator to an `f32`
/// [`DensityGrid`]. ASU-only: no symmetry expansion.
///
/// # Panics
///
/// Panics if the wgpu device cannot be initialized or a grid axis is zero.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::too_many_lines
)]
pub fn gpu_splat_density(
    device: &WgpuDevice,
    params: &SplatParams<'_>,
    dims: [usize; 3],
) -> DensityGrid {
    let [nu, nv, nw] = dims;
    let total = nu * nv * nw;
    let n_atoms = params.positions.len();

    let mut grid = DensityGrid {
        data: vec![0.0; total],
        nu,
        nv,
        nw,
    };
    if n_atoms == 0 || total == 0 {
        return grid;
    }

    let inputs = flatten_inputs(params, dims);

    let client = WgpuRuntime::client(device);

    let g = client.create_from_slice(u32::as_bytes(&vec![0_u32; total]));
    let h_u0 = client.create_from_slice(i32::as_bytes(&inputs.u0));
    let h_off = client.create_from_slice(f32::as_bytes(&inputs.frac_offset));
    let h_rad = client.create_from_slice(i32::as_bytes(&inputs.radius));
    let h_a = client.create_from_slice(f32::as_bytes(&inputs.a_coeffs));
    let h_b = client.create_from_slice(f32::as_bytes(&inputs.b_coeffs));
    let h_occ = client.create_from_slice(f32::as_bytes(&inputs.occupancy));
    let h_csq = client.create_from_slice(f32::as_bytes(&inputs.cutoff_sq));
    let h_orth = client.create_from_slice(f32::as_bytes(&inputs.orth_flat));

    let n_atoms_u32 = n_atoms as u32;
    let blocks = n_atoms_u32.div_ceil(WORKGROUP);

    splat_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count_1d(blocks),
        CubeDim::new_1d(WORKGROUP),
        unsafe { ArrayArg::from_raw_parts(g.clone(), total) },
        unsafe { ArrayArg::from_raw_parts(h_u0, n_atoms * 3) },
        unsafe { ArrayArg::from_raw_parts(h_off, n_atoms * 3) },
        unsafe { ArrayArg::from_raw_parts(h_rad, n_atoms * 3) },
        unsafe { ArrayArg::from_raw_parts(h_a, n_atoms * 5) },
        unsafe { ArrayArg::from_raw_parts(h_b, n_atoms * 5) },
        unsafe { ArrayArg::from_raw_parts(h_occ, n_atoms) },
        unsafe { ArrayArg::from_raw_parts(h_csq, n_atoms) },
        unsafe { ArrayArg::from_raw_parts(h_orth, 9) },
        n_atoms_u32,
        nu as i32,
        nv as i32,
        nw as i32,
    );

    let bytes = client.read_one_unchecked(g);
    let raw = u32::from_bytes(&bytes);
    for (dst, &q) in grid.data.iter_mut().zip(raw.iter()) {
        *dst = q as f32 / SCALE;
    }

    grid
}

/// Gather the real-space B-factor gradient on the GPU (adjoint of the splat).
///
/// `params` carries the orbit-expanded atom copies: `params.positions[i]` is
/// the fractional position of one symmetry/centering copy, and the remaining
/// per-copy fields (form factor, B-factor, occupancy) are replicated across an
/// atom's copies. Copies must be laid out atom-major (all of atom 0's copies,
/// then atom 1's, ...) with a constant orbit size, so `n_copies / n_atoms`
/// copies reduce into each atom's gradient.
///
/// Uploads `g_map` and the per-copy inputs, launches [`gather_kernel`] over the
/// copies, downloads one partial per copy, sums each atom's copies, and applies
/// `scale` exactly as the CPU path (`grad[j] = scale * Σ_copies gather`).
///
/// # Panics
///
/// Panics if the wgpu device cannot be initialized or `n_atoms` does not evenly
/// divide `params.positions.len()`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::too_many_arguments
)]
pub fn gpu_gather_gradient(
    device: &WgpuDevice,
    g_map: &[f32],
    params: &SplatParams<'_>,
    dims: [usize; 3],
    n_atoms: usize,
    scale: f64,
) -> Vec<f64> {
    let [nu, nv, nw] = dims;
    let total = nu * nv * nw;
    let n_copies = params.positions.len();

    let mut grad = vec![0.0_f64; n_atoms];
    if n_copies == 0 || total == 0 || n_atoms == 0 {
        return grad;
    }
    assert_eq!(
        n_copies % n_atoms,
        0,
        "orbit copies must be a whole multiple of the atom count"
    );
    let orbit = n_copies / n_atoms;

    let inputs = flatten_inputs(params, dims);

    let client = WgpuRuntime::client(device);

    let h_gmap = client.create_from_slice(f32::as_bytes(g_map));
    let part =
        client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_copies]));
    let h_u0 = client.create_from_slice(i32::as_bytes(&inputs.u0));
    let h_off = client.create_from_slice(f32::as_bytes(&inputs.frac_offset));
    let h_rad = client.create_from_slice(i32::as_bytes(&inputs.radius));
    let h_a = client.create_from_slice(f32::as_bytes(&inputs.a_coeffs));
    let h_b = client.create_from_slice(f32::as_bytes(&inputs.b_coeffs));
    let h_occ = client.create_from_slice(f32::as_bytes(&inputs.occupancy));
    let h_csq = client.create_from_slice(f32::as_bytes(&inputs.cutoff_sq));
    let h_orth = client.create_from_slice(f32::as_bytes(&inputs.orth_flat));

    let n_copies_u32 = n_copies as u32;
    let blocks = n_copies_u32.div_ceil(WORKGROUP);

    gather_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count_1d(blocks),
        CubeDim::new_1d(WORKGROUP),
        unsafe { ArrayArg::from_raw_parts(h_gmap, total) },
        unsafe { ArrayArg::from_raw_parts(part.clone(), n_copies) },
        unsafe { ArrayArg::from_raw_parts(h_u0, n_copies * 3) },
        unsafe { ArrayArg::from_raw_parts(h_off, n_copies * 3) },
        unsafe { ArrayArg::from_raw_parts(h_rad, n_copies * 3) },
        unsafe { ArrayArg::from_raw_parts(h_a, n_copies * 5) },
        unsafe { ArrayArg::from_raw_parts(h_b, n_copies * 5) },
        unsafe { ArrayArg::from_raw_parts(h_occ, n_copies) },
        unsafe { ArrayArg::from_raw_parts(h_csq, n_copies) },
        unsafe { ArrayArg::from_raw_parts(h_orth, 9) },
        n_copies_u32,
        nu as i32,
        nv as i32,
        nw as i32,
    );

    let bytes = client.read_one_unchecked(part);
    let partials = f32::from_bytes(&bytes);

    for (atom, g) in grad.iter_mut().enumerate() {
        let mut sum = 0.0_f64;
        for c in 0..orbit {
            sum += f64::from(partials[atom * orbit + c]);
        }
        *g = sum * scale;
    }

    grad
}

/// Factor a 5-smooth length into its ordered radix stages (all 2s, then 3s,
/// then 5s). Any factor order yields a correct mixed-radix transform; the
/// product of the returned radices equals `n` for any 5-smooth `n`.
#[allow(clippy::cast_possible_truncation)]
fn factor_235(mut n: usize) -> Vec<u32> {
    let mut radices = Vec::new();
    for f in [2usize, 3, 5] {
        while n.is_multiple_of(f) {
            radices.push(f as u32);
            n /= f;
        }
    }
    radices
}

/// Full complex 3-D FFT on the GPU: a batched mixed-radix (2/3/5) shared-memory
/// 1-D pass along each axis of the row-major `(u,v,w)` grid (`nw` fastest).
///
/// `re`/`im` are the complex input laid out `((u*nv)+v)*nw + w`. Every axis
/// length must be 5-smooth (only 2/3/5 prime factors) and `<= MAX_FFT_LEN`.
/// `FftMode::Forward` is the unnormalized transform matching
/// [`fft_cpu::fft_3d_forward`]; `Inverse` applies the `1/(nu*nv*nw)`
/// normalization (`1/n` folded into each axis) so `Inverse(Forward(x)) == x`.
///
/// [`fft_cpu::fft_3d_forward`]: super::fft_cpu::fft_3d_forward
///
/// # Panics
///
/// Panics if the input lengths do not equal `nu*nv*nw`, an axis is not 5-smooth
/// `<= MAX_FFT_LEN`, or the wgpu device cannot be initialized.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc,
    clippy::too_many_lines
)]
pub fn gpu_cfft_3d(
    device: &WgpuDevice,
    re: &[f32],
    im: &[f32],
    dims: [usize; 3],
    mode: FftMode,
) -> (Vec<f32>, Vec<f32>) {
    let [nu, nv, nw] = dims;
    let total = nu * nv * nw;
    assert_eq!(re.len(), total, "re length must equal nu*nv*nw");
    assert_eq!(im.len(), total, "im length must equal nu*nv*nw");
    assert!(
        gpu_cfft_grid_supported(dims),
        "each axis must be 5-smooth (2/3/5 factors) <= {MAX_FFT_LEN}, got \
         {dims:?}"
    );
    if total == 0 {
        return (Vec::new(), Vec::new());
    }

    let client = WgpuRuntime::client(device);

    let re_buf = client.create_from_slice(f32::as_bytes(re));
    let im_buf = client.create_from_slice(f32::as_bytes(im));

    fft_3d_axes_inplace(&client, &re_buf, &im_buf, dims, mode);

    let re_out = f32::from_bytes(&client.read_one_unchecked(re_buf)).to_vec();
    let im_out = f32::from_bytes(&client.read_one_unchecked(im_buf)).to_vec();
    (re_out, im_out)
}

/// Run the three batched mixed-radix axis passes in place on the device `re`/
/// `im` handles (w fastest, then v, then u), transforming them from natural
/// order to natural order. `mode` selects the unnormalized forward transform
/// or the `1/n`-per-axis inverse. The handles are mutated in place; the caller
/// owns their allocation and any readback.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn fft_3d_axes_inplace(
    client: &ComputeClient<WgpuRuntime>,
    re_buf: &Handle,
    im_buf: &Handle,
    dims: [usize; 3],
    mode: FftMode,
) {
    let [nu, nv, nw] = dims;
    let total = nu * nv * nw;

    let tw_sign = match mode {
        FftMode::Forward => 1.0_f32,
        FftMode::Inverse => -1.0_f32,
    };

    // (axis length, stride, window count) for w (fastest), then v, then u.
    let axes = [
        (nw, 1usize, nu * nv),
        (nv, nw, nu * nw),
        (nu, nv * nw, nv * nw),
    ];

    for &(n, stride, num_windows) in &axes {
        let radices = factor_235(n);
        let tw_re: Vec<f32> = (0..n)
            .map(|k| (-2.0 * std::f32::consts::PI * k as f32 / n as f32).cos())
            .collect();
        let tw_im: Vec<f32> = (0..n)
            .map(|k| (-2.0 * std::f32::consts::PI * k as f32 / n as f32).sin())
            .collect();
        let scale = match mode {
            FftMode::Forward => 1.0_f32,
            FftMode::Inverse => 1.0_f32 / n as f32,
        };

        let h_rad = client.create_from_slice(u32::as_bytes(&radices));
        let h_twr = client.create_from_slice(f32::as_bytes(&tw_re));
        let h_twi = client.create_from_slice(f32::as_bytes(&tw_im));

        fft_axis::launch::<WgpuRuntime>(
            client,
            cube_count_1d(num_windows as u32),
            CubeDim::new_1d(n as u32),
            unsafe { ArrayArg::from_raw_parts(re_buf.clone(), total) },
            unsafe { ArrayArg::from_raw_parts(im_buf.clone(), total) },
            unsafe { ArrayArg::from_raw_parts(h_rad, radices.len()) },
            unsafe { ArrayArg::from_raw_parts(h_twr, n) },
            unsafe { ArrayArg::from_raw_parts(h_twi, n) },
            radices.len() as u32,
            n as u32,
            stride as u32,
            tw_sign,
            scale,
        );
    }
}

/// Gather one complex value per working reflection off a device-resident
/// complex spectrum and read it back: launch [`extract_kernel`] over `refl_idx`
/// (scaling by the voxel volume `V/N`) and reshape the flat readback into
/// `[re, im]` pairs. This per-reflection vector is the only buffer that leaves
/// the device. Consumes the spectrum handles `re_buf`/`im_buf` (their last
/// use).
#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
fn extract_reflections_resident(
    client: &ComputeClient<WgpuRuntime>,
    re_buf: Handle,
    im_buf: Handle,
    refl_idx: &[u32],
    voxel_volume: f32,
    total: usize,
) -> Vec<[f32; 2]> {
    let n_refl = refl_idx.len();
    let idx_buf = client.create_from_slice(u32::as_bytes(refl_idx));
    let out = client.empty(n_refl * 2 * 4);
    let refl_blocks = (n_refl as u32).div_ceil(WORKGROUP);

    extract_kernel::launch::<WgpuRuntime>(
        client,
        cube_count_1d(refl_blocks),
        CubeDim::new_1d(WORKGROUP),
        unsafe { ArrayArg::from_raw_parts(re_buf, total) },
        unsafe { ArrayArg::from_raw_parts(im_buf, total) },
        unsafe { ArrayArg::from_raw_parts(idx_buf, n_refl) },
        unsafe { ArrayArg::from_raw_parts(out.clone(), n_refl * 2) },
        voxel_volume,
        n_refl as u32,
    );

    let bytes = client.read_one_unchecked(out);
    let flat = f32::from_bytes(&bytes);
    flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
}

/// Device-resident forward structure factors: splat -> mixed-radix FFT ->
/// deblur -> extract, with the density grid and complex spectrum kept on the
/// GPU throughout. Only the per-reflection Fc vector is read back.
///
/// `refl_idx[r]` is the flat grid index of reflection `r`'s wrapped-Miller bin
/// (the caller applies the `wrap_miller_index` convention). `recip` packs the
/// reciprocal-cell constants `[a*, b*, c*, cos α*, cos β*, cos γ*]` the deblur
/// kernel forms `d*^2` from; `voxel_volume` is `V/N`, the physical
/// structure-factor scale. Returns `refl_idx.len()` complex `[re, im]` pairs.
///
/// # Panics
///
/// Panics if the wgpu device cannot be initialized.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]
pub fn gpu_forward_fc_resident(
    device: &WgpuDevice,
    params: &SplatParams<'_>,
    dims: [usize; 3],
    blur: f64,
    recip: [f32; 6],
    refl_idx: &[u32],
    voxel_volume: f64,
) -> Vec<[f32; 2]> {
    let [nu, nv, nw] = dims;
    let total = nu * nv * nw;
    let n_atoms = params.positions.len();
    let n_refl = refl_idx.len();
    if n_refl == 0 {
        return Vec::new();
    }
    if n_atoms == 0 || total == 0 {
        return vec![[0.0, 0.0]; n_refl];
    }

    let inputs = flatten_inputs(params, dims);

    let client = WgpuRuntime::client(device);

    // Splat atomic density into the u32 fixed-point accumulator on device.
    let g = client.create_from_slice(u32::as_bytes(&vec![0_u32; total]));
    let h_u0 = client.create_from_slice(i32::as_bytes(&inputs.u0));
    let h_off = client.create_from_slice(f32::as_bytes(&inputs.frac_offset));
    let h_rad = client.create_from_slice(i32::as_bytes(&inputs.radius));
    let h_a = client.create_from_slice(f32::as_bytes(&inputs.a_coeffs));
    let h_b = client.create_from_slice(f32::as_bytes(&inputs.b_coeffs));
    let h_occ = client.create_from_slice(f32::as_bytes(&inputs.occupancy));
    let h_csq = client.create_from_slice(f32::as_bytes(&inputs.cutoff_sq));
    let h_orth = client.create_from_slice(f32::as_bytes(&inputs.orth_flat));

    let n_atoms_u32 = n_atoms as u32;
    let atom_blocks = n_atoms_u32.div_ceil(WORKGROUP);

    splat_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count_1d(atom_blocks),
        CubeDim::new_1d(WORKGROUP),
        unsafe { ArrayArg::from_raw_parts(g.clone(), total) },
        unsafe { ArrayArg::from_raw_parts(h_u0, n_atoms * 3) },
        unsafe { ArrayArg::from_raw_parts(h_off, n_atoms * 3) },
        unsafe { ArrayArg::from_raw_parts(h_rad, n_atoms * 3) },
        unsafe { ArrayArg::from_raw_parts(h_a, n_atoms * 5) },
        unsafe { ArrayArg::from_raw_parts(h_b, n_atoms * 5) },
        unsafe { ArrayArg::from_raw_parts(h_occ, n_atoms) },
        unsafe { ArrayArg::from_raw_parts(h_csq, n_atoms) },
        unsafe { ArrayArg::from_raw_parts(h_orth, 9) },
        n_atoms_u32,
        nu as i32,
        nv as i32,
        nw as i32,
    );

    // Rescale to f32 (real channel) and pair with a zero imaginary channel;
    // both stay on device across the FFT and deblur.
    let re_buf = client.empty(total * 4);
    let im_buf = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; total]));
    let grid_blocks = (total as u32).div_ceil(WORKGROUP);

    scale_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count_1d(grid_blocks),
        CubeDim::new_1d(WORKGROUP),
        unsafe { ArrayArg::from_raw_parts(g, total) },
        unsafe { ArrayArg::from_raw_parts(re_buf.clone(), total) },
        1.0_f32 / SCALE,
        total as u32,
    );

    fft_3d_axes_inplace(&client, &re_buf, &im_buf, dims, FftMode::Forward);

    deblur_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count_1d(grid_blocks),
        CubeDim::new_1d(WORKGROUP),
        unsafe { ArrayArg::from_raw_parts(re_buf.clone(), total) },
        unsafe { ArrayArg::from_raw_parts(im_buf.clone(), total) },
        recip[0],
        recip[1],
        recip[2],
        recip[3],
        recip[4],
        recip[5],
        blur as f32,
        nu as i32,
        nv as i32,
        nw as i32,
        total as u32,
    );

    // Gather one complex value per reflection; this vector is the only
    // readback.
    extract_reflections_resident(
        &client,
        re_buf,
        im_buf,
        refl_idx,
        voxel_volume as f32,
        total,
    )
}

/// Device-resident forward transform of a prebuilt real-space grid: upload the
/// grid once, pair it with a zeroed imaginary channel, mixed-radix FFT, and
/// extract one complex value per reflection, keeping the spectrum on the GPU
/// and reading back only the per-reflection vector.
///
/// This is forward-FFT + extract only: no splat and no deblur. The bulk-solvent
/// mask is a real grid built on host (not a Gaussian density), so it must not
/// be deblurred; keep this distinct from the splat/deblur chain in
/// [`gpu_forward_fc_resident`].
///
/// `refl_idx[r]` is the flat grid index of reflection `r`'s wrapped-Miller bin
/// (same construction the fc-resident wrapper uses); `voxel_volume` is `V/N`,
/// the physical structure-factor scale `extract_kernel` applies. Returns
/// `refl_idx.len()` complex `[re, im]` pairs.
///
/// # Panics
///
/// Panics if the wgpu device cannot be initialized.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::missing_panics_doc)]
pub(crate) fn gpu_forward_grid_extract_resident(
    device: &WgpuDevice,
    real_grid: &[f32],
    dims: [usize; 3],
    refl_idx: &[u32],
    voxel_volume: f32,
) -> Vec<[f32; 2]> {
    let [nu, nv, nw] = dims;
    let total = nu * nv * nw;
    let n_refl = refl_idx.len();
    if n_refl == 0 {
        return Vec::new();
    }
    if total == 0 {
        return vec![[0.0, 0.0]; n_refl];
    }
    assert_eq!(
        real_grid.len(),
        total,
        "real_grid length must equal nu*nv*nw"
    );

    let client = WgpuRuntime::client(device);

    // Upload the real grid and a zeroed imaginary channel; both stay on device
    // across the FFT and extract.
    let re_buf = client.create_from_slice(f32::as_bytes(real_grid));
    let im_buf = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; total]));

    fft_3d_axes_inplace(&client, &re_buf, &im_buf, dims, FftMode::Forward);

    // Gather one complex value per reflection; this vector is the only
    // readback.
    extract_reflections_resident(
        &client,
        re_buf,
        im_buf,
        refl_idx,
        voxel_volume,
        total,
    )
}

/// Device-resident inverse gradient chain: scatter the per-reflection
/// coefficients into a complex grid, inverse-FFT to a real-space gradient map,
/// and gather each atom's B-factor gradient against its Gaussian kernel, with
/// the coefficient grid and the gradient map kept on the GPU throughout. Only
/// the per-reflection scatter data crosses to the device and the per-copy
/// partials cross back.
///
/// `idx`/`idxn` are the wrapped-Miller flat bin and its Friedel mirror; `dre`/
/// `dim` are the real/imaginary coefficients placed there (the host applies the
/// likelihood-weight and deblur math). `gather_params` carries the
/// orbit-expanded atom copies laid out atom-major with a constant orbit size,
/// as in [`gpu_gather_gradient`]; `scale` is the overall `V/2` factor.
///
/// # Panics
///
/// Panics if the wgpu device cannot be initialized or `n_atoms` does not evenly
/// divide the copy count.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
pub fn gpu_inverse_gradient_resident(
    device: &WgpuDevice,
    idx: &[u32],
    idxn: &[u32],
    dre: &[f32],
    dim: &[f32],
    gather_params: &SplatParams<'_>,
    dims: [usize; 3],
    n_atoms: usize,
    scale: f64,
) -> Vec<f64> {
    let [nu, nv, nw] = dims;
    let total = nu * nv * nw;
    let n_copies = gather_params.positions.len();
    let n_refl = idx.len();

    let mut grad = vec![0.0_f64; n_atoms];
    if n_copies == 0 || total == 0 || n_atoms == 0 {
        return grad;
    }
    assert_eq!(
        n_copies % n_atoms,
        0,
        "orbit copies must be a whole multiple of the atom count"
    );
    let orbit = n_copies / n_atoms;

    let client = WgpuRuntime::client(device);

    // Zeroed complex grid; the scatter writes only the reflection bins, so the
    // rest must stay zero for the inverse transform.
    let re_buf = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; total]));
    let im_buf = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; total]));

    if n_refl > 0 {
        let h_idx = client.create_from_slice(u32::as_bytes(idx));
        let h_idxn = client.create_from_slice(u32::as_bytes(idxn));
        let h_dre = client.create_from_slice(f32::as_bytes(dre));
        let h_dim = client.create_from_slice(f32::as_bytes(dim));
        let refl_blocks = (n_refl as u32).div_ceil(WORKGROUP);

        scatter_coeff_kernel::launch::<WgpuRuntime>(
            &client,
            cube_count_1d(refl_blocks),
            CubeDim::new_1d(WORKGROUP),
            unsafe { ArrayArg::from_raw_parts(re_buf.clone(), total) },
            unsafe { ArrayArg::from_raw_parts(im_buf.clone(), total) },
            unsafe { ArrayArg::from_raw_parts(h_idx, n_refl) },
            unsafe { ArrayArg::from_raw_parts(h_idxn, n_refl) },
            unsafe { ArrayArg::from_raw_parts(h_dre, n_refl) },
            unsafe { ArrayArg::from_raw_parts(h_dim, n_refl) },
            n_refl as u32,
        );
    }

    // Inverse FFT in place; the coefficient grid is Hermitian, so `re_buf` now
    // holds the real-space gradient map and stays on device for the gather.
    fft_3d_axes_inplace(&client, &re_buf, &im_buf, dims, FftMode::Inverse);

    let inputs = flatten_inputs(gather_params, dims);
    let part =
        client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n_copies]));
    let h_u0 = client.create_from_slice(i32::as_bytes(&inputs.u0));
    let h_off = client.create_from_slice(f32::as_bytes(&inputs.frac_offset));
    let h_rad = client.create_from_slice(i32::as_bytes(&inputs.radius));
    let h_a = client.create_from_slice(f32::as_bytes(&inputs.a_coeffs));
    let h_b = client.create_from_slice(f32::as_bytes(&inputs.b_coeffs));
    let h_occ = client.create_from_slice(f32::as_bytes(&inputs.occupancy));
    let h_csq = client.create_from_slice(f32::as_bytes(&inputs.cutoff_sq));
    let h_orth = client.create_from_slice(f32::as_bytes(&inputs.orth_flat));

    let n_copies_u32 = n_copies as u32;
    let copy_blocks = n_copies_u32.div_ceil(WORKGROUP);

    gather_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count_1d(copy_blocks),
        CubeDim::new_1d(WORKGROUP),
        unsafe { ArrayArg::from_raw_parts(re_buf, total) },
        unsafe { ArrayArg::from_raw_parts(part.clone(), n_copies) },
        unsafe { ArrayArg::from_raw_parts(h_u0, n_copies * 3) },
        unsafe { ArrayArg::from_raw_parts(h_off, n_copies * 3) },
        unsafe { ArrayArg::from_raw_parts(h_rad, n_copies * 3) },
        unsafe { ArrayArg::from_raw_parts(h_a, n_copies * 5) },
        unsafe { ArrayArg::from_raw_parts(h_b, n_copies * 5) },
        unsafe { ArrayArg::from_raw_parts(h_occ, n_copies) },
        unsafe { ArrayArg::from_raw_parts(h_csq, n_copies) },
        unsafe { ArrayArg::from_raw_parts(h_orth, 9) },
        n_copies_u32,
        nu as i32,
        nv as i32,
        nw as i32,
    );

    let bytes = client.read_one_unchecked(part);
    let partials = f32::from_bytes(&bytes);

    for (atom, g) in grad.iter_mut().enumerate() {
        let mut sum = 0.0_f64;
        for c in 0..orbit {
            sum += f64::from(partials[atom * orbit + c]);
        }
        *g = sum * scale;
    }

    grad
}

/// Device-resident inverse density synthesis: scatter the per-assignment map
/// coefficients into a zeroed complex grid, inverse-FFT in place, and read back
/// the real channel as the electron density, keeping the coefficient grid on
/// the GPU throughout. Only the small per-entry assignment list crosses to the
/// device and the real density crosses back.
///
/// `bins` are the unique row-major flat grid indices and `re`/`im` their final
/// complex coefficient split into channels, as produced by
/// `build_coefficient_assignments`. The list is collision-free, so
/// [`map_scatter_kernel`] assigns each bin from its own thread (no self-Friedel
/// doubling); the inverse transform folds the `1/(nu*nv*nw)` normalization, and
/// the real channel is the density (matching the host `build_coefficient_grid`
/// + [`gpu_cfft_3d`] `Inverse` staged path).
///
/// # Panics
///
/// Panics if the wgpu device cannot be initialized.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::missing_panics_doc,
    clippy::too_many_arguments
)]
pub fn gpu_inverse_map_resident(
    device: &WgpuDevice,
    bins: &[u32],
    re: &[f32],
    im: &[f32],
    dims: [usize; 3],
) -> Vec<f32> {
    let [nu, nv, nw] = dims;
    let total = nu * nv * nw;
    if total == 0 {
        return Vec::new();
    }
    let n_entry = bins.len();

    let client = WgpuRuntime::client(device);

    // Zeroed complex grid; the scatter writes only the coefficient bins, so the
    // rest must stay zero for the inverse transform.
    let re_buf = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; total]));
    let im_buf = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; total]));

    if n_entry > 0 {
        let h_bin = client.create_from_slice(u32::as_bytes(bins));
        let h_re = client.create_from_slice(f32::as_bytes(re));
        let h_im = client.create_from_slice(f32::as_bytes(im));
        let entry_blocks = (n_entry as u32).div_ceil(WORKGROUP);

        map_scatter_kernel::launch::<WgpuRuntime>(
            &client,
            cube_count_1d(entry_blocks),
            CubeDim::new_1d(WORKGROUP),
            unsafe { ArrayArg::from_raw_parts(re_buf.clone(), total) },
            unsafe { ArrayArg::from_raw_parts(im_buf.clone(), total) },
            unsafe { ArrayArg::from_raw_parts(h_bin, n_entry) },
            unsafe { ArrayArg::from_raw_parts(h_re, n_entry) },
            unsafe { ArrayArg::from_raw_parts(h_im, n_entry) },
            n_entry as u32,
        );
    }

    // Inverse FFT in place; the coefficient grid is Hermitian, so `re_buf` now
    // holds the real-space density and `im_buf` is discarded.
    fft_3d_axes_inplace(&client, &re_buf, &im_buf, dims, FftMode::Inverse);

    f32::from_bytes(&client.read_one_unchecked(re_buf)).to_vec()
}

/// Whether `dims` is a valid transform domain for [`gpu_cfft_3d`]: every axis
/// 5-smooth (only 2/3/5 prime factors) and no longer than [`MAX_FFT_LEN`]. This
/// is the same five-smooth family the CPU grid sizer (`round_up_to_smooth`)
/// produces, so the GPU FFT runs on the pipeline's native grid with no pow2
/// inflation.
#[must_use]
pub(crate) fn gpu_cfft_grid_supported(dims: [usize; 3]) -> bool {
    dims.iter().all(|&d| {
        (1..=MAX_FFT_LEN).contains(&d)
            && i32::try_from(d).is_ok_and(has_small_factorization)
    })
}

/// Build a molex GPU device that SHARES foldit's/viso's existing wgpu device
/// with cubecl, instead of letting molex create its own.
///
/// The caller assembles a [`cubecl::wgpu::WgpuSetup`] from its own wgpu
/// resources (instance/adapter/device/queue/backend) and hands it here; cubecl
/// registers the device and returns the [`WgpuDevice`] handle to pass to
/// [`XtalRefinement::with_gpu_device`]. Call ONCE and cache the result (cubecl
/// mints a fresh id per call). Omit entirely to let molex self-create via
/// `WgpuDevice::default()`.
///
/// [`XtalRefinement::with_gpu_device`]: super::refinement::XtalRefinement::with_gpu_device
#[must_use]
pub fn shared_gpu_device(setup: cubecl::wgpu::WgpuSetup) -> WgpuDevice {
    cubecl::wgpu::init_device(setup, cubecl::wgpu::RuntimeOptions::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::refinement_from_cif_pair;
    use crate::xtal::density::{compute_blur, splat_density};
    use crate::xtal::form_factors::form_factor;
    use crate::xtal::types::UnitCell;

    /// Max over the grid of `|gpu - cpu|`, and the peak CPU magnitude, so the
    /// caller can form a peak-normalized relative error robust to the near-zero
    /// tail where the CPU LUT and exact kernel diverge in relative terms.
    fn max_abs_diff_and_peak(gpu: &[f32], cpu: &[f32]) -> (f64, f64) {
        let mut max_diff = 0.0_f64;
        let mut peak = 0.0_f64;
        for (&g, &c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max(f64::from((g - c).abs()));
            peak = peak.max(f64::from(c.abs()));
        }
        (max_diff, peak)
    }

    #[allow(clippy::cast_precision_loss)]
    fn integrated(data: &[f32], cell_volume: f64, total: usize) -> f64 {
        let voxel = cell_volume / total as f64;
        data.iter().map(|&v| f64::from(v)).sum::<f64>() * voxel
    }

    /// Every dispatch count `cube_count_1d` returns must be an exact cover:
    /// `a * b == groups` with both dims inside the backend cap, so no surplus
    /// cube reaches an unguarded window read.
    #[test]
    #[allow(clippy::panic)]
    fn cube_count_1d_is_exact_and_in_range() {
        let cases = [
            1_u32,
            2,
            1000,
            65_534,
            MAX_CUBE_DIM,
            MAX_CUBE_DIM + 1,
            65_536,
            640 * 800,
            768 * 900,
            1024 * 1024,
            1_000_000,
            600 * 625,
        ];
        for &groups in &cases {
            let CubeCount::Static(a, b, c) = cube_count_1d(groups) else {
                panic!("expected Static count for {groups}");
            };
            assert_eq!(c, 1, "third dim must be 1 for {groups}");
            assert_eq!(
                u64::from(a) * u64::from(b),
                u64::from(groups),
                "product must equal groups exactly for {groups} (got {a}*{b})"
            );
            assert!(
                a <= MAX_CUBE_DIM && b <= MAX_CUBE_DIM,
                "both dims must fit MAX_CUBE_DIM for {groups} (got {a}, {b})"
            );
        }
    }

    #[test]
    #[allow(
        clippy::print_stdout,
        clippy::cast_precision_loss,
        clippy::expect_used
    )]
    fn gpu_splat_matches_cpu_synthetic() {
        let cell = UnitCell::new(15.0, 15.0, 15.0, 90.0, 90.0, 90.0);
        let (nu, nv, nw) = (48usize, 48, 48);
        let ff = form_factor(crate::element::Element::C)
            .expect("carbon form factor");
        let positions = [[0.5, 0.5, 0.5], [0.25, 0.6, 0.4]];
        let ffs = [ff, ff];
        let b_factors = [20.0, 35.0];
        let occupancies = [1.0, 0.8];
        let blur = compute_blur(2.0, 1.5, 20.0);

        let params = SplatParams {
            positions: &positions,
            form_factors: &ffs,
            b_factors: &b_factors,
            occupancies: &occupancies,
            unit_cell: &cell,
            blur,
        };

        let mut cpu = DensityGrid {
            data: vec![0.0; nu * nv * nw],
            nu,
            nv,
            nw,
        };
        splat_density(&mut cpu, &params);
        let gpu =
            gpu_splat_density(&WgpuDevice::default(), &params, [nu, nv, nw]);

        let (max_diff, peak) = max_abs_diff_and_peak(&gpu.data, &cpu.data);
        let rel = max_diff / peak;
        let total = nu * nv * nw;
        let z_gpu = integrated(&gpu.data, cell.volume, total);
        let z_cpu = integrated(&cpu.data, cell.volume, total);
        println!(
            "synthetic: peak={peak:.4} max_abs_diff={max_diff:.3e} \
             rel={rel:.3e} Z_gpu={z_gpu:.3} Z_cpu={z_cpu:.3}"
        );
        assert!(rel < 1e-3, "peak-normalized rel error {rel:.3e} >= 1e-3");
        assert!(
            (z_gpu - z_cpu).abs() / z_cpu < 0.10,
            "integrated density mismatch: gpu={z_gpu} cpu={z_cpu}"
        );
    }

    #[test]
    #[allow(clippy::print_stdout, clippy::cast_precision_loss)]
    fn gpu_splat_matches_cpu_real_structure() {
        let Some(case) = refinement_from_cif_pair("1AKI") else {
            println!("1AKI fixtures missing; skipping real-structure case");
            return;
        };
        let [nu, nv, nw] = case.refinement.grid_dims;
        let cell = case.refinement.unit_cell.clone();

        let ffs: Vec<&_> = case
            .elements
            .iter()
            .filter_map(|&e| form_factor(e))
            .collect();
        assert_eq!(
            ffs.len(),
            case.positions.len(),
            "every atom needs a form factor"
        );

        let b_min = case.b_factors.iter().copied().fold(f64::MAX, f64::min);
        let mut max_s2 = 0.0_f64;
        for r in &case.refinement.reflections {
            max_s2 = max_s2.max(cell.d_star_sq(r.h, r.k, r.l));
        }
        let d_min = if max_s2 > 0.0 {
            1.0 / max_s2.sqrt()
        } else {
            2.0
        };
        let blur = compute_blur(d_min, 1.5, b_min);

        let params = SplatParams {
            positions: &case.positions,
            form_factors: &ffs,
            b_factors: &case.b_factors,
            occupancies: &case.occupancies,
            unit_cell: &cell,
            blur,
        };

        let mut cpu = DensityGrid {
            data: vec![0.0; nu * nv * nw],
            nu,
            nv,
            nw,
        };
        splat_density(&mut cpu, &params);
        let gpu =
            gpu_splat_density(&WgpuDevice::default(), &params, [nu, nv, nw]);

        let (max_diff, peak) = max_abs_diff_and_peak(&gpu.data, &cpu.data);
        let rel = max_diff / peak;
        let total = nu * nv * nw;
        let z_gpu = integrated(&gpu.data, cell.volume, total);
        let z_cpu = integrated(&cpu.data, cell.volume, total);
        println!(
            "1AKI ({nu}x{nv}x{nw}, {} atoms): peak={peak:.4} \
             max_abs_diff={max_diff:.3e} rel={rel:.3e} Z_gpu={z_gpu:.1} \
             Z_cpu={z_cpu:.1}",
            case.positions.len()
        );
        assert!(rel < 1e-3, "peak-normalized rel error {rel:.3e} >= 1e-3");
        assert!(
            (z_gpu - z_cpu).abs() / z_cpu < 0.10,
            "integrated density mismatch: gpu={z_gpu} cpu={z_cpu}"
        );
    }
}

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::cast_precision_loss, clippy::expect_used)]
mod cfft_tests {
    use super::*;
    use crate::xtal::fft_cpu::{fft_3d_forward, fft_3d_inverse};

    /// Deterministic `[-1, 1)` pseudo-random stream (SplitMix-style LCG);
    /// avoids pulling in an rng dependency for a fixed-seed correctness
    /// check.
    fn next_unit(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (*state >> 40) as u32; // top 24 bits
        (bits as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0
    }

    /// RMS-floored max relative error between two complex fields, so near-zero
    /// bins do not dominate the ratio.
    fn max_rel_complex(gr: &[f32], gi: &[f32], cr: &[f32], ci: &[f32]) -> f64 {
        let energy: f64 = cr
            .iter()
            .zip(ci)
            .map(|(&r, &i)| {
                f64::from(r) * f64::from(r) + f64::from(i) * f64::from(i)
            })
            .sum();
        let floor = (energy / cr.len() as f64).sqrt();
        let mut worst = 0.0_f64;
        for (((&r, &i), &cru), &ciu) in gr.iter().zip(gi).zip(cr).zip(ci) {
            let err = (f64::from(r) - f64::from(cru))
                .hypot(f64::from(i) - f64::from(ciu));
            let denom = f64::from(cru).hypot(f64::from(ciu)).max(floor);
            worst = worst.max(err / denom);
        }
        worst
    }

    /// Forward GPU-vs-CPU and Inverse(Forward(x)) round-trip on random grids,
    /// on-device (Metal). Covers the pow2 fast path plus the 5-smooth (2/3/5)
    /// grid family the pipeline actually uses, cubic and non-cubic. Prints the
    /// forward max-rel and round-trip abs error for each grid.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn gpu_cfft_3d_matches_cpu() {
        let grids: [[usize; 3]; 10] = [
            // pow2 fast path.
            [8, 8, 8],
            [16, 16, 16],
            [64, 64, 64],
            [32, 16, 64],
            // cubic 5-smooth (pure-3, 2*5, 2*3, 2*3 mixes).
            [27, 27, 27],
            [40, 40, 40],
            [54, 54, 54],
            [72, 72, 72],
            // non-cubic 5-smooth.
            [40, 54, 72],
            [45, 48, 50],
        ];
        let mut seed = 0x1234_5678_9abc_def0_u64;

        for dims in grids {
            let [nu, nv, nw] = dims;
            let total = nu * nv * nw;

            let re: Vec<f32> =
                (0..total).map(|_| next_unit(&mut seed)).collect();
            let im_zero = vec![0.0_f32; total];

            // (a) forward vs CPU (real input, imaginary zero).
            let (gr, gi) = gpu_cfft_3d(
                &WgpuDevice::default(),
                &re,
                &im_zero,
                dims,
                FftMode::Forward,
            );
            let cpu = fft_3d_forward(&re, nu, nv, nw).expect("cpu forward");
            let cr: Vec<f32> = cpu.iter().map(|c| c[0]).collect();
            let ci: Vec<f32> = cpu.iter().map(|c| c[1]).collect();
            let fwd_rel = max_rel_complex(&gr, &gi, &cr, &ci);

            // (b) round-trip on a full complex field.
            let im: Vec<f32> =
                (0..total).map(|_| next_unit(&mut seed)).collect();
            let (fr, fi) = gpu_cfft_3d(
                &WgpuDevice::default(),
                &re,
                &im,
                dims,
                FftMode::Forward,
            );
            let (rr, ri) = gpu_cfft_3d(
                &WgpuDevice::default(),
                &fr,
                &fi,
                dims,
                FftMode::Inverse,
            );
            let mut rt = 0.0_f64;
            for (((&a, &b), &c), &d) in rr.iter().zip(&ri).zip(&re).zip(&im) {
                rt = rt.max(f64::from(a - c).abs());
                rt = rt.max(f64::from(b - d).abs());
            }

            // Cross-check the CPU forward path stays consistent too.
            let cpu_rt = fft_3d_inverse(&cpu, nu, nv, nw).expect("cpu inv");
            let mut cpu_rt_err = 0.0_f64;
            for (&a, &c) in cpu_rt.iter().zip(&re) {
                cpu_rt_err = cpu_rt_err.max(f64::from(a - c).abs());
            }

            println!(
                "grid {nu}x{nv}x{nw}: forward max-rel={fwd_rel:.3e} \
                 round-trip abs={rt:.3e} (cpu round-trip abs={cpu_rt_err:.3e})"
            );

            assert!(
                fwd_rel < 1e-4,
                "grid {nu}x{nv}x{nw} forward rel error {fwd_rel:.3e} >= 1e-4"
            );
            assert!(
                rt < 1e-5,
                "grid {nu}x{nv}x{nw} round-trip abs error {rt:.3e} >= 1e-5"
            );
        }
    }
}
