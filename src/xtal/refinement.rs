//! `XtalRefinement`: main orchestrator for crystallographic maximum-likelihood
//! refinement.
// foldit:allow-long-file - cohesive refinement pipeline plus gradient tests

use super::bessel::bessel_i1_over_i0;
use super::form_factors::{self, FormFactor};
use super::types::{
    CrystalSystem, DensityGrid, GroupOps, Reflection, SpaceGroup, DEN,
};
use super::{
    density, fft_cpu, map_coefficients, scaling, sigma_a, solvent_mask,
    targets, ScalingResult, SigmaAResult,
};
use crate::element::Element;

/// Minimum allowed B-factor (Ų). The softplus transform in
/// [`super::bfactor_refine`] maps every optimizer variable to `B >= B_MIN`.
pub(crate) const B_MIN: f64 = 2.0;

/// Maximum allowed B-factor (Ų). Bounds the initial guess handed to the
/// optimizer; the softplus transform itself is unbounded above.
pub(crate) const B_MAX: f64 = 300.0;

/// Per-reflection complex structure factors for the model and its bulk-solvent
/// mask: `(refl_fc, refl_fmask)`. Each `[f32; 2]` is a `(real, imag)` pair.
type ReflStructureFactors = (Vec<[f32; 2]>, Vec<[f32; 2]>);

/// Per-copy splat inputs for one atom set expanded over its full symmetry
/// orbit: fractional positions, form factors, B-factors, and occupancies, each
/// `n_atoms * n_sym * n_cen` long.
type OrbitSplatInputs<'f> =
    (Vec<[f64; 3]>, Vec<&'f FormFactor>, Vec<f64>, Vec<f64>);

/// How the forward model realizes the crystal's space-group symmetry on the
/// real-space density grid before the FFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ForwardSymmetry {
    /// Splat only the asymmetric-unit atoms, then sum the density grid over
    /// the full symmetry orbit ([`density::symmetrize_sum`]). Requires
    /// grid axes divisible by the space group's grid factors.
    SymmetrizeGrid,
    /// Splat every atom's full symmetry-plus-centering orbit directly onto the
    /// grid; no grid symmetrization. Imposes no grid-factor divisibility, so
    /// it admits arbitrary (e.g. power-of-two) grids.
    // Constructed only through the `forward_symmetry` opt-in (default
    // `SymmetrizeGrid`), so a plain build never selects it.
    #[allow(dead_code)]
    SplatFullOrbit,
}

/// Backend for the real-space grid stencils.
///
/// Gates both the forward density splat and its adjoint, the B-factor gradient
/// gather. Both walk the same covered-voxel boxes, so they share one device
/// decision.
///
/// Defaults to [`StencilBackend::Cpu`]; the GPU variant exists only when the
/// `xtal-gpu` feature is enabled and produces results within `f32` tolerance
/// of the CPU path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StencilBackend {
    /// Run the stencils on the CPU (default).
    #[default]
    Cpu,
    /// Run the stencils on the GPU via CubeCL on the wgpu backend.
    #[cfg(feature = "xtal-gpu")]
    Gpu,
}

/// Wrap a signed Miller index into the range `[0, size)` using modular
/// arithmetic.  Safe for any index magnitude (unlike the bare
/// `(idx + size) as usize` pattern which overflows when `|idx| > size`).
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn wrap_miller_index(idx: i32, size: usize) -> usize {
    let s = size as i32;
    let wrapped = ((idx % s) + s) % s;
    wrapped as usize
}

/// Main orchestrator for crystallographic maximum-likelihood refinement.
///
/// Holds unit cell, space group, reflection data, and grid dimensions. Provides
/// methods to run the full pipeline (density -> FFT -> mask -> scale -> sigma-A
/// -> map coefficients -> IFFT) and to refine B-factors.
pub struct XtalRefinement {
    /// The crystallographic unit cell.
    pub unit_cell: super::UnitCell,
    /// The space group operations.
    pub group_ops: GroupOps,
    /// Crystal system for anisotropic scaling constraints.
    pub crystal_system: CrystalSystem,
    /// Observed reflection data.
    pub reflections: Vec<Reflection>,
    /// Grid dimensions `[nu, nv, nw]`.
    pub grid_dims: [usize; 3],
    /// Current scaling result (fitted parameters).
    pub scaling: ScalingResult,
    /// Current sigma-A estimates.
    pub sigma_a: Option<SigmaAResult>,
    /// Internal working precision for the forward/inverse 3-D FFTs. Defaults
    /// to [`fft_cpu::FftPrecision::F64`]; set to `F32` to measure the GPU
    /// path's precision without changing the default numerical behavior.
    pub fft_precision: fft_cpu::FftPrecision,
    /// Backend for the real-space grid stencils (density splat and gradient
    /// gather). Defaults to [`StencilBackend::Cpu`]; set to `Gpu` (requires
    /// the `xtal-gpu` feature) to run both on the GPU without changing the
    /// default behavior.
    pub stencil_backend: StencilBackend,
    /// How the forward model realizes space-group symmetry before the FFT.
    /// Defaults to [`ForwardSymmetry::SymmetrizeGrid`] (the grid-factor
    /// pipeline); set to `SplatFullOrbit` to run the forward Fc and the map
    /// computation on an arbitrary (e.g. power-of-two) grid, the form the GPU
    /// FFT accepts.
    pub(crate) forward_symmetry: ForwardSymmetry,
}

impl XtalRefinement {
    /// Create a new refinement state.
    ///
    /// # Arguments
    ///
    /// * `unit_cell` - the crystallographic unit cell
    /// * `sg` - the space group (consumed; ops and crystal system are
    ///   extracted)
    /// * `reflections` - observed reflection data
    /// * `grid_dims` - `[nu, nv, nw]` grid dimensions for the density map
    #[must_use]
    pub fn new(
        unit_cell: super::UnitCell,
        sg: SpaceGroup,
        reflections: Vec<Reflection>,
        grid_dims: [usize; 3],
    ) -> Self {
        // Filter out (0,0,0) and systematic absences; these must not
        // participate in scaling, sigma-A, R-factors, or maximum-likelihood
        // targets.
        let filtered: Vec<Reflection> = reflections
            .into_iter()
            .filter(|r| {
                // Skip the origin reflection (unmeasurable, causes 1/d = Inf).
                if r.h == 0 && r.k == 0 && r.l == 0 {
                    return false;
                }
                // Skip systematic absences for this space group.
                !super::is_systematically_absent([r.h, r.k, r.l], &sg.ops)
            })
            .collect();

        Self {
            unit_cell,
            group_ops: sg.ops,
            crystal_system: sg.crystal_system,
            reflections: filtered,
            grid_dims,
            scaling: ScalingResult {
                k_overall: 1.0,
                b_aniso: [0.0; 6],
                k_sol: 0.35,
                b_sol: 46.0,
            },
            sigma_a: None,
            fft_precision: fft_cpu::FftPrecision::F64,
            stencil_backend: StencilBackend::default(),
            forward_symmetry: ForwardSymmetry::SymmetrizeGrid,
        }
    }

    /// Run the full map computation pipeline.
    ///
    /// 1. Splat atomic density onto the grid
    /// 2. FFT -> Fc
    /// 3. Deblur Fc
    /// 4. Compute solvent mask -> FFT -> Fmask
    /// 5. Fit scaling parameters
    /// 6. Estimate sigma-A
    /// 7. Compute map coefficients
    /// 8. Inverse FFT -> real-space density
    ///
    /// Returns the 2mFo-DFc electron density grid, or `None` if the pipeline
    /// fails (e.g. FFT dimension mismatch).
    ///
    /// # Arguments
    ///
    /// * `positions` - fractional coordinates of ASU atoms
    /// * `elements` - element type per atom
    /// * `b_factors` - isotropic B-factor per atom
    /// * `occupancies` - site occupancy per atom
    pub fn compute_map(
        &mut self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &[f64],
        occupancies: &[f64],
    ) -> Option<DensityGrid> {
        let [nu, nv, nw] = self.grid_dims;

        // Steps 1-4: splat -> symmetrize -> FFT -> deblur, plus the real
        // solvent mask FFT that yields a genuine Fmask.
        let (refl_fc, refl_fmask) = self.model_structure_factors(
            positions,
            elements,
            b_factors,
            occupancies,
            true,
            self.forward_symmetry,
        )?;

        // Fc amplitudes for scaling/sigma-A.
        let fc_amps: Vec<f64> = refl_fc
            .iter()
            .map(|c| f64::from(c[0]).hypot(f64::from(c[1])))
            .collect();

        // Step 5: Fit scaling.
        self.scaling = scaling::fit_scaling(
            &self.reflections,
            &refl_fc,
            &refl_fmask,
            &self.unit_cell,
            &self.crystal_system,
        );

        // Step 6: Sigma-A estimation.
        let sa = sigma_a::estimate_sigma_a(
            &self.reflections,
            &fc_amps,
            &self.unit_cell,
        );
        self.sigma_a = Some(sa.clone());

        // Step 7: Map coefficients.
        let map_coeffs = map_coefficients::compute_map_coefficients(
            &self.reflections,
            &refl_fc,
            &sa,
            &self.unit_cell,
        );

        // Step 8: Inverse FFT -> density.
        let density_data = map_coefficients::map_from_coefficients(
            &map_coeffs.two_fo_fc,
            &self.reflections,
            nu,
            nv,
            nw,
        )
        .ok()?;

        Some(DensityGrid {
            data: density_data,
            nu,
            nv,
            nw,
        })
    }

    /// Compute the per-atom B-factor gradient of the maximum-likelihood target
    /// via the FFT-map (Agarwal/Ten Eyck) reformulation.
    ///
    /// Mathematically identical to [`Self::b_factor_gradients_direct`] but
    /// `O(N_grid log N)` instead of `O(reflections × atoms × symmetry)`: the
    /// per-reflection likelihood weights are packed into reciprocal-space
    /// coefficients, one inverse FFT builds a real-space gradient map, and each
    /// atom's gradient is a real-space gather of that map against the atom's
    /// own Gaussian kernel over its symmetry orbit. Requires that
    /// `compute_map` has populated sigma-A.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn b_factor_gradients(
        &self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &[f64],
        occupancies: &[f64],
    ) -> Option<Vec<f64>> {
        // Complex model structure factors on the physical amplitude scale,
        // mask-free; the same Fc the maximum-likelihood target differentiates.
        let refl_fc =
            self.forward_fc(positions, elements, b_factors, occupancies)?;
        self.b_factor_gradients_from_fc(
            &refl_fc,
            positions,
            elements,
            b_factors,
            occupancies,
        )
    }

    /// Compute the B-factor gradient from a precomputed `refl_fc`.
    ///
    /// The Fc-dependent half of [`Self::b_factor_gradients`]: packs the
    /// per-reflection likelihood weights into reciprocal-space coefficients,
    /// runs one inverse FFT, and gathers each atom's real-space gradient.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::excessive_nesting
    )]
    pub(crate) fn b_factor_gradients_from_fc(
        &self,
        refl_fc: &[[f32; 2]],
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &[f64],
        occupancies: &[f64],
    ) -> Option<Vec<f64>> {
        let sa = self.sigma_a.as_ref()?;
        let [nu, nv, nw] = self.grid_dims;

        let ffs: Vec<&FormFactor> = elements
            .iter()
            .filter_map(|e| form_factors::form_factor(*e))
            .collect();
        if ffs.len() != positions.len() {
            return None;
        }

        // The blur the forward splat/deblur used, recomputed identically so the
        // deblur compensation on the coefficients matches the blur baked into
        // the per-atom gather kernels.
        let b_min = b_factors.iter().copied().fold(f64::MAX, f64::min);
        let d_min = self.estimate_d_min();
        let blur = density::compute_blur(d_min, 1.5, b_min);

        // Device-resident inverse chain: on a device-transformable grid the GPU
        // backend packs the coefficients, inverse-FFTs, and gathers with the
        // coefficient grid and gradient map kept on the GPU. Only the small
        // per-reflection scatter data is uploaded; the CPU pack/FFT below never
        // runs. Every other configuration keeps the host path unchanged.
        #[cfg(feature = "xtal-gpu")]
        if self.stencil_backend == StencilBackend::Gpu
            && super::gpu::gpu_cfft_grid_supported([nu, nv, nw])
        {
            let mut idx = Vec::new();
            let mut idxn = Vec::new();
            let mut dre = Vec::new();
            let mut dim = Vec::new();
            for (i, refl) in self.reflections.iter().enumerate() {
                if refl.free_flag {
                    continue;
                }
                let Some((re, im, fc, s2, w_h)) =
                    self.reflection_ml_weight(refl, refl_fc[i], sa)
                else {
                    continue;
                };
                let deblur = (blur * s2 / 4.0).min(80.0).exp();
                let coeff = w_h * -(s2 / 4.0) * deblur;
                let u = wrap_miller_index(refl.h, nu);
                let v = wrap_miller_index(refl.k, nv);
                let w = wrap_miller_index(refl.l, nw);
                let un = wrap_miller_index(-refl.h, nu);
                let vn = wrap_miller_index(-refl.k, nv);
                let wn = wrap_miller_index(-refl.l, nw);
                #[allow(clippy::cast_possible_truncation)]
                {
                    dre.push((coeff * re / fc) as f32);
                    dim.push((coeff * im / fc) as f32);
                    idx.push(((u * nv + v) * nw + w) as u32);
                    idxn.push(((un * nv + vn) * nw + wn) as u32);
                }
            }

            let scale = self.unit_cell.volume / 2.0;
            let (copy_pos, copy_ff, copy_b, copy_occ) =
                self.expand_orbit(positions, &ffs, b_factors, occupancies);
            let gather_params = density::SplatParams {
                positions: &copy_pos,
                form_factors: &copy_ff,
                b_factors: &copy_b,
                occupancies: &copy_occ,
                unit_cell: &self.unit_cell,
                blur,
            };
            return Some(super::gpu::gpu_inverse_gradient_resident(
                &idx,
                &idxn,
                &dre,
                &dim,
                &gather_params,
                [nu, nv, nw],
                positions.len(),
                scale,
            ));
        }

        // Reciprocal-space gradient coefficients. For each working reflection
        // place `coeff · e^{+iφ}` at its wrapped grid index and the conjugate
        // at the Friedel mate, so the inverse FFT yields a purely real
        // map. The per-reflection weight/sign expressions are identical
        // to the direct sum, so the two agree by construction up to
        // grid sampling.
        let n = nu * nv * nw;
        let mut d_grid = vec![[0.0_f64; 2]; n];

        for (i, refl) in self.reflections.iter().enumerate() {
            if refl.free_flag {
                continue;
            }

            let Some((re, im, fc, s2, w_h)) =
                self.reflection_ml_weight(refl, refl_fc[i], sa)
            else {
                continue;
            };

            // -s²/4 is the reflection's Debye-Waller derivative factor; the
            // exp(+blur·s²/4) deblur compensation nets against the blur baked
            // into the gather kernel. Clamp mirrors `deblur_fc`.
            let deblur = (blur * s2 / 4.0).min(80.0).exp();
            let coeff = w_h * -(s2 / 4.0) * deblur;

            let dre = coeff * re / fc;
            let dim = coeff * im / fc;

            let u = wrap_miller_index(refl.h, nu);
            let v = wrap_miller_index(refl.k, nv);
            let w = wrap_miller_index(refl.l, nw);
            let idx = (u * nv + v) * nw + w;
            d_grid[idx][0] += dre;
            d_grid[idx][1] += dim;

            let un = wrap_miller_index(-refl.h, nu);
            let vn = wrap_miller_index(-refl.k, nv);
            let wn = wrap_miller_index(-refl.l, nw);
            let idxn = (un * nv + vn) * nw + wn;
            d_grid[idxn][0] += dre;
            d_grid[idxn][1] -= dim;
        }

        // The shared FFT path stores coefficients as f32 pairs; accumulation
        // above stays f64 to avoid repeated rounding at colliding indices.
        #[allow(clippy::cast_possible_truncation)]
        let d_grid_f32: Vec<[f32; 2]> =
            d_grid.iter().map(|c| [c[0] as f32, c[1] as f32]).collect();

        // The inverse FFT's built-in 1/N cancels the N/V of the real-space
        // sum-to-integral discretization; only V/2 (Friedel doubling over the
        // stored hemisphere) remains as the overall scale. The device-resident
        // GPU inverse returned above; this CPU FFT serves the CPU backend and
        // any GPU grid the device transform does not support.
        let g_map = fft_cpu::fft_3d_inverse_prec(
            &d_grid_f32,
            nu,
            nv,
            nw,
            self.fft_precision,
        )
        .ok()?;

        // Per-atom reverse splat over the full symmetry/centering orbit, the
        // same orbit expansion the forward model and the direct sum use.
        let scale = self.unit_cell.volume / 2.0;

        let grad = match self.stencil_backend {
            StencilBackend::Cpu => {
                let (copy_pos, copy_ff, copy_b, copy_occ) =
                    self.expand_orbit(positions, &ffs, b_factors, occupancies);
                let mult =
                    self.group_ops.sym_ops.len() * self.group_ops.cen_ops.len();
                let mut grad = vec![0.0_f64; positions.len()];
                for (j, chunk) in copy_pos.chunks(mult).enumerate() {
                    let mut sum = 0.0_f64;
                    for (k, &p) in chunk.iter().enumerate() {
                        let idx = j * mult + k;
                        sum += density::gather_gradient(
                            &g_map,
                            p,
                            copy_ff[idx],
                            copy_b[idx],
                            copy_occ[idx],
                            &self.unit_cell,
                            nu,
                            nv,
                            nw,
                            blur,
                        );
                    }
                    grad[j] = sum * scale;
                }
                grad
            }
            #[cfg(feature = "xtal-gpu")]
            StencilBackend::Gpu => {
                // Per-copy inputs are atom-major so the GPU wrapper reduces
                // each atom's copies as a contiguous chunk.
                let (copy_pos, copy_ff, copy_b, copy_occ) =
                    self.expand_orbit(positions, &ffs, b_factors, occupancies);
                let gather_params = density::SplatParams {
                    positions: &copy_pos,
                    form_factors: &copy_ff,
                    b_factors: &copy_b,
                    occupancies: &copy_occ,
                    unit_cell: &self.unit_cell,
                    blur,
                };
                super::gpu::gpu_gather_gradient(
                    &g_map,
                    &gather_params,
                    [nu, nv, nw],
                    positions.len(),
                    scale,
                )
            }
        };

        Some(grad)
    }

    /// Compute the per-atom B-factor gradient by direct reciprocal-space sum.
    ///
    /// For every working reflection the likelihood weight `dW/d|Fc|` is chained
    /// through the analytic derivative of `|Fc|` with respect to each atom's
    /// isotropic B. The per-atom term carries the atom's own scattering factor
    /// and Debye-Waller factor, and is summed over the reflection's symmetry-
    /// and centering-related atom copies so that it differentiates exactly the
    /// symmetric `Fc` the forward model builds. Requires that `compute_map` has
    /// populated sigma-A.
    ///
    /// This is `O(reflections × atoms × symmetry)`. It is the test-only
    /// validation oracle for [`Self::b_factor_gradients`], which computes the
    /// same quantity via an inverse FFT and a per-atom real-space gather.
    #[cfg_attr(not(test), allow(dead_code))]
    #[allow(clippy::too_many_lines)]
    pub(crate) fn b_factor_gradients_direct(
        &self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &[f64],
        occupancies: &[f64],
    ) -> Option<Vec<f64>> {
        let sa = self.sigma_a.as_ref()?;

        // Complex model structure factors on the physical amplitude scale,
        // mask-free; the same Fc the maximum-likelihood target differentiates.
        let (refl_fc, _) = self.model_structure_factors(
            positions,
            elements,
            b_factors,
            occupancies,
            false,
            ForwardSymmetry::SymmetrizeGrid,
        )?;

        let ffs: Vec<&FormFactor> = elements
            .iter()
            .filter_map(|e| form_factors::form_factor(*e))
            .collect();
        if ffs.len() != positions.len() {
            return None;
        }

        // Deduplicate form factors by element so each reflection evaluates the
        // scattering factor once per distinct element, not once per atom.
        let mut unique_elems: Vec<Element> = Vec::new();
        let mut unique_ffs: Vec<&FormFactor> = Vec::new();
        let mut elem_index: Vec<usize> = Vec::with_capacity(elements.len());
        for (&e, ff) in elements.iter().zip(ffs.iter()) {
            if let Some(idx) = unique_elems.iter().position(|&u| u == e) {
                elem_index.push(idx);
            } else {
                elem_index.push(unique_elems.len());
                unique_elems.push(e);
                unique_ffs.push(ff);
            }
        }
        let mut ff_vals = vec![0.0_f64; unique_ffs.len()];

        // Expand every atom into its full orbit of symmetry- and
        // centering-related fractional copies, exactly as the forward model
        // does (splat then `symmetrize_sum` over sym_ops × cen_ops). Fc sums
        // over these copies, so the gradient must too or it will not match the
        // finite difference of Fc.
        let den = f64::from(DEN);
        let n_group =
            self.group_ops.sym_ops.len() * self.group_ops.cen_ops.len();
        let mut expanded: Vec<[f64; 3]> =
            Vec::with_capacity(positions.len() * n_group);
        for pos in positions {
            for sym in &self.group_ops.sym_ops {
                let sp = sym.apply_to_frac(*pos);
                for cen in &self.group_ops.cen_ops {
                    expanded.push([
                        sp[0] + f64::from(cen[0]) / den,
                        sp[1] + f64::from(cen[1]) / den,
                        sp[2] + f64::from(cen[2]) / den,
                    ]);
                }
            }
        }

        let mut grad = vec![0.0_f64; positions.len()];

        for (i, refl) in self.reflections.iter().enumerate() {
            if refl.free_flag {
                continue;
            }

            let Some((re, im, fc, s2, w_h)) =
                self.reflection_ml_weight(refl, refl_fc[i], sa)
            else {
                continue;
            };

            // (sin θ / λ)²; the argument of both the scattering factor and the
            // Debye-Waller factor exp(-B·stol2).
            let stol2 = s2 / 4.0;
            for (val, ff) in ff_vals.iter_mut().zip(unique_ffs.iter()) {
                *val = ff.at_s_sq(stol2);
            }

            let hf = f64::from(refl.h);
            let kf = f64::from(refl.k);
            let lf = f64::from(refl.l);

            for j in 0..positions.len() {
                // Σ over symmetry copies of the phase term, with the forward
                // FFT's e^{-2πi h·x} sign convention.
                let base = j * n_group;
                let mut c_sum = 0.0_f64;
                let mut s_sum = 0.0_f64;
                for p in &expanded[base..base + n_group] {
                    let phase = -std::f64::consts::TAU
                        * hf.mul_add(p[0], kf.mul_add(p[1], lf * p[2]));
                    c_sum += phase.cos();
                    s_sum += phase.sin();
                }

                let f_atom = ff_vals[elem_index[j]];
                let dw = (-b_factors[j] * stol2).exp();
                let g_j = occupancies[j] * f_atom * dw;

                // d|Fc|/dB_j = -(stol2)·g_j·(re·Σcos + im·Σsin) / |Fc|,
                // the projection of dFc/dB_j onto the Fc direction.
                let d_fc_d_b =
                    -stol2 * g_j * re.mul_add(c_sum, im * s_sum) / fc;
                grad[j] += w_h * d_fc_d_b;
            }
        }

        Some(grad)
    }

    /// Per-reflection maximum-likelihood weight shared by both B-factor
    /// gradient paths.
    ///
    /// Returns `(re, im, fc, s2, w_h)` where `w_h = dW/d|Fc|` is the
    /// Rice-target likelihood weight (epsilon = 1), or `None` when `|Fc|`
    /// is negligible and the reflection must be skipped.
    fn reflection_ml_weight(
        &self,
        refl: &Reflection,
        refl_fc_component: [f32; 2],
        sa: &SigmaAResult,
    ) -> Option<(f64, f64, f64, f64, f64)> {
        let re = f64::from(refl_fc_component[0]);
        let im = f64::from(refl_fc_component[1]);
        let fc = re.hypot(im);
        if fc < 1e-30 {
            return None;
        }

        let fo = f64::from(refl.f_obs);
        let s2 = self.unit_cell.d_star_sq(refl.h, refl.k, refl.l);
        let (d_val, sigma_sq) = sa.interpolate(s2);

        let x = 2.0 * fo * d_val * fc / sigma_sq;
        let m = bessel_i1_over_i0(x);
        let w_h = (2.0 * d_val / sigma_sq) * d_val.mul_add(fc, -(m * fo));

        Some((re, im, fc, s2, w_h))
    }

    /// Evaluate the maximum-likelihood target for a candidate B-factor set.
    ///
    /// Recomputes Fc from `b_factors` and scores the negative log-likelihood
    /// against the current sigma-A estimate. This is the objective the B-factor
    /// gradient descends, so a sufficiently small step is guaranteed to reduce
    /// it (unlike R-work, which can disagree with the gradient locally). Bulk
    /// solvent is not needed here, making it cheaper than an R-work evaluation.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn maximum_likelihood_target_for_b(
        &self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &[f64],
        occupancies: &[f64],
    ) -> Option<f64> {
        let refl_fc =
            self.forward_fc(positions, elements, b_factors, occupancies)?;
        self.maximum_likelihood_target_from_fc(&refl_fc)
    }

    /// Mask-free complex model structure factors (`refl_fc`) at each working
    /// reflection; the shared forward Fc that both the maximum-likelihood
    /// target and the B-factor gradient differentiate. Splitting it out lets a
    /// caller evaluating the co-located target and gradient at the same
    /// B-factors compute Fc once and hand it to both.
    pub(crate) fn forward_fc(
        &self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &[f64],
        occupancies: &[f64],
    ) -> Option<Vec<[f32; 2]>> {
        let (refl_fc, _) = self.model_structure_factors(
            positions,
            elements,
            b_factors,
            occupancies,
            false,
            self.forward_symmetry,
        )?;
        Some(refl_fc)
    }

    /// Mask-free complex model structure factors computed by splatting every
    /// atom's full symmetry orbit directly onto the grid, with no grid
    /// symmetrization. Numerically equivalent to [`Self::forward_fc`] up to
    /// grid sampling, but imposes no grid-factor divisibility, so it runs on an
    /// arbitrary (e.g. power-of-two) grid.
    #[allow(dead_code)] // exercised only by the orbit/pow2 comparison tests
    pub(crate) fn forward_fc_orbit(
        &self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &[f64],
        occupancies: &[f64],
    ) -> Option<Vec<[f32; 2]>> {
        let (refl_fc, _) = self.model_structure_factors(
            positions,
            elements,
            b_factors,
            occupancies,
            false,
            ForwardSymmetry::SplatFullOrbit,
        )?;
        Some(refl_fc)
    }

    /// Score the maximum-likelihood target from a precomputed `refl_fc`.
    ///
    /// The Fc-dependent half of [`Self::maximum_likelihood_target_for_b`],
    /// against the current sigma-A estimate.
    pub(crate) fn maximum_likelihood_target_from_fc(
        &self,
        refl_fc: &[[f32; 2]],
    ) -> Option<f64> {
        let sa = self.sigma_a.as_ref()?;

        let fc_amps: Vec<f64> = refl_fc
            .iter()
            .map(|c| f64::from(c[0]).hypot(f64::from(c[1])))
            .collect();

        Some(targets::maximum_likelihood_target(
            &self.reflections,
            &fc_amps,
            sa,
            &self.unit_cell,
        ))
    }

    /// Compute R-work and R-free for the current model.
    ///
    /// Returns `(r_work, r_free)`, or `None` if the pipeline fails.
    #[must_use]
    pub fn r_factors(
        &self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &[f64],
        occupancies: &[f64],
    ) -> Option<(f64, f64)> {
        // R-factors use the deblurred Fc without a separate bulk-solvent mask.
        let (refl_fc, refl_fmask) = self.model_structure_factors(
            positions,
            elements,
            b_factors,
            occupancies,
            false,
            ForwardSymmetry::SymmetrizeGrid,
        )?;

        let rw = targets::r_work(
            &self.reflections,
            &refl_fc,
            &refl_fmask,
            &self.scaling,
            &self.unit_cell,
        );
        let rf = targets::r_free_value(
            &self.reflections,
            &refl_fc,
            &refl_fmask,
            &self.scaling,
            &self.unit_cell,
        );

        Some((rw, rf))
    }

    /// Compute per-reflection model structure factors from the current atoms.
    ///
    /// Splats atomic density, symmetrizes, forward-FFTs, deblurs, and extracts
    /// the complex Fc value at each reflection. When `with_mask` is set, the
    /// real solvent mask is also FFT'd and its per-reflection amplitudes are
    /// returned as the second element; otherwise the second element mirrors
    /// `refl_fc` (no bulk-solvent contribution).
    ///
    /// Returns `(refl_fc, refl_fmask)`, or `None` if any element lacks a form
    /// factor or an FFT fails.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn model_structure_factors(
        &self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &[f64],
        occupancies: &[f64],
        with_mask: bool,
        symmetry: ForwardSymmetry,
    ) -> Option<ReflStructureFactors> {
        let [nu, nv, nw] = self.grid_dims;

        let ffs: Vec<&FormFactor> = elements
            .iter()
            .filter_map(|e| form_factors::form_factor(*e))
            .collect();
        if ffs.len() != positions.len() {
            return None; // some elements have no form factor
        }

        let b_min = b_factors.iter().copied().fold(f64::MAX, f64::min);
        let d_min = self.estimate_d_min();
        let blur = density::compute_blur(d_min, 1.5, b_min);

        // The default path splats the ASU and symmetrizes the grid; the orbit
        // path splats every atom's full orbit and skips symmetrization. The
        // orbit copies live here so the `SplatParams` borrow outlives the
        // splat.
        let orbit_pos: Vec<[f64; 3]>;
        let orbit_ffs: Vec<&FormFactor>;
        let orbit_b: Vec<f64>;
        let orbit_occ: Vec<f64>;
        let splat_params = match symmetry {
            ForwardSymmetry::SymmetrizeGrid => density::SplatParams {
                positions,
                form_factors: &ffs,
                b_factors,
                occupancies,
                unit_cell: &self.unit_cell,
                blur,
            },
            ForwardSymmetry::SplatFullOrbit => {
                (orbit_pos, orbit_ffs, orbit_b, orbit_occ) =
                    self.expand_orbit(positions, &ffs, b_factors, occupancies);
                density::SplatParams {
                    positions: &orbit_pos,
                    form_factors: &orbit_ffs,
                    b_factors: &orbit_b,
                    occupancies: &orbit_occ,
                    unit_cell: &self.unit_cell,
                    blur,
                }
            }
        };

        // Fully device-resident forward chain: the density grid and complex
        // spectrum stay on the GPU through splat, FFT, deblur, and extract, and
        // only the per-reflection Fc vector returns to host. A bulk-solvent
        // mask needs its own host FFT, so it takes the staged path below.
        #[cfg(feature = "xtal-gpu")]
        if self.stencil_backend == StencilBackend::Gpu
            && symmetry == ForwardSymmetry::SplatFullOrbit
            && !with_mask
            && super::gpu::gpu_cfft_grid_supported([nu, nv, nw])
        {
            let refl_fc = self.gpu_forward_fc_resident(&splat_params, blur);
            return Some((refl_fc.clone(), refl_fc));
        }

        let mut grid = DensityGrid {
            data: vec![0.0; nu * nv * nw],
            nu,
            nv,
            nw,
        };
        match self.stencil_backend {
            StencilBackend::Cpu => {
                density::splat_density(&mut grid, &splat_params);
            }
            #[cfg(feature = "xtal-gpu")]
            StencilBackend::Gpu => {
                grid =
                    super::gpu::gpu_splat_density(&splat_params, [nu, nv, nw]);
            }
        }
        if symmetry == ForwardSymmetry::SymmetrizeGrid {
            density::symmetrize_sum(&mut grid, &self.group_ops);
        }

        // Mask-carrying GPU orbit path (the mask-free case took the resident
        // chain above): run the forward FFT on device but read the spectrum
        // back so the host deblur/extract can share the map's solvent grid.
        // Every other configuration keeps the CPU FFT so the default pipeline
        // stays byte-identical. The GPU transform is f32 throughout, so it
        // ignores `fft_precision`.
        #[cfg(feature = "xtal-gpu")]
        let fc_complex = if self.stencil_backend == StencilBackend::Gpu
            && symmetry == ForwardSymmetry::SplatFullOrbit
            && super::gpu::gpu_cfft_grid_supported([nu, nv, nw])
        {
            let im0 = vec![0.0_f32; grid.data.len()];
            let (re, im) = super::gpu::gpu_cfft_3d(
                &grid.data,
                &im0,
                [nu, nv, nw],
                super::gpu::FftMode::Forward,
            );
            re.into_iter().zip(im).map(<[f32; 2]>::from).collect()
        } else {
            fft_cpu::fft_3d_forward_prec(
                &grid.data,
                nu,
                nv,
                nw,
                self.fft_precision,
            )
            .ok()?
        };
        #[cfg(not(feature = "xtal-gpu"))]
        let fc_complex = fft_cpu::fft_3d_forward_prec(
            &grid.data,
            nu,
            nv,
            nw,
            self.fft_precision,
        )
        .ok()?;
        let mut fc_deblurred = fc_complex;
        density::deblur_fc(
            &mut fc_deblurred,
            &self.unit_cell,
            nu,
            nv,
            nw,
            blur,
        );

        if with_mask {
            let mask = solvent_mask::solvent_mask(
                positions,
                elements,
                &self.unit_cell,
                nu,
                nv,
                nw,
                solvent_mask::DEFAULT_R_PROBE,
                solvent_mask::DEFAULT_R_SHRINK,
            );
            let fmask_complex = fft_cpu::fft_3d_forward_prec(
                &mask.data,
                nu,
                nv,
                nw,
                self.fft_precision,
            )
            .ok()?;
            Some(self.extract_reflection_values(
                &fc_deblurred,
                &fmask_complex,
                nu,
                nv,
                nw,
            ))
        } else {
            // No bulk solvent: pass Fc as the mask grid so refl_fmask mirrors
            // refl_fc, exactly as the pre-extraction callers did.
            Some(self.extract_reflection_values(
                &fc_deblurred,
                &fc_deblurred,
                nu,
                nv,
                nw,
            ))
        }
    }

    /// Drive the device-resident forward chain and return the per-reflection
    /// complex Fc. Builds the reciprocal-cell constants and the wrapped-Miller
    /// flat indices the deblur and extract kernels need, then hands the splat
    /// inputs to [`gpu::gpu_forward_fc_resident`]. Every reflection's Miller
    /// bin wraps into `[0, nu*nv*nw)`, so each index is in range.
    #[cfg(feature = "xtal-gpu")]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn gpu_forward_fc_resident(
        &self,
        params: &density::SplatParams<'_>,
        blur: f64,
    ) -> Vec<[f32; 2]> {
        let [nu, nv, nw] = self.grid_dims;
        let cell = &self.unit_cell;
        let recip = [
            cell.ar as f32,
            cell.br as f32,
            cell.cr as f32,
            cell.cos_alphar as f32,
            cell.cos_betar as f32,
            cell.cos_gammar as f32,
        ];
        let refl_idx: Vec<u32> = self
            .reflections
            .iter()
            .map(|r| {
                let u = wrap_miller_index(r.h, nu);
                let v = wrap_miller_index(r.k, nv);
                let w = wrap_miller_index(r.l, nw);
                ((u * nv + v) * nw + w) as u32
            })
            .collect();
        let voxel_volume = cell.volume / (nu * nv * nw) as f64;
        super::gpu::gpu_forward_fc_resident(
            params,
            [nu, nv, nw],
            blur,
            recip,
            &refl_idx,
            voxel_volume,
        )
    }

    /// Expand asymmetric-unit atoms to their full symmetry-plus-centering
    /// orbit, applying the same per-copy transform the reciprocal-space
    /// gradient gather uses (`sym.apply_to_frac` then centering translation).
    ///
    /// Returns per-copy fractional positions, form factors, B-factors, and
    /// occupancies, each `n_atoms * n_sym * n_cen` long and grouped by source
    /// atom. Positions may fall outside the unit cell; the splat wraps them.
    fn expand_orbit<'f>(
        &self,
        positions: &[[f64; 3]],
        form_factors: &[&'f FormFactor],
        b_factors: &[f64],
        occupancies: &[f64],
    ) -> OrbitSplatInputs<'f> {
        let den = f64::from(DEN);
        let mult = self.group_ops.sym_ops.len() * self.group_ops.cen_ops.len();
        let cap = positions.len() * mult;
        let mut pos = Vec::with_capacity(cap);
        let mut ffs = Vec::with_capacity(cap);
        let mut bs = Vec::with_capacity(cap);
        let mut occs = Vec::with_capacity(cap);

        for j in 0..positions.len() {
            for sym in &self.group_ops.sym_ops {
                let sp = sym.apply_to_frac(positions[j]);
                for cen in &self.group_ops.cen_ops {
                    pos.push([
                        sp[0] + f64::from(cen[0]) / den,
                        sp[1] + f64::from(cen[1]) / den,
                        sp[2] + f64::from(cen[2]) / den,
                    ]);
                    ffs.push(form_factors[j]);
                    bs.push(b_factors[j]);
                    occs.push(occupancies[j]);
                }
            }
        }
        (pos, ffs, bs, occs)
    }

    /// Estimate the minimum d-spacing from the reflection data.
    fn estimate_d_min(&self) -> f64 {
        let mut max_s2 = 0.0_f64;
        for r in &self.reflections {
            let s2 = self.unit_cell.d_star_sq(r.h, r.k, r.l);
            if s2 > max_s2 {
                max_s2 = s2;
            }
        }
        if max_s2 > 0.0 {
            1.0 / max_s2.sqrt()
        } else {
            2.0 // default fallback
        }
    }

    /// Extract per-reflection complex values from a full 3D grid.
    ///
    /// Maps each reflection's Miller indices to the grid and reads the complex
    /// value at that position. Indices that exceed the grid dimensions are
    /// treated as zero (missing coefficient).
    ///
    /// The raw forward FFT yields the discrete DFT sum over grid points; the
    /// physical structure factor is `F(h) = dV * DFT(rho)` with voxel volume
    /// `dV = V_cell / N`, `N = nu*nv*nw`. That factor is applied here so both
    /// Fc and Fmask leave this function on the physical amplitude scale.
    #[allow(
        clippy::cast_precision_loss,
        clippy::too_many_arguments,
        clippy::cast_possible_truncation
    )]
    fn extract_reflection_values(
        &self,
        fc_grid: &[[f32; 2]],
        fmask_grid: &[[f32; 2]],
        nu: usize,
        nv: usize,
        nw: usize,
    ) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
        let mut refl_fc = Vec::with_capacity(self.reflections.len());
        let mut refl_fmask = Vec::with_capacity(self.reflections.len());

        let voxel_volume = self.unit_cell.volume / (nu * nv * nw) as f64;
        let scale = |c: [f32; 2]| -> [f32; 2] {
            [
                (f64::from(c[0]) * voxel_volume) as f32,
                (f64::from(c[1]) * voxel_volume) as f32,
            ]
        };

        for r in &self.reflections {
            let u = wrap_miller_index(r.h, nu);
            let v = wrap_miller_index(r.k, nv);
            let w = wrap_miller_index(r.l, nw);

            let idx = (u * nv + v) * nw + w;
            if idx < fc_grid.len() {
                refl_fc.push(scale(fc_grid[idx]));
            } else {
                refl_fc.push([0.0, 0.0]);
            }
            if idx < fmask_grid.len() {
                refl_fmask.push(scale(fmask_grid[idx]));
            } else {
                refl_fmask.push([0.0, 0.0]);
            }
        }

        (refl_fc, refl_fmask)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::types::{Symop, UnitCell};
    use super::super::{space_group, SpaceGroup};
    use super::*;
    use crate::element::Element;

    /// Pearson correlation between two equal-length samples.
    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let (dx, dy) = (xi - mean_x, yi - mean_y);
            cov += dx * dy;
            vx += dx * dx;
            vy += dy * dy;
        }
        let denom = (vx * vy).sqrt();
        if denom < 1e-30 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Splat-full-orbit on a power-of-two grid must reproduce the reference
    /// ASU-splat + symmetrize-sum structure factors (on the smooth grid) to
    /// grid-sampling tolerance, for one real structure of the given space
    /// group. This validates dropping `symmetrize_sum` in favor of splatting
    /// the full orbit directly, and the power-of-two grid it enables.
    #[allow(
        clippy::print_stdout,
        clippy::print_stderr,
        clippy::cast_precision_loss,
        clippy::too_many_lines
    )]
    fn assert_orbit_matches_reference(pdb: &str, sg_number: u16) {
        let Some(case) = crate::testutil::refinement_from_cif_pair(pdb) else {
            // Fixture data absent in this checkout; nothing to validate.
            eprintln!("skipping {pdb}: fixture data not found");
            return;
        };
        let mut refinement = case.refinement;

        // Reference: ASU splat + symmetrize on the smooth grid the deposited
        // pipeline derives.
        let ref_fc = refinement
            .forward_fc(
                &case.positions,
                &case.elements,
                &case.b_factors,
                &case.occupancies,
            )
            .expect("reference forward Fc");

        // Power-of-two grid at the same d_min/3 sampling target.
        let mut d_min = f64::MAX;
        for r in &refinement.reflections {
            let s2 = refinement.unit_cell.d_star_sq(r.h, r.k, r.l);
            if s2 > 0.0 {
                d_min = d_min.min(1.0 / s2.sqrt());
            }
        }
        let cell = &refinement.unit_cell;
        let pow2 = crate::testutil::derive_grid_pow2(
            cell.a,
            cell.b,
            cell.c,
            d_min / 3.0,
            sg_number,
        );
        assert!(
            pow2.iter().all(|n| n.is_power_of_two()),
            "grid must be power-of-two, got {pow2:?}"
        );
        refinement.grid_dims = pow2;

        // New: splat the full orbit directly on the pow2 grid, no symmetrize.
        let orbit_fc = refinement
            .forward_fc_orbit(
                &case.positions,
                &case.elements,
                &case.b_factors,
                &case.occupancies,
            )
            .expect("orbit forward Fc");

        let ref_amp: Vec<f64> = ref_fc
            .iter()
            .map(|c| f64::from(c[0]).hypot(f64::from(c[1])))
            .collect();
        let orbit_amp: Vec<f64> = orbit_fc
            .iter()
            .map(|c| f64::from(c[0]).hypot(f64::from(c[1])))
            .collect();

        let corr = pearson(&ref_amp, &orbit_amp);

        // Max relative error over the structurally significant reflections
        // (amplitude above the mean); weak reflections have near-zero
        // denominators where sampling noise dominates the ratio.
        let mean_ref = ref_amp.iter().sum::<f64>() / ref_amp.len() as f64;
        let mut max_rel = 0.0_f64;
        for (&r, &o) in ref_amp.iter().zip(orbit_amp.iter()) {
            if r > mean_ref {
                max_rel = max_rel.max((r - o).abs() / r);
            }
        }

        println!(
            "{pdb} (SG{sg_number}): grid {pow2:?}, n_refl={}, \
             Pearson(|Fc|)={corr:.6}, max_rel(strong)={max_rel:.4}",
            ref_amp.len()
        );

        assert!(corr > 0.99, "{pdb}: |Fc| Pearson {corr:.6} below 0.99");
        assert!(
            max_rel < 0.20,
            "{pdb}: max relative |Fc| error {max_rel:.4} exceeds 0.20"
        );
    }

    /// Orthorhombic P2₁2₁2₁ (4 symops, no equal-uv constraint).
    #[test]
    fn orbit_pow2_matches_reference_p212121() {
        assert_orbit_matches_reference("1AKI", 19);
    }

    /// Tetragonal P4₃2₁2 (8 symops, requires nu == nv): exercises the highest
    /// orbit multiplicity and the equal-uv branch of the pow2 grid derivation.
    /// Its 256³ pow2 grid makes it too slow for the default run; invoke with
    /// `--ignored`.
    #[test]
    #[ignore = "256^3 grid: ~40s; run with --ignored"]
    fn orbit_pow2_matches_reference_p43212() {
        assert_orbit_matches_reference("1G6X", 96);
    }

    /// Smoke test: construct an XtalRefinement and compute a map.
    #[test]
    fn compute_map_smoke() {
        let cell = UnitCell::new(20.0, 20.0, 20.0, 90.0, 90.0, 90.0);
        let sg = space_group(1); // P1
        let sg = sg.unwrap_or_else(|| SpaceGroup {
            number: 1,
            hm: "P 1",
            ops: GroupOps {
                sym_ops: vec![Symop {
                    rot: [[24, 0, 0], [0, 24, 0], [0, 0, 24]],
                    tran: [0, 0, 0],
                }],
                cen_ops: vec![[0, 0, 0]],
            },
            crystal_system: CrystalSystem::Triclinic,
        });

        // Simple reflections.
        let mut reflections = Vec::new();
        for h in 1..=3_i32 {
            for k in 0..=2_i32 {
                for l in 0..=2_i32 {
                    reflections.push(Reflection {
                        h,
                        k,
                        l,
                        f_obs: 10.0,
                        sigma_f: 1.0,
                        free_flag: false,
                    });
                }
            }
        }

        let grid_dims = [16, 16, 16];
        let mut refinement =
            XtalRefinement::new(cell, sg, reflections, grid_dims);

        let positions = [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]];
        let elements = [Element::C, Element::N];
        let b_factors = [20.0, 25.0];
        let occupancies = [1.0, 1.0];

        let result = refinement.compute_map(
            &positions,
            &elements,
            &b_factors,
            &occupancies,
        );

        assert!(result.is_some(), "compute_map should succeed");
        let grid = result.expect("verified above");
        assert_eq!(grid.data.len(), 16 * 16 * 16);

        // Grid should have non-zero values.
        let has_nonzero = grid.data.iter().any(|&v| v.abs() > 1e-10);
        assert!(has_nonzero, "density grid should have non-zero values");
    }

    /// Verify that r_factors returns something reasonable.
    #[test]
    fn r_factors_smoke() {
        let cell = UnitCell::new(20.0, 20.0, 20.0, 90.0, 90.0, 90.0);
        let sg = space_group(1).expect("P1 should exist");

        let mut reflections = Vec::new();
        for h in 1..=2_i32 {
            for k in 0..=1_i32 {
                for l in 0..=1_i32 {
                    reflections.push(Reflection {
                        h,
                        k,
                        l,
                        f_obs: 10.0,
                        sigma_f: 1.0,
                        free_flag: false,
                    });
                }
            }
        }

        let grid_dims = [16, 16, 16];
        let mut refinement =
            XtalRefinement::new(cell, sg, reflections, grid_dims);

        let positions = [[0.5, 0.5, 0.5]];
        let elements = [Element::C];
        let b_factors = [20.0];
        let occupancies = [1.0];

        // Must compute map first to set scaling.
        let _ = refinement.compute_map(
            &positions,
            &elements,
            &b_factors,
            &occupancies,
        );

        let r = refinement.r_factors(
            &positions,
            &elements,
            &b_factors,
            &occupancies,
        );
        assert!(r.is_some(), "r_factors should return Some");
    }

    /// (0,0,0) and systematic absences are filtered out at construction.
    #[test]
    fn new_filters_origin_and_systematic_absences() {
        let cell = UnitCell::new(50.0, 60.0, 70.0, 90.0, 90.0, 90.0);
        // C 1 2 1 (SG 5): C-centering requires h+k even.
        let sg = space_group(5).expect("C 1 2 1 should exist");

        let reflections = vec![
            // (0,0,0); should be filtered (origin)
            Reflection {
                h: 0,
                k: 0,
                l: 0,
                f_obs: 999.0,
                sigma_f: 1.0,
                free_flag: false,
            },
            // (1,0,1); h+k = 1 (odd) → systematically absent in C lattice
            Reflection {
                h: 1,
                k: 0,
                l: 1,
                f_obs: 50.0,
                sigma_f: 1.0,
                free_flag: false,
            },
            // (2,0,1); h+k = 2 (even) → valid
            Reflection {
                h: 2,
                k: 0,
                l: 1,
                f_obs: 30.0,
                sigma_f: 1.0,
                free_flag: false,
            },
            // (1,1,1); h+k = 2 (even) → valid
            Reflection {
                h: 1,
                k: 1,
                l: 1,
                f_obs: 20.0,
                sigma_f: 1.0,
                free_flag: false,
            },
        ];

        let grid_dims = [16, 16, 16];
        let refinement = XtalRefinement::new(cell, sg, reflections, grid_dims);

        // Should have 2 reflections: (2,0,1) and (1,1,1)
        assert_eq!(
            refinement.reflections.len(),
            2,
            "expected 2 reflections after filtering, got {}",
            refinement.reflections.len()
        );
        // Verify the survivors
        assert_eq!(refinement.reflections[0].h, 2);
        assert_eq!(refinement.reflections[1].h, 1);
        assert_eq!(refinement.reflections[1].k, 1);
    }

    /// A tiny synthetic P1 fixture, millisecond-class: a handful of atoms and a
    /// few hundred reflections, fine enough for the FFT-map gradient to track
    /// the direct target closely.
    fn tiny_synthetic() -> crate::testutil::SyntheticCase {
        crate::testutil::synthetic_refinement(
            8,
            1,
            [26.0, 26.0, 26.0, 90.0, 90.0, 90.0],
            1.0,
            20.0,
        )
    }

    /// The analytic FFT-map B-factor gradient matches a central finite
    /// difference of the maximum-likelihood target. Evaluated at a flat B
    /// offset from the generating B so every atom carries a solidly
    /// non-zero gradient.
    #[test]
    fn gradient_matches_finite_difference() {
        let case = tiny_synthetic();
        let mut refinement = case.refinement;
        let positions = case.positions;
        let elements = case.elements;
        let occupancies = case.occupancies;
        let b: Vec<f64> = vec![35.0; positions.len()];

        // sigma-A is populated here and frozen for every target evaluation.
        let _map = refinement
            .compute_map(&positions, &elements, &b, &occupancies)
            .expect("compute_map");

        let g = refinement
            .b_factor_gradients(&positions, &elements, &b, &occupancies)
            .expect("analytic gradient");
        assert_eq!(g.len(), b.len(), "one gradient per atom");

        let fd_at = |idx: usize, delta: f64| -> f64 {
            let mut b_plus = b.clone();
            let mut b_minus = b.clone();
            b_plus[idx] += delta;
            b_minus[idx] -= delta;
            let t_plus = refinement
                .maximum_likelihood_target_for_b(
                    &positions,
                    &elements,
                    &b_plus,
                    &occupancies,
                )
                .expect("target(+delta)");
            let t_minus = refinement
                .maximum_likelihood_target_for_b(
                    &positions,
                    &elements,
                    &b_minus,
                    &occupancies,
                )
                .expect("target(-delta)");
            (t_plus - t_minus) / (2.0 * delta)
        };

        for (i, &gi) in g.iter().enumerate() {
            let fd = fd_at(i, 0.05);
            assert!(
                gi * fd > 0.0,
                "atom {i}: gradient sign mismatch (analytic {gi:e}, fd {fd:e})"
            );
            let ratio = (gi / fd).abs();
            assert!(
                (0.5..=2.0).contains(&ratio),
                "atom {i}: |analytic/fd| = {ratio} outside [0.5, 2.0] \
                 (analytic {gi:e}, fd {fd:e})"
            );
        }
    }

    /// The FFT-map gradient agrees with the direct-sum oracle: high Pearson
    /// correlation and small max per-atom relative discrepancy.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn fft_gradient_matches_direct_sum() {
        let case = tiny_synthetic();
        let mut refinement = case.refinement;
        let positions = case.positions;
        let elements = case.elements;
        let occupancies = case.occupancies;
        let b: Vec<f64> = vec![30.0; positions.len()];

        let _map = refinement
            .compute_map(&positions, &elements, &b, &occupancies)
            .expect("compute_map");

        let g_fft = refinement
            .b_factor_gradients(&positions, &elements, &b, &occupancies)
            .expect("fft gradient");
        let g_dir = refinement
            .b_factor_gradients_direct(&positions, &elements, &b, &occupancies)
            .expect("direct gradient");
        assert_eq!(g_fft.len(), g_dir.len());

        let max_abs_dir = g_dir.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let denom = max_abs_dir.max(1e-30);
        let max_rel = g_fft
            .iter()
            .zip(g_dir.iter())
            .map(|(x, y)| (x - y).abs() / denom)
            .fold(0.0_f64, f64::max);

        let n = g_fft.len() as f64;
        let mean_f = g_fft.iter().sum::<f64>() / n;
        let mean_d = g_dir.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut var_f = 0.0;
        let mut var_d = 0.0;
        for (x, y) in g_fft.iter().zip(g_dir.iter()) {
            let df = x - mean_f;
            let dd = y - mean_d;
            cov += df * dd;
            var_f += df * df;
            var_d += dd * dd;
        }
        let corr = cov / (var_f * var_d).sqrt();

        assert!(corr > 0.999, "correlation {corr} below 0.999");
        assert!(
            max_rel < 0.05,
            "max relative discrepancy {max_rel} exceeds 5%"
        );
    }

    /// Measure the B-factor gradient computed through the **f32**-internal FFT
    /// against the f64 direct-sum oracle and against a central finite
    /// difference of the (f64) maximum-likelihood target. The GPU path is
    /// f32-only, so this quantifies whether the refinement gradient survives
    /// f32 FFT precision. Prints Pearson correlation and max relative error.
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::print_stdout,
        clippy::too_many_lines
    )]
    fn f32_fft_gradient_precision_report() {
        let case = tiny_synthetic();
        let mut refinement = case.refinement;
        let positions = case.positions;
        let elements = case.elements;
        let occupancies = case.occupancies;
        let b: Vec<f64> = vec![30.0; positions.len()];

        // sigma-A is populated at the default f64 precision and frozen for
        // every evaluation below.
        let _map = refinement
            .compute_map(&positions, &elements, &b, &occupancies)
            .expect("compute_map");

        // f64 references, computed while the default f64 FFT is selected.
        let g_oracle = refinement
            .b_factor_gradients_direct(&positions, &elements, &b, &occupancies)
            .expect("direct-sum oracle");

        let fd_at = |idx: usize, delta: f64| -> f64 {
            let mut b_plus = b.clone();
            let mut b_minus = b.clone();
            b_plus[idx] += delta;
            b_minus[idx] -= delta;
            let t_plus = refinement
                .maximum_likelihood_target_for_b(
                    &positions,
                    &elements,
                    &b_plus,
                    &occupancies,
                )
                .expect("target(+delta)");
            let t_minus = refinement
                .maximum_likelihood_target_for_b(
                    &positions,
                    &elements,
                    &b_minus,
                    &occupancies,
                )
                .expect("target(-delta)");
            (t_plus - t_minus) / (2.0 * delta)
        };
        let g_fd: Vec<f64> = (0..b.len()).map(|i| fd_at(i, 0.05)).collect();

        // Switch to the GPU-equivalent f32 FFT and recompute the analytic
        // FFT-map gradient through it.
        refinement.fft_precision = fft_cpu::FftPrecision::F32;
        let g_f32 = refinement
            .b_factor_gradients(&positions, &elements, &b, &occupancies)
            .expect("f32 gradient");
        assert_eq!(g_f32.len(), b.len(), "one gradient per atom");

        // Pearson correlation and max relative error of `x` against reference
        // `y`, the max-rel normalized by the reference's largest magnitude
        // (same convention as `fft_gradient_matches_direct_sum`).
        let stats = |x: &[f64], y: &[f64]| -> (f64, f64) {
            let n = x.len() as f64;
            let mean_x = x.iter().sum::<f64>() / n;
            let mean_y = y.iter().sum::<f64>() / n;
            let mut cov = 0.0;
            let mut var_x = 0.0;
            let mut var_y = 0.0;
            for (a, c) in x.iter().zip(y.iter()) {
                let dx = a - mean_x;
                let dy = c - mean_y;
                cov += dx * dy;
                var_x += dx * dx;
                var_y += dy * dy;
            }
            let corr = cov / (var_x * var_y).sqrt();

            let denom =
                y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1e-30);
            let max_rel = x
                .iter()
                .zip(y.iter())
                .map(|(a, c)| (a - c).abs() / denom)
                .fold(0.0_f64, f64::max);
            (corr, max_rel)
        };

        let (corr_oracle, maxrel_oracle) = stats(&g_f32, &g_oracle);
        let (corr_fd, maxrel_fd) = stats(&g_f32, &g_fd);

        println!(
            "f32-FFT gradient precision (atoms={}):\n  vs direct-sum oracle: \
             Pearson={corr_oracle:.6}, max_rel={maxrel_oracle:.6}\n  vs \
             finite difference: Pearson={corr_fd:.6}, max_rel={maxrel_fd:.6}",
            b.len()
        );

        assert!(
            g_f32.iter().all(|v| v.is_finite()),
            "f32 gradient produced a non-finite entry"
        );
        assert!(
            corr_oracle > 0.99,
            "f32 vs oracle correlation {corr_oracle} below 0.99"
        );
        assert!(
            corr_fd > 0.99,
            "f32 vs finite-difference correlation {corr_fd} below 0.99"
        );
    }

    /// The GPU real-space gradient gather agrees with the CPU gather and with
    /// the f64 direct-sum oracle. Runs on-device (Metal on macOS) for a
    /// P1 synthetic (orbit size 1) and, when the fixtures are present, a real
    /// structure whose space group exercises the symmetry-orbit reduction.
    #[cfg(feature = "xtal-gpu")]
    #[test]
    #[allow(
        clippy::print_stdout,
        clippy::cast_precision_loss,
        clippy::too_many_lines
    )]
    fn gpu_gather_gradient_matches_cpu_and_oracle() {
        // Pearson correlation and max relative error (normalized by the
        // reference's largest magnitude) of `x` against reference `y`.
        fn stats(x: &[f64], y: &[f64]) -> (f64, f64) {
            let n = x.len() as f64;
            let mean_x = x.iter().sum::<f64>() / n;
            let mean_y = y.iter().sum::<f64>() / n;
            let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
            for (a, c) in x.iter().zip(y.iter()) {
                let dx = a - mean_x;
                let dy = c - mean_y;
                cov += dx * dy;
                var_x += dx * dx;
                var_y += dy * dy;
            }
            let corr = cov / (var_x * var_y).sqrt();
            let denom =
                y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1e-30);
            let max_rel = x
                .iter()
                .zip(y.iter())
                .map(|(a, c)| (a - c).abs() / denom)
                .fold(0.0_f64, f64::max);
            (corr, max_rel)
        }

        let check = |refinement: &mut XtalRefinement,
                     positions: &[[f64; 3]],
                     elements: &[Element],
                     b: &[f64],
                     occ: &[f64],
                     label: &str| {
            let _map = refinement
                .compute_map(positions, elements, b, occ)
                .expect("compute_map populates sigma-A");
            let fc = refinement
                .forward_fc(positions, elements, b, occ)
                .expect("forward_fc");

            refinement.stencil_backend = StencilBackend::Cpu;
            let g_cpu = refinement
                .b_factor_gradients_from_fc(&fc, positions, elements, b, occ)
                .expect("cpu gather");
            let g_oracle = refinement
                .b_factor_gradients_direct(positions, elements, b, occ)
                .expect("direct-sum oracle");

            refinement.stencil_backend = StencilBackend::Gpu;
            let g_gpu = refinement
                .b_factor_gradients_from_fc(&fc, positions, elements, b, occ)
                .expect("gpu gather");
            refinement.stencil_backend = StencilBackend::Cpu;

            assert_eq!(g_gpu.len(), g_cpu.len(), "one gradient per atom");
            assert!(
                g_gpu.iter().all(|v| v.is_finite()),
                "gpu gather produced a non-finite entry"
            );

            let (corr_cpu, maxrel_cpu) = stats(&g_gpu, &g_cpu);
            let (corr_oracle, maxrel_oracle) = stats(&g_gpu, &g_oracle);
            println!(
                "gpu gather [{label}] (atoms={}):\n  vs cpu gather: \
                 Pearson={corr_cpu:.6}, max_rel={maxrel_cpu:.6}\n  vs \
                 direct-sum oracle: Pearson={corr_oracle:.6}, \
                 max_rel={maxrel_oracle:.6}",
                g_gpu.len()
            );
            assert!(
                corr_cpu > 0.99,
                "gpu vs cpu gather correlation {corr_cpu} below 0.99"
            );
            assert!(
                corr_oracle > 0.99,
                "gpu vs oracle correlation {corr_oracle} below 0.99"
            );
        };

        let case = tiny_synthetic();
        let mut refinement = case.refinement;
        let b: Vec<f64> = vec![30.0; case.positions.len()];
        check(
            &mut refinement,
            &case.positions,
            &case.elements,
            &b,
            &case.occupancies,
            "synthetic-P1",
        );

        if let Some(rc) = crate::testutil::refinement_from_cif_pair("1AKI") {
            let mut refinement = rc.refinement;
            check(
                &mut refinement,
                &rc.positions,
                &rc.elements,
                &rc.b_factors,
                &rc.occupancies,
                "1AKI",
            );
        } else {
            println!("1AKI fixtures missing; skipping real-structure case");
        }
    }

    /// End-to-end GPU-FFT pipeline on a real structure: the splat-full-orbit
    /// forward model on a pow2 grid with the device FFT swapped in for the CPU
    /// FFT, both forward (structure factors) and inverse (B-factor gradient).
    ///
    /// The reference is the same splat-full-orbit model with the CPU FFT, which
    /// isolates the FFT swap from the forward-model change (the orbit/pow2
    /// model itself is validated against the deposited symmetrized reference
    /// elsewhere). The gradient additionally clears the f64 direct-sum oracle.
    /// Runs on-device (Metal on macOS).
    #[cfg(feature = "xtal-gpu")]
    #[test]
    #[allow(
        clippy::print_stdout,
        clippy::cast_precision_loss,
        clippy::too_many_lines
    )]
    fn gpu_fft_pipeline_matches_cpu_and_oracle_1aki() {
        let Some(rc) = crate::testutil::refinement_from_cif_pair("1AKI") else {
            println!("1AKI fixtures missing; skipping GPU-FFT pipeline test");
            return;
        };
        let mut refinement = rc.refinement;
        let (pos, el) = (&rc.positions, &rc.elements);
        let (b, occ) = (&rc.b_factors, &rc.occupancies);

        // Populate sigma-A/scaling on the deposited grid (SymmetrizeGrid needs
        // its grid-factor divisibility), and take the f64 direct-sum oracle
        // there — both are grid-robust reciprocal-space quantities.
        let _map = refinement
            .compute_map(pos, el, b, occ)
            .expect("compute_map populates sigma-A");
        let g_oracle = refinement
            .b_factor_gradients_direct(pos, el, b, occ)
            .expect("direct-sum oracle");

        // Switch to the pow2 orbit grid the GPU FFT accepts.
        let cell = &refinement.unit_cell;
        let mut d_min = f64::MAX;
        for r in &refinement.reflections {
            let s2 = cell.d_star_sq(r.h, r.k, r.l);
            if s2 > 0.0 {
                d_min = d_min.min(1.0 / s2.sqrt());
            }
        }
        let pow2 = crate::testutil::derive_grid_pow2(
            cell.a,
            cell.b,
            cell.c,
            d_min / 3.0,
            19,
        );
        assert!(
            pow2.iter().all(|n| n.is_power_of_two()),
            "grid must be power-of-two, got {pow2:?}"
        );
        refinement.grid_dims = pow2;

        // Reference: CPU splat + CPU FFT on the orbit/pow2 model.
        refinement.stencil_backend = StencilBackend::Cpu;
        let ref_fc = refinement
            .forward_fc_orbit(pos, el, b, occ)
            .expect("cpu-fft orbit forward");
        let g_ref = refinement
            .b_factor_gradients_from_fc(&ref_fc, pos, el, b, occ)
            .expect("cpu-fft inverse gradient");

        // GPU-FFT path: GPU splat + GPU FFT, forward and inverse.
        refinement.stencil_backend = StencilBackend::Gpu;
        let gpu_fc = refinement
            .forward_fc_orbit(pos, el, b, occ)
            .expect("gpu-fft orbit forward");
        let g_gpu = refinement
            .b_factor_gradients_from_fc(&gpu_fc, pos, el, b, occ)
            .expect("gpu-fft inverse gradient");
        refinement.stencil_backend = StencilBackend::Cpu;

        assert!(
            gpu_fc.iter().all(|c| c[0].is_finite() && c[1].is_finite()),
            "gpu-fft Fc produced a non-finite entry"
        );
        assert!(
            g_gpu.iter().all(|v| v.is_finite()),
            "gpu-fft gradient produced a non-finite entry"
        );

        // Structure-factor agreement (GPU FFT vs CPU FFT, same model).
        let amp = |fc: &[[f32; 2]]| -> Vec<f64> {
            fc.iter()
                .map(|c| f64::from(c[0]).hypot(f64::from(c[1])))
                .collect()
        };
        let (ref_amp, gpu_amp) = (amp(&ref_fc), amp(&gpu_fc));
        let corr_fc = pearson(&gpu_amp, &ref_amp);
        let mean_ref = ref_amp.iter().sum::<f64>() / ref_amp.len() as f64;
        let mut maxrel_fc = 0.0_f64;
        for (&g, &r) in gpu_amp.iter().zip(&ref_amp) {
            if r > mean_ref {
                maxrel_fc = maxrel_fc.max((g - r).abs() / r);
            }
        }

        // Gradient agreement: max relative error normalized by the reference's
        // largest magnitude, the same convention the gather test uses.
        let corr = |x: &[f64], y: &[f64]| pearson(x, y);
        let corr_grad_ref = corr(&g_gpu, &g_ref);
        let corr_grad_oracle = corr(&g_gpu, &g_oracle);
        let denom = g_oracle
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-30);
        let maxrel_grad = g_gpu
            .iter()
            .zip(&g_oracle)
            .map(|(a, c)| (a - c).abs() / denom)
            .fold(0.0_f64, f64::max);

        println!(
            "gpu-fft pipeline [1AKI]: grid {pow2:?}, n_refl={}\n  |Fc| \
             GPU-FFT vs CPU-FFT: Pearson={corr_fc:.6}, \
             max_rel(strong)={maxrel_fc:.3e}\n  gradient GPU-FFT vs CPU-FFT: \
             Pearson={corr_grad_ref:.6}\n  gradient GPU-FFT vs direct-sum \
             oracle: Pearson={corr_grad_oracle:.6}, max_rel={maxrel_grad:.3e}",
            ref_amp.len()
        );

        assert!(
            corr_fc > 0.999,
            "|Fc| GPU-FFT vs CPU-FFT Pearson {corr_fc:.6} below 0.999"
        );
        assert!(
            corr_grad_oracle > 0.99,
            "gradient GPU-FFT vs oracle Pearson {corr_grad_oracle:.6} below \
             0.99"
        );
    }

    /// `forward_fc` is deterministic and scales with occupancy: doubling every
    /// occupancy doubles each complex structure factor.
    #[test]
    fn forward_fc_is_deterministic_and_scales_with_occupancy() {
        let case = tiny_synthetic();
        let refinement = case.refinement;
        let positions = case.positions;
        let elements = case.elements;
        let b = case.b_true;
        let occ = case.occupancies;

        let fc1 = refinement
            .forward_fc(&positions, &elements, &b, &occ)
            .expect("forward_fc");
        let fc2 = refinement
            .forward_fc(&positions, &elements, &b, &occ)
            .expect("forward_fc");
        assert_eq!(fc1, fc2, "forward_fc should be deterministic");

        let occ2: Vec<f64> = occ.iter().map(|o| o * 2.0).collect();
        let fc_double = refinement
            .forward_fc(&positions, &elements, &b, &occ2)
            .expect("forward_fc");
        for (x, d) in fc1.iter().zip(fc_double.iter()) {
            let amp = f64::from(x[0]).hypot(f64::from(x[1]));
            let amp_d = f64::from(d[0]).hypot(f64::from(d[1]));
            if amp > 1e-3 {
                let ratio = amp_d / amp;
                assert!(
                    (ratio - 2.0).abs() < 1e-3,
                    "doubling occupancy should double |Fc|, ratio {ratio}"
                );
            }
        }
    }

    /// The Fc-driven target and gradient smoke-run: both are finite and the
    /// gradient has one entry per atom.
    #[test]
    fn target_and_gradient_from_fc_smoke() {
        let case = tiny_synthetic();
        let mut refinement = case.refinement;
        let positions = case.positions;
        let elements = case.elements;
        let b = case.b_true;
        let occ = case.occupancies;

        let _map = refinement
            .compute_map(&positions, &elements, &b, &occ)
            .expect("compute_map");
        let fc = refinement
            .forward_fc(&positions, &elements, &b, &occ)
            .expect("forward_fc");

        let target = refinement
            .maximum_likelihood_target_from_fc(&fc)
            .expect("target from fc");
        assert!(target.is_finite(), "target should be finite, got {target}");

        let grad = refinement
            .b_factor_gradients_from_fc(&fc, &positions, &elements, &b, &occ)
            .expect("gradient from fc");
        assert_eq!(grad.len(), b.len(), "one gradient per atom");
        assert!(
            grad.iter().all(|g| g.is_finite()),
            "all gradient entries should be finite"
        );
    }

    /// R-factors on the synthetic fixture are on the physical scale: finite,
    /// bounded in [0, 1], and small at the generating parameters.
    #[test]
    fn r_factors_small_at_truth_on_synthetic() {
        let case = tiny_synthetic();
        let mut refinement = case.refinement;
        let positions = case.positions;
        let elements = case.elements;
        let b = case.b_true;
        let occ = case.occupancies;

        let _map = refinement
            .compute_map(&positions, &elements, &b, &occ)
            .expect("compute_map");
        let (r_work, r_free) = refinement
            .r_factors(&positions, &elements, &b, &occ)
            .expect("r_factors");

        assert!(r_work.is_finite() && (0.0..=1.0).contains(&r_work));
        assert!(r_free.is_finite() && (0.0..=1.0).contains(&r_free));
        assert!(
            r_work < 0.5,
            "R-work should be small at truth, got {r_work}"
        );
    }
}
