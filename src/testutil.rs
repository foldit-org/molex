//! Shared crystallographic refinement fixtures for tests and benchmarks.
//!
//! Two builders back every xtal test and bench:
//! [`crate::testutil::refinement_from_cif_pair`] loads a deposited structure
//! plus its structure factors from `tests/data/` into a ready-to-refine
//! [`crate::xtal::XtalRefinement`], and
//! [`crate::testutil::synthetic_refinement`] constructs a self-consistent cell
//! from scratch with no file I/O and no RNG, placing atoms deterministically
//! and setting each `f_obs` from the forward model so the data fits at truth.
//!
//! Available under `#[cfg(test)]` for in-crate unit tests and behind the
//! `testutil` feature for the external bench and integration-test crates, which
//! see only the `pub` surface.

// These are test-only fixtures: `expect`/`panic` on genuinely-impossible
// construction failures keeps the builder signatures free of `Option`, and the
// numeric casts convert between grid extents and cell dimensions.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::manual_is_multiple_of,
    clippy::too_long_first_doc_paragraph,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use crate::adapters::cif::{parse, AtomSite, CoordinateData, ReflectionData};
use crate::element::Element;
use crate::xtal::{
    grid_factors, reflections_from_cif, requires_equal_uv, round_up_to_pow2,
    round_up_to_smooth, space_group, ForwardSymmetry, Reflection,
    StencilBackend, UnitCell, XtalRefinement,
};

/// The space groups the xtal module supports, for name-to-number lookup.
const SUPPORTED_SG: &[u16] = &[1, 4, 5, 19, 23, 92, 96, 152, 154, 178];

/// A refinement built from a deposited structure and its structure factors.
pub struct RealCase {
    /// The refinement pipeline, grid and reflections already set up.
    pub refinement: XtalRefinement,
    /// International Tables number of the structure's space group.
    pub sg_number: u16,
    /// Fractional coordinates of the non-hydrogen atoms.
    pub positions: Vec<[f64; 3]>,
    /// Element per atom, parallel to `positions`.
    pub elements: Vec<Element>,
    /// Deposited isotropic B-factor per atom.
    pub b_factors: Vec<f64>,
    /// Site occupancy per atom.
    pub occupancies: Vec<f64>,
}

/// A refinement built from scratch with self-consistent synthetic data.
pub struct SyntheticCase {
    /// The refinement pipeline, with `f_obs` set from the forward model at
    /// `b_true` so the working set fits at truth.
    pub refinement: XtalRefinement,
    /// Fractional coordinates of the placed atoms.
    pub positions: Vec<[f64; 3]>,
    /// Element per atom (all carbon), parallel to `positions`.
    pub elements: Vec<Element>,
    /// The B-factor every atom was generated at (one entry per atom).
    pub b_true: Vec<f64>,
    /// Site occupancy per atom (all unit).
    pub occupancies: Vec<f64>,
    /// The reflection set the refinement holds after construction.
    pub reflections: Vec<Reflection>,
}

/// A synthetic refinement with sigma-A already populated, ready for a single
/// [`gradient_once`] evaluation.
pub struct PreparedCase {
    /// The refinement pipeline, with `sigma_a` set by a `compute_map` run at
    /// `b_factors`.
    pub refinement: XtalRefinement,
    /// Fractional coordinates of the placed atoms.
    pub positions: Vec<[f64; 3]>,
    /// Element per atom, parallel to `positions`.
    pub elements: Vec<Element>,
    /// The isotropic B the map and gradient are evaluated at (one per atom).
    pub b_factors: Vec<f64>,
    /// Site occupancy per atom (all unit).
    pub occupancies: Vec<f64>,
    /// Grid dimensions `[nu, nv, nw]` the refinement holds.
    pub grid_dims: [usize; 3],
}

/// Map a Hermann-Mauguin space-group name (as it appears in a coordinate CIF)
/// to its International Tables number, over the supported groups.
fn sg_number_from_name(name: &str) -> Option<u16> {
    let normalized: Vec<&str> = name.split_whitespace().collect();
    SUPPORTED_SG.iter().copied().find(|&n| {
        space_group(n).is_some_and(|sg| {
            sg.hm.split_whitespace().collect::<Vec<_>>() == normalized
        })
    })
}

/// FNV-1a hash of a PDB code, used to seed the free-flag partition
/// deterministically per structure.
fn seed_for(pdb: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for b in pdb.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Raise both axes to `max(nu, nv)` when the space group requires
/// `nu == nv`; otherwise return them unchanged.
fn enforce_equal_uv(nu: usize, nv: usize, sg_number: u16) -> (usize, usize) {
    if requires_equal_uv(sg_number) {
        let m = nu.max(nv);
        (m, m)
    } else {
        (nu, nv)
    }
}

/// Derive grid dimensions `[nu, nv, nw]` from cell edges and a target
/// real-space sampling interval, enforcing the space group's divisibility
/// factors and any equal-`nu`-`nv` constraint.
fn derive_grid(
    a: f64,
    b: f64,
    c: f64,
    spacing: f64,
    sg_number: u16,
) -> [usize; 3] {
    let gf = grid_factors(sg_number).expect("supported space group");
    let mut nu = round_up_to_smooth((a / spacing).ceil());
    let mut nv = round_up_to_smooth((b / spacing).ceil());
    let mut nw = round_up_to_smooth((c / spacing).ceil());
    while nu % gf[0] != 0 {
        nu = round_up_to_smooth((nu + 1) as f64);
    }
    while nv % gf[1] != 0 {
        nv = round_up_to_smooth((nv + 1) as f64);
    }
    while nw % gf[2] != 0 {
        nw = round_up_to_smooth((nw + 1) as f64);
    }
    let (nu, nv) = enforce_equal_uv(nu, nv, sg_number);
    [nu, nv, nw]
}

/// Derive power-of-two grid dimensions `[nu, nv, nw]` from cell edges and a
/// target real-space sampling interval.
///
/// Each axis keeps `spacing <= a/n` by rounding `a/spacing` UP to a power of
/// two, so every reflection resolved at the target spacing stays inside
/// Nyquist. Unlike [`derive_grid`] this imposes no space-group grid-factor
/// divisibility (the orbit-splat forward path needs none); it still honors any
/// equal-`nu`-`nv` constraint.
#[must_use]
pub fn derive_grid_pow2(
    a: f64,
    b: f64,
    c: f64,
    spacing: f64,
    sg_number: u16,
) -> [usize; 3] {
    let nu = round_up_to_pow2((a / spacing).ceil());
    let nv = round_up_to_pow2((b / spacing).ceil());
    let nw = round_up_to_pow2((c / spacing).ceil());
    let (nu, nv) = enforce_equal_uv(nu, nv, sg_number);
    [nu, nv, nw]
}

/// Load `<pdb>.cif` + `<pdb>-sf.cif` from `tests/data/` into a refinement plus
/// its deposited atom arrays.
///
/// Returns `None` if either file is missing, the CIF lacks a cell or a
/// recognized space group, or no reflection carries a measured amplitude. The
/// grid is derived from the reflection resolution exactly as the deposited-data
/// pipeline does; the free-set partition uses a fixed per-PDB seed so the
/// reflection set is reproducible across runs.
#[must_use]
pub fn refinement_from_cif_pair(pdb: &str) -> Option<RealCase> {
    let data_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let coord_path = data_dir.join(format!("{pdb}.cif"));
    let sf_path = data_dir.join(format!("{pdb}-sf.cif"));

    // ── Coordinates ─────────────────────────────────────────────────────
    let coord_str = std::fs::read_to_string(&coord_path).ok()?;
    let coord_doc = parse(&coord_str).ok()?;
    let coord_block = coord_doc.blocks.first()?;
    let coord_data = CoordinateData::try_from(coord_block).ok()?;
    let cell = coord_data.cell.as_ref()?;
    let sg_number = sg_number_from_name(coord_data.spacegroup.as_deref()?)?;

    let atoms: Vec<&AtomSite> = coord_data
        .atoms
        .iter()
        .filter(|&a| element_of(a) != Element::H)
        .collect();

    // ── Reflections ─────────────────────────────────────────────────────
    let sf_str = std::fs::read_to_string(&sf_path).ok()?;
    let sf_doc = parse(&sf_str).ok()?;
    let sf_block = sf_doc.blocks.first()?;
    let mut refl_data = ReflectionData::try_from(sf_block).ok()?;
    if !refl_data.reflections.iter().any(|r| r.free_flag) {
        refl_data.free_flags_from_file = false;
    }
    let reflections = reflections_from_cif(&refl_data, 0.05, seed_for(pdb));
    if reflections.is_empty() {
        return None;
    }

    // ── Grid ────────────────────────────────────────────────────────────
    let sg = space_group(sg_number)?;
    let unit_cell = UnitCell::new(
        cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma,
    );

    let mut d_min = f64::MAX;
    for r in &reflections {
        let s2 = unit_cell.d_star_sq(r.h, r.k, r.l);
        if s2 > 0.0 {
            d_min = d_min.min(1.0 / s2.sqrt());
        }
    }
    let grid = derive_grid(cell.a, cell.b, cell.c, d_min / 3.0, sg_number);

    let refinement = XtalRefinement::new(unit_cell, sg, reflections, grid);

    let positions = atoms
        .iter()
        .map(|a| refinement.unit_cell.fractionalize([a.x, a.y, a.z]))
        .collect();
    let elements = atoms.iter().map(|&a| element_of(a)).collect();
    let b_factors = atoms.iter().map(|a| a.b_factor).collect();
    let occupancies = atoms.iter().map(|a| a.occupancy).collect();

    Some(RealCase {
        refinement,
        sg_number,
        positions,
        elements,
        b_factors,
        occupancies,
    })
}

/// Resolve an atom's element, falling back to the atom name when the CIF omits
/// an explicit element symbol.
fn element_of(a: &AtomSite) -> Element {
    if a.element.is_empty() {
        Element::from_atom_name(&a.label)
    } else {
        Element::from_symbol(&a.element)
    }
}

/// Build a self-consistent synthetic refinement with no file I/O and no RNG.
///
/// The cell is `cell` (`[a, b, c, alpha, beta, gamma]`) in space group
/// `sg_number`, sampled on a grid at `grid_spacing_target`. `n_atoms` carbon
/// atoms are placed deterministically (a golden-ratio low-discrepancy sequence
/// keyed on atom index). A reflection list is generated to the resolution the
/// grid spacing implies, then `forward_fc` at `b_true` sets each reflection's
/// `f_obs` to the model amplitude, so the working set fits at truth.
///
/// `n_atoms` and the cell dimensions are the two independent cost knobs: hold
/// the cell fixed and grow `n_atoms` to scale the density traversal, or hold
/// `n_atoms` fixed and grow the cell (at fixed spacing) to scale the FFT.
#[must_use]
pub fn synthetic_refinement(
    n_atoms: usize,
    sg_number: i32,
    cell: [f64; 6],
    grid_spacing_target: f64,
    b_true: f64,
) -> SyntheticCase {
    let grid = derive_grid(
        cell[0],
        cell[1],
        cell[2],
        grid_spacing_target,
        sg_number as u16,
    );
    build_synthetic(
        n_atoms,
        sg_number,
        cell,
        grid_spacing_target,
        b_true,
        grid,
        ForwardSymmetry::SymmetrizeGrid,
    )
}

/// Build a synthetic refinement on an explicit `grid` under `symmetry`, sizing
/// the reflection set from `grid_spacing_target` and setting each `f_obs` from
/// the forward model at `b_true` so the working set fits at truth. Shared by
/// the smooth-grid [`synthetic_refinement`] and the power-of-two orbit builder.
#[must_use]
#[allow(clippy::too_many_arguments)]
fn build_synthetic(
    n_atoms: usize,
    sg_number: i32,
    cell: [f64; 6],
    grid_spacing_target: f64,
    b_true: f64,
    grid: [usize; 3],
    symmetry: ForwardSymmetry,
) -> SyntheticCase {
    let sg_u16 = sg_number as u16;
    let unit_cell =
        UnitCell::new(cell[0], cell[1], cell[2], cell[3], cell[4], cell[5]);
    let sg = space_group(sg_u16).expect("supported space group");

    // Grid is ~3x finer than the resolution it must represent, mirroring the
    // deposited-data pipeline's `grid_spacing = d_min / 3` relation.
    let d_min = 3.0 * grid_spacing_target;
    let s_max_sq = 1.0 / (d_min * d_min);

    // Generate a reflection hemisphere out to `d_min`.
    let hmax = (cell[0] / d_min).ceil() as i32 + 1;
    let kmax = (cell[1] / d_min).ceil() as i32 + 1;
    let lmax = (cell[2] / d_min).ceil() as i32 + 1;
    let mut reflections = Vec::new();
    let mut idx = 0usize;
    for h in 0..=hmax {
        for k in (if h == 0 { 0 } else { -kmax })..=kmax {
            let l_lo = if h == 0 && k == 0 { 1 } else { -lmax };
            for l in l_lo..=lmax {
                let s2 = unit_cell.d_star_sq(h, k, l);
                if s2 > 0.0 && s2 <= s_max_sq {
                    // ~5% deterministic free set.
                    let free_flag = idx % 20 == 0;
                    reflections.push(Reflection {
                        h,
                        k,
                        l,
                        f_obs: 0.0,
                        sigma_f: 1.0,
                        free_flag,
                    });
                    idx += 1;
                }
            }
        }
    }

    let positions: Vec<[f64; 3]> =
        (0..n_atoms).map(deterministic_frac_position).collect();
    let elements = vec![Element::C; n_atoms];
    let b = vec![b_true; n_atoms];
    let occupancies = vec![1.0; n_atoms];

    let mut refinement = XtalRefinement::new(unit_cell, sg, reflections, grid);
    refinement.forward_symmetry = symmetry;

    // Self-consistency: set each f_obs to the forward-model amplitude at
    // b_true, so R -> 0 at the generating parameters.
    let fc = refinement
        .forward_fc(&positions, &elements, &b, &occupancies)
        .expect("forward Fc for synthetic model");
    for (r, c) in refinement.reflections.iter_mut().zip(fc.iter()) {
        r.f_obs = f64::from(c[0]).hypot(f64::from(c[1])) as f32;
    }

    let reflections = refinement.reflections.clone();

    SyntheticCase {
        refinement,
        positions,
        elements,
        b_true: b,
        occupancies,
        reflections,
    }
}

/// Build a [`synthetic_refinement`] and run `compute_map` once at a flat
/// starting B offset from `b_true`, so `sigma_a` is populated and a subsequent
/// [`gradient_once`] does only the per-evaluation gradient work.
///
/// `b_true` sets `f_obs` (the working set fits at truth); `b_start` is the flat
/// B the map and the gradient are both evaluated at, offset from `b_true` so
/// the gradient is non-trivial.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn prepared_synthetic(
    n_atoms: usize,
    sg_number: i32,
    cell: [f64; 6],
    grid_spacing_target: f64,
    b_true: f64,
    b_start: f64,
) -> PreparedCase {
    let case = synthetic_refinement(
        n_atoms,
        sg_number,
        cell,
        grid_spacing_target,
        b_true,
    );
    prepare_case(case, b_start)
}

/// Build a synthetic refinement on the native five-smooth grid (the same
/// [`derive_grid`] sampling as [`prepared_synthetic`]) with the full-orbit
/// forward model ([`ForwardSymmetry::SplatFullOrbit`]) and prepare it for a
/// single gradient evaluation. Paired with [`StencilBackend::Gpu`] this drives
/// the full resident GPU pipeline (GPU splat + mixed-radix FFT + resident
/// forward/inverse + GPU gather) on the pipeline's native grid, so a bench can
/// time it against the CPU arm on identical grid dimensions.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn prepared_synthetic_native_orbit(
    n_atoms: usize,
    sg_number: i32,
    cell: [f64; 6],
    grid_spacing_target: f64,
    b_true: f64,
    b_start: f64,
) -> PreparedCase {
    let grid = derive_grid(
        cell[0],
        cell[1],
        cell[2],
        grid_spacing_target,
        sg_number as u16,
    );
    let case = build_synthetic(
        n_atoms,
        sg_number,
        cell,
        grid_spacing_target,
        b_true,
        grid,
        ForwardSymmetry::SplatFullOrbit,
    );
    prepare_case(case, b_start)
}

/// Run `compute_map` at a flat `b_start` to populate sigma-A, then package the
/// case for a single gradient evaluation. Shared tail of the smooth-grid and
/// power-of-two prepared builders; `compute_map` honors the case's own forward
/// symmetry, so the sigma-A it fits matches the forward model the gradient
/// differentiates.
#[must_use]
fn prepare_case(case: SyntheticCase, b_start: f64) -> PreparedCase {
    let mut refinement = case.refinement;
    let positions = case.positions;
    let elements = case.elements;
    let occupancies = case.occupancies;
    let b_factors = vec![b_start; positions.len()];

    let _density = refinement
        .compute_map(&positions, &elements, &b_factors, &occupancies)
        .expect("compute_map populates sigma-A for the synthetic case");
    let grid_dims = refinement.grid_dims;

    PreparedCase {
        refinement,
        positions,
        elements,
        b_factors,
        occupancies,
        grid_dims,
    }
}

/// Run one `b_factor_gradients` evaluation against a [`prepared_synthetic`]
/// case, exposing the `pub(crate)` gradient to the external bench crates.
#[must_use]
pub fn gradient_once(case: &PreparedCase) -> Option<Vec<f64>> {
    case.refinement.b_factor_gradients(
        &case.positions,
        &case.elements,
        &case.b_factors,
        &case.occupancies,
    )
}

/// Run [`gradient_once`] with the refinement's real-space stencil backend set
/// to `backend`, so a bench can time the CPU and GPU splat/gather at one
/// prepared case. Both the forward splat and the adjoint gather gate on this
/// backend.
#[must_use]
pub fn gradient_once_with(
    case: &mut PreparedCase,
    backend: StencilBackend,
) -> Option<Vec<f64>> {
    case.refinement.stencil_backend = backend;
    gradient_once(case)
}

/// Outcome of a full-GPU-pipeline B-factor recovery run on a deposited
/// structure.
#[cfg(feature = "xtal-gpu")]
pub struct GpuRecovery {
    /// Pearson correlation of refined vs deposited B-factors.
    pub b_corr: f64,
    /// R-work at the flattened start, before refinement.
    pub r_work_before: f64,
    /// R-work after refinement.
    pub r_work_after: f64,
    /// Non-hydrogen atom count (recovery-vector length).
    pub n_atoms: usize,
    /// Power-of-two FFT grid the recovery ran on, `[nu, nv, nw]`.
    pub grid: [usize; 3],
}

/// Run B-factor recovery for a deposited structure through the full GPU
/// pipeline: a power-of-two grid with the splat-full-orbit forward model and
/// the GPU stencil backend, so the map, the forward Fc, and the gradient all
/// run GPU splat + GPU FFT + GPU gather.
///
/// Flattens every deposited B to their common mean, refines up from that start,
/// and reports the Pearson correlation of the refined B-factors against the
/// deposited ones (the recovery quality). Returns `None` if the fixture is
/// absent.
#[cfg(feature = "xtal-gpu")]
#[must_use]
pub fn recover_b_factors_full_gpu(pdb: &str) -> Option<GpuRecovery> {
    let case = refinement_from_cif_pair(pdb)?;
    let mut refinement = case.refinement;
    let positions = case.positions;
    let elements = case.elements;
    let deposited_b = case.b_factors;
    let occupancies = case.occupancies;

    // Power-of-two orbit grid at the deposited d_min/3 sampling target, run on
    // the GPU backend so the whole refinement is on-device.
    let mut d_min = f64::MAX;
    for r in &refinement.reflections {
        let s2 = refinement.unit_cell.d_star_sq(r.h, r.k, r.l);
        if s2 > 0.0 {
            d_min = d_min.min(1.0 / s2.sqrt());
        }
    }
    let cell = &refinement.unit_cell;
    let pow2 =
        derive_grid_pow2(cell.a, cell.b, cell.c, d_min / 3.0, case.sg_number);
    refinement.grid_dims = pow2;
    refinement.forward_symmetry = ForwardSymmetry::SplatFullOrbit;
    refinement.stencil_backend = StencilBackend::Gpu;

    // Flatten B to the common mean and refit scaling to that start, so the
    // "before" R-work is scaling-consistent with the flattened model.
    let mean_b = deposited_b.iter().sum::<f64>() / deposited_b.len() as f64;
    let mut refined_b = vec![mean_b; deposited_b.len()];
    let _ = refinement
        .compute_map(&positions, &elements, &refined_b, &occupancies)
        .expect("compute_map (flattened, full-GPU orbit)");
    let (r_work_before, _) = refinement
        .r_factors(&positions, &elements, &refined_b, &occupancies)
        .expect("r_factors (flattened)");

    let (r_work_after, _) = refinement
        .refine(&positions, &elements, &mut refined_b, &occupancies, 50)
        .expect("refine (full-GPU orbit)");

    Some(GpuRecovery {
        b_corr: pearson_pair(&deposited_b, &refined_b),
        r_work_before,
        r_work_after,
        n_atoms: deposited_b.len(),
        grid: pow2,
    })
}

/// Pearson correlation between two equal-length samples; `0.0` when either has
/// no variance.
#[cfg(feature = "xtal-gpu")]
fn pearson_pair(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov = dx.mul_add(dy, cov);
        var_x = dx.mul_add(dx, var_x);
        var_y = dy.mul_add(dy, var_y);
    }
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-30 {
        0.0
    } else {
        cov / denom
    }
}

/// Deterministic fractional position for atom `i`, spread across the cell by a
/// golden-ratio low-discrepancy sequence (no RNG, no clock).
fn deterministic_frac_position(i: usize) -> [f64; 3] {
    const IRR: [f64; 3] = [
        0.618_033_988_749_895_f64,
        0.754_877_666_246_692_7,
        0.569_840_290_998_053_2,
    ];
    let t = (i as f64) + 1.0;
    [
        (t * IRR[0]).fract(),
        (t * IRR[1]).fract(),
        (t * IRR[2]).fract(),
    ]
}
