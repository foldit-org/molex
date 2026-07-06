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
    grid_factors, reflections_from_cif, requires_equal_uv, round_up_to_smooth,
    space_group, Reflection, UnitCell, XtalRefinement,
};

/// The space groups the xtal module supports, for name-to-number lookup.
const SUPPORTED_SG: &[u16] = &[1, 4, 5, 19, 23, 92, 96, 152, 154, 178];

/// A refinement built from a deposited structure and its structure factors.
pub struct RealCase {
    /// The refinement pipeline, grid and reflections already set up.
    pub refinement: XtalRefinement,
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
    if requires_equal_uv(sg_number) {
        let m = nu.max(nv);
        nu = m;
        nv = m;
    }
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
    let sg_u16 = sg_number as u16;
    let unit_cell =
        UnitCell::new(cell[0], cell[1], cell[2], cell[3], cell[4], cell[5]);
    let sg = space_group(sg_u16).expect("supported space group");
    let grid =
        derive_grid(cell[0], cell[1], cell[2], grid_spacing_target, sg_u16);

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
    let mut refinement = case.refinement;
    let positions = case.positions;
    let elements = case.elements;
    let occupancies = case.occupancies;
    let b_factors = vec![b_start; n_atoms];

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
