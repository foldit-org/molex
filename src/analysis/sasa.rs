//! Shrake-Rupley solvent-accessible surface area (SASA).
//!
//! For each atom, the accessible surface is the locus of probe-sphere
//! *centers* that touch the atom without overlapping any neighbor: a sphere of
//! radius `vdw + probe` (the solvent-accessible radius). Shrake & Rupley
//! (1973) estimate each atom's accessible area by sampling that sphere at a
//! fixed set of quasi-uniform test points and counting the fraction not buried
//! inside any neighbor's solvent-accessible sphere.
//!
//! Sampling is deterministic: the test points are the Fibonacci/golden-spiral
//! lattice on the unit sphere, so the same input always yields the same area
//! (no RNG). Per-atom area is `(accessible / total) * 4*pi*R^2`; the total is
//! the sum.
//!
//! [`assembly_sasa`] scopes the calculation to protein atoms (water, ions, and
//! other non-polymer entities excluded), matching the convention freesasa and
//! biotite use for their reference protein SASA values. Hydrogens present in
//! the input are included; PDB-derived protein structures are typically heavy-
//! atom only, giving heavy-atom SASA.

use glam::Vec3;

use crate::analysis::grid::SpatialGrid;
use crate::assembly::Assembly;
use crate::entity::molecule::MoleculeType;

/// Default solvent probe radius (water) in Angstroms.
pub const DEFAULT_PROBE_RADIUS: f32 = 1.4;

/// Default number of test points per atom (golden-spiral lattice).
pub const DEFAULT_N_POINTS: usize = 960;

/// Generate `n` quasi-uniform points on the unit sphere via the
/// Fibonacci/golden-spiral lattice. Deterministic; no RNG.
#[allow(
    clippy::cast_precision_loss,
    reason = "point indices are bounded by n (<= a few thousand), well within \
              f32 mantissa precision"
)]
fn fibonacci_sphere(n: usize) -> Vec<Vec3> {
    // Golden angle: pi * (3 - sqrt(5)).
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    let mut points = Vec::with_capacity(n);
    let denom = (n.max(1) - 1).max(1) as f32;
    for i in 0..n {
        // z runs from +1 to -1 in equal steps; ring radius is sqrt(1 - z^2).
        let z = 1.0 - 2.0 * (i as f32) / denom;
        let ring = (1.0 - z * z).max(0.0).sqrt();
        let theta = golden_angle * (i as f32);
        points.push(Vec3::new(ring * theta.cos(), ring * theta.sin(), z));
    }
    points
}

/// Per-atom solvent-accessible surface area in Angstrom^2.
///
/// - `positions`: atom centers (Angstroms)
/// - `radii`: per-atom van der Waals radii (Angstroms), one per position
/// - `probe_radius`: solvent probe radius (1.4 A for water)
/// - `n_points`: test points per atom (more = finer; ~960 is a good default)
///
/// A test point on atom `i`'s solvent-accessible sphere (radius
/// `radii[i] + probe`) is accessible when it lies outside every *other* atom's
/// solvent-accessible sphere. Neighbors are distance-pre-filtered so the inner
/// loop visits only atoms whose spheres can reach the point, not all pairs.
///
/// Returns one area per input atom (empty input -> empty output). `radii` is
/// assumed at least as long as `positions`; extra entries are ignored.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "accessible/total point counts are bounded by n_points, well \
              within f32 mantissa precision"
)]
pub(crate) fn shrake_rupley(
    positions: &[Vec3],
    radii: &[f32],
    probe_radius: f32,
    n_points: usize,
) -> Vec<f32> {
    let n_atoms = positions.len();
    if n_atoms == 0 || n_points == 0 {
        return vec![0.0; n_atoms];
    }

    // Solvent-accessible radius per atom.
    let sa_radii: Vec<f32> =
        radii[..n_atoms].iter().map(|&r| r + probe_radius).collect();
    let max_sa = sa_radii.iter().copied().fold(0.0_f32, f32::max);

    let neighbors = build_neighbor_lists(positions, &sa_radii, max_sa);
    let unit_points = fibonacci_sphere(n_points);

    let four_pi = 4.0 * std::f32::consts::PI;
    let mut areas = Vec::with_capacity(n_atoms);
    for i in 0..n_atoms {
        let center = positions[i];
        let r_i = sa_radii[i];
        let nbrs = &neighbors[i];

        let mut accessible = 0usize;
        for unit in &unit_points {
            let point = center + *unit * r_i;
            let buried = nbrs.iter().any(|&j| {
                let r_j = sa_radii[j];
                point.distance_squared(positions[j]) < r_j * r_j
            });
            if !buried {
                accessible += 1;
            }
        }

        let frac = (accessible as f32) / (n_points as f32);
        areas.push(frac * four_pi * r_i * r_i);
    }
    areas
}

/// For each atom, the indices of atoms whose solvent-accessible spheres can
/// overlap it (centers within `sa_radii[i] + sa_radii[j]`). A uniform grid
/// keyed on the largest accessible radius keeps this near-linear instead of
/// the naive all-pairs scan.
fn build_neighbor_lists(
    positions: &[Vec3],
    sa_radii: &[f32],
    max_sa: f32,
) -> Vec<Vec<usize>> {
    let n = positions.len();
    let mut lists = vec![Vec::new(); n];
    if n == 0 {
        return lists;
    }

    // Two centers' spheres overlap only within max_sa + max_sa; one cell of
    // that width means candidates live in the 27-cell neighborhood.
    let cell = (2.0 * max_sa).max(1e-3);
    let grid = SpatialGrid::build(positions, cell);

    for i in 0..n {
        let r_i = sa_radii[i];
        grid.for_each_candidate(positions[i], |j| {
            if j == i {
                return;
            }
            let reach = r_i + sa_radii[j];
            if positions[i].distance_squared(positions[j]) < reach * reach {
                lists[i].push(j);
            }
        });
    }
    lists
}

/// Total solvent-accessible surface area (Angstrom^2) of an assembly's protein
/// atoms, using Bondi van der Waals radii (`Element::vdw_radius`) and the given
/// probe radius.
///
/// Only `MoleculeType::Protein` entities contribute; water, ions, ligands, and
/// nucleic acids are excluded, matching the protein-scope convention of
/// freesasa and biotite. The neighbor pre-filter still runs over only the
/// protein atom set, so buried-residue areas reflect the protein alone.
///
/// `n_points` controls the angular resolution (use [`DEFAULT_N_POINTS`] for the
/// standard estimate).
#[must_use]
pub fn assembly_sasa(
    assembly: &Assembly,
    probe_radius: f32,
    n_points: usize,
) -> f32 {
    let mut positions = Vec::new();
    let mut radii = Vec::new();
    for entity in assembly.entities() {
        if entity.molecule_type() != MoleculeType::Protein {
            continue;
        }
        for atom in entity.atoms_iter() {
            positions.push(*atom.position);
            radii.push(atom.element.vdw_radius());
        }
    }
    shrake_rupley(&positions, &radii, probe_radius, n_points)
        .iter()
        .sum()
}

impl Assembly {
    /// Total solvent-accessible surface area (Angstrom^2) of this assembly's
    /// protein atoms. See [`assembly_sasa`] for scope and method.
    #[must_use]
    pub fn sasa(&self, probe_radius: f32, n_points: usize) -> f32 {
        assembly_sasa(self, probe_radius, n_points)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn fibonacci_sphere_points_are_unit_length() {
        let pts = fibonacci_sphere(256);
        assert_eq!(pts.len(), 256);
        for p in &pts {
            assert!((p.length() - 1.0).abs() < 1e-4, "len {}", p.length());
        }
    }

    #[test]
    fn fibonacci_sphere_is_deterministic() {
        let a = fibonacci_sphere(960);
        let b = fibonacci_sphere(960);
        assert_eq!(a, b);
    }

    #[test]
    fn single_atom_is_full_sphere() {
        // An isolated atom has nothing to bury its surface: area is the full
        // sphere 4*pi*R^2 with R = vdw + probe.
        let vdw = 1.70_f32;
        let probe = 1.4_f32;
        let r = vdw + probe;
        let expected = 4.0 * std::f32::consts::PI * r * r;

        let areas =
            shrake_rupley(&[Vec3::ZERO], &[vdw], probe, DEFAULT_N_POINTS);
        assert_eq!(areas.len(), 1);
        let rel_err = (areas[0] - expected).abs() / expected;
        assert!(rel_err < 0.01, "area {} vs {expected}", areas[0]);
    }

    #[test]
    fn overlap_reduces_total_below_isolated_sum() {
        // Two atoms 2 A apart (closer than their summed accessible radii) bury
        // part of each other's surface, so the total is strictly less than the
        // sum of two isolated full spheres.
        let vdw = 1.70_f32;
        let probe = 1.4_f32;
        let r = vdw + probe;
        let isolated = 4.0 * std::f32::consts::PI * r * r;

        let positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let radii = [vdw, vdw];
        let areas = shrake_rupley(&positions, &radii, probe, DEFAULT_N_POINTS);
        let total: f32 = areas.iter().sum();

        assert!(
            total < 2.0 * isolated,
            "overlap total {total} should be below isolated sum {}",
            2.0 * isolated
        );
        // Each atom keeps some accessible surface (the contact is partial).
        assert!(areas[0] > 0.0 && areas[1] > 0.0);
    }

    #[test]
    fn empty_input_is_empty() {
        let areas = shrake_rupley(&[], &[], 1.4, DEFAULT_N_POINTS);
        assert!(areas.is_empty());
    }

    #[test]
    fn fully_buried_atom_has_zero_area() {
        // A small central atom surrounded closely by large atoms on all six
        // axes is fully occluded.
        let center = Vec3::ZERO;
        let r_small = 0.1_f32;
        let r_big = 2.0_f32;
        let d = 0.5_f32;
        let positions = vec![
            center,
            Vec3::new(d, 0.0, 0.0),
            Vec3::new(-d, 0.0, 0.0),
            Vec3::new(0.0, d, 0.0),
            Vec3::new(0.0, -d, 0.0),
            Vec3::new(0.0, 0.0, d),
            Vec3::new(0.0, 0.0, -d),
        ];
        let radii = vec![r_small, r_big, r_big, r_big, r_big, r_big, r_big];
        let areas = shrake_rupley(&positions, &radii, 0.0, DEFAULT_N_POINTS);
        assert_eq!(areas[0], 0.0, "buried atom area {}", areas[0]);
    }
}
