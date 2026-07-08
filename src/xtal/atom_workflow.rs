//! Language-neutral atom → map/refine orchestration shared by the Python and C
//! bindings.
//!
//! Both bindings flatten an atom source into an [`AtomTable`], drop hydrogens,
//! build [`RefinementInputs`], and rebuild a fresh [`XtalRefinement`] per call.
//! This module owns that core so the bindings stay thin marshalling shells.

#[cfg(all(feature = "xtal", feature = "gpu"))]
use cubecl::wgpu::WgpuDevice;

#[cfg(all(feature = "xtal", feature = "gpu"))]
use super::StencilBackend;
use super::{DensityGrid, ExperimentalData, XtalRefinement};
use crate::adapters::table::AtomTable;
use crate::element::Element;

/// Non-hydrogen atom columns lifted out of an [`AtomTable`], parallel arrays
/// plus the map back to full-table atom indices for scatter-out.
struct NonHydrogenAtoms {
    positions: Vec<[f32; 3]>,
    elements: Vec<Element>,
    b_factors: Vec<f32>,
    occupancies: Vec<f32>,
    /// Full-table atom index of each retained atom, parallel to the columns.
    full_indices: Vec<usize>,
}

/// Gather the non-hydrogen atoms of `table` into parallel Cartesian-`f32`
/// columns, recording each retained atom's full-table index.
fn non_hydrogen_atoms(table: &AtomTable) -> NonHydrogenAtoms {
    let mut out = NonHydrogenAtoms {
        positions: Vec::new(),
        elements: Vec::new(),
        b_factors: Vec::new(),
        occupancies: Vec::new(),
        full_indices: Vec::new(),
    };
    for i in 0..table.len() {
        if table.element[i] == Element::H {
            continue;
        }
        let p = table.position[i];
        out.positions.push([p.x, p.y, p.z]);
        out.elements.push(table.element[i]);
        out.b_factors.push(table.b_factor[i]);
        out.occupancies.push(table.occupancy[i]);
        out.full_indices.push(i);
    }
    out
}

/// Splat the non-hydrogen atoms of `table` through an already-configured
/// `refinement` and return its 2mFo-DFc map. The caller chooses the stencil
/// backend (CPU or GPU) before handing the refinement in; keeping the device
/// out of this signature lets it compile with the `gpu` feature off.
fn density_from_atom_table_impl(
    data: &ExperimentalData,
    table: &AtomTable,
    mut refinement: XtalRefinement,
) -> Option<DensityGrid> {
    let subset = non_hydrogen_atoms(table);
    let inputs = data.refinement_inputs(
        &subset.positions,
        &subset.elements,
        &subset.b_factors,
        &subset.occupancies,
    );

    refinement.compute_map(
        &inputs.positions,
        &inputs.elements,
        &inputs.b_factors,
        &inputs.occupancies,
    )
}

/// Compute the 2mFo-DFc electron density map for `table`'s atoms against the
/// resident `data`, rebuilding a fresh [`XtalRefinement`] for this call.
///
/// Hydrogens are excluded from the splat, matching the validated refinement
/// path (an X-ray map carries almost no hydrogen signal). Returns `None` if the
/// map cannot be computed (unsupported space group or empty data).
#[must_use]
pub fn density_from_atom_table(
    data: &ExperimentalData,
    table: &AtomTable,
) -> Option<DensityGrid> {
    density_from_atom_table_impl(
        data,
        table,
        XtalRefinement::from_experimental_data(data),
    )
}

/// GPU-backed [`density_from_atom_table`]: routes the density splat onto the
/// caller's shared `dev` instead of a CPU stencil.
///
/// `dev` is cloned because [`XtalRefinement::with_gpu_device`] takes the device
/// by value; the clone duplicates a ref-counted wgpu handle, not the device
/// itself, so a host can reuse one cached handle across calls.
#[cfg(all(feature = "xtal", feature = "gpu"))]
#[must_use]
pub fn density_from_atom_table_gpu(
    data: &ExperimentalData,
    table: &AtomTable,
    dev: &WgpuDevice,
) -> Option<DensityGrid> {
    let mut refinement = XtalRefinement::from_experimental_data(data);
    refinement.stencil_backend = StencilBackend::Gpu;
    let refinement = refinement.with_gpu_device(dev.clone());
    density_from_atom_table_impl(data, table, refinement)
}

/// Score the non-hydrogen atoms of `table` against `data` through an
/// already-configured `refinement`, returning `(r_work, r_free)`.
///
/// A prior [`XtalRefinement::compute_map`] populates the scaling and sigma-A
/// terms that [`XtalRefinement::r_factors`] reads; its density grid is
/// discarded here (only the scaling side effect is wanted). The caller selects
/// the stencil backend before handing the refinement in.
fn r_factors_from_atom_table_impl(
    data: &ExperimentalData,
    table: &AtomTable,
    mut refinement: XtalRefinement,
) -> Option<(f64, f64)> {
    let subset = non_hydrogen_atoms(table);
    let inputs = data.refinement_inputs(
        &subset.positions,
        &subset.elements,
        &subset.b_factors,
        &subset.occupancies,
    );

    let _scaling_grid = refinement.compute_map(
        &inputs.positions,
        &inputs.elements,
        &inputs.b_factors,
        &inputs.occupancies,
    )?;

    refinement.r_factors(
        &inputs.positions,
        &inputs.elements,
        &inputs.b_factors,
        &inputs.occupancies,
    )
}

/// Compute `(r_work, r_free)` for `table`'s atoms against the resident `data`
/// without refining, rebuilding a fresh [`XtalRefinement`] for this call.
///
/// This is a one-shot scoring pass: it scales the current model and reads off
/// the R-factors, doing no B-factor or coordinate minimization. Hydrogens are
/// excluded, matching the validated refinement path. Returns `None` if the
/// pipeline fails (unsupported space group or empty data).
#[must_use]
pub fn r_factors_from_atom_table(
    data: &ExperimentalData,
    table: &AtomTable,
) -> Option<(f64, f64)> {
    r_factors_from_atom_table_impl(
        data,
        table,
        XtalRefinement::from_experimental_data(data),
    )
}

/// GPU-backed [`r_factors_from_atom_table`]: routes the density splat onto the
/// caller's shared `dev` instead of a CPU stencil.
///
/// `dev` is cloned because [`XtalRefinement::with_gpu_device`] takes the device
/// by value; the clone duplicates a ref-counted wgpu handle, letting a host
/// reuse one cached handle across calls.
#[cfg(all(feature = "xtal", feature = "gpu"))]
#[must_use]
pub fn r_factors_from_atom_table_gpu(
    data: &ExperimentalData,
    table: &AtomTable,
    dev: &WgpuDevice,
) -> Option<(f64, f64)> {
    let mut refinement = XtalRefinement::from_experimental_data(data);
    refinement.stencil_backend = StencilBackend::Gpu;
    let refinement = refinement.with_gpu_device(dev.clone());
    r_factors_from_atom_table_impl(data, table, refinement)
}

/// Refine per-atom isotropic B-factors through an already-configured
/// `refinement`, scattering the refined non-H values back over `table`.
///
/// The caller selects the stencil backend before handing the refinement in,
/// keeping the device type out of this signature.
#[cfg(feature = "minimization")]
fn refine_b_from_atom_table_impl(
    data: &ExperimentalData,
    table: &AtomTable,
    mut refinement: XtalRefinement,
    n_macro_cycles: usize,
    progress: impl FnMut(usize, usize, usize) -> bool + 'static,
) -> Option<(Vec<f32>, f64, f64)> {
    let subset = non_hydrogen_atoms(table);
    let inputs = data.refinement_inputs(
        &subset.positions,
        &subset.elements,
        &subset.b_factors,
        &subset.occupancies,
    );

    let mut refined_b = inputs.b_factors;
    let (r_work, r_free) = refinement.refine(
        &inputs.positions,
        &inputs.elements,
        &mut refined_b,
        &inputs.occupancies,
        n_macro_cycles,
        progress,
    )?;

    // Scatter the refined non-H B-factors back over the full atom table;
    // hydrogens retain their input value.
    let mut full_b = table.b_factor.clone();
    for (&idx, &b) in subset.full_indices.iter().zip(refined_b.iter()) {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "B-factors are small positive magnitudes; f64 -> f32 \
                      write-back mirrors the ingest column dtype"
        )]
        {
            full_b[idx] = b as f32;
        }
    }

    Some((full_b, r_work, r_free))
}

/// Refine per-atom isotropic B-factors for `table`'s atoms against the resident
/// `data`, returning `(full_b, r_work, r_free)`.
///
/// Hydrogens are excluded from the refinement, matching the validated
/// refinement path. The refined non-H B-factors are scattered back into a
/// full-length array (length == `table.len()`, aligned 1:1 to the atom-table
/// order) where hydrogen atoms keep their input B-factor. Returns `None` if the
/// refinement pipeline fails.
#[cfg(feature = "minimization")]
#[must_use]
pub fn refine_b_from_atom_table(
    data: &ExperimentalData,
    table: &AtomTable,
    n_macro_cycles: usize,
    progress: impl FnMut(usize, usize, usize) -> bool + 'static,
) -> Option<(Vec<f32>, f64, f64)> {
    refine_b_from_atom_table_impl(
        data,
        table,
        XtalRefinement::from_experimental_data(data),
        n_macro_cycles,
        progress,
    )
}

/// GPU-backed [`refine_b_from_atom_table`]: routes the density splat onto the
/// caller's shared `dev` instead of a CPU stencil.
///
/// `dev` is cloned because [`XtalRefinement::with_gpu_device`] takes the device
/// by value; the clone duplicates a ref-counted wgpu handle, letting a host
/// reuse one cached handle across calls.
#[cfg(all(feature = "xtal", feature = "gpu", feature = "minimization"))]
#[must_use]
pub fn refine_b_from_atom_table_gpu(
    data: &ExperimentalData,
    table: &AtomTable,
    n_macro_cycles: usize,
    dev: &WgpuDevice,
    progress: impl FnMut(usize, usize, usize) -> bool + 'static,
) -> Option<(Vec<f32>, f64, f64)> {
    let mut refinement = XtalRefinement::from_experimental_data(data);
    refinement.stencil_backend = StencilBackend::Gpu;
    let refinement = refinement.with_gpu_device(dev.clone());
    refine_b_from_atom_table_impl(
        data,
        table,
        refinement,
        n_macro_cycles,
        progress,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::adapters::pdb::pdb_str_to_entities;

    const ALA_PDB: &str =
        "\
ATOM      1  N   ALA A   1       5.000   5.000   5.000  1.00 20.00           \
         N\nATOM      2  CA  ALA A   1       6.500   5.000   5.000  1.00 \
         22.00           C\nATOM      3  C   ALA A   1       7.000   6.400   \
         5.000  1.00 24.00           C\nATOM      4  O   ALA A   1       \
         6.200   7.300   5.000  1.00 26.00           O\nATOM      5  CB  ALA \
         A   1       7.000   4.200   6.200  1.00 28.00           C\nEND\n";

    /// Pure-Rust core coverage: parse the 5YPA fixture sf.cif once, flatten a
    /// small atom table, and confirm the density grid dimensions match the
    /// resident cell/resolution grid (independent of the atoms).
    #[test]
    fn density_from_atom_table_grid_matches_derive_grid() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let sf = std::fs::read_to_string(dir.join("5YPA-sf.cif")).unwrap();

        // 5YPA is P 21 21 21 (International Tables number 19).
        let data =
            ExperimentalData::from_sf_cif_with_spacegroup(&sf, 19, 0.05, 7)
                .expect("resident experimental data for 5YPA");

        let entities =
            pdb_str_to_entities(ALA_PDB).expect("parse the ALA fixture");
        let table = AtomTable::from_entities(&entities);

        let grid = density_from_atom_table(&data, &table)
            .expect("density from resident data");
        assert_eq!([grid.nu, grid.nv, grid.nw], data.derive_grid());
    }
}
