//! Language-neutral atom → map/refine orchestration shared by the Python and C
//! bindings.
//!
//! Both bindings flatten an atom source into an [`AtomTable`], drop hydrogens,
//! build [`RefinementInputs`], and rebuild a fresh [`XtalRefinement`] per call.
//! This module owns that core so the bindings stay thin marshalling shells.

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
    let subset = non_hydrogen_atoms(table);
    let inputs = data.refinement_inputs(
        &subset.positions,
        &subset.elements,
        &subset.b_factors,
        &subset.occupancies,
    );

    let mut refinement = XtalRefinement::from_experimental_data(data);
    refinement.compute_map(
        &inputs.positions,
        &inputs.elements,
        &inputs.b_factors,
        &inputs.occupancies,
    )
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
) -> Option<(Vec<f32>, f64, f64)> {
    let subset = non_hydrogen_atoms(table);
    let inputs = data.refinement_inputs(
        &subset.positions,
        &subset.elements,
        &subset.b_factors,
        &subset.occupancies,
    );

    let mut refinement = XtalRefinement::from_experimental_data(data);
    let mut refined_b = inputs.b_factors;
    let (r_work, r_free) = refinement.refine(
        &inputs.positions,
        &inputs.elements,
        &mut refined_b,
        &inputs.occupancies,
        n_macro_cycles,
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
