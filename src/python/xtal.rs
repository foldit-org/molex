//! Python bindings for the resident crystallographic surface.
//!
//! [`PyExperimentalData`] parses an sf.cif once into a cached
//! [`ExperimentalData`] bundle (unit cell, space group, reflections with the
//! free-flag partition resolved) and drives the map/refine workflow against a
//! [`PyAtomTable`]'s native columns. Each call rebuilds a fresh
//! [`XtalRefinement`] from the resident data; the parse cost is paid once, the
//! per-call refinement setup is cheap by comparison.
//!
//! Both the density splat and the B-factor refinement exclude hydrogens,
//! matching the validated refinement path: an X-ray density map carries almost
//! no hydrogen signal, so they are dropped before structure factors are
//! computed. `refine_b_factors` scatters the refined non-H B-factors back into
//! a full-length array aligned 1:1 with the atom table, leaving hydrogen
//! B-factors at their input values.

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;

use super::atom_table::PyAtomTable;
use super::value_err;
use crate::xtal::ExperimentalData;

/// Resident experimental crystallographic data: parse an sf.cif once, then run
/// the map/refine workflow against a `PyAtomTable`.
#[pyclass(name = "ExperimentalData", module = "molex")]
pub struct PyExperimentalData {
    inner: ExperimentalData,
}

#[pymethods]
impl PyExperimentalData {
    /// Parse an sf.cif string, resolving the space group from the file's own
    /// symmetry tag.
    ///
    /// # Errors
    ///
    /// `PyValueError` if the file omits symmetry, lacks a cell or reflections,
    /// or names an unsupported space group.
    #[staticmethod]
    #[pyo3(signature = (sf_cif, free_fraction = 0.05, seed = 0))]
    fn from_sf_cif(
        sf_cif: &str,
        free_fraction: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let inner = ExperimentalData::from_sf_cif(sf_cif, free_fraction, seed)
            .ok_or_else(|| {
                value_err(
                    "failed to parse sf.cif (missing \
                     symmetry/cell/reflections or unsupported space group)",
                )
            })?;
        Ok(Self { inner })
    }

    /// Parse an sf.cif string with an externally supplied International Tables
    /// space-group number, for the common case where the sf.cif omits symmetry.
    ///
    /// # Errors
    ///
    /// `PyValueError` if the file lacks a cell or reflections, or the number is
    /// an unsupported space group.
    #[staticmethod]
    #[pyo3(signature = (
        sf_cif,
        space_group_number,
        free_fraction = 0.05,
        seed = 0,
    ))]
    fn from_sf_cif_with_spacegroup(
        sf_cif: &str,
        space_group_number: u16,
        free_fraction: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let inner = ExperimentalData::from_sf_cif_with_spacegroup(
            sf_cif,
            space_group_number,
            free_fraction,
            seed,
        )
        .ok_or_else(|| {
            value_err(
                "failed to parse sf.cif (missing cell/reflections or \
                 unsupported space group)",
            )
        })?;
        Ok(Self { inner })
    }

    /// Minimum Bragg spacing (Å) over the reflection set.
    #[getter]
    fn d_min(&self) -> f64 {
        self.inner.d_min()
    }

    /// The space-group International Tables number.
    #[getter]
    fn space_group_number(&self) -> i32 {
        self.inner.space_group_number()
    }

    /// Number of reflections in the resident set.
    #[getter]
    fn num_reflections(&self) -> usize {
        self.inner.reflections.len()
    }

    /// Density-map grid dimensions `(nu, nv, nw)` derived from the cell and
    /// resolution.
    #[getter]
    fn grid_dims(&self) -> (usize, usize, usize) {
        self.inner.derive_grid().into()
    }

    /// Compute the 2mFo-DFc electron density map from the atom table's current
    /// coordinates and B-factors, returned as a `(nu, nv, nw)` float32 numpy
    /// array in row-major (u, v, w) order.
    ///
    /// Hydrogens are excluded from the splat, matching the validated refinement
    /// path. A fresh [`XtalRefinement`] is built from the resident data for
    /// this call.
    ///
    /// # Errors
    ///
    /// `PyValueError` if density computation fails (unsupported space group or
    /// empty data).
    fn compute_density(
        &self,
        py: Python<'_>,
        atoms: &PyAtomTable,
    ) -> PyResult<Py<PyAny>> {
        let grid =
            crate::xtal::density_from_atom_table(&self.inner, atoms.table())
                .ok_or_else(|| {
                    value_err(
                        "density computation failed (unsupported space group \
                         or empty data)",
                    )
                })?;

        let arr = grid
            .data
            .into_pyarray(py)
            .reshape((grid.nu, grid.nv, grid.nw))?;
        Ok(arr.into_any().unbind())
    }

    /// Refine per-atom isotropic B-factors against this resident data,
    /// returning `(new_b_factors, r_work, r_free)`.
    ///
    /// Hydrogens are excluded from the refinement, matching the validated
    /// refinement path. The refined non-H B-factors are scattered back into a
    /// full-length float32 array (length == atom table length) where hydrogen
    /// atoms keep their input B-factor, so the returned array maps 1:1 onto the
    /// atom table for write-back. A fresh [`XtalRefinement`] is built from the
    /// resident data for this call.
    ///
    /// # Errors
    ///
    /// `PyValueError` if the refinement pipeline fails at any point.
    #[cfg(feature = "minimization")]
    #[pyo3(signature = (atoms, n_macro_cycles = 1))]
    fn refine_b_factors(
        &self,
        py: Python<'_>,
        atoms: &PyAtomTable,
        n_macro_cycles: usize,
    ) -> PyResult<(Py<PyAny>, f64, f64)> {
        let (full_b, r_work, r_free) = crate::xtal::refine_b_from_atom_table(
            &self.inner,
            atoms.table(),
            n_macro_cycles,
            |_, _, _| {},
        )
        .ok_or_else(|| value_err("B-factor refinement failed"))?;

        let arr = full_b.into_pyarray(py).into_any().unbind();
        Ok((arr, r_work, r_free))
    }

    /// `<ExperimentalData: SG {n}, {r} reflections, d_min {d:.2} Å>`.
    fn __repr__(&self) -> String {
        format!(
            "<ExperimentalData: SG {}, {} reflections, d_min {:.2} Å>",
            self.inner.space_group_number(),
            self.inner.reflections.len(),
            self.inner.d_min(),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use pyo3::types::PyAnyMethods;

    use super::*;
    use crate::adapters::pdb::pdb_str_to_entities;
    use crate::assembly::Assembly;
    use crate::ops::wire::serialize_assembly;
    use crate::python::PyAtomTable;

    const ALA_PDB: &str =
        "\
ATOM      1  N   ALA A   1       5.000   5.000   5.000  1.00 20.00           \
         N\nATOM      2  CA  ALA A   1       6.500   5.000   5.000  1.00 \
         22.00           C\nATOM      3  C   ALA A   1       7.000   6.400   \
         5.000  1.00 24.00           C\nATOM      4  O   ALA A   1       \
         6.200   7.300   5.000  1.00 26.00           O\nATOM      5  CB  ALA \
         A   1       7.000   4.200   6.200  1.00 28.00           C\nEND\n";

    /// A resident round-trip through the Python surface: parse a fixture sf.cif
    /// once, then compute a density map from a small atom table, and confirm
    /// the returned numpy grid's shape matches `grid_dims` (which comes from
    /// the resident cell/resolution, independent of the atoms).
    #[test]
    fn compute_density_grid_matches_grid_dims() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let sf = std::fs::read_to_string(dir.join("5YPA-sf.cif")).unwrap();

        // 5YPA is P 21 21 21 (International Tables number 19).
        let data =
            PyExperimentalData::from_sf_cif_with_spacegroup(&sf, 19, 0.05, 7)
                .expect("resident experimental data for 5YPA");

        let entities =
            pdb_str_to_entities(ALA_PDB).expect("parse the ALA fixture");
        let bytes = serialize_assembly(&Assembly::new(entities))
            .expect("serialize fixture");
        let table = PyAtomTable::from_assembly_bytes(&bytes)
            .expect("build atom table from bytes");

        Python::initialize();
        Python::attach(|py| {
            let grid = data
                .compute_density(py, &table)
                .expect("compute_density on resident data");
            let shape: (usize, usize, usize) = grid
                .bind(py)
                .getattr("shape")
                .expect("numpy shape")
                .extract()
                .expect("shape as 3-tuple");
            assert_eq!(shape, data.grid_dims());
        });
    }
}
