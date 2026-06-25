//! Native (Biotite-free) read path: entities -> per-atom numpy columns.
//!
//! Holds the `AtomArrays` pyclass, the shared `entities_to_arrays` core that
//! the object API (`Assembly.to_arrays` / `Entity.to_arrays`) and the
//! bytes-fed `assembly_bytes_to_arrays` transport helper all route through,
//! so the numpy building lives in exactly one place. The flat per-atom column
//! marshaling (the native -> vocab de-vocab) is shared with the Biotite bridge
//! via [`crate::adapters::atomworks::columns`].

use std::collections::HashMap;

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;

use crate::adapters::atomworks::columns;
use crate::adapters::table::AtomTable;
use crate::entity::molecule::MoleculeEntity;
use crate::ops::wire::deserialize_assembly;

/// Per-atom annotation columns as numpy arrays, returned without Biotite.
///
/// Each field holds a real `numpy.ndarray`. This is the native read path:
/// it builds the same columns as the Biotite `AtomArray` conversion but
/// returns plain numpy arrays, so callers never need Biotite importable.
/// Bonds are intentionally not exposed here (they are a Biotite-only concern).
#[pyclass(name = "AtomArrays", module = "molex")]
pub struct AtomArrays {
    /// `(n, 3)` float32 coordinates.
    #[pyo3(get)]
    coords: Py<PyAny>,
    /// Per-atom PDB atom names.
    #[pyo3(get)]
    atom_names: Py<PyAny>,
    /// Per-atom element symbols.
    #[pyo3(get)]
    elements: Py<PyAny>,
    /// Per-atom residue IDs (int32).
    #[pyo3(get)]
    res_ids: Py<PyAny>,
    /// Per-atom residue names.
    #[pyo3(get)]
    res_names: Py<PyAny>,
    /// Per-atom chain IDs.
    #[pyo3(get)]
    chain_ids: Py<PyAny>,
    /// Per-atom occupancies (float32).
    #[pyo3(get)]
    occupancies: Py<PyAny>,
    /// Per-atom B-factors (float32).
    #[pyo3(get)]
    b_factors: Py<PyAny>,
    /// Per-atom AtomWorks entity IDs (int32).
    #[pyo3(get)]
    entity_ids: Py<PyAny>,
    /// Per-atom AtomWorks `mol_type` strings.
    #[pyo3(get)]
    mol_types: Py<PyAny>,
    /// Per-atom AtomWorks `chain_type` codes (int32).
    #[pyo3(get)]
    chain_types: Py<PyAny>,
}

#[pymethods]
impl AtomArrays {
    /// Number of atoms, read from the always-present `(n, 3)` `coords`
    /// anchor rather than an arbitrary annotation column.
    fn __len__(&self, py: Python) -> PyResult<usize> {
        self.coords.bind(py).len()
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!("<AtomArrays: {} atoms>", self.__len__(py)?))
    }
}

/// Build per-atom numpy columns from a slice of entities, without Biotite.
///
/// The shared core behind every native-read entry point: it collects the
/// flat per-atom annotation columns and turns each into a real
/// `numpy.ndarray` inside an `AtomArrays` handle. The whole-assembly,
/// per-entity, and bytes-fed read paths all route here so the numpy
/// building lives in exactly one place. Imports only `numpy`, never
/// Biotite.
///
/// # Errors
///
/// Returns `PyErr` if any numpy operation fails.
pub(crate) fn entities_to_arrays<E: std::borrow::Borrow<MoleculeEntity>>(
    py: Python,
    entities: &[E],
) -> PyResult<AtomArrays> {
    let table = AtomTable::from_entities(entities);
    let len = table.len();
    table_to_arrays(py, &table, 0..len)
}

/// Lookup from per-entity atom identity to `to_arrays()` flat index space.
///
/// `flat_of` maps `(entity raw id, atom index within that entity)` to the
/// atom's position in canonical `to_arrays()` order (the order
/// [`AtomTable::from_entities`] lays atoms out, the single source of that
/// ordering). Bond and disulfide detection report endpoints as `(entity,
/// per-entity index)` pairs, and this is the lookup that lands them in flat
/// space.
pub(crate) struct FlatAtoms {
    pub flat_of: HashMap<(u32, u32), u32>,
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "flat atom count fits in u32 for valid structures"
)]
pub(crate) fn flat_atoms<E: std::borrow::Borrow<MoleculeEntity>>(
    entities: &[E],
) -> FlatAtoms {
    let flat_of = AtomTable::flat_source_indices(entities)
        .into_iter()
        .enumerate()
        .map(|(flat, src)| (src, flat as u32))
        .collect();

    FlatAtoms { flat_of }
}

/// Marshal a flat-atom `range` of `table` into numpy columns, de-vocabbing
/// through the shared [`columns`] layer.
///
/// The whole-entity read passes `0..table.len()`; the per-residue read passes
/// that residue's contiguous flat run, so a single residue's atoms reach numpy
/// without building any intermediate column bundle. `range.len()` is the row
/// count, so coords reshapes to `(n, 3)`.
///
/// # Errors
///
/// Returns `PyErr` if any numpy operation fails.
pub(crate) fn table_to_arrays(
    py: Python,
    table: &AtomTable,
    range: std::ops::Range<usize>,
) -> PyResult<AtomArrays> {
    let numpy = py.import("numpy")?;
    let n = range.len();

    // Numeric columns: move the owned vecs straight into numpy (no float64/
    // int64 intermediate, no astype copy). `into_pyarray` yields float32 /
    // int32 directly; coords reshape to (n, 3) is a metadata view.
    let coords = columns::coords_flat(table, range.clone())
        .into_pyarray(py)
        .reshape((n, 3))?;
    let res_ids = columns::res_ids(table, range.clone()).into_pyarray(py);
    let occupancies =
        columns::occupancies(table, range.clone()).into_pyarray(py);
    let b_factors = columns::b_factors(table, range.clone()).into_pyarray(py);
    let entity_ids = columns::entity_ids(table, range.clone()).into_pyarray(py);
    let chain_types =
        columns::chain_types(table, range.clone()).into_pyarray(py);

    Ok(AtomArrays {
        coords: coords.into_any().unbind(),
        atom_names: numpy
            .call_method1(
                "array",
                (columns::atom_names(table, range.clone()),),
            )?
            .unbind(),
        elements: numpy
            .call_method1("array", (columns::elements(table, range.clone()),))?
            .unbind(),
        res_ids: res_ids.into_any().unbind(),
        res_names: numpy
            .call_method1("array", (columns::res_names(table, range.clone()),))?
            .unbind(),
        chain_ids: numpy
            .call_method1("array", (columns::chain_ids(table, range.clone()),))?
            .unbind(),
        occupancies: occupancies.into_any().unbind(),
        b_factors: b_factors.into_any().unbind(),
        entity_ids: entity_ids.into_any().unbind(),
        mol_types: numpy
            .call_method1("array", (columns::mol_types(table, range),))?
            .unbind(),
        chain_types: chain_types.into_any().unbind(),
    })
}

/// Convert assembly wire bytes to per-atom numpy arrays without Biotite.
///
/// Transport/serialization helper: decodes wire bytes, then defers to the
/// shared `entities_to_arrays` core. The documented default for a normal
/// caller is the object API (`Assembly.from_pdb` / `.entities` / `.to_arrays`
/// and `Entity.to_arrays`), which reaches the same core without a bytes
/// round-trip; this stays for the plugin transport protocol.
///
/// # Errors
///
/// Returns `PyErr` if the bytes cannot be deserialized or if numpy operations
/// fail.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn assembly_bytes_to_arrays(
    py: Python,
    bytes: Vec<u8>,
) -> PyResult<AtomArrays> {
    let assembly = deserialize_assembly(&bytes).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;
    entities_to_arrays(py, assembly.entities())
}
