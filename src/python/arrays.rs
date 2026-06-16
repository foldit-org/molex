//! Native (Biotite-free) read path: entities -> per-atom numpy columns.
//!
//! Holds the `AtomArrays` pyclass, the shared `entities_to_arrays` core that
//! the object API (`Assembly.to_arrays` / `Entity.to_arrays`) and the
//! bytes-fed `assembly_bytes_to_arrays` transport helper all route through,
//! so the numpy building lives in exactly one place. The flat per-atom column
//! collection is shared with the Biotite bridge via
//! [`crate::adapters::atomworks::collect_atom_data`].

use pyo3::prelude::*;

use crate::adapters::atomworks::collect_atom_data;
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
    /// Number of atoms (length of any column).
    fn __len__(&self, py: Python) -> PyResult<usize> {
        self.atom_names.bind(py).len()
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
    let total_atoms: usize =
        entities.iter().map(|e| e.borrow().atom_count()).sum();
    let data = collect_atom_data(entities, total_atoms);

    let numpy = py.import("numpy")?;

    let coords = numpy.call_method1("array", (&data.coords_flat,))?;
    let coords = coords.call_method1("reshape", ((total_atoms, 3),))?;
    let coords = coords.call_method1("astype", (numpy.getattr("float32")?,))?;

    let res_ids = numpy.call_method1("array", (&data.res_ids,))?;
    let res_ids = res_ids.call_method1("astype", (numpy.getattr("int32")?,))?;

    let occupancies = numpy.call_method1("array", (&data.occupancies,))?;
    let occupancies =
        occupancies.call_method1("astype", (numpy.getattr("float32")?,))?;

    let b_factors = numpy.call_method1("array", (&data.b_factors,))?;
    let b_factors =
        b_factors.call_method1("astype", (numpy.getattr("float32")?,))?;

    let entity_ids = numpy.call_method1("array", (&data.aw_entity_ids,))?;
    let entity_ids =
        entity_ids.call_method1("astype", (numpy.getattr("int32")?,))?;

    let chain_types = numpy.call_method1("array", (&data.aw_chain_types,))?;
    let chain_types =
        chain_types.call_method1("astype", (numpy.getattr("int32")?,))?;

    Ok(AtomArrays {
        coords: coords.unbind(),
        atom_names: numpy.call_method1("array", (&data.atom_names,))?.unbind(),
        elements: numpy.call_method1("array", (&data.elements,))?.unbind(),
        res_ids: res_ids.unbind(),
        res_names: numpy.call_method1("array", (&data.res_names,))?.unbind(),
        chain_ids: numpy.call_method1("array", (&data.chain_ids,))?.unbind(),
        occupancies: occupancies.unbind(),
        b_factors: b_factors.unbind(),
        entity_ids: entity_ids.unbind(),
        mol_types: numpy.call_method1("array", (&data.aw_mol_types,))?.unbind(),
        chain_types: chain_types.unbind(),
    })
}

/// Convert ASSEM01 bytes to per-atom numpy arrays without Biotite.
///
/// Transport/serialization helper: decodes wire bytes, then defers to the
/// shared [`entities_to_arrays`] core. The documented default for a normal
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
