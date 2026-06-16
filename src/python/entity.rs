//! Object-graph pyclasses `PyEntity` and `PyResidue`: the default Python
//! navigation surface reached from `Assembly.entities()`. A submodule of the
//! `python` module, split out to keep each file under the 800-line source cap
//! enforced by `just file-lengths`.
//!
//! `PyEntity` holds an `Arc<MoleculeEntity>` (a cheap refcount clone of the
//! assembly's own entity), so it is `'static` and reads straight through the
//! existing `MoleculeEntity` accessors. `PyResidue` copies its scalars out
//! (no back-reference into the entity), mirroring the `*View` edit-read
//! classes.

use std::sync::Arc;

use pyo3::prelude::*;

use super::arrays::{entities_to_arrays, AtomArrays};
use super::walk::{entity_kind_str, molecule_type_str};
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::MoleculeEntity;

/// Python handle for one entity in an `Assembly`. Reads through the
/// underlying `MoleculeEntity` accessors; cheap to hold (an `Arc` refcount
/// clone of the assembly's entity).
#[pyclass(name = "Entity", module = "molex")]
pub struct PyEntity {
    pub(super) inner: Arc<MoleculeEntity>,
}

#[pymethods]
impl PyEntity {
    /// Raw entity id (`u32`).
    #[must_use]
    #[getter]
    pub fn id(&self) -> u32 {
        self.inner.id().raw()
    }

    /// Structural discriminant, one of `"Protein"`, `"NucleicAcid"`,
    /// `"SmallMolecule"`, `"Bulk"`.
    #[must_use]
    #[getter]
    pub fn kind(&self) -> &'static str {
        entity_kind_str(self.inner.entity_kind())
    }

    /// Molecule-type classification (e.g. `"Protein"`, `"Ligand"`, `"Ion"`,
    /// `"Water"`).
    #[must_use]
    #[getter]
    pub fn molecule_type(&self) -> &'static str {
        molecule_type_str(self.inner.molecule_type())
    }

    /// PDB chain identifier as a one-character string, or `None` when the
    /// entity has no chain id (a non-polymer entity). The printable filter
    /// (`is_ascii_graphic`) is a safety fallback that also yields `None` for a
    /// non-printable chain byte; it is not the primary meaning of `None`.
    #[must_use]
    #[getter]
    pub fn chain_id(&self) -> Option<String> {
        self.inner
            .pdb_chain_id()
            .filter(u8::is_ascii_graphic)
            .map(|b| (b as char).to_string())
    }

    /// Human-readable label (e.g. `"Protein Chain A"`, `"Ligand (ATP)"`).
    #[must_use]
    #[getter]
    pub fn label(&self) -> String {
        self.inner.label()
    }

    /// Total atom count in this entity.
    #[must_use]
    #[getter]
    pub fn atom_count(&self) -> usize {
        self.inner.atom_count()
    }

    /// Residue count; 0 for non-polymer entities.
    #[must_use]
    #[getter]
    pub fn residue_count(&self) -> usize {
        self.inner.residues().map_or(0, <[_]>::len)
    }

    /// The entity's residues as `Residue` objects, in order. Empty for a
    /// non-polymer entity.
    #[must_use]
    pub fn residues(&self) -> Vec<PyResidue> {
        self.inner
            .residues()
            .map(|residues| {
                residues.iter().map(PyResidue::from_residue).collect()
            })
            .unwrap_or_default()
    }

    /// This entity's atoms as per-atom numpy columns (an `AtomArrays`),
    /// via the shared entities -> arrays core.
    ///
    /// # Errors
    ///
    /// `PyErr` if numpy is unavailable or a numpy operation fails.
    pub fn to_arrays(&self, py: Python) -> PyResult<AtomArrays> {
        entities_to_arrays(py, std::slice::from_ref(&self.inner))
    }

    /// Residue count; mirrors `len(assembly)` so `len(entity)` works too.
    #[must_use]
    pub fn __len__(&self) -> usize {
        self.residue_count()
    }

    /// Concise repr: `<Entity {id} {kind} chain={chain_id} {n} atoms>`.
    #[must_use]
    pub fn __repr__(&self) -> String {
        let chain = self.chain_id().unwrap_or_else(|| "None".to_owned());
        format!(
            "<Entity {} {} chain={} {} atoms>",
            self.id(),
            self.kind(),
            chain,
            self.atom_count(),
        )
    }
}

/// Python handle for one residue in a polymer entity. Holds copied-out
/// scalars (no back-reference into the entity), mirroring the edit-read
/// `*View` classes.
#[pyclass(name = "Residue", module = "molex")]
pub struct PyResidue {
    name: String,
    seq_id: i32,
    label_seq_id: i32,
    ins_code: Option<String>,
}

impl PyResidue {
    pub(super) fn from_residue(residue: &Residue) -> Self {
        let name = std::str::from_utf8(&residue.name)
            .unwrap_or("UNK")
            .trim_end()
            .to_owned();
        Self {
            name,
            seq_id: residue.seq_id(),
            label_seq_id: residue.label_seq_id,
            ins_code: residue
                .ins_code
                .filter(u8::is_ascii_graphic)
                .map(|b| (b as char).to_string()),
        }
    }
}

#[pymethods]
impl PyResidue {
    /// Trimmed 3-letter residue name (e.g. `"ALA"`).
    #[must_use]
    #[getter]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Author-side sequence id, falling back to `label_seq_id` when absent.
    #[must_use]
    #[getter]
    pub fn seq_id(&self) -> i32 {
        self.seq_id
    }

    /// Structural-side sequence id (`label_seq_id`).
    #[must_use]
    #[getter]
    pub fn label_seq_id(&self) -> i32 {
        self.label_seq_id
    }

    /// Insertion code as a one-character string, or `None` when blank.
    #[must_use]
    #[getter]
    pub fn ins_code(&self) -> Option<String> {
        self.ins_code.clone()
    }

    /// Concise repr: `<Residue {name} {seq_id}{ins_code}>`.
    #[must_use]
    pub fn __repr__(&self) -> String {
        format!(
            "<Residue {} {}{}>",
            self.name,
            self.seq_id,
            self.ins_code.as_deref().unwrap_or(""),
        )
    }
}
