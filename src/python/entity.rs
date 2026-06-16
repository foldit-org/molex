//! Object-graph pyclasses `PyEntity` and `PyResidue`: the default Python
//! navigation surface reached from `Assembly.entities()`. A submodule of the
//! `python` module, split out to keep each file under the 800-line source cap
//! enforced by `just file-lengths`.
//!
//! `PyEntity` holds an `Arc<MoleculeEntity>` (a cheap refcount clone of the
//! assembly's own entity), so it is `'static` and reads straight through the
//! existing `MoleculeEntity` accessors. `PyResidue` holds the same
//! `Arc<MoleculeEntity>` plus a residue index, so its scalar getters read
//! through the entity's residue list and it can build its own atoms as numpy
//! columns via the shared entities -> arrays core.

use std::sync::Arc;

use pyo3::prelude::*;

use super::arrays::{atom_data_to_arrays, entities_to_arrays, AtomArrays};
use super::resolve_index;
use super::walk::{entity_kind_str, molecule_type_str};
use crate::adapters::atomworks::collect_atom_data;
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
    /// non-polymer entity. Each `Residue` holds an `Arc` refcount clone of this
    /// entity plus its residue index, so the construction is cheap.
    #[must_use]
    pub fn residues(&self) -> Vec<PyResidue> {
        self.inner
            .residues()
            .map(|residues| {
                (0..residues.len())
                    .filter_map(|i| PyResidue::new(Arc::clone(&self.inner), i))
                    .collect()
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

    /// The residue at `index` as a `Residue`, supporting Python negative
    /// indexing (`-1` is the last residue). With `__len__` this makes
    /// `for r in entity:` iterate. Raises `PyIndexError` for a non-polymer
    /// entity (which has no residues) as well as for an out-of-range index.
    ///
    /// # Errors
    ///
    /// `PyIndexError` when the resolved index is out of range.
    pub fn __getitem__(&self, index: isize) -> PyResult<PyResidue> {
        let i = resolve_index(index, self.residue_count(), "Entity")?;
        PyResidue::new(Arc::clone(&self.inner), i).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Entity index {index} out of range (len={})",
                self.residue_count()
            ))
        })
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

/// Python handle for one residue in a polymer entity.
///
/// Holds an `Arc` refcount clone of the parent entity plus the residue's index,
/// so its scalar getters read straight through the entity's residue list and
/// `to_arrays()` can build the residue's own atoms as numpy columns. The index
/// is always in range: every constructor (`Entity.residues` /
/// `Entity.__getitem__`) bounds-checks against a polymer entity's residue list
/// before building a `PyResidue`.
#[pyclass(name = "Residue", module = "molex")]
pub struct PyResidue {
    inner: Arc<MoleculeEntity>,
    residue_index: usize,
}

impl PyResidue {
    /// Build a residue handle for residue `index` of `entity`. Returns `None`
    /// when `entity` is not a polymer or `index` is out of range, so the
    /// pyclass getters can read through a guaranteed-valid index.
    pub(super) fn new(
        entity: Arc<MoleculeEntity>,
        index: usize,
    ) -> Option<Self> {
        let residues = entity.residues()?;
        if index >= residues.len() {
            return None;
        }
        Some(Self {
            inner: entity,
            residue_index: index,
        })
    }

    /// The backing `Residue`. `new` guarantees a polymer entity and an
    /// in-range index, so the lookup resolves to a residue.
    fn residue(&self) -> Option<&Residue> {
        self.inner.residues().map(|r| &r[self.residue_index])
    }
}

#[pymethods]
impl PyResidue {
    /// Trimmed 3-letter residue name (e.g. `"ALA"`).
    #[must_use]
    #[getter]
    pub fn name(&self) -> String {
        self.residue().map_or_else(
            || "UNK".to_owned(),
            |r| {
                std::str::from_utf8(&r.name)
                    .unwrap_or("UNK")
                    .trim_end()
                    .to_owned()
            },
        )
    }

    /// Author-side sequence id, falling back to `label_seq_id` when absent.
    #[must_use]
    #[getter]
    pub fn seq_id(&self) -> i32 {
        self.residue().map_or(0, Residue::seq_id)
    }

    /// Structural-side sequence id (`label_seq_id`).
    #[must_use]
    #[getter]
    pub fn label_seq_id(&self) -> i32 {
        self.residue().map_or(0, |r| r.label_seq_id)
    }

    /// Insertion code as a one-character string, or `None` when blank.
    #[must_use]
    #[getter]
    pub fn ins_code(&self) -> Option<String> {
        self.residue()
            .and_then(|r| r.ins_code)
            .filter(u8::is_ascii_graphic)
            .map(|b| (b as char).to_string())
    }

    /// This residue's atoms as per-atom numpy columns (an `AtomArrays`), via
    /// the shared entities -> arrays core: the parent entity's columns are
    /// collected once, then sliced to this residue's contiguous run of atoms.
    /// The flat column offset is the sum of the prior residues' atom counts,
    /// matching how the shared collector flattens polymer atoms in residue
    /// order.
    ///
    /// # Errors
    ///
    /// `PyErr` if numpy is unavailable or a numpy operation fails.
    pub fn to_arrays(&self, py: Python) -> PyResult<AtomArrays> {
        let Some(residues) = self.inner.residues() else {
            // `new` only builds for polymer entities, so this is unreachable in
            // practice; return empty columns rather than panic.
            let data = collect_atom_data(std::slice::from_ref(&self.inner), 0)
                .slice_atoms(0..0);
            return atom_data_to_arrays(py, &data, 0);
        };
        let offset: usize = residues[..self.residue_index]
            .iter()
            .map(|r| r.atom_range.len())
            .sum();
        let count = residues[self.residue_index].atom_range.len();

        let entity_atoms = self.inner.atom_count();
        let data =
            collect_atom_data(std::slice::from_ref(&self.inner), entity_atoms);
        let sliced = data.slice_atoms(offset..(offset + count));
        atom_data_to_arrays(py, &sliced, count)
    }

    /// Concise repr: `<Residue {name} {seq_id}{ins_code}>`.
    #[must_use]
    pub fn __repr__(&self) -> String {
        format!(
            "<Residue {} {}{}>",
            self.name(),
            self.seq_id(),
            self.ins_code().as_deref().unwrap_or(""),
        )
    }
}
