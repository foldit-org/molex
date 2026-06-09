//! Python bindings for parsing structure files into ASSEM01 bytes and
//! emitting PDB from ASSEM01 bytes. AtomArray interop lives in
//! `adapters::atomworks`.
//!
//! Handle-based surface (`Assembly`, `EditList`, `Variant`, `AtomRow`,
//! `ProtonationState`) parallels the C API in `c_api::edit` per
//! `feedback_molex_api_parity.md`.

use glam::Vec3;
use pyo3::prelude::*;

use crate::adapters::{cif, pdb};
use crate::assembly::Assembly;
use crate::chemistry::variant::{ProtonationState, VariantTag};
use crate::element::Element;
use crate::entity::molecule::atom::Atom as MolexAtom;
use crate::entity::molecule::id::EntityIdAllocator;
use crate::ops::edit::AssemblyEdit;
use crate::ops::wire::delta::{
    deserialize_edits, serialize_edits, DeltaSerializeError,
};
use crate::ops::wire::{
    assembly_bytes, deserialize_assembly, serialize_assembly,
};

/// Parse a PDB string and emit ASSEM01 binary bytes.
///
/// # Errors
///
/// Returns a `PyValueError` if the PDB string cannot be parsed.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn pdb_to_assembly_bytes(pdb_str: String) -> PyResult<Vec<u8>> {
    let entities = pdb::pdb_str_to_entities(&pdb_str).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;
    assembly_bytes(&entities).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })
}

/// Parse an mmCIF string and emit ASSEM01 binary bytes.
///
/// # Errors
///
/// Returns a `PyValueError` if the mmCIF string cannot be parsed.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn mmcif_to_assembly_bytes(cif_str: String) -> PyResult<Vec<u8>> {
    let entities = cif::mmcif_str_to_entities(&cif_str).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;
    assembly_bytes(&entities).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })
}

/// Decode ASSEM01 bytes and emit a PDB-format string.
///
/// # Errors
///
/// Returns a `PyValueError` if the bytes cannot be deserialized.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn assembly_bytes_to_pdb(bytes: Vec<u8>) -> PyResult<String> {
    let assembly = deserialize_assembly(&bytes).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;
    pdb::assembly_to_pdb(&assembly).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })
}

/// Round-trip ASSEM01 bytes through `Assembly` and back (validation).
///
/// # Errors
///
/// Returns a `PyValueError` if the bytes cannot be deserialized or
/// re-serialized.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn deserialize_assembly_bytes(bytes: Vec<u8>) -> PyResult<Vec<u8>> {
    deserialize_assembly(&bytes)
        .and_then(|assembly| serialize_assembly(&assembly))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
        })
}

fn value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

/// Python handle wrapping a `molex::Assembly`. Parallels
/// `molex_Assembly` in the C API and `molex::Assembly` in C++.
#[pyclass(name = "Assembly", module = "molex")]
pub struct PyAssembly {
    inner: Assembly,
}

#[pymethods]
impl PyAssembly {
    /// Decode ASSEM01 / ASSEM02 bytes into an `Assembly`.
    ///
    /// # Errors
    ///
    /// `PyValueError` if the magic header is wrong or the payload is
    /// truncated / malformed.
    #[staticmethod]
    #[pyo3(name = "from_assem01")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_assem01(bytes: Vec<u8>) -> PyResult<Self> {
        let inner = deserialize_assembly(&bytes).map_err(value_err)?;
        Ok(Self { inner })
    }

    /// Emit the assembly as ASSEM02 bytes (latest wire format).
    ///
    /// # Errors
    ///
    /// `PyValueError` propagated from the serializer (currently
    /// infallible in practice; the `Result` is kept for API stability).
    pub fn to_assem01(&self) -> PyResult<Vec<u8>> {
        serialize_assembly(&self.inner).map_err(value_err)
    }

    /// Monotonic generation counter.
    #[must_use]
    #[getter]
    pub fn generation(&self) -> u64 {
        self.inner.generation()
    }

    /// Apply every edit in `edits` to the assembly in order.
    ///
    /// # Errors
    ///
    /// `PyValueError` when an edit references an unknown entity /
    /// residue, when a coord-count doesn't match, or when an edit
    /// targets a non-polymer entity. Edits applied before the failing
    /// one stay applied; the generation counter advances per
    /// successful edit.
    pub fn apply_edits(&mut self, edits: &PyEditList) -> PyResult<()> {
        self.inner.apply_edits(&edits.inner).map_err(value_err)
    }

    /// Decode DELTA01 bytes and apply them in one call.
    ///
    /// # Errors
    ///
    /// `PyValueError` for invalid DELTA01 bytes (bad magic / truncated
    /// payload / unknown tag) or any of the apply failures documented
    /// on `apply_edits`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn apply_delta01(&mut self, bytes: Vec<u8>) -> PyResult<()> {
        let edits = deserialize_edits(&bytes).map_err(value_err)?;
        self.inner.apply_edits(&edits).map_err(value_err)
    }
}

/// Python handle for an ordered list of typed Assembly edits.
/// Parallels `molex_EditList` in C and `molex::EditList` in C++.
#[pyclass(name = "EditList", module = "molex")]
#[derive(Default)]
pub struct PyEditList {
    inner: Vec<AssemblyEdit>,
}

fn make_entity_id(raw: u32) -> crate::entity::molecule::id::EntityId {
    EntityIdAllocator::new().from_raw(raw)
}

fn coords_to_vec3(coords: Vec<(f32, f32, f32)>) -> Vec<Vec3> {
    coords
        .into_iter()
        .map(|(x, y, z)| Vec3::new(x, y, z))
        .collect()
}

#[pymethods]
impl PyEditList {
    /// Construct an empty edit list.
    #[must_use]
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of edits currently in the list.
    #[must_use]
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Append a `SetEntityCoords` edit. `coords` is a sequence of
    /// `(x, y, z)` tuples.
    #[allow(clippy::needless_pass_by_value)]
    pub fn push_set_entity_coords(
        &mut self,
        entity_id: u32,
        coords: Vec<(f32, f32, f32)>,
    ) {
        self.inner.push(AssemblyEdit::SetEntityCoords {
            entity: make_entity_id(entity_id),
            coords: coords_to_vec3(coords),
        });
    }

    /// Append a `SetResidueCoords` edit.
    #[allow(clippy::needless_pass_by_value)]
    pub fn push_set_residue_coords(
        &mut self,
        entity_id: u32,
        residue_idx: usize,
        coords: Vec<(f32, f32, f32)>,
    ) {
        self.inner.push(AssemblyEdit::SetResidueCoords {
            entity: make_entity_id(entity_id),
            residue_idx,
            coords: coords_to_vec3(coords),
        });
    }

    /// Append a `MutateResidue` edit. `new_name` is a 3-byte residue
    /// name (e.g. `b"HIS"`); `atoms` is a list of `AtomRow`;
    /// `variants` is a list of `Variant`.
    #[allow(
        clippy::needless_pass_by_value,
        clippy::too_many_arguments,
        reason = "pyo3 method signature mirrors AssemblyEdit::MutateResidue \
                  variant's fields; Python callers use keyword arguments so \
                  the flat shape is the most ergonomic surface."
    )]
    pub fn push_mutate_residue(
        &mut self,
        entity_id: u32,
        residue_idx: usize,
        new_name: [u8; 3],
        atoms: Vec<PyAtomRow>,
        variants: Vec<PyVariant>,
    ) {
        let new_atoms = atoms.into_iter().map(PyAtomRow::into_atom).collect();
        let new_variants =
            variants.into_iter().map(|v| v.inner).collect::<Vec<_>>();
        self.inner.push(AssemblyEdit::MutateResidue {
            entity: make_entity_id(entity_id),
            residue_idx,
            new_name,
            new_atoms,
            new_variants,
        });
    }

    /// Append a `SetVariants` edit.
    #[allow(clippy::needless_pass_by_value)]
    pub fn push_set_variants(
        &mut self,
        entity_id: u32,
        residue_idx: usize,
        variants: Vec<PyVariant>,
    ) {
        let variants =
            variants.into_iter().map(|v| v.inner).collect::<Vec<_>>();
        self.inner.push(AssemblyEdit::SetVariants {
            entity: make_entity_id(entity_id),
            residue_idx,
            variants,
        });
    }

    /// Serialize this list as DELTA01 bytes.
    ///
    /// # Errors
    ///
    /// `PyValueError` when the list contains a topology edit
    /// (`AddEntity` / `RemoveEntity`) which is not representable in
    /// DELTA01; callers should broadcast a full ASSEM02 snapshot in
    /// that case.
    pub fn to_delta01(&self) -> PyResult<Vec<u8>> {
        serialize_edits(&self.inner)
            .map_err(|e: DeltaSerializeError| value_err(format!("{e}")))
    }

    /// Decode DELTA01 bytes into a new edit list.
    ///
    /// # Errors
    ///
    /// `PyValueError` if the magic header is wrong, the payload is
    /// truncated, or an unknown edit tag is encountered.
    #[staticmethod]
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_delta01(bytes: Vec<u8>) -> PyResult<Self> {
        let inner = deserialize_edits(&bytes).map_err(value_err)?;
        Ok(Self { inner })
    }

    // Read accessors (parallels `molex_edits_*_at` C entry points and
    // `molex::EditList::view_at` in C++). Python doesn't have the
    // borrowed-pointer / sibling-free lifecycle the C side does -- pyo3
    // copies edit fields into freshly-allocated Python objects, which the
    // Python GC owns.

    /// Kind of the edit at `index`. Returns one of the
    /// `EditKind.*` integer constants exposed at module scope.
    ///
    /// # Errors
    ///
    /// `PyIndexError` when `index >= len(self)`.
    pub fn kind_at(&self, index: usize) -> PyResult<i32> {
        let edit = self.inner.get(index).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "EditList.kind_at: index {index} out of range (len={})",
                self.inner.len()
            ))
        })?;
        Ok(edit_kind_int(edit))
    }

    /// Read the `SetEntityCoords` edit at `index`.
    ///
    /// # Errors
    ///
    /// `PyIndexError` when `index` is out of range; `PyValueError`
    /// when the edit at `index` is not a `SetEntityCoords` (consult
    /// `kind_at` first).
    pub fn set_entity_coords_at(
        &self,
        index: usize,
    ) -> PyResult<PySetEntityCoordsView> {
        let edit = self.inner.get(index).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "EditList.set_entity_coords_at: index {index} out of range \
                 (len={})",
                self.inner.len()
            ))
        })?;
        let AssemblyEdit::SetEntityCoords { entity, coords } = edit else {
            return Err(value_err(format!(
                "EditList.set_entity_coords_at: edit at index {index} is not \
                 SetEntityCoords"
            )));
        };
        Ok(PySetEntityCoordsView {
            entity_id: entity.raw(),
            coords: coords.iter().map(|c| (c.x, c.y, c.z)).collect(),
        })
    }

    /// Read the `SetResidueCoords` edit at `index`.
    ///
    /// # Errors
    ///
    /// Same shape as [`Self::set_entity_coords_at`].
    pub fn set_residue_coords_at(
        &self,
        index: usize,
    ) -> PyResult<PySetResidueCoordsView> {
        let edit = self.inner.get(index).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "EditList.set_residue_coords_at: index {index} out of range \
                 (len={})",
                self.inner.len()
            ))
        })?;
        let AssemblyEdit::SetResidueCoords {
            entity,
            residue_idx,
            coords,
        } = edit
        else {
            return Err(value_err(format!(
                "EditList.set_residue_coords_at: edit at index {index} is not \
                 SetResidueCoords"
            )));
        };
        Ok(PySetResidueCoordsView {
            entity_id: entity.raw(),
            residue_idx: *residue_idx,
            coords: coords.iter().map(|c| (c.x, c.y, c.z)).collect(),
        })
    }

    /// Read the `MutateResidue` edit at `index`.
    ///
    /// # Errors
    ///
    /// Same shape as [`Self::set_entity_coords_at`].
    pub fn mutate_residue_at(
        &self,
        index: usize,
    ) -> PyResult<PyMutateResidueView> {
        let edit = self.inner.get(index).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "EditList.mutate_residue_at: index {index} out of range \
                 (len={})",
                self.inner.len()
            ))
        })?;
        let AssemblyEdit::MutateResidue {
            entity,
            residue_idx,
            new_name,
            new_atoms,
            new_variants,
        } = edit
        else {
            return Err(value_err(format!(
                "EditList.mutate_residue_at: edit at index {index} is not \
                 MutateResidue"
            )));
        };
        Ok(PyMutateResidueView {
            entity_id: entity.raw(),
            residue_idx: *residue_idx,
            new_name: *new_name,
            atoms: new_atoms.iter().map(PyAtomRow::from_atom).collect(),
            variants: new_variants
                .iter()
                .map(|v| PyVariant { inner: v.clone() })
                .collect(),
        })
    }

    /// Read the `SetVariants` edit at `index`.
    ///
    /// # Errors
    ///
    /// Same shape as [`Self::set_entity_coords_at`].
    pub fn set_variants_at(&self, index: usize) -> PyResult<PySetVariantsView> {
        let edit = self.inner.get(index).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "EditList.set_variants_at: index {index} out of range (len={})",
                self.inner.len()
            ))
        })?;
        let AssemblyEdit::SetVariants {
            entity,
            residue_idx,
            variants,
        } = edit
        else {
            return Err(value_err(format!(
                "EditList.set_variants_at: edit at index {index} is not \
                 SetVariants"
            )));
        };
        Ok(PySetVariantsView {
            entity_id: entity.raw(),
            residue_idx: *residue_idx,
            variants: variants
                .iter()
                .map(|v| PyVariant { inner: v.clone() })
                .collect(),
        })
    }
}

fn edit_kind_int(edit: &AssemblyEdit) -> i32 {
    match edit {
        AssemblyEdit::SetEntityCoords { .. } => 1,
        AssemblyEdit::SetResidueCoords { .. } => 2,
        AssemblyEdit::MutateResidue { .. } => 3,
        AssemblyEdit::SetVariants { .. } => 4,
        AssemblyEdit::AddEntity { .. } => 5,
        AssemblyEdit::RemoveEntity { .. } => 6,
    }
}

// Edit view pyclasses (parallel `molex::SetEntityCoordsView` et al. in C++).

/// Python view of a `SetEntityCoords` edit. Returned by
/// `EditList.set_entity_coords_at`.
#[pyclass(name = "SetEntityCoordsView", module = "molex")]
#[derive(Clone)]
pub struct PySetEntityCoordsView {
    /// Entity whose atoms get repositioned.
    #[pyo3(get)]
    pub entity_id: u32,
    /// Per-atom positions, one `(x, y, z)` tuple per atom in
    /// declaration order. Length equals the entity's atom count.
    #[pyo3(get)]
    pub coords: Vec<(f32, f32, f32)>,
}

/// Python view of a `SetResidueCoords` edit.
#[pyclass(name = "SetResidueCoordsView", module = "molex")]
#[derive(Clone)]
pub struct PySetResidueCoordsView {
    /// Polymer entity that owns the target residue.
    #[pyo3(get)]
    pub entity_id: u32,
    /// Residue index within the entity's residue list.
    #[pyo3(get)]
    pub residue_idx: usize,
    /// Per-atom positions for the residue, in atom-range order.
    #[pyo3(get)]
    pub coords: Vec<(f32, f32, f32)>,
}

/// Python view of a `MutateResidue` edit.
#[pyclass(name = "MutateResidueView", module = "molex")]
#[derive(Clone)]
pub struct PyMutateResidueView {
    /// Polymer entity that owns the target residue.
    #[pyo3(get)]
    pub entity_id: u32,
    /// Residue index within the entity's residue list.
    #[pyo3(get)]
    pub residue_idx: usize,
    /// New 3-letter residue name (space-padded).
    #[pyo3(get)]
    pub new_name: [u8; 3],
    /// New atom list for the residue.
    #[pyo3(get)]
    pub atoms: Vec<PyAtomRow>,
    /// New variant tag list for the residue.
    #[pyo3(get)]
    pub variants: Vec<PyVariant>,
}

/// Python view of a `SetVariants` edit.
#[pyclass(name = "SetVariantsView", module = "molex")]
#[derive(Clone)]
pub struct PySetVariantsView {
    /// Polymer entity that owns the target residue.
    #[pyo3(get)]
    pub entity_id: u32,
    /// Residue index within the entity's residue list.
    #[pyo3(get)]
    pub residue_idx: usize,
    /// New variant tag list for the residue.
    #[pyo3(get)]
    pub variants: Vec<PyVariant>,
}

/// Python handle wrapping a single `VariantTag`.
#[pyclass(name = "Variant", module = "molex")]
#[derive(Clone)]
pub struct PyVariant {
    inner: VariantTag,
}

#[pymethods]
impl PyVariant {
    /// Chain N-terminus patch.
    #[must_use]
    #[staticmethod]
    pub fn n_terminus() -> Self {
        Self {
            inner: VariantTag::NTerminus,
        }
    }

    /// Chain C-terminus patch.
    #[must_use]
    #[staticmethod]
    pub fn c_terminus() -> Self {
        Self {
            inner: VariantTag::CTerminus,
        }
    }

    /// Participates in a disulfide bond.
    #[must_use]
    #[staticmethod]
    pub fn disulfide() -> Self {
        Self {
            inner: VariantTag::Disulfide,
        }
    }

    /// `HID` — delta-protonated histidine.
    #[must_use]
    #[staticmethod]
    pub fn protonation_his_delta() -> Self {
        Self {
            inner: VariantTag::Protonation(ProtonationState::HisDelta),
        }
    }

    /// `HIE` — epsilon-protonated histidine.
    #[must_use]
    #[staticmethod]
    pub fn protonation_his_epsilon() -> Self {
        Self {
            inner: VariantTag::Protonation(ProtonationState::HisEpsilon),
        }
    }

    /// `HIP` — doubly-protonated histidine.
    #[must_use]
    #[staticmethod]
    pub fn protonation_his_doubly() -> Self {
        Self {
            inner: VariantTag::Protonation(ProtonationState::HisDoubly),
        }
    }

    /// Custom protonation state, e.g. `"ASP_PROTONATED"`.
    #[must_use]
    #[staticmethod]
    pub fn protonation_custom(name: String) -> Self {
        Self {
            inner: VariantTag::Protonation(ProtonationState::Custom(name)),
        }
    }

    /// Open-ended variant tag carrying an arbitrary string.
    #[must_use]
    #[staticmethod]
    pub fn other(name: String) -> Self {
        Self {
            inner: VariantTag::Other(name),
        }
    }
}

/// Python handle wrapping a single atom row (position + name +
/// element). Parallels `molex_AtomRow` in C.
#[pyclass(name = "AtomRow", module = "molex")]
#[derive(Clone)]
pub struct PyAtomRow {
    /// X coordinate.
    #[pyo3(get, set)]
    pub position_x: f32,
    /// Y coordinate.
    #[pyo3(get, set)]
    pub position_y: f32,
    /// Z coordinate.
    #[pyo3(get, set)]
    pub position_z: f32,
    /// PDB atom name (4 bytes, space-padded; e.g. `b"CA  "`).
    #[pyo3(get, set)]
    pub name: [u8; 4],
    /// Element symbol (e.g. `"C"`, `"Fe"`).
    #[pyo3(get, set)]
    pub element: String,
}

#[pymethods]
impl PyAtomRow {
    /// Construct an `AtomRow` from positional fields.
    #[must_use]
    #[new]
    pub fn new(
        position_x: f32,
        position_y: f32,
        position_z: f32,
        name: [u8; 4],
        element: String,
    ) -> Self {
        Self {
            position_x,
            position_y,
            position_z,
            name,
            element,
        }
    }
}

impl PyAtomRow {
    fn into_atom(self) -> MolexAtom {
        let element = Element::from_symbol(self.element.trim());
        MolexAtom {
            position: Vec3::new(
                self.position_x,
                self.position_y,
                self.position_z,
            ),
            occupancy: 1.0,
            b_factor: 0.0,
            element,
            name: self.name,
            formal_charge: 0,
        }
    }

    /// Lossy conversion from internal `Atom` (rich) to `PyAtomRow`
    /// (lean: position + name + element symbol only). Loses occupancy
    /// / b_factor / formal_charge -- mirrors the `c_api`
    /// `atom_to_row` helper exactly.
    fn from_atom(a: &MolexAtom) -> Self {
        Self {
            position_x: a.position.x,
            position_y: a.position.y,
            position_z: a.position.z,
            name: a.name,
            element: a.element.symbol().to_owned(),
        }
    }
}
