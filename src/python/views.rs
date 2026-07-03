//! Edit-view pyclasses returned by `EditList`'s read accessors (parallel
//! `molex::SetEntityCoordsView` et al. in C++). A submodule of the `python`
//! module, split out to keep each file under the 800-line source cap enforced
//! by `just file-lengths`.

use pyo3::prelude::*;

use super::{PyAtomRow, PyVariant};

/// Python view of a `SetEntityCoords` edit. Returned by
/// `EditList.set_entity_coords_at`.
#[pyclass(name = "SetEntityCoordsView", module = "molex", skip_from_py_object)]
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
#[pyclass(name = "SetResidueCoordsView", module = "molex", skip_from_py_object)]
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
#[pyclass(name = "MutateResidueView", module = "molex", skip_from_py_object)]
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
#[pyclass(name = "SetVariantsView", module = "molex", skip_from_py_object)]
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
