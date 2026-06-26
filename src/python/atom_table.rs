//! Biotite-agnostic, bidirectional columnar atom interchange.
//!
//! [`PyAtomTable`] wraps a native [`AtomTable`] and exposes it to Python as
//! per-atom numpy columns (egress) plus row-by-row construction from numpy
//! columns (ingress), so a plugin can round-trip atom data through molex
//! without molex importing Biotite. Egress marshals through the shared de-vocab
//! [`columns`] layer; `atom_names` and `res_names` instead reinterpret the
//! contiguous fixed-width name buffers as numpy `'S4'`/`'S3'` byte-string
//! arrays in one allocation.
//!
//! Atom order is the canonical flat-walk order of [`AtomTable::from_entities`].
//! For a table built from assembly bytes (already-canonical entities) every
//! getter, [`PyAtomTable::bonds`], and the [`PyAtomTable::to_assembly_bytes`]
//! round-trip share that one order. A table built from non-canonical column
//! input is canonicalized only when partitioned into entities (bytes / bonds),
//! so its egress getters can differ from that order until it is round-tripped.

use std::collections::HashMap;

use compact_str::CompactString;
use glam::Vec3;
use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::value_err;
use crate::adapters::atomworks::{
    chain_type_id_to_molecule_type, columns, mol_type_str_to_molecule_type,
};
use crate::adapters::table::AtomTable;
use crate::analysis::bonds::{infer_bonds, BondOrder, DEFAULT_TOLERANCE};
use crate::assembly::Assembly;
use crate::element::Element;
use crate::entity::molecule::atom::{pad_name, Atom};
use crate::entity::molecule::classify::classify_residue;
use crate::entity::molecule::MoleculeType;
use crate::ops::wire::{deserialize_assembly, serialize_assembly};

/// A flat per-atom table that round-trips through molex without Biotite.
///
/// Holds a native [`AtomTable`]. Construct it from assembly wire bytes
/// ([`Self::from_assembly_bytes`]) or from already-extracted numpy columns
/// ([`Self::from_columns`]); read it back as per-atom numpy columns via the
/// getters; emit it as assembly wire bytes ([`Self::to_assembly_bytes`]) or as
/// inferred covalent bonds ([`Self::bonds`]).
#[pyclass(name = "PyAtomTable", module = "molex")]
pub struct PyAtomTable {
    table: AtomTable,
}

#[pymethods]
impl PyAtomTable {
    /// Decode assembly wire bytes into a table.
    ///
    /// # Errors
    ///
    /// `PyValueError` if the bytes cannot be deserialized.
    #[staticmethod]
    pub fn from_assembly_bytes(bytes: &[u8]) -> PyResult<Self> {
        let assembly = deserialize_assembly(bytes).map_err(value_err)?;
        Ok(Self {
            table: AtomTable::from_entities(assembly.entities()),
        })
    }

    /// Build a table row-by-row from per-atom numpy columns.
    ///
    /// `coords` is `(N, 3)` float32; `res_ids`/`entity_ids`/`chain_types` are
    /// `(N,)` int32; `occupancies`/`b_factors` are `(N,)` float32; `observed`
    /// is `(N,)` bool. `atom_names`/`res_names`/`elements`/`mol_types` are each
    /// a length-`N` sequence whose elements are `str` or fixed-width `bytes` (a
    /// numpy `'S'` array works); names are left-justified space-padded into the
    /// 4-/3-byte buffers and elements parse via [`Element::from_symbol`].
    /// `chain_ids` is a length-`N` sequence of `str`.
    ///
    /// Only `coords`, `atom_names`, `elements`, `res_ids`, `res_names`, and
    /// `chain_ids` are required. Absent `occupancies`/`b_factors` default to
    /// `1.0`/`0.0`; absent `observed` defaults to `true`.
    ///
    /// Each atom's [`MoleculeType`] follows the AtomWorks three-tier
    /// precedence: `chain_types` (via the chain-type vocab) if given, else
    /// `mol_types` (via the mol-type vocab), else classification of the residue
    /// name. So an annotation-free protein backbone classifies as protein
    /// rather than ligand.
    ///
    /// When `entity_ids` is omitted, per-atom entity ids are synthesized from
    /// the `(chain_id, mol_type)` pair in first-appearance order. Atoms are
    /// grouped contiguously by entity id (preserving intra-entity order) before
    /// the shared [`AtomTable::into_entities`] partition runs.
    ///
    /// # Errors
    ///
    /// `PyValueError` if any present column's length differs from `N`, if
    /// `coords` is not `(N, 3)`, or if a column element cannot be read.
    #[staticmethod]
    #[pyo3(signature = (
        coords,
        atom_names,
        elements,
        res_ids,
        res_names,
        chain_ids,
        occupancies = None,
        b_factors = None,
        mol_types = None,
        chain_types = None,
        entity_ids = None,
        observed = None,
    ))]
    #[allow(
        clippy::too_many_arguments,
        clippy::needless_pass_by_value,
        clippy::too_many_lines,
        reason = "one parameter per atom column; pyo3 entry point takes the \
                  numpy handles by value"
    )]
    pub fn from_columns(
        coords: PyReadonlyArray2<'_, f32>,
        atom_names: &Bound<'_, PyAny>,
        elements: &Bound<'_, PyAny>,
        res_ids: PyReadonlyArray1<'_, i32>,
        res_names: &Bound<'_, PyAny>,
        chain_ids: Vec<String>,
        occupancies: Option<PyReadonlyArray1<'_, f32>>,
        b_factors: Option<PyReadonlyArray1<'_, f32>>,
        mol_types: Option<&Bound<'_, PyAny>>,
        chain_types: Option<PyReadonlyArray1<'_, i32>>,
        entity_ids: Option<PyReadonlyArray1<'_, i32>>,
        observed: Option<PyReadonlyArray1<'_, bool>>,
    ) -> PyResult<Self> {
        let coords_shape = coords.shape();
        let n = coords_shape[0];
        if coords_shape.get(1) != Some(&3) {
            return Err(value_err(format!(
                "coords must be (N, 3); got {coords_shape:?}"
            )));
        }
        let coords = coords.as_slice()?;

        let atom_name_tokens = read_str_column(atom_names, n, "atom_names")?;
        let element_tokens = read_str_column(elements, n, "elements")?;
        let res_name_tokens = read_str_column(res_names, n, "res_names")?;
        let mol_type_tokens = match mol_types {
            Some(arr) => Some(read_str_column(arr, n, "mol_types")?),
            None => None,
        };
        check_len(chain_ids.len(), n, "chain_ids")?;
        let res_ids = checked_slice(&res_ids, n, "res_ids")?;
        let occupancies = match occupancies.as_ref() {
            Some(arr) => Some(checked_slice(arr, n, "occupancies")?),
            None => None,
        };
        let b_factors = match b_factors.as_ref() {
            Some(arr) => Some(checked_slice(arr, n, "b_factors")?),
            None => None,
        };
        let chain_types = match chain_types.as_ref() {
            Some(arr) => Some(checked_slice(arr, n, "chain_types")?),
            None => None,
        };
        let observed = match observed.as_ref() {
            Some(arr) => Some(checked_slice(arr, n, "observed")?),
            None => None,
        };

        let mol_types = classify_mol_types(
            n,
            chain_types,
            mol_type_tokens.as_deref(),
            &res_name_tokens,
        );

        let per_atom_eid = match entity_ids.as_ref() {
            Some(arr) => checked_slice(arr, n, "entity_ids")?.to_vec(),
            None => synthesize_entity_ids(&chain_ids, &mol_types),
        };

        let mut table = AtomTable::with_capacity(n);
        for (output_idx, eid) in
            unique_in_order(&per_atom_eid).into_iter().enumerate()
        {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "entity count fits in u32 for valid structures"
            )]
            let raw_id = output_idx as u32;
            for i in (0..n).filter(|&i| per_atom_eid[i] == eid) {
                let atom = Atom {
                    position: Vec3::new(
                        coords[i * 3],
                        coords[i * 3 + 1],
                        coords[i * 3 + 2],
                    ),
                    occupancy: occupancies.map_or(1.0, |s| s[i]),
                    b_factor: b_factors.map_or(0.0, |s| s[i]),
                    element: parse_element(
                        &element_tokens[i],
                        &atom_name_tokens[i],
                    ),
                    name: pad_name::<4>(&atom_name_tokens[i]),
                    formal_charge: 0,
                    observed: observed.is_none_or(|o| o[i]),
                };
                table.push_row(
                    &atom,
                    raw_id,
                    CompactString::new(&chain_ids[i]),
                    mol_types[i],
                    res_ids[i],
                    pad_name::<3>(&res_name_tokens[i]),
                );
            }
        }

        Ok(Self { table })
    }

    /// `(N, 3)` float32 coordinates.
    ///
    /// # Errors
    ///
    /// `PyErr` if a numpy operation fails.
    #[getter]
    fn coords(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let n = self.table.len();
        let arr = columns::coords_flat(&self.table, 0..n)
            .into_pyarray(py)
            .reshape((n, 3))?;
        Ok(arr.into_any().unbind())
    }

    /// Per-atom atom names as a numpy `'S4'` byte-string array, trailing
    /// spaces trimmed (e.g. `b"CA"`).
    ///
    /// # Errors
    ///
    /// `PyErr` if a numpy operation fails.
    #[getter]
    fn atom_names(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        fixed_width_bytes::<4>(py, &self.table.name, "S4")
    }

    /// Per-atom residue names as a numpy `'S3'` byte-string array, trailing
    /// spaces trimmed (e.g. `b"ALA"`).
    ///
    /// # Errors
    ///
    /// `PyErr` if a numpy operation fails.
    #[getter]
    fn res_names(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        fixed_width_bytes::<3>(py, &self.table.res_name, "S3")
    }

    /// Per-atom element symbols as a numpy str array.
    ///
    /// # Errors
    ///
    /// `PyErr` if a numpy operation fails.
    #[getter]
    fn elements(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let n = self.table.len();
        str_array(py, columns::elements(&self.table, 0..n))
    }

    /// Per-atom residue ids (int32).
    #[getter]
    fn res_ids(&self, py: Python<'_>) -> Py<PyAny> {
        let n = self.table.len();
        columns::res_ids(&self.table, 0..n)
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    /// Per-atom chain ids as a numpy str array (empty chains fall back to
    /// `"A"`, matching the numpy read path).
    ///
    /// # Errors
    ///
    /// `PyErr` if a numpy operation fails.
    #[getter]
    fn chain_ids(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let n = self.table.len();
        str_array(py, columns::chain_ids(&self.table, 0..n))
    }

    /// Per-atom occupancies (float32).
    #[getter]
    fn occupancies(&self, py: Python<'_>) -> Py<PyAny> {
        let n = self.table.len();
        columns::occupancies(&self.table, 0..n)
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    /// Per-atom B-factors (float32).
    #[getter]
    fn b_factors(&self, py: Python<'_>) -> Py<PyAny> {
        let n = self.table.len();
        columns::b_factors(&self.table, 0..n)
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    /// Per-atom entity ids (int32).
    #[getter]
    fn entity_ids(&self, py: Python<'_>) -> Py<PyAny> {
        let n = self.table.len();
        columns::entity_ids(&self.table, 0..n)
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    /// Per-atom AtomWorks `mol_type` strings as a numpy str array.
    ///
    /// # Errors
    ///
    /// `PyErr` if a numpy operation fails.
    #[getter]
    fn mol_types(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let n = self.table.len();
        str_array(py, columns::mol_types(&self.table, 0..n))
    }

    /// Per-atom AtomWorks `chain_type` codes (int32).
    #[getter]
    fn chain_types(&self, py: Python<'_>) -> Py<PyAny> {
        let n = self.table.len();
        columns::chain_types(&self.table, 0..n)
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    /// Per-atom provenance mask (bool): `true` parsed, `false` fabricated.
    #[getter]
    fn observed_mask(&self, py: Python<'_>) -> Py<PyAny> {
        self.table
            .observed_mask
            .clone()
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    /// Emit the table as assembly wire bytes.
    ///
    /// # Errors
    ///
    /// `PyValueError` propagated from the serializer.
    pub fn to_assembly_bytes(&self) -> PyResult<Vec<u8>> {
        let entities = self.table.clone().into_entities();
        serialize_assembly(&Assembly::new(entities)).map_err(value_err)
    }

    /// Inferred covalent bonds as flat-atom-index triples `(i, j, order)`.
    ///
    /// Distance-inferred bonds for ligand/cofactor/ion entities of 2..=500
    /// atoms, with endpoints in the table's flat atom order. `order` is 1
    /// single, 2 double, 3 triple, 4 aromatic. Polymer template bonds are not
    /// included.
    #[must_use]
    pub fn bonds(&self) -> Vec<(usize, usize, u8)> {
        let entities = self.table.clone().into_entities();
        let mut out = Vec::new();
        let mut offset = 0usize;
        for entity in &entities {
            let needs_inference = matches!(
                entity.molecule_type(),
                MoleculeType::Ligand
                    | MoleculeType::Cofactor
                    | MoleculeType::Ion
            );
            if needs_inference && (2..=500).contains(&entity.atom_count()) {
                let atoms = entity.columns().to_atoms();
                for bond in &infer_bonds(&atoms, DEFAULT_TOLERANCE) {
                    let order = match bond.order {
                        BondOrder::Single => 1u8,
                        BondOrder::Double => 2,
                        BondOrder::Triple => 3,
                        BondOrder::Aromatic => 4,
                    };
                    out.push((
                        bond.atom_a + offset,
                        bond.atom_b + offset,
                        order,
                    ));
                }
            }
            offset += AtomTable::flat_atom_count(entity);
        }
        out
    }

    /// Number of atoms.
    #[must_use]
    pub fn __len__(&self) -> usize {
        self.table.len()
    }

    /// `<PyAtomTable: {n} atoms>`.
    #[must_use]
    pub fn __repr__(&self) -> String {
        format!("<PyAtomTable: {} atoms>", self.table.len())
    }
}

/// Build a numpy str array from owned `String`s via `numpy.array`.
fn str_array(py: Python<'_>, values: Vec<String>) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("numpy")?
        .call_method1("array", (values,))?
        .unbind())
}

/// Reinterpret a contiguous `[u8; N]` name buffer as a numpy `dtype` (`'S4'` /
/// `'S3'`) byte-string array in one allocation.
///
/// numpy's `'S'` arrays strip trailing NULs but not trailing spaces, while the
/// numpy read path trims trailing spaces; a per-element NUL-for-space copy of
/// the (left-justified) buffer keeps the visible values byte-identical to that
/// trimmed output without allocating per element.
fn fixed_width_bytes<const N: usize>(
    py: Python<'_>,
    names: &[[u8; N]],
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let mut buf = Vec::with_capacity(N * names.len());
    for name in names {
        let mut row = *name;
        for slot in row.iter_mut().rev() {
            if *slot == b' ' {
                *slot = 0;
            } else {
                break;
            }
        }
        buf.extend_from_slice(&row);
    }
    let bytes = PyBytes::new(py, &buf);
    Ok(py
        .import("numpy")?
        .call_method1("frombuffer", (bytes, dtype))?
        .unbind())
}

/// Read a length-`name` numpy/sequence column of `str`-or-`bytes` into owned
/// `String`s, erroring if its length is not `n`.
fn read_str_column(
    arr: &Bound<'_, PyAny>,
    n: usize,
    name: &str,
) -> PyResult<Vec<String>> {
    check_len(arr.len()?, n, name)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(token_to_string(&arr.get_item(i)?)?);
    }
    Ok(out)
}

/// Decode one column element to a `String`, accepting a Python `str` or a
/// fixed-width `bytes` (numpy `'S'` scalar; trailing NULs already dropped).
fn token_to_string(item: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = item.extract::<String>() {
        return Ok(s);
    }
    let bytes: Vec<u8> = item.extract()?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Parse an element from its symbol, falling back to atom-name inference when
/// the symbol is unrecognized (e.g. a blank element column).
fn parse_element(symbol: &str, atom_name: &str) -> Element {
    let element = Element::from_symbol(symbol);
    if element == Element::Unknown {
        Element::from_atom_name(atom_name)
    } else {
        element
    }
}

/// Per-atom [`MoleculeType`] via the AtomWorks three-tier precedence (the
/// column-input form of `determine_mol_type`): `chain_types` if given, else
/// `mol_types`, else residue-name classification.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "chain-type codes are small non-negative ids; mirrors the \
              AtomWorks bridge's i32 -> u8 narrowing"
)]
fn classify_mol_types(
    n: usize,
    chain_types: Option<&[i32]>,
    mol_types: Option<&[String]>,
    res_names: &[String],
) -> Vec<MoleculeType> {
    (0..n)
        .map(|i| match (chain_types, mol_types) {
            (Some(ct), _) => chain_type_id_to_molecule_type(ct[i] as u8),
            (None, Some(mt)) => mol_type_str_to_molecule_type(&mt[i]),
            (None, None) => classify_residue(res_names[i].trim()),
        })
        .collect()
}

/// Per-atom entity ids synthesized from the `(chain_id, mol_type)` pair in
/// first-appearance order — the column-input form of the AtomWorks
/// `determine_entity_ids` grouping.
fn synthesize_entity_ids(
    chain_ids: &[String],
    mol_types: &[MoleculeType],
) -> Vec<i32> {
    let mut group_map: HashMap<String, i32> = HashMap::new();
    let mut next_id = 0i32;
    chain_ids
        .iter()
        .zip(mol_types)
        .map(|(cid, mt)| {
            let key = format!("{cid}:{mt:?}");
            *group_map.entry(key).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect()
}

/// Unique values of `ids` in first-appearance order.
fn unique_in_order(ids: &[i32]) -> Vec<i32> {
    let mut seen = std::collections::HashSet::new();
    ids.iter().copied().filter(|&id| seen.insert(id)).collect()
}

/// Error unless `got == want`, naming `column` in the message.
fn check_len(got: usize, want: usize, column: &str) -> PyResult<()> {
    if got == want {
        Ok(())
    } else {
        Err(value_err(format!(
            "{column}: length {got} != atom count {want}"
        )))
    }
}

/// Read a 1-D numpy array as a contiguous slice, erroring if its length is not
/// `n`.
fn checked_slice<'a, T: numpy::Element>(
    arr: &'a PyReadonlyArray1<'_, T>,
    n: usize,
    column: &str,
) -> PyResult<&'a [T]> {
    let slice = arr.as_slice()?;
    check_len(slice.len(), n, column)?;
    Ok(slice)
}
