//! Entities -> AtomArray conversion direction.
//!
//! Contains `collect_atom_data`, `entities_to_atom_array_impl`, and all the
//! `entities_to_*` pyfunction wrappers.

use pyo3::prelude::*;

use super::{molecule_type_to_chain_type_id, molecule_type_to_mol_type_str};
use crate::analysis::bonds::{infer_bonds, BondOrder, DEFAULT_TOLERANCE};
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::{MoleculeEntity, MoleculeType};
use crate::ops::wire::deserialize_assembly;

/// Flat per-atom annotation data collected from entities.
pub(crate) struct AtomData {
    pub coords_flat: Vec<f32>,
    pub chain_ids: Vec<String>,
    pub res_ids: Vec<i32>,
    pub res_names: Vec<String>,
    pub atom_names: Vec<String>,
    pub elements: Vec<String>,
    pub occupancies: Vec<f32>,
    pub b_factors: Vec<f32>,
    pub aw_entity_ids: Vec<i32>,
    pub aw_mol_types: Vec<String>,
    pub aw_chain_types: Vec<i32>,
    pub all_bonds: Vec<(usize, usize, u8)>,
}

impl AtomData {
    /// Restrict every parallel per-atom column to `range` (a half-open run of
    /// flat atom indices), returning a fresh `AtomData`. `coords_flat` is keyed
    /// by `3 * range` since it stores three floats per atom. Bonds are dropped:
    /// they index into the full entity's atom space, so a sliced subset has no
    /// well-defined bond list (and the native numpy read path never reads
    /// them). Used by the per-residue array read to keep the single column
    /// core.
    pub(crate) fn slice_atoms(&self, range: std::ops::Range<usize>) -> Self {
        let coords_range = (range.start * 3)..(range.end * 3);
        Self {
            coords_flat: self.coords_flat[coords_range].to_vec(),
            chain_ids: self.chain_ids[range.clone()].to_vec(),
            res_ids: self.res_ids[range.clone()].to_vec(),
            res_names: self.res_names[range.clone()].to_vec(),
            atom_names: self.atom_names[range.clone()].to_vec(),
            elements: self.elements[range.clone()].to_vec(),
            occupancies: self.occupancies[range.clone()].to_vec(),
            b_factors: self.b_factors[range.clone()].to_vec(),
            aw_entity_ids: self.aw_entity_ids[range.clone()].to_vec(),
            aw_mol_types: self.aw_mol_types[range.clone()].to_vec(),
            aw_chain_types: self.aw_chain_types[range].to_vec(),
            all_bonds: Vec::new(),
        }
    }

    fn with_capacity(total_atoms: usize) -> Self {
        Self {
            coords_flat: Vec::with_capacity(total_atoms * 3),
            chain_ids: Vec::with_capacity(total_atoms),
            res_ids: Vec::with_capacity(total_atoms),
            res_names: Vec::with_capacity(total_atoms),
            atom_names: Vec::with_capacity(total_atoms),
            elements: Vec::with_capacity(total_atoms),
            occupancies: Vec::with_capacity(total_atoms),
            b_factors: Vec::with_capacity(total_atoms),
            aw_entity_ids: Vec::with_capacity(total_atoms),
            aw_mol_types: Vec::with_capacity(total_atoms),
            aw_chain_types: Vec::with_capacity(total_atoms),
            all_bonds: Vec::new(),
        }
    }
}

/// One atom as visited by [`for_each_flat_atom`], with the per-atom
/// residue/chain context the column build needs.
///
/// `entity_raw_id` and `raw_idx` together identify the atom within its entity;
/// `flat` index space is just the order in which these are yielded. `chain_id`,
/// `res_name`, and `res_num` are the residue-level values to stamp on this
/// atom's columns (a non-polymer atom carries the entity's residue name, blank
/// chain, and a synthetic per-atom or fixed residue number).
pub(crate) struct FlatAtom<'a> {
    pub entity_raw_id: u32,
    pub raw_idx: usize,
    pub atom: &'a Atom,
    pub chain_id: u8,
    pub res_name: [u8; 3],
    pub res_num: i32,
}

/// Visit every atom of `entities` in the canonical `to_arrays()` order.
///
/// This is the single producer of `to_arrays()` index space: entities in slice
/// order; within a polymer entity, atoms in residue order via each residue's
/// `atom_range` (an atom not referenced by any range is skipped); within a
/// non-polymer entity, raw atom order. Both the column build
/// ([`collect_atom_data`]) and the flat bond-index map
/// (`crate::python::arrays::flat_atoms`) drive this walk so the ordering rule
/// lives in exactly one place.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "atom counts fit in i32/u32 for valid structures"
)]
pub(crate) fn for_each_flat_atom<E: std::borrow::Borrow<MoleculeEntity>>(
    entities: &[E],
    mut f: impl FnMut(FlatAtom<'_>),
) {
    for entity in entities {
        let entity = entity.borrow();
        let entity_raw_id = entity.id().raw();
        let atoms = entity.atom_set();

        match entity {
            MoleculeEntity::Protein(e) => {
                emit_polymer(&mut f, entity_raw_id, atoms, &e.residues, e.pdb_chain_id);
            }
            MoleculeEntity::NucleicAcid(e) => {
                emit_polymer(&mut f, entity_raw_id, atoms, &e.residues, e.pdb_chain_id);
            }
            MoleculeEntity::SmallMolecule(e) => {
                for (raw_idx, atom) in atoms.iter().enumerate() {
                    f(FlatAtom {
                        entity_raw_id,
                        raw_idx,
                        atom,
                        chain_id: b' ',
                        res_name: e.residue_name,
                        res_num: 1,
                    });
                }
            }
            MoleculeEntity::Bulk(e) => {
                for (raw_idx, atom) in atoms.iter().enumerate() {
                    f(FlatAtom {
                        entity_raw_id,
                        raw_idx,
                        atom,
                        chain_id: b' ',
                        res_name: e.residue_name,
                        res_num: (raw_idx as i32) + 1,
                    });
                }
            }
        }
    }
}

fn emit_polymer(
    f: &mut impl FnMut(FlatAtom<'_>),
    entity_raw_id: u32,
    atoms: &[Atom],
    residues: &[Residue],
    chain_id: u8,
) {
    for residue in residues {
        for raw_idx in residue.atom_range.clone() {
            f(FlatAtom {
                entity_raw_id,
                raw_idx,
                atom: &atoms[raw_idx],
                chain_id,
                res_name: residue.name,
                res_num: residue.label_seq_id,
            });
        }
    }
}

/// Collect per-atom annotation data from entities into flat vectors.
pub(crate) fn collect_atom_data<E: std::borrow::Borrow<MoleculeEntity>>(
    entities: &[E],
    total_atoms: usize,
) -> AtomData {
    let mut data = AtomData::with_capacity(total_atoms);
    let mut atom_offset: usize = 0;

    for entity in entities {
        let entity = entity.borrow();
        let mol_type = entity.molecule_type();
        let ctx = EntityCtx {
            entity_id: entity.id().raw().cast_signed(),
            mol_type_str: molecule_type_to_mol_type_str(mol_type),
            chain_type_id: i32::from(molecule_type_to_chain_type_id(mol_type)),
        };
        for_each_flat_atom(std::slice::from_ref(&entity), |fa| {
            append_atom_row(
                &mut data,
                &ctx,
                fa.atom,
                fa.chain_id,
                fa.res_name,
                fa.res_num,
            );
        });
        append_entity_bonds(&mut data, entity, atom_offset);
        atom_offset += entity.atom_count();
    }

    data
}

struct EntityCtx<'a> {
    entity_id: i32,
    mol_type_str: &'a str,
    chain_type_id: i32,
}

#[allow(
    clippy::too_many_arguments,
    reason = "row payload is best passed positionally; bundling into a struct \
              just to satisfy a lint trades clarity for noise"
)]
fn append_atom_row(
    data: &mut AtomData,
    ctx: &EntityCtx<'_>,
    atom: &Atom,
    chain_id: u8,
    res_name: [u8; 3],
    res_num: i32,
) {
    data.coords_flat.push(atom.position.x);
    data.coords_flat.push(atom.position.y);
    data.coords_flat.push(atom.position.z);

    data.chain_ids.push(if chain_id.is_ascii_alphanumeric() {
        String::from(chain_id as char)
    } else {
        "A".to_owned()
    });

    data.res_ids.push(res_num);

    data.res_names.push(
        std::str::from_utf8(&res_name)
            .unwrap_or("UNK")
            .trim()
            .to_owned(),
    );

    data.atom_names.push(
        std::str::from_utf8(&atom.name)
            .unwrap_or("X")
            .trim()
            .to_owned(),
    );

    data.elements.push(atom.element.symbol().to_owned());

    data.occupancies.push(atom.occupancy);
    data.b_factors.push(atom.b_factor);

    data.aw_entity_ids.push(ctx.entity_id);
    data.aw_mol_types.push(ctx.mol_type_str.to_owned());
    data.aw_chain_types.push(ctx.chain_type_id);
}

/// Infer and append bonds for a single entity (ligands/cofactors/ions only).
fn append_entity_bonds(
    data: &mut AtomData,
    entity: &MoleculeEntity,
    atom_offset: usize,
) {
    let needs_inference = matches!(
        entity.molecule_type(),
        MoleculeType::Ligand | MoleculeType::Cofactor | MoleculeType::Ion
    );

    let atoms = entity.atom_set();
    if needs_inference && atoms.len() >= 2 && atoms.len() <= 500 {
        let inferred = infer_bonds(atoms, DEFAULT_TOLERANCE);
        for bond in &inferred {
            let bt = match bond.order {
                BondOrder::Single => 1u8,
                BondOrder::Double => 2,
                BondOrder::Triple => 3,
                BondOrder::Aromatic => 4,
            };
            data.all_bonds.push((
                bond.atom_a + atom_offset,
                bond.atom_b + atom_offset,
                bt,
            ));
        }
    }
}

/// Convert a `Vec<MoleculeEntity>` to an AtomWorks-compatible Biotite
/// `AtomArray`.
///
/// The resulting AtomArray has:
/// - Standard biotite annotations: `coord`, `chain_id`, `res_id`, `res_name`,
///   `atom_name`, `element`, `occupancy`, `b_factor`
/// - AtomWorks annotations: `entity_id` (per-atom int), `mol_type` (per-atom
///   str), `chain_type` (per-atom int matching `atomworks.enums.ChainType`)
/// - `BondList` populated from entity bond data or distance inference
pub(crate) fn entities_to_atom_array_impl<
    E: std::borrow::Borrow<MoleculeEntity>,
>(
    py: Python,
    entities: &[E],
) -> PyResult<Py<PyAny>> {
    let total_atoms: usize =
        entities.iter().map(|e| e.borrow().atom_count()).sum();
    if total_atoms == 0 {
        let biotite = py.import("biotite.structure")?;
        let arr = biotite.getattr("AtomArray")?.call1((0,))?;
        return Ok(arr.unbind());
    }

    let numpy = py.import("numpy")?;
    let biotite = py.import("biotite.structure")?;
    let atom_array = biotite.getattr("AtomArray")?.call1((total_atoms,))?;

    let data = collect_atom_data(entities, total_atoms);

    set_standard_annotations(&atom_array, numpy.as_any(), &data, total_atoms)?;
    set_atomworks_annotations(&atom_array, numpy.as_any(), &data)?;
    set_bond_list(&atom_array, biotite.as_any(), &data, total_atoms)?;

    Ok(atom_array.unbind())
}

/// Set standard Biotite annotations (coord, chain_id, res_id, etc.).
fn set_standard_annotations(
    atom_array: &Bound<'_, PyAny>,
    numpy: &Bound<'_, PyAny>,
    data: &AtomData,
    total_atoms: usize,
) -> PyResult<()> {
    let coord_np = numpy.call_method1("array", (&data.coords_flat,))?;
    let coord_np = coord_np.call_method1("reshape", ((total_atoms, 3),))?;
    let coord_np =
        coord_np.call_method1("astype", (numpy.getattr("float32")?,))?;
    atom_array.setattr("coord", coord_np)?;

    let chain_np = numpy.call_method1("array", (&data.chain_ids,))?;
    atom_array.setattr("chain_id", chain_np)?;

    let res_np = numpy.call_method1("array", (&data.res_ids,))?;
    let res_np = res_np.call_method1("astype", (numpy.getattr("int32")?,))?;
    atom_array.setattr("res_id", res_np)?;

    let resname_np = numpy.call_method1("array", (&data.res_names,))?;
    atom_array.setattr("res_name", resname_np)?;

    let atomname_np = numpy.call_method1("array", (&data.atom_names,))?;
    atom_array.setattr("atom_name", atomname_np)?;

    let element_np = numpy.call_method1("array", (&data.elements,))?;
    atom_array.setattr("element", element_np)?;

    let occ_np = numpy.call_method1("array", (&data.occupancies,))?;
    let occ_np = occ_np.call_method1("astype", (numpy.getattr("float32")?,))?;
    atom_array.setattr("occupancy", occ_np)?;

    let bf_np = numpy.call_method1("array", (&data.b_factors,))?;
    let bf_np = bf_np.call_method1("astype", (numpy.getattr("float32")?,))?;
    atom_array.setattr("b_factor", bf_np)?;

    Ok(())
}

/// Set AtomWorks-specific annotations (entity_id, mol_type, chain_type).
fn set_atomworks_annotations(
    atom_array: &Bound<'_, PyAny>,
    numpy: &Bound<'_, PyAny>,
    data: &AtomData,
) -> PyResult<()> {
    let eid_np = numpy.call_method1("array", (&data.aw_entity_ids,))?;
    let eid_np = eid_np.call_method1("astype", (numpy.getattr("int32")?,))?;
    let _ = atom_array.call_method1("set_annotation", ("entity_id", eid_np))?;

    let mt_np = numpy.call_method1("array", (&data.aw_mol_types,))?;
    let _ = atom_array.call_method1("set_annotation", ("mol_type", mt_np))?;

    let ct_np = numpy.call_method1("array", (&data.aw_chain_types,))?;
    let ct_np = ct_np.call_method1("astype", (numpy.getattr("int32")?,))?;
    let _ = atom_array.call_method1("set_annotation", ("chain_type", ct_np))?;

    Ok(())
}

/// Build and attach a BondList to the AtomArray.
fn set_bond_list(
    atom_array: &Bound<'_, PyAny>,
    biotite: &Bound<'_, PyAny>,
    data: &AtomData,
    total_atoms: usize,
) -> PyResult<()> {
    let bond_list_cls = biotite.getattr("BondList")?;
    let bond_list = bond_list_cls.call1((total_atoms,))?;
    for (a, b, bt) in &data.all_bonds {
        let _ = bond_list.call_method1("add_bond", (*a, *b, *bt))?;
    }
    atom_array.setattr("bonds", bond_list)?;
    Ok(())
}

/// Convert `Vec<MoleculeEntity>` (from assembly wire bytes) to a Biotite
/// `AtomArray`.
///
/// # Errors
///
/// Returns `PyErr` if the assembly bytes cannot be deserialized or if
/// Python/Biotite operations fail.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn entities_to_atom_array(
    py: Python,
    assembly_bytes: Vec<u8>,
) -> PyResult<Py<PyAny>> {
    let assembly = deserialize_assembly(&assembly_bytes).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;
    entities_to_atom_array_impl(py, assembly.entities())
}

/// Convert `Vec<MoleculeEntity>` (from assembly wire bytes) to an
/// `AtomArrayPlus`.
///
/// `AtomArrayPlus` signals to downstream consumers (e.g. `parse_atom_array`)
/// that the structure is already fully constructed and should skip CCD
/// template rebuilding.
///
/// # Errors
///
/// Returns `PyErr` if the assembly bytes cannot be deserialized or if
/// Python/AtomWorks operations fail.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn entities_to_atom_array_plus(
    py: Python,
    assembly_bytes: Vec<u8>,
) -> PyResult<Py<PyAny>> {
    let assembly = deserialize_assembly(&assembly_bytes).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;
    let atom_array = entities_to_atom_array_impl(py, assembly.entities())?;
    let as_plus = py
        .import("atomworks.io.utils.atom_array_plus")?
        .getattr("as_atom_array_plus")?;
    Ok(as_plus.call1((atom_array,))?.unbind())
}

/// Convert assembly wire bytes to a Biotite `AtomArray`.
///
/// Pass assembly wire bytes (the output of `serialize_assembly` /
/// `assembly_bytes`).
///
/// # Errors
///
/// Returns `PyErr` if the bytes cannot be deserialized or if Python/Biotite
/// operations fail.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn assembly_bytes_to_atom_array(
    py: Python,
    bytes: Vec<u8>,
) -> PyResult<Py<PyAny>> {
    let assembly = deserialize_assembly(&bytes).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;
    entities_to_atom_array_impl(py, assembly.entities())
}

/// Convert assembly wire bytes to an `AtomArrayPlus`.
///
/// # Errors
///
/// Returns `PyErr` if the bytes cannot be deserialized or if
/// Python/AtomWorks operations fail.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn assembly_bytes_to_atom_array_plus(
    py: Python,
    bytes: Vec<u8>,
) -> PyResult<Py<PyAny>> {
    let assembly = deserialize_assembly(&bytes).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })?;
    let atom_array = entities_to_atom_array_impl(py, assembly.entities())?;
    let as_plus = py
        .import("atomworks.io.utils.atom_array_plus")?
        .getattr("as_atom_array_plus")?;
    Ok(as_plus.call1((atom_array,))?.unbind())
}

/// Convert `Vec<MoleculeEntity>` to AtomArray, then run through
/// `atomworks.io.parser.parse()` for full cleaning.
///
/// This first writes a temporary CIF/PDB via the existing export path,
/// then lets AtomWorks re-parse it with its full cleaning pipeline
/// (leaving group removal, charge correction, bond order fixing, etc.).
///
/// Use this when you need maximum data quality for model training or
/// when handling structures with known issues (missing atoms, wrong charges).
/// For interactive use where latency matters, prefer `entities_to_atom_array`.
///
/// # Errors
///
/// Returns `PyErr` if the assembly bytes cannot be deserialized, if the
/// AtomWorks parser fails, or if Python operations fail.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn entities_to_atom_array_parsed(
    py: Python,
    assembly_bytes: Vec<u8>,
    source_path: Option<String>,
) -> PyResult<Py<PyAny>> {
    // If we have the original file path, let AtomWorks parse from source
    // (this gets the best cleaning since AtomWorks can read mmCIF directly
    // and apply its full pipeline including CCD bond lookup, leaving group
    // removal, charge correction, etc.)
    if let Some(path) = source_path {
        let aw_parser = py.import("atomworks.io.parser")?;
        let result = aw_parser.call_method1("parse", (path,))?;
        let asym_unit = result.get_item("asym_unit")?;

        // parse() returns an AtomArrayStack; take model 0
        let atom_array = asym_unit.get_item(0)?;
        return Ok(atom_array.unbind());
    }

    // Fallback: convert through our adapter, then apply AtomWorks transforms
    // manually for cleaning. This is less thorough than parsing from file
    // but still better than raw conversion.
    let atom_array = entities_to_atom_array(py, assembly_bytes)?;

    // Try to apply basic AtomWorks cleaning if available
    match py.import("atomworks.io.cleaning") {
        Ok(cleaning) => cleaning
            .call_method1("clean_atom_array", (atom_array.bind(py),))
            .map_or_else(|_| Ok(atom_array.clone_ref(py)), |c| Ok(c.unbind())),
        Err(_) => Ok(atom_array), /* atomworks not installed or no cleaning
                                   * module */
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    reason = "tests: coords are exact f32 copies, no arithmetic"
)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::element::Element;
    use crate::entity::molecule::id::EntityIdAllocator;
    use crate::entity::molecule::protein::ProteinEntity;
    use crate::entity::molecule::small_molecule::SmallMoleculeEntity;

    fn mk_atom(name: [u8; 4], el: Element, pos: Vec3) -> Atom {
        Atom {
            position: pos,
            occupancy: 1.0,
            b_factor: 0.0,
            element: el,
            name,
            formal_charge: 0,
        }
    }

    /// Two-residue protein (ALA-GLY) laid out in canonical atom order so
    /// `ProteinEntity::new` does not reorder it.
    fn dipeptide(id: crate::entity::molecule::id::EntityId) -> MoleculeEntity {
        let atoms = vec![
            mk_atom(*b"N   ", Element::N, Vec3::new(0.0, 0.0, 0.0)),
            mk_atom(*b"CA  ", Element::C, Vec3::new(1.0, 0.0, 0.0)),
            mk_atom(*b"C   ", Element::C, Vec3::new(2.0, 0.0, 0.0)),
            mk_atom(*b"O   ", Element::O, Vec3::new(2.0, 1.0, 0.0)),
            mk_atom(*b"CB  ", Element::C, Vec3::new(1.0, -1.0, 0.0)),
            mk_atom(*b"N   ", Element::N, Vec3::new(3.2, 0.0, 0.0)),
            mk_atom(*b"CA  ", Element::C, Vec3::new(4.2, 0.0, 0.0)),
            mk_atom(*b"C   ", Element::C, Vec3::new(5.2, 0.0, 0.0)),
            mk_atom(*b"O   ", Element::O, Vec3::new(5.2, 1.0, 0.0)),
        ];
        let residues = vec![
            Residue {
                name: *b"ALA",
                label_seq_id: 1,
                auth_seq_id: None,
                auth_comp_id: None,
                ins_code: None,
                atom_range: 0..5,
                variants: Vec::new(),
            },
            Residue {
                name: *b"GLY",
                label_seq_id: 2,
                auth_seq_id: None,
                auth_comp_id: None,
                ins_code: None,
                atom_range: 5..9,
                variants: Vec::new(),
            },
        ];
        MoleculeEntity::Protein(ProteinEntity::new(
            id, atoms, residues, b'A', None,
        ))
    }

    /// A three-atom ligand (no internal residue structure).
    fn ligand(id: crate::entity::molecule::id::EntityId) -> MoleculeEntity {
        let atoms = vec![
            mk_atom(*b"C1  ", Element::C, Vec3::new(10.0, 0.0, 0.0)),
            mk_atom(*b"O1  ", Element::O, Vec3::new(11.0, 0.0, 0.0)),
            mk_atom(*b"N1  ", Element::N, Vec3::new(12.0, 0.0, 0.0)),
        ];
        MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
            id,
            MoleculeType::Ligand,
            atoms,
            *b"LIG",
        ))
    }

    #[test]
    fn collect_atom_data_empty_entities() {
        let entities: Vec<MoleculeEntity> = vec![];
        let data = collect_atom_data(&entities, 0);
        assert!(data.coords_flat.is_empty());
        assert!(data.chain_ids.is_empty());
        assert!(data.res_ids.is_empty());
        assert!(data.aw_entity_ids.is_empty());
        assert!(data.all_bonds.is_empty());
    }

    #[test]
    fn collect_atom_data_flat_columns_match_entities() {
        let mut alloc = EntityIdAllocator::new();
        let p = dipeptide(alloc.allocate());
        let l = ligand(alloc.allocate());
        let p_id = p.id().raw().cast_signed();
        let l_id = l.id().raw().cast_signed();
        let total: usize = p.atom_count() + l.atom_count();
        let entities = vec![p, l];

        let data = collect_atom_data(&entities, total);

        // Every parallel column has one entry per atom.
        assert_eq!(data.coords_flat.len(), total * 3);
        assert_eq!(data.chain_ids.len(), total);
        assert_eq!(data.res_ids.len(), total);
        assert_eq!(data.res_names.len(), total);
        assert_eq!(data.atom_names.len(), total);
        assert_eq!(data.elements.len(), total);
        assert_eq!(data.occupancies.len(), total);
        assert_eq!(data.b_factors.len(), total);
        assert_eq!(data.aw_entity_ids.len(), total);
        assert_eq!(data.aw_mol_types.len(), total);
        assert_eq!(data.aw_chain_types.len(), total);

        // Coords are stored row-major (x, y, z) per atom in entity order:
        // protein atoms first, then ligand atoms. Walking the entities in
        // order must reproduce coords_flat exactly (catches a transposed or
        // column-major axis).
        let mut flat = Vec::with_capacity(total * 3);
        for e in &entities {
            for a in e.atom_set() {
                flat.push(a.position.x);
                flat.push(a.position.y);
                flat.push(a.position.z);
            }
        }
        assert_eq!(data.coords_flat, flat);

        // entity_id column is segmented by entity: first N atoms carry the
        // protein id, the rest carry the ligand id.
        let n_prot = entities[0].atom_count();
        for &eid in &data.aw_entity_ids[..n_prot] {
            assert_eq!(eid, p_id);
        }
        for &eid in &data.aw_entity_ids[n_prot..] {
            assert_eq!(eid, l_id);
        }

        // res_id mapping: protein residues are 1 (ALA) then 2 (GLY); each
        // value repeats once per atom in that residue. Ligand atoms all map
        // to res_id 1.
        let prot_res = entities[0].residues().unwrap();
        let mut expected_res_ids = Vec::new();
        for r in prot_res {
            for _ in r.atom_range.clone() {
                expected_res_ids.push(r.label_seq_id);
            }
        }
        expected_res_ids
            .extend(std::iter::repeat_n(1, entities[1].atom_count()));
        assert_eq!(data.res_ids, expected_res_ids);

        // res_name column: trimmed 3-char codes, segmented by residue.
        assert_eq!(&data.res_names[..n_prot - 4], &["ALA"; 5]);
        assert_eq!(&data.res_names[n_prot - 4..n_prot], &["GLY"; 4]);
        assert_eq!(&data.res_names[n_prot..], &["LIG"; 3]);

        // chain_id: alphanumeric pdb_chain_id renders to its char; the
        // ligand has no chain so falls back to "A".
        for c in &data.chain_ids[..n_prot] {
            assert_eq!(c, "A");
        }

        // mol_type / chain_type annotations match the source classification.
        assert!(data.aw_mol_types[..n_prot].iter().all(|s| s == "protein"));
        assert!(data.aw_mol_types[n_prot..].iter().all(|s| s == "ligand"));
        assert!(data.aw_chain_types[..n_prot].iter().all(|&c| c == 6));
        assert!(data.aw_chain_types[n_prot..].iter().all(|&c| c == 8));
    }

    #[test]
    fn slice_atoms_extracts_one_residue_run() {
        // The dipeptide lays out ALA (atoms 0..5) then GLY (atoms 5..9). The
        // per-residue read path collects the whole entity, then slices the flat
        // columns to the residue's contiguous run at offset = sum of prior
        // residues' atom counts. Slicing to GLY's run (offset 5, count 4) must
        // reproduce exactly the GLY atoms and only those.
        let mut alloc = EntityIdAllocator::new();
        let p = dipeptide(alloc.allocate());
        let total = p.atom_count();
        let entities = vec![p];
        let data = collect_atom_data(&entities, total);

        // GLY is residue index 1: offset = ALA's 5 atoms, count = GLY's 4.
        let gly = data.slice_atoms(5..9);

        assert_eq!(gly.coords_flat.len(), 4 * 3);
        assert_eq!(gly.coords_flat, data.coords_flat[15..27]);
        assert_eq!(gly.res_names, vec!["GLY"; 4]);
        assert_eq!(gly.res_ids, vec![2, 2, 2, 2]);
        assert_eq!(gly.atom_names, vec!["N", "CA", "C", "O"]);
        assert_eq!(gly.elements, vec!["N", "C", "C", "O"]);
        // Slicing drops bonds (they index the full entity's atom space).
        assert!(gly.all_bonds.is_empty());
    }

    #[test]
    fn collect_atom_data_preserves_atom_order_and_names() {
        let mut alloc = EntityIdAllocator::new();
        let l = ligand(alloc.allocate());
        let total = l.atom_count();
        let entities = vec![l];
        let data = collect_atom_data(&entities, total);

        // Atom names are trimmed and appear in stored order.
        assert_eq!(data.atom_names, vec!["C1", "O1", "N1"]);
        assert_eq!(data.elements, vec!["C", "O", "N"]);
        // Default occupancy/b_factor flow through unchanged.
        assert_eq!(data.occupancies, vec![1.0, 1.0, 1.0]);
        assert_eq!(data.b_factors, vec![0.0, 0.0, 0.0]);
    }
}
