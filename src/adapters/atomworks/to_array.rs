//! Entities -> AtomArray conversion direction.
//!
//! Contains `entities_to_atom_array_impl` and all the `entities_to_*`
//! pyfunction wrappers.

use pyo3::prelude::*;

use super::columns;
use crate::adapters::table::AtomTable;
use crate::analysis::bonds::{infer_bonds, BondOrder, DEFAULT_TOLERANCE};
use crate::entity::molecule::{MoleculeEntity, MoleculeType};
use crate::ops::wire::deserialize_assembly;

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

    let table = AtomTable::from_entities(entities);

    set_standard_annotations(&atom_array, numpy.as_any(), &table, total_atoms)?;
    set_atomworks_annotations(&atom_array, numpy.as_any(), &table)?;
    set_bond_list(&atom_array, biotite.as_any(), entities, total_atoms)?;

    Ok(atom_array.unbind())
}

/// Set standard Biotite annotations (coord, chain_id, res_id, etc.), marshaled
/// off the native `table` through the shared de-vocab [`columns`] layer.
fn set_standard_annotations(
    atom_array: &Bound<'_, PyAny>,
    numpy: &Bound<'_, PyAny>,
    table: &AtomTable,
    total_atoms: usize,
) -> PyResult<()> {
    let full = 0..total_atoms;

    let coord_np = numpy
        .call_method1("array", (columns::coords_flat(table, full.clone()),))?;
    let coord_np = coord_np.call_method1("reshape", ((total_atoms, 3),))?;
    let coord_np =
        coord_np.call_method1("astype", (numpy.getattr("float32")?,))?;
    atom_array.setattr("coord", coord_np)?;

    let chain_np = numpy
        .call_method1("array", (columns::chain_ids(table, full.clone()),))?;
    atom_array.setattr("chain_id", chain_np)?;

    let res_np = numpy
        .call_method1("array", (columns::res_ids(table, full.clone()),))?;
    let res_np = res_np.call_method1("astype", (numpy.getattr("int32")?,))?;
    atom_array.setattr("res_id", res_np)?;

    let resname_np = numpy
        .call_method1("array", (columns::res_names(table, full.clone()),))?;
    atom_array.setattr("res_name", resname_np)?;

    let atomname_np = numpy
        .call_method1("array", (columns::atom_names(table, full.clone()),))?;
    atom_array.setattr("atom_name", atomname_np)?;

    let element_np = numpy
        .call_method1("array", (columns::elements(table, full.clone()),))?;
    atom_array.setattr("element", element_np)?;

    let occ_np = numpy
        .call_method1("array", (columns::occupancies(table, full.clone()),))?;
    let occ_np = occ_np.call_method1("astype", (numpy.getattr("float32")?,))?;
    atom_array.setattr("occupancy", occ_np)?;

    let bf_np =
        numpy.call_method1("array", (columns::b_factors(table, full),))?;
    let bf_np = bf_np.call_method1("astype", (numpy.getattr("float32")?,))?;
    atom_array.setattr("b_factor", bf_np)?;

    Ok(())
}

/// Set AtomWorks-specific annotations (entity_id, mol_type, chain_type),
/// marshaled off the native `table` through the shared de-vocab [`columns`]
/// layer.
fn set_atomworks_annotations(
    atom_array: &Bound<'_, PyAny>,
    numpy: &Bound<'_, PyAny>,
    table: &AtomTable,
) -> PyResult<()> {
    let full = 0..table.len();

    let eid_np = numpy
        .call_method1("array", (columns::entity_ids(table, full.clone()),))?;
    let eid_np = eid_np.call_method1("astype", (numpy.getattr("int32")?,))?;
    let _ = atom_array.call_method1("set_annotation", ("entity_id", eid_np))?;

    let mt_np = numpy
        .call_method1("array", (columns::mol_types(table, full.clone()),))?;
    let _ = atom_array.call_method1("set_annotation", ("mol_type", mt_np))?;

    let ct_np =
        numpy.call_method1("array", (columns::chain_types(table, full),))?;
    let ct_np = ct_np.call_method1("astype", (numpy.getattr("int32")?,))?;
    let _ = atom_array.call_method1("set_annotation", ("chain_type", ct_np))?;

    Ok(())
}

/// Infer bonds for ligand/cofactor/ion entities and attach the resulting
/// `BondList` to the AtomArray. Bonds are a Biotite-only concern: endpoints are
/// reported per-entity, offset into flat `to_arrays()` index space by the
/// running atom count (the same flat order [`AtomTable::from_entities`] lays
/// atoms out).
#[allow(
    clippy::cast_possible_truncation,
    reason = "atom counts fit in i32 for valid structures"
)]
fn set_bond_list<E: std::borrow::Borrow<MoleculeEntity>>(
    atom_array: &Bound<'_, PyAny>,
    biotite: &Bound<'_, PyAny>,
    entities: &[E],
    total_atoms: usize,
) -> PyResult<()> {
    let bond_list_cls = biotite.getattr("BondList")?;
    let bond_list = bond_list_cls.call1((total_atoms,))?;

    let mut atom_offset: usize = 0;
    for entity in entities {
        let entity = entity.borrow();
        let needs_inference = matches!(
            entity.molecule_type(),
            MoleculeType::Ligand | MoleculeType::Cofactor | MoleculeType::Ion
        );
        let count = entity.atom_count();
        if needs_inference && (2..=500).contains(&count) {
            let atoms = entity.columns().to_atoms();
            for bond in &infer_bonds(&atoms, DEFAULT_TOLERANCE) {
                let bt = match bond.order {
                    BondOrder::Single => 1u8,
                    BondOrder::Double => 2,
                    BondOrder::Triple => 3,
                    BondOrder::Aromatic => 4,
                };
                let _ = bond_list.call_method1(
                    "add_bond",
                    (bond.atom_a + atom_offset, bond.atom_b + atom_offset, bt),
                )?;
            }
        }
        atom_offset += count;
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
    use crate::entity::molecule::atom::Atom;
    use crate::entity::molecule::id::EntityIdAllocator;
    use crate::entity::molecule::polymer::Residue;
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
            observed: true,
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
            id,
            atoms,
            residues,
            "A".to_owned(),
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
    fn columns_empty_entities() {
        let entities: Vec<MoleculeEntity> = vec![];
        let table = AtomTable::from_entities(&entities);
        assert_eq!(table.len(), 0);
        assert!(columns::coords_flat(&table, 0..0).is_empty());
        assert!(columns::chain_ids(&table, 0..0).is_empty());
        assert!(columns::res_ids(&table, 0..0).is_empty());
        assert!(columns::entity_ids(&table, 0..0).is_empty());
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one assertion per de-vocab column over the fixture"
    )]
    fn columns_flat_match_entities() {
        let mut alloc = EntityIdAllocator::new();
        let p = dipeptide(alloc.allocate());
        let l = ligand(alloc.allocate());
        let p_id = p.id().raw().cast_signed();
        let l_id = l.id().raw().cast_signed();
        let total: usize = p.atom_count() + l.atom_count();
        let entities = vec![p, l];

        let table = AtomTable::from_entities(&entities);
        let full = 0..total;
        let coords_flat = columns::coords_flat(&table, full.clone());
        let chain_ids = columns::chain_ids(&table, full.clone());
        let res_ids = columns::res_ids(&table, full.clone());
        let res_names = columns::res_names(&table, full.clone());
        let atom_names = columns::atom_names(&table, full.clone());
        let elements = columns::elements(&table, full.clone());
        let occupancies = columns::occupancies(&table, full.clone());
        let b_factors = columns::b_factors(&table, full.clone());
        let aw_entity_ids = columns::entity_ids(&table, full.clone());
        let aw_mol_types = columns::mol_types(&table, full.clone());
        let aw_chain_types = columns::chain_types(&table, full);

        // Every parallel column has one entry per atom.
        assert_eq!(coords_flat.len(), total * 3);
        assert_eq!(chain_ids.len(), total);
        assert_eq!(res_ids.len(), total);
        assert_eq!(res_names.len(), total);
        assert_eq!(atom_names.len(), total);
        assert_eq!(elements.len(), total);
        assert_eq!(occupancies.len(), total);
        assert_eq!(b_factors.len(), total);
        assert_eq!(aw_entity_ids.len(), total);
        assert_eq!(aw_mol_types.len(), total);
        assert_eq!(aw_chain_types.len(), total);

        // Coords are stored row-major (x, y, z) per atom in entity order:
        // protein atoms first, then ligand atoms. Walking the entities in
        // order must reproduce coords_flat exactly (catches a transposed or
        // column-major axis).
        let mut flat = Vec::with_capacity(total * 3);
        for e in &entities {
            for p in e.positions() {
                flat.push(p.x);
                flat.push(p.y);
                flat.push(p.z);
            }
        }
        assert_eq!(coords_flat, flat);

        // entity_id column is segmented by entity: first N atoms carry the
        // protein id, the rest carry the ligand id.
        let n_prot = entities[0].atom_count();
        for &eid in &aw_entity_ids[..n_prot] {
            assert_eq!(eid, p_id);
        }
        for &eid in &aw_entity_ids[n_prot..] {
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
        assert_eq!(res_ids, expected_res_ids);

        // res_name column: trimmed 3-char codes, segmented by residue.
        assert_eq!(&res_names[..n_prot - 4], &["ALA"; 5]);
        assert_eq!(&res_names[n_prot - 4..n_prot], &["GLY"; 4]);
        assert_eq!(&res_names[n_prot..], &["LIG"; 3]);

        // chain_id: the polymer carries its real label_asym_id; the
        // ligand has no chain so falls back to "A".
        for c in &chain_ids[..n_prot] {
            assert_eq!(c, "A");
        }

        // mol_type / chain_type annotations match the source classification.
        assert!(aw_mol_types[..n_prot].iter().all(|s| s == "protein"));
        assert!(aw_mol_types[n_prot..].iter().all(|s| s == "ligand"));
        assert!(aw_chain_types[..n_prot].iter().all(|&c| c == 6));
        assert!(aw_chain_types[n_prot..].iter().all(|&c| c == 8));
    }

    #[test]
    fn columns_range_extracts_one_residue_run() {
        // The dipeptide lays out ALA (atoms 0..5) then GLY (atoms 5..9). The
        // per-residue read path marshals the flat columns over the residue's
        // contiguous run at offset = sum of prior residues' atom counts.
        // Marshaling GLY's run (offset 5, count 4) must reproduce exactly the
        // GLY atoms and only those.
        let mut alloc = EntityIdAllocator::new();
        let p = dipeptide(alloc.allocate());
        let entities = vec![p];
        let table = AtomTable::from_entities(&entities);

        // GLY is residue index 1: offset = ALA's 5 atoms, count = GLY's 4.
        let gly = 5..9;
        let coords = columns::coords_flat(&table, gly.clone());
        assert_eq!(coords.len(), 4 * 3);
        assert_eq!(coords, columns::coords_flat(&table, 0..9)[15..27]);
        assert_eq!(columns::res_names(&table, gly.clone()), vec!["GLY"; 4]);
        assert_eq!(columns::res_ids(&table, gly.clone()), vec![2, 2, 2, 2]);
        assert_eq!(
            columns::atom_names(&table, gly.clone()),
            vec!["N", "CA", "C", "O"]
        );
        assert_eq!(columns::elements(&table, gly), vec!["N", "C", "C", "O"]);
    }

    #[test]
    fn columns_preserve_atom_order_and_names() {
        let mut alloc = EntityIdAllocator::new();
        let l = ligand(alloc.allocate());
        let total = l.atom_count();
        let entities = vec![l];
        let table = AtomTable::from_entities(&entities);
        let full = 0..total;

        // Atom names are trimmed and appear in stored order.
        assert_eq!(
            columns::atom_names(&table, full.clone()),
            vec!["C1", "O1", "N1"]
        );
        assert_eq!(
            columns::elements(&table, full.clone()),
            vec!["C", "O", "N"]
        );
        // Default occupancy/b_factor flow through unchanged.
        assert_eq!(
            columns::occupancies(&table, full.clone()),
            vec![1.0, 1.0, 1.0]
        );
        assert_eq!(columns::b_factors(&table, full), vec![0.0, 0.0, 0.0]);
    }
}
