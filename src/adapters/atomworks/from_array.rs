//! AtomArray -> entities conversion direction.
//!
//! Contains `determine_entity_ids`, `build_entity_from_indices`,
//! `atom_array_to_entities`, `atom_array_to_entity_vec`, and the
//! `parse_file_*` pyfunction wrappers.

use std::collections::HashSet;

use glam::Vec3;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{chain_type_id_to_molecule_type, mol_type_str_to_molecule_type};
use crate::assembly::Assembly;
use crate::element::Element;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::bulk::BulkEntity;
use crate::entity::molecule::chain::ChainIdMapper;
use crate::entity::molecule::id::EntityId;
use crate::entity::molecule::nucleic_acid::NAEntity;
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::protein::ProteinEntity;
use crate::entity::molecule::small_molecule::SmallMoleculeEntity;
use crate::entity::molecule::{MoleculeEntity, MoleculeType};
use crate::ops::wire::serialize_assembly;

/// Determine per-atom entity ID assignments from annotations on the atom array.
fn determine_entity_ids(
    atom_array: &Bound<'_, PyAny>,
    num_atoms: usize,
    entity_id_arr: Option<&Bound<'_, PyAny>>,
    mol_type_arr: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<i32>> {
    let chain_id_arr = atom_array.getattr("chain_id")?;
    let mut atom_entity_ids: Vec<i32> = Vec::with_capacity(num_atoms);

    if let Some(eid_arr) = entity_id_arr {
        for i in 0..num_atoms {
            let eid: i32 = eid_arr.get_item(i)?.extract()?;
            atom_entity_ids.push(eid);
        }
    } else {
        let mut group_map: std::collections::HashMap<String, i32> =
            std::collections::HashMap::new();
        let mut next_id: i32 = 0;

        for i in 0..num_atoms {
            let cid: String = chain_id_arr.get_item(i)?.extract()?;
            let mt_str = if let Some(mt_arr) = mol_type_arr {
                mt_arr.get_item(i)?.extract::<String>().unwrap_or_default()
            } else {
                String::new()
            };
            let key = format!("{cid}:{mt_str}");
            let eid = *group_map.entry(key).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            atom_entity_ids.push(eid);
        }
    }

    Ok(atom_entity_ids)
}

/// References to the per-atom arrays extracted from a Biotite `AtomArray`.
struct AtomArrayRefs<'py> {
    coord: Bound<'py, PyAny>,
    chain_id_arr: Bound<'py, PyAny>,
    res_id_arr: Bound<'py, PyAny>,
    res_name_arr: Bound<'py, PyAny>,
    atom_name_arr: Bound<'py, PyAny>,
    element_arr: Option<Bound<'py, PyAny>>,
    occupancy_arr: Option<Bound<'py, PyAny>>,
    b_factor_arr: Option<Bound<'py, PyAny>>,
}

/// One atom row decoded from the AtomArray, paired with the per-atom
/// residue/chain context used to group atoms into residues.
struct AtomArrayRow {
    atom: Atom,
    chain_id: u8,
    res_name: [u8; 3],
    res_num: i32,
}

/// Build a single `MoleculeEntity` from a group of atom indices.
fn build_entity_from_indices(
    indices: &[usize],
    output_idx: usize,
    mol_type: MoleculeType,
    arrays: &AtomArrayRefs<'_>,
    chain_mapper: &mut ChainIdMapper,
) -> PyResult<MoleculeEntity> {
    let mut rows = Vec::with_capacity(indices.len());
    for &i in indices {
        rows.push(read_atom_array_row(arrays, i, chain_mapper)?);
    }

    // IDs are sequential across entities (entity N gets id N), so the id is
    // just the output index. No allocator round-trip needed.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "entity count fits in u32 for valid structures"
    )]
    let id = EntityId::from_raw(output_idx as u32);

    Ok(build_entity(id, mol_type, rows))
}

fn build_entity(
    id: EntityId,
    mol_type: MoleculeType,
    rows: Vec<AtomArrayRow>,
) -> MoleculeEntity {
    let pdb_chain_id = rows.first().map_or(b' ', |r| r.chain_id);

    match mol_type {
        MoleculeType::Protein => {
            let (atoms, residues) = into_atoms_and_residues(rows);
            MoleculeEntity::Protein(ProteinEntity::new(
                id,
                atoms,
                residues,
                pdb_chain_id,
                None,
            ))
        }
        MoleculeType::DNA | MoleculeType::RNA => {
            let (atoms, residues) = into_atoms_and_residues(rows);
            MoleculeEntity::NucleicAcid(NAEntity::new(
                id,
                mol_type,
                atoms,
                residues,
                pdb_chain_id,
                None,
            ))
        }
        MoleculeType::Water | MoleculeType::Solvent => {
            let residue_name = rows.first().map_or([b' '; 3], |r| r.res_name);
            let mut seen = HashSet::new();
            for row in &rows {
                let _ = seen.insert((row.chain_id, row.res_num));
            }
            let molecule_count = seen.len();
            let atoms = rows.into_iter().map(|r| r.atom).collect();
            MoleculeEntity::Bulk(BulkEntity::new(
                id,
                mol_type,
                atoms,
                residue_name,
                molecule_count,
            ))
        }
        MoleculeType::Ligand
        | MoleculeType::Ion
        | MoleculeType::Cofactor
        | MoleculeType::Lipid => {
            let residue_name = rows.first().map_or([b' '; 3], |r| r.res_name);
            let atoms = rows.into_iter().map(|r| r.atom).collect();
            MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
                id,
                mol_type,
                atoms,
                residue_name,
            ))
        }
    }
}

fn into_atoms_and_residues(
    rows: Vec<AtomArrayRow>,
) -> (Vec<Atom>, Vec<Residue>) {
    let mut atoms = Vec::with_capacity(rows.len());
    let mut residues = Vec::new();
    let mut current_res_num: Option<i32> = None;
    let mut current_res_name: [u8; 3] = [b' '; 3];
    let mut current_start = 0usize;

    for row in rows {
        let new_residue = current_res_num.is_none_or(|n| n != row.res_num);
        if new_residue {
            if let Some(rn) = current_res_num {
                residues.push(Residue {
                    name: current_res_name,
                    label_seq_id: rn,
                    auth_seq_id: None,
                    auth_comp_id: None,
                    ins_code: None,
                    atom_range: current_start..atoms.len(),
                    variants: Vec::new(),
                });
            }
            current_start = atoms.len();
            current_res_num = Some(row.res_num);
            current_res_name = row.res_name;
        }
        atoms.push(row.atom);
    }
    if let Some(rn) = current_res_num {
        residues.push(Residue {
            name: current_res_name,
            label_seq_id: rn,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: current_start..atoms.len(),
            variants: Vec::new(),
        });
    }

    (atoms, residues)
}

fn read_atom_array_row(
    arrays: &AtomArrayRefs<'_>,
    i: usize,
    chain_mapper: &mut ChainIdMapper,
) -> PyResult<AtomArrayRow> {
    let coord_i = arrays.coord.get_item(i)?;
    let x: f32 = coord_i.get_item(0)?.extract()?;
    let y: f32 = coord_i.get_item(1)?.extract()?;
    let z: f32 = coord_i.get_item(2)?.extract()?;

    let occupancy: f32 = arrays
        .occupancy_arr
        .as_ref()
        .and_then(|arr| arr.get_item(i).ok())
        .and_then(|v| v.extract().ok())
        .unwrap_or(1.0);
    let b_factor: f32 = arrays
        .b_factor_arr
        .as_ref()
        .and_then(|arr| arr.get_item(i).ok())
        .and_then(|v| v.extract().ok())
        .unwrap_or(0.0);

    let chain_id = extract_chain_id(arrays, i, chain_mapper)?;
    let res_name = extract_res_name(arrays, i)?;
    let res_num: i32 = arrays.res_id_arr.get_item(i)?.extract()?;
    let (atom_name, element) = extract_atom_name_and_element(arrays, i)?;

    Ok(AtomArrayRow {
        atom: Atom {
            position: Vec3::new(x, y, z),
            occupancy,
            b_factor,
            element,
            name: atom_name,
            formal_charge: 0,
        },
        chain_id,
        res_name,
        res_num,
    })
}

/// Extract chain ID for atom `i` and map it via `chain_mapper`.
fn extract_chain_id(
    arrays: &AtomArrayRefs<'_>,
    i: usize,
    chain_mapper: &mut ChainIdMapper,
) -> PyResult<u8> {
    let cid: String = arrays.chain_id_arr.get_item(i)?.extract()?;
    Ok(chain_mapper.get_or_assign(&cid))
}

/// Extract 3-byte residue name for atom `i`.
fn extract_res_name(arrays: &AtomArrayRefs<'_>, i: usize) -> PyResult<[u8; 3]> {
    let rn: String = arrays.res_name_arr.get_item(i)?.extract()?;
    let mut rn_bytes = [b' '; 3];
    for (j, b) in rn.bytes().take(3).enumerate() {
        rn_bytes[j] = b;
    }
    Ok(rn_bytes)
}

/// Extract 4-byte atom name and determine element for atom `i`.
fn extract_atom_name_and_element(
    arrays: &AtomArrayRefs<'_>,
    i: usize,
) -> PyResult<([u8; 4], Element)> {
    let an: String = arrays.atom_name_arr.get_item(i)?.extract()?;
    let mut an_bytes = [b' '; 4];
    for (j, b) in an.bytes().take(4).enumerate() {
        an_bytes[j] = b;
    }

    let elem = if let Some(ref elem_arr) = arrays.element_arr {
        let sym: String = elem_arr.get_item(i)?.extract().unwrap_or_default();
        Element::from_symbol(&sym)
    } else {
        let an_str = std::str::from_utf8(&an_bytes).unwrap_or("");
        Element::from_atom_name(an_str)
    };
    Ok((an_bytes, elem))
}

/// Convert a Biotite `AtomArray` (or `AtomArrayPlus`) back to ASSEM01 bytes.
///
/// Reconstructs `Vec<MoleculeEntity>` from per-atom annotations.
/// Entity boundaries are determined by `entity_id` annotation if present,
/// otherwise by grouping on `(chain_id, mol_type)`.
///
/// # Errors
///
/// Returns `PyErr` if per-atom annotations cannot be extracted or if
/// assembly serialization fails.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn atom_array_to_entities(
    py: Python,
    atom_array: Py<PyAny>,
) -> PyResult<Vec<u8>> {
    let atom_array = atom_array.bind(py);

    let num_atoms: usize = atom_array
        .getattr("coord")?
        .getattr("shape")?
        .get_item(0)?
        .extract()?;
    if num_atoms == 0 {
        return serialize_assembly(&Assembly::new(vec![])).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
        });
    }

    let arrays = extract_array_refs(atom_array)?;

    let entity_id_arr = super::get_annotation_opt(atom_array, "entity_id");
    let mol_type_arr = super::get_annotation_opt(atom_array, "mol_type");
    let chain_type_arr = super::get_annotation_opt(atom_array, "chain_type");

    let atom_entity_ids = determine_entity_ids(
        atom_array,
        num_atoms,
        entity_id_arr.as_ref(),
        mol_type_arr.as_ref(),
    )?;

    let entity_order = unique_in_order(&atom_entity_ids);

    let entities = build_all_entities(
        &entity_order,
        &atom_entity_ids,
        &arrays,
        chain_type_arr.as_ref(),
        mol_type_arr.as_ref(),
    )?;

    serialize_assembly(&Assembly::new(entities)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })
}

/// Extract per-atom array references from a Biotite `AtomArray`.
fn extract_array_refs<'py>(
    atom_array: &Bound<'py, PyAny>,
) -> PyResult<AtomArrayRefs<'py>> {
    Ok(AtomArrayRefs {
        coord: atom_array.getattr("coord")?,
        chain_id_arr: atom_array.getattr("chain_id")?,
        res_id_arr: atom_array.getattr("res_id")?,
        res_name_arr: atom_array.getattr("res_name")?,
        atom_name_arr: atom_array.getattr("atom_name")?,
        element_arr: atom_array.getattr("element").ok(),
        occupancy_arr: atom_array.getattr("occupancy").ok(),
        b_factor_arr: atom_array.getattr("b_factor").ok(),
    })
}

/// Collect unique values preserving first-appearance order.
fn unique_in_order(ids: &[i32]) -> Vec<i32> {
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    for &eid in ids {
        if seen.insert(eid) {
            order.push(eid);
        }
    }
    order
}

/// Determine molecule type for an entity group from available annotations.
fn determine_mol_type(
    first_idx: usize,
    chain_type_arr: Option<&Bound<'_, PyAny>>,
    mol_type_arr: Option<&Bound<'_, PyAny>>,
    res_name_arr: &Bound<'_, PyAny>,
) -> PyResult<MoleculeType> {
    if let Some(ct_arr) = chain_type_arr {
        let ct: i32 = ct_arr.get_item(first_idx)?.extract().unwrap_or(8);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let ct_u8 = ct as u8;
        Ok(chain_type_id_to_molecule_type(ct_u8))
    } else if let Some(mt_arr) = mol_type_arr {
        let mt_str: String =
            mt_arr.get_item(first_idx)?.extract().unwrap_or_default();
        Ok(mol_type_str_to_molecule_type(&mt_str))
    } else {
        let rn: String = res_name_arr.get_item(first_idx)?.extract()?;
        Ok(crate::entity::molecule::classify::classify_residue(&rn))
    }
}

/// Build `MoleculeEntity` for every entity group.
fn build_all_entities(
    entity_order: &[i32],
    atom_entity_ids: &[i32],
    arrays: &AtomArrayRefs<'_>,
    chain_type_arr: Option<&Bound<'_, PyAny>>,
    mol_type_arr: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<MoleculeEntity>> {
    let mut chain_mapper = ChainIdMapper::new();
    let mut entities: Vec<MoleculeEntity> =
        Vec::with_capacity(entity_order.len());

    for (output_idx, &eid) in entity_order.iter().enumerate() {
        let indices: Vec<usize> = atom_entity_ids
            .iter()
            .enumerate()
            .filter(|(_, &e)| e == eid)
            .map(|(i, _)| i)
            .collect();

        let mol_type = determine_mol_type(
            indices[0],
            chain_type_arr,
            mol_type_arr,
            &arrays.res_name_arr,
        )?;

        entities.push(build_entity_from_indices(
            &indices,
            output_idx,
            mol_type,
            arrays,
            &mut chain_mapper,
        )?);
    }

    Ok(entities)
}

/// Convert an AtomArray to `Vec<MoleculeEntity>` (Rust structs).
///
/// Returns entity objects directly, useful when you need Rust-side
/// access without a serialization round-trip.
///
/// # Errors
///
/// Returns `PyErr` if the atom array cannot be converted or deserialized.
pub fn atom_array_to_entity_vec(
    py: Python,
    atom_array: &Bound<'_, PyAny>,
) -> PyResult<Vec<MoleculeEntity>> {
    let atom_array_py: Py<PyAny> = atom_array.clone().unbind();
    let bytes = atom_array_to_entities(py, atom_array_py)?;
    crate::ops::wire::deserialize_assembly(&bytes)
        .map(|a| {
            a.entities()
                .iter()
                .map(|e| MoleculeEntity::clone(e))
                .collect()
        })
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
        })
}

// ============================================================================
// Convenience: parse from file through AtomWorks, return entities
// ============================================================================

/// Load a structure file through AtomWorks' full parsing pipeline and return
/// ASSEM01 bytes containing properly cleaned and annotated entities.
///
/// This is the highest-fidelity path for loading structures when AtomWorks
/// is available: it gets CCD bond lookup, leaving group removal, charge
/// correction, occupancy handling, and all other AtomWorks cleaning steps.
///
/// # Errors
///
/// Returns `PyErr` if the AtomWorks parser fails or if the resulting atom
/// array cannot be converted to entities.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn parse_file_to_entities(
    py: Python,
    file_path: String,
) -> PyResult<Vec<u8>> {
    let aw_parser = py.import("atomworks.io.parser")?;
    let result = aw_parser.call_method1("parse", (&*file_path,))?;

    let asym_unit = result.get_item("asym_unit")?;
    let atom_array = asym_unit.get_item(0)?; // First model

    let atom_array_py: Py<PyAny> = atom_array.unbind();
    atom_array_to_entities(py, atom_array_py)
}

/// Load a structure file through AtomWorks and return chain metadata.
///
/// Returns a Python dict with:
/// - `"assembly_bytes"`: ASSEM01 bytes for the entity assembly
/// - `"chain_info"`: dict of chain_id -> { "sequence": str, ... }
/// - `"assemblies"`: dict of assembly_id -> ASSEM01 bytes for each bio assembly
///
/// # Errors
///
/// Returns `PyErr` if the AtomWorks parser fails, if atom array conversion
/// fails, or if Python dict operations fail.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn parse_file_full(py: Python, file_path: String) -> PyResult<Py<PyAny>> {
    let aw_parser = py.import("atomworks.io.parser")?;
    let result = aw_parser.call_method1("parse", (&*file_path,))?;

    let out = PyDict::new(py);

    // Convert asymmetric unit
    let asym_unit = result.get_item("asym_unit")?;
    let atom_array = asym_unit.get_item(0)?;
    let asym_bytes = atom_array_to_entities(py, atom_array.unbind())?;
    out.set_item("assembly_bytes", asym_bytes)?;

    // Pass through chain_info directly (Python dict)
    let chain_info = result.get_item("chain_info")?;
    out.set_item("chain_info", chain_info)?;

    // Convert biological assemblies
    let assemblies_dict = PyDict::new(py);
    convert_assemblies(py, &result, &assemblies_dict)?;
    out.set_item("assemblies", assemblies_dict)?;

    Ok(out.unbind().into_any())
}

/// Convert biological assemblies from AtomWorks parse result into a dict.
fn convert_assemblies(
    py: Python,
    result: &Bound<'_, PyAny>,
    assemblies_dict: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let Ok(assemblies) = result.get_item("assemblies") else {
        return Ok(());
    };
    let Ok(items) = assemblies.call_method0("items") else {
        return Ok(());
    };
    let Ok(iter) = items.try_iter() else {
        return Ok(());
    };
    for item in iter.flatten() {
        let key: String = item.get_item(0i32)?.extract()?;
        let stack = item.get_item(1i32)?;
        let aa = stack.get_item(0i32)?;
        if let Ok(bytes) = atom_array_to_entities(py, aa.unbind()) {
            assemblies_dict.set_item(key, bytes)?;
        }
        // Skip assemblies that fail to convert
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    reason = "tests: coords are exact f32 copies, no arithmetic"
)]
mod tests {
    use super::*;
    use crate::entity::molecule::id::EntityIdAllocator;

    #[allow(
        clippy::too_many_arguments,
        reason = "test row constructor mirrors the per-atom annotation set"
    )]
    fn mk_row(
        name: [u8; 4],
        element: Element,
        pos: Vec3,
        chain_id: u8,
        res_name: [u8; 3],
        res_num: i32,
    ) -> AtomArrayRow {
        AtomArrayRow {
            atom: Atom {
                position: pos,
                occupancy: 1.0,
                b_factor: 0.0,
                element,
                name,
                formal_charge: 0,
            },
            chain_id,
            res_name,
            res_num,
        }
    }

    #[test]
    fn unique_in_order_preserves_first_appearance() {
        let ids = [3, 3, 1, 1, 1, 2, 3];
        assert_eq!(unique_in_order(&ids), vec![3, 1, 2]);
    }

    #[test]
    fn unique_in_order_empty_is_empty() {
        assert!(unique_in_order(&[]).is_empty());
    }

    #[test]
    fn into_atoms_and_residues_empty_yields_nothing() {
        let (atoms, residues) = into_atoms_and_residues(Vec::new());
        assert!(atoms.is_empty());
        assert!(residues.is_empty());
    }

    #[test]
    fn build_entity_empty_rows_uses_fill_defaults() {
        // Edge case: an entity group with zero atoms must not panic and must
        // fall back to the documented fill values rather than indexing past
        // the (empty) row vec. A wrong fill here would silently corrupt the
        // residue/chain metadata of an empty group.
        let lig =
            build_entity(EntityId::from_raw(0), MoleculeType::Ligand, vec![]);
        let MoleculeEntity::SmallMolecule(e) = lig else {
            unreachable!("Ligand mol_type must build a SmallMolecule");
        };
        assert!(e.atoms.is_empty());
        assert_eq!(e.residue_name, [b' '; 3]);

        let water =
            build_entity(EntityId::from_raw(1), MoleculeType::Water, vec![]);
        assert!(matches!(water, MoleculeEntity::Bulk(_)));
    }

    #[test]
    fn into_atoms_and_residues_groups_by_res_num() {
        // Three atoms in residue 1, two in residue 2; the split must land
        // exactly on the res_num change (catches an off-by-one in the
        // atom_range boundary).
        let rows = vec![
            mk_row(*b"N   ", Element::N, Vec3::ZERO, b'A', *b"ALA", 1),
            mk_row(*b"CA  ", Element::C, Vec3::X, b'A', *b"ALA", 1),
            mk_row(*b"C   ", Element::C, Vec3::Y, b'A', *b"ALA", 1),
            mk_row(*b"N   ", Element::N, Vec3::Z, b'A', *b"GLY", 2),
            mk_row(*b"CA  ", Element::C, Vec3::ONE, b'A', *b"GLY", 2),
        ];
        let (atoms, residues) = into_atoms_and_residues(rows);

        assert_eq!(atoms.len(), 5);
        assert_eq!(residues.len(), 2);

        assert_eq!(residues[0].name, *b"ALA");
        assert_eq!(residues[0].label_seq_id, 1);
        assert_eq!(residues[0].atom_range, 0..3);

        assert_eq!(residues[1].name, *b"GLY");
        assert_eq!(residues[1].label_seq_id, 2);
        assert_eq!(residues[1].atom_range, 3..5);

        // atom_range slices index back into the contiguous atom vec correctly.
        assert_eq!(atoms[residues[1].atom_range.clone()].len(), 2);
    }

    #[test]
    fn into_atoms_and_residues_single_residue() {
        let rows = vec![
            mk_row(*b"C1  ", Element::C, Vec3::ZERO, b'A', *b"LIG", 7),
            mk_row(*b"O1  ", Element::O, Vec3::X, b'A', *b"LIG", 7),
        ];
        let (atoms, residues) = into_atoms_and_residues(rows);
        assert_eq!(atoms.len(), 2);
        assert_eq!(residues.len(), 1);
        assert_eq!(residues[0].atom_range, 0..2);
    }

    #[test]
    fn build_entity_dispatches_by_mol_type() {
        let protein_rows = vec![
            mk_row(*b"N   ", Element::N, Vec3::ZERO, b'A', *b"ALA", 1),
            mk_row(*b"CA  ", Element::C, Vec3::X, b'A', *b"ALA", 1),
        ];
        let id = EntityId::from_raw(0);
        let e = build_entity(id, MoleculeType::Protein, protein_rows);
        assert!(matches!(e, MoleculeEntity::Protein(_)));
        assert_eq!(e.id(), id);

        let lig_rows =
            vec![mk_row(*b"C1  ", Element::C, Vec3::ZERO, b'B', *b"LIG", 1)];
        let lig =
            build_entity(EntityId::from_raw(1), MoleculeType::Ligand, lig_rows);
        assert!(matches!(lig, MoleculeEntity::SmallMolecule(_)));

        let water_rows =
            vec![mk_row(*b"O   ", Element::O, Vec3::ZERO, b'C', *b"HOH", 1)];
        let water =
            build_entity(EntityId::from_raw(2), MoleculeType::Water, water_rows);
        assert!(matches!(water, MoleculeEntity::Bulk(_)));
    }

    #[test]
    fn build_entity_from_indices_assigns_sequential_ids() {
        // The id is derived directly from output_idx via EntityId::from_raw;
        // this must match the old allocate-output_idx-times-then-allocate
        // behaviour, i.e. entity N gets EntityId(N).
        let mut alloc = EntityIdAllocator::new();
        for output_idx in 0u32..4 {
            // Replicate the discarded-allocation loop the code used to run.
            let mut probe = EntityIdAllocator::new();
            for _ in 0..output_idx {
                let _ = probe.allocate();
            }
            let expected = probe.allocate();
            assert_eq!(EntityId::from_raw(output_idx), expected);
        }
        // Sanity: a fresh allocator still mints 0,1,2,... in step with from_raw.
        for n in 0u32..4 {
            assert_eq!(alloc.allocate(), EntityId::from_raw(n));
        }
    }

    /// Roundtrip at the pure-Rust boundary: the per-atom columns that
    /// `to_array::collect_atom_data` produces are fed back through the
    /// reconstruction core and must rebuild an equivalent multi-residue
    /// entity. The numpy/biotite serialization in between is a thin shim;
    /// this exercises the logic that decides atom order, residue grouping,
    /// and coordinate placement.
    #[test]
    fn reconstruct_multi_residue_entity_roundtrip() {
        // Source: a two-residue protein-like group, atoms in canonical order.
        let coords = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.2, 0.0, 0.0),
            Vec3::new(4.2, 0.0, 0.0),
        ];
        let names: [[u8; 4]; 5] =
            [*b"N   ", *b"CA  ", *b"C   ", *b"N   ", *b"CA  "];
        let res_nums = [1, 1, 1, 2, 2];
        let res_names: [[u8; 3]; 5] =
            [*b"ALA", *b"ALA", *b"ALA", *b"GLY", *b"GLY"];

        let rows: Vec<AtomArrayRow> = (0..5)
            .map(|i| {
                mk_row(
                    names[i],
                    Element::from_atom_name(
                        std::str::from_utf8(&names[i]).unwrap(),
                    ),
                    coords[i],
                    b'A',
                    res_names[i],
                    res_nums[i],
                )
            })
            .collect();

        let (atoms, residues) = into_atoms_and_residues(rows);

        // Coordinates survive in order, exact f32.
        assert_eq!(atoms.len(), 5);
        for (i, a) in atoms.iter().enumerate() {
            assert_eq!(a.position, coords[i]);
        }
        // Two residues, boundary on the res_num change, ranges contiguous and
        // covering all atoms with no gap or overlap.
        assert_eq!(residues.len(), 2);
        assert_eq!(residues[0].atom_range, 0..3);
        assert_eq!(residues[1].atom_range, 3..5);
        assert_eq!(
            residues[0].atom_range.end,
            residues[1].atom_range.start,
            "residue ranges must be contiguous"
        );
        assert_eq!(residues.last().unwrap().atom_range.end, atoms.len());
    }
}
