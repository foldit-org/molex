//! AtomArray -> entities conversion direction.
//!
//! Contains `determine_entity_ids`, `build_entity_from_indices`,
//! `atom_array_to_entities` and the `parse_file_*` pyfunction wrappers.

use std::collections::HashSet;

use compact_str::CompactString;
use glam::Vec3;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{chain_type_id_to_molecule_type, mol_type_str_to_molecule_type};
use crate::adapters::table::{empty_entity, AtomTable};
use crate::assembly::Assembly;
use crate::element::Element;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::id::EntityId;
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

/// Build a single `MoleculeEntity` from a group of atom indices.
///
/// IDs are sequential across entities (entity N gets id N), so each row is
/// tagged with `output_idx` and `AtomTable::into_entities` mints
/// `EntityId::from_raw(output_idx)` for the one entity it yields.
fn build_entity_from_indices(
    indices: &[usize],
    output_idx: usize,
    mol_type: MoleculeType,
    arrays: &AtomArrayRefs<'_>,
) -> PyResult<MoleculeEntity> {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "entity count fits in u32 for valid structures"
    )]
    let entity_id = output_idx as u32;

    let mut table = AtomTable::with_capacity(indices.len());
    for &i in indices {
        push_atom_array_row(&mut table, arrays, i, entity_id, mol_type)?;
    }

    // Every entity group carries at least one atom (groups derive from
    // filtering a non-empty per-atom id list), so into_entities yields
    // exactly one entity. A zero-atom group falls back to the shared empty
    // entity of the right type (no chain id; biotite groups carry none here),
    // matching the prior per-entity constructor's behaviour.
    Ok(table.into_entities().pop().unwrap_or_else(|| {
        empty_entity(EntityId::from_raw(entity_id), mol_type, String::new())
    }))
}

fn push_atom_array_row(
    table: &mut AtomTable,
    arrays: &AtomArrayRefs<'_>,
    i: usize,
    entity_id: u32,
    mol_type: MoleculeType,
) -> PyResult<()> {
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

    let chain_id = extract_chain_id(arrays, i)?;
    let res_name = extract_res_name(arrays, i)?;
    let res_num: i32 = arrays.res_id_arr.get_item(i)?.extract()?;
    let (atom_name, element) = extract_atom_name_and_element(arrays, i)?;

    table.push_row(
        &Atom {
            position: Vec3::new(x, y, z),
            occupancy,
            b_factor,
            element,
            name: atom_name,
            formal_charge: 0,
            observed: true,
        },
        entity_id,
        CompactString::new(chain_id),
        mol_type,
        res_num,
        res_name,
    );
    Ok(())
}

/// Extract chain ID string for atom `i`.
fn extract_chain_id(arrays: &AtomArrayRefs<'_>, i: usize) -> PyResult<String> {
    arrays.chain_id_arr.get_item(i)?.extract()
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

/// Convert a Biotite `AtomArray` (or `AtomArrayPlus`) back to assembly wire
/// bytes.
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
            &indices, output_idx, mol_type, arrays,
        )?);
    }

    Ok(entities)
}

// Convenience: parse from file through AtomWorks, return entities

/// Load a structure file through AtomWorks' full parsing pipeline and return
/// assembly wire bytes containing properly cleaned and annotated entities.
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
/// - `"assembly_bytes"`: assembly wire bytes for the entity assembly
/// - `"chain_info"`: dict of chain_id -> { "sequence": str, ... }
/// - `"assemblies"`: dict of assembly_id -> assembly wire bytes for each bio
///   assembly
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
        // Sanity: a fresh allocator still mints 0,1,2,... in step with
        // from_raw.
        for n in 0u32..4 {
            assert_eq!(alloc.allocate(), EntityId::from_raw(n));
        }
    }
}
