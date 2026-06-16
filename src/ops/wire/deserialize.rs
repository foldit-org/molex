//! Deserialization for the ASSEM02 binary wire format.
//!
//! Accepts both ASSEM01 (no variants) and ASSEM02 (with per-residue
//! variants). ASSEM01 reads as if every residue had empty variants.

use std::collections::HashSet;

use glam::Vec3;

use super::variants::{deserialize_variants_section, EntityVariants};
use super::{molecule_type_from_wire, ASSEMBLY_MAGIC_V1, ASSEMBLY_MAGIC_V2};
use crate::element::Element;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::bulk::BulkEntity;
use crate::entity::molecule::id::{EntityId, EntityIdAllocator};
use crate::entity::molecule::nucleic_acid::NAEntity;
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::protein::ProteinEntity;
use crate::entity::molecule::small_molecule::SmallMoleculeEntity;
use crate::entity::molecule::{MoleculeEntity, MoleculeType};
use crate::ops::codec::AdapterError;

/// Width of one ASSEM01 atom row in bytes.
const ATOM_ROW_BYTES: usize = 26;

/// One atom row decoded from the wire, paired with the per-atom
/// residue/chain context used to group atoms into residues.
pub(crate) struct AtomRow {
    pub(crate) atom: Atom,
    pub(crate) chain_id: u8,
    pub(crate) res_name: [u8; 3],
    pub(crate) res_num: i32,
}

pub(crate) fn read_atom_row(cursor: &[u8]) -> Result<AtomRow, AdapterError> {
    let x = f32::from_be_bytes(cursor[0..4].try_into().map_err(|_| {
        AdapterError::SerializationError("Invalid x coordinate".to_owned())
    })?);
    let y = f32::from_be_bytes(cursor[4..8].try_into().map_err(|_| {
        AdapterError::SerializationError("Invalid y coordinate".to_owned())
    })?);
    let z = f32::from_be_bytes(cursor[8..12].try_into().map_err(|_| {
        AdapterError::SerializationError("Invalid z coordinate".to_owned())
    })?);

    let chain_id = cursor[12];

    let mut res_name = [0u8; 3];
    res_name.copy_from_slice(&cursor[13..16]);

    let res_num =
        i32::from_be_bytes(cursor[16..20].try_into().map_err(|_| {
            AdapterError::SerializationError(
                "Invalid residue number".to_owned(),
            )
        })?);

    let mut atom_name = [0u8; 4];
    atom_name.copy_from_slice(&cursor[20..24]);

    let sym_str = std::str::from_utf8(&cursor[24..26])
        .unwrap_or("")
        .trim_matches('\0')
        .trim();
    let element = Element::from_symbol(sym_str);

    Ok(AtomRow {
        atom: Atom {
            position: Vec3::new(x, y, z),
            occupancy: 1.0,
            b_factor: 0.0,
            element,
            name: atom_name,
            formal_charge: 0,
        },
        chain_id,
        res_name,
        res_num,
    })
}

/// Read `atom_count` atom rows from a cursor, returning the rows and
/// the remaining bytes.
fn read_atom_rows(
    mut cursor: &[u8],
    atom_count: usize,
) -> Result<(Vec<AtomRow>, &[u8]), AdapterError> {
    let mut rows = Vec::with_capacity(atom_count);
    for _ in 0..atom_count {
        rows.push(read_atom_row(cursor)?);
        cursor = &cursor[ATOM_ROW_BYTES..];
    }
    Ok((rows, cursor))
}

/// Split atom rows into `(Vec<Atom>, Vec<Residue>)` by grouping
/// consecutive rows sharing the same residue number.
fn into_atoms_and_residues(rows: Vec<AtomRow>) -> (Vec<Atom>, Vec<Residue>) {
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

fn build_entity(
    id: EntityId,
    mol_type: MoleculeType,
    rows: Vec<AtomRow>,
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

/// One decoded entity header. For ASSEM01 (no inline id),
/// `entity_id_raw` is `None` and the caller allocates a fresh
/// [`EntityId`]; for ASSEM02 it carries the originator's raw value.
struct EntityHeader {
    mol_type: MoleculeType,
    atom_count: usize,
    entity_id_raw: Option<u32>,
}

/// Result of [`parse_entity_headers`]: the headers themselves and the
/// byte offset immediately past them (where atom rows begin).
struct ParsedHeaders {
    headers: Vec<EntityHeader>,
    headers_end: usize,
}

/// Parse entity headers from ASSEM binary format. When `has_entity_ids`
/// is true (ASSEM02), each header carries an extra 4-byte
/// `entity_id_raw`; otherwise (ASSEM01) the slot is absent.
fn parse_entity_headers(
    bytes: &[u8],
    entity_count: usize,
    has_entity_ids: bool,
) -> Result<ParsedHeaders, AdapterError> {
    let mut headers = Vec::with_capacity(entity_count);
    let mut offset = 12;
    for _ in 0..entity_count {
        let mol_byte = *bytes.get(offset).ok_or_else(|| {
            AdapterError::InvalidFormat(
                "Truncated entity header (expected molecule type)".to_owned(),
            )
        })?;
        let mol_type = molecule_type_from_wire(mol_byte).ok_or_else(|| {
            AdapterError::InvalidFormat(format!(
                "Unknown molecule type byte: {mol_byte}",
            ))
        })?;
        offset += 1;
        let atom_count_bytes =
            bytes.get(offset..offset + 4).ok_or_else(|| {
                AdapterError::InvalidFormat(
                    "Truncated entity header (expected atom count)".to_owned(),
                )
            })?;
        let atom_count =
            u32::from_be_bytes(atom_count_bytes.try_into().map_err(|_| {
                AdapterError::InvalidFormat(
                    "Invalid atom count in entity header".to_owned(),
                )
            })?) as usize;
        offset += 4;

        let entity_id_raw = if has_entity_ids {
            let id_bytes = bytes.get(offset..offset + 4).ok_or_else(|| {
                AdapterError::InvalidFormat(
                    "Truncated entity header (expected entity id)".to_owned(),
                )
            })?;
            let raw =
                u32::from_be_bytes(id_bytes.try_into().map_err(|_| {
                    AdapterError::InvalidFormat(
                        "Invalid entity id in entity header".to_owned(),
                    )
                })?);
            offset += 4;
            Some(raw)
        } else {
            None
        };

        headers.push(EntityHeader {
            mol_type,
            atom_count,
            entity_id_raw,
        });
    }
    Ok(ParsedHeaders {
        headers,
        headers_end: offset,
    })
}

/// Deserialize ASSEM01 binary format into an [`Assembly`].
///
/// Runs [`Assembly::new`] over the decoded entities. The result is
/// secondary-structure-free: `ss_types` is empty until a caller opts in
/// via [`Assembly::recompute_ss`].
///
/// # Errors
///
/// Returns `AdapterError::InvalidFormat` if the magic header, entity headers,
/// or atom data are malformed or truncated. Returns
/// `AdapterError::SerializationError` if individual atom fields cannot be
/// parsed.
///
/// [`Assembly`]: crate::Assembly
/// [`Assembly::new`]: crate::Assembly::new
/// [`Assembly::recompute_ss`]: crate::Assembly::recompute_ss
pub fn deserialize_assembly(
    bytes: &[u8],
) -> Result<crate::Assembly, AdapterError> {
    let entities = deserialize_assembly_entities(bytes)?;
    Ok(crate::Assembly::new(entities))
}

/// Deserialize ASSEM01 bytes into a raw entity vec without building
/// an [`Assembly`].
///
/// Internal helper for in-crate paths that mutate entities in place
/// before re-serializing and don't need an [`Assembly`] wrapper.
///
/// # Errors
///
/// Same as [`deserialize_assembly`].
///
/// [`Assembly`]: crate::Assembly
pub(crate) fn deserialize_assembly_entities(
    bytes: &[u8],
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    if bytes.len() < 12 {
        return Err(AdapterError::InvalidFormat(
            "Data too short for ASSEM header".to_owned(),
        ));
    }

    let magic = &bytes[0..8];
    let has_variants = if magic == ASSEMBLY_MAGIC_V2 {
        true
    } else if magic == ASSEMBLY_MAGIC_V1 {
        false
    } else {
        return Err(AdapterError::InvalidFormat(
            "Invalid magic number for ASSEM binary format (expected ASSEM01 \
             or ASSEM02)"
                .to_owned(),
        ));
    };

    let entity_count =
        u32::from_be_bytes(bytes[8..12].try_into().map_err(|_| {
            AdapterError::InvalidFormat("Invalid entity count".to_owned())
        })?) as usize;

    // Reject a count the remaining bytes cannot possibly hold *before*
    // allocating for it. Each entity contributes at least a fixed-size header
    // (mol_type + atom_count, plus entity_id under ASSEM02), so an overstated
    // or corrupt count would otherwise drive a huge `Vec::with_capacity` and
    // abort the process on allocation failure.
    let min_entity_header = if has_variants { 9 } else { 5 };
    if entity_count > (bytes.len() - 12) / min_entity_header {
        return Err(AdapterError::InvalidFormat(
            "Entity count exceeds available data".to_owned(),
        ));
    }

    let ParsedHeaders {
        headers,
        headers_end,
    } = parse_entity_headers(bytes, entity_count, has_variants)?;

    let total_atoms: usize = headers.iter().map(|h| h.atom_count).sum();
    let atoms_end = headers_end + total_atoms * ATOM_ROW_BYTES;
    if bytes.len() < atoms_end {
        return Err(AdapterError::InvalidFormat(
            "Data too short for atom data".to_owned(),
        ));
    }

    let mut cursor = &bytes[headers_end..atoms_end];
    let mut entities = Vec::with_capacity(entity_count);
    let mut allocator = EntityIdAllocator::new();

    for header in headers {
        let (rows, rest) = read_atom_rows(cursor, header.atom_count)?;
        cursor = rest;
        let id = match header.entity_id_raw {
            Some(raw) => allocator.from_raw(raw),
            None => allocator.allocate(),
        };
        entities.push(build_entity(id, header.mol_type, rows));
    }

    if has_variants {
        let per_entity_variants =
            deserialize_variants_section(&bytes[atoms_end..], entity_count)?;
        for (entity, residue_variants) in
            entities.iter_mut().zip(per_entity_variants)
        {
            attach_variants(entity, &residue_variants);
        }
    }

    Ok(entities)
}

/// Attach decoded variants to a polymer entity by matching
/// `label_seq_id`. Non-polymer entities ignore the input (the wire
/// format always emits an empty list for them).
fn attach_variants(entity: &mut MoleculeEntity, decoded: &EntityVariants) {
    let residues: &mut [Residue] = match entity {
        MoleculeEntity::Protein(e) => &mut e.residues,
        MoleculeEntity::NucleicAcid(e) => &mut e.residues,
        MoleculeEntity::SmallMolecule(_) | MoleculeEntity::Bulk(_) => return,
    };
    for block in decoded {
        let Some(target) = residues
            .iter_mut()
            .find(|r| r.label_seq_id == block.label_seq_id)
        else {
            log::warn!(
                "Variants section references unknown label_seq_id {}; skipping",
                block.label_seq_id,
            );
            continue;
        };
        target.variants.clone_from(&block.variants);
    }
}
