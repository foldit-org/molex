//! Deserialization for the assembly binary wire format.
//!
//! Reads the 8-byte magic, then a `u8` version byte. Each entity's chain
//! id lives in its header and atom rows are 25 bytes. Any version other
//! than [`ASSEMBLY_VERSION`] is a clean [`AdapterError::InvalidFormat`].

use compact_str::CompactString;
use glam::Vec3;

use super::variants::{deserialize_variants_section, EntityVariants};
use super::{molecule_type_from_wire, ASSEMBLY_MAGIC, ASSEMBLY_VERSION};
use crate::adapters::table::{empty_entity, AtomTable};
use crate::element::Element;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::id::EntityId;
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::{MoleculeEntity, MoleculeType};
use crate::ops::error::AdapterError;

/// Atom-row width: the chain id lives in the entity header, not the row.
const ATOM_ROW_BYTES: usize = 25;

/// Byte offset where per-entity headers begin: 8-byte magic + 1-byte
/// version + 4-byte entity count.
const HEADERS_START: usize = 13;

/// Decode one wire atom row into its [`Atom`] payload plus the residue keys
/// (name + number) the partition groups on. `occupancy`/`b_factor` are not on
/// the wire and reset to their defaults; `formal_charge` and `observed`
/// likewise reset (the wire carries neither).
pub(crate) fn read_atom_row(
    cursor: &[u8],
) -> Result<(Atom, [u8; 3], i32), AdapterError> {
    let x = f32::from_be_bytes(cursor[0..4].try_into().map_err(|_| {
        AdapterError::SerializationError("Invalid x coordinate".to_owned())
    })?);
    let y = f32::from_be_bytes(cursor[4..8].try_into().map_err(|_| {
        AdapterError::SerializationError("Invalid y coordinate".to_owned())
    })?);
    let z = f32::from_be_bytes(cursor[8..12].try_into().map_err(|_| {
        AdapterError::SerializationError("Invalid z coordinate".to_owned())
    })?);

    let rest_off = 12;

    let mut res_name = [0u8; 3];
    res_name.copy_from_slice(&cursor[rest_off..rest_off + 3]);

    let res_num = i32::from_be_bytes(
        cursor[rest_off + 3..rest_off + 7].try_into().map_err(|_| {
            AdapterError::SerializationError(
                "Invalid residue number".to_owned(),
            )
        })?,
    );

    let mut atom_name = [0u8; 4];
    atom_name.copy_from_slice(&cursor[rest_off + 7..rest_off + 11]);

    let sym_str = std::str::from_utf8(&cursor[rest_off + 11..rest_off + 13])
        .unwrap_or("")
        .trim_matches('\0')
        .trim();
    let element = Element::from_symbol(sym_str);

    Ok((
        Atom {
            position: Vec3::new(x, y, z),
            occupancy: 1.0,
            b_factor: 0.0,
            element,
            name: atom_name,
            formal_charge: 0,
            observed: true,
        },
        res_name,
        res_num,
    ))
}

/// Decode one entity's `atom_count` rows into an [`AtomTable`], tagging every
/// row with the entity's header keys, then partition into the single
/// `MoleculeEntity` and advance the cursor past the consumed rows.
fn build_entity<'a>(
    mut cursor: &'a [u8],
    header: &EntityHeader,
) -> Result<(MoleculeEntity, &'a [u8]), AdapterError> {
    let mut table = AtomTable::with_capacity(header.atom_count);
    let chain = CompactString::new(&header.chain_id);
    for _ in 0..header.atom_count {
        let (atom, res_name, res_num) = read_atom_row(cursor)?;
        table.push_row(
            &atom,
            header.entity_id_raw,
            chain.clone(),
            header.mol_type,
            res_num,
            res_name,
        );
        cursor = &cursor[ATOM_ROW_BYTES..];
    }

    // A zero-atom entity yields nothing from the partition; fall back to the
    // shared empty-entity constructor for the header's molecule type so the
    // entity count still matches the header count (the variants zip below
    // depends on it), carrying the header's chain id for polymers.
    let entity = table.into_entities().pop().unwrap_or_else(|| {
        empty_entity(
            EntityId::from_raw(header.entity_id_raw),
            header.mol_type,
            header.chain_id.clone(),
        )
    });
    Ok((entity, cursor))
}

/// One decoded entity header, carrying the originator's raw `EntityId`
/// value so cross-boundary edit references resolve. `chain_id` is the
/// entity's chain id (empty for non-polymer entities).
struct EntityHeader {
    mol_type: MoleculeType,
    atom_count: usize,
    entity_id_raw: u32,
    chain_id: String,
}

/// Result of [`parse_entity_headers`]: the headers themselves and the
/// byte offset immediately past them (where atom rows begin).
struct ParsedHeaders {
    headers: Vec<EntityHeader>,
    headers_end: usize,
}

/// Parse entity headers from the assembly binary format. Each header is
/// `1 + 4 + 4` bytes (molecule type, atom count, raw entity id) followed
/// by a `u16` BE chain length and the chain bytes.
fn parse_entity_headers(
    bytes: &[u8],
    entity_count: usize,
) -> Result<ParsedHeaders, AdapterError> {
    let mut headers = Vec::with_capacity(entity_count);
    let mut offset = HEADERS_START;
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

        let id_bytes = bytes.get(offset..offset + 4).ok_or_else(|| {
            AdapterError::InvalidFormat(
                "Truncated entity header (expected entity id)".to_owned(),
            )
        })?;
        let entity_id_raw =
            u32::from_be_bytes(id_bytes.try_into().map_err(|_| {
                AdapterError::InvalidFormat(
                    "Invalid entity id in entity header".to_owned(),
                )
            })?);
        offset += 4;

        let (chain_id, next) = read_header_chain(bytes, offset)?;
        offset = next;

        headers.push(EntityHeader {
            mol_type,
            atom_count,
            entity_id_raw,
            chain_id,
        });
    }
    Ok(ParsedHeaders {
        headers,
        headers_end: offset,
    })
}

/// Read a length-prefixed chain string at `offset`: a `u16` BE byte length
/// followed by that many UTF-8 bytes. Returns the string and the offset
/// past it.
fn read_header_chain(
    bytes: &[u8],
    offset: usize,
) -> Result<(String, usize), AdapterError> {
    let len_bytes = bytes.get(offset..offset + 2).ok_or_else(|| {
        AdapterError::InvalidFormat(
            "Truncated entity header (expected chain length)".to_owned(),
        )
    })?;
    let len = usize::from(u16::from_be_bytes(len_bytes.try_into().map_err(
        |_| AdapterError::InvalidFormat("Invalid chain length".to_owned()),
    )?));
    let start = offset + 2;
    let chain_bytes = bytes.get(start..start + len).ok_or_else(|| {
        AdapterError::InvalidFormat(
            "Truncated entity header (chain string)".to_owned(),
        )
    })?;
    let chain = std::str::from_utf8(chain_bytes)
        .map_err(|_| {
            AdapterError::InvalidFormat(
                "Chain id is not valid UTF-8".to_owned(),
            )
        })?
        .to_owned();
    Ok((chain, start + len))
}

/// Deserialize the assembly binary format into an [`Assembly`].
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

/// Deserialize assembly bytes into a raw entity vec without building
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
    if bytes.len() < HEADERS_START {
        return Err(AdapterError::InvalidFormat(
            "Data too short for assembly header".to_owned(),
        ));
    }

    if &bytes[0..8] != ASSEMBLY_MAGIC {
        return Err(AdapterError::InvalidFormat(
            "Invalid magic number for assembly binary format".to_owned(),
        ));
    }

    if bytes[8] != ASSEMBLY_VERSION {
        return Err(AdapterError::InvalidFormat(format!(
            "Unsupported assembly wire version: {}",
            bytes[8],
        )));
    }

    let entity_count =
        u32::from_be_bytes(bytes[9..13].try_into().map_err(|_| {
            AdapterError::InvalidFormat("Invalid entity count".to_owned())
        })?) as usize;

    // Reject a count the remaining bytes cannot possibly hold *before*
    // allocating for it. Each entity contributes at least a fixed-size header
    // (mol_type + atom_count + entity_id = 9 bytes), so an overstated or
    // corrupt count would otherwise drive a huge `Vec::with_capacity` and
    // abort the process on allocation failure.
    if entity_count > (bytes.len() - HEADERS_START) / 9 {
        return Err(AdapterError::InvalidFormat(
            "Entity count exceeds available data".to_owned(),
        ));
    }

    let ParsedHeaders {
        headers,
        headers_end,
    } = parse_entity_headers(bytes, entity_count)?;

    let total_atoms: usize = headers.iter().map(|h| h.atom_count).sum();
    let atoms_end = headers_end + total_atoms * ATOM_ROW_BYTES;
    if bytes.len() < atoms_end {
        return Err(AdapterError::InvalidFormat(
            "Data too short for atom data".to_owned(),
        ));
    }

    let mut cursor = &bytes[headers_end..atoms_end];
    let mut entities = Vec::with_capacity(entity_count);

    for header in headers {
        let (entity, rest) = build_entity(cursor, &header)?;
        cursor = rest;
        entities.push(entity);
    }

    let per_entity_variants =
        deserialize_variants_section(&bytes[atoms_end..], entity_count)?;
    for (entity, residue_variants) in
        entities.iter_mut().zip(per_entity_variants)
    {
        attach_variants(entity, &residue_variants);
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
