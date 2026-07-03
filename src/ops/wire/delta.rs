//! Delta binary wire format: serialized `Vec<AssemblyEdit>`.
//!
//! Companion to the assembly format: where the assembly format carries a
//! full Assembly snapshot, the delta format carries an incremental edit
//! list (the steady-state path). Topology edits (`AddEntity` /
//! `RemoveEntity`) are intentionally not supported on the wire; callers
//! fall back to a full assembly broadcast for that class of change.
//!
//! The header is a fixed 8-byte magic followed by a `u8` version byte
//! that selects the payload layout; the magic never changes when the
//! payload does.
//!
//! Format:
//!
//! ```text
//! 8 bytes  magic b"DELTA\0\0\0"
//! 1 byte   version (currently 1)
//! 4 bytes  edit_count u32 BE
//! per edit:
//!   1 byte tag
//!   tag-specific payload
//! ```
//!
//! Edit tags:
//!
//! - `0x01` SetEntityCoords: u32 entity_id, u32 coord_count, 12 bytes per coord
//!   (xyz f32 BE)
//! - `0x02` SetResidueCoords: u32 entity_id, u32 residue_idx, u32 coord_count,
//!   12 bytes per coord
//! - `0x03` MutateResidue: u32 entity_id, u32 residue_idx, 3 bytes new_name,
//!   u32 atom_count, 25 bytes per atom (same row layout as the assembly
//!   format), u32 variant_count, per-variant payload (same as the assembly
//!   variants block)
//! - `0x04` SetVariants: u32 entity_id, u32 residue_idx, u32 variant_count,
//!   per-variant payload

use glam::Vec3;
use thiserror::Error;

use super::deserialize::read_atom_row;
use super::serialize::write_atom_row;
use super::variants::{read_variant, write_variant};
use crate::chemistry::variant::VariantTag;
use crate::entity::molecule::atom::AtomRef;
use crate::entity::molecule::id::EntityIdAllocator;
use crate::ops::edit::AssemblyEdit;
use crate::ops::error::AdapterError;

/// Magic header for delta byte streams. The version byte that follows
/// selects the payload layout.
pub const DELTA_MAGIC: &[u8; 8] = b"DELTA\0\0\0";

/// Wire format version written after [`DELTA_MAGIC`].
pub const DELTA_VERSION: u8 = 1;

const ATOM_ROW_BYTES: usize = 25;

/// Byte offset where edits begin: 8-byte magic + 1-byte version + 4-byte
/// edit count.
const EDITS_START: usize = 13;

const TAG_SET_ENTITY_COORDS: u8 = 0x01;
const TAG_SET_RESIDUE_COORDS: u8 = 0x02;
const TAG_MUTATE_RESIDUE: u8 = 0x03;
const TAG_SET_VARIANTS: u8 = 0x04;

/// Failure modes for delta serialization.
#[derive(Debug, Error)]
pub enum DeltaSerializeError {
    /// The edit list contained a topology-changing edit
    /// (`AddEntity` / `RemoveEntity`) which is not representable in the
    /// delta format. Callers should broadcast a full assembly snapshot
    /// instead.
    #[error(
        "topology edit at index {index} cannot be serialized as a delta; \
         broadcast a full assembly snapshot instead"
    )]
    TopologyEditNotSupported {
        /// Index of the offending edit in the input slice.
        index: usize,
    },
}

/// Serialize a list of [`AssemblyEdit`]s as delta bytes.
///
/// # Errors
///
/// Returns [`DeltaSerializeError::TopologyEditNotSupported`] if any
/// edit is `AddEntity` / `RemoveEntity`.
pub fn serialize_edits(
    edits: &[AssemblyEdit],
) -> Result<Vec<u8>, DeltaSerializeError> {
    let mut buffer = Vec::with_capacity(16 + edits.len() * 32);
    buffer.extend_from_slice(DELTA_MAGIC);
    buffer.push(DELTA_VERSION);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "edit count fits in u32"
    )]
    buffer.extend_from_slice(&(edits.len() as u32).to_be_bytes());

    for (index, edit) in edits.iter().enumerate() {
        write_edit(edit, index, &mut buffer)?;
    }

    Ok(buffer)
}

/// Deserialize delta bytes into a list of [`AssemblyEdit`]s.
///
/// # Errors
///
/// Returns `AdapterError::InvalidFormat` for magic / version / framing /
/// truncation issues.
pub fn deserialize_edits(
    bytes: &[u8],
) -> Result<Vec<AssemblyEdit>, AdapterError> {
    if bytes.len() < EDITS_START {
        return Err(AdapterError::InvalidFormat(
            "Data too short for delta header".to_owned(),
        ));
    }
    if &bytes[0..8] != DELTA_MAGIC {
        return Err(AdapterError::InvalidFormat(
            "Invalid magic number for delta format".to_owned(),
        ));
    }
    let version = bytes[8];
    if version != DELTA_VERSION {
        return Err(AdapterError::InvalidFormat(format!(
            "Unsupported delta wire version: {version}",
        )));
    }
    let edit_count =
        u32::from_be_bytes(bytes[9..13].try_into().map_err(|_| {
            AdapterError::InvalidFormat("Invalid edit count".to_owned())
        })?) as usize;

    let mut cursor = &bytes[EDITS_START..];
    let mut edits = Vec::with_capacity(edit_count);
    let mut alloc = EntityIdAllocator::new();
    for _ in 0..edit_count {
        let (edit, rest) = read_edit(cursor, &mut alloc)?;
        cursor = rest;
        edits.push(edit);
    }
    Ok(edits)
}

#[allow(
    clippy::too_many_lines,
    reason = "linear match over AssemblyEdit variants; each arm is a tag + \
              field-encoding block. Splitting per-arm into helpers would \
              obscure the variant-to-wire mapping that's the whole point of \
              this function."
)]
fn write_edit(
    edit: &AssemblyEdit,
    index: usize,
    buffer: &mut Vec<u8>,
) -> Result<(), DeltaSerializeError> {
    match edit {
        AssemblyEdit::SetEntityCoords { entity, coords } => {
            buffer.push(TAG_SET_ENTITY_COORDS);
            buffer.extend_from_slice(&entity.raw().to_be_bytes());
            write_coord_list(coords, buffer);
        }
        AssemblyEdit::SetResidueCoords {
            entity,
            residue_idx,
            coords,
        } => {
            buffer.push(TAG_SET_RESIDUE_COORDS);
            buffer.extend_from_slice(&entity.raw().to_be_bytes());
            #[allow(
                clippy::cast_possible_truncation,
                reason = "residue index fits in u32"
            )]
            buffer.extend_from_slice(&(*residue_idx as u32).to_be_bytes());
            write_coord_list(coords, buffer);
        }
        AssemblyEdit::MutateResidue {
            entity,
            residue_idx,
            new_name,
            new_atoms,
            new_variants,
        } => {
            buffer.push(TAG_MUTATE_RESIDUE);
            buffer.extend_from_slice(&entity.raw().to_be_bytes());
            #[allow(
                clippy::cast_possible_truncation,
                reason = "residue index fits in u32"
            )]
            buffer.extend_from_slice(&(*residue_idx as u32).to_be_bytes());
            buffer.extend_from_slice(new_name);
            #[allow(
                clippy::cast_possible_truncation,
                reason = "atom count fits in u32"
            )]
            buffer.extend_from_slice(&(new_atoms.len() as u32).to_be_bytes());
            for atom in new_atoms {
                // The res_name / res_num fields in the atom row are
                // redundant under MutateResidue (the receiver gets
                // `new_name` + residue_idx separately), so we fill them
                // with stable placeholders.
                write_atom_row(AtomRef::from_atom(atom), *new_name, 0, buffer);
            }
            write_variant_list(new_variants, buffer);
        }
        AssemblyEdit::SetVariants {
            entity,
            residue_idx,
            variants,
        } => {
            buffer.push(TAG_SET_VARIANTS);
            buffer.extend_from_slice(&entity.raw().to_be_bytes());
            #[allow(
                clippy::cast_possible_truncation,
                reason = "residue index fits in u32"
            )]
            buffer.extend_from_slice(&(*residue_idx as u32).to_be_bytes());
            write_variant_list(variants, buffer);
        }
        AssemblyEdit::AddEntity { .. } | AssemblyEdit::RemoveEntity { .. } => {
            return Err(DeltaSerializeError::TopologyEditNotSupported {
                index,
            });
        }
    }
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "mirror of write_edit: linear dispatch over tag bytes to \
              AssemblyEdit variants. Each arm decodes one tag's payload."
)]
fn read_edit<'b>(
    cursor: &'b [u8],
    alloc: &mut EntityIdAllocator,
) -> Result<(AssemblyEdit, &'b [u8]), AdapterError> {
    let (tag, rest) = split_first_u8(cursor)?;
    match tag {
        TAG_SET_ENTITY_COORDS => {
            let (entity_raw, rest) = read_u32(rest)?;
            let entity = alloc.from_raw(entity_raw);
            let (coords, rest) = read_coord_list(rest)?;
            Ok((AssemblyEdit::SetEntityCoords { entity, coords }, rest))
        }
        TAG_SET_RESIDUE_COORDS => {
            let (entity_raw, rest) = read_u32(rest)?;
            let entity = alloc.from_raw(entity_raw);
            let (residue_idx, rest) = read_u32(rest)?;
            let (coords, rest) = read_coord_list(rest)?;
            Ok((
                AssemblyEdit::SetResidueCoords {
                    entity,
                    residue_idx: residue_idx as usize,
                    coords,
                },
                rest,
            ))
        }
        TAG_MUTATE_RESIDUE => {
            let (entity_raw, rest) = read_u32(rest)?;
            let entity = alloc.from_raw(entity_raw);
            let (residue_idx, rest) = read_u32(rest)?;
            let (new_name, rest) = read_three_bytes(rest)?;
            let (atom_count, rest) = read_u32(rest)?;
            let mut new_atoms = Vec::with_capacity(atom_count as usize);
            let mut cur = rest;
            for _ in 0..atom_count {
                if cur.len() < ATOM_ROW_BYTES {
                    return Err(AdapterError::InvalidFormat(
                        "Truncated atom row in delta MutateResidue".to_owned(),
                    ));
                }
                // res_name / res_num are placeholders under MutateResidue
                // (the receiver gets new_name + residue_idx separately), so
                // only the atom payload is kept.
                let (atom, _res_name, _res_num) = read_atom_row(cur)?;
                cur = &cur[ATOM_ROW_BYTES..];
                new_atoms.push(atom);
            }
            let (new_variants, cur) = read_variant_list(cur)?;
            Ok((
                AssemblyEdit::MutateResidue {
                    entity,
                    residue_idx: residue_idx as usize,
                    new_name,
                    new_atoms,
                    new_variants,
                },
                cur,
            ))
        }
        TAG_SET_VARIANTS => {
            let (entity_raw, rest) = read_u32(rest)?;
            let entity = alloc.from_raw(entity_raw);
            let (residue_idx, rest) = read_u32(rest)?;
            let (variants, rest) = read_variant_list(rest)?;
            Ok((
                AssemblyEdit::SetVariants {
                    entity,
                    residue_idx: residue_idx as usize,
                    variants,
                },
                rest,
            ))
        }
        other => Err(AdapterError::InvalidFormat(format!(
            "Unknown delta edit tag: {other:#x}",
        ))),
    }
}

fn write_coord_list(coords: &[Vec3], buffer: &mut Vec<u8>) {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "coord count fits in u32"
    )]
    buffer.extend_from_slice(&(coords.len() as u32).to_be_bytes());
    for c in coords {
        buffer.extend_from_slice(&c.x.to_be_bytes());
        buffer.extend_from_slice(&c.y.to_be_bytes());
        buffer.extend_from_slice(&c.z.to_be_bytes());
    }
}

fn read_coord_list(cursor: &[u8]) -> Result<(Vec<Vec3>, &[u8]), AdapterError> {
    let (count, mut rest) = read_u32(cursor)?;
    let count = count as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let (x_bytes, after) =
            rest.split_first_chunk::<4>().ok_or_else(coord_truncated)?;
        let (y_bytes, after) =
            after.split_first_chunk::<4>().ok_or_else(coord_truncated)?;
        let (z_bytes, after) =
            after.split_first_chunk::<4>().ok_or_else(coord_truncated)?;
        out.push(Vec3::new(
            f32::from_be_bytes(*x_bytes),
            f32::from_be_bytes(*y_bytes),
            f32::from_be_bytes(*z_bytes),
        ));
        rest = after;
    }
    Ok((out, rest))
}

fn coord_truncated() -> AdapterError {
    AdapterError::InvalidFormat("Truncated coord list in delta".to_owned())
}

fn write_variant_list(variants: &[VariantTag], buffer: &mut Vec<u8>) {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "variant count fits in u32"
    )]
    buffer.extend_from_slice(&(variants.len() as u32).to_be_bytes());
    for v in variants {
        write_variant(v, buffer);
    }
}

fn read_variant_list(
    cursor: &[u8],
) -> Result<(Vec<VariantTag>, &[u8]), AdapterError> {
    let (count, mut rest) = read_u32(cursor)?;
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let (v, r) = read_variant(rest)?;
        out.push(v);
        rest = r;
    }
    Ok((out, rest))
}

fn split_first_u8(cursor: &[u8]) -> Result<(u8, &[u8]), AdapterError> {
    cursor.split_first().map(|(b, r)| (*b, r)).ok_or_else(|| {
        AdapterError::InvalidFormat(
            "Truncated delta (expected tag byte)".to_owned(),
        )
    })
}

fn read_u32(cursor: &[u8]) -> Result<(u32, &[u8]), AdapterError> {
    let (head, rest) = cursor.split_first_chunk::<4>().ok_or_else(|| {
        AdapterError::InvalidFormat("Truncated delta (expected u32)".to_owned())
    })?;
    Ok((u32::from_be_bytes(*head), rest))
}

fn read_three_bytes(cursor: &[u8]) -> Result<([u8; 3], &[u8]), AdapterError> {
    if cursor.len() < 3 {
        return Err(AdapterError::InvalidFormat(
            "Truncated delta (expected 3-byte residue name)".to_owned(),
        ));
    }
    let mut out = [0u8; 3];
    out.copy_from_slice(&cursor[..3]);
    Ok((out, &cursor[3..]))
}

#[cfg(test)]
#[path = "delta_tests.rs"]
mod tests;
