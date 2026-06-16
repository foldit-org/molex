//! Per-residue variant tag encoding for the assembly wire format.
//!
//! Section layout (after atom payload, in entity-header order):
//!
//! ```text
//! per entity {
//!     u32 BE variant_residue_count
//!     per variant-bearing residue {
//!         i32 BE label_seq_id
//!         u32 BE variant_count
//!         per variant {
//!             u8 tag
//!             tag-specific payload
//!         }
//!     }
//! }
//! ```
//!
//! Tag bytes:
//! - `0x01` NTerminus — 0 payload bytes
//! - `0x02` CTerminus — 0 payload bytes
//! - `0x03` Disulfide — 0 payload bytes
//! - `0x04` Protonation — 1 sub-tag byte, plus optional Custom payload:
//!   - `0x01` HisDelta
//!   - `0x02` HisEpsilon
//!   - `0x03` HisDoubly
//!   - `0xFF` Custom — u16 BE length + UTF-8 bytes
//! - `0xFE` Other — u16 BE length + UTF-8 bytes

use crate::chemistry::variant::{ProtonationState, VariantTag};
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::MoleculeEntity;
use crate::ops::codec::AdapterError;

/// Decoded variant block for a single residue, keyed by its
/// `label_seq_id`. Callers match the id against their reconstructed
/// residue list and attach the variants accordingly.
#[derive(Debug, Clone)]
pub(crate) struct ResidueVariants {
    /// `label_seq_id` of the residue these variants belong to.
    pub label_seq_id: i32,
    /// Variant tags for the residue.
    pub variants: Vec<VariantTag>,
}

/// All decoded variant blocks for a single entity (sparse — only
/// residues that carry at least one variant are listed).
pub(crate) type EntityVariants = Vec<ResidueVariants>;

const TAG_NTERM: u8 = 0x01;
const TAG_CTERM: u8 = 0x02;
const TAG_DISULFIDE: u8 = 0x03;
const TAG_PROTONATION: u8 = 0x04;
const TAG_OTHER: u8 = 0xFE;

const PROT_HIS_DELTA: u8 = 0x01;
const PROT_HIS_EPSILON: u8 = 0x02;
const PROT_HIS_DOUBLY: u8 = 0x03;
const PROT_CUSTOM: u8 = 0xFF;

/// Serialize the per-entity variants section.
///
/// Walks `entities` in order; for each polymer entity (Protein /
/// NucleicAcid), emits residues that have non-empty variants. Non-polymer
/// entities emit a zero count.
pub(crate) fn serialize_variants_section<E>(
    entities: &[E],
    buffer: &mut Vec<u8>,
) where
    E: std::borrow::Borrow<MoleculeEntity>,
{
    for entity in entities {
        let residues_opt: Option<&[Residue]> = match entity.borrow() {
            MoleculeEntity::Protein(e) => Some(&e.residues),
            MoleculeEntity::NucleicAcid(e) => Some(&e.residues),
            MoleculeEntity::SmallMolecule(_) | MoleculeEntity::Bulk(_) => None,
        };

        let Some(residues) = residues_opt else {
            buffer.extend_from_slice(&0u32.to_be_bytes());
            continue;
        };

        let variant_residues: Vec<&Residue> =
            residues.iter().filter(|r| !r.variants.is_empty()).collect();

        #[allow(
            clippy::cast_possible_truncation,
            reason = "residue count per entity fits in u32"
        )]
        buffer
            .extend_from_slice(&(variant_residues.len() as u32).to_be_bytes());

        for residue in variant_residues {
            buffer.extend_from_slice(&residue.label_seq_id.to_be_bytes());
            #[allow(
                clippy::cast_possible_truncation,
                reason = "variant count fits in u32"
            )]
            buffer.extend_from_slice(
                &(residue.variants.len() as u32).to_be_bytes(),
            );
            for variant in &residue.variants {
                write_variant(variant, buffer);
            }
        }
    }
}

/// Deserialize the per-entity variants section.
///
/// Returns one `EntityVariants` per entity in the same order as the
/// entity headers. Callers are responsible for matching each block's
/// `label_seq_id` against their reconstructed residue list and
/// attaching the variants accordingly.
pub(crate) fn deserialize_variants_section(
    bytes: &[u8],
    entity_count: usize,
) -> Result<Vec<EntityVariants>, AdapterError> {
    let mut cursor = bytes;
    let mut out: Vec<EntityVariants> = Vec::with_capacity(entity_count);

    for _ in 0..entity_count {
        let (count, rest) = read_u32(cursor)?;
        cursor = rest;
        let mut per_entity: EntityVariants = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let (label_seq_id, rest) = read_i32(cursor)?;
            cursor = rest;
            let (variant_count, rest) = read_u32(cursor)?;
            cursor = rest;
            let mut variants = Vec::with_capacity(variant_count as usize);
            for _ in 0..variant_count {
                let (variant, rest) = read_variant(cursor)?;
                cursor = rest;
                variants.push(variant);
            }
            per_entity.push(ResidueVariants {
                label_seq_id,
                variants,
            });
        }
        out.push(per_entity);
    }

    Ok(out)
}

pub(crate) fn write_variant(variant: &VariantTag, buffer: &mut Vec<u8>) {
    match variant {
        VariantTag::NTerminus => buffer.push(TAG_NTERM),
        VariantTag::CTerminus => buffer.push(TAG_CTERM),
        VariantTag::Disulfide => buffer.push(TAG_DISULFIDE),
        VariantTag::Protonation(state) => {
            buffer.push(TAG_PROTONATION);
            match state {
                ProtonationState::HisDelta => buffer.push(PROT_HIS_DELTA),
                ProtonationState::HisEpsilon => {
                    buffer.push(PROT_HIS_EPSILON);
                }
                ProtonationState::HisDoubly => buffer.push(PROT_HIS_DOUBLY),
                ProtonationState::Custom(s) => {
                    buffer.push(PROT_CUSTOM);
                    write_string(s, buffer);
                }
            }
        }
        VariantTag::Other(s) => {
            buffer.push(TAG_OTHER);
            write_string(s, buffer);
        }
    }
}

pub(crate) fn read_variant(
    cursor: &[u8],
) -> Result<(VariantTag, &[u8]), AdapterError> {
    let (tag, rest) = read_u8(cursor)?;
    match tag {
        TAG_NTERM => Ok((VariantTag::NTerminus, rest)),
        TAG_CTERM => Ok((VariantTag::CTerminus, rest)),
        TAG_DISULFIDE => Ok((VariantTag::Disulfide, rest)),
        TAG_PROTONATION => {
            let (sub, rest) = read_u8(rest)?;
            let (state, rest) = match sub {
                PROT_HIS_DELTA => (ProtonationState::HisDelta, rest),
                PROT_HIS_EPSILON => (ProtonationState::HisEpsilon, rest),
                PROT_HIS_DOUBLY => (ProtonationState::HisDoubly, rest),
                PROT_CUSTOM => {
                    let (s, rest) = read_string(rest)?;
                    (ProtonationState::Custom(s), rest)
                }
                other => {
                    return Err(AdapterError::InvalidFormat(format!(
                        "Unknown protonation sub-tag: {other:#x}",
                    )))
                }
            };
            Ok((VariantTag::Protonation(state), rest))
        }
        TAG_OTHER => {
            let (s, rest) = read_string(rest)?;
            Ok((VariantTag::Other(s), rest))
        }
        other => Err(AdapterError::InvalidFormat(format!(
            "Unknown variant tag: {other:#x}",
        ))),
    }
}

fn write_string(s: &str, buffer: &mut Vec<u8>) {
    let bytes = s.as_bytes();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "variant string length capped at u16; callers control input"
    )]
    let len = bytes.len().min(u16::MAX as usize) as u16;
    buffer.extend_from_slice(&len.to_be_bytes());
    buffer.extend_from_slice(&bytes[..len as usize]);
}

fn read_string(cursor: &[u8]) -> Result<(String, &[u8]), AdapterError> {
    let (len, rest) = read_u16(cursor)?;
    let len = len as usize;
    if rest.len() < len {
        return Err(AdapterError::InvalidFormat(
            "Truncated variant string payload".to_owned(),
        ));
    }
    let (body, rest) = rest.split_at(len);
    let s = std::str::from_utf8(body)
        .map_err(|e| {
            AdapterError::InvalidFormat(format!(
                "Variant string is not valid UTF-8: {e}",
            ))
        })?
        .to_owned();
    Ok((s, rest))
}

fn read_u8(cursor: &[u8]) -> Result<(u8, &[u8]), AdapterError> {
    cursor
        .split_first()
        .map(|(b, rest)| (*b, rest))
        .ok_or_else(|| {
            AdapterError::InvalidFormat(
                "Truncated variants section (expected u8)".to_owned(),
            )
        })
}

fn read_u16(cursor: &[u8]) -> Result<(u16, &[u8]), AdapterError> {
    let (head, rest) = cursor.split_first_chunk::<2>().ok_or_else(|| {
        AdapterError::InvalidFormat(
            "Truncated variants section (expected u16)".to_owned(),
        )
    })?;
    Ok((u16::from_be_bytes(*head), rest))
}

fn read_u32(cursor: &[u8]) -> Result<(u32, &[u8]), AdapterError> {
    let (head, rest) = cursor.split_first_chunk::<4>().ok_or_else(|| {
        AdapterError::InvalidFormat(
            "Truncated variants section (expected u32)".to_owned(),
        )
    })?;
    Ok((u32::from_be_bytes(*head), rest))
}

fn read_i32(cursor: &[u8]) -> Result<(i32, &[u8]), AdapterError> {
    let (head, rest) = cursor.split_first_chunk::<4>().ok_or_else(|| {
        AdapterError::InvalidFormat(
            "Truncated variants section (expected i32)".to_owned(),
        )
    })?;
    Ok((i32::from_be_bytes(*head), rest))
}
