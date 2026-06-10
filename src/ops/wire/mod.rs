//! ASSEM binary wire format: encoder, decoder, header constants.
//!
//! Current version is ASSEM02 (carries per-residue variant tags after the
//! atom payload). ASSEM01 (no variants) is still accepted by the
//! deserializer and is treated as if every residue had empty variants —
//! existing readers that emit ASSEM01 bytes continue to round-trip
//! through molex without code changes.
//!
//! See [`crate::ops::wire::serialize_assembly`] for the byte layout.

pub mod delta;
pub(crate) mod deserialize;
pub(crate) mod serialize;
pub(crate) mod variants;

use crate::entity::molecule::{MoleculeEntity, MoleculeType};
use crate::ops::codec::AdapterError;

/// Magic header bytes identifying the legacy ASSEM01 binary format
/// (atoms only, no per-residue variants).
pub const ASSEMBLY_MAGIC_V1: &[u8; 8] = b"ASSEM01\0";

/// Magic header bytes for the current ASSEM02 binary format. Carries a
/// trailing variants section after the atom payload.
pub const ASSEMBLY_MAGIC_V2: &[u8; 8] = b"ASSEM02\0";

/// Magic emitted by the current serializer. Alias for the latest
/// supported version.
pub const ASSEMBLY_MAGIC: &[u8; 8] = ASSEMBLY_MAGIC_V2;

/// Encode a `MoleculeType` to its ASSEM02 wire byte.
pub(crate) fn molecule_type_to_wire(mol_type: MoleculeType) -> u8 {
    match mol_type {
        MoleculeType::Protein => 0,
        MoleculeType::DNA => 1,
        MoleculeType::RNA => 2,
        MoleculeType::Ligand => 3,
        MoleculeType::Ion => 4,
        MoleculeType::Water => 5,
        MoleculeType::Lipid => 6,
        MoleculeType::Cofactor => 7,
        MoleculeType::Solvent => 8,
    }
}

/// Decode an ASSEM02 wire byte to a `MoleculeType`.
pub(crate) fn molecule_type_from_wire(b: u8) -> Option<MoleculeType> {
    match b {
        0 => Some(MoleculeType::Protein),
        1 => Some(MoleculeType::DNA),
        2 => Some(MoleculeType::RNA),
        3 => Some(MoleculeType::Ligand),
        4 => Some(MoleculeType::Ion),
        5 => Some(MoleculeType::Water),
        6 => Some(MoleculeType::Lipid),
        7 => Some(MoleculeType::Cofactor),
        8 => Some(MoleculeType::Solvent),
        _ => None,
    }
}

/// ASSEM02-encode a raw entity slice (includes molecule type metadata).
///
/// # Errors
///
/// Returns `AdapterError` if serialization fails.
pub fn assembly_bytes(
    entities: &[MoleculeEntity],
) -> Result<Vec<u8>, AdapterError> {
    serialize::serialize_entities(entities)
}

pub use deserialize::deserialize_assembly;
pub use serialize::serialize_assembly;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
