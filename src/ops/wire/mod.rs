//! Assembly binary wire format: encoder, decoder, header constants.
//!
//! The header is a fixed 8-byte magic followed by a `u8` version byte.
//! The version selects the payload layout; the magic never changes when
//! the payload does. Version 2 carries each entity's chain id as a
//! length-prefixed string in the per-entity header (lifting the
//! single-byte chain cap) and 25-byte atom rows; version 1 carried the
//! chain id as a per-atom byte in 26-byte rows. Both versions decode.
//!
//! See [`crate::ops::wire::serialize_assembly`] for the byte layout.

pub mod delta;
pub(crate) mod deserialize;
pub(crate) mod serialize;
pub(crate) mod variants;

use crate::entity::molecule::{MoleculeEntity, MoleculeType};
use crate::ops::codec::AdapterError;

/// Magic header bytes identifying the assembly binary format. The version
/// byte that follows selects the payload layout.
pub const ASSEMBLY_MAGIC: &[u8; 8] = b"ASSEMBLY";

/// Wire format version written after [`ASSEMBLY_MAGIC`].
///
/// Version 2 carries each entity's chain id in its header and 25-byte atom
/// rows; version 1 (still decodable) carried the chain id per atom in
/// 26-byte rows.
pub const ASSEMBLY_VERSION: u8 = 2;

/// Encode a `MoleculeType` to its wire byte.
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

/// Decode a wire byte to a `MoleculeType`.
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

/// Encode a raw entity slice (includes molecule type metadata).
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
