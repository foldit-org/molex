//! Adapter/wire error type.

use thiserror::Error;

/// Errors returned by molex parsers and serializers.
///
/// Used as the `Err` variant for the public adapter entry points
/// (`pdb_str_to_entities`, `mmcif_str_to_entities`, `bcif_to_entities`) and
/// the assembly wire format (`serialize_assembly` / `deserialize_assembly`).
#[derive(Error, Debug)]
pub enum AdapterError {
    /// The input bytes/text do not conform to the expected format.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    /// A PDB file could not be parsed.
    #[error("Failed to parse PDB: {0}")]
    PdbParseError(String),
    /// An error occurred during binary serialization or deserialization.
    #[error("Serialization error: {0}")]
    SerializationError(String),
}
