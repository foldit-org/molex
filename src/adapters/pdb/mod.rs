//! PDB format parsing and writing.
//!
//! The parser is a hand-rolled column-positional scanner per wwPDB v3.3 section
//! 9. Records other than `ATOM`, `HETATM`, `MODEL`, `ENDMDL`, `TER`, and
//! `END` are silently skipped.

mod parse;
mod refuse;
mod write;

use self::parse::{parse_pdb_to_all_models, parse_pdb_to_entities};
use self::refuse::check_beem_bundle;
pub use self::write::assembly_to_pdb;
use crate::entity::molecule::{Completion, MoleculeEntity};
use crate::ops::error::AdapterError;

// Entity-first API (primary)

/// Parse PDB format string to entity list, completing missing heavy atoms
/// ([`Completion::Heavy`]).
///
/// # Errors
///
/// Returns [`AdapterError`] if parsing fails or the input is recognized
/// as a BeEM split bundle (which the mmCIF adapter must handle instead).
pub fn pdb_str_to_entities(
    pdb_str: &str,
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    pdb_str_to_entities_with(pdb_str, Completion::Heavy)
}

/// Parse PDB format string to entity list at the given completion `level`.
///
/// [`Completion::Raw`] yields the atoms exactly as parsed (no fabricated
/// heavy atoms, input hydrogens preserved); [`Completion::Heavy`] matches
/// [`pdb_str_to_entities`]; [`Completion::AllAtom`] additionally places
/// template hydrogens.
///
/// # Errors
///
/// Returns [`AdapterError`] if parsing fails or the input is recognized
/// as a BeEM split bundle (which the mmCIF adapter must handle instead).
pub fn pdb_str_to_entities_with(
    pdb_str: &str,
    level: Completion,
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    if let Some(err) = check_beem_bundle(pdb_str, None) {
        return Err(err);
    }
    parse_pdb_to_entities(pdb_str, level)
}

/// Load PDB file to entity list.
///
/// # Errors
///
/// Returns [`AdapterError`] if the file cannot be read, the input is
/// recognized as a BeEM split bundle, or parsing fails.
pub fn pdb_file_to_entities(
    path: &std::path::Path,
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        AdapterError::PdbParseError(format!("Failed to read file: {e}"))
    })?;
    if let Some(err) = check_beem_bundle(&content, Some(path)) {
        return Err(err);
    }
    parse_pdb_to_entities(&content, Completion::Heavy)
}

/// Parse a PDB string and return one entity list per `MODEL` block.
///
/// Files without any `MODEL`/`ENDMDL` records are treated as a single
/// implicit model: the returned vector has length 1.
///
/// # Errors
///
/// Returns [`AdapterError`] if parsing fails or the input is a BeEM
/// split bundle.
pub fn pdb_str_to_all_models(
    pdb_str: &str,
) -> Result<Vec<Vec<MoleculeEntity>>, AdapterError> {
    if let Some(err) = check_beem_bundle(pdb_str, None) {
        return Err(err);
    }
    parse_pdb_to_all_models(pdb_str)
}

/// Load a PDB file and return one entity list per `MODEL` block.
///
/// # Errors
///
/// Returns [`AdapterError`] if the file cannot be read, the input is a
/// BeEM split bundle, or parsing fails.
pub fn pdb_file_to_all_models(
    path: &std::path::Path,
) -> Result<Vec<Vec<MoleculeEntity>>, AdapterError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        AdapterError::PdbParseError(format!("Failed to read file: {e}"))
    })?;
    if let Some(err) = check_beem_bundle(&content, Some(path)) {
        return Err(err);
    }
    parse_pdb_to_all_models(&content)
}

/// Load structure file (PDB or mmCIF, detected by extension) to entity list.
///
/// # Errors
///
/// Returns [`AdapterError`] if the file cannot be read or parsing fails.
pub fn structure_file_to_entities(
    path: &std::path::Path,
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "pdb" | "ent" => pdb_file_to_entities(path),
        _ => super::cif::mmcif_file_to_entities(path),
    }
}

impl crate::Assembly {
    /// Build an `Assembly` from a PDB string, completing missing heavy atoms
    /// ([`Completion::Heavy`]).
    ///
    /// # Errors
    ///
    /// Returns [`AdapterError`] if parsing fails or the input is a BeEM
    /// split bundle.
    pub fn from_pdb(pdb_str: &str) -> Result<Self, AdapterError> {
        Ok(Self::new(pdb_str_to_entities(pdb_str)?))
    }

    /// Build an `Assembly` from a PDB string at the given completion `level`.
    ///
    /// # Errors
    ///
    /// Returns [`AdapterError`] if parsing fails or the input is a BeEM
    /// split bundle.
    pub fn from_pdb_with(
        pdb_str: &str,
        level: Completion,
    ) -> Result<Self, AdapterError> {
        Ok(Self::new(pdb_str_to_entities_with(pdb_str, level)?))
    }

    /// Build an `Assembly` from a structure file (PDB or mmCIF, detected by
    /// extension), completing missing heavy atoms ([`Completion::Heavy`]).
    ///
    /// # Errors
    ///
    /// Returns [`AdapterError`] if the file cannot be read or parsing fails.
    pub fn from_file(path: &std::path::Path) -> Result<Self, AdapterError> {
        Ok(Self::new(structure_file_to_entities(path)?))
    }

    /// Emit this assembly as a PDB-format string.
    ///
    /// # Errors
    ///
    /// Returns [`AdapterError`] if PDB serialization fails.
    pub fn to_pdb(&self) -> Result<String, AdapterError> {
        assembly_to_pdb(self)
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
