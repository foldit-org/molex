//! mmCIF text-format adapter.
//!
//! Two parser paths feed `EntityBuilder`: a hand-rolled streaming fast
//! scanner and a DOM-backed fallback. The DOM types are exposed for
//! untyped tag-based queries over a parsed [`Document`].

pub mod dom;
mod dom_build;
mod fast;
mod fast_row;
mod hint;
pub mod parse;
mod refuse;

use std::path::Path;

pub use dom::{Block, ColumnIter, Columns, Document, Loop, RowIter, Value};
pub use parse::{parse, CifParseError};
use refuse::too_many_chains_error;

use crate::element::Element;
use crate::entity::molecule::{BuildError, MoleculeEntity};
use crate::ops::codec::AdapterError;

// ---------------------------------------------------------------------------
// Helpers shared by the fast (`fast_row`) and DOM (`dom_build`) decode paths.
// ---------------------------------------------------------------------------

/// Resolve an atom's [`Element`] from its CIF `type_symbol`, falling back
/// to the atom name when the symbol is absent, blank, or unrecognised.
fn resolve_element(type_symbol: Option<&str>, atom_name: &str) -> Element {
    let trimmed = type_symbol.map(str::trim);
    match trimmed {
        Some(s) if !s.is_empty() && s != "." && s != "?" => {
            let parsed = Element::from_symbol(s);
            if matches!(parsed, Element::Unknown) {
                Element::from_atom_name(atom_name)
            } else {
                parsed
            }
        }
        _ => Element::from_atom_name(atom_name),
    }
}

/// Pack a name into a fixed-width, space-padded byte buffer (truncating at
/// `N` bytes).
fn name_to_bytes<const N: usize>(name: &str) -> [u8; N] {
    let mut buf = [b' '; N];
    for (i, b) in name.bytes().take(N).enumerate() {
        buf[i] = b;
    }
    buf
}

/// Map an [`EntityBuilder`](crate::entity::molecule::EntityBuilder)
/// [`BuildError`] onto the adapter-facing [`AdapterError`].
fn map_build_error(err: &BuildError) -> AdapterError {
    match err {
        BuildError::TooManyChains { limit } => too_many_chains_error(*limit),
        BuildError::InvalidCoordinate { .. } => {
            AdapterError::InvalidFormat(err.to_string())
        }
    }
}

/// Parse mmCIF format string to entity list.
///
/// Returns the model whose `pdbx_PDB_model_num` matches the smallest
/// value present; files without a model column collapse to a single
/// implicit model.
///
/// # Errors
///
/// Returns [`AdapterError`] if parsing fails.
pub fn mmcif_str_to_entities(
    cif_str: &str,
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    if let Some(result) = fast::parse_mmcif_fast(cif_str) {
        return result;
    }
    dom_build::parse_mmcif_dom_to_entities(cif_str)
}

/// Load mmCIF file to entity list.
///
/// # Errors
///
/// Returns [`AdapterError`] if the file cannot be read or parsing fails.
pub fn mmcif_file_to_entities(
    path: &Path,
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        AdapterError::InvalidFormat(format!("Failed to read file: {e}"))
    })?;
    mmcif_str_to_entities(&content)
}

/// Parse an mmCIF string and return one entity list per
/// `pdbx_PDB_model_num`. Files without a model column collapse to a
/// single bucket.
///
/// # Errors
///
/// Returns [`AdapterError`] if parsing fails.
pub fn mmcif_str_to_all_models(
    cif_str: &str,
) -> Result<Vec<Vec<MoleculeEntity>>, AdapterError> {
    if let Some(result) = fast::parse_mmcif_fast_to_all_models(cif_str) {
        return result;
    }
    dom_build::parse_mmcif_dom_to_all_models(cif_str)
}

/// Load an mmCIF file and return one entity list per `pdbx_PDB_model_num`.
///
/// # Errors
///
/// Returns [`AdapterError`] if the file cannot be read or parsing fails.
pub fn mmcif_file_to_all_models(
    path: &Path,
) -> Result<Vec<Vec<MoleculeEntity>>, AdapterError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        AdapterError::InvalidFormat(format!("Failed to read file: {e}"))
    })?;
    mmcif_str_to_all_models(&content)
}
