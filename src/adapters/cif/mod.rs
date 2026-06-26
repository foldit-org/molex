//! mmCIF text-format adapter.
//!
//! Two parser paths feed `EntityBuilder`: a hand-rolled streaming
//! scanner and a DOM-backed fallback. The DOM types are exposed for
//! untyped tag-based queries over a parsed [`Document`].

pub mod dom;
mod dom_build;
mod hint;
pub mod parse;
mod refuse;
mod scan;
mod scan_row;
mod write;

use std::path::Path;

pub use dom::{Block, ColumnIter, Columns, Document, Loop, RowIter, Value};
pub use parse::{parse, CifParseError};
pub(crate) use write::assembly_to_mmcif;

use crate::element::Element;
use crate::entity::molecule::{BuildError, Completion, MoleculeEntity};
use crate::ops::error::AdapterError;

// Helpers shared by the scan (`scan_row`) and DOM (`dom_build`) decode paths.

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
        BuildError::InvalidCoordinate { .. } => {
            AdapterError::InvalidFormat(err.to_string())
        }
    }
}

/// Parse mmCIF format string to entity list, completing missing heavy
/// atoms ([`Completion::Heavy`]).
///
/// Returns the model whose `pdbx_PDB_model_num` matches the smallest
/// value present; files without a model column collapse to a single
/// implicit model.
///
/// # Errors
///
/// Returns [`AdapterError`] if parsing fails.
pub(crate) fn mmcif_str_to_entities(
    cif_str: &str,
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    mmcif_str_to_entities_with(cif_str, Completion::Heavy)
}

/// Parse mmCIF format string to entity list at the given completion
/// `level`.
///
/// [`Completion::Raw`] yields the atoms exactly as parsed (no fabricated
/// heavy atoms, input hydrogens preserved); [`Completion::Heavy`] matches
/// [`mmcif_str_to_entities`]; [`Completion::AllAtom`] additionally places
/// template hydrogens.
///
/// # Errors
///
/// Returns [`AdapterError`] if parsing fails.
pub(crate) fn mmcif_str_to_entities_with(
    cif_str: &str,
    level: Completion,
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    if let Some(result) = scan::parse_mmcif_scan(cif_str, level) {
        return result;
    }
    dom_build::parse_mmcif_dom_to_entities(cif_str, level)
}

/// Load mmCIF file to entity list.
///
/// # Errors
///
/// Returns [`AdapterError`] if the file cannot be read or parsing fails.
pub(crate) fn mmcif_file_to_entities(
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
    if let Some(result) = scan::parse_mmcif_scan_all_models(cif_str) {
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

impl crate::Assembly {
    /// Build an `Assembly` from an mmCIF string, completing missing heavy
    /// atoms ([`Completion::Heavy`]).
    ///
    /// # Errors
    ///
    /// Returns [`AdapterError`] if parsing fails.
    pub fn from_mmcif(cif_str: &str) -> Result<Self, AdapterError> {
        Ok(Self::new(mmcif_str_to_entities(cif_str)?))
    }

    /// Build an `Assembly` from an mmCIF string at the given completion
    /// `level`.
    ///
    /// # Errors
    ///
    /// Returns [`AdapterError`] if parsing fails.
    pub fn from_mmcif_with(
        cif_str: &str,
        level: Completion,
    ) -> Result<Self, AdapterError> {
        Ok(Self::new(mmcif_str_to_entities_with(cif_str, level)?))
    }

    /// Emit this assembly as an mmCIF-format string.
    #[must_use]
    pub fn to_mmcif(&self) -> String {
        assembly_to_mmcif(self)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "test code")]
mod parity_tests {
    use crate::adapters::pdb::pdb_str_to_entities;
    use crate::entity::molecule::{MoleculeEntity, MoleculeType};

    fn atom_count(entities: &[MoleculeEntity]) -> usize {
        entities.iter().map(MoleculeEntity::atom_count).sum()
    }

    fn water_atom_count(entities: &[MoleculeEntity]) -> usize {
        entities
            .iter()
            .filter(|e| e.molecule_type() == MoleculeType::Water)
            .map(MoleculeEntity::atom_count)
            .sum()
    }

    /// RCSB mmCIF gives waters/non-polymer rows `label_seq_id = "."` and puts
    /// their per-instance discriminator in `auth_seq_id`; the PDB equivalent
    /// puts the same author number into the residue-sequence column. The CIF
    /// reader must fall back to `auth_seq_id` so each water keys to a distinct
    /// residue, exactly as the PDB path already does — otherwise the four
    /// waters here collapse onto a single `O` atom. The two formats describe
    /// the identical structure, so their atom counts must agree.
    #[test]
    fn cif_and_pdb_agree_on_waters_with_absent_label_seq_id() {
        const CIF: &str = "\
data_TEST
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.type_symbol
ATOM   1 N  ALA A 1 A 1   0.000 0.000 0.000 1.00 0.00 N
ATOM   2 CA ALA A 1 A 1   1.000 0.000 0.000 1.00 0.00 C
ATOM   3 C  ALA A 1 A 1   1.500 1.000 0.000 1.00 0.00 C
ATOM   4 O  ALA A 1 A 1   1.000 2.000 0.000 1.00 0.00 O
HETATM 5 O  HOH B . B 101 5.000 0.000 0.000 1.00 0.00 O
HETATM 6 O  HOH B . B 102 6.000 0.000 0.000 1.00 0.00 O
HETATM 7 O  HOH B . B 103 7.000 0.000 0.000 1.00 0.00 O
HETATM 8 O  HOH B . B 104 8.000 0.000 0.000 1.00 0.00 O
#
";
        const PDB: &str = "\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.500   1.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.000   2.000   0.000  1.00  0.00           O
TER       5      ALA A   1
HETATM    6  O   HOH B 101       5.000   0.000   0.000  1.00  0.00           O
HETATM    7  O   HOH B 102       6.000   0.000   0.000  1.00  0.00           O
HETATM    8  O   HOH B 103       7.000   0.000   0.000  1.00  0.00           O
HETATM    9  O   HOH B 104       8.000   0.000   0.000  1.00  0.00           O
END
";
        let cif = super::mmcif_str_to_entities(CIF).expect("cif parses");
        let pdb = pdb_str_to_entities(PDB).expect("pdb parses");

        let cif_atoms = atom_count(&cif);
        let pdb_atoms = atom_count(&pdb);
        assert_eq!(
            cif_atoms, pdb_atoms,
            "cif {cif_atoms} != pdb {pdb_atoms} atoms"
        );
        assert_eq!(
            water_atom_count(&cif),
            4,
            "four waters must survive, not collapse to one"
        );
        assert_eq!(water_atom_count(&cif), water_atom_count(&pdb));
    }
}
