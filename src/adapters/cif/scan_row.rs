//! Atom-site column resolution and row -> `AtomRow` decoding used by
//! the streaming scanner.

use compact_str::CompactString;

use super::{map_build_error, name_to_bytes, resolve_element};
use crate::entity::molecule::{AtomRow, EntityBuilder, MoleculeEntity};
use crate::ops::codec::AdapterError;

pub(super) struct AtomSiteCols {
    pub(super) label_atom_id: usize,
    pub(super) label_comp_id: usize,
    pub(super) label_asym_id: usize,
    pub(super) cartn_x: usize,
    pub(super) cartn_y: usize,
    pub(super) cartn_z: usize,
    pub(super) label_seq_id: Option<usize>,
    pub(super) label_entity_id: Option<usize>,
    pub(super) label_alt_id: Option<usize>,
    pub(super) type_symbol: Option<usize>,
    pub(super) occupancy: Option<usize>,
    pub(super) b_iso: Option<usize>,
    pub(super) pdb_ins_code: Option<usize>,
    pub(super) pdb_model_num: Option<usize>,
    pub(super) formal_charge: Option<usize>,
    pub(super) auth_asym_id: Option<usize>,
    pub(super) auth_seq_id: Option<usize>,
    pub(super) auth_comp_id: Option<usize>,
    pub(super) auth_atom_id: Option<usize>,
    pub(super) ncols: usize,
}

impl AtomSiteCols {
    pub(super) fn from_tags(tags: &[String]) -> Option<Self> {
        let lower: Vec<String> =
            tags.iter().map(|t| t.to_ascii_lowercase()).collect();
        let find = |name: &str| lower.iter().position(|t| t == name);
        Some(Self {
            label_atom_id: find("_atom_site.label_atom_id")?,
            label_comp_id: find("_atom_site.label_comp_id")?,
            label_asym_id: find("_atom_site.label_asym_id")?,
            cartn_x: find("_atom_site.cartn_x")?,
            cartn_y: find("_atom_site.cartn_y")?,
            cartn_z: find("_atom_site.cartn_z")?,
            label_seq_id: find("_atom_site.label_seq_id"),
            label_entity_id: find("_atom_site.label_entity_id"),
            label_alt_id: find("_atom_site.label_alt_id"),
            type_symbol: find("_atom_site.type_symbol"),
            occupancy: find("_atom_site.occupancy"),
            b_iso: find("_atom_site.b_iso_or_equiv"),
            pdb_ins_code: find("_atom_site.pdbx_pdb_ins_code"),
            pdb_model_num: find("_atom_site.pdbx_pdb_model_num"),
            formal_charge: find("_atom_site.pdbx_formal_charge"),
            auth_asym_id: find("_atom_site.auth_asym_id"),
            auth_seq_id: find("_atom_site.auth_seq_id"),
            auth_comp_id: find("_atom_site.auth_comp_id"),
            auth_atom_id: find("_atom_site.auth_atom_id"),
            ncols: tags.len(),
        })
    }
}

/// A positionally-indexed row of `_atom_site` cells. Both the borrowed
/// scan path ([`RowSlots`]) and the owned multi-model buffer
/// ([`RowValues`]) expose their cells through this so the field-extraction
/// core ([`build_atom_row`]) runs once over either.
pub(super) trait RowCells {
    fn cell(&self, idx: usize) -> &str;
}

/// One `_atom_site` row as borrowed cells (slices into the input buffer).
/// The outer slice borrow (`'s`) lives only for the row's processing; the
/// cells themselves (`'a`) borrow the input buffer.
pub(super) struct RowSlots<'s, 'a> {
    cells: &'s [&'a str],
}

impl<'s, 'a> RowSlots<'s, 'a> {
    pub(super) fn new(cells: &'s [&'a str]) -> Self {
        Self { cells }
    }

    pub(super) fn model(&self, cols: &AtomSiteCols) -> Option<i32> {
        opt_int(self, cols.pdb_model_num)
    }

    /// Build an owned per-row snapshot for the multi-model bucketing path,
    /// where rows outlive a single scan iteration.
    fn to_owned_values(&self, cols: &AtomSiteCols) -> RowValues {
        let model = self.model(cols);
        RowValues {
            cells: self.cells.iter().map(|c| (*c).to_owned()).collect(),
            model,
        }
    }
}

impl RowCells for RowSlots<'_, '_> {
    fn cell(&self, idx: usize) -> &str {
        self.cells.get(idx).copied().unwrap_or_default()
    }
}

/// Owned per-row snapshot retained across scan iterations for the
/// multi-model path. `model` is extracted once for bucketing before the
/// `AtomRow` is built.
pub(super) struct RowValues {
    cells: Vec<String>,
    pub(super) model: Option<i32>,
}

impl RowValues {
    pub(super) fn from_slots(
        slots: &RowSlots<'_, '_>,
        cols: &AtomSiteCols,
    ) -> Self {
        slots.to_owned_values(cols)
    }
}

impl RowCells for RowValues {
    fn cell(&self, idx: usize) -> &str {
        self.cells.get(idx).map_or("", String::as_str)
    }
}

fn opt_cell(cells: &impl RowCells, idx: Option<usize>) -> Option<&str> {
    idx.map(|i| cells.cell(i)).filter(|s| !is_unknown(s))
}

fn opt_byte(cells: &impl RowCells, idx: Option<usize>) -> Option<u8> {
    opt_cell(cells, idx)
        .and_then(|s| s.bytes().next())
        .filter(|&b| b != b' ')
}

fn opt_int(cells: &impl RowCells, idx: Option<usize>) -> Option<i32> {
    idx.and_then(|i| parse_int_strict(cells.cell(i)))
}

fn opt_float_f32(
    cells: &impl RowCells,
    idx: Option<usize>,
    default: f32,
) -> f32 {
    #[allow(clippy::cast_possible_truncation)]
    idx.and_then(|i| parse_cif_float(cells.cell(i)))
        .map_or(default, |f| f as f32)
}

/// The single field-by-field `_atom_site` -> `AtomRow` decode, shared by the
/// borrowed and owned row paths.
///
/// Returns `None` for rows with missing coordinates or an unknown
/// `label_asym_id` (skipped).
#[allow(clippy::cast_possible_truncation)]
fn build_atom_row(
    cells: &impl RowCells,
    cols: &AtomSiteCols,
) -> Option<AtomRow> {
    let (Some(x), Some(y), Some(z)) = (
        parse_cif_float(cells.cell(cols.cartn_x)),
        parse_cif_float(cells.cell(cols.cartn_y)),
        parse_cif_float(cells.cell(cols.cartn_z)),
    ) else {
        return None;
    };

    let label_asym_id = cells.cell(cols.label_asym_id);
    if label_asym_id.is_empty() || is_unknown(label_asym_id) {
        return None;
    }

    let label_atom_id = cells.cell(cols.label_atom_id);
    let auth_seq_id = opt_int(cells, cols.auth_seq_id);
    let type_symbol = opt_cell(cells, cols.type_symbol);
    let element = resolve_element(type_symbol, label_atom_id);

    Some(AtomRow {
        label_asym_id: CompactString::new(label_asym_id),
        // Waters/non-polymer rows carry `label_seq_id = "."`; their
        // per-instance discriminator lives in `auth_seq_id`. Mirror the
        // PDB path (author number into the seq slot) so each gets a
        // distinct residue key instead of all collapsing onto 0.
        label_seq_id: opt_int(cells, cols.label_seq_id)
            .or(auth_seq_id)
            .unwrap_or(0),
        label_comp_id: name_to_bytes::<3>(cells.cell(cols.label_comp_id)),
        label_atom_id: name_to_bytes::<4>(label_atom_id),
        label_entity_id: opt_cell(cells, cols.label_entity_id)
            .map(CompactString::new),
        auth_asym_id: opt_cell(cells, cols.auth_asym_id)
            .map(CompactString::new),
        auth_seq_id,
        auth_comp_id: opt_cell(cells, cols.auth_comp_id)
            .map(name_to_bytes::<3>),
        auth_atom_id: opt_cell(cells, cols.auth_atom_id)
            .map(name_to_bytes::<4>),
        alt_loc: opt_byte(cells, cols.label_alt_id),
        ins_code: opt_byte(cells, cols.pdb_ins_code),
        element,
        x: x as f32,
        y: y as f32,
        z: z as f32,
        occupancy: opt_float_f32(cells, cols.occupancy, 1.0),
        b_factor: opt_float_f32(cells, cols.b_iso, 0.0),
        formal_charge: opt_int(cells, cols.formal_charge)
            .and_then(|n| i8::try_from(n).ok())
            .unwrap_or(0),
    })
}

/// Decode one borrowed row and push its atom onto the builder.
///
/// Returns `Ok(false)` for rows with missing coordinates or an unknown
/// `label_asym_id` (skipped), `Ok(true)` once the atom is pushed. The
/// chain / entity-id fields are inline `CompactString`s, so a short id
/// (the common case) allocates nothing.
pub(super) fn push_slots(
    builder: &mut EntityBuilder,
    slots: &RowSlots<'_, '_>,
    cols: &AtomSiteCols,
) -> Result<bool, AdapterError> {
    let Some(atom_row) = build_atom_row(slots, cols) else {
        return Ok(false);
    };
    builder
        .push_atom(atom_row)
        .map_err(|e| map_build_error(&e))?;
    Ok(true)
}

pub(super) fn push_row(
    builder: &mut EntityBuilder,
    row: &RowValues,
    cols: &AtomSiteCols,
) -> Result<bool, AdapterError> {
    let Some(atom_row) = build_atom_row(row, cols) else {
        return Ok(false);
    };
    builder
        .push_atom(atom_row)
        .map_err(|e| map_build_error(&e))?;
    Ok(true)
}

pub(super) fn finish(
    builder: EntityBuilder,
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    builder.finish().map_err(|e| map_build_error(&e))
}

/// Number conversion runs off the borrowed token slice (no allocation).
/// `str::parse` is already correctly-rounded (Lemire); a byte-native float
/// reconstruction would diverge from it by a ULP on f64, so the contract
/// (bit-identical output) keeps `parse` here while the tokenizer is what
/// stopped allocating.
fn parse_cif_float(s: &str) -> Option<f64> {
    if is_unknown(s) {
        return None;
    }
    let s = s.find('(').map_or(s, |idx| &s[..idx]);
    s.parse().ok()
}

fn parse_int_strict(s: &str) -> Option<i32> {
    if is_unknown(s) {
        return None;
    }
    s.parse().ok()
}

fn is_unknown(s: &str) -> bool {
    s == "." || s == "?" || s.is_empty()
}
