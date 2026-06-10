//! Atom-site column resolution and row -> `AtomRow` decoding used by
//! the streaming fast path.

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

pub(super) struct RowValues {
    pub(super) label_atom_id: String,
    pub(super) label_comp_id: String,
    pub(super) label_asym_id: String,
    pub(super) label_seq_id: i32,
    pub(super) label_entity_id: Option<String>,
    pub(super) label_alt_id: Option<u8>,
    pub(super) cartn_x: Option<f64>,
    pub(super) cartn_y: Option<f64>,
    pub(super) cartn_z: Option<f64>,
    pub(super) type_symbol: Option<String>,
    pub(super) occupancy: f32,
    pub(super) b_factor: f32,
    pub(super) pdb_ins_code: Option<u8>,
    pub(super) model: Option<i32>,
    pub(super) formal_charge: i8,
    pub(super) auth_asym_id: Option<String>,
    pub(super) auth_seq_id: Option<i32>,
    pub(super) auth_comp_id: Option<String>,
    pub(super) auth_atom_id: Option<String>,
}

impl RowValues {
    pub(super) fn from_values(cols: &AtomSiteCols, values: &[String]) -> Self {
        let take = |idx: usize| values.get(idx).cloned().unwrap_or_default();
        let opt_str = |idx: Option<usize>| -> Option<String> {
            idx.and_then(|i| values.get(i))
                .filter(|s| !is_unknown(s))
                .cloned()
        };
        let opt_byte = |idx: Option<usize>| -> Option<u8> {
            opt_str(idx)
                .and_then(|s| s.bytes().next())
                .filter(|&b| b != b' ')
        };

        Self {
            label_atom_id: take(cols.label_atom_id),
            label_comp_id: take(cols.label_comp_id),
            label_asym_id: take(cols.label_asym_id),
            label_seq_id: opt_int(values, cols.label_seq_id).unwrap_or(0),
            label_entity_id: opt_str(cols.label_entity_id),
            label_alt_id: opt_byte(cols.label_alt_id),
            cartn_x: values.get(cols.cartn_x).and_then(|s| parse_cif_float(s)),
            cartn_y: values.get(cols.cartn_y).and_then(|s| parse_cif_float(s)),
            cartn_z: values.get(cols.cartn_z).and_then(|s| parse_cif_float(s)),
            type_symbol: opt_str(cols.type_symbol),
            occupancy: opt_float_f32(values, cols.occupancy, 1.0),
            b_factor: opt_float_f32(values, cols.b_iso, 0.0),
            pdb_ins_code: opt_byte(cols.pdb_ins_code),
            model: opt_int(values, cols.pdb_model_num),
            formal_charge: opt_int(values, cols.formal_charge)
                .and_then(|n| i8::try_from(n).ok())
                .unwrap_or(0),
            auth_asym_id: opt_str(cols.auth_asym_id),
            auth_seq_id: opt_int(values, cols.auth_seq_id),
            auth_comp_id: opt_str(cols.auth_comp_id),
            auth_atom_id: opt_str(cols.auth_atom_id),
        }
    }
}

fn opt_int(values: &[String], idx: Option<usize>) -> Option<i32> {
    idx.and_then(|i| values.get(i))
        .and_then(|s| parse_int_strict(s))
}

fn opt_float_f32(values: &[String], idx: Option<usize>, default: f32) -> f32 {
    #[allow(clippy::cast_possible_truncation)]
    idx.and_then(|i| values.get(i))
        .and_then(|s| parse_cif_float(s))
        .map_or(default, |f| f as f32)
}

pub(super) fn push_row(
    builder: &mut EntityBuilder,
    row: &RowValues,
) -> Result<bool, AdapterError> {
    let (Some(x), Some(y), Some(z)) = (row.cartn_x, row.cartn_y, row.cartn_z)
    else {
        return Ok(false);
    };
    if row.label_asym_id.is_empty() || is_unknown(&row.label_asym_id) {
        return Ok(false);
    }

    let element =
        resolve_element(row.type_symbol.as_deref(), &row.label_atom_id);

    #[allow(clippy::cast_possible_truncation)]
    let atom_row = AtomRow {
        label_asym_id: row.label_asym_id.clone(),
        label_seq_id: row.label_seq_id,
        label_comp_id: name_to_bytes::<3>(&row.label_comp_id),
        label_atom_id: name_to_bytes::<4>(&row.label_atom_id),
        label_entity_id: row.label_entity_id.clone(),
        auth_asym_id: row.auth_asym_id.clone(),
        auth_seq_id: row.auth_seq_id,
        auth_comp_id: row.auth_comp_id.as_deref().map(name_to_bytes::<3>),
        auth_atom_id: row.auth_atom_id.as_deref().map(name_to_bytes::<4>),
        alt_loc: row.label_alt_id,
        ins_code: row.pdb_ins_code,
        element,
        x: x as f32,
        y: y as f32,
        z: z as f32,
        occupancy: row.occupancy,
        b_factor: row.b_factor,
        formal_charge: row.formal_charge,
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
