//! BinaryCIF decode pipeline: pull the single coordinate data block out of
//! a MessagePack-encoded BinaryCIF byte stream, run the `_entity` /
//! `_entity_poly` hint pre-pass, then iterate `_atom_site` rows into an
//! [`EntityBuilder`].

use std::collections::HashMap;
use std::io::Read;

use super::codec::{decode_column, decode_msgpack, ColData, MsgVal};
use super::hint::resolve_hint;
use super::refuse::{multi_block_error, too_many_chains_error};
use crate::element::Element;
use crate::entity::molecule::{
    AtomRow, BuildError, EntityBuilder, MoleculeEntity,
};
use crate::ops::codec::AdapterError;

// Public entry points (called by adapters::bcif::mod.rs)

pub(super) fn decode_to_entities(
    bytes: &[u8],
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    let block = open_block(bytes)?;
    let atom_site = require_atom_site(&block)?;
    let rows = decode_atom_site(&atom_site)?;
    let target = smallest_model(&rows);

    let mut builder = EntityBuilder::new();
    register_hints(&block, &mut builder)?;

    let mut any_atom = false;
    for row in &rows {
        if let Some(target) = target {
            if row.model.is_some_and(|m| m != target) {
                continue;
            }
        }
        if push_row(&mut builder, row)? {
            any_atom = true;
        }
    }
    if !any_atom {
        return Err(AdapterError::InvalidFormat(
            "No ATOM records found in BinaryCIF".into(),
        ));
    }
    builder.finish().map_err(|e| map_build_error(&e))
}

pub(super) fn decode_to_all_models(
    bytes: &[u8],
) -> Result<Vec<Vec<MoleculeEntity>>, AdapterError> {
    let block = open_block(bytes)?;
    let atom_site = require_atom_site(&block)?;
    let rows = decode_atom_site(&atom_site)?;

    let mut buckets: HashMap<i32, Vec<usize>> = HashMap::new();
    let mut order: Vec<i32> = Vec::new();
    let default_model: i32 = 1;
    for (idx, row) in rows.iter().enumerate() {
        let key = row.model.unwrap_or(default_model);
        buckets
            .entry(key)
            .or_insert_with(|| {
                order.push(key);
                Vec::new()
            })
            .push(idx);
    }
    if buckets.is_empty() {
        return Err(AdapterError::InvalidFormat(
            "No ATOM records found in BinaryCIF".into(),
        ));
    }
    order.sort_unstable();

    let mut out: Vec<Vec<MoleculeEntity>> = Vec::with_capacity(order.len());
    for model in &order {
        let Some(indices) = buckets.remove(model) else {
            continue;
        };
        let mut builder = EntityBuilder::new();
        register_hints(&block, &mut builder)?;
        let mut any_atom = false;
        for idx in indices {
            if push_row(&mut builder, &rows[idx])? {
                any_atom = true;
            }
        }
        if any_atom {
            out.push(builder.finish().map_err(|e| map_build_error(&e))?);
        }
    }
    if out.is_empty() {
        return Err(AdapterError::InvalidFormat(
            "No ATOM records found in BinaryCIF".into(),
        ));
    }
    Ok(out)
}

// Block selection

/// Owning view of the single coordinate `dataBlock` extracted from the file.
struct BlockView {
    categories: Vec<CategoryView>,
}

impl BlockView {
    fn find_category(&self, name: &str) -> Option<&CategoryView> {
        self.categories.iter().find(|c| c.name == name)
    }
}

struct CategoryView {
    name: String,
    row_count: usize,
    columns: HashMap<String, ColumnRaw>,
}

#[derive(Clone)]
struct ColumnRaw {
    data: MsgVal,
    mask: Option<MsgVal>,
}

fn open_block(bytes: &[u8]) -> Result<BlockView, AdapterError> {
    let data = decompress_if_gzip(bytes)?;
    let root = decode_msgpack(&data)?;
    let blocks = root
        .get("dataBlocks")
        .and_then(MsgVal::as_array)
        .ok_or_else(|| {
            AdapterError::InvalidFormat("Missing 'dataBlocks'".into())
        })?;
    if blocks.is_empty() {
        return Err(AdapterError::InvalidFormat("No data blocks found".into()));
    }
    if blocks.len() > 1 {
        return Err(multi_block_error(blocks.len()));
    }
    let raw_categories = blocks[0]
        .get("categories")
        .and_then(MsgVal::as_array)
        .ok_or_else(|| {
            AdapterError::InvalidFormat(
                "Missing 'categories' in data block".into(),
            )
        })?;
    let mut categories: Vec<CategoryView> =
        Vec::with_capacity(raw_categories.len());
    for cat in raw_categories {
        if let Some(view) = parse_category(cat)? {
            categories.push(view);
        }
    }
    Ok(BlockView { categories })
}

fn parse_category(cat: &MsgVal) -> Result<Option<CategoryView>, AdapterError> {
    let Some(name) = cat.get("name").and_then(MsgVal::as_str) else {
        return Ok(None);
    };
    let raw_count =
        cat.get("rowCount")
            .and_then(MsgVal::as_i64)
            .ok_or_else(|| {
                AdapterError::InvalidFormat(format!(
                    "Category {name}: missing 'rowCount'"
                ))
            })?;
    let row_count = usize::try_from(raw_count).map_err(|_| {
        AdapterError::InvalidFormat(format!(
            "Category {name}: negative rowCount {raw_count}"
        ))
    })?;
    let mut columns: HashMap<String, ColumnRaw> = HashMap::new();
    if let Some(cols) = cat.get("columns").and_then(MsgVal::as_array) {
        for col in cols {
            let Some(cname) = col.get("name").and_then(MsgVal::as_str) else {
                continue;
            };
            let Some(data) = col.get("data") else {
                continue;
            };
            let mask = col.get("mask").cloned().filter(is_present_msg);
            let _ = columns.insert(
                cname.to_owned(),
                ColumnRaw {
                    data: data.clone(),
                    mask,
                },
            );
        }
    }
    Ok(Some(CategoryView {
        name: name.to_owned(),
        row_count,
        columns,
    }))
}

fn is_present_msg(v: &MsgVal) -> bool {
    !matches!(v, MsgVal::Nil)
}

fn decompress_if_gzip(bytes: &[u8]) -> Result<Vec<u8>, AdapterError> {
    if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        let mut decoder = flate2::read::GzDecoder::new(bytes);
        let mut out = Vec::new();
        let _ = decoder.read_to_end(&mut out).map_err(|e| {
            AdapterError::InvalidFormat(format!(
                "Gzip decompression failed: {e}"
            ))
        })?;
        Ok(out)
    } else {
        Ok(bytes.to_vec())
    }
}

// Hint pre-pass

fn register_hints(
    block: &BlockView,
    builder: &mut EntityBuilder,
) -> Result<(), AdapterError> {
    let entity = collect_entity_table(block)?;
    let poly = collect_entity_poly_table(block)?;
    let mut seen: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for (id, etype) in &entity {
        let p = poly.get(id).map(String::as_str);
        let hint = resolve_hint(etype, p);
        builder.register_entity(id, hint);
        let _ = seen.insert(id.clone());
    }
    for (id, ptype) in &poly {
        if seen.contains(id) {
            continue;
        }
        let hint = resolve_hint("polymer", Some(ptype));
        builder.register_entity(id, hint);
    }
    Ok(())
}

fn collect_entity_table(
    block: &BlockView,
) -> Result<Vec<(String, String)>, AdapterError> {
    let Some(cat) = block.find_category("_entity") else {
        return Ok(Vec::new());
    };
    let (Some(ids), Some(types)) = (
        strings_with_mask(cat, "id")?,
        strings_with_mask(cat, "type")?,
    ) else {
        return Ok(Vec::new());
    };
    Ok(ids
        .into_iter()
        .zip(types)
        .filter_map(|(id, etype)| match (id, etype) {
            (Some(id), Some(etype)) => Some((id, etype)),
            _ => None,
        })
        .collect())
}

fn collect_entity_poly_table(
    block: &BlockView,
) -> Result<HashMap<String, String>, AdapterError> {
    let mut out: HashMap<String, String> = HashMap::new();
    let Some(cat) = block.find_category("_entity_poly") else {
        return Ok(out);
    };
    let (Some(ids), Some(types)) = (
        strings_with_mask(cat, "entity_id")?,
        strings_with_mask(cat, "type")?,
    ) else {
        return Ok(out);
    };
    for (id, etype) in ids.into_iter().zip(types) {
        if let (Some(id), Some(etype)) = (id, etype) {
            let _ = out.insert(id, etype);
        }
    }
    Ok(out)
}

// _atom_site row decode

fn require_atom_site(block: &BlockView) -> Result<CategoryView, AdapterError> {
    let cat = block.find_category("_atom_site").ok_or_else(|| {
        AdapterError::InvalidFormat("No '_atom_site' category found".into())
    })?;
    Ok(CategoryView {
        name: cat.name.clone(),
        row_count: cat.row_count,
        columns: cat.columns.clone(),
    })
}

struct AtomSiteRow {
    label_asym_id: String,
    label_seq_id: i32,
    label_comp_id: String,
    label_atom_id: String,
    label_entity_id: Option<String>,
    label_alt_id: Option<u8>,
    auth_asym_id: Option<String>,
    auth_seq_id: Option<i32>,
    auth_comp_id: Option<String>,
    auth_atom_id: Option<String>,
    ins_code: Option<u8>,
    type_symbol: Option<String>,
    x: f32,
    y: f32,
    z: f32,
    occupancy: f32,
    b_factor: f32,
    formal_charge: i8,
    model: Option<i32>,
}

#[allow(
    clippy::too_many_lines,
    reason = "single function colocates every _atom_site column extraction"
)]
fn decode_atom_site(
    atom_site: &CategoryView,
) -> Result<Vec<AtomSiteRow>, AdapterError> {
    let n = atom_site.row_count;

    let x = need_floats(atom_site, "Cartn_x")?;
    let y = need_floats(atom_site, "Cartn_y")?;
    let z = need_floats(atom_site, "Cartn_z")?;
    let coord_mask = combine_coord_masks(
        x.mask.as_ref(),
        y.mask.as_ref(),
        z.mask.as_ref(),
        n,
    );

    let label_atom_id = need_strings(atom_site, "label_atom_id")?;
    let label_comp_id = need_strings(atom_site, "label_comp_id")?;
    let label_asym_id = need_strings(atom_site, "label_asym_id")?;
    let label_seq_id = optional_ints(atom_site, "label_seq_id")?;
    let label_entity_id = optional_strings(atom_site, "label_entity_id")?;
    let label_alt_id = optional_strings(atom_site, "label_alt_id")?;
    let type_symbol = optional_strings(atom_site, "type_symbol")?;
    let occupancy = optional_floats(atom_site, "occupancy")?;
    let b_iso = optional_floats(atom_site, "B_iso_or_equiv")?;
    let pdb_ins_code = optional_strings(atom_site, "pdbx_PDB_ins_code")?;
    let pdb_model_num = optional_ints(atom_site, "pdbx_PDB_model_num")?;
    let formal_charge = optional_ints(atom_site, "pdbx_formal_charge")?;
    let auth_asym_id = optional_strings(atom_site, "auth_asym_id")?;
    let auth_seq_id = optional_ints(atom_site, "auth_seq_id")?;
    let auth_comp_id = optional_strings(atom_site, "auth_comp_id")?;
    let auth_atom_id = optional_strings(atom_site, "auth_atom_id")?;

    let mut out: Vec<AtomSiteRow> = Vec::with_capacity(n);
    for i in 0..n {
        if coord_mask.is_masked(i) {
            continue;
        }
        let label_asym = string_at(&label_asym_id, i);
        if label_asym.is_empty() {
            continue;
        }
        let auth_seq = opt_int_at(auth_seq_id.as_ref(), i);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "BCIF doubles narrow to f32 for storage"
        )]
        out.push(AtomSiteRow {
            label_asym_id: label_asym,
            // Waters/non-polymer rows carry an absent `label_seq_id`; their
            // per-instance discriminator lives in `auth_seq_id`. Mirror the
            // PDB path (author number into the seq slot) so each gets a
            // distinct residue key instead of all collapsing onto 0.
            label_seq_id: opt_int_at(label_seq_id.as_ref(), i)
                .or(auth_seq)
                .unwrap_or(0),
            label_comp_id: string_at(&label_comp_id, i),
            label_atom_id: string_at(&label_atom_id, i),
            label_entity_id: opt_string_at(label_entity_id.as_ref(), i),
            label_alt_id: opt_string_at(label_alt_id.as_ref(), i)
                .and_then(|s| s.bytes().next())
                .filter(|&b| b != b' '),
            auth_asym_id: opt_string_at(auth_asym_id.as_ref(), i),
            auth_seq_id: auth_seq,
            auth_comp_id: opt_string_at(auth_comp_id.as_ref(), i),
            auth_atom_id: opt_string_at(auth_atom_id.as_ref(), i),
            ins_code: opt_string_at(pdb_ins_code.as_ref(), i)
                .and_then(|s| s.bytes().next())
                .filter(|&b| b != b' '),
            type_symbol: opt_string_at(type_symbol.as_ref(), i),
            x: float_at_default(&x, i, 0.0) as f32,
            y: float_at_default(&y, i, 0.0) as f32,
            z: float_at_default(&z, i, 0.0) as f32,
            occupancy: float_at_opt(occupancy.as_ref(), i, 1.0) as f32,
            b_factor: float_at_opt(b_iso.as_ref(), i, 0.0) as f32,
            formal_charge: opt_int_at(formal_charge.as_ref(), i)
                .and_then(|n| i8::try_from(n).ok())
                .unwrap_or(0),
            model: opt_int_at(pdb_model_num.as_ref(), i),
        });
    }
    Ok(out)
}

fn smallest_model(rows: &[AtomSiteRow]) -> Option<i32> {
    rows.iter().filter_map(|r| r.model).min()
}

fn push_row(
    builder: &mut EntityBuilder,
    row: &AtomSiteRow,
) -> Result<bool, AdapterError> {
    let element =
        resolve_element(row.type_symbol.as_deref(), &row.label_atom_id);
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
        ins_code: row.ins_code,
        element,
        x: row.x,
        y: row.y,
        z: row.z,
        occupancy: row.occupancy,
        b_factor: row.b_factor,
        formal_charge: row.formal_charge,
    };
    builder
        .push_atom(atom_row)
        .map_err(|e| map_build_error(&e))?;
    Ok(true)
}

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

fn name_to_bytes<const N: usize>(name: &str) -> [u8; N] {
    let mut buf = [b' '; N];
    for (i, b) in name.bytes().take(N).enumerate() {
        buf[i] = b;
    }
    buf
}

fn map_build_error(err: &BuildError) -> AdapterError {
    match err {
        BuildError::TooManyChains { limit } => too_many_chains_error(*limit),
        BuildError::InvalidCoordinate { .. } => {
            AdapterError::InvalidFormat(err.to_string())
        }
    }
}

// Column extraction + mask-aware accessors

/// One BCIF column decoded with its optional mask.
///
/// The mask array (per BCIF v0.3.0) carries one byte per row: 0 = present,
/// 1 = `.` (inapplicable), 2 = `?` (unknown).
struct Floats {
    data: Vec<f64>,
    mask: Option<Vec<u8>>,
}

struct Ints {
    data: Vec<i32>,
    mask: Option<Vec<u8>>,
}

struct Strings {
    data: Vec<String>,
    mask: Option<Vec<u8>>,
}

fn need_floats(cat: &CategoryView, name: &str) -> Result<Floats, AdapterError> {
    let col = require_col(cat, name)?;
    let data = decode_floats_data(&col.data, cat.row_count, name)?;
    let mask = decode_mask(col.mask.as_ref(), cat.row_count, name)?;
    Ok(Floats { data, mask })
}

fn need_strings(
    cat: &CategoryView,
    name: &str,
) -> Result<Strings, AdapterError> {
    let col = require_col(cat, name)?;
    let data = decode_strings_data(&col.data, cat.row_count, name)?;
    let mask = decode_mask(col.mask.as_ref(), cat.row_count, name)?;
    Ok(Strings { data, mask })
}

fn optional_floats(
    cat: &CategoryView,
    name: &str,
) -> Result<Option<Floats>, AdapterError> {
    let Some(col) = cat.columns.get(name) else {
        return Ok(None);
    };
    let data = decode_floats_data(&col.data, cat.row_count, name)?;
    let mask = decode_mask(col.mask.as_ref(), cat.row_count, name)?;
    Ok(Some(Floats { data, mask }))
}

fn optional_ints(
    cat: &CategoryView,
    name: &str,
) -> Result<Option<Ints>, AdapterError> {
    let Some(col) = cat.columns.get(name) else {
        return Ok(None);
    };
    let data = decode_ints_data(&col.data, cat.row_count, name)?;
    let mask = decode_mask(col.mask.as_ref(), cat.row_count, name)?;
    Ok(Some(Ints { data, mask }))
}

fn optional_strings(
    cat: &CategoryView,
    name: &str,
) -> Result<Option<Strings>, AdapterError> {
    let Some(col) = cat.columns.get(name) else {
        return Ok(None);
    };
    let data = decode_strings_data(&col.data, cat.row_count, name)?;
    let mask = decode_mask(col.mask.as_ref(), cat.row_count, name)?;
    Ok(Some(Strings { data, mask }))
}

/// All strings for a category column with mask honoured; masked rows
/// become `None`. Used by `_entity` and `_entity_poly` pre-pass; these
/// don't have a coordinate-mask analogue, so the mask just collapses
/// individual cells.
fn strings_with_mask(
    cat: &CategoryView,
    name: &str,
) -> Result<Option<Vec<Option<String>>>, AdapterError> {
    let Some(s) = optional_strings(cat, name)? else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(s.data.len());
    for i in 0..s.data.len() {
        if cell_masked(s.mask.as_ref(), i) {
            out.push(None);
        } else {
            out.push(Some(s.data[i].clone()));
        }
    }
    Ok(Some(out))
}

fn require_col<'a>(
    cat: &'a CategoryView,
    name: &str,
) -> Result<&'a ColumnRaw, AdapterError> {
    cat.columns.get(name).ok_or_else(|| {
        AdapterError::InvalidFormat(format!(
            "Category {}: missing column '{name}'",
            cat.name
        ))
    })
}

fn decode_floats_data(
    data: &MsgVal,
    expected: usize,
    name: &str,
) -> Result<Vec<f64>, AdapterError> {
    match decode_column(data)? {
        ColData::FloatArray(v) => check_len(v, expected, name),
        ColData::IntArray(v) => {
            check_len(v.iter().map(|&x| f64::from(x)).collect(), expected, name)
        }
        _ => Err(AdapterError::InvalidFormat(format!(
            "Column '{name}': expected float array"
        ))),
    }
}

fn decode_ints_data(
    data: &MsgVal,
    expected: usize,
    name: &str,
) -> Result<Vec<i32>, AdapterError> {
    match decode_column(data)? {
        ColData::IntArray(v) => check_len(v, expected, name),
        _ => Err(AdapterError::InvalidFormat(format!(
            "Column '{name}': expected int array"
        ))),
    }
}

fn decode_strings_data(
    data: &MsgVal,
    expected: usize,
    name: &str,
) -> Result<Vec<String>, AdapterError> {
    match decode_column(data)? {
        ColData::StringArray(v) => check_len(v, expected, name),
        _ => Err(AdapterError::InvalidFormat(format!(
            "Column '{name}': expected string array"
        ))),
    }
}

fn check_len<T>(
    v: Vec<T>,
    expected: usize,
    name: &str,
) -> Result<Vec<T>, AdapterError> {
    if v.len() == expected {
        Ok(v)
    } else {
        Err(AdapterError::InvalidFormat(format!(
            "Column '{name}': expected {expected} rows, got {}",
            v.len()
        )))
    }
}

/// Decode a column's optional `mask` array (BCIF v0.3.0 section 5).
///
/// The mask is itself an encoded-data node with int values 0 (present),
/// 1 (`.`), or 2 (`?`). Returns `Ok(None)` when no mask is attached.
fn decode_mask(
    mask: Option<&MsgVal>,
    expected: usize,
    name: &str,
) -> Result<Option<Vec<u8>>, AdapterError> {
    let Some(mask) = mask else { return Ok(None) };
    let ColData::IntArray(v) = decode_column(mask)? else {
        return Err(AdapterError::InvalidFormat(format!(
            "Column '{name}': mask must decode to int array"
        )));
    };
    if v.len() != expected {
        return Err(AdapterError::InvalidFormat(format!(
            "Column '{name}': mask length {} disagrees with row count \
             {expected}",
            v.len()
        )));
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "mask byte is 0..=2 per spec"
    )]
    let bytes: Vec<u8> = v.into_iter().map(|n| n as u8).collect();
    Ok(Some(bytes))
}

/// Combine per-coordinate masks: a row is dropped if any of its `Cartn_*`
/// columns is masked.
struct CoordMask(Option<Vec<bool>>);

impl CoordMask {
    fn is_masked(&self, i: usize) -> bool {
        self.0
            .as_ref()
            .is_some_and(|m| m.get(i).copied().unwrap_or(true))
    }
}

fn combine_coord_masks(
    mx: Option<&Vec<u8>>,
    my: Option<&Vec<u8>>,
    mz: Option<&Vec<u8>>,
    n: usize,
) -> CoordMask {
    if mx.is_none() && my.is_none() && mz.is_none() {
        return CoordMask(None);
    }
    let mut mask = vec![false; n];
    for src in [mx, my, mz].into_iter().flatten() {
        for (i, &m) in src.iter().enumerate().take(n) {
            if m != 0 {
                mask[i] = true;
            }
        }
    }
    CoordMask(Some(mask))
}

fn cell_masked(mask: Option<&Vec<u8>>, i: usize) -> bool {
    mask.is_some_and(|m| m.get(i).copied().unwrap_or(0) != 0)
}

fn string_at(s: &Strings, i: usize) -> String {
    if cell_masked(s.mask.as_ref(), i) {
        return String::new();
    }
    s.data.get(i).cloned().unwrap_or_default()
}

fn float_at_default(s: &Floats, i: usize, default: f64) -> f64 {
    if cell_masked(s.mask.as_ref(), i) {
        return default;
    }
    s.data.get(i).copied().unwrap_or(default)
}

fn float_at_opt(s: Option<&Floats>, i: usize, default: f64) -> f64 {
    s.map_or(default, |s| float_at_default(s, i, default))
}

fn opt_string_at(s: Option<&Strings>, i: usize) -> Option<String> {
    let s = s?;
    if cell_masked(s.mask.as_ref(), i) {
        return None;
    }
    let v = s.data.get(i)?;
    if v.is_empty() || v == "." || v == "?" {
        None
    } else {
        Some(v.clone())
    }
}

fn opt_int_at(s: Option<&Ints>, i: usize) -> Option<i32> {
    let s = s?;
    if cell_masked(s.mask.as_ref(), i) {
        return None;
    }
    s.data.get(i).copied()
}
