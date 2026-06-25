// foldit:allow-long-file: BinaryCIF decode pipeline + selective-category
// msgpack walk (skip_value/read_value share the marker grammar); cohesive
// decode unit.
//! BinaryCIF decode pipeline: pull the single coordinate data block out of
//! a MessagePack-encoded BinaryCIF byte stream, run the `_entity` /
//! `_entity_poly` hint pre-pass, then iterate `_atom_site` rows into an
//! [`EntityBuilder`].

use std::collections::HashMap;
use std::io::Read;

use super::codec::{
    decode_column, read_array_len, read_map_len, read_str, read_value,
    skip_value, ColData, MsgVal, Reader, StringColumn,
};
use super::hint::resolve_hint;
use super::refuse::multi_block_error;
use crate::element::Element;
use crate::entity::molecule::{
    AtomCells, BuildError, Completion, EntityBuilder, MoleculeEntity,
};
use crate::ops::error::AdapterError;

// Public entry points (called by adapters::bcif::mod.rs)

pub(super) fn decode_to_entities(
    bytes: &[u8],
    completion: Completion,
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    let block = open_block(bytes)?;
    let atom_site = require_atom_site(&block)?;
    let cols = decode_atom_site(atom_site)?;
    let target = cols.smallest_model();

    let mut builder = EntityBuilder::with_completion(completion);
    register_hints(&block, &mut builder)?;

    let mut any_atom = false;
    for i in 0..cols.len() {
        if !cols.is_emitted(i) {
            continue;
        }
        if let Some(target) = target {
            if cols.model_at(i).is_some_and(|m| m != target) {
                continue;
            }
        }
        cols.push_cells(&mut builder, i)?;
        any_atom = true;
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
    let cols = decode_atom_site(atom_site)?;

    let mut buckets: HashMap<i32, Vec<usize>> = HashMap::new();
    let mut order: Vec<i32> = Vec::new();
    let default_model: i32 = 1;
    for i in 0..cols.len() {
        if !cols.is_emitted(i) {
            continue;
        }
        let key = cols.model_at(i).unwrap_or(default_model);
        buckets
            .entry(key)
            .or_insert_with(|| {
                order.push(key);
                Vec::new()
            })
            .push(i);
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
        for i in indices {
            cols.push_cells(&mut builder, i)?;
            any_atom = true;
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

struct ColumnRaw {
    data: MsgVal,
    mask: Option<MsgVal>,
}

/// The only categories the parser reads. Every other category in the file is
/// walked for structure but its column trees are skipped, never allocated.
fn is_read_category(name: &str) -> bool {
    matches!(name, "_atom_site" | "_entity" | "_entity_poly")
}

fn open_block(bytes: &[u8]) -> Result<BlockView, AdapterError> {
    let data = decompress_if_gzip(bytes)?;
    let mut rd: Reader = std::io::Cursor::new(&data);

    // root map -> the single `dataBlocks` array
    let block_count = walk_into_array(&mut rd, "dataBlocks", "root")?;
    if block_count == 0 {
        return Err(AdapterError::InvalidFormat("No data blocks found".into()));
    }
    if block_count > 1 {
        return Err(multi_block_error(block_count));
    }

    // block map -> the `categories` array
    let cat_count = walk_into_array(&mut rd, "categories", "data block")?;
    let mut categories: Vec<CategoryView> = Vec::new();
    for _ in 0..cat_count {
        if let Some(view) = read_category_selective(&mut rd)? {
            categories.push(view);
        }
    }
    Ok(BlockView { categories })
}

/// From a map at the cursor, find the array-valued entry named `key`, skip the
/// map's other entries, descend into that array, and return its element count.
/// `ctx` names the enclosing map for error messages.
fn walk_into_array(
    rd: &mut Reader,
    key: &str,
    ctx: &str,
) -> Result<usize, AdapterError> {
    let entries = read_map_len(rd)?;
    for _ in 0..entries {
        let k = read_str(rd)?;
        if k == key {
            return read_array_len(rd);
        }
        skip_value(rd)?;
    }
    Err(AdapterError::InvalidFormat(format!(
        "Missing '{key}' in {ctx}"
    )))
}

/// Read one category map. Builds the full column tree only when the category is
/// one the parser reads; otherwise the column array is skipped (no `MsgVal`
/// allocation) and the category is dropped from the view. `name` may appear in
/// any key position, so the `columns` value is skipped on first pass and
/// re-read from its recorded offset once the name proves it is wanted.
fn read_category_selective(
    rd: &mut Reader,
) -> Result<Option<CategoryView>, AdapterError> {
    let entries = read_map_len(rd)?;
    let mut name: Option<String> = None;
    let mut row_count: Option<MsgVal> = None;
    let mut columns_at: Option<u64> = None;

    for _ in 0..entries {
        let key = read_str(rd)?;
        match key.as_str() {
            "name" => name = Some(read_str(rd)?),
            "rowCount" => row_count = Some(read_value(rd)?),
            "columns" => {
                columns_at = Some(rd.position());
                skip_value(rd)?;
            }
            _ => skip_value(rd)?,
        }
    }

    let Some(name) = name else {
        return Ok(None);
    };
    if !is_read_category(&name) {
        return Ok(None);
    }

    // The first pass left the cursor at the end of the category map. Re-reading
    // `columns` rewinds it, so restore this end before returning — `columns` is
    // not guaranteed to be the map's last entry.
    let map_end = rd.position();
    let mut pairs: Vec<(MsgVal, MsgVal)> =
        vec![(MsgVal::Str("name".into()), MsgVal::Str(name))];
    if let Some(rc) = row_count {
        pairs.push((MsgVal::Str("rowCount".into()), rc));
    }
    if let Some(at) = columns_at {
        rd.set_position(at);
        pairs.push((MsgVal::Str("columns".into()), read_value(rd)?));
        rd.set_position(map_end);
    }
    parse_category(MsgVal::Map(pairs))
}

fn parse_category(
    mut cat: MsgVal,
) -> Result<Option<CategoryView>, AdapterError> {
    let Some(name) = cat.get("name").and_then(MsgVal::as_str) else {
        return Ok(None);
    };
    let name = name.to_owned();
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
    if let Some(cols) = cat.take("columns").and_then(MsgVal::into_array) {
        for mut col in cols {
            let Some(cname) = col.get("name").and_then(MsgVal::as_str) else {
                continue;
            };
            let cname = cname.to_owned();
            let Some(data) = col.take("data") else {
                continue;
            };
            let mask = col.take("mask").filter(is_present_msg);
            let _ = columns.insert(cname, ColumnRaw { data, mask });
        }
    }
    Ok(Some(CategoryView {
        name,
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

fn require_atom_site(block: &BlockView) -> Result<&CategoryView, AdapterError> {
    block.find_category("_atom_site").ok_or_else(|| {
        AdapterError::InvalidFormat("No '_atom_site' category found".into())
    })
}

/// Every `_atom_site` column decoded once into its native columnar buffer.
///
/// Numeric columns are typed `Vec`s; categorical columns are a unique-value
/// set plus a per-row index ([`StringColumn`]). Rows are materialized into the
/// builder by indexing these buffers — no intermediate per-row value tree and
/// no per-cell `String`.
struct AtomSiteColumns {
    n: usize,
    coord_mask: CoordMask,
    x: Floats,
    y: Floats,
    z: Floats,
    label_atom_id: Strings,
    label_comp_id: Strings,
    label_asym_id: Strings,
    label_seq_id: Option<Ints>,
    label_entity_id: Option<Strings>,
    label_alt_id: Option<Strings>,
    type_symbol: Option<Strings>,
    occupancy: Option<Floats>,
    b_iso: Option<Floats>,
    pdb_ins_code: Option<Strings>,
    pdb_model_num: Option<Ints>,
    formal_charge: Option<Ints>,
    /// Decoded but unread by ingest: the accumulator keys chains on
    /// `label_asym_id` only. Retained in the column set for the egress path.
    #[allow(dead_code, reason = "consumed by the mmCIF/BCIF write path")]
    auth_asym_id: Option<Strings>,
    auth_seq_id: Option<Ints>,
    auth_comp_id: Option<Strings>,
    auth_atom_id: Option<Strings>,
}

impl AtomSiteColumns {
    fn len(&self) -> usize {
        self.n
    }

    /// Whether row `i` survives the row-level filters: coordinate-masked rows
    /// and rows with an empty `label_asym_id` are dropped, as for the PDB path.
    fn is_emitted(&self, i: usize) -> bool {
        !self.coord_mask.is_masked(i)
            && !string_at(&self.label_asym_id, i).is_empty()
    }

    fn model_at(&self, i: usize) -> Option<i32> {
        opt_int_at(self.pdb_model_num.as_ref(), i)
    }

    fn smallest_model(&self) -> Option<i32> {
        (0..self.n)
            .filter(|&i| self.is_emitted(i))
            .filter_map(|i| self.model_at(i))
            .min()
    }

    /// Materialize row `i`'s cells straight from the decoded columns and
    /// route them through the builder's accumulator, skipping the per-atom
    /// `AtomRow` struct and the chain-id `CompactString`s the text-format
    /// `push_atom` path pays. The builder owns coordinate validation and
    /// atom-name normalization, so the cells carry raw (un-normalized)
    /// atom-name bytes.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "BCIF doubles narrow to f32 for storage"
    )]
    fn push_cells(
        &self,
        builder: &mut EntityBuilder,
        i: usize,
    ) -> Result<(), AdapterError> {
        let label_atom_id = string_at(&self.label_atom_id, i);
        let type_symbol = opt_string_at(self.type_symbol.as_ref(), i);
        let element = resolve_element(type_symbol, label_atom_id);
        let auth_seq = opt_int_at(self.auth_seq_id.as_ref(), i);
        builder
            .push_cells(AtomCells {
                label_asym_id: string_at(&self.label_asym_id, i),
                // Waters/non-polymer rows carry an absent `label_seq_id`;
                // their per-instance discriminator lives in `auth_seq_id`.
                // Mirror the PDB path (author number into the seq slot) so
                // each gets a distinct residue key instead of all collapsing
                // onto 0.
                label_seq_id: opt_int_at(self.label_seq_id.as_ref(), i)
                    .or(auth_seq)
                    .unwrap_or(0),
                label_comp_id: name_to_bytes::<3>(string_at(
                    &self.label_comp_id,
                    i,
                )),
                label_atom_id: name_to_bytes::<4>(label_atom_id),
                label_entity_id: opt_string_at(
                    self.label_entity_id.as_ref(),
                    i,
                ),
                auth_seq_id: auth_seq,
                auth_comp_id: opt_string_at(self.auth_comp_id.as_ref(), i)
                    .map(name_to_bytes::<3>),
                auth_atom_id: opt_string_at(self.auth_atom_id.as_ref(), i)
                    .map(name_to_bytes::<4>),
                alt_loc: opt_string_at(self.label_alt_id.as_ref(), i)
                    .and_then(|s| s.bytes().next())
                    .filter(|&b| b != b' '),
                ins_code: opt_string_at(self.pdb_ins_code.as_ref(), i)
                    .and_then(|s| s.bytes().next())
                    .filter(|&b| b != b' '),
                element,
                x: float_at_default(&self.x, i, 0.0) as f32,
                y: float_at_default(&self.y, i, 0.0) as f32,
                z: float_at_default(&self.z, i, 0.0) as f32,
                occupancy: float_at_opt(self.occupancy.as_ref(), i, 1.0) as f32,
                b_factor: float_at_opt(self.b_iso.as_ref(), i, 0.0) as f32,
                formal_charge: opt_int_at(self.formal_charge.as_ref(), i)
                    .and_then(|n| i8::try_from(n).ok())
                    .unwrap_or(0),
            })
            .map_err(|e| map_build_error(&e))?;
        Ok(())
    }
}

fn decode_atom_site(
    atom_site: &CategoryView,
) -> Result<AtomSiteColumns, AdapterError> {
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

    Ok(AtomSiteColumns {
        n,
        coord_mask,
        x,
        y,
        z,
        label_atom_id: need_strings(atom_site, "label_atom_id")?,
        label_comp_id: need_strings(atom_site, "label_comp_id")?,
        label_asym_id: need_strings(atom_site, "label_asym_id")?,
        label_seq_id: optional_ints(atom_site, "label_seq_id")?,
        label_entity_id: optional_strings(atom_site, "label_entity_id")?,
        label_alt_id: optional_strings(atom_site, "label_alt_id")?,
        type_symbol: optional_strings(atom_site, "type_symbol")?,
        occupancy: optional_floats(atom_site, "occupancy")?,
        b_iso: optional_floats(atom_site, "B_iso_or_equiv")?,
        pdb_ins_code: optional_strings(atom_site, "pdbx_PDB_ins_code")?,
        pdb_model_num: optional_ints(atom_site, "pdbx_PDB_model_num")?,
        formal_charge: optional_ints(atom_site, "pdbx_formal_charge")?,
        auth_asym_id: optional_strings(atom_site, "auth_asym_id")?,
        auth_seq_id: optional_ints(atom_site, "auth_seq_id")?,
        auth_comp_id: optional_strings(atom_site, "auth_comp_id")?,
        auth_atom_id: optional_strings(atom_site, "auth_atom_id")?,
    })
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
    data: StringColumn,
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
            out.push(Some(s.data.at(i).to_owned()));
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
) -> Result<StringColumn, AdapterError> {
    match decode_column(data)? {
        ColData::StringArray(col) => {
            if col.len() == expected {
                Ok(col)
            } else {
                Err(AdapterError::InvalidFormat(format!(
                    "Column '{name}': expected {expected} rows, got {}",
                    col.len()
                )))
            }
        }
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

fn string_at(s: &Strings, i: usize) -> &str {
    if cell_masked(s.mask.as_ref(), i) {
        return "";
    }
    s.data.at(i)
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

fn opt_string_at(s: Option<&Strings>, i: usize) -> Option<&str> {
    let s = s?;
    if cell_masked(s.mask.as_ref(), i) {
        return None;
    }
    let v = s.data.at(i);
    if v.is_empty() || v == "." || v == "?" {
        None
    } else {
        Some(v)
    }
}

fn opt_int_at(s: Option<&Ints>, i: usize) -> Option<i32> {
    let s = s?;
    if cell_masked(s.mask.as_ref(), i) {
        return None;
    }
    s.data.get(i).copied()
}
