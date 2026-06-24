//! DOM-fallback path: walk a parsed [`Document`] and push atom rows into
//! an [`EntityBuilder`]. Used when the scanner declines to handle an
//! input.

use std::collections::{HashMap, HashSet};

use compact_str::CompactString;

use super::dom::{Block, Document, Value};
use super::hint::resolve_hint;
use super::parse::parse;
use super::refuse::multi_block_error;
use super::{map_build_error, name_to_bytes, resolve_element};
use crate::entity::molecule::{AtomRow, EntityBuilder, MoleculeEntity};
use crate::ops::codec::AdapterError;

/// Single-model DOM parse: returns the model whose `pdbx_PDB_model_num`
/// matches the smallest value present (typically `1`).
pub(super) fn parse_mmcif_dom_to_entities(
    input: &str,
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    let doc = parse_document(input)?;
    let block = require_single_block(&doc)?;
    let cols = AtomSiteCols::from_block(block)?;
    let target = first_model_number(block, &cols);

    let mut builder = EntityBuilder::new();
    register_hints(block, &mut builder);

    let mut any_atom = false;
    for_each_row(block, &cols, |row| {
        if let Some(target) = target {
            if row.model.is_some_and(|m| m != target) {
                return Ok(());
            }
        }
        if push_row(&mut builder, &cols, &row)? {
            any_atom = true;
        }
        Ok(())
    })?;

    if !any_atom {
        return Err(AdapterError::InvalidFormat(
            "No atoms found in CIF".into(),
        ));
    }
    finish(builder)
}

/// Multi-model DOM parse: partition rows by `pdbx_PDB_model_num`, emit
/// one `Vec<MoleculeEntity>` per distinct value (sorted ascending). A
/// file with no model column collapses to a single bucket.
pub(super) fn parse_mmcif_dom_to_all_models(
    input: &str,
) -> Result<Vec<Vec<MoleculeEntity>>, AdapterError> {
    let doc = parse_document(input)?;
    let block = require_single_block(&doc)?;
    let cols = AtomSiteCols::from_block(block)?;

    let mut model_rows: HashMap<i32, Vec<DomRow>> = HashMap::new();
    let mut order: Vec<i32> = Vec::new();
    let default_model: i32 = 1;

    for_each_row(block, &cols, |row| {
        let key = row.model.unwrap_or(default_model);
        let entry = model_rows.entry(key).or_insert_with(|| {
            order.push(key);
            Vec::new()
        });
        entry.push(row);
        Ok(())
    })?;

    if model_rows.is_empty() {
        return Err(AdapterError::InvalidFormat(
            "No atoms found in CIF".into(),
        ));
    }

    order.sort_unstable();

    let mut out: Vec<Vec<MoleculeEntity>> = Vec::with_capacity(order.len());
    for model in &order {
        let Some(rows) = model_rows.remove(model) else {
            continue;
        };
        let mut builder = EntityBuilder::new();
        register_hints(block, &mut builder);
        let mut any_atom = false;
        for row in &rows {
            if push_row(&mut builder, &cols, row)? {
                any_atom = true;
            }
        }
        if any_atom {
            out.push(finish(builder)?);
        }
    }
    if out.is_empty() {
        return Err(AdapterError::InvalidFormat(
            "No atoms found in CIF".into(),
        ));
    }
    Ok(out)
}

// Block selection + hint pre-pass

fn parse_document(input: &str) -> Result<Document, AdapterError> {
    parse(input).map_err(|e| {
        AdapterError::InvalidFormat(format!("CIF parse error: {e}"))
    })
}

fn require_single_block(doc: &Document) -> Result<&Block, AdapterError> {
    if doc.blocks.len() > 1 {
        return Err(multi_block_error(doc.blocks.len()));
    }
    doc.blocks.first().ok_or_else(|| {
        AdapterError::InvalidFormat("No data blocks in CIF".into())
    })
}

fn register_hints(block: &Block, builder: &mut EntityBuilder) {
    let poly_types = collect_poly_types(block);
    let entity_types = collect_entity_types(block);

    let mut seen: HashSet<String> = HashSet::new();
    for (id, etype) in &entity_types {
        let poly = poly_types.get(id).map(String::as_str);
        let hint = resolve_hint(etype, poly);
        builder.register_entity(id, hint);
        let _ = seen.insert(id.clone());
    }
    for (id, poly) in &poly_types {
        if seen.contains(id) {
            continue;
        }
        let hint = resolve_hint("polymer", Some(poly));
        builder.register_entity(id, hint);
    }
}

fn collect_entity_types(block: &Block) -> Vec<(String, String)> {
    if let Some(cols) = block.columns(&["_entity.id", "_entity.type"]) {
        let mut out = Vec::with_capacity(cols.nrows());
        for row in &cols {
            let (Some(id), Some(etype)) = (row[0].as_str(), row[1].as_str())
            else {
                continue;
            };
            out.push((id.to_owned(), etype.to_owned()));
        }
        return out;
    }
    let id = block
        .get("_entity.id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let etype = block
        .get("_entity.type")
        .and_then(Value::as_str)
        .map(str::to_owned);
    match (id, etype) {
        (Some(id), Some(etype)) => vec![(id, etype)],
        _ => Vec::new(),
    }
}

fn collect_poly_types(block: &Block) -> HashMap<String, String> {
    let mut out: HashMap<String, String> = HashMap::new();
    if let Some(cols) =
        block.columns(&["_entity_poly.entity_id", "_entity_poly.type"])
    {
        for row in &cols {
            let (Some(id), Some(ptype)) = (row[0].as_str(), row[1].as_str())
            else {
                continue;
            };
            let _ = out.insert(id.to_owned(), ptype.to_owned());
        }
        return out;
    }
    if let (Some(id), Some(ptype)) = (
        block.get("_entity_poly.entity_id").and_then(Value::as_str),
        block.get("_entity_poly.type").and_then(Value::as_str),
    ) {
        let _ = out.insert(id.to_owned(), ptype.to_owned());
    }
    out
}

// _atom_site column resolution and row decoding

struct AtomSiteCols {
    label_atom_id: usize,
    label_comp_id: usize,
    label_asym_id: usize,
    cartn_x: usize,
    cartn_y: usize,
    cartn_z: usize,
    label_seq_id: Option<usize>,
    label_entity_id: Option<usize>,
    label_alt_id: Option<usize>,
    type_symbol: Option<usize>,
    occupancy: Option<usize>,
    b_iso: Option<usize>,
    pdb_ins_code: Option<usize>,
    pdb_model_num: Option<usize>,
    formal_charge: Option<usize>,
    auth_asym_id: Option<usize>,
    auth_seq_id: Option<usize>,
    auth_comp_id: Option<usize>,
    auth_atom_id: Option<usize>,
    tags: Vec<String>,
}

impl AtomSiteCols {
    fn from_block(block: &Block) -> Result<Self, AdapterError> {
        let loop_ref =
            block.find_loop("_atom_site.label_atom_id").ok_or_else(|| {
                AdapterError::InvalidFormat("No _atom_site loop in CIF".into())
            })?;
        let find = |tag: &str| {
            loop_ref
                .tags
                .iter()
                .position(|t| t.eq_ignore_ascii_case(tag))
        };
        let need = |tag: &str| {
            find(tag).ok_or_else(|| {
                AdapterError::InvalidFormat(format!(
                    "_atom_site missing required column {tag}"
                ))
            })
        };
        Ok(Self {
            label_atom_id: need("_atom_site.label_atom_id")?,
            label_comp_id: need("_atom_site.label_comp_id")?,
            label_asym_id: need("_atom_site.label_asym_id")?,
            cartn_x: need("_atom_site.Cartn_x")?,
            cartn_y: need("_atom_site.Cartn_y")?,
            cartn_z: need("_atom_site.Cartn_z")?,
            label_seq_id: find("_atom_site.label_seq_id"),
            label_entity_id: find("_atom_site.label_entity_id"),
            label_alt_id: find("_atom_site.label_alt_id"),
            type_symbol: find("_atom_site.type_symbol"),
            occupancy: find("_atom_site.occupancy"),
            b_iso: find("_atom_site.B_iso_or_equiv"),
            pdb_ins_code: find("_atom_site.pdbx_PDB_ins_code"),
            pdb_model_num: find("_atom_site.pdbx_PDB_model_num"),
            formal_charge: find("_atom_site.pdbx_formal_charge"),
            auth_asym_id: find("_atom_site.auth_asym_id"),
            auth_seq_id: find("_atom_site.auth_seq_id"),
            auth_comp_id: find("_atom_site.auth_comp_id"),
            auth_atom_id: find("_atom_site.auth_atom_id"),
            tags: loop_ref.tags.clone(),
        })
    }
}

#[derive(Clone)]
struct DomRow {
    values: Vec<Value>,
    model: Option<i32>,
}

fn for_each_row<F>(
    block: &Block,
    cols: &AtomSiteCols,
    mut visit: F,
) -> Result<(), AdapterError>
where
    F: FnMut(DomRow) -> Result<(), AdapterError>,
{
    let Some(loop_ref) = block.loops.iter().find(|lp| lp.tags == cols.tags)
    else {
        return Err(AdapterError::InvalidFormat(
            "No _atom_site loop in CIF".into(),
        ));
    };
    let stride = loop_ref.tags.len();
    for r in 0..loop_ref.nrows() {
        let base = r * stride;
        let slice = loop_ref.values[base..base + stride].to_vec();
        let model = cols
            .pdb_model_num
            .and_then(|idx| slice.get(idx))
            .and_then(Value::as_i32);
        visit(DomRow {
            values: slice,
            model,
        })?;
    }
    Ok(())
}

fn push_row(
    builder: &mut EntityBuilder,
    cols: &AtomSiteCols,
    row: &DomRow,
) -> Result<bool, AdapterError> {
    let Some(atom_row) = decode_row(cols, row) else {
        return Ok(false);
    };
    builder
        .push_atom(atom_row)
        .map_err(|e| map_build_error(&e))?;
    Ok(true)
}

fn decode_row(cols: &AtomSiteCols, row: &DomRow) -> Option<AtomRow> {
    let cell = |idx: usize| -> Option<&Value> { row.values.get(idx) };
    let cell_str = |idx: Option<usize>| -> Option<&str> {
        idx.and_then(cell).and_then(Value::as_str)
    };
    let cell_f32 = |idx: Option<usize>, default: f32| -> f32 {
        #[allow(clippy::cast_possible_truncation)]
        idx.and_then(cell)
            .and_then(Value::as_f64)
            .map_or(default, |f| f as f32)
    };

    let x = cell(cols.cartn_x).and_then(Value::as_f64)?;
    let y = cell(cols.cartn_y).and_then(Value::as_f64)?;
    let z = cell(cols.cartn_z).and_then(Value::as_f64)?;

    let label_asym = cell_str(Some(cols.label_asym_id)).unwrap_or("");
    if label_asym.is_empty() {
        return None;
    }
    let label_comp = cell_str(Some(cols.label_comp_id)).unwrap_or("");
    let label_atom = cell_str(Some(cols.label_atom_id)).unwrap_or("");

    let formal_charge = cols
        .formal_charge
        .and_then(cell)
        .and_then(Value::as_i32)
        .and_then(|n| i8::try_from(n).ok())
        .unwrap_or(0);

    let auth_seq_id = cols.auth_seq_id.and_then(cell).and_then(Value::as_i32);

    #[allow(clippy::cast_possible_truncation)]
    Some(AtomRow {
        label_asym_id: CompactString::new(label_asym),
        // Waters/non-polymer rows carry `label_seq_id = "."`; their
        // per-instance discriminator lives in `auth_seq_id`. Mirror the
        // PDB path (author number into the seq slot) so each gets a
        // distinct residue key instead of all collapsing onto 0.
        label_seq_id: cols
            .label_seq_id
            .and_then(cell)
            .and_then(Value::as_i32)
            .or(auth_seq_id)
            .unwrap_or(0),
        label_comp_id: name_to_bytes::<3>(label_comp),
        label_atom_id: name_to_bytes::<4>(label_atom),
        label_entity_id: cell_str(cols.label_entity_id).map(CompactString::new),
        auth_asym_id: cell_str(cols.auth_asym_id).map(CompactString::new),
        auth_seq_id,
        auth_comp_id: cell_str(cols.auth_comp_id).map(name_to_bytes::<3>),
        auth_atom_id: cell_str(cols.auth_atom_id).map(name_to_bytes::<4>),
        alt_loc: cell_str(cols.label_alt_id)
            .and_then(|s| s.bytes().next())
            .filter(|&b| b != b' '),
        ins_code: cell_str(cols.pdb_ins_code)
            .and_then(|s| s.bytes().next())
            .filter(|&b| b != b' '),
        element: resolve_element(cell_str(cols.type_symbol), label_atom),
        x: x as f32,
        y: y as f32,
        z: z as f32,
        occupancy: cell_f32(cols.occupancy, 1.0),
        b_factor: cell_f32(cols.b_iso, 0.0),
        formal_charge,
    })
}

fn first_model_number(block: &Block, cols: &AtomSiteCols) -> Option<i32> {
    let model_col = cols.pdb_model_num?;
    let loop_ref = block.loops.iter().find(|lp| lp.tags == cols.tags)?;
    let stride = loop_ref.tags.len();
    let mut min: Option<i32> = None;
    for r in 0..loop_ref.nrows() {
        let v = &loop_ref.values[r * stride + model_col];
        if let Some(n) = v.as_i32() {
            min = Some(min.map_or(n, |cur| cur.min(n)));
        }
    }
    min
}

fn finish(builder: EntityBuilder) -> Result<Vec<MoleculeEntity>, AdapterError> {
    builder.finish().map_err(|e| map_build_error(&e))
}
