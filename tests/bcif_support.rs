//! Test-only BinaryCIF encoder used by `tests/bcif_integration.rs`.
//!
//! Emits the MessagePack envelope and the simplest possible BCIF encodings
//! (ByteArray for ints/floats, StringArray with ByteArray indices/offsets).
//! Production code stays read-only — the encoder lives only in test code.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::struct_excessive_bools,
    clippy::vec_init_then_push,
    unused_results,
    dead_code,
    reason = "test-only helper"
)]

use rmp::encode as rmp_encode;

pub(crate) struct Block {
    pub header: String,
    pub categories: Vec<Category>,
}

pub(crate) struct Category {
    pub name: String,
    pub row_count: usize,
    pub columns: Vec<Column>,
}

pub(crate) struct Column {
    pub name: String,
    pub data: ColumnData,
    pub mask: Option<Vec<u8>>,
}

pub(crate) enum ColumnData {
    Ints(Vec<i32>),
    Floats(Vec<f64>),
    Strings(Vec<String>),
    /// Pre-encoded data + encoding chain. Used to inject malformed inputs
    /// or non-default encodings (RunLength, IntegerPacking byteCount=4).
    Raw {
        bytes: Vec<u8>,
        encoding: Vec<EncStep>,
    },
}

pub(crate) enum EncStep {
    ByteArrayInt8,   // type 1
    ByteArrayInt16,  // type 2
    ByteArrayInt32,  // type 3
    ByteArrayUint8,  // type 4
    ByteArrayUint16, // type 5
    ByteArrayUint32, // type 6
    ByteArrayF32,    // type 32
    ByteArrayF64,    // type 33
    RunLength {
        src_size: i32,
        src_type: i32,
    },
    IntegerPacking {
        byte_count: i32,
        src_size: i32,
        is_unsigned: bool,
    },
    StringArray {
        string_data: String,
        offsets: Vec<i32>,
    },
    Delta {
        origin: i32,
        src_type: i32,
    },
    FixedPoint {
        factor: f64,
        src_type: i32,
    },
    IntervalQuantization {
        min: f64,
        max: f64,
        num_steps: i32,
        src_type: i32,
    },
}

pub(crate) fn build_bcif(blocks: &[Block]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    // top-level map: version, encoder, dataBlocks
    rmp_encode::write_map_len(&mut out, 3).unwrap();
    write_str_kv(&mut out, "version", "0.3.0");
    write_str_kv(&mut out, "encoder", "molex-test");
    rmp_encode::write_str(&mut out, "dataBlocks").unwrap();
    write_array_len(&mut out, blocks.len());
    for block in blocks {
        write_block(&mut out, block);
    }
    out
}

fn write_block(out: &mut Vec<u8>, block: &Block) {
    rmp_encode::write_map_len(out, 2).unwrap();
    write_str_kv(out, "header", &block.header);
    rmp_encode::write_str(out, "categories").unwrap();
    write_array_len(out, block.categories.len());
    for cat in &block.categories {
        write_category(out, cat);
    }
}

fn write_category(out: &mut Vec<u8>, cat: &Category) {
    rmp_encode::write_map_len(out, 3).unwrap();
    write_str_kv(out, "name", &cat.name);
    rmp_encode::write_str(out, "rowCount").unwrap();
    rmp_encode::write_sint(out, cat.row_count as i64).unwrap();
    rmp_encode::write_str(out, "columns").unwrap();
    write_array_len(out, cat.columns.len());
    for col in &cat.columns {
        write_column(out, col, cat.row_count);
    }
}

fn write_column(out: &mut Vec<u8>, col: &Column, row_count: usize) {
    let has_mask = col.mask.is_some();
    let map_len: u32 = if has_mask { 3 } else { 2 };
    rmp_encode::write_map_len(out, map_len).unwrap();
    write_str_kv(out, "name", &col.name);
    rmp_encode::write_str(out, "data").unwrap();
    write_column_data(out, &col.data);
    if let Some(mask) = &col.mask {
        rmp_encode::write_str(out, "mask").unwrap();
        let _ = row_count; // mask length expected to match by caller
        write_encoded_data(out, mask, &[EncStep::ByteArrayUint8]);
    }
}

fn write_column_data(out: &mut Vec<u8>, data: &ColumnData) {
    match data {
        ColumnData::Ints(v) => {
            let bytes = ints_to_bytes_int32_le(v);
            write_encoded_data(out, &bytes, &[EncStep::ByteArrayInt32]);
        }
        ColumnData::Floats(v) => {
            let bytes = floats_to_bytes_f64_le(v);
            write_encoded_data(out, &bytes, &[EncStep::ByteArrayF64]);
        }
        ColumnData::Strings(v) => {
            let (indices, string_data, offsets) = pack_strings(v);
            let index_bytes = ints_to_bytes_int32_le(&indices);
            let enc = vec![EncStep::StringArray {
                string_data,
                offsets,
            }];
            write_encoded_data(out, &index_bytes, &enc);
        }
        ColumnData::Raw { bytes, encoding } => {
            write_encoded_data(out, bytes, encoding);
        }
    }
}

/// Pack a vector of strings into (indices, stringData, offsets) where
/// each row gets its own entry in stringData (no deduplication —
/// adequate for tests).
fn pack_strings(rows: &[String]) -> (Vec<i32>, String, Vec<i32>) {
    let mut indices = Vec::with_capacity(rows.len());
    let mut string_data = String::new();
    let mut offsets: Vec<i32> = vec![0];
    for (i, s) in rows.iter().enumerate() {
        string_data.push_str(s);
        offsets.push(string_data.len() as i32);
        indices.push(i as i32);
    }
    (indices, string_data, offsets)
}

fn write_encoded_data(out: &mut Vec<u8>, bytes: &[u8], encoding: &[EncStep]) {
    rmp_encode::write_map_len(out, 2).unwrap();
    rmp_encode::write_str(out, "data").unwrap();
    rmp_encode::write_bin(out, bytes).unwrap();
    rmp_encode::write_str(out, "encoding").unwrap();
    write_array_len(out, encoding.len());
    for step in encoding {
        write_enc_step(out, step);
    }
}

fn write_enc_step(out: &mut Vec<u8>, step: &EncStep) {
    match step {
        EncStep::ByteArrayInt8 => write_byte_array(out, 1),
        EncStep::ByteArrayInt16 => write_byte_array(out, 2),
        EncStep::ByteArrayInt32 => write_byte_array(out, 3),
        EncStep::ByteArrayUint8 => write_byte_array(out, 4),
        EncStep::ByteArrayUint16 => write_byte_array(out, 5),
        EncStep::ByteArrayUint32 => write_byte_array(out, 6),
        EncStep::ByteArrayF32 => write_byte_array(out, 32),
        EncStep::ByteArrayF64 => write_byte_array(out, 33),
        EncStep::RunLength { src_size, src_type } => {
            rmp_encode::write_map_len(out, 3).unwrap();
            write_str_kv(out, "kind", "RunLength");
            rmp_encode::write_str(out, "srcType").unwrap();
            rmp_encode::write_sint(out, i64::from(*src_type)).unwrap();
            rmp_encode::write_str(out, "srcSize").unwrap();
            rmp_encode::write_sint(out, i64::from(*src_size)).unwrap();
        }
        EncStep::IntegerPacking {
            byte_count,
            src_size,
            is_unsigned,
        } => {
            rmp_encode::write_map_len(out, 4).unwrap();
            write_str_kv(out, "kind", "IntegerPacking");
            rmp_encode::write_str(out, "byteCount").unwrap();
            rmp_encode::write_sint(out, i64::from(*byte_count)).unwrap();
            rmp_encode::write_str(out, "srcSize").unwrap();
            rmp_encode::write_sint(out, i64::from(*src_size)).unwrap();
            rmp_encode::write_str(out, "isUnsigned").unwrap();
            rmp_encode::write_bool(out, *is_unsigned).unwrap();
        }
        EncStep::StringArray {
            string_data,
            offsets,
        } => {
            rmp_encode::write_map_len(out, 5).unwrap();
            write_str_kv(out, "kind", "StringArray");
            rmp_encode::write_str(out, "stringData").unwrap();
            rmp_encode::write_str(out, string_data).unwrap();
            rmp_encode::write_str(out, "offsets").unwrap();
            rmp_encode::write_bin(out, &ints_to_bytes_int32_le(offsets))
                .unwrap();
            rmp_encode::write_str(out, "offsetEncoding").unwrap();
            write_array_len(out, 1);
            write_byte_array(out, 3);
            rmp_encode::write_str(out, "dataEncoding").unwrap();
            write_array_len(out, 1);
            write_byte_array(out, 3);
        }
        EncStep::Delta { origin, src_type } => {
            rmp_encode::write_map_len(out, 3).unwrap();
            write_str_kv(out, "kind", "Delta");
            rmp_encode::write_str(out, "origin").unwrap();
            rmp_encode::write_sint(out, i64::from(*origin)).unwrap();
            rmp_encode::write_str(out, "srcType").unwrap();
            rmp_encode::write_sint(out, i64::from(*src_type)).unwrap();
        }
        EncStep::FixedPoint { factor, src_type } => {
            rmp_encode::write_map_len(out, 3).unwrap();
            write_str_kv(out, "kind", "FixedPoint");
            rmp_encode::write_str(out, "factor").unwrap();
            rmp_encode::write_f64(out, *factor).unwrap();
            rmp_encode::write_str(out, "srcType").unwrap();
            rmp_encode::write_sint(out, i64::from(*src_type)).unwrap();
        }
        EncStep::IntervalQuantization {
            min,
            max,
            num_steps,
            src_type,
        } => {
            rmp_encode::write_map_len(out, 5).unwrap();
            write_str_kv(out, "kind", "IntervalQuantization");
            rmp_encode::write_str(out, "min").unwrap();
            rmp_encode::write_f64(out, *min).unwrap();
            rmp_encode::write_str(out, "max").unwrap();
            rmp_encode::write_f64(out, *max).unwrap();
            rmp_encode::write_str(out, "numSteps").unwrap();
            rmp_encode::write_sint(out, i64::from(*num_steps)).unwrap();
            rmp_encode::write_str(out, "srcType").unwrap();
            rmp_encode::write_sint(out, i64::from(*src_type)).unwrap();
        }
    }
}

fn write_byte_array(out: &mut Vec<u8>, type_id: i32) {
    rmp_encode::write_map_len(out, 2).unwrap();
    write_str_kv(out, "kind", "ByteArray");
    rmp_encode::write_str(out, "type").unwrap();
    rmp_encode::write_sint(out, i64::from(type_id)).unwrap();
}

fn write_str_kv(out: &mut Vec<u8>, key: &str, value: &str) {
    rmp_encode::write_str(out, key).unwrap();
    rmp_encode::write_str(out, value).unwrap();
}

fn write_array_len(out: &mut Vec<u8>, len: usize) {
    rmp_encode::write_array_len(out, len as u32).unwrap();
}

fn write_map_len(out: &mut Vec<u8>, len: usize) {
    rmp_encode::write_map_len(out, len as u32).unwrap();
}

fn ints_to_bytes_int32_le(ints: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ints.len() * 4);
    for &i in ints {
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}

fn floats_to_bytes_f64_le(floats: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(floats.len() * 8);
    for &f in floats {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Non-default encoding transforms (Delta / FixedPoint / IntervalQuantization).
//
// Each mirrors the inverse of the matching `decode_*` in
// `src/adapters/bcif/codec.rs`, so a column run through `build_*_column` and
// then the real decoder reconstructs the original values.
// ---------------------------------------------------------------------------

/// Delta-encode a value sequence against `origin`: the first emitted token is
/// `values[0] - origin`, each later token is the difference from its
/// predecessor. The decoder reverses this with a prefix sum seeded by
/// `origin`.
fn delta_encode(values: &[i32], origin: i32) -> Vec<i32> {
    let mut out = Vec::with_capacity(values.len());
    let mut prev = origin;
    for &v in values {
        out.push(v - prev);
        prev = v;
    }
    out
}

/// Integer-pack a token sequence the way the decoder unpacks it: while a
/// value exceeds the signed range for `byte_count`, emit the saturating
/// limit and carry the remainder; the residual closes the run. Mirrors the
/// limits in `int_packing_params` (signed path).
fn integer_pack(values: &[i32], byte_count: i32) -> Vec<i32> {
    let (upper, lower) = match byte_count {
        1 => (0x7F_i32, -0x7F_i32),
        2 => (0x7FFF_i32, -0x7FFF_i32),
        4 => (i32::MAX, -i32::MAX),
        other => panic!("integer_pack: unsupported byteCount={other}"),
    };
    let mut out = Vec::with_capacity(values.len());
    for &v in values {
        let mut rem = v;
        while rem >= upper {
            out.push(upper);
            rem -= upper;
        }
        while rem <= lower {
            out.push(lower);
            rem -= lower;
        }
        out.push(rem);
    }
    out
}

/// Turn packed `i32` tokens into the byte payload for `byte_count`,
/// little-endian, matching the `ByteArray` type the decoder expects
/// (signed types 1/2/3).
fn packed_ints_to_bytes(tokens: &[i32], byte_count: i32) -> Vec<u8> {
    let mut out = Vec::new();
    for &t in tokens {
        match byte_count {
            1 => out.push((t as i8).to_le_bytes()[0]),
            2 => out.extend_from_slice(&(t as i16).to_le_bytes()),
            4 => out.extend_from_slice(&t.to_le_bytes()),
            other => {
                panic!("packed_ints_to_bytes: unsupported byteCount={other}")
            }
        }
    }
    out
}

/// The `ByteArray` step matching a signed `byte_count` integer-packing
/// payload (type 1 / 2 / 3).
fn signed_byte_array_step(byte_count: i32) -> EncStep {
    match byte_count {
        1 => EncStep::ByteArrayInt8,
        2 => EncStep::ByteArrayInt16,
        4 => EncStep::ByteArrayInt32,
        other => {
            panic!("signed_byte_array_step: unsupported byteCount={other}")
        }
    }
}

/// Build a `Delta` + `IntegerPacking` column over integer `values`, the chain
/// real RCSB files use for coordinate-like columns. The encoding array is
/// emitted in the order the decoder consumes in reverse: `[Delta,
/// IntegerPacking, ByteArray]`.
pub(crate) fn delta_packed_column(
    name: &str,
    values: &[i32],
    origin: i32,
    byte_count: i32,
) -> Column {
    let deltas = delta_encode(values, origin);
    let tokens = integer_pack(&deltas, byte_count);
    let bytes = packed_ints_to_bytes(&tokens, byte_count);
    Column {
        name: name.into(),
        data: ColumnData::Raw {
            bytes,
            encoding: vec![
                EncStep::Delta {
                    origin,
                    src_type: 3,
                },
                EncStep::IntegerPacking {
                    // srcSize is the count of UNPACKED values (the decoder
                    // stops once it has produced this many), not the token
                    // count, which is larger whenever a value spans multiple
                    // saturating limits.
                    byte_count,
                    src_size: values.len() as i32,
                    is_unsigned: false,
                },
                signed_byte_array_step(byte_count),
            ],
        },
        mask: None,
    }
}

/// Build a `FixedPoint` column over float `values`: each is quantized to
/// `round(value * factor)` and stored as int32, so the decoder recovers
/// `int / factor`. Chain is `[FixedPoint, ByteArray(int32)]`.
pub(crate) fn fixed_point_column(
    name: &str,
    values: &[f64],
    factor: f64,
) -> Column {
    let ints: Vec<i32> = values
        .iter()
        .map(|&v| (v * factor).round() as i32)
        .collect();
    let bytes = ints_to_bytes_int32_le(&ints);
    Column {
        name: name.into(),
        data: ColumnData::Raw {
            bytes,
            encoding: vec![
                EncStep::FixedPoint {
                    factor,
                    src_type: 33,
                },
                EncStep::ByteArrayInt32,
            ],
        },
        mask: None,
    }
}

/// Build an `IntervalQuantization` column over float `values`: each maps to
/// the nearest bucket index in `[0, num_steps-1]` over `[min, max]`, stored
/// as int32. The decoder recovers `index * step + min`. Chain is
/// `[IntervalQuantization, ByteArray(int32)]`.
pub(crate) fn interval_quantized_column(
    name: &str,
    values: &[f64],
    min: f64,
    max: f64,
    num_steps: i32,
) -> Column {
    let step = (max - min) / f64::from(num_steps - 1);
    let ints: Vec<i32> = values
        .iter()
        .map(|&v| {
            let idx = ((v - min) / step).round();
            idx.clamp(0.0, f64::from(num_steps - 1)) as i32
        })
        .collect();
    let bytes = ints_to_bytes_int32_le(&ints);
    Column {
        name: name.into(),
        data: ColumnData::Raw {
            bytes,
            encoding: vec![
                EncStep::IntervalQuantization {
                    min,
                    max,
                    num_steps,
                    src_type: 33,
                },
                EncStep::ByteArrayInt32,
            ],
        },
        mask: None,
    }
}

// ---------------------------------------------------------------------------
// Convenience builders for the atom_site test fixtures.
// ---------------------------------------------------------------------------

/// One row's worth of atom_site fields for the test fixtures. Helpers
/// transpose a `&[AtomSite]` into BCIF column arrays.
pub(crate) struct AtomSite {
    pub label_atom_id: &'static str,
    pub label_comp_id: &'static str,
    pub label_asym_id: &'static str,
    pub label_seq_id: i32,
    pub label_entity_id: &'static str,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub occupancy: f64,
    pub b_factor: f64,
    pub type_symbol: &'static str,
    pub model: i32,
    pub label_alt_id: Option<&'static str>,
    pub auth_asym_id: Option<&'static str>,
    pub auth_seq_id: Option<i32>,
    pub auth_comp_id: Option<&'static str>,
    pub auth_atom_id: Option<&'static str>,
    pub ins_code: Option<&'static str>,
    pub formal_charge: Option<i32>,
}

pub(crate) fn entity_polymer_category() -> Category {
    Category {
        name: "_entity".into(),
        row_count: 1,
        columns: vec![
            Column {
                name: "id".into(),
                data: ColumnData::Strings(vec!["1".into()]),
                mask: None,
            },
            Column {
                name: "type".into(),
                data: ColumnData::Strings(vec!["polymer".into()]),
                mask: None,
            },
        ],
    }
}

pub(crate) fn entity_poly_protein_category() -> Category {
    Category {
        name: "_entity_poly".into(),
        row_count: 1,
        columns: vec![
            Column {
                name: "entity_id".into(),
                data: ColumnData::Strings(vec!["1".into()]),
                mask: None,
            },
            Column {
                name: "type".into(),
                data: ColumnData::Strings(vec!["polypeptide(L)".into()]),
                mask: None,
            },
        ],
    }
}

pub(crate) fn entity_poly_rna_category() -> Category {
    Category {
        name: "_entity_poly".into(),
        row_count: 1,
        columns: vec![
            Column {
                name: "entity_id".into(),
                data: ColumnData::Strings(vec!["1".into()]),
                mask: None,
            },
            Column {
                name: "type".into(),
                data: ColumnData::Strings(vec!["polyribonucleotide".into()]),
                mask: None,
            },
        ],
    }
}

pub(crate) struct AtomSiteOpts {
    pub include_auth: bool,
    pub include_ins_code: bool,
    pub include_formal_charge: bool,
    pub include_alt_id: bool,
    pub include_model_num: bool,
    pub coord_masks: Option<Vec<u8>>,
    pub occupancy_mask: Option<Vec<u8>>,
}

impl Default for AtomSiteOpts {
    fn default() -> Self {
        Self {
            include_auth: false,
            include_ins_code: false,
            include_formal_charge: false,
            include_alt_id: false,
            include_model_num: true,
            coord_masks: None,
            occupancy_mask: None,
        }
    }
}

pub(crate) fn atom_site_category(
    rows: &[AtomSite],
    opts: &AtomSiteOpts,
) -> Category {
    let row_count = rows.len();
    let pick_string = |get: &dyn Fn(&AtomSite) -> &str| -> Vec<String> {
        rows.iter().map(|r| get(r).to_owned()).collect()
    };
    let pick_int = |get: &dyn Fn(&AtomSite) -> i32| -> Vec<i32> {
        rows.iter().map(get).collect()
    };
    let pick_float = |get: &dyn Fn(&AtomSite) -> f64| -> Vec<f64> {
        rows.iter().map(get).collect()
    };

    let mut cols = Vec::new();

    cols.push(Column {
        name: "label_atom_id".into(),
        data: ColumnData::Strings(pick_string(&|r| r.label_atom_id)),
        mask: None,
    });
    cols.push(Column {
        name: "label_comp_id".into(),
        data: ColumnData::Strings(pick_string(&|r| r.label_comp_id)),
        mask: None,
    });
    cols.push(Column {
        name: "label_asym_id".into(),
        data: ColumnData::Strings(pick_string(&|r| r.label_asym_id)),
        mask: None,
    });
    cols.push(Column {
        name: "label_seq_id".into(),
        data: ColumnData::Ints(pick_int(&|r| r.label_seq_id)),
        mask: None,
    });
    cols.push(Column {
        name: "label_entity_id".into(),
        data: ColumnData::Strings(pick_string(&|r| r.label_entity_id)),
        mask: None,
    });
    cols.push(Column {
        name: "Cartn_x".into(),
        data: ColumnData::Floats(pick_float(&|r| r.x)),
        mask: opts.coord_masks.clone(),
    });
    cols.push(Column {
        name: "Cartn_y".into(),
        data: ColumnData::Floats(pick_float(&|r| r.y)),
        mask: opts.coord_masks.clone(),
    });
    cols.push(Column {
        name: "Cartn_z".into(),
        data: ColumnData::Floats(pick_float(&|r| r.z)),
        mask: opts.coord_masks.clone(),
    });
    cols.push(Column {
        name: "occupancy".into(),
        data: ColumnData::Floats(pick_float(&|r| r.occupancy)),
        mask: opts.occupancy_mask.clone(),
    });
    cols.push(Column {
        name: "B_iso_or_equiv".into(),
        data: ColumnData::Floats(pick_float(&|r| r.b_factor)),
        mask: None,
    });
    cols.push(Column {
        name: "type_symbol".into(),
        data: ColumnData::Strings(pick_string(&|r| r.type_symbol)),
        mask: None,
    });
    if opts.include_model_num {
        cols.push(Column {
            name: "pdbx_PDB_model_num".into(),
            data: ColumnData::Ints(pick_int(&|r| r.model)),
            mask: None,
        });
    }
    if opts.include_alt_id {
        let values: Vec<String> = rows
            .iter()
            .map(|r| r.label_alt_id.unwrap_or("").to_owned())
            .collect();
        let mask: Vec<u8> = rows
            .iter()
            .map(|r| u8::from(r.label_alt_id.is_none()))
            .collect();
        cols.push(Column {
            name: "label_alt_id".into(),
            data: ColumnData::Strings(values),
            mask: if mask.iter().any(|&m| m != 0) {
                Some(mask)
            } else {
                None
            },
        });
    }
    if opts.include_ins_code {
        let values: Vec<String> = rows
            .iter()
            .map(|r| r.ins_code.unwrap_or("").to_owned())
            .collect();
        let mask: Vec<u8> = rows
            .iter()
            .map(|r| u8::from(r.ins_code.is_none()))
            .collect();
        cols.push(Column {
            name: "pdbx_PDB_ins_code".into(),
            data: ColumnData::Strings(values),
            mask: if mask.iter().any(|&m| m != 0) {
                Some(mask)
            } else {
                None
            },
        });
    }
    if opts.include_formal_charge {
        let values: Vec<i32> =
            rows.iter().map(|r| r.formal_charge.unwrap_or(0)).collect();
        cols.push(Column {
            name: "pdbx_formal_charge".into(),
            data: ColumnData::Ints(values),
            mask: None,
        });
    }
    if opts.include_auth {
        cols.push(Column {
            name: "auth_asym_id".into(),
            data: ColumnData::Strings(
                rows.iter()
                    .map(|r| r.auth_asym_id.unwrap_or("").to_owned())
                    .collect(),
            ),
            mask: None,
        });
        cols.push(Column {
            name: "auth_seq_id".into(),
            data: ColumnData::Ints(
                rows.iter().map(|r| r.auth_seq_id.unwrap_or(0)).collect(),
            ),
            mask: None,
        });
        cols.push(Column {
            name: "auth_comp_id".into(),
            data: ColumnData::Strings(
                rows.iter()
                    .map(|r| r.auth_comp_id.unwrap_or("").to_owned())
                    .collect(),
            ),
            mask: None,
        });
        cols.push(Column {
            name: "auth_atom_id".into(),
            data: ColumnData::Strings(
                rows.iter()
                    .map(|r| r.auth_atom_id.unwrap_or("").to_owned())
                    .collect(),
            ),
            mask: None,
        });
    }

    Category {
        name: "_atom_site".into(),
        row_count,
        columns: cols,
    }
}
