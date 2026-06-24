//! Column-positional ATOM/HETATM scanner and MODEL/ENDMDL state machine.

use crate::element::Element;
use crate::entity::molecule::{AtomRow, EntityBuilder, MoleculeEntity};
use crate::ops::codec::AdapterError;

/// Record-kind dispatch based on cols 1-6.
#[derive(Clone, Copy)]
enum RecordKind {
    Atom,
    Hetatm,
    Model,
    EndMdl,
    Ter,
    End,
    Skip,
}

/// Classify a line by its first six columns. Short lines and unknown
/// record names map to [`RecordKind::Skip`].
fn record_kind(line: &[u8]) -> RecordKind {
    let mut prefix = [b' '; 6];
    let n = line.len().min(6);
    prefix[..n].copy_from_slice(&line[..n]);
    match &prefix {
        b"ATOM  " => RecordKind::Atom,
        b"HETATM" => RecordKind::Hetatm,
        b"MODEL " => RecordKind::Model,
        b"ENDMDL" => RecordKind::EndMdl,
        b"TER   " => RecordKind::Ter,
        b"END   " => RecordKind::End,
        _ => RecordKind::Skip,
    }
}

/// Drive the column scanner across the input.
///
/// Single-model API: emits the entities for `MODEL 1`. Files without
/// `MODEL`/`ENDMDL` are treated as one implicit model.
pub(super) fn parse_pdb_to_entities(
    input: &str,
) -> Result<Vec<MoleculeEntity>, AdapterError> {
    let mut builder = EntityBuilder::new();
    let mut current_model: i32 = 1;
    let target_model: i32 = 1;
    let mut any_atom = false;

    for line in input.lines() {
        let bytes = line.as_bytes();
        match record_kind(bytes) {
            RecordKind::Atom | RecordKind::Hetatm => {
                if current_model != target_model {
                    continue;
                }
                let row = parse_atom_record(bytes)?;
                builder
                    .push_atom(row)
                    .map_err(|e| AdapterError::PdbParseError(e.to_string()))?;
                any_atom = true;
            }
            RecordKind::Model => {
                current_model = parse_model_serial(bytes);
            }
            RecordKind::EndMdl => {
                if current_model == target_model {
                    break;
                }
            }
            RecordKind::Ter | RecordKind::Skip => {}
            RecordKind::End => break,
        }
    }

    if !any_atom {
        return Err(AdapterError::PdbParseError(
            "No atoms found in structure".to_owned(),
        ));
    }
    builder
        .finish()
        .map_err(|e| AdapterError::PdbParseError(e.to_string()))
}

/// All-models variant: emit one `Vec<MoleculeEntity>` per `MODEL` block.
///
/// Files without any `MODEL` records flush to a single entry. Atoms
/// before the first `MODEL` are treated as belonging to an implicit
/// model and flushed when the first `MODEL` arrives.
pub(super) fn parse_pdb_to_all_models(
    input: &str,
) -> Result<Vec<Vec<MoleculeEntity>>, AdapterError> {
    let mut models: Vec<Vec<MoleculeEntity>> = Vec::new();
    let mut builder = EntityBuilder::new();
    let mut any_atom_in_model = false;
    let mut any_atom_total = false;

    for line in input.lines() {
        let bytes = line.as_bytes();
        match record_kind(bytes) {
            RecordKind::Atom | RecordKind::Hetatm => {
                let row = parse_atom_record(bytes)?;
                builder
                    .push_atom(row)
                    .map_err(|e| AdapterError::PdbParseError(e.to_string()))?;
                any_atom_in_model = true;
                any_atom_total = true;
            }
            RecordKind::Model => {
                // Flush any pre-MODEL atoms as their own implicit model.
                if any_atom_in_model {
                    let entities = builder.finish().map_err(|e| {
                        AdapterError::PdbParseError(e.to_string())
                    })?;
                    models.push(entities);
                    builder = EntityBuilder::new();
                    any_atom_in_model = false;
                }
            }
            RecordKind::EndMdl => {
                if any_atom_in_model {
                    let entities = builder.finish().map_err(|e| {
                        AdapterError::PdbParseError(e.to_string())
                    })?;
                    models.push(entities);
                    builder = EntityBuilder::new();
                    any_atom_in_model = false;
                }
            }
            RecordKind::End => break,
            RecordKind::Ter | RecordKind::Skip => {}
        }
    }

    if any_atom_in_model {
        let entities = builder
            .finish()
            .map_err(|e| AdapterError::PdbParseError(e.to_string()))?;
        models.push(entities);
    }

    if !any_atom_total {
        return Err(AdapterError::PdbParseError(
            "No atoms found in structure".to_owned(),
        ));
    }
    Ok(models)
}

/// Parse a `MODEL` record's serial-number field (cols 11-14).
fn parse_model_serial(bytes: &[u8]) -> i32 {
    let lo = 10.min(bytes.len());
    let hi = 14.min(bytes.len());
    std::str::from_utf8(&bytes[lo..hi])
        .ok()
        .map(str::trim)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

/// Slice cols `lo..hi` (1-indexed, inclusive on lo, exclusive on hi-as-len)
/// from the line; returns an empty slice if the line is too short.
fn col_slice(bytes: &[u8], lo: usize, hi: usize) -> &[u8] {
    let start = (lo - 1).min(bytes.len());
    let end = hi.min(bytes.len());
    if start > end {
        &[]
    } else {
        &bytes[start..end]
    }
}

pub(super) fn trim_ascii(s: &[u8]) -> &[u8] {
    let start = s
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(s.len());
    let end = s[start..]
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map_or(start, |i| start + i + 1);
    &s[start..end]
}

/// View an all-ASCII byte slice as `&str` without re-validating UTF-8.
/// Falls back to checked decoding if a non-ASCII byte is present (a column
/// straddling a multibyte char in a malformed file).
fn ascii_str(s: &[u8]) -> &str {
    if s.is_ascii() {
        // SAFETY: every byte is ASCII (< 0x80), so the slice is valid UTF-8
        // and sits on char boundaries.
        unsafe { std::str::from_utf8_unchecked(s) }
    } else {
        std::str::from_utf8(s).unwrap_or("")
    }
}

/// Parse a fixed-width PDB decimal float field directly off bytes.
///
/// Accepts an optional leading `+`/`-`, decimal digits, and a single `.`;
/// returns `default` for an empty or malformed field. The significand is
/// accumulated as `i64` and scaled by an exact power of ten in `f64`
/// before the final `f32` cast, which is bit-identical to
/// `str::parse::<f32>()` across the F8.3 / F6.2 value ranges PDB columns
/// hold (exponent/`inf`/`nan` forms never appear in these fixed columns).
fn parse_f32_bytes(field: &[u8], default: f32) -> f32 {
    let s = trim_ascii(field);
    if s.is_empty() {
        return default;
    }
    let mut idx = 0;
    let neg = match s[0] {
        b'-' => {
            idx = 1;
            true
        }
        b'+' => {
            idx = 1;
            false
        }
        _ => false,
    };
    let mut mant: i64 = 0;
    let mut frac_digits: i32 = 0;
    let mut seen_dot = false;
    let mut any_digit = false;
    while idx < s.len() {
        let b = s[idx];
        if b == b'.' {
            if seen_dot {
                return default;
            }
            seen_dot = true;
        } else if b.is_ascii_digit() {
            any_digit = true;
            mant = mant * 10 + i64::from(b - b'0');
            if seen_dot {
                frac_digits += 1;
            }
        } else {
            return default;
        }
        idx += 1;
    }
    if !any_digit {
        return default;
    }
    if neg {
        mant = -mant;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "mant has <=8 significant digits in F8.3/F6.2 fields; the f64 \
                  product is exact and the f32 cast matches str::parse"
    )]
    let value = (mant as f64) * pow10_neg(frac_digits);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "narrowing the correctly-rounded f64 to f32 mirrors \
                  str::parse::<f32>()"
    )]
    let out = value as f32;
    out
}

/// Exact `10^-n` for the small `n` that fixed PDB float fields produce.
fn pow10_neg(n: i32) -> f64 {
    const POS: [f64; 8] = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7];
    let m = n.unsigned_abs() as usize;
    if m < POS.len() {
        1.0 / POS[m]
    } else {
        10f64.powi(-n)
    }
}

/// Parse a fixed-width PDB decimal integer field directly off bytes.
/// Returns `None` for an empty or non-decimal field.
fn parse_i32_bytes(field: &[u8]) -> Option<i32> {
    let s = trim_ascii(field);
    if s.is_empty() {
        return None;
    }
    let mut idx = 0;
    let neg = match s[0] {
        b'-' => {
            idx = 1;
            true
        }
        b'+' => {
            idx = 1;
            false
        }
        _ => false,
    };
    if idx >= s.len() {
        return None;
    }
    let mut v: i64 = 0;
    while idx < s.len() {
        let b = s[idx];
        if !b.is_ascii_digit() {
            return None;
        }
        v = v * 10 + i64::from(b - b'0');
        if v > i64::from(i32::MAX) + 1 {
            return None;
        }
        idx += 1;
    }
    if neg {
        v = -v;
    }
    i32::try_from(v).ok()
}

/// Left-justified, space-padded copy of `raw` into an `N`-byte buffer.
/// Keeps `Atom.name` storage stable across files that differ on col-13
/// vs col-14 alignment.
pub(super) fn normalize_name<const N: usize>(raw: &[u8]) -> [u8; N] {
    let trimmed = trim_ascii(raw);
    let mut out = [b' '; N];
    for (i, &b) in trimmed.iter().take(N).enumerate() {
        out[i] = b;
    }
    out
}

/// Decode a width-`W` PDB hybrid-36 numeric field.
///
/// Width-strict: the raw slice must be exactly `width` bytes, all
/// alphanumeric ASCII, with at least one alphabetic character. Returns
/// `None` for any other input; callers fall back to decimal parsing.
///
/// Per the Grosse-Kunstleve hybrid-36 spec: an uppercase first char
/// encodes the upper-base-36 range starting at `10^width`; a lowercase
/// first char encodes the lower range starting after the upper range
/// exhausts.
pub(super) fn decode_hybrid36(raw: &[u8], width: usize) -> Option<i32> {
    if raw.len() != width {
        return None;
    }
    let first = *raw.first()?;
    if first.is_ascii_uppercase() {
        let mut v: i64 = 0;
        for &b in raw {
            let d = if b.is_ascii_digit() {
                i64::from(b - b'0')
            } else if b.is_ascii_uppercase() {
                i64::from(b - b'A') + 10
            } else {
                return None;
            };
            v = v * 36 + d;
        }
        let offset = 10_i64 * pow36(width - 1) - pow10(width);
        i32::try_from(v - offset).ok()
    } else if first.is_ascii_lowercase() {
        let mut v: i64 = 0;
        for &b in raw {
            let d = if b.is_ascii_digit() {
                i64::from(b - b'0')
            } else if b.is_ascii_lowercase() {
                i64::from(b - b'a') + 10
            } else {
                return None;
            };
            v = v * 36 + d;
        }
        let offset = -(16_i64 * pow36(width - 1) + pow10(width));
        i32::try_from(v - offset).ok()
    } else {
        None
    }
}

/// Hybrid-36 decode of the 5-byte atom serial field (cols 7-11).
#[allow(
    dead_code,
    reason = "named wrapper kept for symmetry with the resseq decoder"
)]
pub(super) fn decode_hybrid36_serial(raw: &[u8]) -> Option<i32> {
    decode_hybrid36(raw, 5)
}

/// Hybrid-36 decode of the 4-byte residue sequence field (cols 23-26).
#[allow(
    dead_code,
    reason = "named wrapper used by tests; in-tree resseq parsing flows \
              through parse_seq_field"
)]
pub(super) fn decode_hybrid36_resseq(raw: &[u8]) -> Option<i32> {
    decode_hybrid36(raw, 4)
}

fn pow10(n: usize) -> i64 {
    10_i64.pow(u32::try_from(n).unwrap_or(0))
}

fn pow36(n: usize) -> i64 {
    36_i64.pow(u32::try_from(n).unwrap_or(0))
}

/// Decode a fixed-width PDB numeric field that may be encoded as either
/// decimal or hybrid-36. Hybrid-36 detection is per-column independent
/// (matches mixed-encoding files produced by Phenix / MODELLER / OpenMM).
pub(super) fn parse_seq_field(raw: &[u8], width: usize) -> Option<i32> {
    if raw.len() == width
        && raw.iter().all(u8::is_ascii_alphanumeric)
        && raw.iter().any(u8::is_ascii_alphabetic)
    {
        return decode_hybrid36(raw, width);
    }
    parse_i32_bytes(raw)
}

/// Parse a PDB charge field (`"2+"`, `"1-"`, `""`).
pub(super) fn parse_pdb_charge(raw: &[u8]) -> i8 {
    let b = trim_ascii(raw);
    if b.is_empty() {
        return 0;
    }
    if b.len() == 2 && b[0].is_ascii_digit() {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "single-digit charge fits in i8"
        )]
        let signed = i8::try_from(b[0] - b'0').unwrap_or(0);
        return match b[1] {
            b'+' => signed,
            b'-' => -signed,
            _ => 0,
        };
    }
    parse_i32_bytes(b)
        .and_then(|v| i8::try_from(v).ok())
        .unwrap_or(0)
}

/// Decode element from cols 77-78, falling back to atom-name inference.
fn resolve_element(bytes: &[u8], atom_name_str: &str) -> Element {
    let raw = col_slice(bytes, 77, 78);
    let trimmed = trim_ascii(raw);
    if trimmed.is_empty() {
        return Element::from_atom_name(atom_name_str);
    }
    let parsed = Element::from_symbol(ascii_str(trimmed));
    if matches!(parsed, Element::Unknown) {
        Element::from_atom_name(atom_name_str)
    } else {
        parsed
    }
}

/// Parse one `ATOM`/`HETATM` row.
fn parse_atom_record(bytes: &[u8]) -> Result<AtomRow, AdapterError> {
    // Require enough length to cover x/y/z (cols 31-54).
    if bytes.len() < 54 {
        return Err(AdapterError::PdbParseError(format!(
            "ATOM/HETATM row too short ({} bytes); need >=54 for coordinates",
            bytes.len()
        )));
    }

    let label_atom_id = normalize_name::<4>(col_slice(bytes, 13, 16));
    let alt_loc = col_slice(bytes, 17, 17)
        .first()
        .copied()
        .filter(|&b| b != b' ');
    let label_comp_id = normalize_name::<3>(col_slice(bytes, 18, 20));
    let chain_byte = col_slice(bytes, 22, 22).first().copied().unwrap_or(b' ');
    let label_seq_id = parse_seq_field(col_slice(bytes, 23, 26), 4)
        .ok_or_else(|| {
            AdapterError::PdbParseError(
                "ATOM/HETATM row missing residue sequence number".to_owned(),
            )
        })?;
    let ins_code = col_slice(bytes, 27, 27)
        .first()
        .copied()
        .filter(|&b| b != b' ');
    let x = parse_f32_bytes(col_slice(bytes, 31, 38), f32::NAN);
    let y = parse_f32_bytes(col_slice(bytes, 39, 46), f32::NAN);
    let z = parse_f32_bytes(col_slice(bytes, 47, 54), f32::NAN);
    let occupancy = parse_f32_bytes(col_slice(bytes, 55, 60), 1.0);
    let b_factor = parse_f32_bytes(col_slice(bytes, 61, 66), 0.0);
    let formal_charge = parse_pdb_charge(col_slice(bytes, 79, 80));

    let atom_name_str = ascii_str(trim_ascii(&label_atom_id));
    let element = resolve_element(bytes, atom_name_str);

    // Chain ID: single byte per PDB spec. Stored as `String` to keep the
    // `AtomRow` shape uniform with mmCIF (multi-char chain IDs).
    let chain_str = if chain_byte.is_ascii() {
        (chain_byte as char).to_string()
    } else {
        " ".to_owned()
    };

    Ok(AtomRow {
        label_asym_id: chain_str,
        label_seq_id,
        label_comp_id,
        label_atom_id,
        label_entity_id: None,
        auth_asym_id: None,
        auth_seq_id: None,
        auth_comp_id: None,
        auth_atom_id: None,
        alt_loc,
        ins_code,
        element,
        x,
        y,
        z,
        occupancy,
        b_factor,
        formal_charge,
    })
}
