//! Generator for `src/chemistry/completion_data.rs`.
//!
//! Reads the vendored PDB Chemical Component Dictionary files in
//! `data/ccd/*.cif`, parses each `_chem_comp_atom` loop, and emits a
//! committed Rust const table of ideal-geometry residue templates.
//!
//! Run manually from the crate root:
//!
//! ```text
//! cargo run --example gen_completion_table
//! ```
//!
//! The output is deterministic and idempotent: a second run produces a
//! byte-identical file. The normal build and test do NOT invoke this;
//! it touches the network never and the filesystem only to read the
//! vendored CIFs and write the generated module.

// This is a developer-run code generator, not shipped library code: it
// fails loud on malformed input (the vendored dictionaries are known
// good), prints the file it wrote, and intentionally narrows the f64
// source coordinates to the f32 used by the template storage.
#![allow(
    clippy::panic,
    clippy::print_stdout,
    clippy::cast_possible_truncation,
    clippy::struct_excessive_bools
)]

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

/// Component codes to vendor, in emit order: the 20 amino acids then the
/// 4 DNA bases. Order here fixes the order of the generated consts.
const COMPONENTS: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "DA",
    "DC", "DG", "DT",
];

/// One parsed atom row from a `_chem_comp_atom` loop.
struct AtomRow {
    atom_id: String,
    type_symbol: String,
    is_leaving: bool,
    is_backbone: bool,
    is_n_terminal: bool,
    is_c_terminal: bool,
    x: f64,
    y: f64,
    z: f64,
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let ccd_dir = Path::new(manifest).join("data").join("ccd");
    let out_path = Path::new(manifest)
        .join("src")
        .join("chemistry")
        .join("completion_data.rs");

    let mut body = String::new();
    body.push_str(HEADER);

    for &comp in COMPONENTS {
        let path = ccd_dir.join(format!("{comp}.cif"));
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let rows = parse_chem_comp_atom(&text)
            .unwrap_or_else(|| panic!("no _chem_comp_atom loop in {comp}"));
        assert!(!rows.is_empty(), "{comp} has zero atom rows");
        emit_component(&mut body, comp, &rows);
    }

    // A single trailing newline, matching rustfmt's expectation; the
    // per-component emit leaves a blank line between blocks.
    let body = format!("{}\n", body.trim_end());
    fs::write(&out_path, body)
        .unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));
    println!("wrote {}", out_path.display());
}

/// Parse the `_chem_comp_atom` loop. Builds the field-name to column
/// map from the header order (column order can vary between files), then
/// reads data rows until the next `loop_` or a `#` line. Returns `None`
/// if no such loop is present.
fn parse_chem_comp_atom(text: &str) -> Option<Vec<AtomRow>> {
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim() == "loop_" {
            // Collect the contiguous header lines that follow.
            let mut headers: Vec<&str> = Vec::new();
            let mut j = i + 1;
            while j < lines.len() && lines[j].trim_start().starts_with('_') {
                headers.push(lines[j].trim());
                j += 1;
            }
            if headers.iter().any(|h| h.starts_with("_chem_comp_atom.")) {
                let cols = column_map(&headers);
                return Some(read_rows(&lines, j, &cols));
            }
        }
        i += 1;
    }
    None
}

/// Indices of the fields the template needs, resolved from header order.
struct ColumnMap {
    atom_id: usize,
    type_symbol: usize,
    leaving: usize,
    backbone: usize,
    n_terminal: usize,
    c_terminal: usize,
    x_ideal: usize,
    y_ideal: usize,
    z_ideal: usize,
}

fn column_map(headers: &[&str]) -> ColumnMap {
    let find = |suffix: &str| -> usize {
        headers
            .iter()
            .position(|h| *h == format!("_chem_comp_atom.{suffix}"))
            .unwrap_or_else(|| {
                panic!("missing column _chem_comp_atom.{suffix}")
            })
    };
    ColumnMap {
        atom_id: find("atom_id"),
        type_symbol: find("type_symbol"),
        leaving: find("pdbx_leaving_atom_flag"),
        backbone: find("pdbx_backbone_atom_flag"),
        n_terminal: find("pdbx_n_terminal_atom_flag"),
        c_terminal: find("pdbx_c_terminal_atom_flag"),
        x_ideal: find("pdbx_model_Cartn_x_ideal"),
        y_ideal: find("pdbx_model_Cartn_y_ideal"),
        z_ideal: find("pdbx_model_Cartn_z_ideal"),
    }
}

/// Read data rows starting at `start`, stopping at the next `loop_` or a
/// line beginning with `#`.
fn read_rows(lines: &[&str], start: usize, cols: &ColumnMap) -> Vec<AtomRow> {
    let mut out = Vec::new();
    let mut i = start;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        if trimmed.is_empty() {
            i += 1;
            continue;
        }
        if trimmed == "loop_" || trimmed.starts_with('#') {
            break;
        }
        let toks = tokenize(line);
        let yes = |idx: usize| toks.get(idx).map(String::as_str) == Some("Y");
        out.push(AtomRow {
            atom_id: toks[cols.atom_id].clone(),
            type_symbol: toks[cols.type_symbol].clone(),
            is_leaving: yes(cols.leaving),
            is_backbone: yes(cols.backbone),
            is_n_terminal: yes(cols.n_terminal),
            is_c_terminal: yes(cols.c_terminal),
            x: parse_coord(&toks[cols.x_ideal]),
            y: parse_coord(&toks[cols.y_ideal]),
            z: parse_coord(&toks[cols.z_ideal]),
        });
        i += 1;
    }
    out
}

/// Split a CIF data line into tokens on whitespace, honoring
/// double-quoted tokens (DNA atom names contain primes and are quoted,
/// e.g. `"O5'"`, `"H5''"`). The surrounding quotes are stripped; the
/// primes are kept.
fn tokenize(line: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let mut chars = line.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            let _ = chars.next();
            continue;
        }
        if c == '"' {
            let _ = chars.next();
            let mut tok = String::new();
            for ch in chars.by_ref() {
                if ch == '"' {
                    break;
                }
                tok.push(ch);
            }
            toks.push(tok);
        } else {
            let mut tok = String::new();
            while let Some(&ch) = chars.peek() {
                if ch.is_whitespace() {
                    break;
                }
                tok.push(ch);
                let _ = chars.next();
            }
            toks.push(tok);
        }
    }
    toks
}

/// Parse an ideal coordinate. The dictionary uses `?` for an unknown
/// value; treat that as a hard error here since the standard residues
/// always carry ideal coordinates.
fn parse_coord(s: &str) -> f64 {
    s.parse::<f64>()
        .unwrap_or_else(|e| panic!("bad coordinate {s:?}: {e}"))
}

/// Append one component's `ResidueTemplate` const plus its atom slice.
fn emit_component(out: &mut String, comp: &str, rows: &[AtomRow]) {
    let atoms_const = format!("{comp}_ATOMS");
    let _ =
        writeln!(out, "/// Ideal-geometry atoms for the `{comp}` component.");
    let _ = writeln!(out, "const {atoms_const}: &[TemplateAtom] = &[");
    for row in rows {
        let element = element_variant(&row.type_symbol);
        let _ = writeln!(out, "    TemplateAtom {{");
        let _ = writeln!(
            out,
            "        name: AtomName::from_bytes(b{:?}),",
            row.atom_id
        );
        let _ = writeln!(out, "        element: Element::{element},");
        let _ = writeln!(
            out,
            "        ideal: Vec3::new({}, {}, {}),",
            fmt_f32(row.x),
            fmt_f32(row.y),
            fmt_f32(row.z),
        );
        let _ = writeln!(out, "        is_leaving: {},", row.is_leaving);
        let _ = writeln!(out, "        is_backbone: {},", row.is_backbone);
        let _ = writeln!(out, "        is_n_terminal: {},", row.is_n_terminal);
        let _ = writeln!(out, "        is_c_terminal: {},", row.is_c_terminal);
        let _ = writeln!(out, "    }},");
    }
    let _ = writeln!(out, "];");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "/// Ideal-geometry template for the `{comp}` component."
    );
    let _ = writeln!(
        out,
        "pub(crate) const {comp}: ResidueTemplate = ResidueTemplate {{"
    );
    let [c0, c1, c2] = pad3(comp);
    let _ = writeln!(out, "    comp: [{c0}, {c1}, {c2}],");
    let _ = writeln!(out, "    atoms: {atoms_const},");
    let _ = writeln!(out, "}};");
    let _ = writeln!(out);
}

/// Map a CIF `type_symbol` to an `Element` enum variant name.
fn element_variant(symbol: &str) -> &'static str {
    match symbol.to_ascii_uppercase().as_str() {
        "H" | "D" => "H",
        "C" => "C",
        "N" => "N",
        "O" => "O",
        "S" => "S",
        "P" => "P",
        "SE" => "Se",
        other => panic!("unexpected element symbol {other:?}"),
    }
}

/// Right-pad a component code to three bytes with NUL for the `[u8; 3]`
/// `comp` field. Codes longer than three characters are not expected.
fn pad3(comp: &str) -> [u8; 3] {
    let bytes = comp.as_bytes();
    assert!(bytes.len() <= 3, "component code {comp} exceeds 3 bytes");
    let mut out = [0u8; 3];
    out[..bytes.len()].copy_from_slice(bytes);
    out
}

/// Format a coordinate as a stable `f32` literal. Going through `f32`
/// keeps the emitted value consistent with the `Vec3` (f32) storage and
/// avoids dependence on the f64 source precision in the output text.
fn fmt_f32(v: f64) -> String {
    let f = v as f32;
    // `{:?}` on f32 yields the shortest round-tripping decimal, which is
    // stable across runs and always includes a decimal point.
    format!("{f:?}")
}

const HEADER: &str = "\
// foldit:allow-long-file
//! Generated ideal-geometry residue templates. DO NOT EDIT BY HAND.
//!
//! Produced by `cargo run --example gen_completion_table` from the
//! vendored PDB Chemical Component Dictionary files in `data/ccd/`.
//! Re-run the generator to regenerate; edits here will be overwritten.

use glam::Vec3;

use super::atom_name::AtomName;
use super::completion::{ResidueTemplate, TemplateAtom};
use crate::element::Element;

";
