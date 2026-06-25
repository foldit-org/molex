//! Entity -> PDB-text emission. `ATOM` for polymer entities (with a
//! closing `TER`), `HETATM` for non-polymer entities. Atom-name
//! alignment follows wwPDB v3.3 section 9.

use std::fmt::Write as _;

use super::parse::trim_ascii;
use super::refuse::validate_writable;
use crate::assembly::Assembly;
use crate::element::Element;
use crate::entity::molecule::atom::{AtomColumns, AtomRef};
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::traits::Entity;
use crate::entity::molecule::MoleculeEntity;
use crate::ops::error::AdapterError;

/// Emit an `Assembly` as a PDB-format string. All entities are flattened
/// into a single atom stream in declaration order.
///
/// # Errors
///
/// Returns [`AdapterError`] if the assembly exceeds any legacy PDB
/// limit (>99,999 atoms, >62 polymer chains, >9,999 residues per chain),
/// directing the caller to [`crate::adapters::cif::assembly_to_mmcif`].
pub fn assembly_to_pdb(assembly: &Assembly) -> Result<String, AdapterError> {
    entities_to_pdb(assembly.entities())
}

/// Emit an entity slice as a PDB-format string.
///
/// `ATOM` records are emitted for polymer entities (Protein, NucleicAcid)
/// and a `TER` record closes each polymer chain. `HETATM` records are
/// emitted for non-polymer entities (SmallMolecule, Bulk). Atom-name
/// alignment follows wwPDB v3.3 section 9: single-letter elements indent into
/// col 14, two-letter or 4-char names start at col 13.
///
/// # Errors
///
/// Returns [`AdapterError`] if the slice exceeds any legacy PDB limit
/// (see [`assembly_to_pdb`]).
pub(crate) fn entities_to_pdb<E: std::borrow::Borrow<MoleculeEntity>>(
    entities: &[E],
) -> Result<String, AdapterError> {
    validate_writable(entities)?;
    let mut pdb = String::new();
    let mut serial: usize = 0;
    for entity in entities {
        write_entity_atoms(entity.borrow(), &mut serial, &mut pdb);
    }
    pdb.push_str("END\n");
    Ok(pdb)
}

fn write_entity_atoms(
    entity: &MoleculeEntity,
    serial: &mut usize,
    out: &mut String,
) {
    match entity {
        MoleculeEntity::Protein(e) => {
            let chain = chain_id_byte(&e.pdb_chain_id);
            write_polymer_atoms(e.columns(), &e.residues, chain, serial, out);
            write_polymer_terminator(&e.residues, chain, serial, out);
        }
        MoleculeEntity::NucleicAcid(e) => {
            let chain = chain_id_byte(&e.pdb_chain_id);
            write_polymer_atoms(e.columns(), &e.residues, chain, serial, out);
            write_polymer_terminator(&e.residues, chain, serial, out);
        }
        MoleculeEntity::SmallMolecule(e) => {
            for atom in e.atoms_iter() {
                *serial += 1;
                write_atom_line(
                    "HETATM",
                    *serial,
                    atom,
                    ResidueCtx {
                        chain_id: b' ',
                        res_name: e.residue_name,
                        res_num: 1,
                    },
                    out,
                );
            }
        }
        MoleculeEntity::Bulk(e) =>
        {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                reason = "atom count fits in i32 for valid structures"
            )]
            for (i, atom) in e.atoms_iter().enumerate() {
                *serial += 1;
                write_atom_line(
                    "HETATM",
                    *serial,
                    atom,
                    ResidueCtx {
                        chain_id: b' ',
                        res_name: e.residue_name,
                        res_num: (i as i32) + 1,
                    },
                    out,
                );
            }
        }
    }
}

/// Single-byte chain id for the PDB col-22 field. `validate_writable`
/// already refused any multi-character chain, so this only ever runs on a
/// one-char (or empty) id; a non-single-byte id falls back to a blank.
fn chain_id_byte(chain_id: &str) -> u8 {
    let bytes = chain_id.as_bytes();
    if bytes.len() == 1 {
        bytes[0]
    } else {
        b' '
    }
}

fn write_polymer_atoms(
    columns: &AtomColumns,
    residues: &[Residue],
    chain_id: u8,
    serial: &mut usize,
    out: &mut String,
) {
    for residue in residues {
        for idx in residue.atom_range.clone() {
            *serial += 1;
            write_atom_line(
                "ATOM  ",
                *serial,
                columns.atom_ref(idx),
                ResidueCtx {
                    chain_id,
                    res_name: residue.name,
                    res_num: residue.label_seq_id,
                },
                out,
            );
        }
    }
}

fn write_polymer_terminator(
    residues: &[Residue],
    chain_id: u8,
    serial: &mut usize,
    out: &mut String,
) {
    let Some(last) = residues.last() else {
        return;
    };
    *serial += 1;
    let res_name_str = std::str::from_utf8(&last.name).unwrap_or("UNK");
    let _ = writeln!(
        out,
        "TER   {:>5}      {:>3} {}{:>4}",
        serial, res_name_str, chain_id as char, last.label_seq_id,
    );
}

#[derive(Copy, Clone)]
struct ResidueCtx {
    chain_id: u8,
    res_name: [u8; 3],
    res_num: i32,
}

/// Format an atom name for the cols 13-16 field per wwPDB v3.3 section 9.
///
/// Single-letter elements with <=3-char names indent so the name starts
/// at col 14 (e.g. `" CA "`). Two-letter elements and 4-char names
/// occupy all four cols left-justified (e.g. `"FE  "`, `"1HD1"`).
pub(super) fn format_atom_name(name: [u8; 4], element: Element) -> [u8; 4] {
    let trimmed = trim_ascii(&name);
    let mut out = [b' '; 4];
    let two_letter_element = element.symbol().len() == 2;
    if two_letter_element || trimmed.len() >= 4 {
        for (i, &b) in trimmed.iter().take(4).enumerate() {
            out[i] = b;
        }
    } else {
        for (i, &b) in trimmed.iter().take(3).enumerate() {
            out[i + 1] = b;
        }
    }
    out
}

fn write_atom_line(
    record_kind: &str,
    serial: usize,
    atom: AtomRef<'_>,
    ctx: ResidueCtx,
    out: &mut String,
) {
    let name_bytes = format_atom_name(*atom.name, *atom.element);
    let name_str = std::str::from_utf8(&name_bytes).unwrap_or("X   ");
    let res_name_str = std::str::from_utf8(&ctx.res_name).unwrap_or("UNK");
    let _ = writeln!(
        out,
        "{}{:>5} {} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}",
        record_kind,
        serial,
        name_str,
        res_name_str,
        ctx.chain_id as char,
        ctx.res_num,
        atom.position.x,
        atom.position.y,
        atom.position.z,
        atom.occupancy,
        atom.b_factor,
    );
}
