//! Entity -> mmCIF `_atom_site` emission. The inverse of the cif reader:
//! the emitted column set is exactly the one `scan_row::AtomSiteCols`
//! consumes, so output round-trips faithfully through `mmcif_str_to_entities`.
//!
//! Polymer atoms are `ATOM` rows carrying the residue's `label_seq_id`;
//! non-polymer atoms (ligands, ions, waters) are `HETATM` rows with
//! `label_seq_id = .` and a per-instance `auth_seq_id`, mirroring how RCSB
//! mmCIF discriminates waters (the reader keys on the author number when
//! `label_seq_id` is absent).

use std::collections::HashSet;
use std::fmt::Write as _;

use crate::assembly::Assembly;
use crate::entity::molecule::atom::{AtomColumns, AtomRef};
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::traits::Entity;
use crate::entity::molecule::MoleculeEntity;

/// Emit an `Assembly` as an mmCIF string with a single `_atom_site` loop.
///
/// Every entity is flattened into one model (`pdbx_PDB_model_num = 1`) in
/// declaration order. Polymer chains keep their `label_asym_id`; each
/// non-polymer entity is assigned a fresh unique `label_asym_id` so waters
/// and ligands re-parse as distinct chains rather than merging.
#[must_use]
pub(crate) fn assembly_to_mmcif(assembly: &Assembly) -> String {
    entities_to_mmcif(assembly.entities())
}

/// Emit an entity slice as an mmCIF string. See [`assembly_to_mmcif`].
#[must_use]
pub(crate) fn entities_to_mmcif<E: std::borrow::Borrow<MoleculeEntity>>(
    entities: &[E],
) -> String {
    let mut chains = ChainLabeller::new(entities);
    let mut out = String::new();
    out.push_str("data_molex\n#\n");
    write_atom_site_header(&mut out);

    let mut serial: usize = 0;
    for entity in entities {
        write_entity(entity.borrow(), &mut chains, &mut serial, &mut out);
    }
    out.push_str("#\n");
    out
}

/// The `_atom_site` column order, emitted once before the rows. Mirrors the
/// fields `scan_row::AtomSiteCols` resolves so the loop round-trips.
const COLUMNS: &[&str] = &[
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_seq_id",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
    "pdbx_formal_charge",
    "pdbx_PDB_ins_code",
    "auth_seq_id",
    "auth_asym_id",
    "pdbx_PDB_model_num",
];

fn write_atom_site_header(out: &mut String) {
    out.push_str("loop_\n");
    for col in COLUMNS {
        let _ = writeln!(out, "_atom_site.{col}");
    }
}

/// Per-entity `label_asym_id` resolution. Polymer entities keep their real
/// chain id; non-polymer entities (whose `pdb_chain_id()` is `None`) get a
/// fresh single/double-letter label that does not collide with any polymer
/// chain or another synthesized one.
struct ChainLabeller {
    used: HashSet<String>,
    next: u32,
}

impl ChainLabeller {
    fn new<E: std::borrow::Borrow<MoleculeEntity>>(entities: &[E]) -> Self {
        let used = entities
            .iter()
            .filter_map(|e| e.borrow().pdb_chain_id())
            .map(str::to_owned)
            .collect();
        Self { used, next: 0 }
    }

    /// A label for a non-polymer entity: the next base-26 letter string
    /// (`A`, `B`, ... `Z`, `AA`, ...) not already claimed by a polymer chain.
    fn synth_label(&mut self) -> String {
        loop {
            let label = base26(self.next);
            self.next += 1;
            if self.used.insert(label.clone()) {
                return label;
            }
        }
    }
}

/// Encode `n` as an uppercase base-26 string with digits `A..Z` (spreadsheet
/// column style): 0 -> "A", 25 -> "Z", 26 -> "AA".
fn base26(mut n: u32) -> String {
    let mut buf = Vec::new();
    loop {
        buf.push(b'A' + u8::try_from(n % 26).unwrap_or(0));
        n /= 26;
        if n == 0 {
            break;
        }
        n -= 1;
    }
    buf.reverse();
    String::from_utf8(buf).unwrap_or_else(|_| "A".to_owned())
}

fn write_entity(
    entity: &MoleculeEntity,
    chains: &mut ChainLabeller,
    serial: &mut usize,
    out: &mut String,
) {
    match entity {
        MoleculeEntity::Protein(e) => {
            write_polymer(
                e.columns(),
                &e.residues,
                &e.pdb_chain_id,
                serial,
                out,
            );
        }
        MoleculeEntity::NucleicAcid(e) => {
            write_polymer(
                e.columns(),
                &e.residues,
                &e.pdb_chain_id,
                serial,
                out,
            );
        }
        MoleculeEntity::SmallMolecule(e) => {
            // One molecule => one residue: all atoms share seq id 1.
            let chain = chains.synth_label();
            let ctx = NonPolymerCtx {
                res_name: e.residue_name,
                chain: &chain,
                seq: 1,
            };
            for atom in e.atoms_iter() {
                write_nonpolymer_row(atom, &ctx, serial, out);
            }
        }
        MoleculeEntity::Bulk(e) => {
            // Many identical molecules (water/solvent): one residue per atom
            // so each survives the reader's distinct-residue keying.
            let chain = chains.synth_label();
            for (i, atom) in e.atoms_iter().enumerate() {
                let ctx = NonPolymerCtx {
                    res_name: e.residue_name,
                    chain: &chain,
                    seq: i32::try_from(i + 1).unwrap_or(i32::MAX),
                };
                write_nonpolymer_row(atom, &ctx, serial, out);
            }
        }
    }
}

fn write_polymer(
    columns: &AtomColumns,
    residues: &[Residue],
    chain: &str,
    serial: &mut usize,
    out: &mut String,
) {
    for residue in residues {
        let res_name = bytes3_to_str(&residue.name);
        let ins = residue.ins_code.map_or_else(
            || ".".to_owned(),
            |c| cif_quote(&(c as char).to_string()),
        );
        let auth_seq = residue.seq_id();
        for idx in residue.atom_range.clone() {
            *serial += 1;
            write_row(
                out,
                &RowFields {
                    group: "ATOM",
                    serial: *serial,
                    atom: columns.atom_ref(idx),
                    res_name,
                    chain,
                    label_seq: &residue.label_seq_id.to_string(),
                    ins_code: &ins,
                    auth_seq,
                },
            );
        }
    }
}

struct NonPolymerCtx<'a> {
    res_name: [u8; 3],
    chain: &'a str,
    seq: i32,
}

fn write_nonpolymer_row(
    atom: AtomRef<'_>,
    ctx: &NonPolymerCtx<'_>,
    serial: &mut usize,
    out: &mut String,
) {
    *serial += 1;
    write_row(
        out,
        &RowFields {
            group: "HETATM",
            serial: *serial,
            atom,
            res_name: bytes3_to_str(&ctx.res_name),
            chain: ctx.chain,
            // Non-polymer rows carry no structural seq id; the reader
            // falls back to auth_seq_id for the distinct-residue key.
            label_seq: ".",
            ins_code: ".",
            auth_seq: ctx.seq,
        },
    );
}

struct RowFields<'a> {
    group: &'a str,
    serial: usize,
    atom: AtomRef<'a>,
    res_name: &'a str,
    chain: &'a str,
    label_seq: &'a str,
    ins_code: &'a str,
    auth_seq: i32,
}

fn write_row(out: &mut String, f: &RowFields<'_>) {
    let atom = f.atom;
    let atom_name = std::str::from_utf8(atom.name).unwrap_or("X").trim();
    let charge = if *atom.formal_charge == 0 {
        "?".to_owned()
    } else {
        atom.formal_charge.to_string()
    };
    // molex deduplicates alternate locations at parse time and keeps no
    // alt-loc on `Atom`; `.` (no alt id) is emitted for every row. gemmi's
    // structure builder requires this column to be present.
    let _ = writeln!(
        out,
        "{group} {serial} {sym} {atom} . {comp} {asym} {lseq} {x:.3} {y:.3} \
         {z:.3} {occ:.2} {b:.2} {charge} {ins} {aseq} {auth} 1",
        group = f.group,
        serial = f.serial,
        sym = atom.element.symbol(),
        atom = cif_quote(atom_name),
        comp = cif_quote(f.res_name),
        asym = cif_quote(f.chain),
        lseq = f.label_seq,
        x = atom.position.x,
        y = atom.position.y,
        z = atom.position.z,
        occ = atom.occupancy,
        b = atom.b_factor,
        charge = charge,
        ins = f.ins_code,
        aseq = f.auth_seq,
        auth = cif_quote(f.chain),
    );
}

fn bytes3_to_str(b: &[u8; 3]) -> &str {
    std::str::from_utf8(b).unwrap_or("UNK").trim()
}

/// Quote a CIF data value when it cannot stand bare: empty -> `.`, embedded
/// whitespace / leading quote -> single-quoted. A value with no special
/// character is returned as-is.
fn cif_quote(s: &str) -> String {
    if s.is_empty() {
        return ".".to_owned();
    }
    let needs_quote = s.chars().any(char::is_whitespace)
        || s.starts_with('\'')
        || s.starts_with('"')
        || s.starts_with('_')
        || s == "."
        || s == "?";
    if needs_quote {
        format!("'{s}'")
    } else {
        s.to_owned()
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    reason = "test code"
)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::adapters::cif::mmcif_str_to_entities;
    use crate::assembly::Assembly;
    use crate::element::Element;
    use crate::entity::molecule::atom::Atom;
    use crate::entity::molecule::bulk::BulkEntity;
    use crate::entity::molecule::id::EntityIdAllocator;
    use crate::entity::molecule::protein::ProteinEntity;
    use crate::entity::molecule::MoleculeType;

    fn mk_atom(name: [u8; 4], el: Element, pos: Vec3) -> Atom {
        Atom {
            position: pos,
            occupancy: 1.0,
            b_factor: 0.0,
            element: el,
            name,
            formal_charge: 0,
            observed: true,
        }
    }

    fn dipeptide(chain: &str) -> MoleculeEntity {
        let atoms = vec![
            mk_atom(*b"N   ", Element::N, Vec3::new(0.0, 0.0, 0.0)),
            mk_atom(*b"CA  ", Element::C, Vec3::new(1.0, 0.0, 0.0)),
            mk_atom(*b"C   ", Element::C, Vec3::new(2.0, 0.0, 0.0)),
            mk_atom(*b"O   ", Element::O, Vec3::new(2.0, 1.0, 0.0)),
            mk_atom(*b"CB  ", Element::C, Vec3::new(1.0, -1.0, 0.0)),
            mk_atom(*b"N   ", Element::N, Vec3::new(3.2, 0.0, 0.0)),
            mk_atom(*b"CA  ", Element::C, Vec3::new(4.2, 0.0, 0.0)),
            mk_atom(*b"C   ", Element::C, Vec3::new(5.2, 0.0, 0.0)),
            mk_atom(*b"O   ", Element::O, Vec3::new(5.2, 1.0, 0.0)),
        ];
        let residues = vec![
            Residue {
                name: *b"ALA",
                label_seq_id: 1,
                auth_seq_id: None,
                auth_comp_id: None,
                ins_code: None,
                atom_range: 0..5,
                variants: Vec::new(),
            },
            Residue {
                name: *b"GLY",
                label_seq_id: 2,
                auth_seq_id: None,
                auth_comp_id: None,
                ins_code: None,
                atom_range: 5..9,
                variants: Vec::new(),
            },
        ];
        let id = EntityIdAllocator::new().allocate();
        MoleculeEntity::Protein(ProteinEntity::new(
            id,
            atoms,
            residues,
            chain.to_owned(),
        ))
    }

    fn waters(n: usize) -> MoleculeEntity {
        let atoms: Vec<Atom> = (0..n)
            .map(|i| {
                mk_atom(
                    *b"O   ",
                    Element::O,
                    Vec3::new(10.0 + i as f32, 0.0, 0.0),
                )
            })
            .collect();
        let id = EntityIdAllocator::new().allocate();
        MoleculeEntity::Bulk(BulkEntity::new(
            id,
            MoleculeType::Water,
            atoms,
            *b"HOH",
            n,
        ))
    }

    fn atom_count(entities: &[MoleculeEntity]) -> usize {
        entities.iter().map(MoleculeEntity::atom_count).sum()
    }

    #[test]
    fn protein_and_waters_round_trip() {
        // `from_mmcif` completes missing heavy atoms at parse time, so the
        // round-trip invariant is write -> parse -> write -> parse being
        // stable, not raw-input atom count equality. Parse once to absorb
        // completion, then assert the second cycle preserves everything.
        let entities = vec![dipeptide("A"), waters(4)];
        let p1 =
            mmcif_str_to_entities(&Assembly::new(entities).to_mmcif()).unwrap();
        let p2 = mmcif_str_to_entities(&Assembly::new(p1.clone()).to_mmcif())
            .unwrap();

        assert_eq!(atom_count(&p2), atom_count(&p1));

        // The two polymer residues come back as two residues, and the four
        // waters do not collapse onto one O.
        let protein = p2
            .iter()
            .find(|e| e.molecule_type() == MoleculeType::Protein)
            .expect("protein survives");
        assert_eq!(protein.residues().unwrap().len(), 2);

        let water_atoms: usize = p2
            .iter()
            .filter(|e| e.molecule_type() == MoleculeType::Water)
            .map(MoleculeEntity::atom_count)
            .sum();
        assert_eq!(water_atoms, 4, "four waters must survive distinctly");
    }

    #[test]
    fn multi_char_chain_survives() {
        let entities = vec![dipeptide("AA")];
        let cif = Assembly::new(entities).to_mmcif();
        let parsed = mmcif_str_to_entities(&cif).unwrap();

        let chains: Vec<&str> =
            parsed.iter().filter_map(|e| e.pdb_chain_id()).collect();
        assert!(
            chains.contains(&"AA"),
            "multi-char chain 'AA' must round-trip, got {chains:?}"
        );
    }
}
