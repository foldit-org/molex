//! Classification dispatch: picks an entity kind for each chain and
//! emits one or more `MoleculeEntity` values into the `ChainCtx`.

use super::{
    ChainCtx, ExpectedEntityType, ResidueAccum, DEFAULT_WATER_RESNAME,
};
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::bulk::BulkEntity;
use crate::entity::molecule::classify::classify_residue;
use crate::entity::molecule::complete::CompletionMode;
use crate::entity::molecule::nucleic_acid::NAEntity;
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::protein::ProteinEntity;
use crate::entity::molecule::small_molecule::SmallMoleculeEntity;
use crate::entity::molecule::{MoleculeEntity, MoleculeType};

pub(super) fn classify_chain(
    hint: ExpectedEntityType,
    chain_id: &str,
    residues: &[ResidueAccum],
    ctx: &mut ChainCtx,
) {
    match hint {
        ExpectedEntityType::Protein => {
            emit_polymer_chain(residues, MoleculeType::Protein, chain_id, ctx);
        }
        ExpectedEntityType::DNA => {
            emit_polymer_chain(residues, MoleculeType::DNA, chain_id, ctx);
        }
        ExpectedEntityType::RNA => {
            emit_polymer_chain(residues, MoleculeType::RNA, chain_id, ctx);
        }
        ExpectedEntityType::Water => emit_chain_bulk(
            residues,
            MoleculeType::Water,
            DEFAULT_WATER_RESNAME,
            ctx,
        ),
        ExpectedEntityType::NonPolymer => {
            emit_non_polymer_chain(residues, ctx);
        }
        ExpectedEntityType::Unknown => {
            emit_unknown_chain(chain_id, residues, ctx);
        }
    }
}

fn emit_polymer_chain(
    residues: &[ResidueAccum],
    mol_type: MoleculeType,
    chain_id: &str,
    ctx: &mut ChainCtx,
) {
    if residues.is_empty() {
        return;
    }
    let (atoms, res_vec) = flatten_residues(residues);
    let id = ctx.allocator.allocate();
    match mol_type {
        MoleculeType::Protein => {
            ctx.out.push(MoleculeEntity::Protein(
                ProteinEntity::new_normalized(
                    id,
                    atoms,
                    res_vec,
                    chain_id.to_owned(),
                    CompletionMode::HeavyOnly,
                ),
            ));
        }
        MoleculeType::DNA | MoleculeType::RNA => {
            ctx.out.push(MoleculeEntity::NucleicAcid(
                NAEntity::new_normalized(
                    id,
                    mol_type,
                    atoms,
                    res_vec,
                    chain_id.to_owned(),
                    CompletionMode::HeavyOnly,
                ),
            ));
        }
        _ => unreachable!("emit_polymer_chain called with non-polymer type"),
    }
}

fn emit_chain_bulk(
    residues: &[ResidueAccum],
    mol_type: MoleculeType,
    default_resname: [u8; 3],
    ctx: &mut ChainCtx,
) {
    if residues.is_empty() {
        return;
    }
    let mut atoms: Vec<Atom> = Vec::new();
    for r in residues {
        for name in &r.atom_order {
            atoms.push(r.atoms[name].to_atom());
        }
    }
    let residue_name = residues
        .first()
        .map_or(default_resname, |r| r.label_comp_id);
    let id = ctx.allocator.allocate();
    ctx.out.push(MoleculeEntity::Bulk(BulkEntity::new(
        id,
        mol_type,
        atoms,
        residue_name,
        residues.len(),
    )));
}

fn emit_single_residue_small_molecule(
    r: &ResidueAccum,
    mol_type: MoleculeType,
    ctx: &mut ChainCtx,
) {
    let atoms = residue_to_atoms(r);
    let id = ctx.allocator.allocate();
    ctx.out
        .push(MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
            id,
            mol_type,
            atoms,
            r.label_comp_id,
        )));
}

fn emit_non_polymer_chain(residues: &[ResidueAccum], ctx: &mut ChainCtx) {
    for r in residues {
        let mol_type = classify_residue(trim_res_name(&r.label_comp_id));
        match mol_type {
            MoleculeType::Water => ctx.water.ingest(r),
            MoleculeType::Solvent => ctx.solvent.ingest(r),
            MoleculeType::Protein | MoleculeType::DNA | MoleculeType::RNA => {
                // Hint says non-polymer; backbone-bearing residues fall
                // back to Ligand rather than building a polymer.
                emit_single_residue_small_molecule(
                    r,
                    MoleculeType::Ligand,
                    ctx,
                );
            }
            MoleculeType::Ligand
            | MoleculeType::Ion
            | MoleculeType::Cofactor
            | MoleculeType::Lipid => {
                emit_single_residue_small_molecule(r, mol_type, ctx);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::upper_case_acronyms,
    reason = "DNA / RNA mirror MoleculeType's variant names"
)]
enum UnknownBucket {
    Protein,
    /// A nucleic-acid residue. DNA vs RNA is decided once per chain by a
    /// chemistry tally (see [`chain_na_type`]), not per residue, so a
    /// single chain never splits across both.
    NucleicAcid,
    Water,
    Solvent,
    Small(MoleculeType),
}

fn assign_unknown_bucket(r: &ResidueAccum, has_protein: bool) -> UnknownBucket {
    let mol_type = classify_residue(trim_res_name(&r.label_comp_id));
    match mol_type {
        MoleculeType::Protein => UnknownBucket::Protein,
        MoleculeType::DNA | MoleculeType::RNA => UnknownBucket::NucleicAcid,
        MoleculeType::Water => UnknownBucket::Water,
        MoleculeType::Solvent => UnknownBucket::Solvent,
        MoleculeType::Ligand
        | MoleculeType::Ion
        | MoleculeType::Cofactor
        | MoleculeType::Lipid => {
            if has_protein && residue_has_protein_backbone(r) {
                UnknownBucket::Protein
            } else if residue_has_na_backbone(r)
                && !matches!(
                    residue_sugar_chemistry(r),
                    SugarChemistry::Indeterminate
                )
            {
                // Legacy single-char names that the residue-name map mis-
                // sorts (bare `T` -> Ligand) but whose sugar-phosphate
                // chemistry is unambiguously nucleic acid.
                UnknownBucket::NucleicAcid
            } else {
                UnknownBucket::Small(mol_type)
            }
        }
    }
}

/// True if the residue carries the NA sugar-phosphate backbone atoms the
/// canonicalize partition requires (`O5'`, `C5'`, `C4'`, `C3'`, `O3'`;
/// `P` optional for a 5'-terminal residue). Names are v3-normalized at
/// the builder choke point, so this matches the apostrophe-prime form.
fn residue_has_na_backbone(r: &ResidueAccum) -> bool {
    const REQUIRED: [&[u8]; 5] = [b"O5'", b"C5'", b"C4'", b"C3'", b"O3'"];
    REQUIRED
        .iter()
        .all(|want| r.atoms.keys().any(|name| trim_atom_key(name) == *want))
}

/// Per-residue sugar chemistry: a `C2'`-bearing residue with `O2'`
/// present is a ribose (RNA); `C2'` present and `O2'` absent is a
/// deoxyribose (DNA). Atom names are already v3-normalized at the
/// builder choke point, so the probe matches the canonical `O2'`/`C2'`.
#[derive(Clone, Copy)]
enum SugarChemistry {
    Ribose,
    Deoxyribose,
    Indeterminate,
}

fn residue_sugar_chemistry(r: &ResidueAccum) -> SugarChemistry {
    sugar_chemistry_from_keys(r.atoms.keys())
}

fn sugar_chemistry_from_keys<'a>(
    keys: impl IntoIterator<Item = &'a [u8; 4]>,
) -> SugarChemistry {
    let mut has_o2prime = false;
    let mut has_c2prime = false;
    for name in keys {
        match trim_atom_key(name) {
            b"O2'" => has_o2prime = true,
            b"C2'" => has_c2prime = true,
            _ => {}
        }
    }
    if has_o2prime {
        SugarChemistry::Ribose
    } else if has_c2prime {
        SugarChemistry::Deoxyribose
    } else {
        SugarChemistry::Indeterminate
    }
}

/// Decide a single `na_type` for a chain's nucleic-acid residues.
///
/// Counts ribose- vs deoxyribose-bearing residues across the selected
/// residues and picks the majority (ties favor DNA: a deoxyribose strand
/// must not gain a spurious 2'-OH). Falls back to the residue-name vote
/// only when no residue carries a probeable sugar atom (e.g. a base-only
/// parse), where chemistry is silent.
fn chain_na_type(selected: &[&ResidueAccum]) -> MoleculeType {
    let mut ribose = 0usize;
    let mut deoxy = 0usize;
    for r in selected {
        match residue_sugar_chemistry(r) {
            SugarChemistry::Ribose => ribose += 1,
            SugarChemistry::Deoxyribose => deoxy += 1,
            SugarChemistry::Indeterminate => {}
        }
    }
    if ribose == 0 && deoxy == 0 {
        let name_votes_dna = selected.iter().any(|r| {
            classify_residue(trim_res_name(&r.label_comp_id))
                == MoleculeType::DNA
        });
        return if name_votes_dna {
            MoleculeType::DNA
        } else {
            MoleculeType::RNA
        };
    }
    if ribose > deoxy {
        MoleculeType::RNA
    } else {
        MoleculeType::DNA
    }
}

/// 4-byte atom-name key trimmed of trailing space / NUL padding.
fn trim_atom_key(name: &[u8; 4]) -> &[u8] {
    let end = name
        .iter()
        .rposition(|&b| b != b' ' && b != b'\0')
        .map_or(0, |i| i + 1);
    &name[..end]
}

/// Full residue-name heuristic with the protein-merge logic: a residue
/// that classifies as Ligand / Ion / Cofactor / Lipid but carries
/// protein backbone atoms gets folded into the chain's protein bucket
/// when one exists.
fn emit_unknown_chain(
    chain_id: &str,
    residues: &[ResidueAccum],
    ctx: &mut ChainCtx,
) {
    let has_protein = residues.iter().any(|r| {
        classify_residue(trim_res_name(&r.label_comp_id))
            == MoleculeType::Protein
    });
    let buckets: Vec<UnknownBucket> = residues
        .iter()
        .map(|r| assign_unknown_bucket(r, has_protein))
        .collect();

    emit_unknown_protein(residues, &buckets, chain_id, ctx);
    emit_unknown_nucleic_acid(residues, &buckets, chain_id, ctx);

    for (r, bucket) in residues.iter().zip(buckets.iter()) {
        match bucket {
            UnknownBucket::Water => ctx.water.ingest(r),
            UnknownBucket::Solvent => ctx.solvent.ingest(r),
            UnknownBucket::Small(mt) => {
                emit_single_residue_small_molecule(r, *mt, ctx);
            }
            UnknownBucket::Protein | UnknownBucket::NucleicAcid => {}
        }
    }
}

fn select_bucket<'a>(
    residues: &'a [ResidueAccum],
    buckets: &[UnknownBucket],
    target: UnknownBucket,
) -> Vec<&'a ResidueAccum> {
    residues
        .iter()
        .zip(buckets.iter())
        .filter(|(_, b)| **b == target)
        .map(|(r, _)| r)
        .collect()
}

fn emit_unknown_protein(
    residues: &[ResidueAccum],
    buckets: &[UnknownBucket],
    chain_id: &str,
    ctx: &mut ChainCtx,
) {
    let selected = select_bucket(residues, buckets, UnknownBucket::Protein);
    if selected.is_empty() {
        return;
    }
    let (atoms, res_vec) = flatten_residues(selected.iter().copied());
    let id = ctx.allocator.allocate();
    ctx.out
        .push(MoleculeEntity::Protein(ProteinEntity::new_normalized(
            id,
            atoms,
            res_vec,
            chain_id.to_owned(),
            CompletionMode::HeavyOnly,
        )));
}

fn emit_unknown_nucleic_acid(
    residues: &[ResidueAccum],
    buckets: &[UnknownBucket],
    chain_id: &str,
    ctx: &mut ChainCtx,
) {
    let selected = select_bucket(residues, buckets, UnknownBucket::NucleicAcid);
    if selected.is_empty() {
        return;
    }
    let na_type = chain_na_type(&selected);
    let (atoms, res_vec) = flatten_residues(selected.iter().copied());
    let id = ctx.allocator.allocate();
    ctx.out
        .push(MoleculeEntity::NucleicAcid(NAEntity::new_normalized(
            id,
            na_type,
            atoms,
            res_vec,
            chain_id.to_owned(),
            CompletionMode::HeavyOnly,
        )));
}

fn trim_res_name(name: &[u8; 3]) -> &str {
    std::str::from_utf8(name).unwrap_or("").trim()
}

/// True if the residue has PDB-style N, CA, and C atoms; the trigger
/// for modified-residue merging into a Protein chain.
fn residue_has_protein_backbone(residue: &ResidueAccum) -> bool {
    let mut has_n = false;
    let mut has_ca = false;
    let mut has_c = false;
    for name in residue.atoms.keys() {
        match name {
            [b' ', b'N', b' ', b' '] | [b'N', b' ', b' ', b' '] => has_n = true,
            [b' ', b'C', b'A', b' '] | [b'C', b'A', b' ', b' '] => {
                has_ca = true;
            }
            [b' ', b'C', b' ', b' '] | [b'C', b' ', b' ', b' '] => has_c = true,
            _ => {}
        }
    }
    has_n && has_ca && has_c
}

fn residue_to_atoms(r: &ResidueAccum) -> Vec<Atom> {
    r.atom_order
        .iter()
        .map(|name| r.atoms[name].to_atom())
        .collect()
}

fn flatten_residues<'a>(
    residues: impl IntoIterator<Item = &'a ResidueAccum>,
) -> (Vec<Atom>, Vec<Residue>) {
    let mut atoms: Vec<Atom> = Vec::new();
    let mut out_residues: Vec<Residue> = Vec::new();
    for r in residues {
        let start = atoms.len();
        for name in &r.atom_order {
            atoms.push(r.atoms[name].to_atom());
        }
        let end = atoms.len();
        out_residues.push(Residue {
            name: r.label_comp_id,
            label_seq_id: r.label_seq_id,
            auth_seq_id: r.auth_seq_id,
            auth_comp_id: r.auth_comp_id,
            ins_code: r.ins_code,
            atom_range: start..end,
            variants: Vec::new(),
        });
    }
    (atoms, out_residues)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(s: &str) -> [u8; 4] {
        let mut b = [b' '; 4];
        for (i, c) in s.bytes().take(4).enumerate() {
            b[i] = c;
        }
        b
    }

    #[test]
    fn ribose_keys_classify_as_rna() {
        let keys = [key("C2'"), key("O2'"), key("C1'")];
        assert!(matches!(
            sugar_chemistry_from_keys(keys.iter()),
            SugarChemistry::Ribose
        ));
    }

    #[test]
    fn deoxyribose_keys_classify_as_dna() {
        let keys = [key("C2'"), key("C1'"), key("O4'")];
        assert!(matches!(
            sugar_chemistry_from_keys(keys.iter()),
            SugarChemistry::Deoxyribose
        ));
    }

    #[test]
    fn no_sugar_atoms_is_indeterminate() {
        let keys = [key("N1"), key("C2"), key("N3")];
        assert!(matches!(
            sugar_chemistry_from_keys(keys.iter()),
            SugarChemistry::Indeterminate
        ));
    }

    #[test]
    fn trim_atom_key_strips_padding() {
        assert_eq!(trim_atom_key(&key("O2'")), b"O2'");
        assert_eq!(trim_atom_key(&key("P")), b"P");
    }
}
