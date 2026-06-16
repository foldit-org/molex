//! Classification dispatch: picks an entity kind for each chain and
//! emits one or more `MoleculeEntity` values into the `ChainCtx`.

use super::{
    ChainBytes, ChainCtx, ExpectedEntityType, ResidueAccum,
    DEFAULT_WATER_RESNAME,
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
    chain: ChainBytes,
    residues: &[ResidueAccum],
    ctx: &mut ChainCtx,
) {
    match hint {
        ExpectedEntityType::Protein => {
            emit_polymer_chain(residues, MoleculeType::Protein, chain, ctx);
        }
        ExpectedEntityType::DNA => {
            emit_polymer_chain(residues, MoleculeType::DNA, chain, ctx);
        }
        ExpectedEntityType::RNA => {
            emit_polymer_chain(residues, MoleculeType::RNA, chain, ctx);
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
            emit_unknown_chain(chain, residues, ctx);
        }
    }
}

fn emit_polymer_chain(
    residues: &[ResidueAccum],
    mol_type: MoleculeType,
    chain: ChainBytes,
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
                    chain.pdb_chain_id,
                    chain.auth_asym_id,
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
                    chain.pdb_chain_id,
                    chain.auth_asym_id,
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
    DNA,
    RNA,
    Water,
    Solvent,
    Small(MoleculeType),
}

fn assign_unknown_bucket(r: &ResidueAccum, has_protein: bool) -> UnknownBucket {
    let mol_type = classify_residue(trim_res_name(&r.label_comp_id));
    match mol_type {
        MoleculeType::Protein => UnknownBucket::Protein,
        MoleculeType::DNA => UnknownBucket::DNA,
        MoleculeType::RNA => UnknownBucket::RNA,
        MoleculeType::Water => UnknownBucket::Water,
        MoleculeType::Solvent => UnknownBucket::Solvent,
        MoleculeType::Ligand
        | MoleculeType::Ion
        | MoleculeType::Cofactor
        | MoleculeType::Lipid => {
            if has_protein && residue_has_protein_backbone(r) {
                UnknownBucket::Protein
            } else {
                UnknownBucket::Small(mol_type)
            }
        }
    }
}

/// Full residue-name heuristic with the protein-merge logic: a residue
/// that classifies as Ligand / Ion / Cofactor / Lipid but carries
/// protein backbone atoms gets folded into the chain's protein bucket
/// when one exists.
fn emit_unknown_chain(
    chain: ChainBytes,
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

    emit_unknown_polymer(
        UnknownBucket::Protein,
        residues,
        &buckets,
        chain,
        ctx,
    );
    emit_unknown_polymer(UnknownBucket::DNA, residues, &buckets, chain, ctx);
    emit_unknown_polymer(UnknownBucket::RNA, residues, &buckets, chain, ctx);

    for (r, bucket) in residues.iter().zip(buckets.iter()) {
        match bucket {
            UnknownBucket::Water => ctx.water.ingest(r),
            UnknownBucket::Solvent => ctx.solvent.ingest(r),
            UnknownBucket::Small(mt) => {
                emit_single_residue_small_molecule(r, *mt, ctx);
            }
            UnknownBucket::Protein
            | UnknownBucket::DNA
            | UnknownBucket::RNA => {}
        }
    }
}

fn emit_unknown_polymer(
    target: UnknownBucket,
    residues: &[ResidueAccum],
    buckets: &[UnknownBucket],
    chain: ChainBytes,
    ctx: &mut ChainCtx,
) {
    let selected: Vec<&ResidueAccum> = residues
        .iter()
        .zip(buckets.iter())
        .filter(|(_, b)| **b == target)
        .map(|(r, _)| r)
        .collect();
    if selected.is_empty() {
        return;
    }
    let (atoms, res_vec) = flatten_residues(selected.iter().copied());
    let id = ctx.allocator.allocate();
    match target {
        UnknownBucket::Protein => {
            ctx.out.push(MoleculeEntity::Protein(
                ProteinEntity::new_normalized(
                    id,
                    atoms,
                    res_vec,
                    chain.pdb_chain_id,
                    chain.auth_asym_id,
                    CompletionMode::HeavyOnly,
                ),
            ));
        }
        UnknownBucket::DNA => {
            ctx.out.push(MoleculeEntity::NucleicAcid(
                NAEntity::new_normalized(
                    id,
                    MoleculeType::DNA,
                    atoms,
                    res_vec,
                    chain.pdb_chain_id,
                    chain.auth_asym_id,
                    CompletionMode::HeavyOnly,
                ),
            ));
        }
        UnknownBucket::RNA => {
            ctx.out.push(MoleculeEntity::NucleicAcid(
                NAEntity::new_normalized(
                    id,
                    MoleculeType::RNA,
                    atoms,
                    res_vec,
                    chain.pdb_chain_id,
                    chain.auth_asym_id,
                    CompletionMode::HeavyOnly,
                ),
            ));
        }
        UnknownBucket::Water
        | UnknownBucket::Solvent
        | UnknownBucket::Small(_) => {
            unreachable!("emit_unknown_polymer called with non-polymer bucket")
        }
    }
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
