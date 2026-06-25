//! End-to-end legacy nucleic-acid ingest through `EntityBuilder`:
//! legacy atom-name normalization, chemistry-based DNA/RNA
//! classification, and residue-name canonicalization, exercised together.
//! Confirms a legacy-named single-char B-DNA strand ingests as DNA and a
//! modern single-char RNA strand stays RNA through the full ingest path.

#![allow(clippy::unwrap_used, clippy::too_many_arguments)]

use super::tests::{raw_atomname, RowBuilder};
use super::EntityBuilder;
use crate::chemistry::nucleotides::Nucleotide;
use crate::element::Element;
use crate::entity::molecule::nucleic_acid::NAEntity;
use crate::entity::molecule::{MoleculeEntity, MoleculeType};

/// Map a v3 atom name to the legacy pre-2007 PDB spelling: sugar primes
/// become star-primes (`O5'` -> `O5*`) and phosphate oxygens become
/// `O1P`/`O2P`/`O3P`. Other names pass through.
fn legacy_atom_name(v3: &str) -> [u8; 4] {
    let legacy = match v3 {
        "OP1" => "O1P".to_owned(),
        "OP2" => "O2P".to_owned(),
        "OP3" => "O3P".to_owned(),
        _ => v3.replace('\'', "*"),
    };
    raw_atomname(&legacy)
}

/// Push one nucleotide residue into `b` using legacy atom names and the
/// strand-template ideal coordinates, keeping only heavy atoms whose v3
/// name passes `keep`. `comp` is the (legacy single-char or two-char)
/// residue name written into the row.
fn push_legacy_na_residue(
    b: &mut EntityBuilder,
    chain: &str,
    seq: i32,
    comp: &str,
    nt: Nucleotide,
    na_type: MoleculeType,
    keep: impl Fn(&str) -> bool,
) {
    let template = nt.template(na_type).unwrap();
    for a in template.atoms {
        if a.element == Element::H || a.is_leaving {
            continue;
        }
        let v3 = a.name.as_str();
        if !keep(v3) {
            continue;
        }
        let row = RowBuilder::new(chain, seq, comp, "X")
            .raw_atom(legacy_atom_name(v3))
            .at(a.ideal.x, a.ideal.y, a.ideal.z)
            .elem(a.element);
        b.push_atom(row.build()).unwrap();
    }
}

fn na_entity(entities: &[MoleculeEntity]) -> &NAEntity {
    entities
        .iter()
        .find_map(|e| match e {
            MoleculeEntity::NucleicAcid(na) => Some(na),
            _ => None,
        })
        .unwrap()
}

fn residue_names(na: &NAEntity) -> Vec<String> {
    na.residues
        .iter()
        .map(|r| std::str::from_utf8(&r.name).unwrap().trim().to_owned())
        .collect()
}

/// A legacy-named B-DNA strand (single-char `A`/`C`/`G`/`T`, star-prime
/// sugar atoms, `O1P`/`O2P` phosphate oxygens, deoxyribose) must parse to
/// a kept DNA entity with two-char canonical residue names and v3
/// backbone atom names: N1 normalizes the atoms, N2 classifies by sugar
/// chemistry (not name), N3 renames to the bridge contract.
#[test]
fn legacy_single_char_dna_strand_ingests_as_dna() {
    let mut b = EntityBuilder::new();
    let strand = [
        (1, "A", Nucleotide::A),
        (2, "C", Nucleotide::C),
        (3, "G", Nucleotide::G),
        (4, "T", Nucleotide::T),
    ];
    for (seq, comp, nt) in strand {
        push_legacy_na_residue(
            &mut b,
            "A",
            seq,
            comp,
            nt,
            MoleculeType::DNA,
            |_| true,
        );
    }
    let entities = b.finish().unwrap();
    let na = na_entity(&entities);

    // NA residues survived canonicalize (chemistry classified as NA even
    // though bare `T` classifies by name as Ligand).
    assert_eq!(na.residues.len(), 4, "all four NA residues kept");

    // Chemistry classification: deoxyribose -> DNA, not name-driven RNA.
    assert_eq!(na.na_type, MoleculeType::DNA);

    // Names canonicalized to the two-char DNA contract.
    assert_eq!(residue_names(na), vec!["DA", "DC", "DG", "DT"]);

    // Backbone atoms present under v3 names after normalization.
    use crate::entity::molecule::traits::Entity;
    let na_atoms = na.columns().to_atoms();
    let r0_names: Vec<&str> = na.residues[0]
        .atom_range
        .clone()
        .map(|i| std::str::from_utf8(&na_atoms[i].name).unwrap().trim())
        .collect();
    for v3 in ["P", "O5'", "C5'", "C4'", "C3'", "O3'"] {
        assert!(
            r0_names.contains(&v3),
            "expected v3 backbone atom {v3}; got {r0_names:?}"
        );
    }
    // No legacy star-prime leaked through.
    assert!(
        !r0_names.iter().any(|n| n.contains('*')),
        "no star-prime atom names should survive; got {r0_names:?}"
    );
}

/// A modern single-char RNA strand (`A`/`C`/`G`/`U` with ribose `O2'`)
/// stays RNA with single-char names: the DNA/RNA split is chemistry-keyed.
#[test]
fn modern_single_char_rna_strand_stays_rna() {
    let mut b = EntityBuilder::new();
    let strand = [
        (1, "A", Nucleotide::A),
        (2, "C", Nucleotide::C),
        (3, "G", Nucleotide::G),
        (4, "U", Nucleotide::U),
    ];
    for (seq, comp, nt) in strand {
        push_legacy_na_residue(
            &mut b,
            "A",
            seq,
            comp,
            nt,
            MoleculeType::RNA,
            |_| true,
        );
    }
    let entities = b.finish().unwrap();
    let na = na_entity(&entities);

    assert_eq!(na.na_type, MoleculeType::RNA);
    assert_eq!(residue_names(na), vec!["A", "C", "G", "U"]);
}
