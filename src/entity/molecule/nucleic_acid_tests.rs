//! Tests for `nucleic_acid.rs`. Split into a sibling file to keep the
//! primary module under the 800-line source cap enforced by
//! `just file-lengths`.

#![allow(clippy::unwrap_used)]

use super::*;
use crate::element::Element;
use crate::entity::molecule::id::EntityIdAllocator;

fn atom_with(name: &str, el: Element, x: f32) -> Atom {
    let mut n = [b' '; 4];
    for (i, b) in name.bytes().take(4).enumerate() {
        n[i] = b;
    }
    Atom {
        position: Vec3::new(x, 0.0, 0.0),
        occupancy: 1.0,
        b_factor: 0.0,
        element: el,
        name: n,
        formal_charge: 0,
    }
}

fn res_name_bytes(s: &str) -> [u8; 3] {
    let mut n = [b' '; 3];
    for (i, b) in s.bytes().take(3).enumerate() {
        n[i] = b;
    }
    n
}

/// Build an NAEntity with a single canonical adenine (DA) residue
/// from a scrambled atom order.
fn scrambled_adenine_entity() -> NAEntity {
    // Intended atoms for a single DA residue (DNA adenine):
    // Backbone: P, O5', C5', C4', C3', O3'
    // Sugar extras: O4', C1', C2'
    // Base: N9, C8, N7, C5, C4, N3, C2, N1, C6, N6
    // H: HA1' (non-existent but used for test)
    // Let's scramble.
    let atoms = vec![
        atom_with("O3'", Element::O, 1.0), // backbone position 5
        atom_with("C4'", Element::C, 2.0), // backbone position 3
        atom_with("N9", Element::N, 3.0),  // base
        atom_with("P", Element::P, 4.0),   // backbone position 0
        atom_with("C5'", Element::C, 5.0), // backbone position 2
        atom_with("O5'", Element::O, 6.0), // backbone position 1
        atom_with("C3'", Element::C, 7.0), // backbone position 4
        atom_with("H8", Element::H, 8.0),  // hydrogen
        atom_with("O4'", Element::O, 9.0), /* base (but really sugar);
                                            * treated as non-backbone */
        atom_with("C1'", Element::C, 10.0), // non-backbone heavy
    ];
    let residues = vec![Residue {
        name: res_name_bytes("DA "),
        label_seq_id: 1,
        auth_seq_id: None,
        auth_comp_id: None,
        ins_code: None,
        atom_range: 0..atoms.len(),
        variants: Vec::new(),
    }];
    let mut alloc = EntityIdAllocator::new();
    let id = alloc.allocate();
    NAEntity::new(id, MoleculeType::DNA, atoms, residues, "A".to_owned())
}

#[test]
fn na_canonical_ordering_places_backbone_first() {
    let na = scrambled_adenine_entity();
    assert_eq!(na.residues.len(), 1);
    let r = &na.residues[0];
    let na_atoms = na.columns().to_atoms();
    let names: Vec<&str> = r
        .atom_range
        .clone()
        .map(|i| std::str::from_utf8(&na_atoms[i].name).unwrap().trim())
        .collect();
    // First six must be the canonical backbone.
    assert_eq!(&names[..6], &["P", "O5'", "C5'", "C4'", "C3'", "O3'"]);
}

#[test]
fn na_bonds_include_sugar_phosphate_and_purine_anchor() {
    let na = scrambled_adenine_entity();
    let na_atoms = na.columns().to_atoms();
    // bonds are AtomId-endpoint. Reconstruct by name.
    let name_of = |aid: AtomId| {
        std::str::from_utf8(&na_atoms[aid.index as usize].name)
            .unwrap()
            .trim()
            .to_owned()
    };
    let pairs: Vec<(String, String)> = na
        .bonds
        .iter()
        .map(|b| (name_of(b.a), name_of(b.b)))
        .collect();
    // P-O5' should be present
    assert!(pairs.iter().any(|(a, b)| {
        (a == "P" && b == "O5'") || (a == "O5'" && b == "P")
    }));
    // C1'-N9 (sugar-base anchor for purine)
    assert!(pairs.iter().any(|(a, b)| {
        (a == "C1'" && b == "N9") || (a == "N9" && b == "C1'")
    }));
}

/// Build an RNA adenine residue from real ribose-component ideal
/// geometry with the 2'-OH (`O2'`/`HO2'`) deleted, then run
/// completion. A correct ribose template fit must fabricate `O2'`.
fn rna_adenine_missing_o2_prime() -> NAEntity {
    use crate::chemistry::nucleotides::Nucleotide;
    // RNA adenine has a ribose template.
    let template = Nucleotide::A.template(MoleculeType::RNA).unwrap();
    // Carry every in-scope heavy atom EXCEPT the 2'-OH, at the
    // template's own ideal coordinates (any rigid frame works; the
    // fit recovers it). Skipping O2'/HO2' makes them the missing set.
    let atoms: Vec<Atom> = template
        .atoms
        .iter()
        .filter(|a| {
            a.element != Element::H && !a.is_leaving && a.name.as_str() != "O2'"
        })
        .map(|a| {
            let mut name = [b' '; 4];
            for (i, b) in a.name.as_str().bytes().take(4).enumerate() {
                name[i] = b;
            }
            Atom {
                position: a.ideal,
                occupancy: 1.0,
                b_factor: 0.0,
                element: a.element,
                name,
                formal_charge: 0,
            }
        })
        .collect();
    let residues = vec![Residue {
        name: res_name_bytes("A  "),
        label_seq_id: 1,
        auth_seq_id: None,
        auth_comp_id: None,
        ins_code: None,
        atom_range: 0..atoms.len(),
        variants: Vec::new(),
    }];
    let mut alloc = EntityIdAllocator::new();
    let id = alloc.allocate();
    NAEntity::new_normalized(
        id,
        MoleculeType::RNA,
        atoms,
        residues,
        "A".to_owned(),
        Completion::Heavy,
    )
}

#[test]
fn rna_residue_completes_with_ribose_o2_prime() {
    let na = rna_adenine_missing_o2_prime();
    assert_eq!(na.residues.len(), 1, "RNA adenine should be kept");
    let r = &na.residues[0];
    let na_atoms = na.columns().to_atoms();
    let names: Vec<String> = r
        .atom_range
        .clone()
        .map(|i| {
            std::str::from_utf8(&na_atoms[i].name)
                .unwrap()
                .trim()
                .to_owned()
        })
        .collect();
    // The deleted ribose 2'-OH oxygen must have been fabricated from
    // the RNA (ribose) template, not silently dropped.
    assert!(
        names.iter().any(|n| n == "O2'"),
        "completion should fabricate the ribose O2'; got {names:?}"
    );
}

/// DA heavy atoms at real DNA-template ideal geometry, keeping every
/// in-scope heavy atom whose name passes `keep`. Real coordinates let the
/// completion rigid fit anchor (so base atoms fill) and give the
/// canonicalize/bond stages chemically sane positions.
fn da_heavy_atoms(keep: impl Fn(&str) -> bool) -> Vec<Atom> {
    use crate::chemistry::nucleotides::Nucleotide;
    let template = Nucleotide::A.template(MoleculeType::DNA).unwrap();
    template
        .atoms
        .iter()
        .filter(|a| a.element != Element::H && !a.is_leaving)
        .filter(|a| keep(a.name.as_str()))
        .map(|a| {
            let mut name = [b' '; 4];
            for (i, b) in a.name.as_str().bytes().take(4).enumerate() {
                name[i] = b;
            }
            Atom {
                position: a.ideal,
                occupancy: 1.0,
                b_factor: 0.0,
                element: a.element,
                name,
                formal_charge: 0,
            }
        })
        .collect()
}

fn na_residue(name: &str, range: std::ops::Range<usize>, seq: i32) -> Residue {
    Residue {
        name: res_name_bytes(name),
        label_seq_id: seq,
        auth_seq_id: None,
        auth_comp_id: None,
        ins_code: None,
        atom_range: range,
        variants: Vec::new(),
    }
}

#[test]
fn na_five_prime_oh_residue_is_kept() {
    // A free 5'-OH DA (sugar + base, no P/OP1/OP2): canonicalize must KEEP
    // it as the 5'-terminal residue rather than dropping it for the
    // missing P, and no phantom phosphate appears in its atom set.
    let phosphate = |n: &str| matches!(n, "P" | "OP1" | "OP2" | "OP3");
    let atoms = da_heavy_atoms(|n| !phosphate(n));
    let n = atoms.len();
    let mut alloc = EntityIdAllocator::new();
    let id = alloc.allocate();
    let na = NAEntity::new_normalized(
        id,
        MoleculeType::DNA,
        atoms,
        vec![na_residue("DA", 0..n, 1)],
        "A".to_owned(),
        Completion::Heavy,
    );
    assert_eq!(na.residues.len(), 1, "5'-OH residue must be kept");
    let r = &na.residues[0];
    let na_atoms = na.columns().to_atoms();
    let names: Vec<&str> = r
        .atom_range
        .clone()
        .map(|i| std::str::from_utf8(&na_atoms[i].name).unwrap().trim())
        .collect();
    for forged in ["P", "OP1", "OP2", "OP3"] {
        assert!(
            !names.contains(&forged),
            "kept 5'-OH residue must carry no fabricated {forged}: {names:?}"
        );
    }
    // The kept backbone starts at O5' (no leading P).
    assert_eq!(names[0], "O5'", "P-less 5' residue starts at O5'");

    // No bond names a (nonexistent) phosphate or its OP oxygens.
    for bond in &na.bonds {
        for aid in [bond.a, bond.b] {
            let nm = std::str::from_utf8(&na_atoms[aid.index as usize].name)
                .unwrap()
                .trim();
            assert!(
                !matches!(nm, "P" | "OP1" | "OP2" | "OP3"),
                "no phosphate bond for a 5'-OH residue, found {nm}"
            );
        }
    }
    // No spurious segment break introduced (single residue).
    assert!(na.segment_breaks.is_empty());
}

#[test]
fn na_five_prime_oh_then_phosphodiester_to_next() {
    // 5'-OH DA (no phosphate) followed by a normal phosphate-bearing DA.
    // The phosphodiester bond O3'(0) -> P(1) must still form, using the
    // P-less residue's shorter (5-atom) backbone offset for O3'.
    let phosphate = |n: &str| matches!(n, "P" | "OP1" | "OP2" | "OP3");
    let first = da_heavy_atoms(|n| !phosphate(n));
    // Shift the second residue along x so its P sits within bonding range
    // of the first residue's O3' (the ideal frames otherwise overlap).
    let mut second = da_heavy_atoms(|_| true);
    for a in &mut second {
        a.position += Vec3::new(6.0, 0.0, 0.0);
    }
    let first_len = first.len();
    let mut atoms = first;
    atoms.extend(second);
    let residues = vec![
        na_residue("DA", 0..first_len, 1),
        na_residue("DA", first_len..atoms.len(), 2),
    ];
    let mut alloc = EntityIdAllocator::new();
    let id = alloc.allocate();
    let na = NAEntity::new_normalized(
        id,
        MoleculeType::DNA,
        atoms,
        residues,
        "A".to_owned(),
        Completion::Heavy,
    );
    assert_eq!(na.residues.len(), 2, "both residues kept");
    let na_atoms = na.columns().to_atoms();
    let name_of = |aid: AtomId| {
        std::str::from_utf8(&na_atoms[aid.index as usize].name)
            .unwrap()
            .trim()
            .to_owned()
    };
    let pairs: Vec<(String, String)> = na
        .bonds
        .iter()
        .map(|b| (name_of(b.a), name_of(b.b)))
        .collect();
    assert!(
        pairs.iter().any(|(a, b)| {
            (a == "O3'" && b == "P") || (a == "P" && b == "O3'")
        }),
        "O3'(0)->P(1) phosphodiester must form across the 5'-OH terminus; got \
         {pairs:?}"
    );
}

#[test]
fn protonated_na_strips_h_heavy_only_keeps_h_otherwise() {
    // A protonated DA parse (every heavy atom at real ideal geometry plus
    // an input base proton H8): under Heavy the canonical entity must
    // carry NO hydrogen; the pure `new` path (Completion::Raw) keeps
    // the input proton verbatim.
    let protonated = || {
        let mut atoms = da_heavy_atoms(|_| true);
        atoms.push(atom_with("H8", Element::H, 50.0));
        atoms
    };
    let n = protonated().len();
    let mut alloc = EntityIdAllocator::new();
    let id = alloc.allocate();

    let heavy = NAEntity::new_normalized(
        id,
        MoleculeType::DNA,
        protonated(),
        vec![na_residue("DA", 0..n, 1)],
        "A".to_owned(),
        Completion::Heavy,
    );
    let r = &heavy.residues[0];
    let heavy_atoms = heavy.columns().to_atoms();
    assert!(
        r.atom_range
            .clone()
            .all(|i| heavy_atoms[i].element != Element::H),
        "heavy-only NA ingest strips every input hydrogen"
    );

    let kept = NAEntity::new(
        id,
        MoleculeType::DNA,
        protonated(),
        vec![na_residue("DA", 0..n, 1)],
        "A".to_owned(),
    );
    let r = &kept.residues[0];
    let kept_atoms = kept.columns().to_atoms();
    assert!(
        r.atom_range
            .clone()
            .any(|i| kept_atoms[i].element == Element::H),
        "the pure (None) path keeps the input hydrogen"
    );
}

#[test]
fn canonical_na_name_maps_to_bridge_contract() {
    let nm = |s: &str, t: MoleculeType| {
        std::str::from_utf8(&canonical_na_name(res_name_bytes(s), t))
            .unwrap()
            .trim()
            .to_owned()
    };
    // DNA: single-char and alias forms collapse to the two-char contract.
    assert_eq!(nm("A", MoleculeType::DNA), "DA");
    assert_eq!(nm("DA", MoleculeType::DNA), "DA");
    assert_eq!(nm("T", MoleculeType::DNA), "DT");
    assert_eq!(nm("DG", MoleculeType::DNA), "DG");
    // RNA: collapses to the single-char contract.
    assert_eq!(nm("A", MoleculeType::RNA), "A");
    assert_eq!(nm("U", MoleculeType::RNA), "U");
    assert_eq!(nm("ADE", MoleculeType::RNA), "A");
    // Unknown base name passes through verbatim.
    assert_eq!(nm("XYZ", MoleculeType::DNA), "XYZ");
}

#[test]
fn na_drops_residue_missing_backbone() {
    // Residue with only P and O5': missing four backbone atoms.
    // Two present atoms are below the rigid-fit anchor minimum, so
    // completion fills nothing and the residue still drops.
    let atoms = vec![
        atom_with("P", Element::P, 0.0),
        atom_with("O5'", Element::O, 1.0),
    ];
    let residues = vec![Residue {
        name: res_name_bytes("DA "),
        label_seq_id: 1,
        auth_seq_id: None,
        auth_comp_id: None,
        ins_code: None,
        atom_range: 0..atoms.len(),
        variants: Vec::new(),
    }];
    let mut alloc = EntityIdAllocator::new();
    let id = alloc.allocate();
    let na =
        NAEntity::new(id, MoleculeType::DNA, atoms, residues, "A".to_owned());
    assert_eq!(na.residues.len(), 0);
    // Atoms still present (carried through, unreferenced).
    assert_eq!(na.columns.len(), 2);
}
