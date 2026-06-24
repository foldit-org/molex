//! Tests for `EntityBuilder`: heuristic, hint, altloc, and error paths.
//!
//! Author-side / metadata round-trip tests live in
//! `super::roundtrip_tests`; this module exports the shared
//! `RowBuilder` helpers via `pub(super)` so the sibling test module can
//! reuse them.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::too_many_arguments,
    clippy::suboptimal_flops,
    clippy::match_wildcard_for_single_variants,
    clippy::panic
)]

use super::*;

// Row construction helpers

pub(super) fn resname(s: &str) -> [u8; 3] {
    let mut n = [b' '; 3];
    for (i, b) in s.bytes().take(3).enumerate() {
        n[i] = b;
    }
    n
}

pub(super) fn atomname(s: &str) -> [u8; 4] {
    let mut n = [b' '; 4];
    // PDB-style names: " CA " (single-letter element, padded col 13-16
    // with the element in col 14). Single-letter callers ("N", "C")
    // produce " N  " / " C  "; two-letter ("CA") produces " CA "; the
    // four-letter case ("OXT") fills " OXT".
    if s.len() <= 3 {
        for (i, b) in s.bytes().enumerate() {
            n[i + 1] = b;
        }
    } else {
        for (i, b) in s.bytes().take(4).enumerate() {
            n[i] = b;
        }
    }
    n
}

pub(super) fn raw_atomname(s: &str) -> [u8; 4] {
    let mut n = [b' '; 4];
    for (i, b) in s.bytes().take(4).enumerate() {
        n[i] = b;
    }
    n
}

pub(super) struct RowBuilder {
    chain: String,
    seq: i32,
    comp: [u8; 3],
    atom: [u8; 4],
    element: Element,
    pos: (f32, f32, f32),
    occupancy: f32,
    alt_loc: Option<u8>,
    ins_code: Option<u8>,
    entity_id: Option<String>,
    auth_chain: Option<String>,
    auth_seq: Option<i32>,
    auth_comp: Option<[u8; 3]>,
    auth_atom: Option<[u8; 4]>,
    formal_charge: i8,
}

impl RowBuilder {
    pub(super) fn new(chain: &str, seq: i32, comp: &str, atom: &str) -> Self {
        Self {
            chain: chain.to_owned(),
            seq,
            comp: resname(comp),
            atom: atomname(atom),
            element: Element::C,
            pos: (0.0, 0.0, 0.0),
            occupancy: 1.0,
            alt_loc: None,
            ins_code: None,
            entity_id: None,
            auth_chain: None,
            auth_seq: None,
            auth_comp: None,
            auth_atom: None,
            formal_charge: 0,
        }
    }

    pub(super) fn at(mut self, x: f32, y: f32, z: f32) -> Self {
        self.pos = (x, y, z);
        self
    }

    pub(super) fn elem(mut self, element: Element) -> Self {
        self.element = element;
        self
    }

    pub(super) fn alt(mut self, alt_loc: u8, occupancy: f32) -> Self {
        self.alt_loc = Some(alt_loc);
        self.occupancy = occupancy;
        self
    }

    pub(super) fn ins(mut self, ins_code: u8) -> Self {
        self.ins_code = Some(ins_code);
        self
    }

    pub(super) fn entity(mut self, eid: &str) -> Self {
        self.entity_id = Some(eid.to_owned());
        self
    }

    pub(super) fn raw_atom(mut self, atom: [u8; 4]) -> Self {
        self.atom = atom;
        self
    }

    pub(super) fn auth_chain(mut self, chain: &str) -> Self {
        self.auth_chain = Some(chain.to_owned());
        self
    }

    pub(super) fn auth_seq(mut self, seq: i32) -> Self {
        self.auth_seq = Some(seq);
        self
    }

    pub(super) fn auth_comp(mut self, comp: &str) -> Self {
        self.auth_comp = Some(resname(comp));
        self
    }

    pub(super) fn auth_atom(mut self, atom: [u8; 4]) -> Self {
        self.auth_atom = Some(atom);
        self
    }

    pub(super) fn formal_charge(mut self, charge: i8) -> Self {
        self.formal_charge = charge;
        self
    }

    pub(super) fn build(self) -> AtomRow {
        AtomRow {
            label_asym_id: self.chain,
            label_seq_id: self.seq,
            label_comp_id: self.comp,
            label_atom_id: self.atom,
            label_entity_id: self.entity_id,
            auth_asym_id: self.auth_chain,
            auth_seq_id: self.auth_seq,
            auth_comp_id: self.auth_comp,
            auth_atom_id: self.auth_atom,
            alt_loc: self.alt_loc,
            ins_code: self.ins_code,
            element: self.element,
            x: self.pos.0,
            y: self.pos.1,
            z: self.pos.2,
            occupancy: self.occupancy,
            b_factor: 0.0,
            formal_charge: self.formal_charge,
        }
    }
}

/// Push a residue with the four standard protein backbone atoms in a
/// straight 3.8 A spacing along x, enough for `ProteinEntity::new` to
/// keep it.
pub(super) fn push_protein_residue(
    b: &mut EntityBuilder,
    chain: &str,
    seq: i32,
    comp: &str,
    base_x: f32,
    entity: Option<&str>,
) {
    let names = ["N", "CA", "C", "O"];
    let elements = [Element::N, Element::C, Element::C, Element::O];
    for ((dx, name), el) in
        [0.0, 1.45, 2.4, 2.8].into_iter().zip(names).zip(elements)
    {
        let mut row = RowBuilder::new(chain, seq, comp, name)
            .at(base_x + dx, 0.0, 0.0)
            .elem(el);
        if let Some(e) = entity {
            row = row.entity(e);
        }
        b.push_atom(row.build()).unwrap();
    }
}

fn push_dna_residue(
    b: &mut EntityBuilder,
    chain: &str,
    seq: i32,
    comp: &str,
    base_x: f32,
    entity: Option<&str>,
) {
    let names = ["P", "O5'", "C5'", "C4'", "C3'", "O3'"];
    let elements = [
        Element::P,
        Element::O,
        Element::C,
        Element::C,
        Element::C,
        Element::O,
    ];
    for (i, (name, el)) in names.iter().zip(elements).enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let mut row = RowBuilder::new(chain, seq, comp, name)
            .at(base_x + (i as f32) * 1.5, 0.0, 0.0)
            .elem(el);
        if let Some(e) = entity {
            row = row.entity(e);
        }
        b.push_atom(row.build()).unwrap();
    }
}

// PDB-style heuristic (no hints)

#[test]
fn pdb_path_single_protein_chain() {
    let mut b = EntityBuilder::new();
    push_protein_residue(&mut b, "A", 1, "ALA", 0.0, None);
    push_protein_residue(&mut b, "A", 2, "GLY", 3.8, None);
    let entities = b.finish().unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].molecule_type(), MoleculeType::Protein);
    let protein = entities[0].as_protein().unwrap();
    assert_eq!(protein.residues.len(), 2);
    assert_eq!(protein.pdb_chain_id, "A");
}

#[test]
fn pdb_path_dna_chain() {
    let mut b = EntityBuilder::new();
    push_dna_residue(&mut b, "B", 1, "DA", 0.0, None);
    push_dna_residue(&mut b, "B", 2, "DT", 10.0, None);
    let entities = b.finish().unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].molecule_type(), MoleculeType::DNA);
}

#[test]
fn pdb_path_rna_chain() {
    let mut b = EntityBuilder::new();
    push_dna_residue(&mut b, "C", 1, "A", 0.0, None);
    push_dna_residue(&mut b, "C", 2, "U", 10.0, None);
    let entities = b.finish().unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].molecule_type(), MoleculeType::RNA);
}

#[test]
fn pdb_path_multi_chain_protein_dna() {
    let mut b = EntityBuilder::new();
    push_protein_residue(&mut b, "A", 1, "ALA", 0.0, None);
    push_protein_residue(&mut b, "A", 2, "GLY", 3.8, None);
    push_dna_residue(&mut b, "B", 1, "DA", 0.0, None);
    let entities = b.finish().unwrap();
    assert_eq!(entities.len(), 2);
    assert_eq!(entities[0].molecule_type(), MoleculeType::Protein);
    assert_eq!(entities[1].molecule_type(), MoleculeType::DNA);
    assert_eq!(entities[0].pdb_chain_id(), Some("A"));
    assert_eq!(entities[1].pdb_chain_id(), Some("B"));
}

#[test]
fn pdb_path_modified_residue_with_backbone_merges_into_protein() {
    let mut b = EntityBuilder::new();
    push_protein_residue(&mut b, "A", 1, "ALA", 0.0, None);
    // SEP (phospho-serine) classifies as Ligand but carries backbone.
    push_protein_residue(&mut b, "A", 2, "SEP", 3.8, None);
    push_protein_residue(&mut b, "A", 3, "GLY", 7.6, None);
    let entities = b.finish().unwrap();
    assert_eq!(entities.len(), 1, "SEP should merge into the protein");
    assert_eq!(entities[0].molecule_type(), MoleculeType::Protein);
    assert_eq!(entities[0].as_protein().unwrap().residues.len(), 3);
}

#[test]
fn pdb_path_modified_residue_without_backbone_stays_separate() {
    let mut b = EntityBuilder::new();
    push_protein_residue(&mut b, "A", 1, "ALA", 0.0, None);
    push_protein_residue(&mut b, "A", 2, "GLY", 3.8, None);
    // SEP without backbone atoms: just a sidechain marker.
    b.push_atom(
        RowBuilder::new("A", 3, "SEP", "OG")
            .at(10.0, 0.0, 0.0)
            .elem(Element::O)
            .build(),
    )
    .unwrap();
    let entities = b.finish().unwrap();
    let kinds: Vec<MoleculeType> =
        entities.iter().map(MoleculeEntity::molecule_type).collect();
    assert!(kinds.contains(&MoleculeType::Protein));
    assert!(kinds.contains(&MoleculeType::Ligand));
}

#[test]
fn pdb_path_heterogeneous_non_polymer_chain_splits() {
    let mut b = EntityBuilder::new();
    b.push_atom(
        RowBuilder::new("X", 1, "HOH", "O")
            .at(0.0, 0.0, 0.0)
            .elem(Element::O)
            .build(),
    )
    .unwrap();
    b.push_atom(
        RowBuilder::new("X", 2, "ZN", "ZN")
            .raw_atom(raw_atomname("ZN  "))
            .at(5.0, 0.0, 0.0)
            .elem(Element::Zn)
            .build(),
    )
    .unwrap();
    b.push_atom(
        RowBuilder::new("X", 3, "ATP", "PA")
            .raw_atom(raw_atomname("PA  "))
            .at(10.0, 0.0, 0.0)
            .elem(Element::P)
            .build(),
    )
    .unwrap();

    let entities = b.finish().unwrap();
    assert_eq!(entities.len(), 3);
    let types: Vec<_> =
        entities.iter().map(MoleculeEntity::molecule_type).collect();
    assert!(types.contains(&MoleculeType::Ion), "{types:?}");
    assert!(types.contains(&MoleculeType::Cofactor), "{types:?}");
    assert!(types.contains(&MoleculeType::Water), "{types:?}");
}

// mmCIF hint path

#[test]
fn hint_protein_keeps_modified_residues() {
    let mut b = EntityBuilder::new();
    b.register_entity("1", ExpectedEntityType::Protein);
    push_protein_residue(&mut b, "A", 1, "ALA", 0.0, Some("1"));
    push_protein_residue(&mut b, "A", 2, "SEP", 3.8, Some("1"));
    push_protein_residue(&mut b, "A", 3, "MSE", 7.6, Some("1"));
    push_protein_residue(&mut b, "A", 4, "PTR", 11.4, Some("1"));
    let entities = b.finish().unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].molecule_type(), MoleculeType::Protein);
    assert_eq!(entities[0].as_protein().unwrap().residues.len(), 4);
}

#[test]
fn hint_dna_keeps_modified_nucleotides() {
    let mut b = EntityBuilder::new();
    b.register_entity("2", ExpectedEntityType::DNA);
    push_dna_residue(&mut b, "B", 1, "DA", 0.0, Some("2"));
    // 5BU (5-bromouridine) classifies as Ligand without hint.
    push_dna_residue(&mut b, "B", 2, "5BU", 10.0, Some("2"));
    let entities = b.finish().unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].molecule_type(), MoleculeType::DNA);
    assert_eq!(entities[0].as_nucleic_acid().unwrap().residues.len(), 2);
}

#[test]
fn hint_rna_keeps_modified_nucleotides() {
    let mut b = EntityBuilder::new();
    b.register_entity("3", ExpectedEntityType::RNA);
    push_dna_residue(&mut b, "R", 1, "A", 0.0, Some("3"));
    push_dna_residue(&mut b, "R", 2, "PSU", 10.0, Some("3"));
    let entities = b.finish().unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].molecule_type(), MoleculeType::RNA);
}

#[test]
fn hint_non_polymer_sub_classifies_per_residue() {
    let mut b = EntityBuilder::new();
    b.register_entity("4", ExpectedEntityType::NonPolymer);
    b.push_atom(
        RowBuilder::new("L", 1, "HOH", "O")
            .at(0.0, 0.0, 0.0)
            .elem(Element::O)
            .entity("4")
            .build(),
    )
    .unwrap();
    b.push_atom(
        RowBuilder::new("L", 2, "ZN", "ZN")
            .raw_atom(raw_atomname("ZN  "))
            .at(3.0, 0.0, 0.0)
            .elem(Element::Zn)
            .entity("4")
            .build(),
    )
    .unwrap();
    b.push_atom(
        RowBuilder::new("L", 3, "ATP", "PA")
            .raw_atom(raw_atomname("PA  "))
            .at(6.0, 0.0, 0.0)
            .elem(Element::P)
            .entity("4")
            .build(),
    )
    .unwrap();
    let entities = b.finish().unwrap();
    let types: Vec<_> =
        entities.iter().map(MoleculeEntity::molecule_type).collect();
    assert!(types.contains(&MoleculeType::Ion));
    assert!(types.contains(&MoleculeType::Cofactor));
    assert!(types.contains(&MoleculeType::Water));
}

#[test]
fn hint_water_emits_single_bulk() {
    let mut b = EntityBuilder::new();
    b.register_entity("5", ExpectedEntityType::Water);
    for i in 1..=3 {
        #[allow(clippy::cast_precision_loss)]
        b.push_atom(
            RowBuilder::new("W", i, "HOH", "O")
                .at(i as f32, 0.0, 0.0)
                .elem(Element::O)
                .entity("5")
                .build(),
        )
        .unwrap();
    }
    let entities = b.finish().unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].molecule_type(), MoleculeType::Water);
    let bulk = entities[0].as_bulk().unwrap();
    assert_eq!(bulk.molecule_count, 3);
}

#[test]
fn hint_unknown_falls_back_to_pdb_heuristic() {
    let mut b = EntityBuilder::new();
    // No register_entity call -> label_entity_id resolves to Unknown.
    push_protein_residue(&mut b, "A", 1, "ALA", 0.0, Some("missing"));
    push_protein_residue(&mut b, "A", 2, "SEP", 3.8, Some("missing"));
    let entities = b.finish().unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].molecule_type(), MoleculeType::Protein);
    assert_eq!(entities[0].as_protein().unwrap().residues.len(), 2);
}

// AltLoc dedup

fn solo_chain_residue_atom_count(
    rows: impl IntoIterator<Item = AtomRow>,
) -> usize {
    let mut b = EntityBuilder::new();
    for r in rows {
        b.push_atom(r).unwrap();
    }
    let entities = b.finish().unwrap();
    entities[0].atom_count()
}

#[test]
fn altloc_higher_occupancy_wins() {
    let row_a = RowBuilder::new("A", 1, "ALA", "CB")
        .at(1.0, 0.0, 0.0)
        .elem(Element::C)
        .alt(b'A', 0.6)
        .build();
    let row_b = RowBuilder::new("A", 1, "ALA", "CB")
        .at(2.0, 0.0, 0.0)
        .elem(Element::C)
        .alt(b'B', 0.4)
        .build();
    // Need backbone too so the residue survives.
    let mut rows = vec![row_a, row_b];
    for (name, el, x) in [
        ("N", Element::N, 0.0),
        ("CA", Element::C, 1.0),
        ("C", Element::C, 2.0),
        ("O", Element::O, 3.0),
    ] {
        rows.push(
            RowBuilder::new("A", 1, "ALA", name)
                .at(x, 0.0, 0.0)
                .elem(el)
                .build(),
        );
    }

    let mut b = EntityBuilder::new();
    for r in rows {
        b.push_atom(r).unwrap();
    }
    let entities = b.finish().unwrap();
    let protein = entities[0].as_protein().unwrap();
    let cb = protein
        .atoms
        .iter()
        .find(|a| std::str::from_utf8(&a.name).unwrap().trim() == "CB")
        .unwrap();
    assert_eq!(cb.position.x, 1.0, "altloc A (occ=0.6) should win");
}

#[test]
fn altloc_alphabetic_tiebreak() {
    let mut b = EntityBuilder::new();
    for (alt, x) in [(b'A', 1.0), (b'B', 2.0)] {
        b.push_atom(
            RowBuilder::new("X", 1, "LIG", "C1")
                .raw_atom(raw_atomname("C1  "))
                .at(x, 0.0, 0.0)
                .elem(Element::C)
                .alt(alt, 0.5)
                .build(),
        )
        .unwrap();
    }
    let entities = b.finish().unwrap();
    let sm = entities[0].as_small_molecule().unwrap();
    assert_eq!(sm.atoms.len(), 1);
    assert_eq!(sm.atoms[0].position.x, 1.0, "A wins alphabetic tiebreak");
}

#[test]
fn altloc_blank_arrives_first_then_non_blank() {
    let count = solo_chain_residue_atom_count(vec![
        RowBuilder::new("X", 1, "LIG", "C1")
            .raw_atom(raw_atomname("C1  "))
            .at(0.0, 0.0, 0.0)
            .elem(Element::C)
            .build(),
        RowBuilder::new("X", 1, "LIG", "C1")
            .raw_atom(raw_atomname("C1  "))
            .at(9.0, 0.0, 0.0)
            .elem(Element::C)
            .alt(b'A', 1.0)
            .build(),
    ]);
    assert_eq!(count, 1, "blank atom is the kept one");
}

#[test]
fn altloc_blank_arrives_after_non_blank_replaces_it() {
    let mut b = EntityBuilder::new();
    b.push_atom(
        RowBuilder::new("X", 1, "LIG", "C1")
            .raw_atom(raw_atomname("C1  "))
            .at(9.0, 0.0, 0.0)
            .elem(Element::C)
            .alt(b'A', 1.0)
            .build(),
    )
    .unwrap();
    b.push_atom(
        RowBuilder::new("X", 1, "LIG", "C1")
            .raw_atom(raw_atomname("C1  "))
            .at(0.0, 0.0, 0.0)
            .elem(Element::C)
            .build(),
    )
    .unwrap();
    let entities = b.finish().unwrap();
    let sm = entities[0].as_small_molecule().unwrap();
    assert_eq!(sm.atoms.len(), 1);
    assert_eq!(sm.atoms[0].position.x, 0.0, "blank replaces alt");
}

#[test]
fn altloc_same_atom_name_different_residues_is_independent() {
    let mut b = EntityBuilder::new();
    // MSE residues so the missing-atom completion pass is a no-op; this
    // test fixes the backbone-only atom count and is about altloc
    // independence across residues, not chemistry completion.
    push_protein_residue(&mut b, "A", 1, "MSE", 0.0, None);
    push_protein_residue(&mut b, "A", 2, "MSE", 3.8, None);
    let entities = b.finish().unwrap();
    let protein = entities[0].as_protein().unwrap();
    // 4 backbone atoms x 2 residues = 8 atoms total.
    assert_eq!(protein.atoms.len(), 8);
}

#[test]
fn altloc_numeric_alphabetic_tiebreak() {
    let mut b = EntityBuilder::new();
    for (alt, x) in [(b'2', 2.0), (b'1', 1.0)] {
        b.push_atom(
            RowBuilder::new("X", 1, "LIG", "C1")
                .raw_atom(raw_atomname("C1  "))
                .at(x, 0.0, 0.0)
                .elem(Element::C)
                .alt(alt, 0.5)
                .build(),
        )
        .unwrap();
    }
    let entities = b.finish().unwrap();
    let sm = entities[0].as_small_molecule().unwrap();
    assert_eq!(sm.atoms[0].position.x, 1.0, "'1' < '2' alphabetic");
}

// Insertion codes

#[test]
fn insertion_codes_distinguish_residues() {
    let mut b = EntityBuilder::new();
    // Two residues sharing label_seq_id but with different ins_codes.
    b.push_atom(
        RowBuilder::new("A", 100, "GOL", "C1")
            .raw_atom(raw_atomname("C1  "))
            .at(0.0, 0.0, 0.0)
            .elem(Element::C)
            .build(),
    )
    .unwrap();
    b.push_atom(
        RowBuilder::new("A", 100, "GOL", "C1")
            .raw_atom(raw_atomname("C1  "))
            .at(5.0, 0.0, 0.0)
            .elem(Element::C)
            .ins(b'A')
            .build(),
    )
    .unwrap();
    let entities = b.finish().unwrap();
    // Both residues are GOL -> Solvent -> bulked into one BulkEntity
    // with residue_count=2.
    assert_eq!(entities.len(), 1);
    let bulk = entities[0].as_bulk().unwrap();
    assert_eq!(bulk.molecule_count, 2);
}

// Error paths

#[test]
fn over_ninety_chains_all_build() {
    // The single-byte chain cap is gone: 120 distinct protein chains all
    // build, each carrying its own label_asym_id verbatim.
    let mut b = EntityBuilder::new();
    for i in 0..120 {
        push_protein_residue(&mut b, &format!("CH{i}"), 1, "ALA", 0.0, None);
    }
    let entities = b.finish().unwrap();
    let chains: Vec<&str> = entities
        .iter()
        .filter_map(|e| e.pdb_chain_id())
        .collect();
    assert_eq!(chains.len(), 120);
    assert!(chains.contains(&"CH0"));
    assert!(chains.contains(&"CH119"));
}

#[test]
fn nan_coordinate_errors() {
    let mut b = EntityBuilder::new();
    let row = RowBuilder::new("A", 1, "ALA", "CA")
        .at(f32::NAN, 0.0, 0.0)
        .elem(Element::C)
        .build();
    let err = b.push_atom(row).unwrap_err();
    let BuildError::InvalidCoordinate {
        axis,
        label_atom_id,
        ..
    } = err;
    assert_eq!(axis, 'x');
    assert_eq!(label_atom_id, "CA");
}

#[test]
fn infinite_z_errors() {
    let mut b = EntityBuilder::new();
    let row = RowBuilder::new("A", 1, "ALA", "CA")
        .at(0.0, 0.0, f32::INFINITY)
        .elem(Element::C)
        .build();
    let err = b.push_atom(row).unwrap_err();
    assert!(matches!(
        err,
        BuildError::InvalidCoordinate { axis: 'z', .. }
    ));
}

// Output stability

fn build_sample(prefix: &str) -> Vec<MoleculeEntity> {
    let mut b = EntityBuilder::new();
    push_protein_residue(&mut b, &format!("{prefix}A"), 1, "ALA", 0.0, None);
    push_protein_residue(&mut b, &format!("{prefix}A"), 2, "GLY", 3.8, None);
    push_dna_residue(&mut b, &format!("{prefix}B"), 1, "DA", 0.0, None);
    b.finish().unwrap()
}

#[test]
fn output_is_stable_across_runs() {
    let lhs = build_sample("");
    let rhs = build_sample("");
    assert_eq!(lhs.len(), rhs.len());
    for (l, r) in lhs.iter().zip(rhs.iter()) {
        assert_eq!(l.molecule_type(), r.molecule_type());
        assert_eq!(l.atom_count(), r.atom_count());
        assert_eq!(l.pdb_chain_id(), r.pdb_chain_id());
    }
}

#[test]
fn chain_order_follows_insertion_order() {
    let mut b = EntityBuilder::new();
    push_dna_residue(&mut b, "B", 1, "DA", 0.0, None);
    push_protein_residue(&mut b, "A", 1, "ALA", 0.0, None);
    push_protein_residue(&mut b, "A", 2, "GLY", 3.8, None);
    let entities = b.finish().unwrap();
    // Insertion order: chain B (DNA) first, chain A (protein) second; each
    // entity carries its own label_asym_id verbatim.
    assert_eq!(entities[0].pdb_chain_id(), Some("B"));
    assert_eq!(entities[0].molecule_type(), MoleculeType::DNA);
    assert_eq!(entities[1].pdb_chain_id(), Some("A"));
    assert_eq!(entities[1].molecule_type(), MoleculeType::Protein);
}

// register_entity conflicts

#[test]
fn register_entity_idempotent_on_same_hint() {
    let mut b = EntityBuilder::new();
    b.register_entity("1", ExpectedEntityType::Protein);
    b.register_entity("1", ExpectedEntityType::Protein);
    push_protein_residue(&mut b, "A", 1, "ALA", 0.0, Some("1"));
    let entities = b.finish().unwrap();
    assert_eq!(entities[0].molecule_type(), MoleculeType::Protein);
}

#[test]
fn register_entity_first_wins_on_conflict() {
    let mut b = EntityBuilder::new();
    b.register_entity("1", ExpectedEntityType::Protein);
    b.register_entity("1", ExpectedEntityType::DNA);
    push_protein_residue(&mut b, "A", 1, "ALA", 0.0, Some("1"));
    let entities = b.finish().unwrap();
    assert_eq!(
        entities[0].molecule_type(),
        MoleculeType::Protein,
        "first registration wins",
    );
}
