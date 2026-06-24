//! Round-trip tests proving that `pdb_chain_id`, `auth_seq_id`,
//! `auth_comp_id`, `auth_atom_id`, `ins_code`, and `formal_charge`
//! survive the `AtomRow` -> builder -> entity-type-constructor ->
//! `Vec<MoleculeEntity>` path. `None` on an auth-side row column
//! propagates as `None` on the output (the "default to label" rule is
//! applied downstream at read time).

#![allow(clippy::unwrap_used, clippy::float_cmp, clippy::too_many_arguments)]

use super::tests::{
    atomname, push_protein_residue, raw_atomname, resname, RowBuilder,
};
use super::*;

/// Push four backbone rows for a single ALA residue with overrides
/// applied uniformly through `decorate`. `decorate` runs per-row so the
/// caller can drop in `auth_chain` / `auth_seq` / etc. on every row.
fn push_protein_residue_with<F>(
    b: &mut EntityBuilder,
    chain: &str,
    seq: i32,
    comp: &str,
    base_x: f32,
    decorate: F,
) where
    F: Fn(RowBuilder) -> RowBuilder,
{
    let names = ["N", "CA", "C", "O"];
    let elements = [Element::N, Element::C, Element::C, Element::O];
    for ((dx, name), el) in
        [0.0, 1.45, 2.4, 2.8].into_iter().zip(names).zip(elements)
    {
        let row = RowBuilder::new(chain, seq, comp, name)
            .at(base_x + dx, 0.0, 0.0)
            .elem(el);
        b.push_atom(decorate(row).build()).unwrap();
    }
}

#[test]
fn label_chain_id_round_trips() {
    // The entity carries its `label_asym_id` verbatim as `pdb_chain_id`;
    // a distinct author chain string on the rows does not displace it.
    let mut b = EntityBuilder::new();
    push_protein_residue_with(&mut b, "A", 1, "ALA", 0.0, |r| {
        r.auth_chain("H")
    });
    let entities = b.finish().unwrap();
    let protein = entities[0].as_protein().unwrap();
    assert_eq!(protein.pdb_chain_id, "A", "label chain carried verbatim");
}

#[test]
fn auth_seq_id_round_trips() {
    let mut b = EntityBuilder::new();
    push_protein_residue_with(&mut b, "A", 1, "ALA", 0.0, |r| r.auth_seq(100));
    let entities = b.finish().unwrap();
    let residues = &entities[0].as_protein().unwrap().residues;
    assert_eq!(residues.len(), 1);
    assert_eq!(residues[0].label_seq_id, 1);
    assert_eq!(residues[0].auth_seq_id, Some(100));
}

#[test]
fn auth_comp_id_round_trips() {
    let mut b = EntityBuilder::new();
    push_protein_residue_with(&mut b, "A", 1, "ALA", 0.0, |r| {
        r.auth_comp("XAA")
    });
    let entities = b.finish().unwrap();
    let residues = &entities[0].as_protein().unwrap().residues;
    assert_eq!(residues[0].name, resname("ALA"));
    assert_eq!(residues[0].auth_comp_id, Some(resname("XAA")));
}

#[test]
fn auth_atom_id_round_trips() {
    // Sidechain atom (CB) carries an author-side rename. Protein
    // canonicalization keys on backbone names (N, CA, C, O), so leaving
    // the backbone label-only keeps the residue alive while the auth
    // name reaches the output through the sidechain slot.
    let mut b = EntityBuilder::new();
    push_protein_residue(&mut b, "A", 1, "ALA", 0.0, None);
    b.push_atom(
        RowBuilder::new("A", 1, "ALA", "CB")
            .at(1.5, -1.0, 0.0)
            .elem(Element::C)
            .auth_atom(raw_atomname("QB  "))
            .build(),
    )
    .unwrap();
    let entities = b.finish().unwrap();
    let protein = entities[0].as_protein().unwrap();
    assert_eq!(protein.residues.len(), 1);
    let cb_atom = protein
        .atoms
        .iter()
        .find(|a| a.name == raw_atomname("QB  "))
        .unwrap();
    assert_eq!(cb_atom.element, Element::C);
}

#[test]
fn ins_code_round_trips() {
    let mut b = EntityBuilder::new();
    push_protein_residue_with(&mut b, "A", 100, "ALA", 0.0, |r| r.ins(b'A'));
    let entities = b.finish().unwrap();
    let residues = &entities[0].as_protein().unwrap().residues;
    assert_eq!(residues[0].ins_code, Some(b'A'));
}

#[test]
fn formal_charge_round_trips() {
    let mut b = EntityBuilder::new();
    // Charge only on N to prove per-atom propagation.
    let rows = [
        ("N", Element::N, 0.0_f32, 2_i8),
        ("CA", Element::C, 1.45, 0),
        ("C", Element::C, 2.4, 0),
        ("O", Element::O, 2.8, 0),
    ];
    for (name, el, x, charge) in rows {
        b.push_atom(
            RowBuilder::new("A", 1, "ALA", name)
                .at(x, 0.0, 0.0)
                .elem(el)
                .formal_charge(charge)
                .build(),
        )
        .unwrap();
    }
    let entities = b.finish().unwrap();
    let protein = entities[0].as_protein().unwrap();
    let n_atom = &protein.atoms[protein.residues[0].atom_range.start];
    assert_eq!(n_atom.name, atomname("N"));
    assert_eq!(n_atom.formal_charge, 2);
    // Sibling atoms stay neutral.
    let ca_atom = &protein.atoms[protein.residues[0].atom_range.start + 1];
    assert_eq!(ca_atom.formal_charge, 0);
}

#[test]
fn auth_none_defaults_to_label() {
    let mut b = EntityBuilder::new();
    // All auth_* fields left None on every row.
    push_protein_residue(&mut b, "A", 1, "ALA", 0.0, None);
    let entities = b.finish().unwrap();
    let protein = entities[0].as_protein().unwrap();
    let residue = &protein.residues[0];
    assert_eq!(residue.auth_seq_id, None);
    assert_eq!(residue.auth_comp_id, None);
    assert_eq!(residue.ins_code, None);
    for idx in residue.atom_range.clone() {
        assert_eq!(
            protein.atoms[idx].formal_charge, 0,
            "default formal_charge is 0",
        );
    }
}
