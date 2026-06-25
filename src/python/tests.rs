//! Tests for the `python` module. A submodule split out to keep the primary
//! module file under the 800-line source cap enforced by `just file-lengths`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

use glam::Vec3;

use super::walk::{entity_kind_str, molecule_type_str};
use super::{coords_to_vec3, make_entity_id, PyEditList};
use crate::adapters::pdb::pdb_str_to_entities;
use crate::assembly::Assembly;

// `coords_to_vec3` takes already-extracted tuples; well-formed mapping
// is testable without the interpreter.
#[test]
fn coords_to_vec3_maps_tuples_in_order() {
    let got = coords_to_vec3(vec![(1.0, 2.0, 3.0), (-4.0, 5.5, 6.25)]);
    assert_eq!(
        got,
        vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(-4.0, 5.5, 6.25)]
    );
}

// Edit-application path: `push_set_entity_coords` runs `coords_to_vec3`
// and stamps the `EntityId`; read back through the public accessor.
#[test]
fn push_set_entity_coords_records_edit() {
    let mut list = PyEditList::new();
    list.push_set_entity_coords(7, vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]);

    assert_eq!(list.__len__(), 1);
    assert_eq!(list.kind_at(0).expect("kind at 0"), 1);

    let view = list
        .set_entity_coords_at(0)
        .expect("edit at index 0 is a SetEntityCoords view");
    assert_eq!(view.entity_id, make_entity_id(7).raw());
    assert_eq!(view.coords, vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]);
}

// Wrong view kind returns a PyErr, never a panic (no interpreter); the
// concrete `ValueError` type is asserted in `wrong_kind_is_value_error`.
#[test]
fn wrong_kind_errors_cleanly() {
    let mut list = PyEditList::new();
    list.push_set_variants(1, 0, Vec::new());
    assert!(list.set_entity_coords_at(0).is_err(), "not SetEntityCoords");
}

// Reading past the end returns a PyErr, never a panic (no interpreter).
#[test]
fn out_of_range_errors_cleanly() {
    let list = PyEditList::new();
    assert!(list.set_entity_coords_at(0).is_err(), "no edit at index 0");
}

// A minimal two-residue, two-entity PDB: ALA-GLY on chain A plus a single
// zinc ion. The object-graph getters (`Entity` / `Residue` `#[pymethods]`)
// need a live interpreter to dispatch and are owed a runtime test; the tests
// below exercise the same molex accessors and string renderers the objects
// wrap, plus the `PyResidue` read-through getters, which run without libpython.
const WALK_PDB: &str = "\
ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00           N\n\
ATOM      2  CA  ALA A   1      1.458   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C   ALA A   1      2.009   1.420   0.000  1.00  0.00           C\n\
ATOM      4  O   ALA A   1      1.251   2.390   0.000  1.00  0.00           O\n\
ATOM      5  CB  ALA A   1      1.988  -0.773   1.199  1.00  0.00           C\n\
ATOM      6  N   GLY A   2      3.332   1.539   0.000  1.00  0.00           N\n\
ATOM      7  CA  GLY A   2      3.999   2.835   0.000  1.00  0.00           C\n\
ATOM      8  C   GLY A   2      5.508   2.700   0.000  1.00  0.00           C\n\
ATOM      9  O   GLY A   2      6.040   1.593   0.000  1.00  0.00           O\n\
HETATM   10 ZN    ZN B   1     10.000  10.000  10.000  1.00  0.00          ZN\n\
END\n";

// The parse path the `Assembly.from_pdb` staticmethod wraps: raw ingest,
// SS-free `Assembly::new`. The object getters read straight through these
// `MoleculeEntity` accessors.
#[test]
fn parsed_entities_expose_object_accessors() {
    let entities = pdb_str_to_entities(WALK_PDB).expect("parse minimal PDB");
    let asm = Assembly::new(entities);
    let entities = asm.entities();
    assert_eq!(entities.len(), 2);

    // Protein chain A: kind / molecule-type / chain / residue count.
    let prot = &entities[0];
    assert_eq!(entity_kind_str(prot.entity_kind()), "Protein");
    assert_eq!(molecule_type_str(prot.molecule_type()), "Protein");
    assert_eq!(prot.pdb_chain_id(), Some("A"));
    assert_eq!(prot.residues().map_or(0, <[_]>::len), 2);
    assert!(prot.atom_count() >= 9);

    // Zinc ion: a non-polymer entity has no chain id and no residues.
    let ion = &entities[1];
    assert_eq!(ion.pdb_chain_id(), None);
    assert!(ion.residues().is_none());
}

// The `Residue` handle reads through the parent entity's residue list: it
// trims the name, applies the auth/label `seq_id` fallback, and renders the
// (blank-here) insertion code.
#[test]
fn residue_reads_through_trims_name_and_keeps_seq_ids() {
    use super::entity::PyResidue;

    let entities = pdb_str_to_entities(WALK_PDB).expect("parse minimal PDB");
    let asm = Assembly::new(entities);
    let prot = std::sync::Arc::clone(&asm.entities()[0]);

    let ala =
        PyResidue::new(std::sync::Arc::clone(&prot), 0).expect("residue 0");
    assert_eq!(ala.name(), "ALA");
    assert_eq!(ala.seq_id(), 1);
    assert_eq!(ala.label_seq_id(), 1);
    assert_eq!(ala.ins_code(), None);

    let gly = PyResidue::new(prot, 1).expect("residue 1");
    assert_eq!(gly.name(), "GLY");
    assert_eq!(gly.seq_id(), 2);
}

// `Assembly::secondary_structure` (what `Assembly.secondary_structure`
// delegates to): one string per entity in declaration order, all-`C` per
// residue without `recompute_ss`, and an empty string for the non-polymer
// ion. Aligns 1:1 with `entities()`, unlike `sequence()` which drops it.
#[test]
fn secondary_structure_aligns_with_entities_and_is_coil_without_recompute() {
    let entities = pdb_str_to_entities(WALK_PDB).expect("parse minimal PDB");
    let asm = Assembly::new(entities);

    let ss = asm.secondary_structure();
    assert_eq!(ss.len(), asm.entities().len(), "one entry per entity");
    let n_res = asm.entities()[0].residues().map_or(0, <[_]>::len);
    assert_eq!(ss[0], "C".repeat(n_res), "all-coil before recompute_ss");
    assert_eq!(ss[1], "", "non-polymer ion contributes an empty string");
}

// `MoleculeEntity::residue_flat_range` (what `Residue.to_arrays` slices with):
// the prefix-sum over prior residues' atom counts in flat residue order. The
// ranges tile the protein's atoms contiguously from 0; a non-polymer or
// out-of-range index yields `0..0`.
#[test]
fn residue_flat_range_tiles_atoms_in_residue_order() {
    let entities = pdb_str_to_entities(WALK_PDB).expect("parse minimal PDB");
    let asm = Assembly::new(entities);
    let prot = &asm.entities()[0];
    let n_res = prot.residues().map_or(0, <[_]>::len);
    assert_eq!(n_res, 2, "ALA-GLY dipeptide");

    let r0 = prot.residue_flat_range(0);
    let r1 = prot.residue_flat_range(1);
    assert_eq!(r0.start, 0, "first residue starts the flat order");
    assert_eq!(r1.start, r0.end, "residue 1 follows residue 0 contiguously");
    assert_eq!(r1.end, prot.atom_count(), "ranges cover every atom");

    // Out-of-range and non-polymer both yield an empty range.
    assert_eq!(prot.residue_flat_range(n_res), 0..0);
    assert_eq!(asm.entities()[1].residue_flat_range(0), 0..0);
}

// GIL-dependent tests below need a live interpreter. The
// `extension-module` build links no libpython, so the test binary
// fails to link and no test here runs in that environment; they cover
// the boundary wherever libpython is linked.

// Malformed coords fail at pyo3's `FromPyObject` boundary: a wrong
// inner length fails extraction with a PyErr before `coords_to_vec3`.
#[test]
fn extract_malformed_coords_is_pyerr_not_panic() {
    use pyo3::types::{PyAnyMethods, PyList};
    pyo3::Python::attach(|py| {
        let bad = PyList::new(py, [(1.0_f64, 2.0_f64)])
            .expect("build py list of one 2-tuple");
        let res: Result<Vec<(f32, f32, f32)>, pyo3::PyErr> = bad.extract();
        assert!(res.is_err(), "wrong inner length must not extract");
    });
}

// Confirm the wrong-kind read produces a ValueError specifically.
#[test]
fn wrong_kind_is_value_error() {
    use pyo3::types::{PyStringMethods, PyTypeMethods};
    let mut list = PyEditList::new();
    list.push_set_variants(1, 0, Vec::new());
    let err = list.set_entity_coords_at(0).err().expect("must error");
    pyo3::Python::attach(|py| {
        let bound = err.get_type(py).name().expect("read type name");
        let name = bound.to_str().expect("utf-8 type name");
        assert!(name.ends_with("ValueError"), "got {name}");
    });
}
