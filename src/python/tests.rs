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

    // Zinc ion: a non-polymer entity carries its source chain but no
    // residues.
    let ion = &entities[1];
    assert_eq!(ion.pdb_chain_id(), Some("B"));
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

// `MoleculeEntity::residue_flat_range`: the prefix-sum over prior residues'
// atom counts in flat residue order. The
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

// A two-residue protein on chain A plus a three-carbon ligand on chain B. The
// carbons sit 1.5 A apart in a line, so distance inference yields two bonds
// (C1-C2, C2-C3) and the table exercises a polymer + non-polymer pair.
const ATOM_TABLE_PDB: &str = "\
ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00           N\n\
ATOM      2  CA  ALA A   1      1.458   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C   ALA A   1      2.009   1.420   0.000  1.00  0.00           C\n\
ATOM      4  O   ALA A   1      1.251   2.390   0.000  1.00  0.00           O\n\
ATOM      5  CB  ALA A   1      1.988  -0.773   1.199  1.00  0.00           C\n\
ATOM      6  N   GLY A   2      3.332   1.539   0.000  1.00  0.00           N\n\
ATOM      7  CA  GLY A   2      3.999   2.835   0.000  1.00  0.00           C\n\
ATOM      8  C   GLY A   2      5.508   2.700   0.000  1.00  0.00           C\n\
ATOM      9  O   GLY A   2      6.040   1.593   0.000  1.00  0.00           O\n\
HETATM   10  C1  LIG B   1      0.000   0.000   5.000  1.00  0.00           C\n\
HETATM   11  C2  LIG B   1      1.500   0.000   5.000  1.00  0.00           C\n\
HETATM   12  C3  LIG B   1      3.000   0.000   5.000  1.00  0.00           C\n\
END\n";

fn atom_table_bytes() -> Vec<u8> {
    use crate::ops::wire::serialize_assembly;
    let entities =
        pdb_str_to_entities(ATOM_TABLE_PDB).expect("parse protein+ligand PDB");
    serialize_assembly(&Assembly::new(entities)).expect("serialize fixture")
}

// `PyAtomTable.from_assembly_bytes(...)` then feeding every egress getter back
// into `from_columns(...)` reproduces the original wire bytes exactly: the
// table is biotite-free and bidirectional, and the canonical fixture makes the
// entity partition idempotent so byte identity (not just column identity)
// holds.
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "feeds every egress column back through from_columns"
)]
fn py_atom_table_columns_round_trip_to_identical_bytes() {
    use pyo3::types::{PyAnyMethods, PyBytes};

    let bytes = atom_table_bytes();
    pyo3::Python::attach(|py| {
        let cls = py.get_type::<super::PyAtomTable>();
        let table = cls
            .call_method1("from_assembly_bytes", (PyBytes::new(py, &bytes),))
            .expect("from_assembly_bytes");

        let col = |name: &str| table.getattr(name).expect("getter");
        let rebuilt = cls
            .call_method1(
                "from_columns",
                (
                    col("coords"),
                    col("atom_names"),
                    col("elements"),
                    col("res_ids"),
                    col("res_names"),
                    col("chain_ids"),
                    col("occupancies"),
                    col("b_factors"),
                    col("mol_types"),
                    py.None(), // chain_types: exercise the mol_types tier
                    col("entity_ids"),
                    col("observed_mask"),
                ),
            )
            .expect("from_columns");

        let round_tripped: Vec<u8> = rebuilt
            .call_method0("to_assembly_bytes")
            .expect("to_assembly_bytes")
            .extract()
            .expect("bytes");
        assert_eq!(round_tripped, bytes, "column round-trip is byte-identical");
    });
}

// The egress getters carry the dtypes/shapes a Python caller relies on, the
// `'S4'`/`'S3'` name arrays hold trailing-space-trimmed values, and `bonds()`
// returns the distance-inferred ligand bonds as flat-index triples.
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "dtype/itemsize and value assertions per egress column"
)]
fn py_atom_table_getter_dtypes_and_bonds() {
    use pyo3::types::{PyAnyMethods, PyBytes};

    let bytes = atom_table_bytes();
    pyo3::Python::attach(|py| {
        let cls = py.get_type::<super::PyAtomTable>();
        let table = cls
            .call_method1("from_assembly_bytes", (PyBytes::new(py, &bytes),))
            .expect("from_assembly_bytes");

        let n: usize = table.len().expect("__len__");

        let dtype = |name: &str, attr: &str| {
            table
                .getattr(name)
                .expect("getter")
                .getattr("dtype")
                .expect("dtype")
                .getattr(attr)
                .expect("dtype attr")
        };
        let kind =
            |name: &str| dtype(name, "kind").extract::<String>().expect("kind");
        let itemsize = |name: &str| {
            dtype(name, "itemsize")
                .extract::<usize>()
                .expect("itemsize")
        };

        // coords: float32, shape (N, 3).
        assert_eq!(kind("coords"), "f");
        assert_eq!(itemsize("coords"), 4);
        let shape: (usize, usize) = table
            .getattr("coords")
            .unwrap()
            .getattr("shape")
            .unwrap()
            .extract()
            .expect("coords 2-D shape");
        assert_eq!(shape, (n, 3));

        assert_eq!(kind("res_ids"), "i");
        assert_eq!(itemsize("res_ids"), 4);
        assert_eq!(kind("entity_ids"), "i");
        assert_eq!(kind("occupancies"), "f");
        assert_eq!(kind("observed_mask"), "b");

        // atom_names / res_names: numpy 'S4' / 'S3', trailing-space-trimmed.
        assert_eq!(kind("atom_names"), "S");
        assert_eq!(itemsize("atom_names"), 4);
        assert_eq!(kind("res_names"), "S");
        assert_eq!(itemsize("res_names"), 3);

        let atom_name1: Vec<u8> = table
            .getattr("atom_names")
            .unwrap()
            .get_item(1)
            .unwrap()
            .extract()
            .expect("atom_names[1] as bytes");
        assert_eq!(atom_name1, b"CA", "second atom is the trimmed CA");
        let res_name0: Vec<u8> = table
            .getattr("res_names")
            .unwrap()
            .get_item(0)
            .unwrap()
            .extract()
            .expect("res_names[0] as bytes");
        assert_eq!(res_name0, b"ALA", "first residue name trimmed to ALA");

        // bonds(): the ligand's two inferred C-C bonds, flat-index triples
        // offset past the protein's atoms.
        let bonds: Vec<(usize, usize, u8)> = table
            .call_method0("bonds")
            .expect("bonds")
            .extract()
            .expect("bond triples");
        assert_eq!(bonds.len(), 2, "two inferred ligand bonds");
        for (i, j, order) in &bonds {
            assert!(*i >= 9 && *j >= 9, "ligand bonds index past the protein");
            assert_eq!(*order, 1, "single bonds");
        }
    });
}

// The `coords` egress carries the fixture's atom positions verbatim: the wire
// round-trip and the columnar flatten preserve each f32 coordinate. The first
// parsed atom (N of ALA 1) sits at the origin, and the three ligand carbons are
// the only atoms in the z = 5 plane, at x = 0 / 1.5 / 3.0 (all exactly
// representable in f32, so no ULP slack). Heavy-atom completion adds protein
// atoms, so the ligand carbons are matched by their plane rather than by a
// fixed flat index.
#[test]
fn py_atom_table_coords_preserve_fixture_values() {
    use pyo3::types::{PyAnyMethods, PyBytes};

    let bytes = atom_table_bytes();
    pyo3::Python::attach(|py| {
        let cls = py.get_type::<super::PyAtomTable>();
        let table = cls
            .call_method1("from_assembly_bytes", (PyBytes::new(py, &bytes),))
            .expect("from_assembly_bytes");

        let coords: Vec<Vec<f32>> = table
            .getattr("coords")
            .expect("coords")
            .extract()
            .expect("coords as (N, 3) f32 rows");

        assert_eq!(
            coords[0],
            vec![0.0, 0.0, 0.0],
            "first parsed atom (N of ALA 1) at the origin"
        );

        let mut ligand: Vec<Vec<f32>> =
            coords.iter().filter(|p| p[2] == 5.0).cloned().collect();
        ligand.sort_by(|a, b| a[0].total_cmp(&b[0]));
        assert_eq!(
            ligand,
            vec![
                vec![0.0, 0.0, 5.0],
                vec![1.5, 0.0, 5.0],
                vec![3.0, 0.0, 5.0],
            ],
            "the three ligand carbons keep their z = 5 coordinates"
        );
    });
}

// With no `mol_types`/`chain_types`, `from_columns` falls back to residue-name
// classification: a protein backbone classifies as protein (not ligand), and a
// ligand residue name classifies as ligand. Guards the annotation-free ingest
// path that hand-built frames / boltz outputs rely on.
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "builds two fixtures and asserts per-column classification"
)]
fn py_atom_table_from_columns_classifies_without_annotations() {
    use pyo3::types::PyAnyMethods;

    pyo3::Python::attach(|py| {
        let cls = py.get_type::<super::PyAtomTable>();
        let numpy = py.import("numpy").expect("numpy");
        let f32_ty = numpy.getattr("float32").expect("float32");
        let i32_ty = numpy.getattr("int32").expect("int32");

        // Build a table from the six required columns only (no mol_types,
        // chain_types, occupancies, b_factors, entity_ids, observed).
        let build = |res_names: Vec<String>,
                     atom_names: Vec<String>,
                     elements: Vec<String>,
                     res_ids: Vec<i32>,
                     chain_ids: Vec<String>| {
            let n = res_names.len();
            let coords = numpy
                .call_method1("zeros", ((n, 3),))
                .unwrap()
                .call_method1("astype", (&f32_ty,))
                .unwrap();
            let res_ids = numpy
                .call_method1("array", (res_ids,))
                .unwrap()
                .call_method1("astype", (&i32_ty,))
                .unwrap();
            cls.call_method1(
                "from_columns",
                (coords, atom_names, elements, res_ids, res_names, chain_ids),
            )
            .expect("from_columns with required columns only")
        };

        let mol_types = |table: &pyo3::Bound<'_, pyo3::PyAny>| -> Vec<String> {
            table
                .getattr("mol_types")
                .unwrap()
                .extract()
                .expect("mol_types as str list")
        };

        let protein = build(
            vec!["ALA".into(), "ALA".into(), "ALA".into(), "GLY".into()],
            vec!["N".into(), "CA".into(), "C".into(), "N".into()],
            vec!["N".into(), "C".into(), "C".into(), "N".into()],
            vec![1, 1, 1, 2],
            vec!["A".into(); 4],
        );
        let protein_mt = mol_types(&protein);
        assert!(
            protein_mt.iter().all(|s| s == "protein"),
            "annotation-free protein backbone classifies as protein, got \
             {protein_mt:?}"
        );

        let ligand = build(
            vec!["LIG".into(), "LIG".into()],
            vec!["C1".into(), "C2".into()],
            vec!["C".into(), "C".into()],
            vec![1, 1],
            vec!["B".into(); 2],
        );
        let ligand_mt = mol_types(&ligand);
        assert!(
            ligand_mt.iter().all(|s| s == "ligand"),
            "ligand residue name classifies as ligand, got {ligand_mt:?}"
        );
    });
}
