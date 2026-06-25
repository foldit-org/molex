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

// The native numpy egress (`entities_to_arrays`, what `Assembly.to_arrays`
// and `Entity.to_arrays` call) builds its six numeric columns straight from
// owned Rust vecs via `into_pyarray`, with no float64/int64 intermediate or
// `astype` copy. Guard the dtype, shape, AND values of the returned numpy
// arrays directly, so a regression in that path — wrong dtype, wrong shape, or
// wrong/transposed values — is caught at the numpy boundary the Python caller
// actually sees.
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "dtype, shape, and value assertions per numeric column"
)]
fn to_arrays_numeric_columns_have_expected_dtype_shape_and_values() {
    use pyo3::types::PyAnyMethods;

    use crate::adapters::atomworks::columns;
    use crate::adapters::table::AtomTable;

    let entities = pdb_str_to_entities(WALK_PDB).expect("parse minimal PDB");
    let asm = Assembly::new(entities);
    let entities = asm.entities();
    // ALA (5 atoms) + GLY (5, one completed by the protein builder) + ZN (1).
    let total: usize = entities.iter().map(|e| e.atom_count()).sum();
    assert_eq!(total, 11, "fixture atom count");

    // Independent Rust oracle for the values the numpy columns must carry,
    // marshaled through the same flatten + de-vocab the egress uses.
    let table = AtomTable::from_entities(entities);
    let coords_flat = columns::coords_flat(&table, 0..total);
    let oracle_res_ids = columns::res_ids(&table, 0..total);
    let oracle_occ = columns::occupancies(&table, 0..total);

    pyo3::Python::attach(|py| {
        let numpy = py.import("numpy").expect("import numpy");
        let arrays =
            super::arrays::entities_to_arrays(py, entities).expect("to_arrays");
        // Read each column through the `#[pyo3(get)]` attribute surface a
        // Python caller sees, by binding the pyclass instance into the
        // interpreter.
        let arrays = pyo3::Py::new(py, arrays).expect("bind AtomArrays");
        let arrays = arrays.bind(py);
        let col = |name: &str| arrays.getattr(name).expect("column attr");

        let f32_ty = numpy.getattr("float32").expect("numpy.float32");
        let i32_ty = numpy.getattr("int32").expect("numpy.int32");
        // `np.dtype == np.<scalar>` is True for the matching scalar type; this
        // checks the column's element type without depending on the numpy
        // crate's dtype downcast generics.
        let assert_dtype = |name: &str, ty: &pyo3::Bound<'_, pyo3::PyAny>| {
            let got = col(name).getattr("dtype").expect("column dtype");
            assert!(got.eq(ty).expect("compare dtype"), "{name} dtype");
        };
        let f32 = |name: &str| assert_dtype(name, &f32_ty);
        let i32 = |name: &str| assert_dtype(name, &i32_ty);

        // coords: float32, shape (N, 3); every numeric dtype guarded.
        f32("coords");
        for name in ["occupancies", "b_factors"] {
            f32(name);
        }
        for name in ["res_ids", "entity_ids", "chain_types"] {
            i32(name);
        }
        assert_eq!(
            col("coords")
                .getattr("shape")
                .unwrap()
                .extract::<(usize, usize)>()
                .expect("coords shape is 2-D"),
            (total, 3),
            "coords shape must be (N, 3)"
        );

        // VALUES via the `into_pyarray` path, against the Rust oracle (catches
        // a transposed reshape or a value-corrupting move into numpy) plus
        // independent fixture anchors. `tolist` flattens, so coords come back
        // in the same row-major order as `coords_flat`.
        let to_list =
            |name: &str| col(name).call_method0("tolist").expect("tolist");
        let coords_rows: Vec<Vec<f32>> = to_list("coords")
            .extract()
            .expect("coords as nested f32 list");
        let numpy_coords: Vec<f32> =
            coords_rows.into_iter().flatten().collect();
        assert_eq!(numpy_coords, coords_flat, "coords values");
        // ALA's first atom (N) sits at the origin; the trailing ZN ion sits at
        // (10, 10, 10) — anchors that do not lean on the oracle.
        assert_eq!(&coords_flat[..3], &[0.0_f32, 0.0, 0.0], "ALA N at origin");
        assert_eq!(
            &coords_flat[total * 3 - 3..],
            &[10.0_f32, 10.0, 10.0],
            "ZN ion at (10, 10, 10)"
        );

        let res_ids: Vec<i32> =
            to_list("res_ids").extract().expect("res_ids list");
        assert_eq!(res_ids, oracle_res_ids, "res_ids values");
        // ALA atoms carry seq id 1, GLY carry 2, the trailing ZN carries 1.
        assert_eq!(res_ids, vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1], "res ids");

        let occ: Vec<f32> = to_list("occupancies").extract().expect("occ list");
        assert_eq!(occ, oracle_occ, "occupancy values");
        assert_eq!(occ, vec![1.0_f32; total], "fixture occupancy is 1.00");
    });
}
