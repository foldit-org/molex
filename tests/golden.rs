//! Golden correctness tests: each asserts a molex-computed value against a
//! constant captured from molex's *own* output (never a foreign library), so
//! every optimization-target bench has a paired correctness guard. A speedup
//! that perturbs a result trips the relevant test here.
//!
//! Constants were bootstrapped by running the call once and baking the value
//! in with a tight tolerance; the inline comment on each records the params it
//! was computed under. Fixtures are the committed `benches/data/*` set only;
//! the synthetic cases (RMSD, disulfide) need no fixture.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::unreadable_literal
)]

use std::path::PathBuf;

use glam::{Mat3, Vec3};
use molex::adapters::cif::{mmcif_str_to_entities, mmcif_str_to_entities_with};
use molex::analysis::sasa::DEFAULT_N_POINTS;
use molex::analysis::{assembly_sasa, ContactLevel};
use molex::entity::molecule::atom::Atom;
use molex::entity::molecule::id::EntityIdAllocator;
use molex::entity::molecule::protein::ProteinEntity;
use molex::ops::transform::rmsd;
use molex::{
    detect_disulfides, Assembly, BondOrder, Completion, Element,
    MoleculeEntity, Residue,
};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches/data")
        .join(name)
}

fn ubq_assembly(level: Completion) -> Assembly {
    let cif = std::fs::read_to_string(data("1ubq.cif")).unwrap();
    Assembly::new(mmcif_str_to_entities_with(&cif, level).unwrap())
}

/// Total protein SASA of 1UBQ.
///
/// Params: Bondi vdW radii (`Element::vdw_radius`) + 960 golden-spiral points
/// (`DEFAULT_N_POINTS`), probe 1.4 A, protein atoms only. Golden against
/// molex's own number — NOT freesasa (Lee-Richards + ProtOr radii) or a
/// BioPython 100-point figure; those diverge by construction.
#[test]
fn uc7_sasa_1ubq_total() {
    let cif = std::fs::read_to_string(data("1ubq.cif")).unwrap();
    let asm = Assembly::new(mmcif_str_to_entities(&cif).unwrap());

    let total = assembly_sasa(&asm, 1.4, DEFAULT_N_POINTS);

    // Captured 2026-06-24: 4872.2393 A^2. Tolerance ~0.5%.
    let expected = 4872.24_f32;
    let rel_err = (total - expected).abs() / expected;
    assert!(
        rel_err < 0.005,
        "1UBQ SASA = {total}, expected ~{expected} (rel err {rel_err})"
    );
}

// --- UC-9 · phi/psi torsions --------------------------------------------

/// φ/ψ of interior 1UBQ residues, in degrees in (-180, 180] (IUPAC sign),
/// with `None` at the termini.
///
/// Params: `ProteinEntity::phi_psi()` over the Heavy-completed 1UBQ protein
/// entity; degree values, not radians.
#[test]
fn uc9_phi_psi_1ubq() {
    let asm = ubq_assembly(Completion::Heavy);

    let angles = asm
        .entities()
        .iter()
        .find_map(|e| e.as_protein())
        .map(ProteinEntity::phi_psi)
        .expect("1UBQ has a protein entity");

    // 1UBQ has 76 residues.
    assert_eq!(angles.len(), 76);

    // Termini: first has no phi (no preceding C), last has no psi.
    assert_eq!(angles[0].0, None, "first residue phi must be None");
    assert_eq!(angles[75].1, None, "last residue psi must be None");

    // Interior residues, captured 2026-06-24 (degrees). Tolerance 1e-2.
    let tol = 1e-2_f32;
    let check = |idx: usize, phi_exp: f32, psi_exp: f32| {
        let (phi, psi) = angles[idx];
        let phi = phi.expect("interior residue must have phi");
        let psi = psi.expect("interior residue must have psi");
        assert!(
            (phi - phi_exp).abs() < tol,
            "residue {idx} phi = {phi}, expected {phi_exp}"
        );
        assert!(
            (psi - psi_exp).abs() < tol,
            "residue {idx} psi = {psi}, expected {psi_exp}"
        );
    };
    check(1, -91.02025, 138.26411);
    check(2, -131.0989, 163.04655);
    check(7, -73.42523, -6.9358196);
}

/// RMSD is pure geometry (no algorithmic freedom): a known rigid transform
/// must superpose back to ~0, and a known displacement noise yields a known
/// nonzero value. No fixture needed.
///
/// Params: `ops::transform::rmsd` (optimal-superposition RMSD via Kabsch).
#[test]
fn uc10_kabsch_rmsd_synthetic() {
    let reference: Vec<Vec3> = (0..12)
        .map(|i| {
            let t = i as f32 * 0.37;
            Vec3::new(t.cos() * 4.0, t.sin() * 4.0, t * 0.8)
        })
        .collect();

    // Known rigid rotation + translation -> RMSD must recover ~0.
    let rotation = Mat3::from_rotation_z(0.6) * Mat3::from_rotation_y(-0.4);
    let translation = Vec3::new(2.0, -5.0, 1.5);
    let moved: Vec<Vec3> = reference
        .iter()
        .map(|p| rotation * *p + translation)
        .collect();

    let r_rigid = rmsd(&reference, &moved).unwrap();
    assert!(
        r_rigid < 1e-4,
        "rigid-motion RMSD must be ~0, got {r_rigid}"
    );

    // Add deterministic per-point displacement -> known nonzero RMSD.
    let noisy: Vec<Vec3> = moved
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let k = i as f32;
            *p + Vec3::new(
                0.1 * (k * 1.1).sin(),
                0.1 * (k * 0.7).cos(),
                0.1 * (k * 1.9).sin(),
            )
        })
        .collect();

    let r_noisy = rmsd(&reference, &noisy).unwrap();
    // Captured 2026-06-24: 0.118532404. Tolerance 1e-4.
    let expected = 0.118_532_4_f32;
    assert!(
        (r_noisy - expected).abs() < 1e-4,
        "noisy RMSD = {r_noisy}, expected ~{expected}"
    );
}

fn mk_atom(name: [u8; 4], el: Element, pos: Vec3) -> Atom {
    Atom {
        position: pos,
        occupancy: 1.0,
        b_factor: 0.0,
        element: el,
        name,
        formal_charge: 0,
    }
}

/// Minimal CYS residue (N, CA, C, O, CB, SG). `sg_offset` places the SG atom
/// relative to `origin`, so two such residues can be set at a chosen SG-SG
/// distance.
fn cys_residue(origin: Vec3, sg_offset: Vec3) -> (Vec<Atom>, Residue) {
    let atoms = vec![
        mk_atom(*b"N   ", Element::N, origin),
        mk_atom(*b"CA  ", Element::C, origin + Vec3::new(1.0, 0.0, 0.0)),
        mk_atom(*b"C   ", Element::C, origin + Vec3::new(2.0, 0.0, 0.0)),
        mk_atom(*b"O   ", Element::O, origin + Vec3::new(2.0, 1.0, 0.0)),
        mk_atom(*b"CB  ", Element::C, origin + Vec3::new(1.0, -1.0, 0.0)),
        mk_atom(*b"SG  ", Element::S, origin + sg_offset),
    ];
    let residue = Residue {
        name: *b"CYS",
        label_seq_id: 1,
        auth_seq_id: None,
        auth_comp_id: None,
        ins_code: None,
        atom_range: 0..atoms.len(),
        variants: Vec::new(),
    };
    (atoms, residue)
}

/// Two CYS residues with SG atoms 2.03 A apart (inside molex's 1.5-2.5 A,
/// distance-only band; `disulfide.rs`) must yield exactly that one bond,
/// spanning the two entities.
#[test]
fn uc11_detect_disulfides_one_pair() {
    // Chain A SG at the origin; chain B SG 2.03 A away along +x.
    let (mut atoms_a, res_a) = cys_residue(Vec3::ZERO, Vec3::ZERO);
    let (mut atoms_b, res_b) =
        cys_residue(Vec3::new(2.03, 0.0, 0.0), Vec3::ZERO);

    let mut alloc = EntityIdAllocator::new();
    let id_a = alloc.allocate();
    let id_b = alloc.allocate();

    let ent_a = ProteinEntity::new(
        id_a,
        std::mem::take(&mut atoms_a),
        vec![res_a],
        "A".to_owned(),
    );
    let ent_b = ProteinEntity::new(
        id_b,
        std::mem::take(&mut atoms_b),
        vec![res_b],
        "B".to_owned(),
    );
    let entities = vec![
        MoleculeEntity::Protein(ent_a),
        MoleculeEntity::Protein(ent_b),
    ];

    let bonds = detect_disulfides(&entities);
    assert_eq!(bonds.len(), 1, "expected exactly one disulfide");
    assert_ne!(
        bonds[0].a.entity, bonds[0].b.entity,
        "the bond must span the two cysteine entities"
    );
}

fn total_atoms(asm: &Assembly) -> usize {
    asm.entities().iter().map(|e| e.atom_count()).sum()
}

fn count_hydrogens(asm: &Assembly) -> usize {
    asm.entities()
        .iter()
        .flat_map(|e| e.columns().to_atoms())
        .filter(|a| a.element == Element::H)
        .count()
}

/// First N residues of the first protein entity, as `(trimmed name, atom
/// count)` pairs.
fn residue_atom_counts(asm: &Assembly, take: usize) -> Vec<(String, usize)> {
    asm.entities()
        .iter()
        .find_map(|e| e.as_protein())
        .map(|p| {
            p.residues
                .iter()
                .take(take)
                .map(|r| {
                    let name =
                        std::str::from_utf8(&r.name).unwrap().trim().to_owned();
                    (name, r.atom_range.clone().count())
                })
                .collect()
        })
        .expect("1UBQ has a protein entity")
}

/// After `complete(Completion::Heavy)`, standard residues carry their full
/// CCD heavy-atom complement; `to_all_atom()` then adds template hydrogens,
/// so the H count strictly exceeds heavy-only.
///
/// Params: 1UBQ parsed at `Completion::Raw`, then `complete(Heavy)` /
/// `to_all_atom()`. Golden against molex's own template output. (1UBQ is
/// deposited heavy-complete, so these per-residue counts equal the canonical
/// CCD heavy complement for each residue.)
#[test]
fn uc13_completion_heavy_and_all_atom() {
    let raw = ubq_assembly(Completion::Raw);

    let heavy = raw.complete(Completion::Heavy);
    // Captured 2026-06-24. CCD heavy complement per residue:
    // MET 8, GLN 9, ILE 8, PHE 11.
    let heavy_counts = residue_atom_counts(&heavy, 4);
    assert_eq!(
        heavy_counts,
        vec![
            ("MET".to_owned(), 8),
            ("GLN".to_owned(), 9),
            ("ILE".to_owned(), 8),
            ("PHE".to_owned(), 11),
        ]
    );
    assert_eq!(count_hydrogens(&heavy), 0, "Heavy carries no hydrogens");

    let all_atom = raw.to_all_atom();
    let all_atom_counts = residue_atom_counts(&all_atom, 4);
    // Same residues, protonated: MET 18, GLN 16, ILE 18, PHE 19.
    assert_eq!(
        all_atom_counts,
        vec![
            ("MET".to_owned(), 18),
            ("GLN".to_owned(), 16),
            ("ILE".to_owned(), 18),
            ("PHE".to_owned(), 19),
        ]
    );

    // H count strictly grows from heavy-only to all-atom.
    let heavy_h = count_hydrogens(&heavy);
    let all_h = count_hydrogens(&all_atom);
    assert!(
        all_h > heavy_h,
        "all-atom H count ({all_h}) must exceed heavy-only ({heavy_h})"
    );
    // Captured 2026-06-24: all-atom adds 569 hydrogens (660 -> 1229 atoms).
    assert_eq!(all_h, 569);
    assert_eq!(total_atoms(&all_atom), 1229);
}

// --- UC-12 · aggregated covalent bonds ----------------------------------

/// `Assembly::covalent_bonds()` aggregates each entity's stored intra-entity
/// bond topology. 1UBQ at `Completion::Heavy` is one protein chain plus a
/// water bulk; the water carries no topology, so the total equals the
/// protein's bond count and the first bonds are residue-0 backbone.
///
/// Params: 1UBQ parsed at `Completion::Heavy`, `asm.covalent_bonds()`. Golden
/// against molex's own template + peptide bond graph (the count is invariant
/// across completion levels — bonds derive from residue templates, not from
/// fabricated atoms).
#[test]
fn uc12_covalent_bonds() {
    let asm = ubq_assembly(Completion::Heavy);

    let bonds = asm.covalent_bonds();

    // Captured 2026-06-24: 604 bonds, entirely from the single protein
    // entity (the 58-atom water bulk contributes none). Exact.
    assert_eq!(bonds.len(), 604, "1UBQ Heavy covalent bond count");

    // Aggregate equals the sum of per-entity bond slices, and the water
    // bulk contributes exactly zero.
    let per_entity: usize =
        asm.entities().iter().map(|e| e.bonds().len()).sum();
    assert_eq!(bonds.len(), per_entity);
    let water_bonds: usize = asm
        .entities()
        .iter()
        .filter(|e| e.molecule_type() == molex::MoleculeType::Water)
        .map(|e| e.bonds().len())
        .sum();
    assert_eq!(water_bonds, 0, "bulk water carries no bond topology");

    // Residue-0 backbone, emitted first (entity-declaration order):
    // N(0)-CA(1), CA(1)-C(2) single; C(2)=O(3) double.
    assert_eq!((bonds[0].a.index, bonds[0].b.index), (0, 1));
    assert_eq!(bonds[0].order, BondOrder::Single);
    assert_eq!((bonds[1].a.index, bonds[1].b.index), (1, 2));
    assert_eq!(bonds[1].order, BondOrder::Single);
    assert_eq!((bonds[2].a.index, bonds[2].b.index), (2, 3));
    assert_eq!(bonds[2].order, BondOrder::Double);
}

// --- UC-14 · center of mass + radius of gyration ------------------------

/// Mass-weighted center of mass and radius of gyration of 1UBQ.
///
/// Params: `Assembly::center_of_mass()` / `radius_of_gyration()` over the
/// Heavy-completed 1UBQ assembly, standard atomic weights
/// (`Element::mass`), every atom of every entity. Golden against molex's own
/// numbers.
#[test]
fn uc14_center_of_mass_1ubq() {
    let asm = ubq_assembly(Completion::Heavy);

    let com = asm.center_of_mass();
    let rg = asm.radius_of_gyration();

    // Captured 2026-06-25. Tolerance 1e-2.
    let com_exp = Vec3::new(30.303543, 28.722134, 15.058919);
    let rg_exp = 12.151875_f32;
    assert!(
        (com - com_exp).length() < 1e-2,
        "1UBQ com = {com}, expected {com_exp}"
    );
    assert!(
        (rg - rg_exp).abs() < 1e-2,
        "1UBQ R_g = {rg}, expected {rg_exp}"
    );
}

// --- UC-15 · sidechain chi torsions -------------------------------------

/// χ angles of an ARG sidechain in 1UBQ, plus a synthetic known-angle
/// `dihedral` check.
///
/// Params: `ProteinEntity::chi_angles()` over the Heavy-completed 1UBQ
/// protein entity; degrees, IUPAC sign. Golden against molex's own numbers.
#[test]
fn uc15_chi_1ubq() {
    // Synthetic dihedral: four points in known geometry. p0/p1/p2 fix the
    // first plane; p3 placed to give a +90 and a 180 dihedral about p1->p2.
    let p0 = Vec3::new(0.0, 1.0, 0.0);
    let p1 = Vec3::ZERO;
    let p2 = Vec3::new(1.0, 0.0, 0.0);
    // p3 = p2 + (+z) -> the p1->p2 axis is +x; rotating the p0 plane (xy) by
    // +90 about +x lands the far bond on +z, a +90 dihedral.
    let p3_90 = p2 + Vec3::new(0.0, 0.0, 1.0);
    let d90 = molex::analysis::geometry::dihedral(p0, p1, p2, p3_90);
    assert!((d90.abs() - 90.0).abs() < 1e-3, "expected +/-90, got {d90}");
    // p3 in the same plane, anti to p0 -> 180.
    let p3_180 = p2 + Vec3::new(0.0, -1.0, 0.0);
    let d180 = molex::analysis::geometry::dihedral(p0, p1, p2, p3_180);
    assert!(
        (d180.abs() - 180.0).abs() < 1e-3,
        "expected 180, got {d180}"
    );

    let asm = ubq_assembly(Completion::Heavy);
    let p = asm
        .entities()
        .iter()
        .find_map(|e| e.as_protein())
        .expect("1UBQ has a protein entity");
    let chi = p.chi_angles();

    // 1UBQ residue 41 is ARG42 (Arg with four full χ torsions).
    let arg_name = std::str::from_utf8(&p.residues[41].name).unwrap().trim();
    assert_eq!(arg_name, "ARG", "residue 41 should be ARG");

    // Captured 2026-06-25 (degrees). Tolerance 1e-2.
    let expected = [161.69246, 173.62587, 174.22984, -106.69687];
    let got = &chi[41];
    assert_eq!(got.len(), 4, "ARG has four χ angles");
    for (i, &exp) in expected.iter().enumerate() {
        let v = got[i].expect("complete ARG sidechain has all χ");
        assert!(
            (v - exp).abs() < 1e-2,
            "ARG χ{} = {v}, expected {exp}",
            i + 1
        );
    }
}

// --- UC-16 · one-letter sequence ----------------------------------------

/// One-letter sequence of 1UBQ (the canonical 76-residue ubiquitin string),
/// plus a synthetic single-MSE-residue protein asserting MSE -> 'M'.
///
/// Params: `Assembly::sequence()` over the Heavy-completed 1UBQ assembly.
/// Golden against molex's own output (which matches the canonical ubiquitin
/// sequence).
#[test]
fn uc16_sequence_1ubq() {
    let asm = ubq_assembly(Completion::Heavy);
    let seqs = asm.sequence();
    assert_eq!(seqs.len(), 1, "1UBQ has one polymer entity");

    // Captured 2026-06-25: canonical ubiquitin sequence.
    let expected = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";
    assert_eq!(seqs[0].len(), 76);
    assert_eq!(seqs[0], expected);

    // Synthetic: a one-residue MSE (selenomethionine) protein resolves to 'M'.
    let atoms = vec![
        mk_atom(*b"N   ", Element::N, Vec3::ZERO),
        mk_atom(*b"CA  ", Element::C, Vec3::new(1.0, 0.0, 0.0)),
        mk_atom(*b"C   ", Element::C, Vec3::new(2.0, 0.0, 0.0)),
        mk_atom(*b"O   ", Element::O, Vec3::new(2.0, 1.0, 0.0)),
    ];
    let residue = Residue {
        name: *b"MSE",
        label_seq_id: 1,
        auth_seq_id: None,
        auth_comp_id: None,
        ins_code: None,
        atom_range: 0..atoms.len(),
        variants: Vec::new(),
    };
    let mut alloc = EntityIdAllocator::new();
    let ent = ProteinEntity::new(
        alloc.allocate(),
        atoms,
        vec![residue],
        "A".to_owned(),
    );
    let asm = Assembly::new(vec![MoleculeEntity::Protein(ent)]);
    assert_eq!(asm.sequence(), vec!["M".to_owned()]);
}

// --- UC-17 · spatial contacts + neighbor search -------------------------

/// Atom-atom contacts and a point neighbor search over heavy-atom 1UBQ.
///
/// Params: `ubq_assembly(Completion::Heavy)`. `contacts(4.0, Atom, false)` and
/// `neighbor_search` at the first atom's position. Golden against molex's own
/// counts; the full-cover cross-check asserts `n*(n-1)/2` independent of the
/// captured constants.
#[test]
fn uc17_contacts_1ubq() {
    let asm = ubq_assembly(Completion::Heavy);

    let n_atoms: usize = asm.entities().iter().map(|e| e.atom_count()).sum();

    let contacts = asm.contacts(4.0, ContactLevel::Atom, false);
    // Captured 2026-06-25: heavy-atom 1UBQ, 4.0 A atom-atom contacts.
    assert_eq!(contacts.len(), CONTACT_COUNT_4A);

    // neighbor_search at the first atom's own position must return at least
    // that atom plus its close covalent neighbors.
    let first = &asm.entities()[0];
    let center = first.positions()[0];
    let center_id = molex::AtomId {
        entity: first.id(),
        index: 0,
    };
    let hits = asm.neighbor_search(center, 4.0);
    // Captured 2026-06-25: atoms within 4.0 A of atom 0, including atom 0.
    assert_eq!(hits.len(), NEIGHBOR_COUNT_4A);
    assert!(hits.contains(&center_id), "must find the query atom itself");

    // Sanity: at a cutoff spanning the whole structure every atom pairs with
    // every other exactly once, independent of geometry.
    let full = asm.contacts(1000.0, ContactLevel::Atom, false);
    assert_eq!(full.len(), n_atoms * (n_atoms - 1) / 2);
}

const CONTACT_COUNT_4A: usize = 3611;
const NEIGHBOR_COUNT_4A: usize = 11;
