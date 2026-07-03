//! Tests for `dssp.rs`. Split into a sibling file to keep the primary module
//! under the 800-line source cap enforced by `just file-lengths`.

#![allow(clippy::unwrap_used, deprecated)]

use super::*;
use crate::analysis::bonds::hydrogen::HBond;

/// Helper: create an H-bond with dummy energy.
fn hb(donor: usize, acceptor: usize) -> HBond {
    HBond {
        donor,
        acceptor,
        energy: -1.0,
    }
}

// Basic edge cases

#[test]
fn empty_input() {
    let result = classify(&[], 0);
    assert!(result.is_empty());
}

#[test]
fn single_residue() {
    let result = classify(&[], 1);
    assert_eq!(result, vec![SSType::Coil]);
}

#[test]
fn two_residues_no_hbonds() {
    let result = classify(&[], 2);
    assert_eq!(result, vec![SSType::Coil; 2]);
}

// Helix detection

#[test]
fn alpha_helix_minimal() {
    // Minimal alpha-helix: 2 consecutive 4-turns at i=0 and i=1.
    // Turn at 0: hbond(4, 0). Turn at 1: hbond(5, 1).
    // Marks residues 1..=4 as Helix.
    let hbonds = vec![hb(4, 0), hb(5, 1)];
    let result = classify(&hbonds, 6);
    assert_eq!(result[0], SSType::Coil);
    assert_eq!(result[1], SSType::Helix);
    assert_eq!(result[2], SSType::Helix);
    assert_eq!(result[3], SSType::Helix);
    assert_eq!(result[4], SSType::Helix);
    assert_eq!(result[5], SSType::Coil);
}

#[test]
fn alpha_helix_longer() {
    // 3 consecutive 4-turns at 0,1,2 -> marks 1..=5
    // (first_turn+1=1 through last_turn+turn_size-1=2+3=5)
    let hbonds = vec![hb(4, 0), hb(5, 1), hb(6, 2)];
    let result = classify(&hbonds, 8);
    assert_eq!(result[0], SSType::Coil);
    assert!(result[1..=5].iter().all(|&s| s == SSType::Helix));
    assert_eq!(result[6], SSType::Coil);
}

#[test]
fn three_ten_helix_minimal() {
    // Minimal 3_10 helix: 2 consecutive 3-turns at i=0 and i=1.
    // Turn at 0: hbond(3, 0). Turn at 1: hbond(4, 1).
    // Marks residues 1..=4 (start+1=1 to last_turn+turn_size=1+3=4).
    let hbonds = vec![hb(3, 0), hb(4, 1)];
    let result = classify(&hbonds, 5);
    assert_eq!(result[0], SSType::Coil);
    assert_eq!(result[1], SSType::Helix);
    assert_eq!(result[2], SSType::Helix);
    assert_eq!(result[3], SSType::Helix);
    assert_eq!(result[4], SSType::Coil);
}

#[test]
fn single_turn_not_helix() {
    // A single 4-turn at i=0 (no consecutive partner) should NOT form a
    // helix.
    let hbonds = vec![hb(4, 0)];
    let result = classify(&hbonds, 5);
    assert_eq!(result, vec![SSType::Coil; 5]);
}

#[test]
fn alpha_priority_over_310() {
    // Alpha turns at 0,1 mark 1..=4. 3_10 turns at 2,3 would mark 3..=5.
    // Residues 3,4 already alpha, so only 5 gets added by 3_10.
    let hbonds = vec![
        hb(4, 0),
        hb(5, 1), // alpha turns
        hb(5, 2),
        hb(6, 3), // 3_10 turns
    ];
    let result = classify(&hbonds, 7);
    assert!(result[1..=5].iter().all(|&s| s == SSType::Helix));
}

// Sheet detection

#[test]
fn antiparallel_bridge_pair() {
    // Antiparallel bridge: hbond(j, i) && hbond(i, j)
    let hbonds = vec![hb(15, 5), hb(5, 15)];
    let result = classify(&hbonds, 20);
    assert_eq!(result[5], SSType::Sheet);
    assert_eq!(result[15], SSType::Sheet);
    assert_eq!(result[0], SSType::Coil);
    assert_eq!(result[10], SSType::Coil);
}

#[test]
fn parallel_bridge_pair() {
    // Parallel bridge: hbond(j, i-1) && hbond(i+1, j)
    let hbonds = vec![hb(15, 4), hb(6, 15)];
    let result = classify(&hbonds, 20);
    assert_eq!(result[5], SSType::Sheet);
    assert_eq!(result[15], SSType::Sheet);
}

#[test]
fn antiparallel_ladder_extends() {
    // Antiparallel ladder: bridges (5,15) and (6,14).
    let hbonds = vec![hb(15, 5), hb(5, 15), hb(14, 6), hb(6, 14)];
    let result = classify(&hbonds, 20);
    assert_eq!(result[5], SSType::Sheet);
    assert_eq!(result[6], SSType::Sheet);
    assert_eq!(result[14], SSType::Sheet);
    assert_eq!(result[15], SSType::Sheet);
}

#[test]
fn parallel_ladder_extends() {
    // Parallel ladder: bridges (5,15) and (6,16).
    let hbonds = vec![hb(15, 4), hb(6, 15), hb(16, 5), hb(7, 16)];
    let result = classify(&hbonds, 20);
    assert_eq!(result[5], SSType::Sheet);
    assert_eq!(result[6], SSType::Sheet);
    assert_eq!(result[15], SSType::Sheet);
    assert_eq!(result[16], SSType::Sheet);
}

#[test]
fn helix_takes_priority_over_sheet() {
    // Helix at 2-5, bridge at 5 and 15. Residue 5 should stay Helix.
    let hbonds = vec![
        hb(5, 1),
        hb(6, 2), // alpha turns
        hb(15, 5),
        hb(5, 15), // antiparallel bridge
    ];
    let result = classify(&hbonds, 20);
    assert_eq!(result[5], SSType::Helix);
    assert_eq!(result[15], SSType::Sheet);
}

#[test]
fn bridge_requires_separation() {
    // Minimum separation is 3 (j >= i+3).
    let hbonds = vec![hb(8, 5), hb(5, 8)]; // separation 3: works
    let result = classify(&hbonds, 10);
    assert_eq!(result[5], SSType::Sheet);
    assert_eq!(result[8], SSType::Sheet);

    let hbonds2 = vec![hb(7, 5), hb(5, 7)]; // separation 2: too close
    let result2 = classify(&hbonds2, 10);
    assert_eq!(result2[5], SSType::Coil);
    assert_eq!(result2[7], SSType::Coil);
}

// Integration tests with real structures

#[test]
#[allow(clippy::too_many_lines)]
fn ubiquitin_secondary_structure() {
    let path = std::path::Path::new("../viso/assets/models/1ubq.cif");
    if !path.exists() {
        return;
    }

    let entities =
        crate::adapters::pdb::structure_file_to_entities(path).unwrap();

    let protein_id = entities.iter().find_map(|e| e.as_protein()).unwrap().id;
    let mut assembly = crate::Assembly::new(entities);
    assembly.recompute_ss();
    let ss =
        crate::analysis::merge_short_segments(assembly.ss_types(protein_id));

    // 76 residues: M1-G76. Ground truth from mkdssp 4.4.10 run on
    // our exact 1ubq.cif, converted to Q3 (H,G,I->H; E,B->E; else->C).
    // DSSP8: -EEEEEETTS-EEEEE--TTSBHHHHHHHHHHHH---GGGEEEEETTEEPPTTSBTGGGTPPTT-EEEEEE--S--
    let expected =
            "CEEEEEECCCCEEEEECCCCCEHHHHHHHHHHHHCCCHHHEEEEECCEECCCCCECHHHCCCCCCEEEEEECCCCC";
    assert_eq!(
        ss.len(),
        expected.len(),
        "SS length {} != expected {}",
        ss.len(),
        expected.len()
    );

    let expected_ss: Vec<SSType> = expected
        .chars()
        .map(|c| match c {
            'H' => SSType::Helix,
            'E' => SSType::Sheet,
            _ => SSType::Coil,
        })
        .collect();

    // Count matches.
    let mut matches = 0;
    let mut mismatches = Vec::new();
    for (i, (got, exp)) in ss.iter().zip(expected_ss.iter()).enumerate() {
        if got == exp {
            matches += 1;
        } else {
            mismatches.push((i, *exp, *got));
        }
    }

    let accuracy = f64::from(u32::try_from(matches).unwrap())
        / f64::from(u32::try_from(ss.len()).unwrap());

    let got_str: String = ss
        .iter()
        .map(|s| match s {
            SSType::Helix => 'H',
            SSType::Sheet => 'E',
            SSType::Coil => 'C',
        })
        .collect();

    assert!(
        accuracy > 0.90,
        "1UBQ accuracy {:.1}% below 90%\n  expected: {expected}\n  got:      \
         {got_str}\n  mismatches: {mismatches:?}",
        accuracy * 100.0
    );

    // Alpha helix: indices 22-33 should be mostly Helix.
    let helix_count =
        ss[22..=33].iter().filter(|&&s| s == SSType::Helix).count();
    assert!(
        helix_count >= 10,
        "Alpha helix (22-33): only {helix_count}/12 residues are Helix"
    );

    // Beta strands.
    let strand_ranges = [(1, 6), (9, 16), (39, 44), (47, 49), (63, 71)];
    for (start, end) in strand_ranges {
        let sheet_count = ss[start..=end]
            .iter()
            .filter(|&&s| s == SSType::Sheet)
            .count();
        let len = end - start + 1;
        assert!(
            sheet_count >= len / 2,
            "Strand ({start}-{end}): only {sheet_count}/{len} residues are \
             Sheet"
        );
    }

    // 3_10 helix: indices 55-58 should have some Helix.
    let helix_310 = ss[55..=58].iter().filter(|&&s| s == SSType::Helix).count();
    assert!(
        helix_310 >= 2,
        "3_10 helix (55-58): only {helix_310}/4 residues are Helix"
    );
}

// End-to-end geometry pipeline (real coordinates)

/// Internal coordinates placing one atom relative to three prior ones:
/// the new bond length plus the bond angle and dihedral (degrees) that
/// orient it.
#[derive(Clone, Copy)]
struct Internal {
    length: f32,
    angle_deg: f32,
    dihedral_deg: f32,
}

/// Place a new atom from the three most recently placed atoms `prior`
/// (oldest first) and the internal coordinates `geom`. Natural
/// Extension Reference Frame construction; lets us grow a backbone
/// from internal coordinates so every bond length, angle, and torsion
/// is physically real.
fn nerf(prior: [glam::Vec3; 3], geom: Internal) -> glam::Vec3 {
    let angle = geom.angle_deg.to_radians();
    let dih = geom.dihedral_deg.to_radians();
    let [first, mid, last] = prior;
    let axis = (last - mid).normalize();
    let normal = (mid - first).cross(axis).normalize();
    let binormal = normal.cross(axis);
    let local = glam::Vec3::new(
        -geom.length * angle.cos(),
        geom.length * angle.sin() * dih.cos(),
        geom.length * angle.sin() * dih.sin(),
    );
    last + local.x * axis + local.y * binormal + local.z * normal
}

/// Build an idealized right-handed alpha-helix backbone of `n_res`
/// residues from canonical internal coordinates (bond lengths/angles
/// and phi=-57 deg, psi=-47 deg, omega=180 deg). Returns one
/// `ResidueBackbone` (N, CA, C, O) per residue with realistic
/// geometry: ~1.46 A N-CA, ~1.52 A CA-C, ~1.33 A peptide C-N,
/// ~3.8 A CA-CA, which is what `detect_hbonds` actually consumes.
#[allow(clippy::too_many_lines)]
fn ideal_alpha_helix(n_res: usize) -> Vec<crate::ResidueBackbone> {
    use glam::Vec3;

    // Canonical backbone bond lengths (A).
    let l_n_ca = 1.458_f32;
    let l_ca_c = 1.525_f32;
    let l_c_n = 1.329_f32;
    let l_c_o = 1.231_f32;
    // Canonical backbone bond angles (deg).
    let a_n_ca_c = 111.2_f32;
    let a_ca_c_n = 116.2_f32;
    let a_c_n_ca = 121.7_f32;
    let a_ca_c_o = 120.8_f32;
    // Ideal alpha-helix torsions (deg).
    let phi = -57.0_f32;
    let psi = -47.0_f32;
    let omega = 180.0_f32;

    // Seed the first three atoms of residue 0 in the xy-plane.
    let mut n_pos = vec![Vec3::ZERO];
    let mut ca_pos = vec![Vec3::new(l_n_ca, 0.0, 0.0)];
    let seed_ang = a_n_ca_c.to_radians();
    let mut c_pos = vec![Vec3::new(
        (-l_ca_c).mul_add(seed_ang.cos(), ca_pos[0].x),
        l_ca_c * seed_ang.sin(),
        0.0,
    )];
    let mut o_pos = Vec::with_capacity(n_res);

    let mut i = 0;
    while i < n_res {
        let prior = [n_pos[i], ca_pos[i], c_pos[i]];
        // O(i): dihedral N-CA-C-O = psi + 180 (carbonyl anti to next N).
        o_pos.push(nerf(
            prior,
            Internal {
                length: l_c_o,
                angle_deg: a_ca_c_o,
                dihedral_deg: psi + 180.0,
            },
        ));
        if i + 1 < n_res {
            // N(i+1): dihedral N(i)-CA(i)-C(i)-N = psi.
            let n_next = nerf(
                prior,
                Internal {
                    length: l_c_n,
                    angle_deg: a_ca_c_n,
                    dihedral_deg: psi,
                },
            );
            // CA(i+1): dihedral CA(i)-C(i)-N(i+1)-CA = omega.
            let ca_next = nerf(
                [ca_pos[i], c_pos[i], n_next],
                Internal {
                    length: l_n_ca,
                    angle_deg: a_c_n_ca,
                    dihedral_deg: omega,
                },
            );
            // C(i+1): dihedral C(i)-N(i+1)-CA(i+1)-C = phi.
            let c_next = nerf(
                [c_pos[i], n_next, ca_next],
                Internal {
                    length: l_ca_c,
                    angle_deg: a_n_ca_c,
                    dihedral_deg: phi,
                },
            );
            n_pos.push(n_next);
            ca_pos.push(ca_next);
            c_pos.push(c_next);
        }
        i += 1;
    }

    n_pos
        .iter()
        .zip(ca_pos.iter())
        .zip(c_pos.iter())
        .zip(o_pos.iter())
        .map(|(((&n, &ca), &c), &o)| crate::ResidueBackbone { n, ca, c, o })
        .collect()
}

/// End-to-end: drive a real-coordinate ideal alpha-helix through the
/// crate's secondary-structure pipeline (`Assembly` -> `to_backbone`
/// -> `detect_hbonds` -> `classify`). Unlike the hand-built-HBond
/// tests above, this exercises the Kabsch-Sander electrostatic
/// geometry that actually feeds DSSP. The interior residues must
/// classify as Helix; the terminal residue on each end is allowed to
/// stay Coil (no i->i+4 partner to anchor a turn).
#[test]
#[allow(clippy::too_many_lines)]
fn ideal_helix_classifies_as_helix_via_real_geometry() {
    use crate::element::Element;
    use crate::entity::molecule::id::EntityIdAllocator;
    use crate::entity::molecule::protein::ProteinEntity;
    use crate::entity::molecule::{MoleculeEntity, Residue};
    use crate::Atom;

    const N_RES: usize = 14;
    let backbone = ideal_alpha_helix(N_RES);
    assert_eq!(backbone.len(), N_RES);

    // Sanity-check the generated geometry is physically real before we
    // rely on it: peptide C(i)->N(i+1) must look like a real bond, or
    // ProteinEntity::new would insert segment breaks and split the
    // chain. Allow a generous band around the canonical ~1.33 A.
    for w in backbone.windows(2) {
        let peptide = (w[0].c - w[1].n).length();
        assert!(
            (1.0..1.6).contains(&peptide),
            "peptide bond {peptide:.2} A is not bond-length; geometry is wrong"
        );
    }

    // Build a real protein entity from the helix coordinates. One ALA
    // per residue, atoms in canonical N, CA, C, O order.
    let atom = |name: &[u8; 4], element: Element, pos: glam::Vec3| Atom {
        position: pos,
        occupancy: 1.0,
        b_factor: 0.0,
        element,
        name: *name,
        formal_charge: 0,
        observed: true,
    };
    let mut atoms = Vec::with_capacity(N_RES * 4);
    let mut residues = Vec::with_capacity(N_RES);
    for (idx, bb) in backbone.iter().enumerate() {
        let start = atoms.len();
        atoms.push(atom(b"N   ", Element::N, bb.n));
        atoms.push(atom(b"CA  ", Element::C, bb.ca));
        atoms.push(atom(b"C   ", Element::C, bb.c));
        atoms.push(atom(b"O   ", Element::O, bb.o));
        let seq = i32::try_from(idx + 1).unwrap();
        residues.push(Residue {
            name: *b"ALA",
            label_seq_id: seq,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: start..atoms.len(),
            variants: Vec::new(),
        });
    }

    let id = EntityIdAllocator::new().allocate();
    let protein = ProteinEntity::new(id, atoms, residues, "A".to_owned());
    // No segment breaks: the chain must be one continuous helix or the
    // geometry isn't peptide-bonded.
    assert!(
        protein.segment_breaks.is_empty(),
        "ideal helix should be one continuous chain, got breaks {:?}",
        protein.segment_breaks
    );

    let mut assembly =
        crate::Assembly::new(vec![MoleculeEntity::Protein(protein)]);
    assembly.recompute_ss();
    let ss = assembly.ss_types(id);
    assert_eq!(ss.len(), N_RES);

    let ss_str: String = ss
        .iter()
        .map(|s| match s {
            SSType::Helix => 'H',
            SSType::Sheet => 'E',
            SSType::Coil => 'C',
        })
        .collect();

    // Interior residues (1..=N-2) must be Helix; the two ends may fray
    // to Coil. No residue may be classified Sheet.
    let interior_all_helix =
        ss[1..N_RES - 1].iter().all(|&s| s == SSType::Helix);
    assert!(
        interior_all_helix,
        "interior of ideal alpha-helix not all Helix; SS = {ss_str}"
    );
    assert!(
        ss.iter().all(|&s| s != SSType::Sheet),
        "ideal helix should contain no Sheet; SS = {ss_str}"
    );
}

#[test]
fn hemoglobin_mostly_helical() {
    let path = std::path::Path::new("../viso/assets/models/4hhb.cif");
    if !path.exists() {
        return;
    }

    let entities =
        crate::adapters::pdb::structure_file_to_entities(path).unwrap();

    let protein_id = entities.iter().find_map(|e| e.as_protein()).unwrap().id;
    let mut assembly = crate::Assembly::new(entities);
    assembly.recompute_ss();
    let ss =
        crate::analysis::merge_short_segments(assembly.ss_types(protein_id));

    let helix_count = ss.iter().filter(|&&s| s == SSType::Helix).count();
    let helix_frac = f64::from(u32::try_from(helix_count).unwrap())
        / f64::from(u32::try_from(ss.len()).unwrap());
    assert!(
        helix_frac > 0.50,
        "4HHB should be mostly helical, got {:.1}% helix",
        helix_frac * 100.0
    );

    let sheet_count = ss.iter().filter(|&&s| s == SSType::Sheet).count();
    let sheet_frac = f64::from(u32::try_from(sheet_count).unwrap())
        / f64::from(u32::try_from(ss.len()).unwrap());
    assert!(
        sheet_frac < 0.15,
        "4HHB should have very little sheet, got {:.1}%",
        sheet_frac * 100.0
    );
}
