use super::*;
use crate::adapters::cif::parse::parse;

const MMCIF_SNIPPET: &str = r"data_1ABC
_cell.length_a   50.000
_cell.length_b   60.000
_cell.length_c   70.000
_cell.angle_alpha 90.00
_cell.angle_beta  90.00
_cell.angle_gamma 90.00
_symmetry.space_group_name_H-M 'P 21 21 21'
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.type_symbol
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM N   ALA A 1 10.000 20.000 30.000 N 1.00 15.0
ATOM CA  ALA A 1 11.000 21.000 31.000 C 1.00 16.0
ATOM C   ALA A 1 12.000 22.000 32.000 C 1.00 17.0
HETATM O   HOH B 100 5.000 6.000 7.000 O 1.00 20.0
";

const SF_SNIPPET: &str = r"data_r1abcsf
_cell.length_a   50.000
_cell.length_b   60.000
_cell.length_c   70.000
_cell.angle_alpha 90.00
_cell.angle_beta  90.00
_cell.angle_gamma 90.00
_symmetry.space_group_name_H-M 'P 21 21 21'
loop_
_refln.index_h
_refln.index_k
_refln.index_l
_refln.F_meas_au
_refln.F_meas_sigma_au
_refln.status
1  0  0  100.5 2.3 o
0  1  0  200.1 3.4 o
0  0  1  150.7 2.8 f
-1 2  3  .     .   o
";

#[test]
fn extract_coordinates() {
    let doc = parse(MMCIF_SNIPPET).unwrap();
    let coords = CoordinateData::try_from(&doc.blocks[0]).unwrap();

    assert_eq!(coords.atoms.len(), 4);
    assert_eq!(coords.atoms[0].label, "N");
    assert_eq!(coords.atoms[0].residue, "ALA");
    assert_eq!(coords.atoms[0].chain, "A");
    assert_eq!(coords.atoms[0].seq_id, Some(1));
    assert!((coords.atoms[0].x - 10.0).abs() < 1e-6);
    assert!((coords.atoms[1].y - 21.0).abs() < 1e-6);
    assert_eq!(coords.atoms[1].element, "C");
    assert!((coords.atoms[2].b_factor - 17.0).abs() < 1e-6);

    // HETATM
    assert_eq!(coords.atoms[3].group, "HETATM");
    assert_eq!(coords.atoms[3].residue, "HOH");
    assert_eq!(coords.atoms[3].chain, "B");
}

#[test]
fn extract_unit_cell() {
    let doc = parse(MMCIF_SNIPPET).unwrap();
    let cell = UnitCell::try_from(&doc.blocks[0]).unwrap();
    assert!((cell.a - 50.0).abs() < 1e-6);
    assert!((cell.b - 60.0).abs() < 1e-6);
    assert!((cell.c - 70.0).abs() < 1e-6);
    assert!((cell.alpha - 90.0).abs() < 1e-6);
}

#[test]
fn extract_spacegroup_from_block() {
    let doc = parse(MMCIF_SNIPPET).unwrap();
    let coords = CoordinateData::try_from(&doc.blocks[0]).unwrap();
    assert_eq!(coords.spacegroup.as_deref(), Some("P 21 21 21"));
}

#[test]
fn extract_reflections() {
    let doc = parse(SF_SNIPPET).unwrap();
    let sf = ReflectionData::try_from(&doc.blocks[0]).unwrap();

    assert_eq!(sf.obs_data_type, ObsDataType::Amplitude);
    assert_eq!(sf.reflections.len(), 4);
    assert_eq!(sf.reflections[0].h, 1);
    assert_eq!(sf.reflections[0].k, 0);
    assert_eq!(sf.reflections[0].l, 0);
    assert!((sf.reflections[0].f_meas.unwrap() - 100.5).abs() < 1e-6);
    assert!((sf.reflections[0].sigma_f_meas.unwrap() - 2.3).abs() < 1e-6);
    assert!(!sf.reflections[0].free_flag); // 'o' = working

    // Free-set flag
    assert!(sf.reflections[2].free_flag); // 'f' = free

    // Negative index
    assert_eq!(sf.reflections[3].h, -1);

    // Inapplicable values -> None
    assert!(sf.reflections[3].f_meas.is_none());
    assert!(sf.reflections[3].sigma_f_meas.is_none());
}

/// SF-CIF with intensity columns instead of amplitudes -- the parser should
/// auto-detect and convert via F = sqrt(I), sigma_F = sigma_I / (2*sqrt(I)).
#[test]
fn extract_reflections_from_intensities() {
    let input = r"data_rint
_cell.length_a   50.0
_cell.length_b   60.0
_cell.length_c   70.0
_cell.angle_alpha 90.0
_cell.angle_beta  90.0
_cell.angle_gamma 90.0
loop_
_refln.index_h
_refln.index_k
_refln.index_l
_refln.intensity_meas
_refln.intensity_sigma
_refln.status
1 0 0  10000.0 200.0 o
0 1 0  400.0   40.0  f
0 0 1  .       .     o
";
    let doc = parse(input).unwrap();
    let sf = ReflectionData::try_from(&doc.blocks[0]).unwrap();

    assert_eq!(sf.obs_data_type, ObsDataType::Intensity);
    assert_eq!(sf.reflections.len(), 3);

    // I=10000 -> F=100, sigma_I=200 -> sigma_F = 200/(2*100) = 1.0
    let r0 = &sf.reflections[0];
    assert!((r0.f_meas.unwrap() - 100.0).abs() < 1e-6);
    assert!((r0.sigma_f_meas.unwrap() - 1.0).abs() < 1e-6);
    assert!(!r0.free_flag);

    // I=400 -> F=20, sigma_I=40 -> sigma_F = 40/(2*20) = 1.0
    let r1 = &sf.reflections[1];
    assert!((r1.f_meas.unwrap() - 20.0).abs() < 1e-6);
    assert!((r1.sigma_f_meas.unwrap() - 1.0).abs() < 1e-6);
    assert!(r1.free_flag); // 'f'

    // Inapplicable intensity -> None amplitude
    assert!(sf.reflections[2].f_meas.is_none());
}

/// SF-CIF using the older `_refln.F_obs` convention instead of `F_meas_au`.
#[test]
fn extract_reflections_f_obs_convention() {
    let input = r"data_rfobs
_cell.length_a   50.0
_cell.length_b   60.0
_cell.length_c   70.0
_cell.angle_alpha 90.0
_cell.angle_beta  90.0
_cell.angle_gamma 90.0
loop_
_refln.index_h
_refln.index_k
_refln.index_l
_refln.F_obs
_refln.F_obs_sigma
1 0 0  55.5 1.2
";
    let doc = parse(input).unwrap();
    let sf = ReflectionData::try_from(&doc.blocks[0]).unwrap();

    assert_eq!(sf.obs_data_type, ObsDataType::Amplitude);
    assert!((sf.reflections[0].f_meas.unwrap() - 55.5).abs() < 1e-6);
    assert!((sf.reflections[0].sigma_f_meas.unwrap() - 1.2).abs() < 1e-6);
}

/// SF-CIF using `_refln.pdbx_r_free_flag` (modern PDB convention).
#[test]
fn extract_reflections_pdbx_rfree_flag() {
    let input = r"data_rpdbx
_cell.length_a   50.0
_cell.length_b   60.0
_cell.length_c   70.0
_cell.angle_alpha 90.0
_cell.angle_beta  90.0
_cell.angle_gamma 90.0
loop_
_refln.index_h
_refln.index_k
_refln.index_l
_refln.F_meas_au
_refln.pdbx_r_free_flag
1 0 0  100.0 0
0 1 0  200.0 1
0 0 1  150.0 0
";
    let doc = parse(input).unwrap();
    let sf = ReflectionData::try_from(&doc.blocks[0]).unwrap();

    assert!(!sf.reflections[0].free_flag); // flag=0 -> working
    assert!(sf.reflections[1].free_flag); // flag=1 -> free
    assert!(!sf.reflections[2].free_flag); // flag=0 -> working
}

/// Amplitudes take priority over intensities when both are present.
#[test]
fn amplitudes_preferred_over_intensities() {
    let input = r"data_rboth
_cell.length_a   50.0
_cell.length_b   60.0
_cell.length_c   70.0
_cell.angle_alpha 90.0
_cell.angle_beta  90.0
_cell.angle_gamma 90.0
loop_
_refln.index_h
_refln.index_k
_refln.index_l
_refln.F_meas_au
_refln.intensity_meas
1 0 0  100.0 10000.0
";
    let doc = parse(input).unwrap();
    let sf = ReflectionData::try_from(&doc.blocks[0]).unwrap();

    assert_eq!(sf.obs_data_type, ObsDataType::Amplitude);
    // Should use F_meas_au (100), not sqrt(intensity) (100).
    assert!((sf.reflections[0].f_meas.unwrap() - 100.0).abs() < 1e-6);
}

/// Negative intensities produce None (physically meaningless).
#[test]
fn negative_intensity_yields_none() {
    let (f, sigma) = intensity_to_amplitude(Some(-5.0), Some(1.0));
    assert!(f.is_none());
    assert!(sigma.is_none());
}

#[test]
fn reflection_cell_required() {
    let input = "data_test\nloop_\n_refln.index_h\n_refln.index_k\n_refln.\
                 index_l\n1 0 0\n";
    let doc = parse(input).unwrap();
    let err = ReflectionData::try_from(&doc.blocks[0]).unwrap_err();
    assert!(matches!(err, ExtractionError::MissingTag(_)));
}

#[test]
fn autodetect_coordinates() {
    let doc = parse(MMCIF_SNIPPET).unwrap();
    let content = CifContent::from(doc.blocks.into_iter().next().unwrap());
    assert!(matches!(content, CifContent::Coordinates(_)));
}

#[test]
fn autodetect_reflections() {
    let doc = parse(SF_SNIPPET).unwrap();
    let content = CifContent::from(doc.blocks.into_iter().next().unwrap());
    assert!(matches!(content, CifContent::Reflections(_)));
}

#[test]
fn autodetect_unknown() {
    let input = "data_mystery\n_some.tag value\n";
    let doc = parse(input).unwrap();
    let content = CifContent::from(doc.blocks.into_iter().next().unwrap());
    assert!(matches!(content, CifContent::Unknown(_)));
}

#[test]
fn missing_atom_site_category() {
    let input = "data_empty\n_cell.length_a 50.0\n";
    let doc = parse(input).unwrap();
    let err = CoordinateData::try_from(&doc.blocks[0]).unwrap_err();
    assert!(matches!(err, ExtractionError::MissingCategory(_)));
}

#[test]
fn optional_columns_absent() {
    // Minimal atom_site with only required columns
    let input = r"data_min
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
CA ALA A 1.0 2.0 3.0
";
    let doc = parse(input).unwrap();
    let coords = CoordinateData::try_from(&doc.blocks[0]).unwrap();
    assert_eq!(coords.atoms.len(), 1);
    assert_eq!(coords.atoms[0].label, "CA");
    assert_eq!(coords.atoms[0].group, "ATOM"); // default
    assert_eq!(coords.atoms[0].element, ""); // absent
    assert!((coords.atoms[0].occupancy - 1.0).abs() < 1e-6); // default
    assert!((coords.atoms[0].b_factor - 0.0).abs() < 1e-6); // default
    assert!(coords.atoms[0].seq_id.is_none());
    assert!(coords.cell.is_none());
    assert!(coords.spacegroup.is_none());
}

#[test]
fn cell_with_uncertainty() {
    let input = "data_test\n_cell.length_a 50.123(4)\n_cell.length_b \
                 60.456(5)\n_cell.length_c 70.789(6)\n_cell.angle_alpha \
                 90.00(1)\n_cell.angle_beta 90.00(1)\n_cell.angle_gamma \
                 90.00(1)\n";
    let doc = parse(input).unwrap();
    let cell = UnitCell::try_from(&doc.blocks[0]).unwrap();
    assert!((cell.a - 50.123).abs() < 1e-6);
    assert!((cell.b - 60.456).abs() < 1e-6);
}

#[test]
fn free_flags_from_file_true_when_status_present() {
    let doc = parse(SF_SNIPPET).unwrap();
    let sf = ReflectionData::try_from(&doc.blocks[0]).unwrap();
    assert!(sf.free_flags_from_file);
}

#[test]
fn free_flags_from_file_false_when_no_flags() {
    let input = r"data_rnoflag
_cell.length_a   50.0
_cell.length_b   60.0
_cell.length_c   70.0
_cell.angle_alpha 90.0
_cell.angle_beta  90.0
_cell.angle_gamma 90.0
loop_
_refln.index_h
_refln.index_k
_refln.index_l
_refln.F_meas_au
1 0 0 100.0
0 1 0 200.0
";
    let doc = parse(input).unwrap();
    let sf = ReflectionData::try_from(&doc.blocks[0]).unwrap();

    assert!(!sf.free_flags_from_file);
    // All reflections default to working set.
    assert!(sf.reflections.iter().all(|r| !r.free_flag));
}
