use super::*;

/// Parameters for building a minimal valid MRC header + data.
struct TestMrcParams<'a> {
    nc: i32,
    nr: i32,
    ns: i32,
    mode: i32,
    mapc: i32,
    mapr: i32,
    maps: i32,
    data_values: &'a [f32],
}

/// Build a minimal valid MRC header + data for testing.
fn make_test_mrc(p: &TestMrcParams<'_>) -> Vec<u8> {
    let mut header = vec![0u8; HEADER_SIZE];

    let put_i32 = |buf: &mut [u8], offset: usize, val: i32| {
        buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    };
    let put_f32 = |buf: &mut [u8], offset: usize, val: f32| {
        buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    };

    // Grid dimensions
    put_i32(&mut header, 0, p.nc);
    put_i32(&mut header, 4, p.nr);
    put_i32(&mut header, 8, p.ns);

    // Mode
    put_i32(&mut header, 12, p.mode);

    // Start indices (all 0)
    put_i32(&mut header, 16, 0);
    put_i32(&mut header, 20, 0);
    put_i32(&mut header, 24, 0);

    // MX, MY, MZ = grid dims
    put_i32(&mut header, 28, p.nc);
    put_i32(&mut header, 32, p.nr);
    put_i32(&mut header, 36, p.ns);

    // Cell dimensions (10 A each)
    put_f32(&mut header, 40, 10.0);
    put_f32(&mut header, 44, 10.0);
    put_f32(&mut header, 48, 10.0);

    // Cell angles (90 deg)
    put_f32(&mut header, 52, 90.0);
    put_f32(&mut header, 56, 90.0);
    put_f32(&mut header, 60, 90.0);

    // Axis mapping
    put_i32(&mut header, 64, p.mapc);
    put_i32(&mut header, 68, p.mapr);
    put_i32(&mut header, 72, p.maps);

    // DMIN, DMAX, DMEAN
    put_f32(&mut header, 76, 0.0);
    put_f32(&mut header, 80, 1.0);
    put_f32(&mut header, 84, 0.5);

    // Space group
    put_i32(&mut header, 88, 1);

    // NSYMBT = 0
    put_i32(&mut header, 92, 0);

    // Origin (0, 0, 0)
    put_f32(&mut header, 196, 0.0);
    put_f32(&mut header, 200, 0.0);
    put_f32(&mut header, 204, 0.0);

    // MAP_ magic
    header[208..212].copy_from_slice(b"MAP ");

    // MACHST: little-endian
    header[212] = 0x44;

    // RMS
    put_f32(&mut header, 216, 0.25);

    // Append data as mode-2 f32
    let mut bytes = header;
    for &v in p.data_values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    bytes
}

#[test]
fn parse_basic_xyz() {
    // 2x3x4 grid, XYZ axis order (MAPC=1, MAPR=2, MAPS=3)
    let nc = 2;
    let nr = 3;
    let ns = 4;
    let total = (nc * nr * ns) as usize;
    let data: Vec<f32> = (0..total).map(|i| i as f32).collect();

    let mrc = make_test_mrc(&TestMrcParams {
        nc,
        nr,
        ns,
        mode: 2,
        mapc: 1,
        mapr: 2,
        maps: 3,
        data_values: &data,
    });
    let map = mrc_to_density(&mrc).unwrap();

    assert_eq!(map.nx, 2);
    assert_eq!(map.ny, 3);
    assert_eq!(map.nz, 4);
    assert_eq!(map.dmin, 0.0);
    assert_eq!(map.dmax, 1.0);
    assert_eq!(map.dmean, 0.5);
    assert_eq!(map.rms, 0.25);

    // File stores section(Z)->row(Y)->col(X):
    // index in flat = z * nr * nc + y * nc + x
    // So flat[0] = data[z=0,y=0,x=0], flat[1] = data[z=0,y=0,x=1], etc.
    assert_eq!(map.data[[0, 0, 0]], 0.0);
    assert_eq!(map.data[[1, 0, 0]], 1.0); // x=1, col index 1
    assert_eq!(map.data[[0, 1, 0]], 2.0); // y=1, row index 1 -> offset nc
    assert_eq!(map.data[[0, 0, 1]], 6.0); // z=1, section index 1 -> offset
                                          // nr*nc
}

#[test]
fn parse_zxy_axis_reorder() {
    // File stores cols=Z, rows=X, sections=Y (MAPC=3, MAPR=1, MAPS=2)
    // NC=4(Z), NR=2(X), NS=3(Y)
    let nc = 4; // maps to Z
    let nr = 2; // maps to X
    let ns = 3; // maps to Y

    let total = (nc * nr * ns) as usize;
    let data: Vec<f32> = (0..total).map(|i| i as f32).collect();

    let mrc = make_test_mrc(&TestMrcParams {
        nc,
        nr,
        ns,
        mode: 2,
        mapc: 3,
        mapr: 1,
        maps: 2,
        data_values: &data,
    });
    let map = mrc_to_density(&mrc).unwrap();

    // Spatial: nx=NR=2, ny=NS=3, nz=NC=4
    assert_eq!(map.nx, 2);
    assert_eq!(map.ny, 3);
    assert_eq!(map.nz, 4);

    // Flat index for file (s, r, c) = s * nr * nc + r * nc + c
    // Spatial: xyz[MAPC-1]=xyz[2]=c, xyz[MAPR-1]=xyz[0]=r,
    // xyz[MAPS-1]=xyz[1]=s So map[x,y,z] should equal flat[y * nr *
    // nc + x * nc + z]

    // (x=0,y=0,z=0) -> (s=0,r=0,c=0) -> flat[0] = 0
    assert_eq!(map.data[[0, 0, 0]], 0.0);
    // (x=1,y=0,z=0) -> (s=0,r=1,c=0) -> flat[0*2*4 + 1*4 + 0] = 4
    assert_eq!(map.data[[1, 0, 0]], 4.0);
    // (x=0,y=1,z=0) -> (s=1,r=0,c=0) -> flat[1*2*4 + 0] = 8
    assert_eq!(map.data[[0, 1, 0]], 8.0);
    // (x=0,y=0,z=1) -> (s=0,r=0,c=1) -> flat[1] = 1
    assert_eq!(map.data[[0, 0, 1]], 1.0);
}

#[test]
fn voxel_size_and_grid_to_cartesian() {
    let nc = 2;
    let nr = 3;
    let ns = 4;
    let total = (nc * nr * ns) as usize;
    let data: Vec<f32> = vec![0.0; total];

    let mut mrc = make_test_mrc(&TestMrcParams {
        nc,
        nr,
        ns,
        mode: 2,
        mapc: 1,
        mapr: 2,
        maps: 3,
        data_values: &data,
    });

    // Set cell dims to 20, 30, 40 A with MX=2, MY=3, MZ=4
    mrc[40..44].copy_from_slice(&20.0f32.to_le_bytes());
    mrc[44..48].copy_from_slice(&30.0f32.to_le_bytes());
    mrc[48..52].copy_from_slice(&40.0f32.to_le_bytes());

    let map = mrc_to_density(&mrc).unwrap();

    let vs = map.voxel_size();
    assert!((vs[0] - 10.0).abs() < 1e-6); // 20/2
    assert!((vs[1] - 10.0).abs() < 1e-6); // 30/3
    assert!((vs[2] - 10.0).abs() < 1e-6); // 40/4

    let pos = map.grid_to_cartesian(1, 2, 3);
    assert!((pos[0] - 10.0).abs() < 1e-4, "x: got {}", pos[0]);
    assert!((pos[1] - 20.0).abs() < 1e-4, "y: got {}", pos[1]);
    assert!((pos[2] - 30.0).abs() < 1e-4, "z: got {}", pos[2]);
}

#[test]
fn sigma_level() {
    let nc = 1;
    let nr = 1;
    let ns = 1;
    let data = make_test_mrc(&TestMrcParams {
        nc,
        nr,
        ns,
        mode: 2,
        mapc: 1,
        mapr: 2,
        maps: 3,
        data_values: &[0.5f32],
    });

    // dmean=0.5, rms=0.25
    let map = mrc_to_density(&data).unwrap();
    let level = map.sigma_level(2.0);
    assert!((level - 1.0).abs() < 1e-6); // 0.5 + 2.0 * 0.25 = 1.0
}

#[test]
fn reject_invalid_magic() {
    let mut mrc = make_test_mrc(&TestMrcParams {
        nc: 1,
        nr: 1,
        ns: 1,
        mode: 2,
        mapc: 1,
        mapr: 2,
        maps: 3,
        data_values: &[0.0],
    });
    mrc[208..212].copy_from_slice(b"NOPE");

    let err = mrc_to_density(&mrc).unwrap_err();
    assert!(matches!(err, DensityError::InvalidFormat(_)));
}

#[test]
fn reject_unsupported_mode() {
    let mut mrc = make_test_mrc(&TestMrcParams {
        nc: 1,
        nr: 1,
        ns: 1,
        mode: 2,
        mapc: 1,
        mapr: 2,
        maps: 3,
        data_values: &[0.0],
    });
    // Set mode to 3 (complex i16, unsupported)
    mrc[12..16].copy_from_slice(&3i32.to_le_bytes());

    let err = mrc_to_density(&mrc).unwrap_err();
    assert!(matches!(err, DensityError::UnsupportedMode(3)));
}

#[test]
fn reject_too_small() {
    let err = mrc_to_density(&[0u8; 100]).unwrap_err();
    assert!(matches!(err, DensityError::InvalidFormat(_)));
}

#[test]
fn mode_1_i16() {
    // Test mode 1 (signed 16-bit)
    let mut header = make_test_mrc(&TestMrcParams {
        nc: 2,
        nr: 1,
        ns: 1,
        mode: 1,
        mapc: 1,
        mapr: 2,
        maps: 3,
        data_values: &[],
    });
    // Remove the empty f32 data appended by make_test_mrc (none since &[]
    // was passed) Now manually add i16 data
    let values: [i16; 2] = [100, -200];
    for v in &values {
        header.extend_from_slice(&v.to_le_bytes());
    }

    let map = mrc_to_density(&header).unwrap();
    assert_eq!(map.data[[0, 0, 0]], 100.0);
    assert_eq!(map.data[[1, 0, 0]], -200.0);
}

#[test]
fn mode_0_i8() {
    let mut header = make_test_mrc(&TestMrcParams {
        nc: 2,
        nr: 1,
        ns: 1,
        mode: 0,
        mapc: 1,
        mapr: 2,
        maps: 3,
        data_values: &[],
    });
    header.push(50u8); // 50 as i8
    header.push((-30i8).cast_unsigned()); // -30 as i8

    let map = mrc_to_density(&header).unwrap();
    assert_eq!(map.data[[0, 0, 0]], 50.0);
    assert_eq!(map.data[[1, 0, 0]], -30.0);
}

#[test]
fn mode_6_u16() {
    let mut header = make_test_mrc(&TestMrcParams {
        nc: 2,
        nr: 1,
        ns: 1,
        mode: 6,
        mapc: 1,
        mapr: 2,
        maps: 3,
        data_values: &[],
    });
    let values: [u16; 2] = [1000, 65000];
    for v in &values {
        header.extend_from_slice(&v.to_le_bytes());
    }

    let map = mrc_to_density(&header).unwrap();
    assert_eq!(map.data[[0, 0, 0]], 1000.0);
    assert_eq!(map.data[[1, 0, 0]], 65000.0);
}

#[test]
fn endianness_fallback() {
    // Set MACHST to unknown value, but leave mode as valid LE
    let mut mrc = make_test_mrc(&TestMrcParams {
        nc: 1,
        nr: 1,
        ns: 1,
        mode: 2,
        mapc: 1,
        mapr: 2,
        maps: 3,
        data_values: &[1.0],
    });
    mrc[212] = 0x00; // unknown MACHST

    let map = mrc_to_density(&mrc).unwrap();
    assert_eq!(map.data[[0, 0, 0]], 1.0);
}

#[test]
fn extended_header_skip() {
    let nc = 1;
    let nr = 1;
    let ns = 1;

    let mut mrc = make_test_mrc(&TestMrcParams {
        nc,
        nr,
        ns,
        mode: 2,
        mapc: 1,
        mapr: 2,
        maps: 3,
        data_values: &[],
    });

    // Set NSYMBT = 64 (skip 64 bytes of extended header)
    mrc[92..96].copy_from_slice(&64i32.to_le_bytes());

    // Add 64 bytes of garbage extended header
    mrc.extend_from_slice(&[0xAB; 64]);

    // Then the actual data
    mrc.extend_from_slice(&42.0f32.to_le_bytes());

    let map = mrc_to_density(&mrc).unwrap();
    assert_eq!(map.data[[0, 0, 0]], 42.0);
}

#[test]
fn origin_and_start_indices() {
    let nc = 2;
    let nr = 2;
    let ns = 2;
    let total = (nc * nr * ns) as usize;
    let data: Vec<f32> = vec![0.0; total];

    let mut mrc = make_test_mrc(&TestMrcParams {
        nc,
        nr,
        ns,
        mode: 2,
        mapc: 1,
        mapr: 2,
        maps: 3,
        data_values: &data,
    });

    // Set origin to (5.0, 10.0, 15.0)
    mrc[196..200].copy_from_slice(&5.0f32.to_le_bytes());
    mrc[200..204].copy_from_slice(&10.0f32.to_le_bytes());
    mrc[204..208].copy_from_slice(&15.0f32.to_le_bytes());

    // Set start indices to (1, 2, 3) -- these are in file axis order
    mrc[16..20].copy_from_slice(&1i32.to_le_bytes());
    mrc[20..24].copy_from_slice(&2i32.to_le_bytes());
    mrc[24..28].copy_from_slice(&3i32.to_le_bytes());

    let map = mrc_to_density(&mrc).unwrap();

    assert_eq!(map.nxstart, 1);
    assert_eq!(map.nystart, 2);
    assert_eq!(map.nzstart, 3);

    // voxel_size = 10/2 = 5 A per axis
    // grid_to_cartesian(0,0,0) = origin + start * voxel_size
    let pos = map.grid_to_cartesian(0, 0, 0);
    assert!((pos[0] - 10.0).abs() < 1e-6); // 5.0 + 1*5.0
    assert!((pos[1] - 20.0).abs() < 1e-6); // 10.0 + 2*5.0
    assert!((pos[2] - 30.0).abs() < 1e-6); // 15.0 + 3*5.0
}
