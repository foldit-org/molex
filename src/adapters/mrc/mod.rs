//! MRC/CCP4 density map parser.
//!
//! Parses the MRC2014 / CCP4 binary format: a 1024-byte header followed by a
//! 3D grid of density values. Supports modes 0 (i8), 1 (i16), 2 (f32), and
//! 6 (u16), with automatic endianness detection via MACHST.

use std::fs;
use std::path::Path;

use ndarray::Array3;

use crate::entity::surface::density::{Density, DensityError, VoxelGrid};

const HEADER_SIZE: usize = 1024;
const MAP_MAGIC: &[u8; 4] = b"MAP ";

fn word4(b: &[u8]) -> [u8; 4] {
    [b[0], b[1], b[2], b[3]]
}
fn read_i32_at(b: &[u8], le: bool) -> i32 {
    let w = word4(b);
    if le {
        i32::from_le_bytes(w)
    } else {
        i32::from_be_bytes(w)
    }
}
fn read_f32_at(b: &[u8], le: bool) -> f32 {
    let w = word4(b);
    if le {
        f32::from_le_bytes(w)
    } else {
        f32::from_be_bytes(w)
    }
}

/// Parse an MRC/CCP4 density map from a file path.
///
/// # Errors
///
/// Returns [`DensityError`] if the file cannot be read or does not contain a
/// valid MRC/CCP4 density map.
pub fn mrc_file_to_density(path: &Path) -> Result<Density, DensityError> {
    let bytes = fs::read(path)?;
    mrc_to_density(&bytes)
}

/// Validated MRC header fields.
struct MrcHeader {
    nc: i32,
    nr: i32,
    ns: i32,
    mode: i32,
    ncstart: i32,
    nrstart: i32,
    nsstart: i32,
    mx: i32,
    my: i32,
    mz: i32,
    cell_dims: [f32; 3],
    cell_angles: [f32; 3],
    mapc: i32,
    mapr: i32,
    maps: i32,
    dmin: f32,
    dmax: f32,
    dmean: f32,
    space_group: u32,
    nsymbt: i32,
    origin: [f32; 3],
    rms: f32,
    little_endian: bool,
}

/// Validate raw header bytes: size, magic, endianness.
fn validate_header(bytes: &[u8]) -> Result<(bool, &[u8]), DensityError> {
    if bytes.len() < HEADER_SIZE {
        return Err(DensityError::InvalidFormat(format!(
            "file too small: {} bytes (need {HEADER_SIZE})",
            bytes.len()
        )));
    }
    let h = &bytes[..HEADER_SIZE];
    if &h[208..212] != MAP_MAGIC {
        return Err(DensityError::InvalidFormat(format!(
            "missing MAP magic at offset 208: got {:?}",
            &h[208..212]
        )));
    }
    Ok((detect_endianness(h)?, h))
}

/// Check that three i32 values are positive.
fn require_positive(vals: [i32; 3], msg: &str) -> Result<(), DensityError> {
    if vals.iter().any(|&v| v <= 0) {
        return Err(DensityError::InvalidFormat(format!(
            "non-positive {msg}: {vals:?}"
        )));
    }
    Ok(())
}

/// Check that axis mapping is a valid permutation of {1, 2, 3}.
fn validate_axes(mapc: i32, mapr: i32, maps: i32) -> Result<(), DensityError> {
    let ok = (1..=3).contains(&mapc)
        && (1..=3).contains(&mapr)
        && (1..=3).contains(&maps)
        && mapc != mapr
        && mapc != maps
        && mapr != maps;
    if !ok {
        return Err(DensityError::InvalidFormat(format!(
            "invalid axes: MAPC={mapc}, MAPR={mapr}, MAPS={maps}"
        )));
    }
    Ok(())
}

/// Parse and validate the 1024-byte MRC header.
#[allow(clippy::cast_sign_loss, reason = "validated positive")]
fn parse_mrc_header(bytes: &[u8]) -> Result<MrcHeader, DensityError> {
    let (le, h) = validate_header(bytes)?;
    let i = |off: usize| read_i32_at(&h[off..off + 4], le);
    let f = |off: usize| read_f32_at(&h[off..off + 4], le);
    let (nc, nr, ns) = (i(0), i(4), i(8));
    require_positive([nc, nr, ns], "grid (NC, NR, NS)")?;
    let mode = i(12);
    if !matches!(mode, 0 | 1 | 2 | 6) {
        return Err(DensityError::UnsupportedMode(mode));
    }
    let (mx, my, mz) = (i(28), i(32), i(36));
    require_positive([mx, my, mz], "sampling (MX, MY, MZ)")?;
    let (mapc, mapr, maps) = (i(64), i(68), i(72));
    validate_axes(mapc, mapr, maps)?;
    let nsymbt = i(92);
    if nsymbt < 0 {
        return Err(DensityError::InvalidFormat(format!(
            "negative extended header size: {nsymbt}"
        )));
    }
    Ok(MrcHeader {
        nc,
        nr,
        ns,
        mode,
        ncstart: i(16),
        nrstart: i(20),
        nsstart: i(24),
        mx,
        my,
        mz,
        cell_dims: [f(40), f(44), f(48)],
        cell_angles: [f(52), f(56), f(60)],
        mapc,
        mapr,
        maps,
        dmin: f(76),
        dmax: f(80),
        dmean: f(84),
        space_group: i(88).cast_unsigned(),
        nsymbt,
        origin: [f(196), f(200), f(204)],
        rms: f(216),
        little_endian: le,
    })
}

/// Parse an MRC/CCP4 density map from raw bytes.
///
/// # Errors
///
/// Returns [`DensityError`] if the bytes do not represent a valid MRC/CCP4
/// density map (e.g. missing magic, unsupported mode, truncated data).
#[allow(clippy::cast_sign_loss, reason = "header validated positive")]
pub fn mrc_to_density(bytes: &[u8]) -> Result<Density, DensityError> {
    let h = parse_mrc_header(bytes)?;

    let data_offset = HEADER_SIZE + h.nsymbt as usize;
    let dims = [h.nc as usize, h.nr as usize, h.ns as usize];
    let total = dims[0] * dims[1] * dims[2];
    let flat = read_density_values(
        &bytes[data_offset..],
        h.mode,
        total,
        h.little_endian,
    )?;

    let axes = [h.mapc as usize, h.mapr as usize, h.maps as usize];
    let (data, nx, ny, nz) = reorder_axes(flat, dims, axes)?;

    let fs = [h.ncstart, h.nrstart, h.nsstart];
    let (mc, mr, ms) = (h.mapc as usize, h.mapr as usize, h.maps as usize);
    Ok(Density {
        grid: VoxelGrid {
            nx,
            ny,
            nz,
            nxstart: fs[axis_file_index(mc, mr, ms, 1)],
            nystart: fs[axis_file_index(mc, mr, ms, 2)],
            nzstart: fs[axis_file_index(mc, mr, ms, 3)],
            mx: h.mx as usize,
            my: h.my as usize,
            mz: h.mz as usize,
            cell_dims: h.cell_dims,
            cell_angles: h.cell_angles,
            origin: h.origin,
            data,
        },
        dmin: h.dmin,
        dmax: h.dmax,
        dmean: h.dmean,
        rms: h.rms,
        space_group: h.space_group,
    })
}

/// Serialize a [`Density`] to MRC/CCP4 bytes (MODE 2, little-endian).
///
/// The inverse of [`mrc_to_density`]: writes a 1024-byte header in canonical
/// axis order (MAPC/MAPR/MAPS = 1/2/3) followed by the f32 grid in X-fastest
/// layout, so feeding the result back through [`mrc_to_density`] reconstructs
/// an identical [`Density`]. The layout also satisfies Rosetta's
/// `readMRCandResize`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "grid dims and space group fit the header's i32 fields"
)]
pub fn density_to_mrc_bytes(density: &Density) -> Vec<u8> {
    let g = &density.grid;
    let mut buf = vec![0u8; HEADER_SIZE];

    let put_i32 = |buf: &mut [u8], off: usize, v: i32| {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    };
    let put_f32 = |buf: &mut [u8], off: usize, v: f32| {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    };

    put_i32(&mut buf, 0, g.nx as i32);
    put_i32(&mut buf, 4, g.ny as i32);
    put_i32(&mut buf, 8, g.nz as i32);
    put_i32(&mut buf, 12, 2); // MODE = 2 (f32)
    put_i32(&mut buf, 16, g.nxstart);
    put_i32(&mut buf, 20, g.nystart);
    put_i32(&mut buf, 24, g.nzstart);
    put_i32(&mut buf, 28, g.mx as i32);
    put_i32(&mut buf, 32, g.my as i32);
    put_i32(&mut buf, 36, g.mz as i32);
    put_f32(&mut buf, 40, g.cell_dims[0]);
    put_f32(&mut buf, 44, g.cell_dims[1]);
    put_f32(&mut buf, 48, g.cell_dims[2]);
    put_f32(&mut buf, 52, g.cell_angles[0]);
    put_f32(&mut buf, 56, g.cell_angles[1]);
    put_f32(&mut buf, 60, g.cell_angles[2]);
    put_i32(&mut buf, 64, 1); // MAPC = X
    put_i32(&mut buf, 68, 2); // MAPR = Y
    put_i32(&mut buf, 72, 3); // MAPS = Z
    put_f32(&mut buf, 76, density.dmin);
    put_f32(&mut buf, 80, density.dmax);
    put_f32(&mut buf, 84, density.dmean);
    put_i32(&mut buf, 88, density.space_group as i32);
    put_i32(&mut buf, 92, 0); // NSYMBT
    put_f32(&mut buf, 196, g.origin[0]);
    put_f32(&mut buf, 200, g.origin[1]);
    put_f32(&mut buf, 204, g.origin[2]);
    buf[208..212].copy_from_slice(MAP_MAGIC);
    buf[212..216].copy_from_slice(&[0x44, 0x41, 0x00, 0x00]);
    put_f32(&mut buf, 216, density.rms);

    // Data block: X fastest, then Y, then Z. This is the inverse of
    // `reorder_axes` for the identity axis map (flat index = z*ny*nx + y*nx +
    // x).
    buf.reserve(g.nx * g.ny * g.nz * 4);
    for z in 0..g.nz {
        for y in 0..g.ny {
            for x in 0..g.nx {
                buf.extend_from_slice(&g.data[[x, y, z]].to_le_bytes());
            }
        }
    }
    buf
}

/// Reorder flat file data into an `[X, Y, Z]` `Array3`.
#[allow(clippy::cast_sign_loss, reason = "axis values 1-3, validated")]
fn reorder_axes(
    flat: Vec<f32>,
    dims: [usize; 3],
    axes: [usize; 3],
) -> Result<(Array3<f32>, usize, usize, usize), DensityError> {
    let [nc, nr, ns] = dims;
    let [mapc, mapr, maps] = axes;
    let mut sz = [0usize; 3];
    sz[mapc - 1] = nc;
    sz[mapr - 1] = nr;
    sz[maps - 1] = ns;
    let [nx, ny, nz] = sz;

    let arr = if mapc == 1 && mapr == 2 && maps == 3 {
        let a = Array3::from_shape_vec((ns, nr, nc), flat).map_err(|e| {
            DensityError::InvalidFormat(format!("reshape: {e}"))
        })?;
        a.permuted_axes([2, 1, 0]).as_standard_layout().to_owned()
    } else {
        let mut a = Array3::<f32>::zeros((nx, ny, nz));
        for s in 0..ns {
            for r in 0..nr {
                let base = (s * nr + r) * nc;
                for c in 0..nc {
                    let mut xyz = [0usize; 3];
                    xyz[mapc - 1] = c;
                    xyz[mapr - 1] = r;
                    xyz[maps - 1] = s;
                    a[xyz] = flat[base + c];
                }
            }
        }
        a
    };
    Ok((arr, nx, ny, nz))
}

/// Given axis mapping values, return the file dimension index (0=col, 1=row,
/// 2=section) for a given spatial axis (1=X, 2=Y, 3=Z).
fn axis_file_index(
    mapc: usize,
    mapr: usize,
    maps: usize,
    spatial: usize,
) -> usize {
    if mapc == spatial {
        0
    } else if mapr == spatial {
        1
    } else {
        debug_assert_eq!(maps, spatial);
        2
    }
}

/// Detect endianness from MACHST (byte offset 212-213).
fn detect_endianness(header: &[u8]) -> Result<bool, DensityError> {
    let machst = header[212];
    match machst {
        0x44 => Ok(true),  // little-endian
        0x11 => Ok(false), // big-endian
        _ => {
            // Fallback: try LE, check if MODE is valid
            let mode_le = i32::from_le_bytes([
                header[12], header[13], header[14], header[15],
            ]);
            if matches!(mode_le, 0 | 1 | 2 | 6) {
                Ok(true)
            } else {
                let mode_be = i32::from_be_bytes([
                    header[12], header[13], header[14], header[15],
                ]);
                if matches!(mode_be, 0 | 1 | 2 | 6) {
                    Ok(false)
                } else {
                    Err(DensityError::InvalidFormat(format!(
                        "cannot determine endianness: MACHST={machst:#x}, \
                         MODE(LE)={mode_le}, MODE(BE)={mode_be}"
                    )))
                }
            }
        }
    }
}

/// Read and convert density data to f32 based on MODE.
fn read_density_values(
    data: &[u8],
    mode: i32,
    count: usize,
    le: bool,
) -> Result<Vec<f32>, DensityError> {
    let bpv = match mode {
        0 => 1,
        1 | 6 => 2,
        2 => 4,
        _ => return Err(DensityError::UnsupportedMode(mode)),
    };
    let needed = count * bpv;
    if data.len() < needed {
        return Err(DensityError::InvalidFormat(format!(
            "not enough data: need {needed} bytes for {count} voxels (mode \
             {mode}), got {}",
            data.len()
        )));
    }
    let data = &data[..needed];
    let values = match mode {
        0 => data.iter().map(|&b| f32::from(b.cast_signed())).collect(),
        1 => read_chunks(data, |b| {
            f32::from(if le {
                i16::from_le_bytes(b)
            } else {
                i16::from_be_bytes(b)
            })
        }),
        2 => read_chunks(data, |b| {
            if le {
                f32::from_le_bytes(b)
            } else {
                f32::from_be_bytes(b)
            }
        }),
        6 => read_chunks(data, |b| {
            f32::from(if le {
                u16::from_le_bytes(b)
            } else {
                u16::from_be_bytes(b)
            })
        }),
        _ => unreachable!(),
    };
    Ok(values)
}

/// Decode a byte slice in fixed-size chunks via a conversion function.
fn read_chunks<const N: usize>(
    data: &[u8],
    convert: impl Fn([u8; N]) -> f32,
) -> Vec<f32> {
    data.chunks_exact(N)
        .map(|c| {
            let mut buf = [0u8; N];
            buf.copy_from_slice(c);
            convert(buf)
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
#[path = "tests.rs"]
mod tests;
