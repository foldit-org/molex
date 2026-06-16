//! DCD trajectory file parser (CHARMM/NAMD binary format).
//!
//! DCD files store molecular dynamics trajectories as Fortran-style binary
//! records:
//! - Header: magic `CORD`, frame/atom counts, timestep, title
//! - Per-frame: optional unit cell, then X\[N\], Y\[N\], Z\[N\] as f32 arrays
//!
//! No atom metadata is stored; topology must come from the loaded structure.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Parsed DCD file header.
#[derive(Debug, Clone)]
pub struct DcdHeader {
    /// Total number of trajectory frames.
    pub num_frames: u32,
    /// Number of atoms per frame.
    pub num_atoms: u32,
    /// Starting timestep number.
    pub start_step: u32,
    /// Interval between saved steps.
    pub step_interval: u32,
    /// Integration timestep in picoseconds.
    pub timestep: f32,
    /// Whether frames contain a unit cell record.
    pub has_extra_block: bool,
    /// Whether frames contain a 4th dimension.
    pub has_four_dims: bool,
    /// Title lines from the DCD header.
    pub title: String,
}

/// A single trajectory frame: flat arrays of x, y, z positions.
#[derive(Debug, Clone)]
pub struct DcdFrame {
    /// X coordinates for all atoms.
    pub x: Vec<f32>,
    /// Y coordinates for all atoms.
    pub y: Vec<f32>,
    /// Z coordinates for all atoms.
    pub z: Vec<f32>,
}

/// Streaming DCD reader over any `Read + Seek` source.
pub struct DcdReader<R: Read + Seek> {
    reader: R,
    /// Parsed header from the DCD file.
    pub header: DcdHeader,
    frames_read: u32,
}

impl DcdReader<BufReader<File>> {
    /// Open a DCD file and parse its header.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or the header is invalid.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::new(reader)
    }
}

impl<R: Read + Seek> DcdReader<R> {
    /// Parse header from any seekable reader.
    ///
    /// # Errors
    ///
    /// Returns an error if the header cannot be parsed from the reader.
    pub fn new(mut reader: R) -> io::Result<Self> {
        let header = parse_header(&mut reader)?;
        Ok(Self {
            reader,
            header,
            frames_read: 0,
        })
    }

    /// Read the next frame. Returns `None` at EOF.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame data is malformed or unreadable.
    pub fn read_frame(&mut self) -> io::Result<Option<DcdFrame>> {
        if self.frames_read >= self.header.num_frames {
            return Ok(None);
        }

        let n = self.header.num_atoms as usize;

        // Skip unit cell record if present
        if self.header.has_extra_block {
            skip_fortran_record(&mut self.reader)?;
        }

        // Read X, Y, Z coordinate arrays
        let x = read_f32_fortran_record(&mut self.reader, n)?;
        let y = read_f32_fortran_record(&mut self.reader, n)?;
        let z = read_f32_fortran_record(&mut self.reader, n)?;

        // Skip 4th dimension if present
        if self.header.has_four_dims {
            skip_fortran_record(&mut self.reader)?;
        }

        self.frames_read += 1;
        Ok(Some(DcdFrame { x, y, z }))
    }

    /// Read all remaining frames into memory.
    ///
    /// # Errors
    ///
    /// Returns an error if any frame is malformed or unreadable.
    pub fn read_all_frames(&mut self) -> io::Result<Vec<DcdFrame>> {
        let remaining = (self.header.num_frames - self.frames_read) as usize;
        let mut frames = Vec::with_capacity(remaining);
        while let Some(frame) = self.read_frame()? {
            frames.push(frame);
        }
        Ok(frames)
    }
}

/// Convenience: open a DCD file, parse header and all frames.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or any data is malformed.
pub fn dcd_file_to_frames(
    path: &Path,
) -> io::Result<(DcdHeader, Vec<DcdFrame>)> {
    let mut reader = DcdReader::open(path)?;
    let header = reader.header.clone();
    let frames = reader.read_all_frames()?;
    Ok((header, frames))
}

// Internal parsing helpers

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

/// Read a Fortran record: i32 size, N bytes payload, i32 size (must match).
fn read_fortran_record(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let size = read_i32(r)?;
    if size < 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("negative Fortran record size: {size}"),
        ));
    }
    let size = size.cast_unsigned() as usize;
    let mut buf = vec![0u8; size];
    r.read_exact(&mut buf)?;
    let end_size = read_i32(r)?.cast_unsigned() as usize;
    if size != end_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Fortran record size mismatch: start={size}, end={end_size}"
            ),
        ));
    }
    Ok(buf)
}

/// Skip over a Fortran record without allocating.
fn skip_fortran_record(r: &mut (impl Read + Seek)) -> io::Result<()> {
    let size = read_i32(r)?;
    if size < 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("negative Fortran record size: {size}"),
        ));
    }
    let _ = r.seek(SeekFrom::Current(i64::from(size)))?;
    let end_size = read_i32(r)?;
    if size != end_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Fortran record size mismatch: start={size}, end={end_size}"
            ),
        ));
    }
    Ok(())
}

/// Read a Fortran record containing exactly `n` f32 values.
fn read_f32_fortran_record(
    r: &mut impl Read,
    n: usize,
) -> io::Result<Vec<f32>> {
    let data = read_fortran_record(r)?;
    let expected = n * 4;
    if data.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "f32 record size mismatch: expected {expected} bytes ({n} \
                 floats), got {}",
                data.len()
            ),
        ));
    }
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(floats)
}

/// Parse the 84-byte control block (record 1) into header fields.
/// Returns (num_frames, start_step, step_interval, timestep,
/// has_extra_block, has_four_dims).
fn parse_control_block(
    rec1: &[u8],
) -> io::Result<(u32, u32, u32, f32, bool, bool)> {
    if rec1.len() != 84 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "DCD header record 1: expected 84 bytes, got {}",
                rec1.len()
            ),
        ));
    }
    let magic = &rec1[0..4];
    if magic != b"CORD" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("DCD magic: expected CORD, got {magic:?}"),
        ));
    }
    let icntrl: Vec<i32> = rec1[4..84]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let timestep = f32::from_bits(icntrl[9].cast_unsigned());
    Ok((
        icntrl[0].cast_unsigned(),
        icntrl[1].cast_unsigned(),
        icntrl[2].cast_unsigned(),
        timestep,
        icntrl[10] != 0,
        icntrl[11] != 0,
    ))
}

/// Parse the title block (record 2) into a newline-joined string.
fn parse_title_block(rec2: &[u8]) -> String {
    if rec2.len() < 4 {
        return String::new();
    }
    let ntitle = i32::from_le_bytes([rec2[0], rec2[1], rec2[2], rec2[3]])
        .cast_unsigned() as usize;
    let mut lines = Vec::with_capacity(ntitle);
    for i in 0..ntitle {
        let start = 4 + i * 80;
        let end = start + 80;
        if end <= rec2.len() {
            let line =
                String::from_utf8_lossy(&rec2[start..end]).trim().to_owned();
            if !line.is_empty() {
                lines.push(line);
            }
        }
    }
    lines.join("\n")
}

/// Parse the atom-count record (record 3).
fn parse_atom_count_record(rec3: &[u8]) -> io::Result<u32> {
    if rec3.len() != 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("DCD natom record: expected 4 bytes, got {}", rec3.len()),
        ));
    }
    Ok(
        i32::from_le_bytes([rec3[0], rec3[1], rec3[2], rec3[3]])
            .cast_unsigned(),
    )
}

/// Parse the DCD header (first three Fortran records).
fn parse_header(r: &mut impl Read) -> io::Result<DcdHeader> {
    let rec1 = read_fortran_record(r)?;
    let (
        num_frames,
        start_step,
        step_interval,
        timestep,
        has_extra_block,
        has_four_dims,
    ) = parse_control_block(&rec1)?;

    let rec2 = read_fortran_record(r)?;
    let title = parse_title_block(&rec2);

    let rec3 = read_fortran_record(r)?;
    let num_atoms = parse_atom_count_record(&rec3)?;

    Ok(DcdHeader {
        num_frames,
        num_atoms,
        start_step,
        step_interval,
        timestep,
        has_extra_block,
        has_four_dims,
        title,
    })
}
