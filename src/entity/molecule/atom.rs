//! Core atom type: position and chemistry.

use glam::Vec3;

use crate::element::Element;

/// A single atom with position, chemistry, and crystallographic data.
///
/// Residue and chain context lives on the entity that contains the atom.
#[derive(Debug, Clone)]
pub struct Atom {
    /// 3D position in angstroms.
    pub position: Vec3,
    /// Crystallographic occupancy (0.0 to 1.0).
    pub occupancy: f32,
    /// Temperature factor (B-factor) in square angstroms.
    pub b_factor: f32,
    /// Chemical element.
    pub element: Element,
    /// PDB-style 4-character atom name (e.g. b"CA  ", b"N   ").
    pub name: [u8; 4],
    /// Formal charge (signed). 0 means neutral.
    pub formal_charge: i8,
}

/// Pack an atom-name string into the 4-byte, left-justified, space-padded
/// buffer every parser uses for [`Atom::name`] (PDB-column convention).
/// Inputs longer than 4 bytes are truncated to the first 4.
#[must_use]
pub(crate) fn pad_atom_name(name: &str) -> [u8; 4] {
    let mut buf = [b' '; 4];
    for (slot, byte) in buf.iter_mut().zip(name.bytes().take(4)) {
        *slot = byte;
    }
    buf
}
