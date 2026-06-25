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

/// A borrowed, layout-agnostic view of one atom's fields.
///
/// Every field is a reference so the same view can front either the
/// array-of-structs storage (one `&Atom`) or a struct-of-arrays storage
/// (cells gathered from parallel columns). Readers that touch several
/// fields of one atom take an `AtomRef` instead of `&Atom`, decoupling
/// them from the underlying storage layout.
#[derive(Debug, Clone, Copy)]
pub struct AtomRef<'a> {
    /// 3D position in angstroms.
    pub position: &'a Vec3,
    /// Crystallographic occupancy (0.0 to 1.0).
    pub occupancy: &'a f32,
    /// Temperature factor (B-factor) in square angstroms.
    pub b_factor: &'a f32,
    /// Chemical element.
    pub element: &'a Element,
    /// PDB-style 4-character atom name.
    pub name: &'a [u8; 4],
    /// Formal charge (signed). 0 means neutral.
    pub formal_charge: &'a i8,
}

impl<'a> AtomRef<'a> {
    /// View an array-of-structs [`Atom`] as borrowed cells.
    #[must_use]
    pub fn from_atom(atom: &'a Atom) -> Self {
        Self {
            position: &atom.position,
            occupancy: &atom.occupancy,
            b_factor: &atom.b_factor,
            element: &atom.element,
            name: &atom.name,
            formal_charge: &atom.formal_charge,
        }
    }
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
