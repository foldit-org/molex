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
    /// Provenance: `true` if parsed from the input, `false` if fabricated by
    /// completion. Metadata only, never part of atom identity.
    pub observed: bool,
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

/// Struct-of-arrays storage for an entity's atoms.
///
/// Each [`Atom`] field becomes a parallel column. The columns are kept the
/// same length and in the same order, so index `i` addresses one atom
/// across all seven.
///
/// Built from a `Vec<Atom>` via [`AtomColumns::from_atoms`]; that transpose
/// is the bit-identical contract — the column at index `i` carries exactly
/// the bytes the `i`-th `Atom` held. Single atoms are read back with
/// [`AtomColumns::gather`] (by value) or [`AtomColumns::atom_ref`]
/// (borrowed cells).
#[derive(Debug, Clone)]
pub struct AtomColumns {
    /// 3D positions in angstroms.
    pub position: Vec<Vec3>,
    /// Crystallographic occupancies (0.0 to 1.0).
    pub occupancy: Vec<f32>,
    /// Temperature factors (B-factor) in square angstroms.
    pub b_factor: Vec<f32>,
    /// Chemical elements.
    pub element: Vec<Element>,
    /// PDB-style 4-character atom names.
    pub name: Vec<[u8; 4]>,
    /// Formal charges (signed). 0 means neutral.
    pub formal_charge: Vec<i8>,
    /// Per-atom provenance: `true` parsed, `false` fabricated by completion.
    pub observed: Vec<bool>,
}

impl AtomColumns {
    /// Number of atoms (the shared column length).
    #[must_use]
    pub fn len(&self) -> usize {
        self.position.len()
    }

    /// Whether this column set holds no atoms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.position.is_empty()
    }

    /// Transpose a `Vec<Atom>` into parallel columns, field by field in
    /// struct-declaration order. The bit-identical boundary: every column
    /// cell carries the matching `Atom` field verbatim.
    #[must_use]
    pub fn from_atoms(atoms: Vec<Atom>) -> Self {
        let mut out = Self {
            position: Vec::with_capacity(atoms.len()),
            occupancy: Vec::with_capacity(atoms.len()),
            b_factor: Vec::with_capacity(atoms.len()),
            element: Vec::with_capacity(atoms.len()),
            name: Vec::with_capacity(atoms.len()),
            formal_charge: Vec::with_capacity(atoms.len()),
            observed: Vec::with_capacity(atoms.len()),
        };
        for atom in atoms {
            out.position.push(atom.position);
            out.occupancy.push(atom.occupancy);
            out.b_factor.push(atom.b_factor);
            out.element.push(atom.element);
            out.name.push(atom.name);
            out.formal_charge.push(atom.formal_charge);
            out.observed.push(atom.observed);
        }
        out
    }

    /// Gather every atom back into a `Vec<Atom>`, the inverse of
    /// [`Self::from_atoms`]. Reconstructs each `Atom` in struct-field order.
    #[must_use]
    pub fn to_atoms(&self) -> Vec<Atom> {
        (0..self.len()).map(|i| self.gather(i)).collect()
    }

    /// Gather one atom by index into a by-value [`Atom`]. Panics if `i` is
    /// out of bounds, matching slice indexing.
    #[must_use]
    pub fn gather(&self, i: usize) -> Atom {
        Atom {
            position: self.position[i],
            occupancy: self.occupancy[i],
            b_factor: self.b_factor[i],
            element: self.element[i],
            name: self.name[i],
            formal_charge: self.formal_charge[i],
            observed: self.observed[i],
        }
    }

    /// Borrow one atom's six cells as an [`AtomRef`]. Panics if `i` is out
    /// of bounds.
    #[must_use]
    pub fn atom_ref(&self, i: usize) -> AtomRef<'_> {
        AtomRef {
            position: &self.position[i],
            occupancy: &self.occupancy[i],
            b_factor: &self.b_factor[i],
            element: &self.element[i],
            name: &self.name[i],
            formal_charge: &self.formal_charge[i],
        }
    }

    /// Replace `range` across all six columns with `atoms`, in lockstep.
    /// Equivalent to splicing the same range of a `Vec<Atom>` and re-
    /// transposing: each column drops its `range` cells and inserts the
    /// matching field of every atom, preserving element order.
    ///
    /// Each `Vec::splice` returns a drain iterator that performs the
    /// replacement only once consumed/dropped, so each is dropped eagerly.
    pub fn splice(&mut self, range: std::ops::Range<usize>, atoms: &[Atom]) {
        drop(
            self.position
                .splice(range.clone(), atoms.iter().map(|a| a.position)),
        );
        drop(
            self.occupancy
                .splice(range.clone(), atoms.iter().map(|a| a.occupancy)),
        );
        drop(
            self.b_factor
                .splice(range.clone(), atoms.iter().map(|a| a.b_factor)),
        );
        drop(
            self.element
                .splice(range.clone(), atoms.iter().map(|a| a.element)),
        );
        drop(
            self.name
                .splice(range.clone(), atoms.iter().map(|a| a.name)),
        );
        drop(
            self.formal_charge
                .splice(range.clone(), atoms.iter().map(|a| a.formal_charge)),
        );
        drop(
            self.observed
                .splice(range, atoms.iter().map(|a| a.observed)),
        );
    }
}

/// Pack a token into an `N`-byte, left-justified, space-padded buffer
/// (truncating inputs longer than `N`). The shared name convention for atom
/// names (`N = 4`) and residue names (`N = 3`).
#[must_use]
pub(crate) fn pad_name<const N: usize>(token: &str) -> [u8; N] {
    let mut buf = [b' '; N];
    for (slot, byte) in buf.iter_mut().zip(token.bytes().take(N)) {
        *slot = byte;
    }
    buf
}

/// Pack an atom-name string into the 4-byte, left-justified, space-padded
/// buffer every parser uses for [`Atom::name`] (PDB-column convention).
#[must_use]
pub(crate) fn pad_atom_name(name: &str) -> [u8; 4] {
    pad_name::<4>(name)
}
