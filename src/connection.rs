//! Inter-entity rendering connections on [`Assembly`].
//!
//! A connection is atom-connecting geometry the viewer draws on top of
//! the structure: hydrogen bonds, disulfides, clashes, and (eventually)
//! pull bands. Connections are rendering metadata, not structural source
//! of truth; they are populated by the assembly's owner (which selects a
//! provider) rather than by `recompute_derived`, and stored keyed by
//! [`ConnectionType`]. The category key selects the renderer; there is no
//! per-connection style.

use crate::atom_id::AtomId;

/// Category of an atom-connecting geometry. The category KEY selects the
/// renderer; there is no per-connection style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConnectionType {
    /// Backbone hydrogen bonds.
    HBond,
    /// Cys-Cys disulfide bridges.
    Disulfide,
    /// Steric clashes between atoms.
    Clash,
    /// Pull bands (e.g. a space-pull target). Not yet produced.
    Band,
}

/// One endpoint of a connection: either a specific atom (stable across
/// coordinate-only changes) or a fixed point in space (e.g. a space-pull
/// target).
///
/// `Anchor` has no producer yet (the band path is not wired); it is
/// defined here so the endpoint vocabulary is complete and a fixed-point
/// endpoint needs no later type change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AtomEnd {
    /// A specific atom, stable across coordinate-only changes.
    Atom(AtomId),
    /// A fixed point in space. No producer yet (the band path is not
    /// wired); defined so a fixed-point endpoint needs no later type
    /// change.
    Anchor(glam::Vec3),
}

/// One atom-connecting geometry: an unordered pair of endpoints plus an
/// optional per-connection intensity.
///
/// `magnitude` is a scalar a renderer may use to modulate its visual; its
/// meaning is per `ConnectionType` (e.g. clash severity scales the
/// lightning bolt amplitude). It is NOT a style/renderer selector - the
/// `ConnectionType` key still chooses the renderer. `None` for connection
/// types that carry no intensity (hydrogen bonds and disulfides today).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomLink {
    /// First endpoint.
    pub a: AtomEnd,
    /// Second endpoint.
    pub b: AtomEnd,
    /// Optional per-connection intensity; interpretation is per
    /// `ConnectionType`.
    pub magnitude: Option<f32>,
}

impl AtomLink {
    /// A connection between two endpoints with no intensity.
    #[must_use]
    pub fn new(a: AtomEnd, b: AtomEnd) -> Self {
        Self {
            a,
            b,
            magnitude: None,
        }
    }

    /// A connection between two endpoints carrying an intensity scalar.
    #[must_use]
    pub fn with_magnitude(a: AtomEnd, b: AtomEnd, magnitude: f32) -> Self {
        Self {
            a,
            b,
            magnitude: Some(magnitude),
        }
    }
}
