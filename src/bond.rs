//! Covalent bonds between atoms identified by [`AtomId`].
//!
//! [`CovalentBond`] is role-free: callers read the bond list (e.g.
//! `ProteinEntity::bonds`) and partition by endpoint role themselves, and
//! disulfides are detected on demand (`detect_disulfides` /
//! `detect_fallback_connections`) rather than tagged on the bond itself.

use crate::analysis::BondOrder;
use crate::atom_id::AtomId;

/// A covalent bond between two atoms.
///
/// Endpoints are unordered: `CovalentBond { a, b, .. }` and
/// `CovalentBond { a: b, b: a, .. }` are not `==`-equal but represent
/// the same chemistry. Consumers that need canonical ordering should
/// sort the endpoints themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CovalentBond {
    /// First endpoint.
    pub a: AtomId,
    /// Second endpoint.
    pub b: AtomId,
    /// Bond order.
    pub order: BondOrder,
}
