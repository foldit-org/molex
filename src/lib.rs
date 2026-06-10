//! Molecular structure conversion library for parsing, transforming, and
//! serializing molecular coordinate data.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use molex::{MoleculeEntity, MoleculeType, SSType};
//! use molex::adapters::pdb::structure_file_to_entities;
//!
//! let entities = structure_file_to_entities("1ubq.pdb")?;
//! for e in &entities {
//!     println!("{:?}: {} atoms", e.molecule_type(), e.atom_count());
//! }
//! ```

pub mod adapters;
/// Structural analysis: bond detection and secondary structure classification.
pub mod analysis;
pub mod assembly;
/// Cross-cutting atom identifier (`AtomId`).
pub mod atom_id;
/// Cross-cutting covalent bond (`CovalentBond`).
pub mod bond;
/// Static chemistry tables: amino acids, nucleotides, atom names.
pub mod chemistry;
pub mod element;
pub mod entity;
pub mod ops;

#[cfg(feature = "c-api")]
pub mod c_api;

#[cfg(feature = "python")]
pub mod python;

// -- Entity-first public API ---------------------------------------------
// The most commonly used types, re-exported at the crate root.

pub use analysis::{detect_disulfides, BondOrder, HBond, SSType};
pub use assembly::{Assembly, CoordinateSnapshot};
pub use atom_id::AtomId;
pub use bond::CovalentBond;
pub use element::Element;
pub use entity::molecule::atom::Atom;
pub use entity::molecule::id::EntityId;
pub use entity::molecule::protein::{ResidueBackbone, Sidechain};
pub use entity::molecule::{
    EntityKind, MoleculeEntity, MoleculeType, NucleotideRing,
};
pub use ops::codec::AdapterError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule(name = "molex")]
fn molex(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // ASSEM01-based IO helpers (replaces the retired COORDS01 surface).
    m.add_function(wrap_pyfunction!(python::pdb_to_assembly_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(python::mmcif_to_assembly_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(python::assembly_bytes_to_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(python::deserialize_assembly_bytes, m)?)?;
    // AtomArray / AtomArrayPlus converters (entity-aware, preserves molecule
    // types and bonds)
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::entities_to_atom_array,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::entities_to_atom_array_plus,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::atom_array_to_entities,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::assembly_bytes_to_atom_array,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::assembly_bytes_to_atom_array_plus,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::entities_to_atom_array_parsed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        adapters::atomworks::parse_file_to_entities,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(adapters::atomworks::parse_file_full, m)?)?;

    // Handle-based edit / DELTA01 surface (parallels c_api::edit).
    m.add_class::<python::PyAssembly>()?;
    m.add_class::<python::PyEditList>()?;
    m.add_class::<python::PyVariant>()?;
    m.add_class::<python::PyAtomRow>()?;
    m.add_class::<python::PySetEntityCoordsView>()?;
    m.add_class::<python::PySetResidueCoordsView>()?;
    m.add_class::<python::PyMutateResidueView>()?;
    m.add_class::<python::PySetVariantsView>()?;

    Ok(())
}
