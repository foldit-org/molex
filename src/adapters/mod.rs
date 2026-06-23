//! Format adapters for parsing molecular structure files.

pub mod bcif;
pub mod cif;
pub mod dcd;
pub mod mrc;
pub mod pdb;

#[cfg(feature = "python")]
pub mod atomworks;
#[cfg(feature = "python")]
pub mod ml;
#[cfg(feature = "python")]
pub(crate) mod reconstruct;

// Entity-first re-exports.
pub use bcif::{
    bcif_file_to_all_models, bcif_file_to_entities, bcif_to_all_models,
    bcif_to_entities,
};
pub use cif::{
    mmcif_file_to_all_models, mmcif_file_to_entities, mmcif_str_to_all_models,
    mmcif_str_to_entities,
};
pub use dcd::{dcd_file_to_frames, DcdFrame, DcdHeader, DcdReader};
pub use mrc::{mrc_file_to_density, mrc_to_density};
pub use pdb::{
    assembly_to_pdb, entities_to_pdb, pdb_file_to_all_models,
    pdb_file_to_entities, pdb_str_to_all_models, pdb_str_to_entities,
    structure_file_to_entities,
};
