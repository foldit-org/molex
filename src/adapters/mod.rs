//! Format adapters for parsing molecular structure files.

pub mod bcif;
pub mod cif;
pub mod dcd;
pub mod mrc;
pub mod pdb;
/// Native, key-bearing flat atom interchange and the single partition core.
pub mod table;

#[cfg(feature = "python")]
pub mod atomworks;

// Entity-first re-exports.
pub use bcif::{bcif_file_to_all_models, bcif_to_all_models};
pub use cif::{mmcif_file_to_all_models, mmcif_str_to_all_models};
pub use dcd::{dcd_file_to_frames, DcdFrame, DcdHeader, DcdReader};
pub use mrc::{mrc_file_to_density, mrc_to_density};
pub use pdb::{pdb_file_to_all_models, pdb_str_to_all_models};
