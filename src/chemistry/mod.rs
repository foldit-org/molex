//! Chemistry tables: amino acids, nucleotides, and atom name handling.
//!
//! These modules own the static residue chemistry that consumers
//! (entity construction, bond inference) build on. The data here is
//! pure: no entity references, no positions, no I/O.

pub mod amino_acids;
pub mod atom_name;
pub mod completion;
mod completion_data;
pub mod nucleotides;
pub mod variant;

pub use amino_acids::AminoAcid;
pub use atom_name::{is_protein_backbone_atom_name, AtomName};
pub use completion::{ResidueTemplate, TemplateAtom};
pub use nucleotides::Nucleotide;
pub use variant::{ProtonationState, VariantTag};
