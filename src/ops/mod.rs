//! Operations on molecular data.

/// Typed Assembly edits + apply path.
pub mod edit;
/// Adapter / wire error type.
pub mod error;
pub mod transform;
/// Assembly + delta binary wire format encoder/decoder.
pub mod wire;

pub use edit::{AssemblyEdit, BulkEditError, EditError};
pub use transform::kabsch_alignment;
