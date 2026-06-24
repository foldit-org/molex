//! Refusal helpers shared by the fast and DOM ingest paths.

use crate::ops::codec::AdapterError;

/// Refusal for files containing multiple `data_*` blocks.
///
/// Production coordinate files are single-block; multi-block inputs come
/// from chemical-component dictionaries (`components.cif`) which the
/// coordinate adapter intentionally does not handle.
pub(super) fn multi_block_error(count: usize) -> AdapterError {
    AdapterError::InvalidFormat(format!(
        "mmCIF input has {count} data blocks; the coordinate adapter consumes \
         single-block files only. Use a chemical-component dictionary adapter \
         for components.cif-style inputs."
    ))
}
