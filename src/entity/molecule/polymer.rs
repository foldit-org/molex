//! Polymer residue type.

use std::ops::Range;

use crate::chemistry::variant::VariantTag;

/// A single residue within a polymer entity.
///
/// Carries both the structural-side identifier (`label_seq_id`) and the
/// optional author-side identifiers (`auth_seq_id`, `auth_comp_id`,
/// `ins_code`) sourced from mmCIF / BinaryCIF. `None` on any author
/// field means "fall back to the label-side value". Internal grouping
/// uses `label_*`, user-facing output uses `auth_*`.
#[derive(Debug, Clone)]
pub struct Residue {
    /// 3-character residue name (e.g. b"ALA"). The structural-side name.
    pub name: [u8; 3],
    /// `label_seq_id` (mmCIF) or `resSeq` (PDB). The structural-side
    /// residue number used for internal ordering.
    pub label_seq_id: i32,
    /// `auth_seq_id` from mmCIF / BinaryCIF. `None` defaults to
    /// [`Self::label_seq_id`].
    pub auth_seq_id: Option<i32>,
    /// `auth_comp_id` from mmCIF / BinaryCIF. `None` defaults to
    /// [`Self::name`].
    pub auth_comp_id: Option<[u8; 3]>,
    /// Insertion code (PDB `iCode` / mmCIF `pdbx_PDB_ins_code`).
    /// `None` = blank.
    pub ins_code: Option<u8>,
    /// Index range into the parent entity's atom list.
    pub atom_range: Range<usize>,
    /// Chemistry variants attached to this residue (terminus patches,
    /// disulfide participation, protonation state). Default-empty.
    /// Parsers populate this from format-specific tags when available;
    /// most parsers leave it empty and let downstream consumers
    /// re-derive what they need.
    pub variants: Vec<VariantTag>,
}

impl Residue {
    /// Author-side sequence id, falling back to [`Self::label_seq_id`]
    /// when the author id is absent.
    #[must_use]
    pub fn seq_id(&self) -> i32 {
        self.auth_seq_id.unwrap_or(self.label_seq_id)
    }
}
