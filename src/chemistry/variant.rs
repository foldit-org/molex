//! Per-residue chemistry variants: terminus patches, disulfide
//! participation, and protonation states.
//!
//! Variants are tags that describe how a residue's chemistry differs
//! from its canonical form. They are owned by `Residue::variants` and
//! survive Assembly round-trips. Consumers that care about residue
//! chemistry (the rosetta bridge in particular) translate `VariantTag`
//! into their own variant taxonomy on apply.
//!
//! Most variants are redundant with state that's derivable from
//! coordinates (termini = chain endpoints; disulfides = S-S distance
//! within `detect_disulfides` cutoff), but they're carried explicitly
//! so consumers don't have to re-derive on every snapshot.
//! `Protonation` is the load-bearing case: HID vs HIE differ only in
//! which nitrogen carries the hydrogen, and protonation isn't
//! determined by heavy-atom positions alone; without a wire-level
//! tag, a snapshot round-trip would silently revert to default
//! protonation.

/// Per-residue chemistry variant.
///
/// Multiple tags may apply to the same residue (e.g., a CYS that's
/// both N-terminal and disulfide-bonded). Tag order within a
/// `Residue::variants` vector is not semantically significant;
/// duplicates are tolerated by readers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VariantTag {
    /// Chain N-terminus patch.
    NTerminus,
    /// Chain C-terminus patch.
    CTerminus,
    /// Participates in a disulfide bond. The partner is not stored on
    /// the tag; callers recover the SG-SG pairing on demand from geometry
    /// via `detect_disulfides` (surfaced for rendering through
    /// `Assembly::detect_fallback_connections`). The tag is purely a flag.
    Disulfide,
    /// Non-canonical protonation state.
    Protonation(ProtonationState),
    /// Open-ended escape hatch for variants not yet enumerated.
    /// Consumers that don't recognize the string ignore the tag.
    Other(String),
}

/// Non-canonical protonation state for a residue.
///
/// Histidine tautomers are the most common case in practice
/// (rosetta's `HID` / `HIE` / `HIP` variants). Other residues
/// (ASH, GLH, LYN, CYM) are reached via `Custom`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProtonationState {
    /// Histidine, delta-protonated (rosetta `HID`).
    HisDelta,
    /// Histidine, epsilon-protonated (rosetta `HIE`).
    HisEpsilon,
    /// Histidine, doubly-protonated (rosetta `HIP`).
    HisDoubly,
    /// Anything not enumerated above. Consumers that don't recognize
    /// the string ignore the variant.
    Custom(String),
}
