//! Packed PDB-style atom name (`[u8; 4]`).

/// Packed PDB-style atom name (4 bytes, NUL-padded on the right).
///
/// PDB atom names are 1-4 ASCII characters (`"N"`, `"CA"`, `"OD1"`,
/// `"HG21"`). Storing them as `[u8; 4]` enables integer-comparison
/// matching in hot paths and keeps the type `Copy` + `Hash`-friendly.
///
/// The packing convention is **NUL-padded on the right**, distinct
/// from the PDB column convention of space-padding. Use
/// [`AtomName::from_bytes`] to construct from a slice; it strips
/// trailing padding internally for you only if the slice itself
/// already excludes it. PDB columns should be trimmed to non-space
/// bytes before passing in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtomName([u8; 4]);

impl AtomName {
    /// Construct from a byte slice. Inputs longer than 4 bytes are
    /// truncated to the first 4; shorter inputs are right-padded with
    /// NUL.
    #[must_use]
    pub const fn from_bytes(s: &[u8]) -> Self {
        let mut out = [0u8; 4];
        let n = if s.len() < 4 { s.len() } else { 4 };
        let mut i = 0;
        while i < n {
            out[i] = s[i];
            i += 1;
        }
        Self(out)
    }

    /// View as a UTF-8 string slice with trailing NUL bytes trimmed.
    ///
    /// PDB atom names are ASCII; the UTF-8 fallback returns an empty
    /// slice on the (unexpected) chance the bytes are not valid UTF-8.
    #[must_use]
    pub fn as_str(&self) -> &str {
        let end = self.0.iter().position(|&b| b == 0).unwrap_or(4);
        core::str::from_utf8(&self.0[..end]).unwrap_or("")
    }

    /// Whether this name denotes a protein backbone heavy atom
    /// (`N`, `CA`, `C`, `O`, or terminal `OXT`). Returns `false` for
    /// sidechain atoms and for non-protein atoms.
    #[must_use]
    pub fn is_protein_backbone(&self) -> bool {
        is_protein_backbone_atom_name(self.as_str())
    }
}

/// Whether a PDB atom-name string denotes a protein backbone heavy
/// atom (`N`, `CA`, `C`, `O`, or terminal `OXT`).
#[must_use]
pub fn is_protein_backbone_atom_name(name: &str) -> bool {
    matches!(name, "N" | "CA" | "C" | "O" | "OXT")
}

impl core::fmt::Display for AtomName {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_one_to_four_chars() {
        assert_eq!(AtomName::from_bytes(b"N").as_str(), "N");
        assert_eq!(AtomName::from_bytes(b"CA").as_str(), "CA");
        assert_eq!(AtomName::from_bytes(b"CG1").as_str(), "CG1");
        assert_eq!(AtomName::from_bytes(b"HG21").as_str(), "HG21");
    }

    #[test]
    fn truncates_oversize_input() {
        assert_eq!(AtomName::from_bytes(b"TOOBIG").as_str(), "TOOB");
    }

    #[test]
    fn empty_input_is_empty_str() {
        assert_eq!(AtomName::from_bytes(b"").as_str(), "");
    }

    #[test]
    fn equality_by_packed_bytes() {
        assert_eq!(AtomName::from_bytes(b"CA"), AtomName::from_bytes(b"CA"),);
        assert_ne!(AtomName::from_bytes(b"CA"), AtomName::from_bytes(b"CB"),);
    }

    #[test]
    fn display_strips_padding() {
        let name = AtomName::from_bytes(b"CA");
        assert_eq!(format!("{name}"), "CA");
    }
}
