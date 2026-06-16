//! Canonical nucleotides: enum + 3-letter code parsing.
//!
//! Per-base bond tables cover sugar-backbone bonds shared across all
//! nucleotides, sugar-to-base anchor, and the heterocyclic base ring
//! bonds. Inter-residue phosphodiester bonds live at the
//! [`crate::entity::molecule::nucleic_acid::NAEntity`] level and are
//! not part of these tables.

use super::atom_name::AtomName;

/// Canonical nucleotide bases. The DNA/RNA distinction lives on the
/// owning [`crate::entity::molecule::MoleculeType`]; this enum names
/// only the base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nucleotide {
    /// Adenine.
    A,
    /// Cytosine.
    C,
    /// Guanine.
    G,
    /// Thymine (DNA).
    T,
    /// Uracil (RNA, occasionally DNA).
    U,
}

impl Nucleotide {
    /// Parse a 1-, 2-, or 3-letter PDB nucleotide code.
    ///
    /// Accepts the canonical mmCIF and PDB aliases that
    /// [`crate::entity::molecule::classify_residue`] already
    /// recognizes:
    ///
    /// - 1-letter RNA: `A`, `C`, `G`, `U`, and `T`
    /// - 2-letter DNA: `DA`, `DC`, `DG`, `DT`, `DU`
    /// - 3-letter aliases: `ADE`, `CYT`, `GUA`, `URA`, `THY`, `RAD`, `RCY`,
    ///   `RGU`
    ///
    /// Comparison is ASCII-case-insensitive. Trailing space or NUL
    /// padding is stripped before matching. Returns `None` for unknown
    /// codes.
    #[must_use]
    pub fn from_code(code: [u8; 3]) -> Option<Self> {
        let upper = [
            code[0].to_ascii_uppercase(),
            code[1].to_ascii_uppercase(),
            code[2].to_ascii_uppercase(),
        ];
        let mut end = 3usize;
        while end > 0 && (upper[end - 1] == 0 || upper[end - 1] == b' ') {
            end -= 1;
        }
        match &upper[..end] {
            b"A" | b"DA" | b"ADE" | b"RAD" => Some(Self::A),
            b"C" | b"DC" | b"CYT" | b"RCY" => Some(Self::C),
            b"G" | b"DG" | b"GUA" | b"RGU" => Some(Self::G),
            b"T" | b"DT" | b"THY" => Some(Self::T),
            b"U" | b"DU" | b"URA" => Some(Self::U),
            _ => None,
        }
    }

    /// Heavy-atom intra-residue bonds for this nucleotide.
    ///
    /// Covers: the shared sugar ring (C1'-C2'-C3'-C4'-O4'-C1'), the
    /// sugar-phosphate backbone bonds (P-O5'-C5'-C4', C3'-O3'), the
    /// 2'-OH on ribose (C2'-O2'), the sugar-to-base anchor (C1'-N for
    /// purines at N9, pyrimidines at N1), and the heterocyclic base
    /// ring bonds. Phosphate terminal oxygens (OP1, OP2, OP3) are
    /// intentionally absent from sugar-phosphate heavy-heavy coverage
    /// because their variable presence in PDB data would generate
    /// dangling bonds; `NAEntity::new` emits P-OP* via name-matching
    /// when the terminal oxygens are present.
    ///
    /// Inter-residue phosphodiester bonds (O3'(i)-P(i+1)) are emitted
    /// at the entity level, not here.
    #[must_use]
    pub const fn bonds(self) -> &'static [(AtomName, AtomName)] {
        match self {
            Self::A => &ADENINE_BONDS,
            Self::C => &CYTOSINE_BONDS,
            Self::G => &GUANINE_BONDS,
            Self::T => &THYMINE_BONDS,
            Self::U => &URACIL_BONDS,
        }
    }
}

// Bond tables

const fn an(b: &'static [u8]) -> AtomName {
    AtomName::from_bytes(b)
}

/// Sugar-phosphate backbone bonds shared by every nucleotide.
const SUGAR_PHOSPHATE_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"P"), an(b"O5'")),
    (an(b"O5'"), an(b"C5'")),
    (an(b"C5'"), an(b"C4'")),
    (an(b"C4'"), an(b"C3'")),
    (an(b"C3'"), an(b"O3'")),
    (an(b"C4'"), an(b"O4'")),
    (an(b"O4'"), an(b"C1'")),
    (an(b"C1'"), an(b"C2'")),
    (an(b"C2'"), an(b"C3'")),
    (an(b"C2'"), an(b"O2'")),
];

/// Purine (A/G) shared ring skeleton: 6-membered + fused 5-membered,
/// plus the sugar-to-base anchor C1'-N9.
const PURINE_RING_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"C1'"), an(b"N9")),
    (an(b"N9"), an(b"C8")),
    (an(b"C8"), an(b"N7")),
    (an(b"N7"), an(b"C5")),
    (an(b"C5"), an(b"C4")),
    (an(b"C4"), an(b"N9")),
    (an(b"C5"), an(b"C6")),
    (an(b"C6"), an(b"N1")),
    (an(b"N1"), an(b"C2")),
    (an(b"C2"), an(b"N3")),
    (an(b"N3"), an(b"C4")),
];

/// Pyrimidine (C/T/U) shared ring: 6-membered with sugar anchor C1'-N1.
const PYRIMIDINE_RING_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"C1'"), an(b"N1")),
    (an(b"N1"), an(b"C2")),
    (an(b"C2"), an(b"N3")),
    (an(b"N3"), an(b"C4")),
    (an(b"C4"), an(b"C5")),
    (an(b"C5"), an(b"C6")),
    (an(b"C6"), an(b"N1")),
];

const ADENINE_BASE_BONDS: &[(AtomName, AtomName)] = &[(an(b"C6"), an(b"N6"))];
const GUANINE_BASE_BONDS: &[(AtomName, AtomName)] =
    &[(an(b"C6"), an(b"O6")), (an(b"C2"), an(b"N2"))];
const CYTOSINE_BASE_BONDS: &[(AtomName, AtomName)] =
    &[(an(b"C2"), an(b"O2")), (an(b"C4"), an(b"N4"))];
const URACIL_BASE_BONDS: &[(AtomName, AtomName)] =
    &[(an(b"C2"), an(b"O2")), (an(b"C4"), an(b"O4"))];
const THYMINE_BASE_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"C2"), an(b"O2")),
    (an(b"C4"), an(b"O4")),
    (an(b"C5"), an(b"C7")),
];

// Tables below are built by const concatenation of the shared sugar-
// phosphate skeleton with each base's ring + substituent bonds.
// Sizes: 10 sugar-phosphate + 11 purine-ring = 21;
//        10 sugar-phosphate + 7 pyrimidine-ring = 17.
const ADENINE_BONDS: [(AtomName, AtomName); 22] =
    concat_three(SUGAR_PHOSPHATE_BONDS, PURINE_RING_BONDS, ADENINE_BASE_BONDS);
const GUANINE_BONDS: [(AtomName, AtomName); 23] =
    concat_three(SUGAR_PHOSPHATE_BONDS, PURINE_RING_BONDS, GUANINE_BASE_BONDS);
const CYTOSINE_BONDS: [(AtomName, AtomName); 19] = concat_three(
    SUGAR_PHOSPHATE_BONDS,
    PYRIMIDINE_RING_BONDS,
    CYTOSINE_BASE_BONDS,
);
const URACIL_BONDS: [(AtomName, AtomName); 19] = concat_three(
    SUGAR_PHOSPHATE_BONDS,
    PYRIMIDINE_RING_BONDS,
    URACIL_BASE_BONDS,
);
const THYMINE_BONDS: [(AtomName, AtomName); 20] = concat_three(
    SUGAR_PHOSPHATE_BONDS,
    PYRIMIDINE_RING_BONDS,
    THYMINE_BASE_BONDS,
);

/// `const fn` concatenation of three bond slices, sized by explicit
/// output length `N`. Callers infer `N` from the surrounding slice
/// binding's length expectation.
const fn concat_three<const N: usize>(
    a: &[(AtomName, AtomName)],
    b: &[(AtomName, AtomName)],
    c: &[(AtomName, AtomName)],
) -> [(AtomName, AtomName); N] {
    let zero = an(b"");
    let mut out = [(zero, zero); N];
    let mut idx = 0;
    let mut i = 0;
    while i < a.len() {
        out[idx] = a[i];
        idx += 1;
        i += 1;
    }
    let mut i = 0;
    while i < b.len() {
        out[idx] = b[i];
        idx += 1;
        i += 1;
    }
    let mut i = 0;
    while i < c.len() {
        out[idx] = c[i];
        idx += 1;
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_code_canonical_one_letter() {
        assert_eq!(Nucleotide::from_code(*b"A  "), Some(Nucleotide::A));
        assert_eq!(Nucleotide::from_code(*b"C  "), Some(Nucleotide::C));
        assert_eq!(Nucleotide::from_code(*b"G  "), Some(Nucleotide::G));
        assert_eq!(Nucleotide::from_code(*b"T  "), Some(Nucleotide::T));
        assert_eq!(Nucleotide::from_code(*b"U  "), Some(Nucleotide::U));
    }

    #[test]
    fn from_code_dna_two_letter() {
        assert_eq!(Nucleotide::from_code(*b"DA\0"), Some(Nucleotide::A));
        assert_eq!(Nucleotide::from_code(*b"DC\0"), Some(Nucleotide::C));
        assert_eq!(Nucleotide::from_code(*b"DG\0"), Some(Nucleotide::G));
        assert_eq!(Nucleotide::from_code(*b"DT\0"), Some(Nucleotide::T));
    }

    #[test]
    fn from_code_three_letter_aliases() {
        assert_eq!(Nucleotide::from_code(*b"ADE"), Some(Nucleotide::A));
        assert_eq!(Nucleotide::from_code(*b"GUA"), Some(Nucleotide::G));
        assert_eq!(Nucleotide::from_code(*b"THY"), Some(Nucleotide::T));
    }

    #[test]
    fn from_code_unknown_returns_none() {
        assert_eq!(Nucleotide::from_code(*b"XXX"), None);
        assert_eq!(Nucleotide::from_code(*b"DI\0"), None);
    }

    #[test]
    fn from_code_handles_padding() {
        assert_eq!(Nucleotide::from_code(*b"A  "), Some(Nucleotide::A));
    }

    #[test]
    fn bonds_tables_non_empty() {
        for nt in [
            Nucleotide::A,
            Nucleotide::C,
            Nucleotide::G,
            Nucleotide::T,
            Nucleotide::U,
        ] {
            assert!(!nt.bonds().is_empty(), "{nt:?} should have bonds");
        }
    }

    #[test]
    fn purine_bonds_include_sugar_base_anchor() {
        let table = Nucleotide::A.bonds();
        let anchor =
            (AtomName::from_bytes(b"C1'"), AtomName::from_bytes(b"N9"));
        assert!(table.contains(&anchor));
    }

    #[test]
    fn pyrimidine_bonds_include_sugar_base_anchor() {
        let table = Nucleotide::C.bonds();
        let anchor =
            (AtomName::from_bytes(b"C1'"), AtomName::from_bytes(b"N1"));
        assert!(table.contains(&anchor));
    }

    #[test]
    fn thymine_has_methyl_c7_bond() {
        let table = Nucleotide::T.bonds();
        let methyl = (AtomName::from_bytes(b"C5"), AtomName::from_bytes(b"C7"));
        assert!(table.contains(&methyl));
    }
}
