//! Canonical amino acids: enum, 3-letter code parsing, intra-residue
//! bond tables, and hydrophobicity classification.

use super::atom_name::AtomName;

/// One of the 20 canonical L-amino acids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AminoAcid {
    /// Alanine (A).
    Ala,
    /// Arginine (R).
    Arg,
    /// Asparagine (N).
    Asn,
    /// Aspartate (D).
    Asp,
    /// Cysteine (C).
    Cys,
    /// Glutamine (Q).
    Gln,
    /// Glutamate (E).
    Glu,
    /// Glycine (G).
    Gly,
    /// Histidine (H).
    His,
    /// Isoleucine (I).
    Ile,
    /// Leucine (L).
    Leu,
    /// Lysine (K).
    Lys,
    /// Methionine (M).
    Met,
    /// Phenylalanine (F).
    Phe,
    /// Proline (P).
    Pro,
    /// Serine (S).
    Ser,
    /// Threonine (T).
    Thr,
    /// Tryptophan (W).
    Trp,
    /// Tyrosine (Y).
    Tyr,
    /// Valine (V).
    Val,
}

impl AminoAcid {
    /// Parse a 3-letter PDB residue code (case-insensitive ASCII).
    /// Returns `None` for unknown codes.
    #[must_use]
    pub fn from_code(code: [u8; 3]) -> Option<Self> {
        let upper = [
            code[0].to_ascii_uppercase(),
            code[1].to_ascii_uppercase(),
            code[2].to_ascii_uppercase(),
        ];
        match &upper {
            b"ALA" => Some(Self::Ala),
            b"ARG" => Some(Self::Arg),
            b"ASN" => Some(Self::Asn),
            b"ASP" => Some(Self::Asp),
            b"CYS" => Some(Self::Cys),
            b"GLN" => Some(Self::Gln),
            b"GLU" => Some(Self::Glu),
            b"GLY" => Some(Self::Gly),
            b"HIS" => Some(Self::His),
            b"ILE" => Some(Self::Ile),
            b"LEU" => Some(Self::Leu),
            b"LYS" => Some(Self::Lys),
            b"MET" => Some(Self::Met),
            b"PHE" => Some(Self::Phe),
            b"PRO" => Some(Self::Pro),
            b"SER" => Some(Self::Ser),
            b"THR" => Some(Self::Thr),
            b"TRP" => Some(Self::Trp),
            b"TYR" => Some(Self::Tyr),
            b"VAL" => Some(Self::Val),
            _ => None,
        }
    }

    /// 3-letter PDB residue code (uppercase ASCII).
    #[must_use]
    pub const fn code(self) -> [u8; 3] {
        match self {
            Self::Ala => *b"ALA",
            Self::Arg => *b"ARG",
            Self::Asn => *b"ASN",
            Self::Asp => *b"ASP",
            Self::Cys => *b"CYS",
            Self::Gln => *b"GLN",
            Self::Glu => *b"GLU",
            Self::Gly => *b"GLY",
            Self::His => *b"HIS",
            Self::Ile => *b"ILE",
            Self::Leu => *b"LEU",
            Self::Lys => *b"LYS",
            Self::Met => *b"MET",
            Self::Phe => *b"PHE",
            Self::Pro => *b"PRO",
            Self::Ser => *b"SER",
            Self::Thr => *b"THR",
            Self::Trp => *b"TRP",
            Self::Tyr => *b"TYR",
            Self::Val => *b"VAL",
        }
    }

    /// 1-letter amino acid code (uppercase ASCII).
    #[must_use]
    pub const fn one_letter(self) -> u8 {
        match self {
            Self::Ala => b'A',
            Self::Arg => b'R',
            Self::Asn => b'N',
            Self::Asp => b'D',
            Self::Cys => b'C',
            Self::Gln => b'Q',
            Self::Glu => b'E',
            Self::Gly => b'G',
            Self::His => b'H',
            Self::Ile => b'I',
            Self::Leu => b'L',
            Self::Lys => b'K',
            Self::Met => b'M',
            Self::Phe => b'F',
            Self::Pro => b'P',
            Self::Ser => b'S',
            Self::Thr => b'T',
            Self::Trp => b'W',
            Self::Tyr => b'Y',
            Self::Val => b'V',
        }
    }

    /// Heavy-atom intra-residue bonds beyond the universal N-CA, CA-C,
    /// and C=O backbone bonds.
    ///
    /// Includes the CA-CB anchor for every non-Gly residue plus all
    /// sidechain-internal heavy bonds. Glycine has no sidechain heavy
    /// atoms and returns an empty slice.
    ///
    /// Consumers add backbone (N-CA, CA-C, C=O) and inter-residue
    /// peptide bonds separately when populating `ProteinEntity` bond
    /// graphs.
    ///
    /// The proline ring-closure bond CD-N is intentionally omitted.
    #[must_use]
    pub const fn bonds(self) -> &'static [(AtomName, AtomName)] {
        match self {
            Self::Ala => ALA_BONDS,
            Self::Arg => ARG_BONDS,
            Self::Asn => ASN_BONDS,
            Self::Asp => ASP_BONDS,
            Self::Cys => CYS_BONDS,
            Self::Gln => GLN_BONDS,
            Self::Glu => GLU_BONDS,
            Self::Gly => GLY_BONDS,
            Self::His => HIS_BONDS,
            Self::Ile => ILE_BONDS,
            Self::Leu => LEU_BONDS,
            Self::Lys => LYS_BONDS,
            Self::Met => MET_BONDS,
            Self::Phe => PHE_BONDS,
            Self::Pro => PRO_BONDS,
            Self::Ser => SER_BONDS,
            Self::Thr => THR_BONDS,
            Self::Trp => TRP_BONDS,
            Self::Tyr => TYR_BONDS,
            Self::Val => VAL_BONDS,
        }
    }

    /// Sidechain torsion (χ) atom quads, in order χ1, χ2, ...
    ///
    /// Each quad is the four atom names defining one χ dihedral under the
    /// standard IUPAC convention (χ1 = N-CA-CB-*G, with the branched/heteroatom
    /// exceptions: Ser OG, Thr OG1, Cys SG, Ile/Val CG1). Ala and Gly have no
    /// rotatable sidechain torsions and return an empty slice.
    #[must_use]
    pub const fn chi_atoms(self) -> &'static [[AtomName; 4]] {
        match self {
            Self::Ala | Self::Gly => &[],
            Self::Arg => ARG_CHI,
            Self::Asn => ASN_CHI,
            Self::Asp => ASP_CHI,
            Self::Cys => CYS_CHI,
            Self::Gln => GLN_CHI,
            Self::Glu => GLU_CHI,
            Self::His => HIS_CHI,
            Self::Ile => ILE_CHI,
            Self::Leu => LEU_CHI,
            Self::Lys => LYS_CHI,
            Self::Met => MET_CHI,
            Self::Phe => PHE_CHI,
            Self::Pro => PRO_CHI,
            Self::Ser => SER_CHI,
            Self::Thr => THR_CHI,
            Self::Trp => TRP_CHI,
            Self::Tyr => TYR_CHI,
            Self::Val => VAL_CHI,
        }
    }

    /// Whether this amino acid carries Rosetta's POLAR residue-type property.
    ///
    /// Polar set (Rosetta `fa_standard` POLAR property): Arg, Asn, Asp, Gln,
    /// Glu, His, Lys, Ser, Thr.
    #[must_use]
    pub const fn is_polar(self) -> bool {
        matches!(
            self,
            Self::Arg
                | Self::Asn
                | Self::Asp
                | Self::Gln
                | Self::Glu
                | Self::His
                | Self::Lys
                | Self::Ser
                | Self::Thr
        )
    }

    /// Whether this amino acid is hydrophobic, i.e. not polar.
    ///
    /// Defined as the complement of [`Self::is_polar`], matching Rosetta's
    /// binary polar/non-polar partition. Hydrophobic set: Ala, Cys, Gly,
    /// Ile, Leu, Met, Phe, Pro, Trp, Tyr, Val.
    #[must_use]
    pub const fn is_hydrophobic(self) -> bool {
        !self.is_polar()
    }
}

/// One-letter code for a modified amino-acid residue name that
/// [`AminoAcid::from_code`] does not parse but [`crate::entity::molecule`]
/// classifies as protein.
///
/// Maps the standard genetically/selenium-substituted residues to their parent:
/// MSE (selenomethionine) -> M, SEC (selenocysteine) -> U, PYL (pyrrolysine)
/// -> O. Returns `None` for any other code.
#[must_use]
pub fn modified_aa_one_letter(code: [u8; 3]) -> Option<u8> {
    let upper = [
        code[0].to_ascii_uppercase(),
        code[1].to_ascii_uppercase(),
        code[2].to_ascii_uppercase(),
    ];
    match &upper {
        b"MSE" => Some(b'M'),
        b"SEC" => Some(b'U'),
        b"PYL" => Some(b'O'),
        _ => None,
    }
}

// Bond tables

const fn an(b: &'static [u8]) -> AtomName {
    AtomName::from_bytes(b)
}

const ALA_BONDS: &[(AtomName, AtomName)] = &[(an(b"CA"), an(b"CB"))];

const ARG_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"CD")),
    (an(b"CD"), an(b"NE")),
    (an(b"NE"), an(b"CZ")),
    (an(b"CZ"), an(b"NH1")),
    (an(b"CZ"), an(b"NH2")),
];

const ASN_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"OD1")),
    (an(b"CG"), an(b"ND2")),
];

const ASP_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"OD1")),
    (an(b"CG"), an(b"OD2")),
];

const CYS_BONDS: &[(AtomName, AtomName)] =
    &[(an(b"CA"), an(b"CB")), (an(b"CB"), an(b"SG"))];

const GLN_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"CD")),
    (an(b"CD"), an(b"OE1")),
    (an(b"CD"), an(b"NE2")),
];

const GLU_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"CD")),
    (an(b"CD"), an(b"OE1")),
    (an(b"CD"), an(b"OE2")),
];

const GLY_BONDS: &[(AtomName, AtomName)] = &[];

const HIS_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"ND1")),
    (an(b"ND1"), an(b"CE1")),
    (an(b"CE1"), an(b"NE2")),
    (an(b"NE2"), an(b"CD2")),
    (an(b"CD2"), an(b"CG")),
];

const ILE_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG1")),
    (an(b"CG1"), an(b"CD1")),
    (an(b"CB"), an(b"CG2")),
];

const LEU_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"CD1")),
    (an(b"CG"), an(b"CD2")),
];

const LYS_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"CD")),
    (an(b"CD"), an(b"CE")),
    (an(b"CE"), an(b"NZ")),
];

const MET_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"SD")),
    (an(b"SD"), an(b"CE")),
];

const PHE_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"CD1")),
    (an(b"CD1"), an(b"CE1")),
    (an(b"CE1"), an(b"CZ")),
    (an(b"CZ"), an(b"CE2")),
    (an(b"CE2"), an(b"CD2")),
    (an(b"CD2"), an(b"CG")),
];

const PRO_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"CD")),
];

const SER_BONDS: &[(AtomName, AtomName)] =
    &[(an(b"CA"), an(b"CB")), (an(b"CB"), an(b"OG"))];

const THR_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"OG1")),
    (an(b"CB"), an(b"CG2")),
];

const TRP_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"CD1")),
    (an(b"CD1"), an(b"NE1")),
    (an(b"NE1"), an(b"CE2")),
    (an(b"CE2"), an(b"CD2")),
    (an(b"CD2"), an(b"CG")),
    (an(b"CE2"), an(b"CZ2")),
    (an(b"CZ2"), an(b"CH2")),
    (an(b"CH2"), an(b"CZ3")),
    (an(b"CZ3"), an(b"CE3")),
    (an(b"CE3"), an(b"CD2")),
];

const TYR_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG")),
    (an(b"CG"), an(b"CD1")),
    (an(b"CD1"), an(b"CE1")),
    (an(b"CE1"), an(b"CZ")),
    (an(b"CZ"), an(b"OH")),
    (an(b"CZ"), an(b"CE2")),
    (an(b"CE2"), an(b"CD2")),
    (an(b"CD2"), an(b"CG")),
];

const VAL_BONDS: &[(AtomName, AtomName)] = &[
    (an(b"CA"), an(b"CB")),
    (an(b"CB"), an(b"CG1")),
    (an(b"CB"), an(b"CG2")),
];

// Sidechain χ tables (IUPAC convention)

const fn chi(
    a: &'static [u8],
    b: &'static [u8],
    c: &'static [u8],
    d: &'static [u8],
) -> [AtomName; 4] {
    [an(a), an(b), an(c), an(d)]
}

const ARG_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"CD"),
    chi(b"CB", b"CG", b"CD", b"NE"),
    chi(b"CG", b"CD", b"NE", b"CZ"),
];
const ASN_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"OD1"),
];
const ASP_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"OD1"),
];
const CYS_CHI: &[[AtomName; 4]] = &[chi(b"N", b"CA", b"CB", b"SG")];
const GLN_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"CD"),
    chi(b"CB", b"CG", b"CD", b"OE1"),
];
const GLU_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"CD"),
    chi(b"CB", b"CG", b"CD", b"OE1"),
];
const HIS_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"ND1"),
];
const ILE_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG1"),
    chi(b"CA", b"CB", b"CG1", b"CD1"),
];
const LEU_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"CD1"),
];
const LYS_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"CD"),
    chi(b"CB", b"CG", b"CD", b"CE"),
    chi(b"CG", b"CD", b"CE", b"NZ"),
];
const MET_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"SD"),
    chi(b"CB", b"CG", b"SD", b"CE"),
];
const PHE_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"CD1"),
];
const PRO_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"CD"),
];
const SER_CHI: &[[AtomName; 4]] = &[chi(b"N", b"CA", b"CB", b"OG")];
const THR_CHI: &[[AtomName; 4]] = &[chi(b"N", b"CA", b"CB", b"OG1")];
const TRP_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"CD1"),
];
const TYR_CHI: &[[AtomName; 4]] = &[
    chi(b"N", b"CA", b"CB", b"CG"),
    chi(b"CA", b"CB", b"CG", b"CD1"),
];
const VAL_CHI: &[[AtomName; 4]] = &[chi(b"N", b"CA", b"CB", b"CG1")];

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[AminoAcid] = &[
        AminoAcid::Ala,
        AminoAcid::Arg,
        AminoAcid::Asn,
        AminoAcid::Asp,
        AminoAcid::Cys,
        AminoAcid::Gln,
        AminoAcid::Glu,
        AminoAcid::Gly,
        AminoAcid::His,
        AminoAcid::Ile,
        AminoAcid::Leu,
        AminoAcid::Lys,
        AminoAcid::Met,
        AminoAcid::Phe,
        AminoAcid::Pro,
        AminoAcid::Ser,
        AminoAcid::Thr,
        AminoAcid::Trp,
        AminoAcid::Tyr,
        AminoAcid::Val,
    ];

    #[test]
    fn from_code_round_trips_every_variant() {
        for &aa in ALL {
            assert_eq!(AminoAcid::from_code(aa.code()), Some(aa));
        }
    }

    #[test]
    fn from_code_is_case_insensitive() {
        assert_eq!(AminoAcid::from_code(*b"ala"), Some(AminoAcid::Ala));
        assert_eq!(AminoAcid::from_code(*b"Gly"), Some(AminoAcid::Gly));
        assert_eq!(AminoAcid::from_code(*b"tRp"), Some(AminoAcid::Trp));
    }

    #[test]
    fn from_code_unknown_returns_none() {
        assert_eq!(AminoAcid::from_code(*b"XXX"), None);
        assert_eq!(AminoAcid::from_code(*b"MSE"), None);
    }

    #[test]
    fn chi_atoms_counts_and_first_quad() {
        // Expected χ count per residue; Ala/Gly have none.
        let n = |aa: AminoAcid| aa.chi_atoms().len();
        assert_eq!(n(AminoAcid::Ala), 0);
        assert_eq!(n(AminoAcid::Gly), 0);
        assert_eq!(n(AminoAcid::Ser), 1);
        assert_eq!(n(AminoAcid::Val), 1);
        assert_eq!(n(AminoAcid::Phe), 2);
        assert_eq!(n(AminoAcid::Met), 3);
        assert_eq!(n(AminoAcid::Arg), 4);
        assert_eq!(n(AminoAcid::Lys), 4);

        // χ1 of a standard residue is N-CA-CB-CG; Ser swaps in OG.
        assert_eq!(
            AminoAcid::Arg.chi_atoms()[0],
            [an(b"N"), an(b"CA"), an(b"CB"), an(b"CG")]
        );
        assert_eq!(
            AminoAcid::Ser.chi_atoms()[0],
            [an(b"N"), an(b"CA"), an(b"CB"), an(b"OG")]
        );
    }

    #[test]
    fn modified_aa_one_letter_known_and_unknown() {
        assert_eq!(modified_aa_one_letter(*b"MSE"), Some(b'M'));
        assert_eq!(modified_aa_one_letter(*b"SEC"), Some(b'U'));
        assert_eq!(modified_aa_one_letter(*b"PYL"), Some(b'O'));
        assert_eq!(modified_aa_one_letter(*b"mse"), Some(b'M'));
        assert_eq!(modified_aa_one_letter(*b"ALA"), None);
        assert_eq!(modified_aa_one_letter(*b"XXX"), None);
    }

    #[test]
    fn bonds_non_empty_for_every_variant_except_gly() {
        for &aa in ALL {
            let bonds = aa.bonds();
            if matches!(aa, AminoAcid::Gly) {
                assert!(bonds.is_empty(), "Gly bonds should be empty");
            } else {
                assert!(!bonds.is_empty(), "{aa:?} bonds should not be empty");
            }
        }
    }

    #[test]
    fn phenylalanine_has_ring_plus_anchor_bonds() {
        // 6-bond benzene ring + 1 CG-CB bond + 1 CA-CB anchor = 8.
        assert_eq!(AminoAcid::Phe.bonds().len(), 8);
    }

    #[test]
    fn is_hydrophobic_matches_reference_table() {
        let hydrophobic = [
            AminoAcid::Ala,
            AminoAcid::Cys,
            AminoAcid::Gly,
            AminoAcid::Ile,
            AminoAcid::Leu,
            AminoAcid::Met,
            AminoAcid::Phe,
            AminoAcid::Pro,
            AminoAcid::Trp,
            AminoAcid::Tyr,
            AminoAcid::Val,
        ];
        for &aa in ALL {
            let expected = hydrophobic.contains(&aa);
            assert_eq!(aa.is_hydrophobic(), expected, "{aa:?} hydrophobicity");
        }
    }
}
