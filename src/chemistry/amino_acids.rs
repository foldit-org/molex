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

// ---------------------------------------------------------------------------
// Bond tables
// ---------------------------------------------------------------------------

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
