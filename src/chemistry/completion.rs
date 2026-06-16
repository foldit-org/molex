//! Ideal-geometry reference templates for standard residues.
//!
//! Each standard amino acid and DNA nucleotide has a full all-atom
//! (heavy + hydrogen) ideal conformation taken from the PDB Chemical
//! Component Dictionary. These templates are the source geometry for
//! completing residues that are missing atoms: a rigid fit of the
//! template's present atoms onto a parsed residue's present atoms
//! places the absent atoms.
//!
//! This module owns the template types and the per-residue lookup. The
//! const atom data itself is generated and lives in the private
//! `completion_data` sibling module; do not hand-edit that file.

use glam::Vec3;

use super::amino_acids::AminoAcid;
use super::atom_name::AtomName;
use super::nucleotides::Nucleotide;
use crate::element::Element;

/// One atom of an ideal-geometry residue template.
///
/// `ideal` is the component's ideal Cartesian position in angstroms,
/// expressed in the dictionary's own arbitrary frame; only the relative
/// geometry within a residue is meaningful, so consumers rigid-fit the
/// whole template onto a parsed residue rather than using these
/// coordinates directly.
///
/// The four boolean role flags mirror the source dictionary's per-atom
/// flags one-to-one; they are independent facts about an atom, not a
/// state machine, so they stay as plain bools.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemplateAtom {
    /// PDB-v3 atom name, used to match against a parsed residue's atoms.
    pub name: AtomName,
    /// Chemical element.
    pub element: Element,
    /// Ideal position in the dictionary's local frame (angstroms).
    pub ideal: Vec3,
    /// Whether this atom leaves on polymer-bond formation (e.g. `OXT`
    /// and the extra backbone hydrogens lost when residues link).
    pub is_leaving: bool,
    /// Whether the dictionary flags this atom as a backbone atom.
    pub is_backbone: bool,
    /// Whether this atom is present only at a chain's N-terminus.
    pub is_n_terminal: bool,
    /// Whether this atom is present only at a chain's C-terminus.
    pub is_c_terminal: bool,
}

/// Ordered ideal-geometry template for one standard residue.
///
/// `atoms` preserves the Chemical Component Dictionary's canonical atom
/// order. `comp` is the dictionary component code (e.g. `b"ALA"`,
/// `b"DA"`), right-NUL-padded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidueTemplate {
    /// Dictionary component code, right-NUL-padded (e.g. `b"ALA\0"`).
    pub comp: [u8; 3],
    /// Template atoms in canonical dictionary order.
    pub atoms: &'static [TemplateAtom],
}

impl ResidueTemplate {
    /// Component code with trailing NUL padding trimmed.
    #[must_use]
    pub fn comp_str(&self) -> &str {
        let end = self.comp.iter().position(|&b| b == 0).unwrap_or(3);
        core::str::from_utf8(&self.comp[..end]).unwrap_or("")
    }

    /// Find a template atom by name, if present.
    #[must_use]
    pub fn atom(&self, name: AtomName) -> Option<&'static TemplateAtom> {
        self.atoms.iter().find(|a| a.name == name)
    }
}

impl AminoAcid {
    /// Ideal-geometry template for this amino acid.
    #[must_use]
    pub fn template(self) -> &'static ResidueTemplate {
        use super::completion_data as data;
        match self {
            Self::Ala => &data::ALA,
            Self::Arg => &data::ARG,
            Self::Asn => &data::ASN,
            Self::Asp => &data::ASP,
            Self::Cys => &data::CYS,
            Self::Gln => &data::GLN,
            Self::Glu => &data::GLU,
            Self::Gly => &data::GLY,
            Self::His => &data::HIS,
            Self::Ile => &data::ILE,
            Self::Leu => &data::LEU,
            Self::Lys => &data::LYS,
            Self::Met => &data::MET,
            Self::Phe => &data::PHE,
            Self::Pro => &data::PRO,
            Self::Ser => &data::SER,
            Self::Thr => &data::THR,
            Self::Trp => &data::TRP,
            Self::Tyr => &data::TYR,
            Self::Val => &data::VAL,
        }
    }
}

impl Nucleotide {
    /// Ideal-geometry template for this nucleotide.
    ///
    /// Only the four DNA bases (A, C, G, T) have templates. RNA uracil
    /// ([`Nucleotide::U`]) returns `None`; an RNA template set is a
    /// later addition.
    #[must_use]
    pub fn template(self) -> Option<&'static ResidueTemplate> {
        use super::completion_data as data;
        match self {
            Self::A => Some(&data::DA),
            Self::C => Some(&data::DC),
            Self::G => Some(&data::DG),
            Self::T => Some(&data::DT),
            Self::U => None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    const AMINO_ACIDS: &[AminoAcid] = &[
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

    const DNA: &[Nucleotide] =
        &[Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::T];

    fn name(s: &[u8]) -> AtomName {
        AtomName::from_bytes(s)
    }

    #[test]
    fn every_template_loads_with_finite_coords() {
        for &aa in AMINO_ACIDS {
            let t = aa.template();
            assert!(!t.atoms.is_empty(), "{aa:?} template should not be empty");
            for atom in t.atoms {
                assert!(
                    atom.ideal.is_finite(),
                    "{aa:?} atom {} has non-finite ideal coords",
                    atom.name
                );
            }
        }
        for &nt in DNA {
            let t = nt.template().expect("DNA base has a template");
            assert!(!t.atoms.is_empty(), "{nt:?} template should not be empty");
            for atom in t.atoms {
                assert!(
                    atom.ideal.is_finite(),
                    "{nt:?} atom {} has non-finite ideal coords",
                    atom.name
                );
            }
        }
    }

    #[test]
    fn uracil_has_no_template_yet() {
        assert!(Nucleotide::U.template().is_none());
    }

    #[test]
    fn accessors_return_matching_comp_code() {
        assert_eq!(AminoAcid::Ala.template().comp_str(), "ALA");
        assert_eq!(AminoAcid::Trp.template().comp_str(), "TRP");
        assert_eq!(AminoAcid::Gly.template().comp_str(), "GLY");
        assert_eq!(
            Nucleotide::A.template().map(ResidueTemplate::comp_str),
            Some("DA")
        );
        assert_eq!(
            Nucleotide::T.template().map(ResidueTemplate::comp_str),
            Some("DT")
        );
    }

    #[test]
    fn alanine_has_backbone_cb_oxt_and_hydrogens() {
        let t = AminoAcid::Ala.template();
        for heavy in [b"N".as_slice(), b"CA", b"C", b"O", b"CB"] {
            assert!(
                t.atom(name(heavy)).is_some(),
                "ALA missing heavy atom {}",
                core::str::from_utf8(heavy).unwrap_or("?")
            );
        }
        let oxt = t.atom(name(b"OXT")).expect("ALA has OXT");
        assert!(oxt.is_leaving, "OXT should be flagged is_leaving");
        let has_h = t.atoms.iter().any(|a| a.element == Element::H);
        assert!(has_h, "ALA template should carry hydrogens");
    }

    #[test]
    fn tryptophan_has_indole_ring_heavy_atoms() {
        let t = AminoAcid::Trp.template();
        // Indole ring (5-membered pyrrole fused to benzene) heavy set.
        for ring in [
            b"CG".as_slice(),
            b"CD1",
            b"CD2",
            b"NE1",
            b"CE2",
            b"CE3",
            b"CZ2",
            b"CZ3",
            b"CH2",
        ] {
            assert!(
                t.atom(name(ring)).is_some(),
                "TRP missing indole atom {}",
                core::str::from_utf8(ring).unwrap_or("?")
            );
        }
    }

    #[test]
    fn da_has_primed_sugar_and_base_atoms() {
        let t = Nucleotide::A.template().expect("DA template");
        // Phosphate + primed sugar names must survive the quote stripping.
        for atom in [
            b"P".as_slice(),
            b"O5'",
            b"C5'",
            b"C4'",
            b"C3'",
            b"C1'",
            b"O4'",
        ] {
            assert!(
                t.atom(name(atom)).is_some(),
                "DA missing sugar/phosphate atom {}",
                core::str::from_utf8(atom).unwrap_or("?")
            );
        }
        // Adenine base atoms.
        for base in [b"N9".as_slice(), b"C8", b"N7", b"C5", b"C4", b"N6"] {
            assert!(
                t.atom(name(base)).is_some(),
                "DA missing base atom {}",
                core::str::from_utf8(base).unwrap_or("?")
            );
        }
    }
}
