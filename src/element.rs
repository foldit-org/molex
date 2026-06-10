//! Chemical element enum and associated data (CPK colors, radii, etc.).

/// Chemical element for atoms in a molecular structure.
///
/// Covers biologically-relevant elements found in proteins, nucleic acids,
/// ligands, ions, and waters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    /// Hydrogen
    H,
    /// Carbon
    C,
    /// Nitrogen
    N,
    /// Oxygen
    O,
    /// Sulfur
    S,
    /// Phosphorus
    P,
    /// Selenium
    Se,
    /// Iron
    Fe,
    /// Zinc
    Zn,
    /// Magnesium
    Mg,
    /// Calcium
    Ca,
    /// Sodium
    Na,
    /// Chlorine
    Cl,
    /// Potassium
    K,
    /// Manganese
    Mn,
    /// Cobalt
    Co,
    /// Nickel
    Ni,
    /// Copper
    Cu,
    /// Bromine
    Br,
    /// Iodine
    I,
    /// Fluorine
    F,
    /// Unrecognized or unsupported element.
    Unknown,
}

impl Element {
    /// Parse element from a 1-2 character symbol string (case-insensitive).
    #[must_use]
    pub fn from_symbol(s: &str) -> Self {
        match s.trim().to_uppercase().as_str() {
            "H" => Element::H,
            "C" => Element::C,
            "N" => Element::N,
            "O" => Element::O,
            "S" => Element::S,
            "P" => Element::P,
            "SE" => Element::Se,
            "FE" => Element::Fe,
            "ZN" => Element::Zn,
            "MG" => Element::Mg,
            "CA" => Element::Ca,
            "NA" => Element::Na,
            "CL" => Element::Cl,
            "K" => Element::K,
            "MN" => Element::Mn,
            "CO" => Element::Co,
            "NI" => Element::Ni,
            "CU" => Element::Cu,
            "BR" => Element::Br,
            "I" => Element::I,
            "F" => Element::F,
            _ => Element::Unknown,
        }
    }

    /// Infer element from a standard protein atom name (e.g., "CA" -> C, "OG"
    /// -> O, "SD" -> S).
    ///
    /// For standard protein atoms, the first alphabetic character of the
    /// atom name reliably identifies the element (e.g. `CA`, `CB`, `CG` all
    /// begin with `C` for carbon). Only the common biomolecular elements
    /// (C, N, O, S, H, P) are recognized; anything else maps to
    /// [`Element::Unknown`].
    #[must_use]
    pub fn from_atom_name(name: &str) -> Self {
        let name = name.trim();
        // Find first alphabetic character
        name.chars().find(|c| c.is_alphabetic()).map_or(
            Element::Unknown,
            |ch| match ch.to_ascii_uppercase() {
                'C' => Element::C,
                'N' => Element::N,
                'O' => Element::O,
                'S' => Element::S,
                'H' => Element::H,
                'P' => Element::P,
                _ => Element::Unknown,
            },
        )
    }

    /// Atomic number. Returns 0 for [`Element::Unknown`].
    #[must_use]
    pub fn atomic_number(self) -> u8 {
        match self {
            Element::H => 1,
            Element::C => 6,
            Element::N => 7,
            Element::O => 8,
            Element::F => 9,
            Element::Na => 11,
            Element::Mg => 12,
            Element::P => 15,
            Element::S => 16,
            Element::Cl => 17,
            Element::K => 19,
            Element::Ca => 20,
            Element::Mn => 25,
            Element::Fe => 26,
            Element::Co => 27,
            Element::Ni => 28,
            Element::Cu => 29,
            Element::Zn => 30,
            Element::Se => 34,
            Element::Br => 35,
            Element::I => 53,
            Element::Unknown => 0,
        }
    }

    /// Lookup by atomic number. Returns [`Element::Unknown`] for any
    /// number not in this enum.
    #[must_use]
    pub fn from_atomic_number(z: u8) -> Self {
        match z {
            1 => Element::H,
            6 => Element::C,
            7 => Element::N,
            8 => Element::O,
            9 => Element::F,
            11 => Element::Na,
            12 => Element::Mg,
            15 => Element::P,
            16 => Element::S,
            17 => Element::Cl,
            19 => Element::K,
            20 => Element::Ca,
            25 => Element::Mn,
            26 => Element::Fe,
            27 => Element::Co,
            28 => Element::Ni,
            29 => Element::Cu,
            30 => Element::Zn,
            34 => Element::Se,
            35 => Element::Br,
            53 => Element::I,
            _ => Element::Unknown,
        }
    }

    /// Standard CPK coloring (Corey-Pauling-Koltun).
    #[must_use]
    pub fn cpk_color(&self) -> [f32; 3] {
        match self {
            Element::H => [1.0, 1.0, 1.0],     // White
            Element::C => [0.4, 0.4, 0.4],     // Dark gray
            Element::N => [0.2, 0.2, 1.0],     // Blue
            Element::O => [1.0, 0.2, 0.2],     // Red
            Element::S => [1.0, 0.85, 0.2],    // Yellow
            Element::P => [1.0, 0.5, 0.0],     // Orange
            Element::Se => [1.0, 0.63, 0.0],   // Orange-yellow
            Element::Fe => [0.56, 0.25, 0.08], // Rust brown
            Element::Zn => [0.49, 0.50, 0.69], // Slate blue
            Element::Mg | Element::Ca => [0.0, 0.55, 0.0], // Dark green
            Element::Na => [0.67, 0.36, 0.95], // Purple
            Element::Cl => [0.12, 0.94, 0.12], // Green
            Element::K => [0.56, 0.25, 0.83],  // Violet
            Element::Mn => [0.61, 0.48, 0.78], // Purple-gray
            Element::Co => [0.94, 0.56, 0.63], // Pink
            Element::Ni => [0.31, 0.82, 0.31], // Green
            Element::Cu => [0.78, 0.50, 0.20], // Copper
            Element::Br => [0.65, 0.16, 0.16], // Dark red
            Element::I => [0.58, 0.0, 0.58],   // Purple
            Element::F => [0.56, 0.88, 0.31],  // Yellow-green
            Element::Unknown => [0.7, 0.7, 0.7], // Light gray
        }
    }

    /// Covalent radius in angstroms (Cambridge CSD values).
    #[must_use]
    pub fn covalent_radius(&self) -> f32 {
        match self {
            Element::H => 0.31,
            Element::C => 0.76,
            Element::N => 0.71,
            Element::O => 0.66,
            Element::S => 1.05,
            Element::P => 1.07,
            Element::Se | Element::Br => 1.20,
            Element::Fe | Element::Cu => 1.32,
            Element::Zn => 1.22,
            Element::Mg => 1.41,
            Element::Ca => 1.76,
            Element::Na => 1.66,
            Element::Cl => 1.02,
            Element::K => 2.03,
            Element::Mn | Element::I => 1.39,
            Element::Co => 1.26,
            Element::Ni => 1.24,
            Element::F => 0.57,
            Element::Unknown => 0.77,
        }
    }

    /// Van der Waals radius in angstroms.
    #[must_use]
    pub fn vdw_radius(&self) -> f32 {
        match self {
            Element::H => 1.20,
            Element::C | Element::Unknown => 1.70,
            Element::N => 1.55,
            Element::O => 1.52,
            Element::S | Element::P => 1.80,
            Element::Se => 1.90,
            Element::Fe | Element::Mn | Element::Co => 2.00,
            Element::Zn => 1.39,
            Element::Mg => 1.73,
            Element::Ca => 2.31,
            Element::Na => 2.27,
            Element::Cl => 1.75,
            Element::K => 2.75,
            Element::Ni => 1.63,
            Element::Cu => 1.40,
            Element::Br => 1.85,
            Element::I => 1.98,
            Element::F => 1.47,
        }
    }

    /// Two-character symbol (padded with space if single char).
    #[must_use]
    pub fn symbol(&self) -> &'static str {
        match self {
            Element::H => "H",
            Element::C => "C",
            Element::N => "N",
            Element::O => "O",
            Element::S => "S",
            Element::P => "P",
            Element::Se => "Se",
            Element::Fe => "Fe",
            Element::Zn => "Zn",
            Element::Mg => "Mg",
            Element::Ca => "Ca",
            Element::Na => "Na",
            Element::Cl => "Cl",
            Element::K => "K",
            Element::Mn => "Mn",
            Element::Co => "Co",
            Element::Ni => "Ni",
            Element::Cu => "Cu",
            Element::Br => "Br",
            Element::I => "I",
            Element::F => "F",
            Element::Unknown => "X",
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    // -- from_symbol --

    #[test]
    fn from_symbol_common_elements() {
        assert_eq!(Element::from_symbol("C"), Element::C);
        assert_eq!(Element::from_symbol("N"), Element::N);
        assert_eq!(Element::from_symbol("O"), Element::O);
        assert_eq!(Element::from_symbol("S"), Element::S);
        assert_eq!(Element::from_symbol("H"), Element::H);
        assert_eq!(Element::from_symbol("Fe"), Element::Fe);
        assert_eq!(Element::from_symbol("Zn"), Element::Zn);
        assert_eq!(Element::from_symbol("P"), Element::P);
    }

    #[test]
    fn from_symbol_case_insensitive() {
        assert_eq!(Element::from_symbol("fe"), Element::Fe);
        assert_eq!(Element::from_symbol("FE"), Element::Fe);
        assert_eq!(Element::from_symbol("zn"), Element::Zn);
        assert_eq!(Element::from_symbol("ZN"), Element::Zn);
        assert_eq!(Element::from_symbol("se"), Element::Se);
    }

    #[test]
    fn from_symbol_with_whitespace() {
        assert_eq!(Element::from_symbol(" C "), Element::C);
        assert_eq!(Element::from_symbol("  Fe  "), Element::Fe);
    }

    #[test]
    fn from_symbol_unknown() {
        assert_eq!(Element::from_symbol("Xx"), Element::Unknown);
        assert_eq!(Element::from_symbol(""), Element::Unknown);
        assert_eq!(Element::from_symbol("???"), Element::Unknown);
    }

    #[test]
    fn from_symbol_all_variants() {
        let cases = vec![
            ("H", Element::H),
            ("C", Element::C),
            ("N", Element::N),
            ("O", Element::O),
            ("S", Element::S),
            ("P", Element::P),
            ("SE", Element::Se),
            ("FE", Element::Fe),
            ("ZN", Element::Zn),
            ("MG", Element::Mg),
            ("CA", Element::Ca),
            ("NA", Element::Na),
            ("CL", Element::Cl),
            ("K", Element::K),
            ("MN", Element::Mn),
            ("CO", Element::Co),
            ("NI", Element::Ni),
            ("CU", Element::Cu),
            ("BR", Element::Br),
            ("I", Element::I),
            ("F", Element::F),
        ];
        for (sym, expected) in cases {
            assert_eq!(Element::from_symbol(sym), expected, "failed for {sym}");
        }
    }

    // -- from_atom_name --

    #[test]
    fn from_atom_name_standard_backbone() {
        assert_eq!(Element::from_atom_name("CA"), Element::C);
        assert_eq!(Element::from_atom_name("N"), Element::N);
        assert_eq!(Element::from_atom_name("O"), Element::O);
        assert_eq!(Element::from_atom_name("C"), Element::C);
    }

    #[test]
    fn from_atom_name_sidechain() {
        assert_eq!(Element::from_atom_name("OG"), Element::O);
        assert_eq!(Element::from_atom_name("SD"), Element::S);
        assert_eq!(Element::from_atom_name("NZ"), Element::N);
        assert_eq!(Element::from_atom_name("CB"), Element::C);
    }

    #[test]
    fn from_atom_name_with_leading_space() {
        assert_eq!(Element::from_atom_name(" CA "), Element::C);
        assert_eq!(Element::from_atom_name(" N  "), Element::N);
    }

    #[test]
    fn from_atom_name_hydrogen() {
        assert_eq!(Element::from_atom_name("H"), Element::H);
        assert_eq!(Element::from_atom_name("HA"), Element::H);
    }

    #[test]
    fn from_atom_name_phosphorus() {
        assert_eq!(Element::from_atom_name("P"), Element::P);
    }

    #[test]
    fn from_atom_name_unknown_element() {
        assert_eq!(Element::from_atom_name(""), Element::Unknown);
        assert_eq!(Element::from_atom_name("   "), Element::Unknown);
    }

    // -- symbol --

    #[test]
    fn symbol_roundtrip() {
        assert_eq!(Element::C.symbol(), "C");
        assert_eq!(Element::Fe.symbol(), "Fe");
        assert_eq!(Element::Zn.symbol(), "Zn");
        assert_eq!(Element::Unknown.symbol(), "X");
        assert_eq!(Element::H.symbol(), "H");
    }

    #[test]
    fn symbol_from_symbol_roundtrip() {
        let elements = vec![
            Element::H,
            Element::C,
            Element::N,
            Element::O,
            Element::S,
            Element::P,
            Element::Se,
            Element::Fe,
            Element::Zn,
            Element::Mg,
            Element::Ca,
            Element::Na,
            Element::Cl,
            Element::K,
            Element::Mn,
            Element::Co,
            Element::Ni,
            Element::Cu,
            Element::Br,
            Element::I,
            Element::F,
        ];
        for elem in elements {
            assert_eq!(
                Element::from_symbol(elem.symbol()),
                elem,
                "roundtrip failed for {elem:?}"
            );
        }
    }

    // -- covalent_radius --

    #[test]
    fn covalent_radius_known_values() {
        assert_eq!(Element::H.covalent_radius(), 0.31);
        assert_eq!(Element::C.covalent_radius(), 0.76);
        assert_eq!(Element::N.covalent_radius(), 0.71);
        assert_eq!(Element::O.covalent_radius(), 0.66);
        assert_eq!(Element::S.covalent_radius(), 1.05);
        assert_eq!(Element::Fe.covalent_radius(), 1.32);
        assert_eq!(Element::Zn.covalent_radius(), 1.22);
        assert_eq!(Element::Unknown.covalent_radius(), 0.77);
    }

    #[test]
    fn covalent_radius_positive() {
        let all = vec![
            Element::H,
            Element::C,
            Element::N,
            Element::O,
            Element::S,
            Element::P,
            Element::Se,
            Element::Fe,
            Element::Zn,
            Element::Mg,
            Element::Ca,
            Element::Na,
            Element::Cl,
            Element::K,
            Element::Mn,
            Element::Co,
            Element::Ni,
            Element::Cu,
            Element::Br,
            Element::I,
            Element::F,
            Element::Unknown,
        ];
        for e in all {
            assert!(
                e.covalent_radius() > 0.0,
                "{e:?} radius should be positive"
            );
        }
    }

    // -- cpk_color --

    #[test]
    fn cpk_color_carbon_dark_gray() {
        let c = Element::C.cpk_color();
        assert_eq!(c, [0.4, 0.4, 0.4]);
    }

    #[test]
    fn cpk_color_nitrogen_blue() {
        let c = Element::N.cpk_color();
        assert_eq!(c, [0.2, 0.2, 1.0]);
    }

    #[test]
    fn cpk_color_oxygen_red() {
        let c = Element::O.cpk_color();
        assert_eq!(c, [1.0, 0.2, 0.2]);
    }

    #[test]
    fn cpk_color_hydrogen_white() {
        let c = Element::H.cpk_color();
        assert_eq!(c, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn cpk_color_sulfur_yellow() {
        let c = Element::S.cpk_color();
        assert_eq!(c, [1.0, 0.85, 0.2]);
    }

    #[test]
    fn cpk_color_unknown_light_gray() {
        let c = Element::Unknown.cpk_color();
        assert_eq!(c, [0.7, 0.7, 0.7]);
    }

    #[test]
    fn cpk_color_all_in_range() {
        let all = vec![
            Element::H,
            Element::C,
            Element::N,
            Element::O,
            Element::S,
            Element::P,
            Element::Se,
            Element::Fe,
            Element::Zn,
            Element::Mg,
            Element::Ca,
            Element::Na,
            Element::Cl,
            Element::K,
            Element::Mn,
            Element::Co,
            Element::Ni,
            Element::Cu,
            Element::Br,
            Element::I,
            Element::F,
            Element::Unknown,
        ];
        for e in all {
            let color = e.cpk_color();
            for (ch, &val) in ["r", "g", "b"].iter().zip(color.iter()) {
                assert!(
                    (0.0..=1.0).contains(&val),
                    "{e:?} cpk_color {ch}={val} out of [0,1]"
                );
            }
        }
    }

    // -- vdw_radius --

    #[test]
    fn vdw_radius_known_values() {
        assert_eq!(Element::H.vdw_radius(), 1.20);
        assert_eq!(Element::C.vdw_radius(), 1.70);
        assert_eq!(Element::N.vdw_radius(), 1.55);
        assert_eq!(Element::O.vdw_radius(), 1.52);
    }

    #[test]
    fn vdw_radius_greater_than_covalent() {
        let all = vec![
            Element::H,
            Element::C,
            Element::N,
            Element::O,
            Element::S,
            Element::P,
            Element::Fe,
            Element::Zn,
            Element::Unknown,
        ];
        for e in all {
            assert!(
                e.vdw_radius() > e.covalent_radius(),
                "{:?}: vdw {} should be > covalent {}",
                e,
                e.vdw_radius(),
                e.covalent_radius()
            );
        }
    }
}
