//! Parse DSSP-style secondary structure annotation strings.

use crate::analysis::SSType;

/// Parse a DSSP-style secondary structure annotation string.
///
/// Character mapping:
/// - `H`, `G`, `I` -> Helix (alpha-helix, 3_1_0-helix, pi-helix)
/// - `E`, `B` -> Sheet (strand, isolated bridge)
/// - Everything else -> Coil
///
/// Also handles simplified notation where lowercase is accepted.
#[must_use]
pub fn from_string(ss: &str) -> Vec<SSType> {
    ss.chars()
        .map(|c| match c.to_ascii_uppercase() {
            'H' | 'G' | 'I' => SSType::Helix,
            'E' | 'B' => SSType::Sheet,
            _ => SSType::Coil,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let result = from_string("HHHEEECCC");
        assert_eq!(
            result,
            vec![
                SSType::Helix,
                SSType::Helix,
                SSType::Helix,
                SSType::Sheet,
                SSType::Sheet,
                SSType::Sheet,
                SSType::Coil,
                SSType::Coil,
                SSType::Coil,
            ]
        );
    }

    #[test]
    fn dssp_codes() {
        let result = from_string("HGIEBS T");
        assert_eq!(result[0], SSType::Helix);
        assert_eq!(result[1], SSType::Helix);
        assert_eq!(result[2], SSType::Helix);
        assert_eq!(result[3], SSType::Sheet);
        assert_eq!(result[4], SSType::Sheet);
        assert_eq!(result[5], SSType::Coil);
        assert_eq!(result[6], SSType::Coil);
        assert_eq!(result[7], SSType::Coil);
    }

    #[test]
    fn lowercase() {
        let result = from_string("hhe");
        assert_eq!(result, vec![SSType::Helix, SSType::Helix, SSType::Sheet]);
    }

    #[test]
    fn empty() {
        assert!(from_string("").is_empty());
    }
}
