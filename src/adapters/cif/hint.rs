//! Entity-type hint resolution from `_entity.type` and `_entity_poly.type`.

use crate::entity::molecule::ExpectedEntityType;

/// Map `_entity.type` (and the joined `_entity_poly.type` for polymers) to
/// an [`ExpectedEntityType`]. Unrecognised values collapse to
/// [`ExpectedEntityType::Unknown`].
pub(super) fn resolve_hint(
    entity_type: &str,
    poly_type: Option<&str>,
) -> ExpectedEntityType {
    match entity_type.trim() {
        "water" => ExpectedEntityType::Water,
        "polymer" => match poly_type.map(str::trim) {
            Some("polypeptide(L)" | "polypeptide(D)") => {
                ExpectedEntityType::Protein
            }
            Some(
                "polydeoxyribonucleotide"
                | "polydeoxyribonucleotide/polyribonucleotide hybrid",
            ) => ExpectedEntityType::DNA,
            Some("polyribonucleotide") => ExpectedEntityType::RNA,
            other => {
                log::debug!(
                    "cif: unrecognised _entity_poly.type {other:?}; falling \
                     back to Unknown"
                );
                ExpectedEntityType::Unknown
            }
        },
        "non-polymer" | "branched" | "macrolide" => {
            ExpectedEntityType::NonPolymer
        }
        other => {
            log::debug!(
                "cif: unrecognised _entity.type {other:?}; falling back to \
                 Unknown"
            );
            ExpectedEntityType::Unknown
        }
    }
}
