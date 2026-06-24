//! Legacy atom-name normalization to PDB v3 conventions.
//!
//! Pre-2007 PDB files name nucleic-acid sugar atoms with star-primes
//! (`O5*`, `C4*`, ...) and phosphate oxygens as `O1P`/`O2P`. The
//! classification atom-set probe and the canonical NA backbone partition
//! both match against v3 names (apostrophe-prime, `OP1`/`OP2`), so the
//! raw parsed name is rewritten here at ingest before either consumer
//! sees it. Applied at the builder choke point so PDB, mmCIF, and
//! BinaryCIF all inherit it.

/// Rewrite one parsed 4-byte atom name into PDB v3 form.
///
/// Two heavy-atom rules, in order:
///
/// 1. Every `*` byte becomes `'` (apostrophe). No canonical atom name contains
///    `*`, so this blanket swap is safe and fixes all sugar primes (`O5*` ->
///    `O5'`). It also rewrites the prime inside numeric-prefix hydrogen names
///    (`1H5*` -> `1H5'`); those H names are otherwise left untouched (the
///    heavy-completion path sorts hydrogens by element and never matches them
///    by name, and the rosetta bridge already aliases the numeric-prefix form).
/// 2. Phosphate oxygen rename: `O1P` -> `OP1`, `O2P` -> `OP2`, `O3P` -> `OP3`.
///    These carry no `*`, so rule 1 leaves them.
#[must_use]
pub(super) fn normalize_legacy_atom_name(name: [u8; 4]) -> [u8; 4] {
    let mut out = name;
    for b in &mut out {
        if *b == b'*' {
            *b = b'\'';
        }
    }
    match &out {
        b"O1P " => *b"OP1 ",
        b"O2P " => *b"OP2 ",
        b"O3P " => *b"OP3 ",
        _ => out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(s: &str) -> [u8; 4] {
        let mut b = [b' '; 4];
        for (i, c) in s.bytes().take(4).enumerate() {
            b[i] = c;
        }
        b
    }

    #[test]
    fn star_primes_become_apostrophes() {
        assert_eq!(normalize_legacy_atom_name(n("O5*")), n("O5'"));
        assert_eq!(normalize_legacy_atom_name(n("C4*")), n("C4'"));
        assert_eq!(normalize_legacy_atom_name(n("C1*")), n("C1'"));
        assert_eq!(normalize_legacy_atom_name(n("O2*")), n("O2'"));
    }

    #[test]
    fn phosphate_oxygens_renamed() {
        assert_eq!(normalize_legacy_atom_name(n("O1P")), n("OP1"));
        assert_eq!(normalize_legacy_atom_name(n("O2P")), n("OP2"));
        assert_eq!(normalize_legacy_atom_name(n("O3P")), n("OP3"));
    }

    #[test]
    fn numeric_prefix_hydrogen_prime_swapped_name_left_otherwise() {
        // The prime is fixed; the numeric-prefix H form is otherwise
        // unchanged.
        assert_eq!(normalize_legacy_atom_name(n("1H5*")), n("1H5'"));
        assert_eq!(normalize_legacy_atom_name(n("2H2*")), n("2H2'"));
    }

    #[test]
    fn v3_names_pass_through_unchanged() {
        assert_eq!(normalize_legacy_atom_name(n("O5'")), n("O5'"));
        assert_eq!(normalize_legacy_atom_name(n("OP1")), n("OP1"));
        assert_eq!(normalize_legacy_atom_name(n("P")), n("P"));
        assert_eq!(normalize_legacy_atom_name(n("CA")), n("CA"));
    }
}
