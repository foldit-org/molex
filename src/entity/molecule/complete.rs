//! Missing-atom completion for standard residues.
//!
//! Parsed structures routinely omit atoms (hydrogens, distal sidechain
//! atoms, sometimes whole sidechains beyond the backbone). This pass
//! brings each residue's atom set up to its ideal-geometry template: for
//! every residue that resolves to a standard amino acid or DNA base, it
//! rigid-fits the template's present atoms onto the parsed residue's
//! present atoms (reusing the Kabsch alignment) and places each absent
//! template atom at its transformed ideal coordinate.
//!
//! The pass is purely additive. Parsed atoms keep their coordinates;
//! only absent atoms are created, with chemically sane (not
//! bit-identical) positions. Protein C-termini gain their carboxylate
//! `OXT` (and, under all-atom, `HXT`). Leaving atoms and the N-terminal
//! extra protons stay out of scope and are never fabricated. A free
//! 5'-OH nucleotide keeps its absent 5'-phosphate group (`P`/`OP1`/`OP2`)
//! absent rather than having a phantom phosphate forged.
//!
//! Residues that do not resolve to a template, that resolve to a base
//! with no template in their strand (RNA thymine, DNA uridine), that
//! have fewer than three matched atoms, or whose matched atoms are
//! collinear (all on a single line) pass through unchanged: those are
//! the only configurations too
//! under-determined to anchor a unique rigid fit. Three or more matched
//! atoms that are not collinear fix the fit completely (coplanar is
//! enough; the reflection is resolved by the Kabsch determinant
//! correction). The pass runs before the canonicalize reorder, so a
//! residue still missing backbone atoms after completion drops exactly
//! as it did before.

use glam::{Mat3, Vec3};

use super::atom::{pad_atom_name, Atom};
use super::polymer::Residue;
use super::protein::trimmed_atom_name;
use super::MoleculeType;
use crate::chemistry::amino_acids::AminoAcid;
use crate::chemistry::atom_name::AtomName;
use crate::chemistry::completion::{ResidueTemplate, TemplateAtom};
use crate::chemistry::nucleotides::Nucleotide;
use crate::element::Element;
use crate::ops::kabsch_alignment;

/// Minimum number of template atoms that must be present in both the
/// parse and the template to anchor a rigid fit (Kabsch needs >= 3).
const MIN_ANCHOR_ATOMS: usize = 3;

/// Relative tolerance for the collinearity screen in [`is_collinear`].
/// Matched anchors count as collinear when the largest perpendicular
/// excursion from their dominant axis is below this fraction of that
/// axis's length. At 1e-3 a residue must bulge off the line by at least
/// a thousandth of its own anchor span to be accepted; real residue
/// backbones bulge by tens of percent, so this rejects only genuinely
/// straight-line inputs while staying comfortably clear of f32 round-off.
const COLLINEAR_TOLERANCE: f32 = 1e-3;

/// A monotonic completion level: `Raw < Heavy < AllAtom`.
///
/// `Raw` skips completion entirely: the residue's atom set is used
/// exactly as given, fabricating nothing and preserving any input
/// hydrogens. It is the pure construction level (wire round-trip, array
/// ingest, continuous rebuild), where inputs are already complete and
/// re-running the rigid fit is pure waste. `Heavy` fabricates absent
/// heavy template atoms and drops input hydrogens; the host and rosetta
/// consume heavy atoms only and re-derive hydrogens downstream, so it is
/// the file-ingest level. `AllAtom` additionally places template
/// hydrogens for callers that need a fully protonated model.
///
/// Callers select a level at construction through the opt-in completing
/// constructors ([`super::protein::ProteinEntity::new_normalized`] /
/// [`super::nucleic_acid::NAEntity::new_normalized`]); the plain `new`
/// constructors are pure (`Raw`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completion {
    /// Atoms exactly as given: skip completion, keep input hydrogens.
    Raw,
    /// Fabricate absent heavy template atoms (the file-ingest level).
    Heavy,
    /// Fabricate absent heavy atoms and template hydrogens.
    AllAtom,
}

/// Whether canonicalize should carry parsed hydrogens through into the
/// output entity. Only [`Completion::Heavy`] strips them: it is the
/// file-ingest level, where the host and rosetta consume heavy atoms
/// only, so a fully-protonated input PDB must not leak its hydrogens into
/// the canonical entity. `Raw` and `AllAtom` keep them (the former by
/// preserving the parse verbatim, the latter because hydrogens are part
/// of the all-atom model).
pub(super) fn keep_hydrogens(level: Completion) -> bool {
    level != Completion::Heavy
}

/// What a single residue's completion is allowed to fabricate: the
/// completion `level` plus which polymer end the residue sits at (a lone
/// residue is both termini at once). Decided by [`in_scope`].
#[derive(Clone, Copy)]
struct Scope {
    level: Completion,
    is_chain_first: bool,
    is_chain_last: bool,
}

/// Complete missing atoms in protein residues against amino-acid
/// templates.
///
/// Returns fresh `(Vec<Atom>, Vec<Residue>)` with absent template atoms
/// appended to each completable residue and every residue's `atom_range`
/// recomputed contiguously. Residues whose name does not resolve to a
/// standard amino acid, or that are too sparse to fit, are emitted
/// unchanged.
///
/// `level` selects what is fabricated: [`Completion::Raw`] skips
/// the pass and returns the atoms unchanged; [`Completion::Heavy`]
/// (the file-ingest path) places only heavy atoms; [`Completion::AllAtom`]
/// additionally places template hydrogens.
pub(super) fn complete_protein_residues(
    atoms: &[Atom],
    residues: &[Residue],
    level: Completion,
) -> (Vec<Atom>, Vec<Residue>) {
    complete_residues(atoms, residues, level, |name| {
        AminoAcid::from_code(name).map(AminoAcid::template)
    })
}

/// Complete missing atoms in nucleic-acid residues against the strand's
/// base templates.
///
/// Same contract as [`complete_protein_residues`] (including the `level`
/// semantics); resolution goes through [`Nucleotide`] in the chemistry
/// the entity's `na_type` selects, so an RNA chain completes against
/// ribose (2'-OH) templates and a DNA chain against deoxyribose ones.
/// Bases with no template in that strand (RNA thymine, DNA uridine) pass
/// through unchanged.
pub(super) fn complete_na_residues(
    atoms: &[Atom],
    residues: &[Residue],
    level: Completion,
    na_type: MoleculeType,
) -> (Vec<Atom>, Vec<Residue>) {
    complete_residues(atoms, residues, level, |name| {
        Nucleotide::from_code(name).and_then(|n| n.template(na_type))
    })
}

/// Shared completion driver. `resolve` maps a residue name to its
/// template (or `None` for residues this pass leaves untouched).
fn complete_residues(
    atoms: &[Atom],
    residues: &[Residue],
    level: Completion,
    resolve: impl Fn([u8; 3]) -> Option<&'static ResidueTemplate>,
) -> (Vec<Atom>, Vec<Residue>) {
    let mut new_atoms: Vec<Atom> = Vec::with_capacity(atoms.len());
    let mut new_residues: Vec<Residue> = Vec::with_capacity(residues.len());

    // The completing path receives residues in true N->C sequence order
    // (the builder sorts by `(label_seq_id, ins_code)` and the flatten
    // preserves it), so the first and last residues are the polymer
    // termini. Terminus-only completion keys off these indices.
    for (idx, residue) in residues.iter().enumerate() {
        let scope = Scope {
            level,
            is_chain_first: idx == 0,
            is_chain_last: idx + 1 == residues.len(),
        };
        let start = new_atoms.len();

        // Always carry the parsed atoms through unchanged.
        for idx in residue.atom_range.clone() {
            new_atoms.push(atoms[idx].clone());
        }

        if level != Completion::Raw {
            if let Some(template) = resolve(residue.name) {
                append_missing_atoms(
                    atoms,
                    residue.atom_range.clone(),
                    template,
                    scope,
                    &mut new_atoms,
                );
            }
        }

        let end = new_atoms.len();
        new_residues.push(Residue {
            name: residue.name,
            label_seq_id: residue.label_seq_id,
            auth_seq_id: residue.auth_seq_id,
            auth_comp_id: residue.auth_comp_id,
            ins_code: residue.ins_code,
            atom_range: start..end,
            variants: residue.variants.clone(),
        });
    }

    (new_atoms, new_residues)
}

/// Rigid-fit `template` onto the parsed atoms in `range` and append each
/// missing in-scope template atom to `out`. No-op when there is
/// nothing missing, when fewer than [`MIN_ANCHOR_ATOMS`] atoms match, or
/// when the matched anchors are collinear. `kabsch_alignment` itself has
/// no rank check (it only rejects length mismatch or `< 3` points), so
/// the collinear case is screened here at the call site: a unique
/// rotation needs the anchors to span at least two dimensions, and the
/// only way `>= 3` points fail that is by lying on one line.
///
/// `scope` carries the completion level and which polymer end this
/// residue sits at, so [`in_scope`] can admit the terminus-only atoms
/// there.
fn append_missing_atoms(
    atoms: &[Atom],
    range: std::ops::Range<usize>,
    template: &ResidueTemplate,
    scope: Scope,
    out: &mut Vec<Atom>,
) {
    // Matched: template atoms whose name is present in the parse, paired
    // (parsed_position, template_ideal) in one consistent order. Missing:
    // in-scope template atoms whose name is absent from the parse.
    let mut parsed_matched: Vec<Vec3> = Vec::new();
    let mut template_matched: Vec<Vec3> = Vec::new();
    let mut missing: Vec<&TemplateAtom> = Vec::new();

    // A free 5'-OH nucleotide carries no phosphate, and the 5'-phosphate
    // group (P/OP1/OP2, identifiable only by name) must not be invented
    // there: fabricating it would forge a terminus the structure does not
    // have. Detect the group's presence by `P`; when it is present (a real
    // 5' phosphate, even a partial one) the rest of the group completes
    // normally. Protein templates name no P/OP1/OP2, so this is inert for
    // them.
    let phosphate_present =
        present_position(atoms, range.clone(), AtomName::from_bytes(b"P"))
            .is_some();

    for t in template.atoms {
        if scope.is_chain_first
            && !phosphate_present
            && is_five_prime_phosphate_atom(t.name)
        {
            continue;
        }
        match present_position(atoms, range.clone(), t.name) {
            Some(pos) => {
                parsed_matched.push(pos);
                template_matched.push(t.ideal);
            }
            None if in_scope(t, scope) => {
                missing.push(t);
            }
            None => {}
        }
    }

    if missing.is_empty() {
        return;
    }
    if parsed_matched.len() < MIN_ANCHOR_ATOMS {
        log::debug!(
            "atom completion: residue {} has {} matched atoms (need {}); \
             leaving {} missing atom(s) unfilled",
            template.comp_str(),
            parsed_matched.len(),
            MIN_ANCHOR_ATOMS,
            missing.len(),
        );
        return;
    }
    if is_collinear(&parsed_matched) {
        log::debug!(
            "atom completion: residue {} has {} collinear matched atoms (no \
             unique rigid fit); leaving {} missing atom(s) unfilled",
            template.comp_str(),
            parsed_matched.len(),
            missing.len(),
        );
        return;
    }

    // kabsch_alignment(reference, target) returns (rot, trans) with
    // aligned = rot * target + trans. We want template-frame ideals
    // mapped into the parsed frame, so reference = parsed, target =
    // template (same atoms, same order), then a missing atom's placed
    // position is rot * ideal + trans. The collinearity screen above
    // guarantees the matched anchors span >= 2 dimensions, so the fit is
    // well-determined and kabsch_alignment cannot return None here (the
    // length and count preconditions are met by construction).
    let Some((rot, trans)) =
        kabsch_alignment(&parsed_matched, &template_matched)
    else {
        return;
    };

    for t in missing {
        out.push(placed_atom(t, rot, trans));
    }
}

/// Whether all of `points` lie within tolerance of a single straight
/// line, i.e. their spread is effectively one-dimensional. Such a set
/// fixes a rotation only up to a free turn about that line, so a rigid
/// fit anchored on it is not unique.
///
/// The test takes the longest edge from the first point as the candidate
/// line direction `u`, then measures every point's perpendicular
/// distance from that line. The set is treated as collinear when the
/// largest such perpendicular distance is below
/// [`COLLINEAR_TOLERANCE`] times the line's own length, i.e. when no
/// point bulges off the axis by even a small fraction of the anchor
/// spread. Scaling the threshold by the spread keeps the test
/// scale-free (it does not care how large the residue is in angstroms).
fn is_collinear(points: &[Vec3]) -> bool {
    // Longest edge from points[0]; this is the most numerically stable
    // axis estimate. `< 2` points can never be non-collinear, but the
    // caller has already enforced MIN_ANCHOR_ATOMS >= 3.
    let p0 = points[0];
    let mut axis = Vec3::ZERO;
    let mut axis_len_sq = 0.0f32;
    for &p in &points[1..] {
        let edge = p - p0;
        let len_sq = edge.length_squared();
        if len_sq > axis_len_sq {
            axis_len_sq = len_sq;
            axis = edge;
        }
    }
    // All points coincide: degenerate, and the < MIN_ANCHOR_ATOMS guard
    // is the intended gate, but treat as collinear for safety.
    if axis_len_sq <= f32::EPSILON {
        return true;
    }
    let axis_len = axis_len_sq.sqrt();
    let u = axis / axis_len;
    // Largest perpendicular component of any point from the line through
    // p0 along u.
    let mut max_perp = 0.0f32;
    for &p in &points[1..] {
        let edge = p - p0;
        let perp = (edge - u * edge.dot(u)).length();
        max_perp = max_perp.max(perp);
    }
    max_perp < COLLINEAR_TOLERANCE * axis_len
}

/// Whether a template atom is one of the three 5'-terminal phosphate-group
/// atoms (`P`, `OP1`, `OP2`). These carry no CCD terminus flag, so the
/// group is recognized by name alone. Used to suppress fabricating a
/// 5'-phosphate that a free 5'-OH nucleotide does not carry.
fn is_five_prime_phosphate_atom(name: AtomName) -> bool {
    name == AtomName::from_bytes(b"P")
        || name == AtomName::from_bytes(b"OP1")
        || name == AtomName::from_bytes(b"OP2")
}

/// Whether a template atom is in scope for completion. Interior body
/// atoms are always in scope. Terminus-only atoms are admitted only at
/// the matching polymer end: C-terminal atoms when `is_chain_last`,
/// N-terminal atoms when `is_chain_first`. Leaving atoms are never
/// fabricated, and under [`Completion::Heavy`] hydrogens are excluded.
///
/// In practice, on a protein C-terminus under `Heavy` the only NEW heavy
/// atom this admits is the carboxylate `OXT`: the C-terminal C and O
/// carry `is_c_terminal` but are present in every residue (matched as
/// anchors, never reaching the missing list), and `HXT` is dropped as a
/// hydrogen. N-terminal extras are all hydrogens, so the N-terminus is a
/// heavy no-op. Under `AllAtom` the same rule additionally places `HXT`
/// and the N-terminal protons, consistently.
///
/// The NA 5'-terminal phosphate group is handled separately in
/// [`append_missing_atoms`] (it carries no CCD flag this could key on).
fn in_scope(t: &TemplateAtom, scope: Scope) -> bool {
    if scope.level == Completion::Heavy && t.element == Element::H {
        return false;
    }
    if !t.is_leaving && !t.is_n_terminal && !t.is_c_terminal {
        return true;
    }
    if t.is_c_terminal && scope.is_chain_last {
        return true;
    }
    if t.is_n_terminal && scope.is_chain_first {
        return true;
    }
    false
}

/// Position of the parsed atom named `name` within `range`, if present.
/// Matching is by trimmed name only, never by index.
///
/// On a residue that carries duplicate atom names (e.g. altloc remnants),
/// this anchors the rigid fit on the FIRST occurrence; the later
/// duplicates are not matched here and so flow through `complete_residues`
/// unchanged. This is deliberate, not an oversight: do not "harden" it
/// into a duplicate-name panic.
fn present_position(
    atoms: &[Atom],
    mut range: std::ops::Range<usize>,
    name: AtomName,
) -> Option<Vec3> {
    range
        .find(|&idx| {
            AtomName::from_bytes(trimmed_atom_name(&atoms[idx].name)) == name
        })
        .map(|idx| atoms[idx].position)
}

/// Build a fresh `Atom` for a missing template atom at its transformed
/// ideal position. Occupancy 1.0, B-factor 0.0, neutral charge: these
/// are placed, not observed.
fn placed_atom(t: &TemplateAtom, rot: Mat3, trans: Vec3) -> Atom {
    let position = rot * t.ideal + trans;
    Atom {
        position,
        occupancy: 1.0,
        b_factor: 0.0,
        element: t.element,
        // Space-padded to the convention every parser uses for
        // `Atom.name`; `trimmed_atom_name` strips it back off.
        name: pad_atom_name(t.name.as_str()),
        formal_charge: 0,
    }
}

#[cfg(test)]
#[path = "complete_tests.rs"]
mod tests;
