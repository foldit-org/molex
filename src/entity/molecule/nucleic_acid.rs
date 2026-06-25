//! Nucleic acid entity: a single DNA or RNA chain instance.

use std::collections::HashMap;

use glam::Vec3;

use super::atom::{Atom, AtomColumns};
use super::complete::{complete_na_residues, keep_hydrogens, Completion};
use super::id::EntityId;
use super::protein::trimmed_atom_name;
use super::traits::{Entity, Polymer};
use super::{MoleculeType, Residue};
use crate::analysis::BondOrder;
use crate::atom_id::AtomId;
use crate::bond::CovalentBond;
use crate::chemistry::atom_name::AtomName;
use crate::chemistry::nucleotides::Nucleotide;
use crate::element::Element;

/// Pre-extracted ring geometry for a single nucleotide base.
#[derive(Debug, Clone)]
pub struct NucleotideRing {
    /// Hexagonal ring atom positions in order: N1, C2, N3, C4, C5, C6.
    pub hex_ring: Vec<Vec3>,
    /// Pentagonal ring for purines: C4, C5, N7, C8, N9 (empty for
    /// pyrimidines).
    pub pent_ring: Vec<Vec3>,
    /// NDB color for this base.
    pub color: [f32; 3],
    /// C1' sugar carbon position (for anchoring stem to backbone spline).
    pub c1_prime: Option<Vec3>,
}

/// A single DNA or RNA entity (one PDB chain instance).
#[derive(Debug, Clone)]
pub struct NAEntity {
    /// Unique entity identifier.
    pub id: EntityId,
    /// Molecule type (DNA or RNA).
    pub na_type: MoleculeType,
    /// All atoms in this entity, stored as parallel columns.
    ///
    /// After [`NAEntity::new`], atoms belonging to a kept residue are
    /// laid out in canonical order: `P, O5', C5', C4', C3', O3',
    /// base heavy..., hydrogens...`. A free 5'-OH terminal residue omits
    /// the leading `P` (its backbone starts at `O5'`). Atoms of dropped
    /// residues remain in `columns` but are not referenced by any
    /// `Residue::atom_range`.
    pub columns: AtomColumns,
    /// Ordered residues (nucleotides). Residues missing any required
    /// canonical backbone atom are dropped during construction; `P` is
    /// required except at the 5'-terminal residue.
    pub residues: Vec<Residue>,
    /// Backbone segment break indices.
    pub segment_breaks: Vec<usize>,
    /// Intra-entity covalent bonds with `AtomId` endpoints. Populated
    /// at construction from [`Nucleotide::bonds()`] tables plus
    /// inter-residue phosphodiester bonds `O3'(i)-P(i+1)` across kept
    /// consecutive residues.
    pub bonds: Vec<CovalentBond>,
    /// PDB chain identifier (the mmCIF `label_asym_id`, an arbitrary
    /// string).
    pub pdb_chain_id: String,
}

impl Entity for NAEntity {
    fn id(&self) -> EntityId {
        self.id
    }
    fn molecule_type(&self) -> MoleculeType {
        self.na_type
    }
    fn columns(&self) -> &AtomColumns {
        &self.columns
    }
    fn bonds(&self) -> &[CovalentBond] {
        &self.bonds
    }
}

const HEX_RING_ATOMS: &[&str] = &["N1", "C2", "N3", "C4", "C5", "C6"];
const PENT_RING_ATOMS: &[&str] = &["C4", "C5", "N7", "C8", "N9"];

fn ndb_base_color(res_name: &str) -> Option<[f32; 3]> {
    match res_name {
        "DA" | "A" | "ADE" | "RAD" => Some([0.85, 0.20, 0.20]),
        "DG" | "G" | "GUA" | "RGU" => Some([0.20, 0.80, 0.20]),
        "DC" | "C" | "CYT" | "RCY" => Some([0.90, 0.90, 0.20]),
        "DT" | "THY" => Some([0.20, 0.20, 0.85]),
        "DU" | "U" | "URA" => Some([0.20, 0.85, 0.85]),
        _ => None,
    }
}

fn is_purine(res_name: &str) -> bool {
    matches!(
        res_name,
        "DA" | "DG" | "DI" | "A" | "G" | "ADE" | "GUA" | "I" | "RAD" | "RGU"
    )
}

impl NAEntity {
    /// Construct an NA entity, enforcing canonical atom ordering.
    ///
    /// For every kept residue, atoms are laid out as
    /// `P, O5', C5', C4', C3', O3', base heavy..., hydrogens...` (the 5'
    /// terminus may omit the leading `P`). Residues missing any required
    /// canonical backbone atom are dropped from `residues` with a
    /// `log::warn!`; their atoms remain in `atoms` but are unreferenced.
    ///
    /// Construction is pure: the atom set is assembled exactly as given
    /// and no missing-atom completion runs. To fabricate missing atoms at
    /// construction (and rescue backbone-incomplete residues before the
    /// drop), use [`Self::new_normalized`]; to complete an already-built
    /// entity's surviving residues, use [`Self::normalize`].
    #[must_use]
    pub fn new(
        id: EntityId,
        na_type: MoleculeType,
        atoms: Vec<Atom>,
        residues: Vec<Residue>,
        pdb_chain_id: String,
    ) -> Self {
        build_na(id, na_type, atoms, residues, pdb_chain_id, Completion::Raw)
    }

    /// Construct an NA entity, fabricating missing atoms at construction
    /// in the given completion `mode`.
    ///
    /// Same canonicalize / segment-break / bond-graph construction as
    /// [`Self::new`], but the rigid-fit completion pass runs first (to
    /// `completion`), before the canonicalize reorder. Because completion
    /// runs before the drop, a backbone-incomplete but anchorable residue
    /// can have its backbone fabricated and survive, whereas the pure
    /// [`Self::new`] would drop it. The file-ingest path uses this with
    /// [`Completion::Heavy`].
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors NAEntity::new's argument set plus the completion \
                  level"
    )]
    pub fn new_normalized(
        id: EntityId,
        na_type: MoleculeType,
        atoms: Vec<Atom>,
        residues: Vec<Residue>,
        pdb_chain_id: String,
        completion: Completion,
    ) -> Self {
        build_na(id, na_type, atoms, residues, pdb_chain_id, completion)
    }

    /// Rebuild this entity with heavy-atom completion, fabricating the
    /// missing heavy base atoms a sparse parse omits. Identity, type,
    /// segment breaks, and existing atoms are preserved; completion
    /// matches every present atom against its template by name and places
    /// only the absent in-scope heavy atoms (idempotent: a second
    /// projection adds nothing).
    ///
    /// This re-runs completion on the entity's already-surviving
    /// residues; it cannot resurrect a residue dropped at construction
    /// for missing backbone atoms. To rescue backbone-incomplete
    /// residues, fabricate at construction with [`Self::new_normalized`].
    ///
    /// Atom indices shift because fabricated atoms are appended per
    /// residue and the atom set is re-canonicalized, so `AtomId`s are not
    /// stable across this projection.
    #[must_use]
    pub fn normalize(&self) -> Self {
        self.complete(Completion::Heavy)
    }

    /// Rebuild this entity with all-atom completion, additionally
    /// fabricating the template hydrogens [`Self::normalize`] omits.
    /// Same identity / index-shift / survivors-only caveats as
    /// [`Self::normalize`].
    #[must_use]
    pub fn to_all_atom(&self) -> Self {
        self.complete(Completion::AllAtom)
    }

    /// Re-extract this entity's atoms and residues and rebuild them at
    /// the given completion `level`. The canonical re-completion entry;
    /// [`Self::normalize`] and [`Self::to_all_atom`] are thin wrappers
    /// over it.
    #[must_use]
    pub fn complete(&self, level: Completion) -> Self {
        build_na(
            self.id,
            self.na_type,
            self.columns.to_atoms(),
            self.residues.clone(),
            self.pdb_chain_id.clone(),
            level,
        )
    }

    /// Extract phosphorus (P) atom positions, split into segments at
    /// gaps where consecutive P-P distance exceeds ~8 A.
    #[must_use]
    #[allow(
        clippy::excessive_nesting,
        reason = "gap-splitting logic is inherently nested"
    )]
    pub fn extract_p_atom_segments(&self) -> Vec<Vec<Vec3>> {
        const MAX_PHOSPHATE_BOND_DIST_SQ: f32 =
            MAX_PHOSPHATE_BOND_DIST * MAX_PHOSPHATE_BOND_DIST;

        let mut p_positions = Vec::new();
        for residue in &self.residues {
            for idx in residue.atom_range.clone() {
                let a = self.atom(idx);
                let name = std::str::from_utf8(&a.name).unwrap_or("").trim();
                if name == "P" {
                    p_positions.push(a.position);
                }
            }
        }

        // Split at large gaps (missing residues)
        let mut result = Vec::new();
        let mut segment = Vec::new();
        for pos in p_positions {
            if let Some(&prev) = segment.last() {
                if pos.distance_squared(prev) > MAX_PHOSPHATE_BOND_DIST_SQ {
                    if segment.len() >= 2 {
                        result.push(std::mem::take(&mut segment));
                    } else {
                        segment.clear();
                    }
                }
            }
            segment.push(pos);
        }
        if segment.len() >= 2 {
            result.push(segment);
        }

        result
    }

    /// Extract base ring geometry for each nucleotide residue.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn extract_base_rings(&self) -> Vec<NucleotideRing> {
        let mut rings = Vec::new();

        for residue in &self.residues {
            let res_name =
                std::str::from_utf8(&residue.name).unwrap_or("").trim();

            let Some(color) = ndb_base_color(res_name) else {
                continue;
            };

            let mut atom_map: HashMap<String, Vec3> = HashMap::new();
            for idx in residue.atom_range.clone() {
                let a = self.atom(idx);
                let name = std::str::from_utf8(&a.name)
                    .unwrap_or("")
                    .trim()
                    .trim_matches('\0')
                    .to_owned();
                let _ = atom_map.insert(name, a.position);
            }

            let hex_ring: Vec<Vec3> = HEX_RING_ATOMS
                .iter()
                .filter_map(|name| atom_map.get(*name).copied())
                .collect();
            if hex_ring.len() != 6 {
                continue;
            }

            let pent_ring = if is_purine(res_name) {
                let pent: Vec<Vec3> = PENT_RING_ATOMS
                    .iter()
                    .filter_map(|name| atom_map.get(*name).copied())
                    .collect();
                if pent.len() == 5 {
                    pent
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            let c1_prime =
                atom_map.get("C1'").or_else(|| atom_map.get("C1*")).copied();

            rings.push(NucleotideRing {
                hex_ring,
                pent_ring,
                color,
                c1_prime,
            });
        }

        rings
    }
}

/// Shared NA-entity construction: complete (to `level`), canonicalize,
/// derive segment breaks, and build the bond graph.
#[allow(
    clippy::needless_pass_by_value,
    reason = "owns the inputs; canonicalize borrows before reassigning new \
              owned vecs"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors NAEntity::new's argument set plus the completion level"
)]
fn build_na(
    id: EntityId,
    na_type: MoleculeType,
    atoms: Vec<Atom>,
    residues: Vec<Residue>,
    pdb_chain_id: String,
    level: Completion,
) -> NAEntity {
    let (atoms, residues) =
        complete_na_residues(&atoms, &residues, level, na_type);
    let (atoms, residues) = canonicalize_na_residues(
        &atoms,
        &residues,
        &pdb_chain_id,
        na_type,
        keep_hydrogens(level),
    );
    let segment_breaks = compute_na_segment_breaks(&atoms, &residues);
    let bonds = build_na_bonds(id, &atoms, &residues, &segment_breaks);
    NAEntity {
        id,
        na_type,
        columns: AtomColumns::from_atoms(atoms),
        residues,
        segment_breaks,
        bonds,
        pdb_chain_id,
    }
}

/// Maximum P->P distance (A) for consecutive nucleotides. Gaps exceeding
/// this indicate a backbone break.
const MAX_PHOSPHATE_BOND_DIST: f32 = 8.0;

/// Whether a kept residue carries a 5'-phosphate, i.e. its canonical
/// first atom is `P`. Canonicalize places `P` at local offset 0 for every
/// kept residue that has one; only the 5'-terminal residue may lack it (a
/// free 5'-OH nucleotide), in which case offset 0 is a sugar atom instead.
fn residue_has_phosphate(atoms: &[Atom], r: &Residue) -> bool {
    r.atom_range.start < atoms.len()
        && trimmed_atom_name(&atoms[r.atom_range.start].name) == b"P"
}

/// Compute segment break indices from P(i-1)->P(i) distances.
///
/// `P` sits at local offset 0 of every kept residue that has one; a
/// P-less 5'-terminal residue has a sugar atom there instead. A pair
/// whose earlier residue lacks `P` cannot have a P-P distance measured,
/// so it is left unbroken: the 5' terminus is the chain start and carries
/// no break of its own.
fn compute_na_segment_breaks(
    atoms: &[Atom],
    residues: &[Residue],
) -> Vec<usize> {
    let mut breaks = Vec::new();
    for i in 1..residues.len() {
        if !residue_has_phosphate(atoms, &residues[i - 1]) {
            continue;
        }
        let prev_p = atoms[residues[i - 1].atom_range.start].position;
        let curr_p = atoms[residues[i].atom_range.start].position;
        if prev_p.distance(curr_p) > MAX_PHOSPHATE_BOND_DIST {
            breaks.push(i);
        }
    }
    breaks
}

/// Canonical backbone atom names, in canonical order, for NA residues.
const NA_BACKBONE_NAMES: [&[u8]; 6] =
    [b"P", b"O5'", b"C5'", b"C4'", b"C3'", b"O3'"];

/// Reorganize NA atoms into canonical per-residue order and drop
/// residues missing a required canonical backbone atom. `P` is required
/// except at the 5'-terminal residue, where a free 5'-OH nucleotide
/// legitimately carries no phosphate.
///
/// Layout for kept residues: `P, O5', C5', C4', C3', O3', base
/// heavy..., hydrogens...` (the 5' terminus may omit the leading `P`).
/// Dropped residues' atoms are appended in input order and left
/// unreferenced.
///
/// When `keep_hydrogens` is false (the heavy-only file-ingest mode), the
/// parsed hydrogen bucket is dropped from every kept residue, yielding a
/// heavy-only entity from a protonated input.
struct NaResiduePartition {
    backbone_slots: [Option<usize>; 6],
    base_heavy: Vec<usize>,
    hydrogens: Vec<usize>,
}

fn partition_na_residue(
    atoms: &[Atom],
    range: std::ops::Range<usize>,
) -> NaResiduePartition {
    let mut out = NaResiduePartition {
        backbone_slots: [None; 6],
        base_heavy: Vec::new(),
        hydrogens: Vec::new(),
    };
    for idx in range {
        let trimmed = trimmed_atom_name(&atoms[idx].name);
        let is_h =
            atoms[idx].element == Element::H || trimmed.first() == Some(&b'H');
        let mut matched = false;
        for (slot_i, name) in NA_BACKBONE_NAMES.iter().enumerate() {
            if out.backbone_slots[slot_i].is_none() && trimmed == *name {
                out.backbone_slots[slot_i] = Some(idx);
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        if is_h {
            out.hydrogens.push(idx);
        } else {
            out.base_heavy.push(idx);
        }
    }
    out
}

/// Resolve the six canonical backbone slots into a kept layout.
///
/// All six (`P, O5', C5', C4', C3', O3'`) are required for an interior
/// residue. At the 5'-terminal residue (`is_chain_first`) the `P` slot is
/// optional: a free 5'-OH nucleotide carries the sugar/backbone atoms but
/// no phosphate, and must be kept rather than dropped. The returned vec
/// holds the present backbone atom indices in canonical order (length 5
/// for a P-less 5' terminus, 6 otherwise); the remaining sugar/base atoms
/// are not required and ride along separately.
fn na_backbone_complete(
    slots: &[Option<usize>; 6],
    is_chain_first: bool,
) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(6);
    for (i, slot) in slots.iter().enumerate() {
        match *slot {
            Some(idx) => out.push(idx),
            None if i == 0 && is_chain_first => {}
            None => return None,
        }
    }
    Some(out)
}

/// Canonical residue name for the rosetta-bridge contract: two-char
/// `DA`/`DC`/`DG`/`DT` for DNA, single-char `A`/`C`/`G`/`U` for RNA. The
/// raw parsed name (single-char DNA, three-letter alias, ...) is mapped
/// to this form so the bridge accepts the residue. A name that does not
/// resolve to a known base is left verbatim.
///
/// A DNA `U`/`DU` has no deoxy base type and is canonicalized to `DT`
/// here only if it parses to thymine; a true DNA uracil round-trips
/// through ribose-uracil chemistry elsewhere and is out of this map.
fn canonical_na_name(raw: [u8; 3], na_type: MoleculeType) -> [u8; 3] {
    let Some(nt) = Nucleotide::from_code(raw) else {
        return raw;
    };
    let name: &[u8] = match (na_type, nt) {
        (MoleculeType::DNA, Nucleotide::A) => b"DA",
        (MoleculeType::DNA, Nucleotide::C) => b"DC",
        (MoleculeType::DNA, Nucleotide::G) => b"DG",
        (MoleculeType::DNA, Nucleotide::T | Nucleotide::U) => b"DT",
        (_, Nucleotide::A) => b"A",
        (_, Nucleotide::C) => b"C",
        (_, Nucleotide::G) => b"G",
        (_, Nucleotide::T | Nucleotide::U) => b"U",
    };
    let mut out = [b' '; 3];
    out[..name.len()].copy_from_slice(name);
    out
}

fn canonicalize_na_residues(
    atoms: &[Atom],
    residues: &[Residue],
    pdb_chain_id: &str,
    na_type: MoleculeType,
    keep_hydrogens: bool,
) -> (Vec<Atom>, Vec<Residue>) {
    let mut new_atoms: Vec<Atom> = Vec::with_capacity(atoms.len());
    let mut new_residues: Vec<Residue> = Vec::new();

    // Residues arrive in N->C sequence order, so index 0 is the 5'
    // terminus (the same builder guarantee the completion pass relies on);
    // the `P` slot is optional only there.
    for (idx, residue) in residues.iter().enumerate() {
        let is_chain_first = idx == 0;
        let range = residue.atom_range.clone();
        let start = new_atoms.len();
        let p = partition_na_residue(atoms, range.clone());

        if let Some(bb) =
            na_backbone_complete(&p.backbone_slots, is_chain_first)
        {
            for &idx in &bb {
                new_atoms.push(atoms[idx].clone());
            }
            for idx in p.base_heavy {
                new_atoms.push(atoms[idx].clone());
            }
            if keep_hydrogens {
                for idx in p.hydrogens {
                    new_atoms.push(atoms[idx].clone());
                }
            }
            let end = new_atoms.len();
            new_residues.push(Residue {
                name: canonical_na_name(residue.name, na_type),
                label_seq_id: residue.label_seq_id,
                auth_seq_id: residue.auth_seq_id,
                auth_comp_id: residue.auth_comp_id,
                ins_code: residue.ins_code,
                atom_range: start..end,
                variants: residue.variants.clone(),
            });
        } else {
            let res_name = std::str::from_utf8(&residue.name).unwrap_or("???");
            log::warn!(
                "NAEntity chain '{}': dropping residue {} (name {}); missing \
                 backbone atoms (need O5', C5', C4', C3', O3'; P required \
                 except at the 5' terminus)",
                pdb_chain_id,
                residue.label_seq_id,
                res_name.trim(),
            );
            for idx in range {
                new_atoms.push(atoms[idx].clone());
            }
        }
    }

    (new_atoms, new_residues)
}

/// Emit intra-residue bonds for one NA residue (chemistry-table
/// bonds + terminal phosphate oxygens) into `bonds`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "atom indices are bounded by NA chain size (< u32::MAX)"
)]
fn emit_na_residue_bonds(
    entity_id: EntityId,
    atoms: &[Atom],
    r: &Residue,
    bonds: &mut Vec<CovalentBond>,
) {
    let mut name_to_idx: HashMap<AtomName, usize> = HashMap::new();
    for idx in r.atom_range.clone() {
        let key = AtomName::from_bytes(trimmed_atom_name(&atoms[idx].name));
        let _ = name_to_idx.insert(key, idx);
    }
    let Some(nt) = Nucleotide::from_code(r.name) else {
        return;
    };
    for (name_a, name_b) in nt.bonds() {
        if let (Some(&ia), Some(&ib)) =
            (name_to_idx.get(name_a), name_to_idx.get(name_b))
        {
            bonds.push(CovalentBond {
                a: AtomId {
                    entity: entity_id,
                    index: ia as u32,
                },
                b: AtomId {
                    entity: entity_id,
                    index: ib as u32,
                },
                order: BondOrder::Single,
            });
        }
    }
    // Phosphate terminal oxygens (OP1 / OP2 / OP3): emit only when
    // present. Not part of Nucleotide::bonds() because their PDB
    // presence is variable (5' terminus, deprotonated forms). A P-less
    // 5'-terminal residue has no phosphate at offset 0 and no OP oxygens,
    // so it gets no phosphate bonds.
    if residue_has_phosphate(atoms, r) {
        let phosphate_idx = r.atom_range.start; // canonical P
        for oxygen in [b"OP1", b"OP2", b"OP3"] {
            let key = AtomName::from_bytes(oxygen);
            if let Some(&ox_idx) = name_to_idx.get(&key) {
                bonds.push(CovalentBond {
                    a: AtomId {
                        entity: entity_id,
                        index: phosphate_idx as u32,
                    },
                    b: AtomId {
                        entity: entity_id,
                        index: ox_idx as u32,
                    },
                    order: BondOrder::Single,
                });
            }
        }
    }
}

/// Populate an NA entity's covalent bond graph.
#[allow(
    clippy::cast_possible_truncation,
    reason = "atom indices are bounded by NA chain size (< u32::MAX)"
)]
fn build_na_bonds(
    entity_id: EntityId,
    atoms: &[Atom],
    residues: &[Residue],
    segment_breaks: &[usize],
) -> Vec<CovalentBond> {
    use std::collections::HashSet;

    let mut bonds: Vec<CovalentBond> = Vec::new();
    for r in residues {
        emit_na_residue_bonds(entity_id, atoms, r, &mut bonds);
    }

    // Inter-residue phosphodiester bonds: O3'(i-1) -> P(i) where no break.
    // O3' is the last of the canonical backbone slots, at offset 5 of a
    // P-bearing residue but offset 4 of a P-less 5'-terminal residue (its
    // backbone is one atom shorter). `P(i)` is always present: only the
    // 5'-terminal residue may lack one, and it is never a `curr` here.
    let breaks: HashSet<usize> = segment_breaks.iter().copied().collect();
    for i in 1..residues.len() {
        if breaks.contains(&i) {
            continue;
        }
        let prev = &residues[i - 1];
        let curr = &residues[i];
        let o3_prime_offset = if residue_has_phosphate(atoms, prev) {
            5
        } else {
            4
        };
        bonds.push(CovalentBond {
            a: AtomId {
                entity: entity_id,
                index: (prev.atom_range.start + o3_prime_offset) as u32, // O3'
            },
            b: AtomId {
                entity: entity_id,
                index: curr.atom_range.start as u32, // P
            },
            order: BondOrder::Single,
        });
    }

    bonds
}

impl Polymer for NAEntity {
    fn residues(&self) -> &[Residue] {
        &self.residues
    }
    fn segment_breaks(&self) -> &[usize] {
        &self.segment_breaks
    }
}

#[cfg(test)]
#[path = "nucleic_acid_tests.rs"]
mod tests;
