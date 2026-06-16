//! Nucleic acid entity: a single DNA or RNA chain instance.

use std::collections::HashMap;

use glam::Vec3;

use super::atom::Atom;
use super::complete::complete_na_residues;
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
    /// All atoms in this entity.
    ///
    /// After [`NAEntity::new`], atoms belonging to a kept residue are
    /// laid out in canonical order: `P, O5', C5', C4', C3', O3',
    /// base heavy..., hydrogens...`. Atoms of dropped residues remain
    /// in `atoms` but are not referenced by any `Residue::atom_range`.
    pub atoms: Vec<Atom>,
    /// Ordered residues (nucleotides). Residues missing any of the
    /// six canonical backbone atoms are dropped during construction.
    pub residues: Vec<Residue>,
    /// Backbone segment break indices.
    pub segment_breaks: Vec<usize>,
    /// Intra-entity covalent bonds with `AtomId` endpoints. Populated
    /// at construction from [`Nucleotide::bonds()`] tables plus
    /// inter-residue phosphodiester bonds `O3'(i)-P(i+1)` across kept
    /// consecutive residues.
    pub bonds: Vec<CovalentBond>,
    /// PDB chain identifier byte (derived from `label_asym_id`).
    pub pdb_chain_id: u8,
    /// Author-side chain identifier byte, when distinct from
    /// `pdb_chain_id`. `None` means "same as `pdb_chain_id`."
    pub auth_asym_id: Option<u8>,
}

impl Entity for NAEntity {
    fn id(&self) -> EntityId {
        self.id
    }
    fn molecule_type(&self) -> MoleculeType {
        self.na_type
    }
    fn atoms(&self) -> &[Atom] {
        &self.atoms
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
    /// `P, O5', C5', C4', C3', O3', base heavy..., hydrogens...`.
    /// Residues missing any of the six canonical backbone atoms are
    /// dropped from `residues` with a `log::warn!`; their atoms
    /// remain in `atoms` but are unreferenced.
    #[must_use]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "constructor owns the inputs; canonicalize borrows before \
                  reassigning new owned vecs"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "id + atoms + residues + label/auth chain bytes are \
                  required; `na_type` additionally discriminates DNA from RNA \
                  (no equivalent on the protein side because Protein is its \
                  own kind)."
    )]
    pub fn new(
        id: EntityId,
        na_type: MoleculeType,
        atoms: Vec<Atom>,
        residues: Vec<Residue>,
        pdb_chain_id: u8,
        auth_asym_id: Option<u8>,
    ) -> Self {
        let (atoms, residues) = complete_na_residues(&atoms, &residues);
        let (atoms, residues) =
            canonicalize_na_residues(&atoms, &residues, pdb_chain_id);
        let segment_breaks = compute_na_segment_breaks(&atoms, &residues);
        let bonds = build_na_bonds(id, &atoms, &residues, &segment_breaks);
        Self {
            id,
            na_type,
            atoms,
            residues,
            segment_breaks,
            bonds,
            pdb_chain_id,
            auth_asym_id,
        }
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
                let name = std::str::from_utf8(&self.atoms[idx].name)
                    .unwrap_or("")
                    .trim();
                if name == "P" {
                    let a = &self.atoms[idx];
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
                let name = std::str::from_utf8(&self.atoms[idx].name)
                    .unwrap_or("")
                    .trim()
                    .trim_matches('\0')
                    .to_owned();
                let a = &self.atoms[idx];
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

/// Maximum P->P distance (A) for consecutive nucleotides. Gaps exceeding
/// this indicate a backbone break.
const MAX_PHOSPHATE_BOND_DIST: f32 = 8.0;

/// Compute segment break indices from P(i)->P(i+1) distances.
///
/// Relies on canonical atom ordering: kept residues have `P` at
/// local offset 0.
fn compute_na_segment_breaks(
    atoms: &[Atom],
    residues: &[Residue],
) -> Vec<usize> {
    let mut breaks = Vec::new();
    for i in 1..residues.len() {
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
/// residues missing any of the six canonical backbone atoms.
///
/// Layout for kept residues: `P, O5', C5', C4', C3', O3', base
/// heavy..., hydrogens...`. Dropped residues' atoms are appended in
/// input order and left unreferenced.
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

fn na_backbone_complete(slots: &[Option<usize>; 6]) -> Option<[usize; 6]> {
    let mut out = [0usize; 6];
    for (i, slot) in slots.iter().enumerate() {
        out[i] = (*slot)?;
    }
    Some(out)
}

fn canonicalize_na_residues(
    atoms: &[Atom],
    residues: &[Residue],
    pdb_chain_id: u8,
) -> (Vec<Atom>, Vec<Residue>) {
    let mut new_atoms: Vec<Atom> = Vec::with_capacity(atoms.len());
    let mut new_residues: Vec<Residue> = Vec::new();

    for residue in residues {
        let range = residue.atom_range.clone();
        let start = new_atoms.len();
        let p = partition_na_residue(atoms, range.clone());

        if let Some(bb) = na_backbone_complete(&p.backbone_slots) {
            for &idx in &bb {
                new_atoms.push(atoms[idx].clone());
            }
            for idx in p.base_heavy {
                new_atoms.push(atoms[idx].clone());
            }
            for idx in p.hydrogens {
                new_atoms.push(atoms[idx].clone());
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
        } else {
            let res_name = std::str::from_utf8(&residue.name).unwrap_or("???");
            log::warn!(
                "NAEntity chain '{}': dropping residue {} (name {}); missing \
                 backbone atoms (need P, O5', C5', C4', C3', O3')",
                pdb_chain_id as char,
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
    // presence is variable (5' terminus, deprotonated forms).
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

    // Inter-residue phosphodiester bonds: O3'(i) -> P(i+1) where no break.
    let breaks: HashSet<usize> = segment_breaks.iter().copied().collect();
    for i in 1..residues.len() {
        if breaks.contains(&i) {
            continue;
        }
        let prev = &residues[i - 1];
        let curr = &residues[i];
        bonds.push(CovalentBond {
            a: AtomId {
                entity: entity_id,
                index: (prev.atom_range.start + 5) as u32, // O3'
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::element::Element;
    use crate::entity::molecule::id::EntityIdAllocator;

    fn atom_with(name: &str, el: Element, x: f32) -> Atom {
        let mut n = [b' '; 4];
        for (i, b) in name.bytes().take(4).enumerate() {
            n[i] = b;
        }
        Atom {
            position: Vec3::new(x, 0.0, 0.0),
            occupancy: 1.0,
            b_factor: 0.0,
            element: el,
            name: n,
            formal_charge: 0,
        }
    }

    fn res_name_bytes(s: &str) -> [u8; 3] {
        let mut n = [b' '; 3];
        for (i, b) in s.bytes().take(3).enumerate() {
            n[i] = b;
        }
        n
    }

    /// Build an NAEntity with a single canonical adenine (DA) residue
    /// from a scrambled atom order.
    fn scrambled_adenine_entity() -> NAEntity {
        // Intended atoms for a single DA residue (DNA adenine):
        // Backbone: P, O5', C5', C4', C3', O3'
        // Sugar extras: O4', C1', C2'
        // Base: N9, C8, N7, C5, C4, N3, C2, N1, C6, N6
        // H: HA1' (non-existent but used for test)
        // Let's scramble.
        let atoms = vec![
            atom_with("O3'", Element::O, 1.0), // backbone position 5
            atom_with("C4'", Element::C, 2.0), // backbone position 3
            atom_with("N9", Element::N, 3.0),  // base
            atom_with("P", Element::P, 4.0),   // backbone position 0
            atom_with("C5'", Element::C, 5.0), // backbone position 2
            atom_with("O5'", Element::O, 6.0), // backbone position 1
            atom_with("C3'", Element::C, 7.0), // backbone position 4
            atom_with("H8", Element::H, 8.0),  // hydrogen
            atom_with("O4'", Element::O, 9.0), /* base (but really sugar);
                                                * treated as non-backbone */
            atom_with("C1'", Element::C, 10.0), // non-backbone heavy
        ];
        let residues = vec![Residue {
            name: res_name_bytes("DA "),
            label_seq_id: 1,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: 0..atoms.len(),
            variants: Vec::new(),
        }];
        let mut alloc = EntityIdAllocator::new();
        let id = alloc.allocate();
        NAEntity::new(id, MoleculeType::DNA, atoms, residues, b'A', None)
    }

    #[test]
    fn na_canonical_ordering_places_backbone_first() {
        let na = scrambled_adenine_entity();
        assert_eq!(na.residues.len(), 1);
        let r = &na.residues[0];
        let names: Vec<&str> = r
            .atom_range
            .clone()
            .map(|i| std::str::from_utf8(&na.atoms[i].name).unwrap().trim())
            .collect();
        // First six must be the canonical backbone.
        assert_eq!(&names[..6], &["P", "O5'", "C5'", "C4'", "C3'", "O3'"]);
    }

    #[test]
    fn na_bonds_include_sugar_phosphate_and_purine_anchor() {
        let na = scrambled_adenine_entity();
        // bonds are AtomId-endpoint. Reconstruct by name.
        let name_of = |aid: AtomId| {
            std::str::from_utf8(&na.atoms[aid.index as usize].name)
                .unwrap()
                .trim()
                .to_owned()
        };
        let pairs: Vec<(String, String)> = na
            .bonds
            .iter()
            .map(|b| (name_of(b.a), name_of(b.b)))
            .collect();
        // P-O5' should be present
        assert!(pairs.iter().any(|(a, b)| {
            (a == "P" && b == "O5'") || (a == "O5'" && b == "P")
        }));
        // C1'-N9 (sugar-base anchor for purine)
        assert!(pairs.iter().any(|(a, b)| {
            (a == "C1'" && b == "N9") || (a == "N9" && b == "C1'")
        }));
    }

    #[test]
    fn na_drops_residue_missing_backbone() {
        // Residue with only P and O5': missing four backbone atoms, and
        // with just two present atoms it falls below the three-atom
        // anchor floor, so the completion pass cannot rebuild the
        // backbone and the residue is still dropped.
        let atoms = vec![
            atom_with("P", Element::P, 0.0),
            atom_with("O5'", Element::O, 1.0),
        ];
        let residues = vec![Residue {
            name: res_name_bytes("DA "),
            label_seq_id: 1,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: 0..atoms.len(),
            variants: Vec::new(),
        }];
        let mut alloc = EntityIdAllocator::new();
        let id = alloc.allocate();
        let na =
            NAEntity::new(id, MoleculeType::DNA, atoms, residues, b'A', None);
        assert_eq!(na.residues.len(), 0);
        // Atoms still present (carried through, unreferenced).
        assert_eq!(na.atoms.len(), 2);
    }
}
