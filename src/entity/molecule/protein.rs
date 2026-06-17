//! Protein types and entity.
//!
//! Per-residue types (`ResidueBackbone`, `Sidechain`) and the
//! `ProteinEntity` chain instance with segment break metadata.

use glam::Vec3;

use super::atom::Atom;
use super::complete::{
    complete_protein_residues, keep_hydrogens, CompletionMode,
};
use super::id::EntityId;
use super::traits::{Entity, Polymer};
use super::{MoleculeType, Residue};
use crate::analysis::BondOrder;
use crate::atom_id::AtomId;
use crate::bond::CovalentBond;
use crate::chemistry::amino_acids::AminoAcid;
use crate::chemistry::atom_name::AtomName;
use crate::element::Element;

// Per-residue types

/// Backbone atom positions for a single protein residue.
///
/// Named fields, fixed size, `Copy`. Used by DSSP H-bond detection
/// and secondary structure classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidueBackbone {
    /// Nitrogen atom position.
    pub n: Vec3,
    /// Alpha-carbon atom position.
    pub ca: Vec3,
    /// Carbonyl carbon atom position.
    pub c: Vec3,
    /// Carbonyl oxygen atom position.
    pub o: Vec3,
}

/// Sidechain atoms and bond topology for a single protein residue.
///
/// `Vec<Atom>` holds the atoms (positions + names + elements). Bond
/// indices are local to this sidechain's `Vec<Atom>` (0-based). The
/// backbone-to-sidechain bond (CA->CB) is implicit: `ResidueBackbone.ca`
/// connects to the first atom in `atoms` when the sidechain is non-empty.
#[derive(Debug, Clone)]
pub struct Sidechain {
    /// Sidechain atoms (CB, CG, SG, etc.).
    pub atoms: Vec<Atom>,
    /// Intra-residue bonds as `(atom_idx_a, atom_idx_b)` pairs, local
    /// to this sidechain's `Vec<Atom>`.
    pub bonds: Vec<(usize, usize)>,
}

impl Sidechain {
    /// Create an empty sidechain (e.g. for glycine).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            atoms: Vec::new(),
            bonds: Vec::new(),
        }
    }

    /// Whether this sidechain has no atoms (glycine).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }
}

// Protein entity

/// A single protein chain instance.
///
/// Contains all atoms, residue metadata, and precomputed segment breaks
/// (backbone gaps from missing residues, detected by C-N distance).
///
/// Protein-specific derived views (`to_backbone`) are available as methods.
/// These derive from the underlying `atoms` + `residues` on each call; no
/// cached copies.
#[derive(Debug, Clone)]
pub struct ProteinEntity {
    /// Unique entity identifier.
    pub id: EntityId,
    /// All atoms in this protein chain.
    ///
    /// After [`ProteinEntity::new`], atoms belonging to a kept residue are
    /// laid out in canonical order: `N, CA, C, O, sidechain heavy...,
    /// hydrogens...`. Atoms of dropped residues remain in `atoms` but are
    /// not referenced by any `Residue::atom_range`.
    pub atoms: Vec<Atom>,
    /// Ordered residues (name, number, atom range into `atoms`). Residues
    /// missing any of the four backbone atoms (N, CA, C, O or OXT) are
    /// dropped during construction.
    pub residues: Vec<Residue>,
    /// Indices into `residues` where backbone segments start a new
    /// continuous run. Computed from C(i)->N(i+1) distance > 2.0A.
    pub segment_breaks: Vec<usize>,
    /// Intra-entity covalent bonds with `AtomId` endpoints. Populated at
    /// construction from `AminoAcid::bonds()` tables (intra-residue) plus
    /// universal backbone bonds (N-CA, CA-C, C=O) and inter-residue peptide
    /// bonds `C(i)-N(i+1)` across pairs that are not separated by a
    /// segment break.
    pub bonds: Vec<CovalentBond>,
    /// PDB chain identifier byte for this entity (derived from
    /// `label_asym_id`).
    pub pdb_chain_id: u8,
    /// Author-side chain identifier byte, when distinct from
    /// `pdb_chain_id`. `None` means "same as `pdb_chain_id`."
    pub auth_asym_id: Option<u8>,
}

/// Maximum C->N distance (A) for a peptide bond. Pairs exceeding this
/// are treated as segment breaks.
const MAX_PEPTIDE_BOND_DIST: f32 = 2.0;

impl ProteinEntity {
    /// Construct a protein entity.
    ///
    /// Enforces the canonical atom ordering invariant: for every kept
    /// residue, atoms are laid out as `N, CA, C, O, sidechain heavy...,
    /// hydrogens...`. Residues missing any of N/CA/C/O (with OXT as
    /// C-terminal oxygen fallback) are dropped from the `residues` vec
    /// with a `log::warn!` entry; their atoms remain in `atoms` but are
    /// unreferenced.
    ///
    /// Also computes segment breaks from C(i)->N(i+1) distance and
    /// populates `bonds` from the `AminoAcid::bonds()` tables plus
    /// universal backbone + inter-residue peptide bonds.
    ///
    /// Construction is pure: the atom set is assembled exactly as given
    /// and no missing-atom completion runs. A residue still missing any
    /// backbone atom is dropped by canonicalize; sidechain-incomplete
    /// residues are kept as-is with nothing fabricated. To fabricate
    /// missing atoms at construction (and rescue backbone-incomplete
    /// residues before the drop), use [`Self::new_normalized`]; to
    /// complete an already-built entity's surviving residues, use
    /// [`Self::normalize`].
    #[must_use]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "constructor owns the inputs; canonicalize borrows before \
                  reassigning new owned vecs"
    )]
    pub fn new(
        id: EntityId,
        atoms: Vec<Atom>,
        residues: Vec<Residue>,
        pdb_chain_id: u8,
        auth_asym_id: Option<u8>,
    ) -> Self {
        build_protein(
            id,
            atoms,
            residues,
            pdb_chain_id,
            auth_asym_id,
            CompletionMode::None,
            false,
        )
    }

    /// Construct a protein entity, fabricating missing atoms at
    /// construction in the given completion `mode`.
    ///
    /// Same canonicalize / segment-break / bond-graph construction as
    /// [`Self::new`], but the rigid-fit completion pass runs first (in
    /// `mode`), before the canonicalize reorder. Because completion runs
    /// before the drop, a residue that is backbone-incomplete in the
    /// parse but anchorable can have its backbone fabricated and survive,
    /// whereas the pure [`Self::new`] would drop it. The raw-ingest path
    /// (file parses) uses this with [`CompletionMode::HeavyOnly`].
    #[must_use]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "constructor owns the inputs; canonicalize borrows before \
                  reassigning new owned vecs"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors new's argument set plus the completion mode"
    )]
    pub fn new_normalized(
        id: EntityId,
        atoms: Vec<Atom>,
        residues: Vec<Residue>,
        pdb_chain_id: u8,
        auth_asym_id: Option<u8>,
        completion: CompletionMode,
    ) -> Self {
        build_protein(
            id,
            atoms,
            residues,
            pdb_chain_id,
            auth_asym_id,
            completion,
            false,
        )
    }

    /// Construct a protein entity assuming a single continuous chain
    /// with no segment breaks; bypasses the C(i)->N(i+1) distance-based
    /// segment-break detection used by [`Self::new`].
    ///
    /// Useful for synthetic backbones where positions don't yet reflect
    /// real peptide-bond geometry (e.g. ML diffusion intermediates):
    /// noisy positions would otherwise produce a segment break at every
    /// residue and the chain would render as disconnected fragments.
    /// Caller is responsible for knowing that the residues form one
    /// chain.
    #[must_use]
    pub fn new_continuous(
        id: EntityId,
        atoms: Vec<Atom>,
        residues: Vec<Residue>,
        pdb_chain_id: u8,
        auth_asym_id: Option<u8>,
    ) -> Self {
        build_protein(
            id,
            atoms,
            residues,
            pdb_chain_id,
            auth_asym_id,
            CompletionMode::None,
            true,
        )
    }

    /// Rebuild this entity as a single continuous chain, dropping the
    /// distance-based segment breaks [`Self::new`] computed. Identity,
    /// atoms, and residues are preserved.
    ///
    /// For synthetic or noisy backbones (ML diffusion intermediates):
    /// `new`'s C(i)->N(i+1) distance check fragments such a chain into a
    /// segment per residue, so it renders as disconnected pieces. This
    /// reconnects it. Caller asserts the residues form one chain. The
    /// atoms are already complete, so no completion pass is run.
    #[must_use]
    pub fn to_continuous(&self) -> Self {
        build_protein(
            self.id,
            self.atoms.clone(),
            self.residues.clone(),
            self.pdb_chain_id,
            self.auth_asym_id,
            CompletionMode::None,
            true,
        )
    }

    /// Rebuild this entity with heavy-atom completion, fabricating the
    /// missing heavy sidechain/base atoms a sparse parse omits. Identity,
    /// segment-break detection, and existing atoms are preserved;
    /// completion matches every present atom against its template by name
    /// and places only the absent in-scope heavy atoms (idempotent: a
    /// second projection adds nothing).
    ///
    /// This re-runs completion on the entity's already-surviving
    /// residues; it cannot resurrect a residue that was dropped at
    /// construction for missing backbone atoms (those atoms are no longer
    /// referenced by any residue). To rescue backbone-incomplete residues,
    /// fabricate at construction with [`Self::new_normalized`].
    ///
    /// Atom indices shift because fabricated atoms are appended per
    /// residue and the atom set is re-canonicalized, so `AtomId`s are not
    /// stable across this projection.
    #[must_use]
    pub fn normalize(&self) -> Self {
        self.completed(CompletionMode::HeavyOnly)
    }

    /// Rebuild this entity with all-atom completion, additionally
    /// fabricating the template hydrogens [`Self::normalize`] omits.
    /// Same identity / index-shift / survivors-only caveats as
    /// [`Self::normalize`].
    #[must_use]
    pub fn to_all_atom(&self) -> Self {
        self.completed(CompletionMode::AllAtom)
    }

    /// Re-extract this entity's atoms and residues and rebuild them with
    /// the given completion `mode`. Shared core of [`Self::normalize`] and
    /// [`Self::to_all_atom`].
    fn completed(&self, mode: CompletionMode) -> Self {
        build_protein(
            self.id,
            self.atoms.clone(),
            self.residues.clone(),
            self.pdb_chain_id,
            self.auth_asym_id,
            mode,
            false,
        )
    }

    /// Iterate covalent bonds whose endpoints lie in any residue's
    /// canonical backbone region (indices `atom_range.start..+4`).
    ///
    /// Intra-residue N-CA, CA-C, C=O and inter-residue peptide C-N all
    /// qualify because their endpoints occupy positions 0..4 of each
    /// residue after canonical ordering. OXT (C-terminal oxygen) is
    /// stored in the sidechain region and is therefore not considered
    /// a backbone atom by this filter.
    pub fn backbone_bonds(&self) -> impl Iterator<Item = &CovalentBond> + '_ {
        self.bonds.iter().filter(move |b| {
            self.is_backbone_atom(b.a) && self.is_backbone_atom(b.b)
        })
    }

    /// Iterate covalent bonds with at least one heavy-atom endpoint in
    /// the sidechain region (`residue.atom_range.start + 4..`).
    ///
    /// Hydrogen endpoints are excluded (they must be non-hydrogen to
    /// qualify). The CA-CB anchor bond qualifies because CB is a
    /// sidechain heavy atom.
    pub fn sidechain_bonds(&self) -> impl Iterator<Item = &CovalentBond> + '_ {
        self.bonds.iter().filter(move |b| {
            self.is_sidechain_heavy_atom(b.a)
                || self.is_sidechain_heavy_atom(b.b)
        })
    }

    fn residue_local_offset(&self, atom_id: AtomId) -> Option<usize> {
        if atom_id.entity != self.id {
            return None;
        }
        let idx = atom_id.index as usize;
        for r in &self.residues {
            if r.atom_range.contains(&idx) {
                return Some(idx - r.atom_range.start);
            }
        }
        None
    }

    fn is_backbone_atom(&self, atom_id: AtomId) -> bool {
        self.residue_local_offset(atom_id)
            .is_some_and(|off| off < 4)
    }

    fn is_sidechain_heavy_atom(&self, atom_id: AtomId) -> bool {
        let Some(off) = self.residue_local_offset(atom_id) else {
            return false;
        };
        if off < 4 {
            return false;
        }
        let atom = &self.atoms[atom_id.index as usize];
        atom.element != Element::H
    }

    /// Derive backbone (N, CA, C, O) for all residues.
    ///
    /// Returns one `ResidueBackbone` per residue that has all four
    /// backbone atoms. Residues missing any backbone atom are skipped.
    #[must_use]
    pub fn to_backbone(&self) -> Vec<ResidueBackbone> {
        self.residues
            .iter()
            .filter_map(|r| extract_backbone_from_residue(&self.atoms, r))
            .collect()
    }

    /// N/CA/C interleaved positions per segment (for spline renderer).
    ///
    /// Each inner `Vec<Vec3>` is one continuous segment with atoms
    /// laid out as `[N0, CA0, C0, N1, CA1, C1, ...]`.
    #[must_use]
    pub fn to_interleaved_segments(&self) -> Vec<Vec<Vec3>> {
        let n_segments = self.segment_count();
        (0..n_segments)
            .map(|seg_idx| {
                let range = self.segment_range(seg_idx);
                let mut positions = Vec::with_capacity(range.len() * 3);
                for r in &self.residues[range] {
                    if let Some(bb) =
                        extract_backbone_from_residue(&self.atoms, r)
                    {
                        positions.push(bb.n);
                        positions.push(bb.ca);
                        positions.push(bb.c);
                    }
                }
                positions
            })
            .collect()
    }
}

impl Entity for ProteinEntity {
    fn id(&self) -> EntityId {
        self.id
    }
    fn molecule_type(&self) -> MoleculeType {
        MoleculeType::Protein
    }
    fn atoms(&self) -> &[Atom] {
        &self.atoms
    }
}

impl Polymer for ProteinEntity {
    fn residues(&self) -> &[Residue] {
        &self.residues
    }
    fn segment_breaks(&self) -> &[usize] {
        &self.segment_breaks
    }
}

// Internal helpers

/// Shared protein-entity construction: complete (in `mode`), canonicalize,
/// derive segment breaks, and build the bond graph. `continuous` selects
/// whether segment breaks are distance-detected from C(i)->N(i+1) geometry
/// (`false`) or forced empty for a single continuous chain (`true`).
#[allow(
    clippy::needless_pass_by_value,
    reason = "owns the inputs; canonicalize borrows before reassigning new \
              owned vecs"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the two public constructors' argument set plus the \
              completion mode and the continuous-chain selector"
)]
fn build_protein(
    id: EntityId,
    atoms: Vec<Atom>,
    residues: Vec<Residue>,
    pdb_chain_id: u8,
    auth_asym_id: Option<u8>,
    mode: CompletionMode,
    continuous: bool,
) -> ProteinEntity {
    let (atoms, residues) = complete_protein_residues(&atoms, &residues, mode);
    let (atoms, residues) = canonicalize_protein_residues(
        &atoms,
        &residues,
        pdb_chain_id,
        keep_hydrogens(mode),
    );
    let segment_breaks = if continuous {
        Vec::new()
    } else {
        compute_segment_breaks(&residues, &atoms)
    };
    let bonds = build_protein_bonds(id, &atoms, &residues, &segment_breaks);
    ProteinEntity {
        id,
        atoms,
        residues,
        segment_breaks,
        bonds,
        pdb_chain_id,
        auth_asym_id,
    }
}

/// Reorganize atoms into canonical per-residue order and drop residues
/// missing backbone atoms.
///
/// For every residue whose atom slice contains N, CA, C, and O (OXT
/// accepted as C-terminal oxygen fallback), emit the four backbone
/// atoms followed by sidechain heavy atoms (in input order, plus OXT
/// if O and OXT both existed) then hydrogens (in input order).
/// Residues missing any of the four backbone atoms are dropped from
/// the returned `Vec<Residue>`; their atoms are appended to the output
/// `Vec<Atom>` in input order and are left unreferenced.
///
/// When `keep_hydrogens` is false (the heavy-only file-ingest mode), the
/// parsed hydrogen bucket is dropped from every kept residue, so a
/// protonated input yields a heavy-only entity. Dropped residues still
/// carry all their atoms through unreferenced regardless.
struct ProteinResiduePartition {
    n: Option<usize>,
    ca: Option<usize>,
    c: Option<usize>,
    o: Option<usize>,
    oxt: Option<usize>,
    sidechain_heavy: Vec<usize>,
    hydrogens: Vec<usize>,
}

fn partition_protein_residue(
    atoms: &[Atom],
    range: std::ops::Range<usize>,
) -> ProteinResiduePartition {
    let mut out = ProteinResiduePartition {
        n: None,
        ca: None,
        c: None,
        o: None,
        oxt: None,
        sidechain_heavy: Vec::new(),
        hydrogens: Vec::new(),
    };
    for idx in range {
        let trimmed = trimmed_atom_name(&atoms[idx].name);
        let is_h =
            atoms[idx].element == Element::H || trimmed.first() == Some(&b'H');
        match trimmed {
            b"N" if out.n.is_none() => out.n = Some(idx),
            b"CA" if out.ca.is_none() => out.ca = Some(idx),
            b"C" if out.c.is_none() => out.c = Some(idx),
            b"O" if out.o.is_none() => out.o = Some(idx),
            b"OXT" if out.oxt.is_none() => out.oxt = Some(idx),
            _ if is_h => out.hydrogens.push(idx),
            _ => out.sidechain_heavy.push(idx),
        }
    }
    out
}

fn canonicalize_protein_residues(
    atoms: &[Atom],
    residues: &[Residue],
    pdb_chain_id: u8,
    keep_hydrogens: bool,
) -> (Vec<Atom>, Vec<Residue>) {
    let mut new_atoms: Vec<Atom> = Vec::with_capacity(atoms.len());
    let mut new_residues: Vec<Residue> = Vec::new();

    for residue in residues {
        let range = residue.atom_range.clone();
        let start = new_atoms.len();
        let mut p = partition_protein_residue(atoms, range.clone());
        let o_slot = p.o.or(p.oxt);

        if let (Some(n), Some(ca), Some(c), Some(o)) = (p.n, p.ca, p.c, o_slot)
        {
            new_atoms.push(atoms[n].clone());
            new_atoms.push(atoms[ca].clone());
            new_atoms.push(atoms[c].clone());
            new_atoms.push(atoms[o].clone());
            if let (Some(_), Some(xt)) = (p.o, p.oxt) {
                p.sidechain_heavy.push(xt);
            }
            for idx in p.sidechain_heavy {
                new_atoms.push(atoms[idx].clone());
            }
            if keep_hydrogens {
                for idx in p.hydrogens {
                    new_atoms.push(atoms[idx].clone());
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
        } else {
            let res_name = std::str::from_utf8(&residue.name).unwrap_or("???");
            log::warn!(
                "ProteinEntity chain '{}': dropping residue {} (name {}); \
                 missing backbone atoms (need N, CA, C, and O or OXT)",
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

/// Emit intra-residue bonds (backbone + sidechain/anchor) for one
/// protein residue into `bonds`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "atom indices are bounded by protein chain size (< u32::MAX)"
)]
fn emit_protein_residue_bonds(
    entity_id: EntityId,
    atoms: &[Atom],
    r: &Residue,
    bonds: &mut Vec<CovalentBond>,
) {
    use std::collections::HashMap;

    let start = r.atom_range.start;
    let aid = |offset: usize| AtomId {
        entity: entity_id,
        index: (start + offset) as u32,
    };
    bonds.push(CovalentBond {
        a: aid(0),
        b: aid(1),
        order: BondOrder::Single,
    });
    bonds.push(CovalentBond {
        a: aid(1),
        b: aid(2),
        order: BondOrder::Single,
    });
    bonds.push(CovalentBond {
        a: aid(2),
        b: aid(3),
        order: BondOrder::Double,
    });

    let Some(aa) = AminoAcid::from_code(r.name) else {
        return;
    };
    let mut name_to_idx: HashMap<AtomName, usize> = HashMap::new();
    for idx in r.atom_range.clone() {
        let key = AtomName::from_bytes(trimmed_atom_name(&atoms[idx].name));
        let _ = name_to_idx.insert(key, idx);
    }
    for (name_a, name_b) in aa.bonds() {
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
}

/// Populate a protein entity's covalent bond graph.
///
/// Emits: universal backbone bonds (N-CA, CA-C, C=O) per residue,
/// sidechain/anchor bonds from [`AminoAcid::bonds()`] matched by
/// [`AtomName`] within the residue, and inter-residue peptide bonds
/// `C(i)-N(i+1)` between consecutive kept residues that are not
/// separated by a segment break.
#[allow(
    clippy::cast_possible_truncation,
    reason = "atom indices are bounded by protein chain size (< u32::MAX)"
)]
fn build_protein_bonds(
    entity_id: EntityId,
    atoms: &[Atom],
    residues: &[Residue],
    segment_breaks: &[usize],
) -> Vec<CovalentBond> {
    use std::collections::HashSet;

    let mut bonds: Vec<CovalentBond> = Vec::new();
    for r in residues {
        emit_protein_residue_bonds(entity_id, atoms, r, &mut bonds);
    }

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
                index: (prev.atom_range.start + 2) as u32,
            },
            b: AtomId {
                entity: entity_id,
                index: curr.atom_range.start as u32,
            },
            order: BondOrder::Single,
        });
    }

    bonds
}

/// Trim trailing space and NUL padding from a 4-byte atom name.
pub(crate) fn trimmed_atom_name(name: &[u8; 4]) -> &[u8] {
    let mut end = 4;
    while end > 0 && (name[end - 1] == b' ' || name[end - 1] == 0) {
        end -= 1;
    }
    let mut start = 0;
    while start < end && (name[start] == b' ' || name[start] == 0) {
        start += 1;
    }
    &name[start..end]
}

/// Compute segment break indices from C(i)->N(i+1) distances.
///
/// Relies on canonical atom ordering: kept residues have `C` at local
/// offset 2 and `N` at offset 0.
fn compute_segment_breaks(residues: &[Residue], atoms: &[Atom]) -> Vec<usize> {
    let mut breaks = Vec::new();
    for i in 1..residues.len() {
        let prev_c = atoms[residues[i - 1].atom_range.start + 2].position;
        let curr_n = atoms[residues[i].atom_range.start].position;
        if (prev_c - curr_n).length() > MAX_PEPTIDE_BOND_DIST {
            breaks.push(i);
        }
    }
    breaks
}

/// Find a backbone atom by trimmed string name.
fn find_atom_by_name(
    atoms: &[Atom],
    residue: &Residue,
    name: &str,
) -> Option<Vec3> {
    for idx in residue.atom_range.clone() {
        let atom_name =
            std::str::from_utf8(&atoms[idx].name).unwrap_or("").trim();
        if atom_name == name {
            let a = &atoms[idx];
            return Some(a.position);
        }
    }
    None
}

/// Extract backbone atom positions (N, CA, C, O) from a single residue.
fn extract_backbone_from_residue(
    atoms: &[Atom],
    residue: &Residue,
) -> Option<ResidueBackbone> {
    Some(ResidueBackbone {
        n: find_atom_by_name(atoms, residue, "N")?,
        ca: find_atom_by_name(atoms, residue, "CA")?,
        c: find_atom_by_name(atoms, residue, "C")?,
        o: find_atom_by_name(atoms, residue, "O")
            .or_else(|| find_atom_by_name(atoms, residue, "OXT"))?,
    })
}

#[cfg(test)]
#[path = "protein_tests.rs"]
mod tests;
