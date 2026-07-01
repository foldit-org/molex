// foldit:allow-long-file: ProteinEntity plus its build / complete / bond
// derivation.
//! Protein types and entity.
//!
//! Per-residue types (`ResidueBackbone`) and the `ProteinEntity` chain
//! instance with segment break metadata.

use glam::{Mat3, Quat, Vec3};
use thiserror::Error;

use super::atom::{pad_atom_name, Atom, AtomColumns};
use super::complete::{complete_protein_residues, keep_hydrogens, Completion};
use super::id::EntityId;
use super::polymer::residue_name_to_idx;
use super::traits::{Entity, Polymer};
use super::{MoleculeType, Residue};
use crate::analysis::grid::SpatialGrid;
use crate::analysis::{BondOrder, SSType};
use crate::atom_id::AtomId;
use crate::bond::CovalentBond;
use crate::chemistry::amino_acids::AminoAcid;
use crate::chemistry::atom_name::AtomName;
use crate::chemistry::rotamer::{distal_atoms, modal_rotamers};
use crate::element::Element;
use crate::ops::kabsch_alignment;

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
    /// All atoms in this protein chain, stored as parallel columns.
    ///
    /// After [`ProteinEntity::new`], atoms belonging to a kept residue are
    /// laid out in canonical order: `N, CA, C, O, sidechain heavy...,
    /// hydrogens...`. Atoms of dropped residues remain in `columns` but are
    /// not referenced by any `Residue::atom_range`.
    pub columns: AtomColumns,
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
    /// PDB chain identifier for this entity (the mmCIF `label_asym_id`,
    /// an arbitrary string).
    pub pdb_chain_id: String,
}

/// Maximum C->N distance (A) for a peptide bond. Pairs exceeding this
/// are treated as segment breaks.
const MAX_PEPTIDE_BOND_DIST: f32 = 2.0;

/// Failure modes of [`ProteinEntity::mutate_residue`].
///
/// A clash is never a failure; the only error is an out-of-range residue
/// index. (The target identity is a closed [`AminoAcid`], and every
/// residue surviving construction carries a complete backbone, so there
/// is no separate "invalid amino acid" failure to report.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MutateResidueError {
    /// `residue_index` is past the end of the entity's residue list.
    #[error("residue index {0} is out of range")]
    ResidueIndexOutOfRange(usize),
}

/// Fraction of the summed van der Waals radii below which two atoms count
/// as clashing. Picked permissively (a hard-overlap threshold) so that
/// near-contacts of a placed sidechain do not veto otherwise sound
/// rotamers; the placer only needs a non-clashing refinable seat.
const CLASH_FACTOR: f32 = 0.75;

/// Spatial-grid cell size (A) for the clash query. Sized above the largest
/// pairwise clash threshold (`CLASH_FACTOR` times two large vdW radii) so
/// every potential partner falls inside the 27-cell stencil.
const CLASH_CELL: f32 = 4.0;

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
        pdb_chain_id: String,
    ) -> Self {
        build_protein(id, atoms, residues, pdb_chain_id, Completion::Raw, false)
    }

    /// Construct a protein entity, fabricating missing atoms at
    /// construction in the given completion `mode`.
    ///
    /// Same canonicalize / segment-break / bond-graph construction as
    /// [`Self::new`], but the rigid-fit completion pass runs first (to
    /// `completion`), before the canonicalize reorder. Because completion
    /// runs before the drop, a residue that is backbone-incomplete in the
    /// parse but anchorable can have its backbone fabricated and survive,
    /// whereas the pure [`Self::new`] would drop it. The file-ingest path
    /// uses this with [`Completion::Heavy`].
    #[must_use]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "constructor owns the inputs; canonicalize borrows before \
                  reassigning new owned vecs"
    )]
    pub fn new_normalized(
        id: EntityId,
        atoms: Vec<Atom>,
        residues: Vec<Residue>,
        pdb_chain_id: String,
        completion: Completion,
    ) -> Self {
        build_protein(id, atoms, residues, pdb_chain_id, completion, false)
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
        pdb_chain_id: String,
    ) -> Self {
        build_protein(id, atoms, residues, pdb_chain_id, Completion::Raw, true)
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
            self.columns.to_atoms(),
            self.residues.clone(),
            self.pdb_chain_id.clone(),
            Completion::Raw,
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
        build_protein(
            self.id,
            self.columns.to_atoms(),
            self.residues.clone(),
            self.pdb_chain_id.clone(),
            level,
            false,
        )
    }

    /// Derive backbone (N, CA, C, O) for all residues.
    ///
    /// Returns one `ResidueBackbone` per residue that has all four
    /// backbone atoms. Residues missing any backbone atom are skipped.
    #[must_use]
    pub fn to_backbone(&self) -> Vec<ResidueBackbone> {
        self.residues
            .iter()
            .filter_map(|r| extract_backbone_from_residue(self, r))
            .collect()
    }

    /// Spread a per-backbone-residue secondary-structure vector across all
    /// residues, in residue order.
    ///
    /// `backbone_ss` holds one entry per backbone-complete residue, in the
    /// order [`Self::to_backbone`] yields them. This walks every residue
    /// through the same backbone-completeness predicate `to_backbone` uses,
    /// consuming one `backbone_ss` entry per complete residue and emitting
    /// [`SSType::Coil`] for residues `to_backbone` drops. The result has one
    /// entry per residue (`residues.len()`), so it indexes 1:1 against the
    /// residue list. Excess `backbone_ss` entries are ignored; if it runs
    /// short the remaining complete residues also fall back to `Coil`.
    #[must_use]
    pub(crate) fn ss_per_residue(&self, backbone_ss: &[SSType]) -> Vec<SSType> {
        let mut next = backbone_ss.iter();
        self.residues
            .iter()
            .map(|r| {
                if residue_has_backbone(self, r) {
                    next.next().copied().unwrap_or(SSType::Coil)
                } else {
                    SSType::Coil
                }
            })
            .collect()
    }

    /// Backbone (φ, ψ) torsion angles per residue, in degrees, in residue
    /// order, one entry per residue (`residues.len()`).
    ///
    /// - φ(i) = dihedral( C(i-1), N(i), CA(i), C(i) )
    /// - ψ(i) = dihedral( N(i), CA(i), C(i), N(i+1) )
    ///
    /// An angle is `None` when an input atom is unavailable: φ at the first
    /// residue (no preceding C), ψ at the last (no following N), either side of
    /// a segment break (no peptide bond across it), and for any residue (or its
    /// neighbour) lacking a complete N/CA/C backbone — the same completeness
    /// notion `extract_backbone_from_residue` enforces. Results are in
    /// (−180, 180].
    #[must_use]
    pub fn phi_psi(&self) -> Vec<(Option<f32>, Option<f32>)> {
        let n = self.residues.len();
        let bb: Vec<Option<ResidueBackbone>> = self
            .residues
            .iter()
            .map(|r| extract_backbone_from_residue(self, r))
            .collect();
        let breaks: std::collections::HashSet<usize> =
            self.segment_breaks.iter().copied().collect();

        (0..n)
            .map(|i| {
                let curr = bb[i].as_ref();
                let phi = (i > 0 && !breaks.contains(&i))
                    .then(|| (bb[i - 1].as_ref(), curr))
                    .and_then(|(prev, curr)| {
                        let prev = prev?;
                        let curr = curr?;
                        Some(dihedral_deg(prev.c, curr.n, curr.ca, curr.c))
                    });
                let psi = (i + 1 < n && !breaks.contains(&(i + 1)))
                    .then(|| (curr, bb[i + 1].as_ref()))
                    .and_then(|(curr, next)| {
                        let curr = curr?;
                        let next = next?;
                        Some(dihedral_deg(curr.n, curr.ca, curr.c, next.n))
                    });
                (phi, psi)
            })
            .collect()
    }

    /// Sidechain torsion (χ) angles per residue, in degrees in (-180, 180]
    /// (IUPAC sign), one inner `Vec` per residue in residue order.
    ///
    /// The inner vector holds χ1..χN for the residue type
    /// ([`AminoAcid::chi_atoms`]); a χ is `None` when any of its four atoms is
    /// absent from the residue. Non-standard residue names (unparseable by
    /// [`AminoAcid::from_code`]) and residues with no rotatable sidechain
    /// torsion (Ala, Gly) yield an empty inner vector. Uses the same torsion
    /// kernel as [`Self::phi_psi`].
    #[must_use]
    pub fn chi_angles(&self) -> Vec<Vec<Option<f32>>> {
        let positions = self.positions();
        self.residues
            .iter()
            .map(|r| {
                let Some(aa) = AminoAcid::from_code(r.name) else {
                    return Vec::new();
                };
                let name_to_idx =
                    residue_name_to_idx(r.atom_range.clone(), |idx| {
                        self.columns.name[idx]
                    });
                aa.chi_atoms()
                    .iter()
                    .map(|quad| {
                        let p: Option<Vec<Vec3>> = quad
                            .iter()
                            .map(|n| name_to_idx.get(n).map(|&i| positions[i]))
                            .collect();
                        p.map(|p| dihedral_deg(p[0], p[1], p[2], p[3]))
                    })
                    .collect()
            })
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
                    if let Some(bb) = extract_backbone_from_residue(self, r) {
                        positions.push(bb.n);
                        positions.push(bb.ca);
                        positions.push(bb.c);
                    }
                }
                positions
            })
            .collect()
    }

    /// Replace residue `residue_index`'s identity with `aa`, seating a
    /// fresh non-clashing heavy-atom sidechain on the existing backbone.
    ///
    /// The backbone (N, CA, C, O, and any terminal OXT) is preserved
    /// verbatim. The new heavy sidechain is placed by rigid-fitting the
    /// amino-acid template onto that backbone (the same Kabsch anchor as
    /// missing-atom completion), then driving each sidechain torsion to
    /// the least-clashing modal rotamer. Clash is measured only against
    /// the rest of this entity's atoms (intra-entity). The result is a
    /// refinable starting state, not a scored optimum: a clash never
    /// fails the call, and when no rotamer is clash-free the
    /// least-clashing one is kept. Ala/Gly (no rotatable torsion) and Pro
    /// (fixed ring) take the template seat directly.
    ///
    /// The returned entity is freshly built, so its `AtomId`s are not
    /// stable against this one (the atom set is re-canonicalized). Any
    /// sidechain hydrogens on the mutated residue are dropped; the new
    /// sidechain is heavy-atom only.
    ///
    /// # Errors
    /// Returns [`MutateResidueError::ResidueIndexOutOfRange`] when
    /// `residue_index` is past the end of the residue list.
    pub fn mutate_residue(
        &self,
        residue_index: usize,
        aa: AminoAcid,
    ) -> Result<Self, MutateResidueError> {
        let residue = self
            .residues
            .get(residue_index)
            .ok_or(MutateResidueError::ResidueIndexOutOfRange(residue_index))?;
        let sidechain = self.place_sidechain(aa, residue.atom_range.clone());
        Ok(self.rebuild_with_mutation(residue_index, aa, &sidechain))
    }

    /// Place `aa`'s heavy sidechain on the backbone of the residue spanning
    /// `range`, returning each new atom's `(name, element, position)`.
    ///
    /// Seats the template on the N/CA/C frame, then walks `aa`'s modal
    /// rotamers most-populated first and keeps the first clash-free one (or
    /// the least-clashing if none clears). Residues with no rotamer search
    /// (Ala/Gly/Pro) return the template seat directly.
    fn place_sidechain(
        &self,
        aa: AminoAcid,
        range: std::ops::Range<usize>,
    ) -> Vec<(AtomName, Element, Vec3)> {
        let name_to_idx =
            residue_name_to_idx(range.clone(), |i| self.columns.name[i]);
        let real = |nm: &[u8]| {
            name_to_idx
                .get(&AtomName::from_bytes(nm))
                .map(|&i| self.columns.position[i])
        };
        let template = aa.template();
        let ideal = |nm: &[u8]| {
            template.atom(AtomName::from_bytes(nm)).map(|a| a.ideal)
        };

        let (rot, trans) = backbone_fit(
            [real(b"N"), real(b"CA"), real(b"C")],
            [ideal(b"N"), ideal(b"CA"), ideal(b"C")],
        );

        // Scratch buffer: the two fixed backbone references χ quads need
        // (N, CA) followed by the heavy sidechain atoms at their template
        // seat. set_chi rotates only the sidechain entries.
        let mut names =
            vec![AtomName::from_bytes(b"N"), AtomName::from_bytes(b"CA")];
        let mut positions = vec![
            real(b"N").unwrap_or(Vec3::ZERO),
            real(b"CA").unwrap_or(Vec3::ZERO),
        ];
        let mut sidechain: Vec<(AtomName, Element)> = Vec::new();
        for t in template.atoms {
            if t.element == Element::H
                || t.is_leaving
                || t.name.is_protein_backbone()
            {
                continue;
            }
            names.push(t.name);
            positions.push(rot * t.ideal + trans);
            sidechain.push((t.name, t.element));
        }

        let scratch = SidechainScratch {
            names: &names,
            positions: &positions,
            atoms: &sidechain,
        };
        let chosen = self.best_rotamer(aa, &range, &scratch);
        sidechain
            .iter()
            .enumerate()
            .map(|(k, &(name, element))| (name, element, chosen[k]))
            .collect()
    }

    /// Sidechain positions of the least-clashing modal rotamer. Returns the
    /// chosen sidechain positions (parallel to `scratch.atoms`); with no
    /// rotamer search the template seat is returned unchanged.
    fn best_rotamer(
        &self,
        aa: AminoAcid,
        range: &std::ops::Range<usize>,
        scratch: &SidechainScratch,
    ) -> Vec<Vec3> {
        let seat = scratch.positions[2..].to_vec();
        let rotamers = modal_rotamers(aa);
        if rotamers.is_empty() {
            return seat;
        }
        let grid = SpatialGrid::build(&self.columns.position, CLASH_CELL);
        let mut best: Option<(f32, Vec<Vec3>)> = None;
        for rotamer in rotamers {
            let mut buf = scratch.positions.to_vec();
            for (chi_index, &target) in rotamer.iter().enumerate() {
                set_chi(aa, chi_index, target, scratch.names, &mut buf);
            }
            let placed = &buf[2..];
            let score = self.clash_score(&grid, range, scratch.atoms, placed);
            // Only finite scores compete; a non-finite candidate (infinite
            // clash from a non-finite seat) is never selectable, so an
            // all-degenerate search falls back to the template seat below.
            let keep = score.is_finite()
                && match &best {
                    Some((b, _)) => score < *b,
                    None => true,
                };
            if keep {
                best = Some((score, placed.to_vec()));
            }
            if score <= 0.0 {
                break;
            }
        }
        best.map_or(seat, |(_, placed)| placed)
    }

    /// Summed overlap depth of `placed` sidechain atoms against the rest of
    /// the entity (atoms inside `range`, the mutated residue, are skipped).
    /// Zero means clash-free; a larger value means deeper/more overlaps.
    fn clash_score(
        &self,
        grid: &SpatialGrid,
        range: &std::ops::Range<usize>,
        sidechain: &[(AtomName, Element)],
        placed: &[Vec3],
    ) -> f32 {
        let mut score = 0.0f32;
        for (k, &pos) in placed.iter().enumerate() {
            // A non-finite seat has no geometric meaning; treat it as maximally
            // clashing so a garbage rotamer can never read back as clash-free.
            if !pos.is_finite() {
                return f32::INFINITY;
            }
            let element = sidechain[k].1;
            grid.for_each_candidate(pos, |idx| {
                if range.contains(&idx) {
                    return;
                }
                let threshold = CLASH_FACTOR
                    * (element.vdw_radius()
                        + self.columns.element[idx].vdw_radius());
                let dist = pos.distance(self.columns.position[idx]);
                if dist < threshold {
                    score += threshold - dist;
                }
            });
        }
        score
    }

    /// Build a fresh entity with residue `residue_index` re-identified as
    /// `aa` and carrying `sidechain` as its heavy sidechain. Backbone heavy
    /// atoms are preserved; all other residues pass through unchanged.
    fn rebuild_with_mutation(
        &self,
        residue_index: usize,
        aa: AminoAcid,
        sidechain: &[(AtomName, Element, Vec3)],
    ) -> Self {
        let mut new_atoms: Vec<Atom> =
            Vec::with_capacity(self.columns.len() + sidechain.len());
        let mut new_residues: Vec<Residue> =
            Vec::with_capacity(self.residues.len());
        for (ri, res) in self.residues.iter().enumerate() {
            let start = new_atoms.len();
            let name = if ri == residue_index {
                self.push_mutated_residue(
                    res.atom_range.clone(),
                    sidechain,
                    &mut new_atoms,
                );
                aa.code()
            } else {
                for i in res.atom_range.clone() {
                    new_atoms.push(self.columns.gather(i));
                }
                res.name
            };
            new_residues.push(Residue {
                name,
                label_seq_id: res.label_seq_id,
                auth_seq_id: res.auth_seq_id,
                auth_comp_id: res.auth_comp_id,
                ins_code: res.ins_code,
                atom_range: start..new_atoms.len(),
                variants: res.variants.clone(),
            });
        }
        build_protein(
            self.id,
            new_atoms,
            new_residues,
            self.pdb_chain_id.clone(),
            Completion::Raw,
            false,
        )
    }

    /// Emit the mutated residue's atoms: existing backbone heavy atoms
    /// (N, CA, C, O, and OXT when present), then the new heavy sidechain.
    fn push_mutated_residue(
        &self,
        range: std::ops::Range<usize>,
        sidechain: &[(AtomName, Element, Vec3)],
        out: &mut Vec<Atom>,
    ) {
        let name_to_idx = residue_name_to_idx(range, |i| self.columns.name[i]);
        for nm in [b"N".as_slice(), b"CA", b"C", b"O", b"OXT"] {
            if let Some(&i) = name_to_idx.get(&AtomName::from_bytes(nm)) {
                out.push(self.columns.gather(i));
            }
        }
        for &(name, element, position) in sidechain {
            out.push(Atom {
                position,
                occupancy: 1.0,
                b_factor: 0.0,
                element,
                name: pad_atom_name(name.as_str()),
                formal_charge: 0,
                observed: false,
            });
        }
    }
}

impl Entity for ProteinEntity {
    fn id(&self) -> EntityId {
        self.id
    }
    fn molecule_type(&self) -> MoleculeType {
        MoleculeType::Protein
    }
    fn columns(&self) -> &AtomColumns {
        &self.columns
    }
    fn bonds(&self) -> &[CovalentBond] {
        &self.bonds
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

/// Shared protein-entity construction: complete (to `level`), canonicalize,
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
    reason = "mirrors the public constructors' argument set plus the \
              completion level and the continuous-chain selector"
)]
fn build_protein(
    id: EntityId,
    atoms: Vec<Atom>,
    residues: Vec<Residue>,
    pdb_chain_id: String,
    level: Completion,
    continuous: bool,
) -> ProteinEntity {
    let (atoms, residues) = complete_protein_residues(&atoms, &residues, level);
    let (atoms, residues) = canonicalize_protein_residues(
        &atoms,
        &residues,
        &pdb_chain_id,
        keep_hydrogens(level),
    );
    let segment_breaks = if continuous {
        Vec::new()
    } else {
        compute_segment_breaks(&residues, &atoms)
    };
    let bonds = build_protein_bonds(id, &atoms, &residues, &segment_breaks);
    ProteinEntity {
        id,
        columns: AtomColumns::from_atoms(atoms),
        residues,
        segment_breaks,
        bonds,
        pdb_chain_id,
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
    pdb_chain_id: &str,
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
    let name_to_idx =
        residue_name_to_idx(r.atom_range.clone(), |idx| atoms[idx].name);
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
/// [`AtomName`] within the residue,
/// and inter-residue peptide bonds
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
fn find_atom_by_name<E: Entity>(
    entity: &E,
    residue: &Residue,
    name: &str,
) -> Option<Vec3> {
    let cols = entity.columns();
    for idx in residue.atom_range.clone() {
        let atom_name =
            std::str::from_utf8(&cols.name[idx]).unwrap_or("").trim();
        if atom_name == name {
            return Some(cols.position[idx]);
        }
    }
    None
}

/// Extract backbone atom positions (N, CA, C, O) from a single residue.
///
/// Returns `None` for exactly the residues [`residue_has_backbone`] rejects:
/// the two share one notion of a complete backbone (N, CA, C, and O or OXT).
fn extract_backbone_from_residue<E: Entity>(
    entity: &E,
    residue: &Residue,
) -> Option<ResidueBackbone> {
    Some(ResidueBackbone {
        n: find_atom_by_name(entity, residue, "N")?,
        ca: find_atom_by_name(entity, residue, "CA")?,
        c: find_atom_by_name(entity, residue, "C")?,
        o: find_atom_by_name(entity, residue, "O")
            .or_else(|| find_atom_by_name(entity, residue, "OXT"))?,
    })
}

/// `true` when `residue` carries a complete DSSP backbone: N, CA, C, and O
/// (or its OXT terminal substitute). The single source for which residues
/// receive a secondary-structure assignment; [`extract_backbone_from_residue`]
/// returns `Some` for exactly these residues.
pub(crate) fn residue_has_backbone<E: Entity>(
    entity: &E,
    residue: &Residue,
) -> bool {
    extract_backbone_from_residue(entity, residue).is_some()
}

/// Signed dihedral angle about the `p1->p2` bond for the four points
/// `p0, p1, p2, p3`, in degrees in the range (−180, 180].
///
/// Uses the IUPAC-sign convention (`atan2` of the two bond-normal cross
/// products), so φ/ψ match the standard structural-biology definitions.
pub(crate) fn dihedral_deg(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> f32 {
    let b1 = p1 - p0;
    let b2 = p2 - p1;
    let b3 = p3 - p2;
    let n1 = b1.cross(b2);
    let n2 = b2.cross(b3);
    let m1 = b2.normalize().cross(n1);
    let x = n1.dot(n2);
    let y = m1.dot(n2);
    y.atan2(x).to_degrees()
}

/// Per-residue scratch buffer for rotamer placement: parallel `names` and
/// `positions` whose first two slots are the fixed N and CA references and
/// whose remaining slots are the heavy sidechain atoms (named, in order, by
/// `atoms`).
struct SidechainScratch<'a> {
    names: &'a [AtomName],
    positions: &'a [Vec3],
    atoms: &'a [(AtomName, Element)],
}

/// Rigid transform mapping template-frame ideals into the real backbone
/// frame, anchored on N/CA/C (`real`/`ideal` paired by position). Mirrors
/// the Kabsch fit atom completion uses. Falls back to a pure translation
/// aligning CA when an anchor is missing or the three are collinear (the
/// only cases the fit is under-determined).
fn backbone_fit(
    real: [Option<Vec3>; 3],
    ideal: [Option<Vec3>; 3],
) -> (Mat3, Vec3) {
    let mut reference = Vec::new();
    let mut target = Vec::new();
    for (r, t) in real.iter().zip(ideal.iter()) {
        if let (Some(r), Some(t)) = (r, t) {
            reference.push(*r);
            target.push(*t);
        }
    }
    if !backbone_is_collinear(real) {
        if let Some(fit) = kabsch_alignment(&reference, &target) {
            return fit;
        }
    }
    let trans = real[1].zip(ideal[1]).map_or(Vec3::ZERO, |(r, t)| r - t);
    (Mat3::IDENTITY, trans)
}

/// True when the real N/CA/C anchors are all present but (near-)collinear.
/// Collinear anchors cannot define a unique orientation, so Kabsch returns a
/// finite but under-determined rotation that seats the sidechain at a garbage
/// angle; the caller routes such triads to the translation-only fallback.
/// The test is on the normalized cross product of the two edge vectors from N
/// (the sine of the angle at vertex N), a vertex-invariant collinearity
/// measure: zero area means the three anchors are collinear at every vertex.
/// A real backbone gives a sine well away from zero, so a 1e-3 threshold only
/// trips on a triad within ~0.06 deg of a straight line.
fn backbone_is_collinear(real: [Option<Vec3>; 3]) -> bool {
    let (Some(n), Some(ca), Some(c)) = (real[0], real[1], real[2]) else {
        return false;
    };
    let e1 = ca - n;
    let e2 = c - n;
    let denom = e1.length() * e2.length();
    if denom < f32::EPSILON {
        return true;
    }
    e1.cross(e2).length() / denom < 1e-3
}

/// Rotate the atoms distal to sidechain torsion `chi_index` so the dihedral
/// reads `target_deg`. Operates on the scratch buffer `positions` (parallel
/// to `names`); the axis atoms and every proximal/backbone atom stay put.
/// A no-op when the residue lacks that torsion or any quad atom is absent.
pub(crate) fn set_chi(
    aa: AminoAcid,
    chi_index: usize,
    target_deg: f32,
    names: &[AtomName],
    positions: &mut [Vec3],
) {
    let Some(quad) = aa.chi_atoms().get(chi_index) else {
        return;
    };
    let idx_of = |nm: AtomName| names.iter().position(|m| *m == nm);
    let (Some(i0), Some(i1), Some(i2), Some(i3)) = (
        idx_of(quad[0]),
        idx_of(quad[1]),
        idx_of(quad[2]),
        idx_of(quad[3]),
    ) else {
        return;
    };
    let current = dihedral_deg(
        positions[i0],
        positions[i1],
        positions[i2],
        positions[i3],
    );
    let axis_vec = positions[i2] - positions[i1];
    if axis_vec.length_squared() < 1e-12 {
        return;
    }
    let rotation = Quat::from_axis_angle(
        axis_vec.normalize(),
        (target_deg - current).to_radians(),
    );
    let pivot = positions[i2];
    for nm in distal_atoms(aa, quad[1], quad[2]) {
        if let Some(j) = idx_of(nm) {
            positions[j] = pivot + rotation * (positions[j] - pivot);
        }
    }
}

#[cfg(test)]
#[path = "protein_tests.rs"]
mod tests;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod mutate_tests {
    use super::*;
    use crate::entity::molecule::id::EntityIdAllocator;

    fn atom(name: &str, element: Element, pos: Vec3) -> Atom {
        Atom {
            position: pos,
            occupancy: 1.0,
            b_factor: 0.0,
            element,
            name: pad_atom_name(name),
            formal_charge: 0,
            observed: true,
        }
    }

    fn res(name: &str, seq: i32, range: std::ops::Range<usize>) -> Residue {
        Residue {
            name: {
                let mut b = [b' '; 3];
                for (i, c) in name.bytes().take(3).enumerate() {
                    b[i] = c;
                }
                b
            },
            label_seq_id: seq,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: range,
            variants: Vec::new(),
        }
    }

    /// A glycine-only backbone (N, CA, C, O) at a non-collinear frame,
    /// ready to receive a mutated sidechain.
    fn backbone_atoms(origin: Vec3) -> Vec<Atom> {
        vec![
            atom("N", Element::N, origin + Vec3::new(0.0, 0.0, 0.0)),
            atom("CA", Element::C, origin + Vec3::new(1.458, 0.0, 0.0)),
            atom("C", Element::C, origin + Vec3::new(2.0, 1.42, 0.0)),
            atom("O", Element::O, origin + Vec3::new(1.5, 2.5, 0.0)),
        ]
    }

    fn single_gly(origin: Vec3) -> ProteinEntity {
        let mut alloc = EntityIdAllocator::new();
        ProteinEntity::new(
            alloc.allocate(),
            backbone_atoms(origin),
            vec![res("GLY", 1, 0..4)],
            "A".to_owned(),
        )
    }

    fn atom_pos(e: &ProteinEntity, res_idx: usize, name: &str) -> Option<Vec3> {
        let r = &e.residues[res_idx];
        for i in r.atom_range.clone() {
            if std::str::from_utf8(&e.columns.name[i]).unwrap_or("").trim()
                == name
            {
                return Some(e.columns.position[i]);
            }
        }
        None
    }

    #[test]
    fn set_chi_reaches_each_target() {
        // A bare N-CA-CB-CG quad; set_chi must drive the dihedral to the
        // requested value regardless of the starting angle.
        let names = [
            AtomName::from_bytes(b"N"),
            AtomName::from_bytes(b"CA"),
            AtomName::from_bytes(b"CB"),
            AtomName::from_bytes(b"CG"),
        ];
        for target in [60.0f32, -60.0, 170.0, -120.0] {
            let mut pos = vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.5, 0.0, 0.0),
                Vec3::new(2.0, 1.4, 0.0),
                Vec3::new(3.5, 1.4, 0.3),
            ];
            set_chi(AminoAcid::Leu, 0, target, &names, &mut pos);
            let measured = dihedral_deg(pos[0], pos[1], pos[2], pos[3]);
            assert!(
                (measured - target).abs() < 1e-2,
                "target {target}, measured {measured}"
            );
        }
    }

    #[test]
    fn mutation_lands_modal_chi_without_clash() {
        // An isolated residue has no clash pressure, so the placer keeps the
        // first (modal) rotamer; its χ1 must match the table within
        // tolerance.
        let gly = single_gly(Vec3::ZERO);
        let mutated = gly.mutate_residue(0, AminoAcid::Leu).unwrap();
        assert_eq!(&mutated.residues[0].name, b"LEU");
        let chi1 = mutated.chi_angles()[0][0].expect("Leu χ1 present");
        let modal = modal_rotamers(AminoAcid::Leu)[0][0];
        assert!(
            (chi1 - modal).abs() < 5.0,
            "χ1 {chi1} should be near modal {modal}"
        );
    }

    #[test]
    fn clash_pushes_off_the_modal_rotamer() {
        // Place the modal rotamer first in isolation to learn where its CG1
        // tip lands, then drop a blocking residue exactly there: the second
        // mutation must avoid the now-clashing modal and pick another χ1.
        let isolated = single_gly(Vec3::ZERO)
            .mutate_residue(0, AminoAcid::Val)
            .unwrap();
        let modal_chi1 = isolated.chi_angles()[0][0].unwrap();
        let tip = atom_pos(&isolated, 0, "CG1").expect("Val CG1");

        // Two residues: the target (res 0) plus a blocker (res 1) whose CA
        // sits on the modal CG1 tip.
        let mut atoms = backbone_atoms(Vec3::ZERO);
        atoms.extend([
            atom("N", Element::N, tip + Vec3::new(-1.458, 0.0, 0.0)),
            atom("CA", Element::C, tip),
            atom("C", Element::C, tip + Vec3::new(0.5, 1.42, 0.0)),
            atom("O", Element::O, tip + Vec3::new(0.5, 2.5, 0.0)),
        ]);
        let mut alloc = EntityIdAllocator::new();
        let blocked = ProteinEntity::new(
            alloc.allocate(),
            atoms,
            vec![res("GLY", 1, 0..4), res("GLY", 2, 4..8)],
            "A".to_owned(),
        );

        let mutated = blocked.mutate_residue(0, AminoAcid::Val).unwrap();
        let chi1 = mutated.chi_angles()[0][0].unwrap();
        assert!(
            (chi1 - modal_chi1).abs() > 30.0,
            "blocked χ1 {chi1} should differ from modal {modal_chi1}"
        );
    }

    #[test]
    fn collinear_backbone_seats_finite_sidechain() {
        // N/CA/C on a straight line cannot define a unique rotation. The
        // placer must route to the translation-only fallback rather than trust
        // an under-determined Kabsch rotation, leaving every seated atom
        // finite instead of NaN/Inf.
        let mut alloc = EntityIdAllocator::new();
        let atoms = vec![
            atom("N", Element::N, Vec3::new(0.0, 0.0, 0.0)),
            atom("CA", Element::C, Vec3::new(1.458, 0.0, 0.0)),
            atom("C", Element::C, Vec3::new(2.916, 0.0, 0.0)),
            atom("O", Element::O, Vec3::new(3.5, 1.0, 0.0)),
        ];
        let entity = ProteinEntity::new(
            alloc.allocate(),
            atoms,
            vec![res("GLY", 1, 0..4)],
            "A".to_owned(),
        );
        let mutated = entity.mutate_residue(0, AminoAcid::Leu).unwrap();
        assert_eq!(&mutated.residues[0].name, b"LEU");
        for i in mutated.residues[0].atom_range.clone() {
            assert!(
                mutated.columns.position[i].is_finite(),
                "atom {i} seated non-finite on a collinear backbone",
            );
        }
    }

    #[test]
    fn out_of_range_index_is_error() {
        let gly = single_gly(Vec3::ZERO);
        assert_eq!(
            gly.mutate_residue(7, AminoAcid::Ala).unwrap_err(),
            MutateResidueError::ResidueIndexOutOfRange(7)
        );
    }

    #[test]
    fn glycine_target_seats_backbone_only() {
        // Mutating to Gly leaves only the backbone (no rotatable sidechain).
        let ala = {
            let mut alloc = EntityIdAllocator::new();
            let mut atoms = backbone_atoms(Vec3::ZERO);
            atoms.push(atom("CB", Element::C, Vec3::new(2.0, -1.3, 0.5)));
            ProteinEntity::new(
                alloc.allocate(),
                atoms,
                vec![res("ALA", 1, 0..5)],
                "A".to_owned(),
            )
        };
        let mutated = ala.mutate_residue(0, AminoAcid::Gly).unwrap();
        assert_eq!(&mutated.residues[0].name, b"GLY");
        assert!(atom_pos(&mutated, 0, "CB").is_none(), "Gly has no CB");
    }
}
