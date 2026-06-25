//! Builder that turns parser-output atom rows into typed `MoleculeEntity`
//! values. Used by the PDB, mmCIF, and BinaryCIF adapters.

#![allow(
    dead_code,
    reason = "builder internals are exposed for adapters; some helpers are \
              still unwired"
)]

use compact_str::CompactString;
use glam::Vec3;
use rustc_hash::FxHashMap;

use super::atom::Atom;
use super::bulk::BulkEntity;
use super::complete::Completion;
use super::id::EntityIdAllocator;
use super::{MoleculeEntity, MoleculeType};
use crate::element::Element;

mod atom_name_alias;
mod classify;

use atom_name_alias::normalize_legacy_atom_name;
use classify::classify_chain;

const DEFAULT_WATER_RESNAME: [u8; 3] = *b"HOH";
const DEFAULT_SOLVENT_RESNAME: [u8; 3] = *b"GOL";

// Public API

/// Hint about an entity's classification, sourced from mmCIF
/// `_entity.type` joined with `_entity_poly.type`. The PDB ingest path
/// supplies `Unknown` for every atom; mmCIF/BCIF supply explicit hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::upper_case_acronyms,
    reason = "DNA / RNA mirror MoleculeType's variant names"
)]
pub(crate) enum ExpectedEntityType {
    /// `_entity_poly.type` = `polypeptide(L)` or `polypeptide(D)`.
    Protein,
    /// `_entity_poly.type` = `polydeoxyribonucleotide`.
    DNA,
    /// `_entity_poly.type` = `polyribonucleotide`.
    RNA,
    /// `_entity.type` = `non-polymer` or `branched`. Builder
    /// sub-classifies each residue via the 3-letter heuristic.
    NonPolymer,
    /// `_entity.type` = `water`.
    Water,
    /// Entity tables absent (PDB path) or unrecognized polymer subtype.
    /// Builder runs the full heuristic plus modified-residue merge.
    Unknown,
}

/// One parser-emitted atom row.
///
/// Chain / entity ids are [`CompactString`]: ids are short (≤ a few
/// chars) and massively repeated, so storing them inline keeps the
/// per-atom row off the heap while still supporting arbitrary-length
/// ids in the rare long case.
pub(crate) struct AtomRow {
    /// `label_asym_id` (mmCIF) or chain letter (PDB).
    pub label_asym_id: CompactString,
    /// `label_seq_id` (mmCIF) or `resSeq` (PDB). For non-polymer
    /// entities with `.` in the source, parsers pass `1`.
    pub label_seq_id: i32,
    /// `label_comp_id` (mmCIF) or `resName` (PDB), 3 chars, space-padded.
    pub label_comp_id: [u8; 3],
    /// `label_atom_id` (mmCIF) or atom name (PDB), 4 chars.
    pub label_atom_id: [u8; 4],
    /// `label_entity_id` (mmCIF). Joined against registered hints to
    /// pick the entity type. `None` on the PDB path.
    pub label_entity_id: Option<CompactString>,

    /// `auth_asym_id`. `None` defaults to `label_asym_id`.
    pub auth_asym_id: Option<CompactString>,
    /// `auth_seq_id`. `None` defaults to `label_seq_id`.
    pub auth_seq_id: Option<i32>,
    /// `auth_comp_id`. `None` defaults to `label_comp_id`.
    pub auth_comp_id: Option<[u8; 3]>,
    /// `auth_atom_id`. `None` defaults to `label_atom_id`.
    pub auth_atom_id: Option<[u8; 4]>,

    /// Alternate location indicator. `None` = blank (present in every
    /// conformer).
    pub alt_loc: Option<u8>,
    /// Insertion code. `None` = blank.
    pub ins_code: Option<u8>,
    /// Chemical element.
    pub element: Element,
    /// X coordinate in angstroms.
    pub x: f32,
    /// Y coordinate in angstroms.
    pub y: f32,
    /// Z coordinate in angstroms.
    pub z: f32,
    /// Crystallographic occupancy (0.0..=1.0).
    pub occupancy: f32,
    /// Temperature factor (B-factor) in square angstroms.
    pub b_factor: f32,
    /// Formal charge. 0 default.
    pub formal_charge: i8,
}

/// One atom's scalar cells, borrowing its chain/entity ids rather than
/// owning them.
///
/// This is the row-struct-agnostic argument the accumulator highway
/// (`ensure_chain` / `locate_or_create_residue` / `apply_altloc_dedup`)
/// actually consumes. [`push_atom`](EntityBuilder::push_atom) adapts an
/// owned [`AtomRow`] into it; the columnar BinaryCIF path builds it
/// straight from decoded columns, skipping the row struct and the
/// chain-id `CompactString` allocations. Atom-name fields must already be
/// `normalize_legacy_atom_name`-rewritten by the caller.
pub(crate) struct AtomCells<'a> {
    pub label_asym_id: &'a str,
    pub label_seq_id: i32,
    pub label_comp_id: [u8; 3],
    pub label_atom_id: [u8; 4],
    pub label_entity_id: Option<&'a str>,
    pub auth_seq_id: Option<i32>,
    pub auth_comp_id: Option<[u8; 3]>,
    pub auth_atom_id: Option<[u8; 4]>,
    pub alt_loc: Option<u8>,
    pub ins_code: Option<u8>,
    pub element: Element,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub occupancy: f32,
    pub b_factor: f32,
    pub formal_charge: i8,
}

/// Builder-side errors. Adapters wrap these into `AdapterError`.
#[derive(Debug, thiserror::Error)]
pub(crate) enum BuildError {
    /// An atom row contains NaN or infinite coordinates.
    #[error("invalid coordinate: {axis} = {value} (atom {label_atom_id})")]
    InvalidCoordinate {
        /// Offending axis ('x', 'y', or 'z').
        axis: char,
        /// Offending value.
        value: f32,
        /// Trimmed atom name from the offending row.
        label_atom_id: String,
    },
}

/// Builds typed `MoleculeEntity` values from a stream of atom rows.
///
/// Usage:
///
/// 1. [`EntityBuilder::new`].
/// 2. (mmCIF/BCIF only) [`EntityBuilder::register_entity`] once per entity from
///    the `_entity` / `_entity_poly` pre-pass.
/// 3. [`EntityBuilder::push_atom`] for each ATOM/HETATM row in parser order.
/// 4. [`EntityBuilder::finish`] to consume the builder.
pub(crate) struct EntityBuilder {
    allocator: EntityIdAllocator,
    hints: FxHashMap<String, ExpectedEntityType>,
    chains: FxHashMap<String, ChainState>,
    chain_order: Vec<String>,
    completion: Completion,
}

impl EntityBuilder {
    /// Create an empty builder at the default file-ingest completion level
    /// ([`Completion::Heavy`]).
    pub(crate) fn new() -> Self {
        Self::with_completion(Completion::Heavy)
    }

    /// Create an empty builder that emits polymer chains completed to
    /// `completion`. The emitters in `classify` carry this level into the
    /// chain constructors.
    pub(crate) fn with_completion(completion: Completion) -> Self {
        Self {
            allocator: EntityIdAllocator::new(),
            hints: FxHashMap::default(),
            chains: FxHashMap::default(),
            chain_order: Vec::new(),
            completion,
        }
    }

    /// Register an entity-type hint sourced from mmCIF `_entity` +
    /// `_entity_poly`. Idempotent on repeat with the same hint;
    /// first-wins on conflicting hints (with a debug log).
    #[allow(
        clippy::needless_pass_by_value,
        reason = "ExpectedEntityType is Copy; by-value matches the surface"
    )]
    pub(crate) fn register_entity(
        &mut self,
        label_entity_id: &str,
        hint: ExpectedEntityType,
    ) {
        if let Some(&existing) = self.hints.get(label_entity_id) {
            if existing != hint {
                log::debug!(
                    "EntityBuilder: ignoring conflicting hint {hint:?} for \
                     entity {label_entity_id} (first registration: \
                     {existing:?})",
                );
            }
            return;
        }
        let _ = self.hints.insert(label_entity_id.to_owned(), hint);
    }

    /// Push an atom row. Handles altLoc dedup, residue grouping, and
    /// chain bucketing.
    ///
    /// Errors only on invalid coordinates; rows with unknown elements or
    /// unrecognized residue names are accepted.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "by-value AtomRow matches the parser-emits-a-row surface; \
                  its fields are unpacked into AtomCells here"
    )]
    pub(crate) fn push_atom(&mut self, row: AtomRow) -> Result<(), BuildError> {
        self.push_cells(AtomCells {
            label_asym_id: row.label_asym_id.as_str(),
            label_seq_id: row.label_seq_id,
            label_comp_id: row.label_comp_id,
            label_atom_id: row.label_atom_id,
            label_entity_id: row.label_entity_id.as_deref(),
            auth_seq_id: row.auth_seq_id,
            auth_comp_id: row.auth_comp_id,
            auth_atom_id: row.auth_atom_id,
            alt_loc: row.alt_loc,
            ins_code: row.ins_code,
            element: row.element,
            x: row.x,
            y: row.y,
            z: row.z,
            occupancy: row.occupancy,
            b_factor: row.b_factor,
            formal_charge: row.formal_charge,
        })
    }

    /// Route a single atom's scalar cells through the accumulator. The
    /// columnar BinaryCIF path builds [`AtomCells`] straight from decoded
    /// columns and calls this directly — sharing the chain/residue/altLoc
    /// machinery with [`push_atom`](Self::push_atom) without materializing
    /// an [`AtomRow`] or the chain-id `CompactString`s.
    ///
    /// The atom-name choke point lives here: both name fields are rewritten
    /// to v3 form, feeding the residue HashMap key, the classification
    /// atom-set probe, and the flattened `Atom.name`. Coordinates are
    /// validated against the un-normalized name (so the error carries the
    /// raw atom name).
    pub(crate) fn push_cells(
        &mut self,
        mut cells: AtomCells<'_>,
    ) -> Result<(), BuildError> {
        validate_coords(cells.x, cells.y, cells.z, cells.label_atom_id)?;
        cells.label_atom_id = normalize_legacy_atom_name(cells.label_atom_id);
        cells.auth_atom_id = cells.auth_atom_id.map(normalize_legacy_atom_name);
        self.ensure_chain(cells.label_asym_id, cells.label_entity_id);
        let Some(chain) = self.chains.get_mut(cells.label_asym_id) else {
            unreachable!("chain inserted by ensure_chain");
        };
        let res_idx = chain.locate_or_create_residue(&cells);
        chain.residues[res_idx].apply_altloc_dedup(&cells);
        Ok(())
    }

    /// Consume the builder and produce `MoleculeEntity` values.
    /// Chain insertion order; global Water and Solvent bulks
    /// (accumulated under Unknown / NonPolymer hints) appended last.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "Result is part of the API contract; future error paths land \
                  here"
    )]
    pub(crate) fn finish(self) -> Result<Vec<MoleculeEntity>, BuildError> {
        let Self {
            mut allocator,
            hints,
            mut chains,
            chain_order,
            completion,
        } = self;

        let mut out: Vec<MoleculeEntity> = Vec::new();
        let mut water = GlobalBulk::default();
        let mut solvent = GlobalBulk::default();

        {
            let mut ctx = ChainCtx {
                allocator: &mut allocator,
                out: &mut out,
                water: &mut water,
                solvent: &mut solvent,
                completion,
            };
            for chain_key in &chain_order {
                let Some(mut chain) = chains.remove(chain_key) else {
                    unreachable!("chain in chain_order");
                };
                chain.residues.sort_by_key(|r| (r.label_seq_id, r.ins_code));
                let hint = chain
                    .entity_hint_key
                    .as_deref()
                    .and_then(|k| hints.get(k).copied())
                    .unwrap_or(ExpectedEntityType::Unknown);
                classify_chain(
                    hint,
                    &chain.pdb_chain_id,
                    &chain.residues,
                    &mut ctx,
                );
            }
        }

        water.flush(
            MoleculeType::Water,
            DEFAULT_WATER_RESNAME,
            &mut allocator,
            &mut out,
        );
        solvent.flush(
            MoleculeType::Solvent,
            DEFAULT_SOLVENT_RESNAME,
            &mut allocator,
            &mut out,
        );

        Ok(out)
    }

    fn ensure_chain(
        &mut self,
        label_asym_id: &str,
        label_entity_id: Option<&str>,
    ) {
        if self.chains.contains_key(label_asym_id) {
            return;
        }
        let state = ChainState {
            pdb_chain_id: label_asym_id.to_owned(),
            entity_hint_key: label_entity_id.map(str::to_owned),
            residues: Vec::new(),
            residue_index: FxHashMap::default(),
        };
        // The owned chain key lives both as the HashMap key and as the
        // chain_order entry that drives finish() ordering.
        let key = label_asym_id.to_owned();
        self.chain_order.push(key.clone());
        let _ = self.chains.insert(key, state);
    }
}

// Internal state

pub(super) struct ChainState {
    /// The chain's `label_asym_id`, carried through verbatim as the
    /// emitted entity's `pdb_chain_id`.
    pub(super) pdb_chain_id: String,
    pub(super) entity_hint_key: Option<String>,
    pub(super) residues: Vec<ResidueAccum>,
    pub(super) residue_index: FxHashMap<(i32, Option<u8>), usize>,
}

impl ChainState {
    fn locate_or_create_residue(&mut self, cells: &AtomCells<'_>) -> usize {
        let key = (cells.label_seq_id, cells.ins_code);
        if let Some(&idx) = self.residue_index.get(&key) {
            return idx;
        }
        let idx = self.residues.len();
        self.residues.push(ResidueAccum {
            label_seq_id: cells.label_seq_id,
            auth_seq_id: cells.auth_seq_id,
            label_comp_id: cells.label_comp_id,
            auth_comp_id: cells.auth_comp_id,
            ins_code: cells.ins_code,
            atoms: Vec::new(),
        });
        let _ = self.residue_index.insert(key, idx);
        idx
    }
}

pub(super) struct ResidueAccum {
    pub(super) label_seq_id: i32,
    pub(super) auth_seq_id: Option<i32>,
    pub(super) label_comp_id: [u8; 3],
    pub(super) auth_comp_id: Option<[u8; 3]>,
    pub(super) ins_code: Option<u8>,
    pub(super) atoms: Vec<([u8; 4], AtomChoice)>,
}

impl ResidueAccum {
    fn apply_altloc_dedup(&mut self, cells: &AtomCells<'_>) {
        let candidate = AtomChoice::from_cells(cells);
        match self
            .atoms
            .iter_mut()
            .find(|(name, _)| *name == cells.label_atom_id)
        {
            None => self.atoms.push((cells.label_atom_id, candidate)),
            Some(existing) => {
                if candidate_should_replace(&existing.1, &candidate) {
                    existing.1 = candidate;
                }
            }
        }
    }
}

pub(super) struct AtomChoice {
    alt_loc: Option<u8>,
    occupancy: f32,
    label_atom_id: [u8; 4],
    auth_atom_id: Option<[u8; 4]>,
    element: Element,
    position: Vec3,
    b_factor: f32,
    formal_charge: i8,
}

impl AtomChoice {
    fn from_cells(cells: &AtomCells<'_>) -> Self {
        Self {
            alt_loc: cells.alt_loc,
            occupancy: cells.occupancy,
            label_atom_id: cells.label_atom_id,
            auth_atom_id: cells.auth_atom_id,
            element: cells.element,
            position: Vec3::new(cells.x, cells.y, cells.z),
            b_factor: cells.b_factor,
            formal_charge: cells.formal_charge,
        }
    }

    pub(super) fn to_atom(&self) -> Atom {
        // Author-side atom name displaces label when present; the
        // structural-side identifier is still the dedup / grouping key.
        Atom {
            position: self.position,
            occupancy: self.occupancy,
            b_factor: self.b_factor,
            element: self.element,
            name: self.auth_atom_id.unwrap_or(self.label_atom_id),
            formal_charge: self.formal_charge,
            observed: true,
        }
    }
}

/// AltLoc dedup precedence: blank trumps any non-blank; among non-blank
/// higher occupancy wins; ties go to the alphabetically-earlier byte.
fn candidate_should_replace(
    existing: &AtomChoice,
    candidate: &AtomChoice,
) -> bool {
    match (existing.alt_loc, candidate.alt_loc) {
        (None, _) => false,
        (Some(_), None) => true,
        (Some(_), Some(_)) => {
            if candidate.occupancy > existing.occupancy {
                true
            } else if candidate.occupancy < existing.occupancy {
                false
            } else {
                candidate.alt_loc < existing.alt_loc
            }
        }
    }
}

// finish() context

pub(super) struct ChainCtx<'a> {
    pub(super) allocator: &'a mut EntityIdAllocator,
    pub(super) out: &'a mut Vec<MoleculeEntity>,
    pub(super) water: &'a mut GlobalBulk,
    pub(super) solvent: &'a mut GlobalBulk,
    /// Completion level the polymer emitters pass to the chain
    /// constructors.
    pub(super) completion: Completion,
}

#[derive(Default)]
pub(super) struct GlobalBulk {
    atoms: Vec<Atom>,
    residue_count: usize,
    residue_name: Option<[u8; 3]>,
}

impl GlobalBulk {
    pub(super) fn ingest(&mut self, r: &ResidueAccum) {
        for (_, choice) in &r.atoms {
            self.atoms.push(choice.to_atom());
        }
        self.residue_count += 1;
        if self.residue_name.is_none() {
            self.residue_name = Some(r.label_comp_id);
        }
    }

    fn flush(
        self,
        mol_type: MoleculeType,
        default_resname: [u8; 3],
        allocator: &mut EntityIdAllocator,
        out: &mut Vec<MoleculeEntity>,
    ) {
        if self.atoms.is_empty() {
            return;
        }
        let id = allocator.allocate();
        out.push(MoleculeEntity::Bulk(BulkEntity::new(
            id,
            mol_type,
            self.atoms,
            self.residue_name.unwrap_or(default_resname),
            self.residue_count,
        )));
    }
}

// Helpers

fn validate_coords(
    x: f32,
    y: f32,
    z: f32,
    label_atom_id: [u8; 4],
) -> Result<(), BuildError> {
    for (axis, value) in [('x', x), ('y', y), ('z', z)] {
        if !value.is_finite() {
            return Err(BuildError::InvalidCoordinate {
                axis,
                value,
                label_atom_id: trim_atom_name(label_atom_id),
            });
        }
    }
    Ok(())
}

fn trim_atom_name(name: [u8; 4]) -> String {
    std::str::from_utf8(&name)
        .unwrap_or("")
        .trim_matches(|c: char| c == ' ' || c == '\0')
        .to_owned()
}

#[cfg(test)]
mod completion_tests;
#[cfg(test)]
mod na_legacy_tests;
#[cfg(test)]
mod roundtrip_tests;
#[cfg(test)]
mod tests;
