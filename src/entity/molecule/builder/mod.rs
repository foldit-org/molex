//! Builder that turns parser-output atom rows into typed `MoleculeEntity`
//! values. Used by the PDB, mmCIF, and BinaryCIF adapters.

#![allow(
    dead_code,
    reason = "builder internals are exposed for adapters; some helpers are \
              still unwired"
)]

use std::collections::HashMap;

use glam::Vec3;

use super::atom::Atom;
use super::bulk::BulkEntity;
use super::chain::ChainIdMapper;
use super::id::EntityIdAllocator;
use super::{MoleculeEntity, MoleculeType};
use crate::element::Element;

mod classify;

use classify::classify_chain;

/// Maximum number of distinct chains the builder accepts. Matches the
/// printable-byte capacity of `ChainIdMapper`.
const MAX_CHAINS: usize = 90;
const DEFAULT_WATER_RESNAME: [u8; 3] = *b"HOH";
const DEFAULT_SOLVENT_RESNAME: [u8; 3] = *b"GOL";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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
pub(crate) struct AtomRow {
    /// `label_asym_id` (mmCIF) or chain letter (PDB).
    pub label_asym_id: String,
    /// `label_seq_id` (mmCIF) or `resSeq` (PDB). For non-polymer
    /// entities with `.` in the source, parsers pass `1`.
    pub label_seq_id: i32,
    /// `label_comp_id` (mmCIF) or `resName` (PDB), 3 chars, space-padded.
    pub label_comp_id: [u8; 3],
    /// `label_atom_id` (mmCIF) or atom name (PDB), 4 chars.
    pub label_atom_id: [u8; 4],
    /// `label_entity_id` (mmCIF). Joined against registered hints to
    /// pick the entity type. `None` on the PDB path.
    pub label_entity_id: Option<String>,

    /// `auth_asym_id`. `None` defaults to `label_asym_id`.
    pub auth_asym_id: Option<String>,
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
    /// Chain count exceeded printable-byte capacity. Adapters convert
    /// this to the mmCIF redirect error.
    #[error("structure exceeds {limit} chains; use mmCIF instead")]
    TooManyChains {
        /// Capacity that was exceeded.
        limit: usize,
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
    hints: HashMap<String, ExpectedEntityType>,
    chains: HashMap<String, ChainState>,
    chain_mapper: ChainIdMapper,
    chain_order: Vec<String>,
}

impl EntityBuilder {
    /// Create an empty builder.
    pub(crate) fn new() -> Self {
        Self {
            allocator: EntityIdAllocator::new(),
            hints: HashMap::new(),
            chains: HashMap::new(),
            chain_mapper: ChainIdMapper::new(),
            chain_order: Vec::new(),
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
    /// Errors only on invalid coordinates or chain-mapper overflow;
    /// rows with unknown elements or unrecognized residue names are
    /// accepted.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "AtomRow is moved into builder state in mmCIF / BCIF callers"
    )]
    pub(crate) fn push_atom(&mut self, row: AtomRow) -> Result<(), BuildError> {
        validate_coords(&row)?;
        self.ensure_chain(&row)?;
        let Some(chain) = self.chains.get_mut(&row.label_asym_id) else {
            unreachable!("chain inserted by ensure_chain");
        };
        let res_idx = chain.locate_or_create_residue(&row);
        chain.residues[res_idx].apply_altloc_dedup(&row);
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
            mut chain_mapper,
            chain_order,
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
                let chain_bytes = ChainBytes {
                    pdb_chain_id: chain.pdb_chain_id,
                    auth_asym_id: resolve_auth_chain_byte(
                        chain_key,
                        chain.auth_asym_id.as_deref(),
                        &mut chain_mapper,
                    ),
                };
                classify_chain(hint, chain_bytes, &chain.residues, &mut ctx);
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

    fn ensure_chain(&mut self, row: &AtomRow) -> Result<(), BuildError> {
        if self.chains.contains_key(&row.label_asym_id) {
            return Ok(());
        }
        if self.chain_order.len() >= MAX_CHAINS {
            return Err(BuildError::TooManyChains { limit: MAX_CHAINS });
        }
        let pdb_chain_id = self.chain_mapper.get_or_assign(&row.label_asym_id);
        let state = ChainState {
            pdb_chain_id,
            auth_asym_id: row.auth_asym_id.clone(),
            entity_hint_key: row.label_entity_id.clone(),
            residues: Vec::new(),
            residue_index: HashMap::new(),
        };
        // Cloning the chain key: needed both as HashMap key and as the
        // chain_order entry that drives finish() ordering.
        let key = row.label_asym_id.clone();
        self.chain_order.push(key.clone());
        let _ = self.chains.insert(key, state);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// Both chain bytes derived for an emitted entity: the label-side byte
/// (used internally and as the `pdb_chain_id`) and the optional
/// author-side byte (`None` when the auth string matches the label or
/// no auth string was supplied).
#[derive(Clone, Copy)]
pub(super) struct ChainBytes {
    pub(super) pdb_chain_id: u8,
    pub(super) auth_asym_id: Option<u8>,
}

pub(super) struct ChainState {
    pub(super) pdb_chain_id: u8,
    /// `auth_asym_id` captured from the first row that opened the
    /// chain. Resolved to a printable byte at `finish()` time.
    pub(super) auth_asym_id: Option<String>,
    pub(super) entity_hint_key: Option<String>,
    pub(super) residues: Vec<ResidueAccum>,
    pub(super) residue_index: HashMap<(i32, Option<u8>), usize>,
}

impl ChainState {
    fn locate_or_create_residue(&mut self, row: &AtomRow) -> usize {
        let key = (row.label_seq_id, row.ins_code);
        if let Some(&idx) = self.residue_index.get(&key) {
            return idx;
        }
        let idx = self.residues.len();
        self.residues.push(ResidueAccum {
            label_seq_id: row.label_seq_id,
            auth_seq_id: row.auth_seq_id,
            label_comp_id: row.label_comp_id,
            auth_comp_id: row.auth_comp_id,
            ins_code: row.ins_code,
            atoms: HashMap::new(),
            atom_order: Vec::new(),
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
    pub(super) atoms: HashMap<[u8; 4], AtomChoice>,
    pub(super) atom_order: Vec<[u8; 4]>,
}

impl ResidueAccum {
    fn apply_altloc_dedup(&mut self, row: &AtomRow) {
        let candidate = AtomChoice::from_row(row);
        match self.atoms.get_mut(&row.label_atom_id) {
            None => {
                let _ = self.atoms.insert(row.label_atom_id, candidate);
                self.atom_order.push(row.label_atom_id);
            }
            Some(existing) => {
                if candidate_should_replace(existing, &candidate) {
                    *existing = candidate;
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
    fn from_row(row: &AtomRow) -> Self {
        Self {
            alt_loc: row.alt_loc,
            occupancy: row.occupancy,
            label_atom_id: row.label_atom_id,
            auth_atom_id: row.auth_atom_id,
            element: row.element,
            position: Vec3::new(row.x, row.y, row.z),
            b_factor: row.b_factor,
            formal_charge: row.formal_charge,
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

// ---------------------------------------------------------------------------
// finish() context
// ---------------------------------------------------------------------------

pub(super) struct ChainCtx<'a> {
    pub(super) allocator: &'a mut EntityIdAllocator,
    pub(super) out: &'a mut Vec<MoleculeEntity>,
    pub(super) water: &'a mut GlobalBulk,
    pub(super) solvent: &'a mut GlobalBulk,
}

#[derive(Default)]
pub(super) struct GlobalBulk {
    atoms: Vec<Atom>,
    residue_count: usize,
    residue_name: Option<[u8; 3]>,
}

impl GlobalBulk {
    pub(super) fn ingest(&mut self, r: &ResidueAccum) {
        for name in &r.atom_order {
            self.atoms.push(r.atoms[name].to_atom());
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_coords(row: &AtomRow) -> Result<(), BuildError> {
    for (axis, value) in [('x', row.x), ('y', row.y), ('z', row.z)] {
        if !value.is_finite() {
            return Err(BuildError::InvalidCoordinate {
                axis,
                value,
                label_atom_id: trim_atom_name(row.label_atom_id),
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

/// Resolve a chain's `auth_asym_id` to a printable byte.
///
/// Returns `None` when the auth string matches the label key (the
/// common case, no extra mapper entry needed) or when no auth string
/// was supplied. Otherwise allocates a fresh byte in the mapper to
/// disambiguate it from the label-side chain byte.
fn resolve_auth_chain_byte(
    label_key: &str,
    auth: Option<&str>,
    mapper: &mut ChainIdMapper,
) -> Option<u8> {
    let auth = auth?;
    if auth == label_key {
        return None;
    }
    Some(mapper.get_or_assign(auth))
}

#[cfg(test)]
mod completion_tests;
#[cfg(test)]
mod roundtrip_tests;
#[cfg(test)]
mod tests;
