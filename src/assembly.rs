//! Top-level `Assembly` container: entities plus eagerly-recomputed
//! secondary structure.
//!
//! `Assembly` is the host-owned structural source of truth. Every
//! `&mut Assembly` mutation bumps the generation counter and recomputes
//! per-entity secondary structure before returning, so readers of a given
//! snapshot see consistent `ss_types`. Rendering connections (disulfides,
//! backbone H-bonds, etc.) are owner-populated via `set_connections`, not
//! derived here.

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec3;

use crate::analysis::bonds::disulfide::detect_disulfides;
use crate::analysis::bonds::hydrogen::{detect_hbonds, HBond};
use crate::analysis::ss::dssp::classify;
use crate::analysis::SSType;
use crate::atom_id::AtomId;
use crate::connection::{AtomEnd, AtomLink, ConnectionType};
use crate::entity::molecule::id::EntityId;
use crate::entity::molecule::MoleculeEntity;

/// Top-level container of entities plus eagerly-computed secondary
/// structure.
///
/// Each entity is stored behind an `Arc`, so cloning an `Assembly` is
/// O(entities) of refcount bumps, independent of the total atom count.
/// Mutations clone only the touched entity (`Arc::make_mut`) and leave
/// the rest aliased with prior snapshots. Per-entity `ss_types` is also
/// `Arc`-shared so snapshots that didn't trigger a rebuild stay aliased.
/// The generation counter increments on every mutation so consumers can
/// detect snapshots cheaply.
#[derive(Debug, Clone)]
pub struct Assembly {
    entities: Vec<Arc<MoleculeEntity>>,
    ss_types: HashMap<EntityId, Arc<Vec<SSType>>>,
    /// Owner-populated rendering connections, keyed by category. NOT
    /// computed by `recompute_derived`; the assembly's owner selects a
    /// provider and sets this. Empty until populated.
    connections: HashMap<ConnectionType, Vec<AtomLink>>,
    generation: u64,
}

/// Snapshot of per-entity atom positions for whole-assembly replacement.
///
/// Entities present in the snapshot but missing from the target
/// `Assembly` are ignored; entities in the target but missing from the
/// snapshot retain their current positions.
#[derive(Debug, Clone, Default)]
pub struct CoordinateSnapshot {
    per_entity: HashMap<EntityId, Vec<Vec3>>,
}

impl CoordinateSnapshot {
    /// Create a snapshot from a pre-built per-entity map.
    #[must_use]
    pub fn new(per_entity: HashMap<EntityId, Vec<Vec3>>) -> Self {
        Self { per_entity }
    }

    /// Capture the current positions of every entity in an `Assembly`.
    ///
    /// Intended for "reset to original" flows: take a snapshot after
    /// load, run mutations, then replay each entity's coords back via
    /// [`Assembly::apply_edits`] with one
    /// [`AssemblyEdit::SetEntityCoords`](crate::ops::edit::AssemblyEdit::SetEntityCoords)
    /// per entry to restore.
    #[must_use]
    pub fn from_assembly(assembly: &Assembly) -> Self {
        let per_entity = assembly
            .entities
            .iter()
            .map(|e| (e.id(), e.positions()))
            .collect();
        Self { per_entity }
    }

    /// Positions for a specific entity, if present.
    #[must_use]
    pub fn positions(&self, id: EntityId) -> Option<&[Vec3]> {
        self.per_entity.get(&id).map(Vec::as_slice)
    }

    /// Number of entities covered by this snapshot.
    #[must_use]
    pub fn len(&self) -> usize {
        self.per_entity.len()
    }

    /// Whether the snapshot covers no entities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.per_entity.is_empty()
    }
}

impl Assembly {
    /// Build an `Assembly` from a collection of entities.
    ///
    /// Runs per-entity DSSP classification to populate `ss_types`.
    /// `generation` is initialized to 0.
    #[must_use]
    pub fn new(entities: Vec<MoleculeEntity>) -> Self {
        let mut this = Self {
            entities: entities.into_iter().map(Arc::new).collect(),
            ss_types: HashMap::new(),
            connections: HashMap::new(),
            generation: 0,
        };
        this.recompute_derived();
        this
    }

    /// Build an `Assembly` from entities a host already holds behind
    /// `Arc`s, preserving that sharing.
    ///
    /// The `Arc`-preserving sibling of [`Assembly::new`]: hosts that
    /// already keep `Arc<MoleculeEntity>` snapshots (e.g. foldit's
    /// `EntityStore`) can hand them straight in instead of deep-cloning
    /// each payload only to have `new` re-`Arc` it. Runs the same
    /// `recompute_derived` as `new`, so per-entity DSSP classification is
    /// populated identically. `generation` is initialized to 0; stamp it
    /// via [`Assembly::set_generation`] if the host tracks its own version
    /// counter.
    #[must_use]
    pub fn from_arcs(entities: Vec<Arc<MoleculeEntity>>) -> Self {
        let mut this = Self {
            entities,
            ss_types: HashMap::new(),
            connections: HashMap::new(),
            generation: 0,
        };
        this.recompute_derived();
        this
    }

    /// Project to a fresh heavy-complete `Assembly`: every protein and
    /// nucleic-acid entity is rebuilt with its missing heavy atoms
    /// fabricated (see [`MoleculeEntity::normalize`]); non-polymer entities
    /// pass through unchanged. A caller-driven projection that runs
    /// heavy-atom completion on already-built entities (the heavy-only
    /// sibling of [`Assembly::to_all_atom`]). Completion runs on each
    /// chain's surviving residues; residues dropped at construction are
    /// not resurrected.
    ///
    /// The result is a brand-new assembly with re-indexed atoms (the
    /// appended atoms shift every later atom), so atom indices and
    /// `AtomId`s are NOT stable across this projection. Rendering
    /// connections are therefore dropped rather than copied: their stored
    /// `AtomLink` endpoints reference the pre-completion indices and would
    /// mis-point after the shift. Owners re-populate them on the next
    /// publish. The generation counter is carried forward for snapshot
    /// continuity.
    #[must_use]
    pub fn normalize(&self) -> Assembly {
        let entities: Vec<Arc<MoleculeEntity>> = self
            .entities
            .iter()
            .map(|e| Arc::new(e.normalize()))
            .collect();
        let mut projected = Assembly::from_arcs(entities);
        projected.set_generation(self.generation);
        projected
    }

    /// Project to a fresh all-atom `Assembly`: every protein and
    /// nucleic-acid entity is rebuilt with its template hydrogens
    /// fabricated (see [`MoleculeEntity::to_all_atom`]); non-polymer
    /// entities pass through unchanged. The file-ingest path stays
    /// heavy-only; this is the opt-in to a fully protonated model.
    ///
    /// The result is a brand-new assembly with re-indexed atoms (the
    /// appended hydrogens shift every later atom), so atom indices and
    /// `AtomId`s are NOT stable across this projection. Rendering
    /// connections are therefore dropped rather than copied: their stored
    /// `AtomLink` endpoints reference the heavy-only indices and would
    /// mis-point after the shift. Owners re-populate them on the next
    /// publish. The generation counter is carried forward for snapshot
    /// continuity.
    #[must_use]
    pub fn to_all_atom(&self) -> Assembly {
        let entities: Vec<Arc<MoleculeEntity>> = self
            .entities
            .iter()
            .map(|e| Arc::new(e.to_all_atom()))
            .collect();
        let mut projected = Assembly::from_arcs(entities);
        projected.set_generation(self.generation);
        projected
    }

    // -- Read accessors ----------------------------------------------

    /// All entities in declaration order.
    ///
    /// Each entry is an `Arc<MoleculeEntity>` so cloning the slice into
    /// a new `Assembly` is O(entities) of refcount bumps. `&Arc<T>`
    /// derefs to `&T` for method calls, so most call sites do not need
    /// to change shape.
    #[must_use]
    pub fn entities(&self) -> &[Arc<MoleculeEntity>] {
        &self.entities
    }

    /// Look up an entity by id.
    #[must_use]
    pub fn entity(&self, id: EntityId) -> Option<&MoleculeEntity> {
        self.entities.iter().map(Arc::as_ref).find(|e| e.id() == id)
    }

    /// Monotonic counter incremented on every mutation.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Stamp `generation` directly. Use when constructing a fresh
    /// `Assembly` snapshot from a host that tracks its own version
    /// counter (e.g., foldit's `EntityStore::head_assembly`): without
    /// this, a freshly-built `Assembly::new` always starts at
    /// generation 0 and downstream consumers that gate on the
    /// counter (viso's `poll_assembly`) silently skip the second and
    /// subsequent publishes.
    pub fn set_generation(&mut self, generation: u64) {
        self.generation = generation;
    }

    /// Secondary structure classification for an entity.
    ///
    /// Returns an empty slice for entities that don't have an SS
    /// assignment (non-protein, or proteins with fewer than two complete
    /// backbone residues).
    #[must_use]
    pub fn ss_types(&self, id: EntityId) -> &[SSType] {
        self.ss_types.get(&id).map_or(&[], |ss| ss.as_slice())
    }

    /// Current rendering connections, keyed by category.
    #[must_use]
    pub fn connections(&self) -> &HashMap<ConnectionType, Vec<AtomLink>> {
        &self.connections
    }

    /// Replace the rendering connections wholesale. The owner re-applies
    /// the selected provider's set on every publish.
    pub fn set_connections(
        &mut self,
        connections: HashMap<ConnectionType, Vec<AtomLink>>,
    ) {
        self.connections = connections;
    }

    /// Compute the viewer-only fallback connections (disulfides and
    /// backbone hydrogen bonds) from geometry. Used when no host provider
    /// supplies connections. Pure: does not mutate `self`.
    #[must_use]
    pub fn detect_fallback_connections(
        &self,
    ) -> HashMap<ConnectionType, Vec<AtomLink>> {
        let mut out = HashMap::new();

        let disulfides: Vec<AtomLink> = detect_disulfides(&self.entities)
            .iter()
            .map(|b| AtomLink::new(AtomEnd::Atom(b.a), AtomEnd::Atom(b.b)))
            .collect();
        if !disulfides.is_empty() {
            let _ = out.insert(ConnectionType::Disulfide, disulfides);
        }

        let hbonds = self.fallback_hbond_links();
        if !hbonds.is_empty() {
            let _ = out.insert(ConnectionType::HBond, hbonds);
        }

        out
    }

    /// Resolve the flat-backbone hbond list to per-entity atom-pair links,
    /// mirroring the viewer's flat-to-atom resolution: donor N to acceptor
    /// carbonyl C.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "atom indices are bounded by entity size (< u32::MAX)"
    )]
    fn fallback_hbond_links(&self) -> Vec<AtomLink> {
        let mut flat_to_atoms: Vec<[AtomId; 4]> = Vec::new();
        for entity in &self.entities {
            let Some(protein) = entity.as_protein() else {
                continue;
            };
            let eid = protein.id;
            for residue in &protein.residues {
                let start = residue.atom_range.start as u32;
                flat_to_atoms.push([
                    AtomId {
                        entity: eid,
                        index: start,
                    },
                    AtomId {
                        entity: eid,
                        index: start + 1,
                    },
                    AtomId {
                        entity: eid,
                        index: start + 2,
                    },
                    AtomId {
                        entity: eid,
                        index: start + 3,
                    },
                ]);
            }
        }

        compute_flat_hbonds(&self.entities)
            .iter()
            .filter_map(|h| {
                let donor = flat_to_atoms.get(h.donor)?;
                let acceptor = flat_to_atoms.get(h.acceptor)?;
                Some(AtomLink::new(
                    AtomEnd::Atom(donor[0]),
                    AtomEnd::Atom(acceptor[2]),
                ))
            })
            .collect()
    }

    // -- Mutation methods --------------------------------------------
    //
    // All direct content mutators are `pub(crate)`: cross-crate callers
    // must go through [`Self::apply_edit`] / [`Self::apply_edits`] so the
    // host's broadcast routing has a single typed funnel. The reason is
    // protocol-shaped: every host-side Assembly change must produce
    // exactly one `UpdateAssembly` payload (`Full` or `Delta`) per the
    // plugin protocol; a caller that mutates directly bypasses the
    // queue and leaves peer plugins one generation behind.

    /// Append an entity. Bumps the generation counter and recomputes
    /// per-entity secondary structure.
    pub(crate) fn add_entity(&mut self, entity: MoleculeEntity) {
        self.entities.push(Arc::new(entity));
        self.after_mutation();
    }

    /// Remove an entity by id. Bumps the generation counter and
    /// recomputes per-entity secondary structure.
    pub(crate) fn remove_entity(&mut self, id: EntityId) {
        self.entities.retain(|e| e.id() != id);
        let _ = self.ss_types.remove(&id);
        self.after_mutation();
    }

    /// Mutable slice of the entity list. Crate-internal; reserved for
    /// the `ops::edit` apply path which needs `Arc::make_mut` access
    /// to individual entities while running its own derived-data
    /// bookkeeping via [`Self::after_mutation_pub`].
    pub(crate) fn entities_mut(&mut self) -> &mut [Arc<MoleculeEntity>] {
        &mut self.entities
    }

    /// Bump generation and recompute derived data. Crate-internal
    /// counterpart to the private `after_mutation` for the
    /// `ops::edit` apply functions.
    pub(crate) fn after_mutation_pub(&mut self) {
        self.after_mutation();
    }

    // -- Internal helpers --------------------------------------------

    fn after_mutation(&mut self) {
        self.generation = self.generation.saturating_add(1);
        self.recompute_derived();
    }

    fn recompute_derived(&mut self) {
        self.ss_types = compute_per_entity_ss(&self.entities);
    }
}

fn compute_per_entity_ss<E: std::borrow::Borrow<MoleculeEntity>>(
    entities: &[E],
) -> HashMap<EntityId, Arc<Vec<SSType>>> {
    let mut out = HashMap::new();
    for entity in entities {
        let Some(protein) = entity.borrow().as_protein() else {
            continue;
        };
        let backbone = protein.to_backbone();
        if backbone.is_empty() {
            continue;
        }
        let hbonds = detect_hbonds(&backbone);
        let ss = classify(&hbonds, backbone.len());
        let _ = out.insert(protein.id, Arc::new(ss));
    }
    out
}

fn compute_flat_hbonds<E: std::borrow::Borrow<MoleculeEntity>>(
    entities: &[E],
) -> Vec<HBond> {
    let mut flat = Vec::new();
    for entity in entities {
        let Some(protein) = entity.borrow().as_protein() else {
            continue;
        };
        flat.extend(protein.to_backbone());
    }
    detect_hbonds(&flat)
}

#[cfg(test)]
#[path = "assembly_tests.rs"]
mod tests;
