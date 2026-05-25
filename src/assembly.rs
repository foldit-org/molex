//! Top-level `Assembly` container: entities, cross-entity bonds, and
//! eagerly-recomputed derived data (secondary structure + backbone H-bonds).
//!
//! `Assembly` is the host-owned structural source of truth. Every
//! `&mut Assembly` mutation bumps the generation counter and recomputes
//! all derived data before returning, so readers of a given snapshot see
//! consistent `ss_types`, `hbonds`, and `cross_entity_bonds`.

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec3;

use crate::analysis::bonds::disulfide::detect_disulfides;
use crate::analysis::bonds::hydrogen::{detect_hbonds, HBond};
use crate::analysis::ss::dssp::classify;
use crate::analysis::SSType;
use crate::atom_id::AtomId;
use crate::bond::CovalentBond;
use crate::entity::molecule::id::EntityId;
use crate::entity::molecule::MoleculeEntity;

/// Top-level container of entities plus eagerly-computed derived data.
///
/// Each entity is stored behind an `Arc`, so cloning an `Assembly` is
/// O(entities) of refcount bumps, independent of the total atom count.
/// Mutations clone only the touched entity (`Arc::make_mut`) and leave
/// the rest aliased with prior snapshots. Derived data (`ss_types`,
/// `hbonds`) is also `Arc`-shared so snapshots that didn't trigger a
/// rebuild stay aliased. The generation counter increments on every
/// mutation so consumers can detect snapshots cheaply.
#[derive(Debug, Clone)]
pub struct Assembly {
    entities: Vec<Arc<MoleculeEntity>>,
    cross_entity_bonds: Vec<CovalentBond>,
    ss_types: HashMap<EntityId, Arc<Vec<SSType>>>,
    hbonds: Arc<Vec<HBond>>,
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
    /// Runs disulfide detection, per-entity DSSP classification, and
    /// flat-backbone H-bond detection to populate all derived data.
    /// `generation` is initialized to 0.
    #[must_use]
    pub fn new(entities: Vec<MoleculeEntity>) -> Self {
        let mut this = Self {
            entities: entities.into_iter().map(Arc::new).collect(),
            cross_entity_bonds: Vec::new(),
            ss_types: HashMap::new(),
            hbonds: Arc::new(Vec::new()),
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
    /// `recompute_derived` as `new`, so all derived data (disulfides,
    /// DSSP, H-bonds) is populated identically. `generation` is
    /// initialized to 0; stamp it via [`Assembly::set_generation`] if the
    /// host tracks its own version counter.
    #[must_use]
    pub fn from_arcs(entities: Vec<Arc<MoleculeEntity>>) -> Self {
        let mut this = Self {
            entities,
            cross_entity_bonds: Vec::new(),
            ss_types: HashMap::new(),
            hbonds: Arc::new(Vec::new()),
            generation: 0,
        };
        this.recompute_derived();
        this
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

    /// Backbone hydrogen bonds (Kabsch-Sander) across all protein
    /// entities. Donor and acceptor indices refer to the flattened
    /// per-protein-entity backbone sequence produced by concatenating
    /// `ProteinEntity::to_backbone()` in entity order.
    #[must_use]
    pub fn hbonds(&self) -> &[HBond] {
        &self.hbonds
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

    /// Cross-entity covalent bonds. In this migration this contains
    /// disulfides only (decision #12); same-chain SG-SG bridges are
    /// also collected here rather than on the owning `ProteinEntity`.
    #[must_use]
    pub fn cross_entity_bonds(&self) -> &[CovalentBond] {
        &self.cross_entity_bonds
    }

    /// All atoms bonded to `atom`, yielding the far endpoint of each
    /// matching bond. Walks the owning entity's intra-entity bond list
    /// and the assembly's `cross_entity_bonds`.
    pub fn bonds_touching(
        &self,
        atom: AtomId,
    ) -> impl Iterator<Item = AtomId> + '_ {
        let intra = self
            .entity(atom.entity)
            .and_then(entity_bonds)
            .unwrap_or(&[])
            .iter()
            .filter_map(move |b| other_endpoint(b, atom));
        let cross = self
            .cross_entity_bonds
            .iter()
            .filter_map(move |b| other_endpoint(b, atom));
        intra.chain(cross)
    }

    /// Cross-entity disulfide bonds: pairs where both endpoints
    /// resolve to an SG atom inside a CYS residue.
    pub fn disulfides(&self) -> impl Iterator<Item = &CovalentBond> + '_ {
        self.cross_entity_bonds
            .iter()
            .filter(|b| self.is_cys_sg(b.a) && self.is_cys_sg(b.b))
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
    /// all derived data.
    pub(crate) fn add_entity(&mut self, entity: MoleculeEntity) {
        self.entities.push(Arc::new(entity));
        self.after_mutation();
    }

    /// Replace an entity in place, preserving its position in the
    /// internal vec. `entity.id()` must equal `id`. If `id` is not
    /// present, the mutation is logged and skipped (generation not
    /// advanced); readers keep seeing the previous snapshot. Use
    /// this rather than `remove_entity` + `add_entity` whenever the
    /// goal is to swap a single entity's body without disturbing the
    /// vec ordering of the rest.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn replace_entity(
        &mut self,
        id: EntityId,
        entity: MoleculeEntity,
    ) {
        debug_assert_eq!(
            entity.id(),
            id,
            "replace_entity: entity.id() must match id",
        );
        let Some(idx) = self.entities.iter().position(|e| e.id() == id) else {
            log::error!("Assembly::replace_entity: unknown entity id {id}");
            return;
        };
        self.entities[idx] = Arc::new(entity);
        self.after_mutation();
    }

    /// Remove an entity by id. Any `cross_entity_bonds` touching the
    /// removed entity are purged as part of the derived-data
    /// recomputation.
    pub(crate) fn remove_entity(&mut self, id: EntityId) {
        self.entities.retain(|e| e.id() != id);
        let _ = self.ss_types.remove(&id);
        self.after_mutation();
    }

    /// Replace the positions of a single entity. If `coords.len()` does
    /// not match the entity's atom count, the mutation is abandoned
    /// (logged at error level) and the generation counter is not
    /// advanced; readers keep seeing the previous snapshot.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn update_positions(
        &mut self,
        entity: EntityId,
        coords: &[Vec3],
    ) {
        let Some(idx) = self.entities.iter().position(|e| e.id() == entity)
        else {
            log::error!(
                "Assembly::update_positions: unknown entity id {entity}"
            );
            return;
        };
        if self.entities[idx].atom_count() != coords.len() {
            log::error!(
                "Assembly::update_positions: entity {entity} has {} atoms but \
                 {} coords were supplied; skipping update",
                self.entities[idx].atom_count(),
                coords.len(),
            );
            return;
        }
        let atoms = Arc::make_mut(&mut self.entities[idx]).atom_set_mut();
        for (atom, &pos) in atoms.iter_mut().zip(coords.iter()) {
            atom.position = pos;
        }
        self.after_mutation();
    }

    /// Apply positions for every entity covered by `snapshot`.
    ///
    /// Entities in the snapshot whose atom counts don't match their
    /// target entity are skipped (logged at error level) while other
    /// entities still get their positions applied. Entities absent from
    /// the snapshot retain their current positions. Bumps the
    /// generation counter and recomputes derived data exactly once,
    /// regardless of how many entities were touched.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "phase_3.md locks this signature; `snapshot` is the one-shot \
                  \"apply this whole-assembly snapshot\" input; callers hand \
                  over ownership rather than keeping the snapshot alive."
    )]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn set_coordinate_snapshot(
        &mut self,
        snapshot: CoordinateSnapshot,
    ) {
        for entity in &mut self.entities {
            let entity_id = entity.id();
            let Some(coords) = snapshot.per_entity.get(&entity_id) else {
                continue;
            };
            if entity.atom_count() != coords.len() {
                log::error!(
                    "Assembly::set_coordinate_snapshot: entity {entity_id} \
                     has {} atoms but snapshot carries {}; skipping this \
                     entity",
                    entity.atom_count(),
                    coords.len(),
                );
                continue;
            }
            let atoms = Arc::make_mut(entity).atom_set_mut();
            for (atom, &pos) in atoms.iter_mut().zip(coords.iter()) {
                atom.position = pos;
            }
        }
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
        self.cross_entity_bonds = detect_disulfides(&self.entities);
        self.ss_types = compute_per_entity_ss(&self.entities);
        self.hbonds = Arc::new(compute_flat_hbonds(&self.entities));
    }

    fn is_cys_sg(&self, atom: AtomId) -> bool {
        let Some(entity) = self.entity(atom.entity) else {
            return false;
        };
        let Some(protein) = entity.as_protein() else {
            return false;
        };
        let idx = atom.index as usize;
        let Some(residue) = protein
            .residues
            .iter()
            .find(|r| r.atom_range.contains(&idx))
        else {
            return false;
        };
        if trimmed(&residue.name) != b"CYS" {
            return false;
        }
        let Some(atom) = protein.atoms.get(idx) else {
            return false;
        };
        trimmed_atom_name(&atom.name) == b"SG"
    }
}

fn entity_bonds(entity: &MoleculeEntity) -> Option<&[CovalentBond]> {
    match entity {
        MoleculeEntity::Protein(e) => Some(&e.bonds),
        MoleculeEntity::NucleicAcid(e) => Some(&e.bonds),
        MoleculeEntity::SmallMolecule(e) => Some(&e.bonds),
        MoleculeEntity::Bulk(_) => None,
    }
}

fn other_endpoint(bond: &CovalentBond, atom: AtomId) -> Option<AtomId> {
    if bond.a == atom {
        Some(bond.b)
    } else if bond.b == atom {
        Some(bond.a)
    } else {
        None
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

fn trimmed(name: &[u8; 3]) -> &[u8] {
    let mut end = name.len();
    while end > 0 && (name[end - 1] == b' ' || name[end - 1] == 0) {
        end -= 1;
    }
    &name[..end]
}

fn trimmed_atom_name(name: &[u8; 4]) -> &[u8] {
    let mut end = name.len();
    while end > 0 && (name[end - 1] == b' ' || name[end - 1] == 0) {
        end -= 1;
    }
    let mut start = 0;
    while start < end && (name[start] == b' ' || name[start] == 0) {
        start += 1;
    }
    &name[start..end]
}

#[cfg(test)]
#[path = "assembly_tests.rs"]
mod tests;
