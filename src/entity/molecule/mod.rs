//! Entity types and molecule classification.

/// Core atom type.
pub mod atom;
pub(crate) mod builder;
/// Bulk entity (water, solvent).
pub mod bulk;
pub(crate) mod chain;
pub(crate) mod classify;
/// Missing-atom completion against ideal-geometry templates.
pub(crate) mod complete;
/// Structural diff between two views of one entity.
pub mod diff;
/// Opaque entity ID with controlled allocation.
pub mod id;
/// Nucleic acid entity (DNA, RNA).
pub mod nucleic_acid;
/// Polymer residue (shared by protein and nucleic acid entities).
pub mod polymer;
/// Protein entity with residues and segment breaks.
pub mod protein;
/// Small molecule entity (ligand, ion, cofactor).
pub mod small_molecule;
/// Entity and Polymer traits.
pub mod traits;

pub use atom::Atom;
#[allow(
    unused_imports,
    reason = "exported for adapter consumers; not all consumers wired yet"
)]
pub(crate) use builder::{
    AtomRow, BuildError, EntityBuilder, ExpectedEntityType,
};
pub use classify::classify_residue;
pub use diff::EntityDiffError;
use glam::Vec3;
pub use id::{EntityId, EntityIdAllocator};
pub use nucleic_acid::NucleotideRing;
pub use polymer::Residue;

use self::bulk::BulkEntity;
use self::nucleic_acid::NAEntity;
use self::protein::ProteinEntity;
use self::small_molecule::SmallMoleculeEntity;
use self::traits::Entity;
use crate::analysis::aabb::Aabb;
/// Classification of molecule types found in structural biology files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoleculeType {
    /// Amino acid polymer.
    Protein,
    /// Deoxyribonucleic acid polymer.
    DNA,
    /// Ribonucleic acid polymer.
    RNA,
    /// Non-polymer small molecule (drug, substrate, etc.).
    Ligand,
    /// Single-atom metal or halide ion.
    Ion,
    /// Water molecule.
    Water,
    /// Lipid or detergent molecule.
    Lipid,
    /// Enzyme cofactor (heme, NAD, FAD, Fe-S cluster, etc.).
    Cofactor,
    /// Crystallization solvent or buffer artifact.
    Solvent,
}

// ---------------------------------------------------------------------------
// MoleculeEntity enum
// ---------------------------------------------------------------------------

/// Structural discriminant of a [`MoleculeEntity`]: the four-way
/// taxonomy of the variant itself. Distinct from the finer
/// [`MoleculeType`] classification.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EntityKind {
    /// A single protein chain.
    Protein,
    /// A single DNA or RNA chain.
    NucleicAcid,
    /// A single non-polymer molecule.
    SmallMolecule,
    /// A group of identical small molecules.
    Bulk,
}

/// A single entity: one logical molecule (a protein chain, a ligand, waters,
/// etc.).
///
/// This is an enum wrapping concrete entity types. Each variant owns its
/// entity data directly. Use the accessor methods to work with entities
/// polymorphically.
#[derive(Debug, Clone)]
pub enum MoleculeEntity {
    /// A single protein chain.
    Protein(ProteinEntity),
    /// A single DNA or RNA chain.
    NucleicAcid(NAEntity),
    /// A single non-polymer molecule (ligand, ion, cofactor, lipid).
    SmallMolecule(SmallMoleculeEntity),
    /// A group of identical small molecules (water, solvent).
    Bulk(BulkEntity),
}

impl MoleculeEntity {
    /// Return a copy with protein chains rebuilt as a single continuous
    /// segment (see [`ProteinEntity::to_continuous`]); non-protein
    /// entities are cloned unchanged. For synthetic / noisy design
    /// outputs (ML diffusion intermediates) that should render as one
    /// connected chain regardless of the current coordinate geometry.
    #[must_use]
    pub fn to_continuous(&self) -> Self {
        match self {
            MoleculeEntity::Protein(p) => {
                MoleculeEntity::Protein(p.to_continuous())
            }
            other => other.clone(),
        }
    }

    // -- Entity trait delegation --

    /// Unique entity identifier.
    #[must_use]
    pub fn id(&self) -> EntityId {
        match self {
            MoleculeEntity::Protein(e) => e.id(),
            MoleculeEntity::NucleicAcid(e) => e.id(),
            MoleculeEntity::SmallMolecule(e) => e.id(),
            MoleculeEntity::Bulk(e) => e.id(),
        }
    }

    /// Classification of this entity's molecule type.
    #[must_use]
    pub fn molecule_type(&self) -> MoleculeType {
        match self {
            MoleculeEntity::Protein(e) => e.molecule_type(),
            MoleculeEntity::NucleicAcid(e) => e.molecule_type(),
            MoleculeEntity::SmallMolecule(e) => e.molecule_type(),
            MoleculeEntity::Bulk(e) => e.molecule_type(),
        }
    }

    /// The structural discriminant of this entity, 1:1 with the variant.
    #[must_use]
    pub fn entity_kind(&self) -> EntityKind {
        match self {
            Self::Protein(_) => EntityKind::Protein,
            Self::NucleicAcid(_) => EntityKind::NucleicAcid,
            Self::SmallMolecule(_) => EntityKind::SmallMolecule,
            Self::Bulk(_) => EntityKind::Bulk,
        }
    }

    /// Slice over the underlying atoms.
    #[must_use]
    pub fn atom_set(&self) -> &[Atom] {
        match self {
            MoleculeEntity::Protein(e) => e.atoms(),
            MoleculeEntity::NucleicAcid(e) => e.atoms(),
            MoleculeEntity::SmallMolecule(e) => e.atoms(),
            MoleculeEntity::Bulk(e) => e.atoms(),
        }
    }

    /// Mutable reference to the underlying `Vec<Atom>`.
    pub fn atom_set_mut(&mut self) -> &mut [Atom] {
        match self {
            MoleculeEntity::Protein(e) => &mut e.atoms,
            MoleculeEntity::NucleicAcid(e) => &mut e.atoms,
            MoleculeEntity::SmallMolecule(e) => &mut e.atoms,
            MoleculeEntity::Bulk(e) => &mut e.atoms,
        }
    }

    /// All atom positions as Vec3.
    #[must_use]
    pub fn positions(&self) -> Vec<Vec3> {
        self.atom_set().iter().map(|a| a.position).collect()
    }

    /// Number of atoms in this entity.
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.atom_set().len()
    }

    /// Read access to this entity's residue list, or `None` for
    /// non-polymer entities.
    #[must_use]
    pub fn residues(&self) -> Option<&[Residue]> {
        match self {
            MoleculeEntity::Protein(e) => Some(&e.residues),
            MoleculeEntity::NucleicAcid(e) => Some(&e.residues),
            MoleculeEntity::SmallMolecule(_) | MoleculeEntity::Bulk(_) => None,
        }
    }

    /// Mutable access to this entity's `(atoms, residues)` pair for
    /// polymer entities. Returns `None` for non-polymer entities.
    ///
    /// Intended for the `ops::edit` path that re-shapes residue
    /// content; callers are responsible for keeping `atom_range`s
    /// consistent.
    pub fn polymer_parts_mut(
        &mut self,
    ) -> Option<(&mut Vec<Atom>, &mut Vec<Residue>)> {
        match self {
            MoleculeEntity::Protein(e) => Some((&mut e.atoms, &mut e.residues)),
            MoleculeEntity::NucleicAcid(e) => {
                Some((&mut e.atoms, &mut e.residues))
            }
            MoleculeEntity::SmallMolecule(_) | MoleculeEntity::Bulk(_) => None,
        }
    }

    // -- Variant-specific accessors --

    /// If this entity is a protein, return it.
    #[must_use]
    pub fn as_protein(&self) -> Option<&ProteinEntity> {
        match self {
            MoleculeEntity::Protein(e) => Some(e),
            _ => None,
        }
    }

    /// If this entity is a nucleic acid, return it.
    #[must_use]
    pub fn as_nucleic_acid(&self) -> Option<&NAEntity> {
        match self {
            MoleculeEntity::NucleicAcid(e) => Some(e),
            _ => None,
        }
    }

    /// PDB chain identifier byte for polymer entities, `None` for others.
    #[must_use]
    pub fn pdb_chain_id(&self) -> Option<u8> {
        match self {
            MoleculeEntity::Protein(e) => Some(e.pdb_chain_id),
            MoleculeEntity::NucleicAcid(e) => Some(e.pdb_chain_id),
            _ => None,
        }
    }

    /// If this entity is a small molecule, return it.
    #[must_use]
    pub fn as_small_molecule(&self) -> Option<&SmallMoleculeEntity> {
        match self {
            MoleculeEntity::SmallMolecule(e) => Some(e),
            _ => None,
        }
    }

    /// If this entity is a bulk entity, return it.
    #[must_use]
    pub fn as_bulk(&self) -> Option<&BulkEntity> {
        match self {
            MoleculeEntity::Bulk(e) => Some(e),
            _ => None,
        }
    }

    /// Compute the axis-aligned bounding box for this entity's atoms.
    #[must_use]
    pub fn aabb(&self) -> Option<Aabb> {
        Aabb::from_positions(&self.positions())
    }

    /// Human-readable label (e.g. "Protein Chain A", "Ligand (ATP)", "Zn2+
    /// Ion").
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "match arms per molecule type are straightforward"
    )]
    pub fn label(&self) -> String {
        let mol_type = self.molecule_type();
        match mol_type {
            MoleculeType::Protein => polymer_label(self, "Protein"),
            MoleculeType::DNA => polymer_label(self, "DNA"),
            MoleculeType::RNA => polymer_label(self, "RNA"),
            MoleculeType::Ligand => {
                if let MoleculeEntity::SmallMolecule(e) = self {
                    format!("Ligand ({})", e.display_name)
                } else {
                    "Ligand".to_owned()
                }
            }
            MoleculeType::Ion => {
                if let MoleculeEntity::SmallMolecule(e) = self {
                    format!("{} Ion", e.display_name)
                } else {
                    "Ion".to_owned()
                }
            }
            MoleculeType::Water => {
                if let MoleculeEntity::Bulk(e) = self {
                    format!("Water ({} molecules)", e.molecule_count)
                } else {
                    "Water".to_owned()
                }
            }
            MoleculeType::Lipid => {
                if let MoleculeEntity::SmallMolecule(e) = self {
                    format!("Lipid ({})", e.display_name)
                } else {
                    format!("Lipid ({} molecules)", self.residue_count())
                }
            }
            MoleculeType::Cofactor => {
                if let MoleculeEntity::SmallMolecule(e) = self {
                    e.display_name.clone()
                } else {
                    "Cofactor".to_owned()
                }
            }
            MoleculeType::Solvent => {
                if let MoleculeEntity::Bulk(e) = self {
                    format!("Solvent ({} molecules)", e.molecule_count)
                } else {
                    "Solvent".to_owned()
                }
            }
        }
    }

    /// Whether this entity type participates in tab-cycling focus.
    /// Protein: no (focused at group level). Water, Ion: no (ambient).
    /// Ligand, DNA, RNA: yes.
    #[must_use]
    pub fn is_focusable(&self) -> bool {
        !matches!(
            self.molecule_type(),
            MoleculeType::Water | MoleculeType::Ion | MoleculeType::Solvent
        )
    }

    /// Number of residues (for polymer/nucleic) or molecules (for small
    /// mol/ion/water).
    #[must_use]
    pub fn residue_count(&self) -> usize {
        match self {
            MoleculeEntity::Protein(e) => e.residues.len(),
            MoleculeEntity::NucleicAcid(e) => e.residues.len(),
            MoleculeEntity::SmallMolecule(_) => 1,
            MoleculeEntity::Bulk(e) => e.molecule_count,
        }
    }

    /// Set the entity ID. Used when reassembling an entity vec.
    pub fn set_id(&mut self, new_id: EntityId) {
        match self {
            MoleculeEntity::Protein(e) => e.id = new_id,
            MoleculeEntity::NucleicAcid(e) => e.id = new_id,
            MoleculeEntity::SmallMolecule(e) => e.id = new_id,
            MoleculeEntity::Bulk(e) => e.id = new_id,
        }
    }
}

/// Format a polymer entity label from its PDB chain ID.
fn polymer_label(entity: &MoleculeEntity, type_name: &str) -> String {
    entity.pdb_chain_id().map_or_else(
        || type_name.to_owned(),
        |id| format!("{type_name} {}", id as char),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::element::Element;
    use crate::entity::molecule::id::EntityIdAllocator;

    fn atom_at(name: &str, element: Element, x: f32, y: f32, z: f32) -> Atom {
        let mut n = [b' '; 4];
        for (i, b) in name.bytes().take(4).enumerate() {
            n[i] = b;
        }
        Atom {
            position: Vec3::new(x, y, z),
            occupancy: 1.0,
            b_factor: 0.0,
            element,
            name: n,
            formal_charge: 0,
        }
    }

    fn res_bytes(s: &str) -> [u8; 3] {
        let mut n = [b' '; 3];
        for (i, b) in s.bytes().take(3).enumerate() {
            n[i] = b;
        }
        n
    }

    fn residue(name: &str, seq: i32, range: std::ops::Range<usize>) -> Residue {
        Residue {
            name: res_bytes(name),
            label_seq_id: seq,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: range,
            variants: Vec::new(),
        }
    }

    /// Build a 2-residue protein on chain A with backbone atoms at known
    /// positions. The C->N gap between residues exceeds 2 A so a segment
    /// break falls between them.
    ///
    /// Residues are named `UNK` so the missing-atom completion pass (which
    /// only fires on residues that resolve to a standard template) leaves
    /// the backbone-only atom set untouched; these tests assert exact
    /// atom counts and positions and are not about chemistry completion.
    fn two_residue_protein() -> MoleculeEntity {
        let atoms = vec![
            atom_at("N", Element::N, 1.0, 2.0, 3.0),
            atom_at("CA", Element::C, 4.0, 5.0, 6.0),
            atom_at("C", Element::C, 7.0, 8.0, 9.0),
            atom_at("O", Element::O, 10.0, 11.0, 12.0),
            atom_at("N", Element::N, 13.0, 14.0, 15.0),
            atom_at("CA", Element::C, 16.0, 17.0, 18.0),
            atom_at("C", Element::C, 19.0, 20.0, 21.0),
            atom_at("O", Element::O, 22.0, 23.0, 24.0),
        ];
        let residues = vec![residue("UNK", 1, 0..4), residue("UNK", 2, 4..8)];
        let id = EntityIdAllocator::new().allocate();
        MoleculeEntity::Protein(ProteinEntity::new(
            id, atoms, residues, b'A', None,
        ))
    }

    fn water_entity(positions: &[Vec3]) -> MoleculeEntity {
        let atoms: Vec<Atom> = positions
            .iter()
            .map(|p| atom_at("O", Element::O, p.x, p.y, p.z))
            .collect();
        let id = EntityIdAllocator::new().allocate();
        MoleculeEntity::Bulk(BulkEntity::new(
            id,
            MoleculeType::Water,
            atoms,
            res_bytes("HOH"),
            positions.len(),
        ))
    }

    fn zinc_ion() -> MoleculeEntity {
        let atoms = vec![atom_at("ZN", Element::Zn, 5.0, 6.0, 7.0)];
        let id = EntityIdAllocator::new().allocate();
        MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
            id,
            MoleculeType::Ion,
            atoms,
            res_bytes("ZN"),
        ))
    }

    #[test]
    fn protein_classifies_correctly() {
        let entity = two_residue_protein();
        assert_eq!(entity.molecule_type(), MoleculeType::Protein);
    }

    #[test]
    fn entity_id_is_set() {
        let entity = two_residue_protein();
        let _id = entity.id();
    }

    #[test]
    fn atom_set_and_atom_count() {
        let entity = two_residue_protein();
        assert_eq!(entity.atom_count(), 8);
        assert_eq!(entity.atom_set().len(), 8);
    }

    #[test]
    fn as_protein_returns_some_for_protein() {
        let entity = two_residue_protein();
        assert!(entity.as_protein().is_some());
        assert!(entity.as_nucleic_acid().is_none());
        assert!(entity.as_small_molecule().is_none());
        assert!(entity.as_bulk().is_none());
    }

    #[test]
    fn label_for_protein() {
        let entity = two_residue_protein();
        let label = entity.label();
        assert!(label.contains("Protein"), "label={label}");
        assert!(label.contains('A'), "label should contain chain A: {label}");
    }

    #[test]
    fn residue_count_for_protein() {
        let entity = two_residue_protein();
        assert_eq!(entity.residue_count(), 2);
    }

    #[test]
    fn aabb_is_some_for_nonempty_entity() {
        let entity = two_residue_protein();
        let aabb = entity.aabb();
        assert!(aabb.is_some());
        let bb = aabb.unwrap();
        assert!(bb.min.x <= 1.0);
        assert!(bb.max.x >= 22.0);
    }

    #[test]
    fn positions_returns_all_atom_positions() {
        let entity = two_residue_protein();
        let positions = entity.positions();
        assert_eq!(positions.len(), 8);
        assert!((positions[0].x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn water_entity_accessors() {
        let water =
            water_entity(&[Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)]);
        assert_eq!(water.molecule_type(), MoleculeType::Water);
        assert!(water.as_bulk().is_some());
        assert_eq!(water.atom_count(), 2);
        assert_eq!(water.residue_count(), 2);
        assert!(water.label().contains("Water"));
    }

    #[test]
    fn ion_entity_accessors() {
        let ion = zinc_ion();
        assert_eq!(ion.molecule_type(), MoleculeType::Ion);
        assert!(ion.as_small_molecule().is_some());
        assert_eq!(ion.residue_count(), 1);
    }

    #[test]
    fn is_focusable_for_protein_and_water() {
        let protein = two_residue_protein();
        assert!(protein.is_focusable());

        let water = water_entity(&[Vec3::ZERO]);
        assert!(!water.is_focusable());
    }

    #[test]
    fn pdb_chain_id_for_protein() {
        let entity = two_residue_protein();
        assert_eq!(entity.pdb_chain_id(), Some(b'A'));
    }
}
