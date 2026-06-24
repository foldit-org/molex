//! Serialization for the assembly binary wire format.

use super::variants::serialize_variants_section;
use super::{molecule_type_to_wire, ASSEMBLY_MAGIC, ASSEMBLY_VERSION};
use crate::assembly::Assembly;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::MoleculeEntity;
use crate::ops::codec::AdapterError;

/// Serialize an [`Assembly`] to the assembly binary format.
///
/// Writes the entity list only. `generation` resets to 0 on the
/// [`deserialize_assembly`](super::deserialize::deserialize_assembly)
/// rebuild via [`Assembly::new`], and `ss_types` comes back empty until a
/// caller opts in via [`Assembly::recompute_ss`].
///
/// Format (version 2):
/// - 8 bytes: magic `b"ASSEMBLY"`
/// - 1 byte:  version (currently 2)
/// - 4 bytes: entity_count (u32 BE)
/// - Per entity header (variable):
///   - 1 byte: `molecule_type` wire byte
///   - 4 bytes: `atom_count` (`u32` BE)
///   - 4 bytes: `entity_id` (`u32` BE) — the originator's `EntityId.raw()`.
///     Receivers reconstruct entities with the same id so cross-boundary edit
///     references resolve.
///   - 2 bytes: `chain_len` (`u16` BE)
///   - `chain_len` bytes: the entity's chain id (UTF-8 `label_asym_id`). Empty
///     for non-polymer entities.
/// - Per atom (25 bytes):
///   - 12 bytes: x, y, z (`f32` BE x 3)
///   - 3 bytes:  `res_name`
///   - 4 bytes:  `res_num` (`i32` BE)
///   - 4 bytes:  `atom_name`
///   - 2 bytes:  element symbol (byte 0, byte 1 or 0)
/// - Per-entity variants section (after all atoms, in entity order). See the
///   `variants` submodule for the inner layout.
///
/// Chain id moved from a per-atom byte (version 1) to a per-entity header
/// string here, lifting the single-printable-byte chain cap. The version-1
/// decode path is retained in `deserialize`.
///
/// Occupancy and b_factor are not preserved on the wire; deserialize
/// resets them to 1.0 and 0.0 respectively.
///
/// # Errors
///
/// Currently infallible but returns `Result` for API consistency.
pub fn serialize_assembly(
    assembly: &Assembly,
) -> Result<Vec<u8>, AdapterError> {
    serialize_entities(assembly.entities())
}

/// Serialize a raw entity slice to the assembly binary format.
///
/// Internal helper for in-crate paths that operate on owned entity
/// slices and don't need an [`Assembly`] wrapper.
#[allow(
    clippy::unnecessary_wraps,
    reason = "mirrors `serialize_assembly` which keeps the `Result` for API \
              consistency"
)]
pub(crate) fn serialize_entities<E: std::borrow::Borrow<MoleculeEntity>>(
    entities: &[E],
) -> Result<Vec<u8>, AdapterError> {
    let total_atoms: usize =
        entities.iter().map(|e| e.borrow().atom_count()).sum();
    let chain_bytes: usize = entities
        .iter()
        .filter_map(|e| e.borrow().pdb_chain_id().map(str::len))
        .sum();
    let header_size = 8 + 1 + 4 + entities.len() * 11 + chain_bytes;
    let atom_size = total_atoms * 25;
    let mut buffer = Vec::with_capacity(header_size + atom_size);

    // Magic + version byte
    buffer.extend_from_slice(ASSEMBLY_MAGIC);
    buffer.push(ASSEMBLY_VERSION);

    // Entity count
    #[allow(clippy::cast_possible_truncation)] // entity count fits in u32
    buffer.extend_from_slice(&(entities.len() as u32).to_be_bytes());

    // Per-entity headers (chain string lives here, not per atom).
    for entity in entities {
        let entity = entity.borrow();
        buffer.push(molecule_type_to_wire(entity.molecule_type()));
        #[allow(clippy::cast_possible_truncation)] // atom count fits in u32
        buffer.extend_from_slice(&(entity.atom_count() as u32).to_be_bytes());
        buffer.extend_from_slice(&entity.id().raw().to_be_bytes());
        write_chain_id(entity.pdb_chain_id().unwrap_or(""), &mut buffer);
    }

    // Atom data per entity, walking residues directly.
    for entity in entities {
        write_entity_atoms(entity.borrow(), &mut buffer);
    }

    // Per-entity variants section. Empty when no residue carries
    // variants — costs 4 bytes per entity in that case.
    serialize_variants_section(entities, &mut buffer);

    Ok(buffer)
}

/// Write a length-prefixed chain string: a `u16` BE byte length followed
/// by the UTF-8 bytes. A chain id longer than `u16::MAX` bytes (never seen
/// in practice) is truncated to fit the prefix.
fn write_chain_id(chain_id: &str, buffer: &mut Vec<u8>) {
    let bytes = chain_id.as_bytes();
    let len = bytes.len().min(usize::from(u16::MAX));
    #[allow(clippy::cast_possible_truncation)] // clamped to u16::MAX above
    buffer.extend_from_slice(&(len as u16).to_be_bytes());
    buffer.extend_from_slice(&bytes[..len]);
}

fn write_entity_atoms(entity: &MoleculeEntity, buffer: &mut Vec<u8>) {
    match entity {
        MoleculeEntity::Protein(e) => {
            write_polymer_atoms(&e.atoms, &e.residues, buffer);
        }
        MoleculeEntity::NucleicAcid(e) => {
            write_polymer_atoms(&e.atoms, &e.residues, buffer);
        }
        MoleculeEntity::SmallMolecule(e) => {
            for atom in &e.atoms {
                write_atom_row(atom, e.residue_name, 1, buffer);
            }
        }
        MoleculeEntity::Bulk(e) =>
        {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                reason = "atom count fits in i32 for valid structures"
            )]
            for (i, atom) in e.atoms.iter().enumerate() {
                write_atom_row(atom, e.residue_name, (i as i32) + 1, buffer);
            }
        }
    }
}

fn write_polymer_atoms(
    atoms: &[Atom],
    residues: &[Residue],
    buffer: &mut Vec<u8>,
) {
    for residue in residues {
        for idx in residue.atom_range.clone() {
            write_atom_row(
                &atoms[idx],
                residue.name,
                residue.label_seq_id,
                buffer,
            );
        }
    }
}

pub(crate) fn write_atom_row(
    atom: &Atom,
    res_name: [u8; 3],
    res_num: i32,
    buffer: &mut Vec<u8>,
) {
    buffer.extend_from_slice(&atom.position.x.to_be_bytes());
    buffer.extend_from_slice(&atom.position.y.to_be_bytes());
    buffer.extend_from_slice(&atom.position.z.to_be_bytes());
    buffer.extend_from_slice(&res_name);
    buffer.extend_from_slice(&res_num.to_be_bytes());
    buffer.extend_from_slice(&atom.name);
    let sym = atom.element.symbol();
    let sym_bytes = sym.as_bytes();
    buffer.push(sym_bytes.first().copied().unwrap_or(b'X'));
    buffer.push(sym_bytes.get(1).copied().unwrap_or(0));
}
