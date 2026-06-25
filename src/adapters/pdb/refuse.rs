//! Refusal helpers: BeEM-bundle detection on read, legacy-cap checks on
//! write.

use crate::entity::molecule::MoleculeEntity;
use crate::ops::error::AdapterError;

/// Legacy PDB format caps. Set by wwPDB v3.3 column widths.
pub(super) const MAX_PDB_ATOMS: usize = 99_999;
pub(super) const MAX_PDB_POLYMER_CHAINS: usize = 62;
pub(super) const MAX_PDB_RESIDUES_PER_CHAIN: usize = 9_999;

/// Returns `Some(error)` if the input is recognized as a BeEM split
/// bundle (which the mmCIF adapter handles instead). Detection is
/// either filename-based (`<id>-pdb-bundle<n>.pdb`) or content-based
/// (a `HEADER` record containing the text `PDB BUNDLE`).
pub(super) fn check_beem_bundle(
    input: &str,
    path: Option<&std::path::Path>,
) -> Option<AdapterError> {
    if let Some(id) = path
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .and_then(extract_pdb_id_from_bundle_filename)
    {
        return Some(beem_redirect_error(Some(&id)));
    }
    for (i, line) in input.lines().enumerate() {
        if i >= 50 {
            break;
        }
        if line.starts_with("HEADER") && line.contains("PDB BUNDLE") {
            let id = line
                .get(62..66)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            return Some(beem_redirect_error(id.as_deref()));
        }
    }
    None
}

/// Recover the PDB ID from a BeEM bundle filename of the form
/// `<id>-pdb-bundle<digits>.pdb` (case-insensitive on the bundle part).
pub(super) fn extract_pdb_id_from_bundle_filename(
    name: &str,
) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    let stem = lower.strip_suffix(".pdb")?;
    let idx = stem.rfind("-pdb-bundle")?;
    let suffix = &stem[idx + "-pdb-bundle".len()..];
    if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let id = &name[..idx];
    (!id.is_empty()).then(|| id.to_owned())
}

fn beem_redirect_error(pdb_id: Option<&str>) -> AdapterError {
    let url = pdb_id.map_or_else(
        || "https://www.rcsb.org/structure/".to_owned(),
        |id| format!("https://files.rcsb.org/download/{id}.cif.gz"),
    );
    AdapterError::PdbParseError(format!(
        "BeEM split PDB bundle detected; the molex PDB adapter does not \
         support split bundles. Use the mmCIF adapter (download {url}) \
         instead."
    ))
}

/// Validate that an entity slice fits inside legacy PDB column widths.
pub(super) fn validate_writable<E: std::borrow::Borrow<MoleculeEntity>>(
    entities: &[E],
) -> Result<(), AdapterError> {
    let total_atoms: usize =
        entities.iter().map(|e| e.borrow().atom_count()).sum();
    if total_atoms > MAX_PDB_ATOMS {
        return Err(legacy_limit_error(&format!(
            "{total_atoms} atoms exceed the legacy cap of {MAX_PDB_ATOMS}"
        )));
    }
    let polymer_chains = entities
        .iter()
        .filter(|e| {
            matches!(
                e.borrow(),
                MoleculeEntity::Protein(_) | MoleculeEntity::NucleicAcid(_)
            )
        })
        .count();
    if polymer_chains > MAX_PDB_POLYMER_CHAINS {
        return Err(legacy_limit_error(&format!(
            "{polymer_chains} polymer chains exceed the legacy cap of \
             {MAX_PDB_POLYMER_CHAINS}"
        )));
    }
    // Legacy PDB col 22 holds exactly one chain character. A multi-char
    // mmCIF `label_asym_id` (ribosomes, capsids) cannot round-trip; refuse
    // rather than truncate.
    for entity in entities {
        if let Some(chain) = entity.borrow().pdb_chain_id() {
            if chain.chars().count() > 1 {
                return Err(legacy_limit_error(&format!(
                    "chain id {chain:?} is multi-character and cannot fit the \
                     single-column legacy PDB chain field"
                )));
            }
        }
    }
    for entity in entities {
        let e = entity.borrow();
        let res_count = match e {
            MoleculeEntity::Protein(p) => p.residues.len(),
            MoleculeEntity::NucleicAcid(n) => n.residues.len(),
            MoleculeEntity::Bulk(b) => b.columns.len(),
            MoleculeEntity::SmallMolecule(_) => 1,
        };
        if res_count > MAX_PDB_RESIDUES_PER_CHAIN {
            return Err(legacy_limit_error(&format!(
                "entity has {res_count} residues exceeding the legacy \
                 per-chain cap of {MAX_PDB_RESIDUES_PER_CHAIN}"
            )));
        }
    }
    Ok(())
}

fn legacy_limit_error(detail: &str) -> AdapterError {
    AdapterError::PdbParseError(format!(
        "Structure does not fit in legacy PDB format ({detail}); use the \
         mmCIF writer (`Assembly.to_mmcif` in Python, `assembly_to_mmcif` in \
         Rust) for assemblies of this size."
    ))
}
