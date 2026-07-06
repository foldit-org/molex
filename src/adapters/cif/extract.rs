//! Typed extractors for CIF data blocks.
//!
//! Each extractor implements `TryFrom<&Block>`, pulling typed data out of the
//! untyped DOM. Extraction fails gracefully when required categories or tags
//! are missing.

use super::dom::{Block, Value};
use crate::element::Element;

/// Unit cell parameters.
#[derive(Debug, Clone)]
pub struct UnitCell {
    /// Cell length a (angstroms).
    pub a: f64,
    /// Cell length b (angstroms).
    pub b: f64,
    /// Cell length c (angstroms).
    pub c: f64,
    /// Cell angle alpha (degrees).
    pub alpha: f64,
    /// Cell angle beta (degrees).
    pub beta: f64,
    /// Cell angle gamma (degrees).
    pub gamma: f64,
}

/// A single atom from the `_atom_site` category.
#[derive(Debug, Clone)]
pub struct AtomSite {
    /// `ATOM` or `HETATM`.
    pub group: String,
    /// Atom name (e.g. `CA`, `N`, `OG1`).
    pub label: String,
    /// Residue name (e.g. `ALA`, `GLY`).
    pub residue: String,
    /// Chain ID.
    pub chain: String,
    /// Residue sequence number.
    pub seq_id: Option<i32>,
    /// Cartesian x coordinate (angstroms).
    pub x: f64,
    /// Cartesian y coordinate (angstroms).
    pub y: f64,
    /// Cartesian z coordinate (angstroms).
    pub z: f64,
    /// Element symbol (e.g. `C`, `N`, `O`).
    pub element: String,
    /// Fractional occupancy (0.0 to 1.0).
    pub occupancy: f64,
    /// Isotropic B-factor (temperature factor).
    pub b_factor: f64,
}

impl AtomSite {
    /// Resolve this atom's [`Element`], falling back to the atom name when the
    /// CIF omits an explicit element symbol.
    #[must_use]
    pub fn element(&self) -> Element {
        if self.element.is_empty() {
            Element::from_atom_name(&self.label)
        } else {
            Element::from_symbol(&self.element)
        }
    }
}

/// Coordinate data extracted from an mmCIF block.
#[derive(Debug, Clone)]
pub struct CoordinateData {
    /// All atom sites in the block.
    pub atoms: Vec<AtomSite>,
    /// Unit cell parameters, if present.
    pub cell: Option<UnitCell>,
    /// Space group name, if present.
    pub spacegroup: Option<String>,
}

/// How the observed data was originally stored in the SF-CIF file.
///
/// Structure factor amplitudes (F) and intensities (I) are related by I = F².
/// The parser always converts to amplitudes in [`Reflection::f_meas`], but this
/// tag records what was actually present in the file so downstream code can
/// audit the provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObsDataType {
    /// Amplitudes (F) were present in the file. No conversion needed.
    Amplitude,
    /// Intensities (I) were present and converted via F = √I, σ_F = σ_I /
    /// (2√I).
    Intensity,
}

/// A single reflection from the `_refln` category.
///
/// The [`f_meas`](Self::f_meas) and [`sigma_f_meas`](Self::sigma_f_meas) fields
/// are always in **amplitude** (F) units, regardless of whether the source file
/// stored amplitudes or intensities. See [`ReflectionData::obs_data_type`]
/// to check what was originally present.
#[derive(Debug, Clone)]
pub struct Reflection {
    /// Miller index h.
    pub h: i32,
    /// Miller index k.
    pub k: i32,
    /// Miller index l.
    pub l: i32,
    /// Measured structure factor amplitude (always F, even if the file had I).
    pub f_meas: Option<f64>,
    /// Standard uncertainty of measured amplitude (always σ_F).
    pub sigma_f_meas: Option<f64>,
    /// R-free flag: `true` = free/test set, `false` = working set.
    pub free_flag: bool,
}

/// Structure factor / reflection data extracted from an SF-CIF block.
#[derive(Debug, Clone)]
pub struct ReflectionData {
    /// Unit cell parameters (required for reflection data).
    pub cell: UnitCell,
    /// Space group name, if present.
    pub spacegroup: Option<String>,
    /// All reflections in the block.
    pub reflections: Vec<Reflection>,
    /// Whether the source file contained amplitudes (F) or intensities (I).
    pub obs_data_type: ObsDataType,
    /// Whether the R-free flags were read from the file (`true`) or all
    /// reflections defaulted to the working set (`false`).
    pub free_flags_from_file: bool,
}

/// Auto-detected content of a CIF data block.
#[derive(Debug)]
pub enum CifContent {
    /// Block contains atomic coordinate data.
    Coordinates(CoordinateData),
    /// Block contains structure factor / reflection data.
    Reflections(ReflectionData),
    /// Block could not be classified; raw DOM retained.
    Unknown(Block),
}

/// Errors from typed extraction.
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    /// A required CIF category (loop) is missing.
    #[error("missing category: {0}")]
    MissingCategory(String),
    /// A required CIF tag is missing.
    #[error("missing required tag: {0}")]
    MissingTag(String),
    /// A value could not be parsed to the expected type.
    #[error("parse error in {tag} row {row}: {detail}")]
    ParseError {
        /// The tag that failed to parse.
        tag: String,
        /// The row index (0-based) where parsing failed.
        row: usize,
        /// Description of the parse failure.
        detail: String,
    },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_cell(block: &Block) -> Option<UnitCell> {
    Some(UnitCell {
        a: block.get("_cell.length_a")?.as_f64()?,
        b: block.get("_cell.length_b")?.as_f64()?,
        c: block.get("_cell.length_c")?.as_f64()?,
        alpha: block.get("_cell.angle_alpha")?.as_f64()?,
        beta: block.get("_cell.angle_beta")?.as_f64()?,
        gamma: block.get("_cell.angle_gamma")?.as_f64()?,
    })
}

fn extract_spacegroup(block: &Block) -> Option<String> {
    block
        .get("_symmetry.space_group_name_H-M")
        .or_else(|| block.get("_space_group.name_H-M_alt"))
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

fn require_f64(
    v: &Value,
    tag: &str,
    row: usize,
) -> Result<f64, ExtractionError> {
    v.as_f64().ok_or_else(|| ExtractionError::ParseError {
        tag: tag.into(),
        row,
        detail: format!("expected float, got {v:?}"),
    })
}

fn require_i32(
    v: &Value,
    tag: &str,
    row: usize,
) -> Result<i32, ExtractionError> {
    v.as_i32().ok_or_else(|| ExtractionError::ParseError {
        tag: tag.into(),
        row,
        detail: format!("expected integer, got {v:?}"),
    })
}

// ---------------------------------------------------------------------------
// TryFrom impls
// ---------------------------------------------------------------------------

impl TryFrom<&Block> for UnitCell {
    type Error = ExtractionError;

    fn try_from(block: &Block) -> Result<Self, Self::Error> {
        extract_cell(block)
            .ok_or_else(|| ExtractionError::MissingTag("_cell.length_*".into()))
    }
}

/// Optional per-atom columns for extraction.
struct OptionalAtomColumns<'a> {
    elements: Option<Vec<&'a Value>>,
    bfactors: Option<Vec<&'a Value>>,
    occupancies: Option<Vec<&'a Value>>,
    seq_ids: Option<Vec<&'a Value>>,
    groups: Option<Vec<&'a Value>>,
}

fn extract_atom(
    row: &[&Value],
    i: usize,
    opt: &OptionalAtomColumns<'_>,
) -> Result<AtomSite, ExtractionError> {
    Ok(AtomSite {
        group: opt
            .groups
            .as_ref()
            .and_then(|v| v.get(i))
            .and_then(|v| v.as_str())
            .unwrap_or("ATOM")
            .to_owned(),
        label: row[0].as_str().unwrap_or("").to_owned(),
        residue: row[1].as_str().unwrap_or("").to_owned(),
        chain: row[2].as_str().unwrap_or("").to_owned(),
        seq_id: opt
            .seq_ids
            .as_ref()
            .and_then(|v| v.get(i))
            .and_then(|v| v.as_i32()),
        x: require_f64(row[3], "_atom_site.Cartn_x", i)?,
        y: require_f64(row[4], "_atom_site.Cartn_y", i)?,
        z: require_f64(row[5], "_atom_site.Cartn_z", i)?,
        element: opt
            .elements
            .as_ref()
            .and_then(|v| v.get(i))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned(),
        occupancy: opt
            .occupancies
            .as_ref()
            .and_then(|v| v.get(i))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0),
        b_factor: opt
            .bfactors
            .as_ref()
            .and_then(|v| v.get(i))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
    })
}

impl TryFrom<&Block> for CoordinateData {
    type Error = ExtractionError;

    fn try_from(block: &Block) -> Result<Self, Self::Error> {
        let cols = block
            .columns(&[
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
            ])
            .ok_or_else(|| {
                ExtractionError::MissingCategory("_atom_site".into())
            })?;

        let opt = OptionalAtomColumns {
            elements: block
                .column("_atom_site.type_symbol")
                .map(Iterator::collect),
            bfactors: block
                .column("_atom_site.B_iso_or_equiv")
                .map(Iterator::collect),
            occupancies: block
                .column("_atom_site.occupancy")
                .map(Iterator::collect),
            seq_ids: block
                .column("_atom_site.label_seq_id")
                .map(Iterator::collect),
            groups: block.column("_atom_site.group_PDB").map(Iterator::collect),
        };

        let mut atoms = Vec::with_capacity(cols.nrows());
        for (i, row) in cols.iter().enumerate() {
            atoms.push(extract_atom(&row, i, &opt)?);
        }

        Ok(CoordinateData {
            atoms,
            cell: extract_cell(block),
            spacegroup: extract_spacegroup(block),
        })
    }
}

/// Try the first column name that exists in the block.
fn first_available_column<'a>(
    block: &'a Block,
    names: &[&str],
) -> Option<Vec<&'a Value>> {
    for &name in names {
        if let Some(col) = block.column(name) {
            return Some(col.collect());
        }
    }
    None
}

/// Determine whether a `_refln.status` character means "free set".
///
/// PDB convention: `f` = free-set, `o` = observed/working.
/// Some files use `0`/`1` in the status column too.
fn status_is_free(ch: char) -> bool {
    matches!(ch, 'f' | 'F')
}

/// Determine whether a `_refln.pdbx_r_free_flag` integer means "free set".
///
/// PDB convention: typically 0 = free, 1 = working in some files, but the
/// dominant convention (CCP4 / Phenix / Refmac) is: the flag value matching
/// the "test" set index is free. Since most depositors use flag=1 for free,
/// we treat 1 as free. This is configurable downstream if needed.
fn rfree_flag_is_free(val: i32) -> bool {
    val == 1
}

impl TryFrom<&Block> for ReflectionData {
    type Error = ExtractionError;

    #[allow(clippy::too_many_lines)]
    fn try_from(block: &Block) -> Result<Self, Self::Error> {
        let cell = extract_cell(block).ok_or_else(|| {
            ExtractionError::MissingTag("_cell.length_*".into())
        })?;

        let cols = block
            .columns(&["_refln.index_h", "_refln.index_k", "_refln.index_l"])
            .ok_or_else(|| ExtractionError::MissingCategory("_refln".into()))?;

        // ── Observed data: try amplitudes first, fall back to intensities ──
        //
        // Amplitude columns (F):
        //   _refln.F_meas_au  /  _refln.F_meas_sigma_au   (modern mmCIF)
        //   _refln.F_obs      /  _refln.F_obs_sigma        (older convention)
        //
        // Intensity columns (I = F²):
        //   _refln.intensity_meas  /  _refln.intensity_sigma   (modern)
        //   _refln.I_obs           /  _refln.I_obs_sigma       (older)

        let amp_col = first_available_column(
            block,
            &["_refln.F_meas_au", "_refln.F_obs"],
        );
        let amp_sigma_col = first_available_column(
            block,
            &["_refln.F_meas_sigma_au", "_refln.F_obs_sigma"],
        );

        let int_col = first_available_column(
            block,
            &["_refln.intensity_meas", "_refln.I_obs"],
        );
        let int_sigma_col = first_available_column(
            block,
            &["_refln.intensity_sigma", "_refln.I_obs_sigma"],
        );

        // Decide: amplitudes take priority over intensities.
        let use_intensities = amp_col.is_none() && int_col.is_some();
        let obs_data_type = if use_intensities {
            ObsDataType::Intensity
        } else {
            ObsDataType::Amplitude
        };

        let (obs_col, obs_sigma_col) = if use_intensities {
            (int_col, int_sigma_col)
        } else {
            (amp_col, amp_sigma_col)
        };

        // ── R-free flags ──
        // Try modern pdbx_r_free_flag (integer) first, then legacy status
        // (char).
        let rfree_int: Option<Vec<_>> = block
            .column("_refln.pdbx_r_free_flag")
            .map(Iterator::collect);
        let status_char: Option<Vec<_>> =
            block.column("_refln.status").map(Iterator::collect);
        let free_flags_from_file = rfree_int.is_some() || status_char.is_some();

        let mut reflections = Vec::with_capacity(cols.nrows());
        for (i, row) in cols.iter().enumerate() {
            // Raw observed value from whichever column we found.
            let raw_obs = obs_col
                .as_ref()
                .and_then(|v| v.get(i))
                .and_then(|v| v.as_f64());
            let raw_sigma = obs_sigma_col
                .as_ref()
                .and_then(|v| v.get(i))
                .and_then(|v| v.as_f64());

            // Convert intensities → amplitudes if needed.
            let (f_meas, sigma_f_meas) = if use_intensities {
                intensity_to_amplitude(raw_obs, raw_sigma)
            } else {
                (raw_obs, raw_sigma)
            };

            let free_flag = extract_free_flag(
                rfree_int.as_deref(),
                status_char.as_deref(),
                i,
            );

            reflections.push(Reflection {
                h: require_i32(row[0], "_refln.index_h", i)?,
                k: require_i32(row[1], "_refln.index_k", i)?,
                l: require_i32(row[2], "_refln.index_l", i)?,
                f_meas,
                sigma_f_meas,
                free_flag,
            });
        }

        Ok(ReflectionData {
            cell,
            spacegroup: extract_spacegroup(block),
            reflections,
            obs_data_type,
            free_flags_from_file,
        })
    }
}

/// Determine the R-free flag for reflection `i` from available columns.
///
/// Prefers the modern `pdbx_r_free_flag` integer column, falling back to the
/// legacy `status` character column, then defaulting to `false` (working set).
fn extract_free_flag(
    rfree_int: Option<&[&Value]>,
    status_char: Option<&[&Value]>,
    i: usize,
) -> bool {
    if let Some(vals) = rfree_int {
        return vals
            .get(i)
            .and_then(|v| v.as_i32())
            .is_some_and(rfree_flag_is_free);
    }
    if let Some(vals) = status_char {
        return vals
            .get(i)
            .and_then(|v| v.as_str())
            .and_then(|s| s.chars().next())
            .is_some_and(status_is_free);
    }
    false
}

/// Convert intensity (I) and its uncertainty (σ_I) to amplitude (F) and σ_F.
///
/// `F = √I` and `σ_F = σ_I / (2√I)` (first-order error propagation).
///
/// Negative or zero intensities yield `None` since F is undefined.
fn intensity_to_amplitude(
    intensity: Option<f64>,
    sigma_i: Option<f64>,
) -> (Option<f64>, Option<f64>) {
    let Some(i_val) = intensity else {
        return (None, None);
    };
    if i_val <= 0.0 {
        return (None, None);
    }
    let f = i_val.sqrt();
    let sigma_f = sigma_i.map(|si| si / (2.0 * f));
    (Some(f), sigma_f)
}

impl From<Block> for CifContent {
    fn from(block: Block) -> Self {
        if let Ok(coords) = CoordinateData::try_from(&block) {
            return CifContent::Coordinates(coords);
        }
        if let Ok(reflns) = ReflectionData::try_from(&block) {
            return CifContent::Reflections(reflns);
        }
        CifContent::Unknown(block)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "extract_tests.rs"]
mod tests;
