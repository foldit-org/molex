//! Resident experimental crystallographic data.
//!
//! [`ExperimentalData`] parses an sf.cif once into a cached, reusable bundle of
//! unit cell, space group, and reflection set (with its free-flag partition
//! already baked in). A refinement is bound to it with
//! [`XtalRefinement::from_experimental_data`](crate::xtal::XtalRefinement::from_experimental_data),
//! so the parse cost is paid once and the cached reflections drive every
//! subsequent map/refine cycle.

use super::{
    reflections_from_cif, space_group, space_group_number_from_name,
    Reflection, SpaceGroup, UnitCell,
};
use crate::adapters::cif::{parse, ReflectionData};
use crate::element::Element;

/// Minimum Bragg spacing over `reflections`, evaluated against `cell`.
///
/// The minimum of `1/sqrt(d*²)` over reflections with positive `d*²`.
fn min_bragg_spacing(reflections: &[Reflection], cell: &UnitCell) -> f64 {
    let mut d_min = f64::MAX;
    for r in reflections {
        let s2 = cell.d_star_sq(r.h, r.k, r.l);
        if s2 > 0.0 {
            d_min = d_min.min(1.0 / s2.sqrt());
        }
    }
    d_min
}

/// A resident bundle of experimental data: unit cell, space group, and the
/// reflection set with its free-flag partition already resolved.
///
/// Parse once, then reuse across many refine/map cycles.
pub struct ExperimentalData {
    /// The crystallographic unit cell.
    pub unit_cell: UnitCell,
    /// The space group.
    pub space_group: SpaceGroup,
    /// The reflection set, free-flag partition already baked in.
    pub reflections: Vec<Reflection>,
    /// Minimum Bragg spacing over `reflections`, cached at construction.
    d_min: f64,
}

impl ExperimentalData {
    /// Assemble from fully-resolved parts. `reflections` must already carry
    /// their final free-flag partition.
    #[must_use]
    pub fn from_parts(
        reflections: Vec<Reflection>,
        unit_cell: UnitCell,
        space_group: SpaceGroup,
    ) -> Self {
        let d_min = min_bragg_spacing(&reflections, &unit_cell);
        Self {
            unit_cell,
            space_group,
            reflections,
            d_min,
        }
    }

    /// Build from parsed reflection data and an explicit International Tables
    /// space-group number. Uses `data.cell`.
    ///
    /// The free-flag partition comes from [`reflections_from_cif`]: deposited
    /// flags are respected when present, and only seed-derived when the file
    /// carries none. Returns `None` if the reflection set is empty or the space
    /// group is unsupported.
    #[must_use]
    pub fn from_reflection_data(
        data: &ReflectionData,
        sg_number: u16,
        free_fraction: f64,
        seed: u64,
    ) -> Option<Self> {
        let space_group = space_group(sg_number)?;
        let reflections = reflections_from_cif(data, free_fraction, seed);
        if reflections.is_empty() {
            return None;
        }
        let unit_cell = UnitCell::new(
            data.cell.a,
            data.cell.b,
            data.cell.c,
            data.cell.alpha,
            data.cell.beta,
            data.cell.gamma,
        );
        Some(Self::from_parts(reflections, unit_cell, space_group))
    }

    /// Parse an sf.cif string, resolving the space group from the file's own
    /// symmetry tag. Returns `None` if the file omits symmetry, lacks a cell or
    /// reflections, or names an unsupported space group.
    #[must_use]
    pub fn from_sf_cif(
        sf_cif: &str,
        free_fraction: f64,
        seed: u64,
    ) -> Option<Self> {
        let doc = parse(sf_cif).ok()?;
        let block = doc.blocks.first()?;
        let data = ReflectionData::try_from(block).ok()?;
        let sg_number =
            space_group_number_from_name(data.spacegroup.as_deref()?)?;
        Self::from_reflection_data(&data, sg_number, free_fraction, seed)
    }

    /// Parse an sf.cif string with an externally supplied International Tables
    /// space-group number, for the common case where the sf.cif omits symmetry
    /// and the number is resolved from the coordinate CIF. Uses the sf.cif
    /// cell.
    #[must_use]
    pub fn from_sf_cif_with_spacegroup(
        sf_cif: &str,
        sg_number: u16,
        free_fraction: f64,
        seed: u64,
    ) -> Option<Self> {
        let doc = parse(sf_cif).ok()?;
        let block = doc.blocks.first()?;
        let data = ReflectionData::try_from(block).ok()?;
        Self::from_reflection_data(&data, sg_number, free_fraction, seed)
    }

    /// Minimum Bragg spacing (Å) over the reflection set.
    #[must_use]
    pub fn d_min(&self) -> f64 {
        self.d_min
    }

    /// Derive the density-map grid dimensions `[nu, nv, nw]` from the cell and
    /// resolution, at the deposited-data sampling of `d_min / 3`.
    #[must_use]
    pub fn derive_grid(&self) -> [usize; 3] {
        super::derive_grid(
            self.unit_cell.a,
            self.unit_cell.b,
            self.unit_cell.c,
            self.d_min / 3.0,
            self.space_group.number,
        )
    }

    /// The space-group International Tables number, as the `i32` metadata
    /// [`density_from_grid`](crate::xtal::density_from_grid) expects.
    #[must_use]
    pub fn space_group_number(&self) -> i32 {
        i32::from(self.space_group.number)
    }

    /// Lift Cartesian `f32` atom data into fractional `f64` refinement inputs
    /// using this data's cell.
    ///
    /// The slices are parallel; the caller does any hydrogen filtering.
    /// `positions_cart` are Cartesian ångström and are fractionalized through
    /// this data's unit cell. Refined `f64` B-factors write back to `f32` via
    /// `as f32` at the call site.
    #[must_use]
    pub fn refinement_inputs(
        &self,
        positions_cart: &[[f32; 3]],
        elements: &[Element],
        b_factors: &[f32],
        occupancies: &[f32],
    ) -> RefinementInputs {
        let positions = positions_cart
            .iter()
            .map(|p| {
                self.unit_cell.fractionalize([
                    f64::from(p[0]),
                    f64::from(p[1]),
                    f64::from(p[2]),
                ])
            })
            .collect();
        RefinementInputs {
            positions,
            elements: elements.to_vec(),
            b_factors: b_factors.iter().map(|&b| f64::from(b)).collect(),
            occupancies: occupancies.iter().map(|&o| f64::from(o)).collect(),
        }
    }
}

/// Fractional `f64` atom data ready to feed the refinement pipeline, produced
/// by [`ExperimentalData::refinement_inputs`]. The four vectors are parallel.
pub struct RefinementInputs {
    /// Fractional atom positions.
    pub positions: Vec<[f64; 3]>,
    /// Element per atom.
    pub elements: Vec<Element>,
    /// Isotropic B-factor per atom.
    pub b_factors: Vec<f64>,
    /// Site occupancy per atom.
    pub occupancies: Vec<f64>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::xtal::XtalRefinement;

    /// A resident round-trip: parse a fixture sf.cif once, bind a refinement to
    /// it, and confirm the map runs on the cached grid and the deposited free
    /// flags survive the partition.
    #[test]
    fn resident_roundtrip_preserves_grid_and_free_flags() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let sf = std::fs::read_to_string(dir.join("5YPA-sf.cif")).unwrap();

        // 5YPA is P 21 21 21 (International Tables number 19).
        let data =
            ExperimentalData::from_sf_cif_with_spacegroup(&sf, 19, 0.05, 7)
                .expect("resident experimental data for 5YPA");

        // The file carries R-free flags, so some reflections are free.
        assert!(
            data.reflections.iter().any(|r| r.free_flag),
            "5YPA-sf.cif deposits R-free flags"
        );

        // Coordinates for the map come from the paired coordinate CIF.
        let coord = std::fs::read_to_string(dir.join("5YPA.cif")).unwrap();
        let doc = parse(&coord).unwrap();
        let block = doc.blocks.first().unwrap();
        let coord_data =
            crate::adapters::cif::CoordinateData::try_from(block).unwrap();
        let atoms: Vec<_> = coord_data
            .atoms
            .iter()
            .filter(|a| a.element() != Element::H)
            .collect();
        let positions: Vec<[f64; 3]> = atoms
            .iter()
            .map(|a| data.unit_cell.fractionalize([a.x, a.y, a.z]))
            .collect();
        let elements: Vec<Element> =
            atoms.iter().map(|a| a.element()).collect();
        let b_factors: Vec<f64> = atoms.iter().map(|a| a.b_factor).collect();
        let occupancies: Vec<f64> = atoms.iter().map(|a| a.occupancy).collect();

        let expected_grid = data.derive_grid();
        let mut refinement = XtalRefinement::from_experimental_data(&data);
        assert_eq!(refinement.grid_dims, expected_grid);

        let grid = refinement
            .compute_map(&positions, &elements, &b_factors, &occupancies)
            .expect("compute_map on resident experimental data");
        assert_eq!([grid.nu, grid.nv, grid.nw], expected_grid);
    }
}
