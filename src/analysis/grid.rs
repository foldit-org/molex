//! Uniform spatial hash for near-linear neighbor queries.
//!
//! Atom positions are binned into a uniform grid keyed on a caller-chosen cell
//! size. A candidate query visits the 27 cells around a point, so the caller
//! must size the cell so every relevant neighbor lands within that stencil:
//! pick `cell` >= the maximum query reach and any neighbor closer than `cell`
//! is guaranteed to be in one of the 27 cells.

use glam::Vec3;

/// Uniform spatial hash over atom positions for neighbor queries.
pub(crate) struct SpatialGrid {
    cell: f32,
    min: Vec3,
    dims: [usize; 3],
    /// Flattened cell -> atom-index buckets.
    buckets: Vec<Vec<usize>>,
}

impl SpatialGrid {
    /// Bin `positions` into a uniform grid of the given `cell` size. Callers
    /// pass a `cell` at least as large as their query reach so candidates fall
    /// within the 27-cell stencil. Panics on empty `positions`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "span/cell is non-negative and bounded by any realistic \
                  coordinate range, so the floored count fits usize"
    )]
    pub(crate) fn build(positions: &[Vec3], cell: f32) -> Self {
        let mut min = positions[0];
        let mut max = positions[0];
        for &p in positions {
            min = min.min(p);
            max = max.max(p);
        }
        let span = max - min;
        let dims = [
            (span.x / cell).floor() as usize + 1,
            (span.y / cell).floor() as usize + 1,
            (span.z / cell).floor() as usize + 1,
        ];
        let total = dims[0] * dims[1] * dims[2];
        let mut buckets = vec![Vec::new(); total];
        let grid = Self {
            cell,
            min,
            dims,
            buckets: Vec::new(),
        };
        for (idx, &p) in positions.iter().enumerate() {
            let [cx, cy, cz] = grid.cell_coord(p);
            let lin = grid.lin(cx, cy, cz);
            buckets[lin].push(idx);
        }
        Self {
            cell,
            min,
            dims,
            buckets,
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "coordinate offset is clamped to [0, dim) before the cast"
    )]
    fn cell_coord(&self, p: Vec3) -> [usize; 3] {
        let rel = (p - self.min) / self.cell;
        let clamp = |v: f32, dim: usize| -> usize {
            (v.floor().max(0.0) as usize).min(dim - 1)
        };
        [
            clamp(rel.x, self.dims[0]),
            clamp(rel.y, self.dims[1]),
            clamp(rel.z, self.dims[2]),
        ]
    }

    /// Flat bucket index for in-range cell coordinates (each `0 <= c < dim`).
    fn lin(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }

    /// Flat indices of the up-to-27 cells around `p`'s cell, each range
    /// clamped to the grid extent so out-of-bounds neighbors are skipped.
    fn neighbor_cells(&self, p: Vec3) -> Vec<usize> {
        let [cx, cy, cz] = self.cell_coord(p);
        let ranges = [
            clamped_range(cx, self.dims[0]),
            clamped_range(cy, self.dims[1]),
            clamped_range(cz, self.dims[2]),
        ];
        let mut cells = Vec::new();
        for z in ranges[2].clone() {
            for y in ranges[1].clone() {
                for x in ranges[0].clone() {
                    cells.push(self.lin(x, y, z));
                }
            }
        }
        cells
    }

    /// Invoke `f` for every atom index in the 27 cells around `p`.
    pub(crate) fn for_each_candidate<F: FnMut(usize)>(
        &self,
        p: Vec3,
        mut f: F,
    ) {
        for cell in self.neighbor_cells(p) {
            for &idx in &self.buckets[cell] {
                f(idx);
            }
        }
    }
}

/// Inclusive `[c-1, c+1]` cell band clamped to `[0, dim)`, as a `usize` range.
/// `dim` is the cell count along the axis (always >= 1); `c` is in `[0, dim)`.
fn clamped_range(c: usize, dim: usize) -> std::ops::Range<usize> {
    let lo = c.saturating_sub(1);
    let hi = (c + 1).min(dim - 1);
    lo..(hi + 1)
}
