//! Python bindings for volumetric scalar-field generation: one free function
//! per pipeline, each taking an explicit grid and returning a numpy array.

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;

use super::{read_coords_nx3, value_err};
use crate::analysis::volumetric::{
    compute_gaussian_field_on_grid, GaussianOptions, GridSpec,
};

/// Read a 1-D array-like of numbers (numpy array or sequence) into a
/// `Vec<f32>`.
fn read_f32_vec(arr: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let n = arr.len()?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(arr.get_item(i)?.extract::<f32>()?);
    }
    Ok(out)
}

/// Rasterize atoms as summed Gaussian blobs onto an explicit voxel grid.
///
/// `positions` is an array-like of shape `(N, 3)` and `radii` an array-like of
/// `N` floats, both in Angstroms; the grid is `dims` voxels from world-space
/// `origin` at per-axis `spacing`. Returns a float32 numpy array of shape
/// `(nx, ny, nz)` in C order. `sigma_scale` sets the Gaussian width as a
/// fraction of each atom's radius, `amplitude` fixes the peak density (`None`
/// uses the atom's radius), and `quadratic_tail` replaces the Gaussian beyond
/// `2 * sigma` with a quadratic reaching zero at `3 * sigma`.
///
/// # Errors
///
/// `PyValueError` for mismatched lengths, nonfinite inputs, a nonpositive
/// radius, width, or spacing, unrepresentable grid geometry, or allocation
/// failure. Also propagates read errors from the two array-likes.
#[pyfunction]
#[pyo3(signature = (
    positions,
    radii,
    dims,
    origin,
    spacing,
    *,
    sigma_scale = 0.7,
    amplitude = None,
    quadratic_tail = false,
))]
#[allow(
    clippy::too_many_arguments,
    reason = "grid geometry and kernel options are flat keyword arguments \
              rather than Python-side wrapper objects"
)]
pub fn gaussian_field(
    py: Python<'_>,
    positions: &Bound<'_, PyAny>,
    radii: &Bound<'_, PyAny>,
    dims: [usize; 3],
    origin: [f32; 3],
    spacing: [f32; 3],
    sigma_scale: f32,
    amplitude: Option<f32>,
    quadratic_tail: bool,
) -> PyResult<Py<PyAny>> {
    let positions = read_coords_nx3(positions)?;
    let radii = read_f32_vec(radii)?;
    let grid = compute_gaussian_field_on_grid(
        &positions,
        &radii,
        GridSpec {
            dims,
            origin,
            spacing,
        },
        GaussianOptions {
            sigma_scale,
            amplitude,
            quadratic_tail,
        },
    )
    .map_err(value_err)?;

    let shape = grid.dims;
    let arr = grid.data.into_pyarray(py).reshape(shape)?;
    Ok(arr.into_any().unbind())
}
