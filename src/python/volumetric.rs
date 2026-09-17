//! Python bindings for volumetric scalar-field generation.

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;

use super::{read_coords_nx3, value_err};
use crate::analysis::volumetric::{
    compute_gaussian_field_on_grid, GaussianOptions, GridSpec,
};

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
/// Returns a float32 array of shape `(nx, ny, nz)` in C order.
///
/// # Errors
/// `PyValueError` for unreadable atom arrays or unrepresentable grid geometry.
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
#[allow(clippy::too_many_arguments)]
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
