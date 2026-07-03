//! 3D FFT wrapper using `rustfft` for the CPU fallback path.
//!
//! Provides forward and inverse 3D FFTs on real-valued grids stored in
//! row-major order (index = u * nv * nw + v * nw + w).  The implementation
//! performs three batched 1-D FFT passes (w, v, u axes) which is
//! mathematically equivalent to a single 3-D DFT.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during FFT operations.
#[derive(Debug)]
pub enum FftError {
    /// Grid dimensions do not match the length of the supplied data.
    DimensionMismatch {
        /// The expected number of elements (`nu * nv * nw`).
        expected: usize,
        /// The actual number of elements provided.
        got: usize,
    },
}

impl std::fmt::Display for FftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "FFT dimension mismatch: expected {expected} elements, \
                     got {got}"
                )
            }
        }
    }
}

impl std::error::Error for FftError {}

// ---------------------------------------------------------------------------
// Forward 3-D FFT  (real → complex)
// ---------------------------------------------------------------------------

/// Compute the forward (unnormalized) 3-D FFT of a real-valued grid.
///
/// # Arguments
///
/// * `grid` – real-valued input of length `nu * nv * nw`, stored row-major.
/// * `nu`, `nv`, `nw` – grid dimensions along the three axes.
///
/// # Returns
///
/// A `Vec<[f32; 2]>` of the same total length where each element is
/// `[real, imag]`.  The output is **not** normalized (consistent with the
/// `rustfft` convention).
///
/// # Errors
///
/// Returns [`FftError::DimensionMismatch`] when `grid.len() != nu * nv * nw`.
pub fn fft_3d_forward(
    grid: &[f32],
    nu: usize,
    nv: usize,
    nw: usize,
) -> Result<Vec<[f32; 2]>, FftError> {
    let n_total = nu * nv * nw;
    if grid.len() != n_total {
        return Err(FftError::DimensionMismatch {
            expected: n_total,
            got: grid.len(),
        });
    }

    // Working buffer in f64 complex.
    let mut buf: Vec<Complex<f64>> = grid
        .iter()
        .map(|&v| Complex::new(f64::from(v), 0.0))
        .collect();

    let mut planner = FftPlanner::<f64>::new();

    // Pass 1 – along w (innermost, contiguous).
    pass_forward(&mut planner, &mut buf, nu * nv, nw, 1);

    // Pass 2 – along v (stride = nw).
    pass_forward(&mut planner, &mut buf, nu * nw, nv, nw);

    // Pass 3 – along u (stride = nv * nw).
    let stride_u = nv * nw;
    pass_forward(&mut planner, &mut buf, nv * nw, nu, stride_u);

    Ok(complex_to_f32_pairs(&buf))
}

// ---------------------------------------------------------------------------
// Inverse 3-D FFT  (complex → real)
// ---------------------------------------------------------------------------

/// Compute the inverse 3-D FFT, returning a real-valued grid.
///
/// After the three inverse passes the result is divided by `N = nu * nv * nw`
/// so that a forward-then-inverse round-trip recovers the original data.
///
/// # Arguments
///
/// * `data` – complex input of length `nu * nv * nw`, each element `[re, im]`.
/// * `nu`, `nv`, `nw` – grid dimensions.
///
/// # Returns
///
/// A `Vec<f32>` of length `nu * nv * nw` containing the real parts of the
/// normalized inverse transform.
///
/// # Errors
///
/// Returns [`FftError::DimensionMismatch`] when `data.len() != nu * nv * nw`.
pub fn fft_3d_inverse(
    data: &[[f32; 2]],
    nu: usize,
    nv: usize,
    nw: usize,
) -> Result<Vec<f32>, FftError> {
    let n_total = nu * nv * nw;
    if data.len() != n_total {
        return Err(FftError::DimensionMismatch {
            expected: n_total,
            got: data.len(),
        });
    }

    let mut buf: Vec<Complex<f64>> = data
        .iter()
        .map(|pair| Complex::new(f64::from(pair[0]), f64::from(pair[1])))
        .collect();

    let mut planner = FftPlanner::<f64>::new();

    // Inverse passes in reverse axis order (u, v, w) — order does not
    // mathematically matter, but we mirror the forward convention.
    let stride_u = nv * nw;
    pass_inverse(&mut planner, &mut buf, nv * nw, nu, stride_u);

    pass_inverse(&mut planner, &mut buf, nu * nw, nv, nw);

    pass_inverse(&mut planner, &mut buf, nu * nv, nw, 1);

    // Normalize.
    #[allow(clippy::cast_precision_loss)]
    let inv_n = 1.0 / n_total as f64;
    #[allow(clippy::cast_possible_truncation)]
    let result = buf.iter().map(|c| (c.re * inv_n) as f32).collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Execute a batched 1-D forward FFT along one axis.
///
/// * `n_batches` – how many independent 1-D transforms to perform.
/// * `len` – length of each 1-D transform.
/// * `stride` – element stride between consecutive entries of a single 1-D
///   slice (1 for contiguous).
///
/// The caller is responsible for ensuring that all indices accessed are within
/// `buf`.
fn pass_forward(
    planner: &mut FftPlanner<f64>,
    buf: &mut [Complex<f64>],
    n_batches: usize,
    len: usize,
    stride: usize,
) {
    let fft = planner.plan_fft_forward(len);
    let mut scratch = vec![Complex::new(0.0, 0.0); len];

    for batch in 0..n_batches {
        let start = batch_start(batch, stride, len);
        // Gather.
        for i in 0..len {
            scratch[i] = buf[start + i * stride];
        }
        fft.process(&mut scratch);
        // Scatter.
        for i in 0..len {
            buf[start + i * stride] = scratch[i];
        }
    }
}

/// Execute a batched 1-D inverse FFT along one axis.
fn pass_inverse(
    planner: &mut FftPlanner<f64>,
    buf: &mut [Complex<f64>],
    n_batches: usize,
    len: usize,
    stride: usize,
) {
    let fft = planner.plan_fft_inverse(len);
    let mut scratch = vec![Complex::new(0.0, 0.0); len];

    for batch in 0..n_batches {
        let start = batch_start(batch, stride, len);
        for i in 0..len {
            scratch[i] = buf[start + i * stride];
        }
        fft.process(&mut scratch);
        for i in 0..len {
            buf[start + i * stride] = scratch[i];
        }
    }
}

/// Compute the starting linear index for a given batch.
///
/// When `stride == 1` (contiguous), the batches are simply consecutive blocks
/// of `len` elements.  For larger strides the batches interleave, so we must
/// compute the correct origin.
#[must_use]
const fn batch_start(batch: usize, stride: usize, len: usize) -> usize {
    if stride == 1 {
        // Contiguous: each batch is a consecutive run.
        batch * len
    } else {
        // Non-contiguous: the batch index encodes position in the
        // complementary dimensions.  The stride gives the step between
        // consecutive elements of the 1-D slice, and the block size along
        // that axis is `len * stride`.  Within each such block the batches
        // are offset by 1, and across blocks they are offset by
        // `len * stride`.
        let block = stride * len;
        let outer = batch / stride;
        let inner = batch % stride;
        outer * block + inner
    }
}

/// Convert a slice of `Complex<f64>` to a `Vec<[f32; 2]>`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
fn complex_to_f32_pairs(data: &[Complex<f64>]) -> Vec<[f32; 2]> {
    data.iter().map(|c| [c.re as f32, c.im as f32]).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    /// Helper: build a flat grid of zeros with a single 1.0 at the origin.
    fn delta_grid(nu: usize, nv: usize, nw: usize) -> Vec<f32> {
        let mut g = vec![0.0_f32; nu * nv * nw];
        g[0] = 1.0;
        g
    }

    /// Helper: build a constant grid.
    fn constant_grid(nu: usize, nv: usize, nw: usize, val: f32) -> Vec<f32> {
        vec![val; nu * nv * nw]
    }

    /// Round-trip: forward then inverse recovers the original values.
    #[test]
    fn round_trip_4x4x4() {
        let (nu, nv, nw) = (4, 4, 4);
        #[allow(clippy::cast_precision_loss)]
        let grid: Vec<f32> =
            (0..nu * nv * nw).map(|i| i as f32 * 0.1).collect();

        let freq =
            fft_3d_forward(&grid, nu, nv, nw).expect("forward should succeed");
        let recovered =
            fft_3d_inverse(&freq, nu, nv, nw).expect("inverse should succeed");

        assert_eq!(grid.len(), recovered.len());
        for (i, (&orig, &rec)) in grid.iter().zip(recovered.iter()).enumerate()
        {
            assert!(
                (orig - rec).abs() < 1e-4,
                "mismatch at index {i}: orig={orig}, recovered={rec}"
            );
        }
    }

    /// Delta function at the origin should produce flat Fourier magnitudes.
    #[test]
    fn delta_function_4x4x4() {
        let (nu, nv, nw) = (4, 4, 4);
        let grid = delta_grid(nu, nv, nw);

        let freq =
            fft_3d_forward(&grid, nu, nv, nw).expect("forward should succeed");

        for (i, coeff) in freq.iter().enumerate() {
            let mag = coeff[0].hypot(coeff[1]);
            assert!(
                (mag - 1.0).abs() < 1e-4,
                "magnitude at index {i} = {mag}, expected 1.0"
            );
        }
    }

    /// Constant grid: only DC component (index 0) is nonzero.
    #[test]
    fn constant_grid_4x4x4() {
        let (nu, nv, nw) = (4, 4, 4);
        let n_total = nu * nv * nw;
        let grid = constant_grid(nu, nv, nw, 1.0);

        let freq =
            fft_3d_forward(&grid, nu, nv, nw).expect("forward should succeed");

        // DC component should equal N (unnormalized).
        let dc_mag = freq[0][0].hypot(freq[0][1]);
        #[allow(clippy::cast_precision_loss)]
        let expected_dc = n_total as f32;
        assert!(
            (dc_mag - expected_dc).abs() < 1e-3,
            "DC magnitude = {dc_mag}, expected {n_total}"
        );

        // All other components should be ~0.
        for (i, coeff) in freq.iter().enumerate().skip(1) {
            let mag = coeff[0].hypot(coeff[1]);
            assert!(
                mag < 1e-3,
                "non-DC magnitude at index {i} = {mag}, expected ~0"
            );
        }
    }

    /// Dimension mismatch on forward.
    #[test]
    fn dimension_mismatch_forward() {
        let result = fft_3d_forward(&[1.0, 2.0, 3.0], 2, 2, 2);
        assert!(result.is_err());
    }

    /// Dimension mismatch on inverse.
    #[test]
    fn dimension_mismatch_inverse() {
        let result = fft_3d_inverse(&[[1.0, 0.0]; 3], 2, 2, 2);
        assert!(result.is_err());
    }

    /// Non-cubic grid round-trip (8x6x4).
    #[test]
    fn round_trip_8x6x4() {
        let (nu, nv, nw) = (8, 6, 4);
        #[allow(clippy::cast_precision_loss)]
        let grid: Vec<f32> = (0..nu * nv * nw)
            .map(|i| (i as f32 * 0.037).sin())
            .collect();

        let freq =
            fft_3d_forward(&grid, nu, nv, nw).expect("forward should succeed");
        let recovered =
            fft_3d_inverse(&freq, nu, nv, nw).expect("inverse should succeed");

        assert_eq!(grid.len(), recovered.len());
        for (i, (&orig, &rec)) in grid.iter().zip(recovered.iter()).enumerate()
        {
            assert!(
                (orig - rec).abs() < 1e-4,
                "mismatch at index {i}: orig={orig}, recovered={rec}"
            );
        }
    }
}
