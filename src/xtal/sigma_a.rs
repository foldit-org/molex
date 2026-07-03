//! Binned sigma-A estimation for ML map coefficient weighting.
//!
//! Implements the Read (1986) sigma-A scheme: observed and calculated
//! amplitudes are binned by resolution, and per-bin D (figure of merit proxy)
//! and sigma² (residual variance) values are estimated, smoothed, and
//! interpolated for each reflection.

use super::types::{Reflection, UnitCell};

/// Number of resolution bins.
const NUM_BINS: usize = 20;

/// Minimum number of reflections for a bin to be considered populated.
const MIN_BIN_POP: usize = 3;

/// Floor value for sigma-squared estimates.
const SIGMA_SQ_FLOOR: f64 = 1e-6;

/// Result of sigma-A estimation, containing per-bin D and sigma² values
/// together with the resolution binning parameters needed for interpolation.
#[derive(Debug, Clone)]
pub struct SigmaAResult {
    /// D values per resolution bin (20 bins).
    pub d_bins: [f64; NUM_BINS],
    /// σ² values per resolution bin (20 bins).
    pub sigma_sq_bins: [f64; NUM_BINS],
    /// Minimum s² (1/d²) value.
    pub s2_min: f64,
    /// Width of each resolution bin in s² units.
    pub bin_width: f64,
}

impl SigmaAResult {
    /// Linearly interpolate D and sigma² for a given s² value.
    ///
    /// Returns `(D, sigma_sq)` by interpolating between bin centers.
    /// Values outside the binned range are clamped to the nearest edge bin.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn interpolate(&self, s2: f64) -> (f64, f64) {
        // Bin centers are at s2_min + (i + 0.5) * bin_width
        // Fractional bin position relative to first center
        let frac = (s2 - self.s2_min) / self.bin_width - 0.5;

        if frac <= 0.0 {
            return (self.d_bins[0], self.sigma_sq_bins[0]);
        }

        let last = NUM_BINS - 1;

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let lo = frac as usize;

        if lo >= last {
            return (self.d_bins[last], self.sigma_sq_bins[last]);
        }

        let hi = lo + 1;
        let t = frac - lo as f64;

        let d = self.d_bins[lo].mul_add(1.0 - t, self.d_bins[hi] * t);
        let s =
            self.sigma_sq_bins[lo].mul_add(1.0 - t, self.sigma_sq_bins[hi] * t);

        (d, s)
    }
}

/// Compute the bin index for a given s² value, clamped to `[0, NUM_BINS - 1]`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
fn bin_index(s2: f64, s2_min: f64, bin_width: f64) -> usize {
    let idx = ((s2 - s2_min) / bin_width).floor() as isize;
    if idx < 0 {
        0
    } else if idx >= NUM_BINS as isize {
        NUM_BINS - 1
    } else {
        idx as usize
    }
}

/// Apply a 3-point moving average with 2:1 edge weighting.
fn smooth_bins(vals: &mut [f64; NUM_BINS]) {
    let mut smoothed = [0.0; NUM_BINS];
    smoothed[0] = 2.0f64.mul_add(vals[0], vals[1]) / 3.0;
    for i in 1..(NUM_BINS - 1) {
        smoothed[i] = (vals[i - 1] + vals[i] + vals[i + 1]) / 3.0;
    }
    smoothed[NUM_BINS - 1] =
        2.0f64.mul_add(vals[NUM_BINS - 1], vals[NUM_BINS - 2]) / 3.0;
    *vals = smoothed;
}

/// Fill any remaining zero-valued bins by linear interpolation from populated
/// neighbors.
#[allow(clippy::cast_precision_loss)]
fn interpolate_empty(vals: &mut [f64; NUM_BINS], populated: &[bool; NUM_BINS]) {
    // Find first populated bin
    let first_pop = populated.iter().position(|&p| p);
    let last_pop = populated.iter().rposition(|&p| p);

    let (Some(first_pop), Some(last_pop)) = (first_pop, last_pop) else {
        return; // no populated bins at all
    };

    // Fill before first populated bin
    for i in 0..first_pop {
        vals[i] = vals[first_pop];
    }
    // Fill after last populated bin
    for i in (last_pop + 1)..NUM_BINS {
        vals[i] = vals[last_pop];
    }
    // Interpolate gaps between populated bins
    let mut left = first_pop;
    for right in (first_pop + 1)..=last_pop {
        if populated[right] {
            if right - left > 1 {
                let span = (right - left) as f64;
                for j in (left + 1)..right {
                    let t = (j - left) as f64 / span;
                    vals[j] = vals[left].mul_add(1.0 - t, vals[right] * t);
                }
            }
            left = right;
        }
    }
}

/// Estimate sigma-A parameters (D and σ²) from observed amplitudes, calculated
/// amplitudes, and the unit cell.
///
/// The algorithm bins reflections by resolution, computes per-bin D values
/// (correlation between Fo and Fc), smooths them with a moving average,
/// enforces monotonic decrease with resolution, and estimates residual variance
/// σ².
///
/// Only working-set reflections (those with `free_flag == false`) are used for
/// estimation; free-set reflections are reserved for cross-validation.
#[must_use]
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::excessive_nesting
)]
pub fn estimate_sigma_a(
    reflections: &[Reflection],
    fc_amplitudes: &[f64],
    unit_cell: &UnitCell,
) -> SigmaAResult {
    // Step 1: Find resolution range
    let mut s2_min = f64::MAX;
    let mut s2_max = f64::MIN;

    let s2_values: Vec<f64> = reflections
        .iter()
        .map(|r| unit_cell.d_star_sq(r.h, r.k, r.l))
        .collect();

    for &s2 in &s2_values {
        if s2 < s2_min {
            s2_min = s2;
        }
        if s2 > s2_max {
            s2_max = s2;
        }
    }

    // Guard against degenerate cases
    let range = s2_max - s2_min;
    let bin_width = if range < 1e-12 {
        1.0
    } else {
        range / NUM_BINS as f64
    };

    // Step 2: Bin using working set only
    let mut sum_fo_fc = [0.0_f64; NUM_BINS];
    let mut sum_fc2 = [0.0_f64; NUM_BINS];
    let mut count = [0_usize; NUM_BINS];

    for (i, refl) in reflections.iter().enumerate() {
        if refl.free_flag {
            continue;
        }
        let b = bin_index(s2_values[i], s2_min, bin_width);
        // Use absolute value: negative Fobs (data error) must not corrupt
        // the Fo·Fc correlation that estimates D.
        let fo = f64::from(refl.f_obs).abs();
        let fc = fc_amplitudes[i];
        sum_fo_fc[b] += fo * fc;
        sum_fc2[b] += fc * fc;
        count[b] += 1;
    }

    // Step 3: Merge underpopulated bins into nearest populated neighbor
    for b in 0..NUM_BINS {
        if count[b] > 0 && count[b] < MIN_BIN_POP {
            // Find nearest populated neighbor (check left first, then right)
            let mut target = None;

            // Search left
            for j in (0..b).rev() {
                if count[j] >= MIN_BIN_POP {
                    target = Some(j);
                    break;
                }
            }
            // Search right if no left neighbor found
            if target.is_none() {
                for (j, &cnt) in count.iter().enumerate().skip(b + 1) {
                    if cnt >= MIN_BIN_POP {
                        target = Some(j);
                        break;
                    }
                }
            }

            if let Some(t) = target {
                sum_fo_fc[t] += sum_fo_fc[b];
                sum_fc2[t] += sum_fc2[b];
                count[t] += count[b];
                sum_fo_fc[b] = 0.0;
                sum_fc2[b] = 0.0;
                count[b] = 0;
            }
        }
    }

    // Step 4: Compute D
    let mut d_bins = [0.0_f64; NUM_BINS];
    let mut populated = [false; NUM_BINS];

    for b in 0..NUM_BINS {
        if count[b] > 0 && sum_fc2[b] > 0.0 {
            d_bins[b] = (sum_fo_fc[b] / sum_fc2[b]).clamp(0.0, 1.0);
            populated[b] = true;
        }
    }

    // Interpolate empty bins from neighbors
    interpolate_empty(&mut d_bins, &populated);

    // Step 5: Smooth with 3-point moving average
    smooth_bins(&mut d_bins);

    // Step 6: Enforce monotonicity (D should decrease with resolution)
    for i in 1..NUM_BINS {
        if d_bins[i] > d_bins[i - 1] {
            d_bins[i] = d_bins[i - 1];
        }
    }

    // Step 7: Compute sigma²
    let mut sigma_sq_bins = [0.0_f64; NUM_BINS];
    let mut sigma_count = [0_usize; NUM_BINS];

    for (i, refl) in reflections.iter().enumerate() {
        if refl.free_flag {
            continue;
        }
        let b = bin_index(s2_values[i], s2_min, bin_width);
        let fo = f64::from(refl.f_obs).abs();
        let fc = fc_amplitudes[i];
        let residual = d_bins[b].mul_add(-fc, fo);
        sigma_sq_bins[b] += residual * residual;
        sigma_count[b] += 1;
    }

    let mut sigma_populated = [false; NUM_BINS];
    for b in 0..NUM_BINS {
        if sigma_count[b] > 0 {
            sigma_sq_bins[b] /= sigma_count[b] as f64;
            if sigma_sq_bins[b] < SIGMA_SQ_FLOOR {
                sigma_sq_bins[b] = SIGMA_SQ_FLOOR;
            }
            sigma_populated[b] = true;
        } else {
            sigma_sq_bins[b] = SIGMA_SQ_FLOOR;
            // leave sigma_populated false so interpolation fills it
        }
    }

    // Step 8: Interpolate empty sigma² bins and smooth
    interpolate_empty(&mut sigma_sq_bins, &sigma_populated);
    smooth_bins(&mut sigma_sq_bins);

    // Floor again after smoothing
    for s in &mut sigma_sq_bins {
        if *s < SIGMA_SQ_FLOOR {
            *s = SIGMA_SQ_FLOOR;
        }
    }

    SigmaAResult {
        d_bins,
        sigma_sq_bins,
        s2_min,
        bin_width,
    }
}

/// Compute the free-set R factor.
///
/// R-free = Σ|Fobs − k·Fc| / Σ|Fobs| over reflections with `free_flag == true`.
/// Returns 0.0 if there are no free-set reflections.
#[must_use]
pub fn r_free(
    reflections: &[Reflection],
    fc_amplitudes: &[f64],
    k_overall: f64,
) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;

    for (i, refl) in reflections.iter().enumerate() {
        if !refl.free_flag {
            continue;
        }
        let fo = f64::from(refl.f_obs).abs();
        let fc = fc_amplitudes[i];
        num += k_overall.mul_add(-fc, fo).abs();
        den += fo;
    }

    if den < f64::EPSILON {
        0.0
    } else {
        num / den
    }
}

#[cfg(test)]
#[allow(
    clippy::excessive_nesting,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops
)]
mod tests {
    use super::*;

    /// Build a simple orthorhombic unit cell for testing.
    fn test_cell() -> UnitCell {
        UnitCell::new(50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
    }

    /// Generate a grid of reflections spanning a range of Miller indices.
    /// Every 10th reflection is flagged as free.
    fn make_reflections(cell: &UnitCell) -> Vec<Reflection> {
        let mut refls = Vec::new();
        let mut idx = 0_usize;
        for h in -5..=5_i32 {
            for k in -5..=5_i32 {
                for l in 1..=10_i32 {
                    // Skip (0,0,0)
                    if h == 0 && k == 0 && l == 0 {
                        continue;
                    }
                    let _ = cell; // used by caller
                    refls.push(Reflection {
                        h,
                        k,
                        l,
                        f_obs: 100.0 / (1.0 + (h * h + k * k + l * l) as f32),
                        sigma_f: 1.0,
                        free_flag: idx.is_multiple_of(10),
                    });
                    idx += 1;
                }
            }
        }
        refls
    }

    #[test]
    fn perfect_model_d_near_one() {
        let cell = test_cell();
        let refls = make_reflections(&cell);
        let fc: Vec<f64> = refls.iter().map(|r| f64::from(r.f_obs)).collect();

        let result = estimate_sigma_a(&refls, &fc, &cell);

        for (i, &d) in result.d_bins.iter().enumerate() {
            assert!(d > 0.85, "bin {i}: D = {d}, expected near 1.0");
        }
    }

    #[test]
    fn random_fc_low_d() {
        let cell = test_cell();
        let refls = make_reflections(&cell);
        // Use Fc values that are unrelated to Fo
        let fc: Vec<f64> = refls
            .iter()
            .enumerate()
            .map(|(i, _)| 10.0 + (i as f64 * 7.3).sin() * 5.0)
            .collect();

        let result = estimate_sigma_a(&refls, &fc, &cell);

        let low_count = result.d_bins.iter().filter(|&&d| d < 0.5).count();
        assert!(
            low_count > NUM_BINS / 2,
            "expected most bins to have D < 0.5, got {low_count}/{NUM_BINS}"
        );
    }

    #[test]
    fn monotonicity_enforced() {
        let cell = test_cell();
        let refls = make_reflections(&cell);
        let fc: Vec<f64> =
            refls.iter().map(|r| f64::from(r.f_obs) * 0.9).collect();

        let result = estimate_sigma_a(&refls, &fc, &cell);

        for i in 1..NUM_BINS {
            assert!(
                result.d_bins[i] <= result.d_bins[i - 1] + 1e-12,
                "monotonicity violated at bin {i}: D[{i}]={} > D[{}]={}",
                result.d_bins[i],
                i - 1,
                result.d_bins[i - 1]
            );
        }
    }

    #[test]
    fn sigma_sq_always_positive() {
        let cell = test_cell();
        let refls = make_reflections(&cell);
        let fc: Vec<f64> =
            refls.iter().map(|r| f64::from(r.f_obs) * 1.1).collect();

        let result = estimate_sigma_a(&refls, &fc, &cell);

        for (i, &s) in result.sigma_sq_bins.iter().enumerate() {
            assert!(
                s >= SIGMA_SQ_FLOOR,
                "bin {i}: sigma_sq = {s}, expected >= {SIGMA_SQ_FLOOR}"
            );
        }
    }

    #[test]
    fn r_free_perfect_model() {
        let cell = test_cell();
        let refls = make_reflections(&cell);
        let fc: Vec<f64> = refls.iter().map(|r| f64::from(r.f_obs)).collect();

        let rf = r_free(&refls, &fc, 1.0);
        assert!(rf < 1e-6, "R-free of perfect model should be ~0, got {rf}");
    }

    #[test]
    fn r_free_no_free_reflections() {
        let refls = vec![Reflection {
            h: 1,
            k: 0,
            l: 0,
            f_obs: 10.0,
            sigma_f: 1.0,
            free_flag: false,
        }];
        let fc = vec![10.0];
        let rf = r_free(&refls, &fc, 1.0);
        assert!((rf - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn interpolation_between_bins() {
        let cell = test_cell();
        let refls = make_reflections(&cell);
        let fc: Vec<f64> =
            refls.iter().map(|r| f64::from(r.f_obs) * 0.95).collect();

        let result = estimate_sigma_a(&refls, &fc, &cell);

        // Pick s² values at first and second bin centers
        let center0 = result.s2_min + 0.5 * result.bin_width;
        let center1 = result.s2_min + 1.5 * result.bin_width;
        let mid = f64::midpoint(center0, center1);

        let (d0, s0) = result.interpolate(center0);
        let (d1, s1) = result.interpolate(center1);
        let (dm, sm) = result.interpolate(mid);

        // Interpolated value should be between neighbors (or equal if they
        // match)
        let d_lo = d0.min(d1);
        let d_hi = d0.max(d1);
        assert!(
            dm >= d_lo - 1e-12 && dm <= d_hi + 1e-12,
            "D interpolation: {dm} not between {d_lo} and {d_hi}"
        );

        let s_lo = s0.min(s1);
        let s_hi = s0.max(s1);
        assert!(
            sm >= s_lo - 1e-12 && sm <= s_hi + 1e-12,
            "sigma_sq interpolation: {sm} not between {s_lo} and {s_hi}"
        );
    }

    #[test]
    fn negative_fobs_does_not_corrupt_d() {
        let cell = test_cell();
        let mut refls = make_reflections(&cell);
        // Inject a few negative Fobs values (data errors).
        refls[0].f_obs = -50.0;
        refls[5].f_obs = -20.0;
        let fc: Vec<f64> =
            refls.iter().map(|r| f64::from(r.f_obs).abs()).collect();

        let result = estimate_sigma_a(&refls, &fc, &cell);

        // D values should still be valid (0..1 range) and not NaN.
        for (i, &d) in result.d_bins.iter().enumerate() {
            assert!(d.is_finite(), "bin {i}: D is not finite");
            assert!(d >= 0.0, "bin {i}: D = {d} is negative");
            assert!(d <= 1.0, "bin {i}: D = {d} exceeds 1.0");
        }
    }

    #[test]
    fn interpolation_clamps_at_edges() {
        let result = SigmaAResult {
            d_bins: [0.9; NUM_BINS],
            sigma_sq_bins: [0.01; NUM_BINS],
            s2_min: 0.001,
            bin_width: 0.01,
        };

        // Far below range
        let (d_lo, s_lo) = result.interpolate(-1.0);
        assert!((d_lo - 0.9).abs() < 1e-12);
        assert!((s_lo - 0.01).abs() < 1e-12);

        // Far above range
        let (d_hi, s_hi) = result.interpolate(100.0);
        assert!((d_hi - 0.9).abs() < 1e-12);
        assert!((s_hi - 0.01).abs() < 1e-12);
    }
}
