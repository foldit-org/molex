//! L-BFGS B-factor refinement for [`XtalRefinement`].
//!
//! All argmin-facing code lives here. The optimizer is unconstrained, so
//! physical B-factor bounds are imposed by a variable transform rather than a
//! penalty: an unconstrained variable `u` maps to `B = B_MIN + softplus(u)`,
//! which is smooth and keeps `B > B_MIN` for every `u`. The objective handed
//! to argmin is the pure maximum-likelihood target, so the analytic gradient
//! that descends it is reused verbatim (chained through the transform).

use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

use super::refinement::{XtalRefinement, B_MAX, B_MIN};
use crate::element::Element;

/// L-BFGS history length (number of past `(s, y)` pairs kept for the
/// two-loop Hessian approximation).
const LBFGS_MEMORY: usize = 7;

/// Inner-solver iteration cap per macro-cycle. sigma-A is frozen inside a
/// macro-cycle, so a full solve of the transformed problem is cheap; the
/// outer loop revisits it once scaling and sigma-A are refit.
const INNER_ITERS: u64 = 25;

/// Early-stop tolerance on the relative macro-cycle-to-macro-cycle improvement
/// of the maximum-likelihood target.
const CONVERGENCE_TOL: f64 = 1e-4;

/// Floor on `B - B_MIN` when inverting softplus for the initial guess, so a
/// starting B pinned at (or below) `B_MIN` yields a finite `u`.
const SOFTPLUS_FLOOR: f64 = 1e-6;

/// Memoized forward Fc paired with the B-factor set it was computed for:
/// `(b_factors, fc)`.
type CachedFc = (Vec<f64>, Vec<[f32; 2]>);

/// Numerically stable softplus, `ln(1 + e^u)`.
fn softplus(u: f64) -> f64 {
    u.max(0.0) + (-u.abs()).exp().ln_1p()
}

/// Numerically stable logistic sigmoid, `1 / (1 + e^-u)`, the derivative of
/// [`softplus`].
fn sigmoid(u: f64) -> f64 {
    if u >= 0.0 {
        1.0 / (1.0 + (-u).exp())
    } else {
        let z = u.exp();
        z / (1.0 + z)
    }
}

/// Map an unconstrained variable to a physical B-factor.
fn b_from_u(u: f64) -> f64 {
    B_MIN + softplus(u)
}

/// d`B`/d`u` = softplus'(u).
fn db_du(u: f64) -> f64 {
    sigmoid(u)
}

/// Invert the transform for the initial guess: `u = softplus⁻¹(B - B_MIN)`.
///
/// The starting B is clamped to `[B_MIN, B_MAX]` first (the optimizer is
/// unbounded above, but a pathological start should not seed it with a huge
/// `u`). Written as `s + ln(1 - e^-s)` to stay stable for large `s`.
fn u_from_b(b: f64) -> f64 {
    let s = (b.clamp(B_MIN, B_MAX) - B_MIN).max(SOFTPLUS_FLOOR);
    s + (-(-s).exp()).ln_1p()
}

/// Adapts one B-factor refinement problem to argmin's `CostFunction` /
/// `Gradient` traits over the unconstrained variable `u`.
struct BFactorProblem<'a> {
    refinement: &'a XtalRefinement,
    positions: &'a [[f64; 3]],
    elements: &'a [Element],
    occupancies: &'a [f64],
    /// Forward Fc memoized for the most recent B-factor set. MoreThuente
    /// evaluates `cost` and `gradient` at the same trial `u` (hence the same
    /// B), and each otherwise recomputes the full splat -> FFT -> deblur
    /// pipeline; caching Fc lets the co-located pair share one computation.
    fc_cache: std::cell::RefCell<Option<CachedFc>>,
}

impl BFactorProblem<'_> {
    /// Forward Fc for `b`, reused from the cache when `b` matches the cached
    /// B-factor set elementwise and recomputed (and stored) otherwise.
    fn fc_for(&self, b: &[f64]) -> Option<Vec<[f32; 2]>> {
        if let Some((cached_b, cached_fc)) = self.fc_cache.borrow().as_ref() {
            if cached_b == b {
                return Some(cached_fc.clone());
            }
        }
        let fc = self.refinement.forward_fc(
            self.positions,
            self.elements,
            b,
            self.occupancies,
        )?;
        *self.fc_cache.borrow_mut() = Some((b.to_vec(), fc.clone()));
        Some(fc)
    }
}

impl CostFunction for BFactorProblem<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, u: &Self::Param) -> Result<Self::Output, Error> {
        let b: Vec<f64> = u.iter().map(|&ui| b_from_u(ui)).collect();
        let fc = self.fc_for(&b).ok_or_else(|| {
            Error::msg("forward Fc unavailable (form factor / FFT)")
        })?;
        self.refinement
            .maximum_likelihood_target_from_fc(&fc)
            .ok_or_else(|| {
                Error::msg("maximum-likelihood target unavailable (sigma-A)")
            })
    }
}

impl Gradient for BFactorProblem<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, u: &Self::Param) -> Result<Self::Gradient, Error> {
        let b: Vec<f64> = u.iter().map(|&ui| b_from_u(ui)).collect();
        let fc = self.fc_for(&b).ok_or_else(|| {
            Error::msg("forward Fc unavailable (form factor / FFT)")
        })?;
        let dcost_db = self
            .refinement
            .b_factor_gradients_from_fc(
                &fc,
                self.positions,
                self.elements,
                &b,
                self.occupancies,
            )
            .ok_or_else(|| {
                Error::msg("B-factor gradient unavailable (sigma-A)")
            })?;
        // Chain rule: dCost/du_i = (dCost/dB_i) * (dB_i/du_i).
        Ok(dcost_db
            .iter()
            .zip(u.iter())
            .map(|(&g, &ui)| g * db_du(ui))
            .collect())
    }
}

impl XtalRefinement {
    /// Refine per-atom isotropic B-factors by maximum likelihood.
    ///
    /// Each macro-cycle recomputes the map (refitting scaling and sigma-A),
    /// then runs an L-BFGS solve of the pure maximum-likelihood target over the
    /// softplus-transformed variable `u`, so the returned B-factors stay above
    /// `B_MIN`. The outer loop stops early once a macro-cycle improves the
    /// target by less than `CONVERGENCE_TOL` relative to the previous cycle,
    /// or after `n_macro_cycles`.
    ///
    /// `b_factors` is updated in place. Returns `(r_work, r_free)` after the
    /// final cycle, or `None` if the pipeline fails at any point.
    #[allow(clippy::too_many_arguments)]
    pub fn refine(
        &mut self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &mut [f64],
        occupancies: &[f64],
        n_macro_cycles: usize,
    ) -> Option<(f64, f64)> {
        let mut prev_target: Option<f64> = None;

        for _cycle in 0..n_macro_cycles {
            // Refit scaling + sigma-A to the current B-factors; the inner solve
            // treats them as frozen.
            let _map =
                self.compute_map(positions, elements, b_factors, occupancies)?;

            let u0: Vec<f64> = b_factors.iter().map(|&b| u_from_b(b)).collect();

            // The executor borrows `&self` (via the problem) for the whole
            // solve. Scope it and copy the result out before the borrow drops,
            // so the next cycle's `compute_map(&mut self)` is free to run.
            let (best_u, best_target) = {
                let problem = BFactorProblem {
                    refinement: &*self,
                    positions,
                    elements,
                    occupancies,
                    fc_cache: std::cell::RefCell::new(None),
                };
                // A fresh solver each cycle resets the curvature history, which
                // is stale once sigma-A changes.
                let solver =
                    LBFGS::new(MoreThuenteLineSearch::new(), LBFGS_MEMORY);
                let res = Executor::new(problem, solver)
                    .configure(|state| state.param(u0).max_iters(INNER_ITERS))
                    .run()
                    .ok()?;
                let u = res.state().get_best_param()?.clone();
                (u, res.state().get_best_cost())
            };

            for (b, &u) in b_factors.iter_mut().zip(best_u.iter()) {
                *b = b_from_u(u);
            }

            if let Some(prev) = prev_target {
                let rel = (prev - best_target) / prev.abs().max(1.0);
                if rel < CONVERGENCE_TOL {
                    break;
                }
            }
            prev_target = Some(best_target);
        }

        self.r_factors(positions, elements, b_factors, occupancies)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The softplus B-transform round-trips: `b_from_u(u_from_b(b)) == b`
    /// across the physical range.
    #[test]
    fn u_b_transform_round_trips() {
        for &b in &[2.5, 5.0, 20.0, 50.0, 100.0, 250.0] {
            let recovered = b_from_u(u_from_b(b));
            assert!(
                (recovered - b).abs() < 1e-6,
                "round trip b={b}: recovered {recovered}"
            );
        }
    }

    /// `db_du` is the analytic derivative of `b_from_u`: it matches a central
    /// finite difference at every sampled `u`.
    #[test]
    fn db_du_matches_finite_difference() {
        let h = 1e-6;
        for &u in &[-3.0, -1.0, 0.0, 0.5, 2.0, 5.0] {
            let fd = (b_from_u(u + h) - b_from_u(u - h)) / (2.0 * h);
            let analytic = db_du(u);
            assert!(
                (analytic - fd).abs() < 1e-4,
                "u={u}: analytic {analytic} vs fd {fd}"
            );
        }
    }

    /// `sigmoid` is the derivative of `softplus`, bounded to (0, 1).
    #[test]
    fn sigmoid_is_softplus_derivative_and_bounded() {
        let h = 1e-6;
        for &u in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
            let fd = (softplus(u + h) - softplus(u - h)) / (2.0 * h);
            let s = sigmoid(u);
            assert!((s - fd).abs() < 1e-4, "u={u}: sigmoid {s} vs fd {fd}");
            assert!(s > 0.0 && s < 1.0, "sigmoid {s} out of (0,1) at u={u}");
        }
    }
}
