//! L-BFGS B-factor refinement for [`XtalRefinement`].
//!
//! All argmin-facing code lives here. The optimizer is unconstrained, so
//! physical B-factor bounds are imposed by a variable transform rather than a
//! penalty: an unconstrained variable `u` maps to `B = B_MIN + softplus(u)`,
//! which is smooth and keeps `B > B_MIN` for every `u`. The objective handed
//! to argmin is the pure maximum-likelihood target, so the analytic gradient
//! that descends it is reused verbatim (chained through the transform).

use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{CostFunction, Error, Executor, Gradient, State, KV};
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

/// argmin observer that forwards per-inner-iteration progress out of an L-BFGS
/// solve. One is attached per macro-cycle carrying that cycle's 1-based index;
/// argmin invokes `observe_iter` after each inner iteration until the solver
/// converges or reaches `INNER_ITERS`.
struct ProgressObserver<F> {
    cycle: usize,
    /// `Arc<Mutex>` so the fresh per-cycle observers share one user `FnMut`
    /// (argmin takes each observer by value).
    cb: std::sync::Arc<std::sync::Mutex<F>>,
}

impl<F, I> Observe<I> for ProgressObserver<F>
where
    F: FnMut(usize, usize, usize),
    I: State,
{
    #[allow(
        clippy::cast_possible_truncation,
        reason = "argmin iteration counters are bounded by INNER_ITERS; the \
                  u64 -> usize casts cannot truncate a value this small"
    )]
    fn observe_iter(&mut self, state: &I, _kv: &KV) -> Result<(), Error> {
        {
            let mut cb = self
                .cb
                .lock()
                .map_err(|_| Error::msg("progress callback mutex poisoned"))?;
            cb(
                self.cycle,
                state.get_iter() as usize + 1,
                state.get_max_iters() as usize,
            );
        }
        Ok(())
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
    ///
    /// `progress` is invoked once per inner L-BFGS iteration with
    /// `(macro_cycle, inner_iter, inner_total)`: `macro_cycle` and `inner_iter`
    /// are 1-based, `inner_total` is the per-cycle iteration cap. A caller can
    /// fill a determinate bar within each macro-cycle and reset it when
    /// `macro_cycle` advances. L-BFGS often converges before `inner_total`, in
    /// which case that cycle's ticks simply stop early.
    #[allow(clippy::too_many_arguments)]
    pub fn refine(
        &mut self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &mut [f64],
        occupancies: &[f64],
        n_macro_cycles: usize,
        progress: impl FnMut(usize, usize, usize) + 'static,
    ) -> Option<(f64, f64)> {
        let progress = std::sync::Arc::new(std::sync::Mutex::new(progress));
        let mut prev_target: Option<f64> = None;

        for cycle in 0..n_macro_cycles {
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
                    .add_observer(
                        ProgressObserver {
                            cycle: cycle + 1,
                            cb: std::sync::Arc::clone(&progress),
                        },
                        ObserverMode::Always,
                    )
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

    /// The progress callback fires once per inner L-BFGS iteration, carrying
    /// `(macro_cycle, inner_iter, inner_total)`: macro-cycle indices stay
    /// within `1..=n_macro_cycles`, inner-iteration indices start at 1 and
    /// increase within a cycle, and `inner_total` reports the `INNER_ITERS`
    /// cap.
    ///
    /// Self-consistent synthetic data converges in ~2 cycles, which would cut
    /// the run short of the requested count. To keep multiple cycles running,
    /// the observed amplitudes get a resolution-dependent B-offset so the model
    /// must inflate every B-factor to match; the sigma-A refit lags that
    /// motion, so each macro-cycle keeps improving past `CONVERGENCE_TOL`.
    /// The cell and grid stay small so the whole refine is
    /// millisecond-class.
    #[test]
    #[allow(clippy::expect_used, clippy::cast_possible_truncation)]
    fn progress_callback_fires_per_inner_iter() {
        let case = crate::testutil::synthetic_refinement(
            32,
            1,
            [20.0, 20.0, 20.0, 90.0, 90.0, 90.0],
            1.2,
            20.0,
        );
        let mut refinement = case.refinement;

        // Damp F_obs as if the data carried an extra +80 A^2 of isotropic B,
        // pulling the best-fit model well above the b_true start.
        const DATA_B_OFFSET: f64 = 80.0;
        let unit_cell = refinement.unit_cell.clone();
        for r in &mut refinement.reflections {
            let s2 = unit_cell.d_star_sq(r.h, r.k, r.l);
            r.f_obs *= (-DATA_B_OFFSET * s2 / 4.0).exp() as f32;
        }

        let mut b = vec![20.0; case.positions.len()];

        const N_MACRO_CYCLES: usize = 3;
        let ticks = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let ticks_cb = std::sync::Arc::clone(&ticks);
        let _ = refinement
            .refine(
                &case.positions,
                &case.elements,
                &mut b,
                &case.occupancies,
                N_MACRO_CYCLES,
                move |macro_cycle, inner_iter, inner_total| {
                    ticks_cb.lock().expect("ticks mutex").push((
                        macro_cycle,
                        inner_iter,
                        inner_total,
                    ));
                },
            )
            .expect("synthetic refine");

        let ticks = ticks.lock().expect("ticks mutex");
        assert!(!ticks.is_empty(), "at least one inner-iteration tick fired");

        // Group inner ticks by macro-cycle, checking cycle range, inner_total,
        // and that inner_iter starts at 1 and strictly increases within a
        // cycle.
        let mut cycle_of_prev = 0usize;
        let mut prev_inner = 0usize;
        for &(macro_cycle, inner_iter, inner_total) in ticks.iter() {
            assert!(
                (1..=N_MACRO_CYCLES).contains(&macro_cycle),
                "macro_cycle {macro_cycle} out of 1..={N_MACRO_CYCLES}"
            );
            assert_eq!(inner_total, INNER_ITERS as usize, "inner_total is cap");
            if macro_cycle == cycle_of_prev {
                assert_eq!(
                    inner_iter,
                    prev_inner + 1,
                    "inner_iter increments by 1 within a macro-cycle"
                );
            } else {
                assert_eq!(
                    inner_iter, 1,
                    "inner_iter restarts at 1 for a new macro-cycle"
                );
            }
            cycle_of_prev = macro_cycle;
            prev_inner = inner_iter;
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
