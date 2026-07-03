use super::*;

/// A trivial 1-parameter LM test: minimise `(x - 3)²`.
///
/// We reuse `solve_spd` and verify convergence of a manual LM loop.
/// Convention matches `fit_scaling`: Jacobian is `d(model)/d(param)`,
/// residual `r = obs - model`, and the normal equation is
/// `(J^T J + lambda * diag) delta = J^T r`.
#[test]
fn lm_quadratic_converges() {
    // Model: f(x) = x. Observation: 3.  Residual: r = 3 - x.
    // Jacobian of model w.r.t. param: df/dx = 1.
    let mut x = 0.0_f64;
    let mut lambda = 1e-3_f64;
    let mut prev_cost = (x - 3.0) * (x - 3.0);

    for _ in 0..50 {
        let r = 3.0 - x;
        let j = 1.0_f64; // d(model)/d(param)
        let jtj = j * j;
        let jtr_val = j * r;

        let a = [jtj + lambda * jtj.max(1e-12)];
        let b = [jtr_val];

        if let Some(delta) = solve_spd(&a, &b, 1) {
            let trial = x + delta[0];
            let trial_cost = (trial - 3.0) * (trial - 3.0);
            if trial_cost < prev_cost {
                x = trial;
                prev_cost = trial_cost;
                lambda *= 0.1;
            } else {
                lambda *= 10.0;
            }
        }
    }

    assert!((x - 3.0).abs() < 1e-6, "expected x ≈ 3.0, got {x}");
}

/// Wilson estimate on synthetic data with known k=1.5, B=20.
#[test]
fn wilson_estimate_recovers_known() {
    let cell = UnitCell::new(50.0, 60.0, 70.0, 90.0, 90.0, 90.0);

    let k_true = 1.5_f64;
    let b_true = 20.0_f64;

    let mut reflections = Vec::new();
    let mut fc_amps = Vec::new();

    // Generate synthetic reflections over a range of resolution.
    for h in 1..=8 {
        for k in 0..=5 {
            for l in 0..=5 {
                let dss = cell.d_star_sq(h, k, l);
                let stol2 = dss * 0.25;
                // fc_amp = 100 * exp(-2 * stol2) (some smooth fall-off)
                let fc_amp = 100.0 * (-2.0 * stol2).exp();
                if fc_amp < 1.0 {
                    continue;
                }
                // f_obs = k_true * exp(-b_true * stol2) * fc_amp
                let f_obs = k_true * (-b_true * stol2).exp() * fc_amp;
                if f_obs < 1.0 {
                    continue;
                }
                #[allow(clippy::cast_possible_truncation)]
                let fo_f32 = f_obs as f32;
                reflections.push(Reflection {
                    h,
                    k,
                    l,
                    f_obs: fo_f32,
                    sigma_f: 1.0,
                    free_flag: false,
                });
                fc_amps.push(fc_amp);
            }
        }
    }

    let (k_est, b_est) = wilson_estimate(&reflections, &fc_amps, &cell);

    assert!(
        (k_est - k_true).abs() < 0.3,
        "k_est={k_est}, expected ~{k_true}"
    );
    assert!(
        (b_est - b_true).abs() < 5.0,
        "b_est={b_est}, expected ~{b_true}"
    );
}

/// Orthorhombic system has exactly 3 basis vectors.
#[test]
fn constraint_basis_orthorhombic() {
    let basis = aniso_constraint_basis(&CrystalSystem::Orthorhombic);
    assert_eq!(basis.len(), 3);
}

/// Cubic system has exactly 1 basis vector.
#[test]
fn constraint_basis_cubic() {
    let basis = aniso_constraint_basis(&CrystalSystem::Cubic);
    assert_eq!(basis.len(), 1);
    assert!((basis[0][0] - 1.0).abs() < 1e-12);
    assert!((basis[0][1] - 1.0).abs() < 1e-12);
    assert!((basis[0][2] - 1.0).abs() < 1e-12);
}

/// Verify that every crystal system returns the expected number of basis
/// vectors.
#[test]
fn constraint_basis_dimensions() {
    assert_eq!(aniso_constraint_basis(&CrystalSystem::Triclinic).len(), 6);
    assert_eq!(aniso_constraint_basis(&CrystalSystem::Monoclinic).len(), 4);
    assert_eq!(
        aniso_constraint_basis(&CrystalSystem::Orthorhombic).len(),
        3
    );
    assert_eq!(aniso_constraint_basis(&CrystalSystem::Tetragonal).len(), 2);
    assert_eq!(aniso_constraint_basis(&CrystalSystem::Trigonal).len(), 2);
    assert_eq!(aniso_constraint_basis(&CrystalSystem::Hexagonal).len(), 2);
    assert_eq!(aniso_constraint_basis(&CrystalSystem::Cubic).len(), 1);
}

/// Cholesky solver on a trivial 2×2 SPD system.
#[test]
fn solve_spd_2x2() {
    // [4 2; 2 3] x = [8; 7] → x = [1; 1]... nope.
    // 4*1 + 2*1 = 6 ≠ 8 — let's just use a correct RHS.
    // [4 2; 2 3] x = [1; 2] → x = [−1/8; 6/8] = [−0.125; 0.75]
    let a = [4.0, 2.0, 2.0, 3.0];
    let b = [1.0, 2.0];
    let x = solve_spd(&a, &b, 2);
    assert!(x.is_some());
    if let Some(x) = x {
        let expected_0 = -0.125;
        let expected_1 = 0.75;
        assert!(
            (x[0] - expected_0).abs() < 1e-10,
            "x[0]={}, expected {}",
            x[0],
            expected_0
        );
        assert!(
            (x[1] - expected_1).abs() < 1e-10,
            "x[1]={}, expected {}",
            x[1],
            expected_1
        );
    }
}

/// `fit_scaling` runs without panicking on a small synthetic dataset.
#[test]
fn fit_scaling_smoke() {
    let cell = UnitCell::new(50.0, 50.0, 80.0, 90.0, 90.0, 90.0);
    let system = CrystalSystem::Tetragonal;

    let mut reflections = Vec::new();
    let mut fc_vals = Vec::new();
    let mut fm_vals = Vec::new();

    for h in -3..=3_i32 {
        for k in -3..=3_i32 {
            for l in 0..=5_i32 {
                if h == 0 && k == 0 && l == 0 {
                    continue;
                }
                let dss = cell.d_star_sq(h, k, l);
                let stol2 = dss * 0.25;
                let fc_amp = 50.0 * (-3.0 * stol2).exp();
                // Simulate observed as scaled Fc.
                let f_obs = 1.2 * (-5.0 * stol2).exp() * fc_amp;
                #[allow(clippy::cast_possible_truncation)]
                let fo = f_obs as f32;
                if fo < 0.5 {
                    continue;
                }
                reflections.push(Reflection {
                    h,
                    k,
                    l,
                    f_obs: fo,
                    sigma_f: fo.max(1.0) * 0.05,
                    free_flag: false,
                });
                #[allow(clippy::cast_possible_truncation)]
                let fc_f32 = fc_amp as f32;
                fc_vals.push([fc_f32, 0.0_f32]);
                fm_vals.push([fc_f32 * 0.05, 0.0_f32]);
            }
        }
    }

    let result = fit_scaling(&reflections, &fc_vals, &fm_vals, &cell, &system);

    // Just verify it produced reasonable numbers.
    assert!(result.k_overall > 0.0, "k_overall should be positive");
    assert!(result.b_sol > 0.0, "b_sol should be positive");
}
