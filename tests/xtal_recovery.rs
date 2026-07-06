//! Real-structure B-factor recovery validation.
//!
//! Each case flattens every deposited B-factor to their common mean, refits and
//! refines from that degraded start, and asserts that refinement both lowers
//! R-work and recovers a positive correlation with the deposited B-factors.
//!
//! Even the cheapest real structure takes several seconds, so every case is
//! `#[ignore]`d; run them with `cargo test --features xtal,testutil --
//! --ignored`.

#![cfg(all(feature = "xtal", feature = "testutil"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::suboptimal_flops,
    clippy::cast_precision_loss
)]

use molex::testutil::refinement_from_cif_pair;

/// Pearson correlation between two equal-length samples.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-30 {
        0.0
    } else {
        cov / denom
    }
}

/// Flatten all B-factors to their mean, refine up from that start, and assert
/// recovery: R-work drops and the refined B-factors correlate with the
/// deposited ones above `min_b_corr`.
fn assert_recovers(pdb: &str, min_b_corr: f64) {
    let case = refinement_from_cif_pair(pdb)
        .unwrap_or_else(|| panic!("load {pdb} from tests/data"));
    let mut refinement = case.refinement;
    let positions = case.positions;
    let elements = case.elements;
    let deposited_b = case.b_factors;
    let occupancies = case.occupancies;

    // Sanity gate: the deposited model must itself fit, or the probe is moot.
    let _ = refinement
        .compute_map(&positions, &elements, &deposited_b, &occupancies)
        .expect("compute_map (deposited)");
    let (r_work_dep, r_free_dep) = refinement
        .r_factors(&positions, &elements, &deposited_b, &occupancies)
        .expect("r_factors (deposited)");
    assert!(
        r_work_dep < 0.60 && r_free_dep < 0.60,
        "{pdb}: deposited model fails the fit gate (R-work {r_work_dep:.4}, \
         R-free {r_free_dep:.4})"
    );

    // Flatten B to the common mean and refit scaling to that start, so the
    // "before" R-work is scaling-consistent with the flattened model.
    let mean_b = deposited_b.iter().sum::<f64>() / deposited_b.len() as f64;
    let mut refined_b = vec![mean_b; deposited_b.len()];
    let _ = refinement
        .compute_map(&positions, &elements, &refined_b, &occupancies)
        .expect("compute_map (flattened)");
    let (r_work_before, _) = refinement
        .r_factors(&positions, &elements, &refined_b, &occupancies)
        .expect("r_factors (flattened)");

    let (r_work_after, _) = refinement
        .refine(&positions, &elements, &mut refined_b, &occupancies, 50)
        .expect("refine");

    assert!(
        r_work_after < r_work_before,
        "{pdb}: refinement did not improve R-work ({r_work_before:.4} -> \
         {r_work_after:.4})"
    );

    let b_corr = pearson(&deposited_b, &refined_b);
    assert!(
        b_corr > min_b_corr,
        "{pdb}: recovered B-correlation {b_corr:.3} below {min_b_corr}"
    );
}

#[test]
#[ignore = "real-structure recovery (~8s); run with --ignored"]
fn recovers_1aki() {
    assert_recovers("1AKI", 0.3);
}

#[test]
#[ignore = "real-structure recovery (~8s); run with --ignored"]
fn recovers_4xdx() {
    assert_recovers("4XDX", 0.3);
}

#[test]
#[ignore = "real-structure recovery (~8s); run with --ignored"]
fn recovers_6e6o() {
    assert_recovers("6E6O", 0.3);
}

#[test]
#[ignore = "real-structure recovery (~8s); run with --ignored"]
fn recovers_7kom() {
    assert_recovers("7KOM", 0.3);
}

#[test]
#[ignore = "real-structure recovery (~8s); run with --ignored"]
fn recovers_2oxy() {
    assert_recovers("2OXY", 0.3);
}
