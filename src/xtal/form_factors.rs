//! IT92 scattering form factors and van der Waals radii.
//!
//! Provides 4-Gaussian approximations to atomic scattering factors from the
//! International Tables for Crystallography, Volume C (1992), and Bondi/Mantina
//! van der Waals radii for supported elements.

use crate::element::Element;

/// Four-Gaussian approximation to the atomic scattering factor.
///
/// The scattering factor is evaluated as:
///
/// f(s) = c + sum_i a_i * exp(-b_i * s^2)
///
/// where s = sin(theta) / lambda.
#[derive(Debug, Clone, Copy)]
pub struct FormFactor {
    /// Gaussian amplitudes (4 terms).
    pub a: [f64; 4],
    /// Gaussian decay constants (4 terms).
    pub b: [f64; 4],
    /// Constant offset.
    pub c: f64,
}

impl FormFactor {
    /// Evaluate the scattering factor at a given value of s squared.
    ///
    /// `s_sq` is (sin(theta) / lambda)^2 in inverse Angstroms squared.
    #[must_use]
    pub fn at_s_sq(&self, s_sq: f64) -> f64 {
        self.a[3].mul_add(
            (-self.b[3] * s_sq).exp(),
            self.a[2].mul_add(
                (-self.b[2] * s_sq).exp(),
                self.a[1].mul_add(
                    (-self.b[1] * s_sq).exp(),
                    self.a[0].mul_add((-self.b[0] * s_sq).exp(), self.c),
                ),
            ),
        )
    }
}

/// IT92 4-Gaussian scattering-factor coefficients for supported elements.
///
/// Indexed by the internal element order; use [`form_factor`] for lookups.
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
static FORM_FACTORS: [FormFactor; 18] = [
    // H
    FormFactor {
        a: [0.493002, 0.322912, 0.140191, 0.04081],
        b: [10.5109, 26.1257, 3.14236, 57.7997],
        c: 0.003038,
    },
    // C
    FormFactor {
        a: [2.31, 1.02, 1.5886, 0.865],
        b: [20.8439, 10.2075, 0.5687, 51.6512],
        c: 0.2156,
    },
    // N
    FormFactor {
        a: [12.2126, 3.1322, 2.0125, 1.1663],
        b: [0.0057, 9.8933, 28.9975, 0.5826],
        c: -11.529,
    },
    // O
    FormFactor {
        a: [3.0485, 2.2868, 1.5463, 0.867],
        b: [13.2771, 5.7011, 0.3239, 32.9089],
        c: 0.2508,
    },
    // S
    FormFactor {
        a: [6.9053, 5.2034, 1.4379, 1.5863],
        b: [1.4679, 22.2151, 0.2536, 56.172],
        c: 0.8669,
    },
    // P
    FormFactor {
        a: [6.4345, 4.1791, 1.78, 1.4908],
        b: [1.9067, 27.157, 0.526, 68.1645],
        c: 1.1149,
    },
    // Se
    FormFactor {
        a: [17.0006, 5.8196, 3.9731, 4.3543],
        b: [2.4098, 0.2726, 15.2372, 43.8163],
        c: 2.8409,
    },
    // Fe
    FormFactor {
        a: [11.7695, 7.3573, 3.5222, 2.3045],
        b: [4.7611, 0.3072, 15.3535, 76.8805],
        c: 1.0369,
    },
    // Zn
    FormFactor {
        a: [14.0743, 7.0318, 5.1652, 2.41],
        b: [3.2655, 0.2333, 10.3163, 58.7097],
        c: 1.3041,
    },
    // Mg
    FormFactor {
        a: [5.4204, 2.1735, 1.2269, 2.3073],
        b: [2.8275, 79.2611, 0.3808, 7.1937],
        c: 0.8584,
    },
    // Ca
    FormFactor {
        a: [8.6266, 7.3873, 1.5899, 1.0211],
        b: [10.4421, 0.6599, 85.7484, 178.437],
        c: 1.3751,
    },
    // Na
    FormFactor {
        a: [4.7626, 3.1736, 1.2674, 1.1128],
        b: [3.285, 8.8422, 0.3136, 129.424],
        c: 0.676,
    },
    // Cl
    FormFactor {
        a: [11.4604, 7.1964, 6.2556, 1.6455],
        b: [0.0104, 1.1662, 18.5194, 47.7784],
        c: -9.5574,
    },
    // K
    FormFactor {
        a: [8.2186, 7.4398, 1.0519, 0.8659],
        b: [12.7949, 0.7748, 213.187, 41.6841],
        c: 1.4228,
    },
    // Mn
    FormFactor {
        a: [11.2819, 7.3573, 3.0193, 2.2441],
        b: [5.3409, 0.3432, 17.8674, 83.7543],
        c: 1.0896,
    },
    // Co
    FormFactor {
        a: [12.2841, 7.3409, 4.0034, 2.3488],
        b: [4.2791, 0.2784, 13.5359, 71.1692],
        c: 1.0118,
    },
    // Ni
    FormFactor {
        a: [12.8376, 7.292, 4.4438, 2.38],
        b: [3.8785, 0.2565, 12.1763, 66.3421],
        c: 1.0341,
    },
    // Cu
    FormFactor {
        a: [13.338, 7.1676, 5.6158, 1.6735],
        b: [3.5828, 0.247, 11.3966, 64.8126],
        c: 1.191,
    },
];

/// Look up the IT92 scattering form factor for an element.
///
/// Returns `None` for elements without tabulated data (`Unknown`, `Br`, `I`,
/// `F`).
#[must_use]
pub fn form_factor(element: Element) -> Option<&'static FormFactor> {
    let index = match element {
        Element::H => 0,
        Element::C => 1,
        Element::N => 2,
        Element::O => 3,
        Element::S => 4,
        Element::P => 5,
        Element::Se => 6,
        Element::Fe => 7,
        Element::Zn => 8,
        Element::Mg => 9,
        Element::Ca => 10,
        Element::Na => 11,
        Element::Cl => 12,
        Element::K => 13,
        Element::Mn => 14,
        Element::Co => 15,
        Element::Ni => 16,
        Element::Cu => 17,
        Element::Br | Element::I | Element::F | Element::Unknown => {
            return None
        }
    };
    Some(&FORM_FACTORS[index])
}

/// Bondi/Mantina van der Waals radii in Angstroms.
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
static VDW_RADII: [f64; 18] = [
    1.20, // H
    1.70, // C
    1.55, // N
    1.52, // O
    1.80, // S
    1.80, // P
    1.90, // Se
    1.26, // Fe
    1.39, // Zn
    1.73, // Mg
    2.31, // Ca
    2.27, // Na
    1.75, // Cl
    2.75, // K
    1.19, // Mn
    1.13, // Co
    1.63, // Ni
    1.40, // Cu
];

/// Look up the van der Waals radius for an element (Angstroms).
///
/// Returns `None` for elements without tabulated data (`Unknown`, `Br`, `I`,
/// `F`).
#[must_use]
pub fn vdw_radius(element: Element) -> Option<f64> {
    let index = match element {
        Element::H => 0,
        Element::C => 1,
        Element::N => 2,
        Element::O => 3,
        Element::S => 4,
        Element::P => 5,
        Element::Se => 6,
        Element::Fe => 7,
        Element::Zn => 8,
        Element::Mg => 9,
        Element::Ca => 10,
        Element::Na => 11,
        Element::Cl => 12,
        Element::K => 13,
        Element::Mn => 14,
        Element::Co => 15,
        Element::Ni => 16,
        Element::Cu => 17,
        Element::Br | Element::I | Element::F | Element::Unknown => {
            return None
        }
    };
    Some(VDW_RADII[index])
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    /// Check that the sum of Gaussian amplitudes plus c equals the atomic
    /// number.
    fn check_z(element: Element, z: f64) {
        let ff = form_factor(element).expect("element should have form factor");
        let sum: f64 = ff.a.iter().sum::<f64>() + ff.c;
        assert!(
            (sum - z).abs() < 0.05,
            "sum(a)+c for {element:?} = {sum}, expected {z}"
        );
    }

    #[test]
    fn carbon_z() {
        check_z(Element::C, 6.0);
    }

    #[test]
    fn oxygen_z() {
        check_z(Element::O, 8.0);
    }

    #[test]
    fn iron_z() {
        check_z(Element::Fe, 26.0);
    }

    #[test]
    fn carbon_at_zero_approx_six() {
        let ff = form_factor(Element::C).expect("C should have form factor");
        let f0 = ff.at_s_sq(0.0);
        assert!((f0 - 6.0).abs() < 0.05, "C at s=0 should be ~6.0, got {f0}");
    }

    #[test]
    fn carbon_decreases_with_s() {
        let ff = form_factor(Element::C).expect("C should have form factor");
        let f0 = ff.at_s_sq(0.0);
        let f_half = ff.at_s_sq(0.25); // s=0.5 => s_sq=0.25
        assert!(
            f_half < f0,
            "f(s=0.5)={f_half} should be less than f(s=0)={f0}"
        );
    }

    #[test]
    fn vdw_carbon() {
        let r = vdw_radius(Element::C);
        assert_eq!(r, Some(1.70));
    }

    #[test]
    fn unknown_returns_none() {
        assert!(form_factor(Element::Unknown).is_none());
        assert!(vdw_radius(Element::Unknown).is_none());
    }
}
