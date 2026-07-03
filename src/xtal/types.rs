//! Core crystallographic types: unit cell, symmetry operations, space groups.

use std::f64::consts::PI;

// ── Constants ─────────────────────────────────────────────────────────

/// Common denominator for integer-encoded symmetry operations.
pub const DEN: i32 = 24;

// ── Symop ─────────────────────────────────────────────────────────────

/// A symmetry operation consisting of a rotation matrix and translation vector,
/// both encoded with integer arithmetic scaled by [`DEN`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Symop {
    /// 3×3 rotation matrix, each element scaled by [`DEN`].
    pub rot: [[i32; 3]; 3],
    /// Translation vector, each element scaled by [`DEN`].
    pub tran: [i32; 3],
}

impl Symop {
    /// Apply this symmetry operation to fractional coordinates.
    ///
    /// Computes `R·xyz + t`, where `R` and `t` are decoded from integer form.
    #[must_use]
    pub fn apply_to_frac(&self, xyz: [f64; 3]) -> [f64; 3] {
        let d = f64::from(DEN);
        let mut out = [0.0; 3];
        for (i, out_val) in out.iter_mut().enumerate() {
            *out_val = f64::from(self.rot[i][2]).mul_add(
                xyz[2],
                f64::from(self.rot[i][0])
                    .mul_add(xyz[0], f64::from(self.rot[i][1]) * xyz[1]),
            ) / d
                + f64::from(self.tran[i]) / d;
        }
        out
    }

    /// Apply this symmetry operation to Miller indices `(h, k, l)`.
    ///
    /// Uses the *transpose* of the rotation matrix and divides by [`DEN`].
    #[must_use]
    pub fn apply_to_hkl(&self, hkl: [i32; 3]) -> [i32; 3] {
        let raw = self.apply_to_hkl_raw(hkl);
        [raw[0] / DEN, raw[1] / DEN, raw[2] / DEN]
    }

    /// Apply the transpose of the rotation matrix to Miller indices *without*
    /// dividing by [`DEN`]. The returned values are still scaled by `DEN`.
    #[must_use]
    pub fn apply_to_hkl_raw(&self, hkl: [i32; 3]) -> [i32; 3] {
        [
            self.rot[0][0] * hkl[0]
                + self.rot[1][0] * hkl[1]
                + self.rot[2][0] * hkl[2],
            self.rot[0][1] * hkl[0]
                + self.rot[1][1] * hkl[1]
                + self.rot[2][1] * hkl[2],
            self.rot[0][2] * hkl[0]
                + self.rot[1][2] * hkl[1]
                + self.rot[2][2] * hkl[2],
        ]
    }

    /// Phase shift for reflection `(h, k, l)` due to the translational part.
    ///
    /// Returns `-2π / DEN * (h·t₀ + k·t₁ + l·t₂)`.
    #[must_use]
    pub fn phase_shift(&self, hkl: [i32; 3]) -> f64 {
        let dot = hkl[0] * self.tran[0]
            + hkl[1] * self.tran[1]
            + hkl[2] * self.tran[2];
        -2.0 * PI / f64::from(DEN) * f64::from(dot)
    }
}

// ── GroupOps ──────────────────────────────────────────────────────────

/// A set of symmetry and centering operations that define a space group.
#[derive(Debug, Clone)]
pub struct GroupOps {
    /// Symmetry operations (rotation + translation).
    pub sym_ops: Vec<Symop>,
    /// Centering translations (e.g. `[0,0,0]` for primitive).
    pub cen_ops: Vec<[i32; 3]>,
}

// ── Free functions ────────────────────────────────────────────────────

/// Compute the epsilon factor for reflection `hkl` under the given group.
///
/// Counts how many symmetry operations map `hkl` to itself (using raw/unscaled
/// comparison), then multiplies by the number of centering operations.
#[must_use]
pub fn epsilon_factor(hkl: [i32; 3], group: &GroupOps) -> i32 {
    let target = [DEN * hkl[0], DEN * hkl[1], DEN * hkl[2]];
    let mut count = 0i32;
    for op in &group.sym_ops {
        let mapped = op.apply_to_hkl_raw(hkl);
        if mapped == target {
            count += 1;
        }
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let n_cen = group.cen_ops.len() as i32;
    count * n_cen
}

/// Check whether reflection `hkl` is centric under the given symmetry
/// operations.
///
/// Returns `true` if any operation maps `hkl` to `−hkl` (raw comparison).
#[must_use]
pub fn is_centric(hkl: [i32; 3], ops: &[Symop]) -> bool {
    let neg_target = [-DEN * hkl[0], -DEN * hkl[1], -DEN * hkl[2]];
    for op in ops {
        let mapped = op.apply_to_hkl_raw(hkl);
        if mapped == neg_target {
            return true;
        }
    }
    false
}

/// Determine whether reflection `hkl` is systematically absent.
///
/// Checks both centering violations and screw-axis / glide-plane absences.
#[must_use]
pub fn is_systematically_absent(hkl: [i32; 3], group: &GroupOps) -> bool {
    // Centering check: for each centering vector c, h·c must be an integer
    // (i.e. divisible by DEN).
    for cen in &group.cen_ops {
        let dot = hkl[0] * cen[0] + hkl[1] * cen[1] + hkl[2] * cen[2];
        if dot % DEN != 0 {
            return true;
        }
    }

    // Screw-axis / glide-plane check: for every op that maps hkl to itself,
    // the phase shift must be a multiple of 2π (i.e. h·t must be 0 mod DEN).
    let target = [DEN * hkl[0], DEN * hkl[1], DEN * hkl[2]];
    for op in &group.sym_ops {
        let mapped = op.apply_to_hkl_raw(hkl);
        if mapped == target {
            let dot =
                hkl[0] * op.tran[0] + hkl[1] * op.tran[1] + hkl[2] * op.tran[2];
            if dot % DEN != 0 {
                return true;
            }
        }
    }

    false
}

// ── Space-group data (DEN = 24) ──────────────────────────────────────

/// Compact representation of a symmetry operation for const data tables.
pub type SymopData = ([[i32; 3]; 3], [i32; 3]);

/// P 1 (SG 1) — centering translations.
pub const SG1_CEN: &[[i32; 3]] = &[[0, 0, 0]];
/// P 1 (SG 1) — symmetry operations.
#[allow(clippy::unreadable_literal)]
pub const SG1_OPS: &[SymopData] =
    &[([[24, 0, 0], [0, 24, 0], [0, 0, 24]], [0, 0, 0])];

/// P 1 21 1 (SG 4) — centering translations.
pub const SG4_CEN: &[[i32; 3]] = &[[0, 0, 0]];
/// P 1 21 1 (SG 4) — symmetry operations.
#[allow(clippy::unreadable_literal)]
pub const SG4_OPS: &[SymopData] = &[
    ([[24, 0, 0], [0, 24, 0], [0, 0, 24]], [0, 0, 0]),
    ([[-24, 0, 0], [0, 24, 0], [0, 0, -24]], [0, 12, 0]),
];

/// C 1 2 1 (SG 5) — centering translations.
pub const SG5_CEN: &[[i32; 3]] = &[[0, 0, 0], [12, 12, 0]];
/// C 1 2 1 (SG 5) — symmetry operations.
#[allow(clippy::unreadable_literal)]
pub const SG5_OPS: &[SymopData] = &[
    ([[24, 0, 0], [0, 24, 0], [0, 0, 24]], [0, 0, 0]),
    ([[-24, 0, 0], [0, 24, 0], [0, 0, -24]], [0, 0, 0]),
];

/// P 21 21 21 (SG 19) — centering translations.
pub const SG19_CEN: &[[i32; 3]] = &[[0, 0, 0]];
/// P 21 21 21 (SG 19) — symmetry operations.
#[allow(clippy::unreadable_literal)]
pub const SG19_OPS: &[SymopData] = &[
    ([[24, 0, 0], [0, 24, 0], [0, 0, 24]], [0, 0, 0]),
    ([[-24, 0, 0], [0, -24, 0], [0, 0, 24]], [12, 0, 12]),
    ([[24, 0, 0], [0, -24, 0], [0, 0, -24]], [12, 12, 0]),
    ([[-24, 0, 0], [0, 24, 0], [0, 0, -24]], [0, 12, 12]),
];

/// I 2 2 2 (SG 23) — centering translations.
pub const SG23_CEN: &[[i32; 3]] = &[[0, 0, 0], [12, 12, 12]];
/// I 2 2 2 (SG 23) — symmetry operations.
#[allow(clippy::unreadable_literal)]
pub const SG23_OPS: &[SymopData] = &[
    ([[24, 0, 0], [0, 24, 0], [0, 0, 24]], [0, 0, 0]),
    ([[-24, 0, 0], [0, -24, 0], [0, 0, 24]], [0, 0, 0]),
    ([[24, 0, 0], [0, -24, 0], [0, 0, -24]], [0, 0, 0]),
    ([[-24, 0, 0], [0, 24, 0], [0, 0, -24]], [0, 0, 0]),
];

/// P 41 21 2 (SG 92) — centering translations.
pub const SG92_CEN: &[[i32; 3]] = &[[0, 0, 0]];
/// P 41 21 2 (SG 92) — symmetry operations.
#[allow(clippy::unreadable_literal)]
pub const SG92_OPS: &[SymopData] = &[
    ([[24, 0, 0], [0, 24, 0], [0, 0, 24]], [0, 0, 0]),
    ([[0, -24, 0], [24, 0, 0], [0, 0, 24]], [12, 12, 6]),
    ([[-24, 0, 0], [0, -24, 0], [0, 0, 24]], [0, 0, 12]),
    ([[0, 24, 0], [-24, 0, 0], [0, 0, 24]], [12, 12, 18]),
    ([[24, 0, 0], [0, -24, 0], [0, 0, -24]], [12, 12, 18]),
    ([[0, 24, 0], [24, 0, 0], [0, 0, -24]], [0, 0, 0]),
    ([[-24, 0, 0], [0, 24, 0], [0, 0, -24]], [12, 12, 6]),
    ([[0, -24, 0], [-24, 0, 0], [0, 0, -24]], [0, 0, 12]),
];

/// P 43 21 2 (SG 96) — centering translations.
pub const SG96_CEN: &[[i32; 3]] = &[[0, 0, 0]];
/// P 43 21 2 (SG 96) — symmetry operations.
#[allow(clippy::unreadable_literal)]
pub const SG96_OPS: &[SymopData] = &[
    ([[24, 0, 0], [0, 24, 0], [0, 0, 24]], [0, 0, 0]),
    ([[0, -24, 0], [24, 0, 0], [0, 0, 24]], [12, 12, 18]),
    ([[-24, 0, 0], [0, -24, 0], [0, 0, 24]], [0, 0, 12]),
    ([[0, 24, 0], [-24, 0, 0], [0, 0, 24]], [12, 12, 6]),
    ([[24, 0, 0], [0, -24, 0], [0, 0, -24]], [12, 12, 6]),
    ([[0, 24, 0], [24, 0, 0], [0, 0, -24]], [0, 0, 0]),
    ([[-24, 0, 0], [0, 24, 0], [0, 0, -24]], [12, 12, 18]),
    ([[0, -24, 0], [-24, 0, 0], [0, 0, -24]], [0, 0, 12]),
];

/// P 31 2 1 (SG 152) — centering translations.
pub const SG152_CEN: &[[i32; 3]] = &[[0, 0, 0]];
/// P 31 2 1 (SG 152) — symmetry operations.
#[allow(clippy::unreadable_literal)]
pub const SG152_OPS: &[SymopData] = &[
    ([[24, 0, 0], [0, 24, 0], [0, 0, 24]], [0, 0, 0]),
    ([[0, -24, 0], [24, -24, 0], [0, 0, 24]], [0, 0, 8]),
    ([[-24, 24, 0], [-24, 0, 0], [0, 0, 24]], [0, 0, 16]),
    ([[0, 24, 0], [24, 0, 0], [0, 0, -24]], [0, 0, 0]),
    ([[-24, 0, 0], [-24, 24, 0], [0, 0, -24]], [0, 0, 8]),
    ([[24, -24, 0], [0, -24, 0], [0, 0, -24]], [0, 0, 16]),
];

/// P 32 2 1 (SG 154) — centering translations.
pub const SG154_CEN: &[[i32; 3]] = &[[0, 0, 0]];
/// P 32 2 1 (SG 154) — symmetry operations.
#[allow(clippy::unreadable_literal)]
pub const SG154_OPS: &[SymopData] = &[
    ([[24, 0, 0], [0, 24, 0], [0, 0, 24]], [0, 0, 0]),
    ([[0, -24, 0], [24, -24, 0], [0, 0, 24]], [0, 0, 16]),
    ([[-24, 24, 0], [-24, 0, 0], [0, 0, 24]], [0, 0, 8]),
    ([[0, 24, 0], [24, 0, 0], [0, 0, -24]], [0, 0, 0]),
    ([[-24, 0, 0], [-24, 24, 0], [0, 0, -24]], [0, 0, 16]),
    ([[24, -24, 0], [0, -24, 0], [0, 0, -24]], [0, 0, 8]),
];

/// P 61 2 2 (SG 178) — centering translations.
pub const SG178_CEN: &[[i32; 3]] = &[[0, 0, 0]];
/// P 61 2 2 (SG 178) — symmetry operations.
#[allow(clippy::unreadable_literal)]
pub const SG178_OPS: &[SymopData] = &[
    ([[24, 0, 0], [0, 24, 0], [0, 0, 24]], [0, 0, 0]),
    ([[24, -24, 0], [24, 0, 0], [0, 0, 24]], [0, 0, 4]),
    ([[0, -24, 0], [24, -24, 0], [0, 0, 24]], [0, 0, 8]),
    ([[-24, 0, 0], [0, -24, 0], [0, 0, 24]], [0, 0, 12]),
    ([[-24, 24, 0], [-24, 0, 0], [0, 0, 24]], [0, 0, 16]),
    ([[0, 24, 0], [-24, 24, 0], [0, 0, 24]], [0, 0, 20]),
    ([[0, -24, 0], [-24, 0, 0], [0, 0, -24]], [0, 0, 20]),
    ([[24, -24, 0], [0, -24, 0], [0, 0, -24]], [0, 0, 0]),
    ([[24, 0, 0], [24, -24, 0], [0, 0, -24]], [0, 0, 4]),
    ([[0, 24, 0], [24, 0, 0], [0, 0, -24]], [0, 0, 8]),
    ([[-24, 24, 0], [0, 24, 0], [0, 0, -24]], [0, 0, 12]),
    ([[-24, 0, 0], [-24, 24, 0], [0, 0, -24]], [0, 0, 16]),
];

// ── Crystal system ────────────────────────────────────────────────────

/// Classification of the crystal system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrystalSystem {
    /// No symmetry constraints on cell parameters.
    Triclinic,
    /// One unique axis (conventionally b), β ≠ 90°.
    Monoclinic,
    /// Three mutually perpendicular axes, all angles 90°.
    Orthorhombic,
    /// a = b, all angles 90°.
    Tetragonal,
    /// a = b, α = β = 90°, γ = 120°.
    Trigonal,
    /// a = b, α = β = 90°, γ = 120°.
    Hexagonal,
    /// a = b = c, α = β = γ = 90°.
    Cubic,
}

// ── SpaceGroup ────────────────────────────────────────────────────────

/// A crystallographic space group with its symmetry operations and metadata.
#[derive(Debug, Clone)]
pub struct SpaceGroup {
    /// International Tables number.
    pub number: u16,
    /// Hermann–Mauguin symbol.
    pub hm: &'static str,
    /// Group operations (symmetry + centering).
    pub ops: GroupOps,
    /// Crystal system classification.
    pub crystal_system: CrystalSystem,
}

/// Build a [`GroupOps`] from const data slices.
fn build_group_ops(cen: &[[i32; 3]], ops: &[SymopData]) -> GroupOps {
    GroupOps {
        sym_ops: ops
            .iter()
            .map(|(rot, tran)| Symop {
                rot: *rot,
                tran: *tran,
            })
            .collect(),
        cen_ops: cen.to_vec(),
    }
}

/// Look up a space group by its International Tables number.
///
/// Returns `None` for space groups not included in the hardcoded table.
#[must_use]
pub fn space_group(number: u16) -> Option<SpaceGroup> {
    let (hm, cen, ops, system): (
        &str,
        &[[i32; 3]],
        &[SymopData],
        CrystalSystem,
    ) = match number {
        1 => ("P 1", SG1_CEN, SG1_OPS, CrystalSystem::Triclinic),
        4 => ("P 1 21 1", SG4_CEN, SG4_OPS, CrystalSystem::Monoclinic),
        5 => ("C 1 2 1", SG5_CEN, SG5_OPS, CrystalSystem::Monoclinic),
        19 => (
            "P 21 21 21",
            SG19_CEN,
            SG19_OPS,
            CrystalSystem::Orthorhombic,
        ),
        23 => ("I 2 2 2", SG23_CEN, SG23_OPS, CrystalSystem::Orthorhombic),
        92 => ("P 41 21 2", SG92_CEN, SG92_OPS, CrystalSystem::Tetragonal),
        96 => ("P 43 21 2", SG96_CEN, SG96_OPS, CrystalSystem::Tetragonal),
        152 => ("P 31 2 1", SG152_CEN, SG152_OPS, CrystalSystem::Trigonal),
        154 => ("P 32 2 1", SG154_CEN, SG154_OPS, CrystalSystem::Trigonal),
        178 => ("P 61 2 2", SG178_CEN, SG178_OPS, CrystalSystem::Hexagonal),
        _ => return None,
    };

    Some(SpaceGroup {
        number,
        hm,
        ops: build_group_ops(cen, ops),
        crystal_system: system,
    })
}

// ── Grid helper functions ─────────────────────────────────────────────

/// Return the grid divisibility factors `[nu_factor, nv_factor, nw_factor]` for
/// the given space group number, or `None` if the group is unsupported.
#[must_use]
pub fn grid_factors(sg_number: u16) -> Option<[usize; 3]> {
    match sg_number {
        1 => Some([1, 1, 1]),
        4 => Some([1, 2, 1]),
        5 => Some([2, 2, 1]),
        19 | 23 => Some([2, 2, 2]),
        92 | 96 => Some([2, 2, 4]),
        152 | 154 => Some([1, 1, 3]),
        178 => Some([1, 1, 6]),
        _ => None,
    }
}

/// Whether the space group requires `nu == nv` (tetragonal, trigonal,
/// hexagonal).
#[must_use]
pub fn requires_equal_uv(sg_number: u16) -> bool {
    matches!(sg_number, 92 | 96 | 152 | 154 | 178)
}

/// Check whether `n` factors entirely into primes 2, 3, and 5 (a "smooth"
/// number).
#[must_use]
pub fn has_small_factorization(n: i32) -> bool {
    if n <= 0 {
        return false;
    }
    let mut v = n;
    while v % 2 == 0 {
        v /= 2;
    }
    while v % 3 == 0 {
        v /= 3;
    }
    while v % 5 == 0 {
        v /= 5;
    }
    v == 1
}

/// Round a floating-point value up to the nearest integer that has only 2, 3,
/// and 5 as prime factors.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn round_up_to_smooth(exact: f64) -> usize {
    #[allow(clippy::cast_sign_loss)]
    let mut n = exact.ceil() as i32;
    if n < 1 {
        n = 1;
    }
    while !has_small_factorization(n) {
        n += 1;
    }
    #[allow(clippy::cast_sign_loss)]
    let result = n as usize;
    result
}

// ── UnitCell ──────────────────────────────────────────────────────────

/// Crystallographic unit cell with precomputed orthogonalization and
/// fractionalization matrices, reciprocal-space parameters, and volume.
#[derive(Debug, Clone)]
pub struct UnitCell {
    /// Cell dimension a (Å).
    pub a: f64,
    /// Cell dimension b (Å).
    pub b: f64,
    /// Cell dimension c (Å).
    pub c: f64,
    /// Cell angle α (degrees).
    pub alpha: f64,
    /// Cell angle β (degrees).
    pub beta: f64,
    /// Cell angle γ (degrees).
    pub gamma: f64,
    /// Cell volume (ų).
    pub volume: f64,
    /// Reciprocal-space length a* (Å⁻¹).
    pub ar: f64,
    /// Reciprocal-space length b* (Å⁻¹).
    pub br: f64,
    /// Reciprocal-space length c* (Å⁻¹).
    pub cr: f64,
    /// Cosine of reciprocal angle α*.
    pub cos_alphar: f64,
    /// Cosine of reciprocal angle β*.
    pub cos_betar: f64,
    /// Cosine of reciprocal angle γ*.
    pub cos_gammar: f64,
    /// Orthogonalization matrix (fractional → Cartesian).
    pub orth: [[f64; 3]; 3],
    /// Fractionalization matrix (Cartesian → fractional).
    pub frac: [[f64; 3]; 3],
}

impl UnitCell {
    /// Create a new unit cell from the six cell parameters (a, b, c in Å;
    /// alpha, beta, gamma in degrees).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a: f64,
        b: f64,
        c: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
    ) -> Self {
        let mut cell = Self {
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
            volume: 0.0,
            ar: 0.0,
            br: 0.0,
            cr: 0.0,
            cos_alphar: 0.0,
            cos_betar: 0.0,
            cos_gammar: 0.0,
            orth: [[0.0; 3]; 3],
            frac: [[0.0; 3]; 3],
        };
        cell.calculate_properties();
        cell
    }

    /// Compute all derived properties from the six basic cell parameters.
    #[allow(clippy::too_many_lines, clippy::suboptimal_flops)]
    fn calculate_properties(&mut self) {
        let to_rad = PI / 180.0;

        // Use exact values for 90° angles to avoid floating-point noise.
        let cos_alpha = if (self.alpha - 90.0).abs() < 1e-10 {
            0.0
        } else {
            (self.alpha * to_rad).cos()
        };
        let sin_alpha = if (self.alpha - 90.0).abs() < 1e-10 {
            1.0
        } else {
            (self.alpha * to_rad).sin()
        };
        let cos_beta = if (self.beta - 90.0).abs() < 1e-10 {
            0.0
        } else {
            (self.beta * to_rad).cos()
        };
        let sin_beta = if (self.beta - 90.0).abs() < 1e-10 {
            1.0
        } else {
            (self.beta * to_rad).sin()
        };
        let cos_gamma = if (self.gamma - 90.0).abs() < 1e-10 {
            0.0
        } else {
            (self.gamma * to_rad).cos()
        };
        let sin_gamma = if (self.gamma - 90.0).abs() < 1e-10 {
            1.0
        } else {
            (self.gamma * to_rad).sin()
        };

        // Volume
        let val = 1.0
            - cos_alpha * cos_alpha
            - cos_beta * cos_beta
            - cos_gamma * cos_gamma
            + 2.0 * cos_alpha * cos_beta * cos_gamma;
        let sqrt_val = val.sqrt();
        self.volume = self.a * self.b * self.c * sqrt_val;

        // Reciprocal cell parameters
        self.ar = self.b * self.c * sin_alpha / self.volume;
        self.br = self.a * self.c * sin_beta / self.volume;
        self.cr = self.a * self.b * sin_gamma / self.volume;

        self.cos_alphar =
            (cos_beta * cos_gamma - cos_alpha) / (sin_beta * sin_gamma);
        self.cos_betar =
            (cos_alpha * cos_gamma - cos_beta) / (sin_alpha * sin_gamma);
        self.cos_gammar =
            (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta);

        let sin_alphar = (1.0 - self.cos_alphar * self.cos_alphar).sqrt();

        // Orthogonalization matrix (PDB convention, a along x)
        //   | a   b*cos_gamma   c*cos_beta                          |
        //   | 0   b*sin_gamma   c*(cos_alpha - cos_beta*cos_gamma)/sin_gamma |
        //   | 0   0             c*sin_beta*sin_alphar               |
        self.orth[0][0] = self.a;
        self.orth[0][1] = self.b * cos_gamma;
        self.orth[0][2] = self.c * cos_beta;
        self.orth[1][0] = 0.0;
        self.orth[1][1] = self.b * sin_gamma;
        self.orth[1][2] =
            self.c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
        self.orth[2][0] = 0.0;
        self.orth[2][1] = 0.0;
        self.orth[2][2] = self.c * sin_beta * sin_alphar;

        // Fractionalization matrix (inverse of orth, upper-triangular)
        self.frac[0][0] = 1.0 / self.a;
        self.frac[0][1] = -cos_gamma / (self.a * sin_gamma);
        self.frac[0][2] = (cos_gamma * (cos_alpha - cos_beta * cos_gamma)
            / sin_gamma
            - cos_beta * sin_gamma)
            / (self.a * sin_beta * sin_alphar * sin_gamma);
        self.frac[1][0] = 0.0;
        self.frac[1][1] = 1.0 / (self.b * sin_gamma);
        self.frac[1][2] = -(cos_alpha - cos_beta * cos_gamma)
            / (self.b * sin_gamma * sin_beta * sin_alphar);
        self.frac[2][0] = 0.0;
        self.frac[2][1] = 0.0;
        self.frac[2][2] = 1.0 / (self.c * sin_beta * sin_alphar);
    }

    /// Compute 1/d² for reflection `(h, k, l)`.
    #[must_use]
    #[allow(clippy::suboptimal_flops)]
    pub fn d_star_sq(&self, h: i32, k: i32, l: i32) -> f64 {
        let hf = f64::from(h);
        let kf = f64::from(k);
        let lf = f64::from(l);

        // d*² = h²a*² + k²b*² + l²c*²
        //       + 2hk a*b* cos(γ*) + 2hl a*c* cos(β*) + 2kl b*c* cos(α*)
        let diag = (hf * self.ar).mul_add(
            hf * self.ar,
            (kf * self.br).mul_add(kf * self.br, lf * lf * self.cr * self.cr),
        );
        let cross = (2.0 * kf * lf * self.br * self.cr).mul_add(
            self.cos_alphar,
            (2.0 * hf * lf * self.ar * self.cr).mul_add(
                self.cos_betar,
                2.0 * hf * kf * self.ar * self.br * self.cos_gammar,
            ),
        );
        diag + cross
    }

    /// Convert fractional coordinates to Cartesian (Å).
    #[must_use]
    pub fn orthogonalize(&self, frac: [f64; 3]) -> [f64; 3] {
        [
            self.orth[0][2].mul_add(
                frac[2],
                self.orth[0][0].mul_add(frac[0], self.orth[0][1] * frac[1]),
            ),
            self.orth[1][2].mul_add(
                frac[2],
                self.orth[1][0].mul_add(frac[0], self.orth[1][1] * frac[1]),
            ),
            self.orth[2][2].mul_add(
                frac[2],
                self.orth[2][0].mul_add(frac[0], self.orth[2][1] * frac[1]),
            ),
        ]
    }

    /// Convert Cartesian coordinates (Å) to fractional.
    #[must_use]
    pub fn fractionalize(&self, cart: [f64; 3]) -> [f64; 3] {
        [
            self.frac[0][2].mul_add(
                cart[2],
                self.frac[0][0].mul_add(cart[0], self.frac[0][1] * cart[1]),
            ),
            self.frac[1][2].mul_add(
                cart[2],
                self.frac[1][0].mul_add(cart[0], self.frac[1][1] * cart[1]),
            ),
            self.frac[2][2].mul_add(
                cart[2],
                self.frac[2][0].mul_add(cart[0], self.frac[2][1] * cart[1]),
            ),
        ]
    }
}

// ── Reflection & DensityGrid ──────────────────────────────────────────

/// A single measured (or calculated) reflection.
#[derive(Debug, Clone)]
pub struct Reflection {
    /// Miller index h.
    pub h: i32,
    /// Miller index k.
    pub k: i32,
    /// Miller index l.
    pub l: i32,
    /// Observed structure factor amplitude.
    pub f_obs: f32,
    /// Measurement uncertainty of `f_obs`.
    pub sigma_f: f32,
    /// Whether this reflection belongs to the free (test) set.
    pub free_flag: bool,
}

/// A 3D grid of electron-density values.
#[derive(Debug, Clone)]
pub struct DensityGrid {
    /// Flattened density values in row-major (u, v, w) order.
    pub data: Vec<f32>,
    /// Grid size along the a axis.
    pub nu: usize,
    /// Grid size along the b axis.
    pub nv: usize,
    /// Grid size along the c axis.
    pub nw: usize,
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::cast_possible_wrap)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    #[test]
    fn orthorhombic_orth_is_diagonal() {
        let cell = UnitCell::new(10.0, 20.0, 30.0, 90.0, 90.0, 90.0);
        // Diagonal elements
        assert!((cell.orth[0][0] - 10.0).abs() < TOL);
        assert!((cell.orth[1][1] - 20.0).abs() < TOL);
        assert!((cell.orth[2][2] - 30.0).abs() < TOL);
        // Off-diagonal elements should be zero
        assert!(cell.orth[0][1].abs() < TOL);
        assert!(cell.orth[0][2].abs() < TOL);
        assert!(cell.orth[1][0].abs() < TOL);
        assert!(cell.orth[1][2].abs() < TOL);
        assert!(cell.orth[2][0].abs() < TOL);
        assert!(cell.orth[2][1].abs() < TOL);
    }

    #[test]
    fn monoclinic_orth_has_nonzero_02() {
        let cell = UnitCell::new(10.0, 20.0, 30.0, 90.0, 100.0, 90.0);
        // orth[0][2] = c * cos(beta), beta=100° → cos < 0
        assert!(cell.orth[0][2].abs() > 0.1);
    }

    #[test]
    fn lysozyme_volume() {
        // P 43 21 2, a=b=79.1, c=37.9
        let cell = UnitCell::new(79.1, 79.1, 37.9, 90.0, 90.0, 90.0);
        let expected = 79.1 * 79.1 * 37.9; // ~237,050
        let rel_error = (cell.volume - expected).abs() / expected;
        assert!(
            rel_error < 0.01,
            "volume {:.0} vs expected {:.0}",
            cell.volume,
            expected
        );
    }

    #[test]
    fn d_star_sq_100() {
        let cell = UnitCell::new(50.0, 60.0, 70.0, 90.0, 90.0, 90.0);
        let dss = cell.d_star_sq(1, 0, 0);
        let expected = cell.ar * cell.ar;
        assert!(
            (dss - expected).abs() < 1e-12,
            "d*²={dss} vs ar²={expected}"
        );
    }

    #[test]
    fn epsilon_factor_general_reflection() {
        let sg = space_group(19);
        assert!(sg.is_some());
        if let Some(sg) = sg {
            // General reflection (1,2,3) should only match identity → epsilon =
            // 1 * n_cen
            let eps = epsilon_factor([1, 2, 3], &sg.ops);
            #[allow(clippy::cast_possible_truncation)]
            let n_cen = sg.ops.cen_ops.len() as i32;
            assert_eq!(eps, n_cen);
        }
    }

    #[test]
    fn systematically_absent_c_lattice() {
        let sg = space_group(5);
        assert!(sg.is_some());
        if let Some(sg) = sg {
            // C centering: h+k must be even. (1,0,0) has h+k=1 (odd) → absent
            assert!(is_systematically_absent([1, 0, 0], &sg.ops));
            // (2,0,0) has h+k=2 (even) → not absent from centering
            assert!(!is_systematically_absent([2, 0, 0], &sg.ops));
        }
    }

    #[test]
    fn round_up_to_smooth_cases() {
        assert_eq!(round_up_to_smooth(7.0), 8);
        assert_eq!(round_up_to_smooth(11.0), 12);
        assert_eq!(round_up_to_smooth(13.0), 15);
        assert_eq!(round_up_to_smooth(31.0), 32);
    }

    #[test]
    fn has_small_factorization_cases() {
        assert!(has_small_factorization(30)); // 2*3*5
        assert!(!has_small_factorization(7)); // prime > 5
        assert!(has_small_factorization(1));
        assert!(has_small_factorization(16)); // 2^4
        assert!(!has_small_factorization(0));
    }
}
