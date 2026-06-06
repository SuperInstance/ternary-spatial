//! # Ternary Spatial — Zero-Drift Direction + Hexagonal Position
//!
//! Integrates two fundamental spatial primitives for balanced ternary systems:
//!
//! 1. **Pythagorean48** — Exact unit vectors from integer Pythagorean triples.
//!    Zero floating-point drift when using rational arithmetic (a/c, b/c).
//!    Provides 128+ unique directions in the full 360° circle.
//!
//! 2. **Eisenstein Quantize** — Hexagonal A₂ lattice quantization.
//!    The densest 2D lattice packing, natural for balanced ternary coordinates.
//!
//! Together, they form the **TensorSpine** — a combined direction + position
//! state that is exact, deterministic, and drift-free.
//!
//! ## Zero-Drift Guarantee
//!
//! The Pythagorean48 direction type uses `num_bigint::BigInt` internally for
//! arbitrary-precision integer arithmetic during composition. This means:
//!
//! - **No overflow** — compose 10 or 10,000 rotations, magnitude² = 1.0 exactly
//! - **No drift** — integer arithmetic never accumulates floating-point error
//! - **No renormalization needed** — the unit vector is always exactly unit length

pub use eisenstein_quantize::lattice::{HexPoint, quantize_batch};
pub use eisenstein_quantize::integer::Eisenstein;
pub use eisenstein_quantize::error::EisensteinError;

use num_bigint::BigInt;
use num_traits::{Zero, One, Signed, ToPrimitive};
use std::cmp::Ordering;
use std::sync::OnceLock;

// ──────────────────────────────────────────────
//  Helpers
// ──────────────────────────────────────────────

/// GCD of two BigInt values.
fn gcd_big(a: &BigInt, b: &BigInt) -> BigInt {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_zero() {
        let t = b.clone();
        b = a % &t;
        a = t;
    }
    a
}

/// GCD of three BigInt values.
fn gcd3_big(a: &BigInt, b: &BigInt, c: &BigInt) -> BigInt {
    gcd_big(&gcd_big(a, b), c)
}

// ──────────────────────────────────────────────
//  Module: pythagorean48
// ──────────────────────────────────────────────

/// An exact direction on the unit circle, represented as a Pythagorean triple
/// (a, b, c) where a² + b² = c² exactly.
///
/// The unit vector is (a/c, b/c). Since a, b, c are integers and a² + b² = c²
/// exactly, the magnitude is always exactly 1.0 — zero drift, forever.
///
/// Internally uses `BigInt` for arbitrary-precision composition arithmetic.
/// For compact embedded deployments, construct from small triples or use
/// `Direction48::to_compact()` to get an i64 snapshot.
#[derive(Debug, Clone)]
pub struct Direction48 {
    /// First component of the direction vector × c
    pub a: BigInt,
    /// Second component of the direction vector × c
    pub b: BigInt,
    /// Hypotenuse (magnitude scaling factor)
    pub c: BigInt,
}

impl Direction48 {
    /// Create a new direction from a Pythagorean triple with i64 values.
    pub fn new(a: i64, b: i64, c: i64) -> Self {
        Self {
            a: BigInt::from(a),
            b: BigInt::from(b),
            c: BigInt::from(c),
        }
    }

    /// Create a new direction from BigInt values, automatically reduced by gcd.
    pub fn from_big(a: BigInt, b: BigInt, c: BigInt) -> Self {
        Self { a, b, c }.normalized()
    }

    /// Create a direction from any integers that satisfy a² + b² = c².
    /// Panics in debug mode if the triple is invalid.
    pub fn new_checked(a: i64, b: i64, c: i64) -> Self {
        let d = Self::new(a, b, c);
        debug_assert!(d.is_valid(), "({a}, {b}, {c}) is not a valid triple");
        d
    }

    /// Identity direction (1, 0) — angle 0°.
    pub fn identity() -> Self {
        Self {
            a: BigInt::from(1),
            b: BigInt::from(0),
            c: BigInt::from(1),
        }
    }

    /// Normalize by dividing out gcd(a, b, c).
    pub fn normalized(&self) -> Self {
        let g = gcd3_big(&self.a.abs(), &self.b.abs(), &self.c.abs());
        if g.is_one() {
            return Self {
                a: self.a.clone(),
                b: self.b.clone(),
                c: self.c.clone(),
            };
        }
        Self {
            a: &self.a / &g,
            b: &self.b / &g,
            c: &self.c / &g,
        }
    }

    /// Verify this is a valid Pythagorean triple: a² + b² = c².
    pub fn is_valid(&self) -> bool {
        let lhs = (&self.a).pow(2u32) + (&self.b).pow(2u32);
        let rhs = (&self.c).pow(2);
        lhs == rhs
    }

    /// Convert BigInt to f64 for angle computation.
    fn big_to_f64(x: &BigInt) -> f64 {
        x.to_f64().unwrap_or_else(|| {
            // Fallback: approximate from magnitude digits
            let mut val = 0.0f64;
            for d in x.magnitude().iter_u32_digits().rev() {
                val = val * (u32::MAX as f64 + 1.0) + d as f64;
            }
            if x.is_negative() { -val } else { val }
        })
    }

    /// Angle in degrees (for display / nearest-neighbor search).
    pub fn angle_deg(&self) -> f64 {
        let a_f = Self::big_to_f64(&self.a);
        let b_f = Self::big_to_f64(&self.b);
        b_f.atan2(a_f).to_degrees()
    }

    /// Angle in radians.
    pub fn angle_rad(&self) -> f64 {
        let a_f = Self::big_to_f64(&self.a);
        let b_f = Self::big_to_f64(&self.b);
        b_f.atan2(a_f)
    }

    /// Compose this direction with another via exact rational arithmetic.
    ///
    /// Uses arbitrary-precision BigInt arithmetic so there is no overflow.
    /// The result is always a valid Pythagorean triple with magnitude exactly 1.
    ///
    /// Mathematically, this is the rotation matrix composition:
    ///
    /// ```text
    /// new_a = a₁a₂ - b₁b₂
    /// new_b = a₁b₂ + b₁a₂
    /// new_c = c₁c₂
    /// ```
    ///
    /// Guaranteed: new_a² + new_b² = new_c²
    pub fn compose(&self, other: &Direction48) -> Self {
        // Rotation matrix composition with BigInt
        let num_a = &self.a * &other.a - &self.b * &other.b;
        let num_b = &self.a * &other.b + &self.b * &other.a;
        let denom = &self.c * &other.c;

        if num_a.is_zero() && num_b.is_zero() {
            return Self::identity();
        }

        // Reduce by gcd
        let g = gcd3_big(&num_a.abs(), &num_b.abs(), &denom.abs());
        if g.is_zero() {
            return Self::identity();
        }

        let final_a = num_a / &g;
        let final_b = num_b / &g;
        let final_c = denom / &g;

        // Ensure c is positive
        if final_c.is_negative() {
            Self {
                a: -final_a,
                b: -final_b,
                c: -final_c,
            }
        } else {
            Self {
                a: final_a,
                b: final_b,
                c: final_c,
            }
        }
    }

    /// Get the unit vector as (cos, sin) in floating-point.
    pub fn unit_vector_f64(&self) -> (f64, f64) {
        (self.angle_deg().to_radians().cos(), self.angle_deg().to_radians().sin())
    }

    /// Get the magnitude squared (always 1.0 by construction).
    pub fn magnitude_sq(&self) -> f64 {
        let a_f = Self::big_to_f64(&self.a);
        let b_f = Self::big_to_f64(&self.b);
        let c_f = Self::big_to_f64(&self.c);
        if c_f == 0.0 { return 0.0; }
        (a_f * a_f + b_f * b_f) / (c_f * c_f)
    }

    /// Deep equality check (normalized comparison).
    pub fn eq_normalized(&self, other: &Direction48) -> bool {
        let sn = self.normalized();
        let on = other.normalized();
        sn.a == on.a && sn.b == on.b && sn.c == on.c
    }

    /// Get a compact (i64) snapshot of this direction.
    /// Returns None if the direction values exceed i64 range.
    pub fn to_compact(&self) -> Option<Direction48Compact> {
        let a = i64::try_from(&self.a).ok()?;
        let b = i64::try_from(&self.b).ok()?;
        let c = i64::try_from(&self.c).ok()?;
        Some(Direction48Compact { a, b, c })
    }
}

impl PartialEq for Direction48 {
    fn eq(&self, other: &Self) -> bool {
        self.eq_normalized(other)
    }
}

impl Eq for Direction48 {}

impl std::hash::Hash for Direction48 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let n = self.normalized();
        n.a.hash(state);
        n.b.hash(state);
        n.c.hash(state);
    }
}

/// A compact i64 version of Direction48 for embedded use.
///
/// Limited to compositions where the values stay within i64 range
/// (~15-20 compositions depending on the triple magnitudes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Direction48Compact {
    pub a: i64,
    pub b: i64,
    pub c: i64,
}

impl Direction48Compact {
    pub fn new(a: i64, b: i64, c: i64) -> Self {
        Self { a, b, c }
    }
}

// ──────────────────────────────────────────────
//  Pythagorean48 Direction Database
// ──────────────────────────────────────────────

/// Pre-computed database of all unique Pythagorean48 directions.
struct DirectionDatabase {
    directions: Vec<Direction48>,
}

impl DirectionDatabase {
    fn build() -> Self {
        let triples = enumerate_pythagorean_triples(100);
        let mut dir_set: Vec<Direction48> = Vec::new();

        for &(a, b, c) in &triples {
            // 8 sign/swap symmetries: (±a, ±b), (±b, ±a)
            let syms = [
                (a, b, c),
                (-a, b, c),
                (a, -b, c),
                (-a, -b, c),
                (b, a, c),
                (-b, a, c),
                (b, -a, c),
                (-b, -a, c),
            ];

            for &(sa, sb, sc) in &syms {
                let dir = Direction48::new(sa, sb, sc).normalized();
                if !dir_set.iter().any(|d| d == &dir) {
                    dir_set.push(dir);
                }
            }
        }

        // Sort by angle
        dir_set.sort_by(|a, b| {
            a.angle_deg()
                .partial_cmp(&b.angle_deg())
                .unwrap_or(Ordering::Equal)
        });

        Self {
            directions: dir_set,
        }
    }

    fn nearest(&self, angle_deg: f64) -> &Direction48 {
        let target = angle_deg.rem_euclid(360.0);
        self.directions
            .iter()
            .min_by(|a, b| {
                let da = ((a.angle_deg() - target + 180.0).rem_euclid(360.0) - 180.0).abs();
                let db = ((b.angle_deg() - target + 180.0).rem_euclid(360.0) - 180.0).abs();
                da.partial_cmp(&db).unwrap()
            })
            .expect("DirectionDatabase should not be empty")
    }

    fn get(&self, index: usize) -> &Direction48 {
        &self.directions[index % self.directions.len()]
    }
}

/// Global lazy-initialized direction database.
fn direction_db() -> &'static DirectionDatabase {
    static DB: OnceLock<DirectionDatabase> = OnceLock::new();
    DB.get_or_init(DirectionDatabase::build)
}

/// Enumerate all Pythagorean triples with c ≤ max_c.
///
/// Uses Euclid's formula: a = k·(m²−n²), b = k·2mn, c = k·(m²+n²)
/// where m > n, gcd(m,n) = 1, and (m − n) is odd.
pub fn enumerate_pythagorean_triples(max_c: i64) -> Vec<(i64, i64, i64)> {
    use std::collections::BTreeSet;
    let mut triples: BTreeSet<(i64, i64, i64)> = BTreeSet::new();
    let max_m = (max_c as f64).sqrt() as i64 + 1;

    for m in 1..=max_m {
        for n in 1..m {
            if (m - n) % 2 == 0 {
                continue;
            }
            if gcd_i64(m, n) != 1 {
                continue;
            }

            let a0 = m * m - n * n;
            let b0 = 2 * m * n;
            let c0 = m * m + n * n;

            if c0 > max_c {
                continue;
            }

            let mut k = 1i64;
            while k * c0 <= max_c {
                let a = k * a0;
                let b = k * b0;
                let c = k * c0;
                triples.insert((a.min(b), a.max(b), c));
                k += 1;
            }
        }
    }

    triples.into_iter().collect()
}

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Get the total number of unique Pythagorean48 directions.
pub fn pythagorean48_count() -> usize {
    direction_db().directions.len()
}

/// Get all unique Pythagorean48 directions (lazy-initialized).
pub fn all_directions() -> &'static [Direction48] {
    &direction_db().directions
}

/// Find the nearest Pythagorean48 direction to a given angle (in degrees).
pub fn nearest_direction(angle_deg: f64) -> &'static Direction48 {
    direction_db().nearest(angle_deg)
}

/// Chain N rotations using exact BigInt arithmetic and verify zero drift.
///
/// Returns a vector of (direction, magnitude_sq) for each step.
/// magnitude_sq should always be exactly 1.0.
pub fn chain_rotations(count: usize, seed: u64) -> Vec<Direction48> {
    use rand::Rng;
    use rand::SeedableRng;
    let dirs = all_directions();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut current = Direction48::identity();
    let mut results = Vec::with_capacity(count);

    for _ in 0..count {
        let idx = rng.gen_range(0..dirs.len());
        current = dirs[idx].compose(&current);
        results.push(current.clone());
    }

    results
}

// ──────────────────────────────────────────────
//  Module: tensor_spine
// ──────────────────────────────────────────────

/// A combined spatial state — Pythagorean48 direction + Eisenstein position.
///
/// The `TensorSpine` represents a point in 2D space with an exact direction:
///
/// - **direction**: exact unit vector via Pythagorean48 (zero drift)
/// - **position**: hexagonal lattice point via Eisenstein quantization
/// - **spacing**: lattice spacing for Euclidean ↔ Hex conversion
///
/// This is the fundamental spatial primitive for ternary systems —
/// deterministic, drift-free, and embeddable.
#[derive(Debug, Clone)]
pub struct TensorSpine {
    /// Exact direction (zero-drift unit vector)
    pub direction: Direction48,
    /// Position on the hexagonal (A₂) lattice
    pub position: HexPoint,
    /// Lattice spacing
    pub spacing: f64,
}

impl TensorSpine {
    /// Create a new TensorSpine from a direction and hexagonal position.
    pub fn new(direction: Direction48, position: HexPoint, spacing: f64) -> Self {
        Self {
            direction,
            position,
            spacing,
        }
    }

    /// Create a TensorSpine at the origin, facing the identity direction.
    pub fn origin(spacing: f64) -> Self {
        Self {
            direction: Direction48::identity(),
            position: HexPoint::origin(),
            spacing,
        }
    }

    /// Compose this spine's direction with another direction exactly.
    ///
    /// Returns a new TensorSpine with the composed direction at the same position.
    pub fn compose_rotation(&self, other: &Direction48) -> Self {
        Self {
            direction: self.direction.compose(other),
            position: self.position,
            spacing: self.spacing,
        }
    }

    /// Rotate the spine by the i-th Pythagorean48 direction.
    pub fn rotate(&self, index: u32) -> Self {
        let dir = direction_db().get(index as usize);
        self.compose_rotation(dir)
    }

    /// Translate (move) the spine to the nearest Eisenstein lattice point
    /// from the given Euclidean offset.
    pub fn translate(&self, dx: f64, dy: f64) -> Self {
        let (cx, cy) = self.position.to_euclidean(self.spacing);
        let new_pos = HexPoint::from_euclidean(cx + dx, cy + dy, self.spacing);
        Self {
            direction: self.direction.clone(),
            position: new_pos,
            spacing: self.spacing,
        }
    }

    /// Set a new position directly from Eisenstein coordinates.
    pub fn set_position(&self, a: i64, b: i64) -> Self {
        Self {
            direction: self.direction.clone(),
            position: HexPoint::new(a, b),
            spacing: self.spacing,
        }
    }

    /// Get the Euclidean (x, y) position of this spine.
    pub fn euclidean_position(&self) -> (f64, f64) {
        self.position.to_euclidean(self.spacing)
    }

    /// Get the exact direction magnitude squared (should always be 1.0).
    pub fn direction_magnitude_sq(&self) -> f64 {
        self.direction.magnitude_sq()
    }

    /// Get a deterministic state hash for this spine.
    pub fn state_hash(&self) -> String {
        format!(
            "TS(d={}/{}@{}|p=({},{}))",
            self.direction.a,
            self.direction.b,
            self.direction.c,
            self.position.a,
            self.position.b,
        )
    }

    /// Number of unique Pythagorean48 directions available.
    pub fn num_directions() -> usize {
        pythagorean48_count()
    }
}

/// Query the hexagonal lattice for the nearest point to any (x, y).
///
/// Returns the HexPoint on the Eisenstein A₂ lattice closest to the
/// given Euclidean coordinates.
pub fn hex_grid_query(x: f64, y: f64, spacing: f64) -> HexPoint {
    HexPoint::from_euclidean(x, y, spacing)
}

/// Count of primitive + non-primitive Pythagorean triples with c ≤ 100.
pub fn pythagorean_triple_count() -> usize {
    enumerate_pythagorean_triples(100).len()
}

// ──────────────────────────────────────────────
//  Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// ZERO DRIFT PROOF
    ///
    /// Chain 1000 rotations using exact BigInt arithmetic.
    /// At every step, the direction must satisfy a² + b² = c² exactly.
    /// This proves that Pythagorean48 gives zero drift — the magnitude
    /// stays exactly 1.0 regardless of how many rotations are chained.
    #[test]
    fn test_zero_drift_proof() {
        let dirs = all_directions();
        assert!(dirs.len() >= 100, "should have ≥100 unique directions");

        // Deterministic sequence through all directions
        let mut current = Direction48::identity();
        let chain_len = 1000;

        for i in 0..chain_len {
            let step = &dirs[i % dirs.len()];
            current = current.compose(step);

            // ZERO DRIFT ASSERTION:
            // Every composed direction MUST be a valid Pythagorean triple.
            // This is the core proof that Pythagorean48 gives zero drift.
            assert!(
                current.is_valid(),
                "ZERO DRIFT VIOLATED at step {}: direction ({},{},{}) is INVALID.\n\
                 This proves BigInt composition preserves exactness.\n\
                 a²+b² = {}, c² = {}",
                i + 1,
                current.a,
                current.b,
                current.c,
                (&current.a).pow(2u32) + (&current.b).pow(2u32),
                (&current.c).pow(2u32),
            );
        }

        // Verify final magnitude is still exactly 1.0
        let mag = current.magnitude_sq();
        let diff = (mag - 1.0).abs();
        assert!(
            diff < 1e-12,
            "After {} rotations, magnitude drifted by {:.2e} — should be ZERO.",
            chain_len, diff
        );

        // Also verify by is_valid() as the definitive check
        assert!(
            current.is_valid(),
            "After {} rotations, the direction is not a valid Pythagorean triple.",
            chain_len
        );
    }

    /// Test that HexPoint roundtrips correctly for various positions.
    #[test]
    fn test_hex_grid_query() {
        let test_cases: Vec<(f64, f64)> = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (3.0, 2.0),
            (-1.5, 2.7),
            (10.0, -5.0),
            (-3.0, -1.0),
            (7.5, 0.8660254),
        ];

        let spacing = 1.0;

        for &(x, y) in &test_cases {
            let hp = hex_grid_query(x, y, spacing);
            let (qx, qy) = hp.to_euclidean(spacing);
            let dx = x - qx;
            let dy = y - qy;
            let dist = (dx * dx + dy * dy).sqrt();

            // Covering radius of hexagonal lattice = sqrt(3)/3 ≈ 0.577
            assert!(
                dist < 0.6,
                "hex_grid_query({}, {}) → ({}, {}) → ({:.6}, {:.6}) dist {:.6}",
                x, y, hp.a, hp.b, qx, qy, dist
            );
        }
    }

    /// Test that HexPoint → Euclidean → HexPoint roundtrips exactly.
    #[test]
    fn test_hex_roundtrip() {
        let points = [
            HexPoint::new(0, 0),
            HexPoint::new(1, 0),
            HexPoint::new(0, 1),
            HexPoint::new(3, -2),
            HexPoint::new(-5, 7),
            HexPoint::new(100, -200),
        ];

        let spacing = 1.5;
        for &p in &points {
            let (x, y) = p.to_euclidean(spacing);
            let q = hex_grid_query(x, y, spacing);
            assert_eq!(
                p, q,
                "Roundtrip failed: {:?} → ({}, {}) → {:?}",
                p, x, y, q
            );
        }
    }

    /// Test that all stored directions are valid Pythagorean triples.
    #[test]
    fn test_all_directions_are_unit() {
        let dirs = all_directions();
        assert!(!dirs.is_empty(), "should have at least one direction");

        for (i, dir) in dirs.iter().enumerate() {
            assert!(
                dir.is_valid(),
                "Direction {} ({:?}) is not a valid Pythagorean triple",
                i, dir
            );
        }
    }

    /// Test that composing any two directions preserves magnitude = 1.
    #[test]
    fn test_direction_composition_magnitude() {
        let dirs = all_directions();
        assert!(dirs.len() >= 10);

        // Test all pairs of first 10 directions
        let n = 10.min(dirs.len());
        for i in 0..n {
            for j in 0..n {
                let composed = dirs[i].compose(&dirs[j]);
                assert!(
                    composed.is_valid(),
                    "Composition {:?} ∘ {:?} gave invalid triple {:?}",
                    dirs[i], dirs[j], composed
                );
            }
        }
    }

    /// Test that identity composition is a true no-op.
    #[test]
    fn test_identity_composition() {
        let dirs = all_directions();
        let id = Direction48::identity();

        for dir in dirs.iter().take(20) {
            let composed = dir.compose(&id);
            // Since both the DB and compose normalize by gcd,
            // and identity has explicit (1, 0, 1),
            // compose with identity should leave any normalized direction unchanged.
            assert!(
                composed.eq_normalized(dir),
                "Direction {:?} composed with identity should be itself, got {:?}",
                dir, composed
            );
        }
    }

    /// Test Pythagorean triple enumeration.
    #[test]
    fn test_pythagorean_triple_count() {
        let triples = enumerate_pythagorean_triples(100);
        assert!(
            triples.len() >= 30,
            "Expected ≥30 Pythagorean triples with c ≤ 100, found {}",
            triples.len()
        );

        for &(a, b, c) in &triples {
            assert_eq!(
                a * a + b * b,
                c * c,
                "({}, {}, {}) is not a valid Pythagorean triple",
                a,
                b,
                c
            );
        }

        // Classic triples
        assert!(
            triples.contains(&(3, 4, 5)),
            "Expected (3,4,5) to be found"
        );
        assert!(
            triples.contains(&(5, 12, 13)),
            "Expected (5,12,13) to be found"
        );
    }

    /// Test that the database contains recognizable directions.
    #[test]
    fn test_database_contains_expected_directions() {
        let dirs = all_directions();

        // Should have direction pointing up (angle ≈ 90°)
        // (0, 1, 1) is not generated by Euclid's formula for primitive triples,
        // but some direction should be very close to 90°.
        let has_near_90 = dirs.iter().any(|d| {
            let angle = d.angle_deg().rem_euclid(360.0);
            (angle - 90.0).abs() < 10.0 || (angle - 270.0).abs() < 10.0
        });
        assert!(has_near_90,
            "Expected a direction near 90° or 270°. Found none.");

        // Should have (3,4,5) or (4,3,5) in some symmetry
        let has_345 = dirs.iter().any(|d| {
            d.a.abs() == BigInt::from(3) && d.b.abs() == BigInt::from(4) && d.c == BigInt::from(5)
        }) || dirs.iter().any(|d| {
            d.a.abs() == BigInt::from(4) && d.b.abs() == BigInt::from(3) && d.c == BigInt::from(5)
        });
        assert!(has_345, "Expected (3,4,5) direction in database");

        // Should have (5,12,13)
        let has_51213 = dirs.iter().any(|d| {
            d.a.abs() == BigInt::from(5) && d.b.abs() == BigInt::from(12) && d.c == BigInt::from(13)
        }) || dirs.iter().any(|d| {
            d.a.abs() == BigInt::from(12) && d.b.abs() == BigInt::from(5) && d.c == BigInt::from(13)
        });
        assert!(has_51213, "Expected (5,12,13) direction in database");
    }

    /// Test TensorSpine integration.
    #[test]
    fn test_tensor_spine_integration() {
        let spacing = 1.0;

        // Create spine at origin
        let spine = TensorSpine::origin(spacing);
        assert_eq!(spine.position, HexPoint::origin());
        assert_eq!(spine.direction, Direction48::identity());

        // Compose a rotation
        let dirs = all_directions();
        let rotated = spine.compose_rotation(&dirs[1]);
        assert_eq!(rotated.position, HexPoint::origin());
        assert!(rotated.direction.is_valid());

        // Translate
        let translated = spine.translate(2.0, 3.0);
        let expected = HexPoint::from_euclidean(2.0, 3.0, spacing);
        assert_eq!(
            translated.position, expected,
            "Translation should put spine at nearest hex point"
        );

        // Chain: compose + translate
        let dir = direction_db().get(5);
        let chained = spine
            .compose_rotation(dir)
            .translate(1.0, -1.0)
            .rotate(3);

        assert!(
            chained.direction.is_valid(),
            "Chained operations should preserve direction validity"
        );

        // State hash should be deterministic
        assert_eq!(
            rotated.state_hash(),
            rotated.state_hash(),
            "state_hash should be deterministic"
        );
    }

    /// Test that nearest_direction returns a valid direction.
    #[test]
    fn test_nearest_direction() {
        let angles = [0.0, 45.0, 90.0, 180.0, 270.0, 360.0, 13.7, -47.3];
        for &a in &angles {
            let dir = nearest_direction(a);
            assert!(
                dir.is_valid(),
                "nearest_direction({}) returned invalid triple",
                a
            );
        }
    }

    /// Test that chain_rotations produces zero drift.
    #[test]
    fn test_chain_rotations_zero_drift() {
        let results = chain_rotations(100, 42);
        assert_eq!(results.len(), 100);

        for (i, dir) in results.iter().enumerate() {
            assert!(
                dir.is_valid(),
                "Step {}: direction is not a valid triple — zero drift violated",
                i + 1
            );
        }
    }

    /// Test hex_grid_query with exact lattice sites.
    #[test]
    fn test_hex_grid_lattice_sites() {
        let spacing = 2.0;

        let sites = [
            HexPoint::new(0, 0),
            HexPoint::new(3, 1),
            HexPoint::new(-2, 5),
        ];

        for &site in &sites {
            let (ex, ey) = site.to_euclidean(spacing);
            let result = hex_grid_query(ex, ey, spacing);
            assert_eq!(
                site, result,
                "Lattice site {:?} → ({}, {}) should map to itself, got {:?}",
                site, ex, ey, result
            );
        }
    }

    /// Test TensorSpine.num_directions() returns a positive count.
    #[test]
    fn test_tensor_spine_num_directions() {
        let n = TensorSpine::num_directions();
        assert!(n >= 100, "expected ≥100 directions, got {}", n);
    }

    /// Test that compose is associative: (a ∘ b) ∘ c = a ∘ (b ∘ c).
    #[test]
    fn test_composition_associativity() {
        let dirs = all_directions();
        let n = 5.min(dirs.len());

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let a = dirs[i].compose(&dirs[j]).compose(&dirs[k]);
                    let b = dirs[i].compose(&dirs[j].compose(&dirs[k]));
                    assert!(
                        a.eq_normalized(&b),
                        "Associativity violated for ({i},{j},{k})"
                    );
                }
            }
        }
    }

    /// Test that the database has reasonable angular coverage.
    #[test]
    fn test_angular_coverage() {
        let dirs = all_directions();

        // Compute angular gaps
        let mut angles: Vec<f64> = dirs.iter().map(|d| d.angle_deg().rem_euclid(360.0)).collect();
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = angles.len();
        let mut gaps = Vec::new();
        for i in 0..n {
            let gap = (angles[(i + 1) % n] - angles[i] + 360.0) % 360.0;
            if gap > 0.0 {
                gaps.push(gap);
            }
        }

        assert!(!gaps.is_empty(), "should have angular gaps");

        let mean_gap: f64 = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let max_gap = gaps.iter().cloned().fold(0.0_f64, f64::max);

        // With 128+ directions, mean gap should be about 2.8°
        assert!(
            mean_gap < 10.0,
            "Mean angular gap {:.2}° is too large — expected < 10°",
            mean_gap
        );
        assert!(
            max_gap < 60.0,
            "Max angular gap {:.2}° is too large — expected < 60°",
            max_gap
        );
    }
}
