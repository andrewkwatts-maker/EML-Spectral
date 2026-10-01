//! Fixed, published invariants of the E8 root system and the Leech lattice.
//!
//! The crate exposes the two minimum norms as constants, so the tests build the
//! actual root system and check the constants against it rather than restating
//! them.

use std::collections::HashSet;

use eml_spectral_core::lattice::{e8_min_norm_squared, e8_norms_squared_n, leech_min_norm_squared};

const TOL: f64 = 1e-12;
const E8_DIM: usize = 8;
const LEECH_DIM: usize = 24;
const E8_ROOT_COUNT: usize = 240;

/// The 240 roots of E8: the 112 vectors `(+/-1, +/-1, 0^6)` in every position,
/// and the 128 vectors `(+/-1/2)^8` carrying an even number of minus signs.
fn e8_roots() -> Vec<Vec<f64>> {
    let mut roots: Vec<Vec<f64>> = Vec::with_capacity(E8_ROOT_COUNT);

    for i in 0..E8_DIM {
        for j in (i + 1)..E8_DIM {
            for si in [1.0f64, -1.0] {
                for sj in [1.0f64, -1.0] {
                    let mut r = vec![0.0; E8_DIM];
                    r[i] = si;
                    r[j] = sj;
                    roots.push(r);
                }
            }
        }
    }
    debug_assert_eq!(roots.len(), 112, "there are 4 * C(8,2) integer roots in E8");

    for mask in 0u32..(1u32 << E8_DIM) {
        if mask.count_ones() % 2 != 0 {
            continue;
        }
        let r: Vec<f64> = (0..E8_DIM)
            .map(|k| if mask & (1 << k) == 0 { 0.5 } else { -0.5 })
            .collect();
        roots.push(r);
    }
    debug_assert_eq!(
        roots.len(),
        E8_ROOT_COUNT,
        "the half-integer roots complete the 240"
    );
    debug_assert!(
        roots.iter().all(|r| r.len() == E8_DIM),
        "every E8 root has eight coordinates"
    );
    roots
}

/// Exact key for a root whose coordinates are always integers or half-integers.
fn key(point: &[f64]) -> Vec<i64> {
    debug_assert!(!point.is_empty(), "a lattice point has coordinates");
    let scaled: Vec<i64> = point.iter().map(|c| (c * 2.0).round() as i64).collect();
    debug_assert!(
        scaled
            .iter()
            .zip(point.iter())
            .all(|(s, c)| ((*s as f64) / 2.0 - c).abs() < TOL),
        "doubling must be exact for integer and half-integer coordinates"
    );
    scaled
}

/// A Leech vector in the standard 1/sqrt(8) normalisation, from integer
/// coordinates.
fn leech_scaled(coords: [i32; LEECH_DIM]) -> Vec<f64> {
    let scale = 1.0 / (8.0f64).sqrt();
    let v: Vec<f64> = coords.iter().map(|c| f64::from(*c) * scale).collect();
    debug_assert_eq!(v.len(), LEECH_DIM, "a Leech point has 24 coordinates");
    debug_assert!(
        v.iter().all(|c| c.is_finite()),
        "scaling integer coordinates cannot leave the finite range"
    );
    v
}

#[test]
fn the_e8_root_system_has_exactly_240_distinct_roots() {
    let roots = e8_roots();
    assert_eq!(roots.len(), E8_ROOT_COUNT, "E8 has 240 roots");

    let distinct: HashSet<Vec<i64>> = roots.iter().map(|r| key(r)).collect();
    assert_eq!(
        distinct.len(),
        E8_ROOT_COUNT,
        "the 240 roots must all be distinct"
    );
}

#[test]
fn every_e8_root_has_the_published_minimum_squared_norm_of_two() {
    let roots = e8_roots();
    let norms = e8_norms_squared_n(roots.clone()).expect("well-formed points must have norms");
    assert_eq!(norms.len(), roots.len(), "one squared norm per root");

    let minimum = e8_min_norm_squared();
    assert_eq!(minimum, 2.0, "the E8 minimum squared norm is 2");
    for (i, n) in norms.iter().enumerate() {
        assert!(
            (n - minimum).abs() <= TOL,
            "root {i} has squared norm {n}, expected {minimum}"
        );
    }
}

#[test]
fn the_e8_root_set_is_closed_under_negation() {
    let roots = e8_roots();
    let present: HashSet<Vec<i64>> = roots.iter().map(|r| key(r)).collect();
    for r in &roots {
        let negated: Vec<f64> = r.iter().map(|c| -c).collect();
        assert!(
            present.contains(&key(&negated)),
            "the negation of {r:?} must also be a root"
        );
    }
}

#[test]
fn e8_roots_have_integer_inner_products_of_absolute_value_at_most_two() {
    // A root system over an integral lattice: every pairwise inner product is a
    // whole number, and 2 is reached only by a root with itself.
    let roots = e8_roots();
    for (i, a) in roots.iter().enumerate() {
        for b in roots.iter().skip(i) {
            let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let rounded = dot.round();
            assert!(
                (dot - rounded).abs() <= TOL,
                "inner product {dot} is not an integer"
            );
            assert!(
                rounded.abs() <= 2.0,
                "inner product {rounded} exceeds the root-length bound"
            );
        }
    }
}

#[test]
fn every_shape_of_shortest_leech_vector_has_squared_norm_four() {
    // The three orbits of minimal vectors, in the normalisation where Leech
    // coordinates carry a factor 1/sqrt(8):
    //   (+/-4, +/-4, 0^22), (+/-2^8, 0^16) on an octad, and (-/+3, +/-1^23).
    let mut two_fours = [0i32; LEECH_DIM];
    two_fours[0] = 4;
    two_fours[7] = -4;

    let mut octad = [0i32; LEECH_DIM];
    for (i, c) in octad.iter_mut().enumerate().take(8) {
        *c = if i % 3 == 0 { -2 } else { 2 };
    }

    let mut threes = [1i32; LEECH_DIM];
    threes[0] = -3;

    let points = vec![
        leech_scaled(two_fours),
        leech_scaled(octad),
        leech_scaled(threes),
    ];
    let norms = e8_norms_squared_n(points).expect("well-formed points must have norms");

    let minimum = leech_min_norm_squared();
    assert_eq!(minimum, 4.0, "the Leech minimum squared norm is 4");
    for (i, n) in norms.iter().enumerate() {
        assert!(
            (n - minimum).abs() <= TOL,
            "Leech shape {i} has squared norm {n}, expected {minimum}"
        );
    }
}

#[test]
fn the_leech_minimum_is_strictly_deeper_than_the_e8_minimum() {
    let e8 = e8_min_norm_squared();
    let leech = leech_min_norm_squared();
    assert!(
        leech > e8,
        "the Leech lattice has no vectors as short as an E8 root"
    );
    assert_eq!(leech, 2.0 * e8, "4 is twice 2 in these normalisations");
}

#[test]
fn the_batch_squared_norm_agrees_with_the_scalar_sum_of_squares() {
    let points = vec![
        vec![1.0, -2.0, 3.0, 0.5, 0.0, -0.25, 4.0, 1.0],
        vec![0.5; LEECH_DIM],
        vec![1e-8, -1e-8],
        vec![3.0],
    ];
    let norms = e8_norms_squared_n(points.clone()).expect("well-formed points must have norms");
    assert_eq!(norms.len(), points.len(), "one squared norm per point");
    for (i, p) in points.iter().enumerate() {
        let expected: f64 = p.iter().map(|c| c * c).sum();
        assert!(
            (norms[i] - expected).abs() <= TOL * expected.max(1.0),
            "point {i}: squared norm was {}, expected {expected}",
            norms[i]
        );
    }
}
