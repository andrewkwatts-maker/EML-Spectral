//! Properties that define a geometric (Clifford) product.
//!
//! The defining relation is `e_i e_j + e_j e_i = 2 g_ij`, which splits into
//! "a basis vector squares to its signature entry" and "distinct basis vectors
//! anticommute". Both are checked here, along with associativity -- the
//! property that separates a Clifford algebra from the octonions.

use eml_spectral_core::clifford::{geometric_product_n, MAX_CLIFFORD_DIM};

const TOL: f64 = 1e-12;

const EUCLIDEAN_3: [i8; 3] = [1, 1, 1];
const SPACETIME: [i8; 4] = [1, -1, -1, -1];
const PLANE: [i8; 2] = [1, 1];

/// Geometric product of a single pair, through the batch entry point.
fn product(a: &[f64], b: &[f64], signature: &[i8]) -> Vec<f64> {
    debug_assert!(
        !signature.is_empty() && signature.len() <= MAX_CLIFFORD_DIM,
        "test signatures stay inside the supported range"
    );
    debug_assert!(
        a.len() == 1usize << signature.len() && b.len() == 1usize << signature.len(),
        "test multivectors carry one coefficient per blade"
    );
    let mut out = geometric_product_n(vec![a.to_vec()], vec![b.to_vec()], signature.to_vec())
        .expect("well-formed multivectors must multiply");
    assert_eq!(out.len(), 1, "one product per input pair");
    out.remove(0)
}

/// The blade `e_I` selected by the bitmask `mask`, in an algebra with `n`
/// generators.
fn blade(n: usize, mask: usize) -> Vec<f64> {
    debug_assert!(
        (1..=MAX_CLIFFORD_DIM).contains(&n),
        "n names the generators"
    );
    debug_assert!(
        mask < 1usize << n,
        "a blade mask indexes the generators of n"
    );
    let mut v = vec![0.0; 1usize << n];
    v[mask] = 1.0;
    v
}

/// A grade-1 element, i.e. an ordinary vector, from its `n` coordinates.
fn vector(coords: &[f64]) -> Vec<f64> {
    let n = coords.len();
    debug_assert!(
        (1..=MAX_CLIFFORD_DIM).contains(&n),
        "a vector has one coordinate per generator"
    );
    let mut v = vec![0.0; 1usize << n];
    for (i, c) in coords.iter().enumerate() {
        v[1usize << i] = *c;
    }
    debug_assert_eq!(v[0], 0.0, "a grade-1 element has no scalar part");
    v
}

fn scalar(n: usize, value: f64) -> Vec<f64> {
    debug_assert!(
        (1..=MAX_CLIFFORD_DIM).contains(&n),
        "n names the generators"
    );
    let mut v = vec![0.0; 1usize << n];
    v[0] = value;
    debug_assert_eq!(v.len(), 1usize << n, "a multivector has 2^n coefficients");
    v
}

fn assert_close(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = e.abs().max(1.0);
        assert!(
            (a - e).abs() <= TOL * scale,
            "{what}: blade {i} was {a}, expected {e}"
        );
    }
}

#[test]
fn clifford_basis_vectors_square_to_their_signature_entry() {
    for signature in [SPACETIME.as_slice(), EUCLIDEAN_3.as_slice()] {
        let n = signature.len();
        for (i, &sign) in signature.iter().enumerate() {
            let e_i = blade(n, 1usize << i);
            let square = product(&e_i, &e_i, signature);
            assert_close(
                &square,
                &scalar(n, f64::from(sign)),
                "e_i squared equals the signature entry",
            );
        }
    }
}

#[test]
fn clifford_orthogonal_basis_vectors_anticommute() {
    for signature in [SPACETIME.as_slice(), EUCLIDEAN_3.as_slice()] {
        let n = signature.len();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let ij = product(&blade(n, 1usize << i), &blade(n, 1usize << j), signature);
                let ji = product(&blade(n, 1usize << j), &blade(n, 1usize << i), signature);
                let negated: Vec<f64> = ji.iter().map(|c| -c).collect();
                assert_close(&ij, &negated, "e_i e_j = -(e_j e_i)");
                assert_ne!(
                    ij, ji,
                    "distinct generators must not commute (i={i}, j={j})"
                );
            }
        }
    }
}

#[test]
fn clifford_vectors_satisfy_the_defining_anticommutator() {
    // u v + v u = 2 <u, v>, where the inner product uses the signature.
    let signature = SPACETIME.as_slice();
    let u_coords = [1.5, -2.0, 0.5, 3.0];
    let v_coords = [-0.5, 1.0, 2.5, -1.5];
    let u = vector(&u_coords);
    let v = vector(&v_coords);

    let uv = product(&u, &v, signature);
    let vu = product(&v, &u, signature);
    let sum: Vec<f64> = uv.iter().zip(vu.iter()).map(|(a, b)| a + b).collect();

    let inner: f64 = signature
        .iter()
        .zip(u_coords.iter().zip(v_coords.iter()))
        .map(|(&s, (a, b))| f64::from(s) * a * b)
        .sum();
    assert_close(
        &sum,
        &scalar(signature.len(), 2.0 * inner),
        "uv + vu = 2<u,v>",
    );
}

#[test]
fn clifford_vector_squared_is_its_quadratic_form() {
    let signature = SPACETIME.as_slice();
    let coords = [2.0, 1.0, -3.0, 0.5];
    let v = vector(&coords);
    let expected: f64 = signature
        .iter()
        .zip(coords.iter())
        .map(|(&s, c)| f64::from(s) * c * c)
        .sum();
    assert_close(
        &product(&v, &v, signature),
        &scalar(signature.len(), expected),
        "v squared is the quadratic form of v",
    );
}

#[test]
fn clifford_geometric_product_is_associative_unlike_the_octonion_product() {
    let signature = EUCLIDEAN_3.as_slice();
    let a = vec![1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 1.5, 0.25];
    let b = vec![-2.0, 1.0, 0.5, 2.0, -1.5, 3.0, 0.75, -1.0];
    let c = vec![0.5, -1.5, 2.0, 1.0, -0.25, 0.5, -3.0, 2.0];

    let left = product(&product(&a, &b, signature), &c, signature);
    let right = product(&a, &product(&b, &c, signature), signature);
    assert_close(&left, &right, "(ab)c = a(bc)");
}

#[test]
fn clifford_unit_scalar_is_a_two_sided_identity() {
    let signature = SPACETIME.as_slice();
    let n = signature.len();
    let one = scalar(n, 1.0);
    let a: Vec<f64> = (0..(1usize << n)).map(|i| (i as f64) * 0.5 - 2.0).collect();
    assert_close(&product(&one, &a, signature), &a, "1 * a");
    assert_close(&product(&a, &one, signature), &a, "a * 1");
}

#[test]
fn clifford_pseudoscalar_squares_to_minus_one_in_the_euclidean_plane_and_space() {
    // e_0 e_1 squares to -1 in Cl(2,0): the even subalgebra is the complex
    // numbers.
    let e01 = blade(2, 0b11);
    assert_close(
        &product(&e01, &e01, PLANE.as_slice()),
        &scalar(2, -1.0),
        "(e0 e1)^2 in Cl(2,0)",
    );

    // e_0 e_1 e_2 squares to -1 in Cl(3,0).
    let e012 = blade(3, 0b111);
    assert_close(
        &product(&e012, &e012, EUCLIDEAN_3.as_slice()),
        &scalar(3, -1.0),
        "(e0 e1 e2)^2 in Cl(3,0)",
    );
}

#[test]
fn clifford_spacetime_pseudoscalar_squares_to_minus_one() {
    // In Cl(1,3) the volume element e_0 e_1 e_2 e_3 squares to -1.
    let i = blade(4, 0b1111);
    assert_close(
        &product(&i, &i, SPACETIME.as_slice()),
        &scalar(4, -1.0),
        "(e0 e1 e2 e3)^2 in Cl(1,3)",
    );
}

#[test]
fn clifford_batch_product_agrees_pairwise_with_single_products() {
    let signature = SPACETIME.to_vec();
    let n = signature.len();
    let total = 1usize << n;
    let a_batch: Vec<Vec<f64>> = (0..5)
        .map(|k| (0..total).map(|i| ((i + k) as f64) * 0.25 - 1.0).collect())
        .collect();
    let b_batch: Vec<Vec<f64>> = (0..5)
        .map(|k| {
            (0..total)
                .map(|i| 2.0 - ((i * 2 + k) as f64) * 0.125)
                .collect()
        })
        .collect();

    let batched = geometric_product_n(a_batch.clone(), b_batch.clone(), signature.clone())
        .expect("well-formed batches must multiply");
    assert_eq!(batched.len(), a_batch.len(), "one product per input pair");

    for (k, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
        assert_eq!(
            batched[k],
            product(a, b, &signature),
            "batch row {k} must be bit-identical to the single product"
        );
    }
}
