//! Properties that define the octonions.
//!
//! These are the identities that separate a correct Fano-plane table from a
//! plausible-looking one: a wrong sign anywhere in the table breaks norm
//! multiplicativity or alternativity even though every product still returns
//! eight numbers.

use eml_spectral_core::add_n;
use eml_spectral_core::octonion::{octonion_mul, octonion_mul_n, octonion_norm_n, OCTONION_DIM};

/// Relative tolerance for products of order-100 magnitudes.
const TOL: f64 = 1e-12;

/// The basis unit e_i.
fn unit(i: usize) -> Vec<f64> {
    debug_assert!(i < OCTONION_DIM, "octonions have eight basis units");
    let mut v = vec![0.0; OCTONION_DIM];
    v[i] = 1.0;
    debug_assert_eq!(
        v.iter().filter(|c| **c != 0.0).count(),
        1,
        "a basis unit has exactly one non-zero component"
    );
    v
}

/// Product of two octonions, refusing to continue if the crate refuses input
/// the test built as well-formed.
fn mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), OCTONION_DIM, "test operands are full octonions");
    debug_assert_eq!(b.len(), OCTONION_DIM, "test operands are full octonions");
    octonion_mul(a.to_vec(), b.to_vec()).expect("well-formed octonions must multiply")
}

/// Euclidean norm through the crate's own batch entry point.
fn norm(a: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), OCTONION_DIM, "test operands are full octonions");
    let norms = octonion_norm_n(vec![a.to_vec()]).expect("a well-formed octonion must have a norm");
    debug_assert_eq!(norms.len(), 1, "one norm per input octonion");
    norms[0]
}

/// Conjugate: negate the seven imaginary components.
fn conjugate(a: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), OCTONION_DIM, "test operands are full octonions");
    let mut out = a.to_vec();
    for c in out.iter_mut().skip(1) {
        *c = -*c;
    }
    debug_assert_eq!(out[0], a[0], "conjugation fixes the real part");
    out
}

fn assert_close(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = e.abs().max(1.0);
        assert!(
            (a - e).abs() <= TOL * scale,
            "{what}: component {i} was {a}, expected {e}"
        );
    }
}

/// A pair of octonions with every component populated, so no zero-skip
/// shortcut in the kernel can hide a wrong table entry.
fn generic_pair() -> (Vec<f64>, Vec<f64>) {
    (
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![8.0, -7.0, 6.0, -5.0, 4.0, -3.0, 2.0, -1.0],
    )
}

#[test]
fn octonion_multiplication_is_not_associative() {
    // (e1 e2) e3 = e4 e3 = -e6, while e1 (e2 e3) = e1 e5 = +e6.
    let left = mul(&mul(&unit(1), &unit(2)), &unit(3));
    let right = mul(&unit(1), &mul(&unit(2), &unit(3)));

    assert_ne!(
        left, right,
        "octonion multiplication must not associate; an associative result here \
         means the Fano table has collapsed to a commutative or quaternionic one"
    );

    let mut minus_e6 = vec![0.0; OCTONION_DIM];
    minus_e6[6] = -1.0;
    assert_close(&left, &minus_e6, "(e1 e2) e3");
    assert_close(&right, &unit(6), "e1 (e2 e3)");
}

#[test]
fn octonion_multiplication_has_a_multiplicative_norm() {
    let (a, b) = generic_pair();
    let cases = [
        (a.clone(), b.clone()),
        (unit(3), unit(5)),
        (a.clone(), unit(7)),
        (vec![0.5, 0.0, 0.0, -0.25, 0.0, 1.5, 0.0, 0.0], b.clone()),
    ];
    for (i, (x, y)) in cases.iter().enumerate() {
        let product_norm = norm(&mul(x, y));
        let norm_product = norm(x) * norm(y);
        let scale = norm_product.abs().max(1.0);
        assert!(
            (product_norm - norm_product).abs() <= TOL * scale,
            "case {i}: |ab| was {product_norm}, |a||b| was {norm_product}"
        );
    }
}

#[test]
fn octonion_one_is_a_two_sided_identity() {
    let one = unit(0);
    let (a, _) = generic_pair();
    for i in 0..OCTONION_DIM {
        let e = unit(i);
        assert_close(&mul(&one, &e), &e, "1 * e_i");
        assert_close(&mul(&e, &one), &e, "e_i * 1");
    }
    assert_close(&mul(&one, &a), &a, "1 * a");
    assert_close(&mul(&a, &one), &a, "a * 1");
}

#[test]
fn octonion_imaginary_units_square_to_minus_one() {
    let mut minus_one = vec![0.0; OCTONION_DIM];
    minus_one[0] = -1.0;
    for i in 1..OCTONION_DIM {
        assert_close(&mul(&unit(i), &unit(i)), &minus_one, "e_i * e_i");
    }
}

#[test]
fn octonion_imaginary_units_anticommute() {
    for i in 1..OCTONION_DIM {
        for j in 1..OCTONION_DIM {
            if i == j {
                continue;
            }
            let ij = mul(&unit(i), &unit(j));
            let ji = mul(&unit(j), &unit(i));
            let negated: Vec<f64> = ji.iter().map(|c| -c).collect();
            assert_close(&ij, &negated, "e_i e_j = -(e_j e_i)");
            assert_ne!(
                ij, ji,
                "distinct imaginary units must not commute (i={i}, j={j})"
            );
        }
    }
}

#[test]
fn octonion_multiplication_is_alternative_even_though_it_is_not_associative() {
    // Alternativity is the weakened associativity the octonions do satisfy:
    // any subalgebra generated by two elements associates.
    let (a, b) = generic_pair();
    assert_close(
        &mul(&mul(&a, &a), &b),
        &mul(&a, &mul(&a, &b)),
        "(aa)b = a(ab)",
    );
    assert_close(
        &mul(&mul(&b, &a), &a),
        &mul(&b, &mul(&a, &a)),
        "(ba)a = b(aa)",
    );
}

#[test]
fn octonion_times_its_conjugate_is_its_squared_norm() {
    let (a, b) = generic_pair();
    for x in [&a, &b] {
        let expected_scalar = norm(x) * norm(x);
        let mut expected = vec![0.0; OCTONION_DIM];
        expected[0] = expected_scalar;
        assert_close(&mul(x, &conjugate(x)), &expected, "a * conj(a)");
        assert_close(&mul(&conjugate(x), x), &expected, "conj(a) * a");
    }
}

#[test]
fn octonion_multiplication_distributes_over_addition() {
    let (a, b) = generic_pair();
    let c = vec![-1.0, 0.5, 2.0, -2.5, 3.0, 0.0, -4.0, 1.25];
    let sum = add_n(b.clone(), c.clone()).expect("equal-length vectors must add");
    let left = mul(&a, &sum);
    let right = add_n(mul(&a, &b), mul(&a, &c)).expect("equal-length vectors must add");
    assert_close(&left, &right, "a(b + c) = ab + ac");
}

#[test]
fn octonion_batch_multiply_agrees_elementwise_with_the_scalar_version() {
    let (a, b) = generic_pair();
    let a_batch = vec![
        a.clone(),
        unit(1),
        unit(4),
        vec![0.0; OCTONION_DIM],
        b.clone(),
    ];
    let b_batch = vec![b.clone(), unit(2), a.clone(), a.clone(), unit(0)];

    let batched = octonion_mul_n(a_batch.clone(), b_batch.clone())
        .expect("well-formed batches must multiply");
    assert_eq!(batched.len(), a_batch.len(), "one product per input pair");

    for (i, (x, y)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
        let scalar = mul(x, y);
        assert_eq!(
            batched[i], scalar,
            "batch row {i} must be bit-identical to the scalar product"
        );
    }
}

#[test]
fn octonion_batch_norm_agrees_with_the_scalar_definition() {
    let (a, b) = generic_pair();
    let batch = vec![a.clone(), b.clone(), unit(5), vec![0.0; OCTONION_DIM]];
    let norms = octonion_norm_n(batch.clone()).expect("well-formed batch must have norms");
    assert_eq!(norms.len(), batch.len(), "one norm per input octonion");
    for (i, v) in batch.iter().enumerate() {
        let expected = v.iter().map(|c| c * c).sum::<f64>().sqrt();
        assert!(
            (norms[i] - expected).abs() <= TOL * expected.max(1.0),
            "norm {i} was {}, expected {expected}",
            norms[i]
        );
    }
}
