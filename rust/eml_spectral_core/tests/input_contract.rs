//! The refusal paths.
//!
//! Structurally invalid input must come back as a `SpectralError`, never as a
//! panic (which PyO3 would surface to Python as the near-uncatchable
//! `PanicException`), and never as a silently shortened or padded result.
//!
//! Non-finite *values* are a separate case and are deliberately not refused:
//! the last tests here pin that documented behaviour so it cannot drift into a
//! panic either.

use eml_spectral_core::clifford::{geometric_product_n, MAX_CLIFFORD_DIM};
use eml_spectral_core::lattice::e8_norms_squared_n;
use eml_spectral_core::octonion::{octonion_mul, octonion_mul_n, octonion_norm_n};
use eml_spectral_core::spectral::{spectral_flow_batch, spectral_flow_n, MAX_FLOW_STEPS};
use eml_spectral_core::{add_n, SpectralError};

fn octonion(fill: f64) -> Vec<f64> {
    debug_assert!(fill.is_finite(), "test fixtures start from finite values");
    let v = vec![fill; 8];
    debug_assert_eq!(v.len(), 8, "an octonion fixture has eight components");
    v
}

#[test]
fn octonion_multiply_refuses_an_operand_that_is_not_eight_components() {
    for bad_len in [0usize, 1, 7, 9, 16] {
        let bad = vec![1.0; bad_len];
        assert_eq!(
            octonion_mul(bad.clone(), octonion(1.0)),
            Err(SpectralError::WrongLength {
                what: "left octonion",
                expected: 8,
                actual: bad_len,
            }),
            "a {bad_len}-component left operand must be refused, not padded"
        );
        assert_eq!(
            octonion_mul(octonion(1.0), bad),
            Err(SpectralError::WrongLength {
                what: "right octonion",
                expected: 8,
                actual: bad_len,
            }),
            "a {bad_len}-component right operand must be refused, not truncated"
        );
    }
}

#[test]
fn octonion_batch_multiply_refuses_batches_of_different_lengths() {
    let a = vec![octonion(1.0); 3];
    let b = vec![octonion(2.0); 2];
    assert_eq!(
        octonion_mul_n(a, b),
        Err(SpectralError::LengthMismatch {
            what: "octonion batches",
            left: 3,
            right: 2,
        }),
        "the shorter batch must not silently decide the result length"
    );
}

#[test]
fn octonion_batch_multiply_reports_which_element_is_malformed() {
    let a = vec![octonion(1.0), vec![1.0, 2.0, 3.0], octonion(1.0)];
    let b = vec![octonion(1.0); 3];
    assert_eq!(
        octonion_mul_n(a, b),
        Err(SpectralError::BatchElementWrongLength {
            what: "left batch",
            index: 1,
            expected: 8,
            actual: 3,
        })
    );
}

#[test]
fn octonion_batch_norm_refuses_a_malformed_element() {
    assert_eq!(
        octonion_norm_n(vec![octonion(1.0), Vec::new()]),
        Err(SpectralError::BatchElementWrongLength {
            what: "norm batch",
            index: 1,
            expected: 8,
            actual: 0,
        }),
        "an empty vector has a defined Euclidean norm but is not an octonion"
    );
}

#[test]
fn empty_octonion_batches_multiply_to_an_empty_batch() {
    // Documented sentinel: nothing in, nothing out -- this is not an error.
    let out = octonion_mul_n(Vec::new(), Vec::new()).expect("empty batches are well formed");
    assert!(out.is_empty(), "no pairs means no products");
}

#[test]
fn the_geometric_product_refuses_a_signature_with_no_generators() {
    assert_eq!(
        geometric_product_n(vec![vec![1.0]], vec![vec![1.0]], Vec::new()),
        Err(SpectralError::EmptySignature),
        "there is no algebra to multiply in without generators"
    );
}

#[test]
fn the_geometric_product_refuses_a_signature_longer_than_the_blade_encoding() {
    let too_long = vec![1i8; MAX_CLIFFORD_DIM + 1];
    assert_eq!(
        geometric_product_n(vec![vec![1.0]], vec![vec![1.0]], too_long),
        Err(SpectralError::SignatureTooLong {
            requested: MAX_CLIFFORD_DIM + 1,
            max: MAX_CLIFFORD_DIM,
        }),
        "the 2^n allocation and the blade shift both have to stay bounded"
    );
}

#[test]
fn the_geometric_product_refuses_a_signature_entry_that_is_not_plus_or_minus_one() {
    for bad in [0i8, 2, -3] {
        let signature = vec![1i8, bad];
        assert_eq!(
            geometric_product_n(vec![vec![0.0; 4]], vec![vec![0.0; 4]], signature),
            Err(SpectralError::SignatureEntryNotUnit {
                index: 1,
                value: bad,
            }),
            "a diagonal metric entry of {bad} is not a signature"
        );
    }
}

#[test]
fn the_geometric_product_refuses_a_multivector_that_is_not_two_to_the_n_long() {
    let signature = vec![1i8, -1, -1];
    assert_eq!(
        geometric_product_n(vec![vec![0.0; 8]], vec![vec![0.0; 7]], signature),
        Err(SpectralError::BatchElementWrongLength {
            what: "right batch",
            index: 0,
            expected: 8,
            actual: 7,
        }),
        "Cl(1,2) has exactly 8 blades"
    );
}

#[test]
fn the_geometric_product_refuses_batches_of_different_lengths() {
    let signature = vec![1i8, 1];
    assert_eq!(
        geometric_product_n(vec![vec![0.0; 4]; 2], vec![vec![0.0; 4]; 5], signature),
        Err(SpectralError::LengthMismatch {
            what: "multivector batches",
            left: 2,
            right: 5,
        })
    );
}

#[test]
fn vector_addition_refuses_operands_of_different_lengths() {
    assert_eq!(
        add_n(vec![1.0, 2.0, 3.0], vec![1.0, 2.0]),
        Err(SpectralError::LengthMismatch {
            what: "addends",
            left: 3,
            right: 2,
        }),
        "a zip would have silently returned the shorter sum"
    );
}

#[test]
fn the_squared_norm_refuses_a_point_with_no_coordinates() {
    assert_eq!(
        e8_norms_squared_n(vec![vec![1.0, 2.0], Vec::new()]),
        Err(SpectralError::EmptyPoint {
            what: "norm batch",
            index: 1,
        }),
        "a zero-dimensional point would report 0.0 and look like the origin"
    );
}

#[test]
fn the_flow_refuses_a_step_count_it_cannot_allocate() {
    assert_eq!(
        spectral_flow_n(1.0, 1.0, MAX_FLOW_STEPS + 1),
        Err(SpectralError::TooManySteps {
            requested: MAX_FLOW_STEPS + 1,
            max: MAX_FLOW_STEPS,
        })
    );
    assert_eq!(
        spectral_flow_batch(vec![(1.0, 1.0)], usize::MAX),
        Err(SpectralError::TooManySteps {
            requested: usize::MAX,
            max: MAX_FLOW_STEPS,
        }),
        "the bound is checked before n_steps + 1 could overflow"
    );
}

#[test]
fn a_refusal_carries_an_ascii_message_naming_the_offending_shape() {
    let err = octonion_mul(vec![1.0; 3], vec![1.0; 8]).expect_err("a 3-vector is not an octonion");
    let message = err.to_string();
    assert!(
        message.contains("8") && message.contains("3"),
        "the message must name both the expected and the actual length: {message}"
    );
    assert!(
        message.is_ascii(),
        "messages must stay ASCII for every log sink: {message}"
    );
}

#[test]
fn non_finite_components_are_carried_through_rather_than_refused() {
    // Documented sentinel for values (as opposed to shapes): IEEE-754
    // propagation, and never a panic.
    let mut a = vec![1.0; 8];
    a[3] = f64::NAN;
    let product = octonion_mul(a.clone(), vec![1.0; 8]).expect("shape is valid, so it computes");
    assert!(
        product.iter().any(|c| c.is_nan()),
        "a NaN component must reach the product"
    );

    let norms = octonion_norm_n(vec![a]).expect("shape is valid, so it computes");
    assert!(norms[0].is_nan(), "a NaN component must reach the norm");

    let infinite =
        e8_norms_squared_n(vec![vec![f64::INFINITY, 1.0]]).expect("shape is valid, so it computes");
    assert!(
        infinite[0].is_infinite(),
        "an infinite coordinate must reach the squared norm"
    );
}
