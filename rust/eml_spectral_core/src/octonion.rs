//! Octonion (Cayley-Dickson) multiplication via the Fano-plane table.
//!
//! ## Input contract
//!
//! An octonion is exactly [`OCTONION_DIM`] real components ordered
//! `(e0, e1, ..., e7)`, with `e0` the real unit. Any other length is refused
//! with a [`SpectralError`] -- never padded, never truncated.
//!
//! Non-finite components are not refused. They propagate through products and
//! norms under IEEE-754 rules (a NaN component yields NaN outputs); that is the
//! documented sentinel. No function here panics on non-finite input.

use crate::error::SpectralError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;

/// Number of real components in an octonion.
pub const OCTONION_DIM: usize = 8;

const FANO_LINES: [(usize, usize, usize); 7] = [
    (1, 2, 4),
    (2, 3, 5),
    (3, 4, 6),
    (4, 5, 7),
    (1, 5, 6),
    (2, 6, 7),
    (1, 3, 7),
];

const fn build_oct_table() -> [[(i8, usize); OCTONION_DIM]; OCTONION_DIM] {
    let mut t = [[(0i8, 0usize); OCTONION_DIM]; OCTONION_DIM];
    let mut i = 0;
    while i < OCTONION_DIM {
        t[0][i] = (1, i);
        t[i][0] = (1, i);
        i += 1;
    }
    let mut i = 1;
    while i < OCTONION_DIM {
        t[i][i] = (-1, 0);
        i += 1;
    }
    let mut li = 0;
    while li < FANO_LINES.len() {
        let (a, b, c) = FANO_LINES[li];
        t[a][b] = (1, c);
        t[b][a] = (-1, c);
        t[b][c] = (1, a);
        t[c][b] = (-1, a);
        t[c][a] = (1, b);
        t[a][c] = (-1, b);
        li += 1;
    }
    t
}

const OCT_TABLE: [[(i8, usize); OCTONION_DIM]; OCTONION_DIM] = build_oct_table();

/// Refuses anything that is not a well-formed octonion.
fn check_octonion(v: &[f64], what: &'static str) -> Result<(), SpectralError> {
    if v.len() != OCTONION_DIM {
        return Err(SpectralError::WrongLength {
            what,
            expected: OCTONION_DIM,
            actual: v.len(),
        });
    }
    Ok(())
}

/// Refuses a batch in which any element is not a well-formed octonion. The
/// scan is bounded by the batch length, which the caller already owns.
fn check_octonion_batch(batch: &[Vec<f64>], what: &'static str) -> Result<(), SpectralError> {
    let mut checked = 0usize;
    for (index, v) in batch.iter().enumerate() {
        if v.len() != OCTONION_DIM {
            return Err(SpectralError::BatchElementWrongLength {
                what,
                index,
                expected: OCTONION_DIM,
                actual: v.len(),
            });
        }
        checked += 1;
    }
    debug_assert_eq!(
        checked,
        batch.len(),
        "the Ok path must have visited every element of the batch"
    );
    debug_assert!(
        batch.iter().all(|v| v.len() == OCTONION_DIM),
        "the Ok path must leave every element at the octonion dimension"
    );
    Ok(())
}

/// Fano-plane product of two operands that a caller has already validated.
#[inline]
fn mul_components(a: &[f64], b: &[f64]) -> [f64; OCTONION_DIM] {
    debug_assert_eq!(
        a.len(),
        OCTONION_DIM,
        "left operand reaches the kernel only after length validation"
    );
    debug_assert_eq!(
        b.len(),
        OCTONION_DIM,
        "right operand reaches the kernel only after length validation"
    );
    let mut out = [0.0f64; OCTONION_DIM];
    for (i, &ai) in a.iter().enumerate().take(OCTONION_DIM) {
        if ai == 0.0 {
            continue;
        }
        let row = &OCT_TABLE[i];
        for (j, &bj) in b.iter().enumerate().take(OCTONION_DIM) {
            if bj == 0.0 {
                continue;
            }
            let (sign, k) = row[j];
            out[k] += f64::from(sign) * ai * bj;
        }
    }
    out
}

/// Multiply two octonions. Inputs are 8-element float vectors.
///
/// # Errors
///
/// [`SpectralError::WrongLength`] if either operand is not [`OCTONION_DIM`]
/// components long.
#[cfg_attr(feature = "python", pyfunction)]
pub fn octonion_mul(a: Vec<f64>, b: Vec<f64>) -> Result<Vec<f64>, SpectralError> {
    check_octonion(&a, "left octonion")?;
    check_octonion(&b, "right octonion")?;
    debug_assert_eq!(
        a.len(),
        OCTONION_DIM,
        "validation above admits only full octonions"
    );
    let out = mul_components(&a, &b).to_vec();
    debug_assert_eq!(
        out.len(),
        OCTONION_DIM,
        "the product of two octonions is an octonion"
    );
    Ok(out)
}

/// Batch octonion multiply, Rayon-parallel. Element `i` of the result is
/// `a_batch[i] * b_batch[i]`, identical to calling [`octonion_mul`] on the pair.
///
/// # Errors
///
/// [`SpectralError::LengthMismatch`] if the two batches differ in length --
/// the pairing is never silently truncated to the shorter one --
/// or [`SpectralError::BatchElementWrongLength`] if any element is not
/// [`OCTONION_DIM`] components long.
#[cfg_attr(feature = "python", pyfunction)]
pub fn octonion_mul_n(
    a_batch: Vec<Vec<f64>>,
    b_batch: Vec<Vec<f64>>,
) -> Result<Vec<Vec<f64>>, SpectralError> {
    if a_batch.len() != b_batch.len() {
        return Err(SpectralError::LengthMismatch {
            what: "octonion batches",
            left: a_batch.len(),
            right: b_batch.len(),
        });
    }
    check_octonion_batch(&a_batch, "left batch")?;
    check_octonion_batch(&b_batch, "right batch")?;
    let out: Vec<Vec<f64>> = a_batch
        .par_iter()
        .zip(b_batch.par_iter())
        .map(|(a, b)| mul_components(a, b).to_vec())
        .collect();
    debug_assert_eq!(
        out.len(),
        a_batch.len(),
        "the zip cannot drop a pair once both batches are the same length"
    );
    debug_assert!(
        out.iter().all(|v| v.len() == OCTONION_DIM),
        "every product must itself be a full octonion"
    );
    Ok(out)
}

/// Batch octonion norm, Rayon-parallel. Returns `sqrt(sum a_i^2)` per element.
///
/// # Errors
///
/// [`SpectralError::BatchElementWrongLength`] if any element is not
/// [`OCTONION_DIM`] components long.
#[cfg_attr(feature = "python", pyfunction)]
pub fn octonion_norm_n(batch: Vec<Vec<f64>>) -> Result<Vec<f64>, SpectralError> {
    check_octonion_batch(&batch, "norm batch")?;
    let out: Vec<f64> = batch
        .par_iter()
        .map(|v| v.iter().map(|c| c * c).sum::<f64>().sqrt())
        .collect();
    debug_assert_eq!(out.len(), batch.len(), "one norm per input octonion");
    // sqrt of a sum of squares is never negative, and can only be NaN when a
    // component was already non-finite -- a sum of finite squares overflows to
    // +inf rather than to NaN.
    debug_assert!(
        out.iter()
            .zip(batch.iter())
            .all(|(n, v)| (*n >= 0.0) || (n.is_nan() && v.iter().any(|c| !c.is_finite()))),
        "a norm may only be non-negative, or NaN carried in from a non-finite component"
    );
    Ok(out)
}
