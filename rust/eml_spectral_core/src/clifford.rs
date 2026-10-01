//! Geometric product for Cl(p, q) -- bitmask blade encoding.
//!
//! A multivector is `2^n` coefficients where `n` is the number of generators
//! named by the signature. Index `I` of the coefficient array is a bitmask:
//! bit `k` set means generator `e_k` appears in that blade, so index `0b101` is
//! the coefficient of `e_0 e_2`.
//!
//! ## Input contract
//!
//! The signature must name between 1 and [`MAX_CLIFFORD_DIM`] generators, each
//! `+1` or `-1`, and every multivector must carry exactly `2^n` coefficients.
//! Anything else is refused with a [`SpectralError`]. Non-finite coefficients
//! are not refused; they propagate under IEEE-754 rules.

use crate::error::SpectralError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;

/// Largest number of generators the blade encoding accepts.
///
/// This is a memory and shift-overflow guard, not a performance promise: the
/// product is O(4^n), so the useful range is far smaller than the cap. At the
/// cap a multivector is 2^16 coefficients (512 KiB), and the largest shift
/// `blade_product` performs is 16, comfortably inside `usize`.
pub const MAX_CLIFFORD_DIM: usize = 16;

/// e_I . e_J in a Clifford algebra with the given diagonal signature.
/// Returns (sign, blade_index_of_result).
fn blade_product(ia: usize, jb: usize, signature: &[i8]) -> (f64, usize) {
    debug_assert!(
        signature.len() <= MAX_CLIFFORD_DIM,
        "callers validate the signature length before reaching the kernel"
    );
    debug_assert!(
        ia < (1usize << signature.len()) && jb < (1usize << signature.len()),
        "blade indices are bitmasks over the signature's generators"
    );
    // Anticommutation parity: count swaps needed to permute the
    // generators of e_I e_J into canonical order. For each set bit k of jb,
    // count the bits of ia at positions strictly > k.
    let mut swaps: u32 = 0;
    let mut bit = 1usize;
    for k in 0..signature.len() {
        if jb & bit != 0 {
            let higher = ia >> (k + 1);
            swaps += higher.count_ones();
        }
        bit <<= 1;
    }
    let mut sign: f64 = if swaps & 1 == 1 { -1.0 } else { 1.0 };
    // Signature factor for shared generators.
    let shared = ia & jb;
    let mut s = shared;
    let mut k = 0usize;
    while s != 0 {
        if s & 1 == 1 {
            sign *= f64::from(signature[k]);
        }
        s >>= 1;
        k += 1;
    }
    (sign, ia ^ jb)
}

/// Geometric product of two multivectors a caller has already validated.
fn product_one(a: &[f64], b: &[f64], signature: &[i8]) -> Vec<f64> {
    let n = signature.len();
    debug_assert!(
        (1..=MAX_CLIFFORD_DIM).contains(&n),
        "the signature reaches the kernel only after range validation"
    );
    let total = 1usize << n;
    debug_assert!(
        a.len() == total && b.len() == total,
        "both operands reach the kernel only after 2^n length validation"
    );
    let mut out = vec![0.0f64; total];
    for (ia, &av) in a.iter().enumerate() {
        if av == 0.0 {
            continue;
        }
        for (jb, &bv) in b.iter().enumerate() {
            if bv == 0.0 {
                continue;
            }
            let (sign, k) = blade_product(ia, jb, signature);
            out[k] += sign * av * bv;
        }
    }
    out
}

/// Refuses a signature the blade encoding cannot represent.
fn check_signature(signature: &[i8]) -> Result<usize, SpectralError> {
    if signature.is_empty() {
        return Err(SpectralError::EmptySignature);
    }
    if signature.len() > MAX_CLIFFORD_DIM {
        return Err(SpectralError::SignatureTooLong {
            requested: signature.len(),
            max: MAX_CLIFFORD_DIM,
        });
    }
    for (index, &value) in signature.iter().enumerate() {
        if value != 1 && value != -1 {
            return Err(SpectralError::SignatureEntryNotUnit { index, value });
        }
    }
    let total = 1usize << signature.len();
    debug_assert!(
        total >= 2,
        "an accepted signature always yields at least the scalar and one vector blade"
    );
    debug_assert!(
        total.is_power_of_two() && total <= 1usize << MAX_CLIFFORD_DIM,
        "the blade count stays a power of two inside the supported range"
    );
    debug_assert!(
        signature.iter().all(|&v| v == 1 || v == -1),
        "the Ok path leaves every signature entry a unit"
    );
    Ok(total)
}

/// Refuses a batch in which any multivector is not `2^n` coefficients long.
fn check_multivector_batch(
    batch: &[Vec<f64>],
    total: usize,
    what: &'static str,
) -> Result<(), SpectralError> {
    for (index, v) in batch.iter().enumerate() {
        if v.len() != total {
            return Err(SpectralError::BatchElementWrongLength {
                what,
                index,
                expected: total,
                actual: v.len(),
            });
        }
    }
    debug_assert!(
        batch.iter().all(|v| v.len() == total),
        "the Ok path leaves every multivector at 2^n coefficients"
    );
    debug_assert!(
        total.is_power_of_two(),
        "a Clifford algebra's dimension is a power of two"
    );
    Ok(())
}

/// Batch geometric product for Cl(p, q) -- Rayon-parallel.
///
/// # Errors
///
/// [`SpectralError::EmptySignature`], [`SpectralError::SignatureTooLong`] or
/// [`SpectralError::SignatureEntryNotUnit`] for a signature the encoding
/// cannot represent; [`SpectralError::LengthMismatch`] if the two batches
/// differ in length -- the pairing is never silently truncated to the shorter
/// one -- and [`SpectralError::BatchElementWrongLength`] if a multivector does
/// not carry exactly `2^n` coefficients.
#[cfg_attr(feature = "python", pyfunction)]
pub fn geometric_product_n(
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    signature: Vec<i8>,
) -> Result<Vec<Vec<f64>>, SpectralError> {
    let total = check_signature(&signature)?;
    if a.len() != b.len() {
        return Err(SpectralError::LengthMismatch {
            what: "multivector batches",
            left: a.len(),
            right: b.len(),
        });
    }
    check_multivector_batch(&a, total, "left batch")?;
    check_multivector_batch(&b, total, "right batch")?;
    let out: Vec<Vec<f64>> = a
        .par_iter()
        .zip(b.par_iter())
        .map(|(av, bv)| product_one(av, bv, &signature))
        .collect();
    debug_assert_eq!(
        out.len(),
        a.len(),
        "the zip cannot drop a pair once both batches are the same length"
    );
    debug_assert!(
        out.iter().all(|v| v.len() == total),
        "a geometric product stays inside the same 2^n-dimensional algebra"
    );
    Ok(out)
}
