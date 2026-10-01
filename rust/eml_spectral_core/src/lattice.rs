//! Lattice helpers -- E8 and Leech norm-squared.
//!
//! The two minimum-norm constants are the defining invariants of the lattices:
//! the E8 root system's 240 shortest vectors all have squared norm 2, and the
//! Leech lattice's 196560 shortest vectors all have squared norm 4 (in the
//! standard normalisation where Leech coordinates carry a factor 1/sqrt(8)).

use crate::error::SpectralError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;

/// Batch squared norm for points -- Rayon-parallel.
///
/// Dimension-agnostic: it is used for 8-component E8 points and 24-component
/// Leech points alike, and it does not require the points to share a
/// dimension. Non-finite coordinates propagate under IEEE-754 rules.
///
/// # Errors
///
/// [`SpectralError::EmptyPoint`] if any point has no coordinates. A
/// zero-dimensional point would otherwise sum to `0.0` and be indistinguishable
/// from a genuine lattice point at the origin.
#[cfg_attr(feature = "python", pyfunction)]
pub fn e8_norms_squared_n(points: Vec<Vec<f64>>) -> Result<Vec<f64>, SpectralError> {
    for (index, p) in points.iter().enumerate() {
        if p.is_empty() {
            return Err(SpectralError::EmptyPoint {
                what: "norm batch",
                index,
            });
        }
    }
    let out: Vec<f64> = points
        .par_iter()
        .map(|p| p.iter().map(|c| c * c).sum())
        .collect();
    debug_assert_eq!(out.len(), points.len(), "one squared norm per input point");
    // A sum of squares is never negative, and can only be NaN when a coordinate
    // was already non-finite -- a sum of finite squares overflows to +inf.
    debug_assert!(
        out.iter()
            .zip(points.iter())
            .all(|(n, p)| (*n >= 0.0) || (n.is_nan() && p.iter().any(|c| !c.is_finite()))),
        "a squared norm may only be non-negative, or NaN carried in from a non-finite coordinate"
    );
    Ok(out)
}

/// E8 minimum-vector squared norm (= 2.0).
#[cfg_attr(feature = "python", pyfunction)]
pub fn e8_min_norm_squared() -> f64 {
    2.0
}

/// Leech minimum-vector squared norm (= 4.0).
#[cfg_attr(feature = "python", pyfunction)]
pub fn leech_min_norm_squared() -> f64 {
    4.0
}
