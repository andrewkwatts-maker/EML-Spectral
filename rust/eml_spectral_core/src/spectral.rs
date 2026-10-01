//! Discrete spectral flow operator Phi on EML-coordinate pairs.
//!
//! `Phi((x, y)) = (y_safe(y), exp(xv_safe(x)) - ln(y_safe(y)))`
//! where `y_safe(y) = max(|y|, 1e-300)` (Axiom 8 frame-shift guard)
//! and   `xv_safe(x) = ln(x)` when `x > 709.78` (overflow guard).
//!
//! ## Invariants the two guards buy
//!
//! For any *finite* `(x, y)`:
//!
//! 1. `x'` is strictly positive -- `y_safe` never returns zero, so no later
//!    step can take `ln(0)` and produce `-inf`. Every state of a trajectory
//!    after the starting state therefore has a strictly positive first
//!    coordinate, whatever the caller started from.
//! 2. `x'` and `y'` are both finite -- `xv_safe` caps its result at
//!    `ln(f64::MAX)`, so `exp` cannot overflow, and `ln(y_safe(y))` is bounded
//!    below by `ln(1e-300)`. Finiteness is therefore preserved for arbitrarily
//!    many steps.
//!
//! Non-finite input is not refused: a NaN coordinate propagates to NaN, and an
//! infinite coordinate to +/-inf. That is the documented sentinel; the flow
//! never panics.

use crate::error::SpectralError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;

const OVERFLOW_THRESHOLD: f64 = 709.78;

/// Largest number of steps a single trajectory may hold.
///
/// A trajectory is materialised eagerly, so the step count doubles as an
/// allocation bound: at the cap one trajectory is 2^22 pairs (64 MiB).
pub const MAX_FLOW_STEPS: usize = 1 << 22;

#[inline]
fn y_safe(y: f64) -> f64 {
    if y <= 0.0 {
        y.abs().max(1e-300)
    } else {
        y
    }
}

#[inline]
fn xv_safe(x: f64) -> f64 {
    if x > OVERFLOW_THRESHOLD {
        x.ln()
    } else {
        x
    }
}

#[inline]
fn step(x: f64, y: f64) -> (f64, f64) {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    // Guard invariants, checked here rather than at any caller boundary
    // because they are established by y_safe / xv_safe themselves. Both are
    // written to hold for NaN, which is documented to propagate untouched.
    debug_assert!(
        ys > 0.0 || ys.is_nan(),
        "y_safe returns a strictly positive magnitude for every non-NaN input"
    );
    debug_assert!(
        !x.is_finite() || xv.exp().is_finite(),
        "the overflow guard keeps exp(xv_safe(x)) finite for every finite x"
    );
    let t = xv.exp() - ys.ln();
    (ys, t)
}

/// Refuses a step count larger than one trajectory may hold.
fn check_steps(n_steps: usize) -> Result<usize, SpectralError> {
    if n_steps > MAX_FLOW_STEPS {
        return Err(SpectralError::TooManySteps {
            requested: n_steps,
            max: MAX_FLOW_STEPS,
        });
    }
    let capacity = n_steps + 1;
    debug_assert!(
        capacity > n_steps,
        "the starting state always adds one slot without wrapping"
    );
    debug_assert!(
        capacity <= MAX_FLOW_STEPS + 1,
        "an accepted step count stays inside the trajectory bound"
    );
    Ok(capacity)
}

/// Walks one trajectory of `n_steps` steps into a freshly allocated buffer.
fn trajectory(x0: f64, y0: f64, n_steps: usize, capacity: usize) -> Vec<(f64, f64)> {
    debug_assert_eq!(
        capacity,
        n_steps + 1,
        "the capacity is the validated step count plus the starting state"
    );
    let mut out = Vec::with_capacity(capacity);
    out.push((x0, y0));
    let mut x = x0;
    let mut y = y0;
    for _ in 0..n_steps {
        let (xn, yn) = step(x, y);
        x = xn;
        y = yn;
        out.push((x, y));
    }
    debug_assert_eq!(
        out.len(),
        capacity,
        "the loop runs exactly once per requested step"
    );
    out
}

/// One Phi step: returns (x', y').
///
/// Total on `f64`: every input pair, finite or not, has a defined result, so
/// there is nothing here to refuse.
#[cfg_attr(feature = "python", pyfunction)]
pub fn spectral_flow_step(x: f64, y: f64) -> (f64, f64) {
    step(x, y)
}

/// Generate a length-(n_steps+1) trajectory starting from (x0, y0). Index 0 is
/// the starting state; index `i+1` is [`spectral_flow_step`] applied to index
/// `i`.
///
/// # Errors
///
/// [`SpectralError::TooManySteps`] if `n_steps` exceeds [`MAX_FLOW_STEPS`].
#[cfg_attr(feature = "python", pyfunction)]
pub fn spectral_flow_n(x0: f64, y0: f64, n_steps: usize) -> Result<Vec<(f64, f64)>, SpectralError> {
    let capacity = check_steps(n_steps)?;
    let out = trajectory(x0, y0, n_steps, capacity);
    debug_assert_eq!(
        out.len(),
        n_steps + 1,
        "a trajectory holds the starting state plus one state per step"
    );
    debug_assert!(
        out[0].0.to_bits() == x0.to_bits() && out[0].1.to_bits() == y0.to_bits(),
        "index 0 is the starting state, untouched"
    );
    Ok(out)
}

/// Generate one trajectory per starting point -- Rayon-parallel over starts.
/// Row `i` is identical to [`spectral_flow_n`] on `starts[i]`.
///
/// # Errors
///
/// [`SpectralError::TooManySteps`] if `n_steps` exceeds [`MAX_FLOW_STEPS`].
#[cfg_attr(feature = "python", pyfunction)]
pub fn spectral_flow_batch(
    starts: Vec<(f64, f64)>,
    n_steps: usize,
) -> Result<Vec<Vec<(f64, f64)>>, SpectralError> {
    let capacity = check_steps(n_steps)?;
    let out: Vec<Vec<(f64, f64)>> = starts
        .par_iter()
        .map(|&(x0, y0)| trajectory(x0, y0, n_steps, capacity))
        .collect();
    debug_assert_eq!(out.len(), starts.len(), "one trajectory per starting point");
    debug_assert!(
        out.iter().all(|traj| traj.len() == capacity),
        "every trajectory in a batch has the same length"
    );
    Ok(out)
}
