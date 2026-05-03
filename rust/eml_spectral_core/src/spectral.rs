//! Discrete spectral flow operator Φ on EML-coordinate pairs.
//!
//! Φ((x, y)) = (y_safe, exp(xv_safe(x)) − ln(y_safe))
//! where y_safe = max(|y|, 1e-300)  (Axiom 8 frame-shift guard)
//! and   xv_safe(x) = ln(x) when x > 709.78 (overflow guard).

use pyo3::prelude::*;
use rayon::prelude::*;

const OVERFLOW_THRESHOLD: f64 = 709.78;

#[inline]
fn y_safe(y: f64) -> f64 {
    if y <= 0.0 { y.abs().max(1e-300) } else { y }
}

#[inline]
fn xv_safe(x: f64) -> f64 {
    if x > OVERFLOW_THRESHOLD { x.ln() } else { x }
}

#[inline]
fn step(x: f64, y: f64) -> (f64, f64) {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t = xv.exp() - ys.ln();
    (ys, t)
}

/// One Φ step: returns (x', y').
#[pyfunction]
pub fn spectral_flow_step(x: f64, y: f64) -> (f64, f64) {
    step(x, y)
}

/// Generate a length-(n_steps+1) trajectory starting from (x0, y0).
#[pyfunction]
pub fn spectral_flow_n(x0: f64, y0: f64, n_steps: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(n_steps + 1);
    out.push((x0, y0));
    let mut x = x0;
    let mut y = y0;
    for _ in 0..n_steps {
        let (xn, yn) = step(x, y);
        x = xn;
        y = yn;
        out.push((x, y));
    }
    out
}

/// Generate one trajectory per starting point — Rayon-parallel over starts.
#[pyfunction]
pub fn spectral_flow_batch(
    starts: Vec<(f64, f64)>,
    n_steps: usize,
) -> Vec<Vec<(f64, f64)>> {
    starts
        .par_iter()
        .map(|&(x0, y0)| {
            let mut traj = Vec::with_capacity(n_steps + 1);
            traj.push((x0, y0));
            let mut x = x0;
            let mut y = y0;
            for _ in 0..n_steps {
                let (xn, yn) = step(x, y);
                x = xn;
                y = yn;
                traj.push((x, y));
            }
            traj
        })
        .collect()
}
