//! Lattice helpers — E₈ and Leech norm-squared.

use pyo3::prelude::*;
use rayon::prelude::*;

/// Batch ‖x‖² for points (Rayon-parallel; works for any dimension).
#[pyfunction]
pub fn e8_norms_squared_n(points: Vec<Vec<f64>>) -> Vec<f64> {
    points
        .par_iter()
        .map(|p| p.iter().map(|c| c * c).sum())
        .collect()
}

/// E₈ minimum-vector squared norm (= 2.0).
#[pyfunction]
pub fn e8_min_norm_squared() -> f64 {
    2.0
}

/// Leech minimum-vector squared norm (= 4.0).
#[pyfunction]
pub fn leech_min_norm_squared() -> f64 {
    4.0
}
