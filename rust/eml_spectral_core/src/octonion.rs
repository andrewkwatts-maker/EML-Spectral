//! Octonion (Cayley-Dickson) multiplication via the Fano-plane table.

use pyo3::prelude::*;
use rayon::prelude::*;

const FANO_LINES: [(usize, usize, usize); 7] =
    [(1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7),
     (1, 5, 6), (2, 6, 7), (1, 3, 7)];

const fn build_oct_table() -> [[(i8, usize); 8]; 8] {
    let mut t = [[(0i8, 0usize); 8]; 8];
    let mut i = 0;
    while i < 8 { t[0][i] = (1, i); t[i][0] = (1, i); i += 1; }
    let mut i = 1;
    while i < 8 { t[i][i] = (-1, 0); i += 1; }
    let mut li = 0;
    while li < 7 {
        let (a, b, c) = FANO_LINES[li];
        t[a][b] = (1, c); t[b][a] = (-1, c);
        t[b][c] = (1, a); t[c][b] = (-1, a);
        t[c][a] = (1, b); t[a][c] = (-1, b);
        li += 1;
    }
    t
}

const OCT_TABLE: [[(i8, usize); 8]; 8] = build_oct_table();

#[inline]
fn _mul(a: &[f64], b: &[f64]) -> [f64; 8] {
    let mut out = [0.0f64; 8];
    for i in 0..8 {
        if a[i] == 0.0 { continue; }
        for j in 0..8 {
            if b[j] == 0.0 { continue; }
            let (sign, k) = OCT_TABLE[i][j];
            out[k] += (sign as f64) * a[i] * b[j];
        }
    }
    out
}

/// Multiply two octonions. Inputs are 8-element float vectors.
#[pyfunction]
pub fn octonion_mul(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    let r = _mul(&a, &b);
    r.to_vec()
}

/// Batch octonion multiply, Rayon-parallel.
#[pyfunction]
pub fn octonion_mul_n(a_batch: Vec<Vec<f64>>, b_batch: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    a_batch
        .par_iter()
        .zip(b_batch.par_iter())
        .map(|(a, b)| _mul(a, b).to_vec())
        .collect()
}

/// Batch octonion norm, Rayon-parallel.
#[pyfunction]
pub fn octonion_norm_n(batch: Vec<Vec<f64>>) -> Vec<f64> {
    batch
        .par_iter()
        .map(|v| v.iter().map(|c| c * c).sum::<f64>().sqrt())
        .collect()
}
