//! Geometric product for Cl(p, q) — bitmask blade encoding.

use pyo3::prelude::*;
use rayon::prelude::*;

/// e_I · e_J  in a Clifford algebra with the given diagonal signature.
/// Returns (sign, blade_index_of_result).
fn blade_product(ia: usize, jb: usize, signature: &[i8]) -> (f64, usize) {
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
            sign *= signature[k] as f64;
        }
        s >>= 1;
        k += 1;
    }
    (sign, ia ^ jb)
}

fn product_one(a: &[f64], b: &[f64], signature: &[i8]) -> Vec<f64> {
    let n = signature.len();
    let total = 1usize << n;
    let mut out = vec![0.0f64; total];
    for ia in 0..total {
        if a[ia] == 0.0 { continue; }
        for jb in 0..total {
            if b[jb] == 0.0 { continue; }
            let (sign, k) = blade_product(ia, jb, signature);
            out[k] += sign * a[ia] * b[jb];
        }
    }
    out
}

/// Batch geometric product for Cl(p, q) — Rayon-parallel.
#[pyfunction]
pub fn geometric_product_n(
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    signature: Vec<i8>,
) -> Vec<Vec<f64>> {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(av, bv)| product_one(av, bv, &signature))
        .collect()
}
