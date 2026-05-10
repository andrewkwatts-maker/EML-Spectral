//! eml_spectral_core — Rust acceleration for the eml-spectral package.
//!
//! Mirrors the eml_core (eml-math) crate. Pure-Python paths in eml-spectral
//! always work; this module is opt-in and accessed via
//! `from eml_spectral import eml_spectral_core as _core`.

use pyo3::prelude::*;
use rayon::prelude::*;

mod octonion;
mod spectral;
mod lattice;
mod clifford;

// Optional Arithma symbolic-substrate bridge. Only available when consumed via
// git submodule path-dep (the engine workspace) — never as a PyPI dep. See
// plan §F.11 for the cross-library `with-arithma` pattern.
#[cfg(feature = "with-arithma")]
pub mod arithma_bridge;

use octonion::{octonion_mul, octonion_mul_n, octonion_norm_n};
use spectral::{spectral_flow_step, spectral_flow_n, spectral_flow_batch};
use lattice::{e8_norms_squared_n, e8_min_norm_squared, leech_min_norm_squared};
use clifford::geometric_product_n;

/// Vector-add helper exposed for sanity checks (Rayon-parallel).
#[pyfunction]
fn add_n(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    a.par_iter().zip(b.par_iter()).map(|(x, y)| x + y).collect()
}

#[pymodule]
fn eml_spectral_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(octonion_mul, m)?)?;
    m.add_function(wrap_pyfunction!(octonion_mul_n, m)?)?;
    m.add_function(wrap_pyfunction!(octonion_norm_n, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_flow_step, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_flow_n, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_flow_batch, m)?)?;
    m.add_function(wrap_pyfunction!(e8_norms_squared_n, m)?)?;
    m.add_function(wrap_pyfunction!(e8_min_norm_squared, m)?)?;
    m.add_function(wrap_pyfunction!(leech_min_norm_squared, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_product_n, m)?)?;
    m.add_function(wrap_pyfunction!(add_n, m)?)?;
    Ok(())
}
