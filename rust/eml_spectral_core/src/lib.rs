//! eml_spectral_core -- Rust acceleration for the eml-spectral package.
//!
//! Mirrors the eml_core (eml-math) crate. Pure-Python paths in eml-spectral
//! always work; this module is opt-in and accessed via
//! `from eml_spectral import eml_spectral_core as _core`.
//!
//! Every entry point that can be handed structurally invalid input returns
//! `Result<_, SpectralError>` and refuses it; see [`error`] for the rule
//! separating a refusal from ordinary IEEE-754 propagation.

#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;

pub mod clifford;
pub mod error;
pub mod lattice;
pub mod octonion;
pub mod spectral;

// Optional Arithmos symbolic-substrate bridge. Only available when consumed via
// git submodule path-dep (the engine workspace) -- never as a PyPI dep. See
// plan section F.11 for the cross-library `with-arithmos` pattern. The feature
// is declared to rustc's check-cfg in Cargo.toml rather than to Cargo, because
// its dependencies only exist inside that workspace.
#[cfg(feature = "with-arithmos")]
pub mod arithmos_bridge;

pub use clifford::geometric_product_n;
pub use error::SpectralError;
pub use lattice::{e8_min_norm_squared, e8_norms_squared_n, leech_min_norm_squared};
pub use octonion::{octonion_mul, octonion_mul_n, octonion_norm_n};
pub use spectral::{spectral_flow_batch, spectral_flow_n, spectral_flow_step};

/// Vector-add helper exposed for sanity checks (Rayon-parallel).
///
/// # Errors
///
/// [`SpectralError::LengthMismatch`] if the two vectors differ in length. The
/// sum is never silently truncated to the shorter of the two.
#[cfg_attr(feature = "python", pyfunction)]
pub fn add_n(a: Vec<f64>, b: Vec<f64>) -> Result<Vec<f64>, SpectralError> {
    if a.len() != b.len() {
        return Err(SpectralError::LengthMismatch {
            what: "addends",
            left: a.len(),
            right: b.len(),
        });
    }
    let out: Vec<f64> = a.par_iter().zip(b.par_iter()).map(|(x, y)| x + y).collect();
    debug_assert_eq!(
        out.len(),
        a.len(),
        "the zip cannot drop a term once both vectors are the same length"
    );
    debug_assert!(
        out.iter()
            .zip(a.iter().zip(b.iter()))
            .all(|(s, (x, y))| s.is_nan() || *s == x + y),
        "each output is the sum of the operands at the same position"
    );
    Ok(out)
}

#[cfg(feature = "python")]
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
