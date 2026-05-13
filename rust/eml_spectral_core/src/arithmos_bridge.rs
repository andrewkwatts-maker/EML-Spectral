//! Arithmos symbolic-engine bridge for eml-spectral.
//!
//! Gated behind the `with-arithmos` Cargo feature, which is **only** available
//! when this crate is consumed via git-submodule path-dep (e.g. inside the
//! PlayTow engine workspace). PyPI consumers (`pip install eml-spectral`) do
//! not see this module — Arithmos is never a public dependency of
//! `eml-spectral`.
//!
//! ## Purpose
//!
//! The plan's §F.11 design has every numerics library in our stack opt into a
//! single symbolic substrate (`ArithmosExpression`) when assembled in-engine,
//! so spacetime metrics, Christoffel symbols, octonion products and lattice
//! norms can flow into and out of Arithmos without losing precision or having
//! to re-parse strings.
//!
//! For eml-spectral specifically the high-value targets are:
//! - **MetricTensor parametrisation** — Schwarzschild's `r_s`, FLRW's `a(t)`,
//!   Calabi-Yau moduli, etc. carried as Arithmos sub-trees so Christoffel
//!   batches stay symbolic until the GPU upload step.
//! - **Octonion / Multivector amplitude carriers** — components stored as
//!   ArithmosExpression for symbolic differentiation along trajectories.
//! - **Lattice point predicates** — selection rules on lattice sums become
//!   composable Arithmos expressions instead of opaque closures.
//!
//! ## Status
//!
//! Skeleton only — converters return sensible defaults / `unimplemented!()`.
//! The signatures here are the contract every consumer can rely on; only the
//! bodies are deferred. Wave 3 wires up the real conversion paths once the
//! Arithmos surface stabilises.

use arithmos_core::expression::ArithmosExpression;

/// Trait implemented by any eml-spectral type that can carry an Arithmos
/// sub-tree alongside its native numeric representation. Mirrors the
/// `ArithmosPayload` trait in eml-math's bridge for consistency.
pub trait ArithmosPayload {
    /// The native numeric type the implementor stores by default.
    type Numeric;

    /// Replace the payload's numeric value with an Arithmos expression. The
    /// numeric value is kept as a cached fallback for backends that haven't
    /// adopted Arithmos yet.
    fn with_arithmos_payload(self, expr: ArithmosExpression) -> Self;

    /// Returns a reference to the attached Arithmos sub-tree, if any.
    fn arithmos_payload(&self) -> Option<&ArithmosExpression>;

    /// Strip the attached Arithmos sub-tree and return only the numeric form.
    fn into_numeric(self) -> Self::Numeric;
}

/// Bridge for the Schwarzschild metric. Wave-3 lands the symbolic Christoffel
/// path; the Wave-2 stub returns `Err` so callers know to fall through.
pub fn schwarzschild_arithmos_christoffel(
    _r_s: &ArithmosExpression,
) -> Result<ArithmosExpression, BridgeError> {
    Err(BridgeError::NotYetImplemented("schwarzschild_arithmos_christoffel"))
}

/// Errors returned by the bridge.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("not yet implemented: {0}")]
    NotYetImplemented(&'static str),
    #[error("conversion failed: {0}")]
    ConversionFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schwarzschild_returns_unimplemented_for_now() {
        // Build a placeholder Arithmos expression — the variable form is
        // the cheapest non-trivial value.
        let r_s = ArithmosExpression::Variable("r_s".to_string());
        let r = schwarzschild_arithmos_christoffel(&r_s);
        assert!(matches!(r, Err(BridgeError::NotYetImplemented(_))));
    }
}
