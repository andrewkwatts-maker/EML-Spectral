//! Arithma symbolic-engine bridge for eml-spectral.
//!
//! Gated behind the `with-arithma` Cargo feature, which is **only** available
//! when this crate is consumed via git-submodule path-dep (e.g. inside the
//! PlayTow engine workspace). PyPI consumers (`pip install eml-spectral`) do
//! not see this module — Arithma is never a public dependency of
//! `eml-spectral`.
//!
//! ## Purpose
//!
//! The plan's §F.11 design has every numerics library in our stack opt into a
//! single symbolic substrate (`ArithmaExpression`) when assembled in-engine,
//! so spacetime metrics, Christoffel symbols, octonion products and lattice
//! norms can flow into and out of Arithma without losing precision or having
//! to re-parse strings.
//!
//! For eml-spectral specifically the high-value targets are:
//! - **MetricTensor parametrisation** — Schwarzschild's `r_s`, FLRW's `a(t)`,
//!   Calabi-Yau moduli, etc. carried as Arithma sub-trees so Christoffel
//!   batches stay symbolic until the GPU upload step.
//! - **Octonion / Multivector amplitude carriers** — components stored as
//!   ArithmaExpression for symbolic differentiation along trajectories.
//! - **Lattice point predicates** — selection rules on lattice sums become
//!   composable Arithma expressions instead of opaque closures.
//!
//! ## Status
//!
//! Skeleton only — converters return sensible defaults / `unimplemented!()`.
//! The signatures here are the contract every consumer can rely on; only the
//! bodies are deferred. Wave 3 wires up the real conversion paths once the
//! Arithma surface stabilises.

use arithma_core::expression::ArithmaExpression;

/// Trait implemented by any eml-spectral type that can carry an Arithma
/// sub-tree alongside its native numeric representation. Mirrors the
/// `ArithmaPayload` trait in eml-math's bridge for consistency.
pub trait ArithmaPayload {
    /// The native numeric type the implementor stores by default.
    type Numeric;

    /// Replace the payload's numeric value with an Arithma expression. The
    /// numeric value is kept as a cached fallback for backends that haven't
    /// adopted Arithma yet.
    fn with_arithma_payload(self, expr: ArithmaExpression) -> Self;

    /// Returns a reference to the attached Arithma sub-tree, if any.
    fn arithma_payload(&self) -> Option<&ArithmaExpression>;

    /// Strip the attached Arithma sub-tree and return only the numeric form.
    fn into_numeric(self) -> Self::Numeric;
}

/// Bridge for the Schwarzschild metric. Wave-3 lands the symbolic Christoffel
/// path; the Wave-2 stub returns `Err` so callers know to fall through.
pub fn schwarzschild_arithma_christoffel(
    _r_s: &ArithmaExpression,
) -> Result<ArithmaExpression, BridgeError> {
    Err(BridgeError::NotYetImplemented("schwarzschild_arithma_christoffel"))
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
        // Build a placeholder Arithma expression — the variable form is
        // the cheapest non-trivial value.
        let r_s = ArithmaExpression::Variable("r_s".to_string());
        let r = schwarzschild_arithma_christoffel(&r_s);
        assert!(matches!(r, Err(BridgeError::NotYetImplemented(_))));
    }
}
