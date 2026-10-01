//! Error type shared by every validating entry point in this crate.
//!
//! The rule the whole crate follows: *structural* problems -- a vector that is
//! not the length the algebra requires, two batches that do not line up, a
//! signature the blade encoding cannot represent -- are refused with a
//! [`SpectralError`]. They are never padded, truncated, or silently reduced to
//! a default value.
//!
//! Non-finite *values* (NaN, +/-inf) are a separate matter and are not refused:
//! they propagate through the arithmetic under ordinary IEEE-754 rules. Each
//! module documents that behaviour where it matters. Nothing in this crate
//! panics on non-finite input.

use std::fmt;

#[cfg(feature = "python")]
use pyo3::{exceptions::PyValueError, PyErr};

/// Everything this crate can refuse to compute.
///
/// Messages are plain ASCII so they survive any terminal, log sink, or Python
/// traceback encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpectralError {
    /// A vector was handed over with a length the algebra cannot use.
    WrongLength {
        /// Which operand, for the message.
        what: &'static str,
        /// Length the algebra requires.
        expected: usize,
        /// Length actually supplied.
        actual: usize,
    },
    /// Two collections that must be walked in lockstep had different lengths.
    LengthMismatch {
        /// What was being paired, for the message.
        what: &'static str,
        /// Length of the left collection.
        left: usize,
        /// Length of the right collection.
        right: usize,
    },
    /// One element of a batch had the wrong length.
    BatchElementWrongLength {
        /// Which batch, for the message.
        what: &'static str,
        /// Position of the offending element.
        index: usize,
        /// Length the algebra requires.
        expected: usize,
        /// Length actually supplied.
        actual: usize,
    },
    /// A point with no components was supplied where a norm was requested.
    EmptyPoint {
        /// Which batch, for the message.
        what: &'static str,
        /// Position of the offending element.
        index: usize,
    },
    /// A Clifford signature had no entries, so there is no algebra to work in.
    EmptySignature,
    /// A Clifford signature was longer than the blade encoding supports.
    SignatureTooLong {
        /// Number of generators requested.
        requested: usize,
        /// Largest number of generators supported.
        max: usize,
    },
    /// A Clifford signature entry was not +1 or -1.
    SignatureEntryNotUnit {
        /// Position of the offending entry.
        index: usize,
        /// The value found there.
        value: i8,
    },
    /// More flow steps were requested than the trajectory buffer may hold.
    TooManySteps {
        /// Number of steps requested.
        requested: usize,
        /// Largest number of steps supported.
        max: usize,
    },
}

impl fmt::Display for SpectralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongLength {
                what,
                expected,
                actual,
            } => write!(f, "{what} must have {expected} components, got {actual}"),
            Self::LengthMismatch { what, left, right } => {
                write!(f, "{what} must have equal lengths, got {left} and {right}")
            }
            Self::BatchElementWrongLength {
                what,
                index,
                expected,
                actual,
            } => write!(
                f,
                "{what} element {index} must have {expected} components, got {actual}"
            ),
            Self::EmptyPoint { what, index } => {
                write!(f, "{what} element {index} has no components")
            }
            Self::EmptySignature => {
                write!(f, "signature must name at least one generator")
            }
            Self::SignatureTooLong { requested, max } => write!(
                f,
                "signature of {requested} generators exceeds the supported maximum of {max}"
            ),
            Self::SignatureEntryNotUnit { index, value } => {
                write!(f, "signature entry {index} must be +1 or -1, got {value}")
            }
            Self::TooManySteps { requested, max } => write!(
                f,
                "{requested} steps exceeds the supported maximum of {max}"
            ),
        }
    }
}

impl std::error::Error for SpectralError {}

/// Surfaces a refusal to Python as `ValueError` rather than as a Rust panic
/// (which PyO3 would raise as the near-uncatchable `PanicException`).
#[cfg(feature = "python")]
impl From<SpectralError> for PyErr {
    fn from(err: SpectralError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}
