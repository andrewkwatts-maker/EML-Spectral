"""
eml_spectral.exceptional — exceptional algebras as EML constructions.

Re-exports:
    FreudenthalTripleSystem  — J₃(𝕆) cubic norm + triple product
    E7_56                    — 56-dimensional rep of E₇, quartic invariant
    E8_248                   — adjoint rep of E₈
    E8xE8                    — heterotic E₈×E₈ pair
"""
from eml_spectral.exceptional.freudenthal import FreudenthalTripleSystem
from eml_spectral.exceptional.e7_56 import E7_56
from eml_spectral.exceptional.e8_248 import E8_248, E8xE8

__all__ = [
    "FreudenthalTripleSystem",
    "E7_56",
    "E8_248",
    "E8xE8",
]
