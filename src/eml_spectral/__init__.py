"""
eml_spectral — EML-tree representations of Clifford algebras, octonions,
exceptional algebras (E7/E8/Freudenthal), Lorentz-invariant operations,
named GR metrics (Schwarzschild, FLRW, AdS₅×S⁵, …), and integer lattices
(E8, Leech).

Built on top of eml-math; pulls EMLPoint, operators, etc. from there.

Quickstart
----------
>>> from eml_math import EMLPoint
>>> from eml_spectral import EMLMultivector, Octonion, FreudenthalTripleSystem
>>> from eml_spectral.spacetime import minkowski_delta, boost
>>> from eml_spectral.metrics import MetricTensor

>>> p = EMLPoint(1.0, 2.0)
>>> minkowski_delta(p)
"""

from eml_spectral.pair import EMLPair
from eml_spectral.geometric_algebra import EMLMultivector
from eml_spectral.octonion import Octonion, basis_octonion
from eml_spectral.metric import MetricTensor
from eml_spectral.fourvector import MinkowskiFourVector
from eml_spectral.momentum import FourMomentum
from eml_spectral.ndim import (
    EMLNDVector, e8_lattice_points, leech_lattice_points,
)
from eml_spectral.discrete import (
    planck_delta, lattice_distance, is_lattice_neighbor,
)
from eml_spectral.state import EMLState
from eml_spectral.simulation import (
    simulate_pulses, simulate_flips, quantized_trajectory,
    tension_series, rho_series, phase_series,
    verify_conservation, frame_shift_count, find_resonance_bands,
)
from eml_spectral.exceptional import (
    FreudenthalTripleSystem, E7_56, E8_248, E8xE8,
)
from eml_spectral import spacetime
from eml_spectral.spectral_flow import (
    spectral_flow, racetrack_fixed_point, topology_invariant, G2_SEEDS,
)

# Datasheet API — uniform with eml_math.Get / metaphysica.Get / future
# periodica.Get. Returns a JSON-serialisable dict for any spectral or
# math constant the EML stack knows about.
from eml_spectral.datasheet import Get, list_constants, SPECTRAL_CATALOGUE

# Optional Rust acceleration. Lazy-imported; pure-Python paths always work.
try:
    from eml_spectral import eml_spectral_core as _core   # noqa: F401
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# Convenience aliases
iterate = simulate_pulses

# Companion-app launcher — `eml_spectral.Launch()` finds/clones EML-Spectral-App and runs it.
from eml_spectral._launcher import launch as Launch

__version__ = "2.0.36"
__author__ = "Andrew K Watts"

__all__ = [
    # Two-real complex pair (was eml_math.pair in v1.x)
    "EMLPair",
    # Algebras
    "EMLMultivector",
    "Octonion",
    "basis_octonion",
    "FreudenthalTripleSystem",
    "E7_56",
    "E8_248",
    "E8xE8",
    # Spacetime — both class wrappers AND the functional namespace
    "MinkowskiFourVector",
    "FourMomentum",
    "spacetime",   # functional API: spacetime.minkowski_delta(p), boost(p, rapidity), …
    # Spectral dynamics (salvaged + upgraded mirror_pulse)
    "spectral_flow",
    "racetrack_fixed_point",
    "topology_invariant",
    "G2_SEEDS",
    # Metrics
    "MetricTensor",
    # Lattices
    "EMLNDVector",
    "e8_lattice_points",
    "leech_lattice_points",
    "planck_delta",
    "lattice_distance",
    "is_lattice_neighbor",
    # Iteration
    "EMLState",
    "iterate",
    "simulate_pulses",
    "simulate_flips",
    "quantized_trajectory",
    "tension_series",
    "rho_series",
    "phase_series",
    "verify_conservation",
    "frame_shift_count",
    "find_resonance_bands",
    # Datasheet API (uniform Get convention across the EML stack)
    "Get",
    "list_constants",
    "SPECTRAL_CATALOGUE",
    # Rust availability flag (True when eml_spectral_core extension is built)
    "_HAS_RUST",
    # Companion app
    "Launch",
]
