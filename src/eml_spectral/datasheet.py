"""eml_spectral.datasheet — JSON datasheet API for the spectrum layer.

Sister module to :mod:`eml_math.datasheet`. Implements the same ``Get()``
contract used across the EML stack:

* ``eml_math.Get('pi')``        → math constants (in eml-math)
* ``eml_spectral.Get('E8_dim')`` → algebra dimensions, lattice constants,
                                   spectral invariants (this module)
* ``metaphysica.Get('Up')``     → physics constants + quarks
* (future) ``periodica.Get('Fe')`` → material constants

Each ``Get(name)`` returns a JSON-serialisable dict carrying enough
information for both human display and programmatic re-use.

Catalogue
---------
* **Algebra dimensions** — Cl(p,q), G₂, F₄, E₆, E₇, E₈, J₃(𝕆), 27, 56, 248
* **Lattice constants** — E₈ minimum norm √2, kissing number 240; Leech
  minimum norm 2, kissing number 196560
* **Spectral invariants** — b₃ = 24, χ_eff = 144, Φ topological invariant
* **Heterotic / racetrack** — λ_eff, ε ≈ Cabibbo proxy, portal coupling
  α_leak = 1/√6
* **Pneuma seeds** — G₂ seed values
* **Pass-through to eml-math** — anything in ``list_symbols()`` (pi, e,
  phi, sqrt2, …) is also reachable via ``eml_spectral.Get`` for a single
  uniform entry point.

Quickstart
----------
>>> import eml_spectral
>>> eml_spectral.Get('E8_dim')['value']
248
>>> eml_spectral.Get('alpha_leak')['value']
0.408248...
>>> eml_spectral.Get('lambda_eff')['formula']
'exp(-2π/24)'
>>> eml_spectral.Get('pi')['value']      # delegated to eml-math
3.141592653589793
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Union

__all__ = ["Get", "list_constants", "SPECTRAL_CATALOGUE"]


# ── Spectral catalogue (eml-spectral specific) ──────────────────────────────
#
# Each entry: name → dict containing keys:
#   value:       numeric value
#   formula:     human-readable derivation
#   kind:        "algebra" | "lattice" | "topology" | "heterotic" |
#                "spectral" | "physics" | "dimension"
#   description: one-line what-it-is text

_PI = math.pi
_LAMBDA_EFF = math.exp(-2.0 * _PI / 24.0)   # racetrack stabilised modulus

SPECTRAL_CATALOGUE: Dict[str, Dict[str, Any]] = {

    # ── Algebra dimensions ──────────────────────────────────────────────
    "g2_dim":         {"value": 14,  "formula": "dim G₂ = 14",
                        "kind": "dimension",
                        "description": "Dimension of the exceptional Lie algebra G₂."},
    "f4_dim":         {"value": 52,  "formula": "dim F₄ = 52",
                        "kind": "dimension",
                        "description": "Dimension of the exceptional Lie algebra F₄."},
    "e6_dim":         {"value": 78,  "formula": "dim E₆ = 78",
                        "kind": "dimension",
                        "description": "Dimension of the exceptional Lie algebra E₆."},
    "e7_dim":         {"value": 133, "formula": "dim E₇ = 133",
                        "kind": "dimension",
                        "description": "Dimension of the exceptional Lie algebra E₇."},
    "e8_dim":         {"value": 248, "formula": "dim E₈ = 248",
                        "kind": "dimension",
                        "description": "Dimension of the exceptional Lie algebra E₈ (adjoint)."},
    "e7_56":          {"value": 56,  "formula": "dim 𝟓𝟔 of E₇",
                        "kind": "dimension",
                        "description": "Fundamental 56-dim representation of E₇."},
    "j3o_dim":        {"value": 27,  "formula": "dim J₃(𝕆) = 27",
                        "kind": "dimension",
                        "description": "Real dimension of the exceptional Jordan algebra J₃(𝕆)."},
    "octonion_dim":   {"value": 8,   "formula": "dim 𝕆 = 8",
                        "kind": "dimension",
                        "description": "Real dimension of the octonions 𝕆."},
    "quaternion_dim": {"value": 4,   "formula": "dim ℍ = 4",
                        "kind": "dimension",
                        "description": "Real dimension of the quaternions ℍ."},
    "spacetime_dim":  {"value": 4,   "formula": "(3+1)D",
                        "kind": "dimension",
                        "description": "Minkowski signature: 1 timelike + 3 spacelike."},
    "leech_dim":      {"value": 24,  "formula": "Leech ⊂ ℝ²⁴",
                        "kind": "dimension",
                        "description": "Ambient dimension of the Leech lattice."},

    # ── Lattice constants ───────────────────────────────────────────────
    "e8_min_norm":    {"value": math.sqrt(2.0),     "formula": "√2",
                        "kind": "lattice",
                        "description": "E₈ minimum-vector norm."},
    "e8_min_norm_sq": {"value": 2.0,                "formula": "2",
                        "kind": "lattice",
                        "description": "E₈ minimum-vector squared norm."},
    "e8_kissing":     {"value": 240,                "formula": "240 = 112 + 128",
                        "kind": "lattice",
                        "description": "E₈ kissing number (root system size)."},
    "leech_min_norm": {"value": 2.0,                "formula": "2",
                        "kind": "lattice",
                        "description": "Leech lattice minimum-vector norm."},
    "leech_min_norm_sq": {"value": 4.0,             "formula": "4",
                        "kind": "lattice",
                        "description": "Leech lattice minimum-vector squared norm."},
    "leech_kissing": {"value": 196560,             "formula": "196560",
                        "kind": "lattice",
                        "description": "Leech lattice kissing number (24-D sphere packing optimum)."},
    "d4_kissing":     {"value": 24,                 "formula": "24",
                        "kind": "lattice",
                        "description": "D₄ kissing number — 24-cell vertices."},

    # ── Topology / G₂ holonomy invariants ───────────────────────────────
    "b3":             {"value": 24,                  "formula": "b₃(M_G₂) = 24",
                        "kind": "topology",
                        "description": "Third Betti number of the G₂-holonomy manifold."},
    "chi_eff":        {"value": 144,                 "formula": "χ_eff = 144",
                        "kind": "topology",
                        "description": "Effective Euler characteristic of the G₂ manifold."},
    "topology_invariant": {"value": 144.0,           "formula": "(b₃/24)·χ_eff",
                        "kind": "topology",
                        "description": "Conserved Φ-flow invariant. (b₃/24)·χ_eff = 144."},

    # ── Heterotic / racetrack constants ─────────────────────────────────
    "alpha_leak":     {"value": 1.0 / math.sqrt(6.0), "formula": "1/√6",
                        "kind": "heterotic",
                        "description": "Dark portal coupling from E₇ ⊃ E₆×U(1) Clebsch-Gordan."},
    "portal_coupling": {"value": 1.0 / math.sqrt(6.0), "formula": "1/√6",
                        "kind": "heterotic",
                        "description": "Alias for alpha_leak."},
    "lambda_eff":     {"value": _LAMBDA_EFF,          "formula": "exp(-2π/24)",
                        "kind": "heterotic",
                        "description": "Racetrack-stabilised modulus  λ_eff = exp(-2π/N₁) with N₁ = b₃ = 24."},
    "cabibbo_proxy":  {"value": _LAMBDA_EFF ** 3,     "formula": "exp(-2π/24)³",
                        "kind": "heterotic",
                        "description": "λ_eff³ — proxy for the Cabibbo angle ε."},
    "n1_flux":        {"value": 24,                   "formula": "N₁ = b₃ = 24",
                        "kind": "heterotic",
                        "description": "Dominant racetrack flux quantum (= b₃)."},
    "n2_flux":        {"value": 23,                   "formula": "N₂ = b₃ − 1 = 23",
                        "kind": "heterotic",
                        "description": "Sub-dominant racetrack flux quantum."},

    # ── Spectral / Φ-flow seeds ─────────────────────────────────────────
    "edof":           {"value": 3,                    "formula": "EDOF = 3",
                        "kind": "spectral",
                        "description": "G₂ seed effective degrees of freedom."},
    "g2_seed_t_re":   {"value": 7.086,                "formula": "Re(T) ≈ 7.086",
                        "kind": "spectral",
                        "description": "Real part of the Pneuma modulus T at the G₂ seed."},
    "g2_seed_lambda": {"value": 1.586,                "formula": "λ_VEV ≈ 1.586",
                        "kind": "spectral",
                        "description": "Vacuum-expectation value λ at the G₂ seed."},

    # ── Discrete / Planck ───────────────────────────────────────────────
    "planck_d":       {"value": 1.0,                  "formula": "D = 1.0",
                        "kind": "physics",
                        "description": "Default discrete-quantization scale (eml-spectral.constants.PLANCK_D)."},
}


# ── Public API ──────────────────────────────────────────────────────────────

def Get(name: str, *, as_json: bool = False) -> Union[Dict[str, Any], str]:
    """Return a JSON-serialisable datasheet for *name*.

    Looks up the spectral catalogue first, then falls back to
    :func:`eml_math.Get` for math constants (pi, e, phi, sqrt2, etc.) so
    callers have a single uniform entry point regardless of which layer
    a constant belongs to.

    Parameters
    ----------
    name : str
        Constant name. Case-insensitive; whitespace + underscores are
        normalised. Aliases supported (e.g. ``portal_coupling`` ↔
        ``alpha_leak``).
    as_json : bool, default False
        When True, return the result as a JSON-encoded string instead of
        a dict.

    Raises
    ------
    KeyError
        If *name* is unknown to both eml-spectral and eml-math.

    Returns
    -------
    dict (or str if ``as_json``) with keys:

    * ``name``        — canonical name
    * ``value``       — int or float
    * ``formula``     — human-readable derivation string
    * ``kind``        — category tag
    * ``description`` — one-line what-it-is
    * ``source``      — ``"eml-spectral"`` or ``"eml-math"``

    Examples
    --------
    >>> Get('E8_dim')['value']
    248
    >>> Get('alpha_leak')['value']
    0.408248290463863...
    >>> Get('lambda_eff')['formula']
    'exp(-2π/24)'
    >>> Get('pi')['value']     # delegated to eml-math
    3.141592653589793
    """
    raw = name.strip()
    key = raw.lower().replace(" ", "_").replace("-", "_")

    entry = SPECTRAL_CATALOGUE.get(key)
    if entry is not None:
        out: Dict[str, Any] = {
            "name":        key,
            "value":       entry["value"],
            "formula":     entry["formula"],
            "kind":        entry["kind"],
            "description": entry["description"],
            "source":      "eml-spectral",
        }
        return json.dumps(out, ensure_ascii=False) if as_json else out

    # Delegate to eml-math for math constants (pi, e, phi, …).
    try:
        from eml_math import Get as _eml_Get
    except ImportError as exc:   # pragma: no cover
        raise KeyError(
            f"unknown spectral constant {name!r} and eml-math not available"
        ) from exc

    try:
        delegated = _eml_Get(name)
    except KeyError:
        raise KeyError(f"unknown constant: {name!r}")
    if isinstance(delegated, dict):
        delegated = {**delegated, "source": "eml-math"}
    return json.dumps(delegated, ensure_ascii=False) if as_json else delegated


def list_constants() -> List[str]:
    """Return every constant name eml_spectral.Get knows about.

    Includes both spectral-layer entries (algebra dimensions, lattice
    invariants, racetrack constants, …) and the math-layer entries
    delegated from :func:`eml_math.list_constants`.
    """
    names = set(SPECTRAL_CATALOGUE.keys())
    try:
        from eml_math import list_constants as _eml_list
        names.update(_eml_list())
    except ImportError:   # pragma: no cover
        pass
    return sorted(names)
