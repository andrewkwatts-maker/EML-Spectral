"""
eml_spectral.spacetime — Lorentz-invariant operations on EMLPoint.

In the spacetime interpretation of an EMLPoint(x, y) the time component
is t = exp(x) and the space component is s = ln|y|. The Minkowski
interval Δ_M = √(t² − s²) is invariant under Lorentz boosts; this module
provides functional helpers (each takes an EMLPoint as the first arg).

These were originally methods on EMLPoint in eml-math; they were lifted
out as part of the v1.2.0 split so the core eml-math package has no
spacetime narrative. Same math, same numerics, just a functional API.

Example
-------
>>> from eml_math import EMLPoint
>>> from eml_spectral.spacetime import minkowski_delta, boost
>>> p = EMLPoint(1.0, 2.718)
>>> abs(minkowski_delta(p) - minkowski_delta(boost(p, rapidity=0.5))) < 1e-9
True
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from eml_math.constants import OVERFLOW_THRESHOLD

if TYPE_CHECKING:
    from eml_math import EMLPoint
    from eml_spectral.pair import EMLPair

__all__ = [
    "pair",
    "euclidean_delta",
    "minkowski_delta",
    "is_timelike",
    "is_spacelike",
    "is_lightlike",
    "canonical_frame",
    "rapidity",
    "boost",
    "boost_velocity",
    "light_cone_coordinates",
    "light_cone_type",
    "future_light_cone",
    "rest_energy",
    "proper_time",
]


def pair(point: "EMLPoint") -> "EMLPair":
    """Returns (exp(x), ln(|y|)) as an EMLPair — the canonical frame coordinates."""
    from eml_spectral.pair import EMLPair
    xv, y_safe = _xy_safe(point)
    return EMLPair.from_values(math.exp(xv), math.log(y_safe))


def _xy_safe(point: "EMLPoint"):
    """Return (xv, y_safe) — xv guarded against overflow and y_safe > 0
    so that ln(y_safe) is well-defined. Mirrors the guard previously used
    by the EMLPoint methods."""
    xv = point.x
    if xv > OVERFLOW_THRESHOLD:
        xv = math.log(xv)
    yv = point.y
    y_safe = abs(yv) if yv <= 0 else yv
    if y_safe == 0:
        y_safe = 1e-300
    return xv, y_safe


def euclidean_delta(point: "EMLPoint") -> float:
    """sqrt(exp(2x) + (ln y)²) — Euclidean invariant under 4-frame rotations."""
    xv, y_safe = _xy_safe(point)
    ex = math.exp(xv)
    ly = math.log(y_safe)
    return math.sqrt(ex * ex + ly * ly)


def minkowski_delta(point: "EMLPoint", signature: str = "+---", c: float = 1.0) -> float:
    """Minkowski invariant interval √|exp(2x) − (c·ln y)²| (signature
    ``"+---"`` → time-like positive)."""
    xv, y_safe = _xy_safe(point)
    t_comp = math.exp(xv)
    x_comp = c * math.log(y_safe)
    if signature.startswith("+"):
        ds2 = t_comp * t_comp - x_comp * x_comp
    else:
        ds2 = x_comp * x_comp - t_comp * t_comp
    return math.sqrt(abs(ds2))


def is_timelike(point: "EMLPoint", c: float = 1.0) -> bool:
    xv, y_safe = _xy_safe(point)
    return math.exp(xv) ** 2 > (c * math.log(y_safe)) ** 2


def is_spacelike(point: "EMLPoint", c: float = 1.0) -> bool:
    xv, y_safe = _xy_safe(point)
    return math.exp(xv) ** 2 < (c * math.log(y_safe)) ** 2


def is_lightlike(point: "EMLPoint", c: float = 1.0, tol: float = 1e-9) -> bool:
    xv, y_safe = _xy_safe(point)
    return abs(math.exp(xv) ** 2 - (c * math.log(y_safe)) ** 2) < tol


def canonical_frame(point: "EMLPoint", k: int = 0) -> "EMLPair":
    """Rotate (exp(x), ln y) through one of four Euclidean frames {1, i, -1, -i}.
    Δ_E is invariant across all four."""
    from eml_spectral.pair import EMLPair
    xv, y_safe = _xy_safe(point)
    r  = math.exp(xv)
    im = math.log(y_safe)
    k = k % 4
    if k == 0: return EMLPair.from_values( r,  im)
    if k == 1: return EMLPair.from_values(-im,  r)
    if k == 2: return EMLPair.from_values(-r, -im)
    return            EMLPair.from_values( im, -r)


def rapidity(point: "EMLPoint") -> float:
    """φ = atanh(ln y / exp x). Raises ValueError on spacelike points."""
    xv, y_safe = _xy_safe(point)
    t_comp = math.exp(xv)
    x_comp = math.log(y_safe)
    if abs(t_comp) < 1e-300:
        raise ValueError("Cannot compute rapidity: time component is zero")
    ratio = x_comp / t_comp
    if abs(ratio) >= 1.0:
        raise ValueError(
            f"Cannot compute rapidity: |space/time| = {abs(ratio):.6g} >= 1 "
            "(point is not timelike)"
        )
    return math.atanh(ratio)


def boost(point: "EMLPoint", rapidity: float, c: float = 1.0) -> "EMLPoint":
    """Lorentz boost by `rapidity` φ — returns a new EMLPoint with the
    same Minkowski interval as `point`."""
    from eml_math import EMLPoint
    xv, y_safe = _xy_safe(point)
    t_comp = math.exp(xv)
    x_comp = math.log(y_safe)
    ch = math.cosh(rapidity)
    sh = math.sinh(rapidity)
    t_new = t_comp * ch - (x_comp / c) * sh
    x_new = x_comp * ch - t_comp * c * sh
    if t_new <= 0:
        t_new = 1e-300
    x_out = math.log(t_new)
    if x_new > 709.0:   x_new = 709.0
    elif x_new < -709.0: x_new = -709.0
    y_out = math.exp(x_new)
    return EMLPoint(x_out, y_out, D=getattr(point, "_D", None))


def boost_velocity(point: "EMLPoint", v: float, c: float = 1.0) -> "EMLPoint":
    """Boost by velocity v (= atanh(v/c) rapidity)."""
    if abs(v) >= c:
        raise ValueError(f"Speed |v| = {abs(v):.6g} must be less than c = {c:.6g}")
    return boost(point, rapidity=math.atanh(v / c), c=c)


def light_cone_coordinates(point: "EMLPoint", c: float = 1.0) -> "tuple[float, float]":
    """Null coordinates (u, v) = (t + x/c, t − x/c)."""
    xv, y_safe = _xy_safe(point)
    t_comp = math.exp(xv)
    x_comp = math.log(y_safe) / c
    return t_comp + x_comp, t_comp - x_comp


def light_cone_type(point: "EMLPoint", c: float = 1.0) -> str:
    """Returns 'timelike' | 'spacelike' | 'lightlike'."""
    if is_lightlike(point, c=c): return "lightlike"
    if is_timelike(point, c=c):  return "timelike"
    return "spacelike"


def future_light_cone(point: "EMLPoint", c: float = 1.0) -> bool:
    """True if event is in the future light cone (timelike + exp(x) > 0,
    which is always true for real x)."""
    return is_timelike(point, c=c)


def rest_energy(point: "EMLPoint", c: float = 1.0) -> float:
    """Δ_M in natural units — rest energy E₀ = m·c² (c=1 → rest mass)."""
    return minkowski_delta(point, signature="+---", c=c)


def proper_time(point: "EMLPoint", c: float = 1.0) -> float:
    """Proper time τ = Δ_M / c."""
    return rest_energy(point, c=c) / c
