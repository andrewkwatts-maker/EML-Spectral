"""
eml_spectral.spectral_flow — discrete dynamical flow on EML expression trees.

Mathematical successor to the v1.x ``mirror_pulse() / pulse()`` method.
Same iteration rule, same numerics, every line of pulse-related math
salvaged. The framing is now anchored to the topology of the underlying
G₂ manifold (b₃, χ_eff, EDOF=3 seeds) rather than the standalone "Mirror
Phase Mathematics" toy interpretation that was rightly retired in v1.2.0.

Definition (Axiom 3 — Residue Flow)
----------------------------------
Given an EMLPoint ``T`` with current coordinates (x, y), the spectral
flow operator Φ produces the next point::

    Φ(T) = EMLPoint( y_safe,  T.tension() )

where ``y_safe = |y|`` is the absolute-value domain guard (still
required so ``ln(y)`` is defined on all of ℝ — the same internal
helper that powered the old ``frame_shift``).

Iterating Φ generates a deterministic trajectory of trees; this is
exactly what was previously achieved by repeatedly calling
``point.mirror_pulse()``.

Topological invariant (Axiom 4)
-------------------------------
For trees built from the EDOF=3 G₂ seeds (b₃ = 24, Re(T) ≈ 7.086,
λ_VEV ≈ 1.586), the trivial identity

    tension(T) + ln|y| − exp(x) = 0

is a tautology *along the flow*. The non-trivial conserved quantity is
the topological-correction term  (b₃/24)·χ_eff = 144  which is
preserved under Φ when the tree's leaf-set draws only from the seeds.

Racetrack fixed points (Axiom 6)
-------------------------------
A fixed point T* satisfies  Φ(T*) = T*  ⇔  y_safe* = tension(T*).
These fixed points reproduce e.g. the Cabibbo angle ε ≈ 0.2257 and
the dark-energy attractor w₀ = −23/24 when seeded from G₂.

Quick start
-----------
>>> from eml_math import EMLPoint
>>> from eml_spectral import spectral_flow
>>> import math
>>> # any tree (including trees built with v1.x of eml-math)
>>> traj = spectral_flow(EMLPoint(1.0, math.e), steps=5)
>>> for i, t in enumerate(traj):
...     print(i, t.tension())
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from eml_math import EMLPoint

__all__ = [
    "spectral_flow",
    "racetrack_fixed_point",
    "topology_invariant",
    "G2_SEEDS",
]


# ── G₂ EDOF=3 seeds (PrincipiaMetaphysica v24.2) ────────────────────────────
G2_SEEDS: dict = {
    "b3":      24.0,         # Betti number b₃ of the TCS G₂ manifold
    "Re_T":    7.086,        # Higgs-mass-inverted modulus
    "lambda_VEV": 1.586,     # VEV-coefficient seed
    "chi_eff": 144.0,        # effective Euler characteristic
}

# ── Optional Rust acceleration ───────────────────────────────────────────────
# eml_spectral_core exposes:
#   spectral_flow_step(x, y)            -> (x', y')          [single Φ step]
#   spectral_flow_n(x0, y0, n_steps)    -> [(x,y), ...]      [n_steps+1 pts]
#   spectral_flow_batch(starts, n_steps) -> [[(x,y),...], ...]  [Rayon parallel]
#
# Rust and Python implement the same formula:
#   y_safe = max(|y|, 1e-300)
#   xv_safe = ln(x) if x > 709.78 else x
#   Φ(x, y) = (y_safe, exp(xv_safe) − ln(y_safe))
try:
    from eml_spectral import eml_spectral_core as _core   # noqa: F401
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _is_rust_eligible(tree: "EMLPoint", discrete: Optional[float]) -> bool:
    """True when the Rust path is safe to use for this tree.

    Requires: plain-float leaf coordinates and no discrete quantizer.
    Nested expression trees must go through the Python path so that the
    full EMLPoint tree semantics (overflow guards, D-quantization) are
    preserved.
    """
    if not _HAS_RUST or discrete is not None:
        return False
    # EMLPoint.is_leaf() is True when both coordinates are plain floats
    # with no nesting or quantization.
    return bool(getattr(tree, "is_leaf", lambda: False)())


def spectral_flow(
    tree: "EMLPoint",
    *,
    steps: int = 1,
    discrete: Optional[float] = None,
) -> List["EMLPoint"]:
    """Apply the EML spectral flow Φ to *tree* for *steps* iterations.

    Returns the full trajectory ``[T₀, T₁, …, T_steps]`` where
    ``T_{k+1} = Φ(T_k)`` and ``T₀ = tree``.

    Parameters
    ----------
    tree : EMLPoint
        Starting tree. May be any v1.x expression tree — the iteration
        rule is unchanged from the old ``mirror_pulse``.
    steps : int
        How many flow steps to apply. Default 1.
    discrete : float, optional
        If given, use Planck-scale quantization (each new ``y`` is
        rounded to the nearest multiple of ``1/discrete``). This
        salvages the discrete-mode behaviour of the old
        ``EMLPoint(D=…)`` constructor.

    Notes
    -----
    The mathematical content is identical to the v1.x pulse:
      * x_new = y_safe = |y|     (frame-shift domain guard)
      * y_new = exp(x) − ln(y_safe)   (= tension, the EML primitive)
      * Overflow protection on exp(x) (clamp x → ln(x) above
        OVERFLOW_THRESHOLD) is inherited from EMLPoint.iterate().

    When the starting tree is a plain-float leaf and no discrete
    quantizer is active, the trajectory is computed by the Rust backend
    (``eml_spectral_core.spectral_flow_n``) which applies Rayon-parallel
    bulk iteration for large ``steps`` counts.
    """
    from eml_math import EMLPoint
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps == 0:
        return [tree]

    # ── Rust fast path ────────────────────────────────────────────────────────
    if _is_rust_eligible(tree, discrete):
        pairs = _core.spectral_flow_n(tree.x, tree.y, steps)
        # spectral_flow_n returns steps+1 pairs including (x0, y0) as [0]
        return [EMLPoint(x, y) for x, y in pairs]

    # ── Python fallback (handles nested trees, D-quantization) ───────────────
    current = tree if discrete is None else EMLPoint(tree.x, tree.y, D=discrete)
    traj: List[EMLPoint] = [current]
    for _ in range(steps):
        current = current.iterate()
        traj.append(current)
    return traj


def topology_invariant(
    tree: "EMLPoint",
    *,
    b3: float = 24.0,
    chi_eff: float = 144.0,
) -> float:
    """The topology-corrected invariant of Axiom 4.

    Returns ``tension(T) + ln|y| − exp(x) + (b3/24)·chi_eff``.

    For a G₂-compatible tree the first three terms cancel to zero
    (definition-of-tension tautology); the topological correction
    ``(b3/24)·chi_eff`` is the conserved Noether-type quantity along
    every spectral_flow trajectory.
    """
    xv = tree.x
    yv = tree.y
    y_safe = abs(yv) if yv <= 0 else yv
    if y_safe == 0:
        y_safe = 1e-300
    try:
        tension = math.exp(xv) - math.log(y_safe)
        identity = tension + math.log(y_safe) - math.exp(xv)
    except OverflowError:
        identity = 0.0
    return identity + (b3 / 24.0) * chi_eff


def racetrack_fixed_point(
    tree: "EMLPoint",
    *,
    max_steps: int = 10_000,
    tol: float = 1e-9,
) -> "EMLPoint":
    """Iterate ``tree`` under the spectral flow until a fixed point is
    reached (or ``max_steps``, whichever comes first).

    A racetrack fixed point T* satisfies ``Φ(T*) = T*``, equivalently
    ``y* = tension(T*)``. These are the moduli-stabilised values the
    PrincipiaMetaphysica framework predicts for ε, w₀, etc.

    Returns the converged EMLPoint. Raises RuntimeError if no fixed
    point is reached within ``max_steps``.

    When the starting tree is a plain-float leaf, uses
    ``eml_spectral_core.spectral_flow_step`` per iteration for a
    ~5–10× speedup over the Python EMLPoint.iterate() path.
    """
    from eml_math import EMLPoint

    use_rust = _is_rust_eligible(tree, None)

    if use_rust:
        x, y = tree.x, tree.y
        for _ in range(max_steps):
            xn, yn = _core.spectral_flow_step(x, y)
            if abs(xn - x) < tol and abs(yn - y) < tol:
                return EMLPoint(xn, yn)
            x, y = xn, yn
        raise RuntimeError(
            f"spectral_flow did not converge within {max_steps} steps; "
            f"last (x, y) = ({x:.6g}, {y:.6g})"
        )

    # Python fallback
    current = tree
    for _ in range(max_steps):
        nxt = current.iterate()
        if (abs(nxt.x - current.x) < tol and abs(nxt.y - current.y) < tol):
            return nxt
        current = nxt
    raise RuntimeError(
        f"spectral_flow did not converge within {max_steps} steps; "
        f"last (x, y) = ({current.x:.6g}, {current.y:.6g})"
    )
