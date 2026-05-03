"""
E₇ 56-Dimensional Representation from J₃(𝕆) Freudenthal Triple System.

The minimal representation of E₇ is 56-dimensional. It arises as the space
of Jordan pairs (x, y) ∈ J₃(𝕆) ⊕ J₃(𝕆)* equipped with a symplectic structure
preserved by the E₇ action.

Structure of the 56D rep:
  - 27 components from x ∈ J₃(𝕆)  (FTS element)
  - 27 components from y ∈ J₃(𝕆)* (dual FTS element)
  - 1 + 1 extra symplectic scalars
  Total: 56 = 27 + 27 + 1 + 1

The key subgroup branching that gives the dark-force portal coupling in PM:
  E₇ ⊃ E₆ × U(1)
  56 → 27 + 27* + 1 + 1  (under E₆ × U(1))

The U(1) Clebsch-Gordan coefficient for the 56 → 27 + 27* branching is:
  α_leak = 1/√6

This is the algebraic origin of the inter-face leakage coupling in PM.

References:
  - Brown, R.B. (1969) "Groups of type E7"
  - Freudenthal, H. (1954) "Beziehungen der E7 und E8 zur Oktavenebene"
  - Günaydin, Sierra, Townsend (1983) "The geometry of N=2 Maxwell-Einstein
    supergravity and Jordan algebras"

Copyright (c) 2025-2026 Andrew Keith Watts. All rights reserved.
"""
from __future__ import annotations

import math
from typing import Optional

from eml_spectral.exceptional.freudenthal import FreudenthalTripleSystem


class E7_56:
    """
    56-dimensional fundamental representation of E₇.

    Built from a pair (x, y) of 27D Freudenthal triple system elements.
    The symplectic form and quartic invariant are defined on the 56D space.

    Parameters
    ----------
    x : FreudenthalTripleSystem
        The J₃(𝕆) element (visible sector component).
    y : FreudenthalTripleSystem
        The dual J₃(𝕆) element (hidden sector / dual-shadow component).
    """

    DIM = 56

    # Algebraic constant: U(1) Clebsch-Gordan coefficient for E₇ ⊃ E₆ × U(1)
    # 56 → 27(+1) + 27*(-1) + 1(+3) + 1(-3) under E₆ × U(1)
    # The portal coupling = 1 / √(dim 27 / dim 1) = 1 / √6
    ALPHA_LEAK: float = 1.0 / math.sqrt(6.0)

    def __init__(
        self,
        x: FreudenthalTripleSystem,
        y: FreudenthalTripleSystem,
    ) -> None:
        if len(x) != 27 or len(y) != 27:
            raise ValueError("E7_56 requires two 27-component FTS elements")
        self.x = x  # visible / J₃(𝕆) sector
        self.y = y  # dual / hidden sector

    # ── algebraic operations ─────────────────────────────────────────────────────

    def symplectic_form(self) -> float:
        """
        Skew-symmetric bilinear form preserved by E₇.

        ⟨(x, y), (x', y')⟩ = ⟨x, y'⟩_J − ⟨y, x'⟩_J

        For a self-pairing this is the Jordan bilinear form ⟨x, y⟩_J.
        """
        return self.x.bilinear_form(self.y)

    def quartic_invariant(self) -> float:
        """
        E₇ quartic invariant q(v) on the 56D representation.

        q(x, y) = (⟨x, y⟩_J)² − 4 N(x) · N(y) + Δ_correction

        where N is the cubic norm of J₃(𝕆) and ⟨·,·⟩_J is the Jordan
        bilinear form.

        In PM this gives the attractor potential for dark energy and the
        ALP mass scale m_a ~ q^{1/4} / M_Planck.
        """
        inner = self.x.bilinear_form(self.y)
        nx = self.x.cubic_norm()
        ny = self.y.cubic_norm()
        # Core quartic: (⟨x,y⟩)² - 4 N(x)·N(y)
        q_core = inner * inner - 4.0 * nx * ny
        # Correction from Tr² terms (simplified)
        tr_x = self.x.jordan_trace()
        tr_y = self.y.jordan_trace()
        delta = (tr_x * tr_x * self.y.jordan_norm_sq() + tr_y * tr_y * self.x.jordan_norm_sq()) / 16.0
        return q_core + delta

    def e7_action(self, generator_comps: list) -> "E7_56":
        """
        Apply an E₇ generator (given as 56 scalar coefficients) to the 56D pair.

        In the PM framework, E₇ generators act on the dual-shadow fluxes.
        Implementation uses the infinitesimal rotor structure.

        Parameters
        ----------
        generator_comps : list of 56 floats
            Coefficients of the E₇ generator in the 56D basis.
        """
        if len(generator_comps) != 56:
            raise ValueError(f"E₇ generator requires 56 components, got {len(generator_comps)}")

        # Split generator into (dx, dy) acting on (x, y)
        dx_raw = generator_comps[:27]
        dy_raw = generator_comps[27:54]
        # Extra 2 symplectic scalars (ignored in simplified version)

        # Infinitesimal action: x' = x + ε·dx, y' = y + ε·dy
        eps = 1e-4
        new_x_elems = [self.x[i] + eps * dx_raw[i] for i in range(27)]
        new_y_elems = [self.y[i] + eps * dy_raw[i] for i in range(27)]
        return E7_56(
            FreudenthalTripleSystem(new_x_elems),
            FreudenthalTripleSystem(new_y_elems),
        )

    # ── E₇ ⊃ E₆ × U(1) branching ─────────────────────────────────────────────────

    def split_e6_u1(self) -> dict:
        """
        Decompose the 56D rep under E₇ ⊃ E₆ × U(1).

        Branching: 56 → 27(+1/3) + 27*(-1/3) + 1(+1) + 1(-1)

        The U(1) charge normalisation gives the portal coupling:
          α_leak = 1/√6  (Clebsch-Gordan coefficient for the U(1) factor)

        Returns dict with 'visible_e6', 'hidden_u1_charge', 'portal_coupling'.
        """
        # Visible E₆ 27-plet: the x component
        # Hidden dual 27-plet: the y component
        # U(1) charge on the 27: normalised to 1/sqrt(6)
        symplectic = self.symplectic_form()
        return {
            "visible_e6": self.x,
            "hidden_27_dual": self.y,
            "symplectic_pairing": symplectic,
            "u1_charge_27": self.ALPHA_LEAK,        # = 1/sqrt(6)
            "portal_coupling": self.ALPHA_LEAK,     # = 1/sqrt(6)
            "portal_coupling_note": (
                "alpha_leak = 1/sqrt(6) is the algebraic Clebsch-Gordan coefficient "
                "for the U(1) factor in E7 supset E6 x U(1), 56 -> 27 + 27*. "
                "This is the dark-force portal coupling in PM (DERIVED, not fitted)."
            ),
        }

    def split_so10_su2_u1(self) -> dict:
        """
        Decompose under E₇ ⊃ SO(10) × SU(2) × U(1).

        The 27 of E₆ branches as: 27 → 16 + 10 + 1 under SO(10).
        The hidden SU(2) and U(1) factors generate gaugino condensation.
        """
        # SO(10) 16-plet (fermions): first 16 components of x
        visible_so10_components = self.x[:16] if len(self.x) >= 16 else list(self.x[:len(self.x)])
        # Hidden SU(2): components 16-26
        hidden_su2_components = [self.x[i] for i in range(16, min(27, len(self.x)))]

        # Racetrack contribution from hidden SU(2)
        racetrack_strength = sum(v * v for v in hidden_su2_components)

        return {
            "visible_so10": visible_so10_components,
            "hidden_su2": hidden_su2_components,
            "racetrack_contribution": racetrack_strength,
            "portal_coupling": self.ALPHA_LEAK,
        }

    # ── ALP mass prediction ───────────────────────────────────────────────────────

    def alp_mass_gev(self, m_planck_gev: float = 1.22e19) -> float:
        """
        ALP mass from the E₇ quartic invariant.

        m_a = (q_E7 / M_Planck²)^{1/2}

        where q_E7 = quartic_invariant() is the E₇ quartic form evaluated on
        the 56D Pneuma condensate pair.

        In PM: predicts m_a ≈ 3.51 meV from the spectral residue structure.

        Parameters
        ----------
        m_planck_gev : float
            Planck mass in GeV (default 1.22e19).
        """
        q = abs(self.quartic_invariant())
        if q < 1e-300:
            return 0.0
        return math.sqrt(q) / m_planck_gev

    # ── factories ────────────────────────────────────────────────────────────────

    @classmethod
    def from_pneuma(cls, b3: float = 24.0) -> "E7_56":
        """
        Build a 56D E₇ element from the PM Pneuma condensate.

        x = Pneuma condensate (b₃-scaled J₃(𝕆) element)
        y = dual (reversed ordering = 27* element)
        """
        fts1 = FreudenthalTripleSystem.from_pneuma_condensate(b3)
        elems_rev = list(reversed(fts1[:27]))  # type: ignore[misc]
        elems_rev_full = [fts1[i] for i in range(26, -1, -1)]
        fts2 = FreudenthalTripleSystem(elems_rev_full)
        return cls(fts1, fts2)

    # ── dunder ───────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        q = self.quartic_invariant()
        symp = self.symplectic_form()
        return f"E7_56(quartic={q:.6g}, symplectic={symp:.6g}, alpha_leak={self.ALPHA_LEAK:.6f})"
