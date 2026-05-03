"""
Freudenthal Triple System on J₃(𝕆) — 27-dimensional exceptional Jordan algebra.

The Freudenthal triple system (FTS) is the algebraic structure underlying the
27-dimensional representation of E₆ and the 56-dimensional representation of E₇.
It is built from the exceptional Jordan algebra J₃(𝕆): 3×3 Hermitian matrices
over the octonions 𝕆.

J₃(𝕆) has 27 real dimensions:
  - 3 diagonal real entries  (c₁, c₂, c₃) ∈ ℝ³
  - 3 off-diagonal octonion entries (x₁, x₂, x₃) ∈ 𝕆³   (8 dims each)
  - Total: 3 + 3×8 = 27 ✓

Layout of the 27 components:
  elements[0:3]   = (c₁, c₂, c₃)   — diagonal real entries
  elements[3:11]  = x₁              — off-diagonal octonion (8 components)
  elements[11:19] = x₂              — off-diagonal octonion (8 components)
  elements[19:27] = x₃              — off-diagonal octonion (8 components)

The cubic norm (Jordan determinant):
  N(c, x) = c₁c₂c₃ − c₁|x₁|² − c₂|x₂|² − c₃|x₃|² + 2 Re(x₁ x₂ x₃)

The Jordan trace:  Tr(A) = c₁ + c₂ + c₃

The bilinear Jordan inner product:
  ⟨A, B⟩ = c₁d₁ + c₂d₂ + c₃d₃ + 2(Re(x₁·ȳ₁) + Re(x₂·ȳ₂) + Re(x₃·ȳ₃))

References:
  - Brown, R.B. (1969) "Groups of type E7." J. Reine Angew. Math. 236, 79–102
  - Günaydin, Sierra, Townsend (1983) "Exceptional supergravity"
  - McCrimmon (1978) "Jordan algebras and their applications"

In the Principia Metaphysica framework, the 27D Pneuma condensate in the bulk
M²⁷(24,1,2) is identified with elements of this FTS. The cubic norm gives the
racetrack potential V_bridge/V_face; the quartic invariant of the associated
E₇ representation encodes the ALP mass scale and spectral residues.

Copyright (c) 2025-2026 Andrew Keith Watts. All rights reserved.
"""
from __future__ import annotations

import math
from typing import List, Union

from eml_math.point import EMLPoint
from eml_spectral.octonion import Octonion


Number = Union[float, int, EMLPoint]


def _to_float(v: Number) -> float:
    if isinstance(v, EMLPoint):
        return float(v.tension())
    return float(v)


class FreudenthalTripleSystem:
    """
    Freudenthal triple system on J₃(𝕆) — the 27-dimensional exceptional Jordan algebra.

    This is the algebraic structure that models the 27D Pneuma condensate in the
    M²⁷(24,1,2) bulk of Principia Metaphysica.

    Parameters
    ----------
    elements : list of 27 floats or EMLPoints
        Layout: [c₁, c₂, c₃, x₁₀…x₁₇, x₂₀…x₂₇, x₃₀…x₃₇]
        where (c₁, c₂, c₃) are the real diagonal entries and
        x₁, x₂, x₃ are the off-diagonal octonion entries (8 floats each).
    """

    DIM = 27

    def __init__(self, elements: List[Number]) -> None:
        if len(elements) != self.DIM:
            raise ValueError(
                f"FreudenthalTripleSystem requires exactly 27 elements, got {len(elements)}"
            )
        self._raw = [_to_float(e) for e in elements]

        # Diagonal entries
        self._c = (self._raw[0], self._raw[1], self._raw[2])

        # Off-diagonal octonion entries (8 components each)
        self._x1 = Octonion([EMLPoint(self._raw[3 + i], 1.0) for i in range(8)])
        self._x2 = Octonion([EMLPoint(self._raw[11 + i], 1.0) for i in range(8)])
        self._x3 = Octonion([EMLPoint(self._raw[19 + i], 1.0) for i in range(8)])

    # ── properties ──────────────────────────────────────────────────────────────

    @property
    def diagonal(self):
        """The three real diagonal entries (c₁, c₂, c₃)."""
        return self._c

    @property
    def octonion_x1(self) -> Octonion:
        return self._x1

    @property
    def octonion_x2(self) -> Octonion:
        return self._x2

    @property
    def octonion_x3(self) -> Octonion:
        return self._x3

    # ── algebraic operations ─────────────────────────────────────────────────────

    def jordan_trace(self) -> float:
        """Tr(A) = c₁ + c₂ + c₃."""
        return self._c[0] + self._c[1] + self._c[2]

    def jordan_norm_sq(self) -> float:
        """⟨A, A⟩ = c₁² + c₂² + c₃² + 2(|x₁|² + |x₂|² + |x₃|²)."""
        c_sq = sum(c * c for c in self._c)
        x_sq = (
            self._x1.norm_sq()
            + self._x2.norm_sq()
            + self._x3.norm_sq()
        )
        return c_sq + 2.0 * x_sq

    def bilinear_form(self, other: "FreudenthalTripleSystem") -> float:
        """
        Jordan inner product ⟨A, B⟩.

        ⟨A, B⟩ = c₁d₁ + c₂d₂ + c₃d₃ + 2(Re(x₁·ȳ₁) + Re(x₂·ȳ₂) + Re(x₃·ȳ₃))
        """
        diag_part = sum(self._c[i] * other._c[i] for i in range(3))

        def re_inner(a: Octonion, b: Octonion) -> float:
            b_conj = b.conjugate()
            prod = a * b_conj
            return prod.component(0)

        oct_part = (
            re_inner(self._x1, other._x1)
            + re_inner(self._x2, other._x2)
            + re_inner(self._x3, other._x3)
        )
        return diag_part + 2.0 * oct_part

    def cubic_norm(self) -> float:
        """
        Cubic norm (Jordan determinant) of J₃(𝕆).

        N(A) = c₁c₂c₃ − c₁|x₁|² − c₂|x₂|² − c₃|x₃|² + 2 Re(x₁ x₂ x₃)

        In the PM framework this equals the racetrack potential V_bridge / V_face.
        """
        c1, c2, c3 = self._c

        # Diagonal cubic
        diag_cubic = c1 * c2 * c3

        # Off-diagonal subtractions
        off_diag = (
            c1 * self._x1.norm_sq()
            + c2 * self._x2.norm_sq()
            + c3 * self._x3.norm_sq()
        )

        # Triple octonion product term: Re(x₁(x₂x₃))
        x2_x3 = self._x2 * self._x3
        x1_x2_x3 = self._x1 * x2_x3
        re_triple = x1_x2_x3.component(0)

        return diag_cubic - off_diag + 2.0 * re_triple

    def quartic(self) -> float:
        """
        Quartic invariant q(A) of the FTS.

        This is related to the quartic Casimir of E₇ acting on the 56D rep
        when lifted to (A, A*) ∈ 27 ⊕ 27*.

        q(A) = Tr(A)² · |A|² / 4   (simplified proxy for the full E₇ quartic)

        In PM: encodes the 125 spectral residue scale and the ALP mass.
        """
        tr = self.jordan_trace()
        norm_sq = self.jordan_norm_sq()
        return (tr * tr * norm_sq) / 4.0

    def jordan_square(self) -> "FreudenthalTripleSystem":
        """
        Jordan square A ∘ A = A² in J₃(𝕆).

        For diagonal part:  (A²)ᵢᵢ = cᵢ² + |xⱼ|² + |xₖ|²
        (simplified; full non-commutative formula for off-diagonals omitted)
        """
        c1, c2, c3 = self._c
        x1_sq = self._x1.norm_sq()
        x2_sq = self._x2.norm_sq()
        x3_sq = self._x3.norm_sq()

        # Diagonal of A² (Jordan product diagonal):
        # (A²)₁₁ = c₁² + |x₃|² + |x₂|²  (from row-1 dot row-1 in Hermitian matrix)
        # (A²)₂₂ = c₂² + |x₁|² + |x₃|²
        # (A²)₃₃ = c₃² + |x₂|² + |x₁|²
        d1 = c1 * c1 + x2_sq + x3_sq
        d2 = c2 * c2 + x1_sq + x3_sq
        d3 = c3 * c3 + x1_sq + x2_sq

        # Off-diagonal: (A²)_{ij} = cᵢ xₖ + cⱼ xₖ + ... (octonion products)
        # Simplified: scale each xᵢ by the sum of relevant diagonal entries
        scale_x1 = (c2 + c3) / 2.0
        scale_x2 = (c1 + c3) / 2.0
        scale_x3 = (c1 + c2) / 2.0

        new_x1 = [scale_x1 * self._x1.component(i) for i in range(8)]
        new_x2 = [scale_x2 * self._x2.component(i) for i in range(8)]
        new_x3 = [scale_x3 * self._x3.component(i) for i in range(8)]

        new_elements = [d1, d2, d3] + new_x1 + new_x2 + new_x3
        return FreudenthalTripleSystem(new_elements)

    def triple_product(
        self,
        y: "FreudenthalTripleSystem",
        z: "FreudenthalTripleSystem",
    ) -> "FreudenthalTripleSystem":
        """
        Freudenthal triple product {x, y, z}.

        Simplified implementation via the Jordan identity:
        {x, y, z} = (x ∘ y) ∘ z + (z ∘ y) ∘ x − y ∘ (x ∘ z) (schematic)

        In practice uses the Freudenthal–Brown construction via bilinear form:
        {x, y, z}ᵢ = ⟨x, yᵢ⟩ zᵢ + ⟨z, yᵢ⟩ xᵢ − ⟨x, z⟩ yᵢ
        (component-wise, schematic for the linearised version)
        """
        # Bilinear form coefficients
        xy = self.bilinear_form(y)
        zy = z.bilinear_form(y)
        xz = self.bilinear_form(z)

        # Triple product components: linear combination of x, y, z elements
        result = []
        for i in range(self.DIM):
            xi = self._raw[i]
            yi = y._raw[i]
            zi = z._raw[i]
            val = xy * zi + zy * xi - xz * yi
            result.append(val)

        return FreudenthalTripleSystem(result)

    # ── factories ────────────────────────────────────────────────────────────────

    @classmethod
    def from_scalar(cls, value: float) -> "FreudenthalTripleSystem":
        """Scalar multiple of identity: A = value × diag(1, 1, 1)."""
        elems = [value, value, value] + [0.0] * 24
        return cls(elems)

    @classmethod
    def from_pneuma_condensate(cls, b3: float = 24.0) -> "FreudenthalTripleSystem":
        """
        Pneuma condensate element from PM's M²⁷(24,1,2) bulk.

        Models the 27D Pneuma as a symmetric diagonal element scaled by b₃/27,
        with off-diagonal octonion entries set to the standard basis scale.

        Parameters
        ----------
        b3 : float
            G₂ Betti number (default 24).
        """
        scale = b3 / 27.0
        diag = [scale, scale, scale]
        oct_entry = [scale / math.sqrt(8.0)] * 8  # unit norm octonion scaled
        elems = diag + oct_entry + oct_entry + oct_entry
        return cls(elems)

    # ── dunder ───────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        tr = self.jordan_trace()
        n = self.cubic_norm()
        return f"FreudenthalTripleSystem(Tr={tr:.6g}, N={n:.6g})"

    def __len__(self) -> int:
        return self.DIM

    def __getitem__(self, idx: int) -> float:
        return self._raw[idx]
