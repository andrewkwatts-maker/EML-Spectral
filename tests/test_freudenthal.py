"""Tests for FreudenthalTripleSystem (J₃(𝕆), 27D exceptional Jordan algebra)."""
from __future__ import annotations
import math
import pytest

from eml_spectral.exceptional import FreudenthalTripleSystem as FTS


# ── Construction ─────────────────────────────────────────────────────────────

class TestConstruction:

    def test_zero_element(self):
        z = FTS([0.0] * 27)
        assert z.cubic_norm() == 0.0

    def test_dim_constant(self):
        assert FTS.DIM == 27

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            FTS([0.0] * 26)

    def test_too_long_raises(self):
        with pytest.raises(ValueError):
            FTS([0.0] * 28)

    def test_from_scalar_dim(self):
        s = FTS.from_scalar(2.0)
        assert isinstance(s, FTS)

    def test_from_scalar_diagonal(self):
        s = FTS.from_scalar(3.0)
        assert s._c == (3.0, 3.0, 3.0)

    def test_from_scalar_offdiag_zero(self):
        s = FTS.from_scalar(5.0)
        assert s._x1.norm_sq() == 0.0

    def test_from_pneuma_b3_24(self):
        s = FTS.from_pneuma_condensate(b3=24.0)
        assert isinstance(s, FTS)
        assert math.isfinite(s.cubic_norm())

    @pytest.mark.parametrize("b3", [12.0, 18.0, 24.0, 30.0])
    def test_from_pneuma_finite(self, b3):
        s = FTS.from_pneuma_condensate(b3=b3)
        assert math.isfinite(s.cubic_norm())
        assert math.isfinite(s.quartic())


# ── Cubic norm — homogeneity & invariants ────────────────────────────────────

class TestCubicNorm:

    @pytest.mark.parametrize("v", [1.0, 2.0, -1.0, 0.5, 7.086])
    def test_from_scalar_norm_is_v_cubed(self, v):
        s = FTS.from_scalar(v)
        assert abs(s.cubic_norm() - v ** 3) < 1e-9

    def test_zero_norm(self):
        assert FTS([0.0] * 27).cubic_norm() == 0.0

    def test_diagonal_only(self):
        # diagonal (a, b, c) → cubic norm = a*b*c
        s = FTS([2.0, 3.0, 5.0] + [0.0] * 24)
        assert abs(s.cubic_norm() - 30.0) < 1e-12

    def test_diagonal_negative(self):
        s = FTS([-1.0, 2.0, -3.0] + [0.0] * 24)
        assert abs(s.cubic_norm() - 6.0) < 1e-12

    @pytest.mark.parametrize("lam", [0.5, 1.0, 2.0, 3.0])
    def test_homogeneity_degree_three(self, lam):
        a = FTS([1.0, 2.0, 3.0] + [0.0] * 24)
        b = FTS([lam, 2.0 * lam, 3.0 * lam] + [0.0] * 24)
        assert abs(b.cubic_norm() - lam ** 3 * a.cubic_norm()) < 1e-9

    def test_offdiag_subtracts(self):
        # diagonal (1,1,1) gives c1c2c3 = 1; adding any off-diag norm should reduce
        plain = FTS([1.0, 1.0, 1.0] + [0.0] * 24)
        with_off = FTS([1.0, 1.0, 1.0] + [1.0, 0, 0, 0, 0, 0, 0, 0] + [0.0] * 16)
        assert with_off.cubic_norm() < plain.cubic_norm()


# ── Quartic ──────────────────────────────────────────────────────────────────

class TestQuartic:

    def test_zero_quartic(self):
        assert FTS([0.0] * 27).quartic() == 0.0

    @pytest.mark.parametrize("v", [1.0, 2.0, 3.0])
    def test_from_scalar_quartic(self, v):
        # quartic = (Tr² · |A|²) / 4
        # For from_scalar(v): Tr = 3v, |A|² = 3v², so quartic = (9v² · 3v²)/4 = 27v⁴/4
        got = FTS.from_scalar(v).quartic()
        assert abs(got - 27.0 * v ** 4 / 4.0) < 1e-9

    def test_quartic_finite_for_pneuma(self):
        s = FTS.from_pneuma_condensate(b3=24.0)
        assert math.isfinite(s.quartic())
        assert s.quartic() >= 0.0


# ── Bilinear / Jordan inner product ──────────────────────────────────────────

class TestBilinearForm:

    def test_zero_with_anything(self):
        z = FTS([0.0] * 27)
        a = FTS([1.0, 2.0, 3.0] + [0.0] * 24)
        assert z.bilinear_form(a) == 0.0

    def test_diagonal_diagonal(self):
        a = FTS([1.0, 2.0, 3.0] + [0.0] * 24)
        b = FTS([4.0, 5.0, 6.0] + [0.0] * 24)
        # ⟨A, B⟩ = c1d1 + c2d2 + c3d3 (octonion parts are zero)
        assert abs(a.bilinear_form(b) - (4.0 + 10.0 + 18.0)) < 1e-9

    def test_symmetric_in_diagonal(self):
        a = FTS([1.0, 2.0, 3.0] + [0.0] * 24)
        b = FTS([4.0, 5.0, 6.0] + [0.0] * 24)
        assert abs(a.bilinear_form(b) - b.bilinear_form(a)) < 1e-9

    @pytest.mark.parametrize("v", [1.0, 2.0, 5.0])
    def test_bilinear_self(self, v):
        s = FTS.from_scalar(v)
        # ⟨s, s⟩ = 3v²
        assert abs(s.bilinear_form(s) - 3.0 * v * v) < 1e-9


# ── Jordan trace / norm-sq ───────────────────────────────────────────────────

class TestJordanTraceNormSq:

    def test_trace_diagonal(self):
        a = FTS([1.0, 2.0, 3.0] + [0.0] * 24)
        assert abs(a.jordan_trace() - 6.0) < 1e-12

    @pytest.mark.parametrize("v", [-2.0, 0.5, 1.0, 4.0])
    def test_trace_from_scalar(self, v):
        s = FTS.from_scalar(v)
        assert abs(s.jordan_trace() - 3.0 * v) < 1e-12

    def test_norm_sq_zero_is_zero(self):
        assert FTS([0.0] * 27).jordan_norm_sq() == 0.0

    def test_norm_sq_positive(self):
        a = FTS([1.0, 2.0, 3.0] + [0.0] * 24)
        assert a.jordan_norm_sq() > 0.0

    def test_norm_sq_offdiag_doubled(self):
        # x_off contributes 2*|x|² to norm_sq
        a = FTS([0.0, 0.0, 0.0] + [1.0, 0, 0, 0, 0, 0, 0, 0] + [0.0] * 16)
        # |x1|² = 1, x2 = x3 = 0, c² = 0 → norm_sq = 2
        assert abs(a.jordan_norm_sq() - 2.0) < 1e-9


# ── Jordan square ────────────────────────────────────────────────────────────

class TestJordanSquare:

    def test_returns_fts(self):
        a = FTS([1.0, 2.0, 3.0] + [0.0] * 24)
        sq = a.jordan_square()
        assert isinstance(sq, FTS)

    def test_diagonal_square(self):
        a = FTS([2.0, 3.0, 5.0] + [0.0] * 24)
        sq = a.jordan_square()
        # (A²)₁₁ = c₁² + |x₂|² + |x₃|² = 4 + 0 + 0 = 4
        assert abs(sq._c[0] - 4.0) < 1e-12
        assert abs(sq._c[1] - 9.0) < 1e-12
        assert abs(sq._c[2] - 25.0) < 1e-12

    def test_zero_squared_is_zero(self):
        z = FTS([0.0] * 27)
        sq = z.jordan_square()
        assert sq.cubic_norm() == 0.0


# ── Roundtrip / determinism ──────────────────────────────────────────────────

class TestDeterminism:

    @pytest.mark.parametrize("seed", range(15))
    def test_norm_repeatable(self, seed):
        elems = [(seed + i) * 0.123 for i in range(27)]
        a = FTS(elems)
        b = FTS(elems)
        assert a.cubic_norm() == b.cubic_norm()

    @pytest.mark.parametrize("seed", range(15))
    def test_quartic_repeatable(self, seed):
        elems = [(seed + i) * 0.07 for i in range(27)]
        a = FTS(elems)
        b = FTS(elems)
        assert a.quartic() == b.quartic()
