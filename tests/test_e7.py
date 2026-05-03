"""Tests for E7_56 — 56D representation of E₇ on (J₃(𝕆), J₃(𝕆)*) pairs."""
from __future__ import annotations
import math
import pytest

from eml_spectral.exceptional import FreudenthalTripleSystem as FTS, E7_56


def _scalar_pair(a: float, b: float) -> E7_56:
    return E7_56(FTS.from_scalar(a), FTS.from_scalar(b))


# ── Construction ─────────────────────────────────────────────────────────────

class TestConstruction:

    def test_from_two_fts(self):
        v = _scalar_pair(1.0, 2.0)
        assert isinstance(v, E7_56)

    def test_x_y_components(self):
        v = _scalar_pair(3.0, 5.0)
        assert v.x.cubic_norm() == 27.0   # 3³
        assert v.y.cubic_norm() == 125.0  # 5³

    def test_alpha_leak_constant(self):
        # 1/√6 portal coupling
        assert abs(E7_56.ALPHA_LEAK - 1.0 / math.sqrt(6.0)) < 1e-12

    def test_zero_pair(self):
        z = FTS([0.0] * 27)
        v = E7_56(z, z)
        assert v.symplectic_form() == 0.0
        assert v.quartic_invariant() == 0.0


# ── Symplectic form (delegates to bilinear_form per source) ─────────────────

class TestSymplecticForm:

    @pytest.mark.parametrize("a,b", [(1.0, 2.0), (3.0, 4.0), (-1.0, 5.0)])
    def test_scalar_pair(self, a, b):
        v = _scalar_pair(a, b)
        # x.bilinear_form(y) for from_scalar(a), from_scalar(b) = 3*a*b
        assert abs(v.symplectic_form() - 3.0 * a * b) < 1e-9

    def test_finite_for_pneuma(self):
        a = FTS.from_pneuma_condensate(24.0)
        b = FTS.from_pneuma_condensate(24.0)
        assert math.isfinite(E7_56(a, b).symplectic_form())


# ── Quartic invariant ────────────────────────────────────────────────────────

class TestQuarticInvariant:

    def test_finite(self):
        v = _scalar_pair(1.0, 1.0)
        assert math.isfinite(v.quartic_invariant())

    def test_zero_pair_zero(self):
        z = FTS([0.0] * 27)
        assert E7_56(z, z).quartic_invariant() == 0.0

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 3.0])
    def test_quartic_repeatable(self, a):
        v1 = _scalar_pair(a, a)
        v2 = _scalar_pair(a, a)
        assert v1.quartic_invariant() == v2.quartic_invariant()

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
    def test_self_pair_quartic(self, a):
        # for self-pair (A, A): inner = ⟨A,A⟩ = 3a²; nx = ny = a³;
        # core = (3a²)² − 4a⁶ = 9a⁴ − 4a⁶
        # delta = (3a)² · (3a²) / 16 + (3a)² · (3a²) / 16 = 2 · 27a⁴ / 16 = 27a⁴/8
        v = _scalar_pair(a, a)
        expected = 9.0 * a ** 4 - 4.0 * a ** 6 + 27.0 * a ** 4 / 8.0
        assert abs(v.quartic_invariant() - expected) < 1e-8


# ── e7_action — infinitesimal generator ──────────────────────────────────────

class TestE7Action:

    def test_returns_e7_56(self):
        v = _scalar_pair(1.0, 1.0)
        gen = [0.0] * 56
        out = v.e7_action(gen)
        assert isinstance(out, E7_56)

    def test_zero_generator_near_identity(self):
        v = _scalar_pair(1.0, 2.0)
        out = v.e7_action([0.0] * 56)
        assert abs(out.x.cubic_norm() - v.x.cubic_norm()) < 1e-9

    def test_wrong_length_raises(self):
        v = _scalar_pair(1.0, 1.0)
        with pytest.raises(ValueError):
            v.e7_action([0.0] * 55)

    def test_overlong_raises(self):
        v = _scalar_pair(1.0, 1.0)
        with pytest.raises(ValueError):
            v.e7_action([0.0] * 60)


# ── E₇ ⊃ E₆ × U(1) branching ─────────────────────────────────────────────────

class TestE6U1Split:

    def test_dict_keys(self):
        v = _scalar_pair(1.0, 2.0)
        d = v.split_e6_u1()
        for key in ("visible_e6", "hidden_27_dual", "symplectic_pairing",
                    "u1_charge_27", "portal_coupling"):
            assert key in d

    def test_portal_coupling_value(self):
        v = _scalar_pair(1.0, 2.0)
        d = v.split_e6_u1()
        assert abs(d["portal_coupling"] - 1.0 / math.sqrt(6.0)) < 1e-12

    def test_visible_is_x(self):
        v = _scalar_pair(3.0, 7.0)
        d = v.split_e6_u1()
        assert d["visible_e6"].cubic_norm() == 27.0

    def test_hidden_is_y(self):
        v = _scalar_pair(3.0, 7.0)
        d = v.split_e6_u1()
        assert d["hidden_27_dual"].cubic_norm() == 343.0

    @pytest.mark.parametrize("a,b", [(1.0, 1.0), (2.0, 3.0), (0.5, 5.0)])
    def test_symplectic_matches(self, a, b):
        v = _scalar_pair(a, b)
        d = v.split_e6_u1()
        assert d["symplectic_pairing"] == v.symplectic_form()
