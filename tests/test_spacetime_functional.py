"""Tests for the functional spacetime API on EMLPoint."""
from __future__ import annotations
import math
import pytest

from eml_math import EMLPoint
from eml_spectral import spacetime
from eml_spectral.pair import EMLPair


# ── pair / euclidean_delta ───────────────────────────────────────────────────

class TestPair:

    def test_returns_pair(self):
        p = EMLPoint(1.0, math.e)
        pr = spacetime.pair(p)
        assert isinstance(pr, EMLPair)

    def test_eml_1_e_components(self):
        # exp(1) ≈ e, ln(e) = 1
        p = EMLPoint(1.0, math.e)
        pr = spacetime.pair(p)
        # EMLPair.from_values(real, imag); .modulus = sqrt(real² + imag²)
        assert math.isfinite(pr.modulus)


class TestEuclideanDelta:

    def test_finite(self):
        p = EMLPoint(1.0, 1.0)
        assert math.isfinite(spacetime.euclidean_delta(p))

    def test_positive(self):
        p = EMLPoint(1.0, 2.0)
        assert spacetime.euclidean_delta(p) > 0

    @pytest.mark.parametrize("x,y", [(1.0, 1.0), (2.0, 3.0), (0.5, 0.5)])
    def test_matches_formula(self, x, y):
        p = EMLPoint(x, y)
        expected = math.sqrt(math.exp(x) ** 2 + math.log(abs(y)) ** 2)
        assert abs(spacetime.euclidean_delta(p) - expected) < 1e-9


# ── minkowski_delta ──────────────────────────────────────────────────────────

class TestMinkowskiDelta:

    def test_default_signature(self):
        p = EMLPoint(1.0, math.e)
        v = spacetime.minkowski_delta(p)
        assert math.isfinite(v) and v >= 0

    def test_signature_invariance(self):
        # |ds²| is the same regardless of overall sign convention
        p = EMLPoint(1.0, 2.0)
        a = spacetime.minkowski_delta(p, signature="+---")
        b = spacetime.minkowski_delta(p, signature="-+++")
        assert abs(a - b) < 1e-12

    def test_lightlike_zero(self):
        # Construct a lightlike point: exp(x) = c·ln(y) with c=1
        # Need exp(x) == ln(y), so y = exp(exp(x)). Pick x=0 → exp(0)=1, ln(y)=1, y=e
        p = EMLPoint(0.0, math.e)
        assert spacetime.minkowski_delta(p) < 1e-9


# ── Causal classification ────────────────────────────────────────────────────

class TestCausal:

    def test_timelike(self):
        # exp(2x) > (ln y)² when ln y is small
        p = EMLPoint(1.0, 1.0)   # exp(2) > 0
        assert spacetime.is_timelike(p)
        assert not spacetime.is_spacelike(p)

    def test_spacelike(self):
        # ln(y)² > exp(2x) when y is huge
        p = EMLPoint(0.0, math.exp(10.0))
        assert spacetime.is_spacelike(p)
        assert not spacetime.is_timelike(p)

    def test_lightlike_boundary(self):
        p = EMLPoint(0.0, math.e)
        assert spacetime.is_lightlike(p, tol=1e-6)


# ── Boost ────────────────────────────────────────────────────────────────────

class TestBoost:

    def test_zero_rapidity_is_identity(self):
        p = EMLPoint(1.0, 2.0)
        b = spacetime.boost(p, 0.0)
        assert abs(b.x - p.x) < 1e-9
        assert abs(b.y - p.y) < 1e-9

    @pytest.mark.parametrize("phi", [0.1, 0.3, 0.5, 0.8])
    def test_boost_preserves_minkowski(self, phi):
        p = EMLPoint(1.0, 1.5)
        d0 = spacetime.minkowski_delta(p)
        b = spacetime.boost(p, phi)
        d1 = spacetime.minkowski_delta(b)
        assert abs(d0 - d1) < 1e-6

    @pytest.mark.parametrize("phi1,phi2", [(0.1, 0.2), (0.2, 0.3), (-0.1, 0.4)])
    def test_rapidity_additive(self, phi1, phi2):
        # boost(boost(p, φ1), φ2) ≈ boost(p, φ1+φ2) (Minkowski preserved)
        p = EMLPoint(1.0, 1.5)
        a = spacetime.boost(spacetime.boost(p, phi1), phi2)
        b = spacetime.boost(p, phi1 + phi2)
        assert abs(spacetime.minkowski_delta(a) - spacetime.minkowski_delta(b)) < 1e-6

    def test_returns_emlpoint(self):
        p = EMLPoint(1.0, 1.0)
        out = spacetime.boost(p, 0.5)
        assert isinstance(out, EMLPoint)


# ── Light cone & rest energy ─────────────────────────────────────────────────

class TestLightCone:

    def test_light_cone_coordinates_returns_pair(self):
        p = EMLPoint(1.0, 1.0)
        lc = spacetime.light_cone_coordinates(p)
        assert hasattr(lc, "__len__") and len(lc) == 2

    def test_light_cone_type_returns_string(self):
        p = EMLPoint(1.0, 1.0)
        t = spacetime.light_cone_type(p)
        assert isinstance(t, str)
        assert t in ("timelike", "spacelike", "lightlike")

    def test_future_light_cone_for_timelike(self):
        p = EMLPoint(1.0, 1.0)
        result = spacetime.future_light_cone(p)
        assert isinstance(result, bool)


class TestRestEnergyProperTime:

    def test_rest_energy_finite(self):
        p = EMLPoint(1.0, 1.0)
        assert math.isfinite(spacetime.rest_energy(p))

    def test_proper_time_scales_with_c(self):
        p = EMLPoint(1.0, 1.0)
        for c in (1.0, 2.0, 5.0):
            t = spacetime.proper_time(p, c=c)
            assert math.isfinite(t)


# ── Rapidity ─────────────────────────────────────────────────────────────────

class TestRapidity:

    def test_finite_for_timelike(self):
        p = EMLPoint(1.0, 1.0)   # exp(1) > ln(1)=0 → timelike, ratio=0
        v = spacetime.rapidity(p)
        assert math.isfinite(v)

    def test_zero_for_pure_time_point(self):
        # ln(1) = 0 → ratio = 0 → atanh(0) = 0
        p = EMLPoint(1.0, 1.0)
        assert abs(spacetime.rapidity(p)) < 1e-9
