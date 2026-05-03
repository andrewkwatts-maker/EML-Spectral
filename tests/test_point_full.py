"""Tests for TensionPoint — the universal EML computation node."""
import math
import pytest
from eml_math import EMLPoint
from eml_spectral import spacetime as _st
from eml_spectral.spacetime import (
    minkowski_delta as _md, euclidean_delta as _ed, is_timelike as _itl,
    is_spacelike as _isl, is_lightlike as _ill, rapidity as _rap,
    boost as _boost, boost_velocity as _bv, canonical_frame as _cf,
    light_cone_coordinates as _lcc, light_cone_type as _lct,
    future_light_cone as _flc, rest_energy as _re, proper_time as _pt,
)
from eml_math.constants import OVERFLOW_THRESHOLD


class TestEMLPrimitive:
    """EMLPoint(x, y).tension() == eml(x, y) = exp(x) - ln(y)."""

    def test_unit_point_gives_e(self):
        assert EMLPoint(1.0, 1.0).tension() == pytest.approx(math.e, rel=1e-14)

    def test_exp_x_is_eml_x_1(self):
        for x in [0.0, 0.5, 1.0, 2.0, 3.0]:
            assert EMLPoint(x, 1.0).tension() == pytest.approx(math.exp(x), rel=1e-14)

    def test_formula_matches_manual(self):
        x, y = 2.0, 3.0
        expected = math.exp(x) - math.log(y)
        assert EMLPoint(x, y).tension() == pytest.approx(expected, rel=1e-14)

    def test_tension_is_always_real(self):
        for x in [0.1, 1.0, 5.0]:
            for y in [0.1, 1.0, 5.0]:
                T = EMLPoint(x, y).tension()
                assert math.isfinite(T)

    def test_frame_shift_guard_when_y_negative(self):
        # y < 0 would make ln(y) undefined; frame guard uses |y|
        p = EMLPoint(1.0, -2.0)
        T = p.tension()
        expected = math.exp(1.0) - math.log(2.0)
        assert T == pytest.approx(expected, rel=1e-12)

    def test_frame_shift_guard_when_y_zero(self):
        p = EMLPoint(1.0, 0.0)
        T = p.tension()
        assert math.isfinite(T)


class TestNestedEML:
    """TensionPoint accepts other TensionPoints as coordinates."""

    def test_ln_nested_knot(self):
        # ln(e) = 1 via depth-3 EML nesting
        e = math.e
        result = EMLPoint(1.0, EMLPoint(EMLPoint(1.0, e), 1.0)).tension()
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_ln_two(self):
        inner1 = EMLPoint(1.0, 2.0)
        inner2 = EMLPoint(inner1, 1.0)
        result = EMLPoint(1.0, inner2).tension()
        assert result == pytest.approx(math.log(2.0), rel=1e-10)

    def test_double_nesting(self):
        # exp(exp(1)) = e^e via nesting
        inner = EMLPoint(1.0, 1.0)          # tension = e
        outer = EMLPoint(inner, 1.0)         # tension = exp(e) - ln(1) = exp(e)
        assert outer.tension() == pytest.approx(math.exp(math.e), rel=1e-10)

    def test_x_coord_evaluates_nested(self):
        nested = EMLPoint(2.0, 1.0)          # tension = exp(2)
        p = EMLPoint(nested, 1.0)            # x = nested.tension() = exp(2)
        assert p.x == pytest.approx(math.exp(2.0), rel=1e-14)


class TestMirrorPulse:
    """mirror_pulse() — continuous mode."""

    def test_standard_update(self, unit_point):
        # Continuous: x_new = y, y_new = T
        y_old = unit_point.y
        T = unit_point.tension()
        nxt = unit_point.mirror_pulse()
        assert nxt.x == pytest.approx(y_old, rel=1e-12)
        assert nxt.y == pytest.approx(T, rel=1e-12)

    def test_frame_shift_on_negative_y(self):
        # When T < 0 (which happens at large y), next pulse uses |y_new|
        p = EMLPoint(0.1, 10.0)   # T = exp(0.1) - ln(10) ≈ 1.105 - 2.303 = -1.198
        nxt = p.mirror_pulse()
        assert math.isfinite(nxt.tension())

    def test_overflow_dampening(self):
        # x near OVERFLOW_THRESHOLD gets ln-dampened
        p = EMLPoint(OVERFLOW_THRESHOLD + 1.0, 1.0)
        nxt = p.mirror_pulse()
        assert math.isfinite(nxt.tension())

    def test_returns_new_object(self, unit_point):
        nxt = unit_point.mirror_pulse()
        assert nxt is not unit_point


class TestDiscreteMode:
    """Discrete mode (D set) quantizes via round(T * D)."""

    def test_discrete_quantization(self):
        p = EMLPoint(1.0, 1.0, D=100)
        nxt = p.mirror_pulse()
        # y_new should be round(T * 100) / 100
        T = p.tension()
        expected_y = round(T * 100) / 100
        assert nxt.y == pytest.approx(expected_y, rel=1e-12)

    def test_d_propagates_to_next(self):
        p = EMLPoint(1.0, 1.0, D=100)
        nxt = p.mirror_pulse()
        assert nxt.D == 100


class TestAxiom10Conservation:
    """Axiom 10: T + x = exp(x) at every step."""

    def test_conservation_at_unit_point(self, unit_point):
        nxt = unit_point.mirror_pulse()
        # mirror update: y_new = old tension
        assert abs(unit_point.tension() - nxt.y) < 1e-10

    def test_conservation_over_multiple_steps(self, unit_knot):
        from eml_spectral.simulation import simulate_pulses, verify_conservation
        traj = simulate_pulses(unit_knot, n_pulses=20)
        assert verify_conservation(traj)


class TestTreeIntrospection:
    """is_leaf, left(), right() for converter traversal."""

    def test_flat_point_is_leaf(self, unit_point):
        assert unit_point.is_leaf()

    def test_nested_point_not_leaf(self):
        p = EMLPoint(EMLPoint(1.0, 1.0), 1.0)
        assert not p.is_leaf()

    def test_left_right_access(self):
        inner = EMLPoint(2.0, 3.0)
        outer = EMLPoint(inner, 5.0)
        assert outer.left() is inner
        assert outer.right() == 5.0


class TestResonance:
    """Axiom 14: resonance as MPM equality."""

    def test_same_point_resonates(self, unit_point):
        other = EMLPoint(1.0, 1.0)
        assert (abs(unit_point.tension() - other.tension()) < 1e-10)

    def test_different_point_does_not_resonate(self, unit_point):
        other = EMLPoint(2.0, 1.0)
        assert not (abs(unit_point.tension() - other.tension()) < 1e-10)


# ── New tests ────────────────────────────────────────────────────────────────


class TestDiscreteModeTension:
    """EMLPoint with D=100: tension is quantized; pulse advances correctly."""

    def test_discrete_tension_is_finite(self):
        p = EMLPoint(1.0, 1.0, D=100)
        assert math.isfinite(p.tension())

    def test_discrete_pulse_y_quantized(self):
        p = EMLPoint(1.0, 1.0, D=100)
        nxt = p.mirror_pulse()
        # y_new must be a multiple of 1/D
        assert abs(nxt.y * 100 - round(nxt.y * 100)) < 1e-9

    def test_discrete_d_propagates(self):
        p = EMLPoint(1.0, 1.0, D=100)
        nxt = p.mirror_pulse()
        assert nxt.D == 100


class TestNestingDepth:
    """Triple-nested EMLPoint gives finite result."""

    def test_triple_nested_is_finite(self):
        inner = EMLPoint(1.0, 1.0)
        mid = EMLPoint(inner, 1.0)
        outer = EMLPoint(mid, 1.0)
        assert math.isfinite(outer.tension())


class TestMirrorPulseInverse:
    """Two mirror_pulses preserves structure for a specific point."""

    def test_two_pulses_returns_emlpoint(self):
        p = EMLPoint(1.0, 2.0)
        p2 = p.mirror_pulse().mirror_pulse()
        assert isinstance(p2, EMLPoint)

    def test_two_pulses_finite(self):
        p = EMLPoint(1.0, 2.0)
        p2 = p.mirror_pulse().mirror_pulse()
        assert math.isfinite(p2.tension())


# ── New point tests (+20) ─────────────────────────────────────────────────────

class TestBoostRoundtrip:
    """boost(phi) then boost(-phi) recovers original Minkowski delta within 1e-9."""

    @pytest.mark.parametrize("phi", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_boost_roundtrip_minkowski_delta(self, phi):
        p = EMLPoint(2.0, 3.0)
        dm_before = _md(p)
        p2 = _boost(_boost(p, phi), -phi)
        dm_after = _md(p2)
        assert abs(dm_after - dm_before) < 1e-9

    @pytest.mark.parametrize("phi", [-1.0, 1.0, 2.0])
    def test_boost_roundtrip_returns_emlpoint(self, phi):
        p = EMLPoint(1.0, math.e)
        result = _boost(_boost(p, phi), -phi)
        assert isinstance(result, EMLPoint)


class TestMinkowskiDeltaLargeRapidity:
    """Minkowski delta preserved for phi=+-5; boost returns finite values for +-10."""

    @pytest.mark.parametrize("phi", [5.0, -5.0])
    def test_delta_m_preserved_large_rapidity(self, phi):
        p = EMLPoint(2.0, 3.0)
        dm0 = _md(p)
        pb = _boost(p, phi)
        dm1 = _md(pb)
        assert abs(dm1 - dm0) < 1e-6

    @pytest.mark.parametrize("phi", [10.0, -10.0])
    def test_boost_finite_very_large_rapidity(self, phi):
        p = EMLPoint(2.0, 3.0)
        pb = _boost(p, phi)
        assert math.isfinite(_md(pb))


class TestCausalConsistency:
    """Exactly one of {is_timelike, is_spacelike, is_lightlike} is True."""

    @pytest.mark.parametrize("x,y", [
        (3.0, 1.0),      # timelike
        (0.0, 100.0),    # spacelike
        (0.0, math.e),   # lightlike
        (2.0, 2.0),      # check
        (1.0, 1.0),      # check
    ])
    def test_exactly_one_causal_type(self, x, y):
        p = EMLPoint(x, y)
        count = sum([_itl(p), _isl(p), _ill(p)])
        assert count == 1


class TestRapidityCausalLink:
    """rapidity defined for timelike; raises ValueError for spacelike."""

    def test_rapidity_defined_for_timelike(self):
        p = EMLPoint(3.0, 1.0)
        assert _itl(p)
        r = _rap(p)
        assert math.isfinite(r)

    def test_rapidity_raises_for_spacelike(self):
        p = EMLPoint(0.0, 1000.0)
        assert _isl(p)
        with pytest.raises(ValueError):
            _rap(p)

    def test_rapidity_finite_for_several_timelike(self):
        for x, y in [(2.0, 1.0), (3.0, 2.0), (5.0, 1.0)]:
            p = EMLPoint(x, y)
            assert math.isfinite(_rap(p))
