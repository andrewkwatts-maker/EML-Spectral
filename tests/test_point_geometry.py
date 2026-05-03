"""Tests for Sprint 1 geometric extensions on EMLPoint and EMLPair."""
from __future__ import annotations

import math
import pytest
from eml_math import EMLPoint
from eml_spectral import spacetime as _st
from eml_spectral import EMLPair


class TestPair:
    def test_returns_emlpair(self):
        p = EMLPoint(0.0, 1.0)
        result = _st.pair(p)
        assert isinstance(result, EMLPair)

    def test_unit_point_pair_values(self):
        # EMLPoint(0, 1): exp(0)=1, ln(1)=0
        p = EMLPoint(0.0, 1.0)
        pair = _st.pair(p)
        assert abs(pair.real_tension - 1.0) < 1e-12
        assert abs(pair.imag_tension - 0.0) < 1e-12

    def test_euler_point_pair_values(self):
        # EMLPoint(1, math.e): exp(1)=e, ln(e)=1
        p = EMLPoint(1.0, math.e)
        pair = _st.pair(p)
        assert abs(pair.real_tension - math.e) < 1e-10
        assert abs(pair.imag_tension - 1.0) < 1e-10

    def test_axiom8_safety_negative_y(self):
        # y=-1: frame shift uses |y|=1, ln(1)=0
        p = EMLPoint(0.0, -1.0)
        pair = _st.pair(p)
        assert pair.real_tension > 0
        assert math.isfinite(pair.imag_tension)

    def test_axiom8_safety_zero_y(self):
        p = EMLPoint(0.0, 0.0)
        pair = _st.pair(p)
        assert math.isfinite(pair.real_tension)
        assert math.isfinite(pair.imag_tension)


class TestEuclideanDelta:
    def test_unit_point_delta(self):
        # pair=(1,0) → Δ=1
        assert abs(_st.euclidean_delta(EMLPoint(0.0, 1.0)) - 1.0) < 1e-12

    def test_euler_point_delta(self):
        # pair=(e,1) → Δ=√(e²+1)
        p = EMLPoint(1.0, math.e)
        expected = math.sqrt(math.e ** 2 + 1.0)
        assert abs(_st.euclidean_delta(p) - expected) < 1e-10

    def test_symmetric_pair_delta(self):
        # EMLPoint(0, math.e): exp(0)=1, ln(e)=1 → Δ=√2
        p = EMLPoint(0.0, math.e)
        assert abs(_st.euclidean_delta(p) - math.sqrt(2)) < 1e-12

    def test_positive_always(self):
        for x, y in [(1.0, 2.0), (2.0, 1.0), (0.5, 3.0), (-1.0, 0.5)]:
            assert _st.euclidean_delta(EMLPoint(x, y)) >= 0.0


class TestMinkowskiDelta:
    def test_timelike_point(self):
        # EMLPoint(0,1): exp(0)=1, ln(1)=0 → Δ_M = √(1-0) = 1
        p = EMLPoint(0.0, 1.0)
        assert abs(_st.minkowski_delta(p) - 1.0) < 1e-12

    def test_lightlike_point(self):
        # EMLPoint(0, math.e): exp(0)=1, ln(e)=1 → Δ_M = √(1-1) = 0
        p = EMLPoint(0.0, math.e)
        assert _st.minkowski_delta(p) < 1e-9

    def test_spacelike_point(self):
        # EMLPoint(0, math.exp(2)): exp(0)=1, ln(e²)=2 → Δ_M=√(4-1)=√3
        p = EMLPoint(0.0, math.exp(2.0))
        assert abs(_st.minkowski_delta(p) - math.sqrt(3.0)) < 1e-10

    def test_signature_minus_plus(self):
        # Same point, different signature — result is the same (abs of ds²)
        p = EMLPoint(0.0, math.exp(2.0))
        plus = _st.minkowski_delta(p, signature="+---")
        minus = _st.minkowski_delta(p, signature="-+++")
        assert abs(plus - minus) < 1e-12

    def test_c_scaling(self):
        # With c=2: space component scaled by 2
        p = EMLPoint(1.0, math.e)  # exp(1)=e, ln(e)=1
        dm = _st.minkowski_delta(p, c=2.0)
        # ds² = e² - (2*1)² = e²-4
        expected = math.sqrt(abs(math.e ** 2 - 4.0))
        assert abs(dm - expected) < 1e-10


class TestCausalClassification:
    def test_timelike(self):
        p = EMLPoint(0.0, 1.0)  # exp(0)=1 > ln(1)=0
        assert _st.is_timelike(p)
        assert not _st.is_spacelike(p)
        assert not _st.is_lightlike(p)
        assert _st.light_cone_type(p) == "timelike"

    def test_lightlike(self):
        p = EMLPoint(0.0, math.e)  # exp(0)=1, ln(e)=1
        assert _st.is_lightlike(p)
        assert not _st.is_timelike(p)
        assert _st.light_cone_type(p) == "lightlike"

    def test_spacelike(self):
        p = EMLPoint(0.0, math.exp(2.0))  # exp(0)=1, ln(e²)=2
        assert _st.is_spacelike(p)
        assert not _st.is_timelike(p)
        assert _st.light_cone_type(p) == "spacelike"

    def test_future_light_cone_timelike(self):
        # exp(x) is always positive, so any timelike point is in future light cone
        p = EMLPoint(0.0, 1.0)
        assert _st.future_light_cone(p)

    def test_light_cone_coordinates(self):
        p = EMLPoint(0.0, 1.0)  # t=1, x=0
        u, v = _st.light_cone_coordinates(p)
        # u = t + x = 1+0 = 1, v = t - x = 1-0 = 1
        assert abs(u - 1.0) < 1e-12
        assert abs(v - 1.0) < 1e-12


class TestBoost:
    def test_zero_rapidity_is_identity(self):
        p = EMLPoint(0.0, 1.0)
        boosted = _st.boost(p, 0.0)
        assert abs(boosted.x - p.x) < 1e-10
        assert abs(boosted.y - p.y) < 1e-10

    def test_boost_preserves_minkowski_delta(self):
        p = EMLPoint(0.0, 1.0)
        original_dm = _st.minkowski_delta(p)
        for phi in [0.1, 0.5, 1.0, -0.3, -0.8]:
            boosted = _st.boost(p, phi)
            assert abs(_st.minkowski_delta(boosted) - original_dm) < 1e-8, \
                f"Δ_M changed after boost(phi={phi})"

    def test_boost_roundtrip(self):
        p = EMLPoint(0.0, 1.0)
        p2 = _st.boost(_st.boost(p, 0.5), -0.5)
        assert abs(p2.x - p.x) < 1e-8
        assert abs(p2.y - p.y) < 1e-8

    def test_boost_velocity_subluminal(self):
        p = EMLPoint(0.0, 1.0)
        boosted = _st.boost_velocity(p, 0.5)
        assert abs(_st.minkowski_delta(boosted) - _st.minkowski_delta(p)) < 1e-8

    def test_boost_velocity_superluminal_raises(self):
        p = EMLPoint(0.0, 1.0)
        with pytest.raises(ValueError):
            _st.boost_velocity(p, 1.0)
        with pytest.raises(ValueError):
            _st.boost_velocity(p, 1.5)

    def test_rapidity_zero_for_rest(self):
        # EMLPoint(0,1): pair=(1,0), rapidity=atanh(0/1)=0
        p = EMLPoint(0.0, 1.0)
        assert abs(_st.rapidity(p) - 0.0) < 1e-12

    def test_rapidity_raises_for_spacelike(self):
        p = EMLPoint(0.0, math.exp(2.0))  # spacelike
        with pytest.raises(ValueError):
            _st.rapidity(p)

    def test_rest_energy_equals_minkowski_delta(self):
        p = EMLPoint(0.0, 1.0)
        assert abs(_st.rest_energy(p) - _st.minkowski_delta(p)) < 1e-12

    def test_proper_time(self):
        p = EMLPoint(0.0, 1.0)
        assert abs(_st.proper_time(p) - _st.minkowski_delta(p)) < 1e-12  # c=1


class TestCanonicalFrame:
    def test_frame0_matches_pair(self):
        p = EMLPoint(1.0, math.e)
        f0 = _st.canonical_frame(p, 0)
        raw = _st.pair(p)
        assert abs(f0.real_tension - raw.real_tension) < 1e-12
        assert abs(f0.imag_tension - raw.imag_tension) < 1e-12

    def test_four_frames_same_euclidean_delta(self):
        p = EMLPoint(1.0, 2.0)
        ref_delta = _st.euclidean_delta(p)
        for k in range(4):
            frame = _st.canonical_frame(p, k)
            frame_delta = math.sqrt(
                frame.real_tension ** 2 + frame.imag_tension ** 2
            )
            assert abs(frame_delta - ref_delta) < 1e-10, \
                f"Frame {k} has different delta: {frame_delta} vs {ref_delta}"

    def test_frame_cycle_mod4(self):
        p = EMLPoint(1.0, 2.0)
        assert _st.canonical_frame(p, 0) == _st.canonical_frame(p, 4)
        assert _st.canonical_frame(p, 1) == _st.canonical_frame(p, 5)

    def test_frame1_is_quarter_rotation(self):
        # Frame 1 multiplies by i: (r, im) → (-im, r)
        p = EMLPoint(1.0, math.e)  # pair=(e, 1)
        f1 = _st.canonical_frame(p, 1)
        pair = _st.pair(p)
        assert abs(f1.real_tension - (-pair.imag_tension)) < 1e-10
        assert abs(f1.imag_tension - pair.real_tension) < 1e-10


class TestEMLPairFrames:
    def test_returns_four_frames(self):
        p = _st.pair(EMLPoint(1.0, 2.0))
        frames = p.frames()
        assert len(frames) == 4
        assert all(isinstance(f, EMLPair) for f in frames)

    def test_all_frames_same_modulus(self):
        p = _st.pair(EMLPoint(1.0, 2.0))
        ref_mod = p.modulus
        for i, f in enumerate(p.frames()):
            assert abs(f.modulus - ref_mod) < 1e-10, \
                f"Frame {i} modulus {f.modulus} != {ref_mod}"

    def test_frame0_is_identity(self):
        p = _st.pair(EMLPoint(1.0, 2.0))
        f0 = p.frames()[0]
        assert abs(f0.real_tension - p.real_tension) < 1e-12
        assert abs(f0.imag_tension - p.imag_tension) < 1e-12


# ── new expanded tests ────────────────────────────────────────────────────────

class TestMinkowskiBoostInvariance:
    @pytest.mark.parametrize("phi", [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    def test_minkowski_delta_conserved(self, phi):
        p = EMLPoint(1.0, math.e)
        dm0 = _st.minkowski_delta(p)
        boosted = _st.boost(p, phi)
        assert abs(_st.minkowski_delta(boosted) - dm0) < 1e-8, \
            f"Δ_M changed after boost(phi={phi}): {_st.minkowski_delta(boosted)} vs {dm0}"

    @pytest.mark.parametrize("phi", [-0.5, -0.25, 0.0, 0.25, 0.5])
    def test_spacelike_point_invariant(self, phi):
        # EMLPoint(0, exp(2)): spacelike — use small rapidities to stay within clamping guard
        p = EMLPoint(0.0, math.exp(2.0))
        dm0 = _st.minkowski_delta(p)
        boosted = _st.boost(p, phi)
        assert abs(_st.minkowski_delta(boosted) - dm0) < 1e-8


class TestMinkowskiDeltaScale:
    @pytest.mark.parametrize("x", [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("y", [0.1, 1.0, math.e, 10.0, 100.0])
    def test_nonnegative_always(self, x, y):
        p = EMLPoint(x, y)
        dm = _st.minkowski_delta(p)
        assert dm >= 0.0

    @pytest.mark.parametrize("x", [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("y", [0.1, 1.0, math.e, 10.0, 100.0])
    def test_finite_always(self, x, y):
        p = EMLPoint(x, y)
        assert math.isfinite(_st.minkowski_delta(p))


class TestLightConeEdgeCases:
    def test_exact_lightlike_unit(self):
        # EMLPoint(0, e): exp(0)=1, ln(e)=1, ds²=0
        p = EMLPoint(0.0, math.e)
        assert _st.is_lightlike(p)
        assert _st.light_cone_type(p) == "lightlike"

    def test_exact_lightlike_nontrivial(self):
        # exp(x) = ln(y) when y = exp(exp(x))
        x = 1.5
        y = math.exp(math.exp(x))
        p = EMLPoint(x, y)
        assert _st.is_lightlike(p, tol=1e-6)

    def test_near_lightlike_tol_tight(self):
        # With very tight tol, the near-lightlike point is NOT lightlike
        p = EMLPoint(0.0, math.e * 1.001)
        assert not _st.is_lightlike(p, tol=1e-12)

    def test_near_lightlike_tol_loose(self):
        # With loose tol it may classify as lightlike
        p = EMLPoint(0.0, math.e * (1.0 + 1e-7))
        assert _st.is_lightlike(p, tol=1.0)

    def test_light_cone_type_timelike_string(self):
        p = EMLPoint(0.0, 1.0)
        assert _st.light_cone_type(p) == "timelike"

    def test_light_cone_type_spacelike_string(self):
        p = EMLPoint(0.0, math.exp(2.0))
        assert _st.light_cone_type(p) == "spacelike"

    def test_lightlike_minkowski_delta_near_zero(self):
        p = EMLPoint(0.0, math.e)
        assert _st.minkowski_delta(p) < 1e-9


class TestBoostLargeRapidity:
    def test_large_positive_rapidity_finite(self):
        p = EMLPoint(1.0, math.e)
        boosted = _st.boost(p, 5.0)
        assert math.isfinite(boosted.x)
        assert math.isfinite(boosted.y)

    def test_large_negative_rapidity_finite(self):
        p = EMLPoint(1.0, math.e)
        boosted = _st.boost(p, -5.0)
        assert math.isfinite(boosted.x)
        assert math.isfinite(boosted.y)

    def test_large_positive_rapidity_invariant(self):
        p = EMLPoint(1.0, math.e)
        dm0 = _st.minkowski_delta(p)
        boosted = _st.boost(p, 5.0)
        assert abs(_st.minkowski_delta(boosted) - dm0) < 1e-6

    def test_large_negative_rapidity_invariant(self):
        p = EMLPoint(1.0, math.e)
        dm0 = _st.minkowski_delta(p)
        boosted = _st.boost(p, -5.0)
        assert abs(_st.minkowski_delta(boosted) - dm0) < 1e-6
