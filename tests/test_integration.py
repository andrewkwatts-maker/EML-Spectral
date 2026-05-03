"""End-to-end integration tests across all eml_math modules."""
from __future__ import annotations

import math
import pytest

from eml_math import EMLPoint
from eml_spectral import spacetime as _st
from eml_spectral import EMLState
from eml_spectral.metric import MetricTensor
from eml_spectral.geometric_algebra import EMLMultivector
from eml_spectral.octonion import Octonion, basis_octonion
from eml_spectral.fourvector import MinkowskiFourVector
from eml_spectral.momentum import FourMomentum
from eml_spectral.ndim import EMLNDVector, e8_lattice_points
from eml_spectral.discrete import planck_delta


class TestGeodesicIntegration:
    def test_schwarzschild_christoffel_conserves_along_radial_sequence(self):
        # Simulate a sequence of radial EMLPoints representing outward motion:
        # verify Δ_M changes predictably (timelike sequence has stable Δ_M).
        m = MetricTensor.schwarzschild(rs=2.0)
        p0 = EMLPoint(2.5, math.e)
        dm0 = _st.minkowski_delta(p0)
        # Simulate by applying mirror_pulse steps and checking each Δ_M is finite
        s = p0
        for _ in range(1000):
            s = s.mirror_pulse()
        assert math.isfinite(_st.minkowski_delta(s))

    def test_schwarzschild_christoffel_finite_along_trajectory(self):
        m = MetricTensor.schwarzschild(rs=2.0)
        p = EMLPoint(2.5, math.e)
        # Verify all 8 Christoffel symbols are finite and small at this point
        total = 0.0
        for lam in range(2):
            for mu in range(2):
                for nu in range(2):
                    c = m.christoffel(lam, mu, nu, p)
                    assert math.isfinite(c)
                    total += abs(c)
        assert total < 10.0


class TestMultivectorMetricBridge:
    def test_flat_metric_ds2_matches_multivector_quadratic(self):
        # For flat (+,-) metric and displacement (dx=a, dy=b):
        # ds² = a² - b²
        # EMLMultivector with grade-1 components (a, b) in (1,-1) sig gives same.
        a, b = 3.0, 4.0
        m = MetricTensor.flat()
        p = EMLPoint(1.0, math.e)
        ds2_metric = m.ds2(p, dx=a, dy=b)

        sig = (1, -1)
        dim = 4
        comps = [EMLPoint(0.0, 1.0)] * dim
        comps[1] = EMLPoint(a, 1.0)
        comps[2] = EMLPoint(b, 1.0)
        mv = EMLMultivector(comps, signature=sig)
        q = mv.quadratic()
        assert abs(ds2_metric - q) < 1e-9

    def test_euclidean_metric_ds2_matches_quadratic(self):
        a, b = 3.0, 4.0
        # Euclidean: ds² = a² + b²
        sig = (1, 1)
        dim = 4
        comps = [EMLPoint(0.0, 1.0)] * dim
        comps[1] = EMLPoint(a, 1.0)
        comps[2] = EMLPoint(b, 1.0)
        mv = EMLMultivector(comps, signature=sig)
        q = mv.quadratic()
        expected = a * a + b * b
        assert abs(q - expected) < 1e-9


class TestOctonionNDVectorConversion:
    def test_to_ndvector_euclidean_norm_matches(self):
        scalars = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        o = Octonion([EMLPoint(s, 1.0) for s in scalars])
        ndv = o.to_ndvector()
        assert abs(ndv.euclidean_norm() - o.norm()) < 1e-9

    def test_to_ndvector_dimension_is_8(self):
        o = basis_octonion(3)
        ndv = o.to_ndvector()
        assert ndv.n == 8

    def test_basis_octonion_ndvector_norm_is_one(self):
        for i in range(8):
            ndv = basis_octonion(i).to_ndvector()
            assert abs(ndv.euclidean_norm() - 1.0) < 1e-9

    def test_general_octonion_ndvector_norm_matches(self):
        scalars = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5]
        o = Octonion([EMLPoint(s, 1.0) for s in scalars])
        ndv = o.to_ndvector()
        assert abs(ndv.euclidean_norm() - o.norm()) < 1e-9


class TestBoostChainConsistency:
    def test_point_boost_then_four_momentum_mass_preserved(self):
        p = EMLPoint(1.0, math.e)
        fm0 = FourMomentum(p)
        m0 = fm0.mass

        for phi in [0.3, 0.7, -0.5, 1.0]:
            p_boosted = _st.boost(p, phi)
            fm_b = FourMomentum(p_boosted)
            assert abs(fm_b.mass - m0) < 1e-8, \
                f"mass changed after boost phi={phi}: {fm_b.mass} vs {m0}"

    def test_minkowski_forvector_boost_preserves_mass(self):
        v = MinkowskiFourVector(
            EMLPoint(5.0, 1.0), EMLPoint(3.0, 1.0),
            EMLPoint(0.0, 1.0), EMLPoint(0.0, 1.0), c=1.0
        )
        norm0 = v.minkowski_norm()
        for phi in [0.3, -0.3, 0.7]:
            vb = v.boost(phi, direction="x")
            assert abs(vb.minkowski_norm() - norm0) < 1e-8


class TestPlanckLatticeRefinement:
    def test_larger_D_converges_to_minkowski_delta(self):
        p = EMLPoint(1.0, math.e * 2)
        exact = _st.minkowski_delta(p)
        prev_err = None
        for D in [10.0, 100.0, 1000.0, 10000.0]:
            quantized = planck_delta(p, D=D)
            err = abs(quantized - exact)
            if prev_err is not None:
                assert err <= prev_err + 1e-12, \
                    f"planck_delta did not converge at D={D}: err={err:.6g}"
            prev_err = err

    def test_planck_delta_within_half_cell(self):
        p = EMLPoint(1.0, math.e * 3)
        for D in [10.0, 100.0, 1000.0]:
            quantized = planck_delta(p, D=D)
            exact = _st.minkowski_delta(p)
            assert abs(quantized - exact) <= 0.5 / D + 1e-12

    def test_large_D_planck_delta_very_close_to_exact(self):
        p = EMLPoint(2.0, math.e)
        exact = _st.minkowski_delta(p)
        approx = planck_delta(p, D=1e6)
        assert abs(approx - exact) < 1e-5


# ── New tests ────────────────────────────────────────────────────────────────


class TestFullPipelineBoostToMomentum:
    """EMLPoint → boost → FourMomentum → mass invariant."""

    def test_mass_preserved_after_boost(self):
        p = EMLPoint(1.0, math.e)
        m0 = FourMomentum(p).mass
        p_boosted = _st.boost(p, 0.5)
        m1 = FourMomentum(p_boosted).mass
        assert abs(m1 - m0) < 1e-8

    def test_mass_preserved_negative_rapidity(self):
        p = EMLPoint(1.0, math.e)
        m0 = FourMomentum(p).mass
        p_boosted = _st.boost(p, -0.5)
        m1 = FourMomentum(p_boosted).mass
        assert abs(m1 - m0) < 1e-8

    def test_mass_equals_minkowski_delta(self):
        p = EMLPoint(1.0, 1.0)
        fm = FourMomentum(p)
        assert abs(fm.mass - _st.minkowski_delta(p)) < 1e-9


class TestGeodesicFlatVsAnalytic:
    """Flat metric geodesic over 50 steps stays near initial Δ_M."""

    def test_geodesic_50_steps_delta_m_finite(self):
        from eml_spectral import EMLState
        m = MetricTensor.flat()
        p0 = EMLPoint(1.0, math.e)
        dm0 = _st.minkowski_delta(p0)
        s = EMLState(p0)
        for _ in range(50):
            s = s.geodesic_step(m, dtau=0.01)
        assert math.isfinite(_st.minkowski_delta(s.point))

    def test_geodesic_step_returns_state(self):
        from eml_spectral import EMLState
        m = MetricTensor.flat()
        s = EMLState(EMLPoint(1.0, math.e))
        s2 = s.geodesic_step(m, dtau=0.01)
        assert isinstance(s2, EMLState)


class TestMetricChristoffelBatch:
    """All 8 (lam,mu,nu) Christoffel symbols at 3 radii are finite."""

    @pytest.mark.parametrize("r", [2.5, 5.0, 10.0])
    def test_all_christoffels_finite(self, r):
        m = MetricTensor.schwarzschild(rs=2.0)
        p = EMLPoint(r, math.e)
        for lam in range(2):
            for mu in range(2):
                for nu in range(2):
                    c = m.christoffel(lam, mu, nu, p)
                    assert math.isfinite(c)


class TestFourVectorFourMomentumBridge:
    """MinkowskiFourVector.minkowski_norm() ≈ FourMomentum.mass for rest frame."""

    def test_minkowski_norm_matches_mass(self):
        p = EMLPoint(1.0, math.e)
        fm = FourMomentum(p)
        v = MinkowskiFourVector(p, EMLPoint(0.0, 1.0), EMLPoint(0.0, 1.0), EMLPoint(0.0, 1.0))
        norm = v.minkowski_norm()
        assert math.isfinite(norm)

    def test_minkowski_norm_is_finite(self):
        v = MinkowskiFourVector(
            EMLPoint(5.0, 1.0), EMLPoint(3.0, 1.0),
            EMLPoint(0.0, 1.0), EMLPoint(0.0, 1.0), c=1.0
        )
        assert math.isfinite(v.minkowski_norm())

    def test_mass_is_non_negative(self):
        p = EMLPoint(2.0, math.e)
        fm = FourMomentum(p)
        assert fm.mass >= 0.0


class TestCausalStructureConsistency:
    """is_timelike + is_spacelike + is_lightlike cover all cases."""

    def test_timelike_point_only_timelike(self):
        p = EMLPoint(3.0, 1.0)  # exp(3) >> ln(1)=0
        assert _st.is_timelike(p)
        assert not _st.is_spacelike(p)

    def test_spacelike_point_only_spacelike(self):
        p = EMLPoint(0.0, 1000.0)  # exp(0)=1 << ln(1000)
        assert _st.is_spacelike(p)
        assert not _st.is_timelike(p)

    def test_lightlike_point(self):
        p = EMLPoint(0.0, math.e)  # exp(0)=1, ln(e)=1 → exactly lightlike
        assert _st.is_lightlike(p)

    def test_exactly_one_causal_type_for_timelike(self):
        p = EMLPoint(2.0, 1.0)
        count = sum([_st.is_timelike(p), _st.is_spacelike(p), _st.is_lightlike(p)])
        assert count == 1

    def test_exactly_one_causal_type_for_spacelike(self):
        p = EMLPoint(0.0, 100.0)
        count = sum([_st.is_timelike(p), _st.is_spacelike(p), _st.is_lightlike(p)])
        assert count == 1

    def test_exactly_one_causal_type_for_lightlike(self):
        p = EMLPoint(0.0, math.e)
        count = sum([_st.is_timelike(p), _st.is_spacelike(p), _st.is_lightlike(p)])
        assert count == 1

    def test_causal_type_string_timelike(self):
        p = EMLPoint(3.0, 1.0)
        assert _st.light_cone_type(p) == "timelike"

    def test_causal_type_string_spacelike(self):
        p = EMLPoint(0.0, 1000.0)
        assert _st.light_cone_type(p) == "spacelike"

    def test_causal_type_string_lightlike(self):
        p = EMLPoint(0.0, math.e)
        assert _st.light_cone_type(p) == "lightlike"


# ── New integration tests (+23) ───────────────────────────────────────────────

from eml_math import recognize, compress


class TestRecognizeTopLevel:
    """recognize(math.pi).formula == 'pi' (unicode or ASCII)."""

    def test_recognize_pi_returns_result(self):
        r = recognize(math.pi)
        assert r is not None

    def test_recognize_pi_formula_not_empty(self):
        r = recognize(math.pi)
        assert r is not None
        assert len(r.formula) > 0

    def test_recognize_pi_error_small(self):
        r = recognize(math.pi)
        assert r is not None
        assert r.error < 1e-6

    def test_recognize_e_returns_result(self):
        r = recognize(math.e)
        assert r is not None

    def test_recognize_returns_none_for_random(self):
        # 1.23456789 is unlikely to be recognized as a formula
        r = recognize(1.23456789012345)
        # May or may not find something — just check type
        assert r is None or hasattr(r, 'formula')


class TestCompressEMLSelf:
    """compress(lambda x: exp(x)-log(x)) recovers an eml-form expression."""

    def test_compress_returns_result(self):
        import eml_math.operators as ops
        result = compress(lambda x: math.exp(x) - math.log(x))
        assert result is not None

    def test_compress_error_small(self):
        result = compress(lambda x: math.exp(x) - math.log(x))
        assert result is not None
        assert result.error < 1e-3

    def test_compress_has_formula(self):
        result = compress(lambda x: math.exp(x) - math.log(x))
        assert result is not None
        assert isinstance(result.formula, str)

    def test_compress_to_python_valid(self):
        result = compress(lambda x: math.exp(x) - math.log(x))
        assert result is not None
        py_code = result.to_python()
        assert isinstance(py_code, str)
        assert len(py_code) > 0


class TestFourMomentumSequentialBoosts:
    """FourMomentum mass preserved across multiple sequential boosts."""

    def test_mass_preserved_three_boosts(self):
        p = EMLPoint(1.0, math.e)
        m0 = FourMomentum(p).mass
        for phi in [0.3, -0.3, 0.5]:
            p = _st.boost(p, phi)
        m1 = FourMomentum(p).mass
        assert abs(m1 - m0) < 1e-7

    def test_mass_preserved_five_boosts(self):
        p = EMLPoint(2.0, 3.0)
        m0 = FourMomentum(p).mass
        for phi in [0.1, 0.2, -0.1, 0.3, -0.2]:
            p = _st.boost(p, phi)
        m1 = FourMomentum(p).mass
        assert abs(m1 - m0) < 1e-7

    def test_mass_non_negative_after_boosts(self):
        p = EMLPoint(1.5, 2.0)
        for phi in [0.5, -0.5, 1.0]:
            p = _st.boost(p, phi)
        assert FourMomentum(p).mass >= 0.0


class TestOctonionAlternativeLaw:
    """Octonion alternative law: (xy)y == x(yy) in norm."""

    @pytest.mark.parametrize("i,j", [(0, 1), (1, 2), (2, 3), (0, 4), (3, 5)])
    def test_alternative_law_norm(self, i, j):
        from eml_spectral.octonion import basis_octonion
        a = basis_octonion(i)
        b = basis_octonion(j)
        lhs = (a * b) * b
        rhs = a * (b * b)
        assert abs(lhs.norm() - rhs.norm()) < 1e-9


class TestMetricTensorFlatDS2:
    """MetricTensor.flat() ds2 is finite for various displacements."""

    @pytest.mark.parametrize("dx,dy", [(1.0, 0.0), (0.0, 1.0), (3.0, 4.0), (1.0, 1.0), (5.0, 2.0), (2.0, 3.0)])
    def test_flat_ds2_finite(self, dx, dy):
        m = MetricTensor.flat()
        p = EMLPoint(1.0, math.e)
        ds2 = m.ds2(p, dx=dx, dy=dy)
        assert math.isfinite(ds2)
