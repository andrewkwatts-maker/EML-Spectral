"""Tests for E8_248 and E8xE8 — adjoint E₈ representation and heterotic pair."""
from __future__ import annotations
import math
import pytest

from eml_spectral.exceptional import (
    FreudenthalTripleSystem as FTS,
    E7_56,
    E8_248,
    E8xE8,
)


# ── E8_248 construction ──────────────────────────────────────────────────────

class TestE8Construction:

    def test_construct(self):
        e = E8_248()
        assert isinstance(e, E8_248)

    def test_repr_mentions_dim(self):
        e = E8_248()
        assert "248" in repr(e)


# ── Gaugino condensation ─────────────────────────────────────────────────────

class TestGauginoCondensation:

    def test_returns_float(self):
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        v = e.gaugino_condensation(fts)
        assert isinstance(v, float)

    def test_finite(self):
        e = E8_248()
        for v in (0.5, 1.0, 2.0, 3.0):
            fts = FTS.from_scalar(v)
            assert math.isfinite(e.gaugino_condensation(fts))

    def test_zero_trace_returns_one(self):
        # T = 0 → exp(-a*0) = 1
        e = E8_248()
        fts = FTS([0.0] * 27)
        assert abs(e.gaugino_condensation(fts) - 1.0) < 1e-12

    @pytest.mark.parametrize("v", [0.1, 1.0, 2.0])
    def test_decreasing_in_trace(self, v):
        e = E8_248()
        small = FTS.from_scalar(v)
        big = FTS.from_scalar(v * 2.0)
        assert e.gaugino_condensation(big) < e.gaugino_condensation(small)

    def test_a_parameter_effect(self):
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        small_a = e.gaugino_condensation(fts, a=0.1)
        big_a = e.gaugino_condensation(fts, a=1.0)
        # bigger exponent → faster decay → smaller value
        assert big_a < small_a


# ── Racetrack potential (E8_248) ─────────────────────────────────────────────

class TestRacetrackPotential:

    def test_dict_keys(self):
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        d = e.racetrack_potential(fts)
        for k in ("W1", "W2", "W_total", "T_min", "lambda_eff", "cabibbo_proxy",
                  "N1", "N2"):
            assert k in d

    def test_n1_n2_passthrough(self):
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        d = e.racetrack_potential(fts, N1=24, N2=23)
        assert d["N1"] == 24 and d["N2"] == 23

    def test_lambda_eff_value(self):
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        d = e.racetrack_potential(fts, N1=24, N2=23)
        # lambda_eff = exp(-2π/24) ≈ 0.7676
        assert abs(d["lambda_eff"] - math.exp(-2.0 * math.pi / 24.0)) < 1e-12

    def test_cabibbo_is_lambda_cubed(self):
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        d = e.racetrack_potential(fts)
        assert abs(d["cabibbo_proxy"] - d["lambda_eff"] ** 3) < 1e-12

    def test_w_total_is_sum(self):
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        d = e.racetrack_potential(fts)
        assert abs(d["W_total"] - (d["W1"] + d["W2"])) < 1e-12

    def test_tmin_finite_when_n1_ne_n2(self):
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        d = e.racetrack_potential(fts, N1=24, N2=23)
        assert math.isfinite(d["T_min"])

    def test_tmin_inf_when_n1_eq_n2(self):
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        d = e.racetrack_potential(fts, N1=24, N2=24)
        assert d["T_min"] == float("inf")


# ── act_on_27 ────────────────────────────────────────────────────────────────

class TestActOn27:

    @pytest.mark.parametrize("idx", [0, 1, 5, 50, 100, 200])
    def test_returns_fts(self, idx):
        pytest.importorskip("numpy")  # _build_simple_roots imports numpy
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        out = e.act_on_27(fts, generator_index=idx)
        assert isinstance(out, FTS)

    def test_out_of_range_raises(self):
        pytest.importorskip("numpy")
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        with pytest.raises(ValueError):
            e.act_on_27(fts, generator_index=10_000)

    def test_zero_scale_near_identity(self):
        pytest.importorskip("numpy")
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        out = e.act_on_27(fts, generator_index=0, scale=0.0)
        assert abs(out.cubic_norm() - fts.cubic_norm()) < 1e-9


# ── hidden_sector_action ─────────────────────────────────────────────────────

class TestHiddenSectorAction:

    def test_returns_fts(self):
        pytest.importorskip("numpy")
        e = E8_248()
        fts = FTS.from_scalar(1.0)
        out = e.hidden_sector_action(fts, n_roots=5)
        assert isinstance(out, FTS)

    def test_finite_norm(self):
        pytest.importorskip("numpy")
        e = E8_248()
        fts = FTS.from_scalar(2.0)
        out = e.hidden_sector_action(fts, n_roots=10)
        assert math.isfinite(out.cubic_norm())


# ── E8xE8 ────────────────────────────────────────────────────────────────────

class TestE8xE8:

    def test_construct(self):
        ee = E8xE8()
        assert isinstance(ee.visible, E8_248)
        assert isinstance(ee.hidden, E8_248)

    def test_split_adjoint(self):
        ee = E8xE8()
        v, h = ee.split_adjoint()
        assert isinstance(v, E8_248)
        assert isinstance(h, E8_248)

    def test_portal_coupling_value(self):
        ee = E8xE8()
        assert abs(ee.portal_coupling() - 1.0 / math.sqrt(6.0)) < 1e-12

    def test_portal_matches_e7_alpha_leak(self):
        ee = E8xE8()
        assert ee.portal_coupling() == E7_56.ALPHA_LEAK

    def test_racetrack_potential_keys(self):
        ee = E8xE8()
        d = ee.racetrack_potential()
        for k in ("epsilon_derived", "lambda_eff", "W_hidden", "W_visible",
                  "W_total", "T_min", "N1", "N2", "stabilized"):
            assert k in d

    def test_epsilon_is_lambda_cubed(self):
        ee = E8xE8()
        d = ee.racetrack_potential()
        assert abs(d["epsilon_derived"] - d["lambda_eff"] ** 3) < 1e-12

    def test_lambda_eff_default(self):
        ee = E8xE8()
        d = ee.racetrack_potential(N1=24, N2=23)
        assert abs(d["lambda_eff"] - math.exp(-2.0 * math.pi / 24.0)) < 1e-12

    def test_stabilized_true(self):
        ee = E8xE8()
        d = ee.racetrack_potential()
        assert d["stabilized"] is True

    @pytest.mark.parametrize("n1,n2", [(24, 23), (12, 11), (30, 29)])
    def test_alternative_flux_quanta(self, n1, n2):
        ee = E8xE8()
        d = ee.racetrack_potential(N1=n1, N2=n2)
        assert math.isfinite(d["lambda_eff"])
        assert math.isfinite(d["epsilon_derived"])

    def test_repr_mentions_visible_hidden(self):
        ee = E8xE8()
        s = repr(ee)
        assert "visible" in s.lower()
        assert "hidden" in s.lower()
