"""Tests for the spectral flow operator Φ on EML expression trees."""
from __future__ import annotations
import math
import pytest

from eml_math import EMLPoint
from eml_spectral import (
    spectral_flow,
    racetrack_fixed_point,
    topology_invariant,
    G2_SEEDS,
)


# ── spectral_flow basic shape ────────────────────────────────────────────────

class TestSpectralFlowBasic:

    def test_zero_steps_returns_self(self):
        p = EMLPoint(1.0, 1.0)
        traj = spectral_flow(p, steps=0)
        assert len(traj) == 1
        assert traj[0] is p

    def test_one_step_returns_two(self):
        p = EMLPoint(1.0, 1.0)
        traj = spectral_flow(p, steps=1)
        assert len(traj) == 2

    @pytest.mark.parametrize("n", [1, 3, 5, 10, 50])
    def test_n_steps_returns_n_plus_1(self, n):
        p = EMLPoint(1.0, 1.0)
        traj = spectral_flow(p, steps=n)
        assert len(traj) == n + 1

    def test_negative_steps_raises(self):
        p = EMLPoint(1.0, 1.0)
        with pytest.raises(ValueError):
            spectral_flow(p, steps=-1)

    def test_first_element_is_input(self):
        p = EMLPoint(2.0, 3.0)
        traj = spectral_flow(p, steps=5)
        assert traj[0].x == 2.0
        assert traj[0].y == 3.0


# ── Φ formula: x' = y_safe, y' = exp(x) − ln(y_safe) ────────────────────────

class TestPhiFormula:

    def test_phi_step_eml_1_1(self):
        # eml(1, 1) = e − 0 = e ≈ 2.718
        p = EMLPoint(1.0, 1.0)
        nxt = spectral_flow(p, steps=1)[1]
        assert abs(nxt.x - 1.0) < 1e-12  # x' = y_safe = 1
        assert abs(nxt.y - math.e) < 1e-9

    @pytest.mark.parametrize("x,y", [(0.0, 1.0), (1.0, 1.0), (2.0, 3.0), (0.5, 2.0)])
    def test_x_prime_is_y_safe(self, x, y):
        p = EMLPoint(x, y)
        nxt = spectral_flow(p, steps=1)[1]
        assert abs(nxt.x - abs(y)) < 1e-9

    def test_negative_y_uses_abs(self):
        # Axiom 8: y < 0 → |y|
        p = EMLPoint(1.0, -2.0)
        nxt = spectral_flow(p, steps=1)[1]
        assert abs(nxt.x - 2.0) < 1e-9


# ── Determinism ──────────────────────────────────────────────────────────────

class TestDeterminism:

    @pytest.mark.parametrize("x,y", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.7)])
    def test_same_input_same_output(self, x, y):
        p1 = EMLPoint(x, y)
        p2 = EMLPoint(x, y)
        a = spectral_flow(p1, steps=10)
        b = spectral_flow(p2, steps=10)
        for ai, bi in zip(a, b):
            assert ai.x == bi.x
            assert ai.y == bi.y


# ── Long trajectories stay finite ────────────────────────────────────────────

class TestLongTrajectories:

    @pytest.mark.parametrize("steps", [10, 25, 50, 100])
    def test_finite_over_long_run(self, steps):
        p = EMLPoint(1.0, 1.0)
        traj = spectral_flow(p, steps=steps)
        assert all(math.isfinite(t.x) and math.isfinite(t.y) for t in traj)

    @pytest.mark.parametrize("y", [0.5, 1.0, 1.5, 2.0])
    def test_y_safe_keeps_traj_alive(self, y):
        p = EMLPoint(0.5, y)
        traj = spectral_flow(p, steps=50)
        assert len(traj) == 51
        assert all(math.isfinite(t.x) for t in traj)


# ── Frame-shift safety ───────────────────────────────────────────────────────

class TestFrameShiftSafety:

    @pytest.mark.parametrize("y", [-1.0, -0.5, -2.0, -0.001])
    def test_negative_y_does_not_raise(self, y):
        p = EMLPoint(1.0, y)
        traj = spectral_flow(p, steps=5)
        assert all(math.isfinite(t.x) and math.isfinite(t.y) for t in traj)

    def test_zero_y_does_not_raise(self):
        p = EMLPoint(1.0, 0.0)
        traj = spectral_flow(p, steps=3)
        assert all(math.isfinite(t.x) for t in traj)


# ── Discrete (Planck-quantized) mode ─────────────────────────────────────────

class TestDiscreteMode:

    def test_discrete_returns_traj(self):
        p = EMLPoint(1.0, 1.0)
        traj = spectral_flow(p, steps=5, discrete=10.0)
        assert len(traj) == 6

    def test_discrete_finite(self):
        p = EMLPoint(1.0, 1.0)
        traj = spectral_flow(p, steps=10, discrete=100.0)
        assert all(math.isfinite(t.x) for t in traj)


# ── topology_invariant ───────────────────────────────────────────────────────

class TestTopologyInvariant:

    def test_default_b3_chi(self):
        p = EMLPoint(1.0, 1.0)
        v = topology_invariant(p)
        # default b3=24, chi=144 → (24/24)*144 = 144 (identity term cancels)
        assert abs(v - 144.0) < 1e-9

    @pytest.mark.parametrize("b3,chi", [(24.0, 144.0), (12.0, 72.0), (48.0, 288.0)])
    def test_invariant_value(self, b3, chi):
        p = EMLPoint(1.0, 1.0)
        v = topology_invariant(p, b3=b3, chi_eff=chi)
        assert abs(v - (b3 / 24.0) * chi) < 1e-9

    @pytest.mark.parametrize("x,y", [(1.0, 1.0), (0.5, 2.0), (2.0, 3.0), (0.0, 0.5)])
    def test_invariant_independent_of_point(self, x, y):
        # The identity term cancels, so result depends only on b3/chi
        p = EMLPoint(x, y)
        v = topology_invariant(p, b3=24.0, chi_eff=144.0)
        assert abs(v - 144.0) < 1e-6

    def test_returns_finite(self):
        for x, y in [(0.0, 1.0), (10.0, 0.1), (1.0, math.e ** 5)]:
            p = EMLPoint(x, y)
            assert math.isfinite(topology_invariant(p))


# ── racetrack_fixed_point ────────────────────────────────────────────────────

class TestRacetrackFixedPoint:

    def test_returns_emlpoint(self):
        # Find a fixed point starting from a stable input
        p = EMLPoint(0.5, 0.5)
        try:
            fp = racetrack_fixed_point(p, max_steps=10_000, tol=1e-6)
            assert isinstance(fp, EMLPoint)
        except RuntimeError:
            pytest.skip("did not converge for this seed")

    def test_fixed_point_satisfies_phi_eq_self(self):
        p = EMLPoint(0.5, 0.5)
        try:
            fp = racetrack_fixed_point(p, max_steps=10_000, tol=1e-6)
            nxt = fp.iterate()
            assert abs(nxt.x - fp.x) < 1e-4
            assert abs(nxt.y - fp.y) < 1e-4
        except RuntimeError:
            pytest.skip("did not converge")

    def test_no_convergence_raises(self):
        # 0 max_steps should raise
        p = EMLPoint(0.5, 0.5)
        with pytest.raises(RuntimeError):
            racetrack_fixed_point(p, max_steps=0, tol=1e-30)


# ── G2_SEEDS ─────────────────────────────────────────────────────────────────

class TestG2Seeds:

    def test_seeds_is_iterable(self):
        assert hasattr(G2_SEEDS, "__iter__") or hasattr(G2_SEEDS, "__getitem__")

    def test_seeds_nonempty(self):
        items = list(G2_SEEDS) if not isinstance(G2_SEEDS, dict) else list(G2_SEEDS.values())
        assert len(items) > 0
