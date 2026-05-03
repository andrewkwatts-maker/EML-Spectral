"""Cross-cutting EML axiom tests as they manifest in eml-spectral."""
from __future__ import annotations
import math
import pytest

from eml_math import EMLPoint
from eml_spectral import spacetime
from eml_spectral import spectral_flow, topology_invariant


# ── Axiom 3 (Residue Flow) ───────────────────────────────────────────────────

class TestAxiom3ResidueFlow:

    @pytest.mark.parametrize("x,y", [(0.0, 1.0), (1.0, 1.0), (2.0, 3.0), (0.5, 2.0)])
    def test_phi_step(self, x, y):
        p = EMLPoint(x, y)
        nxt = spectral_flow(p, steps=1)[1]
        # x' = y_safe = |y|
        assert abs(nxt.x - abs(y)) < 1e-9


# ── Axiom 4 (Topological Invariant) ──────────────────────────────────────────

class TestAxiom4TopologyInvariant:

    @pytest.mark.parametrize("x,y", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.7)])
    def test_invariant_along_flow(self, x, y):
        # topology_invariant should equal (b3/24)·chi_eff = 144 along every step
        p = EMLPoint(x, y)
        traj = spectral_flow(p, steps=10)
        for tp in traj:
            assert abs(topology_invariant(tp) - 144.0) < 1e-6

    @pytest.mark.parametrize("b3,chi", [(24.0, 144.0), (12.0, 72.0), (48.0, 288.0)])
    def test_invariant_value_scales(self, b3, chi):
        p = EMLPoint(1.0, 1.0)
        assert abs(topology_invariant(p, b3=b3, chi_eff=chi) - (b3/24.0)*chi) < 1e-9


# ── Axiom 6 (Racetrack Fixed Points) ─────────────────────────────────────────

class TestAxiom6RacetrackFixedPoints:

    def test_fixed_point_definition(self):
        # If T* is a fixed point: Φ(T*) = T*  ⇔  y* = exp(x*) − ln(y*) AND x* = y*
        # Try an explicit fixed point candidate: x* = y* = solve x = e^x − ln(x)
        # Doesn't have a clean closed form, so just verify Φ(p).x = |p.y|
        p = EMLPoint(0.6, 0.6)
        nxt = p.iterate()
        assert abs(nxt.x - abs(p.y)) < 1e-9


# ── Axiom 7 (Lorentz Invariance) ─────────────────────────────────────────────

class TestAxiom7LorentzInvariance:

    @pytest.mark.parametrize("phi", [0.1, 0.2, 0.5, 0.7])
    def test_minkowski_preserved_under_boost(self, phi):
        p = EMLPoint(1.0, 1.5)
        d0 = spacetime.minkowski_delta(p)
        d1 = spacetime.minkowski_delta(spacetime.boost(p, phi))
        assert abs(d0 - d1) < 1e-6

    @pytest.mark.parametrize("phi1,phi2", [(0.1, 0.2), (0.3, 0.4), (-0.1, 0.5)])
    def test_boost_composition_law(self, phi1, phi2):
        p = EMLPoint(1.0, 1.5)
        d_a = spacetime.minkowski_delta(spacetime.boost(spacetime.boost(p, phi1), phi2))
        d_b = spacetime.minkowski_delta(spacetime.boost(p, phi1 + phi2))
        assert abs(d_a - d_b) < 1e-6

    @pytest.mark.parametrize("x,y", [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0)])
    def test_boost_inverse(self, x, y):
        # boost(p, φ) then boost(., -φ) should preserve Minkowski delta
        p = EMLPoint(x, y)
        d0 = spacetime.minkowski_delta(p)
        b = spacetime.boost(spacetime.boost(p, 0.3), -0.3)
        assert abs(spacetime.minkowski_delta(b) - d0) < 1e-6


# ── Axiom 8 (Frame-Shift Guard) ──────────────────────────────────────────────

class TestAxiom8FrameShift:

    @pytest.mark.parametrize("y", [-0.001, -0.5, -1.0, -2.0, -10.0])
    def test_negative_y_treated_as_abs(self, y):
        # spectral_flow should give x' = |y|
        p = EMLPoint(1.0, y)
        nxt = spectral_flow(p, steps=1)[1]
        assert abs(nxt.x - abs(y)) < 1e-9

    @pytest.mark.parametrize("y", [-1.0, -0.5, -2.0])
    def test_minkowski_handles_negative_y(self, y):
        p = EMLPoint(1.0, y)
        d = spacetime.minkowski_delta(p)
        assert math.isfinite(d) and d >= 0

    def test_zero_y_does_not_raise(self):
        p = EMLPoint(1.0, 0.0)
        spacetime.minkowski_delta(p)   # doesn't raise
        spectral_flow(p, steps=1)


# ── Axiom 5 (Sheffer Operator on Trajectories) ──────────────────────────────

class TestAxiom5Sheffer:

    @pytest.mark.parametrize("x,y", [(0.0, 1.0), (1.0, 1.0), (2.0, 3.0)])
    def test_y_prime_is_eml(self, x, y):
        # y' of next state = exp(x) − ln(y_safe)
        p = EMLPoint(x, y)
        nxt = spectral_flow(p, steps=1)[1]
        expected = math.exp(x) - math.log(abs(y) if y <= 0 else y)
        assert abs(nxt.y - expected) < 1e-6


# ── Composition / chain rules ────────────────────────────────────────────────

class TestComposition:

    @pytest.mark.parametrize("steps", [3, 5, 10])
    def test_n_steps_equals_chained_one_steps(self, steps):
        p = EMLPoint(1.0, 1.5)
        # Apply N steps in one call
        traj_one = spectral_flow(p, steps=steps)
        # Apply 1 step at a time, reusing each result
        traj_chain = [p]
        cur = p
        for _ in range(steps):
            cur = spectral_flow(cur, steps=1)[1]
            traj_chain.append(cur)
        for a, b in zip(traj_one, traj_chain):
            assert abs(a.x - b.x) < 1e-9
            assert abs(a.y - b.y) < 1e-9
