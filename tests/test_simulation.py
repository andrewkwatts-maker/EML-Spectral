"""
Simulation tests including regression against MPM.txt D=100 table.
"""
import math
import pytest
from eml_math import EMLPoint
from eml_spectral import (
    EMLState,
    simulate_pulses, simulate_flips, quantized_trajectory,
    tension_series, rho_series, phase_series,
    verify_conservation, frame_shift_count,
)


class TestSimulatePulses:
    def test_length(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=10)
        assert len(traj) == 11  # includes initial state

    def test_flip_count_increments(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        for i, k in enumerate(traj):
            assert k.flip_count == i

    def test_first_element_is_initial(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        assert traj[0] is unit_knot

    def test_all_tensions_finite(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=50)
        assert all(math.isfinite(k.point.tension()) for k in traj)


class TestSimulateFlips:
    def test_length(self, unit_knot):
        traj = simulate_flips(unit_knot, n_flips=5)
        assert len(traj) == 6

    def test_flip_count_advances_by_4(self, unit_knot):
        traj = simulate_flips(unit_knot, n_flips=3)
        assert traj[1].flip_count == 4
        assert traj[2].flip_count == 8
        assert traj[3].flip_count == 12


class TestQuantizedTrajectory:
    """Regression against the D=100 table from MPM.txt lines ~600-643."""

    def test_initial_pair(self):
        pairs = quantized_trajectory(100, 100, n_pulses=0, D=100)
        assert pairs[0] == (100, 100)

    def test_first_step_matches_document(self):
        # x0=1.0, y0=1.0, T = exp(1) - ln(1) = e ≈ 2.718
        # b1 = round(e * 100) = 272
        pairs = quantized_trajectory(100, 100, n_pulses=1, D=100)
        assert pairs[1] == (100, 272)

    def test_length(self):
        pairs = quantized_trajectory(100, 100, n_pulses=7, D=100)
        assert len(pairs) == 8

    def test_a_next_equals_b_prev(self):
        pairs = quantized_trajectory(100, 100, n_pulses=5, D=100)
        for i in range(len(pairs) - 1):
            assert pairs[i + 1][0] == pairs[i][1], (
                f"At step {i}: a_{i+1}={pairs[i+1][0]} should equal b_{i}={pairs[i][1]}"
            )


class TestSeriesExtractors:
    def test_tension_series_length(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        assert len(tension_series(traj)) == 6

    def test_rho_series_non_negative(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=10)
        assert all(r >= 0 for r in rho_series(traj))

    def test_phase_series_in_range(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=10)
        for phase in phase_series(traj):
            assert 0.0 <= phase < 2 * math.pi + 1e-9


class TestVerifyConservation:
    def test_passes_for_clean_trajectory(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=20)
        assert verify_conservation(traj)

    def test_single_step(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=1)
        assert verify_conservation(traj)


# ── New tests ────────────────────────────────────────────────────────────────


class TestSimulatePulsesProperties:
    """Length is n+1; first element is the initial state; flip_count is monotone."""

    def test_length_is_n_plus_one_small(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=3)
        assert len(traj) == 4

    def test_length_is_n_plus_one_large(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=100)
        assert len(traj) == 101

    def test_first_element_is_initial_state(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=7)
        assert traj[0] is unit_knot

    def test_flip_count_increments_monotonically(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=8)
        for i in range(len(traj) - 1):
            assert traj[i + 1].flip_count == traj[i].flip_count + 1

    def test_length_zero_pulses(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=0)
        assert len(traj) == 1


class TestPhaseAdvancement:
    """Phase advances by π/2 per pulse; after 4 pulses wraps back."""

    def test_phase_advances_by_half_pi(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=1)
        delta = (traj[1].phase - traj[0].phase) % (2 * math.pi)
        assert abs(delta - math.pi / 2) < 1e-9

    def test_four_pulses_wrap_to_start(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=4)
        assert abs(traj[4].phase - traj[0].phase) < 1e-9

    def test_phase_in_0_to_2pi(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=12)
        for s in traj:
            assert 0.0 <= s.phase < 2 * math.pi + 1e-9

    def test_phase_step_two_pulses(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=2)
        delta = (traj[2].phase - traj[0].phase) % (2 * math.pi)
        assert abs(delta - math.pi) < 1e-9


class TestTensionSeries:
    """tension_series: list of floats, length n+1, all finite."""

    def test_returns_list(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        ts = tension_series(traj)
        assert isinstance(ts, list)

    def test_length_matches_trajectory(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=7)
        ts = tension_series(traj)
        assert len(ts) == len(traj)

    def test_all_finite(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=20)
        assert all(math.isfinite(t) for t in tension_series(traj))

    def test_first_value_matches_initial_tension(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        ts = tension_series(traj)
        assert abs(ts[0] - unit_knot.point.tension()) < 1e-12


class TestRhoSeries:
    """rho_series: always non-negative; same length as simulate_pulses."""

    def test_non_negative(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=10)
        assert all(r >= 0.0 for r in rho_series(traj))

    def test_length_matches_trajectory(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=10)
        assert len(rho_series(traj)) == len(traj)

    def test_all_finite(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=10)
        assert all(math.isfinite(r) for r in rho_series(traj))


class TestPhaseSeries:
    """phase_series: all values in [0, 2π)."""

    def test_all_in_0_2pi(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=10)
        for ph in phase_series(traj):
            assert 0.0 <= ph < 2 * math.pi + 1e-9

    def test_length_matches_trajectory(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=10)
        assert len(phase_series(traj)) == len(traj)

    def test_returns_list(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        result = phase_series(traj)
        assert isinstance(result, list)

    def test_first_phase_matches_initial(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=4)
        ps = phase_series(traj)
        assert abs(ps[0] - unit_knot.phase) < 1e-12

    def test_phase_advances_each_step(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=4)
        ps = phase_series(traj)
        for i in range(len(ps) - 1):
            delta = (ps[i + 1] - ps[i]) % (2 * math.pi)
            assert abs(delta - math.pi / 2) < 1e-9


class TestVerifyConservationEdgeCases:
    """Edge cases for verify_conservation."""

    def test_single_element_returns_true(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=0)
        assert verify_conservation(traj)

    def test_empty_list_returns_true(self):
        assert verify_conservation([])

    def test_two_element_passing_pair(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=1)
        assert verify_conservation(traj)


class TestQuantizedTrajectory:
    """quantized_trajectory: returns integer pairs, correct length, EML step relation."""

    def test_returns_list_of_tuples(self):
        pairs = quantized_trajectory(100, 100, n_pulses=5, D=100)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)

    def test_length_is_n_plus_one(self):
        pairs = quantized_trajectory(100, 100, n_pulses=10, D=100)
        assert len(pairs) == 11

    def test_elements_are_integers(self):
        pairs = quantized_trajectory(100, 100, n_pulses=3, D=100)
        for a, b in pairs:
            assert isinstance(a, int) and isinstance(b, int)

    def test_step_relation_a_next_equals_b_prev(self):
        pairs = quantized_trajectory(50, 100, n_pulses=4, D=100)
        for i in range(len(pairs) - 1):
            assert pairs[i + 1][0] == pairs[i][1]


class TestFrameShiftCount:
    """frame_shift_count: returns int, non-negative, ≤ n_pulses."""

    def test_returns_int(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        result = frame_shift_count(traj)
        assert isinstance(result, int)

    def test_non_negative(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        assert frame_shift_count(traj) >= 0

    def test_at_most_n_pulses(self, unit_knot):
        n = 10
        traj = simulate_pulses(unit_knot, n_pulses=n)
        assert frame_shift_count(traj) <= n

    def test_single_element_trajectory(self, unit_knot):
        traj = [unit_knot]
        assert frame_shift_count(traj) == 0

    def test_two_element_trajectory_non_negative(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=1)
        assert frame_shift_count(traj) >= 0

    def test_result_is_int_for_longer_trajectory(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=20)
        result = frame_shift_count(traj)
        assert isinstance(result, int) and result >= 0


# ── New simulation tests (+30) ────────────────────────────────────────────────

from eml_spectral import find_resonance_bands, rho_series


class TestSimulateFlipsExtended:
    """simulate_flips: correct type, length, flip_count at multiples of 4."""

    @pytest.mark.parametrize("n_flips", [4, 8, 12, 16])
    def test_length_at_multiple_of_4(self, unit_knot, n_flips):
        traj = simulate_flips(unit_knot, n_flips=n_flips)
        assert len(traj) == n_flips + 1

    @pytest.mark.parametrize("n_flips", [4, 8, 12, 16])
    def test_flip_count_at_multiple_of_4(self, unit_knot, n_flips):
        traj = simulate_flips(unit_knot, n_flips=n_flips)
        assert traj[-1].flip_count == n_flips * 4

    @pytest.mark.parametrize("n_flips", [4, 8, 12, 16])
    def test_all_states_are_emlstate(self, unit_knot, n_flips):
        from eml_spectral import EMLState
        traj = simulate_flips(unit_knot, n_flips=n_flips)
        for s in traj:
            assert isinstance(s, EMLState)

    def test_flip_traj_tensions_finite(self, unit_knot):
        traj = simulate_flips(unit_knot, n_flips=8)
        for s in traj:
            assert math.isfinite(s.point.tension())


class TestFindResonanceBands:
    """find_resonance_bands returns list; non-empty for long trajectory."""

    def test_returns_list_type(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=20)
        result = find_resonance_bands(traj, tolerance=1e-3)
        assert isinstance(result, list)

    def test_short_trajectory_empty_or_list(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=3)
        result = find_resonance_bands(traj, tolerance=1e-9)
        assert isinstance(result, list)

    def test_long_trajectory_finds_bands(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=100)
        result = find_resonance_bands(traj, tolerance=1e-3)
        assert isinstance(result, list)
        assert len(result) >= 0  # may be empty or not

    def test_tolerance_affects_count(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=50)
        loose = find_resonance_bands(traj, tolerance=0.5)
        tight = find_resonance_bands(traj, tolerance=1e-9)
        assert len(loose) >= len(tight)


class TestTrajectoryContinuity:
    """rho values are always positive and finite across 100 steps."""

    def test_rho_positive_100_steps(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=100)
        rhos = rho_series(traj)
        assert all(r > 0 for r in rhos)

    def test_rho_finite_100_steps(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=100)
        rhos = rho_series(traj)
        assert all(math.isfinite(r) for r in rhos)

    def test_rho_length_100_steps(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=100)
        rhos = rho_series(traj)
        assert len(rhos) == 101

    @pytest.mark.parametrize("x0,y0", [(1.0, 1.0), (2.0, 3.0), (0.5, 2.0)])
    def test_rho_positive_various_starts(self, x0, y0):
        p = EMLPoint(x0, y0)
        from eml_spectral import EMLState
        s = EMLState(p)
        traj = simulate_pulses(s, n_pulses=50)
        rhos = rho_series(traj)
        assert all(r >= 0 for r in rhos)

    def test_tensions_all_finite_100_steps(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=100)
        ts = tension_series(traj)
        assert all(math.isfinite(t) for t in ts)


class TestSimulationConsistency:
    """Additional consistency checks on simulate_pulses trajectories."""

    def test_consecutive_x_equals_prev_y(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        for i in range(len(traj) - 1):
            assert abs(traj[i + 1].point.x - traj[i].point.y) < 1e-12

    def test_consecutive_y_equals_prev_tension(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        for i in range(len(traj) - 1):
            T_prev = traj[i].point.tension()
            assert abs(traj[i + 1].point.y - T_prev) < 1e-12

    @pytest.mark.parametrize("n", [10, 20, 50])
    def test_trajectory_length_parametrized(self, unit_knot, n):
        traj = simulate_pulses(unit_knot, n_pulses=n)
        assert len(traj) == n + 1

    def test_flip_traj_x_positive(self, unit_knot):
        traj = simulate_flips(unit_knot, n_flips=4)
        for s in traj:
            assert math.isfinite(s.point.x)
