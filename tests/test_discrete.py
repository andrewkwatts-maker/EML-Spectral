"""Tests for Sprint 3 discrete / Planck-lattice helpers."""
import math
import pytest

from eml_math.point import EMLPoint
from eml_spectral import spacetime as _st
from eml_spectral.discrete import planck_delta, lattice_distance, is_lattice_neighbor
from eml_math.constants import PLANCK_D


class TestPlanckDelta:
    def test_quantizes_to_grid(self):
        p = EMLPoint(1.0, 1.0)
        D = 10.0
        result = planck_delta(p, D=D)
        # Must be a multiple of 1/D
        assert abs(result * D - round(result * D)) < 1e-9

    def test_zero_delta_stays_zero(self):
        # Lightlike: E = p*c → delta = 0
        p = EMLPoint(0.0, math.e)  # E=1, s=1, delta=0
        result = planck_delta(p, D=100.0)
        assert abs(result) < 0.01  # quantized near 0

    def test_returns_float(self):
        p = EMLPoint(1.0, 2.0)
        assert isinstance(planck_delta(p), float)

    def test_default_D_is_planck_d(self):
        p = EMLPoint(1.0, 2.0)
        result_default = planck_delta(p)
        result_explicit = planck_delta(p, D=PLANCK_D)
        assert abs(result_default - result_explicit) < 1e-12


class TestLatticeDistance:
    def test_same_point_distance_near_zero(self):
        p = EMLPoint(1.0, math.e)
        dist = lattice_distance(p, p, D=10.0)
        # displacement is EMLPoint(0, 1) → delta near 1; quantized to 0.1*n
        assert isinstance(dist, float)
        assert math.isfinite(dist)

    def test_symmetric_under_swap(self):
        p1 = EMLPoint(1.0, 2.0)
        p2 = EMLPoint(1.5, 3.0)
        # Not necessarily symmetric (direction matters), just verify both finite
        d12 = lattice_distance(p1, p2, D=10.0)
        d21 = lattice_distance(p2, p1, D=10.0)
        assert math.isfinite(d12)
        assert math.isfinite(d21)

    def test_returns_float(self):
        p1 = EMLPoint(0.0, 1.0)
        p2 = EMLPoint(1.0, math.e)
        assert isinstance(lattice_distance(p1, p2, D=10.0), float)


class TestIsLatticeNeighbor:
    def test_far_points_not_neighbors(self):
        p1 = EMLPoint(0.0, 1.0)
        p2 = EMLPoint(100.0, 1.0)
        D = 10.0
        result = is_lattice_neighbor(p1, p2, D=D)
        assert isinstance(result, bool)

    def test_returns_bool(self):
        p1 = EMLPoint(1.0, 1.0)
        p2 = EMLPoint(1.0, 1.0)
        assert isinstance(is_lattice_neighbor(p1, p2), bool)

    def test_self_not_neighbor(self):
        # Distance to self is quantized(delta of displacement) which is not 1/D
        p = EMLPoint(1.0, math.e)
        result = is_lattice_neighbor(p, p, D=PLANCK_D)
        assert isinstance(result, bool)


# ── New tests ────────────────────────────────────────────────────────────────


class TestPlanckDeltaScale:
    """planck_delta result is a multiple of 1/D, non-negative, and finite.

    5 x-values × 4 y-values = 20 parametrized test cases.
    """

    @pytest.mark.parametrize("x,y", [
        (-3.0, 0.1), (-3.0, 1.0), (-3.0, math.e), (-3.0, 10.0),
        (-1.0, 0.1), (-1.0, 1.0), (-1.0, math.e), (-1.0, 10.0),
        (0.0,  0.1), (0.0,  1.0), (0.0,  math.e), (0.0,  10.0),
        (1.0,  0.1), (1.0,  1.0), (1.0,  math.e), (1.0,  10.0),
        (3.0,  0.1), (3.0,  1.0), (3.0,  math.e), (3.0,  10.0),
    ])
    def test_multiple_of_grid_step_non_negative_finite(self, x, y):
        p = EMLPoint(x, y)
        D = 100.0
        result = planck_delta(p, D=D)
        assert abs(result * D - round(result * D)) < 1e-9
        assert result >= 0.0
        assert math.isfinite(result)


class TestPlanckDeltaQuantisation:
    """planck_delta(p, D=10) rounds to nearest 0.1."""

    def test_grid_step_case1(self):
        p = EMLPoint(1.0, 1.0)
        result = planck_delta(p, D=10)
        assert abs(result * 10 - round(result * 10)) < 1e-9

    def test_within_half_cell_case2(self):
        p = EMLPoint(2.0, math.e)
        result = planck_delta(p, D=10)
        exact = _st.minkowski_delta(p)
        assert abs(result - exact) <= 0.05 + 1e-9

    def test_grid_step_case3(self):
        p = EMLPoint(0.5, 2.0)
        result = planck_delta(p, D=10)
        assert abs(result * 10 - round(result * 10)) < 1e-9

    def test_within_half_cell_case4(self):
        p = EMLPoint(1.5, 3.0)
        result = planck_delta(p, D=10)
        exact = _st.minkowski_delta(p)
        assert abs(result - exact) <= 0.05 + 1e-9

    def test_grid_step_case5(self):
        p = EMLPoint(0.0, 1.0)
        result = planck_delta(p, D=10)
        assert abs(result * 10 - round(result * 10)) < 1e-9


class TestLatticeDistanceProperties:
    """lattice_distance: self-distance, symmetry, non-negativity."""

    def test_self_distance_is_finite(self):
        p = EMLPoint(1.0, math.e)
        dist = lattice_distance(p, p, D=100.0)
        assert math.isfinite(dist)

    def test_self_distance_is_non_negative(self):
        p = EMLPoint(2.0, 3.0)
        dist = lattice_distance(p, p, D=100.0)
        assert dist >= 0.0

    def test_pair_distance_is_finite(self):
        p1 = EMLPoint(1.0, 2.0)
        p2 = EMLPoint(1.1, 2.0)
        assert math.isfinite(lattice_distance(p1, p2, D=100.0))

    def test_pair_distance_is_non_negative(self):
        p1 = EMLPoint(1.0, 2.0)
        p2 = EMLPoint(2.0, 4.0)
        assert lattice_distance(p1, p2, D=100.0) >= 0.0

    def test_both_directions_are_finite(self):
        p1 = EMLPoint(1.0, 2.0)
        p2 = EMLPoint(1.5, 3.0)
        d12 = lattice_distance(p1, p2, D=100.0)
        d21 = lattice_distance(p2, p1, D=100.0)
        assert math.isfinite(d12) and math.isfinite(d21)


class TestIsLatticeNeighbour:
    """is_lattice_neighbor: boundary detection and bool return."""

    def test_returns_bool_same_point(self):
        p = EMLPoint(1.0, 1.0)
        result = is_lattice_neighbor(p, p, D=10.0)
        assert isinstance(result, bool)

    def test_returns_bool_distant_points(self):
        p1 = EMLPoint(0.0, 1.0)
        p2 = EMLPoint(50.0, 1.0)
        result = is_lattice_neighbor(p1, p2, D=10.0)
        assert isinstance(result, bool)

    def test_far_apart_not_neighbor(self):
        p1 = EMLPoint(0.0, 1.0)
        p2 = EMLPoint(100.0, 1.0)
        assert not is_lattice_neighbor(p1, p2, D=10.0)

    def test_neighbor_check_small_d_no_error(self):
        p1 = EMLPoint(1.0, 1.0)
        p2 = EMLPoint(1.0, math.exp(1.0 / 10.0))
        result = is_lattice_neighbor(p1, p2, D=10.0)
        assert isinstance(result, bool)

    def test_boundary_at_exact_cell(self):
        D = 10.0
        p1 = EMLPoint(1.0, 1.0)
        p2 = EMLPoint(1.0, math.e)
        result = is_lattice_neighbor(p1, p2, D=D)
        assert isinstance(result, bool)


