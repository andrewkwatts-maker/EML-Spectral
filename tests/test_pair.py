"""Tests for TensionPair — real replacement for complex numbers."""
import math
import pytest
from eml_spectral import EMLPair


class TestConstruction:
    def test_unit_i(self):
        i = EMLPair.unit_i()
        assert i.real_tension == pytest.approx(0.0, abs=1e-10)
        assert i.imag_tension == pytest.approx(1.0, rel=1e-10)

    def test_from_values(self):
        z = EMLPair.from_values(3.0, 4.0)
        assert z.real_tension == pytest.approx(3.0, rel=1e-10)
        assert z.imag_tension == pytest.approx(4.0, rel=1e-10)

    def test_one(self):
        one = EMLPair.one()
        assert one.real_tension == pytest.approx(1.0, rel=1e-10)
        assert one.imag_tension == pytest.approx(0.0, abs=1e-10)


class TestModulusArgument:
    def test_modulus_345(self):
        z = EMLPair.from_values(3.0, 4.0)
        assert z.modulus == pytest.approx(5.0, rel=1e-8)

    def test_modulus_unit_i(self):
        assert EMLPair.unit_i().modulus == pytest.approx(1.0, rel=1e-8)

    def test_argument_unit_i(self):
        i = EMLPair.unit_i()
        # arctan(1/0) → π/2 (handled via arctan of large number)
        assert abs(i.argument) > 1.0  # approaches π/2


class TestArithmetic:
    def test_add(self):
        z1 = EMLPair.from_values(1.0, 2.0)
        z2 = EMLPair.from_values(3.0, 4.0)
        z3 = z1 + z2
        assert z3.real_tension == pytest.approx(4.0, rel=1e-10)
        assert z3.imag_tension == pytest.approx(6.0, rel=1e-10)

    def test_mul(self):
        # (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
        z1 = EMLPair.from_values(1.0, 2.0)
        z2 = EMLPair.from_values(3.0, 4.0)
        z3 = z1 * z2
        assert z3.real_tension == pytest.approx(-5.0, abs=1e-9)
        assert z3.imag_tension == pytest.approx(10.0, rel=1e-9)

    def test_i_squared_is_neg_one(self):
        i = EMLPair.unit_i()
        i2 = i * i    # i² = -1 + 0i
        assert i2.real_tension == pytest.approx(-1.0, abs=1e-9)
        assert i2.imag_tension == pytest.approx(0.0, abs=1e-9)

    def test_conjugate(self):
        z = EMLPair.from_values(3.0, 4.0)
        zc = z.conjugate()
        assert zc.real_tension == pytest.approx(3.0, rel=1e-10)
        assert zc.imag_tension == pytest.approx(-4.0, abs=1e-9)


class TestRotatePhase:
    def test_rotate_half_pi_gives_i_times(self):
        # Rotating (1, 0) by π/2 should give (0, 1) = i
        one = EMLPair.from_values(1.0, 0.0)
        rotated = one.rotate_phase(math.pi / 2)
        assert rotated.real_tension == pytest.approx(0.0, abs=1e-8)
        assert rotated.imag_tension == pytest.approx(1.0, rel=1e-8)

    def test_rotate_pi_gives_negation(self):
        # Rotating (1, 0) by π should give (-1, 0)
        one = EMLPair.from_values(1.0, 0.0)
        rotated = one.rotate_phase(math.pi)
        assert rotated.real_tension == pytest.approx(-1.0, abs=1e-8)
        assert rotated.imag_tension == pytest.approx(0.0, abs=1e-8)


class TestFromPolar:
    def test_unit_magnitude(self):
        z = EMLPair.from_polar(1.0, 0.0)
        assert z.real_tension == pytest.approx(1.0, rel=1e-10)
        assert z.imag_tension == pytest.approx(0.0, abs=1e-10)

    def test_45_degrees(self):
        z = EMLPair.from_polar(math.sqrt(2), math.pi / 4)
        assert z.real_tension == pytest.approx(1.0, rel=1e-8)
        assert z.imag_tension == pytest.approx(1.0, rel=1e-8)


# ── New tests ────────────────────────────────────────────────────────────────


class TestEMLPairArithmetic:
    """add, subtract, multiply between EMLPair instances."""

    def test_add_real_components(self):
        z1 = EMLPair.from_values(2.0, 3.0)
        z2 = EMLPair.from_values(5.0, 7.0)
        result = z1 + z2
        assert abs(result.real_tension - 7.0) < 1e-9

    def test_add_imag_components(self):
        z1 = EMLPair.from_values(2.0, 3.0)
        z2 = EMLPair.from_values(5.0, 7.0)
        result = z1 + z2
        assert abs(result.imag_tension - 10.0) < 1e-9

    def test_subtract_real(self):
        z1 = EMLPair.from_values(5.0, 3.0)
        z2 = EMLPair.from_values(2.0, 1.0)
        result = z1 - z2
        assert abs(result.real_tension - 3.0) < 1e-9

    def test_subtract_imag(self):
        z1 = EMLPair.from_values(5.0, 3.0)
        z2 = EMLPair.from_values(2.0, 1.0)
        result = z1 - z2
        assert abs(result.imag_tension - 2.0) < 1e-9

    def test_multiply_real_part(self):
        # (2+3i)(4+5i) = (8-15) + (10+12)i = -7 + 22i
        z1 = EMLPair.from_values(2.0, 3.0)
        z2 = EMLPair.from_values(4.0, 5.0)
        result = z1 * z2
        assert abs(result.real_tension - (-7.0)) < 1e-9

    def test_multiply_imag_part(self):
        z1 = EMLPair.from_values(2.0, 3.0)
        z2 = EMLPair.from_values(4.0, 5.0)
        result = z1 * z2
        assert abs(result.imag_tension - 22.0) < 1e-9

    def test_add_zero(self):
        z = EMLPair.from_values(3.0, 4.0)
        zero = EMLPair.zero()
        result = z + zero
        assert abs(result.real_tension - 3.0) < 1e-9
        assert abs(result.imag_tension - 4.0) < 1e-9


class TestModulusProperties:
    """modulus ≥ 0 always; |z|² = real² + imag²; unit_i has modulus 1."""

    def test_modulus_non_negative(self):
        z = EMLPair.from_values(-3.0, -4.0)
        assert z.modulus >= 0.0

    def test_modulus_squared_equals_components_squared(self):
        z = EMLPair.from_values(3.0, 4.0)
        r, im = z.real_tension, z.imag_tension
        assert abs(z.modulus ** 2 - (r * r + im * im)) < 1e-9

    def test_unit_i_modulus_is_one(self):
        assert abs(EMLPair.unit_i().modulus - 1.0) < 1e-9

    def test_modulus_zero_pair(self):
        z = EMLPair.zero()
        assert z.modulus >= 0.0

    def test_modulus_real_only(self):
        z = EMLPair.from_values(5.0, 0.0)
        assert abs(z.modulus - 5.0) < 1e-9


class TestFromValues:
    """from_values(r, i) round-trips through .real_tension and .imag_tension."""

    def test_round_trip_positive(self):
        z = EMLPair.from_values(7.0, 8.0)
        assert abs(z.real_tension - 7.0) < 1e-9
        assert abs(z.imag_tension - 8.0) < 1e-9

    def test_round_trip_negative_imag(self):
        z = EMLPair.from_values(3.0, -5.0)
        assert abs(z.real_tension - 3.0) < 1e-9
        assert abs(z.imag_tension - (-5.0)) < 1e-9

    def test_round_trip_zero(self):
        z = EMLPair.from_values(0.0, 0.0)
        assert abs(z.real_tension) < 1e-9
        assert abs(z.imag_tension) < 1e-9

    def test_round_trip_large_values(self):
        z = EMLPair.from_values(1e6, -1e6)
        assert abs(z.real_tension - 1e6) < 1.0
        assert abs(z.imag_tension - (-1e6)) < 1.0


class TestFrames:
    """frames() returns 4 EMLPairs; all have same modulus; successive 90° rotations."""

    def test_returns_four_frames(self):
        z = EMLPair.from_values(3.0, 4.0)
        fs = z.frames()
        assert len(fs) == 4

    def test_all_frames_same_modulus(self):
        z = EMLPair.from_values(3.0, 4.0)
        m0 = z.modulus
        for f in z.frames():
            assert abs(f.modulus - m0) < 1e-9

    def test_frame_0_is_original(self):
        z = EMLPair.from_values(3.0, 4.0)
        f0 = z.frames()[0]
        assert abs(f0.real_tension - z.real_tension) < 1e-9
        assert abs(f0.imag_tension - z.imag_tension) < 1e-9

    def test_frames_are_emlpair_instances(self):
        z = EMLPair.from_values(1.0, 0.0)
        for f in z.frames():
            assert isinstance(f, EMLPair)


class TestRotatePhase:
    """rotate_phase: 0 is identity, π negates, π/2 maps (1,0)→(0,1)."""

    def test_rotate_zero_is_identity(self):
        z = EMLPair.from_values(3.0, 4.0)
        rotated = z.rotate_phase(0.0)
        assert abs(rotated.real_tension - z.real_tension) < 1e-8
        assert abs(rotated.imag_tension - z.imag_tension) < 1e-8

    def test_rotate_pi_negates_both(self):
        z = EMLPair.from_values(3.0, 4.0)
        rotated = z.rotate_phase(math.pi)
        assert abs(rotated.real_tension - (-3.0)) < 1e-8
        assert abs(rotated.imag_tension - (-4.0)) < 1e-8

    def test_rotate_half_pi_maps_real_to_imag(self):
        z = EMLPair.from_values(1.0, 0.0)
        rotated = z.rotate_phase(math.pi / 2)
        assert abs(rotated.real_tension - 0.0) < 1e-8
        assert abs(rotated.imag_tension - 1.0) < 1e-8

    def test_rotate_full_cycle_is_identity(self):
        z = EMLPair.from_values(2.0, 3.0)
        rotated = z.rotate_phase(2 * math.pi)
        assert abs(rotated.real_tension - z.real_tension) < 1e-8
        assert abs(rotated.imag_tension - z.imag_tension) < 1e-8


class TestSchrodingerStep:
    """mirror_pulse on EMLPair: returns EMLPair with finite components."""

    def test_returns_emlpair(self):
        z = EMLPair.from_values(1.0, 1.0)
        result = z.mirror_pulse()
        assert isinstance(result, EMLPair)

    def test_result_is_finite_real(self):
        z = EMLPair.from_values(1.0, 1.0)
        result = z.mirror_pulse()
        assert math.isfinite(result.real_tension)

    def test_result_is_finite_imag(self):
        z = EMLPair.from_values(1.0, 1.0)
        result = z.mirror_pulse()
        assert math.isfinite(result.imag_tension)

    def test_two_pulses_finite(self):
        z = EMLPair.from_values(1.0, 1.0)
        result = z.mirror_pulse().mirror_pulse()
        assert math.isfinite(result.real_tension)
        assert math.isfinite(result.imag_tension)

    def test_returns_new_object(self):
        z = EMLPair.from_values(1.0, 1.0)
        result = z.mirror_pulse()
        assert result is not z

    def test_abs_returns_modulus(self):
        z = EMLPair.from_values(3.0, 4.0)
        assert abs(abs(z) - 5.0) < 1e-9

    def test_conjugate_imag_negated(self):
        z = EMLPair.from_values(2.0, 5.0)
        zc = z.conjugate()
        assert abs(zc.imag_tension - (-5.0)) < 1e-9

    def test_conjugate_real_unchanged(self):
        z = EMLPair.from_values(2.0, 5.0)
        zc = z.conjugate()
        assert abs(zc.real_tension - 2.0) < 1e-9
