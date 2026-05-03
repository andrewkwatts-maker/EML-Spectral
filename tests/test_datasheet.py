"""Tests for the eml-spectral datasheet Get() API."""
from __future__ import annotations
import math
import json
import pytest

from eml_spectral import Get, list_constants, SPECTRAL_CATALOGUE


# ── Catalogue coverage ──────────────────────────────────────────────────────

class TestCatalogue:

    def test_catalogue_nonempty(self):
        assert len(SPECTRAL_CATALOGUE) >= 25

    def test_list_constants_includes_spectral(self):
        names = list_constants()
        for required in ("e8_dim", "alpha_leak", "lambda_eff",
                          "topology_invariant", "leech_kissing"):
            assert required in names

    def test_list_constants_includes_math(self):
        # Delegated from eml-math
        names = list_constants()
        for required in ("pi", "e", "phi", "sqrt2"):
            assert required in names


# ── Algebra dimensions ──────────────────────────────────────────────────────

class TestAlgebraDimensions:

    @pytest.mark.parametrize("name,expected", [
        ("g2_dim",         14),
        ("f4_dim",         52),
        ("e6_dim",         78),
        ("e7_dim",         133),
        ("e8_dim",         248),
        ("e7_56",          56),
        ("j3o_dim",        27),
        ("octonion_dim",   8),
        ("quaternion_dim", 4),
        ("spacetime_dim",  4),
        ("leech_dim",      24),
    ])
    def test_dimension_value(self, name, expected):
        d = Get(name)
        assert d["value"] == expected
        assert d["kind"] == "dimension"

    def test_case_insensitive(self):
        a = Get("E8_dim")
        b = Get("e8_dim")
        c = Get("E8_DIM")
        assert a["value"] == b["value"] == c["value"] == 248


# ── Lattice constants ───────────────────────────────────────────────────────

class TestLatticeConstants:

    def test_e8_min_norm(self):
        d = Get("e8_min_norm")
        assert abs(d["value"] - math.sqrt(2)) < 1e-12
        assert d["kind"] == "lattice"

    def test_e8_min_norm_squared(self):
        assert Get("e8_min_norm_sq")["value"] == 2.0

    def test_e8_kissing_240(self):
        assert Get("e8_kissing")["value"] == 240

    def test_leech_min_norm(self):
        assert Get("leech_min_norm")["value"] == 2.0

    def test_leech_min_norm_squared(self):
        assert Get("leech_min_norm_sq")["value"] == 4.0

    def test_leech_kissing_196560(self):
        assert Get("leech_kissing")["value"] == 196560

    def test_d4_kissing_24(self):
        assert Get("d4_kissing")["value"] == 24


# ── Topology invariants ─────────────────────────────────────────────────────

class TestTopology:

    def test_b3_24(self):
        assert Get("b3")["value"] == 24

    def test_chi_eff_144(self):
        assert Get("chi_eff")["value"] == 144

    def test_topology_invariant_144(self):
        # (b3 / 24) · chi_eff = 1 · 144 = 144
        d = Get("topology_invariant")
        assert abs(d["value"] - 144.0) < 1e-9


# ── Heterotic / racetrack ───────────────────────────────────────────────────

class TestHeterotic:

    def test_alpha_leak_value(self):
        d = Get("alpha_leak")
        assert abs(d["value"] - 1.0 / math.sqrt(6.0)) < 1e-12

    def test_portal_coupling_alias(self):
        a = Get("alpha_leak")
        b = Get("portal_coupling")
        assert abs(a["value"] - b["value"]) < 1e-15

    def test_lambda_eff_formula(self):
        d = Get("lambda_eff")
        assert abs(d["value"] - math.exp(-2 * math.pi / 24)) < 1e-12
        assert "exp(-2π/24)" in d["formula"]

    def test_cabibbo_proxy_is_lambda_cubed(self):
        lam = Get("lambda_eff")["value"]
        cabb = Get("cabibbo_proxy")["value"]
        assert abs(cabb - lam ** 3) < 1e-12

    def test_n1_24(self):
        assert Get("n1_flux")["value"] == 24

    def test_n2_23(self):
        assert Get("n2_flux")["value"] == 23


# ── Spectral / G₂ seeds ─────────────────────────────────────────────────────

class TestSpectralSeeds:

    def test_edof(self):
        assert Get("edof")["value"] == 3

    def test_g2_seed_T_re_finite(self):
        assert math.isfinite(Get("g2_seed_T_re")["value"])

    def test_g2_seed_lambda_finite(self):
        assert math.isfinite(Get("g2_seed_lambda")["value"])


# ── Delegation to eml-math ──────────────────────────────────────────────────

class TestDelegation:

    @pytest.mark.parametrize("name,expected", [
        ("pi",     math.pi),
        ("e",      math.e),
        ("phi",    (1 + math.sqrt(5)) / 2),
        ("sqrt2",  math.sqrt(2)),
        ("tau",    2 * math.pi),
    ])
    def test_math_constants_passthrough(self, name, expected):
        d = Get(name)
        assert abs(d["value"] - expected) < 1e-9
        assert d["source"] == "eml-math"

    def test_spectral_marked_as_eml_spectral(self):
        d = Get("e8_dim")
        assert d["source"] == "eml-spectral"


# ── Return shape & JSON serialisation ───────────────────────────────────────

class TestShape:

    @pytest.mark.parametrize("name", [
        "e8_dim", "alpha_leak", "lambda_eff", "leech_kissing",
        "topology_invariant", "edof",
    ])
    def test_required_keys(self, name):
        d = Get(name)
        for key in ("name", "value", "formula", "kind", "description", "source"):
            assert key in d, f"missing key {key!r} in Get({name!r})"

    def test_json_round_trip(self):
        d = Get("e8_dim")
        encoded = json.dumps(d)
        restored = json.loads(encoded)
        assert restored == d

    def test_as_json_returns_string(self):
        s = Get("e8_dim", as_json=True)
        assert isinstance(s, str)
        d = json.loads(s)
        assert d["value"] == 248


# ── Unknown names ───────────────────────────────────────────────────────────

class TestUnknown:

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            Get("totally-unknown-constant-zzz")

    def test_known_in_neither_passthrough_raises(self):
        with pytest.raises(KeyError):
            Get("xyzqwerty")
