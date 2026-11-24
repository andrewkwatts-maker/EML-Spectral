"""Rust core against the pure-Python fallback: one answer per call.

Every product and Christoffel symbol is computed twice on shared, seeded
inputs, once with the Rust flag on and once with it forced off, and the
relative difference is gated by ``PARITY_REL_GATE``.
"""
import itertools
import random

import pytest
from eml_math.point import EMLPoint

from eml_spectral import geometric_algebra as ga_mod
from eml_spectral import metric as metric_mod
from eml_spectral import octonion as oct_mod
from eml_spectral.geometric_algebra import EMLMultivector
from eml_spectral.metric import MetricTensor
from eml_spectral.octonion import Octonion

PARITY_REL_GATE = 1e-15
SEED = 20261124
SIGNATURES_PER_N = 20
GA_MAX_N = 10
OCTONION_PAIRS = 200
RANDOM_POINTS = 600
EDGE_X = (-3.0, -1.0, 0.0, 0.5, 0.6931471805599453, 1.0, 3.0, 50.0, 700.0,
          709.0, 709.78, 709.7813, 800.0)
EDGE_RS = (0.5, 1.0, 2.0)
TRIPLES = tuple(itertools.product((0, 1), repeat=3))


def _rel(value: float, ref: float) -> float:
    """Difference relative to max(|ref|, 1)."""
    return abs(value - ref) / max(abs(ref), 1.0)


def _points(rng: random.Random, count: int) -> list:
    return [EMLPoint(rng.uniform(-2.0, 2.0), 1.0) for _ in range(count)]


def _worst_product(mod, flag, pairs) -> float:
    """Worst relative error of ``a * b`` between Rust and fallback."""
    worst = 0.0
    for a, b in pairs:
        rust = a * b
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mod, flag, False)
            ref = a * b
        for got, want in zip(rust._scalars(), ref._scalars()):
            worst = max(worst, _rel(got, want))
    return worst


class _CoreSpy:
    """Delegates to the real core and counts calls per function name."""

    def __init__(self, real) -> None:
        self._real = real
        self.calls = {}

    def __getattr__(self, name):
        target = getattr(self._real, name)
        if not callable(target):
            return target

        def counted(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            return target(*args, **kwargs)

        return counted


def test_rust_core_is_loaded():
    assert ga_mod._RUST_GA is True
    assert oct_mod._RUST_OCTONION is True
    assert metric_mod._RUST_METRIC is True


def test_geometric_product_parity_n1_to_n10():
    rng = random.Random(SEED)
    worst = 0.0
    for n in range(1, GA_MAX_N + 1):
        pairs = []
        for _ in range(SIGNATURES_PER_N if n <= 8 else 2):
            sig = tuple(rng.choice((-1, 1)) for _ in range(n))
            dim = 1 << n
            pairs.append((EMLMultivector(_points(rng, dim), sig),
                          EMLMultivector(_points(rng, dim), sig)))
        worst = max(worst, _worst_product(ga_mod, "_RUST_GA", pairs))
    print(f"geometric product n=1..{GA_MAX_N} worst rel error: {worst:.3e}")
    assert worst <= PARITY_REL_GATE


def test_octonion_product_parity():
    rng = random.Random(SEED)
    pairs = [(Octonion(_points(rng, 8)), Octonion(_points(rng, 8)))
             for _ in range(OCTONION_PAIRS)]
    worst = _worst_product(oct_mod, "_RUST_OCTONION", pairs)
    print(f"octonion product worst rel error: {worst:.3e}")
    assert worst <= PARITY_REL_GATE


def _christoffel_cases() -> list:
    rng = random.Random(SEED)
    cases = [(rng.uniform(0.75, 6.0), rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0))
             for _ in range(RANDOM_POINTS)]
    cases += [(x, 1.0, rs) for x in EDGE_X for rs in EDGE_RS]
    return cases


def test_schwarzschild_christoffel_parity():
    worst = 0.0
    for x, y, rs in _christoffel_cases():
        metric = MetricTensor.schwarzschild(rs)
        p = EMLPoint(x, y)
        for lam, mu, nu in TRIPLES:
            rust = metric.christoffel(lam, mu, nu, p)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(metric_mod, "_RUST_METRIC", False)
                ref = metric.christoffel(lam, mu, nu, p)
            worst = max(worst, _rel(rust, ref))
    print(f"schwarzschild christoffel worst rel error: {worst:.3e}")
    assert worst <= PARITY_REL_GATE


def test_rust_path_is_called(monkeypatch):
    rng = random.Random(SEED)
    sig = (1, -1, -1)
    mv = EMLMultivector(_points(rng, 8), sig)
    octo = Octonion(_points(rng, 8))
    metric = MetricTensor.schwarzschild(2.0)
    point = EMLPoint(1.5, 1.0)
    cases = (
        (ga_mod, "_RUST_GA", "geometric_product_n", lambda: mv * mv),
        (oct_mod, "_RUST_OCTONION", "octonion_mul_n", lambda: octo * octo),
        (metric_mod, "_RUST_METRIC", "christoffel_batch_n",
         lambda: metric.christoffel(0, 0, 1, point)),
    )
    for mod, flag, name, call in cases:
        spy = _CoreSpy(mod._core)
        monkeypatch.setattr(mod, "_core", spy)
        monkeypatch.setattr(mod, flag, True)
        call()
        assert spy.calls.get(name, 0) == 1
        monkeypatch.setattr(mod, flag, False)
        call()
        assert spy.calls.get(name, 0) == 1


def test_generic_metric_keeps_finite_difference():
    flat = MetricTensor.flat()
    point = EMLPoint(1.5, 1.0)
    for lam, mu, nu in TRIPLES:
        assert flat.christoffel(lam, mu, nu, point) == 0.0
