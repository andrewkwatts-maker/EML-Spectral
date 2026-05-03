# EML-Spectral

**v1.3.0** · The physics-and-algebra companion to
[`eml-math`](https://github.com/andrewkwatts-maker/EML-Math). Pure Python core,
zero required dependencies, optional Rust acceleration and C API.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

EML-tree representations of:

- **Clifford algebras** Cl(p, q) with the geometric product, rotors, and the sandwich product
- **Octonions** — 8-component non-associative normed division algebra (Fano-plane multiplication)
- **Exceptional algebras** — `FreudenthalTripleSystem` (J₃(𝕆), 27D), `E7_56` (56D rep), `E8_248` (adjoint), `E8xE8` (heterotic)
- **Lorentz-invariant operations** on EMLPoints (functional `spacetime` namespace)
- **Named GR metrics** — `MetricTensor.flat / schwarzschild / flrw / ads5_x_s5 / calabi_yau_3 / klebanov_strassler / heterotic_e8x8 / g2_holonomy`
- **Lattices** — E₈ root lattice, Leech lattice, Planck-scale discrete helpers
- **Spectral flow** — the discrete Φ operator, racetrack fixed points, the (b₃/24)·χ_eff topological invariant

---

## Install

```bash
pip install eml-spectral
```

This pulls in `eml-math >= 1.3.0` automatically. **Zero other required deps.**

Optional extras:

```bash
pip install eml-spectral[ext]        # + numpy, sympy
pip install eml-spectral[precision]  # + mpmath
pip install eml-spectral[rust]       # + maturin (to build the Rust extension)
pip install eml-spectral[dev]        # + pytest, ruff, mypy, maturin
```

---

## Quickstart

```python
from eml_math import EMLPoint
from eml_spectral import (
    EMLPair, EMLMultivector, Octonion, basis_octonion,
    FreudenthalTripleSystem, E7_56, E8_248, E8xE8,
    MetricTensor, EMLNDVector, MinkowskiFourVector, FourMomentum,
    spectral_flow, racetrack_fixed_point, topology_invariant,
)
from eml_spectral import spacetime

# Lorentz-invariant Minkowski interval (preserved under boost)
p = EMLPoint(1.0, 2.718)
print(spacetime.minkowski_delta(p))
p2 = spacetime.boost(p, 0.693)
print(abs(spacetime.minkowski_delta(p) - spacetime.minkowski_delta(p2)))   # ≈ 0

# Discrete spectral flow Φ — racetrack fixed points
traj = spectral_flow(EMLPoint(1.0, 1.0), steps=10)
print(traj[-1].x, traj[-1].y)

# Schwarzschild metric — analytic Christoffel symbols
g = MetricTensor.schwarzschild(rs=2.0)
print(g.is_curved(EMLPoint(3.0, 1.0)))                                    # True

# Octonion multiplication via the Fano plane
e1, e2, e4 = basis_octonion(1), basis_octonion(2), basis_octonion(4)
print((e1 * e2).norm())                                                   # 1.0  (= e4)

# Clifford algebra rotor — preserves quadratic form
mv = EMLMultivector([EMLPoint(1, 1)] * 4, signature=(1, -1))
R = mv.rotor(angle=0.5, plane=(0, 1))
print(abs(mv.rotate(R).quadratic() - mv.quadratic()))                      # ≈ 0

# E₈ × E₈ heterotic gauge group — analytic Cabibbo proxy
ee = E8xE8()
print(ee.portal_coupling())                                                # 1/√6
print(ee.racetrack_potential()["lambda_eff"])                              # exp(−2π/24)
```

---

## What's new in v1.3.0

| Layer | What's added |
|-------|------------|
| **Datasheet `Get()` API** | Uniform with `eml_math.Get()` / `metaphysica.Get()` / future `periodica.Get()`. `eml_spectral.Get('E8_dim')` returns a JSON-serialisable dict for any spectral or math constant. **167 total constants** reachable via one entry point (31 spectral + 136 math, delegated). |
| **Rust acceleration** | `eml_spectral_core` PyO3+Rayon module: `octonion_mul_n`, `octonion_norm_n`, `spectral_flow_n`, `spectral_flow_batch`, `geometric_product_n`, `e8_norms_squared_n`. Pure-Python fallback always present. |
| **C / C++ API** | `c_api/eml_spectral.h` plus `libeml_spectral.{dll,so,a}` — 14 exported functions covering spectral flow, octonions, Lorentz boosts, Schwarzschild Christoffels, Clifford geometric product, lattice norms. Zero deps. |
| **Documentation** | Static-HTML site under `docs/` — index, API reference, user guide, mathematical concepts. Mirrors `eml-math`'s docs style. |
| **Tests** | 1166 tests (was 873) — full coverage of `exceptional/`, spectral flow, functional spacetime, axiom invariants, frame-shift safety, datasheet API. |
| **Polish** | Versions kept in sync with `eml-math` v1.3.0; `_HAS_RUST` flag exported; pyproject classified as `Production/Stable`. |

### Datasheet API

```python
from eml_spectral import Get, list_constants

Get('E8_dim')          # {value: 248, kind: 'dimension', source: 'eml-spectral', ...}
Get('alpha_leak')      # {value: 1/√6, formula: '1/√6', kind: 'heterotic', ...}
Get('lambda_eff')      # racetrack-stabilised modulus exp(-2π/24)
Get('leech_kissing')   # 196560 — kissing number of the Leech lattice
Get('topology_invariant')  # 144 — the conserved Φ-flow Noether charge

Get('pi')              # delegated to eml-math: math.pi with EML derivation

list_constants()       # 167 total names (spectral + math, deduplicated)
```

Categories surfaced via the spectral catalogue:

* **Algebra dimensions**: `g2_dim`, `f4_dim`, `e6_dim`, `e7_dim`, `e8_dim`,
  `e7_56`, `j3o_dim`, `octonion_dim`, `quaternion_dim`, `spacetime_dim`,
  `leech_dim`
* **Lattice constants**: `e8_min_norm`, `e8_min_norm_sq`, `e8_kissing` (240),
  `leech_min_norm`, `leech_min_norm_sq`, `leech_kissing` (196560),
  `d4_kissing` (24)
* **Topology**: `b3` (24), `chi_eff` (144), `topology_invariant`
* **Heterotic / racetrack**: `alpha_leak` (= `portal_coupling`),
  `lambda_eff`, `cabibbo_proxy`, `n1_flux`, `n2_flux`
* **Spectral / G₂ seeds**: `edof`, `g2_seed_t_re`, `g2_seed_lambda`
* **Discrete / Planck**: `planck_d`

---

## Optional Rust acceleration

When the wheel ships with the prebuilt extension (or you build it yourself with
`maturin develop --release`), batch operations dispatch to Rayon-parallel Rust:

```python
from eml_spectral import _HAS_RUST, eml_spectral_core as core

print("Rust available:", _HAS_RUST)
print(core.spectral_flow_step(1.0, 1.0))                          # → (1.0, e)
print(core.octonion_norm_n([[1, 0, 0, 0, 0, 0, 0, 0]] * 1000))     # parallel batch
print(core.e8_min_norm_squared())                                  # 2.0
```

To build from source:

```bash
$env:PYO3_PYTHON = "C:\path\to\python.exe"   # Windows
cargo build --release -p eml_spectral_core
# the .pyd / .so lands in target/release; copy into src/eml_spectral/
```

Listed via `dir(eml_spectral_core)`:

| Function | Purpose |
|---|---|
| `octonion_mul`, `octonion_mul_n`, `octonion_norm_n` | Fano-plane multiplication, batched |
| `spectral_flow_step`, `spectral_flow_n`, `spectral_flow_batch` | Φ iteration, batched starts |
| `geometric_product_n` | Clifford geometric product for Cl(p, q), Rayon-parallel |
| `e8_norms_squared_n`, `e8_min_norm_squared`, `leech_min_norm_squared` | Lattice norms |
| `add_n` | Element-wise addition (sanity check / template) |

---

## Optional C API

A standalone C shared library — no Python needed — lives under `c_api/`. Build:

```bash
cargo build --release -p eml_spectral_c_api
```

Outputs (in `target/release/`):

- `eml_spectral.dll` (Windows) · `libeml_spectral.so` (Linux) · `libeml_spectral.a` (static)
- `c_api/eml_spectral.h` — full header with extern "C" guards

Exports 14 `extern "C"` functions:

| Topic | Functions |
|---|---|
| **Spectral flow** | `els_spectral_flow_step`, `els_spectral_flow`, `els_spectral_flow_batch` |
| **Octonion** | `els_octonion_mul`, `els_octonion_norm`, `els_octonion_mul_batch` |
| **Lorentz** | `els_minkowski_delta`, `els_rapidity`, `els_boost`, `els_boost_batch` |
| **Schwarzschild** | `els_schwarzschild_christoffel` |
| **Clifford** | `els_geometric_product` (any Cl(p, q)) |
| **Lattice** | `els_e8_norm_squared`, `els_leech_norm_squared` |

Minimal C example (`c_api/example.c`):

```c
#include <stdio.h>
#include "eml_spectral.h"

int main(void) {
    double x, y;
    els_spectral_flow_step(1.0, 1.0, &x, &y);
    printf("Phi(1, 1) = (%g, %g)\n", x, y);   /* (1.0, 2.71828) */
    return 0;
}
```

Build the example (Linux/macOS):

```bash
gcc example.c -L./target/release -leml_spectral -lm -o example
```

---

## Documentation

`docs/` contains a static-HTML site (no build step, open files directly):

- `docs/index.html` — landing page with quickstart
- `docs/api.html` — full API reference (every public name in `__all__`)
- `docs/guide.html` — 11-section user guide
- `docs/concepts.html` — mathematical / physics primer
- `docs/style.css` — shared with `eml-math` for visual parity

---

## API surface

Major exports from `eml_spectral`:

```python
# Algebras
EMLPair, EMLMultivector, Octonion, basis_octonion,
FreudenthalTripleSystem, E7_56, E8_248, E8xE8

# Spacetime  (functional namespace + class wrappers)
spacetime,                         # spacetime.minkowski_delta(p), boost(p, phi), …
MinkowskiFourVector, FourMomentum

# Spectral dynamics
spectral_flow, racetrack_fixed_point, topology_invariant, G2_SEEDS,
EMLState, simulate_pulses, simulate_flips, quantized_trajectory,
verify_conservation, frame_shift_count, find_resonance_bands

# Metrics  (factories: flat / schwarzschild / flrw / ads5_x_s5 / calabi_yau_3 /
#           klebanov_strassler / heterotic_e8x8 / g2_holonomy)
MetricTensor

# Lattices
EMLNDVector, e8_lattice_points, leech_lattice_points,
planck_delta, lattice_distance, is_lattice_neighbor

# Optional Rust flag
_HAS_RUST                          # True when eml_spectral_core is importable
```

---

## Why a separate package?

`eml-math` is the universal-math toolkit (operator, expression trees, symbolic
regression, flow rendering). It stays small and physics-agnostic.

`eml-spectral` houses every EML-tree construction that's domain-specific to
physics or exceptional algebra. Splitting them lets casual users of `eml-math`
install the core without pulling in the algebra/physics weight, while
researchers add `pip install eml-spectral` for the full toolkit.

---

## Project layout

```
eml-spectral/
├─ src/eml_spectral/          # Python sources (pure stdlib core)
│  ├─ pair.py                 # EMLPair (two-real complex)
│  ├─ geometric_algebra.py    # EMLMultivector — Clifford Cl(p, q)
│  ├─ octonion.py             # Octonion — Fano-plane product
│  ├─ exceptional/            # FreudenthalTripleSystem, E7_56, E8_248, E8xE8
│  ├─ metric.py               # MetricTensor + 8 factories
│  ├─ ndim.py                 # EMLNDVector + E₈ / Leech lattice generators
│  ├─ discrete.py             # Planck-scale lattice helpers
│  ├─ spacetime.py            # Functional Lorentz operations
│  ├─ momentum.py             # FourMomentum
│  ├─ fourvector.py           # MinkowskiFourVector
│  ├─ state.py                # EMLState
│  ├─ simulation.py           # simulate_pulses, verify_conservation, …
│  └─ spectral_flow.py        # Φ operator + topology_invariant + racetrack
├─ rust/eml_spectral_core/    # Optional PyO3+Rayon Rust extension
├─ c_api/                     # C/C++/Rust shared library
├─ docs/                      # Static-HTML documentation
├─ tests/                     # 1116 pytest tests
└─ examples/
```

---

## License

MIT — © Andrew K Watts. Based on the EML Sheffer operator established by
Andrzej Odrzywolek ([arXiv:2603.21852v2](https://arxiv.org/html/2603.21852v2),
CC BY 4.0).
