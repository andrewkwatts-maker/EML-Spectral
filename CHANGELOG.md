# Changelog

## [1.4.0] — 2026-05-10

EML / Arithma / metaphysica / periodica synchronised v1.4.0 cut. Adds an
optional Arithma symbolic-substrate bridge for engine consumers; the
PyPI install path is unchanged.

### Added

- **`with-arithma` Cargo feature** (off by default) — pulls in
  `arithma_core` as an optional dependency via a `git-submodule`-only
  path so engine consumers can carry an `ArithmaExpression` payload
  alongside the native `EMLMultivector` / `Octonion` / `EMLNDVector`
  types. Strictly absent from the PyPI dep tree —
  `pip install eml-spectral` is unaffected.
- **`src/arithma_bridge.rs`** — converters + `ArithmaPayload` trait
  contract mirroring eml-math's pattern. Skeleton only for v1.4.0; the
  high-value targets (Schwarzschild Christoffel symbols carried as
  Arithma sub-trees, FLRW `a(t)` parametrisation, octonion-amplitude
  carriers) populate as the Arithma surface stabilises.

### Changed

- Crate version bumped from `1.0.0` → `1.4.0` so it matches the
  `pyproject.toml` PyPI version line for downstream sanity-check
  tooling that compares both.

### Notes

- No functional regressions; existing PyO3 attributes are untouched.
- `eml_spectral_core` already had `crate-type = ["cdylib", "lib"]`
  upstream — no `rlib` patch needed unlike eml-math.
