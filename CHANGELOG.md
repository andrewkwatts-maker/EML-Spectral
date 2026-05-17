# Changelog

---

## [2.0.1] — 2026-05-17

### Added

- **`eml-spectral-app` CLI command** — companion-app launcher installed as a console script
  alongside the library. On first run it locates or clones the
  [EML-Spectral-App](https://github.com/andrewkwatts-maker/EML-Spectral-App) KivyMD desktop/Android
  explorer at the matching version tag (`v2.0.1`) and launches it. Developer checkouts are
  detected automatically via sibling-directory search from `__file__`; end-user installs clone
  to `~/.eml-spectral-app`.

### Fixed

- **CI:** Removed stale `arithmos_core` path dependency from `rust/eml_spectral_core/Cargo.toml`
  which caused all PyPI CI builds to fail at cargo manifest load time.
- **Build backend:** Switched from `setuptools` to `maturin`; CI and publish workflow updated to
  use Rust toolchain + cibuildwheel, matching the eml-math pattern. The previous setuptools build
  was producing wheels with no compiled Rust extension.
- **pyproject.toml:** Removed UTF-8 BOM that caused `tomllib.TOMLDecodeError` on `pip install`.

---

## [2.0.0] — 2026-05-14

### Added

- **`spectral_flow.py` — Rust fast path.** `spectral_flow_n()` now dispatches to
  `eml_spectral_core::spectral_flow_batch()` for leaf-coordinate, non-discrete mode inputs.
  A `_is_rust_eligible` guard ensures the fast path is only taken when safe; a `steps==0`
  short-circuit preserves object identity required by the test suite.

### Changed

- **`eml_spectral_core` — PyO3 feature-gated.** `pyo3` is now an optional dependency behind
  a `python` Cargo feature (`default = []`). The engine workspace can now compile
  `eml_spectral_core` without a Python interpreter on the path. All public Rust functions remain
  available natively; PyO3 bindings activate only via maturin or `--features python`:
  - `Cargo.toml`: `pyo3` moved to optional under `python` feature
  - `lib.rs`, `octonion.rs`, `spectral.rs`, `lattice.rs`, `clifford.rs`: all PyO3 attributes
    wrapped in `#[cfg(feature = "python")]`

- Crate version bumped to `2.0.0` to align with the PyPI package version.

---

## [1.4.1] — 2026-05-12

### Changed

- Renamed `rust/eml_spectral_core/src/arithma_bridge.rs` → `arithmos_bridge.rs` to align with
  the upstream crate rename (`arithma_core` → `arithmos_core`).
- Updated `[features]` dep reference accordingly; `Cargo.lock` regenerated.

---

## [1.4.0] — 2026-05-10

### Added

- **`with-arithmos` Cargo feature** *(off by default)* — optional `arithmos_core` path dependency
  for engine consumers. Allows `eml_spectral_core` to carry an `ArithmosExpression` payload
  alongside the native `EMLMultivector` / `Octonion` / `EMLNDVector` types. Strictly absent from
  the PyPI dependency tree — `pip install eml-spectral` is unaffected.
- **`rust/eml_spectral_core/src/arithmos_bridge.rs`** — `ArithmosPayload` trait + converter
  skeleton. Planned targets: Schwarzschild Christoffel symbols as Arithmos sub-trees, FLRW
  `a(t)` parametrisation, octonion-amplitude carriers.

### Changed

- Crate version bumped `1.0.0` → `1.4.0` to match `pyproject.toml` for downstream version-parity
  tooling.

### Notes

- No functional regressions. `eml_spectral_core` already had `crate-type = ["cdylib", "lib"]`
  upstream — no `rlib` patch required (unlike eml-math).

---

## [1.3.0] — 2026-05-03

- Rust core (`eml_spectral_core`) + C API + `Get()` datasheet API, synchronised with eml-math
  1.3.0. CI green. PyPI publish workflow added (Trusted Publishing).
