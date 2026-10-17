General engineering standards: ~/.claude/docs/standards.md.

# EML-Spectral

## Versioning — 4 locations must stay in sync

When bumping the package version, **all four locations must be updated together** or the PyPI release will be broken:

| File | Field |
|------|-------|
| `pyproject.toml` | `version = "X.Y.Z"` under `[project]` |
| `rust/eml_spectral_core/Cargo.toml` | `version = "X.Y.Z"` under `[package]` |
| `src/eml_spectral/__init__.py` | `__version__ = "X.Y.Z"` |
| Git tag | `vX.Y.Z` pushed to origin |

After committing and tagging, create a **GitHub Release** from the tag to trigger the PyPI publish workflow (`release: published` event). The workflow does **not** fire on a tag push alone.

> **Important:** `pyproject.toml version` is what PyPI indexes — not the git tag. Pre-release suffixes like `2.0.0a0` are **not** matched by `>=2.0.0` in pip. Always use clean `X.Y.Z` for stable releases.
