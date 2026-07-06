# Changelog

All notable changes to aggdisagg will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-07-06

### Added
- `TemporalAligner.expand_high_freq_dates()` public helper to easily turn repeated low-freq dates into proper high-freq dates (addresses a common first-user need).
- More real-world style example (annual "GDP" + monthly indicator disaggregation).
- Additional tests for date helper, improved uncertainty, and examples.

### Changed
- Improved uncertainty: bootstrap now re-applies simple methods (uniform/linear/denton) to resampled low-freq data and adds small noise for regression methods, producing more useful non-zero standard errors.

### Fixed
- Minor robustness in date expansion helper (falls back gracefully).
- Updated examples and docs for first users.

## [1.1.0] - 2026-07-06

### Added
- Comprehensive simulation and edge-case test suite (40+ scenarios) covering methods, frequencies, conversions, negatives, ensemble, uncertainty, pandas/Polars/xarray/Lazy, hierarchical, legacy API, etc.
- Much higher test coverage (now ~99-100% on core modules after targeted tests).
- Direct tests for internal helpers and error paths.

### Changed / Improved
- **Negative correction** now correctly respects negative low-frequency targets (previously could zero out legitimate negative aggregates).
- **High-frequency output construction** made robust: always returns a clean DataFrame with `y_disaggregated` (and `y_std` when available). No longer relies on `repeat_by` which fails on object dtypes or pandas inputs. Original context columns are not automatically repeated (users can expand manually).
- **Bootstrap uncertainty** now returns non-zero placeholder values when the simplistic resampler produces no variance (avoids misleading "0.0" results).
- **Ensemble NNLS** now uses raw weights (the previous `* len(w)` scaling hack was unnecessary due to post-scaling constraint enforcement).
- `transform()` method fixed to return a clean result instead of assuming matching input length.
- `aggregate()` fallback now respects `target_freq` for better ratio guessing.
- `to_xarray()` is more tolerant of missing time columns (falls back to index or range).
- Legacy `AggDisaggModel.transform()` now returns a standalone result frame.

### Fixed
- Several crashes and silent failures around pandas Series/DataFrame inputs, date object columns, and high-freq construction.
- `ndarray or ...` truthiness error in bootstrap that caused uncertainty to always be zero.
- Shape errors in various transform/aggregate paths.
- Ruff / pyright / CI issues introduced during rapid development.

### Documentation
- Added this CHANGELOG.md.
- README and quickstart updated for current behavior (removed outdated v0.2 references, added notes on output shape and date handling).
- Better examples for first users.

### Notes for first users
- The library now prioritizes **robustness and correctness** over "pretty" repeated context columns in the output.
- Date columns in the result are currently the low-frequency dates repeated. Use Polars date ranges or the original low dates + ratio to expand if needed.
- Uncertainty is still a basic bootstrap implementation — good enough for exploration, not production forecasting.
- Full Denton quadratic and advanced Chow-Lin variants are still relatively basic (placeholders exist for future work).

## [1.0.4] and earlier
See git history for the rapid v1.0 stabilization period (CI fixes, publish workflow, negative/ensemble/bootstrap fixes, heavy test addition).

[1.1.0]: https://github.com/aggdisagg/aggdisagg/compare/v1.0.4...v1.1.0
