# Changelog

All notable changes to aggdisagg will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.1] - 2026-07-07

### Fixed
- **Regression (include_dates)**: `expand_high_freq_dates` (and thus `disaggregate_columns(..., include_dates=True)`) now reliably expands to distinct high-frequency dates (e.g. 47 quarters → 141 unique monthly pl.Date). The pandas path is hardened and a pure-Python calendar stepper ensures the repeat-low fallback is never taken for dates. Updated docstring example (fit_transform does not emit a date column).
- **Missing public API**: `extrapolate` is now accepted on `fit_transform(..., extrapolate=...)` and `disaggregate_columns(..., extrapolate=...)` (forwarded with per-call override/restore). All four policies run and produce distinct length/value behavior.
- **Dangerous default (NaN inputs)**: Default `extrapolate="nan"` (was "hold"). NaN low-freq input values now produce honest NaN in the corresponding high-freq output months by default — the library no longer silently fabricates data for unreported periods. `extrapolate="hold"`/`"linear"` will fill when explicitly requested. `"drop"` shortens output by truncating after the last valid anchor. Warnings now specifically call out "NaN-input periods".
- Updated related tests, README, and docstrings. Existing behavior for fully-observed series is unchanged.

### Changed
- Default for `TemporalAligner(extrapolate=...)` is now `"nan"`.

## [1.4.0] - 2026-07-07

### Added
- `TemporalAligner.disaggregate_columns()` now supports per-column `stock` vs `flow` semantics via `col_semantics`, `default_semantics`, and `autodetect_semantics` (with heuristic and override). Exposes `_detected_semantics`.
- New `extrapolate` parameter on `TemporalAligner` ("hold" default, "linear", "nan", "drop") to control handling of NaN in low-freq input / end-of-range in disaggregation.
- `detect_semantics()` convenience on aligner.
- Excel extra in packaging (`aggdisagg[excel]`) for fastexcel/openpyxl.
- Regression tests for NaN handling, exact aggregation per method, stock/flow auto+override, multi-col roundtrip, date dtype, README examples.

### Changed
- `aggregate()` now supports multi-column high-freq frames (from `disaggregate_columns`); preserves original column names and recovers correctly using per-col semantics/agg when available.
- `include_dates=True` (and `expand_high_freq_dates`) now returns native `pl.Date` (was Object of python dates).
- Final per-group C scaling now robust to NaN in y_h (prevents pollution of valid groups).
- Default `extrapolate="hold"` ensures the final period is never silently dropped to NaN.
- README updated with prominent Python >=3.10 note and Excel extra guidance; added detailed quarterly-to-monthly multi-series example using the helper (now runs as test).
- Version bumped to 1.4.0 (breaking behavior changes for NaN policy, aggregate multi-col, date dtype in some paths; documented).

### Fixed
- Silent NaN tail in disagg when low-freq input had NaN or at end-of-range (now warns + holds by default).
- False exact guarantee claim for "linear" + sum: now the post-scaling always enforces (for valid groups); docs updated to scope the claim appropriately.
- README roundtrip example now runs without crash (aggregate works on named multi-col output).
- Date column from include_dates is now pl.Date.
- Onboarding notes for Python version and Excel deps.

## [1.3.0] - 2026-07-07

### Added
- `disaggregate_columns()` helper for multi-target DataFrames (all columns as targets).
- Date-aware ratio inference (Q→M now correctly 3).

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
