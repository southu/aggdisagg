# Robust Test Plan for aggdisagg - 100 Scenarios

## Goal
- Achieve near-100% coverage of core logic, edge cases, APIs, input types.
- Stress test for correctness (aggregation constraint always held), numerical stability, error handling.
- Validate new features: date expansion helper, improved uncertainty.
- Simulate real usage patterns for first users (econometrics, indicators, negatives, mixed freqs).
- Run in real pytest for ~30 minutes to catch intermittent/numerical issues.
- Find bugs, fix in src/, re-test, push.

## Test Strategy
- Use a single parametrized or looped test: `test_robust_100_scenarios()`
- Generate 100 distinct scenarios using combinations + random seeds.
- Categories covered (aim for diversity):
  1-20: Basic methods + conversions + sizes (uniform/linear/denton with sum/mean/first/last, n_low=5 to 200)
  21-40: Advanced methods + indicators (chow-lin-opt, litterman, fernandez with/without inds)
  41-60: Edge data (zeros, all-neg, mixed signs, small/large vals, n_low=1, n_low=2)
  61-70: Ensemble + corrections (use_ensemble=True, correct_negatives=True on neg data)
  71-80: Input type variations (Polars DF, Pandas DF, Pandas Series+DTI, LazyFrame, xarray)
  81-90: Freq + date handling (1mo, 1q, daily-ish; test expand_high_freq_dates)
  91-95: Uncertainty & bootstrap (n_bootstrap=0/20/100; check non-zero for simple methods)
  96-98: Legacy API + hierarchical (AggDisaggModel, reconcile)
  99-100: Error paths & stress (bad inputs, large n=500, many bootstraps)

- For each scenario:
  - Generate deterministic synthetic low-freq data (trend + noise + occasional negs)
  - Create df in target format
  - Instantiate TemporalAligner with params
  - fit_transform, assert:
    - No NaN/Inf
    - Correct length = n_low * ratio
    - If _C available: np.allclose( C @ y_high , y_low , atol=1e-8 )
    - aggregate() recovers low within tol
    - predict_with_uncertainty(): shapes match, std >=0
  - Call expand_high_freq_dates, assert len correct, monotonic
  - Call summary(), rho_ etc.
  - For 20% of cases: roundtrip check with legacy dis/agg
  - Occasionally: to_xarray/from_xarray if available
  - Time the operation, log if >5s

- Assertions:
  - Strict constraint for methods that promise it (uniform, linear, denton, chow-lin etc.)
  - For ensemble: still exact after correction
  - Uncertainty: for uniform/linear after improvement, std >0 on varied data
  - No exceptions on valid inputs

- To reach ~30 min:
  - Use n_low up to 500 for 10 cases (chow-lin matrix ops + bootstrap=200 slow)
  - 20 cases with bootstrap=100+
  - Include xarray conversions (pandas interop)
  - Loop some scenarios 3x with diff seeds
  - Total ~100 core scenarios + overhead

- Execution:
  - pytest tests/test_simulation.py::test_robust_100_scenarios -s --durations=10
  - Capture failures, fix in core.py / methods.py
  - Re-run subset or full
  - At end, full pytest + coverage

- Post-run:
  - Update test plan with findings
  - Commit fixes + new tests
  - Push

## Implementation Notes
- Use itertools.product + random.sample for 100 unique combos
- Seeds = range(100) + offset for reproducibility
- Helper funcs: make_synthetic_data(n, seed, neg_prob=0.1)
- Compute ratio from target_freq string
- Skip xarray/sk time if not installed (pytest.importorskip)
- For legacy: use AggDisaggModel where applicable
- Track passed/failed in dict, assert all 100 passed at end

This plan ensures real execution, diversity, and will surface issues in date handling, uncertainty, negatives, large data, pandas paths, etc.