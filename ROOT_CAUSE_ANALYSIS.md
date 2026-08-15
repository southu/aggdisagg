# Root-cause analysis — aggdisagg disaggregation pipeline

**Mission:** `aggdisagg-root-cause-review` (iteration 1), documentation only.
**Scope:** `src/aggdisagg/core.py` (`_compute_high_lengths`, `_apply_denton`,
`_prepare_data`, `fit_transform`, `disaggregate_columns`) and the implicated
method classes in `src/aggdisagg/methods.py`.
**Prioritized risk areas (from `SESSION_HANDOFF_2026-07.md`):** fiscal / offset-quarter
anchor handling, and Denton-Cholette boundary accuracy.
**Code under review:** current `main` (library version 1.11.0). No function
signatures or behavior in `core.py` / `methods.py` were changed for this pass.

This note reconstructs the candidate failure set from (a) the risk areas in
`SESSION_HANDOFF_2026-07.md`, (b) currently failing or skipped tests in this
checkout, and (c) a search for open GitHub issues on
`https://github.com/southu/aggdisagg`. For each candidate: **PASS** or
**CONFIRMED-BUG**, the implicated file / function / line(s), why it fails or
does not, and any known limitation stated as-is.

---

## How the candidate set was reconstructed

### (a) `SESSION_HANDOFF_2026-07.md` flagged risk areas

1. **Issue 1 — fiscal / offset-quarter anchors.** `_compute_high_lengths` used
   to call `pd.to_period(low_f)`, which snaps to calendar quarters and produced
   (usually) 1 child per period for Sep-start / Jul-start / other fiscal labels
   → silent pass-through or wrong expansion. The 1.11 work inferred `step_months`
   from label spacing and added an explicit error when `n_high == n_low`.
2. **Issue 2 — Denton-Cholette boundary.** Endpoints extended the trend instead
   of damping the transient like R `tempdisagg`. The 1.11 change uses first-difference
   Q plus a preliminary `p` placed at ~0.2 into each block. Toy series improved
   to `<0.5%`; Kraft / B&G still miss the strict `<1%` / `<1.02` movement-distortion
   target. Special-case overrides that forced exact reference outputs were removed.

### (b) Currently failing or skipped tests

Captured in `TEST_REPORT.md` (`uv run pytest --cov=aggdisagg`: 61 failed, 59
passed, 2 skipped). Grouped here as candidates, not as 61 independent library
bugs:

| Group | Tests | Symptom |
|---|---|---|
| B1 | `test_issue1_*`, `test_issue2_*` (in-repo fixtures) | **Passed.** Fiscal expansion and Denton-Cholette tolerances hold. |
| B2 | 18 `test_basic.py` cases + 2 skips | Hard-coded `/Users/dev/Documents/GitHub/...` paths missing in this checkout. |
| B3 | `test_simulation.py::test_messy_incomplete_data_batch_{3,6–43}` (39 tests) | `ValueError: lengths length must match n_low` from `_build_c_matrix`. First reproduced failure: batch 3 case 13 (`object_mixed` + `nat_date`). |
| B4 | `test_messy_incomplete_data_batch_{46,47}` | `ValueError: Disaggregation did not expand the series (output length 4 == input 4)`. First reproduced: batch 46 case 13 (`low_f="Q"`, `tf="1q"`); batch 47 case 19 (same Q→Q). |
| B5 | `test_improved_uncertainty` | `predict_with_uncertainty()` returns an all-zero std of length 48. |
| B6 | `test_robust_100_scenarios` | Round-trip `np.allclose` fails because `back.to_numpy().ravel()` mixes the date ordinal with values (`17897` vs generated `y`). |

The fixture × method messy-loop matrix in `TEST_REPORT.md` (Kraft / PepsiCo /
B&G × 8 methods × 5 repeats) was **clean**: no exceptions, exact sum constraint,
deterministic.

### (c) Open issues

`gh issue list --repo southu/aggdisagg` returned no issues (empty / no visible
open items referencing these functions). No additional candidates from (c).

---

## Function-level implication (acceptance checklist)

Each of the five `core.py` functions, plus the `methods.py` classes on the
reviewed path, states whether it is implicated in a **confirmed** failure.

| Function | File | Implicated in a confirmed failure? |
|---|---|---|
| `_compute_high_lengths` | `src/aggdisagg/core.py` | **YES** — NaT/`dropna` shortens the lengths vector (C3). Fiscal/offset-quarter handling itself is **not** a current failure (C1 PASS). |
| `_apply_denton` | `src/aggdisagg/core.py` | **YES** — preliminary `p` ignores variable `_high_lengths` (C10). The SESSION_HANDOFF Issue 2 R-parity gap on volatile series is **not** a confirmed bug (C2: no bug found; known limitation). |
| `_prepare_data` | `src/aggdisagg/core.py` | **YES** — forwards mismatched lengths into `_build_c_matrix` (C3) and raises on any all-ones expansion, including legitimate same-frequency requests; advertised `source_freq` is never read (C4 / C8). |
| `fit_transform` | `src/aggdisagg/core.py` | **YES** — public caller of the C3/C4 path; also fails to populate `_std_errors` unless `with_uncertainty=True`, so a later `predict_with_uncertainty()` returns zeros (C5). |
| `disaggregate_columns` | `src/aggdisagg/core.py` | **YES** — calls `fit_transform` (lines 1564 and 1580) and therefore inherits C3/C4. No independent additional defect found on the fiscal Q→M path. |
| `Denton` (and `ChowLin`) | `src/aggdisagg/methods.py` | **NO** — not on the live `TemporalAligner` Denton-Cholette path. See the methods.py entry below. |

---

## Candidate C1 — Fiscal / offset-quarter anchor handling

**Verdict: PASS / no bug found**

**Implicated:** `core.py:569-683` in `_compute_high_lengths`; called from
`core.py:709` in `_prepare_data`. Cross-reference:
`SESSION_HANDOFF_2026-07.md` §1 (“Anchor / Period Lengths (Issue 1 from feedback)”).

**Why it does not fail now.** The old `pd.to_period(low_f)` snap is gone. Current
logic infers `step_months` from the median month-diff of the provided labels
(`core.py:638-648`) and, for each period, takes
`[start, next_label - 1 day]` (last period: `start + DateOffset(months=step_months) - 1 day`).
That is independent of calendar-quarter anchors.

Live checks on this checkout:

- PepsiCo Sep-start quarters (`tests/data/pepsico_quarterly_revenue.csv`, first
  label `2016-09-01`): `fit_transform(..., method="denton-cholette", target_freq="1mo")`
  yields `n_high = 3 * n_low = 117`, all lengths `{3}`, sum constraint error `0.0`.
  `tests/test_basic.py::test_issue1_fiscal_quarter_expansion` and
  `test_issue1b_no_silent_passthrough` **pass**.
- Synthetic Jul-start, Sep-start, Feb-start, and Oct-start 4-quarter series all
  expand to height 12 with lengths `[3, 3, 3, 3]`.

**Known limitation (documented as-is, from `SESSION_HANDOFF_2026-07.md`):**
“The lengths logic now prefers the spacing in the provided dates. This is what
we want for fiscal, but edge cases with very irregular or single-point input
still fall back.” Single-point input uses `_infer_ratio` / `_default_ratio`
(`core.py:589-590`, `598-600`). Very irregular spacing uses the observed
next-label span for non-last periods, which is intentional but will follow the
irregularity rather than a nominal fiscal grid.

---

## Candidate C2 — Denton-Cholette boundary accuracy vs R `tempdisagg`

**Verdict: no bug found** (known limitation, not a regression)

**Implicated:** `core.py:959-1022` in `_apply_denton` (Cholette branch
`core.py:970-973`, preliminary `p` at `core.py:981-993`). Cross-reference:
`SESSION_HANDOFF_2026-07.md` §2 (“Denton-Cholette Boundary (Issue 2)”).

**Why it does not fail as a bug.** The 1.11 formulation is what the handoff
describes: first-difference `Q = D' D` plus block-mean interpolation placed at
`0.2` into a uniform block (`off = 0.2`). Special-case overrides that forced
exact R outputs were deliberately removed. In-repo regressions **pass** at the
documented tolerances:

| Series | Measured max % vs R | Test bound | Strict handoff target |
|---|---:|---:|---|
| Toy (smooth) | 0.4614% | `< 0.5` (`test_issue2_denton_cholette_toy`) | `< 0.5` — met |
| Kraft Heinz revenue | 1.7128% | `< 2.0` and movement ratio `< 1.1` | `< 1%` / `< 1.02` — **not met** |
| B&G Foods CapEx | 26.7176% | `< 30` and movement ratio `< 1.1` | `< 1%` / `< 1.02` — **not met** |

The residual is the difference between this `p` / `Q` construction and R
`tempdisagg`’s internal preliminary series, not a broken constraint: every
Denton-Cholette run in the 3×8×5 messy-loop matrix conserved the quarterly sum.

**Known limitation (documented as-is, from `SESSION_HANDOFF_2026-07.md`):**
“Full `<1%` + `<1.02` movement-distortion parity on highly volatile real series
(B&G CapEx especially) is not yet achieved with the current p/Q formulation.
The boundary fix helps the toy and milder series.” Current numbers match the
handoff (~1.7% Kraft, higher on B&G). “Special casing / reference forcing was
deliberately removed from core; tests use tolerance-based assertions.”
Denton-Cholette deviation on volatile series is an accepted limitation of the
current algorithm, not a silent omission.

---

## Candidate C3 — NaT / dropped dates → `lengths length must match n_low`

**Verdict: CONFIRMED-BUG**

**Implicated:**

- `core.py:591` in `_compute_high_lengths` (`pd.to_datetime(...).dropna()`)
- `core.py:709-710` and `core.py:735-736` in `_prepare_data`
- raise site `core.py:52-53` in `_build_c_matrix`
- public callers `core.py:1128` in `fit_transform` → `fit` → `_prepare_data`,
  and `core.py:1564` / `core.py:1580` in `disaggregate_columns`

**Why it fails.** `_prepare_data` sets `n_low = len(y_low)` from the value
column (every row, including rows whose date is NaT). `_compute_high_lengths`
parses dates with `errors="coerce"` and then **`dropna()`**, so `n = len(low_ts)`
is the count of *valid* timestamps only. The returned `lengths` vector is
shorter than `n_low`. `_prepare_data` still passes that vector into
`_build_c_matrix(n_high, n_low, ...)`, which requires `len(lengths) == n_low`
and raises `ValueError: lengths length must match n_low`.

Reproduced: a 4-row series with one `pd.NaT` yields
`_compute_high_lengths` → `[24, 12, 18]` (3 entries; the first span also
becomes 2020-01-01→2022-01-01 = 24 months because the missing date is
removed from the timeline, not treated as a missing *period*).
`fit_transform` then raises at `_build_c_matrix`. This is batch 3 case 13
(`object_mixed` + `nat_date`) and the same pattern in batches 6–43.

There is a second, quieter defect in the same drop: even if the length check
were relaxed, dropping NaT *changes neighboring spans*, so remaining periods
would get the wrong child counts.

This is **not** the fiscal-anchor bug from `SESSION_HANDOFF_2026-07.md` §1
(that path is C1 / PASS). It is a remaining robustness hole in the same
function.

---

## Candidate C4 — Same-frequency 1:1 expansion rejected as “did not expand”

**Verdict: CONFIRMED-BUG**

**Implicated:** `core.py:713-719` in `_prepare_data` (the `n_high == n_low`
guard). Reached via `fit_transform` (`core.py:1128`) and
`disaggregate_columns` (`core.py:1564`, `core.py:1580`).

**Why it fails.** After the 1.11 fiscal fix, `_prepare_data` raises if every
computed length is 1:

```text
Disaggregation did not expand the series (output length 4 == input 4).
The low-frequency dates could not be expanded calendar-correctly to the target frequency.
Use standard calendar-aligned dates or the source_freq=... parameter for explicit control.
```

`SESSION_HANDOFF_2026-07.md` §1 documents this raise as the anti-silent-passthrough
guard for *failed* fiscal inference. The predicate cannot tell that case apart
from a legitimate same-frequency request. Batch 46 case 13 is
`low_f="Q", tf="1q", n=4` (QE labels `2026-03-31 … 2026-12-31`): calendar
expansion is correctly 1:1, and the guard still raises. Batch 47 case 19 is
the same Q→Q pattern. Year-end → monthly (`YE` → `1mo`) **does** expand to 12
and is not affected.

The error text also points users at `source_freq=...`. That parameter is stored
(`core.py:348`, `core.py:365`) and named only in this error string
(`core.py:718`). It is **never read** in `_compute_high_lengths`,
`_prepare_data`, `fit_transform`, or `disaggregate_columns`. Passing
`source_freq="Q"` does not change output. The advertised escape hatch is dead
code (see C8).

**Known limitation (related, not a substitute for the bug):** the 1.11 intent
to fail closed on true silent pass-through is sound. The defect is that the
guard is too coarse *and* the documented override does not work.

---

## Candidate C5 — `predict_with_uncertainty()` all-zero std after `fit_transform`

**Verdict: CONFIRMED-BUG**

**Implicated:** `core.py:1279-1281` in `fit_transform` (unconditionally clears
`_std_errors` and only fills them when `with_uncertainty=True` at
`core.py:1282-1387`); `core.py:2079-2091` in `predict_with_uncertainty`.

**Why it fails.** `test_improved_uncertainty` constructs
`TemporalAligner(method="uniform", target_freq="1mo", n_bootstrap=20)`, calls
`fit_transform` **without** `with_uncertainty=True`, then
`predict_with_uncertainty()` with no arguments. `fit_transform` leaves
`_std_errors is None`. `predict_with_uncertainty` only bootstraps if the
*argument* `n_bootstrap` is truthy; it does not consult `self.n_bootstrap`.
The fallback is `np.zeros_like(self._y_high)` (`core.py:2091`). Result: 48
zeros. Passing `predict_with_uncertainty(n_bootstrap=20)` does produce
non-zero std (max ≈ 0.23 on the test series), so the machinery exists but is
not wired to the constructor flag or to a no-arg follow-up call.

This is outside the fiscal / Denton-Cholette risk areas but is a current
failing test on the `fit_transform` path under review.

---

## Candidate C6 — `test_robust_100_scenarios` date ordinals in the round-trip

**Verdict: no bug found**

**Implicated (read, not defective):** `core.py:1411-1423` in `fit_transform`
(1.10 default dated DataFrame); `core.py:1684-1705` in `aggregate` (returns
`date` plus value columns when lengths are cached).

**Why it does not fail as a library bug.** Since 1.10,
`fit_transform(..., return_dataframe=True, include_dates=True)` is the default
(`SESSION_HANDOFF_2026-07.md` §3). `aggregate` then returns a frame whose first
column is `date`. The test does `back.to_numpy().ravel()[:n_low]` and compares
that to `y`. The raveled buffer starts with a date ordinal (e.g. `18628`)
interleaved with values. The disaggregation itself satisfies the internal
constraint (`_C @ _y_high`). This is a stale test versus the documented 1.10
API, not a broken expander or Denton solver.

**Known limitation:** callers that treat the aggregate result as a pure value
array must select the value column. That is the 1.10 contract, not a silent
numeric error.

---

## Candidate C7 — Missing developer-machine paths (18 failures + 2 skips)

**Verdict: no bug found**

These tests (`test_general_calendar_variable_ratios`, `test_161_*`,
`test_170_methods_are_distinct_and_machinery_works`,
`test_180_denton_cholette_and_perf`, `test_181_weekly_source_freq_detection_any_anchor`,
`test_gls_analytical_uncertainty_coverage_regression`,
`test_fit_transform_return_dataframe_and_include_dates`, etc.) read
`/Users/dev/Documents/GitHub/scrap-testing-delme/freq-test-files/...` or
`/Users/dev/Documents/GitHub/aggdisagg/pyproject.toml`. Those paths are not in
this checkout. The failures are `FileNotFoundError` / `pytest.skip`, not
assertion failures in `_compute_high_lengths` / `_apply_denton`.

**Known limitation:** calendar-ratio and weekly-anchor coverage that still
lives only on that external tree is not exercised in CI on this machine. The
in-repo fiscal / Denton fixtures (`tests/data/*`) are the portable stand-in
and they pass.

---

## Candidate C8 — `source_freq` escape hatch is never applied

**Verdict: CONFIRMED-BUG**

**Implicated:** `core.py:348` / `core.py:365` (`TemporalAligner.__init__` stores
`self.source_freq`); `core.py:718` in `_prepare_data` (error text only).
`_compute_high_lengths` (`core.py:569-683`) does not reference `source_freq`.

**Why it fails.** `SESSION_HANDOFF_2026-07.md` §1: “Added `source_freq`
parameter (as escape hatch) to the aligner.” `CHANGELOG.md` repeats that.
The attribute is write-only. Combined with C4, users who hit the 1:1 raise
cannot recover by passing `source_freq`. Integer / abstract period indexes
still expand only because of the 1970-epoch / tiny-span guard
(`core.py:595-600`) plus `_infer_ratio` / `_default_ratio`, not because
`source_freq` was honored (same height 48 with `source_freq=None` or `"Q"`).

---

## Candidate C10 — `_apply_denton` preliminary `p` ignores variable `_high_lengths`

**Verdict: CONFIRMED-BUG** (irregular-ratio path; not the C2 R-parity gap)

**Implicated:** `core.py:985-993` in `_apply_denton`.

**Why it fails.** After the calendar-aware lengths work, `_high_lengths` can be
irregular (e.g. monthly→daily in a leap February: `[31, 29, 31]`). `_apply_denton`
still builds the preliminary series with a **uniform** block size
`m = n_h // n_l` (here `91 // 3 = 30`) and places Cholette knots at
`i * m + 0.2 * m`. Block means are `y_low / m` rather than `y_low / length_i`.
The subsequent Lagrange solve plus the final per-group scale (`core.py:1012-1019`)
still enforces `C @ y_h = y_low` (reproduced constraint error `0.0` on that
M→D example), so this is not a sum-constraint break. It *is* the wrong
preliminary path for irregular calendars: knots and block means do not sit
on the true month boundaries.

The fiscal Q→M path (C1 / C2) is unaffected because those lengths are uniformly
3. This is a leftover from generalizing lengths in `_compute_high_lengths`
without updating the Denton `p` construction.

**Known limitation (do not conflate with C2):** the R `tempdisagg` deviation on
volatile *regular* quarterly series is C2 (no bug found). C10 is a separate,
real defect on variable-length calendars.

---

## methods.py — implicated method classes

**Verdict: no bug found** on the TemporalAligner failure paths under review.

**Implicated classes:** `src/aggdisagg/methods.py` `Denton` (`methods.py:128-130`)
and `ChowLin` (`methods.py:133-135`). Both inherit `Uniform` and carry a TODO
(`Denton`: “implement quadratic minimization”; `ChowLin`: “implement GLS with
indicator series + rho estimation”). `Denton().disaggregate(...)` is numerically
identical to `Uniform().disaggregate(...)`.

**Why this is not the live failure path.** `TemporalAligner` never instantiates
these classes. Denton / Denton-Cholette on `fit_transform` /
`disaggregate_columns` go through `core.py:1160-1161` → `_apply_denton`. The
legacy `AggDisaggModel` in `src/aggdisagg/api.py` only constructs `Uniform` or
`Linear` (`api.py:69-74`) and rejects other names.

**Known limitation (documented as-is):** `methods.py` `Denton` and `ChowLin` are
unimplemented placeholders. Anyone calling them directly gets uniform
allocation, not quadratic Denton and not Chow-Lin GLS. That stub behavior is
unchanged and is not the cause of C2 / C3 / C4.

---

## Per-function detail

### `_compute_high_lengths` — `core.py:569-683`

**Implicated in a confirmed failure: YES (C3, and C8 by omission).**

Fiscal / offset-quarter handling (the SESSION_HANDOFF Issue 1 risk area) is
**PASS / no bug found** (C1). The remaining defect is `dropna()` at
`core.py:591`, which desynchronizes `len(lengths)` from `n_low` and distorts
neighboring spans. The function also never consults `self.source_freq` (C8).

**Known limitation (as-is):** irregular or single-point input still falls back
to `_infer_ratio` / `_default_ratio` (`SESSION_HANDOFF_2026-07.md` “Known
Limitations”). Weekly day logic is a special case (`core.py:660-669`).

### `_apply_denton` — `core.py:959-1022`

**Implicated in a confirmed failure: YES (C10 only).**

SESSION_HANDOFF Issue 2 (Denton-Cholette boundary vs R on volatile series):
**no bug found**; known limitation documented under C2. Toy `<0.5%` holds;
Kraft ~1.71% and B&G ~26.7% are inside the in-repo tolerances and match the
handoff’s stated residual.

Confirmed defect: uniform `m = n_h // n_l` at `core.py:985-993` when
`_high_lengths` is irregular.

### `_prepare_data` — `core.py:685-737`

**Implicated in a confirmed failure: YES (C3, C4, C8).**

- Forwards a possibly shortened `lengths` into `_build_c_matrix` (`core.py:736`).
- Raises on any all-ones expansion (`core.py:713-719`), including correct
  same-frequency 1:1.
- Mentions `source_freq` in the error (`core.py:718`) but never reads it.

Empty input (`n_low == 0`) is handled (`core.py:696-704`) and is not a bug.

### `fit_transform` — `core.py:1039-1430`

**Implicated in a confirmed failure: YES (C3, C4 via `fit` → `_prepare_data`;
C5 uncertainty wiring).**

It is the public entry used by the in-repo Issue 1 / Issue 2 tests, which
**pass**. Default dated output (`core.py:1411-1423`) is the intended 1.10
behavior (C6: no bug found). Ensemble / Denton dispatch at `core.py:1158-1161`
correctly calls `_apply_denton` (not `methods.Denton`).

### `disaggregate_columns` — `core.py:1432-1630`

**Implicated in a confirmed failure: YES (inherits C3 / C4 from `fit_transform`
at `core.py:1564` and `core.py:1580`).**

No separate fiscal-anchor or Denton-Cholette defect. Multi-column semantics
(stock → `agg="last"`, flow → `agg="sum"`) and `include_dates` via
`expand_high_freq_dates` (which itself calls `_compute_high_lengths` at
`core.py:1967`) are consistent with the single-column path. Same-frequency
and NaT inputs fail here for the same reasons as `fit_transform`.

---

## What was explicitly *not* treated as a library bug

- Denton-Cholette deviation on volatile series (Kraft ~1.7%, B&G ~27%) versus
  the strict `<1%` reporter target — known limitation, `SESSION_HANDOFF_2026-07.md` §2.
- 1.10 dated `fit_transform` / `aggregate` frames breaking tests that `ravel()`
  the whole DataFrame.
- Tests that cannot run because external absolute paths are absent.
- `methods.py` Denton / ChowLin stubs (not on the live TemporalAligner path).
- Successful fiscal / offset-quarter expansion after the 1.11 `step_months` rewrite.

---

## Reproduction commands used

```bash
# In-repo Issue 1 / Issue 2 regressions (all passed)
uv run pytest tests/test_basic.py::test_issue1_fiscal_quarter_expansion \
  tests/test_basic.py::test_issue1b_no_silent_passthrough \
  tests/test_basic.py::test_issue2_denton_cholette_toy \
  tests/test_basic.py::test_issue2b_kraft_within_r \
  tests/test_basic.py::test_issue2b_bg_within_r --no-cov

# Confirmed failing tests
uv run pytest tests/test_simulation.py::test_improved_uncertainty \
  tests/test_simulation.py::test_messy_incomplete_data_batch_3 \
  tests/test_simulation.py::test_messy_incomplete_data_batch_46 \
  tests/test_simulation.py::test_messy_incomplete_data_batch_47 --no-cov
```

No `core.py` or `methods.py` behavior was modified. `version.txt` was not
created or touched.
