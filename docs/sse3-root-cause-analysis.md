# SSE3 `RuntimeError` on `import polars` — root-cause analysis

## Pinned Polars version (from `pyproject.toml`)

aggdisagg does not pin an exact Polars release. The sole constraint, quoted verbatim from
`pyproject.toml` `[project].dependencies`, is:

    "polars>=1.0.0",

That is the string `polars>=1.0.0`.

That floor admits any Polars 1.x (and would admit 2.x if published). A user who
`pip install`s aggdisagg therefore receives whatever current default `polars` wheel
satisfies `polars>=1.0.0`, not a single frozen binary.

## Symptom

`import polars` (and therefore `from aggdisagg import TemporalAligner`) raises:

```
RuntimeError: unknown feature flag: 'sse3'
```

The traceback ends in Polars' own import path (`polars/__init__.py` → `polars/_plr.py`
→ `polars/_cpu_check.py::check_cpu_flags`) **before any aggdisagg module body
executes**. The failure is entirely inside the Polars Python/Rust wheel.

## How the error is produced

Since December 2023 Polars has run a pure-Python CPUID gate *before* loading the
native extension, so the Rust compiler cannot emit an illegal instruction before the
check runs. The check is in
[`py-polars/src/polars/_cpu_check.py`](https://github.com/pola-rs/polars/blob/main/py-polars/src/polars/_cpu_check.py)
(at the `py-1.0.0` tag:
[`py-polars/polars/_cpu_check.py`](https://github.com/pola-rs/polars/blob/py-1.0.0/py-polars/polars/_cpu_check.py)).

`check_cpu_flags` walks the compile-time feature list baked into the wheel
(`BUILD_FEATURE_FLAGS` / `_POLARS_FEATURE_FLAGS`). Two outcomes are possible:

1. A name in that list is **not** a key of `_read_cpu_flags()` → hard
   `RuntimeError: unknown feature flag: '…'`. This is the observed exception.
2. The name is known but CPUID reports it absent → `RuntimeWarning: Missing
   required CPU features` (import continues; a later SIGILL is likely).

`sse3` **is** a first-class key of `_read_cpu_flags()` in every 1.x tree inspected,
including the `py-1.0.0` tag. Therefore `'sse3'` is reported as *unknown* only when
the flags dict is empty or mismatched — typically because:

- `platform.machine()` is not treated as x86, so `_SUPPORTS_CPUID` is false and
  `_read_cpu_flags()` returns `{}` (current `main`); or
- CPUID is unavailable / masked (some hypervisors, older Xeons, Rosetta, containers
  that hide feature bits); or
- mixed `polars` / `polars-lts-cpu` / `polars-runtime-*` installs leave a binary
  whose `BUILD_FEATURE_FLAGS` do not match the installed `_cpu_check.py`
  ([pola-rs/polars#26534](https://github.com/pola-rs/polars/issues/26534)).

The same `RuntimeError: unknown feature flag: '…'` path is a documented Polars
import failure mode: [pola-rs/polars#26047](https://github.com/pola-rs/polars/issues/26047)
(`unknown feature flag: '-crt-static'` on musl, fixed by
[pola-rs/polars#26076](https://github.com/pola-rs/polars/pull/26076)). The identical
`'sse3'` message has been reported independently of aggdisagg
([NEXTAltair/LoRAIro#264](https://github.com/NEXTAltair/LoRAIro/issues/264)).

## Does this Polars range require SSE3 at runtime?

Yes. That is by design for every official x86-64 wheel in the `>=1.0.0` range,
including the “legacy CPU” build.

At tag `py-1.0.0` the release workflow compiles x86-64 wheels with
`-C target-feature=…` as follows
([`.github/workflows/release-python.yml` at `py-1.0.0`](https://github.com/pola-rs/polars/blob/py-1.0.0/.github/workflows/release-python.yml)):

| Wheel | Compile-time `target-feature` list |
| --- | --- |
| default `polars` (Linux/Windows) | `+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+avx,+avx2,+fma,+bmi1,+bmi2,+lzcnt,+pclmulqdq` |
| default `polars` (macOS at 1.0.0) | `+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+avx,+fma,+pclmulqdq` |
| `polars-lts-cpu` | `+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt` |

Current `main` still starts **both** the default and the compat lists with `+sse3`
([`.github/workflows/release-python.yml` on `main`](https://github.com/pola-rs/polars/blob/main/.github/workflows/release-python.yml)):

- `NONCOMPAT_FEATURES` (default / `rt32`):
  `+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+fma,+bmi1,+bmi2,+lzcnt,+pclmulqdq,+movbe`
- `COMPAT_FEATURES` (`polars[rtcompat]` / former LTS):
  `+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b`

SSE3 is therefore a **minimum** of every official x86-64 Polars 1.x binary. The
default wheel additionally requires AVX2-class features. Polars documents this as
expected behaviour, not a regression:

- Installation guide: install `polars[rtcompat]` “for legacy CPUs without AVX2
  support” —
  <https://docs.pola.rs/user-guide/installation/>
- PyPI `polars-lts-cpu`: “Do you want Polars to run on an old CPU (e.g. dating
  from before 2011), or on an x86-64 build of Python on Apple Silicon under
  Rosetta? Install `pip install polars-lts-cpu`.” —
  <https://pypi.org/project/polars-lts-cpu/>
- [pola-rs/polars#15404](https://github.com/pola-rs/polars/issues/15404) (“permit
  run polars on old CPU”): missing-feature warning on import; maintainers directed
  users to `polars-lts-cpu` and declined to drop the SIMD baseline (even LTS keeps
  `popcnt`).
- [pola-rs/polars#19936](https://github.com/pola-rs/polars/issues/19936)
  (`POLARS_SKIP_CPU_CHECK` does not prevent `illegal hardware instruction` on
  bare `import polars`): closed as **invalid** — import-time use of the compiled
  SIMD baseline is expected.

No changelog or issue in the `1.0.0`–current 1.x window describes an SSE3
*regression* that was later fixed by moving to a different 1.x release. The
`+sse3` flag is present at `py-1.0.0` and still present on `main`.

## Why aggdisagg's constraint does not cause this

1. The error is raised inside Polars' import-time CPU check, with no aggdisagg
   frame on the stack.
2. `polars>=1.0.0` is a **minimum**, not a pin to a single buggy build. Users
   already resolve to the latest 1.x default wheel.
3. Every default 1.x x86-64 wheel is compiled with `+sse3` (and, except LTS/compat,
   AVX2). Pinning `polars==1.0.0`, `polars==1.14.0`, or any other 1.x default
   wheel still ships a binary that lists `sse3` in `BUILD_FEATURE_FLAGS`.
4. The official escape hatch is an **alternate wheel** (`polars-lts-cpu` /
   `polars[rtcompat]`), not a different version number. Even that compat wheel
   still requires SSE3. A host whose CPUID does not expose SSE3 (or whose
   `platform.machine()` prevents the check from populating flags) will fail the
   same way on any 1.x x86-64 wheel.
5. Lowering the floor below 1.0.0 would be an API break and would not remove the
   SIMD baseline (0.20.x already shipped the same CPU check and `+sse3` list).

Workarounds belong on the **user machine**, not in aggdisagg's dependency list:

- Confirm the CPU / hypervisor actually advertises SSE3 (`/proc/cpuinfo` flags,
  `sysctl machdep.cpu.features`, or Windows CPU-Z). Feature masking on older
  Xeons and some cloud/hypervisor profiles is a known source of this class of
  failure.
- On a genuine pre-AVX2 (but SSE3+) host, install the compat wheel:
  `pip install polars-lts-cpu` or `pip install 'polars[rtcompat]'`.
- Avoid mixing `polars` and `polars-lts-cpu` at different versions
  ([pola-rs/polars#26534](https://github.com/pola-rs/polars/issues/26534)).
- `POLARS_SKIP_CPU_CHECK=1` only bypasses the *warning* path; it does not make
  a no-SSE3 CPU execute a `+sse3` binary
  ([pola-rs/polars#19936](https://github.com/pola-rs/polars/issues/19936)).

## Classification

**(a)** This is purely an **environment/CPU issue** on the user's machine.
Polars' behaviour for every release admitted by `polars>=1.0.0` is
expected/by-design: official x86-64 wheels are compiled with `+sse3` and
import-time `check_cpu_flags` will raise `RuntimeError: unknown feature flag:
'sse3'` when that compile-time flag cannot be resolved against the host.
aggdisagg's dependency constraint did not introduce an SSE3 regression and
cannot mitigate the failure by pinning a different Polars version.

## Citations

- aggdisagg constraint: `pyproject.toml` `[project].dependencies` → `polars>=1.0.0`
- Polars CPU check (current): <https://github.com/pola-rs/polars/blob/main/py-polars/src/polars/_cpu_check.py>
- Polars CPU check at 1.0.0: <https://github.com/pola-rs/polars/blob/py-1.0.0/py-polars/polars/_cpu_check.py>
- Wheel `target-feature` lists at 1.0.0: <https://github.com/pola-rs/polars/blob/py-1.0.0/.github/workflows/release-python.yml>
- Wheel `target-feature` lists on current `main`: <https://github.com/pola-rs/polars/blob/main/.github/workflows/release-python.yml>
- Installation / legacy-CPU docs: <https://docs.pola.rs/user-guide/installation/>
- `polars-lts-cpu` legacy note: <https://pypi.org/project/polars-lts-cpu/>
- [pola-rs/polars#15404](https://github.com/pola-rs/polars/issues/15404) — old-CPU import warning; LTS recommended; SIMD baseline kept
- [pola-rs/polars#19936](https://github.com/pola-rs/polars/issues/19936) — import-time illegal instruction is expected, not a skippable check
- [pola-rs/polars#26047](https://github.com/pola-rs/polars/issues/26047) / [pola-rs/polars#26076](https://github.com/pola-rs/polars/pull/26076) — `unknown feature flag` raised from `_cpu_check.py`
- [pola-rs/polars#26534](https://github.com/pola-rs/polars/issues/26534) — mixed `polars` / `polars-lts-cpu` versions
- [NEXTAltair/LoRAIro#264](https://github.com/NEXTAltair/LoRAIro/issues/264) — same `RuntimeError: unknown feature flag: 'sse3'` on `import polars`
