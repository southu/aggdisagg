# SSE3 `RuntimeError` on `import polars` — root-cause analysis

## Classification (unambiguous)

**(a)** This is purely an **environment/CPU issue** on the user's machine.
Polars' behaviour for every release admitted by `polars>=1.0.0` is
expected/by-design and is not something aggdisagg's pinning caused.

Supporting citation:
<https://github.com/pola-rs/polars/issues/15404>

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

## How the error is produced — two mutually exclusive paths

Since December 2023 Polars has run a pure-Python CPUID gate *before* loading the
native extension, so the Rust compiler cannot emit an illegal instruction before the
check runs. The check is in
[`py-polars/src/polars/_cpu_check.py`](https://github.com/pola-rs/polars/blob/main/py-polars/src/polars/_cpu_check.py)
(at the `py-1.0.0` tag:
[`py-polars/polars/_cpu_check.py`](https://github.com/pola-rs/polars/blob/py-1.0.0/py-polars/polars/_cpu_check.py)).

`check_cpu_flags` walks the compile-time feature list baked into the wheel
(`BUILD_FEATURE_FLAGS` / `_POLARS_FEATURE_FLAGS`). The two outcomes below are
**mutually exclusive**. Polars' own unit tests encode that split
([`test_cpu_check.py` on `main`](https://github.com/pola-rs/polars/blob/main/py-polars/tests/unit/meta/test_cpu_check.py)):

| Path | Condition | Result | Official test |
| --- | --- | --- | --- |
| **Unknown flag** | A name in the compile-time list is **not a key** of `_read_cpu_flags()` | hard `RuntimeError: unknown feature flag: '…'` | `test_check_cpu_flags_unknown_flag` (`'HelloWorld!'`) |
| **Missing feature** | The name **is** a key, but CPUID reports the bit **absent** (`False`) | `RuntimeWarning: Missing required CPU features`; import **continues** | `test_check_cpu_flags_missing_features` (`ssse3: False`) |

The observed exception is the **unknown-flag** path with the name `'sse3'`.
It is **not** the missing-feature path.

Consequences that follow from that split, and that this report will not
contradict later:

1. A host whose CPUID reader **does** know the name `sse3` and reports the
   SSE3 bit as **false** cannot produce `RuntimeError: unknown feature flag:
   'sse3'`. That host takes the warning path. This is the case the unit test
   `test_check_cpu_flags_missing_features` covers, and it is the case
   [pola-rs/polars#15404](https://github.com/pola-rs/polars/issues/15404)
   documented (old CPU → import-time *warning*; maintainers pointed users at
   `polars-lts-cpu`).
2. The reported `RuntimeError` means `_read_cpu_flags()` returned a mapping
   that **did not contain the key `'sse3'` at all**. On a consistent x86
   install that mapping always includes `'sse3'` (value `True` or `False`).
   The key is missing only when the mapping is empty or is from a mismatched
   checker — not when SSE3 is merely unsupported.

`'sse3'` **is** a first-class key of `_read_cpu_flags()` in every 1.x tree
inspected whenever CPUID is actually consulted, including the `py-1.0.0` tag.
Therefore `'sse3'` is reported as *unknown* only when that reader does not
populate the usual x86 dict.

## When is the `'sse3'` key actually absent?

### 1. CPUID is not consulted (`_SUPPORTS_CPUID` is false) — Polars ≥ 1.38.1

At `py-1.0.0`, `_read_cpu_flags()` always constructed a CPUID reader and
always returned a dict that included `"sse3"`. On a consistent 1.0.0 install
the unknown-flag path was therefore unreachable for the name `'sse3'`.

Python Polars **1.38.1** shipped
[pola-rs/polars#26439](https://github.com/pola-rs/polars/pull/26439)
("Don't run CPU check on aarch64 musl"; listed in the
[1.38.1 release notes](https://github.com/pola-rs/polars/releases/tag/py-1.38.1)).
That change introduced:

```python
_SUPPORTS_CPUID = platform.machine().lower() in {
    "x86_64", "x64", "amd64", "x86", "i368", "i686", "i686-64",
}

def _read_cpu_flags() -> dict[str, bool]:
    if not _SUPPORTS_CPUID:
        return {}
    # ... otherwise the usual dict, still including "sse3"
```

`check_cpu_flags` was **not** taught to skip the walk when the dict is empty.
If the installed runtime still lists compile-time x86 flags (every official
x86-64 wheel starts that list with `+sse3`), the first name looked up in `{}`
is `'sse3'` and the check raises exactly:

```
RuntimeError: unknown feature flag: 'sse3'
```

`_SUPPORTS_CPUID` is false when `platform.machine()` is not in that hardcoded
set. That is an **environment / platform-identification** condition, for
example:

- non-x86 `uname` (`aarch64`, `arm64`, `ppc64le`, …) together with an x86
  runtime whose `BUILD_FEATURE_FLAGS` still contain `+sse3` (wrong wheel,
  emulation, mixed site-packages);
- an x86 host whose machine string is unrecognized (`i386` is a common
  32-bit name but the set lists the typo `"i368"` rather than `"i386"`;
  some hypervisors and containers report empty or unusual strings);
- Windows-on-ARM or similar, where `platform.machine()` may be `ARM64`
  even if an x86-64 wheel was installed.

The same `RuntimeError: unknown feature flag: '…'` constructor is the
documented import failure in
[pola-rs/polars#26047](https://github.com/pola-rs/polars/issues/26047)
(`unknown feature flag: '-crt-static'` on musl; fixed by
[pola-rs/polars#26076](https://github.com/pola-rs/polars/pull/26076)).
#26439 was itself a follow-up so that leftover non-CPU rustc flags would not
drive the check on aarch64. The empty-dict implementation is what makes the
**first real compile flag** (`sse3` on every x86-64 wheel) surface as
"unknown" whenever CPUID is skipped.

The identical `'sse3'` message has been reported independently of aggdisagg
on a modern `_plr.py` / `rt_32` install
([NEXTAltair/LoRAIro#264](https://github.com/NEXTAltair/LoRAIro/issues/264)).

### 2. Mixed Polars / runtime packages

Current Polars selects a runtime (`_polars_runtime_32` / `_64` / `_compat`)
and passes that package's `BUILD_FEATURE_FLAGS` into `check_cpu_flags`
([`_plr.py` on `main`](https://github.com/pola-rs/polars/blob/main/py-polars/src/polars/_plr.py)).
Installing `polars` and `polars-lts-cpu` (or mismatched runtime packages)
at different versions can leave a `BUILD_FEATURE_FLAGS` string that the
installed `_cpu_check.py` cannot name
([pola-rs/polars#26534](https://github.com/pola-rs/polars/issues/26534)).
That is an **install** mismatch on the user machine, not a missing SSE3 bit.

### What this is *not*

It is **not** "CPUID ran, the SSE3 feature bit was 0, therefore
`unknown feature flag: 'sse3'`". That sequence is the missing-feature
**warning** path. Equating the two is how an earlier draft of this note
contradicted itself.

## Does this Polars range require SSE3 at runtime?

Yes, **on x86-64, once CPUID is consulted**. That is by design for every
official x86-64 wheel in the `>=1.0.0` range, including the “legacy CPU”
build. It is a separate fact from the observed `RuntimeError`.

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

SSE3 is therefore a **minimum** of every official x86-64 Polars 1.x binary
*when the flag reader can name it*. The default wheel additionally requires
AVX2-class features. Polars documents the missing-feature outcome as expected
behaviour, not a regression:

- Installation guide: install `polars[rtcompat]` “for legacy CPUs without AVX2
  support” —
  <https://docs.pola.rs/user-guide/installation/>
- PyPI `polars-lts-cpu`: “Do you want Polars to run on an old CPU (e.g. dating
  from before 2011), or on an x86-64 build of Python on Apple Silicon under
  Rosetta? Install `pip install polars-lts-cpu`.” —
  <https://pypi.org/project/polars-lts-cpu/>
- [pola-rs/polars#15404](https://github.com/pola-rs/polars/issues/15404) (“permit
  run polars on old CPU”): **missing-feature warning** on import; maintainers
  directed users to `polars-lts-cpu` and declined to drop the SIMD baseline
  (even LTS keeps `popcnt`).
- [pola-rs/polars#19936](https://github.com/pola-rs/polars/issues/19936)
  (`POLARS_SKIP_CPU_CHECK` does not prevent `illegal hardware instruction` on
  bare `import polars`): closed as **invalid** — import-time use of the compiled
  SIMD baseline is expected.

No changelog or issue in the `1.0.0`–current 1.x window describes an SSE3
*codegen* regression that was later fixed by moving to a different 1.x release.
The `+sse3` compile flag is present at `py-1.0.0` and still present on `main`.
#26439 did not add or remove SSE3 from the binary; it changed when the Python
checker consults CPUID, which is how the **unknown-flag** spelling of this
failure became reachable.

## Why aggdisagg's constraint does not cause this

1. The error is raised inside Polars' import-time CPU check, with no aggdisagg
   frame on the stack.
2. `polars>=1.0.0` is a **minimum**, not a pin to a single buggy build. Users
   already resolve to the latest 1.x default wheel.
3. Every default 1.x x86-64 wheel is compiled with `+sse3` (and, except
   LTS/compat, AVX2). Pinning `polars==1.0.0`, `polars==1.14.0`,
   `polars==1.38.0`, or any other 1.x default wheel still ships a binary that
   lists `sse3` in `BUILD_FEATURE_FLAGS`.
4. Pinning **below** 1.38.1 would make `_read_cpu_flags()` always insert the
   `'sse3'` key (so this particular `RuntimeError` would not fire), but that
   is not a fix for the observed failure mode: it reverts
   [pola-rs/polars#26439](https://github.com/pola-rs/polars/pull/26439), which
   exists so aarch64/musl hosts do not execute x86 CPUID stubs, and it still
   leaves every x86-64 wheel compiled with `+sse3`. On a host where CPUID is
   skipped today, the next failure would be a different environment crash, not
   a working import. Pinning **above** 1.38.1 cannot help: current `main` still
   returns `{}` when `_SUPPORTS_CPUID` is false.
5. The official escape hatch is an **alternate wheel** (`polars-lts-cpu` /
   `polars[rtcompat]`) plus a consistent install, not a different version
   number. Even the compat wheel still requires SSE3 **once CPUID runs**. A
   host whose flag reader never names `'sse3'` fails the unknown-flag path
   before that warning can appear.
6. Lowering the floor below 1.0.0 would be an API break and would not remove
   the SIMD baseline (0.20.x already shipped the same CPU check and `+sse3`
   list).

Workarounds belong on the **user machine**, not in aggdisagg's dependency list:

- Confirm `platform.machine()` is a recognized x86 name (`x86_64`, `amd64`,
  …). An unrecognized or non-x86 string is what makes ≥1.38.1 return `{}` and
  then raise `unknown feature flag: 'sse3'` against an x86 `BUILD_FEATURE_FLAGS`
  list ([pola-rs/polars#26439](https://github.com/pola-rs/polars/pull/26439),
  [1.38.1 notes](https://github.com/pola-rs/polars/releases/tag/py-1.38.1)).
- Use a native-arch Python and the matching wheel. An x86-64 wheel on ARM
  (or the reverse) is the typical way x86 flags meet a checker that refuses
  CPUID.
- On a genuine pre-AVX2 **x86** host where CPUID *does* run and the
  **warning** path fires, install the compat wheel:
  `pip install polars-lts-cpu` or `pip install 'polars[rtcompat]'`.
  That addresses missing AVX2, not the unknown-flag `RuntimeError`.
- Avoid mixing `polars` and `polars-lts-cpu` (or mismatched runtimes) at
  different versions
  ([pola-rs/polars#26534](https://github.com/pola-rs/polars/issues/26534)).
- `POLARS_SKIP_CPU_CHECK=1` bypasses `check_cpu_flags` entirely (both
  paths). It does not make a no-SSE3 CPU execute a `+sse3` binary
  ([pola-rs/polars#19936](https://github.com/pola-rs/polars/issues/19936)).

## Classification

**(a)** This is purely an **environment/CPU issue** on the user's machine
(the CPU-flag reader never named `'sse3'`: CPUID was not consulted, or the
installed Polars packages disagree). It is **not** a missing-SSE3 feature
bit, and it is not something aggdisagg's `polars>=1.0.0` constraint caused.

Polars' behaviour for every release admitted by `polars>=1.0.0` is
expected/by-design on the two paths the checker implements:

- Official x86-64 wheels are compiled with `+sse3`. When CPUID runs and the
  SSE3 bit is absent, `check_cpu_flags` emits `RuntimeWarning: Missing
  required CPU features` and import continues
  ([pola-rs/polars#15404](https://github.com/pola-rs/polars/issues/15404);
  `test_check_cpu_flags_missing_features`).
- The **observed** `RuntimeError: unknown feature flag: 'sse3'` is the other
  path: `'sse3'` is not a key of `_read_cpu_flags()`. After 1.38.1 that is
  the empty-dict result of `_SUPPORTS_CPUID is False`
  ([pola-rs/polars#26439](https://github.com/pola-rs/polars/pull/26439)),
  or a mixed-package flag list
  ([pola-rs/polars#26534](https://github.com/pola-rs/polars/issues/26534)).
  Both are host/environment conditions. The same constructor is how Polars
  reports unreadable compile flags
  ([pola-rs/polars#26047](https://github.com/pola-rs/polars/issues/26047)).

aggdisagg's dependency constraint did not introduce an SSE3 codegen
regression. Pinning a different 1.x version cannot make `_read_cpu_flags()`
grow a `'sse3'` key on a host where CPUID is skipped, and cannot remove
`+sse3` from official x86-64 wheels.

## Citations

- aggdisagg constraint: `pyproject.toml` `[project].dependencies` → `polars>=1.0.0`
- Polars CPU check (current): <https://github.com/pola-rs/polars/blob/main/py-polars/src/polars/_cpu_check.py>
- Polars CPU check at 1.0.0: <https://github.com/pola-rs/polars/blob/py-1.0.0/py-polars/polars/_cpu_check.py>
- Official two-path unit tests: <https://github.com/pola-rs/polars/blob/main/py-polars/tests/unit/meta/test_cpu_check.py>
- Runtime selector passing `BUILD_FEATURE_FLAGS`: <https://github.com/pola-rs/polars/blob/main/py-polars/src/polars/_plr.py>
- Wheel `target-feature` lists at 1.0.0: <https://github.com/pola-rs/polars/blob/py-1.0.0/.github/workflows/release-python.yml>
- Wheel `target-feature` lists on current `main`: <https://github.com/pola-rs/polars/blob/main/.github/workflows/release-python.yml>
- Installation / legacy-CPU docs: <https://docs.pola.rs/user-guide/installation/>
- `polars-lts-cpu` legacy note: <https://pypi.org/project/polars-lts-cpu/>
- [pola-rs/polars#15404](https://github.com/pola-rs/polars/issues/15404) — old-CPU import **warning**; LTS recommended; SIMD baseline kept
- [pola-rs/polars#19936](https://github.com/pola-rs/polars/issues/19936) — import-time illegal instruction is expected, not a skippable check
- [pola-rs/polars#26047](https://github.com/pola-rs/polars/issues/26047) / [pola-rs/polars#26076](https://github.com/pola-rs/polars/pull/26076) — `unknown feature flag` raised from `_cpu_check.py`
- [pola-rs/polars#26439](https://github.com/pola-rs/polars/pull/26439) — `_SUPPORTS_CPUID`; empty flags dict when `platform.machine()` is not x86
- [Python Polars 1.38.1 release notes](https://github.com/pola-rs/polars/releases/tag/py-1.38.1) — “Don't run CPU check on aarch64 musl (#26439)”
- [pola-rs/polars#26534](https://github.com/pola-rs/polars/issues/26534) — mixed `polars` / `polars-lts-cpu` versions
- [NEXTAltair/LoRAIro#264](https://github.com/NEXTAltair/LoRAIro/issues/264) — same `RuntimeError: unknown feature flag: 'sse3'` on `import polars`
