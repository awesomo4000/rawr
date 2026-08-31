<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 47-00: Portability machinery and its falsification controls

Toplevel: [47-portability-matrix.md](47-portability-matrix.md).

Builds the checks. **Runs nothing across the matrix** — that is `47-01`. **No evidence table, no README
change**, which is `47-02`.

**Every guard here must be demonstrated to fail against a seeded defect.** The toplevel review found four
proposed guards that would have passed with the defect they claimed to detect fully present. A guard whose
failure mode is untested is a guard nobody can rely on, and this chunk exists as much for the controls as
for the checks.

## 1. Extend the API probe

`tools/check_32_api.zig` never references `OwnedBitmap`, so a defect confined to its six methods passes
today. Spec 40-01 established that **the probe's enumerated surface is the guard boundary** — an
unenumerated type is invisible, not implicitly covered.

- Add `OwnedBitmap` and its full method surface to the probe.
- **Keep `check-32` and `check-portability` as separate steps.** They cover different axes — pointer
  width versus arch × OS. Merging them silently drops 32-bit coverage.

**Control:** seed a defect reachable only through `OwnedBitmap` and show the extended probe fails on a
target where the unextended probe passed. Without that, "extended" is an assertion.

## 2. Two per-target checks, not one

### 2.1 API probe compile (`check-portability`)

Cross-compile the probe for each of the 16 target triples of toplevel §3. No execution, no host.

### 2.2 Allowlist-only package consumer — the one that reaches `build.zig`

**The probe alone does not exercise the shipped build script.** `check-32` compiles source modules
directly and bypasses the dependency path, which is exactly where toplevel §1's OpenBSD and FreeBSD
branches live. Building only the probe would reprioritise the BSDs and then never test what made them a
priority.

Add a **per-target, allowlist-only package consumer that passes the target into `b.dependency`**, so the
shipped `build.zig` is resolved and executed for every cell.

**Control:** add a reference to a file outside `.paths` on the consumer path and show the check fails.
This is the real defect shape — `build.zig`'s OpenBSD branch names `src/bench_openbsd.c`, which is not in
the allowlist.

## 3. Build-option coverage

rawr has exactly one documented option, `-Dcroaring-avx512`, and it did not build until `52-00` fixed it.
**100% of the option surface was broken and nothing noticed**, because every check exercises default
values only.

- **Must run through a CRoaring-backed step** such as `bench-parity-worker`. A plain
  `zig build -Dcroaring-avx512=true` builds the library and **need not instantiate translate-c** — which
  was the broken path, so that command would pass with the defect present.
- **`true` panics deliberately on non-x86_64** (`build.zig:13`), so it runs on an x86_64 target only;
  `false` runs everywhere.
- **Pin both commands verbatim in the spec record**, not as a description.

**Control:** revert the translate-c macro definition and show the `true` command fails. Also show the
plain library build still *passes* with that defect present — which is the evidence that the step choice
matters and not just the flag.

## 4. Baseline-feature dispatch assertions

`array_simd.zig` gates on features, not architecture: `has_x86_simd` needs AVX **and** SSSE3, `has_neon`
needs the NEON bit, and `array_kernels.zig` consumes them to pick the array-intersection kernel.

Add cells for **`x86_64` without AVX** and **`aarch64` without NEON**. They are cross-compile-only, so the
deliverable is **compile plus an assertion on which kernel was selected** — "pass" is not available to a
target that never runs.

**State the fact narrowly.** These targets take the **scalar array-intersection path**. They do *not* lack
SIMD: `simdBitsetOp` (`bitset_container.zig:213`) uses `@Vector(8, u64)` with **no feature gate at all**.
An earlier toplevel draft claimed otherwise.

**Control:** invert the predicate and show the assertion fails.

## 5. Report every cell

A single aggregate step cannot produce an evidence table: one failing or non-targetable target stops it,
and every later cell goes unrecorded while looking merely absent.

Require **independently invocable per-target substeps, or a controller** that runs each cell and records
**pass, failure, or not-targetable** without skipping the rest.

**Control:** make one mid-matrix target fail and show every other cell is still reported. This is the one
most likely to be skipped, and the failure it prevents — a table with silent holes read as coverage — is
the failure this whole spec exists to avoid.

## Acceptance

- Probe extended to `OwnedBitmap`; `check-32` and `check-portability` remain separate steps.
- Per-target API probe compile **and** per-target allowlist-only package consumer, the latter passing the
  target into `b.dependency`.
- Build-option check wired to a **CRoaring-backed step**, both values, `true` on x86_64 only, **both
  commands pinned verbatim**.
- Baseline-feature cells assert **selected kernel**, with the narrow wording of §4.
- Per-cell reporting per §5.
- **All five controls exercised and their results recorded** — §1 probe, §2.2 non-allowlisted file, §3
  reverted macro *plus* the demonstration that the plain library build still passes, §4 inverted
  predicate, §5 mid-matrix failure. **A control that was not run is a failed acceptance criterion**, not a
  detail.
- No matrix run, no evidence table, no README change.
- Existing suites plus `check-32`, `check-docs`, `check-package` green.

## Estimate

**M** — the checks are small; the controls are most of the work and all of the value.
