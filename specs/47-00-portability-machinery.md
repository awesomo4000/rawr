<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 47-00: Portability machinery and its falsification controls

Toplevel: [47-portability-matrix.md](47-portability-matrix.md).

> **Outcome — complete.** `check-portability` now has independently invocable control, public-probe,
> and allowlist-package steps for all 16 target triples and both baseline-feature configurations. Its
> controller runs every requested cell after failures, requires the exact requested cell count, and
> reports the three phase results separately. `tools/check_32_api.zig` reaches all six `OwnedBitmap`
> methods and all nine `RoaringBitmap` `*Owned` producers, and native `check-package` retains its
> build-and-run behaviour while portability cells use an explicit-target build-only mode.
>
> The documented option commands are pinned as:
>
> ```bash
> zig build bench-parity-worker -Dtarget=aarch64-linux-gnu -Dcpu=baseline -Dcroaring-avx512=false
> zig build bench-parity-worker -Dtarget=x86_64-linux-gnu -Dcpu=x86_64_v4+evex512 -Dcroaring-avx512=true
> ```
>
> `scripts/check-portability-controls.sh` exercises all five controls in disposable `/tmp` copies. The
> extended probe caught an `OwnedBitmap`-only width defect while the call-removed probe passed; the
> allowlist consumer built the OpenBSD dependency path from the full checkout and then caught the real
> missing `src/bench_openbsd.c` file in the packaged copy; removing the translate-c macro made the affected
> option check fail while the plain library still built; inverting the x86 predicate tripped the baseline
> dispatch assertion; the reporting controller printed the cell after a seeded middle-cell failure, and
> a deliberately truncated controller failed its exact cell-count assertion.
> The full target matrix was not run in this chunk.

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

- Add `OwnedBitmap`, its full method surface, and every public `RoaringBitmap` `*Owned` producer to the
  probe. This states the guard boundary explicitly rather than claiming coverage beyond its enumerated
  calls.
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

**Control:** add a reference to an existing repository file outside `.paths` on the consumer path, show
the in-repository build succeeds, and show the allowlist-only package build fails.
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

**Control:** make one mid-matrix target fail and show every other cell is still reported. Also truncate
the controller after that failure and show the exact expected-cell-count assertion fails. This is the one
most likely to be skipped, and the failure it prevents — a table with silent holes read as coverage — is
the failure this whole spec exists to avoid.

## Acceptance

- Probe extended to the six-method `OwnedBitmap` surface and all nine `RoaringBitmap` `*Owned` producers;
  `check-32` and `check-portability` remain separate steps.
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

## Verification record — implemented, reviewed, ACCEPTED

**Checked in the tree, not taken on report.** All six `OwnedBitmap` methods (`deinit`, `contains`,
`asBitmap`, `cardinality`, `iterator`, `serialize`) and all nine `*Owned` producers are reached by the
probe. The controller asserts `cells != expected_cells` (`check_portability.zig:69`), so truncation now
fails rather than passing on a nonzero broken count.

**Three review findings, all of which were guards that could not fail for their stated reason.** That is
the failure this chunk was written to prevent, and finding them here rather than in `47-01` is the chunk
working as intended:

| finding | before | after |
| --- | --- | --- |
| reporting control | asserted only `broken != 0`; a controller halting at the middle cell gave `cells=2, broken=1` and **passed** | exact cell count required |
| package control | referenced `src/not-in-package.zig`, which exists nowhere — failed on *file-not-found* in any mode, proving nothing about allowlist membership | real defect reproduced |
| probe boundary | six `OwnedBitmap` methods covered, nine `*Owned` producers unenumerated and therefore invisible per spec 40-01 | all nine added |

**The package control is now stronger than what was asked for.** The request was to reference an existing
non-allowlisted file. The implementation instead injects `addBenchmarkPlatformShim(b, lib_mod, target)` —
attaching the OpenBSD shim to `lib_mod`, which is **the exact risk the toplevel §1 finding describes** —
then shows the full checkout builds under `-Dtarget=x86_64-openbsd` while the allowlist-only package build
fails on the missing `src/bench_openbsd.c`. It reproduces the mechanism rather than simulating its
symptom.

**The cell-count fix was made falsifiable without being asked.** `mid-matrix-truncated` seeds a truncating
controller and requires `CellCountMismatch`, so the assertion that catches truncation is itself
controlled. Together with the two-sided `owned-bitmap-unextended` and `build-option-plain-library` arms,
three of the six controls now prove their guard is load-bearing rather than merely correlated with a pass.

**Controls run in disposable `/tmp` copies with a guarded cleanup** — the right handling for work that
seeds defects into a source tree.

## Estimate

**M** — the checks are small; the controls are most of the work and all of the value.
