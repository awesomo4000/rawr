<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 11-00: Kernel extraction + benchmark corpus

First chunk of [array-kernel performance](11-array-kernel-perf.md). Pure setup:
extract the array kernels into a shared module and stand up the benchmark that
makes every later win visible. **No behavior change** — after this chunk the
production path still calls gallop exactly as before.

**Dependency order:** none — this is the base everything else builds on. 11-01,
11-03, 11-05 all depend on the module and bench created here.

Why first: the current compare board reported ~1.03× "parity" against CRoaring
while the array∩array kernel was **6× slower** — the corpus never exercised
balanced array pairs, so the regression was invisible. Without this chunk the later
wins can't be measured.

## Task 1 — Extract the kernels into `src/array_kernels.zig`

Create an internal, exported module `src/array_kernels.zig` and move the **existing
production array∩array kernels** into it. Extract **all three operation shapes**,
each in its current gallop form, and add the corresponding **merge** form
(initially unused — 11-01 wires the dispatch that selects between them):

| shape | current (gallop) source | new module API |
|---|---|---|
| writing intersection | `arrayIntersectArray` (`container_ops.zig:707`) | `intersectWriteGallop`, `intersectWriteMerge` |
| cardinality-only | `arrayIntersectArrayCard` (`:889`) | `intersectCardGallop`, `intersectCardMerge` |
| boolean | `arrayIntersectsArray` (`:976`) | `intersectBoolGallop`, `intersectBoolMerge` |

`container_ops.zig` imports the module and calls the **gallop** variants exactly as
today (no dispatch yet, no behavior change). Defining all six here means 11-01 is
genuinely dispatch-only and needs no further extraction. The merge kernel shape is
the branchless walk already used by `arrayDifferenceArray` — reuse it, don't
re-derive.

**Module layout (fixed for the whole spec):**
- `array_kernels.zig` — production dispatch + portable kernels (gallop, merge); the
  *only* kernel module `container_ops.zig` and `bench_aa.zig` import.
- `array_simd.zig` (added in [11-05](11-05-x86-simd.md)) — arch-specific SIMD,
  imported **privately by `array_kernels.zig`** only.
- `bench_aa.zig` — imports **only** `array_kernels.zig`.

## Task 2 — Package allowlist

`build.zig.zon` `.paths` now explicitly enumerates the files shipped to downstream
consumers. **Add `src/array_kernels.zig` to `.paths`** (it's a production module
imported by `container_ops.zig`). **`bench_aa.zig` stays repo-only** — do *not* add
it. Run a lightweight downstream-package build check (a consumer `zig build` against
this package) as part of this chunk's acceptance so a missing path is caught here.

## Task 3 — Compare board additions (`src/bench_croaring.zig`)

Add the balanced scenarios the board was missing:
- two bitmaps, each ~200 containers, all arrays of cardinality 1024–4096, ≥80% key
  overlap; ops: `and`, `andCardinality`, `xor`;
- a skewed variant (16–64 vs 4096) guarding the gallop path against regression.

## Task 4 — `src/bench_aa.zig` — standalone kernel bench

The per-kernel bench, isolated from allocation and container-walk noise. Fully
defined here (it does not exist yet):

- **Build:** `src/bench_aa.zig`, wired as build step **`bench-aa`** in `build.zig`
  mirroring `bench` (link libc, `addBenchmarkPlatformShim`, `ReleaseFast`).
- **Kernels under test come from `array_kernels.zig`** (Task 1) — bench the
  **production** gallop/merge (and later SIMD), never a re-implementation.
- **Kernel enumeration API (required so SIMD can be timed separately).** `bench_aa`
  **cannot import `array_simd.zig`**, and calling the *dispatch* entry point can't
  produce separate SIMD-vs-merge numbers. So `array_kernels.zig` exposes a
  **comptime kernel-descriptor list** (or direct internal bench entry points) — e.g.
  `pub const bench_kernels = .{ .{ "gallop-write", intersectWriteGallop }, .{
  "merge-write", intersectWriteMerge }, … }` — that `bench_aa` iterates to time and
  cross-check each kernel individually. 11-05 **registers its SIMD kernels in this
  list**, which is what "included automatically" means concretely; without the list
  it isn't automatic.
- **Deterministic, exact-cardinality inputs:** fixed-seed `std.Random.DefaultPrng`
  (seed logged), no `Date`/wallclock. Raw uniform u16 draws contain **duplicates**,
  so N draws yield fewer than N distinct values. For each requested `(cardA, cardB)`:
  draw, **sort + dedupe, and refill** until the set is **exactly** the requested
  cardinality. Otherwise "4096×4096" isn't 4096×4096.
- **Correctness check (mandatory):** cross-verify **every kernel available in this
  build** for each input pair before timing; abort on mismatch. In 11-00 that's
  gallop vs merge; **SIMD is included automatically once 11-05 adds it** (the check
  iterates whatever kernels `array_kernels.zig` exposes, not a hardcoded list). By
  shape: **write** kernels must produce **byte-identical output arrays**; the
  **cardinality** kernel must return a value equal to that output's length; the
  **boolean** kernel must equal that output's non-emptiness. This is the safety net
  that makes the SIMD numbers trustworthy.
- **Timing methodology (required — some cases are ~260 ns, so one call per trial is
  pure noise):**
  - Generate/allocate all inputs **and** output buffers **outside** the timed region.
  - Each trial runs the kernel in a **repeat loop** until the batch takes ~1–10 ms;
    report `batch_time / iterations`. Use the **same iteration count** for kernels
    being compared in a case.
  - `std.mem.doNotOptimizeAway` the results (output lengths, booleans, cardinalities,
    **and** the output data) so the loop isn't optimized out.
  - Run the **correctness cross-check outside** the timed region (once per case, not
    per timed iteration).
  - Median of **≥9 trials** per case (matches the evidence methodology). Print
    `case | kernel | ns/op` + ratio vs merge, plus the seed and build target/host.
- **Invocation:** build-only, so two commands — document this in the file header:

  ```bash
  zig build bench-aa -Dcpu=native   # or explicit -Dcpu=x86_64_v3 / baseline
  ./zig-out/bin/bench_aa
  ```

  Alternatively make `bench-aa` a `b.addRunArtifact` run step + a separate build-only
  step; pick one and document it.

## Acceptance

- `array_kernels.zig` exists with the six intersect functions (three gallop + three
  merge); `container_ops.zig` calls the gallop forms — **differential suites green,
  zero behavior change** (`zig build test test64 validate validate64 difftest
  difftest64`).
- `array_kernels.zig` in `build.zig.zon` `.paths`; `bench_aa.zig` not; downstream
  build check passes.
- `bench-aa` builds and runs on the named host; correctness check passes for all
  kernels; balanced + skewed scenarios present on the compare board.
- Baseline numbers recorded (this is the "before" the later chunks are measured
  against).
