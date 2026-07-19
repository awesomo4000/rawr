<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 14: Benchmark environment header

Small side quest. Every benchmark should print a **CPU / features / build header**
as its first output, so when the output is redirected to a file the numbers carry
the machine + build that produced them. Today a redirected bench file is just a
results table with no record of the target it ran on — useless for comparing runs.

Behavior-only add (bench tooling); no library API change.

## What to print

All values are **comptime** (`builtin.*` + the `array_simd` flags) — which is
correct, because the SIMD kernels are comptime-gated, so the header reports what's
actually compiled into *this* binary. With `-Dcpu=native` that's the host; with an
explicit `-Dcpu=…` it's that target. Say so in the header so a reader isn't misled
into thinking it's runtime host detection.

Fields:
- **Zig version** (`builtin.zig_version`), **optimize mode** (`builtin.mode`),
  **OS** (`builtin.os.tag`), **arch** (`builtin.cpu.arch`).
- **CPU model** (`builtin.cpu.model.name`) — e.g. `apple_m4`, `znver4`, or
  `baseline`/generic for a non-native target.
- **Relevant CPU features present**, arch-gated (only the ones the kernels care
  about, not the full set):
  - x86_64: `sse2`, `ssse3`, `sse4_2`, `avx`, `avx2` (via
    `std.Target.x86.featureSetHas`).
  - aarch64: `neon` (via `std.Target.aarch64.featureSetHas`).
- **Active array-intersect kernel path** — the most useful line: derive from
  `array_simd.has_x86_simd` / `array_simd.has_neon` → `x86-simd` / `neon` /
  `scalar`. Tells the reader at a glance whether the SIMD kernel is even in this
  build.

## Where it lives

A single shared helper `printBenchEnvironment()` in **`src/bench_time.zig`**,
alongside the existing `printRunTimestamp()` (that file is already the shared bench
utility). Every bench `main()` calls it as its **first** output:

- `bench.zig`, `bench_croaring.zig`, `bench_aa.zig`, `bench_allocators.zig`.
- Future `bench64` / `bench-compare64` (spec 10-21) use the same helper.

> Coupling note: `bench_time.zig` is also imported by `validate*`/`difftest*` (for
> the timer shim). Importing `array_simd.zig` into `bench_time` for the two
> `has_*` bools pulls it into those too — harmless (pure Zig, comptime bools, no
> C). If we'd rather not couple them, print the arch+features lines from
> `bench_time` and let each bench print the one kernel-path line itself. Implementor's
> call; the simple import is fine.

## Format

Comment-prefixed (`#`) so the header reads as metadata above the results and
survives being pasted/committed. Clinical — facts only, no commentary:

```
# rawr bench env
# zig 0.16.0 | ReleaseFast | macos aarch64
# cpu: apple_m4 | features: neon
# array-intersect kernel: neon
```

(x86 example: `# cpu: znver4 | features: sse2 ssse3 sse4_2 avx avx2` /
`# array-intersect kernel: x86-simd`.)

## Acceptance

- Each of the four bench programs prints the header block as its first output;
  redirecting to a file captures cpu model + gated features + active kernel path +
  zig/mode/os/arch.
- Values reflect the compiled target (documented as such in the header).
- Builds green across the target matrix (x86-64 baseline/AVX, aarch64, windows) —
  the feature queries are comptime and total on every arch.
- No library/API change; `zig build test` unaffected.
