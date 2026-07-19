<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 11: Array-kernel performance (umbrella)

Perf work identified by kernel-level benchmarking against CRoaring.
**Behavior-preserving throughout, no container-ABI change:** public API, results,
and wire format unchanged; replacement kernels must be bit-identical to the current
ones (the reference bench cross-verifies kernel outputs — keep that check). A purely
*computational-throughput* effort — container-layout changes are a separate track
and out of scope here.

## Differential suites (referenced by every chunk)

Every chunk must pass, before merge, what the chunks call **the differential
suites**:

```
zig build test test64 validate validate64 difftest difftest64
```

(`validate`/`validate64` = CRoaring interop round-trips; `difftest`/`difftest64` =
randomized differential rigs; `test`/`test64` = unit + property tests.)

## Package allowlist

`build.zig.zon` `.paths` explicitly enumerates the files shipped to downstream
consumers, so **each new production module must be added when introduced** and a
lightweight downstream-package build check run in that chunk:

- `src/array_kernels.zig` → added in [11-00](11-00-kernel-extraction-bench.md).
- `src/array_simd.zig` → added in [11-05](11-05-x86-simd.md).
- `src/bench_aa.zig` → **repository-only, never in `.paths`** (a bench tool, not
  shipped).

## Build configuration for performance thresholds

All perf numbers/thresholds are measured under a **fixed config**, else they mean
nothing:

- **Mode:** `ReleaseFast` (correctness also under `ReleaseSafe`). Caveat: the
  `validate*`/`difftest*` modules are hardcoded to `ReleaseFast`, so
  `-Doptimize=ReleaseSafe` does not rebuild them — the SIMD chunk (11-05) specifies
  the build wiring or the exact direct command that makes "green under ReleaseSafe"
  real.
- **Target:** portable chunks use an explicit baseline (`-Dcpu=baseline`/`x86_64_v2`)
  so the merge path is what's measured; SIMD chunks use an explicit feature target
  (`-Dcpu=x86_64_v3`, or `-Dcpu=native` on the recording host). **State which in the
  result.**
- **Authoritative machine:** each number names the host it came from (e.g. "x86-64
  AVX2, Zen 4" / "Apple M-series"). Thresholds are checked on that host class; others
  are informational.

## Background / evidence

Standalone kernel bench (x86-64 AVX2, uniform random u16 draws from the 64K container
domain, median of 9 trials, verified identical output):

| scenario        | rawr-gallop | branchless merge | CRoaring vec16 (AVX2) | skewed gallop |
|-----------------|------------:|-----------------:|----------------------:|--------------:|
| 4096×4096 (1:1) |     43.2 µs |          25.3 µs |                 7.2 µs |       49.0 µs |
| 1024×1024 (1:1) |     10.5 µs |           6.3 µs |                 1.1 µs |       10.8 µs |
| 256×256 (1:1)   |      2.5 µs |           1.6 µs |                0.26 µs |        2.3 µs |
| 1024×4096 (1:4) |     16.6 µs |          15.9 µs |                 3.2 µs |       20.1 µs |
| 256×4096 (1:16) |      6.6 µs |          13.6 µs |                 2.2 µs |        7.2 µs |
| 64×4096 (1:64)  |      2.6 µs |          13.0 µs |                 2.0 µs |        2.2 µs |
| 16×4096 (1:256) |     0.67 µs |          12.2 µs |                 1.8 µs |       0.33 µs |

Findings, confirmed against vendored CRoaring (`vendor/roaring.c:6930`,
`array_container_intersection`):

1. CRoaring **never gallops unconditionally** — gallop only when `card_small * 64 <
   card_big`, else SSE `intersect_vector16` on x86 / scalar merge elsewhere. rawr made
   gallop the only path (`container_ops.zig:707`), losing ~1.7× to a branchless merge
   at balanced ratios **everywhere**, and 5–10× to the SIMD kernel on x86. Rule:
   **balanced → merge; highly skewed → gallop.**
2. CRoaring's kernel algorithms (shuffle-table compaction, threshold=64 *) are the
   reference to port — no better-known alternative.
3. CRoaring's non-x64 balanced path is plain scalar — there's no NEON array kernel
   to port from, so 11-06 is a from-scratch kernel for ARM targets.

## Chunk plan (execution order)

Numbering preserves the established identifiers (gap at 11-04, which was reassigned
to the separate container-layout track); execution order is the order below. Each
chunk states its own dependencies.

| # | chunk | deps | size | platforms |
|---|---|---|---|---|
| [11-00](11-00-kernel-extraction-bench.md) | kernel extraction + bench corpus + `bench-aa` | — | S | all |
| [11-01](11-01-ratio-dispatch.md) | ratio dispatch (uses extracted kernels) | 11-00 | S | all |
| [11-02](11-02-array-bitset-loops.md) | array→bitset conversion loops | 11-00 | S | all |
| [11-03](11-03-findkey.md) | `findKey` lookup tuning | 11-00 | XS | all |
| [11-05](11-05-x86-simd.md) | x86 SIMD `vector16` | 11-00, 11-01 | M | x86-64 |
| [11-06](11-06-neon-simd.md) | aarch64 NEON `tbl` | 11-05 | S atop 11-05 | aarch64 |

**Framing:** this is *fixing our own regression + idiomatic perf* on rawr's own
terms. CRoaring is the correctness oracle and the perf yardstick; the kernels are
ports of known-best algorithms. Comparisons are cited as facts (ratios, oracle
results), not as a scoreboard.
The SIMD tier (11-05/06) is a deliberate, heavily-fuzzed opt-in — a different risk
class (hand asm, per-arch maintenance) weighed against the "idiomatic Zig, clean
API" goal.

## Out of scope

- Runtime CPU dispatch (per-target builds cover the port matrix).
- AVX-512 kernels (`vp2intersect` etc.) — revisit after 11-05; note the LLVM
  feature-detection issues that motivated disabling CRoaring's AVX512 in the compare
  bench.
- Container-layout / single-allocation changes — a separate track.
- `SKEW_THRESHOLD` tuning beyond confirming 64 is sane on the board. *

---

\* **Threshold outcome (as implemented).** The scalar merge-vs-gallop crossover
stayed 64, but 11-05/11-06 introduced a *new* decision — SIMD-vs-gallop — with a
different, measured crossover per architecture. As shipped: **scalar 64, x86 SIMD
12, AArch64 NEON 40** (`array_kernels.zig`). The write/cardinality dispatch uses the
arch-appropriate SIMD threshold; the boolean path (no SIMD) keeps 64. This is the
"record the crossover" outcome the SIMD chunks' acceptance asked for, not a change to
the scalar-64 baseline.
