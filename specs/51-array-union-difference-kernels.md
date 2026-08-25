<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 51: Array union and difference kernels

**Target.** The one real-data gap that reproduced on both hosts (spec 50-02): `wikileaks-noquotes`
pairwise OR at **2.586x** (M4) / **2.682x** (Zen 4), ANDNOT at **2.005x** / **4.073x**.

**Diagnosis first. No production change until a prototype earns it.**

## 1. A specific hypothesis, from reading the source

**rawr vectorizes intersection only.** `src/array_simd.zig` exports exactly four kernels:

```
intersectWriteX86   intersectCardX86   intersectWriteNeon   intersectCardNeon
```

`src/array_kernels.zig` adds scalar intersect variants and a `gallopSearch` selector. There is **no union
kernel and no difference kernel anywhere**. `arrayUnionArray` (`container_ops.zig:509`) is a scalar
branchless merge; `arrayDifferenceArray` (`:996`) is its counterpart.

**CRoaring vectorizes all three**: `intersect_vector16`, `union_vector16`, `difference_vector16`
(`vendor/roaring.c`).

**And the corpus that shows the gap is entirely array containers.** From 50-01's histograms:

| dataset | array | bitset | run | median card | OR gap (M4 / Zen 4) |
| --- | ---: | ---: | ---: | ---: | --- |
| `wikileaks-noquotes` | **1,892** | 0 | 0 | **280** | **2.586x / 2.682x** |
| `census1881` | 1,459 | 5 | 0 | 4 | 1.496x / **0.549x** |
| `uscensus2000` | 2,221 | 0 | 0 | 2 | 0.583x / 0.843x |

The pattern fits: rawr wins AND where it has a tuned kernel, loses OR and ANDNOT where it does not, and
the effect disappears on corpora whose arrays are too small for kernel choice to matter.

**This is a hypothesis with a mechanism, not a proven cause.** It must be measured before anything is
built.

## 2. Why this is not the parked kernel work

Two prior kernel efforts should not be read as precedent against this one:

- **Spec 34 (NO-GO)** unrolled an *existing* kernel for `select`. The row closed later through Run-header
  locality instead. That says loop shape did not help, not that a missing kernel is fine.
- **Spec 15 (parked)** asks whether *wider* SIMD beats the existing 128-bit intersect. Also about an
  existing kernel.

**This spec is about a kernel that does not exist at all**, where the current path is scalar and the
reference is vectorized. Different proposition, and the measured gap is 2-4x rather than a few percent.

## 3. Stage 1: confirm the mechanism before writing a kernel

**Do not start with SIMD.** Establish first that the array union and difference *kernels* account for the
gap, rather than container dispatch, allocation, or result sizing.

Extend the existing real-data harness (spec 50) with a **kernel-level microbenchmark** over the actual
`wikileaks-noquotes` container pairs:

- extract the array pairs that pairwise OR and ANDNOT actually visit;
- time **rawr's scalar merge** against **CRoaring's `union_uint16` / `union_vector16` and
  `difference_vector16`** on identical inputs;
- report ns per output element and the input-size distribution.

**Gate:** the kernel-level ratio must reproduce the end-to-end ratio to within a stated tolerance. If
rawr's scalar merge is close to CRoaring's vectorized union at these sizes, **the kernel is not the cause
and this spec stops** — the gap is elsewhere and building SIMD would be chasing the wrong thing.

Record the **input-size distribution**, because it decides whether a vectorized kernel can help at all: a
SIMD union has a fixed setup cost and loses on short inputs.

## 4. Stage 2: prototype, gated

Only if Stage 1 confirms the kernel is the cause.

- Implement **scalar-improved** union and difference first if the profile suggests it (galloping when
  sizes are skewed, matching what `intersectWriteGallop` already does for AND). **A scalar win needs no
  new per-arch code and should be tried before SIMD.**
- Then, and only if scalar is insufficient, a vectorized `unionWrite` / `differenceWrite` pair.

**Cost to weigh explicitly:** a SIMD union plus difference means **four new per-arch kernels** (x86 and
NEON for each), roughly doubling `array_simd.zig`. zroar declined per-arch paths entirely; spec 15 has
stayed parked over the same cost. That is a real maintenance and correctness surface, and the decision
belongs to the owner.

## 5. Measurement

- **Canonical real-data protocol from spec 50**: one process per cell, 5 processes, 1 warmup + 7 timed
  cycles, aggregate medians, semantic digests, cross-host audit.
- **Both hosts.** The gap is one of the few findings that held on both, and it must be shown to close on
  both.
- **All three corpora.** `uscensus2000` and `census1881` are the controls: a change that helps
  `wikileaks` while hurting them is not a win.
- **Parity board unaffected** and re-run to confirm it. The board's array rows are synthetic and a kernel
  change touches them.

## 6. Gates

- **Stage 1:** kernel-level ratio reproduces the end-to-end ratio, or **stop**.
- **Stage 2:** `wikileaks-noquotes` OR and ANDNOT improve materially on **both** hosts, with
  non-overlapping ranges, and neither control corpus regresses beyond 5%.
- **No parity-board row regresses** beyond the spec-28 layout tolerance.
- Correctness unchanged: semantic digests still match CRoaring across all 42 cells.

## 7. Out of scope

- The other two 50-02 findings, each of which deserves its own spec:
  - **`census1881` serialize + deserialize**, 1.640x M4 / 2.824x Zen 4, consistent on both hosts;
  - **`toArray` on Zen 4**, 2.546x and 2.237x, against 0.918x / 0.837x on M4.
- Wider SIMD for the existing intersect kernel (spec 15).
- Bitset or run kernels. The corpus that shows this gap has neither.

## 8. Estimate

**M** — Stage 1 is a focused microbenchmark reusing the spec 50 harness. Stage 2 depends on whether
scalar improvements suffice.

## 9. Chunking

Not chunked. Stage 1 and Stage 2 are the natural split, and **Stage 2 should not be written until Stage 1
reports**, since its content depends on what the profile shows.
