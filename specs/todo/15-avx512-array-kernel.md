<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 15: AVX-512 array-intersect kernel (prototype / experiment)

Draft, parked. A possible wider array∩array kernel on top of the shipped 128-bit
SSE/NEON one (spec 11). **Prototype-first, decision-gated** — the deliverable is a
measured answer to "does 512-bit beat the 128-bit kernel enough to justify a third
per-arch code path?", not a commitment to ship. (The allocation-work initiatives that
once out-prioritized this — specs 13/16/17 — have since concluded, so this is no longer
sequenced behind them; it is a standalone parked experiment.)

## Why it's uncertain up front

512-bit is the width where intersection *compaction* vectorizes cleanly, but the
gain is not guaranteed:

- **`vpshufb` can't compact across 128-bit lanes**, so AVX2 (256-bit) is a dead end
  for this. The AVX-512 enabler is **`vpcompressw`** (AVX512BW + AVX512VBMI2), which
  compacts matching lanes across the whole 512-bit register in one instruction.
- **`vp2intersect`** (one-instruction cross-vector match) is not broadly available —
  absent on current AMD parts and removed from Intel after Tiger Lake — so the
  prototype must use **compare-any + `vpcompressw`**, not the `vp2intersect` fast
  path.
- **Some AVX-512 implementations double-pump 512-bit ops on a 256-bit datapath**, so
  the throughput win over the 128-bit kernel may be *modest* (fewer instructions and
  a cleaner compaction, not 2× raw width). Only a bench decides.

## Approach

- **512-bit block = 32×u16.** Compare-any: broadcast each lane of B against the A
  block, OR the equalities — the same shape as the 128-bit kernel, widened. On
  AVX-512 the compares land directly in a **k-mask register** (no separate movemask
  step), and the k-mask feeds `vpcompressw` to write only the matching lanes.
- **Scope: write + cardinality intersect only** (matching 11-05); boolean stays on
  the gallop/merge dispatch.
- Gate comptime on `.avx512f and .avx512bw and .avx512vbmi2`. It likely does **not**
  slot into the existing `(shuffle, movemask)` skeleton cleanly (k-masks + compress
  differ from pshufb/tbl + movemask) — expect a separate kernel function registered
  alongside the others, not a reuse of the skeleton. Baseline builds without the
  features keep the existing dispatch.

## How to evaluate

- Add it to `array_kernels.zig`'s `bench_kernels` descriptor list so `bench_aa`
  **cross-checks it byte-identical** against gallop / merge / 128-bit for every input
  pair, and times it separately.
- **Differential suites are the safety net** for a hand-SIMD kernel — run long, under
  both `ReleaseSafe` and `ReleaseFast`, on an AVX-512-enabled build target.
- **Bench vs the 128-bit kernel** on the balanced-array corpus (spec 11-00) to get the
  win number. Compile-check a baseline target (no AVX-512 → falls back) and an
  AVX-512 target.
- **Decision:** if the measured gain over the 128-bit kernel is marginal, keep this
  parked — the prototype's whole job is to produce that number cheaply before any
  commitment to a maintained third path.

## Estimate

S–M for the prototype + bench. The kernel itself is small; the deliverable is the
go/no-go measurement, not shipped code.
