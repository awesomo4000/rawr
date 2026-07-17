<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 11-01: Ratio dispatch for array∩array (portable)

Second chunk of [array-kernel performance](11-array-kernel-perf.md). The single
highest value/risk item: replace rawr's unconditional gallop with CRoaring's
ratio dispatch, fixing a measured regression on **every** architecture with **zero
SIMD**. Behavior-preserving (bit-identical results); this only changes *which*
kernel runs.

**Dependency order:** after [11-00](11-00-kernel-extraction-bench.md) — the gallop
and merge kernels already live in `array_kernels.zig`, so this chunk is
**dispatch-only** (no extraction).

## The regression

rawr made gallop the *only* array∩array path (`container_ops.zig:707`). Gallop is
right for skewed inputs but loses ~1.7× to a plain branchless merge at balanced
ratios. CRoaring dispatches: **balanced → linear merge; highly skewed → gallop.**

## Change — wire dispatch for all three shapes

For each of the three intersect shapes extracted in 11-00 (write / cardinality /
boolean), add a dispatch entry point in `array_kernels.zig` that picks gallop vs
merge by the skew threshold, and point `container_ops.zig` at it:

```zig
const SKEW_THRESHOLD = 64; // CRoaring's value; "subject to tuning"

if (@as(u32, small.len) * SKEW_THRESHOLD <= big.len) {   // NOTE: inclusive — see below
    // gallop (existing kernel)
} else {
    // merge (the branchless walk added in 11-00; same shape as arrayDifferenceArray)
}
```

**Inclusive boundary (`<=`, not `<`).** At exactly 1:64 the evidence table has
merge at ~13.0 µs vs gallop ~2.6 µs — merge is ~5× *slower* there. A `<` boundary
would pick merge at 1:64 and violate this chunk's own "no regression at ≥1:64 skew"
acceptance. `<=` keeps the 1:64 case on gallop, matching the data.

Notes:
- `arrayDifferenceArray` / `arrayUnionArray` already use branchless merge; this makes
  the intersect family consistent with them.
- The **boolean** variant's gallop is correct at any skew (it early-exits); keep the
  threshold dispatch for the balanced case anyway.
- When [11-05](11-05-x86-simd.md) lands, the balanced arm becomes
  `if (comptime has_simd) vec16 else merge` — one more branch in the same dispatch.
- Optional symmetric skew check on `arrayDifferenceArray` — low value, measure
  first, implementor's call.

## Acceptance

- Differential suites green (`zig build test test64 validate validate64 difftest
  difftest64`) — bit-identical results.
- `bench-aa` (from 11-00): **≥1.5× at 1024×1024**, **no regression at ≥1:64 skew**,
  under the fixed build config.
- Compare board's balanced `and` improves ~1.6–1.7×.
