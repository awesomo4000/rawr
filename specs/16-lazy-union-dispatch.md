<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 16: Size-conditional lazy-union dispatch

Close the one remaining benchmark op where rawr trails the reference: `lazyOr+repair
(sparse)`. On the standard M4/NEON comparison it runs ~1.36x (16.54 ms vs 12.14 ms,
~+4.4 ms) — the only operation above the noise floor still on the wrong side of 1.0.
Every other tracked op is at or below parity.

This is **not** a kernel or hardware gap. rawr's own eager `bitwiseOr (sparse)` is
already at parity (~0.98x) on the *same input*. The lazy path loses because it does
strictly more work than the eager path for the same result. The fix reuses machinery
that already exists in the tree.

## Root cause

`lazyMergeTwo` (`src/bitmap.zig`) decides per overlapping key pair:

```zig
const use_lazy_bitset = op == .xor or bitset_conversion
    or isBitsetContainer(c_a) or isBitsetContainer(c_b);
```

When the caller passes `bitset_conversion = true`, **every** overlapping array∪array
pair is forced down the bitset-accumulator branch: allocate and zero a fresh 8 KB
`BitsetContainer`, set bits one at a time (`acc.lazySet` scalar loop), then append the
bitset. `repairAfterLazy` later runs `computeCardinality()` (popcount of 1024 words)
on each such container and, when the true cardinality is `<= MAX_CARDINALITY`,
`bitsetToArray` demotes it straight back to an array.

The sparse corpus is the pathological case for this: ~500k random `u32` values dedup
across ~65k containers at ~7 elements each. Overlapping pairs union two ~7-element
arrays into a ~15-element result — three orders of magnitude under the 4096 array/
bitset threshold — yet each one pays a full 8 KB alloc + zero + per-element bit-set +
1024-word popcount + demote round-trip. Thousands of times.

The reference implementation's equivalent flag promotes to bitset only where the
result is actually dense; small array unions stay arrays and its repair barely touches
them.

## The fix is already in the codebase

The `else` branch immediately below the forced-bitset branch calls
`ops.containerUnion` → `arrayUnionArray` (`src/container_ops.zig`), which **already**
selects the output type by size:

```zig
if (max_card > ArrayContainer.MAX_CARDINALITY) { ...bitset... }
else { ...array merge... }
```

That is the exact path the eager `bitwiseOr` uses — the one already at parity. The
lazy path bypasses it whenever `bitset_conversion` is set.

Change: treat `bitset_conversion` as a **size-conditional hint, not an unconditional
mandate**. For an overlapping pair whose inputs are not already bitsets, route through
`containerUnion` and let it promote to bitset only when the union genuinely crosses the
array threshold. Keep the forced bitset-accumulator branch for the cases that still
need it (an input already a bitset; `xor`, which the lazy contract keeps on the bitset
path). Small sparse unions then stay arrays, and `repairAfterLazy` passes them straight
through (the `.array` arm already does nothing but sum cardinality).

Net effect: `lazyOr+repair (sparse)` collapses toward the eager `bitwiseOr (sparse)`
number, and the dense n-way case — the reason `bitset_conversion` exists — is
untouched, because those unions legitimately exceed the threshold and still become
bitsets that defer their cardinality.

## Scope / tasks

1. **`lazyMergeTwo`** — make the bitset-accumulator branch size-conditional as above.
   Preserve exact behavior when either input is already a bitset and for `xor`.
2. **`lazyOrInPlace`** (`src/bitmap.zig`) — apply the same dispatch change to the
   in-place path so both entry points behave identically.
3. **Callers of the flag** — audit `lazyOr` / `lazyOrInPlace` call sites (including the
   internal n-way fold used by `orMany`) for any that depend on the old "force every
   overlap to bitset" behavior. The public `bitset_conversion` parameter's observable
   meaning shifts from "force" to "convert when beneficial"; document it at the
   definition site.

## Correctness / safety

- No result changes. An array produced by `containerUnion` carries an exact
  cardinality, which is a valid pre-repair state; `repairAfterLazy` already handles
  array, bitset, and run containers and recomputes the total regardless of type.
- The lazy footgun contract is unchanged: the result still requires `repairAfterLazy`
  before normal use (the cached bitmap cardinality is still `-1` until repair).
- The differential harness is the safety net. `assertLazyOrCase` /
  `assertLazyXorCase` (`src/diff_test.zig`) compare the repaired output — by value and
  in-place — against the reference oracle's `repair_after_lazy` result. These must stay
  green, and the lazy footgun / edge cases must continue to pass.

## Acceptance (GO)

- **`lazyOr+repair (sparse)` lands within the benchmark noise band of eager
  `bitwiseOr (sparse)`** on the standard M4/NEON comparison (target ≈ parity, ~1.0x;
  concretely well under the current 1.36x). Report the before/after ratio.
- **No regression on the dense n-way path**: `orMany` / `orManyHeap` / `xorMany` and
  `bitwiseOr (dense)` stay within noise of their current numbers.
- **Differential tests green**: lazy-or, lazy-xor, in-place variants, footgun, and
  edge cases all pass under both `ReleaseSafe` and `ReleaseFast`.
- No beyond-noise regression elsewhere in the `bench-croaring` comparison.

## Estimate

S. One dispatch predicate in two places plus a caller audit; the union and repair
machinery already exists. The deliverable is the closed benchmark gap plus green
differential tests.
