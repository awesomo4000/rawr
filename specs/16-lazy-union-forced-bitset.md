<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 16: Forced-bitset lazy-union construction cost

Close the one benchmark op where rawr still trails the reference: `lazyOr+repair
(sparse)`. On the standard M4/NEON comparison it runs ~1.36x (16.54 ms vs 12.14 ms,
~+4.4 ms) — the only tracked operation above the noise floor still on the wrong side
of 1.0. Everything else is at or below parity.

The optimization is entirely within the **forced-bitset** path. Both implementations
are told to force conversion and both do; the work is equivalent, so the gap is
construction/repair *cost*, not a difference in what gets built. Semantics stay
identical — this spec does not change dispatch or any public contract.

## Correcting an earlier framing

An earlier draft of this spec assumed the reference keeps small array unions as arrays
under this flag and proposed routing rawr through its size-selecting `containerUnion`.
That is wrong and is withdrawn:

- The `bitsetconversion` flag is documented as forcing a bitset conversion
  (`vendor/roaring.h:8717`), and the reference performs it unconditionally for
  non-bitset overlaps in both the allocating and in-place paths
  (`vendor/roaring.c:16808`, `vendor/roaring.c:16895`).
- The benchmark passes `true` to both sides (`src/bench_croaring.zig:338` and `:792`),
  so both force conversion. Routing rawr through eager size-selecting union would
  compare size-selecting rawr against forced-bitset reference — improving the number by
  diverging semantics rather than by optimizing equivalent work.

If a size-selecting lazy OR is ever wanted, it belongs behind a separately named
policy/API, benchmarked on its own — not folded into this flag.

## What each forced path does

Both build a bitset for every overlapping (non-bitset) key pair and leave its
cardinality unresolved until repair. The construction differs:

- **Reference** (`vendor/roaring.c:16808`): converts **one** input to a bitset via
  `container_to_bitset`, then `container_lazy_ior`s the second input **in place** into
  that same destination bitset. The first conversion is itself a per-element loop
  (`bitset_container_from_array` → `bitset_container_set`, `vendor/roaring.c:9334`); only
  the *second* input is applied with the bulk `bitset_set_list`. Bitset creation is two
  allocator calls — container struct + words (`vendor/roaring.c:7272`) — same as rawr
  (`src/bitset_container.zig:22`). So: **one destination bitset**, first input filled
  element-by-element, second input bulk, cardinality left as its unknown-sentinel.
- **rawr** (`lazyMergeTwo`, `src/bitmap.zig`): allocates a fresh zeroed
  `BitsetContainer`, then calls `lazyAccumulateIntoBitset` for **both** inputs. For
  array inputs that is a per-element `acc.lazySet(value)` loop
  (`src/bitset_container.zig:299`) for *every* input, and `lazySet` re-stores the
  `cardinality = -1` sentinel on **every element**. rawr has a bulk `setList`
  (`src/bitset_container.zig:195`) that is not used on this path. The divergence from
  the reference is therefore the second input (per-element `lazySet` vs bulk
  `setList`) and the per-element sentinel store — not the first-input fill or the
  allocation count, which match.

`repairAfterLazy` (`src/bitmap.zig:1520`) then runs `computeCardinality()` (popcount of
1024 words) on each such bitset and, when the true cardinality is `<=
MAX_CARDINALITY`, `bitsetToArray` demotes it back to an array. This is the same shape
as the reference's repair.

The sparse corpus makes this the worst case: ~500k random `u32` values dedup across
~65k containers at ~7 elements each (`src/bench_croaring.zig` sparse setup), so every
overlapping pair inflates two ~7-element arrays through a full 8 KB
alloc + zero + per-element accumulate + 1024-word popcount + demote round-trip,
thousands of times.

## Approach — measure first, then optimize the forced path

Semantics are fixed (forced bitset, both sides `true`). The task is to find and remove
rawr's excess cost in that path.

1. **Split the measurement.** Time lazy **construction** and **repair** separately on
   the sparse corpus, for rawr and the reference, both with `bitset_conversion = true`.
   The current bench folds them into one number; the split says whether the cost is in
   accumulation, in repair, or both. **Timing boundaries:**
   - *Construction-only* excludes result destruction — deinit the result outside the
     timed sample.
   - *Repair-only* starts from a fresh, unrepaired result prepared **outside** each
     timed sample, so construction and free do not leak into the repair number.
   - Apply identical setup/teardown to rawr and the reference so allocator and
     deallocation time land on neither side.
2. **Attribute the construction cost.** Within construction, account for allocation +
   zeroing, per-element accumulation, and (in repair) popcount + demotion. Confirm
   where rawr diverges from the reference before changing code.
3. **Candidate optimizations to validate against the split** (implement only what the
   measurement justifies):
   - Apply the second (and subsequent) array input with the existing bulk `setList`
     rather than a per-element `lazySet` loop — this is the concrete divergence from the
     reference, which applies its second input via `bitset_set_list`.
   - Drop the redundant per-element `cardinality = -1` store — invalidate the sentinel
     once per container, not once per bit.
   - Only if the split shows repair dominates: evaluate reducing repair's popcount/demote
     cost. This must **preserve the unknown-cardinality sentinel until repair** — do not
     change the internal lazy-state invariant, and do not add side metadata or extra
     allocation merely to save the repair popcount. If a repair change appears
     warranted, raise it as an explicit follow-up decision rather than folding it in
     here.

## Scope / accurate boundaries

- **`lazyMergeTwo`** (`src/bitmap.zig`) is the hot two-way path and the target.
- **`lazyOrInPlace`** (`src/bitmap.zig:1480`) has no separate dispatch — it calls
  `lazyOr` and replaces the bitmap — so it inherits any construction improvement
  automatically; there is nothing separate to change there.
- **`orMany` does not use this path.** Its `foldManyKey` (`src/bitmap.zig:1928`) is an
  independent n-way fold that always creates a bitset for shared keys. It cannot be
  directly improved by this change; n-way ops are **regression coverage only**.

## Correctness / safety

- No result or representation change: the forced-bitset semantics are preserved
  exactly, on both the by-value and in-place paths. The lazy contract is unchanged —
  the result still requires `repairAfterLazy` before normal use.
- **Add pre-repair representation assertions.** The existing differential tests repair
  both outputs and compare values (`src/diff_test.zig:567`), which erases any
  representation difference before comparison. Add a **controlled sparse array/array
  fixture** and assert, on the pre-repair result:
  - `bitset_conversion = true` → the overlapping container is a **bitset**;
  - `bitset_conversion = false` → the overlapping container is an **array**.

  Use a fixture whose types are known, not a blanket "every non-bitset pair is a
  bitset" claim — that has exceptions (e.g. union with a full run container; a
  size-selected result can also land as a run).
- Existing lazy-or / lazy-xor / in-place / footgun / edge differential cases stay green
  under both `ReleaseSafe` and `ReleaseFast`.

## Acceptance (GO)

- **Establish the noise band empirically** — **five independent process runs** of the
  sparse construction and repair benchmarks, reporting **median and range/IQR** per
  measurement, not an informal target.
- **`lazyOr+repair (sparse)` clears a concrete gate** with `bitset_conversion = true` on
  both sides (apples-to-apples preserved): overall ratio **≤ 1.10x**, or removal of **at
  least 75% of the current excess time** (the ~+4.4 ms over the reference), whichever is
  the cleaner bar to state on the final numbers. Report the construction/repair split
  before and after and the overall before/after ratio.
- **No regression** on the n-way path (`orMany` / `orManyHeap` / `xorMany`,
  `bitwiseOr (dense)`) or elsewhere in the `bench-croaring` comparison, beyond noise.
- **Representation tests + differential tests green** under both build modes.

## Estimate

S–M. The change is a localized accumulation-path optimization, but the deliverable is
the measured before/after (construction vs repair split) plus the new pre-repair
representation assertions, not just the code edit.
