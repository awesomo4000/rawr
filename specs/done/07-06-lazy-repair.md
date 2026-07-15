<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 07-06: Lazy union/xor + repair

Sixth piece of the [CRoaring parity effort](07-parity-inventory.md). Closes the
n-way performance gap from `07-05` and adds the lazy primitives CRoaring exposes.

**Baseline to beat** (from `07-05`, 32 mixed bitmaps): `orMany` ~85× slower,
`xorMany` ~40× slower than CRoaring. The gap is eager per-step normalization: the
k-way fold runs ~`M-1` `containerUnion`/`containerXor` per key, each allocating,
popcounting, and re-checking container type. CRoaring instead accumulates without
maintaining cardinality or container type, then does **one** repair pass.

This is the design-heavy chunk — treat the shapes below as the intended approach,
open to refinement during implementation.

## Two levers (this chunk = lever 1; lever 2 is `07-06b`)

1. **Lazy fold + single repair** — skip cardinality maintenance and container
   normalization during the fold; fix it all up once at the end. The dominant
   win. **This spec.**
2. **Heap-based k-way cursor** — replace `nextManyKey`'s O(N·distinct_keys) linear
   min-scan with a priority queue (CRoaring's `or_many_heap`). The remaining cost
   for large N. **Deferred to `07-06b`.**

## What "lazy" means in rawr

rawr's bitset container already has a lazy-cardinality state: `cardinality == -1`
means "unknown, recompute on demand" (`getCardinality`/`computeCardinality`). So a
bitset with `cardinality == -1` *is* a lazy container. Lazy ops exploit this:
accumulate into a bitset with raw word operations, leave `cardinality == -1`, and
**don't** demote/normalize — defer all of that to repair.

## Task 0 — Wrapper decls (parity)

```c
roaring_bitmap_t* roaring_bitmap_lazy_or(const roaring_bitmap_t*, const roaring_bitmap_t*, bool bitsetconversion);
void              roaring_bitmap_lazy_or_inplace(roaring_bitmap_t*, const roaring_bitmap_t*, bool bitsetconversion);
roaring_bitmap_t* roaring_bitmap_lazy_xor(const roaring_bitmap_t*, const roaring_bitmap_t*);
void              roaring_bitmap_lazy_xor_inplace(roaring_bitmap_t*, const roaring_bitmap_t*);
void              roaring_bitmap_repair_after_lazy(roaring_bitmap_t*);
```

## Task 1 — `repairAfterLazy` (the normalization pass)

```zig
pub fn repairAfterLazy(self: *Self) !void   // !void: demotion allocates
```
It's `!void`, not `void` — demoting a bitset to an array allocates (via
`self.allocator`), which can fail. (Note this diverges from the C
`roaring_bitmap_repair_after_lazy` signature, which is `void`; the public parity
wrapper around it stays `void`, but rawr's own method propagates the alloc error.)
Every caller uses `try`.

A single walk over `self.containers[0..size]` that restores all invariants the
lazy ops deferred. It must tolerate **any** container — normal array/run/bitset as
well as lazy bitsets — and be idempotent on an already-clean bitmap (single-input
keys are cloned as-is and pass through untouched):
- For each container, ensure cardinality is computed (bitset left at `-1` →
  `computeCardinality`; array/run already valid).
- **Demote** a bitset whose cardinality is now `≤ ArrayContainer.MAX_CARDINALITY`
  to an array (the conversion lazy skipped).
- **Drop** any container that ended up empty (cardinality 0), compacting
  `keys`/`containers` — same ghost-container discipline as the set ops.
- Recompute `self.cached_cardinality`.

rawr's repair right-sizes **array↔bitset only** — it does not create run
containers (run optimization stays an explicit `runOptimize`). Note CRoaring's
`repair_after_lazy` *does* convert to efficient containers including runs, so this
is **not** exact representation parity — which is fine, because the differential
check is `assertSameValues` (semantic), not byte-identity.

This walk is also the public parity API (`roaring_bitmap_repair_after_lazy`).

## Task 2 — Lazy primitives + make the n-way fold lazy

### Lazy accumulation primitive

The fold needs an "OR/XOR a container into a bitset accumulator **without**
maintaining cardinality or normalizing." Add the minimal container-level support:
- a raw bitset word-OR / word-XOR that skips the popcount (the existing
  `simdBitsetOp` computes cardinality inline — lazy needs a variant that doesn't,
  or a flag),
- raw "set bits of an array" / "set ranges of a run" into a bitset accumulator
  without cardinality tracking,
- leave the accumulator at `cardinality == -1`.

### Apply it in `foldManyKey` (`manyMerge`)

Keep the k-way structure from `07-05`; change only the per-key fold:
- **single-input key** → clone as-is (cardinality already known; passes through
  repair untouched).
- **multi-input key** → **always accumulate into a bitset** (the
  `bitset_conversion = true` behavior — this forced-bitset mode *is* the perf
  fix): build a bitset accumulator and lazily OR/XOR every same-key container into
  it (raw, no per-step cardinality/normalize), leaving it `cardinality == -1`.
- After `manyMerge` finishes building the result, call `try repairAfterLazy`
  **once**.

Result correctness is unchanged from `07-05` (same set), so the existing
`orMany`/`xorMany` differential tests must keep passing — this is a
behavior-preserving perf rewrite of the fold.

## Task 3 — Public lazy APIs (parity)

Expose the 2-way lazy ops CRoaring has, built on the same primitives:

```zig
pub fn lazyOr(self: *const Self, allocator, other: *const Self, bitset_conversion: bool) !Self
pub fn lazyOrInPlace(self: *Self, other: *const Self, bitset_conversion: bool) !void
pub fn lazyXor(self: *const Self, allocator, other: *const Self) !Self
pub fn lazyXorInPlace(self: *Self, other: *const Self) !void
```

These behave like `bitwiseOr`/`bitwiseXor` but leave the result in a **lazy
state** — cardinality and container types are **not valid** until `try
repairAfterLazy()` is called. Document this loudly on each (it's a footgun; the
intended use is "do several lazy ops, repair once").

`bitset_conversion` semantics for rawr:
- **`true`** → forced bitset accumulation for matched keys (always merge into a
  bitset). Right for bulk/dense work; **the n-way fold uses this mode.**
- **`false`** → only use a bitset accumulator when at least one operand is already
  a bitset; otherwise keep the cheaper eager-ish container ops for that key.

Do **not** change the eager `bitwiseOr`/`bitwiseXor`.

## Task 4 — Tests

- **Lazy == eager (pure rawr):** `a.lazyOr(b, …)` then `repairAfterLazy` equals
  `a.bitwiseOr(b)`; same for xor. Over the mixed generator, both run-optimized and
  not.
- **Differential:** `lazyOr` + `repairAfterLazy` vs `roaring_bitmap_lazy_or` +
  `roaring_bitmap_repair_after_lazy` by `assertSameValues` (representation may
  differ; semantic equality, as elsewhere). And the existing `07-05`
  `orMany`/`xorMany` differential tests must still pass after the fold rewrite.
- **`validate()` after repair:** a repaired bitmap must pass `validate()` (correct
  container types, cardinalities, no empties) — reuse the validator from spec 06.
- **Repair is idempotent / no-op on an already-clean bitmap** (calling it on a
  normally-built bitmap — array/run/bitset containers — changes nothing
  observable and doesn't error).
- **`serialize` after repair works:** repair a lazily-built bitmap, then
  `serialize` it. `serialize` validates internally, so this catches a repair that
  forgot to restore an invariant, on a user-facing path.
- **Lazy footgun (pinned contract):** build a lazy result that contains a lazy
  bitset (cardinality `-1`, un-normalized) and assert that serializing/validating
  it **before** repair does not succeed cleanly. We don't guarantee *every* lazy
  result fails pre-repair, but one pinned case documents that lazy state isn't
  usable until repaired.
- **Edge:** lazy ops that produce empties (xor of equal containers) → repair drops
  them; lazy accumulation that stays large → stays a bitset.

## Task 5 — Benchmark (the point of this chunk)

Re-run the `07-05` `orMany`/`xorMany` benches (same 32-mixed setup) and record
before/after against the committed `07-05` baseline rows — **`orMany` 85.78×,
`xorMany` 40.25×** (unless bench noise has shifted them). Target: lazy fold +
single repair should recover most of the gap — expect low single-digit× or better. The residual (from the linear `nextManyKey`
scan) is what `07-06b`'s heap closes; note it rather than chasing it here. Also
worth a quick bench of the public `lazyOr`+`repair` vs eager `bitwiseOr` to
confirm lazy isn't slower for the 2-way case it's not meant for (it may be — lazy
is a bulk optimization; just don't regress the eager path, which is untouched).

## Acceptance criteria

1. `repairAfterLazy` is `!void`, restores cardinalities, demotes small bitsets to
   arrays, drops empties, fixes `cached_cardinality`; tolerates any container type;
   a repaired bitmap passes `validate()` and `serialize`; idempotent on clean
   bitmaps. Callers use `try`.
2. `orMany`/`xorMany` use the lazy fold + single repair; their `07-05`
   differential tests still pass (behavior-preserving).
3. Public `lazyOr`/`lazyOrInPlace`/`lazyXor`/`lazyXorInPlace` exist with the
   lazy-state contract documented; lazy+repair matches eager and the CRoaring
   lazy oracle (`assertSameValues`).
4. `orMany`/`xorMany` benches recorded before/after with a substantial recovery
   from the ~85×/~40× baseline; eager `bitwiseOr`/`bitwiseXor` unchanged.
5. No leaks; `zig build test`, `validate`, `difftest`, `bench-compare` pass.

## Notes

- Heap-based k-way cursor (`or_many_heap`/`xor_many_heap`) → `07-06b`.
- Mark `lazy_or(_inplace)`, `lazy_xor(_inplace)`, `repair_after_lazy` ✅ in the
  [inventory](07-parity-inventory.md) when done; leave the `_many_heap` rows for
  `07-06b`.
