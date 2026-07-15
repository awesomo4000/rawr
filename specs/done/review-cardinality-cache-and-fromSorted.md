<!-- SPDX-License-Identifier: MPL-2.0 -->

# Review: Cardinality Caching + fromSorted Bug

## Cardinality Cache Review

The cardinality caching implementation is solid overall. Benchmarks confirm it: `0.00ms` vs CRoaring's `0.04ms`. The fundamental design — `i64` field, `-1` sentinel for "unknown", incremental maintenance on add/remove, invalidation on bulk ops — is right.

### What's correct

- **add()**: `if (added and self.cached_cardinality >= 0) self.cached_cardinality += 1` — perfect. Guards both the "already existed" case and the "cache unknown" case.
- **addRange()**: accumulates from the return value of `addRangeToChunk`, which already computes the delta. Clean.
- **remove()**: same pattern as add, decrement on actual removal.
- **In-place ops** (OrInPlace, AndInPlace, DiffInPlace, XorInPlace): all invalidate at the top of the function. Correct — the result cardinality is expensive to track incrementally during a merge.
- **clone()**: copies the cache. Correct.
- **init()**: sets cache to 0. Correct — empty bitmap has 0 elements.
- **deserialize()**: sums from the header cardinalities array. This is free (data already parsed) and correct — the header stores `cardinality - 1` per container, and the code does `@as(u32, desc_buf[i * 2 + 1]) + 1`.

### Minor issues (non-blocking)

**1. RunContainer.add always invalidates instead of incrementing**

Every successful `add()` path in RunContainer sets `self.cardinality = -1`. Since `add` inserts exactly 1 element, these could all be `if (self.cardinality >= 0) self.cardinality += 1`. Same logic as BitsetContainer and ArrayContainer already use. Not a bug — invalidation is always safe — but it forces a recompute on the next `getCardinality()` call, which walks all runs.

**2. addRangeNewContainer sets `rc.cardinality = -1` when the answer is known**

In `addRangeToChunk`, when creating a brand-new RunContainer with a single run `[start, end]`, the cardinality is trivially `range_size`. Setting -1 forces a recompute. Same thing in the array→run conversion path in `addRangeToContainer`.

**3. Non-allocating set ops return -1 on results**

`bitwiseOr`, `bitwiseAnd`, etc. all set `result.cached_cardinality = -1` on the newly created bitmap. These *could* accumulate container cardinalities during the merge loop essentially for free. Low priority — the caller usually doesn't immediately query cardinality on a freshly computed result.

**4. runOptimize invalidation**

The comment says "cardinality doesn't actually change" but invalidates anyway. This is fine defensively. If you trust the invariant, you could skip it. Personal preference.

None of these are worth fixing now. They're all "safe but suboptimal" — the cache falls back to recompute, which is the pre-cache behavior. Flag them for a future pass if profiling shows they matter.

### Const-correctness

`cardinality()` changed from `*const Self` to `*Self`. This is the right call — caching requires mutation. The ripple to `OwnedBitmap.cardinality` is correct.

`RunContainer.getCardinality` also changed to `*Self`. This works because `Container` stores `*RunContainer` (mutable pointer), so even when a `const Container` is destructured, the inner pointer is still mutable. Checked all callers in `compare.zig`, `container_ops.zig`, `serialize.zig` — no const violations.

---

## The fromSorted Bug

This is the real issue. `fromSorted` has a **preexisting correctness bug** that the cardinality cache now makes more visible.

### What happens with duplicates

```zig
// Current code in fromSorted, array path:
for (values[chunk_start..chunk_end], 0..) |v, i| {
    ac.values[i] = lowBits(v);
}
ac.cardinality = @intCast(chunk_size);
```

If `values` contains `[5, 5, 5, 10]`, this creates an ArrayContainer with:
- `values = [5, 5, 5, 10]`
- `cardinality = 4`

But the *actual* cardinality is 2. The container is now corrupt — binary search on `values` may work by accident (finding *a* 5), but:
- `cardinality` is wrong (4 vs 2)
- `contains(6)` might return true (binary search lands on a duplicate 5 and checks neighbors)
- `iterate()` yields duplicate values
- Set operations produce wrong results (merge algorithms assume sorted *unique* arrays)
- `serialize → deserialize` roundtrip may produce different bitmap

The bitset path is *partially* OK — `bc.add()` is idempotent (sets a bit), and `bc.computeCardinality()` counts actual bits, so the container-level cardinality is correct. But the bitmap-level `cached_cardinality = values.len` is still wrong.

### What the cache adds

Before caching: `cardinality()` recomputed from containers every time. The array containers had wrong internal cardinality, but nobody noticed because `ArrayContainer.getCardinality()` just returns the stored field — garbage in, garbage out, but consistently so.

After caching: `cached_cardinality = values.len` at the bitmap level. Now there are *two* wrong numbers that might disagree with each other. The bitmap cache says 4, the sum of container cardinalities says 4 (for the array case) or maybe 2 (for the bitset case, where computeCardinality is correct). Invalidating the cache and recomputing could give a different answer than the cached value. This is the kind of bug that shows up three weeks later in a Datalog join producing wrong results.

### The fix

Two parts:

1. **Debug assertion** — catch bad input in test/debug builds, zero cost in release
2. **Test-first** — prove you understand the invariant before writing the fix

---

## Task: Write Tests First, Then Fix

The contract for `fromSorted` is: **input must be sorted in strictly ascending order (no duplicates)**. This matches CRoaring's `roaring_bitmap_of_ptr`: "values must be sorted and have no duplicates."

### Step 1: Write these tests

Write all tests first, run them, see which fail, *then* fix the code.

**Test A: fromSorted basic correctness**

Build a bitmap from a known sorted-unique array. Verify:
- `cardinality()` matches input length
- `contains()` returns true for every input value
- `contains()` returns false for values not in input
- Iteration yields exactly the input values in order

**Test B: fromSorted matches incremental add**

Build two bitmaps with the same values — one via `fromSorted`, one via repeated `add()`. Verify they're `equals()`. This is the "oracle test" — `add()` is well-tested, so it's the ground truth.

**Test C: fromSorted cardinality cache consistency**

Build via `fromSorted`. Check `cardinality()` (cached). Then invalidate the cache (e.g., add one more value, which will increment the cache; or do a no-op in-place AND with self). Check `cardinality()` again (recomputed from containers). Both values must match. This catches the case where the bitmap-level cache disagrees with the container-level truth.

**Test D: fromSorted with cross-container values**

Use values that span multiple 65536-boundaries. E.g., `[0, 1, 65536, 65537, 131072]`. Verify cardinality is 5, verify contains for all, verify container count is 3.

**Test E: fromSorted roundtrip**

`fromSorted → serialize → deserialize`. Verify the deserialized bitmap `equals()` the original and has the same cardinality. This catches container-level corruption that survives serialization.

**Test F: fromSorted with duplicate input (the bug)**

Pass `[1, 1, 2, 3, 3]` to `fromSorted`. What *should* happen?

This is the design question. Two valid answers:
- **Option A (strict):** Debug assert / panic. The precondition is "sorted, unique." Violation is caller error.
- **Option B (lenient):** Silently dedup and produce correct bitmap with cardinality 3.

CRoaring takes Option A (documents the precondition, undefined behavior on violation). We should too — `fromSorted` is a power-user bulk load function. If you don't know your data is deduped, use `add()` in a loop.

So the test should verify that in debug builds, passing duplicates triggers an assertion failure (or returns an error if we prefer that). In release builds, it's undefined — we don't need to test UB.

### Step 2: Observe failures

Tests C and F will likely fail (or reveal interesting behavior). Test C because the cache might disagree with recomputed cardinality if you construct the right input. Test F because there's no assertion at all right now.

### Step 3: Fix

1. Add the debug assertion at the top of `fromSorted`:
```zig
if (std.debug.runtime_safety) {
    for (values[1..], 0..) |cur, i| {
        std.debug.assert(cur > values[i]); // not sorted or contains duplicates
    }
}
```

2. Change `cached_cardinality = @intCast(values.len)` — this is technically fine *after* the assert guarantees uniqueness. But as a belt-and-suspenders measure, consider computing it from the containers instead:
```zig
// After all containers built:
var total: u64 = 0;
for (result.containers[0..result.size]) |tp| {
    total += Container.fromTagged(tp).getCardinality();
}
result.cached_cardinality = @intCast(total);
```
This costs ~nothing (iterating a handful of containers) and makes the cache provably correct regardless of input. Up to you — the assert already guarantees correctness, this is just defense in depth.

3. Add a doc comment:
```zig
/// Build from pre-sorted, deduplicated values. O(n), no binary searches.
/// Caller must ensure values are in strictly ascending order with no duplicates.
/// Debug builds assert this precondition. In release, duplicates cause undefined behavior
/// (incorrect cardinality, corrupt containers).
```

### Step 4: Run tests again — all green

---

## Checklist Before Merge

- [ ] Tests A-F written and passing
- [ ] Debug assertion added to `fromSorted`
- [ ] Doc comment updated on `fromSorted`
- [ ] `cached_cardinality` in `fromSorted` is trustworthy (either via assert guarantee or recompute)
- [ ] Errdefer fix in `bitwiseOrInPlace` and `bitwiseXorInPlace` (separate issue from previous review — `new_containers` allocated without immediate errdefer)
- [ ] All existing tests still pass (`zig build test`)
