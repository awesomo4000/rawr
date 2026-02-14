# rawr API Guide

## Bitmap Types

rawr provides three bitmap types for different access patterns.

**`RoaringBitmap`** — Mutable bitmap with caller-managed allocation. Supports add, remove, and in-place set operations. You provide the allocator and call `deinit()` when done.

**`OwnedBitmap`** — Immutable, arena-backed bitmap. Returned by `*Owned` factory methods. Fastest for read-only results you'll query and discard. A single `deinit()` frees all internal memory at once.

**`FrozenBitmap`** — Zero-copy view over serialized bytes. No allocation or deserialization cost. Supports `contains`, `cardinality`, and `iterator`. The backing bytes must outlive the bitmap.

```
                    ┌─────────────────┐
   Mutable          │ RoaringBitmap   │  ← add, remove, in-place ops
                    └────────┬────────┘
                             │ serialize
                    ┌────────▼────────┐
   Immutable        │ OwnedBitmap     │  ← set op results, deserialized
   (arena-backed)   └────────┬────────┘
                             │ or skip deserialization entirely
                    ┌────────▼────────┐
   Zero-copy        │ FrozenBitmap    │  ← mmap, embedded bytes, lookup-only
                    └─────────────────┘
```

| Scenario | Type | Why |
|---|---|---|
| Building a set incrementally | `RoaringBitmap` | Need add/remove |
| Accumulating results in a loop | `RoaringBitmap` | Need in-place ops |
| Set operation result (read then discard) | `OwnedBitmap` via `bitwiseAndOwned` | Fast alloc, bulk free |
| Lookup against stored/mmap'd data | `FrozenBitmap` | Zero deserialization cost |
| Estimating overlap between two sets | Neither — use `andCardinality` | No allocation at all |

---

## Construction

### From individual values

```zig
var bm = try RoaringBitmap.init(allocator);
defer bm.deinit();

_ = try bm.add(42);      // returns true (new value)
_ = try bm.add(42);      // returns false (already present)
_ = try bm.add(100);
```

### Bulk: sorted, unique data — O(n)

```zig
const values = [_]u32{ 1, 5, 10, 100 };
var bm = try RoaringBitmap.fromSorted(allocator, &values);
defer bm.deinit();
```

Input must be strictly ascending with no duplicates. Debug builds assert this. Violation in release builds causes silent corruption. Use when the source is already ordered (database cursor output, pre-processed pipelines).

### Bulk: arbitrary data — O(n log n)

```zig
var values = [_]u32{ 10, 3, 3, 7, 1, 10 };
var bm = try RoaringBitmap.fromSlice(allocator, &values);
defer bm.deinit();
// bm contains {1, 3, 7, 10}
```

Sorts and deduplicates in-place. Takes `[]u32` (mutable) — the signature signals that your data gets modified. Use when you don't control input ordering.

### Bulk: ranges

```zig
_ = try bm.addRange(1000, 1999); // inclusive [1000, 1999]
```

Returns count of newly added values. O(R) where R is the number of affected run containers.

### From serialized bytes

```zig
// Standard
var bm = try RoaringBitmap.deserialize(allocator, bytes);
defer bm.deinit();

// Arena-backed (faster, recommended for read-mostly use)
var owned = try RoaringBitmap.deserializeOwned(page_allocator, bytes);
defer owned.deinit();

// Zero-copy (no allocation — bytes must outlive the bitmap)
var frozen = try FrozenBitmap.init(bytes);
```

---

## Querying

```zig
bm.contains(42)        // bool — is this value present?
bm.cardinality()       // u64  — total number of values (O(1), cached)
bm.isEmpty()           // bool — cardinality == 0?
bm.minimum()           // ?u32 — smallest value, or null
bm.maximum()           // ?u32 — largest value, or null
```

`cardinality()` is O(1) — the count is cached and maintained incrementally through mutations. The cache is invalidated on bulk operations and recomputed on the next call.

---

## Iteration

```zig
var it = bm.iterator();
while (it.next()) |value| {
    // value: u32, always in ascending order
}
```

Works on all three bitmap types. O(n) total cost.

---

## Set Operations

Every set operation exists in three forms:

| Form | Signature | Mutates self | Returns | Use for |
|---|---|---|---|---|
| Allocating | `bitwiseAnd(allocator, other)` | No | new `RoaringBitmap` | General use |
| In-place | `bitwiseAndInPlace(other)` | Yes | void | Accumulation loops |
| Owned | `bitwiseAndOwned(backing, other)` | No | `OwnedBitmap` | Fast temporary results |

Same pattern for **Or**, **Difference**, and **Xor**.

### Allocating — new result

```zig
var result = try a.bitwiseAnd(allocator, &b);
defer result.deinit();
```

Both inputs are `*const` — not modified.

### In-place — mutate self

```zig
try working_set.bitwiseOrInPlace(&additions);         // self |= other
try candidates.bitwiseAndInPlace(&filter);             // self &= other
try remaining.bitwiseDifferenceInPlace(&processed);    // self -= other
try changed.bitwiseXorInPlace(&previous);              // self ^= other
```

The `other` argument is `*const` — not modified.

### Owned — arena-backed result

```zig
var result = try a.bitwiseAndOwned(page_allocator, &b);
defer result.deinit();
```

Fastest for results you'll read and discard. The backing allocator feeds an internal arena.

---

## Analytics (No Allocation)

### andCardinality — |A ∩ B| without materializing

```zig
const overlap = a.andCardinality(&b);
```

Returns the number of values common to both bitmaps. Zero allocation — uses SIMD popcount for bitset containers and galloping merge for arrays. Use for selectivity estimation, overlap analysis, or any case where you need the intersection size but not the intersection itself.

### intersects — boolean overlap check

```zig
if (a.intersects(&b)) {
    // at least one shared value
}
```

Early-exits on first match. Cheaper than `andCardinality` when you only need yes/no.

---

## Comparison

```zig
a.equals(&b)       // true if both contain the same values
a.isSubsetOf(&b)   // true if every value in a is also in b
```

Both are single-pass O(n).

---

## Serialization

Wire-format compatible with CRoaring (RoaringFormatSpec). Bitmaps serialized by CRoaring can be deserialized by rawr and vice versa.

```zig
// Serialize to bytes
const bytes = try bm.serialize(allocator);
defer allocator.free(bytes);

// Serialize to any std.io writer
try bm.serializeToWriter(writer);

// Check size before allocating
const size = bm.serializedSizeInBytes();
```

---

## Optimization

```zig
const converted = try bm.runOptimize();
```

Converts containers to run-length encoding where it saves space. Call after bulk construction (many `add` or `addRange` calls) before serialization. Returns the number of containers converted. Does not change the values in the bitmap.

---

## Allocator Guide

| Allocator | Speed | Notes |
|---|---|---|
| `std.heap.c_allocator` | **Avoid** | 10-40x slower due to alignment overhead |
| `std.heap.smp_allocator` | Fast | Best default for long-lived mutable bitmaps |
| `std.heap.ArenaAllocator` | Faster | Batch operations, scoped lifetimes, bulk free |
| `OwnedBitmap` (internal arena) | Fastest convenience | Use the `*Owned` methods |
| `std.heap.FixedBufferAllocator` | Fastest possible | Pre-sized buffers for hot loops |

The `c_allocator` penalty comes from Zig's wrapper enforcing 32-byte alignment and vtable dispatch on every allocation. Rawr creates many small containers — the per-allocation cost adds up. Use `smp_allocator` as a drop-in replacement.

### Scoped arena pattern

```zig
var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
defer arena.deinit();
const alloc = arena.allocator();

var result = try set_a.bitwiseAnd(alloc, &set_b);
var filtered = try result.bitwiseAnd(alloc, &set_c);
// use filtered...
// arena.deinit() frees everything at once
```

---

## Quick Reference

```
CONSTRUCT        init → add/addRange    fromSorted(sorted unique)    fromSlice(anything)
                 deserialize(bytes)     FrozenBitmap.init(bytes)

QUERY            contains(u32) → bool   cardinality() → u64          isEmpty() → bool
                 minimum() → ?u32       maximum() → ?u32

SET OPS          bitwiseAnd             bitwiseOr                    bitwiseDifference
(new result)     bitwiseXor

SET OPS          bitwiseAndInPlace      bitwiseOrInPlace             bitwiseDifferenceInPlace
(mutate self)    bitwiseXorInPlace

ANALYTICS        andCardinality(other) → u64     intersects(other) → bool

ARENA RESULTS    bitwiseAndOwned        bitwiseOrOwned               bitwiseDifferenceOwned
                 deserializeOwned       fromSliceOwned

ITERATE          iterator() → Iterator { .next() → ?u32 }

SERIALIZE        serialize(alloc) → []u8          serializeToWriter(writer)
                 serializedSizeInBytes() → usize

COMPARE          equals(other) → bool             isSubsetOf(other) → bool

OPTIMIZE         runOptimize() → u32
```
