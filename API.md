# rawr API Guide

rawr's stable public API is:

- `RoaringBitmap`: mutable, caller-allocated bitmap.
- `Roaring64Bitmap`: mutable 64-bit bitmap layered over 32-bit roaring buckets.
- `OwnedBitmap`: arena-backed owned result for read-heavy temporary values.
- `FrozenBitmap`: zero-copy read-only view over serialized bytes.
- `ValidateError`: structural validation errors from `RoaringBitmap.validate()`.

The module root also exposes container internals for rawr's own benchmarks,
validation, and differential tooling. Those internal exports are not part of the
stable API and may change.

## Bitmap Types

| Type | Use When | Notes |
|---|---|---|
| `RoaringBitmap` | You need mutation or in-place operations | Call `deinit()` when done. |
| `OwnedBitmap` | You need a read-only result and bulk-free lifetime | Returned by `*Owned` helpers. Use `asBitmap()` for the full read-only API. |
| `FrozenBitmap` | You have serialized bytes and want zero-copy lookup | Backing bytes must outlive the view. |

```zig
var bm = try rawr.RoaringBitmap.init(allocator);
defer bm.deinit();

var owned = try rawr.RoaringBitmap.deserializeSafeOwned(allocator, bytes);
defer owned.deinit();
const min = owned.asBitmap().minimum();

var frozen = try rawr.FrozenBitmap.init(bytes);
defer frozen.deinit();
```

## Footguns

### Ranges Are Inclusive

All range APIs use inclusive endpoints:

```zig
_ = try bm.addRange(0, 100);       // adds 101 values: 0 through 100
_ = try bm.removeRange(10, 20);    // removes 11 values if all are present
const n = bm.rangeCardinality(0, 100);
```

This applies to `addRange`, `removeRange`, `flip`, `flipInplace`,
`rangeCardinality`, `containsRange`, and `intersectsRange`.

### Use `deserializeSafe` For Untrusted Bytes

`deserialize` is bounds-safe, but it does not semantically validate every
container invariant. Use `deserializeSafe` for untrusted input, or call
`validate()` after `deserialize`.

```zig
var trusted = try RoaringBitmap.deserialize(allocator, bytes);
var untrusted = try RoaringBitmap.deserializeSafe(allocator, bytes);
```

### Iterators Are Invalidated By Mutation

Mutating a bitmap while iterating invalidates the iterator. Finish iterating, or
snapshot the bitmap first.

### Lazy Results Must Be Repaired

`lazyOr`, `lazyXor`, `lazyOrInPlace`, and `lazyXorInPlace` leave the result in an
invalid intermediate state until `repairAfterLazy()` succeeds. For ordinary
two-way set operations, prefer eager `bitwiseOr` and `bitwiseXor`.

```zig
var result = try a.lazyOr(allocator, &b, true);
defer result.deinit();
try result.repairAfterLazy();
```

### `fromSorted` Requires Strictly Sorted Unique Input

`fromSorted` is for data already known to be strictly ascending with no
duplicates. Debug builds assert this; release builds may silently corrupt the
bitmap if the precondition is violated. Use `fromSlice` for arbitrary input.

```zig
var arbitrary = [_]u32{ 10, 3, 3, 7 };
var bm = try RoaringBitmap.fromSlice(allocator, &arbitrary);
```

### Querying `OwnedBitmap`

`OwnedBitmap` keeps a few convenience methods, and `asBitmap()` exposes the full
read-only `RoaringBitmap` surface:

```zig
const card = owned.cardinality();
const max = owned.asBitmap().maximum();
const overlap = owned.asBitmap().andCardinality(&other);
```

## Construction

```zig
var bm = try RoaringBitmap.init(allocator);
defer bm.deinit();

_ = try bm.add(42);
try bm.addMany(&values);
_ = try bm.addRange(1000, 1999); // inclusive

var sorted = try RoaringBitmap.fromSorted(allocator, sorted_unique_values);
defer sorted.deinit();

var arbitrary = try RoaringBitmap.fromSlice(allocator, mutable_values);
defer arbitrary.deinit();
```

Owned construction helpers:

```zig
var from_slice = try RoaringBitmap.fromSliceOwned(backing_allocator, mutable_values);
defer from_slice.deinit();

var owned = try RoaringBitmap.deserializeSafeOwned(backing_allocator, bytes);
defer owned.deinit();
```

## Mutation

```zig
_ = try bm.add(value);
try bm.addMany(values);
_ = try bm.addRange(lo, hi);       // inclusive

_ = try bm.remove(value);
try bm.removeMany(values);
_ = try bm.removeRange(lo, hi);    // inclusive
```

## Queries

```zig
bm.contains(value)                 // bool
bm.cardinality()                   // u64, const-safe
bm.isEmpty()                       // bool
bm.minimum()                       // ?u32
bm.maximum()                       // ?u32
bm.validate()                      // ValidateError!void
```

Positional queries:

```zig
bm.rank(value)                     // count values <= value
bm.select(index)                   // ?u32, 0-based
bm.getIndex(value)                 // ?u64, null if absent
bm.rankMany(sorted_values, out)    // batched ranks; input must be sorted
```

Range queries:

```zig
bm.rangeCardinality(lo, hi)        // u64, inclusive
bm.containsRange(lo, hi)           // bool, inclusive
bm.intersectsRange(lo, hi)         // bool, inclusive
```

## Extraction And Iteration

```zig
var it = bm.iterator();
while (it.next()) |value| {
    // values in ascending order
}

const values = try bm.toArrayAlloc(allocator);
defer allocator.free(values);

const written = bm.toArray(out);
```

## Set Operations

Allocating two-way operations:

```zig
var and_result = try a.bitwiseAnd(allocator, &b);
var or_result = try a.bitwiseOr(allocator, &b);
var xor_result = try a.bitwiseXor(allocator, &b);
var diff_result = try a.bitwiseDifference(allocator, &b); // a \ b
```

In-place two-way operations:

```zig
try a.bitwiseAndInPlace(&b);
try a.bitwiseOrInPlace(&b);
try a.bitwiseXorInPlace(&b);
try a.bitwiseDifferenceInPlace(&b);
```

Owned two-way operations:

```zig
var result = try a.bitwiseAndOwned(backing_allocator, &b);
defer result.deinit();
```

N-way operations:

```zig
var union_result = try RoaringBitmap.orMany(allocator, bitmaps);
var heap_union = try RoaringBitmap.orManyHeap(allocator, bitmaps);
var xor_result = try RoaringBitmap.xorMany(allocator, bitmaps);

var owned_union = try RoaringBitmap.orManyOwned(backing_allocator, bitmaps);
var owned_xor = try RoaringBitmap.xorManyOwned(backing_allocator, bitmaps);
```

Range-producing operations:

```zig
var flipped = try bm.flip(allocator, lo, hi); // inclusive
try bm.flipInplace(lo, hi);                   // inclusive
```

Lazy operations:

```zig
var lazy = try a.lazyOr(allocator, &b, true);
try lazy.repairAfterLazy();

var lazy_xor = try a.lazyXor(allocator, &b);
try lazy_xor.repairAfterLazy();

try a.lazyOrInPlace(&b, true);
try a.repairAfterLazy();

try a.lazyXorInPlace(&b);
try a.repairAfterLazy();
```

## Analytics And Comparison

No-allocation cardinality variants:

```zig
a.andCardinality(&b)               // |A intersection B|
a.orCardinality(&b)                // |A union B|
a.xorCardinality(&b)               // |A symmetric-difference B|
a.differenceCardinality(&b)        // |A \ B|
a.jaccardIndex(&b)                 // |A intersection B| / |A union B|
a.intersects(&b)                   // early-exit overlap check
```

Comparison:

```zig
a.equals(&b)
a.isSubsetOf(&b)
a.isStrictSubsetOf(&b)
```

## Serialization

rawr uses the CRoaring-compatible RoaringFormatSpec wire format.
`Roaring64Bitmap` uses CRoaring's `roaring64` portable format. Java 64-bit
Roaring layouts are different and are not supported or tested by rawr's 64-bit
serializer.

```zig
const bytes = try bm.serialize(allocator);
defer allocator.free(bytes);

try bm.serializeToWriter(writer);
const size = bm.serializedSizeInBytes();

var trusted = try RoaringBitmap.deserialize(allocator, bytes);
defer trusted.deinit();

var safe = try RoaringBitmap.deserializeSafe(allocator, bytes);
defer safe.deinit();

var from_reader = try RoaringBitmap.deserializeFromReader(allocator, reader, data_len);
defer from_reader.deinit();
```

`OwnedBitmap.serialize(out_allocator)` allocates the output with the provided
allocator, not the internal arena.

## Optimization

```zig
const converted = try bm.runOptimize();
```

`runOptimize` converts containers to run-length encoding where it saves space.
It does not change the represented values.

## `OwnedBitmap`

```zig
owned.contains(value)
owned.cardinality()
owned.iterator()
owned.serialize(out_allocator)
owned.asBitmap()                   // *const RoaringBitmap
owned.deinit()
```

`OwnedBitmap` is read-only by convention. Use `asBitmap()` for read-only APIs
such as `minimum`, `maximum`, `rank`, `select`, `equals`, `isSubsetOf`, and the
cardinality variants.

## `FrozenBitmap`

```zig
var frozen = try FrozenBitmap.init(bytes);
defer frozen.deinit();

frozen.contains(value)
frozen.cardinality()
frozen.isEmpty()
frozen.iterator()
```

The serialized bytes must remain alive for the lifetime of the `FrozenBitmap`.

## Allocator Guide

| Allocator | Use When |
|---|---|
| `OwnedBitmap` helpers | Fast temporary read-only results. |
| `std.heap.smp_allocator` | Long-lived mutable bitmaps. |
| `std.heap.ArenaAllocator` | Batch operations with bulk-free lifetime. |
| `std.heap.FixedBufferAllocator` | Hot paths with known memory bounds. |
| `std.heap.c_allocator` | Avoid for rawr's many small allocations. |

## Quick Reference

```text
PUBLIC TYPES      RoaringBitmap, OwnedBitmap, FrozenBitmap, ValidateError

CONSTRUCT         init, fromSorted, fromSlice, fromSliceOwned
                  deserialize, deserializeSafe, deserializeFromReader
                  deserializeOwned, deserializeSafeOwned

MUTATE            add, addMany, addRange, remove, removeMany, removeRange

QUERY             contains, cardinality, isEmpty, minimum, maximum, validate

POSITIONAL        rank, select, getIndex, rankMany

RANGES            flip, flipInplace, rangeCardinality, containsRange,
                  intersectsRange

SET OPS           bitwiseAnd, bitwiseOr, bitwiseXor, bitwiseDifference
                  bitwiseAndInPlace, bitwiseOrInPlace, bitwiseXorInPlace,
                  bitwiseDifferenceInPlace
                  bitwiseAndOwned, bitwiseOrOwned, bitwiseDifferenceOwned,
                  flipOwned

N-WAY             orMany, orManyHeap, xorMany, orManyOwned, xorManyOwned

LAZY              lazyOr, lazyXor, lazyOrInPlace, lazyXorInPlace,
                  repairAfterLazy

ANALYTICS         andCardinality, orCardinality, xorCardinality,
                  differenceCardinality, jaccardIndex, intersects

COMPARE           equals, isSubsetOf, isStrictSubsetOf

EXTRACT           iterator, toArray, toArrayAlloc

SERIALIZE         serialize, serializeToWriter, serializedSizeInBytes

OPTIMIZE          runOptimize

OWNED             asBitmap, contains, cardinality, iterator, serialize, deinit

FROZEN            init, contains, cardinality, isEmpty, iterator, deinit
```
