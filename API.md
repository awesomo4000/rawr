<!-- SPDX-License-Identifier: MPL-2.0 -->

# rawr API Guide

rawr's stable public API is:

- `RoaringBitmap`: mutable, caller-allocated bitmap.
- `Roaring64Bitmap`: mutable 64-bit bitmap layered over 32-bit roaring buckets.
- `OwnedBitmap`: arena-backed owned result for read-heavy temporary values.
- `FrozenBitmap`: zero-copy read-only view over serialized bytes.
- `Frozen64Bitmap`: zero-copy read-only view over rawr-native frozen64 bytes.
- `ValidateError`: structural validation errors from `RoaringBitmap.validate()`.

The module root also exposes container internals for rawr's own benchmarks,
validation, and differential tooling. Those internal exports are not part of the
stable API and may change without notice.

## Bitmap Types

| Type | Use When | Notes |
|---|---|---|
| `RoaringBitmap` | You need mutation or in-place operations | Call `deinit()` when done. |
| `Roaring64Bitmap` | Values may exceed `u32` | Uses sorted high-32 buckets containing 32-bit bitmaps. |
| `OwnedBitmap` | You need a read-only result and bulk-free lifetime | Returned by `*Owned` helpers. Use `asBitmap()` for the full read-only API. |
| `FrozenBitmap` | You have serialized bytes and want zero-copy lookup | Backing bytes must outlive the view. |
| `Frozen64Bitmap` | You have rawr frozen64 bytes and want zero-copy lookup | rawr-native format, not CRoaring frozen interop. |

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

### Mutation And Query Ranges Are Inclusive

Mutation and query range APIs use inclusive endpoints:

```zig
_ = try bm.addRange(0, 100);       // adds 101 values: 0 through 100
_ = try bm.removeRange(10, 20);    // removes 11 values if all are present
const n = bm.rangeCardinality(0, 100);
```

This applies to `addRange`, `removeRange`, `flip`, `flipInplace`,
`flipInPlace`, `rangeCardinality`, `containsRange`, and `intersectsRange`.
`Roaring64Bitmap.fromRange` is the exception: it is a stepped constructor over
the half-open range `[min, max)`.

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

var copy = try bm.clone(other_allocator);
defer copy.deinit();
```

`clone` creates an independent deep copy using the allocator passed to it. The
source and clone can be mutated and deinited independently.

Owned construction helpers:

```zig
var from_slice = try RoaringBitmap.fromSliceOwned(backing_allocator, mutable_values);
defer from_slice.deinit();

var owned = try RoaringBitmap.deserializeSafeOwned(backing_allocator, bytes);
defer owned.deinit();
```

### How rawr bitmaps allocate

A Roaring bitmap is a sorted array of containers, one per distinct high-16 chunk
of the value space, with up to 65,536 containers. There are two independent
allocation axes:

1. The container index: the top-level `keys` and `containers` arrays. These start
   with four entries and grow geometrically as data spans more 16-bit chunks.
2. Each container's storage: array, bitset, or run storage sized and selected
   dynamically as values are inserted.

When the expected number of chunks is known, pre-size the container index to
avoid geometric regrowth:

```zig
// Expect values spread across about 1000 distinct high-16 chunks.
var bm = try RoaringBitmap.initCapacity(allocator, 1000);
defer bm.deinit();

// The capacity can also be raised later without changing the contents.
try bm.ensureTotalCapacity(2000);
```

Capacity is measured in containers, approximately the number of distinct high-16
chunks. For `Roaring64Bitmap`, it is the number of high-32 buckets. It is not an
element count: values map to containers unpredictably, and each container's type
and storage are selected dynamically.

`shrinkToFit()` releases unused index capacity and unused capacity in shrinkable
containers, returning the approximate number of payload bytes freed.
`clearRetainingCapacity()` empties the bitmap while keeping only the top-level
container index, allowing it to be refilled without regrowing that index. Storage
owned by the cleared containers is freed because those containers no longer
exist.

## Mutation

```zig
_ = try bm.add(value);
try bm.addMany(values);
_ = try bm.addRange(lo, hi);       // inclusive

_ = try bm.remove(value);
try bm.removeMany(values);
_ = try bm.removeRange(lo, hi);    // inclusive

var without_range = try bm.removeRangeCopy(allocator, lo, hi);
defer without_range.deinit();
```

`removeRangeCopy` leaves the source unchanged and constructs an independently
owned result from only the surviving containers instead of cloning containers
that the range removal would immediately discard.

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

`bitwiseOrInPlaceConsume` can move right-only containers instead of cloning
them. Both operands must be distinct and use the exact same allocator handle.
On success, the right operand is a valid empty bitmap that retains its top-level
capacity; it may be reused or deinited normally, but its previous contents have
been consumed. Allocator mismatch and aliased operands fail before mutation. On
allocation failure, the right operand is unchanged and the left remains valid,
but the left may already contain unions completed for earlier matching chunks.

```zig
try a.bitwiseOrInPlaceConsume(&b);
// b is now empty and valid; a contains the union.
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

`repairAfterLazyWithOptions` has the same semantic result as
`repairAfterLazy`, with an opt-in that changes only the order in which transient
bitsets are freed:

```zig
try a.repairAfterLazyWithOptions(.{
    .allocator_benefits_from_descending_free_order = true,
});
```

The default repair path is unchanged. The effect of descending free order is
allocator- and workload-dependent, so callers should enable it only after
measuring their allocator and workload.

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

## `Roaring64Bitmap`

`Roaring64Bitmap` partitions `u64` values by their high 32 bits. Each sorted
high-key bucket contains a `RoaringBitmap` for the low 32 bits. It supports the
same mutation, query, positional, set, comparison, range, iteration, and
optimization families as `RoaringBitmap`, using `u64` values.

Construction and conversion:

```zig
var bm64 = try Roaring64Bitmap.init(allocator);
defer bm64.deinit();

var reserved = try Roaring64Bitmap.initCapacity(allocator, expected_buckets);
defer reserved.deinit();

var stepped = try Roaring64Bitmap.fromRange(allocator, min, max, step);
defer stepped.deinit();

var sorted = try Roaring64Bitmap.fromSortedSlice(allocator, sorted_values);
defer sorted.deinit();

var arbitrary = try Roaring64Bitmap.fromSlice(allocator, mutable_values);
defer arbitrary.deinit();

var copy = try bm64.clone(other_allocator);
defer copy.deinit();
```

`fromRange` covers the stepped half-open range `[min, max)`; `step == 0` or
`max <= min` produces an empty bitmap. `fromSortedSlice` requires ascending
input and accepts adjacent duplicates. `fromSlice` accepts arbitrary values but
sorts and deduplicates its mutable input slice in place. `clone` is a deep copy
using the supplied allocator.

`fromRoaring32` clones a 32-bit bitmap into high-key bucket zero.
`toRoaring32` returns an independently allocated 32-bit bitmap when every value
fits in `u32`, or `null` when a nonzero high-key bucket is present.

```zig
var widened = try Roaring64Bitmap.fromRoaring32(allocator, &bm);
defer widened.deinit();

if (try widened.toRoaring32(allocator)) |narrowed_value| {
    var narrowed = narrowed_value;
    defer narrowed.deinit();
}
```

For repeated operations with locality in the high 32 bits, initialize a
`BulkContext` and pass it to `addBulk`, `containsBulk`, or `removeBulk`. Keep a
context associated with one bitmap; it caches a bucket position and refreshes
itself when that bitmap mutates.

```zig
var bulk = Roaring64Bitmap.BulkContext.init();
try bm64.addBulk(&bulk, value);
const present = bm64.containsBulk(&bulk, value);
try bm64.removeBulk(&bulk, value);
```

`addRange`, `removeRange`, range queries, `flip`, and `flipInPlace` use inclusive
endpoints. The 64-bit mutating range methods do not return added or removed
counts because a range can contain more values than fit in `u64`.

```zig
try bm64.addRange(lo, hi);
try bm64.flipInPlace(lo, hi);
const count = bm64.rangeCardinality(lo, hi);
```

`statistics` reports bucket count, cardinality, minimum and maximum values, and
the container mix across all buckets. Its byte fields describe rawr allocation
bytes. Portable serialization uses CRoaring's roaring64 format as described in
[Serialization](#serialization); `frozenSerialize` instead produces the
rawr-native image consumed by `Frozen64Bitmap`.

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
frozen.minimum()
frozen.maximum()
frozen.rank(value)
frozen.select(rank)
frozen.getIndex(value)
frozen.iterator()
```

The serialized bytes must remain alive for the lifetime of the `FrozenBitmap`.
`rank`, `select`, and `getIndex` scan container descriptors and then probe one
container; the frozen descriptor table does not store prefix sums. `minimum`
and `maximum` read array and run endpoints directly, while a bitset container
may require scanning up to 1,024 words.

## `Frozen64Bitmap`

```zig
const size = try bm64.frozenSizeInBytes();
const bytes = try allocator.alloc(u8, size);
defer allocator.free(bytes);

try bm64.frozenSerialize(bytes);

var frozen64 = try Frozen64Bitmap.view(bytes);
defer frozen64.deinit();

frozen64.contains(value)
frozen64.cardinality()
frozen64.minimum()
frozen64.maximum()
frozen64.rank(value)
frozen64.select(rank)
frozen64.getIndex(value)
frozen64.iterator()
```

`Frozen64Bitmap` uses rawr's own frozen64 image: a 64-bit bucket table plus
rawr 32-bit frozen sub-images. It is not byte-compatible with CRoaring's frozen
format. The backing bytes must remain alive for the lifetime of the view.

## Allocator Guide

| Allocator | Use When |
|---|---|
| `OwnedBitmap` helpers | Read-only results whose arena-backed storage should be released in one `deinit`. |
| `std.heap.smp_allocator` | General-purpose mutable bitmaps with independently managed lifetimes. |
| `std.heap.ArenaAllocator` | Groups of bitmaps or operations that share a bulk-free lifetime. |
| `std.heap.FixedBufferAllocator` | Workloads with a known memory bound and caller-provided storage. |
| `std.heap.c_allocator` | Applications that require libc-backed allocation and link libc. |

## Quick Reference

The guarded region below inventories direct public methods on rawr's five stable
bitmap types. Nested public types and constants are outside this guard's scope.

<!-- check-docs:begin -->

### `RoaringBitmap`

- `RoaringBitmap.init`
- `RoaringBitmap.initCapacity`
- `RoaringBitmap.deinit`
- `RoaringBitmap.clone`
- `RoaringBitmap.validate`
- `RoaringBitmap.ensureTotalCapacity`
- `RoaringBitmap.clearRetainingCapacity`
- `RoaringBitmap.shrinkToFit`
- `RoaringBitmap.contains`
- `RoaringBitmap.rangeCardinality`
- `RoaringBitmap.containsRange`
- `RoaringBitmap.intersectsRange`
- `RoaringBitmap.add`
- `RoaringBitmap.addMany`
- `RoaringBitmap.addRange`
- `RoaringBitmap.removeRange`
- `RoaringBitmap.removeRangeCopy`
- `RoaringBitmap.fromSorted`
- `RoaringBitmap.fromSlice`
- `RoaringBitmap.remove`
- `RoaringBitmap.removeMany`
- `RoaringBitmap.cardinality`
- `RoaringBitmap.toArrayAlloc`
- `RoaringBitmap.toArray`
- `RoaringBitmap.isEmpty`
- `RoaringBitmap.minimum`
- `RoaringBitmap.maximum`
- `RoaringBitmap.bitwiseOr`
- `RoaringBitmap.bitwiseAnd`
- `RoaringBitmap.andCardinality`
- `RoaringBitmap.orCardinality`
- `RoaringBitmap.xorCardinality`
- `RoaringBitmap.differenceCardinality`
- `RoaringBitmap.jaccardIndex`
- `RoaringBitmap.rank`
- `RoaringBitmap.getIndex`
- `RoaringBitmap.select`
- `RoaringBitmap.rankMany`
- `RoaringBitmap.intersects`
- `RoaringBitmap.bitwiseDifference`
- `RoaringBitmap.bitwiseXor`
- `RoaringBitmap.orMany`
- `RoaringBitmap.orManyHeap`
- `RoaringBitmap.xorMany`
- `RoaringBitmap.lazyOr`
- `RoaringBitmap.lazyXor`
- `RoaringBitmap.flip`
- `RoaringBitmap.bitwiseOrInPlace`
- `RoaringBitmap.bitwiseOrInPlaceConsume`
- `RoaringBitmap.bitwiseAndInPlace`
- `RoaringBitmap.bitwiseDifferenceInPlace`
- `RoaringBitmap.bitwiseXorInPlace`
- `RoaringBitmap.lazyOrInPlace`
- `RoaringBitmap.lazyXorInPlace`
- `RoaringBitmap.flipInplace`
- `RoaringBitmap.runOptimize`
- `RoaringBitmap.repairAfterLazy`
- `RoaringBitmap.repairAfterLazyWithOptions`
- `RoaringBitmap.isSubsetOf`
- `RoaringBitmap.isStrictSubsetOf`
- `RoaringBitmap.equals`
- `RoaringBitmap.iterator`
- `RoaringBitmap.serializedSizeInBytes`
- `RoaringBitmap.serialize`
- `RoaringBitmap.serializeToWriter`
- `RoaringBitmap.deserialize`
- `RoaringBitmap.deserializeSafe`
- `RoaringBitmap.deserializeFromReader`
- `RoaringBitmap.deserializeOwned`
- `RoaringBitmap.deserializeSafeOwned`
- `RoaringBitmap.bitwiseAndOwned`
- `RoaringBitmap.bitwiseOrOwned`
- `RoaringBitmap.bitwiseDifferenceOwned`
- `RoaringBitmap.flipOwned`
- `RoaringBitmap.fromSliceOwned`
- `RoaringBitmap.orManyOwned`
- `RoaringBitmap.xorManyOwned`

### `Roaring64Bitmap`

- `Roaring64Bitmap.init`
- `Roaring64Bitmap.initCapacity`
- `Roaring64Bitmap.deinit`
- `Roaring64Bitmap.clearRetainingCapacity`
- `Roaring64Bitmap.clone`
- `Roaring64Bitmap.fromRange`
- `Roaring64Bitmap.fromSortedSlice`
- `Roaring64Bitmap.fromSlice`
- `Roaring64Bitmap.fromRoaring32`
- `Roaring64Bitmap.toRoaring32`
- `Roaring64Bitmap.isEmpty`
- `Roaring64Bitmap.cardinality`
- `Roaring64Bitmap.add`
- `Roaring64Bitmap.addMany`
- `Roaring64Bitmap.contains`
- `Roaring64Bitmap.remove`
- `Roaring64Bitmap.addBulk`
- `Roaring64Bitmap.containsBulk`
- `Roaring64Bitmap.removeBulk`
- `Roaring64Bitmap.minimum`
- `Roaring64Bitmap.maximum`
- `Roaring64Bitmap.toArrayAlloc`
- `Roaring64Bitmap.toArray`
- `Roaring64Bitmap.iterator`
- `Roaring64Bitmap.bitwiseOr`
- `Roaring64Bitmap.bitwiseAnd`
- `Roaring64Bitmap.bitwiseXor`
- `Roaring64Bitmap.bitwiseDifference`
- `Roaring64Bitmap.andCardinality`
- `Roaring64Bitmap.orCardinality`
- `Roaring64Bitmap.xorCardinality`
- `Roaring64Bitmap.differenceCardinality`
- `Roaring64Bitmap.jaccardIndex`
- `Roaring64Bitmap.intersects`
- `Roaring64Bitmap.isSubsetOf`
- `Roaring64Bitmap.isStrictSubsetOf`
- `Roaring64Bitmap.equals`
- `Roaring64Bitmap.bitwiseOrInPlace`
- `Roaring64Bitmap.bitwiseAndInPlace`
- `Roaring64Bitmap.bitwiseXorInPlace`
- `Roaring64Bitmap.bitwiseDifferenceInPlace`
- `Roaring64Bitmap.rank`
- `Roaring64Bitmap.getIndex`
- `Roaring64Bitmap.select`
- `Roaring64Bitmap.addRange`
- `Roaring64Bitmap.removeRange`
- `Roaring64Bitmap.rangeCardinality`
- `Roaring64Bitmap.containsRange`
- `Roaring64Bitmap.intersectsRange`
- `Roaring64Bitmap.flip`
- `Roaring64Bitmap.flipInPlace`
- `Roaring64Bitmap.runOptimize`
- `Roaring64Bitmap.shrinkToFit`
- `Roaring64Bitmap.validate`
- `Roaring64Bitmap.statistics`
- `Roaring64Bitmap.serializedSizeInBytes`
- `Roaring64Bitmap.frozenSizeInBytes`
- `Roaring64Bitmap.frozenSerialize`
- `Roaring64Bitmap.serialize`
- `Roaring64Bitmap.serializeToWriter`
- `Roaring64Bitmap.deserialize`
- `Roaring64Bitmap.deserializeSafe`
- `Roaring64Bitmap.ensureTotalCapacity`

### `OwnedBitmap`

- `OwnedBitmap.deinit`
- `OwnedBitmap.contains`
- `OwnedBitmap.asBitmap`
- `OwnedBitmap.cardinality`
- `OwnedBitmap.iterator`
- `OwnedBitmap.serialize`

### `FrozenBitmap`

- `FrozenBitmap.init`
- `FrozenBitmap.deinit`
- `FrozenBitmap.isEmpty`
- `FrozenBitmap.contains`
- `FrozenBitmap.rank`
- `FrozenBitmap.getIndex`
- `FrozenBitmap.select`
- `FrozenBitmap.minimum`
- `FrozenBitmap.maximum`
- `FrozenBitmap.cardinality`
- `FrozenBitmap.iterator`

### `Frozen64Bitmap`

- `Frozen64Bitmap.view`
- `Frozen64Bitmap.deinit`
- `Frozen64Bitmap.isEmpty`
- `Frozen64Bitmap.contains`
- `Frozen64Bitmap.cardinality`
- `Frozen64Bitmap.minimum`
- `Frozen64Bitmap.maximum`
- `Frozen64Bitmap.rank`
- `Frozen64Bitmap.getIndex`
- `Frozen64Bitmap.select`
- `Frozen64Bitmap.iterator`
<!-- check-docs:end -->
