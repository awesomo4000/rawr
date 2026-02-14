# rawr

Roaring bitmap library in pure Zig. Wire-compatible with CRoaring (serialized
bitmaps interoperate across implementations). No C dependencies.

## Interop

Implements the [RoaringFormatSpec](https://github.com/RoaringBitmap/RoaringFormatSpec)
portable serialization format. Bitmaps serialized by rawr can be read by CRoaring,
Java RoaringBitmap, Go roaring, and any other compliant implementation — and vice
versa. Validated by `zig build validate` which round-trips through both rawr and
CRoaring.

## Usage

```zig
const rawr = @import("rawr");
const RoaringBitmap = rawr.RoaringBitmap;

// Create and populate
var bm = try RoaringBitmap.init(std.heap.smp_allocator);
defer bm.deinit();

_ = try bm.add(1);
_ = try bm.add(2);
_ = try bm.add(3);
_ = try bm.addRange(100, 200);  // adds 100..200 inclusive

// Query
assert(bm.contains(150));
assert(!bm.contains(50));
const card = bm.cardinality(); // 104

// Iterate
var it = bm.iterator();
while (it.next()) |value| {
    // values in sorted order: 1, 2, 3, 100, 101, ..., 200
}

// Set operations (allocate new bitmap)
var other = try RoaringBitmap.init(std.heap.smp_allocator);
defer other.deinit();
_ = try other.addRange(150, 250);

var intersection = try bm.bitwiseAnd(std.heap.smp_allocator, &other);
defer intersection.deinit();
// intersection contains 150..200

// Set operations (in-place, no allocation)
try bm.bitwiseOrInPlace(&other);
// bm now contains 1, 2, 3, 100..250

// Serialize (CRoaring-compatible wire format)
const bytes = try bm.serialize(std.heap.smp_allocator);
defer std.heap.smp_allocator.free(bytes);

// Deserialize
var restored = try RoaringBitmap.deserialize(std.heap.smp_allocator, bytes);
defer restored.deinit();
```

### OwnedBitmap (recommended for read-heavy patterns)

`OwnedBitmap` uses arena allocation internally — all container memory is freed
in one operation. Faster for deserialize and set operations, but no individual
`remove()`.

```zig
const OwnedBitmap = rawr.OwnedBitmap;

// Deserialize with arena allocation
var owned = try RoaringBitmap.deserializeOwned(std.heap.smp_allocator, bytes);
defer owned.deinit(); // frees everything at once

assert(owned.contains(42));
const card = owned.cardinality();
var it = owned.iterator();

// Set operations
var result = try bm.bitwiseAndOwned(std.heap.smp_allocator, &other);
defer result.deinit();
```

### FrozenBitmap (zero-copy, zero-alloc)

Operates directly on a serialized byte buffer. No deserialization, no heap
allocation. Read-only.

```zig
const FrozenBitmap = rawr.FrozenBitmap;

var frozen = try FrozenBitmap.init(bytes);
defer frozen.deinit();

assert(frozen.contains(42));
const card = frozen.cardinality();
var it = frozen.iterator();
```

## Allocator guidance

Avoid `std.heap.c_allocator` — it is 10-40x slower than alternatives for rawr's
allocation patterns (many small containers). Measured on macOS M4; gap may vary
on other platforms.

Recommended allocators, fastest to most flexible:

| Allocator | Speed | Use when |
|-----------|-------|----------|
| `OwnedBitmap` API | Fastest | Deserialize → query → discard |
| `ArenaAllocator` | Fast | Bounded lifetime, bulk free |
| `smp_allocator` | Good | Long-lived mutable bitmaps |
| `c_allocator` | Avoid | Don't use with rawr |

For hot loops with bounded lifetime (evaluation rounds, request handling),
pre-allocate a `FixedBufferAllocator` and reuse it across iterations.

## Building

Requires Zig 0.15.2+.

```bash
zig build              # build library
zig build test         # run tests
zig build validate     # CRoaring interop validation (18 tests)
zig build bench        # rawr-only benchmarks
zig build bench-compare # rawr vs CRoaring comparison
zig build bench-alloc  # allocator matrix experiment
```

Run benchmarks (results saved to `misc/`):

```bash
./scripts/run-bench.sh           # rawr benchmarks
./scripts/run-compare-bench.sh   # rawr vs CRoaring comparison
./scripts/run-bench-alloc.sh     # allocator matrix experiment
```

## Internals

~9400 lines of Zig across 18 source files. Three container types per the
Roaring spec:

- **Array containers** — sorted u16 arrays for sparse chunks (<4096 values)
- **Bitset containers** — 8KB bitmaps for dense chunks, SIMD via `@Vector(8, u64)`
- **Run containers** — run-length encoded for sequential ranges

Key implementation details:

- SIMD bitset operations (OR/AND/XOR/ANDNOT) via `@Vector`, lowers to
  AVX-512/AVX2/NEON depending on target
- Branchless merge walks for array container intersection/union
- Run-aware `addRange` — creates run containers directly instead of
  element-by-element insertion
- Bulk I/O serialization — descriptive headers and container data written
  in single operations, no per-element loops
- Arena-friendly allocation — all container init/deinit goes through
  `std.mem.Allocator`, works with any Zig allocator

## Project structure

```
src/
  bitmap.zig          # RoaringBitmap, OwnedBitmap (public API)
  bitmap_tests.zig    # unit tests for bitmap.zig
  array_container.zig # sorted u16 array container
  bitset_container.zig# 8KB bitset container (SIMD)
  run_container.zig   # run-length encoded container
  container.zig       # tagged union over container types
  container_ops.zig   # cross-container set operations (9 type pairs)
  serialize.zig       # RoaringFormatSpec serialize/deserialize
  frozen.zig          # FrozenBitmap (zero-copy)
  optimize.zig        # runOptimize, container type conversions
  compare.zig         # isSubsetOf, equals
  format.zig          # format constants
  roaring.zig         # public module root
  bench.zig           # standalone benchmarks
  bench_croaring.zig  # CRoaring comparison benchmarks
  bench_allocators.zig# allocator matrix experiment
  validate_croaring.zig# CRoaring interop validation
  property_tests.zig  # randomized property tests
vendor/
  roaring.c, roaring.h # CRoaring amalgamation (for benchmarks/validation only)
```

## License

TODO
