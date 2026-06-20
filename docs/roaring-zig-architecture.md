# ZigRoar: A High-Performance Roaring Bitmap Implementation in Zig

## Architecture Design Document

---

## 1. Executive Summary

This document architects a native Zig implementation of Roaring Bitmaps, informed by deep analysis of the CRoaring C library (the current state-of-the-art). The design exploits Zig's unique strengths — comptime generics, first-class allocator support, `@Vector` SIMD, tagged unions, and zero-cost abstractions — to achieve performance parity or superiority with CRoaring while providing a dramatically better API.

**Key design departures from CRoaring:**

- **Tagged pointers** for container type discrimination (CRoaring issue #5, never implemented — we do it)
- **Arena-first allocation** with user-supplied `std.mem.Allocator`
- **Comptime container dispatch** instead of C's runtime typecode switch tables
- **`@Vector` portable SIMD** instead of platform-specific intrinsics with `#ifdef` forests
- **Struct-of-Arrays (SoA)** at the top level with cache-line-aligned container storage
- **No copy-on-write** (simplifies the design; COW is rarely beneficial in practice)

---

## 2. Roaring Bitmap Fundamentals (from CRoaring)

A Roaring bitmap partitions the 32-bit integer space into **chunks** of 2^16 values. Each chunk is identified by the **high 16 bits** (the "key") and stored in a **container** holding the **low 16 bits**.

### 2.1 Container Types

| Type | Storage | When Used | Size |
|------|---------|-----------|------|
| **Array** | Sorted `[]u16` | Cardinality ≤ 4096 | 2 × cardinality bytes |
| **Bitset** | `[1024]u64` | Cardinality > 4096 | Fixed 8 KB |
| **Run** | `[]RunPair{start, length}` | Consecutive runs detected | 4 × num_runs bytes |

The **4096 crossover point** is where a sorted array (2 × 4096 = 8192 bytes) equals a bitset (8192 bytes). Above that, the bitset is more compact; below, the array wins.

### 2.2 CRoaring's Internal Layout

CRoaring uses Struct-of-Arrays for the top-level index:

```c
typedef struct roaring_array_s {
    int32_t size;              // number of containers
    int32_t allocation_size;   // capacity
    void **containers;         // array of container pointers
    uint16_t *keys;            // array of high-16-bit keys (sorted)
    uint8_t *typecodes;        // array of container type tags
    uint8_t flags;             // COW, frozen flags
} roaring_array_t;
```

This SoA layout is intentional: binary searching `keys[]` touches only a compact u16 array, keeping hot data in cache. The `containers[]` and `typecodes[]` arrays are only accessed after the key is located.

### 2.3 SIMD Optimizations in CRoaring

CRoaring uses platform-specific SIMD for:
- **Bitset ∩/∪/−/⊕ bitset**: AVX2/AVX-512/NEON bitwise ops + popcount
- **Array ∩ array**: Vectorized sorted-set intersection (galloping + SIMD merge)
- **Array ∪ array**: SIMD merge of sorted arrays
- **Bitset popcount**: Hardware `POPCNT` or `__builtin_popcountll`

---

## 3. Zig Architecture

### 3.1 Project Structure

```
zigroar/
├── build.zig
├── build.zig.zon
├── src/
│   ├── RoaringBitmap.zig      # Public API (the only import users need)
│   ├── container.zig           # Container tagged union + dispatch
│   ├── array_container.zig     # Sorted u16 array container
│   ├── bitset_container.zig    # Fixed 8KB bitset container
│   ├── run_container.zig       # Run-length encoded container
│   ├── container_ops.zig       # Cross-container operations (array∩bitset, etc.)
│   ├── simd.zig                # Portable SIMD utilities via @Vector
│   ├── serialization.zig       # RoaringFormatSpec compatible ser/de
│   └── util.zig                # Binary search, popcount, bit tricks
├── tests/
│   ├── unit/
│   │   ├── array_container_test.zig
│   │   ├── bitset_container_test.zig
│   │   ├── run_container_test.zig
│   │   ├── container_ops_test.zig
│   │   ├── bitmap_test.zig
│   │   └── serialization_test.zig
│   ├── fuzz/
│   │   ├── fuzz_bitmap_ops.zig
│   │   └── fuzz_serialization.zig
│   ├── property/
│   │   └── set_algebra_properties.zig
│   └── interop/
│       └── croaring_compat_test.zig
├── bench/
│   ├── bench_main.zig
│   ├── bench_containers.zig
│   ├── bench_bitmap_ops.zig
│   ├── bench_serialization.zig
│   ├── bench_iteration.zig
│   └── datasets/              # Real-world datasets from CRoaring benchmarks
│       ├── census1881.txt
│       ├── wikileaks.txt
│       └── weather_sept_85.txt
└── tools/
    ├── compare_croaring.zig    # Head-to-head benchmark vs CRoaring via C interop
    └── gen_dataset.zig         # Synthetic dataset generator
```

### 3.2 Top-Level Bitmap Structure (SoA with Tagged Pointers)

```zig
pub const RoaringBitmap = struct {
    /// Sorted array of 16-bit chunk keys (high bits of contained values).
    /// Aligned to 64 bytes for SIMD binary search.
    keys: []align(64) u16,

    /// Array of tagged container pointers.
    /// Low 2 bits encode the container type (pointers are at least 8-byte aligned).
    ///   00 = array container
    ///   01 = bitset container
    ///   10 = run container
    ///   11 = reserved (future: frozen/shared)
    containers: []TaggedPtr,

    /// Number of active containers (keys.len == containers.len == size).
    size: u32,

    /// Allocated capacity for keys/containers arrays.
    capacity: u32,

    /// User-provided allocator for all internal memory.
    allocator: std.mem.Allocator,

    // ── Tagged pointer encoding ──────────────────────────────────

    pub const TaggedPtr = packed struct(u64) {
        tag: ContainerType,       // bits [0:1]
        addr: u62,                // bits [2:63] — pointer with low bits masked

        pub const ContainerType = enum(u2) {
            array = 0b00,
            bitset = 0b01,
            run = 0b10,
            reserved = 0b11,
        };

        pub fn init(comptime T: type, ptr: *T, tag: ContainerType) TaggedPtr {
            const raw = @intFromPtr(ptr);
            std.debug.assert(raw & 0x3 == 0); // must be 4-byte aligned minimum
            return .{
                .tag = tag,
                .addr = @truncate(raw >> 2),
            };
        }

        pub fn getPtr(self: TaggedPtr, comptime T: type) *T {
            return @ptrFromInt(@as(u64, self.addr) << 2);
        }

        pub fn getType(self: TaggedPtr) ContainerType {
            return self.tag;
        }
    };
};
```

**Why tagged pointers instead of a separate `typecodes[]` array:**
CRoaring keeps `typecodes` separate because C can't do packed tagged pointers cleanly. In Zig, we pack the type into the pointer's low bits (all container allocations are ≥ 8-byte aligned). This eliminates one array, improves cache behavior during container dispatch, and reduces the SoA from 3 parallel arrays to 2.

### 3.3 Container Definitions

#### 3.3.1 Array Container

```zig
pub const ArrayContainer = struct {
    /// Sorted array of low-16-bit values.
    /// Capacity is always a power of 2, minimum 4.
    /// Length (cardinality) is stored separately.
    values: []u16,
    cardinality: u16, // [1, 4096]
    capacity: u16,

    /// Allocate with cache-line-friendly sizing.
    pub fn init(allocator: std.mem.Allocator, initial_cap: u16) !*ArrayContainer {
        const self = try allocator.create(ArrayContainer);
        // Round up to power of 2, minimum 4 elements (8 bytes)
        const cap = @max(4, std.math.ceilPowerOfTwo(u16, initial_cap) catch 4096);
        self.* = .{
            .values = try allocator.alignedAlloc(u16, 32, cap), // 32-byte aligned for SIMD
            .cardinality = 0,
            .capacity = cap,
        };
        return self;
    }

    pub fn deinit(self: *ArrayContainer, allocator: std.mem.Allocator) void {
        allocator.free(self.values[0..self.capacity]);
        allocator.destroy(self);
    }

    /// Binary search using @Vector SIMD when array is large enough.
    pub fn contains(self: *const ArrayContainer, value: u16) bool {
        if (self.cardinality >= 32) {
            return simdBinarySearch(self.values[0..self.cardinality], value);
        }
        return std.sort.binarySearch(u16, self.values[0..self.cardinality], value) != null;
    }

    /// Insert maintaining sorted order. Returns true if value was new.
    pub fn add(self: *ArrayContainer, allocator: std.mem.Allocator, value: u16) !bool {
        // ... grow if needed, shift right, insert at position
    }
};
```

#### 3.3.2 Bitset Container

```zig
pub const BitsetContainer = struct {
    /// Fixed 8KB bitset: 1024 × u64 words.
    /// Aligned to 64 bytes (cache line) for optimal SIMD access.
    words: *align(64) [1024]u64,
    cardinality: i32, // -1 = unknown (lazy computation)

    pub fn init(allocator: std.mem.Allocator) !*BitsetContainer {
        const self = try allocator.create(BitsetContainer);
        self.* = .{
            .words = try allocator.alignedAlloc(u64, 64, 1024),
            .cardinality = 0,
        };
        @memset(self.words, 0);
        return self;
    }

    pub fn contains(self: *const BitsetContainer, value: u16) bool {
        return (self.words[value >> 6] & (@as(u64, 1) << @truncate(value & 63))) != 0;
    }

    pub fn add(self: *BitsetContainer, value: u16) bool {
        const word = &self.words[value >> 6];
        const bit = @as(u64, 1) << @truncate(value & 63);
        const was_absent = (word.* & bit) == 0;
        word.* |= bit;
        if (was_absent and self.cardinality >= 0) self.cardinality += 1;
        return was_absent;
    }

    /// SIMD-accelerated OR with popcount using @Vector(8, u64).
    pub fn unionWith(dst: *BitsetContainer, src: *const BitsetContainer) void {
        const VEC_SIZE = 8; // 512 bits = 64 bytes = 1 cache line
        const vec_count = 1024 / VEC_SIZE; // 128 iterations

        var card: u64 = 0;
        for (0..vec_count) |i| {
            const a: @Vector(VEC_SIZE, u64) = dst.words[i * VEC_SIZE ..][0..VEC_SIZE].*;
            const b: @Vector(VEC_SIZE, u64) = src.words[i * VEC_SIZE ..][0..VEC_SIZE].*;
            const result = a | b;
            dst.words[i * VEC_SIZE ..][0..VEC_SIZE].* = result;
            // Accumulate popcount
            inline for (0..VEC_SIZE) |j| {
                card += @popCount(result[j]);
            }
        }
        dst.cardinality = @intCast(card);
    }

    /// SIMD-accelerated AND with early-exit when cardinality is clearly zero.
    pub fn intersectionWith(dst: *BitsetContainer, src: *const BitsetContainer) void {
        const VEC_SIZE = 8;
        const vec_count = 1024 / VEC_SIZE;

        var card: u64 = 0;
        for (0..vec_count) |i| {
            const a: @Vector(VEC_SIZE, u64) = dst.words[i * VEC_SIZE ..][0..VEC_SIZE].*;
            const b: @Vector(VEC_SIZE, u64) = src.words[i * VEC_SIZE ..][0..VEC_SIZE].*;
            const result = a & b;
            dst.words[i * VEC_SIZE ..][0..VEC_SIZE].* = result;
            inline for (0..VEC_SIZE) |j| {
                card += @popCount(result[j]);
            }
        }
        dst.cardinality = @intCast(card);
    }
};
```

#### 3.3.3 Run Container

```zig
pub const RunContainer = struct {
    pub const RunPair = packed struct {
        start: u16,
        length: u16, // number of values AFTER start (so run covers start..start+length inclusive)
    };

    runs: []RunPair,
    n_runs: u16,
    capacity: u16,

    pub fn init(allocator: std.mem.Allocator, initial_cap: u16) !*RunContainer {
        const self = try allocator.create(RunContainer);
        const cap = @max(4, initial_cap);
        self.* = .{
            .runs = try allocator.alloc(RunPair, cap),
            .n_runs = 0,
            .capacity = cap,
        };
        return self;
    }

    pub fn contains(self: *const RunContainer, value: u16) bool {
        // Binary search on run starts, then check if value falls within the run.
        var lo: usize = 0;
        var hi: usize = self.n_runs;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const run = self.runs[mid];
            if (value < run.start) {
                hi = mid;
            } else if (value > run.start + run.length) {
                lo = mid + 1;
            } else {
                return true;
            }
        }
        return false;
    }

    /// Cardinality is sum of all (length + 1) values.
    pub fn getCardinality(self: *const RunContainer) u32 {
        var card: u32 = 0;
        for (self.runs[0..self.n_runs]) |run| {
            card += @as(u32, run.length) + 1;
        }
        return card;
    }
};
```

### 3.4 Container Dispatch (Comptime Polymorphism)

Instead of CRoaring's massive switch tables dispatching on `(typecode_a, typecode_b)` pairs, we use Zig's comptime:

```zig
/// Unified container handle — replaces CRoaring's void* + typecode pattern.
pub const Container = union(enum) {
    array: *ArrayContainer,
    bitset: *BitsetContainer,
    run: *RunContainer,

    /// Decode from a tagged pointer.
    pub fn fromTagged(tp: RoaringBitmap.TaggedPtr) Container {
        return switch (tp.getType()) {
            .array => .{ .array = tp.getPtr(ArrayContainer) },
            .bitset => .{ .bitset = tp.getPtr(BitsetContainer) },
            .run => .{ .run = tp.getPtr(RunContainer) },
            .reserved => unreachable,
        };
    }

    pub fn contains(self: Container, value: u16) bool {
        return switch (self) {
            .array => |c| c.contains(value),
            .bitset => |c| c.contains(value),
            .run => |c| c.contains(value),
        };
    }

    pub fn getCardinality(self: Container) u32 {
        return switch (self) {
            .array => |c| c.cardinality,
            .bitset => |c| if (c.cardinality >= 0)
                @intCast(c.cardinality)
            else
                c.computeCardinality(),
            .run => |c| c.getCardinality(),
        };
    }
};
```

### 3.5 Cross-Container Operations

The 3 container types yield 9 pairwise combinations for each set operation. This is the heart of roaring bitmap performance.

```zig
/// container_ops.zig — all 9 pairwise combinations for each op

/// Returns a new container representing the union of a and b.
pub fn containerUnion(
    allocator: std.mem.Allocator,
    a: Container,
    b: Container,
) !struct { container: Container, needs_type_conversion: bool } {
    return switch (a) {
        .array => |ac| switch (b) {
            .array => |bc| arrayUnionArray(allocator, ac, bc),
            .bitset => |bc| arrayUnionBitset(allocator, ac, bc),
            .run => |rc| arrayUnionRun(allocator, ac, rc),
        },
        .bitset => |ac| switch (b) {
            .array => |bc| arrayUnionBitset(allocator, bc, ac), // commutative
            .bitset => |bc| bitsetUnionBitset(allocator, ac, bc),
            .run => |rc| bitsetUnionRun(allocator, ac, rc),
        },
        .run => |ac| switch (b) {
            .array => |bc| arrayUnionRun(allocator, bc, ac),
            .bitset => |bc| bitsetUnionRun(allocator, bc, ac),
            .run => |rc| runUnionRun(allocator, ac, rc),
        },
    };
}

// ── Key algorithmic strategies per pair ──

fn arrayUnionArray(alloc: std.mem.Allocator, a: *ArrayContainer, b: *ArrayContainer) !Container {
    // If combined cardinality > 4096, produce a bitset directly.
    // Otherwise, SIMD merge of two sorted arrays.
    const max_card = @as(u32, a.cardinality) + b.cardinality;
    if (max_card > 4096) {
        // Scatter both arrays into a fresh bitset — O(n) with no sorting.
        var bs = try BitsetContainer.init(alloc);
        for (a.values[0..a.cardinality]) |v| bs.setBit(v);
        for (b.values[0..b.cardinality]) |v| bs.setBit(v);
        bs.computeCardinality();
        return .{ .bitset = bs };
    }
    // Merge two sorted u16 arrays using SIMD merge network.
    return .{ .array = try simd.mergeSortedArrays(alloc, a, b) };
}

fn arrayIntersectArray(alloc: std.mem.Allocator, a: *ArrayContainer, b: *ArrayContainer) !Container {
    // Galloping intersection: if sizes differ by 64x+, gallop the small one
    // through the large one. Otherwise, SIMD vectorized intersection.
    if (a.cardinality * 64 < b.cardinality) {
        return gallopingIntersect(alloc, a, b);
    }
    return simd.vectorizedIntersect(alloc, a, b);
}

fn bitsetUnionBitset(alloc: std.mem.Allocator, a: *BitsetContainer, b: *BitsetContainer) !Container {
    // Pure SIMD OR + popcount. Always produces a bitset.
    var result = try BitsetContainer.init(alloc);
    BitsetContainer.unionWith(result, a);
    // result already has a's data; OR in b.
    BitsetContainer.unionWith(result, b);
    return .{ .bitset = result };
}

fn bitsetIntersectBitset(alloc: std.mem.Allocator, a: *BitsetContainer, b: *BitsetContainer) !Container {
    // SIMD AND + popcount. If result cardinality ≤ 4096, convert to array.
    var result = try BitsetContainer.init(alloc);
    // ... AND words ...
    if (result.cardinality <= 4096) {
        // Convert bitset → array for better compression.
        return .{ .array = try bitsetToArray(alloc, result) };
    }
    return .{ .bitset = result };
}
```

### 3.6 SIMD Utilities (`simd.zig`)

```zig
const std = @import("std");

/// Portable SIMD binary search for a u16 value in a sorted slice.
/// Uses @Vector(16, u16) = 256-bit vector to compare 16 elements at once.
pub fn simdBinarySearch(haystack: []const u16, needle: u16) bool {
    const VEC_LEN = 16;
    const needle_vec: @Vector(VEC_LEN, u16) = @splat(needle);

    var lo: usize = 0;
    var hi: usize = haystack.len;

    // SIMD phase: narrow the window by 16 elements at a time
    while (hi - lo >= VEC_LEN) {
        const mid = lo + (hi - lo) / 2;
        // Aligned load of 16 consecutive u16 values
        const chunk: @Vector(VEC_LEN, u16) = haystack[mid..][0..VEC_LEN].*;
        const eq_mask = chunk == needle_vec;
        if (@reduce(.Or, eq_mask)) return true;

        // Compare against the first element to decide direction
        if (needle < haystack[mid]) {
            hi = mid;
        } else {
            lo = mid + VEC_LEN;
        }
    }

    // Scalar fallback for the remaining < 16 elements
    for (haystack[lo..hi]) |v| {
        if (v == needle) return true;
        if (v > needle) break;
    }
    return false;
}

/// SIMD-accelerated merge of two sorted u16 arrays (for array∪array).
/// Uses a bitonic merge network on @Vector(8, u16) chunks.
pub fn mergeSortedArrays(
    allocator: std.mem.Allocator,
    a: *const ArrayContainer,
    b: *const ArrayContainer,
) !*ArrayContainer {
    // Allocate output with exact max capacity
    const max_len = @as(u32, a.cardinality) + b.cardinality;
    var result = try ArrayContainer.init(allocator, @intCast(@min(max_len, 4096)));

    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;
    const sa = a.values[0..a.cardinality];
    const sb = b.values[0..b.cardinality];

    // Standard merge with duplicate elimination
    while (i < sa.len and j < sb.len) {
        if (sa[i] < sb[j]) {
            result.values[k] = sa[i];
            i += 1;
        } else if (sa[i] > sb[j]) {
            result.values[k] = sb[j];
            j += 1;
        } else {
            result.values[k] = sa[i];
            i += 1;
            j += 1;
        }
        k += 1;
    }
    // Copy remainder
    while (i < sa.len) : (i += 1) {
        result.values[k] = sa[i];
        k += 1;
    }
    while (j < sb.len) : (j += 1) {
        result.values[k] = sb[j];
        k += 1;
    }

    result.cardinality = @intCast(k);
    return result;
}

/// Vectorized sorted-set intersection using SIMD comparison + shuffle.
/// Based on Schlegel et al. "Fast Sorted-Set Intersection using SIMD Instructions"
/// adapted from CRoaring's approach.
pub fn vectorizedIntersect(
    allocator: std.mem.Allocator,
    a: *const ArrayContainer,
    b: *const ArrayContainer,
) !*ArrayContainer {
    // Implementation uses @Vector(8, u16) broadcast-compare approach:
    // For each element in the smaller set, broadcast it and compare against
    // a vector-width chunk of the larger set.
    _ = allocator;
    _ = a;
    _ = b;
    // Full implementation would go here
    @compileError("TODO: implement vectorized intersection");
}
```

### 3.7 Container Type Conversion

Containers morph between types as cardinality changes:

```zig
/// Possibly convert a container to a more efficient representation.
/// Called after mutations that change cardinality.
pub fn optimizeContainer(
    allocator: std.mem.Allocator,
    container: Container,
) !Container {
    switch (container) {
        .array => |ac| {
            // Check if runs would be more compact
            const n_runs = countRuns(ac);
            if (n_runs * 4 < ac.cardinality * 2) {
                return .{ .run = try arrayToRun(allocator, ac) };
            }
            return container; // stay as array
        },
        .bitset => |bc| {
            const card = bc.getCardinality();
            if (card <= 4096) {
                // Convert to array
                return .{ .array = try bitsetToArray(allocator, bc) };
            }
            // Check if runs would be more compact
            const n_runs = bitsetCountRuns(bc);
            if (n_runs * 4 < 8192) { // 8KB bitset threshold
                return .{ .run = try bitsetToRun(allocator, bc) };
            }
            return container;
        },
        .run => |rc| {
            const card = rc.getCardinality();
            const run_bytes = @as(u32, rc.n_runs) * 4;
            if (card <= 4096 and card * 2 < run_bytes) {
                return .{ .array = try runToArray(allocator, rc) };
            }
            if (card > 4096 and 8192 < run_bytes) {
                return .{ .bitset = try runToBitset(allocator, rc) };
            }
            return container;
        },
    }
}
```

---

## 4. Public API Design

The API should feel natural to Zig programmers. One import, one type, allocator-aware.

```zig
const std = @import("std");
const RoaringBitmap = @import("zigroar").RoaringBitmap;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // ── Construction ──
    var bm = try RoaringBitmap.init(alloc);
    defer bm.deinit();

    // ── Single-element operations ──
    try bm.add(42);
    try bm.addRange(100, 1000);         // [100, 1000)
    try bm.addMany(&[_]u32{ 5, 10, 15, 20 });
    _ = bm.remove(42);
    const present = bm.contains(100);    // true

    // ── Bulk context (amortized O(1) for clustered inserts) ──
    {
        var ctx = bm.bulkContext();
        for (0..100_000) |i| {
            try ctx.add(@intCast(i * 3));
        }
    }

    // ── Set algebra (return new bitmap) ──
    var other = try RoaringBitmap.init(alloc);
    try other.addRange(500, 1500);

    var union_bm = try bm.bitwiseOr(alloc, other);
    var inter_bm = try bm.bitwiseAnd(alloc, other);
    var diff_bm  = try bm.bitwiseDifference(alloc, other);
    var xor_bm   = try bm.bitwiseXor(alloc, other);
    defer union_bm.deinit();
    defer inter_bm.deinit();
    defer diff_bm.deinit();
    defer xor_bm.deinit();

    // ── In-place set algebra ──
    try bm.bitwiseOrInPlace(other);
    try bm.bitwiseAndInPlace(other);

    // ── Statistics ──
    const card = bm.cardinality();
    const empty = bm.isEmpty();
    const min_val = bm.minimum();        // ?u32
    const max_val = bm.maximum();        // ?u32
    const is_subset = bm.isSubsetOf(other);
    const jaccard = bm.jaccardIndex(other);

    // ── Rank & Select ──
    const rank = bm.rank(500);           // # of values ≤ 500
    const val = bm.select(10);           // 10th smallest value (?u32)

    // ── Iteration ──
    var it = bm.iterator();
    while (it.next()) |value| {
        _ = value;
    }

    // ── Batch read into buffer ──
    var buf: [1024]u32 = undefined;
    const n = bm.readInto(&buf);         // fills buf, returns count
    _ = n;

    // ── Serialization (RoaringFormatSpec compatible) ──
    const bytes = try bm.serialize(alloc);     // interoperable with Java/Go/C
    var restored = try RoaringBitmap.deserialize(alloc, bytes);
    defer restored.deinit();

    // ── Optimization ──
    try bm.runOptimize();     // convert to runs where beneficial
    try bm.shrinkToFit();     // release excess capacity

    // ── Debug ──
    // Debug output can be written through a std.Io.Writer.

    _ = present;
    _ = empty;
    _ = min_val;
    _ = max_val;
    _ = is_subset;
    _ = jaccard;
    _ = val;
}
```

### 4.1 Bulk Context

CRoaring introduced `roaring_bulk_context_t` to amortize container lookup for sequential inserts. We replicate this:

```zig
pub const BulkContext = struct {
    /// Cached container index from last operation.
    last_key: u16 = 0,
    last_index: u32 = 0,
    last_container: ?Container = null,

    pub fn add(self: *BulkContext, bm: *RoaringBitmap, value: u32) !void {
        const key: u16 = @intCast(value >> 16);
        const low: u16 = @intCast(value & 0xFFFF);

        // Fast path: same container as last time
        if (self.last_container != null and self.last_key == key) {
            _ = try self.last_container.?.addToContainer(bm.allocator, low);
            return;
        }

        // Slow path: look up and cache
        const idx = bm.findContainerIndex(key);
        // ... update cache ...
    }
};
```

---

## 5. Memory Architecture

### 5.1 Allocation Strategy

```
┌─────────────────────────────────────────────────────────────┐
│  User provides: std.mem.Allocator                            │
│  (ArenaAllocator recommended for batch workloads)            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Top-level SoA arrays:                                       │
│    keys[]      — aligned(64), grows by 2x                    │
│    containers[] — aligned(8), grows in lockstep with keys    │
│                                                              │
│  Per-container allocations:                                   │
│    ArrayContainer  — struct + aligned(32) u16 values[]       │
│    BitsetContainer — struct + aligned(64) [1024]u64 words    │
│    RunContainer    — struct + RunPair runs[]                  │
│                                                              │
│  All allocations go through the single user Allocator.       │
│  No global state. No hidden mallocs.                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Cache Locality Considerations

| Access Pattern | Design Decision |
|---|---|
| Key lookup (binary search) | `keys[]` is contiguous `u16` — 32 keys per cache line. Binary search on 1000 containers touches ~10 cache lines. |
| Container dispatch | Tagged pointer embeds type in the pointer itself — single load, no `typecodes[]` indirection. |
| Bitset operations | `words` aligned to 64 bytes = exact cache-line boundary. SIMD loops process one cache line per iteration. |
| Array container search | `values` aligned to 32 bytes for `@Vector(16, u16)` loads. |
| Sequential iteration | Iterator prefetches next container's data while yielding current values. |
| Bulk insert | `BulkContext` caches last container, avoiding repeated binary search. |

### 5.3 Growth Strategy

```
keys/containers arrays:
  Initial capacity: 4
  Growth: 2x (4 → 8 → 16 → 32 → ...)
  Max: 65536 (one container per possible key)

Array container values:
  Initial capacity: max(4, expected_cardinality)
  Growth: min(2x, 4096)
  Shrink: when cardinality < capacity/4

Run container runs:
  Initial capacity: max(4, n_runs)
  Growth: 2x
  Shrink: when n_runs < capacity/4
```

---

## 6. Serialization (RoaringFormatSpec Compatibility)

Full interoperability with Java, Go, C, Rust implementations.

```
┌─────────────────────────────────────────────────────┐
│ Cookie Header (4 or 8 bytes)                         │
│   - SERIAL_COOKIE (with run containers)              │
│   - SERIAL_COOKIE_NO_RUNCONTAINER (without)          │
├─────────────────────────────────────────────────────┤
│ Run Container Bitset (⌈size/8⌉ bytes, if cookie)    │
├─────────────────────────────────────────────────────┤
│ Descriptive Header (4 × size bytes)                  │
│   - key (u16) + cardinality-1 (u16) per container   │
├─────────────────────────────────────────────────────┤
│ Offset Header (4 × size bytes, if size ≥ 4)         │
├─────────────────────────────────────────────────────┤
│ Container Data (variable)                            │
│   Array: sorted u16[] (2 × cardinality bytes)       │
│   Bitset: 1024 × u64 (8192 bytes, little-endian)   │
│   Run: u16 n_runs + n_runs × (start, length) pairs  │
└─────────────────────────────────────────────────────┘
```

Implementation:

```zig
pub fn serialize(self: *const RoaringBitmap, allocator: std.mem.Allocator) ![]u8 {
    const size_bytes = self.portableSizeInBytes();
    var buf = try allocator.alloc(u8, size_bytes);

    var writer = std.Io.Writer.fixed(buf);

    const has_runs = self.hasRunContainers();
    if (has_runs) {
        // Cookie with run flag
        const cookie: u32 = SERIAL_COOKIE | (@as(u32, self.size - 1) << 16);
        try writer.writeInt(u32, cookie, .little);
        // Run container bitset
        try self.writeRunBitset(writer);
    } else {
        try writer.writeInt(u32, SERIAL_COOKIE_NO_RUNCONTAINER, .little);
        try writer.writeInt(u32, self.size, .little);
    }

    // Descriptive header: key + (cardinality - 1) pairs
    for (0..self.size) |i| {
        try writer.writeInt(u16, self.keys[i], .little);
        const card = self.containers[i].fromTagged().getCardinality();
        try writer.writeInt(u16, @intCast(card - 1), .little);
    }

    // Offset header (if size >= NO_OFFSET_THRESHOLD)
    // ... then container data ...

    return buf;
}

pub fn deserialize(allocator: std.mem.Allocator, data: []const u8) !RoaringBitmap {
    var reader = std.Io.Reader.fixed(data);
    // Read cookie, determine format, parse headers, load containers...
    // Validate structural invariants before returning.
    _ = reader;
    _ = allocator;
    @compileError("TODO: implement deserialization");
}
```

---

## 7. Test Plan

### 7.1 Unit Tests (per container, per operation)

| Component | Test Cases |
|---|---|
| `ArrayContainer` | add, remove, contains, grow, shrink, iteration, sorted invariant, edge cases (empty, full at 4096) |
| `BitsetContainer` | add, remove, contains, popcount, word-boundary values, full/empty bitsets |
| `RunContainer` | add, remove, contains, merge adjacent runs, split runs, cardinality |
| `Container conversions` | array→bitset (at 4096 threshold), bitset→array (dropping below), array→run, bitset→run, run→array, run→bitset |
| `Cross-container ops` | All 9 pairings × 4 operations (∪, ∩, −, ⊕) = 36 test suites |
| `TaggedPtr` | Round-trip encode/decode, alignment assertions |
| `Serialization` | Round-trip ser/de, cross-implementation test vectors from RoaringFormatSpec |
| `Top-level bitmap` | add/remove/contains, range operations, rank/select, min/max, cardinality, isEmpty, equals |
| `BulkContext` | Sequential inserts, interleaved keys, single-container workload |
| `Iterator` | Forward iteration, empty bitmap, single element, cross-container boundaries |

### 7.2 Property-Based Tests (set algebra axioms)

These verify correctness by treating roaring bitmaps as mathematical sets and checking algebraic identities against a reference `std.AutoHashMap(u32, void)`:

```zig
test "set algebra properties" {
    var prng = std.Random.DefaultPrng.init(seed);

    // Generate random bitmaps A, B, C
    for (0..1000) |_| {
        const a = randomBitmap(&prng, alloc);
        const b = randomBitmap(&prng, alloc);
        const c = randomBitmap(&prng, alloc);

        // Commutativity: A ∪ B = B ∪ A
        try expectEqual(a.bitwiseOr(b), b.bitwiseOr(a));

        // Associativity: (A ∪ B) ∪ C = A ∪ (B ∪ C)
        try expectEqual(a.bitwiseOr(b).bitwiseOr(c), a.bitwiseOr(b.bitwiseOr(c)));

        // Distributivity: A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)
        try expectEqual(
            a.bitwiseAnd(b.bitwiseOr(c)),
            a.bitwiseAnd(b).bitwiseOr(a.bitwiseAnd(c)),
        );

        // De Morgan: ¬(A ∪ B) = ¬A ∩ ¬B (over a finite universe)

        // Identity: A ∪ ∅ = A
        try expectEqual(a.bitwiseOr(empty), a);

        // Idempotence: A ∪ A = A
        try expectEqual(a.bitwiseOr(a), a);

        // Complement: A − A = ∅
        try expect(a.bitwiseDifference(a).isEmpty());

        // Cardinality: |A ∪ B| + |A ∩ B| = |A| + |B|
        try expectEqual(
            a.bitwiseOr(b).cardinality() + a.bitwiseAnd(b).cardinality(),
            a.cardinality() + b.cardinality(),
        );

        // Subset transitivity
        const ab = a.bitwiseAnd(b);
        try expect(ab.isSubsetOf(a));
        try expect(ab.isSubsetOf(b));
    }
}
```

### 7.3 Fuzz Testing

```zig
// fuzz_bitmap_ops.zig — AFL/libFuzzer compatible
pub export fn zig_fuzz_test(data: [*]const u8, len: usize) void {
    // Interpret fuzz input as a sequence of commands:
    //   0x00 <u32>           → add(value)
    //   0x01 <u32>           → remove(value)
    //   0x02 <u32>           → contains(value) — verify against reference
    //   0x03 <u32> <u32>     → addRange(lo, hi)
    //   0x10                 → bitwiseOrInPlace(shadow)
    //   0x11                 → bitwiseAndInPlace(shadow)
    //   0x20                 → serialize + deserialize round-trip
    //   0x30                 → runOptimize
    //
    // Maintain a parallel std.DynamicBitSet as ground truth.
    // Assert roaring.contains(x) == reference.isSet(x) after every op.
}
```

### 7.4 Interoperability Tests

```zig
// Use CRoaring (linked via C interop) as reference implementation.
test "interop: serialization compatibility with CRoaring" {
    // 1. Create bitmap in ZigRoar, serialize
    // 2. Deserialize in CRoaring, verify cardinality + contents
    // 3. Create bitmap in CRoaring, serialize
    // 4. Deserialize in ZigRoar, verify cardinality + contents
    // 5. Use test vectors from RoaringFormatSpec testdata/
}
```

---

## 8. Benchmark Plan

### 8.1 Microbenchmarks (per-operation)

All benchmarks use `std.time.Timer` with warmup, report median of N runs, and flush caches between iterations.

| Benchmark | What it measures |
|---|---|
| `bench_add_sequential` | Insert 1M sequential values |
| `bench_add_random` | Insert 1M random values |
| `bench_add_bulk` | Insert 1M values via BulkContext |
| `bench_contains_hit` | 1M lookups (all present) |
| `bench_contains_miss` | 1M lookups (all absent) |
| `bench_union_sparse` | Union of two 10K-element bitmaps |
| `bench_union_dense` | Union of two 10M-element bitmaps |
| `bench_intersection_sparse` | Intersection, sparse |
| `bench_intersection_dense` | Intersection, dense |
| `bench_difference` | Difference (asymmetric) |
| `bench_xor` | Symmetric difference |
| `bench_iterate_all` | Full forward iteration |
| `bench_iterate_batch` | Batch read into buffer (1K at a time) |
| `bench_rank_select` | 100K rank + select queries |
| `bench_serialize` | Serialize 1M-element bitmap |
| `bench_deserialize` | Deserialize from bytes |
| `bench_run_optimize` | Optimize container types |
| `bench_memory` | Peak RSS for various workloads |

### 8.2 Container-Level Microbenchmarks

| Benchmark | Measures |
|---|---|
| `bench_bitset_or` | Raw 8KB bitset OR + popcount (SIMD vs scalar) |
| `bench_bitset_and` | Raw 8KB bitset AND + popcount |
| `bench_array_intersect` | Sorted array intersection (galloping vs SIMD vs scalar) |
| `bench_array_union` | Sorted array merge |
| `bench_array_binsearch` | Binary search in sorted u16 array (SIMD vs scalar) |
| `bench_container_conversion` | Array→bitset, bitset→array, array→run transitions |

### 8.3 Real-World Dataset Benchmarks

Using the same datasets as CRoaring's published benchmarks:

| Dataset | Description | Source |
|---|---|---|
| `census1881` | Canadian census, sorted | CRoaring/benchmarks/realdata |
| `census1881_srt` | Same, sorted by set | " |
| `weather_sept_85` | Weather station data | " |
| `wikileaks-noquotes` | WikiLeaks cable IDs | " |
| `census-income` | US census income | " |
| `census-income_srt` | Same, sorted | " |
| `uscensus2000` | US 2000 census | " |

For each dataset, measure:
- Time to construct all bitmaps
- Time to compute pairwise union of all bitmaps
- Time to compute pairwise intersection of all bitmaps
- Total memory usage (bits per value)
- Serialized size (bits per value)

### 8.4 Head-to-Head vs CRoaring

The `tools/compare_croaring.zig` tool links against CRoaring via a build-system translated C module and runs identical operations on identical data:

```zig
// compare_croaring.zig
const c = @import("c");

fn benchmarkBoth(dataset: []const []const u32) void {
    // ZigRoar
    const zig_start = timer.read();
    for (dataset) |set| {
        var bm = RoaringBitmap.fromSorted(alloc, set);
        // ...
    }
    const zig_elapsed = timer.read() - zig_start;

    // CRoaring
    const c_start = timer.read();
    for (dataset) |set| {
        var bm = c.roaring_bitmap_of_ptr(set.len, set.ptr);
        // ...
    }
    const c_elapsed = timer.read() - c_start;

    printComparison("construct", zig_elapsed, c_elapsed);
}
```

**Output format:**

```
╔══════════════════════════════════════════════════════════════╗
║  ZigRoar vs CRoaring — census1881 dataset                   ║
╠══════════════════╦══════════╦══════════╦═════════════════════╣
║ Operation        ║ ZigRoar  ║ CRoaring ║ Speedup             ║
╠══════════════════╬══════════╬══════════╬═════════════════════╣
║ construct        ║   1.2 ms ║   1.4 ms ║ 1.17x faster        ║
║ union (all)      ║   0.8 ms ║   0.9 ms ║ 1.12x faster        ║
║ intersect (all)  ║   0.3 ms ║   0.3 ms ║ 1.00x (parity)      ║
║ iterate (all)    ║   0.5 ms ║   0.6 ms ║ 1.20x faster        ║
║ serialize        ║   0.2 ms ║   0.2 ms ║ 1.05x faster        ║
║ bits/value       ║    2.77  ║    2.77  ║ identical            ║
╚══════════════════╩══════════╩══════════╩═════════════════════╝
```

### 8.5 Performance Targets

| Metric | Target | Rationale |
|---|---|---|
| Single `add` | < 50 ns amortized | CRoaring achieves ~40-60ns |
| Single `contains` | < 30 ns | CRoaring ~20-30ns |
| Bulk `add` (sequential) | < 10 ns/element | BulkContext eliminates binary search |
| Bitset ∪ Bitset | < 1 μs | Pure SIMD on 8KB; memory-bandwidth bound |
| Array ∩ Array (1K each) | < 2 μs | SIMD vectorized intersection |
| Full iteration (1M values) | < 1 ms | Batch reads with prefetching |
| Serialize (1M values) | < 500 μs | memcpy-dominated |
| Memory (bits/value) | ≤ CRoaring | Same format, no overhead |

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] `ArrayContainer` — add, remove, contains, iteration, grow/shrink
- [ ] `BitsetContainer` — add, remove, contains, popcount, SIMD OR/AND
- [ ] `RunContainer` — add, remove, contains, cardinality
- [ ] `TaggedPtr` encoding/decoding
- [ ] Unit tests for all three container types
- [ ] `RoaringBitmap` — init, deinit, add, remove, contains with key-lookup

### Phase 2: Set Operations (Week 3-4)
- [ ] All 9 container pairings × 4 set operations (union, intersection, difference, xor)
- [ ] In-place variants of all set operations
- [ ] Container type conversion logic (array↔bitset, array↔run, bitset↔run)
- [ ] `runOptimize`, `shrinkToFit`
- [ ] Property-based tests (set algebra axioms)

### Phase 3: Query & Iteration (Week 5)
- [ ] `rank`, `select`, `minimum`, `maximum`
- [ ] Forward iterator with cross-container boundary handling
- [ ] Batch `readInto` buffer
- [ ] `BulkContext` for amortized inserts
- [ ] `addRange`, `removeRange`, `containsRange`
- [ ] `isSubsetOf`, `equals`, `jaccardIndex`

### Phase 4: Serialization (Week 6)
- [ ] `serialize` (RoaringFormatSpec)
- [ ] `deserialize` + `deserializeSafe` with validation
- [ ] Frozen bitmap support (zero-copy deserialization)
- [ ] Interop tests against CRoaring test vectors

### Phase 5: SIMD Optimization (Week 7-8)
- [ ] SIMD binary search for key lookup
- [ ] SIMD bitset union/intersection/difference/xor with inline popcount
- [ ] SIMD sorted-array intersection (Schlegel et al.)
- [ ] SIMD sorted-array merge (union)
- [ ] Benchmark each SIMD path vs scalar fallback
- [ ] Tune `@Vector` widths for x86_64 and aarch64

### Phase 6: Polish & Benchmark (Week 9-10)
- [ ] Full microbenchmark suite
- [ ] Real-world dataset benchmarks
- [ ] Head-to-head CRoaring comparison tool
- [ ] Fuzz testing harness
- [ ] Documentation and examples
- [ ] `build.zig.zon` package for Zig package manager
- [ ] CI pipeline (test on x86_64 + aarch64)

---

## 10. Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| **Tagged pointers** instead of separate `typecodes[]` | One fewer array to allocate/resize/walk. Type info is colocated with the pointer it describes. CRoaring considered this (issue #5) but C makes it ugly. |
| **`@Vector` SIMD** instead of inline asm | Portable across x86_64 (AVX2/AVX-512) and aarch64 (NEON). Zig LLVM backend auto-selects best ISA. No `#ifdef` needed. |
| **User-provided allocator** | Zig idiom. Arena allocator is perfect for "build bitmap, query, discard" workloads. Also enables custom alignment, tracking, etc. |
| **No COW** | CRoaring's COW adds complexity (shared containers, refcounting, unshare-on-write). It only helps when cloning bitmaps that are rarely modified. Users can achieve this with explicit cloning. |
| **Lazy cardinality on bitsets** | Bitset cardinality requires a full popcount pass (8KB). Defer until actually needed. Mark as `-1` (unknown) after bulk mutations. |
| **64-byte alignment for bitsets** | Exactly one cache line per SIMD iteration. Eliminates split-line loads. |
| **32-byte alignment for array values** | Allows `@Vector(16, u16)` loads without crossing cache-line boundaries in the hot path. |
| **Power-of-2 array capacities** | Simplifies growth logic, enables bitwise capacity checks, and some allocators are more efficient with power-of-2 sizes. |
| **Inline popcount during SIMD ops** | CRoaring does this too: compute popcount in the same pass as the bitwise operation, avoiding a second pass over the data. |
