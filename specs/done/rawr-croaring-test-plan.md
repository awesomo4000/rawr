<!-- SPDX-License-Identifier: MPL-2.0 -->

# CRoaring Interop Validation & Perf Comparison

## Prerequisites

- Refactor merged to main ✓
- Comptime cleanups applied ✓
- Straggler constants fixed ✓
- `zig build test` passes

## Setup

CRoaring ships as an amalgamated single-file C library (`roaring.h` + `roaring.c`).
Zig links C natively — no CMake, no FFI bindings.

### Get CRoaring amalgamation

```bash
git clone --depth 1 https://github.com/RoaringBitmap/CRoaring.git /tmp/CRoaring
cd /tmp/CRoaring && bash amalgamation.sh
cp roaring.h roaring.c /path/to/rawr/vendor/
```

This produces two files: `roaring.h` (~10K lines) and `roaring.c` (~21K lines).

### Build integration

Add to `build.zig` after the existing bench section:

```zig
// CRoaring validation executable
const validate_mod = b.createModule(.{
    .root_source_file = b.path("src/validate_croaring.zig"),
    .target = target,
    .optimize = .ReleaseFast,
});
validate_mod.addImport("rawr", bench_lib_mod);
validate_mod.addIncludePath(b.path("vendor/"));
validate_mod.addCSourceFile(.{
    .file = b.path("vendor/roaring.c"),
    .flags = &.{"-std=c11", "-O3", "-DNDEBUG"},
});
validate_mod.link_libc = true;

const validate_exe = b.addExecutable(.{
    .name = "validate_croaring",
    .root_module = validate_mod,
});
b.installArtifact(validate_exe);

const validate_step = b.step("validate", "Run CRoaring interop validation");
const run_validate = b.addRunArtifact(validate_exe);
validate_step.dependOn(&run_validate.step);

// CRoaring benchmark comparison
const bench_cr_mod = b.createModule(.{
    .root_source_file = b.path("src/bench_croaring.zig"),
    .target = target,
    .optimize = .ReleaseFast,
});
bench_cr_mod.addImport("rawr", bench_lib_mod);
bench_cr_mod.addIncludePath(b.path("vendor/"));
bench_cr_mod.addCSourceFile(.{
    .file = b.path("vendor/roaring.c"),
    .flags = &.{"-std=c11", "-O3", "-DNDEBUG"},
});
bench_cr_mod.link_libc = true;

const bench_cr_exe = b.addExecutable(.{
    .name = "bench_croaring",
    .root_module = bench_cr_mod,
});
b.installArtifact(bench_cr_exe);

const bench_cr_step = b.step("bench-compare", "Build CRoaring comparison benchmarks");
bench_cr_step.dependOn(&b.addInstallArtifact(bench_cr_exe, .{}).step);
```

**Note on C bindings:** CRoaring's header has heavy preprocessor usage. If
direct translation of `roaring.h` fails, write a thin `vendor/croaring_wrapper.h`
that includes only the functions you need:

```c
// croaring_wrapper.h
#include "roaring.h"
// If direct translation chokes, trim this to just the function declarations you use.
```

### File structure

```
vendor/
  roaring.h
  roaring.c
src/
  validate_croaring.zig   # Part 1: format validation
  bench_croaring.zig      # Part 2: performance comparison
```

### API notes

CRoaring functions use `char *` for buffers. In Zig, cast with `@ptrCast`.

CRoaring's `roaring_bitmap_add_range(bm, min, max)` uses **exclusive** end `[min, max)`.
rawr's `addRange(start, end)` uses **inclusive** end `[start, end]`. When creating
identical bitmaps in both, use `max = end + 1` on the CRoaring side.

---

## Part 1: Serialization Format Validation

The acid test. If rawr and CRoaring produce bit-identical bytes for the same input,
the serialization is spec-correct.

### Test matrix

Each test generates the same bitmap in both rawr and CRoaring, serializes with both,
and verifies:
1. Byte-level identity (rawr bytes == CRoaring bytes)
2. Cross-deserialize (rawr bytes → CRoaring, CRoaring bytes → rawr)
3. Content verification (cardinality + every value present)

| Test case                     | Why it matters                              |
|-------------------------------|---------------------------------------------|
| Empty bitmap                  | Minimal header (just cookie + size=0)       |
| Single element (value=0)      | Edge: low 16 bits all zero                  |
| Single element (value=MAX)    | Edge: 0xFFFFFFFF, chunk 65535               |
| 1 array container (100 vals)  | Basic no-run format                         |
| 1 bitset container (5000 vals)| Card > 4096 triggers bitset                 |
| Multiple array containers     | Tests offset header correctness             |
| Exactly 3 containers          | Below NO_OFFSET_THRESHOLD (run format only) |
| Exactly 4 containers          | At NO_OFFSET_THRESHOLD boundary             |
| 5+ containers, mixed types    | Full format exercise                        |
| Post-runOptimize              | Run containers in serialized data           |
| Dense range (0..1M)           | Large bitset containers                     |
| Sparse random (500K values)   | Many array containers across u32 space      |
| Values at chunk boundaries    | 65535, 65536, 131071, 131072                |
| All values in one chunk       | Single container, max cardinality (65536)   |
| Alternating values (0,2,4..)  | Array that doesn't compress to runs         |

### Implementation

```zig
// src/validate_croaring.zig
const std = @import("std");
const rawr = @import("rawr");
const RoaringBitmap = rawr.RoaringBitmap;
const c = @import("c");

const allocator = std.heap.c_allocator;

/// Build identical bitmaps in rawr and CRoaring from a value list.
/// Serialize both, compare bytes, cross-deserialize, verify contents.
fn validateRoundTrip(values: []const u32) !void {
    // --- Build rawr bitmap ---
    var rbm = try RoaringBitmap.init(allocator);
    defer rbm.deinit();
    for (values) |v| {
        _ = try rbm.add(v);
    }

    // --- Build CRoaring bitmap ---
    const cr = c.roaring_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(cr);
    for (values) |v| {
        c.roaring_bitmap_add(cr, v);
    }

    // --- Serialize both ---
    const rawr_bytes = try rbm.serialize(allocator);
    defer allocator.free(rawr_bytes);

    const cr_size = c.roaring_bitmap_portable_size_in_bytes(cr);
    const cr_buf = try allocator.alloc(u8, cr_size);
    defer allocator.free(cr_buf);
    _ = c.roaring_bitmap_portable_serialize(cr, @ptrCast(cr_buf.ptr));

    // --- Byte-level comparison ---
    if (!std.mem.eql(u8, rawr_bytes, cr_buf)) {
        std.debug.print("FAIL: bytes differ! rawr={d} bytes, croaring={d} bytes\n",
            .{ rawr_bytes.len, cr_buf.len });
        // Print first divergence point for debugging
        const min_len = @min(rawr_bytes.len, cr_buf.len);
        for (0..min_len) |i| {
            if (rawr_bytes[i] != cr_buf[i]) {
                std.debug.print("  First difference at byte {d}: rawr=0x{x:0>2} cr=0x{x:0>2}\n",
                    .{ i, rawr_bytes[i], cr_buf[i] });
                break;
            }
        }
        return error.ByteMismatch;
    }

    // --- Cross-deserialize: rawr bytes → CRoaring ---
    const cr2 = c.roaring_bitmap_portable_deserialize_safe(
        @ptrCast(rawr_bytes.ptr), rawr_bytes.len
    ) orelse return error.CRoaringDeserializeFailed;
    defer c.roaring_bitmap_free(cr2);

    if (c.roaring_bitmap_get_cardinality(cr2) != rbm.cardinality()) {
        return error.CardinalityMismatch;
    }
    for (values) |v| {
        if (!c.roaring_bitmap_contains(cr2, v)) return error.MissingValue;
    }

    // --- Cross-deserialize: CRoaring bytes → rawr ---
    var rbm2 = try RoaringBitmap.deserialize(allocator, cr_buf);
    defer rbm2.deinit();

    if (rbm2.cardinality() != rbm.cardinality()) return error.CardinalityMismatch;
    if (!rbm2.equals(&rbm)) return error.ContentMismatch;
}

/// Same as above but with runOptimize applied before serialization.
fn validateRunRoundTrip(values: []const u32) !void {
    var rbm = try RoaringBitmap.init(allocator);
    defer rbm.deinit();
    for (values) |v| _ = try rbm.add(v);
    _ = try rbm.runOptimize();

    const cr = c.roaring_bitmap_create() orelse return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(cr);
    for (values) |v| c.roaring_bitmap_add(cr, v);
    _ = c.roaring_bitmap_run_optimize(cr);

    // Same serialize/compare/cross-deserialize flow as validateRoundTrip...
    // (factor out common code)
}

pub fn main() !void {
    std.debug.print("CRoaring Interop Validation\n", .{});
    std.debug.print("===========================\n\n", .{});

    // Empty
    try runTest("empty", &.{});

    // Single elements
    try runTest("single_zero", &.{0});
    try runTest("single_max", &.{0xFFFFFFFF});

    // Array container
    var arr100: [100]u32 = undefined;
    for (0..100) |i| arr100[i] = @intCast(i * 10);
    try runTest("array_100", &arr100);

    // Bitset container (5000 values in one chunk)
    var bitset5k: [5000]u32 = undefined;
    for (0..5000) |i| bitset5k[i] = @intCast(i);
    try runTest("bitset_5000", &bitset5k);

    // Multiple containers, chunk boundaries, etc.
    try runTest("chunk_boundaries", &.{ 65535, 65536, 131071, 131072 });

    // ... etc for each row in the test matrix.
    // Use a PRNG with fixed seed for random test cases.

    // Run format tests
    // ... same cases but with runOptimize

    std.debug.print("\nAll validation tests passed.\n", .{});
}

fn runTest(name: []const u8, values: []const u32) !void {
    validateRoundTrip(values) catch |err| {
        std.debug.print("FAIL: {s} ({s})\n", .{ name, @errorName(err) });
        return err;
    };
    std.debug.print("  PASS: {s} ({d} values)\n", .{ name, values.len });
}
```

---

## Part 2: Performance Comparison

Side-by-side benchmarks. Same data, same operations, same allocator (`c_allocator`),
same binary.

### Operations to benchmark

| Operation              | rawr call                          | CRoaring call                                 |
|------------------------|------------------------------------|-----------------------------------------------|
| add (random 1M)       | `bm.add(v)`                        | `roaring_bitmap_add(bm, v)`                  |
| add (sequential 1M)   | `bm.add(v)`                        | `roaring_bitmap_add(bm, v)`                  |
| addRange               | `bm.addRange(0, N-1)`             | `roaring_bitmap_add_range(bm, 0, N)`         |
| contains (hit)         | `bm.contains(v)`                  | `roaring_bitmap_contains(bm, v)`             |
| contains (miss)        | `bm.contains(v \| 0x80000000)`     | `roaring_bitmap_contains(bm, v \| 0x80000000)` |
| bitwiseAnd (sparse)    | `bm.bitwiseAnd(alloc, &other)`    | `roaring_bitmap_and(a, b)`                   |
| bitwiseAnd (dense)     | `bm.bitwiseAnd(alloc, &other)`    | `roaring_bitmap_and(a, b)`                   |
| bitwiseOr (sparse)     | `bm.bitwiseOr(alloc, &other)`     | `roaring_bitmap_or(a, b)`                    |
| bitwiseOr (dense)      | `bm.bitwiseOr(alloc, &other)`     | `roaring_bitmap_or(a, b)`                    |
| bitwiseAndInPlace      | `bm.bitwiseAndInPlace(&other)`    | `roaring_bitmap_and_inplace(a, b)`           |
| bitwiseOrInPlace       | `bm.bitwiseOrInPlace(&other)`     | `roaring_bitmap_or_inplace(a, b)`            |
| iterate                | `bm.iterator()` loop + sum        | `roaring_iterate(bm, callback, ctx)`         |
| serialize              | `bm.serialize(alloc)`             | `roaring_bitmap_portable_serialize(bm, buf)` |
| deserialize            | `RoaringBitmap.deserialize(data)` | `roaring_bitmap_portable_deserialize_safe`    |
| cardinality            | `bm.cardinality()`                | `roaring_bitmap_get_cardinality(bm)`         |
| runOptimize            | `bm.runOptimize()`                | `roaring_bitmap_run_optimize(bm)`            |

For in-place operations on CRoaring, clone first with `roaring_bitmap_copy(bm)` so
each iteration starts from the same state.

### Datasets

Same as `bench.zig`, built once, shared by both:

1. **Sparse random**: 500K values uniform across u32 (~65K containers, mostly array)
2. **Dense range**: 500K consecutive values (8 bitset containers)
3. **Clustered**: 1M values in variable-size clusters
4. **Mixed post-optimize**: clustered data after runOptimize (run containers)

Use a fixed PRNG seed (same as bench.zig: `12345` / `54321`) so data is reproducible.

### Measurement

- **21 timed runs**, 3 warmup, report **median**.
- Report P25 and P75 alongside median so noisy benchmarks are visible.
- Use `c_allocator` for both sides. CRoaring uses `malloc` internally.
- Use `std.mem.doNotOptimizeAway` on results to prevent dead code elimination.
- For CRoaring iteration, use a callback that sums values into a `volatile` counter.

For best results on **macOS M4**:
- Close background apps (Safari, Spotlight, Time Machine).
- Run with `sudo nice -n -20 ./zig-out/bin/bench_croaring`.
- Optionally add P-core pinning via QoS at top of main:

```zig
if (comptime @import("builtin").os.tag == .macos) {
    // Request user-interactive QoS to pin to performance cores.
    // std.os.darwin.pthread_set_qos_class_self_np may need extern decl.
}
```

### Output format

```
Operation                          rawr (ms)     CRoaring (ms)   ratio
─────────────────────────────────  ──────────    ──────────────   ─────
add (random 1M)                       XX.XX           XX.XX       X.Xx
bitwiseAnd (sparse 500K x 500K)      XX.XX           XX.XX       X.Xx
...
```

Ratio = rawr / CRoaring. Values < 1.0 mean rawr is faster.

### What to expect on M4

We inspected CRoaring's actual source code. On aarch64:

**Bitset-on-bitset ops (AND/OR/XOR/ANDNOT):** Both use NEON. CRoaring has
hand-written NEON intrinsics (`vandq_u64`, etc.), rawr uses `@Vector(8, u64)`
through LLVM which emits the same instructions. **Expect parity.**

**Array-on-array intersection (similar sizes):** Both do scalar merge walks. CRoaring
uses a slightly different branch structure (goto-based) but fundamentally the same
algorithm. **Expect parity**, with rawr potentially slightly ahead due to cross-module
inlining.

**Array-on-array intersection (skewed sizes):** CRoaring switches to galloping
search (`intersect_skewed_uint16`) when one array is 64x larger. rawr currently
does linear merge regardless. **Expect CRoaring wins on skewed cases.** This is
a known gap with a planned fix (see optimization-galloping-intersect.md).

**contains:** Identical algorithm (binary search on array, bit lookup on bitset,
run search on run). **Expect parity.**

**iterate:** Both do `@ctz` / `__builtin_ctzll` loops on bitset words. **Expect
parity.**

**serialize/deserialize:** Same format, similar code paths. **Expect parity.**

**add (random):** Container lookup + sorted insert / bit set. rawr may be slightly
slower due to binary search + memmove on array containers vs CRoaring's potentially
tighter inner loop. **Expect within 1.5x.**

**Any operation >2x slower than CRoaring** is worth investigating — it likely
indicates a code-level issue, not an architectural limitation.

**Any operation >5x slower** almost certainly means a bug or missing optimization.

---

## Part 3: FrozenBitmap

CRoaring's frozen format (`roaring_bitmap_frozen_serialize` / `roaring_bitmap_frozen_view`)
is a **different layout** from rawr's FrozenBitmap. CRoaring's frozen format is their own
internal representation optimized for mmap. rawr's FrozenBitmap reads the **portable**
serialization format (same bytes as `roaring_bitmap_portable_serialize`).

These are **not interoperable** and that's fine — the frozen/zero-copy format is not part
of the RoaringFormatSpec. Portable format interop (Part 1) is what matters.

**Skip frozen cross-validation.** Document in a comment that the formats differ.

---

## Running

```bash
# Generate amalgamation (one time)
git clone --depth 1 https://github.com/RoaringBitmap/CRoaring.git /tmp/CRoaring
cd /tmp/CRoaring && bash amalgamation.sh
mkdir -p vendor && cp roaring.h roaring.c vendor/

# Format validation
zig build validate
# or: zig build && ./zig-out/bin/validate_croaring

# Performance comparison
zig build bench-compare
sudo nice -n -20 ./zig-out/bin/bench_croaring
```

---

## Success criteria

1. **Byte-identical serialization** for every test case in Part 1 (both directions,
   both with and without runOptimize).
2. **No operation >2x slower** than CRoaring on M4 without a documented reason.
3. **Known gap documented:** array-on-array intersection with skewed sizes will be
   slower until galloping search is implemented.
