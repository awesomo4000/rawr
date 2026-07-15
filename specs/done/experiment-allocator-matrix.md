<!-- SPDX-License-Identifier: MPL-2.0 -->

# Experiment: Allocator Matrix Benchmark

**File:** `src/bench_allocators.zig` (new, separate from `bench_croaring.zig`)
**Build step:** `zig build bench-alloc -- --input=smp --output=arena`

## Motivation

Switching from `c_allocator` to `smp_allocator` revealed that allocator choice
affects both allocation speed AND memory layout. Arena-backed bitwiseAnd got
*slower* (0.61ms → 0.85ms) despite identical arena code, because the **input**
bitmaps had worse pointer locality when built with smp_allocator's size-class
slabs.

Two independent allocator roles:
- **Input allocator** — builds the source bitmaps. Controls memory layout
  (cache/TLB behavior during traversal).
- **Output allocator** — allocates the result bitmap. Controls allocation speed.

This experiment runs a matrix of combinations to find the optimal pairing and
identify any other surprises.

## Allocator options (CLI names)

| CLI name | Allocator | Notes |
|----------|-----------|-------|
| `c`      | `std.heap.c_allocator` | libc malloc. Sequential-ish layout. |
| `smp`    | `std.heap.smp_allocator` | Size-class freelists. Fast alloc, scattered layout. |
| `arena`  | `ArenaAllocator(page_allocator)` | Bump pointer. Sequential layout. Fast alloc. |
| `fba`    | `FixedBufferAllocator` | Pre-allocated flat buffer. Zero overhead. Theoretical floor. |

## CLI interface

```bash
# Run one combination:
zig build bench-alloc -- --input=c --output=arena

# Shortcuts for full matrix:
zig build bench-alloc -- --matrix
```

### Arguments

- `--input=NAME` — allocator for building source bitmaps (default: `smp`)
- `--output=NAME` — allocator for result bitmaps (default: `smp`)
- `--matrix` — run all 16 combinations (4×4), ignore --input/--output
- `--ops=NAMES` — comma-separated ops to run (default: `and,or,deser`)
  Available: `and`, `or`, `deser`, `all`

## Operations benchmarked

Only allocation-sensitive operations. No `contains`, `iterate`, `cardinality`,
`serialize` — those don't allocate result bitmaps.

1. **bitwiseAnd (sparse)** — intersection, small result (~15K containers)
2. **bitwiseOr (sparse)** — union, large result (~65K containers)
3. **deserialize** — output-only (input is a byte buffer, not affected by
   input allocator)

## Implementation

### Build step (build.zig)

Add alongside the existing bench step:

```zig
// Allocator matrix benchmark
const bench_alloc_mod = b.createModule(.{
    .root_source_file = b.path("src/bench_allocators.zig"),
    .target = target,
    .optimize = .ReleaseFast,
});
bench_alloc_mod.addImport("rawr", lib_mod);

const bench_alloc_exe = b.addExecutable(.{
    .name = "bench-alloc",
    .root_module = bench_alloc_mod,
});
b.installArtifact(bench_alloc_exe);

const bench_alloc_run = b.addRunArtifact(bench_alloc_exe);
if (b.args) |args| bench_alloc_run.addArgs(args);
const bench_alloc_step = b.step("bench-alloc", "Allocator matrix benchmark");
bench_alloc_step.dependOn(&bench_alloc_run.step);
```

### Main structure (src/bench_allocators.zig)

```zig
const std = @import("std");
const RoaringBitmap = @import("rawr").RoaringBitmap;

// --- Allocator registry ---

const AllocChoice = enum { c, smp, arena, fba };

fn nameToChoice(name: []const u8) ?AllocChoice {
    if (std.mem.eql(u8, name, "c")) return .c;
    if (std.mem.eql(u8, name, "smp")) return .smp;
    if (std.mem.eql(u8, name, "arena")) return .arena;
    if (std.mem.eql(u8, name, "fba")) return .fba;
    return null;
}

// --- Managed allocator context ---
// Wraps the different allocator types so we can select at runtime.

const AllocContext = struct {
    allocator: std.mem.Allocator,
    // Owned state for arena/fba — null for c/smp
    arena: ?*std.heap.ArenaAllocator = null,
    fba_buf: ?[]u8 = null,

    fn init(choice: AllocChoice) AllocContext {
        switch (choice) {
            .c => return .{ .allocator = std.heap.c_allocator },
            .smp => return .{ .allocator = std.heap.smp_allocator },
            .arena => {
                // Use a long-lived arena backed by page_allocator
                const arena = std.heap.page_allocator.create(
                    std.heap.ArenaAllocator
                ) catch @panic("OOM");
                arena.* = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                return .{
                    .allocator = arena.allocator(),
                    .arena = arena,
                };
            },
            .fba => {
                // 256MB should be plenty for sparse 1M-value bitmaps
                const buf = std.heap.page_allocator.alloc(u8, 256 * 1024 * 1024)
                    catch @panic("OOM");
                // Store the FBA on the heap too
                const fba = std.heap.page_allocator.create(
                    std.heap.FixedBufferAllocator
                ) catch @panic("OOM");
                fba.* = std.heap.FixedBufferAllocator.init(buf);
                return .{
                    .allocator = fba.allocator(),
                    .fba_buf = buf,
                };
            },
        }
    }

    /// Reset arena/fba state between benchmark iterations
    fn reset(self: *AllocContext) void {
        if (self.arena) |a| a.reset(.free_all);
        if (self.fba_buf) |_| {
            // Recreate FBA to reset the offset
            // (FixedBufferAllocator has no reset method)
            // Actually it does: fba.reset()
            // But we stored a pointer... simplest: just track the FBA ptr
        }
    }

    // Note: for the output allocator in timed benchmarks, we create a
    // FRESH arena/fba per iteration (matching how bench_croaring does it).
    // This AllocContext is mainly for the long-lived INPUT bitmaps.
};
```

### Building input bitmaps

Reuse the same data generation as `bench_croaring.zig` — 1M random u32 values,
split into two sets for sparse_a and sparse_b:

```zig
fn buildSparseBitmaps(alloc: std.mem.Allocator, values: []const u32) struct {
    a: RoaringBitmap, b: RoaringBitmap
} {
    var a = RoaringBitmap.init(alloc);
    var b = RoaringBitmap.init(alloc);
    for (values, 0..) |v, i| {
        if (i % 2 == 0) a.add(v) catch unreachable
        else b.add(v) catch unreachable;
    }
    return .{ .a = a, .b = b };
}
```

### Benchmark loop

Same timing approach as bench_croaring: 3 warmup, 21 timed, take median.
Copy the `benchmark()` function from bench_croaring.zig.

For output allocators, **create fresh per iteration:**
- `c` / `smp`: allocate, defer deinit
- `arena`: new ArenaAllocator per iteration, defer arena.deinit()
- `fba`: reset FBA offset per iteration (fba.reset())

```zig
fn benchBitwiseAnd(input_a: *const RoaringBitmap, input_b: *const RoaringBitmap,
                   out_choice: AllocChoice) void {
    switch (out_choice) {
        .c => {
            var result = input_a.bitwiseAnd(std.heap.c_allocator, input_b)
                catch unreachable;
            defer result.deinit();
            std.mem.doNotOptimizeAway(&result);
        },
        .smp => {
            var result = input_a.bitwiseAnd(std.heap.smp_allocator, input_b)
                catch unreachable;
            defer result.deinit();
            std.mem.doNotOptimizeAway(&result);
        },
        .arena => {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            var result = input_a.bitwiseAnd(arena.allocator(), input_b)
                catch unreachable;
            std.mem.doNotOptimizeAway(&result);
        },
        .fba => {
            var buf: [64 * 1024 * 1024]u8 = undefined; // 64MB stack? No.
            // Use a heap-allocated buffer, reset each iteration
            // (pre-allocate outside the timed loop, pass in)
        },
    }
}
```

**FBA detail:** The FBA buffer must be pre-allocated outside the timed loop
(allocation of the buffer itself isn't what we're measuring). Pre-allocate
a 64MB buffer at init, create a FixedBufferAllocator from it, and call
`fba.reset()` at the start of each timed iteration.

### Output format

Single combo:
```
Allocator Experiment: input=c, output=arena
============================================
3 warmup, 21 timed runs (median)

Operation                     ms       vs CRoaring
--------------------------  ------    ----------
bitwiseAnd (sparse)          0.61       0.94x
bitwiseOr (sparse)           1.34       0.61x
deserialize                  1.12       0.50x
```

Matrix mode:
```
Allocator Matrix: bitwiseAnd sparse (ms)
=========================================
              OUTPUT:  c       smp     arena    fba
INPUT: c              1.84    ?.??     0.61    ?.??
INPUT: smp            ?.??    0.93     0.85    ?.??
INPUT: arena          ?.??    ?.??     ?.??    ?.??
INPUT: fba            ?.??    ?.??     ?.??    ?.??

Allocator Matrix: bitwiseOr sparse (ms)
========================================
              OUTPUT:  c       smp     arena    fba
INPUT: c              6.98    ?.??     1.34    ?.??
INPUT: smp            ?.??    2.28     1.69    ?.??
INPUT: arena          ?.??    ?.??     ?.??    ?.??
INPUT: fba            ?.??    ?.??     ?.??    ?.??

Allocator Matrix: deserialize (ms)
===================================
              OUTPUT:  c       smp     arena    fba
(input N/A)           6.18    1.58     1.07    ?.??
```

## Predictions

```
bitwiseAnd sparse (ms):
              OUTPUT:  c       smp     arena    fba
INPUT: c              1.84    ~0.7     0.61    ~0.58    ← fba = floor
INPUT: smp            ~1.5    0.93     0.85    ~0.82
INPUT: arena          ~1.7    ~0.65    ~0.55   ~0.52    ← arena input = best layout?
INPUT: fba            ~1.7    ~0.65    ~0.55   ~0.52    ← same as arena (sequential)
```

Key predictions:
1. **fba output ≈ arena output** — arena's overhead is already near zero,
   fba just confirms it
2. **arena/fba input < c input < smp input** — sequential bump layout beats
   malloc's per-allocation headers, which beats smp's slab scattering
3. **The global optimum is arena×arena or fba×fba** at ~0.52ms
4. **deserialize: fba ≈ arena ≈ 1.07ms** — confirming allocation is fully
   eliminated and remaining time is pure memcpy/parsing

## What we learn

| Finding | Implication |
|---------|-------------|
| fba ≈ arena output | Arena overhead is negligible, no need for FBA in production |
| arena input wins | OwnedBitmap should be the default for kb (deserialize → evaluate) |
| smp input loses | Don't use smp for building long-lived bitmaps that will be traversed |
| c_alloc → smp is a sweet spot | For build-then-query patterns where arena lifetime is awkward |

## What NOT to change based on results

This is an experiment, not an optimization. Don't:
- Change bench_croaring.zig (keep it simple, smp as default)
- Add allocator selection to the rawr library API
- Optimize for FBA (it's a theoretical floor, not practical for real use)

Do:
- Document recommended allocator pairings for common patterns
- Validate that OwnedBitmap (arena-backed) is optimal for kb's use case
- Consider whether smp_allocator should remain the bench_croaring default
  or switch back to c_allocator (depends on what represents "typical" usage)
