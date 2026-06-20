# rawr bitmap.zig Refactor Plan

**Based on:** commit `4a4ea8a` (HEAD of `main`)
**Goal:** Split bitmap.zig (3341 lines) into focused modules, each under 1000 lines.

## Current Layout

```
src/bitmap.zig          3341 lines
├── RoaringBitmap struct (lines 14-1834)
│   ├── Core: init/deinit/clone/add/remove/contains/addRange/fromSorted/min/max  (14-520, ~507 lines)
│   ├── Set Operations: bitwiseOr/And/Difference/Xor                             (521-708, ~188 lines)
│   ├── In-Place Set Operations: bitwiseOr/And/DifferenceInPlace                 (709-936, ~228 lines)
│   ├── Optimization: runOptimize, countRuns*, array/bitsetToRun                 (937-1128, ~192 lines)
│   ├── Comparison: isSubsetOf, equals, 18 cross-type helpers                    (1129-1374, ~246 lines)
│   ├── Helpers: appendContainer, cloneContainer                                 (1375-1412, ~38 lines)
│   ├── Iterator: Iterator struct + iterator()                                   (1413-1566, ~154 lines)
│   └── Serialization: serialize/deserialize + format constants                  (1567-1834, ~268 lines)
├── FrozenBitmap struct (lines 1842-2218, ~377 lines)
└── Tests (lines 2220-3341, ~1122 lines)
    ├── Core tests                        (2222-2370, ~149 lines)
    ├── Set operation tests               (2372-2600, ~229 lines)
    ├── Iterator tests                    (2602-2678, ~77 lines)
    ├── In-place operation tests          (2680-2821, ~142 lines)
    ├── Serialization tests               (2823-2918, ~96 lines)
    ├── addRange/fromSorted tests         (2920-3040, ~121 lines)
    ├── runOptimize tests                 (3042-3170, ~129 lines)
    └── FrozenBitmap tests                (3171-3341, ~171 lines)

src/container_ops.zig   1052 lines  (cross-container set ops + containerUnionInPlace)
src/container.zig        251 lines  (TaggedPtr + Container union)
src/array_container.zig  393 lines  (includes unionInPlace)
src/bitset_container.zig 419 lines
src/run_container.zig    416 lines
src/property_tests.zig   365 lines
src/bench.zig            487 lines
src/roaring.zig           23 lines  (module root, re-exports + test block)
build.zig                 62 lines
```

## Strategy

Thin method wrappers (Option A) for methods that stay on RoaringBitmap's API.
Standalone export for FrozenBitmap. Each extraction is one commit. Run `zig build test`
after each.

## Extraction Order

### Step 1: `format.zig` — Shared constants (prerequisite, ~10 lines)

Three format constants currently live inside the `RoaringBitmap` struct (line 1571-1573).
Both serialization and FrozenBitmap need them. Extract to a tiny shared file first.

**Create `src/format.zig`:**
```zig
/// RoaringFormatSpec serialization constants.
pub const SERIAL_COOKIE_NO_RUNCONTAINER: u32 = 12346;
pub const SERIAL_COOKIE: u32 = 12347;
pub const NO_OFFSET_THRESHOLD: u32 = 4;
```

**In bitmap.zig, replace the three `const` lines (1571-1573) with:**
```zig
const fmt = @import("format.zig");
pub const SERIAL_COOKIE = fmt.SERIAL_COOKIE;
pub const SERIAL_COOKIE_NO_RUNCONTAINER = fmt.SERIAL_COOKIE_NO_RUNCONTAINER;
pub const NO_OFFSET_THRESHOLD = fmt.NO_OFFSET_THRESHOLD;
```

All existing references (`SERIAL_COOKIE`, `RoaringBitmap.SERIAL_COOKIE`) continue
to work unchanged.

**Risk: None.** Three constants moved. Enables steps 2 and 4.

---

### Step 2: `frozen.zig` — FrozenBitmap (~548 lines)

FrozenBitmap is already a standalone `pub const` struct. Zero access to RoaringBitmap
internals. It only references format constants and `ArrayContainer.MAX_CARDINALITY`.

**Move to `src/frozen.zig`:**
- `FrozenBitmap` struct (lines 1842-2218)
- FrozenBitmap tests (lines 3171-3341)

**Imports needed in frozen.zig:**
```zig
const std = @import("std");
const fmt = @import("format.zig");
const ArrayContainer = @import("array_container.zig").ArrayContainer;
```

Replace `RoaringBitmap.SERIAL_COOKIE` → `fmt.SERIAL_COOKIE` (3 occurrences),
`RoaringBitmap.NO_OFFSET_THRESHOLD` → `fmt.NO_OFFSET_THRESHOLD` (1 occurrence).

**In roaring.zig:**
```zig
pub const FrozenBitmap = @import("frozen.zig").FrozenBitmap;
// In test block:
_ = @import("frozen.zig");
```

**Remove from bitmap.zig:** The `FrozenBitmap` struct and its tests. Also remove
the now-unused `pub const FrozenBitmap` re-export if bitmap.zig had one.

**Lines removed from bitmap.zig:** ~548
**Risk: None.** Pure cut-paste. No wrappers needed.

---

### Step 3: `compare.zig` — Subset/Equality (~296 lines)

Pure read-only functions. `isSubsetOf`, `equals`, and 18 cross-container helpers.
Only call `container.contains()` and iterate values/runs/words.

**Move to `src/compare.zig` (lines 1129-1374 from bitmap.zig):**
- `isSubsetOf`, `equals`
- `containerIsSubset`, `containerEquals` dispatch functions
- 9 `*Subset*` helpers: `arraySubsetArray`, `arraySubsetBitset`, `arraySubsetRun`,
  `bitsetSubsetBitset`, `bitsetSubsetArray`, `bitsetSubsetRun`,
  `runSubsetArray`, `runSubsetBitset`, `runSubsetRun`
- 4 `*Equals*` helpers: `arrayEqualsBitset`, `arrayEqualsRun`, `bitsetEqualsRun`,
  `runEqualsRun`

**Imports needed in compare.zig:**
```zig
const std = @import("std");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
const Container = @import("container.zig").Container;
const TaggedPtr = @import("container.zig").TaggedPtr;
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const RunContainer = @import("run_container.zig").RunContainer;
```

**Note:** This creates a circular import (compare.zig imports bitmap.zig, bitmap.zig
imports compare.zig). Zig handles this fine — no comptime dependency cycle exists.
The functions only need the struct layout (field offsets), not any comptime evaluation.

**In bitmap.zig, replace the comparison section with thin wrappers:**
```zig
const compare = @import("compare.zig");

pub fn isSubsetOf(self: *const Self, other: *const Self) bool {
    return compare.isSubsetOf(self, other);
}
pub fn equals(self: *const Self, other: *const Self) bool {
    return compare.equals(self, other);
}
```

**Tests:** The set operation tests (2372-2600) exercise subset/equality mixed with
other operations. Leave those in bitmap.zig as integration tests. Add focused unit
tests in compare.zig if desired.

**In roaring.zig test block:**
```zig
_ = @import("compare.zig");
```

**Lines removed from bitmap.zig:** ~246 (code only, tests stay)
**Risk: Low.** Two one-liner wrappers. Test the circular import first.

---

### Step 4: `serialize.zig` — Serialization (~364 lines)

`hasRunContainers`, `serializedSizeInBytes`, `serialize`, `serializeToWriter`,
`deserialize`, `deserializeFromReader`.

**Move to `src/serialize.zig` (lines 1567-1834 from bitmap.zig):**
- All 6 serialization functions
- Serialization tests (lines 2823-2918)

**Note:** `deserialize`/`deserializeFromReader` construct a RoaringBitmap from scratch
by calling `init`, `ensureCapacity`, and writing to `keys[]`/`containers[]`. These are
all `pub` fields/methods, so the circular import works.

**Imports needed in serialize.zig:**
```zig
const std = @import("std");
const fmt = @import("format.zig");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
const Container = @import("container.zig").Container;
const TaggedPtr = @import("container.zig").TaggedPtr;
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const RunContainer = @import("run_container.zig").RunContainer;
```

**In bitmap.zig, replace serialization section with thin wrappers:**
```zig
const ser = @import("serialize.zig");

pub fn serializedSizeInBytes(self: *const Self) usize {
    return ser.serializedSizeInBytes(self);
}
pub fn serialize(self: *const Self, allocator: std.mem.Allocator) ![]u8 {
    return ser.serialize(self, allocator);
}
pub fn serializeToWriter(self: *const Self, writer: anytype) !void {
    return ser.serializeToWriter(self, writer);
}
pub fn deserialize(allocator: std.mem.Allocator, data: []const u8) !Self {
    return ser.deserialize(allocator, data);
}
pub fn deserializeFromReader(allocator: std.mem.Allocator, reader: anytype, data_len: usize) !Self {
    return ser.deserializeFromReader(allocator, reader, data_len);
}
```

**Lines removed from bitmap.zig:** ~364
**Risk: Low.** Five wrappers. `serializeToWriter` uses `anytype` for the writer —
the wrapper just forwards it, Zig monomorphizes at the call site.

---

### Step 5: `optimize.zig` — runOptimize (~321 lines)

`runOptimize` and its helpers: `countRunsInArray`, `countRunsInBitset`,
`arrayToRunContainer`, `bitsetToRunContainer`.

**Move to `src/optimize.zig` (lines 937-1128 from bitmap.zig):**
- All 5 functions
- runOptimize tests (lines 3042-3170)

**Imports needed in optimize.zig:**
```zig
const std = @import("std");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
const Container = @import("container.zig").Container;
const TaggedPtr = @import("container.zig").TaggedPtr;
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const RunContainer = @import("run_container.zig").RunContainer;
```

**In bitmap.zig, thin wrapper:**
```zig
const opt = @import("optimize.zig");

pub fn runOptimize(self: *Self) !u32 {
    return opt.runOptimize(self);
}
```

**Lines removed from bitmap.zig:** ~321
**Risk: Minimal.** One wrapper. Only accesses pub fields.

---

## What Stays in bitmap.zig

These are tightly coupled to internal mutation patterns and share helpers:

- **Core** — init/deinit/clone/add/remove/contains/addRange/fromSorted/min/max
- **Set Operations** — bitwiseOr/And/Diff/Xor (scratch arena, merge walks)
- **In-Place Set Operations** — mutation, owned tracking, errdefer, containerUnionInPlace
- **Iterator** — directly indexes keys[]/containers[]
- **Helpers** — appendContainer, cloneContainer (used by set ops + core)
- **Core/set op/iterator/in-place/addRange tests** — integration tests touching multiple subsystems

---

## Result

```
bitmap.zig          ~1462 lines  (507 core + 188 set ops + 228 in-place + 154 iterator
                                  + 38 helpers + ~6 wrapper stubs + ~341 remaining tests)

frozen.zig           ~548 lines  FrozenBitmap struct + tests
serialize.zig        ~364 lines  serialize/deserialize + tests
optimize.zig         ~321 lines  runOptimize + conversion helpers + tests
compare.zig          ~296 lines  subset/equality + 18 cross-type helpers + tests
format.zig            ~10 lines  shared format constants

container_ops.zig   1052 lines   (unchanged)
container.zig        251 lines   (unchanged)
array_container.zig  393 lines   (unchanged)
bitset_container.zig 419 lines   (unchanged)
run_container.zig    416 lines   (unchanged)
property_tests.zig   365 lines   (unchanged)
bench.zig            487 lines   (unchanged)
roaring.zig           ~30 lines  (updated re-exports + test imports)
build.zig             62 lines   (unchanged)
```

**bitmap.zig: 3341 → ~1462 lines.** Every file under 1100 lines.

## Commit Sequence

```
git checkout -b refactor/split-bitmap

# Step 1
git commit -m "refactor: extract format constants to format.zig"

# Step 2
git commit -m "refactor: extract FrozenBitmap to frozen.zig"

# Step 3
git commit -m "refactor: extract subset/equality to compare.zig"

# Step 4
git commit -m "refactor: extract serialization to serialize.zig"

# Step 5
git commit -m "refactor: extract runOptimize to optimize.zig"

git checkout main
git merge refactor/split-bitmap
```

Run `zig build test` between **every** commit. If tests pass, the extraction is correct.
