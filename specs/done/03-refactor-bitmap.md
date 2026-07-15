<!-- SPDX-License-Identifier: MPL-2.0 -->

# Refactor: bitmap.zig (2700 → ~1500 lines)

Two commits, each independently testable.

## Step 1: Move inline imports to file top

The four `@import` declarations buried inside the struct body look block-scoped but aren't. Move them up with the rest.

Before (scattered at lines 1114, 1142, 1352, 1357):
```zig
    // ... 1100 lines of methods ...
    const opt = @import("optimize.zig");
    // ... 30 lines ...
    const compare = @import("compare.zig");
    // ... 200 lines ...
    const fmt = @import("format.zig");
    const ser = @import("serialize.zig");
```

After (all at file top):
```zig
const std = @import("std");
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const RunContainer = @import("run_container.zig").RunContainer;
const container_mod = @import("container.zig");
const Container = container_mod.Container;
const TaggedPtr = container_mod.TaggedPtr;
const ops = @import("container_ops.zig");
const compare = @import("compare.zig");
const opt = @import("optimize.zig");
const ser = @import("serialize.zig");
const fmt = @import("format.zig");
```

Zero behavior change. Five minutes.

## Step 2: Extract tests → `bitmap_tests.zig`

~1200 lines, 59 tests. Create `src/bitmap_tests.zig`:

```zig
const std = @import("std");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
const OwnedBitmap = @import("bitmap.zig").OwnedBitmap;

test "init and deinit" {
    // ...
}
// ... all tests moved here
```

Wire into the build so `zig build test` still finds them. Check how `roaring.zig` currently references test modules — follow the same pattern:

```zig
test {
    _ = @import("bitmap_tests.zig");
}
```

## Result

```
bitmap.zig         ~1500 lines  (core struct, all methods, OwnedBitmap)
bitmap_tests.zig   ~1200 lines  (all 59 tests)
```

## Checklist

- [ ] Imports moved to file top, `zig build test` passes
- [ ] Tests extracted to `bitmap_tests.zig`, `zig build test` passes
- [ ] Benchmarks unchanged
