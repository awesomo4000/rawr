# Cache Cardinality at Bitmap Level

## Problem

`cardinality()` loops over all containers every call:

```zig
pub fn cardinality(self: *const Self) u64 {
    var total: u64 = 0;
    for (self.containers[0..self.size]) |tp| {
        total += Container.fromTagged(tp).getCardinality();
    }
    return total;
}
```

For 65K containers that's 65K tagged pointer unpacks + pointer chases.
Currently 1.42x slower than CRoaring (0.06ms vs 0.04ms). We can make
it O(1).

## Fix

Add a cached cardinality field to RoaringBitmap. Return it directly when
valid. Invalidate it when the bitmap is mutated.

### Step 1: Add field (src/bitmap.zig, RoaringBitmap struct)

```zig
cached_cardinality: i64 = -1, // -1 = unknown
```

### Step 2: Update cardinality() to use cache

```zig
pub fn cardinality(self: *Self) u64 {
    if (self.cached_cardinality >= 0) return @intCast(self.cached_cardinality);
    var total: u64 = 0;
    for (self.containers[0..self.size]) |tp| {
        total += Container.fromTagged(tp).getCardinality();
    }
    self.cached_cardinality = @intCast(total);
    return total;
}
```

Note: signature changes from `*const Self` to `*Self` because we write
the cache. If that causes issues at call sites that have `*const`, add a
separate `computeCardinality` that's `*Self` and have the `*const` version
loop without caching (same as today). But prefer `*Self` if possible.

### Step 3: Keep cache valid on mutations that know the delta

These functions already know exactly how the cardinality changed:

**add** — returns bool (true = new value). Currently ~line 354:
```zig
// After successful add:
if (self.cached_cardinality >= 0) self.cached_cardinality += 1;
return true;
// On duplicate (not added):
return false;
```

**remove** — returns bool (true = was present). Same pattern:
```zig
if (self.cached_cardinality >= 0) self.cached_cardinality -= 1;
return true;
```

**addRange** — returns count added:
```zig
// At end of addRange, you already have the count:
if (self.cached_cardinality >= 0) self.cached_cardinality += @intCast(added);
return added;
```

### Step 4: Invalidate cache on mutations that don't know the delta

These functions modify containers in complex ways. Just set to -1:

```zig
fn invalidateCardinality(self: *Self) void {
    self.cached_cardinality = -1;
}
```

Call `self.invalidateCardinality()` at the top of:
- `bitwiseOrInPlace`
- `bitwiseAndInPlace`
- `bitwiseDifferenceInPlace`
- `bitwiseXorInPlace`
- `runOptimize` (doesn't change cardinality, but safer to invalidate)

### Step 5: init and deserialize

**init** — cardinality is 0:
```zig
pub fn init(allocator: std.mem.Allocator) !Self {
    // ...
    return .{
        // ... existing fields ...
        .cached_cardinality = 0,
    };
}
```

**deserialize** — we know the cardinality from the header. After the
container read loop, the cardinalities array has all the values:
```zig
// At end of deserializeFromReader, before return:
var total: u64 = 0;
for (cardinalities[0..size]) |c| total += c;
result.cached_cardinality = @intCast(total);
```

This is free — we already have the cardinalities array from parsing the
header. No extra work.

### Step 6: clone and fromSorted

**clone** — copy the cache:
```zig
pub fn clone(self: *const Self, allocator: std.mem.Allocator) !Self {
    // ... existing clone logic ...
    result.cached_cardinality = self.cached_cardinality;
    return result;
}
```

**fromSorted** — cardinality equals input length:
```zig
result.cached_cardinality = @intCast(values.len);
```

## Expected result

`cardinality()` becomes a field read. Should be effectively 0.00ms in the
benchmark, well under CRoaring's 0.04ms.

## Testing

All existing tests pass — cardinality behavior is identical, just cached.
Add one test that verifies cache correctness through mutations:

```zig
test "cached cardinality stays correct through mutations" {
    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    try std.testing.expectEqual(@as(u64, 0), bm.cardinality());

    _ = try bm.add(1);
    try std.testing.expectEqual(@as(u64, 1), bm.cardinality());

    _ = try bm.add(1); // duplicate
    try std.testing.expectEqual(@as(u64, 1), bm.cardinality());

    _ = try bm.addRange(100, 199);
    try std.testing.expectEqual(@as(u64, 101), bm.cardinality());

    _ = try bm.remove(1);
    try std.testing.expectEqual(@as(u64, 100), bm.cardinality());

    // In-place op invalidates, next call recomputes
    var other = try RoaringBitmap.init(allocator);
    defer other.deinit();
    _ = try other.addRange(150, 250);
    try bm.bitwiseOrInPlace(&other);
    try std.testing.expectEqual(@as(u64, 151), bm.cardinality());
}
```
