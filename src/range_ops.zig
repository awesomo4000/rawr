// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const array_kernels = @import("array_kernels.zig");
const container_mod = @import("container.zig");
const Container = container_mod.Container;
const container_ops = @import("container_ops.zig");
const RunContainer = @import("run_container.zig").RunContainer;

pub fn removeRange(
    bitmap: anytype,
    lo: u32,
    hi: u32,
) !u64 {
    return removeRangeDirect(bitmap, lo, hi);
}

pub const RemoveRangeCopyCapacity = enum {
    normal_growth,
    exact,
};

pub fn removeRangeCopy(
    bitmap: anytype,
    allocator: std.mem.Allocator,
    lo: u32,
    hi: u32,
) !@TypeOf(bitmap.*) {
    return removeRangeCopyWithCapacity(bitmap, allocator, lo, hi, .normal_growth);
}

/// Repository tooling uses this entry point to compare capacity policies.
pub fn removeRangeCopyWithCapacity(
    bitmap: anytype,
    allocator: std.mem.Allocator,
    lo: u32,
    hi: u32,
    comptime capacity_policy: RemoveRangeCopyCapacity,
) !@TypeOf(bitmap.*) {
    if (lo > hi) return cloneWithCapacity(bitmap, allocator, capacity_policy);

    const Bitmap = @TypeOf(bitmap.*);
    var result = switch (capacity_policy) {
        .normal_growth => try Bitmap.init(allocator),
        .exact => try Bitmap.initCapacity(allocator, exactRemoveRangeCopyCapacity(bitmap, lo, hi)),
    };
    errdefer result.deinit();

    const start_key: u16 = @truncate(lo >> 16);
    const end_key: u16 = @truncate(hi >> 16);
    const start_low: u16 = @truncate(lo);
    const end_low: u16 = @truncate(hi);
    var removed: u64 = 0;

    for (bitmap.keys[0..bitmap.size], bitmap.containers[0..bitmap.size]) |key, tagged| {
        const container = Container.fromTagged(tagged);
        if (key < start_key or key > end_key) {
            try appendClone(&result, key, container);
            continue;
        }

        const low = if (key == start_key) start_low else 0;
        const high = if (key == end_key) end_low else std.math.maxInt(u16);
        removed += container_ops.containerRangeCardinality(container, low, high);

        if (low == 0 and high == std.math.maxInt(u16)) continue;

        const difference = try containerDifferenceRange(allocator, container, low, high);
        if (difference.getCardinality() == 0) {
            difference.deinit(allocator);
        } else {
            try appendOwned(&result, key, difference);
        }
    }

    result.cached_cardinality = if (bitmap.cached_cardinality >= 0)
        bitmap.cached_cardinality - @as(i64, @intCast(removed))
    else
        -1;
    return result;
}

pub fn flip(
    bitmap: anytype,
    allocator: std.mem.Allocator,
    lo: u32,
    hi: u32,
) !@TypeOf(bitmap.*) {
    return flipDirect(bitmap, allocator, lo, hi);
}

pub fn flipInPlace(
    bitmap: anytype,
    lo: u32,
    hi: u32,
) !void {
    return flipInPlaceDirect(bitmap, lo, hi);
}

fn removeRangeDirect(bitmap: anytype, lo: u32, hi: u32) !u64 {
    if (lo > hi or bitmap.size == 0) return 0;

    const start_key: u16 = @truncate(lo >> 16);
    const end_key: u16 = @truncate(hi >> 16);
    const start_low: u16 = @truncate(lo);
    const end_low: u16 = @truncate(hi);
    const original_cache = bitmap.cached_cardinality;
    const original_size: usize = bitmap.size;
    var read = array_kernels.lowerBound(bitmap.keys[0..bitmap.size], start_key);
    var write = read;
    var removed: u64 = 0;

    while (read < original_size and bitmap.keys[read] <= end_key) : (read += 1) {
        const key = bitmap.keys[read];
        const low = if (key == start_key) start_low else 0;
        const high = if (key == end_key) end_low else std.math.maxInt(u16);
        const old_tagged = bitmap.containers[read];
        const old_container = Container.fromTagged(old_tagged);
        const before = old_container.getCardinality();

        if (low == 0 and high == std.math.maxInt(u16)) {
            old_container.deinit(bitmap.allocator);
            removed += before;
            continue;
        }

        const result = containerDifferenceRange(bitmap.allocator, old_container, low, high) catch |err| {
            retainCurrentAndTail(bitmap, write, read, original_size);
            bitmap.cached_cardinality = -1;
            return err;
        };
        const after = result.getCardinality();
        const result_tagged = result.toTagged();

        old_container.deinit(bitmap.allocator);
        if (after == 0) {
            result.deinit(bitmap.allocator);
        } else {
            bitmap.keys[write] = key;
            bitmap.containers[write] = result_tagged;
            write += 1;
        }
        removed += before - after;
    }

    compactTail(bitmap, write, read, original_size);
    if (original_cache >= 0) {
        bitmap.cached_cardinality = original_cache - @as(i64, @intCast(removed));
    }
    return removed;
}

fn cloneWithCapacity(
    bitmap: anytype,
    allocator: std.mem.Allocator,
    comptime capacity_policy: RemoveRangeCopyCapacity,
) !@TypeOf(bitmap.*) {
    if (capacity_policy == .normal_growth) return bitmap.clone(allocator);

    const Bitmap = @TypeOf(bitmap.*);
    var result = try Bitmap.initCapacity(allocator, bitmap.size);
    errdefer result.deinit();
    for (bitmap.keys[0..bitmap.size], bitmap.containers[0..bitmap.size]) |key, tagged| {
        const cloned = try Container.fromTagged(tagged).clone(allocator);
        appendContainer(&result, key, cloned);
    }
    result.cached_cardinality = bitmap.cached_cardinality;
    return result;
}

fn exactRemoveRangeCopyCapacity(bitmap: anytype, lo: u32, hi: u32) u32 {
    const start_key: u16 = @truncate(lo >> 16);
    const end_key: u16 = @truncate(hi >> 16);
    const start_low: u16 = @truncate(lo);
    const end_low: u16 = @truncate(hi);
    var count: u32 = 0;

    for (bitmap.keys[0..bitmap.size], bitmap.containers[0..bitmap.size]) |key, tagged| {
        if (key < start_key or key > end_key) {
            count += 1;
            continue;
        }

        const low = if (key == start_key) start_low else 0;
        const high = if (key == end_key) end_low else std.math.maxInt(u16);
        if (survivingCardinality(Container.fromTagged(tagged), low, high) != 0) count += 1;
    }
    return count;
}

fn survivingCardinality(container: Container, low: u16, high: u16) u32 {
    var count: u32 = 0;
    if (low != 0) count += container_ops.containerRangeCardinality(container, 0, low - 1);
    if (high != std.math.maxInt(u16)) {
        count += container_ops.containerRangeCardinality(container, high + 1, std.math.maxInt(u16));
    }
    return count;
}

fn containerDifferenceRange(
    allocator: std.mem.Allocator,
    container: Container,
    low: u16,
    high: u16,
) !Container {
    var range_pair = RunContainer.RunPair{ .start = low, .length = high - low };
    var range_view = RunContainer{
        .runs = RunContainer.runsStorage(@as(*[1]RunContainer.RunPair, &range_pair)[0..]),
        .n_runs = 1,
        .capacity = 1,
        .cardinality = @intCast(@as(u32, high) - low + 1),
    };
    return container_ops.containerDifference(allocator, container, .{ .run = &range_view });
}

fn appendClone(bitmap: anytype, key: u16, container: Container) !void {
    try ensureTotalCapacity(bitmap, bitmap.size + 1);
    const cloned = try container.clone(bitmap.allocator);
    appendContainer(bitmap, key, cloned);
}

fn appendOwned(bitmap: anytype, key: u16, container: Container) !void {
    errdefer container.deinit(bitmap.allocator);
    try ensureTotalCapacity(bitmap, bitmap.size + 1);
    appendContainer(bitmap, key, container);
}

fn retainCurrentAndTail(bitmap: anytype, write: usize, read: usize, original_size: usize) void {
    bitmap.keys[write] = bitmap.keys[read];
    bitmap.containers[write] = bitmap.containers[read];
    compactTail(bitmap, write + 1, read + 1, original_size);
}

fn compactTail(bitmap: anytype, write: usize, read: usize, original_size: usize) void {
    const tail_len = original_size - read;
    if (tail_len != 0 and write != read) {
        @memmove(bitmap.keys[write .. write + tail_len], bitmap.keys[read..original_size]);
        @memmove(bitmap.containers[write .. write + tail_len], bitmap.containers[read..original_size]);
    }
    bitmap.size = @intCast(write + tail_len);
}

fn flipDirect(
    bitmap: anytype,
    allocator: std.mem.Allocator,
    lo: u32,
    hi: u32,
) !@TypeOf(bitmap.*) {
    if (lo > hi) return bitmap.clone(allocator);

    const Bitmap = @TypeOf(bitmap.*);
    const start_key: u16 = @truncate(lo >> 16);
    const end_key: u16 = @truncate(hi >> 16);
    const covered_count = @as(u32, end_key) - start_key + 1;
    const start_index = array_kernels.lowerBound(bitmap.keys[0..bitmap.size], start_key);
    const after_index = upperBound(bitmap.keys[0..bitmap.size], end_key);
    const existing_count: u32 = @intCast(after_index - start_index);
    const result_capacity = bitmap.size + covered_count - existing_count;
    var result = try Bitmap.initCapacity(allocator, result_capacity);
    errdefer result.deinit();

    var input_index: usize = 0;
    while (input_index < start_index) : (input_index += 1) {
        const cloned = try Container.fromTagged(bitmap.containers[input_index]).clone(allocator);
        appendContainer(&result, bitmap.keys[input_index], cloned);
    }

    var key_value: u32 = start_key;
    while (key_value <= end_key) : (key_value += 1) {
        const key: u16 = @intCast(key_value);
        const low = if (key == start_key) @as(u16, @truncate(lo)) else 0;
        const high = if (key == end_key) @as(u16, @truncate(hi)) else std.math.maxInt(u16);
        const flipped = if (input_index < after_index and bitmap.keys[input_index] == key) blk: {
            defer input_index += 1;
            break :blk try xorRangeContainer(
                allocator,
                Container.fromTagged(bitmap.containers[input_index]),
                low,
                high,
            );
        } else try rangeContainer(allocator, low, high);

        if (flipped.getCardinality() == 0) {
            flipped.deinit(allocator);
        } else {
            appendContainer(&result, key, flipped);
        }
    }

    while (input_index < bitmap.size) : (input_index += 1) {
        const cloned = try Container.fromTagged(bitmap.containers[input_index]).clone(allocator);
        appendContainer(&result, bitmap.keys[input_index], cloned);
    }

    result.cached_cardinality = -1;
    return result;
}

fn flipInPlaceDirect(bitmap: anytype, lo: u32, hi: u32) !void {
    if (lo > hi) return;

    const start_key: u16 = @truncate(lo >> 16);
    const end_key: u16 = @truncate(hi >> 16);
    const start_index = array_kernels.lowerBound(bitmap.keys[0..bitmap.size], start_key);
    const after_index = upperBound(bitmap.keys[0..bitmap.size], end_key);
    const covered_count = @as(u32, end_key) - start_key + 1;
    const existing_count: u32 = @intCast(after_index - start_index);
    const missing_count = covered_count - existing_count;
    const original_size: usize = bitmap.size;
    const upper_size: usize = original_size + missing_count;

    bitmap.cached_cardinality = -1;
    try ensureTotalCapacity(bitmap, @intCast(upper_size));

    var read = original_size;
    var write = upper_size;
    while (read > after_index) {
        read -= 1;
        write -= 1;
        bitmap.keys[write] = bitmap.keys[read];
        bitmap.containers[write] = bitmap.containers[read];
    }

    var key_value = @as(u32, end_key) + 1;
    while (key_value > start_key) {
        key_value -= 1;
        const key: u16 = @intCast(key_value);
        const low = if (key == start_key) @as(u16, @truncate(lo)) else 0;
        const high = if (key == end_key) @as(u16, @truncate(hi)) else std.math.maxInt(u16);

        if (read > start_index and bitmap.keys[read - 1] == key) {
            const old_index = read - 1;
            const old = Container.fromTagged(bitmap.containers[old_index]);
            const flipped = xorRangeContainer(bitmap.allocator, old, low, high) catch |err| {
                finishBackwardMutation(bitmap, read, write, upper_size);
                return err;
            };
            read -= 1;
            old.deinit(bitmap.allocator);
            if (flipped.getCardinality() == 0) {
                flipped.deinit(bitmap.allocator);
            } else {
                write -= 1;
                bitmap.keys[write] = key;
                bitmap.containers[write] = flipped.toTagged();
            }
        } else {
            const added = rangeContainer(bitmap.allocator, low, high) catch |err| {
                finishBackwardMutation(bitmap, read, write, upper_size);
                return err;
            };
            write -= 1;
            bitmap.keys[write] = key;
            bitmap.containers[write] = added.toTagged();
        }
    }

    while (read > 0) {
        read -= 1;
        write -= 1;
        bitmap.keys[write] = bitmap.keys[read];
        bitmap.containers[write] = bitmap.containers[read];
    }
    finishBackwardMutation(bitmap, 0, write, upper_size);
}

fn xorRangeContainer(
    allocator: std.mem.Allocator,
    existing: Container,
    low: u16,
    high: u16,
) !Container {
    var range_pair = RunContainer.RunPair{ .start = low, .length = high - low };
    var range_view = RunContainer{
        .runs = RunContainer.runsStorage(@as(*[1]RunContainer.RunPair, &range_pair)[0..]),
        .n_runs = 1,
        .capacity = 1,
        .cardinality = @intCast(@as(u32, high) - low + 1),
    };
    return container_ops.containerXor(allocator, existing, .{ .run = &range_view });
}

fn rangeContainer(allocator: std.mem.Allocator, low: u16, high: u16) !Container {
    const run = try RunContainer.init(allocator, 1);
    run.runs[0] = .{ .start = low, .length = high - low };
    run.n_runs = 1;
    run.cardinality = @intCast(@as(u32, high) - low + 1);
    return .{ .run = run };
}

fn appendContainer(bitmap: anytype, key: u16, container: Container) void {
    bitmap.keys[bitmap.size] = key;
    bitmap.containers[bitmap.size] = container.toTagged();
    bitmap.size += 1;
}

fn upperBound(keys: []const u16, key: u16) usize {
    if (key == std.math.maxInt(u16)) return keys.len;
    return array_kernels.lowerBound(keys, key + 1);
}

fn ensureTotalCapacity(bitmap: anytype, needed: u32) !void {
    if (needed <= bitmap.capacity) return;

    const new_capacity = @max(bitmap.capacity *| 2, needed);
    const Key = std.meta.Elem(@TypeOf(bitmap.keys));
    const Tagged = std.meta.Elem(@TypeOf(bitmap.containers));
    const new_keys = try bitmap.allocator.alloc(Key, new_capacity);
    errdefer bitmap.allocator.free(new_keys);
    const new_containers = try bitmap.allocator.alloc(Tagged, new_capacity);

    @memcpy(new_keys[0..bitmap.size], bitmap.keys[0..bitmap.size]);
    @memcpy(new_containers[0..bitmap.size], bitmap.containers[0..bitmap.size]);
    bitmap.allocator.free(bitmap.keys[0..bitmap.capacity]);
    bitmap.allocator.free(bitmap.containers[0..bitmap.capacity]);
    bitmap.keys = new_keys;
    bitmap.containers = new_containers;
    bitmap.capacity = new_capacity;
}

fn finishBackwardMutation(bitmap: anytype, prefix_len: usize, suffix_start: usize, upper_size: usize) void {
    const suffix_len = upper_size - suffix_start;
    if (suffix_len != 0 and prefix_len != suffix_start) {
        @memmove(bitmap.keys[prefix_len .. prefix_len + suffix_len], bitmap.keys[suffix_start..upper_size]);
        @memmove(
            bitmap.containers[prefix_len .. prefix_len + suffix_len],
            bitmap.containers[suffix_start..upper_size],
        );
    }
    bitmap.size = @intCast(prefix_len + suffix_len);
}
