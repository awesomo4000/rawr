// SPDX-License-Identifier: MPL-2.0

//! Fresh-process clone/removeRange attribution diagnostic.

const std = @import("std");
const c = @import("c");
const rawr = @import("rawr");
const bench_time = @import("bench_time.zig");
const CountingAllocator = @import("counting_allocator.zig").CountingAllocator;

const RoaringBitmap = rawr.RoaringBitmap;
const warmup_runs = 3;
const timed_runs = 21;
const batch_count = 8192;
const range_lo: u32 = 100_000;
const range_hi: u32 = 650_000;

const Condition = enum {
    timer_control,
    clone_body,
    remove_body,
    clone_remove_body,

    fn id(self: Condition) []const u8 {
        return switch (self) {
            .timer_control => "timer-control",
            .clone_body => "clone-body",
            .remove_body => "remove-body",
            .clone_remove_body => "clone-remove-body",
        };
    }
};

const Implementation = enum { rawr, croaring };
const AllocatorKind = enum { smp, libc };

var rawr_source: ?RoaringBitmap = null;
var cr_source: ?*c.roaring_bitmap_t = null;

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var header = false;
    var inventory = false;
    var condition: ?Condition = null;
    var implementation: ?Implementation = null;
    var allocator_kind: ?AllocatorKind = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.eql(u8, arg, "--inventory")) {
            inventory = true;
        } else if (std.mem.startsWith(u8, arg, "--condition=")) {
            condition = parseCondition(arg[12..]) orelse return error.UnknownCondition;
        } else if (std.mem.startsWith(u8, arg, "--implementation=")) {
            implementation = std.meta.stringToEnum(Implementation, arg[17..]) orelse
                return error.UnknownImplementation;
        } else if (std.mem.startsWith(u8, arg, "--allocator=")) {
            allocator_kind = std.meta.stringToEnum(AllocatorKind, arg[12..]) orelse
                return error.UnknownAllocator;
        } else {
            return error.UnknownArgument;
        }
    }

    if (header) {
        if (inventory or condition != null or implementation != null or allocator_kind != null) {
            return error.ConflictingArguments;
        }
        printHeader();
        return;
    }

    const selected_implementation = implementation orelse return error.MissingImplementation;
    const selected_allocator = allocator_kind orelse return error.MissingAllocator;
    if (selected_implementation == .croaring and selected_allocator != .libc) {
        return error.UnsupportedAllocator;
    }

    if (inventory) {
        if (condition != null) return error.ConflictingArguments;
        try printInventory(selected_implementation, selected_allocator);
        return;
    }

    const selected_condition = condition orelse return error.MissingCondition;
    prepare(selected_implementation);
    defer cleanup();

    const median_ns = measure(selected_condition, selected_implementation, selected_allocator);
    try validateCondition(selected_condition);
    bench_time.print("RESULT\t{s}\t{s}\t{s}\tns/op\t{d}\t{d}\n", .{
        selected_condition.id(),
        @tagName(selected_implementation),
        @tagName(selected_allocator),
        batch_count,
        median_ns,
    });
}

fn parseCondition(value: []const u8) ?Condition {
    inline for (std.meta.fields(Condition)) |field| {
        const condition: Condition = @enumFromInt(field.value);
        if (std.mem.eql(u8, value, condition.id())) return condition;
    }
    return null;
}

fn printHeader() void {
    bench_time.printBenchEnvironment();
    bench_time.print("# requested-cpu: native\n", .{});
    bench_time.print("# protocol: {d}w/{d}t median, batch={d}\n", .{
        warmup_runs,
        timed_runs,
        batch_count,
    });
    bench_time.print("# croaring-avx512: {s}\n", .{
        if (c.CROARING_COMPILER_SUPPORTS_AVX512 != 0) "on" else "off",
    });
}

fn prepare(implementation: Implementation) void {
    switch (implementation) {
        .rawr => ensureRawrSource(),
        .croaring => ensureCRoaringSource(),
    }
}

fn ensureRawrSource() void {
    if (rawr_source != null) return;
    var bitmap = RoaringBitmap.init(std.heap.smp_allocator) catch unreachable;
    _ = bitmap.addRange(0, 499_999) catch unreachable;
    rawr_source = bitmap;
}

fn ensureCRoaringSource() void {
    if (cr_source != null) return;
    const bitmap = c.roaring_bitmap_create() orelse unreachable;
    c.roaring_bitmap_add_range(bitmap, 0, 500_000);
    cr_source = bitmap;
}

fn cleanup() void {
    if (rawr_source) |*bitmap| bitmap.deinit();
    rawr_source = null;
    if (cr_source) |bitmap| c.roaring_bitmap_free(bitmap);
    cr_source = null;
}

fn allocator(kind: AllocatorKind) std.mem.Allocator {
    return switch (kind) {
        .smp => std.heap.smp_allocator,
        .libc => bench_time.cAllocator(),
    };
}

fn measure(condition: Condition, implementation: Implementation, allocator_kind: AllocatorKind) u64 {
    for (0..warmup_runs) |_| _ = runBatch(condition, implementation, allocator_kind);

    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| elapsed.* = runBatch(condition, implementation, allocator_kind);
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

fn runBatch(condition: Condition, implementation: Implementation, allocator_kind: AllocatorKind) u64 {
    var elapsed: u64 = 0;
    for (0..batch_count) |_| {
        elapsed +%= switch (implementation) {
            .rawr => timeRawr(condition, allocator(allocator_kind)),
            .croaring => timeCRoaring(condition),
        };
    }
    std.mem.doNotOptimizeAway(elapsed);
    return elapsed;
}

noinline fn timeRawr(condition: Condition, result_allocator: std.mem.Allocator) u64 {
    const source = &rawr_source.?;
    return switch (condition) {
        .timer_control => result: {
            const start = bench_time.monotonicNanos();
            std.mem.doNotOptimizeAway(start);
            break :result bench_time.monotonicNanos() - start;
        },
        .clone_body => result: {
            const start = bench_time.monotonicNanos();
            var clone = source.clone(result_allocator) catch unreachable;
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(&clone);
            clone.deinit();
            break :result elapsed;
        },
        .remove_body => result: {
            var clone = source.clone(result_allocator) catch unreachable;
            const start = bench_time.monotonicNanos();
            const removed = clone.removeRange(range_lo, range_hi) catch unreachable;
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(removed);
            std.mem.doNotOptimizeAway(&clone);
            clone.deinit();
            break :result elapsed;
        },
        .clone_remove_body => result: {
            const start = bench_time.monotonicNanos();
            var clone = source.clone(result_allocator) catch unreachable;
            const removed = clone.removeRange(range_lo, range_hi) catch unreachable;
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(removed);
            std.mem.doNotOptimizeAway(&clone);
            clone.deinit();
            break :result elapsed;
        },
    };
}

noinline fn timeCRoaring(condition: Condition) u64 {
    const source = cr_source.?;
    return switch (condition) {
        .timer_control => result: {
            const start = bench_time.monotonicNanos();
            std.mem.doNotOptimizeAway(start);
            break :result bench_time.monotonicNanos() - start;
        },
        .clone_body => result: {
            const start = bench_time.monotonicNanos();
            const clone = c.roaring_bitmap_copy(source) orelse unreachable;
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(clone);
            c.roaring_bitmap_free(clone);
            break :result elapsed;
        },
        .remove_body => result: {
            const clone = c.roaring_bitmap_copy(source) orelse unreachable;
            const start = bench_time.monotonicNanos();
            c.roaring_bitmap_remove_range_closed(clone, range_lo, range_hi);
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(clone);
            c.roaring_bitmap_free(clone);
            break :result elapsed;
        },
        .clone_remove_body => result: {
            const start = bench_time.monotonicNanos();
            const clone = c.roaring_bitmap_copy(source) orelse unreachable;
            c.roaring_bitmap_remove_range_closed(clone, range_lo, range_hi);
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(clone);
            c.roaring_bitmap_free(clone);
            break :result elapsed;
        },
    };
}

fn validateCondition(condition: Condition) !void {
    if (condition == .timer_control) return;
    ensureRawrSource();
    ensureCRoaringSource();

    var rawr_result = try rawr_source.?.clone(std.heap.smp_allocator);
    defer rawr_result.deinit();
    const cr_result = c.roaring_bitmap_copy(cr_source.?) orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(cr_result);

    if (condition != .clone_body) {
        _ = try rawr_result.removeRange(range_lo, range_hi);
        c.roaring_bitmap_remove_range_closed(cr_result, range_lo, range_hi);
    }
    try expectPortableEqual(&rawr_result, cr_result);
    if (condition == .clone_body) try expectSourceIdentical(&rawr_source.?, &rawr_result);
}

fn expectSourceIdentical(source: *const RoaringBitmap, clone: *const RoaringBitmap) !void {
    const source_bytes = try source.serialize(std.heap.page_allocator);
    defer std.heap.page_allocator.free(source_bytes);
    const clone_bytes = try clone.serialize(std.heap.page_allocator);
    defer std.heap.page_allocator.free(clone_bytes);
    if (!std.mem.eql(u8, source_bytes, clone_bytes)) return error.CloneMismatch;
}

fn expectPortableEqual(rawr_result: *const RoaringBitmap, cr_result: *const c.roaring_bitmap_t) !void {
    const rawr_bytes = try rawr_result.serialize(std.heap.page_allocator);
    defer std.heap.page_allocator.free(rawr_bytes);
    const cr_len = c.roaring_bitmap_portable_size_in_bytes(cr_result);
    if (rawr_bytes.len != cr_len) return error.SerializedSizeMismatch;
    const cr_bytes = try std.heap.page_allocator.alloc(u8, cr_len);
    defer std.heap.page_allocator.free(cr_bytes);
    if (c.roaring_bitmap_portable_serialize(cr_result, @ptrCast(cr_bytes.ptr)) != cr_len) {
        return error.SerializedSizeMismatch;
    }
    if (!std.mem.eql(u8, rawr_bytes, cr_bytes)) return error.CRoaringMismatch;
}

fn printInventory(implementation: Implementation, allocator_kind: AllocatorKind) !void {
    switch (implementation) {
        .rawr => try printRawrInventory(allocator_kind),
        .croaring => printCRoaringInventory(),
    }
}

fn printRawrInventory(allocator_kind: AllocatorKind) !void {
    ensureRawrSource();
    defer cleanup();

    var counting = CountingAllocator.init(allocator(allocator_kind));
    var clone = try rawr_source.?.clone(counting.allocator());
    const stats = counting.snapshot();

    var arrays: u64 = 0;
    var bitsets: u64 = 0;
    var runs: u64 = 0;
    var copied_bytes: u64 = @as(u64, rawr_source.?.size) * @sizeOf(u16);
    for (rawr_source.?.containers[0..rawr_source.?.size]) |container| {
        switch (container.getType()) {
            .array => {
                arrays += 1;
                copied_bytes += @as(u64, container.getArray().cardinality) * @sizeOf(u16);
            },
            .bitset => {
                bitsets += 1;
                copied_bytes += container.getBitset().words.len * @sizeOf(u64);
            },
            .run => {
                runs += 1;
                copied_bytes += @as(u64, container.getRun().n_runs) * 2 * @sizeOf(u16);
            },
            .reserved => return error.ReservedContainer,
        }
    }

    bench_time.print("INVENTORY\trawr\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        @tagName(allocator_kind),
        rawr_source.?.size,
        arrays,
        bitsets,
        runs,
        stats.alloc_calls,
        stats.cumulative_bytes,
        copied_bytes,
    });
    clone.deinit();
    if (counting.stats.live_bytes != 0) return error.InventoryLeak;
}

fn printCRoaringInventory() void {
    ensureCRoaringSource();
    defer cleanup();
    const inventory = c.rawr_cr_range_clone_inventory(cr_source.?);
    bench_time.print("INVENTORY\tcroaring\tlibc\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        inventory.containers,
        inventory.arrays,
        inventory.bitsets,
        inventory.runs,
        inventory.clone_allocations,
        inventory.clone_requested_bytes,
        inventory.copied_bytes,
    });
}
