// SPDX-License-Identifier: MPL-2.0

//! Fresh-process dense set-operation construction diagnosis for spec 29-00.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const CountingAllocator = @import("counting_allocator.zig").CountingAllocator;

const RoaringBitmap = rawr.RoaringBitmap;
const Container = rawr.Container;
const diag = rawr.dense_result_diag;
const allocator = std.heap.smp_allocator;
const warmup_runs = 3;
const timed_runs = 21;
const batch_count = 8192;

const Implementation = enum {
    rawr,
    croaring,
};

const Phase = enum {
    full,
    construction,
    timer_control,
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var operation: ?diag.Operation = null;
    var cell: ?diag.Cell = null;
    var implementation: ?Implementation = null;
    var phase: ?Phase = null;
    var header = false;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--op=")) {
            operation = parseEnum(diag.Operation, arg[5..]) orelse return error.UnknownOperation;
        } else if (std.mem.startsWith(u8, arg, "--cell=")) {
            cell = parseCell(arg[7..]) orelse return error.UnknownCell;
        } else if (std.mem.startsWith(u8, arg, "--implementation=")) {
            implementation = parseEnum(Implementation, arg[17..]) orelse return error.UnknownImplementation;
        } else if (std.mem.startsWith(u8, arg, "--phase=")) {
            phase = parseEnum(Phase, arg[8..]) orelse return error.UnknownPhase;
        } else {
            return error.UnknownArgument;
        }
    }

    if (header) {
        if (operation != null or cell != null or implementation != null or phase != null) return error.ConflictingArguments;
        bench_time.printBenchEnvironment();
        bench_time.print("# dense-result diagnosis: {d}w/{d}t median; batch={d}; result teardown excluded from construction phase\n", .{
            warmup_runs,
            timed_runs,
            batch_count,
        });
        return;
    }

    const selected_operation = operation orelse return error.MissingOperation;
    const selected_cell = cell orelse return error.MissingCell;
    const selected_implementation = implementation orelse return error.MissingImplementation;
    const selected_phase = phase orelse return error.MissingPhase;
    if (selected_implementation == .croaring and selected_cell != .baseline) return error.CRoaringOnlyHasBaseline;

    var left = try RoaringBitmap.init(allocator);
    defer left.deinit();
    var right = try RoaringBitmap.init(allocator);
    defer right.deinit();
    _ = try left.addRange(0, 499_999);
    _ = try right.addRange(250_000, 749_999);

    const c_left = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(c_left);
    const c_right = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(c_right);
    c.roaring_bitmap_add_range(c_left, 0, 500_000);
    c.roaring_bitmap_add_range(c_right, 250_000, 750_000);

    try assertCorpus(&left, &right);
    const median_ns = try dispatchMeasure(selected_operation, selected_cell, selected_implementation, selected_phase, &left, &right, c_left, c_right);
    try dispatchValidate(selected_operation, selected_cell, &left, &right, c_left, c_right);
    if (selected_implementation == .rawr and selected_phase == .full) {
        try dispatchAccounting(selected_operation, selected_cell, &left, &right);
    }

    bench_time.print("RESULT\t{s}\t{s}\t{s}\t{s}\t{d}\t{d}\n", .{
        @tagName(selected_operation),
        cellName(selected_cell),
        @tagName(selected_implementation),
        @tagName(selected_phase),
        batch_count,
        median_ns,
    });
}

fn parseEnum(comptime T: type, text: []const u8) ?T {
    inline for (std.meta.fields(T)) |field| {
        if (std.mem.eql(u8, text, field.name)) return @enumFromInt(field.value);
    }
    return null;
}

fn parseCell(text: []const u8) ?diag.Cell {
    if (std.mem.eql(u8, text, "baseline")) return .baseline;
    if (std.mem.eql(u8, text, "a")) return .a;
    if (std.mem.eql(u8, text, "b")) return .b;
    if (std.mem.eql(u8, text, "c")) return .c;
    if (std.mem.eql(u8, text, "a-c")) return .a_c;
    if (std.mem.eql(u8, text, "b-c")) return .b_c;
    return null;
}

fn cellName(cell: diag.Cell) []const u8 {
    return switch (cell) {
        .baseline => "baseline",
        .a => "a",
        .b => "b",
        .c => "c",
        .a_c => "a-c",
        .b_c => "b-c",
    };
}

fn assertCorpus(left: *const RoaringBitmap, right: *const RoaringBitmap) !void {
    const Shape = struct { arrays: u32 = 0, bitsets: u32 = 0, runs: u32 = 0 };
    var left_shape = Shape{};
    var right_shape = Shape{};
    for (left.containers[0..left.size]) |tagged| switch (tagged.getType()) {
        .array => left_shape.arrays += 1,
        .bitset => left_shape.bitsets += 1,
        .run => left_shape.runs += 1,
        .reserved => unreachable,
    };
    for (right.containers[0..right.size]) |tagged| switch (tagged.getType()) {
        .array => right_shape.arrays += 1,
        .bitset => right_shape.bitsets += 1,
        .run => right_shape.runs += 1,
        .reserved => unreachable,
    };
    if (left.size != 8 or left_shape.runs != 8 or left_shape.arrays != 0 or left_shape.bitsets != 0) return error.LeftCorpusDrift;
    if (right.size != 9 or right_shape.runs != 9 or right_shape.arrays != 0 or right_shape.bitsets != 0) return error.RightCorpusDrift;

    var matched: u32 = 0;
    var identity_pairs: u32 = 0;
    var i: usize = 0;
    var j: usize = 0;
    while (i < left.size and j < right.size) {
        if (left.keys[i] < right.keys[j]) {
            i += 1;
        } else if (left.keys[i] > right.keys[j]) {
            j += 1;
        } else {
            const a = Container.fromTagged(left.containers[i]);
            const b = Container.fromTagged(right.containers[j]);
            if (a != .run or b != .run) return error.MatchedContainerTypeDrift;
            matched += 1;
            if (diag.isFullRun(a) or diag.isFullRun(b)) identity_pairs += 1;
            i += 1;
            j += 1;
        }
    }
    if (matched != 5 or identity_pairs != 5) return error.MatchedCorpusDrift;
    bench_time.print("SHAPE\tleft=8-run\tright=9-run\tmatched=5-run-run\tfull-identity=5\tand-result=5\tor-result=12\n", .{});
}

fn dispatchMeasure(
    operation: diag.Operation,
    cell: diag.Cell,
    implementation: Implementation,
    phase: Phase,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
    c_left: *const c.roaring_bitmap_t,
    c_right: *const c.roaring_bitmap_t,
) !u64 {
    if (implementation == .croaring) return measureCRoaring(operation, phase, c_left, c_right);
    return switch (operation) {
        .band => switch (cell) {
            .baseline => measureRawr(.band, .baseline, phase, left, right),
            .a => measureRawr(.band, .a, phase, left, right),
            .b => measureRawr(.band, .b, phase, left, right),
            .c => measureRawr(.band, .c, phase, left, right),
            .a_c => measureRawr(.band, .a_c, phase, left, right),
            .b_c => measureRawr(.band, .b_c, phase, left, right),
        },
        .bor => switch (cell) {
            .baseline => measureRawr(.bor, .baseline, phase, left, right),
            .a => measureRawr(.bor, .a, phase, left, right),
            .b => measureRawr(.bor, .b, phase, left, right),
            .c => measureRawr(.bor, .c, phase, left, right),
            .a_c => measureRawr(.bor, .a_c, phase, left, right),
            .b_c => measureRawr(.bor, .b_c, phase, left, right),
        },
    };
}

fn measureRawr(
    comptime operation: diag.Operation,
    comptime cell: diag.Cell,
    phase: Phase,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
) !u64 {
    for (0..warmup_runs) |_| _ = try timeRawrBatch(operation, cell, phase, left, right);
    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        elapsed.* = try timeRawrBatch(operation, cell, phase, left, right);
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

noinline fn timeRawrBatch(
    comptime operation: diag.Operation,
    comptime cell: diag.Cell,
    phase: Phase,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
) !u64 {
    if (phase == .full) {
        const start = bench_time.monotonicNanos();
        for (0..batch_count) |_| {
            var result = try diag.merge(allocator, left, right, operation, cell, null);
            std.mem.doNotOptimizeAway(result.size);
            result.deinit();
        }
        return bench_time.monotonicNanos() - start;
    }

    var elapsed: u64 = 0;
    for (0..batch_count) |_| {
        if (phase == .timer_control) {
            const start = bench_time.monotonicNanos();
            std.mem.doNotOptimizeAway(start);
            elapsed +%= bench_time.monotonicNanos() - start;
        } else {
            const start = bench_time.monotonicNanos();
            var result = try diag.merge(allocator, left, right, operation, cell, null);
            elapsed +%= bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(result.size);
            result.deinit();
        }
    }
    std.mem.doNotOptimizeAway(elapsed);
    return elapsed;
}

fn measureCRoaring(
    operation: diag.Operation,
    phase: Phase,
    left: *const c.roaring_bitmap_t,
    right: *const c.roaring_bitmap_t,
) !u64 {
    for (0..warmup_runs) |_| _ = try timeCRoaringBatch(operation, phase, left, right);
    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        elapsed.* = try timeCRoaringBatch(operation, phase, left, right);
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

noinline fn timeCRoaringBatch(
    operation: diag.Operation,
    phase: Phase,
    left: *const c.roaring_bitmap_t,
    right: *const c.roaring_bitmap_t,
) !u64 {
    if (phase == .full) {
        const start = bench_time.monotonicNanos();
        for (0..batch_count) |_| {
            const result = cMerge(operation, left, right) orelse return error.OutOfMemory;
            std.mem.doNotOptimizeAway(result);
            c.roaring_bitmap_free(result);
        }
        return bench_time.monotonicNanos() - start;
    }

    var elapsed: u64 = 0;
    for (0..batch_count) |_| {
        if (phase == .timer_control) {
            const start = bench_time.monotonicNanos();
            std.mem.doNotOptimizeAway(start);
            elapsed +%= bench_time.monotonicNanos() - start;
        } else {
            const start = bench_time.monotonicNanos();
            const result = cMerge(operation, left, right) orelse return error.OutOfMemory;
            elapsed +%= bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(result);
            c.roaring_bitmap_free(result);
        }
    }
    std.mem.doNotOptimizeAway(elapsed);
    return elapsed;
}

fn cMerge(operation: diag.Operation, left: *const c.roaring_bitmap_t, right: *const c.roaring_bitmap_t) ?*c.roaring_bitmap_t {
    return switch (operation) {
        .band => c.roaring_bitmap_and(left, right),
        .bor => c.roaring_bitmap_or(left, right),
    };
}

fn dispatchValidate(
    operation: diag.Operation,
    cell: diag.Cell,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
    c_left: *const c.roaring_bitmap_t,
    c_right: *const c.roaring_bitmap_t,
) !void {
    switch (operation) {
        .band => switch (cell) {
            inline else => |comptime_cell| try validateCell(.band, comptime_cell, left, right, c_left, c_right),
        },
        .bor => switch (cell) {
            inline else => |comptime_cell| try validateCell(.bor, comptime_cell, left, right, c_left, c_right),
        },
    }
}

fn validateCell(
    comptime operation: diag.Operation,
    comptime cell: diag.Cell,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
    c_left: *const c.roaring_bitmap_t,
    c_right: *const c.roaring_bitmap_t,
) !void {
    var stats = diag.Diagnostics{};
    var result = try diag.merge(allocator, left, right, operation, cell, &stats);
    defer result.deinit();
    const expected_size: u32 = if (operation == .band) 5 else 12;
    if (result.size != expected_size) return error.ResultShapeDrift;
    if (cell.identityEnabled()) {
        if (stats.identity_hits != 5) return error.IdentityHitDrift;
    } else if (stats.identity_hits != 0) return error.UnexpectedIdentityHit;

    const c_result = cMerge(operation, c_left, c_right) orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(c_result);
    const rawr_len = result.serializedSizeInBytes();
    const c_len = c.roaring_bitmap_portable_size_in_bytes(c_result);
    if (rawr_len != c_len) return error.SerializedSizeMismatch;
    const rawr_bytes = try allocator.alloc(u8, rawr_len);
    defer allocator.free(rawr_bytes);
    var writer = std.Io.Writer.fixed(rawr_bytes);
    try result.serializeToWriter(&writer);
    const c_bytes = try allocator.alloc(u8, c_len);
    defer allocator.free(c_bytes);
    if (c.roaring_bitmap_portable_serialize(c_result, @ptrCast(c_bytes.ptr)) != c_len) return error.CRoaringSerializeFailed;
    if (!std.mem.eql(u8, rawr_bytes, c_bytes)) return error.CRoaringByteMismatch;
    bench_time.print("VALIDATION\t{s}\t{s}\tcontainers={d}\tidentity_hits={d}\tbytes={d}\n", .{
        @tagName(operation),
        cellName(cell),
        result.size,
        stats.identity_hits,
        rawr_len,
    });
}

fn dispatchAccounting(operation: diag.Operation, cell: diag.Cell, left: *const RoaringBitmap, right: *const RoaringBitmap) !void {
    switch (operation) {
        .band => switch (cell) {
            inline else => |comptime_cell| try accountCell(.band, comptime_cell, left, right),
        },
        .bor => switch (cell) {
            inline else => |comptime_cell| try accountCell(.bor, comptime_cell, left, right),
        },
    }
}

fn accountCell(
    comptime operation: diag.Operation,
    comptime cell: diag.Cell,
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
) !void {
    var counting = CountingAllocator.init(allocator);
    var diagnostics = diag.Diagnostics{};
    var result = try diag.merge(counting.allocator(), left, right, operation, cell, &diagnostics);
    result.deinit();
    if (counting.stats.live_bytes != 0) return error.DiagnosticAllocationLeak;
    bench_time.print("ALLOC\t{s}\t{s}\tpersistent_allocs={d}\tpersistent_frees={d}\tpersistent_resizes={d}\tpersistent_requested_bytes={d}\tscratch_constructions={d}\tscratch_reservations={d}\tscratch_requested_bytes={d}\n", .{
        @tagName(operation),
        cellName(cell),
        counting.stats.alloc_calls,
        counting.stats.free_calls,
        counting.stats.resize_calls + counting.stats.remap_calls,
        counting.stats.cumulative_bytes,
        diagnostics.scratch_constructions,
        diagnostics.scratch_reservations,
        diagnostics.scratch_requested_bytes,
    });
}
