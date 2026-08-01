// SPDX-License-Identifier: MPL-2.0

//! Fresh-process removeRangeCopy construction and allocation diagnostic.

const std = @import("std");
const c = @import("c");
const rawr = @import("rawr");
const bench_time = @import("bench_time.zig");
const CountingAllocator = @import("counting_allocator.zig").CountingAllocator;

const RoaringBitmap = rawr.RoaringBitmap;
const range_ops = rawr.range_ops;
const warmup_runs = 3;
const timed_runs = 21;
const batch_count = 8192;
const range_lo: u32 = 100_000;
const range_hi: u32 = 650_000;

const Cell = enum {
    baseline,
    fused_default,
    fused_presized,

    fn id(self: Cell) []const u8 {
        return switch (self) {
            .baseline => "baseline",
            .fused_default => "fused-default",
            .fused_presized => "fused-presized",
        };
    }
};

const Implementation = enum { rawr, croaring };

var rawr_source: ?RoaringBitmap = null;
var cr_source: ?*c.roaring_bitmap_t = null;

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var header = false;
    var cell: ?Cell = null;
    var implementation: ?Implementation = null;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--cell=")) {
            cell = parseCell(arg[7..]) orelse return error.UnknownCell;
        } else if (std.mem.startsWith(u8, arg, "--implementation=")) {
            implementation = std.meta.stringToEnum(Implementation, arg[17..]) orelse
                return error.UnknownImplementation;
        } else {
            return error.UnknownArgument;
        }
    }

    if (header) {
        if (cell != null or implementation != null) return error.ConflictingArguments;
        printHeader();
        return;
    }

    const selected_cell = cell orelse return error.MissingCell;
    const selected_implementation = implementation orelse return error.MissingImplementation;
    if (selected_implementation == .croaring and selected_cell != .baseline) {
        return error.UnsupportedCell;
    }

    prepare();
    defer cleanup();
    try assertCanonicalCorpus();

    const median_ns = measure(selected_cell, selected_implementation);
    try validateCell(selected_cell);
    if (selected_implementation == .rawr) try printAllocationAccounting(selected_cell);
    bench_time.print("VALIDATION\t{s}\t{s}\tpass\n", .{
        selected_cell.id(),
        @tagName(selected_implementation),
    });
    bench_time.print("RESULT\t{s}\t{s}\tns/op\t{d}\t{d}\n", .{
        selected_cell.id(),
        @tagName(selected_implementation),
        batch_count,
        median_ns,
    });
}

fn parseCell(value: []const u8) ?Cell {
    inline for (std.meta.fields(Cell)) |field| {
        const cell: Cell = @enumFromInt(field.value);
        if (std.mem.eql(u8, value, cell.id())) return cell;
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
    bench_time.print("# allocator: smp\n", .{});
    bench_time.print("# croaring-avx512: {s}\n", .{
        if (c.CROARING_COMPILER_SUPPORTS_AVX512 != 0) "on" else "off",
    });
    bench_time.print("# croaring-accounting: timing-only\n", .{});
}

fn prepare() void {
    var bitmap = RoaringBitmap.init(std.heap.smp_allocator) catch unreachable;
    _ = bitmap.addRange(0, 499_999) catch unreachable;
    rawr_source = bitmap;

    const cr_bitmap = c.roaring_bitmap_create() orelse unreachable;
    c.rawr_cr_set_copy_on_write(cr_bitmap, false);
    c.roaring_bitmap_add_range(cr_bitmap, 0, 500_000);
    cr_source = cr_bitmap;
}

fn cleanup() void {
    if (rawr_source) |*bitmap| bitmap.deinit();
    rawr_source = null;
    if (cr_source) |bitmap| c.roaring_bitmap_free(bitmap);
    cr_source = null;
}

fn assertCanonicalCorpus() !void {
    const source = &rawr_source.?;
    if (source.size != 8) return error.CorpusContainerCount;
    for (source.keys[0..source.size], source.containers[0..source.size], 0..) |key, tagged, index| {
        if (key != index or tagged.getType() != .run) return error.CorpusShape;
        const run = tagged.getRun();
        const expected_end: u16 = if (index == 7) 41_247 else std.math.maxInt(u16);
        if (run.n_runs != 1 or run.runs[0].start != 0 or
            run.runs[0].length != expected_end)
        {
            return error.CorpusShape;
        }
    }
    bench_time.print("SHAPE\t8\t1\t1\t6\t2\n", .{});
}

fn measure(cell: Cell, implementation: Implementation) u64 {
    for (0..warmup_runs) |_| _ = runBatch(cell, implementation);
    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| elapsed.* = runBatch(cell, implementation);
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

fn runBatch(cell: Cell, implementation: Implementation) u64 {
    var elapsed: u64 = 0;
    for (0..batch_count) |_| {
        elapsed +%= switch (implementation) {
            .rawr => timeRawr(cell),
            .croaring => timeCRoaring(),
        };
    }
    std.mem.doNotOptimizeAway(elapsed);
    return elapsed;
}

noinline fn timeRawr(cell: Cell) u64 {
    const start = bench_time.monotonicNanos();
    var result = buildRawrResult(cell, std.heap.smp_allocator) catch unreachable;
    std.mem.doNotOptimizeAway(&result);
    result.deinit();
    return bench_time.monotonicNanos() - start;
}

noinline fn timeCRoaring() u64 {
    const start = bench_time.monotonicNanos();
    const result = buildCRoaringResult() catch unreachable;
    std.mem.doNotOptimizeAway(result);
    c.roaring_bitmap_free(result);
    return bench_time.monotonicNanos() - start;
}

fn buildRawrResult(cell: Cell, allocator: std.mem.Allocator) !RoaringBitmap {
    const source = &rawr_source.?;
    return switch (cell) {
        .baseline => result: {
            var result = try source.clone(allocator);
            errdefer result.deinit();
            _ = try result.removeRange(range_lo, range_hi);
            break :result result;
        },
        .fused_default => range_ops.removeRangeCopyWithCapacity(
            source,
            allocator,
            range_lo,
            range_hi,
            .normal_growth,
        ),
        .fused_presized => range_ops.removeRangeCopyWithCapacity(
            source,
            allocator,
            range_lo,
            range_hi,
            .exact,
        ),
    };
}

fn buildCRoaringResult() !*c.roaring_bitmap_t {
    const result = c.roaring_bitmap_copy(cr_source.?) orelse return error.OutOfMemory;
    c.roaring_bitmap_remove_range_closed(result, range_lo, range_hi);
    return result;
}

fn validateCell(cell: Cell) !void {
    const before = try rawr_source.?.serialize(std.heap.page_allocator);
    defer std.heap.page_allocator.free(before);

    var rawr_result = try buildRawrResult(cell, std.heap.smp_allocator);
    defer rawr_result.deinit();
    const cr_result = try buildCRoaringResult();
    defer c.roaring_bitmap_free(cr_result);

    try assertResultShape(&rawr_result);
    try expectPortableEqual(&rawr_result, cr_result);
    const after = try rawr_source.?.serialize(std.heap.page_allocator);
    defer std.heap.page_allocator.free(after);
    if (!std.mem.eql(u8, before, after)) return error.SourceMutated;
}

fn assertResultShape(result: *const RoaringBitmap) !void {
    if (result.size != 2 or result.keys[0] != 0 or result.keys[1] != 1) {
        return error.ResultShape;
    }
    if (result.containers[0].getType() != .run or result.containers[1].getType() != .run) {
        return error.ResultShape;
    }
    const boundary = result.containers[1].getRun();
    if (boundary.n_runs != 1 or boundary.runs[0].start != 0 or boundary.runs[0].length != 34_463) {
        return error.ResultShape;
    }
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

fn printAllocationAccounting(cell: Cell) !void {
    var counting = CountingAllocator.init(std.heap.smp_allocator);
    var result = try buildRawrResult(cell, counting.allocator());
    try assertResultShape(&result);
    const construction = counting.snapshot();
    result.deinit();
    const complete = counting.snapshot();
    if (complete.live_bytes != 0) return error.AccountingLeak;

    const constructions: u32 = switch (cell) {
        .baseline => 9,
        .fused_default, .fused_presized => 2,
    };
    bench_time.print("ALLOC\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        cell.id(),
        constructions,
        construction.alloc_calls,
        construction.free_calls,
        construction.cumulative_bytes,
        complete.free_calls - construction.free_calls,
    });
}
