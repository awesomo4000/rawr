// SPDX-License-Identifier: MPL-2.0

//! Fresh-process lazy-OR allocator cost attribution worker.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const dashboard = @import("bench_croaring.zig");

const BitsetContainer = rawr.BitsetContainer;
const AllocatorKind = dashboard.ParityAllocator;

const warmup_runs = 3;
const timed_runs = 21;
const matched_keys = 16_364;
const max_pages_per_words = 3;

const Cell = enum {
    c0,
    p1,
    p2,
    p3,

    fn displayName(self: Cell) []const u8 {
        return switch (self) {
            .c0 => "C0",
            .p1 => "P1",
            .p2 => "P2",
            .p3 => "P3",
        };
    }
};

const Stats = struct {
    median: u64,
    minimum: u64,
    maximum: u64,
};

const AddressStats = struct {
    span: usize,
    distinct_pages: usize,
    straddling: usize,
    contiguous_pairs: usize,
    monotonic_pairs: usize,
    stride_median: usize,
    stride_minimum: usize,
    stride_maximum: usize,
};

var headers: [matched_keys]*BitsetContainer = undefined;
var words: [matched_keys][]align(64) u64 = undefined;
var addresses: [matched_keys]usize = undefined;
var pages: [matched_keys * max_pages_per_words]usize = undefined;
var strides: [matched_keys - 1]usize = undefined;

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var cell: ?Cell = null;
    var allocator_kind: ?AllocatorKind = null;
    var profile_wait = false;
    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--cell=")) {
            cell = parseCell(arg[7..]) orelse return error.UnknownCell;
        } else if (std.mem.startsWith(u8, arg, "--allocator=")) {
            allocator_kind = parseAllocator(arg[12..]) orelse return error.UnknownAllocator;
        } else if (std.mem.eql(u8, arg, "--profile-wait")) {
            profile_wait = true;
        } else {
            return error.UnknownArgument;
        }
    }

    const selected_cell = cell orelse return error.MissingCell;
    const selected_allocator = allocator_kind orelse return error.MissingAllocator;
    if (selected_allocator != .smp and selected_allocator != .libc) return error.UnsupportedAllocator;

    bench_time.printBenchEnvironment();
    bench_time.print("# diagnostic: lazy-or-allocator-cost\n", .{});
    bench_time.print("# protocol: {d}w/{d}t median\n", .{ warmup_runs, timed_runs });
    bench_time.print("# tuple: cell={s} allocator={s}\n", .{
        selected_cell.displayName(),
        @tagName(selected_allocator),
    });
    bench_time.print("# matched-bitsets: {d}\n", .{matched_keys});

    if (selected_cell == .c0) {
        try runCanonical(selected_allocator, profile_wait, init.io);
    } else {
        if (profile_wait) return error.ProfileWaitRequiresC0;
        try runProbe(selected_cell, selected_allocator);
    }
}

fn parseCell(value: []const u8) ?Cell {
    if (std.ascii.eqlIgnoreCase(value, "C0") or std.mem.eql(u8, value, "0")) return .c0;
    if (std.ascii.eqlIgnoreCase(value, "P1") or std.mem.eql(u8, value, "1")) return .p1;
    if (std.ascii.eqlIgnoreCase(value, "P2") or std.mem.eql(u8, value, "2")) return .p2;
    if (std.ascii.eqlIgnoreCase(value, "P3") or std.mem.eql(u8, value, "3")) return .p3;
    return null;
}

fn parseAllocator(value: []const u8) ?AllocatorKind {
    if (std.ascii.eqlIgnoreCase(value, "smp")) return .smp;
    if (std.ascii.eqlIgnoreCase(value, "libc")) return .libc;
    return null;
}

fn allocatorFor(kind: AllocatorKind) std.mem.Allocator {
    return switch (kind) {
        .smp => std.heap.smp_allocator,
        .libc => bench_time.cAllocator(),
        else => unreachable,
    };
}

fn runCanonical(kind: AllocatorKind, profile_wait: bool, io: std.Io) !void {
    dashboard.parityPrepare(.lazy_or_construction, .rawr);
    defer dashboard.parityCleanup();

    for (0..warmup_runs) |_| {
        _ = dashboard.parityRun(.lazy_or_construction, .rawr, kind);
    }

    if (profile_wait) {
        bench_time.print("PROFILE_READY\t{s}\n", .{@tagName(kind)});
        try std.Io.sleep(io, .fromSeconds(1), .awake);
    }

    var times: [timed_runs]u64 = undefined;
    if (profile_wait) {
        for (&times) |*elapsed| {
            const start = bench_time.monotonicNanos();
            var result = dashboard.rawr_prof_timed_lazy_or(kind);
            elapsed.* = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(&result);
            result.deinit();
        }
    } else {
        for (&times) |*elapsed| {
            elapsed.* = dashboard.parityRun(.lazy_or_construction, .rawr, kind);
        }
    }

    try dashboard.parityValidate(.lazy_or_construction, kind);
    printResult(.c0, kind, summarize(times));
    bench_time.print("VALIDATION\tC0\t{s}\tok\n", .{@tagName(kind)});
}

fn runProbe(cell: Cell, kind: AllocatorKind) !void {
    const allocator = allocatorFor(kind);

    for (0..warmup_runs) |_| _ = try probeInvocation(cell, allocator, false);

    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| elapsed.* = try probeInvocation(cell, allocator, false);

    _ = try probeInvocation(cell, allocator, true);
    try validateProbe(cell, allocator);

    printResult(cell, kind, summarize(times));
    printAddressResult(cell, kind, addressStatistics(c.rawr_residency_page_size()));
    bench_time.print("VALIDATION\t{s}\t{s}\tok\n", .{ cell.displayName(), @tagName(kind) });
}

fn probeInvocation(cell: Cell, allocator: std.mem.Allocator, keep_addresses: bool) !u64 {
    var allocated: usize = 0;
    errdefer freeProbe(allocator, allocated);

    const start = bench_time.monotonicNanos();
    for (0..matched_keys) |index| {
        const header = try allocator.create(BitsetContainer);
        errdefer allocator.destroy(header);
        const buffer = try allocator.alignedAlloc(u64, .@"64", BitsetContainer.NUM_WORDS);

        if (cell == .p2) @memset(buffer, 0);
        header.* = .{ .words = buffer[0..BitsetContainer.NUM_WORDS], .cardinality = 0 };
        headers[index] = header;
        words[index] = buffer;
        allocated += 1;
    }
    if (cell == .p3) {
        for (words[0..allocated]) |buffer| @memset(buffer, 0);
    }
    const elapsed = bench_time.monotonicNanos() - start;

    if (keep_addresses) {
        for (words[0..allocated], 0..) |buffer, index| addresses[index] = @intFromPtr(buffer.ptr);
    }
    std.mem.doNotOptimizeAway(headers[0..allocated]);
    freeProbe(allocator, allocated);
    return elapsed;
}

fn freeProbe(allocator: std.mem.Allocator, allocated: usize) void {
    for (0..allocated) |index| {
        allocator.free(words[index]);
        allocator.destroy(headers[index]);
    }
}

fn validateProbe(cell: Cell, allocator: std.mem.Allocator) !void {
    var allocated: usize = 0;
    defer freeProbe(allocator, allocated);

    for (0..matched_keys) |index| {
        const header = try allocator.create(BitsetContainer);
        errdefer allocator.destroy(header);
        const buffer = try allocator.alignedAlloc(u64, .@"64", BitsetContainer.NUM_WORDS);
        if (cell != .p1) @memset(buffer, 0);
        header.* = .{ .words = buffer[0..BitsetContainer.NUM_WORDS], .cardinality = 0 };
        headers[index] = header;
        words[index] = buffer;
        allocated += 1;

        if (@intFromPtr(buffer.ptr) % 64 != 0) return error.InvalidAlignment;
    }

    if (cell != .p1) {
        for (words[0..allocated]) |buffer| {
            for (buffer) |word| if (word != 0) return error.ZeroValidationFailed;
        }
    }
}

fn addressStatistics(page_size_raw: anytype) AddressStats {
    const page_size: usize = @intCast(page_size_raw);
    if (page_size == 0 or !std.math.isPowerOfTwo(page_size)) @panic("invalid page size");

    var minimum = addresses[0];
    var maximum_end = addresses[0] + BitsetContainer.SIZE_BYTES;
    var page_count: usize = 0;
    var straddling: usize = 0;
    for (addresses, 0..) |address, index| {
        minimum = @min(minimum, address);
        maximum_end = @max(maximum_end, address + BitsetContainer.SIZE_BYTES);
        if (address / page_size != (address + BitsetContainer.SIZE_BYTES - 1) / page_size) {
            straddling += 1;
        }

        var page = address - address % page_size;
        const end = address + BitsetContainer.SIZE_BYTES;
        while (page < end) : (page += page_size) {
            if (page_count == pages.len) @panic("page buffer too small");
            pages[page_count] = page;
            page_count += 1;
        }

        if (index != 0) strides[index - 1] = absDiff(address, addresses[index - 1]);
    }

    std.mem.sort(usize, pages[0..page_count], {}, std.sort.asc(usize));
    const distinct_pages = uniqueSorted(pages[0..page_count]);

    var contiguous_pairs: usize = 0;
    var monotonic_pairs: usize = 0;
    for (addresses[1..], addresses[0 .. addresses.len - 1]) |current, previous| {
        if (current == previous + BitsetContainer.SIZE_BYTES) contiguous_pairs += 1;
        if (current > previous) monotonic_pairs += 1;
    }

    var sorted_strides = strides;
    std.mem.sort(usize, &sorted_strides, {}, std.sort.asc(usize));
    return .{
        .span = maximum_end - minimum,
        .distinct_pages = distinct_pages,
        .straddling = straddling,
        .contiguous_pairs = contiguous_pairs,
        .monotonic_pairs = monotonic_pairs,
        .stride_median = sorted_strides[sorted_strides.len / 2],
        .stride_minimum = sorted_strides[0],
        .stride_maximum = sorted_strides[sorted_strides.len - 1],
    };
}

fn absDiff(left: usize, right: usize) usize {
    return if (left >= right) left - right else right - left;
}

fn uniqueSorted(values: []usize) usize {
    if (values.len == 0) return 0;
    var write: usize = 1;
    for (values[1..]) |value| {
        if (value == values[write - 1]) continue;
        values[write] = value;
        write += 1;
    }
    return write;
}

fn summarize(values: [timed_runs]u64) Stats {
    var sorted = values;
    std.mem.sort(u64, &sorted, {}, std.sort.asc(u64));
    return .{
        .median = sorted[timed_runs / 2],
        .minimum = sorted[0],
        .maximum = sorted[timed_runs - 1],
    };
}

fn printResult(cell: Cell, kind: AllocatorKind, stats: Stats) void {
    bench_time.print("RESULT\t{s}\t{s}\t{d}\t{d}\t{d}\n", .{
        cell.displayName(),
        @tagName(kind),
        stats.median,
        stats.minimum,
        stats.maximum,
    });
}

fn printAddressResult(cell: Cell, kind: AllocatorKind, stats: AddressStats) void {
    bench_time.print("ADDRESS\t{s}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        cell.displayName(),
        @tagName(kind),
        stats.span,
        stats.distinct_pages,
        stats.straddling,
        stats.contiguous_pairs,
        stats.monotonic_pairs,
        stats.stride_median,
        stats.stride_minimum,
        stats.stride_maximum,
    });
}
