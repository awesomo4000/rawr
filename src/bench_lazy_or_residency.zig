// SPDX-License-Identifier: MPL-2.0

//! Fresh-process lazy-OR page-residency diagnosis worker.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const dashboard = @import("bench_croaring.zig");

const BitsetContainer = rawr.BitsetContainer;
const Implementation = dashboard.ParityImplementation;

const warmup_runs = 3;
const timed_runs = 21;
const matched_keys = 16_364;
const max_pages_per_words = 3;
const eviction_cache_multiple = 4;

const Cell = enum {
    c0,
    c1,
    c2,
    c3,
    c4,

    fn hasPrepass(self: Cell) bool {
        return self != .c0;
    }

    fn touchesPayload(self: Cell) bool {
        return self == .c2 or self == .c4;
    }

    fn evictsCache(self: Cell) bool {
        return self == .c3 or self == .c4;
    }

    fn displayName(self: Cell) []const u8 {
        return switch (self) {
            .c0 => "C0",
            .c1 => "C1",
            .c2 => "C2",
            .c3 => "C3",
            .c4 => "C4",
        };
    }
};

const FaultDelta = struct {
    primary: u64 = 0,
    major: u64 = 0,
    cow: u64 = 0,
    source: u32 = 0,
    valid: bool = false,
};

const Invocation = struct {
    elapsed_ns: u64,
    operation_faults: FaultDelta,
    prepass_faults: FaultDelta,
};

const Stats = struct {
    median: u64,
    minimum: u64,
    maximum: u64,
    sum: u64,
};

const FaultStats = struct {
    primary: Stats,
    major: Stats,
    cow: Stats,
    source: u32,
    valid: bool,
};

const ReuseStats = struct {
    prepass_pages: usize,
    production_pages: usize,
    overlap_pages: usize,
};

var prepass_headers: [matched_keys]*BitsetContainer = undefined;
var prepass_words: [matched_keys][]align(64) u64 = undefined;
var prepass_pages: [matched_keys * max_pages_per_words]usize = undefined;
var production_addresses: [matched_keys]usize = undefined;
var production_pages: [matched_keys * max_pages_per_words]usize = undefined;

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var cell: ?Cell = null;
    var implementation: ?Implementation = null;
    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--cell=")) {
            cell = parseCell(arg[7..]) orelse return error.UnknownCell;
        } else if (std.mem.startsWith(u8, arg, "--implementation=")) {
            implementation = std.meta.stringToEnum(Implementation, arg[17..]) orelse
                return error.UnknownImplementation;
        } else {
            return error.UnknownArgument;
        }
    }

    const selected_cell = cell orelse return error.MissingCell;
    const selected_implementation = implementation orelse return error.MissingImplementation;

    dashboard.parityPrepare(.lazy_or_construction, selected_implementation);
    defer dashboard.parityCleanup();

    const page_size = c.rawr_residency_page_size();
    if (page_size == 0 or !std.math.isPowerOfTwo(page_size)) return error.InvalidPageSize;
    const cache_size = c.rawr_residency_last_level_cache_size();
    if (cache_size == 0 or cache_size > std.math.maxInt(usize) / eviction_cache_multiple) {
        return error.CacheSizeUnavailable;
    }
    const eviction_size: usize = @intCast(cache_size * eviction_cache_multiple);
    const eviction_buffer = try std.heap.page_allocator.alloc(u8, eviction_size);
    defer std.heap.page_allocator.free(eviction_buffer);
    prefaultRange(eviction_buffer.ptr, eviction_buffer.len, page_size);
    prefaultPrepassStorage(page_size);

    const initial_faults = sampleFaults();
    printHeader(selected_cell, selected_implementation, page_size, cache_size, eviction_size, initial_faults);

    for (0..warmup_runs) |_| {
        _ = try runInvocation(selected_cell, selected_implementation, eviction_buffer, page_size);
    }

    var times: [timed_runs]u64 = undefined;
    var operation_faults: [timed_runs]FaultDelta = undefined;
    var prepass_faults: [timed_runs]FaultDelta = undefined;
    for (0..timed_runs) |index| {
        const invocation = try runInvocation(
            selected_cell,
            selected_implementation,
            eviction_buffer,
            page_size,
        );
        times[index] = invocation.elapsed_ns;
        operation_faults[index] = invocation.operation_faults;
        prepass_faults[index] = invocation.prepass_faults;
    }

    const reuse = if (selected_implementation == .rawr and selected_cell.hasPrepass())
        try provePageReuse(selected_cell, eviction_buffer, page_size)
    else
        null;

    try dashboard.parityValidate(
        .lazy_or_construction,
        if (selected_implementation == .rawr) .smp else .libc,
    );

    const time_stats = summarize(times);
    const operation_stats = summarizeFaults(operation_faults);
    const prepass_stats = summarizeFaults(prepass_faults);
    printResults(
        selected_cell,
        selected_implementation,
        time_stats,
        operation_stats,
        prepass_stats,
        reuse,
    );
}

fn parseCell(value: []const u8) ?Cell {
    if (std.ascii.eqlIgnoreCase(value, "C0") or std.mem.eql(u8, value, "0")) return .c0;
    if (std.ascii.eqlIgnoreCase(value, "C1") or std.mem.eql(u8, value, "1")) return .c1;
    if (std.ascii.eqlIgnoreCase(value, "C2") or std.mem.eql(u8, value, "2")) return .c2;
    if (std.ascii.eqlIgnoreCase(value, "C3") or std.mem.eql(u8, value, "3")) return .c3;
    if (std.ascii.eqlIgnoreCase(value, "C4") or std.mem.eql(u8, value, "4")) return .c4;
    return null;
}

fn printHeader(
    cell: Cell,
    implementation: Implementation,
    page_size: usize,
    cache_size: u64,
    eviction_size: usize,
    faults: dashboard.ParityFaultSnapshot,
) void {
    bench_time.printBenchEnvironment();
    bench_time.print("# diagnostic: lazy-or-residency\n", .{});
    bench_time.print("# protocol: {d}w/{d}t median\n", .{ warmup_runs, timed_runs });
    bench_time.print("# tuple: cell={s} implementation={s}\n", .{ cell.displayName(), @tagName(implementation) });
    bench_time.print("# matched-keys: {d}\n", .{matched_keys});
    bench_time.print("# prepass-allocations: headers={d} words={d} total={d} (mechanical corpus count)\n", .{
        matched_keys,
        matched_keys,
        matched_keys * 2,
    });
    bench_time.print("# page-size: {d}\n", .{page_size});
    bench_time.print("# last-level-cache: {d} source={s}\n", .{ cache_size, cacheSourceName(c.rawr_residency_cache_source()) });
    bench_time.print("# eviction-buffer: {d} bytes ({d}x reported cache)\n", .{ eviction_size, eviction_cache_multiple });
    bench_time.print("# fault-source: {s} valid={d}\n", .{ faultSourceName(faults.source), @intFromBool(faults.valid) });
}

fn cacheSourceName(source: u32) []const u8 {
    return switch (source) {
        c.RAWR_RESIDENCY_CACHE_LINUX_L3 => "linux-sysconf-l3",
        c.RAWR_RESIDENCY_CACHE_DARWIN_L3 => "darwin-hw.l3cachesize",
        c.RAWR_RESIDENCY_CACHE_DARWIN_PERF_L2 => "darwin-hw.perflevel0.l2cachesize",
        c.RAWR_RESIDENCY_CACHE_DARWIN_L2 => "darwin-hw.l2cachesize",
        else => "unavailable",
    };
}

fn faultSourceName(source: u32) []const u8 {
    return switch (source) {
        c.RAWR_RESIDENCY_FAULT_LINUX_RUSAGE => "linux-getrusage-ru_minflt",
        c.RAWR_RESIDENCY_FAULT_DARWIN_TASK_EVENTS => "darwin-task-events-faults",
        else => "unavailable",
    };
}

fn sampleFaults() dashboard.ParityFaultSnapshot {
    var snapshot: c.rawr_residency_fault_snapshot_t = std.mem.zeroes(c.rawr_residency_fault_snapshot_t);
    const ok = c.rawr_residency_fault_snapshot(&snapshot) != 0;
    return .{
        .primary = snapshot.primary,
        .major = snapshot.major,
        .cow = snapshot.cow,
        .source = snapshot.source,
        .valid = ok and snapshot.valid != 0,
    };
}

fn faultDelta(before: dashboard.ParityFaultSnapshot, after: dashboard.ParityFaultSnapshot) FaultDelta {
    const valid = before.valid and after.valid and before.source == after.source and
        after.primary >= before.primary and after.major >= before.major and after.cow >= before.cow;
    if (!valid) return .{ .source = after.source };
    return .{
        .primary = after.primary - before.primary,
        .major = after.major - before.major,
        .cow = after.cow - before.cow,
        .source = after.source,
        .valid = true,
    };
}

fn runInvocation(
    cell: Cell,
    implementation: Implementation,
    eviction_buffer: []u8,
    page_size: usize,
) !Invocation {
    const prepass_before = sampleFaults();
    if (cell.hasPrepass()) _ = try runPrepass(cell.touchesPayload(), page_size, false);
    const prepass_after = sampleFaults();

    if (cell.evictsCache()) evictCache(eviction_buffer);

    const observation = dashboard.parityObserveLazyConstruction(implementation, sampleFaults);
    return .{
        .elapsed_ns = observation.elapsed_ns,
        .operation_faults = faultDelta(observation.before, observation.after),
        .prepass_faults = faultDelta(prepass_before, prepass_after),
    };
}

fn runPrepass(touch_payload: bool, page_size: usize, record_pages: bool) !usize {
    const allocator = std.heap.smp_allocator;
    var allocated: usize = 0;
    errdefer {
        for (0..allocated) |index| {
            allocator.free(prepass_words[index]);
            allocator.destroy(prepass_headers[index]);
        }
    }

    for (0..matched_keys) |index| {
        const header = try allocator.create(BitsetContainer);
        errdefer allocator.destroy(header);
        const words = try allocator.alignedAlloc(u64, .@"64", BitsetContainer.NUM_WORDS);
        prepass_headers[index] = header;
        prepass_words[index] = words;
        allocated += 1;
    }

    if (touch_payload) {
        for (prepass_words[0..allocated]) |words| {
            prefaultRange(@ptrCast(words.ptr), @sizeOf(u64) * words.len, page_size);
        }
    }

    if (record_pages) {
        var page_count: usize = 0;
        for (prepass_words[0..allocated]) |words| {
            appendPages(&prepass_pages, &page_count, @intFromPtr(words.ptr), @sizeOf(u64) * words.len, page_size);
        }
        page_count = uniquePages(prepass_pages[0..page_count]);
        for (0..allocated) |index| {
            allocator.free(prepass_words[index]);
            allocator.destroy(prepass_headers[index]);
        }
        return page_count;
    }

    for (0..allocated) |index| {
        allocator.free(prepass_words[index]);
        allocator.destroy(prepass_headers[index]);
    }
    return 0;
}

fn prefaultPrepassStorage(page_size: usize) void {
    prefaultRange(@ptrCast(&prepass_headers), @sizeOf(@TypeOf(prepass_headers)), page_size);
    prefaultRange(@ptrCast(&prepass_words), @sizeOf(@TypeOf(prepass_words)), page_size);
}

fn prefaultRange(pointer: [*]u8, len: usize, page_size: usize) void {
    if (len == 0) return;
    const start = @intFromPtr(pointer);
    const end = start + len;
    var page = start - start % page_size;
    while (page < end) : (page += page_size) {
        const address = @max(page, start);
        const byte: *volatile u8 = @ptrFromInt(address);
        byte.* = 0;
    }
}

fn evictCache(buffer: []u8) void {
    var checksum: u64 = 0;
    var index: usize = 0;
    while (index < buffer.len) : (index += 64) {
        const byte: *volatile u8 = @ptrCast(&buffer[index]);
        checksum +%= byte.*;
    }
    std.mem.doNotOptimizeAway(checksum);
}

fn provePageReuse(cell: Cell, eviction_buffer: []u8, page_size: usize) !ReuseStats {
    const prepass_page_count = try runPrepass(cell.touchesPayload(), page_size, true);
    if (cell.evictsCache()) evictCache(eviction_buffer);

    const address_count = try dashboard.parityRawrLazyWordAddresses(&production_addresses);
    if (address_count != matched_keys) return error.UnexpectedTransientBitsetCount;

    var production_page_count: usize = 0;
    for (production_addresses[0..address_count]) |address| {
        appendPages(
            &production_pages,
            &production_page_count,
            address,
            BitsetContainer.SIZE_BYTES,
            page_size,
        );
    }

    production_page_count = uniquePages(production_pages[0..production_page_count]);
    const overlap = countIntersection(
        prepass_pages[0..prepass_page_count],
        production_pages[0..production_page_count],
    );
    return .{
        .prepass_pages = prepass_page_count,
        .production_pages = production_page_count,
        .overlap_pages = overlap,
    };
}

fn appendPages(out: []usize, count: *usize, start: usize, len: usize, page_size: usize) void {
    const end = start + len;
    var page = start - start % page_size;
    while (page < end) : (page += page_size) {
        if (count.* == out.len) @panic("page address buffer too small");
        out[count.*] = page;
        count.* += 1;
    }
}

fn uniquePages(pages: []usize) usize {
    if (pages.len == 0) return 0;
    std.mem.sort(usize, pages, {}, std.sort.asc(usize));
    var write: usize = 1;
    for (pages[1..]) |page| {
        if (page == pages[write - 1]) continue;
        pages[write] = page;
        write += 1;
    }
    return write;
}

fn countIntersection(left: []const usize, right: []const usize) usize {
    var i: usize = 0;
    var j: usize = 0;
    var count: usize = 0;
    while (i < left.len and j < right.len) {
        if (left[i] < right[j]) {
            i += 1;
        } else if (left[i] > right[j]) {
            j += 1;
        } else {
            count += 1;
            i += 1;
            j += 1;
        }
    }
    return count;
}

fn summarize(values: [timed_runs]u64) Stats {
    var sorted = values;
    std.mem.sort(u64, &sorted, {}, std.sort.asc(u64));
    var sum: u64 = 0;
    for (values) |value| sum +%= value;
    return .{
        .median = sorted[timed_runs / 2],
        .minimum = sorted[0],
        .maximum = sorted[timed_runs - 1],
        .sum = sum,
    };
}

fn summarizeFaults(values: [timed_runs]FaultDelta) FaultStats {
    var primary: [timed_runs]u64 = undefined;
    var major: [timed_runs]u64 = undefined;
    var cow: [timed_runs]u64 = undefined;
    var valid = true;
    var source: u32 = 0;
    for (values, 0..) |value, index| {
        primary[index] = value.primary;
        major[index] = value.major;
        cow[index] = value.cow;
        valid = valid and value.valid;
        if (source == 0) source = value.source;
        valid = valid and value.source == source;
    }
    return .{
        .primary = summarize(primary),
        .major = summarize(major),
        .cow = summarize(cow),
        .source = source,
        .valid = valid,
    };
}

fn printResults(
    cell: Cell,
    implementation: Implementation,
    time: Stats,
    operation: FaultStats,
    prepass: FaultStats,
    reuse: ?ReuseStats,
) void {
    const cell_name = cell.displayName();
    const implementation_name = @tagName(implementation);
    bench_time.print("RESULT\t{s}\t{s}\t{d}\t{d}\t{d}\n", .{
        cell_name,
        implementation_name,
        time.median,
        time.minimum,
        time.maximum,
    });
    printFaultStats(cell_name, implementation_name, "operation", operation);
    printFaultStats(cell_name, implementation_name, "prepass", prepass);
    if (reuse) |value| {
        bench_time.print("REUSE\t{s}\t{s}\t{d}\t{d}\t{d}\n", .{
            cell_name,
            implementation_name,
            value.prepass_pages,
            value.production_pages,
            value.overlap_pages,
        });
    }
    bench_time.print("VALIDATION\t{s}\t{s}\tok\n", .{ cell_name, implementation_name });
}

fn printFaultStats(cell: []const u8, implementation: []const u8, phase: []const u8, stats: FaultStats) void {
    printMetric(cell, implementation, phase, "primary", stats.source, stats.valid, stats.primary);
    printMetric(cell, implementation, phase, "major", stats.source, stats.valid, stats.major);
    printMetric(cell, implementation, phase, "cow", stats.source, stats.valid, stats.cow);
}

fn printMetric(
    cell: []const u8,
    implementation: []const u8,
    phase: []const u8,
    metric: []const u8,
    source: u32,
    valid: bool,
    stats: Stats,
) void {
    bench_time.print("FAULT\t{s}\t{s}\t{s}\t{s}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        cell,
        implementation,
        phase,
        metric,
        faultSourceName(source),
        @intFromBool(valid),
        stats.median,
        stats.minimum,
        stats.maximum,
        stats.sum,
    });
}
