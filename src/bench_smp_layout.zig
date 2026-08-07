// SPDX-License-Identifier: MPL-2.0

//! Standalone allocator address-order diagnosis. This intentionally imports no
//! rawr or CRoaring code.

const std = @import("std");
const builtin = @import("builtin");

const block_count = 16_364;
const words_per_block = 1024;
const block_bytes = words_per_block * @sizeOf(u64);
const warmup_runs = 3;
const timed_runs = 21;

const Header = struct {
    words: *align(64) [words_per_block]u64,
    cardinality: i32,
};

comptime {
    if (@sizeOf(Header) != 16) @compileError("layout probe requires a 16-byte header");
}

const AllocatorKind = enum { smp, libc };

const Cell = enum {
    alloc_words,
    alloc_header_words,
    interleaved_words,
    interleaved_header_words,
    zero_order_words,
    zero_order_header_words,
    // Sorting is outside the timer: this isolates zeroing traversal order.
    zero_sorted_words,
    zero_sorted_header_words,
    // Sorting is inside the timer: this reports the honest sort-plus-zero cost.
    sort_zero_words,
    sort_zero_header_words,

    fn hasHeader(self: Cell) bool {
        return switch (self) {
            .alloc_header_words,
            .interleaved_header_words,
            .zero_order_header_words,
            .zero_sorted_header_words,
            .sort_zero_header_words,
            => true,
            else => false,
        };
    }

    fn allocationTimed(self: Cell) bool {
        return switch (self) {
            .alloc_words, .alloc_header_words, .interleaved_words, .interleaved_header_words => true,
            else => false,
        };
    }

    fn zeroTimed(self: Cell) bool {
        return switch (self) {
            .interleaved_words,
            .interleaved_header_words,
            .zero_order_words,
            .zero_order_header_words,
            .zero_sorted_words,
            .zero_sorted_header_words,
            .sort_zero_words,
            .sort_zero_header_words,
            => true,
            else => false,
        };
    }
};

const AddressStats = struct {
    span: usize,
    median_stride: usize,
    adjacent_pairs: usize,
    monotonic_pairs: usize,
};

var blocks: [block_count][]align(64) u64 = undefined;
var sorted_blocks: [block_count][]align(64) u64 = undefined;
var headers: [block_count]*Header = undefined;
var addresses: [block_count]usize = undefined;
var stride_scratch: [block_count - 1]usize = undefined;

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var allocator_kind: ?AllocatorKind = null;
    var cell: ?Cell = null;
    var header_only = false;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--header")) {
            header_only = true;
        } else if (std.mem.startsWith(u8, arg, "--allocator=")) {
            allocator_kind = std.meta.stringToEnum(AllocatorKind, arg[12..]) orelse return error.BadAllocator;
        } else if (std.mem.startsWith(u8, arg, "--cell=")) {
            cell = std.meta.stringToEnum(Cell, arg[7..]) orelse return error.BadCell;
        } else {
            return error.UnknownArgument;
        }
    }

    if (header_only) {
        if (allocator_kind != null or cell != null) return error.HeaderWithTuple;
        printHeader();
        return;
    }

    const selected_allocator = allocator_kind orelse return error.MissingAllocator;
    const selected_cell = cell orelse return error.MissingCell;
    const allocator = switch (selected_allocator) {
        .smp => std.heap.smp_allocator,
        .libc => std.heap.c_allocator,
    };

    for (0..warmup_runs) |_| _ = try invoke(allocator, selected_cell);
    var samples: [timed_runs]u64 = undefined;
    for (&samples) |*sample| sample.* = try invoke(allocator, selected_cell);
    std.mem.sort(u64, &samples, {}, std.sort.asc(u64));

    const address_stats = addressStatistics();
    std.debug.print("RESULT\t{s}\t{s}\t{d}\t{d}\t{d}\n", .{
        @tagName(selected_allocator),
        @tagName(selected_cell),
        samples[timed_runs / 2],
        samples[0],
        samples[timed_runs - 1],
    });
    std.debug.print("ADDRESS\t{s}\t{s}\t{d}\t{d}\t{d}\t{d}\n", .{
        @tagName(selected_allocator),
        @tagName(selected_cell),
        address_stats.span,
        address_stats.median_stride,
        address_stats.adjacent_pairs,
        address_stats.monotonic_pairs,
    });

    try validateCell(allocator, selected_cell);
    std.debug.print("VALIDATION\t{s}\t{s}\tok\n", .{
        @tagName(selected_allocator),
        @tagName(selected_cell),
    });
}

fn printHeader() void {
    std.debug.print("# standalone SMP allocator address-order diagnosis\n", .{});
    std.debug.print("# zig {s} | ReleaseFast | {s} {s}\n", .{
        builtin.zig_version_string,
        @tagName(builtin.os.tag),
        @tagName(builtin.cpu.arch),
    });
    std.debug.print("# cpu: {s}\n", .{builtin.cpu.model.name});
    std.debug.print("# protocol: {d}w/{d}t median, {d} blocks x {d} bytes\n", .{
        warmup_runs,
        timed_runs,
        block_count,
        block_bytes,
    });
    std.debug.print("# zero_sorted: sort excluded; sort_zero: sort included\n", .{});
}

fn invoke(allocator: std.mem.Allocator, cell: Cell) !u64 {
    const with_header = cell.hasHeader();
    const allocation_timed = cell.allocationTimed();
    const interleaved = cell == .interleaved_words or cell == .interleaved_header_words;
    const sort_before_timer = cell == .zero_sorted_words or cell == .zero_sorted_header_words;
    const sort_in_timer = cell == .sort_zero_words or cell == .sort_zero_header_words;

    var count: usize = 0;
    errdefer freeAll(allocator, count, with_header);

    if (!allocation_timed) {
        try allocateAll(allocator, with_header, false, &count);
        if (sort_before_timer) prepareSortedBlocks(count);
    }

    const start = monotonicNanos();
    if (allocation_timed) {
        try allocateAll(allocator, with_header, interleaved, &count);
    } else if (cell.zeroTimed()) {
        if (sort_in_timer) prepareSortedBlocks(count);
        const traversal = if (sort_before_timer or sort_in_timer)
            sorted_blocks[0..count]
        else
            blocks[0..count];
        zeroAll(traversal);
    }
    const elapsed = monotonicNanos() - start;

    std.mem.doNotOptimizeAway(blocks[0..count]);
    freeAll(allocator, count, with_header);
    return elapsed;
}

fn allocateAll(
    allocator: std.mem.Allocator,
    with_header: bool,
    zero_interleaved: bool,
    count: *usize,
) !void {
    while (count.* < block_count) {
        var header: ?*Header = null;
        if (with_header) header = try allocator.create(Header);
        errdefer if (header) |value| allocator.destroy(value);

        const block = try allocator.alignedAlloc(u64, .@"64", words_per_block);
        if (zero_interleaved) {
            @memset(block, 0);
            std.mem.doNotOptimizeAway(block.ptr);
        }
        if (header) |value| {
            value.* = .{ .words = block[0..words_per_block], .cardinality = 0 };
            headers[count.*] = value;
        }
        blocks[count.*] = block;
        addresses[count.*] = @intFromPtr(block.ptr);
        count.* += 1;
    }
}

noinline fn zeroAll(memory: []const []align(64) u64) void {
    for (memory) |block| {
        @memset(block, 0);
        std.mem.doNotOptimizeAway(block.ptr);
    }
}

fn prepareSortedBlocks(count: usize) void {
    @memcpy(sorted_blocks[0..count], blocks[0..count]);
    std.mem.sort([]align(64) u64, sorted_blocks[0..count], {}, lessThanAddress);
}

fn freeAll(allocator: std.mem.Allocator, count: usize, with_header: bool) void {
    for (0..count) |index| {
        allocator.free(blocks[index]);
        if (with_header) allocator.destroy(headers[index]);
    }
}

fn validateCell(allocator: std.mem.Allocator, cell: Cell) !void {
    const with_header = cell.hasHeader();
    var count: usize = 0;
    defer freeAll(allocator, count, with_header);

    try allocateAll(allocator, with_header, false, &count);
    if (cell.zeroTimed()) zeroAll(blocks[0..count]);

    for (blocks[0..count], 0..) |block, index| {
        if (@intFromPtr(block.ptr) % 64 != 0) return error.InvalidAlignment;
        if (with_header) {
            if (headers[index].words != block.ptr or headers[index].cardinality != 0) {
                return error.InvalidHeader;
            }
        }
        if (cell.zeroTimed()) {
            for (block) |word| if (word != 0) return error.ZeroValidationFailed;
        }
    }
}

fn addressStatistics() AddressStats {
    var minimum = addresses[0];
    var maximum_end = addresses[0] + block_bytes;
    var adjacent_pairs: usize = 0;
    var monotonic_pairs: usize = 0;

    for (addresses, 0..) |address, index| {
        minimum = @min(minimum, address);
        maximum_end = @max(maximum_end, address + block_bytes);
        if (index == 0) continue;

        const stride = absDiff(address, addresses[index - 1]);
        stride_scratch[index - 1] = stride;
        if (stride == block_bytes) adjacent_pairs += 1;
        if (address > addresses[index - 1]) monotonic_pairs += 1;
    }

    std.mem.sort(usize, &stride_scratch, {}, std.sort.asc(usize));
    return .{
        .span = maximum_end - minimum,
        .median_stride = stride_scratch[stride_scratch.len / 2],
        .adjacent_pairs = adjacent_pairs,
        .monotonic_pairs = monotonic_pairs,
    };
}

fn absDiff(left: usize, right: usize) usize {
    return if (left >= right) left - right else right - left;
}

fn lessThanAddress(_: void, left: []align(64) u64, right: []align(64) u64) bool {
    return @intFromPtr(left.ptr) < @intFromPtr(right.ptr);
}

fn monotonicNanos() u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) @panic("clock_gettime failed");
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
