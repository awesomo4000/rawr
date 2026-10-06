// SPDX-License-Identifier: MPL-2.0
const std = @import("std");
const builtin = @import("builtin");
const clock = @import("bench_time.zig");

// ABI shared with the existing spec-36 counter. No bitmap code is imported.
const Snapshot = extern struct {
    primary: u64 = 0,
    major: u64 = 0,
    cow: u64 = 0,
    valid: u32 = 0,
    source: u32 = 0,
};
extern fn rawr_residency_fault_snapshot(*Snapshot) c_int;
extern fn rawr_residency_page_size() usize;
extern fn rawr_large_marker(c_int) void;
extern fn rawr_large_control_map() void;

pub const Kind = enum {
    smp,
    libc,
    page,
    pub fn allocator(self: Kind) std.mem.Allocator {
        return switch (self) {
            .smp => std.heap.smp_allocator,
            .libc => std.heap.c_allocator,
            .page => std.heap.page_allocator,
        };
    }
};
pub const Mode = enum { write, no_write, retained, serialize, to_array_alloc };
pub const Options = struct {
    kind: Kind,
    mode: Mode,
    bytes: usize,
    batch: usize,
    trace: bool = false,
};
pub const Info = struct { bytes: usize, alignment: usize, offset_min: usize, offset_max: usize };
const Sample = struct { ns: u64, minor: u64, major: u64 };

fn snapshot() !Snapshot {
    var s: Snapshot = .{};
    if (rawr_residency_fault_snapshot(&s) == 0 or s.valid == 0) return error.FaultCounterUnavailable;
    return s;
}

fn batch(comptime Backend: type, state: *Backend, count: usize) !Sample {
    const before = try snapshot();
    const start = clock.monotonicNanos();
    for (0..count) |_| try state.operation();
    const ns = clock.monotonicNanos() - start;
    const after = try snapshot();
    if (after.source != before.source or after.primary < before.primary or after.major < before.major) return error.InvalidFaultDelta;
    return .{ .ns = ns, .minor = after.primary - before.primary, .major = after.major - before.major };
}

pub fn main(comptime Backend: type, init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();
    const first = args.next() orelse return error.MissingArguments;
    if (std.mem.eql(u8, first, "--trace-control")) {
        rawr_large_marker(0);
        rawr_large_control_map();
        rawr_large_marker(1);
        rawr_large_control_map();
        rawr_large_marker(2);
        for (0..2) |_| rawr_large_control_map();
        rawr_large_marker(3);
        for (0..3) |_| rawr_large_control_map();
        rawr_large_marker(4);
        return;
    }
    var options: Options = .{
        .kind = std.meta.stringToEnum(Kind, first) orelse return error.BadAllocator,
        .mode = std.meta.stringToEnum(Mode, args.next() orelse return error.MissingMode) orelse return error.BadMode,
        .bytes = try std.fmt.parseInt(usize, args.next() orelse return error.MissingSize, 10),
        .batch = try std.fmt.parseInt(usize, args.next() orelse return error.MissingBatch, 10),
    };
    if (options.batch == 0 or options.batch > 1_048_576 or options.bytes == 0) return error.InvalidCount;
    if (args.next()) |arg| {
        if (!std.mem.eql(u8, arg, "--trace") or args.next() != null) return error.BadArgument;
        options.trace = true;
    }
    const page_size = rawr_residency_page_size();
    if (page_size == 0) return error.NoPageSize;
    const counter = try snapshot();
    // Prime only measurement infrastructure, never the workload or allocators.
    _ = clock.monotonicNanos();
    if (options.trace) rawr_large_marker(0);
    var state = try Backend.prepare(options, page_size);
    defer state.deinit();
    var warm: [3]Sample = undefined;
    var timed: [21]Sample = undefined;
    if (options.trace) rawr_large_marker(1);
    for (&warm) |*s| s.* = try batch(Backend, &state, options.batch);
    if (options.trace) rawr_large_marker(2);
    for (&timed) |*s| s.* = try batch(Backend, &state, options.batch);
    if (options.trace) rawr_large_marker(3);
    // Validation and production allocation observation must follow all timing.
    const info = try state.validateAndInfo();
    if (options.trace) rawr_large_marker(4);
    const slab = @max(std.heap.page_size_max, 65536);
    clock.print("META\t{s}\t{s}\t{s}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        builtin.zig_version_string, @tagName(builtin.cpu.arch), @tagName(builtin.os.tag),
        @tagName(builtin.mode),     page_size,                  slab,
        slab / 2,                   counter.source,             info.bytes,
        info.alignment,             info.offset_min,            info.offset_max,
    });
    for (warm, 0..) |s, i| clock.print("SAMPLE\twarmup\t{d}\t{d}\t{d}\t{d}\n", .{ i, s.ns, s.minor, s.major });
    var times: [21]u64 = undefined;
    for (timed, 0..) |s, i| {
        times[i] = s.ns;
        clock.print("SAMPLE\ttimed\t{d}\t{d}\t{d}\t{d}\n", .{ i, s.ns, s.minor, s.major });
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    clock.print("RESULT\t{s}\t{s}\t{d}\t{d}\t{d}\n", .{ @tagName(options.kind), @tagName(options.mode), info.bytes, options.batch, times[10] });
}
