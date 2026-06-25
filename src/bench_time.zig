const std = @import("std");
const builtin = @import("builtin");

pub fn monotonicNanos() u64 {
    if (builtin.os.tag == .windows) {
        return windowsPerformanceNanos();
    }
    if (builtin.os.tag == .openbsd) {
        return openbsdBenchMonotonicNanos();
    }
    return posixClockNanos(.MONOTONIC);
}

pub fn realtimeSeconds() u64 {
    if (builtin.os.tag == .windows) {
        return windowsRealtimeSeconds();
    }
    if (builtin.os.tag == .openbsd) {
        return openbsdBenchRealtimeSeconds();
    }
    return posixClockNanos(.REALTIME) / std.time.ns_per_s;
}

pub fn cAllocator() std.mem.Allocator {
    if (builtin.os.tag == .openbsd) {
        return openbsd_c_allocator;
    }
    return std.heap.c_allocator;
}

fn posixClockNanos(clock: std.c.CLOCK) u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(clock, &ts) != 0) {
        @panic("clock_gettime failed");
    }
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

extern fn rawr_bench_monotonic_ns() callconv(.c) u64;
extern fn rawr_bench_realtime_seconds() callconv(.c) u64;
extern fn rawr_bench_malloc(size: usize) callconv(.c) ?*anyopaque;
extern fn rawr_bench_aligned_alloc(alignment: usize, size: usize) callconv(.c) ?*anyopaque;
extern fn rawr_bench_free(ptr: ?*anyopaque) callconv(.c) void;

fn openbsdBenchMonotonicNanos() u64 {
    const ns = rawr_bench_monotonic_ns();
    if (ns == 0) @panic("OpenBSD benchmark timer failed");
    return ns;
}

fn openbsdBenchRealtimeSeconds() u64 {
    const seconds = rawr_bench_realtime_seconds();
    if (seconds == 0) @panic("OpenBSD realtime clock failed");
    return seconds;
}

pub const openbsd_c_allocator = std.mem.Allocator{
    .ptr = undefined,
    .vtable = &.{
        .alloc = openbsdCAlloc,
        .resize = openbsdCResize,
        .remap = openbsdCRemap,
        .free = openbsdCFree,
    },
};

fn openbsdCAlloc(_: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
    std.debug.assert(len > 0);
    if (alignment.toByteUnits() <= @alignOf(std.c.max_align_t)) {
        const actual_len = @max(len, @alignOf(std.c.max_align_t));
        return @ptrCast(rawr_bench_malloc(actual_len) orelse return null);
    }

    const effective_alignment = @max(alignment.toByteUnits(), @sizeOf(usize));
    return @ptrCast(rawr_bench_aligned_alloc(effective_alignment, len) orelse return null);
}

fn openbsdCResize(_: *anyopaque, memory: []u8, _: std.mem.Alignment, new_len: usize, _: usize) bool {
    std.debug.assert(new_len > 0);
    return new_len <= memory.len;
}

fn openbsdCRemap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
    return null;
}

fn openbsdCFree(_: *anyopaque, memory: []u8, alignment: std.mem.Alignment, _: usize) void {
    _ = alignment;
    rawr_bench_free(memory.ptr);
}

fn windowsPerformanceNanos() u64 {
    const windows = std.os.windows;

    var counter: windows.LARGE_INTEGER = undefined;
    var frequency: windows.LARGE_INTEGER = undefined;
    if (!windows.ntdll.RtlQueryPerformanceCounter(&counter).toBool()) {
        @panic("RtlQueryPerformanceCounter failed");
    }
    if (!windows.ntdll.RtlQueryPerformanceFrequency(&frequency).toBool()) {
        @panic("RtlQueryPerformanceFrequency failed");
    }

    const nanos = @divFloor(@as(i128, counter) * std.time.ns_per_s, @as(i128, frequency));
    return @intCast(nanos);
}

fn windowsRealtimeSeconds() u64 {
    const ticks_per_second = 10_000_000;
    const windows_to_unix_seconds = 11_644_473_600;

    const windows_ticks = std.os.windows.ntdll.RtlGetSystemTimePrecise();
    const seconds = @divFloor(windows_ticks, ticks_per_second) - windows_to_unix_seconds;
    return @intCast(seconds);
}
