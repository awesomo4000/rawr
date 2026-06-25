const std = @import("std");
const builtin = @import("builtin");

pub fn monotonicNanos() u64 {
    if (builtin.os.tag == .windows) {
        return windowsPerformanceNanos();
    }
    if (builtin.os.tag == .openbsd) {
        return openbsdWallNanos();
    }
    return posixClockNanos(.MONOTONIC);
}

pub fn realtimeSeconds() u64 {
    if (builtin.os.tag == .windows) {
        return windowsRealtimeSeconds();
    }
    if (builtin.os.tag == .openbsd) {
        return openbsdWallNanos() / std.time.ns_per_s;
    }
    return posixClockNanos(.REALTIME) / std.time.ns_per_s;
}

fn posixClockNanos(clock: std.c.CLOCK) u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(clock, &ts) != 0) {
        @panic("clock_gettime failed");
    }
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn openbsdWallNanos() u64 {
    var tv: std.c.timeval = undefined;
    if (std.c.gettimeofday(&tv, null) != 0) {
        @panic("gettimeofday failed");
    }
    return @as(u64, @intCast(tv.sec)) * std.time.ns_per_s + @as(u64, @intCast(tv.usec)) * std.time.ns_per_us;
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
