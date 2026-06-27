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
        return 0;
    }
    return posixClockNanos(.REALTIME) / std.time.ns_per_s;
}

pub fn print(comptime fmt: []const u8, args: anytype) void {
    if (builtin.os.tag == .openbsd) {
        printOpenBsd(fmt, args);
        return;
    }

    std.debug.print(fmt, args);
}

pub fn printRunTimestamp() void {
    const ts = realtimeSeconds();
    if (ts == 0) {
        print("Run: timestamp unavailable on OpenBSD\n", .{});
        return;
    }

    const epoch_seconds = std.time.epoch.EpochSeconds{ .secs = @intCast(ts) };
    const day_seconds = epoch_seconds.getDaySeconds();
    const year_day = epoch_seconds.getEpochDay().calculateYearDay();
    const month_day = year_day.calculateMonthDay();

    print("Run: {d}-{d:0>2}-{d:0>2} {d:0>2}:{d:0>2}:{d:0>2} UTC\n", .{
        year_day.year,
        @intFromEnum(month_day.month),
        month_day.day_index + 1,
        day_seconds.getHoursIntoDay(),
        day_seconds.getMinutesIntoHour(),
        day_seconds.getSecondsIntoMinute(),
    });
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
extern fn rawr_bench_malloc(size: usize) callconv(.c) ?*anyopaque;
extern fn rawr_bench_aligned_alloc(alignment: usize, size: usize) callconv(.c) ?*anyopaque;
extern fn rawr_bench_free(ptr: ?*anyopaque) callconv(.c) void;
extern fn rawr_bench_write_stderr(ptr: [*]const u8, len: usize) callconv(.c) void;

const FormatKind = enum { auto, string, decimal, hex, optional_decimal };
const FormatAlign = enum { none, left, right };

const FormatSpec = struct {
    kind: FormatKind = .auto,
    justify: FormatAlign = .none,
    fill: u8 = ' ',
    width: usize = 0,
    precision: ?usize = null,
};

fn printOpenBsd(comptime fmt: []const u8, args: anytype) void {
    const fields = @typeInfo(@TypeOf(args)).@"struct".fields;

    comptime var arg_index: usize = 0;
    comptime var literal_start: usize = 0;
    comptime var i: usize = 0;
    inline while (i < fmt.len) {
        if (fmt[i] == '{') {
            if (i + 1 < fmt.len and fmt[i + 1] == '{') {
                writeOpenBsdLiteral(fmt[literal_start..i]);
                writeOpenBsdLiteral("{");
                i += 2;
                literal_start = i;
                continue;
            }

            writeOpenBsdLiteral(fmt[literal_start..i]);
            comptime var end = i + 1;
            inline while (end < fmt.len and fmt[end] != '}') : (end += 1) {}
            if (end >= fmt.len) @compileError("unterminated benchmark format placeholder");
            if (arg_index >= fields.len) @compileError("not enough benchmark format arguments");

            const field = fields[arg_index];
            writeOpenBsdValue(@field(args, field.name), parseFormatSpec(fmt[i + 1 .. end]));
            arg_index += 1;
            i = end + 1;
            literal_start = i;
        } else if (fmt[i] == '}') {
            if (i + 1 < fmt.len and fmt[i + 1] == '}') {
                writeOpenBsdLiteral(fmt[literal_start..i]);
                writeOpenBsdLiteral("}");
                i += 2;
                literal_start = i;
                continue;
            }
            @compileError("unmatched benchmark format closing brace");
        } else {
            i += 1;
        }
    }

    if (arg_index != fields.len) @compileError("too many benchmark format arguments");
    writeOpenBsdLiteral(fmt[literal_start..fmt.len]);
}

fn parseFormatSpec(comptime text: []const u8) FormatSpec {
    comptime var spec = FormatSpec{};
    comptime var i: usize = 0;

    if (text.len >= 2 and text[0] == '?' and text[1] == 'd') {
        spec.kind = .optional_decimal;
        i = 2;
    } else if (text.len > 0) {
        spec.kind = switch (text[0]) {
            's' => .string,
            'd' => .decimal,
            'x' => .hex,
            else => .auto,
        };
        if (spec.kind != .auto) i = 1;
    }

    if (i < text.len and text[i] == ':') {
        i += 1;
        if (i + 1 < text.len and (text[i + 1] == '<' or text[i + 1] == '>')) {
            spec.fill = text[i];
            spec.justify = if (text[i + 1] == '<') .left else .right;
            i += 2;
        } else if (i < text.len and (text[i] == '<' or text[i] == '>')) {
            spec.justify = if (text[i] == '<') .left else .right;
            i += 1;
        }

        inline while (i < text.len and text[i] >= '0' and text[i] <= '9') : (i += 1) {
            spec.width = spec.width * 10 + (text[i] - '0');
        }

        if (i < text.len and text[i] == '.') {
            i += 1;
            comptime var precision: usize = 0;
            inline while (i < text.len and text[i] >= '0' and text[i] <= '9') : (i += 1) {
                precision = precision * 10 + (text[i] - '0');
            }
            spec.precision = precision;
        }
    }

    if (i != text.len) @compileError("unsupported benchmark format spec: " ++ text);
    return spec;
}

fn writeOpenBsdLiteral(comptime bytes: []const u8) void {
    if (bytes.len != 0) rawr_bench_write_stderr(bytes.ptr, bytes.len);
}

fn writeOpenBsdBytes(bytes: []const u8) void {
    if (bytes.len != 0) rawr_bench_write_stderr(bytes.ptr, bytes.len);
}

fn writeOpenBsdRepeat(byte: u8, count: usize) void {
    if (count == 0) return;
    var buf: [64]u8 = undefined;
    @memset(&buf, byte);
    var remaining = count;
    while (remaining > 0) {
        const n = @min(remaining, buf.len);
        writeOpenBsdBytes(buf[0..n]);
        remaining -= n;
    }
}

fn writeOpenBsdPadded(bytes: []const u8, spec: FormatSpec) void {
    const padding = if (spec.width > bytes.len) spec.width - bytes.len else 0;
    if (spec.justify == .right) writeOpenBsdRepeat(spec.fill, padding);
    writeOpenBsdBytes(bytes);
    if (spec.justify == .left) writeOpenBsdRepeat(spec.fill, padding);
}

fn writeOpenBsdValue(value: anytype, spec: FormatSpec) void {
    const T = @TypeOf(value);
    switch (@typeInfo(T)) {
        .bool => writeOpenBsdPadded(if (value) "true" else "false", spec),
        .int, .comptime_int => {
            if (spec.kind == .hex) {
                writeOpenBsdHex(value, spec);
            } else {
                writeOpenBsdDecimal(value, spec);
            }
        },
        .float, .comptime_float => writeOpenBsdFloat(value, spec),
        .optional => {
            if (value) |inner| {
                var inner_spec = spec;
                if (inner_spec.kind == .optional_decimal) inner_spec.kind = .decimal;
                writeOpenBsdValue(inner, inner_spec);
            } else {
                writeOpenBsdPadded("null", spec);
            }
        },
        .pointer => |ptr| switch (ptr.size) {
            .slice => {
                if (ptr.child != u8) @compileError("unsupported benchmark slice format type: " ++ @typeName(T));
                writeOpenBsdPadded(value, spec);
            },
            .one => switch (@typeInfo(ptr.child)) {
                .array => |array| {
                    if (array.child != u8) @compileError("unsupported benchmark pointer format type: " ++ @typeName(T));
                    writeOpenBsdPadded(value[0..], spec);
                },
                else => @compileError("unsupported benchmark pointer format type: " ++ @typeName(T)),
            },
            else => @compileError("unsupported benchmark pointer format type: " ++ @typeName(T)),
        },
        .array => |array| {
            if (array.child != u8) @compileError("unsupported benchmark array format type: " ++ @typeName(T));
            writeOpenBsdPadded(value[0..], spec);
        },
        .@"enum" => writeOpenBsdPadded(@tagName(value), spec),
        else => @compileError("unsupported benchmark format type: " ++ @typeName(T)),
    }
}

fn writeOpenBsdDecimal(value: anytype, spec: FormatSpec) void {
    var buf: [64]u8 = undefined;
    var len: usize = 0;
    const T = @TypeOf(value);
    const negative = switch (@typeInfo(T)) {
        .int => |info| info.signedness == .signed and value < 0,
        .comptime_int => value < 0,
        else => false,
    };

    var magnitude: u128 = switch (@typeInfo(T)) {
        .int => |info| if (info.signedness == .signed and value < 0) magnitude: {
            const signed: i128 = @intCast(value);
            break :magnitude @as(u128, @intCast(-(signed + 1))) + 1;
        } else @as(u128, @intCast(value)),
        .comptime_int => if (value < 0) @as(u128, @intCast(-(value + 1))) + 1 else @as(u128, @intCast(value)),
        else => unreachable,
    };
    if (magnitude == 0) {
        buf[len] = '0';
        len += 1;
    } else {
        while (magnitude != 0) {
            buf[len] = '0' + @as(u8, @intCast(magnitude % 10));
            len += 1;
            magnitude /= 10;
        }
    }
    if (negative) {
        buf[len] = '-';
        len += 1;
    }
    reverse(buf[0..len]);
    writeOpenBsdPadded(buf[0..len], spec);
}

fn writeOpenBsdHex(value: anytype, spec: FormatSpec) void {
    var buf: [64]u8 = undefined;
    var len: usize = 0;
    var magnitude: u128 = @intCast(value);
    if (magnitude == 0) {
        buf[len] = '0';
        len += 1;
    } else {
        while (magnitude != 0) {
            const digit: u8 = @intCast(magnitude & 0xf);
            buf[len] = if (digit < 10) '0' + digit else 'a' + (digit - 10);
            len += 1;
            magnitude >>= 4;
        }
    }
    reverse(buf[0..len]);
    writeOpenBsdPadded(buf[0..len], spec);
}

fn writeOpenBsdFloat(value: anytype, spec: FormatSpec) void {
    if (std.math.isNan(value)) {
        writeOpenBsdPadded("nan", spec);
        return;
    }
    if (std.math.isInf(value)) {
        writeOpenBsdPadded(if (value < 0) "-inf" else "inf", spec);
        return;
    }

    const precision = spec.precision orelse 6;
    var scale: u128 = 1;
    for (0..precision) |_| scale *= 10;

    const negative = value < 0;
    const abs_value = if (negative) -value else value;
    const scaled_float = abs_value * @as(f64, @floatFromInt(scale)) + 0.5;
    const scaled: u128 = @intFromFloat(scaled_float);
    const whole = scaled / scale;
    const frac = scaled % scale;

    var buf: [128]u8 = undefined;
    var len: usize = 0;
    if (negative) {
        buf[len] = '-';
        len += 1;
    }
    len += decimalInto(buf[len..], whole);
    if (precision != 0) {
        buf[len] = '.';
        len += 1;
        const frac_start = len;
        len += decimalInto(buf[len..], frac);
        const frac_len = len - frac_start;
        if (frac_len < precision) {
            const zeros = precision - frac_len;
            std.mem.copyBackwards(u8, buf[frac_start + zeros .. len + zeros], buf[frac_start..len]);
            @memset(buf[frac_start .. frac_start + zeros], '0');
            len += zeros;
        }
    }

    writeOpenBsdPadded(buf[0..len], spec);
}

fn decimalInto(out: []u8, value: u128) usize {
    var tmp: [40]u8 = undefined;
    var len: usize = 0;
    var n = value;
    if (n == 0) {
        out[0] = '0';
        return 1;
    }
    while (n != 0) {
        tmp[len] = '0' + @as(u8, @intCast(n % 10));
        len += 1;
        n /= 10;
    }
    for (0..len) |i| out[i] = tmp[len - 1 - i];
    return len;
}

fn reverse(bytes: []u8) void {
    var left: usize = 0;
    var right = bytes.len;
    while (left < right) {
        right -= 1;
        const tmp = bytes[left];
        bytes[left] = bytes[right];
        bytes[right] = tmp;
        left += 1;
    }
}

fn openbsdBenchMonotonicNanos() u64 {
    const ns = rawr_bench_monotonic_ns();
    if (ns == 0) @panic("OpenBSD benchmark timer failed");
    return ns;
}

var openbsd_c_allocator_state: u8 = 0;
const openbsd_c_allocator_vtable = std.mem.Allocator.VTable{
    .alloc = openbsdCAlloc,
    .resize = openbsdCResize,
    .remap = openbsdCRemap,
    .free = openbsdCFree,
};
pub const openbsd_c_allocator = std.mem.Allocator{
    .ptr = &openbsd_c_allocator_state,
    .vtable = &openbsd_c_allocator_vtable,
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
