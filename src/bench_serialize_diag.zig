// SPDX-License-Identifier: MPL-2.0

//! Fresh-process fixed-buffer serialization diagnosis for spec 28-00.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const ser = rawr.serialize_diag;
const allocator = std.heap.smp_allocator;
const value_count = 1_000_000;
const warmup_runs = 3;
const timed_runs = 21;
const seed = 12345;

const Cell = enum {
    temp_writer,
    direct_writer,
    temp_direct,
    direct_direct,
    croaring,

    fn name(self: Cell) []const u8 {
        return switch (self) {
            .temp_writer => "temp-writer",
            .direct_writer => "direct-writer",
            .temp_direct => "temp-direct",
            .direct_direct => "direct-direct",
            .croaring => "croaring",
        };
    }

    fn parse(text: []const u8) ?Cell {
        inline for (std.meta.fields(Cell)) |field| {
            const value: Cell = @enumFromInt(field.value);
            if (std.mem.eql(u8, text, value.name())) return value;
        }
        return null;
    }

    fn rawrVariant(self: Cell) ?ser.FixedSerializeVariant {
        return switch (self) {
            .temp_writer => .temp_writer,
            .direct_writer => .direct_writer,
            .temp_direct => .temp_direct,
            .direct_direct => .direct_direct,
            .croaring => null,
        };
    }
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var requested_cell: ?Cell = null;
    var header = false;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--cell=")) {
            requested_cell = Cell.parse(arg[7..]) orelse return error.UnknownCell;
        } else {
            return error.UnknownArgument;
        }
    }

    if (header) {
        if (requested_cell != null) return error.ConflictingArguments;
        bench_time.printBenchEnvironment();
        bench_time.print("# serialize diagnosis protocol: {d}w/{d}t median; seed={d}; values={d}\n", .{
            warmup_runs,
            timed_runs,
            seed,
            value_count,
        });
        return;
    }

    const cell = requested_cell orelse return error.MissingCell;
    const values = try allocator.alloc(u32, value_count);
    defer allocator.free(values);
    fillCanonicalValues(values);

    var bitmap = try RoaringBitmap.init(allocator);
    defer bitmap.deinit();
    try bitmap.addMany(values);

    const c_bitmap = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(c_bitmap);
    c.roaring_bitmap_add_many(c_bitmap, values.len, values.ptr);

    const median_ns = measure(cell, &bitmap, c_bitmap);
    try validateCell(cell, &bitmap, c_bitmap);
    const stats = if (cell.rawrVariant()) |variant|
        ser.fixedSerializeAllocationStats(&bitmap, variant)
    else
        ser.FixedSerializeAllocationStats{
            .output_allocations = 1,
            .output_bytes = c.roaring_bitmap_portable_size_in_bytes(c_bitmap),
            .temporary_allocations = 0,
            .temporary_bytes = 0,
        };

    bench_time.print("VALIDATION\t{s}\tbytes={d}\n", .{ cell.name(), stats.output_bytes });
    bench_time.print("ALLOC\t{s}\toutput_allocator={s}\toutput_allocs={d}\toutput_bytes={d}\ttemp_allocator={s}\ttemp_allocs={d}\ttemp_bytes={d}\n", .{
        cell.name(),
        if (cell == .croaring) "libc" else "smp",
        stats.output_allocations,
        stats.output_bytes,
        if (stats.temporary_allocations == 0) "none" else "smp",
        stats.temporary_allocations,
        stats.temporary_bytes,
    });
    bench_time.print("RESULT\t{s}\t{d}\n", .{ cell.name(), median_ns });
}

fn fillCanonicalValues(values: []u32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (values) |*value| {
        value.* = random.int(u32);
        _ = random.uintLessThan(u32, 500_000);
        _ = random.uintLessThan(u32, 500_000);
        _ = random.uintLessThan(u32, 50_000);
        _ = random.uintLessThan(u32, 1024);
        _ = random.uintLessThan(u32, 20_000);
        _ = random.uintLessThan(u32, 20_000);
    }
}

fn validateCell(cell: Cell, bitmap: *const RoaringBitmap, c_bitmap: *const c.roaring_bitmap_t) !void {
    const expected_len = bitmap.serializedSizeInBytes();
    const legacy = try allocator.alloc(u8, expected_len);
    defer allocator.free(legacy);
    var writer = std.Io.Writer.fixed(legacy);
    try bitmap.serializeToWriter(&writer);

    const c_len = c.roaring_bitmap_portable_size_in_bytes(c_bitmap);
    if (c_len != expected_len) return error.SizeMismatch;
    const c_bytes = try allocator.alloc(u8, c_len);
    defer allocator.free(c_bytes);
    if (c.roaring_bitmap_portable_serialize(c_bitmap, @ptrCast(c_bytes.ptr)) != c_len) {
        return error.CRoaringSerializeFailed;
    }
    if (!std.mem.eql(u8, legacy, c_bytes)) return error.CRoaringByteMismatch;

    if (cell.rawrVariant()) |variant| {
        const actual = try ser.serializeFixedDiagnostic(bitmap, allocator, variant);
        defer allocator.free(actual);
        if (!std.mem.eql(u8, legacy, actual)) return error.LegacyByteMismatch;
    }
}

fn measure(cell: Cell, bitmap: *const RoaringBitmap, c_bitmap: *const c.roaring_bitmap_t) u64 {
    for (0..warmup_runs) |_| runOnce(cell, bitmap, c_bitmap);
    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        runOnce(cell, bitmap, c_bitmap);
        elapsed.* = bench_time.monotonicNanos() - start;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

noinline fn runOnce(cell: Cell, bitmap: *const RoaringBitmap, c_bitmap: *const c.roaring_bitmap_t) void {
    if (cell.rawrVariant()) |variant| {
        const bytes = ser.serializeFixedDiagnostic(bitmap, allocator, variant) catch unreachable;
        defer allocator.free(bytes);
        std.mem.doNotOptimizeAway(bytes.ptr);
        return;
    }

    const libc_allocator = bench_time.cAllocator();
    const len = c.roaring_bitmap_portable_size_in_bytes(c_bitmap);
    const bytes = libc_allocator.alloc(u8, len) catch unreachable;
    defer libc_allocator.free(bytes);
    _ = c.roaring_bitmap_portable_serialize(c_bitmap, @ptrCast(bytes.ptr));
    std.mem.doNotOptimizeAway(bytes.ptr);
}
