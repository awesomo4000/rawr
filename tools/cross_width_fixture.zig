// SPDX-License-Identifier: MPL-2.0

//! Deterministic serialization fixture producer and consumer.
//!
//! Usage:
//!   cross_width_fixture produce <file>
//!   cross_width_fixture verify <file>

const std = @import("std");
const rawr = @import("rawr");

const RoaringBitmap = rawr.RoaringBitmap;
const Roaring64Bitmap = rawr.Roaring64Bitmap;

const magic = "RAWRXW01";
const format_version: u32 = 1;
const expected_corpus_hash: u64 = 0x1e4d9768fabb6ac5;
const max_file_size = 64 * 1024 * 1024;
const max_values32 = 6000;
const max_values64 = 128;

const Case = enum(u8) {
    bitmap_empty,
    bitmap_single,
    bitmap_mixed,
    bitmap64_empty,
    bitmap64_single,
    bitmap64_multi,

    fn is64(self: Case) bool {
        return switch (self) {
            .bitmap_empty, .bitmap_single, .bitmap_mixed => false,
            .bitmap64_empty, .bitmap64_single, .bitmap64_multi => true,
        };
    }
};

const cases = [_]Case{
    .bitmap_empty,
    .bitmap_single,
    .bitmap_mixed,
    .bitmap64_empty,
    .bitmap64_single,
    .bitmap64_multi,
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();
    _ = args.skip();

    const command = args.next() orelse return error.MissingCommand;
    const path = args.next() orelse return error.MissingPath;
    if (args.next() != null) return error.TooManyArguments;

    if (std.mem.eql(u8, command, "produce")) {
        try produce(init.io, allocator, path);
    } else if (std.mem.eql(u8, command, "verify")) {
        try verify(init.io, allocator, path);
    } else {
        return error.UnknownCommand;
    }
}

fn produce(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !void {
    const corpus_hash = corpusHash();
    try assertCorpusHash(corpus_hash);

    var payloads: [cases.len][]u8 = undefined;
    var payload_count: usize = 0;
    errdefer for (payloads[0..payload_count]) |payload| allocator.free(payload);

    for (cases, 0..) |case, index| {
        payloads[index] = if (case.is64())
            try serialize64Case(allocator, case)
        else
            try serialize32Case(allocator, case);
        payload_count += 1;
    }
    defer for (payloads) |payload| allocator.free(payload);

    const header_size = magic.len + @sizeOf(u32) + @sizeOf(u64) + @sizeOf(u32);
    const record_header_size = @sizeOf(u8) + @sizeOf(u8) + @sizeOf(u16) + @sizeOf(u64);
    var total_size: usize = header_size;
    for (payloads) |payload| {
        total_size = try std.math.add(usize, total_size, record_header_size);
        total_size = try std.math.add(usize, total_size, payload.len);
    }

    const output = try allocator.alloc(u8, total_size);
    defer allocator.free(output);
    var offset: usize = 0;
    writeBytes(output, &offset, magic);
    writeInt(u32, output, &offset, format_version);
    writeInt(u64, output, &offset, corpus_hash);
    writeInt(u32, output, &offset, @intCast(cases.len));
    for (cases, payloads) |case, payload| {
        writeInt(u8, output, &offset, @intFromEnum(case));
        writeInt(u8, output, &offset, @intFromBool(case.is64()));
        writeInt(u16, output, &offset, 0);
        writeInt(u64, output, &offset, payload.len);
        writeBytes(output, &offset, payload);
    }
    std.debug.assert(offset == output.len);

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = output });
}

fn verify(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !void {
    const corpus_hash = corpusHash();
    try assertCorpusHash(corpus_hash);

    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size > max_file_size) return error.FixtureTooLarge;
    const data = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(data);
    if (try file.readPositionalAll(io, data, 0) != data.len) return error.UnexpectedEndOfFile;

    var offset: usize = 0;
    const found_magic = try readBytes(data, &offset, magic.len);
    if (!std.mem.eql(u8, found_magic, magic)) return error.BadMagic;
    if (try readInt(u32, data, &offset) != format_version) return error.BadVersion;
    if (try readInt(u64, data, &offset) != corpus_hash) return error.CorpusHashMismatch;
    if (try readInt(u32, data, &offset) != @as(u32, cases.len)) return error.BadCaseCount;

    for (cases) |expected_case| {
        const case = std.enums.fromInt(Case, try readInt(u8, data, &offset)) orelse
            return error.BadCase;
        if (case != expected_case) return error.BadCaseOrder;
        const is_64 = try readInt(u8, data, &offset);
        if (is_64 != @intFromBool(case.is64())) return error.BadCaseKind;
        if (try readInt(u16, data, &offset) != 0) return error.BadReservedField;
        const payload_len = try readInt(u64, data, &offset);
        if (payload_len > @as(u64, std.math.maxInt(usize))) return error.FixtureTooLarge;
        const payload = try readBytes(data, &offset, @intCast(payload_len));

        if (case.is64()) {
            try verify64Case(allocator, case, payload);
        } else {
            try verify32Case(allocator, case, payload);
        }
    }
    if (offset != data.len) return error.TrailingData;
}

fn serialize32Case(allocator: std.mem.Allocator, case: Case) ![]u8 {
    var bitmap = try build32Case(allocator, case);
    defer bitmap.deinit();
    return bitmap.serialize(allocator);
}

fn serialize64Case(allocator: std.mem.Allocator, case: Case) ![]u8 {
    var bitmap = try build64Case(allocator, case);
    defer bitmap.deinit();
    return bitmap.serialize(allocator);
}

fn verify32Case(allocator: std.mem.Allocator, case: Case, payload: []const u8) !void {
    var expected = try build32Case(allocator, case);
    defer expected.deinit();
    var actual = try RoaringBitmap.deserializeSafe(allocator, payload);
    defer actual.deinit();
    if (!actual.equals(&expected)) return error.SetMismatch;

    const round_trip = try actual.serialize(allocator);
    defer allocator.free(round_trip);
    if (!std.mem.eql(u8, round_trip, payload)) return error.SerializedBytesMismatch;
}

fn verify64Case(allocator: std.mem.Allocator, case: Case, payload: []const u8) !void {
    var expected = try build64Case(allocator, case);
    defer expected.deinit();
    var actual = try Roaring64Bitmap.deserializeSafe(allocator, payload);
    defer actual.deinit();
    if (!actual.equals(&expected)) return error.SetMismatch;

    const round_trip = try actual.serialize(allocator);
    defer allocator.free(round_trip);
    if (!std.mem.eql(u8, round_trip, payload)) return error.SerializedBytesMismatch;
}

fn build32Case(allocator: std.mem.Allocator, case: Case) !RoaringBitmap {
    var values: [max_values32]u32 = undefined;
    const generated = generate32(case, &values);
    var bitmap = try RoaringBitmap.init(allocator);
    errdefer bitmap.deinit();
    for (generated) |value| _ = try bitmap.add(value);

    if (case == .bitmap_mixed) {
        _ = try bitmap.runOptimize();
        var found_array = false;
        var found_bitset = false;
        var found_run = false;
        for (bitmap.containers[0..bitmap.size]) |tagged| switch (tagged.getType()) {
            .array => found_array = true,
            .bitset => found_bitset = true,
            .run => found_run = true,
            .reserved => return error.ReservedContainer,
        };
        if (!found_array or !found_bitset or !found_run) return error.MissingContainerType;
    }
    return bitmap;
}

fn build64Case(allocator: std.mem.Allocator, case: Case) !Roaring64Bitmap {
    var values: [max_values64]u64 = undefined;
    const generated = generate64(case, &values);
    var bitmap = try Roaring64Bitmap.init(allocator);
    errdefer bitmap.deinit();
    for (generated) |value| _ = try bitmap.add(value);
    return bitmap;
}

fn generate32(case: Case, out: *[max_values32]u32) []const u32 {
    var len: usize = 0;
    switch (case) {
        .bitmap_empty => {},
        .bitmap_single => {
            for ([_]u32{ 0, 7, 65_535 }) |value| append32(out, &len, value);
        },
        .bitmap_mixed => {
            for ([_]u32{ 0, 7, 65_535, std.math.maxInt(u32) }) |value| append32(out, &len, value);

            var present: [1 << 16]bool = @splat(false);
            var prng = std.Random.DefaultPrng.init(0x40_00_32_2026);
            var dense_count: usize = 0;
            while (dense_count < 5000) {
                const low: u16 = @truncate(prng.random().int(u32));
                if (present[low]) continue;
                present[low] = true;
                append32(out, &len, (@as(u32, 1) << 16) | low);
                dense_count += 1;
            }

            var low: u32 = 100;
            while (low <= 500) : (low += 1) {
                append32(out, &len, (@as(u32, 2) << 16) | low);
            }
        },
        else => unreachable,
    }
    std.mem.sort(u32, out[0..len], {}, std.sort.asc(u32));
    return out[0..len];
}

fn generate64(case: Case, out: *[max_values64]u64) []const u64 {
    var len: usize = 0;
    switch (case) {
        .bitmap64_empty => {},
        .bitmap64_single => {
            const high: u32 = 7;
            for ([_]u32{ 0, 11, std.math.maxInt(u32) }) |low| {
                append64(out, &len, join64(high, low));
            }
        },
        .bitmap64_multi => {
            const highs = [_]u32{ 0, 1, 0x1234_5678, std.math.maxInt(u32) };
            var prng = std.Random.DefaultPrng.init(0x40_00_64_2026);
            for (highs) |high| {
                var per_bucket: usize = 0;
                while (per_bucket < 20) {
                    const low = prng.random().int(u32);
                    const value = join64(high, low);
                    if (contains64(out[0..len], value)) continue;
                    append64(out, &len, value);
                    per_bucket += 1;
                }
                append64(out, &len, join64(high, 0));
                append64(out, &len, join64(high, std.math.maxInt(u32)));
            }
        },
        else => unreachable,
    }
    std.mem.sort(u64, out[0..len], {}, std.sort.asc(u64));
    return out[0..len];
}

fn corpusHash() u64 {
    var hash: u64 = 0xcbf29ce484222325;
    for (cases) |case| {
        hash = fnv1a(hash, &.{@intFromEnum(case)});
        if (case.is64()) {
            var values: [max_values64]u64 = undefined;
            const generated = generate64(case, &values);
            hash = hashInt(u32, hash, @intCast(generated.len));
            for (generated) |value| hash = hashInt(u64, hash, value);
        } else {
            var values: [max_values32]u32 = undefined;
            const generated = generate32(case, &values);
            hash = hashInt(u32, hash, @intCast(generated.len));
            for (generated) |value| hash = hashInt(u32, hash, value);
        }
    }
    return hash;
}

fn assertCorpusHash(actual: u64) !void {
    if (actual == expected_corpus_hash) return;
    std.debug.print("cross-width fixture: corpus hash mismatch expected=0x{x} actual=0x{x}\n", .{
        expected_corpus_hash,
        actual,
    });
    return error.UnexpectedCorpusHash;
}

fn append32(out: *[max_values32]u32, len: *usize, value: u32) void {
    out[len.*] = value;
    len.* += 1;
}

fn append64(out: *[max_values64]u64, len: *usize, value: u64) void {
    out[len.*] = value;
    len.* += 1;
}

fn contains64(values: []const u64, target: u64) bool {
    for (values) |value| if (value == target) return true;
    return false;
}

fn join64(high: u32, low: u32) u64 {
    return (@as(u64, high) << 32) | low;
}

fn hashInt(comptime T: type, initial: u64, value: T) u64 {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, value, .little);
    return fnv1a(initial, &bytes);
}

fn fnv1a(initial: u64, bytes: []const u8) u64 {
    var hash = initial;
    for (bytes) |byte| {
        hash ^= byte;
        hash *%= 0x100000001b3;
    }
    return hash;
}

fn writeBytes(output: []u8, offset: *usize, bytes: []const u8) void {
    @memcpy(output[offset.*..][0..bytes.len], bytes);
    offset.* += bytes.len;
}

fn writeInt(comptime T: type, output: []u8, offset: *usize, value: T) void {
    std.mem.writeInt(T, output[offset.*..][0..@sizeOf(T)], value, .little);
    offset.* += @sizeOf(T);
}

fn readBytes(data: []const u8, offset: *usize, len: usize) ![]const u8 {
    const end = std.math.add(usize, offset.*, len) catch return error.UnexpectedEndOfFile;
    if (end > data.len) return error.UnexpectedEndOfFile;
    defer offset.* = end;
    return data[offset.*..end];
}

fn readInt(comptime T: type, data: []const u8, offset: *usize) !T {
    const bytes = try readBytes(data, offset, @sizeOf(T));
    return std.mem.readInt(T, @ptrCast(bytes.ptr), .little);
}
