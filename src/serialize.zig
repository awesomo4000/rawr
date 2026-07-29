// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const fmt = @import("format.zig");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
const Container = @import("container.zig").Container;
const TaggedPtr = @import("container.zig").TaggedPtr;
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const RunContainer = @import("run_container.zig").RunContainer;

const MAX_CONTAINER_COUNT = 65_536;

// Serialization format is little-endian; bulk I/O requires matching host endianness
comptime {
    if (@import("builtin").cpu.arch.endian() != .little) {
        @compileError("rawr serialization assumes little-endian byte order");
    }
}

/// Returns true if any container is a run container.
fn hasRunContainers(bm: *const RoaringBitmap) bool {
    for (bm.containers[0..bm.size]) |tp| {
        if (TaggedPtr.getType(tp) == .run) return true;
    }
    return false;
}

/// Compute serialized size in bytes.
pub fn serializedSizeInBytes(bm: *const RoaringBitmap) usize {
    if (bm.size == 0) return 8; // Just header

    const has_runs = hasRunContainers(bm);
    var size: usize = 0;

    // Cookie + size (or cookie with embedded size for run format)
    if (has_runs) {
        size += 4; // cookie with size embedded
        // Run container bitset: ceil(size / 8) bytes
        size += (bm.size + 7) / 8;
    } else {
        size += 8; // cookie + size
    }

    // Descriptive header: 4 bytes per container (key + cardinality-1)
    size += @as(usize, bm.size) * 4;

    // Offset header:
    // - Always for no-run format (RoaringFormatSpec requirement)
    // - For run format only when size >= NO_OFFSET_THRESHOLD
    if (!has_runs or bm.size >= fmt.NO_OFFSET_THRESHOLD) {
        size += @as(usize, bm.size) * 4; // 4 bytes per container offset
    }

    // Container data
    for (bm.containers[0..bm.size]) |tp| {
        const container = Container.fromTagged(tp);
        size += switch (container) {
            .array => |ac| @as(usize, ac.cardinality) * 2,
            .bitset => BitsetContainer.SIZE_BYTES,
            .run => |rc| 2 + @as(usize, rc.n_runs) * 4, // n_runs prefix + pairs
            .reserved => 0,
        };
    }

    return size;
}

/// Compute the exact byte length of the leading 32-bit portable bitmap in `data`.
pub fn portableSizeInBytes(data: []const u8) !usize {
    if (data.len < 4) return error.InvalidFormat;

    const cookie = std.mem.readInt(u32, data[0..4], .little);
    var offset: usize = 4;
    var has_runs = false;
    var run_bitset_start: usize = 0;
    var run_bitset_len: usize = 0;
    var size: u32 = undefined;

    if ((cookie & 0xFFFF) == fmt.SERIAL_COOKIE) {
        has_runs = true;
        size = ((cookie >> 16) & 0xFFFF) + 1;
        run_bitset_start = offset;
        run_bitset_len = (@as(usize, size) + 7) / 8;
        offset = try checkedAdvance(data, offset, run_bitset_len);
    } else if (cookie == fmt.SERIAL_COOKIE_NO_RUNCONTAINER) {
        try ensureAvailable(data, offset, 4);
        size = std.mem.readInt(u32, data[offset..][0..4], .little);
        if (size > MAX_CONTAINER_COUNT) return error.InvalidFormat;
        offset += 4;
    } else {
        return error.InvalidFormat;
    }

    if (size > MAX_CONTAINER_COUNT) return error.InvalidFormat;

    const container_count: usize = @intCast(size);
    const desc_start = offset;
    offset = try checkedAdvance(data, offset, try checkedMul(container_count, 4));

    if (!has_runs or size >= fmt.NO_OFFSET_THRESHOLD) {
        offset = try checkedAdvance(data, offset, try checkedMul(container_count, 4));
    }

    for (0..container_count) |i| {
        const desc_offset = desc_start + i * 4;
        const card = @as(u32, std.mem.readInt(u16, data[desc_offset + 2 ..][0..2], .little)) + 1;
        const is_run = has_runs and runContainerBit(data[run_bitset_start .. run_bitset_start + run_bitset_len], i);

        const container_size = if (is_run) blk: {
            try ensureAvailable(data, offset, 2);
            const n_runs = std.mem.readInt(u16, data[offset..][0..2], .little);
            break :blk try checkedAdd(2, try checkedMul(@intCast(n_runs), 4));
        } else if (card > ArrayContainer.MAX_CARDINALITY)
            BitsetContainer.SIZE_BYTES
        else
            try checkedMul(@intCast(card), 2);

        offset = try checkedAdvance(data, offset, container_size);
    }

    return offset;
}

fn runContainerBit(bits: []const u8, index: usize) bool {
    return (bits[index / 8] & (@as(u8, 1) << @intCast(index % 8))) != 0;
}

fn checkedAdd(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidFormat;
}

fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidFormat;
}

fn checkedAdvance(data: []const u8, offset: usize, len: usize) !usize {
    const next = try checkedAdd(offset, len);
    if (next > data.len) return error.InvalidFormat;
    return next;
}

fn ensureAvailable(data: []const u8, offset: usize, len: usize) !void {
    _ = try checkedAdvance(data, offset, len);
}

/// Serialize the bitmap to a byte slice (RoaringFormatSpec compatible).
pub fn serialize(bm: *const RoaringBitmap, allocator: std.mem.Allocator) ![]u8 {
    return serializeFixedDirect(bm, allocator, false);
}

fn serializeLegacy(bm: *const RoaringBitmap, allocator: std.mem.Allocator) ![]u8 {
    const size_bytes = serializedSizeInBytes(bm);
    const buf = try allocator.alloc(u8, size_bytes);
    errdefer allocator.free(buf);

    var writer = std.Io.Writer.fixed(buf);

    try serializeToWriter(bm, &writer);

    return buf;
}

/// Repository-only fixed-buffer variants used by serialization diagnostics.
pub const FixedSerializeVariant = enum {
    temp_writer,
    direct_writer,
    temp_direct,
    direct_direct,
};

pub const FixedSerializeAllocationStats = struct {
    output_allocations: u8,
    output_bytes: usize,
    temporary_allocations: u8,
    temporary_bytes: usize,
};

pub fn fixedSerializeAllocationStats(bm: *const RoaringBitmap, variant: FixedSerializeVariant) FixedSerializeAllocationStats {
    var temporary_allocations: u8 = 0;
    var temporary_bytes: usize = 0;
    if (bm.size != 0 and (variant == .temp_writer or variant == .temp_direct)) {
        temporary_allocations = 1;
        temporary_bytes = @as(usize, bm.size) * 4;
        const has_runs = hasRunContainers(bm);
        if (!has_runs or bm.size >= fmt.NO_OFFSET_THRESHOLD) {
            temporary_allocations += 1;
            temporary_bytes += @as(usize, bm.size) * 4;
        }
    }
    return .{
        .output_allocations = 1,
        .output_bytes = serializedSizeInBytes(bm),
        .temporary_allocations = temporary_allocations,
        .temporary_bytes = temporary_bytes,
    };
}

/// Serialize using one of the fixed-buffer diagnostic cells. The current cell
/// deliberately calls the production implementation so it remains the exact baseline.
pub fn serializeFixedDiagnostic(
    bm: *const RoaringBitmap,
    allocator: std.mem.Allocator,
    variant: FixedSerializeVariant,
) ![]u8 {
    return switch (variant) {
        .temp_writer => serializeLegacy(bm, allocator),
        .direct_writer => serializeFixedWriterDirectConstruction(bm, allocator),
        .temp_direct => serializeFixedDirect(bm, allocator, true),
        .direct_direct => serializeFixedDirect(bm, allocator, false),
    };
}

fn serializeFixedWriterDirectConstruction(bm: *const RoaringBitmap, allocator: std.mem.Allocator) ![]u8 {
    const buf = try allocator.alloc(u8, serializedSizeInBytes(bm));
    errdefer allocator.free(buf);
    var writer = std.Io.Writer.fixed(buf);

    if (bm.size == 0) {
        try writer.writeInt(u32, fmt.SERIAL_COOKIE_NO_RUNCONTAINER, .little);
        try writer.writeInt(u32, 0, .little);
        return buf;
    }

    const has_runs = hasRunContainers(bm);
    if (has_runs) {
        const cookie: u32 = fmt.SERIAL_COOKIE | (@as(u32, bm.size - 1) << 16);
        try writer.writeInt(u32, cookie, .little);
        const bitset_bytes = (bm.size + 7) / 8;
        var run_bitset_buf: [8192]u8 = undefined;
        const run_bitset = run_bitset_buf[0..bitset_bytes];
        @memset(run_bitset, 0);
        for (bm.containers[0..bm.size], 0..) |tp, i| {
            if (TaggedPtr.getType(tp) == .run) {
                run_bitset[i / 8] |= @as(u8, 1) << @intCast(i % 8);
            }
        }
        try writer.writeAll(run_bitset);
    } else {
        try writer.writeInt(u32, fmt.SERIAL_COOKIE_NO_RUNCONTAINER, .little);
        try writer.writeInt(u32, bm.size, .little);
    }

    for (bm.containers[0..bm.size], bm.keys[0..bm.size]) |tp, key| {
        try writer.writeInt(u16, key, .little);
        try writer.writeInt(u16, @intCast(Container.fromTagged(tp).getCardinality() - 1), .little);
    }

    if (!has_runs or bm.size >= fmt.NO_OFFSET_THRESHOLD) {
        var offset = fixedContainerDataStart(bm.size, has_runs);
        for (bm.containers[0..bm.size]) |tp| {
            try writer.writeInt(u32, offset, .little);
            offset += containerSerializedSize(Container.fromTagged(tp));
        }
    }

    try writeContainersToWriter(bm, &writer);
    return buf;
}

const FixedCursor = struct {
    buf: []u8,
    pos: usize = 0,

    fn writeInt(self: *FixedCursor, comptime T: type, value: T) void {
        const end = self.pos + @sizeOf(T);
        std.debug.assert(end <= self.buf.len);
        const bytes: *[@sizeOf(T)]u8 = @ptrCast(self.buf[self.pos..end].ptr);
        std.mem.writeInt(T, bytes, value, .little);
        self.pos = end;
    }

    fn writeAll(self: *FixedCursor, bytes: []const u8) void {
        const end = self.pos + bytes.len;
        std.debug.assert(end <= self.buf.len);
        @memcpy(self.buf[self.pos..end], bytes);
        self.pos = end;
    }

    fn reserve(self: *FixedCursor, len: usize) []u8 {
        const end = self.pos + len;
        std.debug.assert(end <= self.buf.len);
        const result = self.buf[self.pos..end];
        self.pos = end;
        return result;
    }
};

fn serializeFixedDirect(bm: *const RoaringBitmap, allocator: std.mem.Allocator, comptime temporary_tables: bool) ![]u8 {
    const buf = try allocator.alloc(u8, serializedSizeInBytes(bm));
    errdefer allocator.free(buf);
    var cursor = FixedCursor{ .buf = buf };

    if (bm.size == 0) {
        cursor.writeInt(u32, fmt.SERIAL_COOKIE_NO_RUNCONTAINER);
        cursor.writeInt(u32, 0);
        std.debug.assert(cursor.pos == buf.len);
        return buf;
    }

    const has_runs = hasRunContainers(bm);
    if (has_runs) {
        cursor.writeInt(u32, fmt.SERIAL_COOKIE | (@as(u32, bm.size - 1) << 16));
        const run_bitset = cursor.reserve((bm.size + 7) / 8);
        @memset(run_bitset, 0);
        for (bm.containers[0..bm.size], 0..) |tp, i| {
            if (TaggedPtr.getType(tp) == .run) {
                run_bitset[i / 8] |= @as(u8, 1) << @intCast(i % 8);
            }
        }
    } else {
        cursor.writeInt(u32, fmt.SERIAL_COOKIE_NO_RUNCONTAINER);
        cursor.writeInt(u32, bm.size);
    }

    if (temporary_tables) {
        const desc_buf = try bm.allocator.alloc(u16, bm.size * 2);
        defer bm.allocator.free(desc_buf);
        for (bm.containers[0..bm.size], bm.keys[0..bm.size], 0..) |tp, key, i| {
            desc_buf[i * 2] = key;
            desc_buf[i * 2 + 1] = @intCast(Container.fromTagged(tp).getCardinality() - 1);
        }
        cursor.writeAll(std.mem.sliceAsBytes(desc_buf));
    } else {
        for (bm.containers[0..bm.size], bm.keys[0..bm.size]) |tp, key| {
            cursor.writeInt(u16, key);
            cursor.writeInt(u16, @intCast(Container.fromTagged(tp).getCardinality() - 1));
        }
    }

    if (!has_runs or bm.size >= fmt.NO_OFFSET_THRESHOLD) {
        var offset = fixedContainerDataStart(bm.size, has_runs);
        if (temporary_tables) {
            const offset_buf = try bm.allocator.alloc(u32, bm.size);
            defer bm.allocator.free(offset_buf);
            for (bm.containers[0..bm.size], 0..) |tp, i| {
                offset_buf[i] = offset;
                offset += containerSerializedSize(Container.fromTagged(tp));
            }
            cursor.writeAll(std.mem.sliceAsBytes(offset_buf));
        } else {
            for (bm.containers[0..bm.size]) |tp| {
                cursor.writeInt(u32, offset);
                offset += containerSerializedSize(Container.fromTagged(tp));
            }
        }
    }

    for (bm.containers[0..bm.size]) |tp| {
        const container = Container.fromTagged(tp);
        switch (container) {
            .array => |ac| cursor.writeAll(std.mem.sliceAsBytes(ac.values[0..ac.cardinality])),
            .bitset => |bc| cursor.writeAll(std.mem.sliceAsBytes(bc.words)),
            .run => |rc| {
                cursor.writeInt(u16, rc.n_runs);
                cursor.writeAll(std.mem.sliceAsBytes(rc.runs[0..rc.n_runs]));
            },
            .reserved => unreachable,
        }
    }
    std.debug.assert(cursor.pos == buf.len);
    return buf;
}

fn fixedContainerDataStart(size: u32, has_runs: bool) u32 {
    if (has_runs) {
        return 4 + (size + 7) / 8 + size * 4 + size * 4;
    }
    return 8 + size * 4 + size * 4;
}

fn containerSerializedSize(container: Container) u32 {
    return switch (container) {
        .array => |ac| @as(u32, ac.cardinality) * 2,
        .bitset => BitsetContainer.SIZE_BYTES,
        .run => |rc| 2 + @as(u32, rc.n_runs) * 4,
        .reserved => unreachable,
    };
}

fn writeContainersToWriter(bm: *const RoaringBitmap, writer: anytype) !void {
    for (bm.containers[0..bm.size]) |tp| {
        switch (Container.fromTagged(tp)) {
            .array => |ac| try writer.writeAll(std.mem.sliceAsBytes(ac.values[0..ac.cardinality])),
            .bitset => |bc| try writer.writeAll(std.mem.sliceAsBytes(bc.words)),
            .run => |rc| {
                try writer.writeInt(u16, rc.n_runs, .little);
                try writer.writeAll(std.mem.sliceAsBytes(rc.runs[0..rc.n_runs]));
            },
            .reserved => unreachable,
        }
    }
}

/// Serialize to any writer.
pub fn serializeToWriter(bm: *const RoaringBitmap, writer: anytype) !void {
    if (bm.size == 0) {
        // Empty bitmap
        try writer.writeInt(u32, fmt.SERIAL_COOKIE_NO_RUNCONTAINER, .little);
        try writer.writeInt(u32, 0, .little);
        return;
    }

    const has_runs = hasRunContainers(bm);

    if (has_runs) {
        // Cookie with size embedded in high 16 bits
        const cookie: u32 = fmt.SERIAL_COOKIE | (@as(u32, bm.size - 1) << 16);
        try writer.writeInt(u32, cookie, .little);

        // Run container bitset (max 8KB for 65536 containers)
        const bitset_bytes = (bm.size + 7) / 8;
        var run_bitset_buf: [8192]u8 = undefined;
        const run_bitset = run_bitset_buf[0..bitset_bytes];
        @memset(run_bitset, 0);

        for (bm.containers[0..bm.size], 0..) |tp, i| {
            if (TaggedPtr.getType(tp) == .run) {
                run_bitset[i / 8] |= @as(u8, 1) << @intCast(i % 8);
            }
        }
        try writer.writeAll(run_bitset);
    } else {
        try writer.writeInt(u32, fmt.SERIAL_COOKIE_NO_RUNCONTAINER, .little);
        try writer.writeInt(u32, bm.size, .little);
    }

    // Descriptive header: key (u16) + cardinality-1 (u16) per container (bulk write)
    var desc_buf = try bm.allocator.alloc(u16, bm.size * 2);
    defer bm.allocator.free(desc_buf);
    for (bm.containers[0..bm.size], bm.keys[0..bm.size], 0..) |tp, key, i| {
        desc_buf[i * 2] = key;
        const card = Container.fromTagged(tp).getCardinality();
        desc_buf[i * 2 + 1] = @intCast(card - 1);
    }
    try writer.writeAll(std.mem.sliceAsBytes(desc_buf[0 .. bm.size * 2]));

    // Offset header:
    // - Always for no-run format (RoaringFormatSpec requirement)
    // - For run format only when size >= NO_OFFSET_THRESHOLD
    // Offsets are ABSOLUTE positions from buffer start per RoaringFormatSpec
    if (!has_runs or bm.size >= fmt.NO_OFFSET_THRESHOLD) {
        // Calculate where container data begins (absolute position from buffer start)
        var data_start: u32 = undefined;
        if (has_runs) {
            // Cookie(4) + run_bitset((size+7)/8) + descriptive(size*4) + offsets(size*4)
            const bitset_bytes: u32 = (bm.size + 7) / 8;
            data_start = 4 + bitset_bytes + (@as(u32, bm.size) * 4) + (@as(u32, bm.size) * 4);
        } else {
            // Cookie(4) + size(4) + descriptive(size*4) + offsets(size*4)
            data_start = 8 + (@as(u32, bm.size) * 4) + (@as(u32, bm.size) * 4);
        }

        const offset_buf = try bm.allocator.alloc(u32, bm.size);
        defer bm.allocator.free(offset_buf);
        var offset: u32 = data_start;
        for (bm.containers[0..bm.size], 0..) |tp, i| {
            offset_buf[i] = offset;
            const container = Container.fromTagged(tp);
            offset += switch (container) {
                .array => |ac| @as(u32, ac.cardinality) * 2,
                .bitset => BitsetContainer.SIZE_BYTES,
                .run => |rc| 2 + @as(u32, rc.n_runs) * 4, // n_runs prefix + pairs
                .reserved => 0,
            };
        }
        try writer.writeAll(std.mem.sliceAsBytes(offset_buf));
    }

    // Container data (bulk write - assumes little-endian, checked at comptime)
    for (bm.containers[0..bm.size]) |tp| {
        const container = Container.fromTagged(tp);
        switch (container) {
            .array => |ac| {
                try writer.writeAll(std.mem.sliceAsBytes(ac.values[0..ac.cardinality]));
            },
            .bitset => |bc| {
                try writer.writeAll(std.mem.sliceAsBytes(bc.words));
            },
            .run => |rc| {
                // RoaringFormatSpec: n_runs prefix followed by run pairs
                try writer.writeInt(u16, rc.n_runs, .little);
                try writer.writeAll(std.mem.sliceAsBytes(rc.runs[0..rc.n_runs]));
            },
            .reserved => {},
        }
    }
}

/// Deserialize a bitmap from bytes (RoaringFormatSpec compatible).
///
/// Performance: Use `std.heap.ArenaAllocator` for ~6x faster deserialization.
/// See `RoaringBitmap.deserialize` doc comment for usage example.
pub fn deserialize(allocator: std.mem.Allocator, data: []const u8) !RoaringBitmap {
    if (data.len < 4) return error.InvalidFormat;

    var reader = std.Io.Reader.fixed(data);

    return deserializeFromReader(allocator, &reader, data.len);
}

/// Deserialize, then validate semantic invariants. Use for untrusted input.
pub fn deserializeSafe(allocator: std.mem.Allocator, data: []const u8) !RoaringBitmap {
    var bm = try deserialize(allocator, data);
    errdefer bm.deinit();

    try bm.validate();
    return bm;
}

/// Deserialize from any reader.
///
/// Performance: Use `std.heap.ArenaAllocator` for ~6x faster deserialization.
pub fn deserializeFromReader(allocator: std.mem.Allocator, reader: anytype, data_len: usize) !RoaringBitmap {
    _ = data_len;

    const cookie = try reader.takeInt(u32, .little);

    var size: u32 = undefined;
    var has_runs = false;
    var run_bitset: ?[]u8 = null;
    defer if (run_bitset) |rb| allocator.free(rb);

    if ((cookie & 0xFFFF) == fmt.SERIAL_COOKIE) {
        // Format with run containers
        has_runs = true;
        size = ((cookie >> 16) & 0xFFFF) + 1;
        std.debug.assert(size <= MAX_CONTAINER_COUNT);

        // Read run container bitset
        const bitset_bytes = (size + 7) / 8;
        run_bitset = try allocator.alloc(u8, bitset_bytes);
        reader.readSliceAll(run_bitset.?) catch return error.InvalidFormat;
    } else if (cookie == fmt.SERIAL_COOKIE_NO_RUNCONTAINER) {
        // Format without run containers
        size = try reader.takeInt(u32, .little);
        if (size > MAX_CONTAINER_COUNT) return error.InvalidFormat;
    } else {
        return error.InvalidFormat;
    }

    if (size == 0) {
        return RoaringBitmap.init(allocator);
    }

    var result = try RoaringBitmap.init(allocator);
    errdefer result.deinit();

    try result.ensureTotalCapacity(size);

    // Read descriptive header (bulk read as packed u16 pairs)
    var cardinalities = try allocator.alloc(u32, size);
    defer allocator.free(cardinalities);

    const desc_len = @as(usize, size) * 2;
    const desc_buf = try allocator.alloc(u16, desc_len);
    defer allocator.free(desc_buf);
    reader.readSliceAll(std.mem.sliceAsBytes(desc_buf[0..desc_len])) catch return error.InvalidFormat;

    for (0..size) |i| {
        result.keys[i] = desc_buf[i * 2];
        cardinalities[i] = @as(u32, desc_buf[i * 2 + 1]) + 1;
    }

    // Skip offset header if present:
    // - Always for no-run format (RoaringFormatSpec requirement)
    // - For run format only when size >= NO_OFFSET_THRESHOLD
    if (!has_runs or size >= fmt.NO_OFFSET_THRESHOLD) {
        reader.discardAll(@as(usize, size) * 4) catch return error.InvalidFormat;
    }

    // Read container data (bulk read - assumes little-endian, checked at comptime)
    for (0..size) |i| {
        const is_run = if (run_bitset) |rb|
            (rb[i / 8] & (@as(u8, 1) << @intCast(i % 8))) != 0
        else
            false;

        const card = cardinalities[i];

        if (is_run) {
            // Run container: n_runs is in the data section prefix, not the header
            // (header stores cardinality-1 which is sum of run lengths, not n_runs)
            const n_runs = try reader.takeInt(u16, .little);
            const rc = try RunContainer.init(allocator, n_runs);
            errdefer rc.deinit(allocator);

            reader.readSliceAll(std.mem.sliceAsBytes(rc.runs[0..n_runs])) catch return error.InvalidFormat;
            rc.n_runs = n_runs;
            rc.cardinality = @intCast(card);
            result.containers[i] = TaggedPtr.initRun(rc);
        } else if (card > ArrayContainer.MAX_CARDINALITY) {
            // Bitset container
            const bc = try BitsetContainer.init(allocator);
            errdefer bc.deinit(allocator);

            reader.readSliceAll(std.mem.sliceAsBytes(bc.words)) catch return error.InvalidFormat;
            bc.cardinality = @intCast(card);
            result.containers[i] = TaggedPtr.initBitset(bc);
        } else {
            // Array container
            const ac = try ArrayContainer.init(allocator, @intCast(card));
            errdefer ac.deinit(allocator);

            reader.readSliceAll(std.mem.sliceAsBytes(ac.values[0..card])) catch return error.InvalidFormat;
            ac.cardinality = @intCast(card);
            result.containers[i] = TaggedPtr.initArray(ac);
        }

        result.size = @intCast(i + 1);
    }

    // Compute total cardinality from header data (free - already parsed)
    var total_cardinality: u64 = 0;
    for (cardinalities[0..size]) |c| total_cardinality += c;
    result.cached_cardinality = @intCast(total_cardinality);

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "serialize and deserialize empty bitmap" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    const bytes = try serialize(&bm, allocator);
    defer allocator.free(bytes);

    var restored = try deserialize(allocator, bytes);
    defer restored.deinit();

    try std.testing.expect(restored.isEmpty());
    try std.testing.expect(bm.equals(&restored));
}

fn expectFixedSerializeMatchesLegacy(bm: *const RoaringBitmap, allocator: std.mem.Allocator) !void {
    const direct = try serialize(bm, allocator);
    defer allocator.free(direct);

    const legacy = try allocator.alloc(u8, serializedSizeInBytes(bm));
    defer allocator.free(legacy);
    var writer = std.Io.Writer.fixed(legacy);
    try serializeToWriter(bm, &writer);

    try std.testing.expectEqualSlices(u8, legacy, direct);
    var restored = try deserialize(allocator, direct);
    defer restored.deinit();
    try std.testing.expect(bm.equals(&restored));
}

test "fixed serialize matches writer across container representations" {
    const allocator = std.testing.allocator;

    var empty = try RoaringBitmap.init(allocator);
    defer empty.deinit();
    try expectFixedSerializeMatchesLegacy(&empty, allocator);

    var mixed = try RoaringBitmap.init(allocator);
    defer mixed.deinit();
    _ = try mixed.add(1);
    _ = try mixed.add(100);
    var value: u32 = 1 << 16;
    while (value < (1 << 16) + 10_000) : (value += 2) _ = try mixed.add(value);
    _ = try mixed.addRange(2 << 16, (2 << 16) + 10_000);
    _ = try mixed.runOptimize();
    try expectFixedSerializeMatchesLegacy(&mixed, allocator);
}

test "fixed serialize matches writer at run offset threshold" {
    const allocator = std.testing.allocator;

    inline for (.{ fmt.NO_OFFSET_THRESHOLD - 1, fmt.NO_OFFSET_THRESHOLD }) |container_count| {
        var bm = try RoaringBitmap.init(allocator);
        defer bm.deinit();
        for (0..container_count) |key| {
            const base: u32 = @as(u32, @intCast(key)) << 16;
            _ = try bm.addRange(base + 10, base + 100);
        }
        _ = try bm.runOptimize();
        try std.testing.expect(hasRunContainers(&bm));
        try expectFixedSerializeMatchesLegacy(&bm, allocator);
    }
}

test "serialize and deserialize array container" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    _ = try bm.add(1);
    _ = try bm.add(100);
    _ = try bm.add(1000);

    const bytes = try serialize(&bm, allocator);
    defer allocator.free(bytes);

    var restored = try deserialize(allocator, bytes);
    defer restored.deinit();

    try std.testing.expectEqual(bm.cardinality(), restored.cardinality());
    try std.testing.expect(restored.contains(1));
    try std.testing.expect(restored.contains(100));
    try std.testing.expect(restored.contains(1000));
    try std.testing.expect(bm.equals(&restored));
}

test "portableSizeInBytes reports leading bitmap length" {
    const allocator = std.testing.allocator;

    var first = try RoaringBitmap.init(allocator);
    defer first.deinit();
    _ = try first.add(1);
    _ = try first.add(100);

    var second = try RoaringBitmap.init(allocator);
    defer second.deinit();
    _ = try second.addRange(10, 100);

    const first_bytes = try serialize(&first, allocator);
    defer allocator.free(first_bytes);
    const second_bytes = try serialize(&second, allocator);
    defer allocator.free(second_bytes);

    const combined = try allocator.alloc(u8, first_bytes.len + second_bytes.len);
    defer allocator.free(combined);
    @memcpy(combined[0..first_bytes.len], first_bytes);
    @memcpy(combined[first_bytes.len..], second_bytes);

    try std.testing.expectEqual(first_bytes.len, try portableSizeInBytes(combined));
    try std.testing.expectEqual(second_bytes.len, try portableSizeInBytes(combined[first_bytes.len..]));
}

test "portableSizeInBytes rejects truncated bitmap" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();
    _ = try bm.addRange(10, 100);

    const bytes = try serialize(&bm, allocator);
    defer allocator.free(bytes);

    for (0..bytes.len) |len| {
        try std.testing.expectError(error.InvalidFormat, portableSizeInBytes(bytes[0..len]));
    }
}

test "serialize and deserialize multiple containers" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Values in different chunks
    _ = try bm.add(100); // chunk 0
    _ = try bm.add(65536 + 200); // chunk 1
    _ = try bm.add(131072 + 300); // chunk 2

    const bytes = try serialize(&bm, allocator);
    defer allocator.free(bytes);

    var restored = try deserialize(allocator, bytes);
    defer restored.deinit();

    try std.testing.expectEqual(@as(u32, 3), restored.size);
    try std.testing.expect(restored.contains(100));
    try std.testing.expect(restored.contains(65536 + 200));
    try std.testing.expect(restored.contains(131072 + 300));
    try std.testing.expect(bm.equals(&restored));
}

test "serialize round-trip preserves all values" {
    const allocator = std.testing.allocator;

    var bm = try RoaringBitmap.init(allocator);
    defer bm.deinit();

    // Add various values across chunks
    const values = [_]u32{ 0, 1, 100, 1000, 65535, 65536, 100000, 0xFFFFFFFF };
    for (values) |v| {
        _ = try bm.add(v);
    }

    const bytes = try serialize(&bm, allocator);
    defer allocator.free(bytes);

    var restored = try deserialize(allocator, bytes);
    defer restored.deinit();

    try std.testing.expectEqual(bm.cardinality(), restored.cardinality());

    // Verify all values via iterator
    var it1 = bm.iterator();
    var it2 = restored.iterator();
    while (it1.next()) |v1| {
        const v2 = it2.next();
        try std.testing.expectEqual(v1, v2.?);
    }
    try std.testing.expectEqual(@as(?u32, null), it2.next());
}

test "deserialize rejects no-run size above maximum container count" {
    const allocator = std.testing.allocator;
    const data = [_]u8{
        0x3A, 0x30, 0x00, 0x00, // SERIAL_COOKIE_NO_RUNCONTAINER
        0x01, 0x00, 0x01, 0x00, // 65537 containers
    };

    try std.testing.expectError(error.InvalidFormat, deserialize(allocator, &data));
}
