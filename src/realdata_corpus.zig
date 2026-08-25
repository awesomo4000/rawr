// SPDX-License-Identifier: MPL-2.0

//! Deterministic loader for the external real-roaring-datasets corpora.
//!
//! Dataset files are intentionally not distributed with rawr. The fetcher
//! verifies pinned archives, while this module pins entry ordering, validates
//! the text format, and computes a stable fingerprint for benchmark workers.

const std = @import("std");

const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const Dataset = enum {
    uscensus2000,
    census1881,
    wikileaks_noquotes,

    pub fn name(self: Dataset) []const u8 {
        return switch (self) {
            .uscensus2000 => "uscensus2000",
            .census1881 => "census1881",
            .wikileaks_noquotes => "wikileaks-noquotes",
        };
    }

    pub fn expectedEntries(_: Dataset) usize {
        return 200;
    }

    pub fn parse(dataset_name: []const u8) ?Dataset {
        for (supported_datasets) |dataset| {
            if (std.mem.eql(u8, dataset_name, dataset.name())) return dataset;
        }
        return null;
    }
};

pub const supported_datasets = [_]Dataset{
    .uscensus2000,
    .census1881,
    .wikileaks_noquotes,
};

pub const Bitmap = struct {
    name: []u8,
    values: []u32,
};

pub const Corpus = struct {
    allocator: Allocator,
    bitmaps: []Bitmap,
    fingerprint: u64,
    total_values: u64,

    pub fn deinit(self: *Corpus) void {
        for (self.bitmaps) |bitmap| {
            self.allocator.free(bitmap.name);
            self.allocator.free(bitmap.values);
        }
        self.allocator.free(self.bitmaps);
        self.* = undefined;
    }
};

const EntryOrder = enum {
    bytewise,
    reverse_bytewise,
};

pub fn loadDataset(
    allocator: Allocator,
    io: Io,
    base_path: []const u8,
    dataset: Dataset,
) !Corpus {
    const dataset_path = try std.fs.path.join(allocator, &.{ base_path, dataset.name() });
    defer allocator.free(dataset_path);

    var dir = try Io.Dir.cwd().openDir(io, dataset_path, .{ .iterate = true });
    defer dir.close(io);
    return loadDir(allocator, io, dir, dataset.expectedEntries(), .bytewise);
}

/// Loads the same files in reverse bytewise order for the ordering mutation
/// control. Production benchmark code must use `loadDataset`.
pub fn loadDatasetWithReversedOrderForTesting(
    allocator: Allocator,
    io: Io,
    base_path: []const u8,
    dataset: Dataset,
) !Corpus {
    const dataset_path = try std.fs.path.join(allocator, &.{ base_path, dataset.name() });
    defer allocator.free(dataset_path);

    var dir = try Io.Dir.cwd().openDir(io, dataset_path, .{ .iterate = true });
    defer dir.close(io);
    return loadDir(allocator, io, dir, dataset.expectedEntries(), .reverse_bytewise);
}

fn loadDir(
    allocator: Allocator,
    io: Io,
    dir: Io.Dir,
    expected_entries: usize,
    order: EntryOrder,
) !Corpus {
    var paths: std.ArrayList([]u8) = .empty;
    defer {
        for (paths.items) |path| {
            if (path.len != 0) allocator.free(path);
        }
        paths.deinit(allocator);
    }

    var walker = try dir.walk(allocator);
    defer walker.deinit();
    while (try walker.next(io)) |entry| {
        switch (entry.kind) {
            .directory => continue,
            .file => try paths.append(allocator, try allocator.dupe(u8, entry.path)),
            else => return error.UnsupportedCorpusEntry,
        }
    }

    if (paths.items.len != expected_entries) return error.UnexpectedEntryCount;
    switch (order) {
        .bytewise => std.mem.sortUnstable([]u8, paths.items, {}, pathLessThan),
        .reverse_bytewise => std.mem.sortUnstable([]u8, paths.items, {}, pathGreaterThan),
    }

    const bitmaps = try allocator.alloc(Bitmap, paths.items.len);
    var loaded: usize = 0;
    errdefer {
        for (bitmaps[0..loaded]) |bitmap| {
            allocator.free(bitmap.name);
            allocator.free(bitmap.values);
        }
        allocator.free(bitmaps);
    }

    var hasher = StableHasher.init();
    var total_values: u64 = 0;
    for (paths.items, 0..) |path, ordinal| {
        const contents = try dir.readFileAlloc(
            io,
            path,
            allocator,
            .limited(256 * 1024 * 1024),
        );
        defer allocator.free(contents);

        const values = try parseBitmap(allocator, contents);
        errdefer allocator.free(values);

        hasher.addU32(@intCast(ordinal));
        hasher.addU64(values.len);
        for (values) |value| hasher.addU32(value);
        total_values = std.math.add(u64, total_values, values.len) catch
            return error.CorpusTooLarge;

        bitmaps[loaded] = .{ .name = path, .values = values };
        paths.items[ordinal] = &.{};
        loaded += 1;
    }

    return .{
        .allocator = allocator,
        .bitmaps = bitmaps,
        .fingerprint = hasher.finish(),
        .total_values = total_values,
    };
}

fn parseBitmap(allocator: Allocator, contents: []const u8) ![]u32 {
    const text = std.mem.trim(u8, contents, " \t\r\n");
    if (text.len == 0) return error.EmptyBitmap;

    const value_count = std.mem.countScalar(u8, text, ',') + 1;
    const values = try allocator.alloc(u32, value_count);
    errdefer allocator.free(values);

    var tokens = std.mem.splitScalar(u8, text, ',');
    var index: usize = 0;
    while (tokens.next()) |raw_token| : (index += 1) {
        const token = std.mem.trim(u8, raw_token, " \t\r\n");
        if (token.len == 0) return error.EmptyValue;
        const value = try std.fmt.parseInt(u32, token, 10);
        if (index != 0 and value <= values[index - 1]) {
            return error.ValuesNotStrictlyAscending;
        }
        values[index] = value;
    }
    std.debug.assert(index == values.len);
    return values;
}

fn pathLessThan(_: void, a: []u8, b: []u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

fn pathGreaterThan(_: void, a: []u8, b: []u8) bool {
    return std.mem.order(u8, a, b) == .gt;
}

const StableHasher = struct {
    state: u64 = offset_basis,

    const offset_basis: u64 = 0xcbf29ce484222325;
    const prime: u64 = 0x100000001b3;

    fn init() StableHasher {
        return .{};
    }

    fn addByte(self: *StableHasher, byte: u8) void {
        self.state = (self.state ^ byte) *% prime;
    }

    fn addU32(self: *StableHasher, value: u32) void {
        inline for (0..4) |shift| self.addByte(@truncate(value >> (shift * 8)));
    }

    fn addU64(self: *StableHasher, value: u64) void {
        inline for (0..8) |shift| self.addByte(@truncate(value >> (shift * 8)));
    }

    fn finish(self: StableHasher) u64 {
        return self.state;
    }
};

test "loader sorts entries bytewise and fingerprints their ordered values" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "csv2.txt", .data = "20, 30\n" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "csv10.txt", .data = "10\n" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "csv1.txt", .data = "1,2,3\n" });

    var first = try loadDir(std.testing.allocator, std.testing.io, tmp.dir, 3, .bytewise);
    defer first.deinit();
    var second = try loadDir(std.testing.allocator, std.testing.io, tmp.dir, 3, .bytewise);
    defer second.deinit();

    try std.testing.expectEqualStrings("csv1.txt", first.bitmaps[0].name);
    try std.testing.expectEqualStrings("csv10.txt", first.bitmaps[1].name);
    try std.testing.expectEqualStrings("csv2.txt", first.bitmaps[2].name);
    try std.testing.expectEqual(@as(u64, 6), first.total_values);
    try std.testing.expectEqual(first.fingerprint, second.fingerprint);

    var reversed = try loadDir(std.testing.allocator, std.testing.io, tmp.dir, 3, .reverse_bytewise);
    defer reversed.deinit();
    try std.testing.expect(first.fingerprint != reversed.fingerprint);
}

test "loader guards entry count and strictly ascending values" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "csv0.txt", .data = "1,3,2\n" });
    try std.testing.expectError(
        error.UnexpectedEntryCount,
        loadDir(std.testing.allocator, std.testing.io, tmp.dir, 2, .bytewise),
    );
    try std.testing.expectError(
        error.ValuesNotStrictlyAscending,
        loadDir(std.testing.allocator, std.testing.io, tmp.dir, 1, .bytewise),
    );
}
