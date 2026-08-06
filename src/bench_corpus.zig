// SPDX-License-Identifier: MPL-2.0

//! Shared repository-only benchmark corpora.

const std = @import("std");
const rawr = @import("rawr");

const RoaringBitmap = rawr.RoaringBitmap;

pub const many_bitmap_count = 32;
pub const many_key_count = 6;
pub const expected_many_corpus_hash: u64 = 0x4826470feff53a55;

pub const TypeCounts = struct {
    array: u8 = 0,
    bitset: u8 = 0,
    run: u8 = 0,

    pub fn total(self: TypeCounts) u8 {
        return self.array + self.bitset + self.run;
    }
};

pub const expected_many_type_counts = [_]TypeCounts{.{
    .array = 16,
    .bitset = 8,
    .run = 8,
}} ** many_key_count;

pub fn initRawrManyBitmaps(
    allocator: std.mem.Allocator,
    bitmaps: *[many_bitmap_count]?RoaringBitmap,
    inputs: *[many_bitmap_count]*const RoaringBitmap,
) !void {
    if (bitmaps[0] != null) return;

    var initialized: usize = 0;
    errdefer {
        for (bitmaps[0..initialized]) |*maybe_bitmap| {
            if (maybe_bitmap.*) |*bitmap| bitmap.deinit();
            maybe_bitmap.* = null;
        }
    }

    for (0..many_bitmap_count) |bitmap_index| {
        var bitmap = try RoaringBitmap.init(allocator);
        errdefer bitmap.deinit();
        try addManyPatternRawr(&bitmap, bitmap_index);
        if (bitmap_index % 3 == 0) _ = try bitmap.runOptimize();
        bitmaps[bitmap_index] = bitmap;
        inputs[bitmap_index] = &bitmaps[bitmap_index].?;
        initialized += 1;
    }

    _ = try assertRawrManyFingerprint(allocator, inputs);
}

pub fn addManyPatternRawr(bitmap: *RoaringBitmap, bitmap_index: usize) !void {
    for (0..many_key_count) |chunk| {
        const base: u32 = @as(u32, @intCast(chunk)) << 16;
        switch ((bitmap_index + chunk) % 4) {
            0 => for (0..128) |value_index| {
                const low: u32 = @intCast((value_index * 521 + bitmap_index * 17) & 0xffff);
                _ = try bitmap.add(base | low);
            },
            1 => for (0..5000) |value_index| {
                const low: u32 = @intCast((value_index * 13 + bitmap_index * 29) & 0xffff);
                _ = try bitmap.add(base | low);
            },
            2 => {
                const start: u32 = @intCast((bitmap_index * 97) % 20_000);
                _ = try bitmap.addRange(base | start, base | (start + 12_000));
            },
            3 => {
                _ = try bitmap.add(base);
                _ = try bitmap.add(base | 1);
                _ = try bitmap.add(base | 65_534);
                _ = try bitmap.add(base | 65_535);
            },
            else => unreachable,
        }
    }
}

pub fn assertRawrManyFingerprint(
    allocator: std.mem.Allocator,
    inputs: *const [many_bitmap_count]*const RoaringBitmap,
) !u64 {
    var actual = [_]TypeCounts{.{}} ** many_key_count;

    for (inputs) |bitmap| {
        if (bitmap.size != many_key_count) return error.UnexpectedManyKeyCount;
        for (0..many_key_count) |key| {
            if (bitmap.keys[key] != key) return error.UnexpectedManyKeyOrder;
            switch (bitmap.containers[key].getType()) {
                .array => actual[key].array += 1,
                .bitset => actual[key].bitset += 1,
                .run => actual[key].run += 1,
                .reserved => return error.UnexpectedReservedContainer,
            }
        }
    }

    for (actual, expected_many_type_counts) |found, expected| {
        if (!std.meta.eql(found, expected) or found.total() != many_bitmap_count) {
            return error.UnexpectedManyTypeCounts;
        }
    }

    var hash: u64 = 0xcbf29ce484222325;
    for (inputs) |bitmap| {
        const bytes = try bitmap.serialize(allocator);
        defer allocator.free(bytes);
        hash = fnv1a(hash, std.mem.asBytes(&bytes.len));
        hash = fnv1a(hash, bytes);
    }
    if (hash != expected_many_corpus_hash) return error.UnexpectedManyCorpusHash;
    return hash;
}

fn fnv1a(initial: u64, bytes: []const u8) u64 {
    var hash = initial;
    for (bytes) |byte| {
        hash ^= byte;
        hash *%= 0x100000001b3;
    }
    return hash;
}
