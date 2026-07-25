// SPDX-License-Identifier: MPL-2.0

//! Fresh-process component diagnosis for the architecture-specific parity cluster.

const std = @import("std");
const rawr = @import("rawr");
const bench_time = @import("bench_time.zig");
const CountingAllocator = @import("counting_allocator.zig").CountingAllocator;

const RoaringBitmap = rawr.RoaringBitmap;
const BitsetContainer = rawr.BitsetContainer;
const Container = rawr.Container;
const ops = rawr.container_ops;

const warmup_runs = 3;
const timed_runs = 21;
const many_count = 32;

const Path = enum {
    dense_and_full,
    dense_and_containers,
    dense_or_full,
    dense_or_containers,
    dense_clone,
    range_mask,
    flip_inplace,
    remove_inplace,
    flip_full,
    remove_full,
    lazy_or_full,
    lazy_or_accumulate,
    or_many_full,
    or_many_accumulate,
    prod_and,
    prod_or,
    prod_lazy_or,
    prod_count,
    and_card_w2,
    and_card_w4,
    and_card_w8,
    and_nocard_w2,
    and_nocard_w4,
    and_nocard_w8,
    or_card_w2,
    or_card_w4,
    or_card_w8,
    or_nocard_w2,
    or_nocard_w4,
    or_nocard_w8,
    lazy_or_w2,
    lazy_or_w4,
    lazy_or_w8,
    count_w2,
    count_w4,
    count_w8,

    fn name(self: Path) []const u8 {
        return @tagName(self);
    }

    fn parse(text: []const u8) ?Path {
        inline for (std.meta.fields(Path)) |field| {
            const value: Path = @enumFromInt(field.value);
            if (std.mem.eql(u8, text, value.name())) return value;
        }
        return null;
    }

    fn batch(self: Path) usize {
        return switch (self) {
            .lazy_or_full, .lazy_or_accumulate => 1,
            .or_many_full, .or_many_accumulate => 128,
            .dense_and_full, .dense_or_full, .dense_clone, .range_mask => 4096,
            .dense_and_containers, .dense_or_containers => 1024,
            .flip_inplace, .remove_inplace, .flip_full, .remove_full => 2048,
            else => 4096,
        };
    }
};

const Context = struct {
    allocator: std.mem.Allocator,
    dense_a: ?RoaringBitmap = null,
    dense_b: ?RoaringBitmap = null,
    sparse_a: ?RoaringBitmap = null,
    sparse_b: ?RoaringBitmap = null,
    many: [many_count]?RoaringBitmap = [_]?RoaringBitmap{null} ** many_count,
    many_inputs: [many_count]*const RoaringBitmap = undefined,
    word_a: ?*BitsetContainer = null,
    word_b: ?*BitsetContainer = null,
    word_dst: ?*BitsetContainer = null,

    fn init(allocator: std.mem.Allocator, path: Path) !Context {
        var self = Context{ .allocator = allocator };
        errdefer self.deinit();

        switch (path) {
            .dense_and_full,
            .dense_and_containers,
            .dense_or_full,
            .dense_or_containers,
            .dense_clone,
            .range_mask,
            .flip_inplace,
            .remove_inplace,
            .flip_full,
            .remove_full,
            => try self.initDense(),
            .lazy_or_full, .lazy_or_accumulate => try self.initSparse(),
            .or_many_full, .or_many_accumulate => try self.initMany(),
            else => try self.initWords(),
        }
        return self;
    }

    fn deinit(self: *Context) void {
        if (self.dense_a) |*bitmap| bitmap.deinit();
        if (self.dense_b) |*bitmap| bitmap.deinit();
        if (self.sparse_a) |*bitmap| bitmap.deinit();
        if (self.sparse_b) |*bitmap| bitmap.deinit();
        for (&self.many) |*maybe_bitmap| {
            if (maybe_bitmap.*) |*bitmap| bitmap.deinit();
        }
        if (self.word_a) |container| container.deinit(self.allocator);
        if (self.word_b) |container| container.deinit(self.allocator);
        if (self.word_dst) |container| container.deinit(self.allocator);
    }

    fn initDense(self: *Context) !void {
        var a = try RoaringBitmap.init(self.allocator);
        errdefer a.deinit();
        var b = try RoaringBitmap.init(self.allocator);
        errdefer b.deinit();
        _ = try a.addRange(0, 499_999);
        _ = try b.addRange(250_000, 749_999);
        self.dense_a = a;
        self.dense_b = b;
    }

    fn initSparse(self: *Context) !void {
        const values = try self.allocator.alloc(u32, 500_000);
        defer self.allocator.free(values);
        var prng = std.Random.DefaultPrng.init(54321);
        for (values) |*value| value.* = prng.random().int(u32);
        std.mem.sort(u32, values, {}, std.sort.asc(u32));

        var unique: usize = 1;
        for (values[1..]) |value| {
            if (value != values[unique - 1]) {
                values[unique] = value;
                unique += 1;
            }
        }

        var a = try RoaringBitmap.init(self.allocator);
        errdefer a.deinit();
        var b = try RoaringBitmap.init(self.allocator);
        errdefer b.deinit();
        const half = unique / 2;
        for (values[0..half]) |value| _ = try a.add(value);
        for (values[half / 2 .. unique]) |value| _ = try b.add(value);
        self.sparse_a = a;
        self.sparse_b = b;
    }

    fn initMany(self: *Context) !void {
        for (0..many_count) |index| {
            var bitmap = try RoaringBitmap.init(self.allocator);
            errdefer bitmap.deinit();
            try addManyPattern(&bitmap, index);
            if (index % 3 == 0) _ = try bitmap.runOptimize();
            self.many[index] = bitmap;
            self.many_inputs[index] = &self.many[index].?;
        }
    }

    fn initWords(self: *Context) !void {
        const a = try BitsetContainer.init(self.allocator);
        errdefer a.deinit(self.allocator);
        const b = try BitsetContainer.init(self.allocator);
        errdefer b.deinit(self.allocator);
        const dst = try BitsetContainer.init(self.allocator);
        errdefer dst.deinit(self.allocator);
        for (0..BitsetContainer.NUM_WORDS) |index| {
            a.words[index] = 0x9e3779b97f4a7c15 *% (index + 1);
            b.words[index] = 0xd6e8feb86659fd93 *% (index + 3);
            dst.words[index] = a.words[index];
        }
        a.cardinality = -1;
        b.cardinality = -1;
        dst.cardinality = -1;
        self.word_a = a;
        self.word_b = b;
        self.word_dst = dst;
    }
};

const Measurement = struct {
    median_ns: u64,
    checksum: u64,
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var requested_path: ?Path = null;
    var header = false;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--path=")) {
            requested_path = Path.parse(arg[7..]) orelse return error.UnknownPath;
        } else {
            return error.UnknownArgument;
        }
    }

    if (header) {
        bench_time.printBenchEnvironment();
        bench_time.print("# M4 cluster diagnosis protocol: {d}w/{d}t median\n", .{ warmup_runs, timed_runs });
        return;
    }

    const path = requested_path orelse return error.MissingPath;
    var context = try Context.init(std.heap.smp_allocator, path);
    defer context.deinit();
    printShapes(&context);
    try printAllocationCounts(path, &context);

    const measurement = measure(path, &context);
    if (measurement.checksum == 0) return error.ZeroChecksum;
    bench_time.print("VALIDATION\t{s}\t{d}\n", .{ path.name(), measurement.checksum });
    bench_time.print("RESULT\t{s}\t{d}\t{d}\t{d}\n", .{
        path.name(),
        path.batch(),
        measurement.checksum,
        measurement.median_ns,
    });
}

fn measure(path: Path, context: *Context) Measurement {
    if (path == .flip_inplace or path == .remove_inplace) {
        return measureInplace(path, context);
    }

    var checksum: u64 = 0;
    for (0..warmup_runs) |_| checksum +%= runPath(path, context);

    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        checksum +%= runPath(path, context);
        elapsed.* = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(checksum);
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return .{ .median_ns = times[timed_runs / 2], .checksum = checksum };
}

fn measureInplace(path: Path, context: *Context) Measurement {
    var checksum: u64 = 0;
    for (0..warmup_runs) |_| {
        const result = timeInplaceBatch(path, context);
        checksum +%= result.checksum;
    }

    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const result = timeInplaceBatch(path, context);
        checksum +%= result.checksum;
        elapsed.* = result.elapsed_ns;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return .{ .median_ns = times[timed_runs / 2], .checksum = checksum };
}

const TimedBatch = struct {
    elapsed_ns: u64,
    checksum: u64,
};

fn timeInplaceBatch(path: Path, context: *Context) TimedBatch {
    const bitmaps = context.allocator.alloc(RoaringBitmap, path.batch()) catch unreachable;
    defer context.allocator.free(bitmaps);
    var initialized: usize = 0;
    defer for (bitmaps[0..initialized]) |*bitmap| bitmap.deinit();
    for (bitmaps) |*bitmap| {
        bitmap.* = context.dense_a.?.clone(context.allocator) catch unreachable;
        initialized += 1;
    }

    var checksum: u64 = 0;
    const start = bench_time.monotonicNanos();
    for (bitmaps) |*bitmap| {
        switch (path) {
            .flip_inplace => {
                bitmap.flipInplace(100_000, 650_000) catch unreachable;
                checksum +%= bitmap.size + 1;
            },
            .remove_inplace => checksum +%= (bitmap.removeRange(100_000, 650_000) catch unreachable) + 1,
            else => unreachable,
        }
    }
    return .{ .elapsed_ns = bench_time.monotonicNanos() - start, .checksum = checksum };
}

noinline fn runPath(path: Path, context: *Context) u64 {
    var checksum: u64 = 0;
    for (0..path.batch()) |_| {
        checksum +%= switch (path) {
            .dense_and_full => fullDenseAnd(context, context.allocator),
            .dense_and_containers => denseContainerSweep(context, .band, context.allocator),
            .dense_or_full => fullDenseOr(context, context.allocator),
            .dense_or_containers => denseContainerSweep(context, .bor, context.allocator),
            .dense_clone => cloneDense(context, context.allocator),
            .range_mask => buildRangeMask(context.allocator),
            .flip_inplace, .remove_inplace => unreachable,
            .flip_full => fullFlip(context, context.allocator),
            .remove_full => fullRemove(context, context.allocator),
            .lazy_or_full => fullLazyOr(context, context.allocator),
            .lazy_or_accumulate => lazyOrAccumulate(context, context.allocator),
            .or_many_full => fullOrMany(context, context.allocator),
            .or_many_accumulate => orManyAccumulate(context, context.allocator),
            .prod_and => productionWordOp(context, .band),
            .prod_or => productionWordOp(context, .bor),
            .prod_lazy_or => productionWordOp(context, .lazy_bor),
            .prod_count => context.word_a.?.computeCardinality(),
            .and_card_w2 => diagnosticWordOp(2, .band, true, context),
            .and_card_w4 => diagnosticWordOp(4, .band, true, context),
            .and_card_w8 => diagnosticWordOp(8, .band, true, context),
            .and_nocard_w2 => diagnosticWordOp(2, .band, false, context),
            .and_nocard_w4 => diagnosticWordOp(4, .band, false, context),
            .and_nocard_w8 => diagnosticWordOp(8, .band, false, context),
            .or_card_w2 => diagnosticWordOp(2, .bor, true, context),
            .or_card_w4 => diagnosticWordOp(4, .bor, true, context),
            .or_card_w8 => diagnosticWordOp(8, .bor, true, context),
            .or_nocard_w2 => diagnosticWordOp(2, .bor, false, context),
            .or_nocard_w4 => diagnosticWordOp(4, .bor, false, context),
            .or_nocard_w8 => diagnosticWordOp(8, .bor, false, context),
            .lazy_or_w2 => diagnosticWordOp(2, .bor, false, context),
            .lazy_or_w4 => diagnosticWordOp(4, .bor, false, context),
            .lazy_or_w8 => diagnosticWordOp(8, .bor, false, context),
            .count_w2 => diagnosticCount(2, context.word_a.?.words),
            .count_w4 => diagnosticCount(4, context.word_a.?.words),
            .count_w8 => diagnosticCount(8, context.word_a.?.words),
        };
    }
    return checksum;
}

const BinaryOp = enum { band, bor, lazy_bor };

fn fullDenseAnd(context: *Context, allocator: std.mem.Allocator) u64 {
    var result = context.dense_a.?.bitwiseAnd(allocator, &context.dense_b.?) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
    return result.size + 1;
}

fn fullDenseOr(context: *Context, allocator: std.mem.Allocator) u64 {
    var result = context.dense_a.?.bitwiseOr(allocator, &context.dense_b.?) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
    return result.size + 1;
}

fn denseContainerSweep(context: *Context, op: BinaryOp, allocator: std.mem.Allocator) u64 {
    const a = &context.dense_a.?;
    const b = &context.dense_b.?;
    var i: usize = 0;
    var j: usize = 0;
    var checksum: u64 = 1;
    while (i < a.size and j < b.size) {
        if (a.keys[i] < b.keys[j]) {
            i += 1;
        } else if (a.keys[i] > b.keys[j]) {
            j += 1;
        } else {
            const result = switch (op) {
                .band => ops.containerIntersection(allocator, Container.fromTagged(a.containers[i]), Container.fromTagged(b.containers[j])) catch unreachable,
                .bor => ops.containerUnion(allocator, Container.fromTagged(a.containers[i]), Container.fromTagged(b.containers[j])) catch unreachable,
                .lazy_bor => unreachable,
            };
            checksum +%= result.getCardinality();
            result.deinit(allocator);
            i += 1;
            j += 1;
        }
    }
    return checksum;
}

fn cloneDense(context: *Context, allocator: std.mem.Allocator) u64 {
    var result = context.dense_a.?.clone(allocator) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
    return result.size + 1;
}

fn buildRangeMask(allocator: std.mem.Allocator) u64 {
    var mask = RoaringBitmap.init(allocator) catch unreachable;
    defer mask.deinit();
    return (mask.addRange(100_000, 650_000) catch unreachable) + 1;
}

fn fullFlip(context: *Context, allocator: std.mem.Allocator) u64 {
    var result = context.dense_a.?.flip(allocator, 100_000, 650_000) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
    return result.size + 1;
}

fn fullRemove(context: *Context, allocator: std.mem.Allocator) u64 {
    var result = context.dense_a.?.clone(allocator) catch unreachable;
    defer result.deinit();
    const removed = result.removeRange(100_000, 650_000) catch unreachable;
    std.mem.doNotOptimizeAway(&result);
    return removed + 1;
}

fn fullLazyOr(context: *Context, allocator: std.mem.Allocator) u64 {
    var result = context.sparse_a.?.lazyOr(allocator, &context.sparse_b.?, true) catch unreachable;
    defer result.deinit();
    return result.size + 1;
}

fn lazyOrAccumulate(context: *Context, allocator: std.mem.Allocator) u64 {
    const a = &context.sparse_a.?;
    const b = &context.sparse_b.?;
    var i: usize = 0;
    var j: usize = 0;
    var checksum: u64 = 1;
    while (i < a.size and j < b.size) {
        if (a.keys[i] < b.keys[j]) {
            i += 1;
        } else if (a.keys[i] > b.keys[j]) {
            j += 1;
        } else {
            const dst = BitsetContainer.init(allocator) catch unreachable;
            accumulateContainer(dst, Container.fromTagged(a.containers[i]));
            accumulateContainer(dst, Container.fromTagged(b.containers[j]));
            checksum +%= dst.words[(i + j) & (BitsetContainer.NUM_WORDS - 1)];
            dst.deinit(allocator);
            i += 1;
            j += 1;
        }
    }
    return checksum;
}

fn fullOrMany(context: *Context, allocator: std.mem.Allocator) u64 {
    var result = RoaringBitmap.orMany(allocator, &context.many_inputs) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
    return result.size + 1;
}

fn orManyAccumulate(context: *Context, allocator: std.mem.Allocator) u64 {
    var checksum: u64 = 1;
    for (0..6) |key| {
        const dst = BitsetContainer.init(allocator) catch unreachable;
        for (context.many_inputs) |bitmap| {
            var index: usize = 0;
            while (index < bitmap.size and bitmap.keys[index] < key) : (index += 1) {}
            if (index < bitmap.size and bitmap.keys[index] == key) {
                accumulateContainer(dst, Container.fromTagged(bitmap.containers[index]));
            }
        }
        checksum +%= dst.computeCardinality();
        dst.deinit(allocator);
    }
    return checksum;
}

fn accumulateContainer(dst: *BitsetContainer, container: Container) void {
    switch (container) {
        .array => |array| dst.setList(array.values[0..array.cardinality]),
        .bitset => |bitset| dst.lazyUnionWith(bitset),
        .run => |run| {
            for (run.runs[0..run.n_runs]) |pair| dst.setRange(pair.start, pair.end());
            dst.cardinality = -1;
        },
        .reserved => unreachable,
    }
}

fn productionWordOp(context: *Context, op: BinaryOp) u64 {
    const dst = context.word_dst.?;
    switch (op) {
        .band => dst.intersectionWith(context.word_b.?),
        .bor => dst.unionWith(context.word_b.?),
        .lazy_bor => dst.lazyUnionWith(context.word_b.?),
    }
    return dst.words[511] +% @as(u64, @bitCast(@as(i64, dst.cardinality))) +% 1;
}

noinline fn diagnosticWordOp(comptime width: usize, comptime op: BinaryOp, comptime card: bool, context: *Context) u64 {
    const dst = context.word_dst.?;
    const src = context.word_b.?;
    var cardinality: @Vector(width, u64) = @splat(0);
    var index: usize = 0;
    while (index < BitsetContainer.NUM_WORDS) : (index += width) {
        const a: @Vector(width, u64) = dst.words[index..][0..width].*;
        const b: @Vector(width, u64) = src.words[index..][0..width].*;
        const result = switch (op) {
            .band => a & b,
            .bor, .lazy_bor => a | b,
        };
        dst.words[index..][0..width].* = result;
        if (card) cardinality += @popCount(result);
    }
    const count = if (card) @reduce(.Add, cardinality) else 0;
    return dst.words[511] +% count +% 1;
}

noinline fn diagnosticCount(comptime width: usize, words: []const u64) u64 {
    var counts: @Vector(width, u64) = @splat(0);
    var index: usize = 0;
    while (index < words.len) : (index += width) {
        const values: @Vector(width, u64) = words[index..][0..width].*;
        counts += @popCount(values);
    }
    return @reduce(.Add, counts) + 1;
}

fn addManyPattern(bitmap: *RoaringBitmap, bitmap_index: usize) !void {
    for (0..6) |chunk| {
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

fn printShapes(context: *const Context) void {
    if (context.dense_a) |bitmap| printShape("dense-a", &bitmap);
    if (context.dense_b) |bitmap| printShape("dense-b", &bitmap);
    if (context.sparse_a) |bitmap| printShape("sparse-a", &bitmap);
    if (context.sparse_b) |bitmap| printShape("sparse-b", &bitmap);
    if (context.many[0]) |_| {
        var arrays: u64 = 0;
        var bitsets: u64 = 0;
        var runs: u64 = 0;
        for (context.many_inputs) |bitmap| countShape(bitmap, &arrays, &bitsets, &runs);
        bench_time.print("SHAPE\tmany\t{d}\t{d}\t{d}\t{d}\n", .{ many_count * 6, arrays, bitsets, runs });
    }
}

fn printShape(name: []const u8, bitmap: *const RoaringBitmap) void {
    var arrays: u64 = 0;
    var bitsets: u64 = 0;
    var runs: u64 = 0;
    countShape(bitmap, &arrays, &bitsets, &runs);
    bench_time.print("SHAPE\t{s}\t{d}\t{d}\t{d}\t{d}\n", .{ name, bitmap.size, arrays, bitsets, runs });
}

fn countShape(bitmap: *const RoaringBitmap, arrays: *u64, bitsets: *u64, runs: *u64) void {
    for (bitmap.containers[0..bitmap.size]) |tagged| switch (tagged.getType()) {
        .array => arrays.* += 1,
        .bitset => bitsets.* += 1,
        .run => runs.* += 1,
        .reserved => unreachable,
    };
}

fn printAllocationCounts(path: Path, context: *Context) !void {
    if (path != .dense_and_full and path != .dense_or_full and path != .flip_full and
        path != .remove_full and path != .lazy_or_full and path != .or_many_full) return;

    var counting = CountingAllocator.init(std.heap.smp_allocator);
    const allocator = counting.allocator();
    const checksum = switch (path) {
        .dense_and_full => fullDenseAnd(context, allocator),
        .dense_or_full => fullDenseOr(context, allocator),
        .flip_full => fullFlip(context, allocator),
        .remove_full => fullRemove(context, allocator),
        .lazy_or_full => fullLazyOr(context, allocator),
        .or_many_full => fullOrMany(context, allocator),
        else => unreachable,
    };
    if (counting.stats.live_bytes != 0) return error.DiagnosticAllocationLeak;
    bench_time.print("ALLOC\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        path.name(),
        counting.stats.alloc_calls,
        counting.stats.free_calls,
        counting.stats.resize_calls + counting.stats.remap_calls,
        counting.stats.cumulative_bytes,
        checksum,
    });
}
