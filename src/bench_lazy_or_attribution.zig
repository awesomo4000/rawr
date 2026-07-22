// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const ArrayContainer = rawr.ArrayContainer;
const BitsetContainer = rawr.BitsetContainer;
const Container = rawr.Container;
const TaggedPtr = rawr.TaggedPtr;

const N_VALUES = 500_000;
const WARMUP_RUNS = 2;
const TIMED_RUNS = 9;

var sparse_values: [N_VALUES]u32 = undefined;

const SharedPair = struct {
    first: *const ArrayContainer,
    second: *const ArrayContainer,
};

const CapturedCorpus = struct {
    shared: []SharedPair,
    unique: []TaggedPtr,
    left_only: usize,
    right_only: usize,

    fn deinit(self: *CapturedCorpus, allocator: std.mem.Allocator) void {
        allocator.free(self.shared);
        allocator.free(self.unique);
    }
};

const Stats = struct {
    median_ns: u64,
    min_ns: u64,
    max_ns: u64,
};

fn initValues() usize {
    var prng = std.Random.DefaultPrng.init(54321);
    for (sparse_values[0..]) |*value| value.* = prng.random().int(u32);
    std.mem.sort(u32, sparse_values[0..], {}, std.sort.asc(u32));

    var len: usize = 1;
    for (sparse_values[1..]) |value| {
        if (value != sparse_values[len - 1]) {
            sparse_values[len] = value;
            len += 1;
        }
    }
    return len;
}

fn buildRawrInputs(allocator: std.mem.Allocator, values: []const u32) !struct { RoaringBitmap, RoaringBitmap } {
    var left = try RoaringBitmap.init(allocator);
    errdefer left.deinit();
    var right = try RoaringBitmap.init(allocator);
    errdefer right.deinit();

    const half = values.len / 2;
    try left.addMany(values[0..half]);
    try right.addMany(values[half / 2 ..]);
    return .{ left, right };
}

fn buildCrInputs(values: []const u32) !struct { *c.roaring_bitmap_t, *c.roaring_bitmap_t } {
    const left = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(left);
    const right = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(right);

    const half = values.len / 2;
    c.roaring_bitmap_add_many(left, half, values.ptr);
    c.roaring_bitmap_add_many(right, values.len - half / 2, values[half / 2 ..].ptr);
    return .{ left, right };
}

fn captureCorpus(allocator: std.mem.Allocator, left: *const RoaringBitmap, right: *const RoaringBitmap) !CapturedCorpus {
    var shared_count: usize = 0;
    var left_only: usize = 0;
    var right_only: usize = 0;
    var i: usize = 0;
    var j: usize = 0;
    while (i < left.size and j < right.size) {
        if (left.keys[i] < right.keys[j]) {
            left_only += 1;
            i += 1;
        } else if (left.keys[i] > right.keys[j]) {
            right_only += 1;
            j += 1;
        } else {
            shared_count += 1;
            i += 1;
            j += 1;
        }
    }
    left_only += left.size - i;
    right_only += right.size - j;

    const shared = try allocator.alloc(SharedPair, shared_count);
    errdefer allocator.free(shared);
    const unique = try allocator.alloc(TaggedPtr, left_only + right_only);
    errdefer allocator.free(unique);

    i = 0;
    j = 0;
    var shared_index: usize = 0;
    var unique_index: usize = 0;
    while (i < left.size and j < right.size) {
        if (left.keys[i] < right.keys[j]) {
            unique[unique_index] = left.containers[i];
            unique_index += 1;
            i += 1;
        } else if (left.keys[i] > right.keys[j]) {
            unique[unique_index] = right.containers[j];
            unique_index += 1;
            j += 1;
        } else {
            const first = Container.fromTagged(left.containers[i]);
            const second = Container.fromTagged(right.containers[j]);
            shared[shared_index] = .{
                .first = switch (first) {
                    .array => |array| array,
                    else => return error.ExpectedArrayContainer,
                },
                .second = switch (second) {
                    .array => |array| array,
                    else => return error.ExpectedArrayContainer,
                },
            };
            shared_index += 1;
            i += 1;
            j += 1;
        }
    }
    while (i < left.size) : (i += 1) {
        unique[unique_index] = left.containers[i];
        unique_index += 1;
    }
    while (j < right.size) : (j += 1) {
        unique[unique_index] = right.containers[j];
        unique_index += 1;
    }

    return .{
        .shared = shared,
        .unique = unique,
        .left_only = left_only,
        .right_only = right_only,
    };
}

fn measure(comptime operation: anytype, args: anytype) !Stats {
    for (0..WARMUP_RUNS) |_| _ = try @call(.auto, operation, args);

    var times: [TIMED_RUNS]u64 = undefined;
    for (&times) |*time| time.* = try @call(.auto, operation, args);
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return .{
        .median_ns = times[TIMED_RUNS / 2],
        .min_ns = times[0],
        .max_ns = times[TIMED_RUNS - 1],
    };
}

fn printResult(component: []const u8, variant: []const u8, stats: Stats) void {
    bench_time.print("{s:<20} {s:<12} {d:>10.3} {d:>10.3} {d:>10.3}\n", .{
        component,
        variant,
        @as(f64, @floatFromInt(stats.median_ns)) / 1_000_000.0,
        @as(f64, @floatFromInt(stats.min_ns)) / 1_000_000.0,
        @as(f64, @floatFromInt(stats.max_ns)) / 1_000_000.0,
    });
    bench_time.print("RESULT\t{s}\t{s}\t{d}\n", .{ component, variant, stats.median_ns });
}

fn rawrFull(left: *const RoaringBitmap, right: *const RoaringBitmap, allocator: std.mem.Allocator) !u64 {
    const start = bench_time.monotonicNanos();
    var result = try left.lazyOr(allocator, right, true);
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(&result);
    result.deinit();
    return elapsed;
}

fn crFull(left: *const c.roaring_bitmap_t, right: *const c.roaring_bitmap_t) !u64 {
    const start = bench_time.monotonicNanos();
    const result = c.roaring_bitmap_lazy_or(left, right, true) orelse return error.OutOfMemory;
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(result);
    c.roaring_bitmap_free(result);
    return elapsed;
}

fn rawrFullAfterRepair(left: *const RoaringBitmap, right: *const RoaringBitmap, allocator: std.mem.Allocator) !u64 {
    var priming_result = try left.lazyOr(allocator, right, true);
    try priming_result.repairAfterLazy();
    priming_result.deinit();
    return rawrFull(left, right, allocator);
}

fn crFullAfterRepair(left: *const c.roaring_bitmap_t, right: *const c.roaring_bitmap_t) !u64 {
    const priming_result = c.roaring_bitmap_lazy_or(left, right, true) orelse return error.OutOfMemory;
    c.roaring_bitmap_repair_after_lazy(priming_result);
    c.roaring_bitmap_free(priming_result);
    return crFull(left, right);
}

fn rawrHeaderAlloc(allocator: std.mem.Allocator, count: usize) !u64 {
    const headers = try std.heap.smp_allocator.alloc(*BitsetContainer, count);
    defer std.heap.smp_allocator.free(headers);
    var created: usize = 0;
    errdefer for (headers[0..created]) |header| allocator.destroy(header);

    const start = bench_time.monotonicNanos();
    for (headers) |*header| {
        header.* = try allocator.create(BitsetContainer);
        created += 1;
    }
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(headers.ptr);
    for (headers) |header| allocator.destroy(header);
    return elapsed;
}

fn rawrWordsAlloc(allocator: std.mem.Allocator, count: usize) !u64 {
    const words = try std.heap.smp_allocator.alloc([]align(64) u64, count);
    defer std.heap.smp_allocator.free(words);
    var created: usize = 0;
    errdefer for (words[0..created]) |slice| allocator.free(slice);

    const start = bench_time.monotonicNanos();
    for (words) |*slice| {
        slice.* = try allocator.alignedAlloc(u64, .@"64", BitsetContainer.NUM_WORDS);
        created += 1;
    }
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(words.ptr);
    for (words) |slice| allocator.free(slice);
    return elapsed;
}

fn rawrBitsetCreate(allocator: std.mem.Allocator, count: usize) !u64 {
    const bitsets = try std.heap.smp_allocator.alloc(*BitsetContainer, count);
    defer std.heap.smp_allocator.free(bitsets);
    var created: usize = 0;
    errdefer for (bitsets[0..created]) |bitset| bitset.deinit(allocator);

    const start = bench_time.monotonicNanos();
    for (bitsets) |*bitset| {
        bitset.* = try BitsetContainer.init(allocator);
        created += 1;
    }
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(bitsets.ptr);
    for (bitsets) |bitset| bitset.deinit(allocator);
    return elapsed;
}

fn allocateBitsets(allocator: std.mem.Allocator, count: usize) ![]*BitsetContainer {
    const bitsets = try std.heap.smp_allocator.alloc(*BitsetContainer, count);
    errdefer std.heap.smp_allocator.free(bitsets);
    var created: usize = 0;
    errdefer for (bitsets[0..created]) |bitset| bitset.deinit(allocator);
    for (bitsets) |*bitset| {
        bitset.* = try BitsetContainer.init(allocator);
        created += 1;
    }
    return bitsets;
}

fn freeBitsets(allocator: std.mem.Allocator, bitsets: []*BitsetContainer) void {
    for (bitsets) |bitset| bitset.deinit(allocator);
    std.heap.smp_allocator.free(bitsets);
}

fn rawrZero(allocator: std.mem.Allocator, count: usize) !u64 {
    const bitsets = try allocateBitsets(allocator, count);
    defer freeBitsets(allocator, bitsets);
    for (bitsets) |bitset| @memset(bitset.words, 0xA5A5A5A5A5A5A5A5);

    const start = bench_time.monotonicNanos();
    for (bitsets) |bitset| @memset(bitset.words, 0);
    const elapsed = bench_time.monotonicNanos() - start;
    var checksum: u64 = 0;
    for (bitsets) |bitset| checksum +%= bitset.words[0];
    std.mem.doNotOptimizeAway(checksum);
    return elapsed;
}

fn rawrZeroFresh(allocator: std.mem.Allocator, count: usize) !u64 {
    const words = try std.heap.smp_allocator.alloc([]align(64) u64, count);
    defer std.heap.smp_allocator.free(words);
    var created: usize = 0;
    errdefer for (words[0..created]) |slice| allocator.free(slice);
    for (words) |*slice| {
        slice.* = try allocator.alignedAlloc(u64, .@"64", BitsetContainer.NUM_WORDS);
        created += 1;
    }

    const start = bench_time.monotonicNanos();
    for (words) |slice| @memset(slice, 0);
    const elapsed = bench_time.monotonicNanos() - start;
    var checksum: u64 = 0;
    for (words) |slice| checksum +%= slice[0];
    std.mem.doNotOptimizeAway(checksum);
    for (words) |slice| allocator.free(slice);
    return elapsed;
}

fn rawrAccumulateFirst(allocator: std.mem.Allocator, shared: []const SharedPair) !u64 {
    const bitsets = try allocateBitsets(allocator, shared.len);
    defer freeBitsets(allocator, bitsets);

    const start = bench_time.monotonicNanos();
    for (bitsets, shared) |bitset, pair| bitset.setList(pair.first.values[0..pair.first.cardinality]);
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(bitsets[bitsets.len - 1].words[0]);
    return elapsed;
}

fn rawrAccumulateSecond(allocator: std.mem.Allocator, shared: []const SharedPair) !u64 {
    const bitsets = try allocateBitsets(allocator, shared.len);
    defer freeBitsets(allocator, bitsets);
    for (bitsets, shared) |bitset, pair| bitset.setList(pair.first.values[0..pair.first.cardinality]);

    const start = bench_time.monotonicNanos();
    for (bitsets, shared) |bitset, pair| bitset.setList(pair.second.values[0..pair.second.cardinality]);
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(bitsets[bitsets.len - 1].words[0]);
    return elapsed;
}

fn rawrSharedPipeline(allocator: std.mem.Allocator, shared: []const SharedPair) !u64 {
    const bitsets = try std.heap.smp_allocator.alloc(*BitsetContainer, shared.len);
    defer std.heap.smp_allocator.free(bitsets);
    var created: usize = 0;
    errdefer for (bitsets[0..created]) |bitset| bitset.deinit(allocator);

    const start = bench_time.monotonicNanos();
    for (bitsets, shared) |*bitset, pair| {
        bitset.* = try BitsetContainer.init(allocator);
        created += 1;
        bitset.*.setList(pair.first.values[0..pair.first.cardinality]);
        bitset.*.setList(pair.second.values[0..pair.second.cardinality]);
    }
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(bitsets[bitsets.len - 1].words[0]);
    for (bitsets) |bitset| bitset.deinit(allocator);
    return elapsed;
}

fn rawrCloneUnique(allocator: std.mem.Allocator, unique: []const TaggedPtr) !u64 {
    const clones = try std.heap.smp_allocator.alloc(TaggedPtr, unique.len);
    defer std.heap.smp_allocator.free(clones);
    var cloned: usize = 0;
    errdefer for (clones[0..cloned]) |clone| Container.fromTagged(clone).deinit(allocator);

    const start = bench_time.monotonicNanos();
    for (clones, unique) |*clone, source| {
        clone.* = (try Container.fromTagged(source).clone(allocator)).toTagged();
        cloned += 1;
    }
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(clones.ptr);
    for (clones) |clone| Container.fromTagged(clone).deinit(allocator);
    return elapsed;
}

fn rawrMergeAppend(left: *const RoaringBitmap, right: *const RoaringBitmap) !u64 {
    const max_count: usize = @as(usize, left.size) + right.size;
    const keys = try std.heap.smp_allocator.alloc(u16, max_count);
    defer std.heap.smp_allocator.free(keys);
    const containers = try std.heap.smp_allocator.alloc(TaggedPtr, max_count);
    defer std.heap.smp_allocator.free(containers);

    var i: usize = 0;
    var j: usize = 0;
    var out: usize = 0;
    const start = bench_time.monotonicNanos();
    while (i < left.size and j < right.size) {
        if (left.keys[i] < right.keys[j]) {
            keys[out] = left.keys[i];
            containers[out] = left.containers[i];
            i += 1;
        } else if (left.keys[i] > right.keys[j]) {
            keys[out] = right.keys[j];
            containers[out] = right.containers[j];
            j += 1;
        } else {
            keys[out] = left.keys[i];
            containers[out] = left.containers[i];
            i += 1;
            j += 1;
        }
        out += 1;
    }
    while (i < left.size) : (i += 1) {
        keys[out] = left.keys[i];
        containers[out] = left.containers[i];
        out += 1;
    }
    while (j < right.size) : (j += 1) {
        keys[out] = right.keys[j];
        containers[out] = right.containers[j];
        out += 1;
    }
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(keys[out - 1]);
    std.mem.doNotOptimizeAway(containers[out - 1]);
    return elapsed;
}

fn crHeaders(context: *c.rawr_cr_attr_context) !u64 {
    const start = bench_time.monotonicNanos();
    if (!c.rawr_cr_attr_alloc_headers(context)) return error.OutOfMemory;
    const elapsed = bench_time.monotonicNanos() - start;
    c.rawr_cr_attr_free_headers(context);
    return elapsed;
}

fn crWords(context: *c.rawr_cr_attr_context) !u64 {
    if (!c.rawr_cr_attr_alloc_headers(context)) return error.OutOfMemory;
    defer c.rawr_cr_attr_free_headers(context);
    const start = bench_time.monotonicNanos();
    if (!c.rawr_cr_attr_alloc_words(context)) return error.OutOfMemory;
    const elapsed = bench_time.monotonicNanos() - start;
    c.rawr_cr_attr_free_words(context);
    return elapsed;
}

fn crBitsetCreate(context: *c.rawr_cr_attr_context) !u64 {
    const start = bench_time.monotonicNanos();
    if (!c.rawr_cr_attr_create_bitsets(context)) return error.OutOfMemory;
    const elapsed = bench_time.monotonicNanos() - start;
    c.rawr_cr_attr_free_bitsets(context);
    return elapsed;
}

fn crZero(context: *c.rawr_cr_attr_context) !u64 {
    if (!c.rawr_cr_attr_create_bitsets(context)) return error.OutOfMemory;
    defer c.rawr_cr_attr_free_bitsets(context);
    c.rawr_cr_attr_dirty_words(context);
    const start = bench_time.monotonicNanos();
    c.rawr_cr_attr_zero_words(context);
    return bench_time.monotonicNanos() - start;
}

fn crZeroFresh(context: *c.rawr_cr_attr_context) !u64 {
    if (!c.rawr_cr_attr_alloc_headers(context)) return error.OutOfMemory;
    defer c.rawr_cr_attr_free_headers(context);
    if (!c.rawr_cr_attr_alloc_words(context)) return error.OutOfMemory;
    defer c.rawr_cr_attr_free_words(context);
    const start = bench_time.monotonicNanos();
    c.rawr_cr_attr_zero_words(context);
    return bench_time.monotonicNanos() - start;
}

fn crAccumulateFirst(context: *c.rawr_cr_attr_context) !u64 {
    if (!c.rawr_cr_attr_create_bitsets(context)) return error.OutOfMemory;
    defer c.rawr_cr_attr_free_bitsets(context);
    const start = bench_time.monotonicNanos();
    c.rawr_cr_attr_accumulate_first(context);
    return bench_time.monotonicNanos() - start;
}

fn crAccumulateSecond(context: *c.rawr_cr_attr_context) !u64 {
    if (!c.rawr_cr_attr_create_bitsets(context)) return error.OutOfMemory;
    defer c.rawr_cr_attr_free_bitsets(context);
    c.rawr_cr_attr_accumulate_first(context);
    const start = bench_time.monotonicNanos();
    c.rawr_cr_attr_accumulate_second(context);
    return bench_time.monotonicNanos() - start;
}

fn crSharedPipeline(context: *c.rawr_cr_attr_context) !u64 {
    const start = bench_time.monotonicNanos();
    if (!c.rawr_cr_attr_shared_pipeline(context)) return error.OutOfMemory;
    const elapsed = bench_time.monotonicNanos() - start;
    c.rawr_cr_attr_free_bitsets(context);
    return elapsed;
}

fn crCloneUnique(context: *c.rawr_cr_attr_context) !u64 {
    const start = bench_time.monotonicNanos();
    if (!c.rawr_cr_attr_clone_unique(context)) return error.OutOfMemory;
    const elapsed = bench_time.monotonicNanos() - start;
    c.rawr_cr_attr_free_clones(context);
    return elapsed;
}

fn crMergeAppend(context: *c.rawr_cr_attr_context) !u64 {
    const start = bench_time.monotonicNanos();
    const checksum = c.rawr_cr_attr_merge_append(context);
    const elapsed = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(checksum);
    return elapsed;
}

fn validateReplica(
    left: *const RoaringBitmap,
    right: *const RoaringBitmap,
    cr_left: *const c.roaring_bitmap_t,
    cr_right: *const c.roaring_bitmap_t,
) !void {
    var smp_result = try left.lazyOr(std.heap.smp_allocator, right, true);
    defer smp_result.deinit();
    try smp_result.repairAfterLazy();

    var libc_result = try left.lazyOr(bench_time.cAllocator(), right, true);
    defer libc_result.deinit();
    try libc_result.repairAfterLazy();
    if (!smp_result.equals(&libc_result)) return error.RawrAllocatorMismatch;

    const cr_result = c.roaring_bitmap_lazy_or(cr_left, cr_right, true) orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(cr_result);
    c.roaring_bitmap_repair_after_lazy(cr_result);

    const rawr_bytes = try smp_result.serialize(std.heap.smp_allocator);
    defer std.heap.smp_allocator.free(rawr_bytes);
    const cr_size = c.roaring_bitmap_portable_size_in_bytes(cr_result);
    if (rawr_bytes.len != cr_size) return error.SerializedSizeMismatch;
    const cr_bytes = try std.heap.smp_allocator.alloc(u8, cr_size);
    defer std.heap.smp_allocator.free(cr_bytes);
    if (c.roaring_bitmap_portable_serialize(cr_result, @ptrCast(cr_bytes.ptr)) != cr_size) {
        return error.SerializedSizeMismatch;
    }
    if (!std.mem.eql(u8, rawr_bytes, cr_bytes)) return error.CRoaringMismatch;
}

fn runComponent(comptime operation: anytype, args: anytype, component: []const u8, variant: []const u8) !void {
    printResult(component, variant, try measure(operation, args));
}

export fn rawr_lazy_attr_zero_probe(words: *align(64) [BitsetContainer.NUM_WORDS]u64) callconv(.c) void {
    @memset(words, 0);
}

export fn rawr_lazy_attr_accumulate_probe(
    words: *align(64) [BitsetContainer.NUM_WORDS]u64,
    values: [*]const u16,
    count: usize,
) callconv(.c) void {
    for (values[0..count]) |value| words[value >> 6] |= @as(u64, 1) << @as(u6, @truncate(value));
}

fn preserveCodegenProbes() void {
    var words: [BitsetContainer.NUM_WORDS]u64 align(64) = undefined;
    c.rawr_cr_attr_call_zig_probes(&words, @ptrCast(&words), 0);
    std.mem.doNotOptimizeAway(&words);
}

pub fn main() !void {
    bench_time.print("Lazy-OR construction attribution\n", .{});
    bench_time.print("================================\n", .{});
    bench_time.printRunTimestamp();
    bench_time.printBenchEnvironment();
    bench_time.print("N={d}, warmup={d}, timed={d}\n\n", .{ N_VALUES, WARMUP_RUNS, TIMED_RUNS });

    const sparse_len = initValues();
    var rawr_inputs = try buildRawrInputs(std.heap.smp_allocator, sparse_values[0..sparse_len]);
    defer rawr_inputs[0].deinit();
    defer rawr_inputs[1].deinit();
    const cr_inputs = try buildCrInputs(sparse_values[0..sparse_len]);
    defer c.roaring_bitmap_free(cr_inputs[0]);
    defer c.roaring_bitmap_free(cr_inputs[1]);

    var corpus = try captureCorpus(std.heap.smp_allocator, &rawr_inputs[0], &rawr_inputs[1]);
    defer corpus.deinit(std.heap.smp_allocator);
    const cr_context = c.rawr_cr_attr_context_create(cr_inputs[0], cr_inputs[1]) orelse return error.OutOfMemory;
    defer c.rawr_cr_attr_context_free(cr_context);
    const cr_counts = c.rawr_cr_attr_get_counts(cr_context);

    if (cr_counts.shared_keys != corpus.shared.len or
        cr_counts.left_only_keys != corpus.left_only or
        cr_counts.right_only_keys != corpus.right_only or
        cr_counts.non_array_shared_keys != 0)
    {
        return error.AttributionCorpusMismatch;
    }
    try validateReplica(&rawr_inputs[0], &rawr_inputs[1], cr_inputs[0], cr_inputs[1]);
    preserveCodegenProbes();

    bench_time.print("COUNT\tleft_keys\t{d}\n", .{rawr_inputs[0].size});
    bench_time.print("COUNT\tright_keys\t{d}\n", .{rawr_inputs[1].size});
    bench_time.print("COUNT\tshared_keys\t{d}\n", .{corpus.shared.len});
    bench_time.print("COUNT\tleft_only_keys\t{d}\n", .{corpus.left_only});
    bench_time.print("COUNT\tright_only_keys\t{d}\n", .{corpus.right_only});
    bench_time.print("COUNT\tbitsets_created\t{d}\n", .{cr_counts.bitsets_created});
    bench_time.print("COUNT\tbytes_cleared\t{d}\n", .{cr_counts.bytes_cleared});
    bench_time.print("VALIDATION\trawr-smp=rawr-libc=croaring-portable\n\n", .{});

    bench_time.print("{s:<20} {s:<12} {s:>10} {s:>10} {s:>10}\n", .{ "component", "variant", "median ms", "min ms", "max ms" });
    bench_time.print("{s:-<20} {s:-<12} {s:->10} {s:->10} {s:->10}\n", .{ "", "", "", "", "" });

    const libc = bench_time.cAllocator();
    try runComponent(rawrFull, .{ &rawr_inputs[0], &rawr_inputs[1], std.heap.smp_allocator }, "full", "rawr-smp");
    try runComponent(rawrFull, .{ &rawr_inputs[0], &rawr_inputs[1], libc }, "full", "rawr-libc");
    try runComponent(crFull, .{ cr_inputs[0], cr_inputs[1] }, "full", "croaring");

    try runComponent(rawrFullAfterRepair, .{ &rawr_inputs[0], &rawr_inputs[1], std.heap.smp_allocator }, "full-after-repair", "rawr-smp");
    try runComponent(rawrFullAfterRepair, .{ &rawr_inputs[0], &rawr_inputs[1], libc }, "full-after-repair", "rawr-libc");
    try runComponent(crFullAfterRepair, .{ cr_inputs[0], cr_inputs[1] }, "full-after-repair", "croaring");

    try runComponent(rawrSharedPipeline, .{ std.heap.smp_allocator, corpus.shared }, "shared-pipeline", "rawr-smp");
    try runComponent(rawrSharedPipeline, .{ libc, corpus.shared }, "shared-pipeline", "rawr-libc");
    try runComponent(crSharedPipeline, .{cr_context}, "shared-pipeline", "croaring");

    try runComponent(rawrHeaderAlloc, .{ std.heap.smp_allocator, corpus.shared.len }, "header-alloc", "rawr-smp");
    try runComponent(rawrHeaderAlloc, .{ libc, corpus.shared.len }, "header-alloc", "rawr-libc");
    try runComponent(crHeaders, .{cr_context}, "header-alloc", "croaring");

    try runComponent(rawrWordsAlloc, .{ std.heap.smp_allocator, corpus.shared.len }, "words-alloc", "rawr-smp");
    try runComponent(rawrWordsAlloc, .{ libc, corpus.shared.len }, "words-alloc", "rawr-libc");
    try runComponent(crWords, .{cr_context}, "words-alloc", "croaring");

    try runComponent(rawrBitsetCreate, .{ std.heap.smp_allocator, corpus.shared.len }, "bitset-create", "rawr-smp");
    try runComponent(rawrBitsetCreate, .{ libc, corpus.shared.len }, "bitset-create", "rawr-libc");
    try runComponent(crBitsetCreate, .{cr_context}, "bitset-create", "croaring");

    try runComponent(rawrZeroFresh, .{ std.heap.smp_allocator, corpus.shared.len }, "zero-fresh", "rawr-smp");
    try runComponent(rawrZeroFresh, .{ libc, corpus.shared.len }, "zero-fresh", "rawr-libc");
    try runComponent(crZeroFresh, .{cr_context}, "zero-fresh", "croaring");

    try runComponent(rawrZero, .{ std.heap.smp_allocator, corpus.shared.len }, "zero-dirty", "rawr-smp");
    try runComponent(rawrZero, .{ libc, corpus.shared.len }, "zero-dirty", "rawr-libc");
    try runComponent(crZero, .{cr_context}, "zero-dirty", "croaring");

    try runComponent(rawrAccumulateFirst, .{ std.heap.smp_allocator, corpus.shared }, "accumulate-first", "rawr-smp");
    try runComponent(rawrAccumulateFirst, .{ libc, corpus.shared }, "accumulate-first", "rawr-libc");
    try runComponent(crAccumulateFirst, .{cr_context}, "accumulate-first", "croaring");

    try runComponent(rawrAccumulateSecond, .{ std.heap.smp_allocator, corpus.shared }, "accumulate-second", "rawr-smp");
    try runComponent(rawrAccumulateSecond, .{ libc, corpus.shared }, "accumulate-second", "rawr-libc");
    try runComponent(crAccumulateSecond, .{cr_context}, "accumulate-second", "croaring");

    try runComponent(rawrCloneUnique, .{ std.heap.smp_allocator, corpus.unique }, "clone-unique", "rawr-smp");
    try runComponent(rawrCloneUnique, .{ libc, corpus.unique }, "clone-unique", "rawr-libc");
    try runComponent(crCloneUnique, .{cr_context}, "clone-unique", "croaring");

    try runComponent(rawrMergeAppend, .{ &rawr_inputs[0], &rawr_inputs[1] }, "merge-append", "rawr-smp");
    try runComponent(rawrMergeAppend, .{ &rawr_inputs[0], &rawr_inputs[1] }, "merge-append", "rawr-libc");
    try runComponent(crMergeAppend, .{cr_context}, "merge-append", "croaring");
}
