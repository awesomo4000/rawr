// SPDX-License-Identifier: MPL-2.0

//! Repository-only orMany source attribution and word-major fusion diagnostic.

const std = @import("std");
const c = @import("c");
const rawr = @import("rawr");
const bench_time = @import("bench_time.zig");
const bench_corpus = @import("bench_corpus.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const ArrayContainer = rawr.ArrayContainer;
const BitsetContainer = rawr.BitsetContainer;
const Container = rawr.Container;
const TaggedPtr = rawr.TaggedPtr;

const warmup_runs = 3;
const timed_runs = 21;
const process_repetitions = 5;
const batch_count = 128;
const many_bitmap_count = bench_corpus.many_bitmap_count;
const canonical_key_count = bench_corpus.many_key_count;
const projection_gate_ppm = 1_100_000;
const vector_width = 8;

const SourceKind = enum {
    array,
    bitset,
    run,
};

const Strategy = enum {
    word_major,
    seed_word_major,

    fn id(self: Strategy) []const u8 {
        return switch (self) {
            .word_major => "word-major",
            .seed_word_major => "seed-word-major",
        };
    }
};

const AccumulationShape = enum {
    input_major,
    first_bitset_seed,
    word_major,
    seed_word_major,
};

const Case = enum {
    attribution_array,
    attribution_bitset,
    attribution_run,
    cell1_baseline,
    cell2_first_bitset_seed,
    cell3_word_major,
    cell4_seed_word_major,
    cell5_ceiling_baseline,
    cell5_ceiling_word_major,
    cell5_ceiling_seed_word_major,
    full_rawr,
    full_croaring,
    full_candidate_word_major,
    full_candidate_seed_word_major,

    fn id(self: Case) []const u8 {
        return switch (self) {
            .attribution_array => "attribution-array",
            .attribution_bitset => "attribution-bitset",
            .attribution_run => "attribution-run",
            .cell1_baseline => "cell1-baseline",
            .cell2_first_bitset_seed => "cell2-first-bitset-seed",
            .cell3_word_major => "cell3-word-major",
            .cell4_seed_word_major => "cell4-seed-word-major",
            .cell5_ceiling_baseline => "cell5-bitset-ceiling-baseline",
            .cell5_ceiling_word_major => "cell5-bitset-ceiling-word-major",
            .cell5_ceiling_seed_word_major => "cell5-bitset-ceiling-seed-word-major",
            .full_rawr => "full-rawr",
            .full_croaring => "full-croaring",
            .full_candidate_word_major => "full-candidate-word-major",
            .full_candidate_seed_word_major => "full-candidate-seed-word-major",
        };
    }

    fn cellNumber(self: Case) u8 {
        return switch (self) {
            .attribution_array, .attribution_bitset, .attribution_run => 0,
            .cell1_baseline => 1,
            .cell2_first_bitset_seed => 2,
            .cell3_word_major => 3,
            .cell4_seed_word_major => 4,
            .cell5_ceiling_baseline,
            .cell5_ceiling_word_major,
            .cell5_ceiling_seed_word_major,
            => 5,
            .full_rawr,
            .full_croaring,
            .full_candidate_word_major,
            .full_candidate_seed_word_major,
            => 6,
        };
    }

    fn scope(self: Case) []const u8 {
        return switch (self) {
            .attribution_array, .attribution_bitset, .attribution_run => "source-attribution",
            .cell1_baseline,
            .cell2_first_bitset_seed,
            .cell3_word_major,
            .cell4_seed_word_major,
            => "mixed-accumulation",
            .cell5_ceiling_baseline,
            .cell5_ceiling_word_major,
            .cell5_ceiling_seed_word_major,
            => "bitset-only",
            .full_rawr, .full_croaring => "canonical-full-row",
            .full_candidate_word_major, .full_candidate_seed_word_major => "candidate-full-row",
        };
    }
};

const expected_type_counts = bench_corpus.expected_many_type_counts;

const CandidateDiagnostics = struct {
    scratch_allocations: usize = 0,
    scratch_capacity: usize = 0,
    keys_folded: usize = 0,
    unknown_accumulators_before_repair: usize = 0,
};

const Context = struct {
    allocator: std.mem.Allocator,
    bitmaps: [many_bitmap_count]?RoaringBitmap = [_]?RoaringBitmap{null} ** many_bitmap_count,
    inputs: [many_bitmap_count]*const RoaringBitmap = undefined,
    croaring: [many_bitmap_count]?*c.roaring_bitmap_t = [_]?*c.roaring_bitmap_t{null} ** many_bitmap_count,
    croaring_inputs: [many_bitmap_count]*c.roaring_bitmap_t = undefined,

    fn init(allocator: std.mem.Allocator) !Context {
        var self = Context{ .allocator = allocator };
        errdefer self.deinit();
        try bench_corpus.initRawrManyBitmaps(allocator, &self.bitmaps, &self.inputs);
        return self;
    }

    fn bindInputs(self: *Context) !void {
        for (0..many_bitmap_count) |index| {
            self.inputs[index] = &self.bitmaps[index].?;

            const bytes = try self.inputs[index].serialize(std.heap.page_allocator);
            defer std.heap.page_allocator.free(bytes);
            const bitmap = c.roaring_bitmap_portable_deserialize_safe(
                @ptrCast(bytes.ptr),
                bytes.len,
            ) orelse return error.CRoaringDeserializeFailed;
            self.croaring[index] = bitmap;
            self.croaring_inputs[index] = bitmap;
        }
    }

    fn deinit(self: *Context) void {
        for (&self.bitmaps) |*maybe_bitmap| {
            if (maybe_bitmap.*) |*bitmap| bitmap.deinit();
            maybe_bitmap.* = null;
        }
        for (&self.croaring) |*maybe_bitmap| {
            if (maybe_bitmap.*) |bitmap| c.roaring_bitmap_free(bitmap);
            maybe_bitmap.* = null;
        }
    }
};

const Measurement = struct {
    median_ns: u64,
    checksum: u64,
};

const TimedBatch = struct {
    elapsed_ns: u64,
    checksum: u64,
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var header = false;
    var suite = false;
    var validate_only = false;
    var selected_case: ?Case = null;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.eql(u8, arg, "--suite")) {
            suite = true;
        } else if (std.mem.eql(u8, arg, "--validate")) {
            validate_only = true;
        } else if (std.mem.startsWith(u8, arg, "--case=")) {
            selected_case = parseCase(arg[7..]) orelse return error.UnknownCase;
        } else {
            return error.UnknownArgument;
        }
    }

    const selected_modes = @as(u8, @intFromBool(header)) +
        @as(u8, @intFromBool(suite)) +
        @as(u8, @intFromBool(validate_only)) +
        @as(u8, @intFromBool(selected_case != null));
    if (selected_modes != 1) return error.ChooseExactlyOneMode;

    if (header) {
        printHeader();
        return;
    }

    var context = try Context.init(std.heap.smp_allocator);
    defer context.deinit();
    try context.bindInputs();
    const fingerprint = try assertCanonicalFingerprint(&context);
    printFingerprint(fingerprint);
    try validateAll(&context);
    bench_time.print("VALIDATION\tcanonical-and-edges\tpass\n", .{});

    if (validate_only) return;
    if (suite) {
        try runSuite(&context);
        return;
    }

    const measurement = measureCase(selected_case.?, &context);
    printMeasurement(selected_case.?, measurement);
}

fn parseCase(value: []const u8) ?Case {
    inline for (std.meta.fields(Case)) |field| {
        const selected: Case = @enumFromInt(field.value);
        if (std.mem.eql(u8, value, selected.id())) return selected;
    }
    return null;
}

fn printHeader() void {
    bench_time.printBenchEnvironment();
    bench_time.print("SCHEMA\tor-many-fusion-v1\n", .{});
    bench_time.print("PROTOCOL\t{d}\t{d}\t{d}\t{d}\tsmp\tnative\n", .{
        warmup_runs,
        timed_runs,
        process_repetitions,
        batch_count,
    });
    bench_time.print("ORDER\tattribution\tcells-1-4\tcell-5\tfull-row\tprojection\tcandidate-if-go\n", .{});
    bench_time.print("CROARING_AVX512\t{s}\n", .{
        if (c.CROARING_COMPILER_SUPPORTS_AVX512 != 0) "on" else "off",
    });
}

fn runSuite(context: *Context) !void {
    const attribution_array = measureAndPrint(.attribution_array, context);
    const attribution_bitset = measureAndPrint(.attribution_bitset, context);
    const attribution_run = measureAndPrint(.attribution_run, context);

    const cell1 = measureAndPrint(.cell1_baseline, context);
    _ = measureAndPrint(.cell2_first_bitset_seed, context);
    const cell3 = measureAndPrint(.cell3_word_major, context);
    const cell4 = measureAndPrint(.cell4_seed_word_major, context);

    const ceiling_baseline = measureAndPrint(.cell5_ceiling_baseline, context);
    const ceiling_word_major = measureAndPrint(.cell5_ceiling_word_major, context);
    const ceiling_seed_word_major = measureAndPrint(.cell5_ceiling_seed_word_major, context);

    const full_rawr = measureAndPrint(.full_rawr, context);
    const full_croaring = measureAndPrint(.full_croaring, context);

    const attribution_total = attribution_array.median_ns +
        attribution_bitset.median_ns + attribution_run.median_ns;
    if (attribution_total == 0 or ceiling_baseline.median_ns == 0 or
        full_croaring.median_ns == 0)
    {
        return error.InvalidZeroMeasurement;
    }
    const bitset_share_ppm = ratioPpm(attribution_bitset.median_ns, attribution_total);

    const strategy: Strategy = if (cell4.median_ns < cell3.median_ns)
        .seed_word_major
    else
        .word_major;
    const ceiling_candidate = switch (strategy) {
        .word_major => ceiling_word_major,
        .seed_word_major => ceiling_seed_word_major,
    };
    const ceiling_improvement_ppm = improvementPpm(
        ceiling_baseline.median_ns,
        ceiling_candidate.median_ns,
    );
    const recoverable_ppm = mulPpm(bitset_share_ppm, ceiling_improvement_ppm);
    const projected_ns = applyImprovement(full_rawr.median_ns, recoverable_ppm);
    const baseline_ratio_ppm = ratioPpm(full_rawr.median_ns, full_croaring.median_ns);
    const projected_ratio_ppm = ratioPpm(projected_ns, full_croaring.median_ns);

    bench_time.print("ATTRIBUTION\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        attribution_array.median_ns,
        attribution_bitset.median_ns,
        attribution_run.median_ns,
        attribution_total,
        bitset_share_ppm,
    });
    bench_time.print("WINNER\t{s}\t{d}\t{d}\n", .{
        strategy.id(),
        cell1.median_ns,
        @min(cell3.median_ns, cell4.median_ns),
    });
    bench_time.print("CEILING\t{s}\t{d}\t{d}\t{d}\n", .{
        strategy.id(),
        ceiling_baseline.median_ns,
        ceiling_candidate.median_ns,
        ceiling_improvement_ppm,
    });
    bench_time.print("PROJECTION\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        baseline_ratio_ppm,
        bitset_share_ppm,
        ceiling_improvement_ppm,
        recoverable_ppm,
        projected_ns,
        projected_ratio_ppm,
    });

    if (projected_ratio_ppm > projection_gate_ppm) {
        bench_time.print("DECISION\tprojection-no-go\t{d}\t{d}\n", .{
            projected_ratio_ppm,
            projection_gate_ppm,
        });
        bench_time.print("DIRECT\tskipped\tprojection-gate\n", .{});
        return;
    }

    bench_time.print("DECISION\tprojection-go\t{d}\t{d}\n", .{
        projected_ratio_ppm,
        projection_gate_ppm,
    });
    const candidate_case: Case = switch (strategy) {
        .word_major => .full_candidate_word_major,
        .seed_word_major => .full_candidate_seed_word_major,
    };
    const direct = measureAndPrint(candidate_case, context);
    const direct_ratio_ppm = ratioPpm(direct.median_ns, full_croaring.median_ns);
    bench_time.print("DIRECT\t{s}\t{d}\t{d}\t{s}\n", .{
        strategy.id(),
        direct.median_ns,
        direct_ratio_ppm,
        if (direct_ratio_ppm <= projection_gate_ppm) "go-33-01" else "no-go",
    });
}

fn measureAndPrint(selected: Case, context: *Context) Measurement {
    const measurement = measureCase(selected, context);
    printMeasurement(selected, measurement);
    return measurement;
}

fn printMeasurement(selected: Case, measurement: Measurement) void {
    bench_time.print("RESULT\t{s}\t{d}\t{s}\tns/batch\t{d}\t{d}\t{d}\n", .{
        selected.id(),
        selected.cellNumber(),
        selected.scope(),
        batch_count,
        measurement.median_ns,
        measurement.checksum,
    });
}

fn measureCase(selected: Case, context: *Context) Measurement {
    var checksum: u64 = 0;
    for (0..warmup_runs) |_| checksum +%= runBatch(selected, context).checksum;

    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const batch = runBatch(selected, context);
        elapsed.* = batch.elapsed_ns;
        checksum +%= batch.checksum;
        std.mem.doNotOptimizeAway(checksum);
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return .{ .median_ns = times[timed_runs / 2], .checksum = checksum };
}

fn runBatch(selected: Case, context: *Context) TimedBatch {
    return switch (selected) {
        .attribution_array => timeAttributionBatch(context, .array),
        .attribution_bitset => timeAttributionBatch(context, .bitset),
        .attribution_run => timeAttributionBatch(context, .run),
        else => result: {
            var checksum: u64 = 0;
            const start = bench_time.monotonicNanos();
            for (0..batch_count) |_| checksum +%= runOne(selected, context);
            const elapsed = bench_time.monotonicNanos() - start;
            std.mem.doNotOptimizeAway(checksum);
            break :result .{ .elapsed_ns = elapsed, .checksum = checksum };
        },
    };
}

noinline fn runOne(selected: Case, context: *Context) u64 {
    return switch (selected) {
        .attribution_array, .attribution_bitset, .attribution_run => unreachable,
        .cell1_baseline => runAccumulationCell(context, .input_major, false),
        .cell2_first_bitset_seed => runAccumulationCell(context, .first_bitset_seed, false),
        .cell3_word_major => runAccumulationCell(context, .word_major, false),
        .cell4_seed_word_major => runAccumulationCell(context, .seed_word_major, false),
        .cell5_ceiling_baseline => runAccumulationCell(context, .input_major, true),
        .cell5_ceiling_word_major => runAccumulationCell(context, .word_major, true),
        .cell5_ceiling_seed_word_major => runAccumulationCell(context, .seed_word_major, true),
        .full_rawr => runFullRawr(context),
        .full_croaring => runFullCRoaring(context),
        .full_candidate_word_major => runFullCandidate(context, .word_major),
        .full_candidate_seed_word_major => runFullCandidate(context, .seed_word_major),
    };
}

fn timeAttributionBatch(context: *Context, source_kind: SourceKind) TimedBatch {
    var destinations: [canonical_key_count]*BitsetContainer = undefined;
    var initialized: usize = 0;
    defer for (destinations[0..initialized]) |destination| destination.deinit(context.allocator);
    for (&destinations) |*destination| {
        destination.* = BitsetContainer.init(context.allocator) catch unreachable;
        initialized += 1;
    }

    const start = bench_time.monotonicNanos();
    for (0..batch_count) |_| {
        for (0..canonical_key_count) |key| {
            for (context.inputs) |bitmap| {
                const container = Container.fromTagged(bitmap.containers[key]);
                if (containerSourceKind(container) == source_kind) {
                    accumulateContainer(destinations[key], container);
                }
            }
        }
    }
    const elapsed = bench_time.monotonicNanos() - start;

    var checksum: u64 = @intFromEnum(source_kind) + 1;
    for (destinations, 0..) |destination, key| {
        checksum +%= @as(u64, destination.computeCardinality()) *% (key + 1);
    }
    std.mem.doNotOptimizeAway(checksum);
    return .{ .elapsed_ns = elapsed, .checksum = checksum };
}

fn runAccumulationCell(
    context: *Context,
    shape: AccumulationShape,
    bitsets_only: bool,
) u64 {
    const needs_scratch = shape == .word_major or shape == .seed_word_major;
    var scratch: []*const BitsetContainer = undefined;
    if (needs_scratch) scratch = context.allocator.alloc(*const BitsetContainer, context.inputs.len) catch unreachable;
    defer if (needs_scratch) context.allocator.free(scratch);

    var checksum: u64 = 1;
    for (0..canonical_key_count) |key| {
        const accumulator = switch (shape) {
            .input_major => BitsetContainer.init(context.allocator) catch unreachable,
            .first_bitset_seed => seedFirstBitset(context.inputs[0..], key, context.allocator) catch unreachable,
            .word_major => BitsetContainer.init(context.allocator) catch unreachable,
            .seed_word_major => seedFirstBitset(context.inputs[0..], key, context.allocator) catch unreachable,
        };
        defer accumulator.deinit(context.allocator);

        switch (shape) {
            .input_major => for (context.inputs) |bitmap| {
                const container = Container.fromTagged(bitmap.containers[key]);
                if (!bitsets_only or container == .bitset) accumulateContainer(accumulator, container);
            },
            .first_bitset_seed => {
                var skipped_seed = false;
                for (context.inputs) |bitmap| {
                    const container = Container.fromTagged(bitmap.containers[key]);
                    if (container == .bitset and !skipped_seed) {
                        skipped_seed = true;
                        continue;
                    }
                    if (!bitsets_only or container == .bitset) accumulateContainer(accumulator, container);
                }
            },
            .word_major, .seed_word_major => {
                var bitset_count: usize = 0;
                var skipped_seed = shape != .seed_word_major;
                for (context.inputs) |bitmap| {
                    const container = Container.fromTagged(bitmap.containers[key]);
                    switch (container) {
                        .bitset => |bitset| {
                            if (!skipped_seed) {
                                skipped_seed = true;
                            } else {
                                scratch[bitset_count] = bitset;
                                bitset_count += 1;
                            }
                        },
                        .array, .run => if (!bitsets_only) accumulateContainer(accumulator, container),
                        .reserved => unreachable,
                    }
                }
                wordMajorOr(accumulator, scratch[0..bitset_count]);
            },
        }

        checksum +%= accumulator.computeCardinality();
        checksum +%= accumulator.words[(key * 149) & (BitsetContainer.NUM_WORDS - 1)];
    }
    return checksum;
}

fn seedFirstBitset(
    inputs: []const *const RoaringBitmap,
    key_index: usize,
    allocator: std.mem.Allocator,
) !*BitsetContainer {
    for (inputs) |bitmap| {
        switch (Container.fromTagged(bitmap.containers[key_index])) {
            .bitset => |bitset| return bitset.clone(allocator),
            .array, .run => {},
            .reserved => unreachable,
        }
    }
    return BitsetContainer.init(allocator);
}

fn wordMajorOr(destination: *BitsetContainer, sources: []const *const BitsetContainer) void {
    var word_index: usize = 0;
    while (word_index < BitsetContainer.NUM_WORDS) : (word_index += vector_width) {
        var accumulated: @Vector(vector_width, u64) = destination.words[word_index..][0..vector_width].*;
        for (sources) |source| {
            const words: @Vector(vector_width, u64) = source.words[word_index..][0..vector_width].*;
            accumulated |= words;
        }
        destination.words[word_index..][0..vector_width].* = accumulated;
    }
    destination.cardinality = -1;
}

fn accumulateContainer(destination: *BitsetContainer, container: Container) void {
    switch (container) {
        .array => |array| destination.setList(array.values[0..array.cardinality]),
        .bitset => |bitset| destination.lazyUnionWith(bitset),
        .run => |run| {
            for (run.runs[0..run.n_runs]) |pair| destination.setRange(pair.start, pair.end());
            destination.cardinality = -1;
        },
        .reserved => unreachable,
    }
}

fn containerSourceKind(container: Container) SourceKind {
    return switch (container) {
        .array => .array,
        .bitset => .bitset,
        .run => .run,
        .reserved => unreachable,
    };
}

fn runFullRawr(context: *Context) u64 {
    var result = RoaringBitmap.orMany(context.allocator, &context.inputs) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
    return result.cardinality() +% result.size +% 1;
}

fn runFullCRoaring(context: *Context) u64 {
    const result = c.roaring_bitmap_or_many(
        context.croaring_inputs.len,
        @ptrCast(&context.croaring_inputs),
    ) orelse unreachable;
    defer c.roaring_bitmap_free(result);
    std.mem.doNotOptimizeAway(result);
    return c.roaring_bitmap_get_cardinality(result) +% 1;
}

fn runFullCandidate(context: *Context, strategy: Strategy) u64 {
    var result = orManyCandidate(context.allocator, &context.inputs, strategy, null) catch unreachable;
    defer result.deinit();
    std.mem.doNotOptimizeAway(&result);
    return result.cardinality() +% result.size +% 1;
}

fn orManyCandidate(
    allocator: std.mem.Allocator,
    bitmaps: []const *const RoaringBitmap,
    strategy: Strategy,
    diagnostics: ?*CandidateDiagnostics,
) !RoaringBitmap {
    if (bitmaps.len == 1) return bitmaps[0].clone(allocator);

    var capacity: usize = 0;
    for (bitmaps) |bitmap| capacity = @min(capacity +| bitmap.size, 1 << 16);
    var result = try RoaringBitmap.initCapacity(allocator, @intCast(capacity));
    errdefer result.deinit();
    if (bitmaps.len == 0) return result;

    const cursors = try allocator.alloc(usize, bitmaps.len);
    defer allocator.free(cursors);
    @memset(cursors, 0);

    const bitset_pointers = try allocator.alloc(*const BitsetContainer, bitmaps.len);
    defer allocator.free(bitset_pointers);
    if (diagnostics) |stats| {
        stats.scratch_allocations += 1;
        stats.scratch_capacity = bitset_pointers.len;
    }

    while (nextManyKey(bitmaps, cursors)) |key| {
        const tagged = try foldCandidateKey(
            allocator,
            bitmaps,
            cursors,
            key,
            bitset_pointers,
            strategy,
            diagnostics,
        );
        if (result.size >= result.capacity) {
            Container.fromTagged(tagged).deinit(allocator);
            return error.InvalidCapacityBound;
        }
        result.keys[result.size] = key;
        result.containers[result.size] = tagged;
        result.size += 1;
        if (diagnostics) |stats| stats.keys_folded += 1;
    }

    result.cached_cardinality = -1;
    try result.repairAfterLazy();
    return result;
}

fn nextManyKey(bitmaps: []const *const RoaringBitmap, cursors: []const usize) ?u16 {
    var minimum: ?u16 = null;
    for (bitmaps, cursors) |bitmap, cursor| {
        if (cursor >= bitmap.size) continue;
        const key = bitmap.keys[cursor];
        if (minimum == null or key < minimum.?) minimum = key;
    }
    return minimum;
}

fn foldCandidateKey(
    allocator: std.mem.Allocator,
    bitmaps: []const *const RoaringBitmap,
    cursors: []usize,
    key: u16,
    bitset_pointers: []*const BitsetContainer,
    strategy: Strategy,
    diagnostics: ?*CandidateDiagnostics,
) !TaggedPtr {
    var source_count: usize = 0;
    for (bitmaps, cursors) |bitmap, cursor| {
        if (cursor < bitmap.size and bitmap.keys[cursor] == key) source_count += 1;
    }

    if (source_count == 1) {
        for (bitmaps, cursors) |bitmap, *cursor| {
            if (cursor.* >= bitmap.size or bitmap.keys[cursor.*] != key) continue;
            const cloned = try Container.fromTagged(bitmap.containers[cursor.*]).clone(allocator);
            cursor.* += 1;
            return cloned.toTagged();
        }
        unreachable;
    }

    var bitset_count: usize = 0;
    for (bitmaps, cursors) |bitmap, cursor| {
        if (cursor >= bitmap.size or bitmap.keys[cursor] != key) continue;
        switch (Container.fromTagged(bitmap.containers[cursor])) {
            .bitset => |bitset| {
                bitset_pointers[bitset_count] = bitset;
                bitset_count += 1;
            },
            .array, .run => {},
            .reserved => unreachable,
        }
    }

    const use_seed = bitset_count == 1 or
        (bitset_count >= 2 and strategy == .seed_word_major);
    const accumulator = if (!use_seed)
        try BitsetContainer.init(allocator)
    else
        try bitset_pointers[0].clone(allocator);
    errdefer accumulator.deinit(allocator);

    if (bitset_count != 0) {
        const first: usize = if (use_seed) 1 else 0;
        wordMajorOr(accumulator, bitset_pointers[first..bitset_count]);
    }
    for (bitmaps, cursors) |bitmap, *cursor| {
        if (cursor.* >= bitmap.size or bitmap.keys[cursor.*] != key) continue;
        const container = Container.fromTagged(bitmap.containers[cursor.*]);
        if (container != .bitset) accumulateContainer(accumulator, container);
        cursor.* += 1;
    }
    accumulator.cardinality = -1;
    if (diagnostics) |stats| stats.unknown_accumulators_before_repair += 1;
    return TaggedPtr.initBitset(accumulator);
}

fn assertCanonicalFingerprint(context: *const Context) !u64 {
    return bench_corpus.assertRawrManyFingerprint(std.heap.page_allocator, &context.inputs);
}

fn printFingerprint(hash: u64) void {
    bench_time.print("CORPUS\t{d}\t{d}\t{x}\n", .{
        many_bitmap_count,
        canonical_key_count,
        hash,
    });
    for (expected_type_counts, 0..) |counts, key| {
        bench_time.print("FINGERPRINT\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
            key,
            counts.array,
            counts.bitset,
            counts.run,
            counts.total(),
        });
    }
    bench_time.print("MULTIPLICITY\tbitset-sources-per-key\t8\t6\n", .{});
}

fn validateAll(context: *Context) !void {
    var snapshots: [many_bitmap_count][]u8 = undefined;
    var initialized: usize = 0;
    defer for (snapshots[0..initialized]) |bytes| std.heap.page_allocator.free(bytes);
    for (context.inputs, 0..) |bitmap, index| {
        snapshots[index] = try bitmap.serialize(std.heap.page_allocator);
        initialized += 1;
    }

    var baseline = try RoaringBitmap.orMany(context.allocator, &context.inputs);
    defer baseline.deinit();
    try assertKnownCardinalities(&baseline);
    try baseline.validate();

    inline for (.{ Strategy.word_major, Strategy.seed_word_major }) |strategy| {
        var diagnostics = CandidateDiagnostics{};
        var candidate = try orManyCandidate(
            context.allocator,
            &context.inputs,
            strategy,
            &diagnostics,
        );
        defer candidate.deinit();
        if (diagnostics.scratch_allocations != 1 or
            diagnostics.scratch_capacity != context.inputs.len or
            diagnostics.keys_folded != canonical_key_count or
            diagnostics.unknown_accumulators_before_repair != canonical_key_count)
        {
            return error.CandidateDiagnosticsMismatch;
        }
        try assertKnownCardinalities(&candidate);
        try expectEquivalent(&baseline, &candidate, true);
        try expectCRoaringEquivalent(context, &candidate);
    }

    try validateEdgeCases(context.allocator);

    for (context.inputs, snapshots) |bitmap, before| {
        const after = try bitmap.serialize(std.heap.page_allocator);
        defer std.heap.page_allocator.free(after);
        if (!std.mem.eql(u8, before, after)) return error.InputMutated;
    }
    bench_time.print("SCRATCH\t1\t{d}\treset-per-key\tfreed-once\n", .{context.inputs.len});
}

fn assertKnownCardinalities(bitmap: *const RoaringBitmap) !void {
    if (bitmap.cached_cardinality < 0) return error.UnknownBitmapCardinality;
    for (bitmap.containers[0..bitmap.size]) |tagged| {
        switch (Container.fromTagged(tagged)) {
            .array => {},
            .bitset => |bitset| if (bitset.cardinality < 0) return error.UnknownContainerCardinality,
            .run => |run| if (run.cardinality < 0) return error.UnknownContainerCardinality,
            .reserved => unreachable,
        }
    }
}

fn expectEquivalent(
    expected: *const RoaringBitmap,
    actual: *const RoaringBitmap,
    require_same_kind: bool,
) !void {
    try expected.validate();
    try actual.validate();
    if (!expected.equals(actual) or expected.cardinality() != actual.cardinality()) {
        return error.ValueMismatch;
    }
    if (expected.size != actual.size) return error.ContainerCountMismatch;
    for (0..expected.size) |index| {
        if (expected.keys[index] != actual.keys[index]) return error.KeyMismatch;
        if (require_same_kind and
            expected.containers[index].getType() != actual.containers[index].getType())
        {
            return error.ContainerKindMismatch;
        }
        if (Container.fromTagged(expected.containers[index]).getCardinality() !=
            Container.fromTagged(actual.containers[index]).getCardinality())
        {
            return error.ContainerCardinalityMismatch;
        }
    }

    const expected_values = try expected.toArrayAlloc(std.heap.page_allocator);
    defer std.heap.page_allocator.free(expected_values);
    const actual_values = try actual.toArrayAlloc(std.heap.page_allocator);
    defer std.heap.page_allocator.free(actual_values);
    if (!std.mem.eql(u32, expected_values, actual_values)) return error.ArrayValueMismatch;

    const expected_bytes = try expected.serialize(std.heap.page_allocator);
    defer std.heap.page_allocator.free(expected_bytes);
    const actual_bytes = try actual.serialize(std.heap.page_allocator);
    defer std.heap.page_allocator.free(actual_bytes);
    if (!std.mem.eql(u8, expected_bytes, actual_bytes)) return error.PortableBytesMismatch;
}

fn expectCRoaringEquivalent(context: *Context, actual: *const RoaringBitmap) !void {
    const expected = c.roaring_bitmap_or_many(
        context.croaring_inputs.len,
        @ptrCast(&context.croaring_inputs),
    ) orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(expected);

    const actual_bytes = try actual.serialize(std.heap.page_allocator);
    defer std.heap.page_allocator.free(actual_bytes);
    const expected_len = c.roaring_bitmap_portable_size_in_bytes(expected);
    if (actual_bytes.len != expected_len) return error.CRoaringSizeMismatch;
    const expected_bytes = try std.heap.page_allocator.alloc(u8, expected_len);
    defer std.heap.page_allocator.free(expected_bytes);
    if (c.roaring_bitmap_portable_serialize(expected, @ptrCast(expected_bytes.ptr)) != expected_len) {
        return error.CRoaringSizeMismatch;
    }
    if (!std.mem.eql(u8, actual_bytes, expected_bytes)) return error.CRoaringMismatch;
}

fn validateEdgeCases(allocator: std.mem.Allocator) !void {
    try validateCandidateCase(allocator, &.{}, null);

    var single = try RoaringBitmap.init(allocator);
    defer single.deinit();
    _ = try single.add(7);
    _ = try single.addRange(100, 220);
    try validateCandidateCase(allocator, &.{&single}, null);
    try validateCandidateCase(allocator, &.{ &single, &single }, null);

    var sparse = try RoaringBitmap.init(allocator);
    defer sparse.deinit();
    for (0..128) |index| _ = try sparse.add(@intCast(index * 31));
    var run = try RoaringBitmap.init(allocator);
    defer run.deinit();
    _ = try run.addRange(2_000, 3_000);
    try validateCandidateCase(allocator, &.{ &sparse, &run }, null);

    var dense_a = try RoaringBitmap.init(allocator);
    defer dense_a.deinit();
    for (0..5000) |index| _ = try dense_a.add(@intCast(index));
    try validateCandidateCase(allocator, &.{ &dense_a, &sparse }, .bitset);

    var dense_b = try RoaringBitmap.init(allocator);
    defer dense_b.deinit();
    for (1000..6000) |index| _ = try dense_b.add(@intCast(index));
    try validateCandidateCase(allocator, &.{ &dense_a, &dense_b }, .bitset);
    try validateCandidateCase(allocator, &.{ &dense_a, &dense_b, &sparse, &run }, .bitset);

    var boundary_even = try RoaringBitmap.init(allocator);
    defer boundary_even.deinit();
    var boundary_odd = try RoaringBitmap.init(allocator);
    defer boundary_odd.deinit();
    for (0..2048) |index| {
        _ = try boundary_even.add(@intCast(index * 2));
        _ = try boundary_odd.add(@intCast(index * 2 + 1));
    }
    try validateCandidateCase(allocator, &.{ &boundary_even, &boundary_odd }, .array);
    _ = try boundary_even.add(4096);
    try validateCandidateCase(allocator, &.{ &boundary_even, &boundary_odd }, .bitset);
}

fn validateCandidateCase(
    allocator: std.mem.Allocator,
    inputs: []const *const RoaringBitmap,
    expected_kind: ?TaggedPtr.ContainerType,
) !void {
    const snapshots = try std.heap.page_allocator.alloc([]u8, inputs.len);
    defer std.heap.page_allocator.free(snapshots);
    var initialized: usize = 0;
    defer for (snapshots[0..initialized]) |bytes| std.heap.page_allocator.free(bytes);
    for (inputs, 0..) |bitmap, index| {
        snapshots[index] = try bitmap.serialize(std.heap.page_allocator);
        initialized += 1;
    }

    var baseline = try RoaringBitmap.orMany(allocator, inputs);
    defer baseline.deinit();
    if (expected_kind) |kind| {
        if (baseline.size != 1 or baseline.containers[0].getType() != kind) {
            return error.EdgeRepresentationMismatch;
        }
    }

    inline for (.{ Strategy.word_major, Strategy.seed_word_major }) |strategy| {
        var candidate = try orManyCandidate(allocator, inputs, strategy, null);
        defer candidate.deinit();
        try expectEquivalent(&baseline, &candidate, true);
    }

    for (inputs, snapshots) |bitmap, before| {
        const after = try bitmap.serialize(std.heap.page_allocator);
        defer std.heap.page_allocator.free(after);
        if (!std.mem.eql(u8, before, after)) return error.EdgeInputMutated;
    }
}

fn ratioPpm(numerator: u64, denominator: u64) u64 {
    std.debug.assert(denominator != 0);
    return @intCast((@as(u128, numerator) * 1_000_000 + denominator / 2) / denominator);
}

fn improvementPpm(baseline: u64, candidate: u64) u64 {
    if (candidate >= baseline) return 0;
    return ratioPpm(baseline - candidate, baseline);
}

fn mulPpm(a: u64, b: u64) u64 {
    return @intCast((@as(u128, a) * b + 500_000) / 1_000_000);
}

fn applyImprovement(value: u64, improvement_ppm: u64) u64 {
    return @intCast((@as(u128, value) * (1_000_000 - improvement_ppm) + 500_000) / 1_000_000);
}

test "orMany candidate is leak-free across allocation failures" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        candidateAllocationFailureCase,
        .{},
    );
}

fn candidateAllocationFailureCase(allocator: std.mem.Allocator) !void {
    var a = try makeFailureInput(0);
    defer a.deinit();
    var b = try makeFailureInput(1);
    defer b.deinit();
    const inputs = [_]*const RoaringBitmap{ &a, &b };

    const before_a = try a.serialize(std.testing.allocator);
    defer std.testing.allocator.free(before_a);
    const before_b = try b.serialize(std.testing.allocator);
    defer std.testing.allocator.free(before_b);

    var result = orManyCandidate(allocator, &inputs, .word_major, null) catch |err| {
        try expectSerializedUnchanged(&a, before_a);
        try expectSerializedUnchanged(&b, before_b);
        return err;
    };
    defer result.deinit();
    try expectSerializedUnchanged(&a, before_a);
    try expectSerializedUnchanged(&b, before_b);
}

test "orMany candidate scratch and partial assembly OOM are clean" {
    var a = try makeFailureInput(0);
    defer a.deinit();
    var b = try makeFailureInput(1);
    defer b.deinit();
    const inputs = [_]*const RoaringBitmap{ &a, &b };

    var scratch_failure = std.testing.FailingAllocator.init(
        std.testing.allocator,
        .{ .fail_index = 3 },
    );
    try std.testing.expectError(
        error.OutOfMemory,
        orManyCandidate(scratch_failure.allocator(), &inputs, .word_major, null),
    );

    var partial_failure = std.testing.FailingAllocator.init(
        std.testing.allocator,
        .{ .fail_index = 6 },
    );
    try std.testing.expectError(
        error.OutOfMemory,
        orManyCandidate(partial_failure.allocator(), &inputs, .word_major, null),
    );
}

fn makeFailureInput(offset: u32) !RoaringBitmap {
    var bitmap = try RoaringBitmap.init(std.testing.allocator);
    errdefer bitmap.deinit();
    for (0..5000) |index| _ = try bitmap.add(@intCast(index + offset));
    for (0..128) |index| _ = try bitmap.add((@as(u32, 1) << 16) | @as(u32, @intCast(index * 31 + offset)));
    return bitmap;
}

fn expectSerializedUnchanged(bitmap: *const RoaringBitmap, before: []const u8) !void {
    const after = try bitmap.serialize(std.testing.allocator);
    defer std.testing.allocator.free(after);
    try std.testing.expectEqualSlices(u8, before, after);
}
