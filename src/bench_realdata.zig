// SPDX-License-Identifier: MPL-2.0

//! Fresh-process rawr/CRoaring worker for pinned real-data corpora.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const corpus_mod = @import("realdata_corpus.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const warmup_runs = 1;
const timed_runs = 7;
const default_root = "misc/realdata";

const Implementation = enum { rawr, croaring };

const Operation = enum {
    pair_and,
    pair_or,
    pair_andnot,
    pair_xor,
    total_union,
    to_array,
    serialize_deserialize,

    fn id(self: Operation) []const u8 {
        return switch (self) {
            .pair_and => "pair-and",
            .pair_or => "pair-or",
            .pair_andnot => "pair-andnot",
            .pair_xor => "pair-xor",
            .total_union => "total-union",
            .to_array => "to-array",
            .serialize_deserialize => "serialize-deserialize",
        };
    }

    fn displayName(self: Operation) []const u8 {
        return switch (self) {
            .pair_and => "successive AND",
            .pair_or => "successive OR",
            .pair_andnot => "successive ANDNOT",
            .pair_xor => "successive XOR",
            .total_union => "total union",
            .to_array => "toArray",
            .serialize_deserialize => "serialize + deserialize",
        };
    }

    fn denominator(self: Operation) usize {
        return switch (self) {
            .pair_and, .pair_or, .pair_andnot, .pair_xor => 199,
            .total_union => 1,
            .to_array, .serialize_deserialize => 200,
        };
    }

    fn parse(name: []const u8) ?Operation {
        for (operations) |operation| {
            if (std.mem.eql(u8, name, operation.id())) return operation;
        }
        return null;
    }
};

const operations = [_]Operation{
    .pair_and,
    .pair_or,
    .pair_andnot,
    .pair_xor,
    .total_union,
    .to_array,
    .serialize_deserialize,
};

const RequestedCell = struct {
    dataset: corpus_mod.Dataset,
    operation: Operation,
    implementation: Implementation,
    root: []const u8,
};

const Histogram = struct {
    arrays: u64 = 0,
    bitsets: u64 = 0,
    runs: u64 = 0,
};

const Validation = struct {
    digest: u64,
    serialized_bytes: u64,
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var list = false;
    var header = false;
    var dataset: ?corpus_mod.Dataset = null;
    var operation: ?Operation = null;
    var implementation: ?Implementation = null;
    var root: []const u8 = default_root;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--list")) {
            list = true;
        } else if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--dataset=")) {
            dataset = corpus_mod.Dataset.parse(arg[10..]) orelse return error.UnknownDataset;
        } else if (std.mem.startsWith(u8, arg, "--operation=")) {
            operation = Operation.parse(arg[12..]) orelse return error.UnknownOperation;
        } else if (std.mem.startsWith(u8, arg, "--implementation=")) {
            implementation = std.meta.stringToEnum(Implementation, arg[17..]) orelse
                return error.UnknownImplementation;
        } else if (std.mem.startsWith(u8, arg, "--root=")) {
            root = arg[7..];
        } else {
            return error.UnknownArgument;
        }
    }

    if (list) {
        if (header or dataset != null or operation != null or implementation != null or
            !std.mem.eql(u8, root, default_root)) return error.ConflictingArguments;
        printManifest();
        return;
    }
    if (header) {
        if (dataset != null or operation != null or implementation != null or
            !std.mem.eql(u8, root, default_root)) return error.ConflictingArguments;
        printHeader();
        return;
    }

    const requested = RequestedCell{
        .dataset = dataset orelse return error.MissingDataset,
        .operation = operation orelse return error.MissingOperation,
        .implementation = implementation orelse return error.MissingImplementation,
        .root = root,
    };

    var corpus = try corpus_mod.loadDataset(
        std.heap.page_allocator,
        init.io,
        requested.root,
        requested.dataset,
    );
    defer corpus.deinit();

    switch (requested.implementation) {
        .rawr => try runRawr(requested, &corpus),
        .croaring => try runCRoaring(requested, &corpus),
    }
}

fn printManifest() void {
    for (corpus_mod.supported_datasets) |dataset| {
        for (operations) |operation| {
            bench_time.print("ROW\t{s}\t{s}\t{s}\t{d}\n", .{
                dataset.name(), operation.id(), operation.displayName(), operation.denominator(),
            });
            inline for (.{ Implementation.rawr, Implementation.croaring }) |implementation| {
                bench_time.print("TUPLE\t{s}\t{s}\t{s}\t{d}\n", .{
                    dataset.name(), operation.id(), @tagName(implementation), operation.denominator(),
                });
            }
        }
    }
}

fn printHeader() void {
    bench_time.printBenchEnvironment();
    bench_time.print("# requested-cpu: native\n", .{});
    bench_time.print("# protocol: {d} warmup cycle, {d} timed cycles, process median\n", .{
        warmup_runs, timed_runs,
    });
    bench_time.print("# allocator-pairing: rawr=smp_allocator, CRoaring=default-libc\n", .{});
    bench_time.print("# construction: rawr=fromSorted, CRoaring=create+add_many, runOptimize=off\n", .{});
    bench_time.print("# croaring-avx512: {s}\n", .{
        if (c.CROARING_COMPILER_SUPPORTS_AVX512 != 0) "on" else "off",
    });
}

const RawrSources = struct {
    bitmaps: []RoaringBitmap,
    pointers: []*const RoaringBitmap,

    fn init(corpus: *const corpus_mod.Corpus, needs_pointers: bool) !RawrSources {
        const allocator = std.heap.page_allocator;
        const bitmaps = try allocator.alloc(RoaringBitmap, corpus.bitmaps.len);
        var built: usize = 0;
        errdefer {
            for (bitmaps[0..built]) |*bitmap| bitmap.deinit();
            allocator.free(bitmaps);
        }

        for (corpus.bitmaps, 0..) |entry, index| {
            bitmaps[index] = try RoaringBitmap.fromSorted(std.heap.smp_allocator, entry.values);
            built += 1;
        }

        const pointers = if (needs_pointers)
            try allocator.alloc(*const RoaringBitmap, bitmaps.len)
        else
            @as([]*const RoaringBitmap, &.{});
        if (needs_pointers) {
            for (bitmaps, pointers) |*bitmap, *pointer| pointer.* = bitmap;
        }
        return .{ .bitmaps = bitmaps, .pointers = pointers };
    }

    fn deinit(self: *RawrSources) void {
        for (self.bitmaps) |*bitmap| bitmap.deinit();
        if (self.pointers.len != 0) std.heap.page_allocator.free(self.pointers);
        std.heap.page_allocator.free(self.bitmaps);
    }

    fn metadata(self: *const RawrSources) struct { cardinality: u64, histogram: Histogram } {
        var cardinality: u64 = 0;
        var histogram = Histogram{};
        for (self.bitmaps) |*bitmap| {
            cardinality += bitmap.cardinality();
            for (bitmap.containers[0..bitmap.size]) |container| switch (container.getType()) {
                .array => histogram.arrays += 1,
                .bitset => histogram.bitsets += 1,
                .run => histogram.runs += 1,
                .reserved => unreachable,
            };
        }
        return .{ .cardinality = cardinality, .histogram = histogram };
    }
};

const CRoaringSources = struct {
    bitmaps: []*c.roaring_bitmap_t,

    fn init(corpus: *const corpus_mod.Corpus) !CRoaringSources {
        const bitmaps = try std.heap.page_allocator.alloc(*c.roaring_bitmap_t, corpus.bitmaps.len);
        var built: usize = 0;
        errdefer {
            for (bitmaps[0..built]) |bitmap| c.roaring_bitmap_free(bitmap);
            std.heap.page_allocator.free(bitmaps);
        }
        for (corpus.bitmaps, 0..) |entry, index| {
            const bitmap = c.roaring_bitmap_create() orelse return error.CRoaringAllocFailed;
            c.roaring_bitmap_add_many(bitmap, entry.values.len, entry.values.ptr);
            bitmaps[index] = bitmap;
            built += 1;
        }
        return .{ .bitmaps = bitmaps };
    }

    fn deinit(self: *CRoaringSources) void {
        for (self.bitmaps) |bitmap| c.roaring_bitmap_free(bitmap);
        std.heap.page_allocator.free(self.bitmaps);
    }

    fn metadata(self: *const CRoaringSources) struct { cardinality: u64, histogram: Histogram } {
        var cardinality: u64 = 0;
        var histogram = Histogram{};
        for (self.bitmaps) |bitmap| {
            cardinality += c.roaring_bitmap_get_cardinality(bitmap);
            var stats: c.roaring_statistics_t = undefined;
            c.roaring_bitmap_statistics(bitmap, &stats);
            histogram.arrays += stats.n_array_containers;
            histogram.bitsets += stats.n_bitset_containers;
            histogram.runs += stats.n_run_containers;
        }
        return .{ .cardinality = cardinality, .histogram = histogram };
    }
};

fn runRawr(requested: RequestedCell, corpus: *const corpus_mod.Corpus) !void {
    var sources = try RawrSources.init(corpus, requested.operation == .total_union);
    defer sources.deinit();
    var timed_output: []u32 = &.{};
    if (requested.operation == .to_array) timed_output = try allocateOutputBuffer(corpus);
    defer if (timed_output.len != 0) std.heap.page_allocator.free(timed_output);

    const median_ns = try measureRawr(requested.operation, &sources, timed_output);
    const validation_output = if (requested.operation == .to_array)
        timed_output
    else
        try allocateOutputBuffer(corpus);
    defer if (requested.operation != .to_array) std.heap.page_allocator.free(validation_output);
    const validation = try validateRawr(requested.operation, &sources, corpus, validation_output);
    const metadata = sources.metadata();
    if (metadata.cardinality != corpus.total_values) return error.SourceCardinalityMismatch;
    printResult(requested, median_ns, corpus, metadata.cardinality, metadata.histogram, validation);
}

fn runCRoaring(requested: RequestedCell, corpus: *const corpus_mod.Corpus) !void {
    var sources = try CRoaringSources.init(corpus);
    defer sources.deinit();
    var timed_output: []u32 = &.{};
    if (requested.operation == .to_array) timed_output = try allocateOutputBuffer(corpus);
    defer if (timed_output.len != 0) std.heap.page_allocator.free(timed_output);

    const median_ns = try measureCRoaring(requested.operation, &sources, timed_output);
    const validation_output = if (requested.operation == .to_array)
        timed_output
    else
        try allocateOutputBuffer(corpus);
    defer if (requested.operation != .to_array) std.heap.page_allocator.free(validation_output);
    const validation = try validateCRoaring(requested.operation, &sources, corpus, validation_output);
    const metadata = sources.metadata();
    if (metadata.cardinality != corpus.total_values) return error.SourceCardinalityMismatch;
    printResult(requested, median_ns, corpus, metadata.cardinality, metadata.histogram, validation);
}

fn allocateOutputBuffer(corpus: *const corpus_mod.Corpus) ![]u32 {
    return std.heap.page_allocator.alloc(u32, @intCast(corpus.total_values));
}

fn measureRawr(operation: Operation, sources: *const RawrSources, output: []u32) !u64 {
    for (0..warmup_runs) |_| _ = try runRawrCycle(operation, sources, output);
    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        const checksum = try runRawrCycle(operation, sources, output);
        elapsed.* = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(checksum);
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

fn measureCRoaring(operation: Operation, sources: *const CRoaringSources, output: []u32) !u64 {
    for (0..warmup_runs) |_| _ = try runCRoaringCycle(operation, sources, output);
    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        const checksum = try runCRoaringCycle(operation, sources, output);
        elapsed.* = bench_time.monotonicNanos() - start;
        std.mem.doNotOptimizeAway(checksum);
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

fn runRawrCycle(operation: Operation, sources: *const RawrSources, output: []u32) !u64 {
    var checksum: u64 = 0;
    switch (operation) {
        .pair_and, .pair_or, .pair_andnot, .pair_xor => {
            for (sources.bitmaps[0 .. sources.bitmaps.len - 1], sources.bitmaps[1..]) |*left, *right| {
                var result = switch (operation) {
                    .pair_and => try left.bitwiseAnd(std.heap.smp_allocator, right),
                    .pair_or => try left.bitwiseOr(std.heap.smp_allocator, right),
                    .pair_andnot => try left.bitwiseDifference(std.heap.smp_allocator, right),
                    .pair_xor => try left.bitwiseXor(std.heap.smp_allocator, right),
                    else => unreachable,
                };
                checksum +%= result.cardinality();
                result.deinit();
            }
        },
        .total_union => {
            var result = try RoaringBitmap.orMany(std.heap.smp_allocator, sources.pointers);
            checksum = result.cardinality();
            result.deinit();
        },
        .to_array => for (sources.bitmaps) |*bitmap| {
            const written = bitmap.toArray(output);
            checksum +%= written;
            std.mem.doNotOptimizeAway(output[0..written]);
        },
        .serialize_deserialize => for (sources.bitmaps) |*bitmap| {
            const bytes = try std.heap.smp_allocator.alloc(u8, bitmap.serializedSizeInBytes());
            errdefer std.heap.smp_allocator.free(bytes);
            var writer = std.Io.Writer.fixed(bytes);
            try bitmap.serializeToWriter(&writer);
            var decoded = try RoaringBitmap.deserialize(std.heap.smp_allocator, bytes);
            checksum +%= decoded.cardinality();
            decoded.deinit();
            std.heap.smp_allocator.free(bytes);
        },
    }
    return checksum;
}

fn runCRoaringCycle(operation: Operation, sources: *const CRoaringSources, output: []u32) !u64 {
    var checksum: u64 = 0;
    switch (operation) {
        .pair_and, .pair_or, .pair_andnot, .pair_xor => {
            for (sources.bitmaps[0 .. sources.bitmaps.len - 1], sources.bitmaps[1..]) |left, right| {
                const result = switch (operation) {
                    .pair_and => c.roaring_bitmap_and(left, right),
                    .pair_or => c.roaring_bitmap_or(left, right),
                    .pair_andnot => c.roaring_bitmap_andnot(left, right),
                    .pair_xor => c.roaring_bitmap_xor(left, right),
                    else => unreachable,
                } orelse return error.CRoaringAllocFailed;
                checksum +%= c.roaring_bitmap_get_cardinality(result);
                c.roaring_bitmap_free(result);
            }
        },
        .total_union => {
            const result = c.roaring_bitmap_or_many(
                sources.bitmaps.len,
                @ptrCast(sources.bitmaps.ptr),
            ) orelse return error.CRoaringAllocFailed;
            checksum = c.roaring_bitmap_get_cardinality(result);
            c.roaring_bitmap_free(result);
        },
        .to_array => for (sources.bitmaps) |bitmap| {
            const cardinality: usize = @intCast(c.roaring_bitmap_get_cardinality(bitmap));
            c.roaring_bitmap_to_uint32_array(bitmap, output.ptr);
            checksum +%= cardinality;
            std.mem.doNotOptimizeAway(output[0..cardinality]);
        },
        .serialize_deserialize => for (sources.bitmaps) |bitmap| {
            const size = c.roaring_bitmap_portable_size_in_bytes(bitmap);
            const bytes = try bench_time.cAllocator().alloc(u8, size);
            errdefer bench_time.cAllocator().free(bytes);
            if (c.roaring_bitmap_portable_serialize(bitmap, @ptrCast(bytes.ptr)) != size) {
                return error.CRoaringSerializeFailed;
            }
            const decoded = c.roaring_bitmap_portable_deserialize_safe(
                @ptrCast(bytes.ptr),
                bytes.len,
            ) orelse return error.CRoaringDeserializeFailed;
            checksum +%= c.roaring_bitmap_get_cardinality(decoded);
            c.roaring_bitmap_free(decoded);
            bench_time.cAllocator().free(bytes);
        },
    }
    return checksum;
}

fn validateRawr(
    operation: Operation,
    sources: *const RawrSources,
    corpus: *const corpus_mod.Corpus,
    output: []u32,
) !Validation {
    var hasher = StableHasher.init();
    var serialized_bytes: u64 = 0;
    switch (operation) {
        .pair_and, .pair_or, .pair_andnot, .pair_xor => {
            for (sources.bitmaps[0 .. sources.bitmaps.len - 1], sources.bitmaps[1..], 0..) |*left, *right, index| {
                var result = switch (operation) {
                    .pair_and => try left.bitwiseAnd(std.heap.smp_allocator, right),
                    .pair_or => try left.bitwiseOr(std.heap.smp_allocator, right),
                    .pair_andnot => try left.bitwiseDifference(std.heap.smp_allocator, right),
                    .pair_xor => try left.bitwiseXor(std.heap.smp_allocator, right),
                    else => unreachable,
                };
                defer result.deinit();
                hashRawrResult(&hasher, index, &result, output);
            }
        },
        .total_union => {
            var result = try RoaringBitmap.orMany(std.heap.smp_allocator, sources.pointers);
            defer result.deinit();
            hashRawrResult(&hasher, 0, &result, output);
        },
        .to_array => for (sources.bitmaps, 0..) |*bitmap, index| {
            hashRawrResult(&hasher, index, bitmap, output);
        },
        .serialize_deserialize => for (sources.bitmaps, corpus.bitmaps, 0..) |*bitmap, entry, index| {
            const bytes = try bitmap.serialize(std.heap.smp_allocator);
            defer std.heap.smp_allocator.free(bytes);
            serialized_bytes += bytes.len;
            var decoded = try RoaringBitmap.deserialize(std.heap.smp_allocator, bytes);
            defer decoded.deinit();
            const written = decoded.toArray(output);
            if (!std.mem.eql(u32, entry.values, output[0..written])) return error.RoundTripMismatch;
            hashValues(&hasher, index, output[0..written]);
        },
    }
    return .{ .digest = hasher.finish(), .serialized_bytes = serialized_bytes };
}

fn validateCRoaring(
    operation: Operation,
    sources: *const CRoaringSources,
    corpus: *const corpus_mod.Corpus,
    output: []u32,
) !Validation {
    var hasher = StableHasher.init();
    var serialized_bytes: u64 = 0;
    switch (operation) {
        .pair_and, .pair_or, .pair_andnot, .pair_xor => {
            for (sources.bitmaps[0 .. sources.bitmaps.len - 1], sources.bitmaps[1..], 0..) |left, right, index| {
                const result = switch (operation) {
                    .pair_and => c.roaring_bitmap_and(left, right),
                    .pair_or => c.roaring_bitmap_or(left, right),
                    .pair_andnot => c.roaring_bitmap_andnot(left, right),
                    .pair_xor => c.roaring_bitmap_xor(left, right),
                    else => unreachable,
                } orelse return error.CRoaringAllocFailed;
                defer c.roaring_bitmap_free(result);
                hashCRoaringResult(&hasher, index, result, output);
            }
        },
        .total_union => {
            const result = c.roaring_bitmap_or_many(
                sources.bitmaps.len,
                @ptrCast(sources.bitmaps.ptr),
            ) orelse return error.CRoaringAllocFailed;
            defer c.roaring_bitmap_free(result);
            hashCRoaringResult(&hasher, 0, result, output);
        },
        .to_array => for (sources.bitmaps, 0..) |bitmap, index| {
            hashCRoaringResult(&hasher, index, bitmap, output);
        },
        .serialize_deserialize => for (sources.bitmaps, corpus.bitmaps, 0..) |bitmap, entry, index| {
            const size = c.roaring_bitmap_portable_size_in_bytes(bitmap);
            const bytes = try bench_time.cAllocator().alloc(u8, size);
            defer bench_time.cAllocator().free(bytes);
            if (c.roaring_bitmap_portable_serialize(bitmap, @ptrCast(bytes.ptr)) != size) {
                return error.CRoaringSerializeFailed;
            }
            serialized_bytes += bytes.len;
            const decoded = c.roaring_bitmap_portable_deserialize_safe(
                @ptrCast(bytes.ptr),
                bytes.len,
            ) orelse return error.CRoaringDeserializeFailed;
            defer c.roaring_bitmap_free(decoded);
            const written: usize = @intCast(c.roaring_bitmap_get_cardinality(decoded));
            c.roaring_bitmap_to_uint32_array(decoded, output.ptr);
            if (!std.mem.eql(u32, entry.values, output[0..written])) return error.RoundTripMismatch;
            hashValues(&hasher, index, output[0..written]);
        },
    }
    return .{ .digest = hasher.finish(), .serialized_bytes = serialized_bytes };
}

fn hashRawrResult(hasher: *StableHasher, index: usize, bitmap: *const RoaringBitmap, output: []u32) void {
    const written = bitmap.toArray(output);
    hashValues(hasher, index, output[0..written]);
}

fn hashCRoaringResult(
    hasher: *StableHasher,
    index: usize,
    bitmap: *const c.roaring_bitmap_t,
    output: []u32,
) void {
    const written: usize = @intCast(c.roaring_bitmap_get_cardinality(bitmap));
    c.roaring_bitmap_to_uint32_array(bitmap, output.ptr);
    hashValues(hasher, index, output[0..written]);
}

fn hashValues(hasher: *StableHasher, index: usize, values: []const u32) void {
    hasher.addU32(@intCast(index));
    hasher.addU64(values.len);
    for (values) |value| hasher.addU32(value);
}

fn printResult(
    requested: RequestedCell,
    median_ns: u64,
    corpus: *const corpus_mod.Corpus,
    source_cardinality: u64,
    histogram: Histogram,
    validation: Validation,
) void {
    bench_time.print("RESULT\t{s}\t{s}\t{s}\t{d}\t{d}\t0x{x:0>16}\t0x{x:0>16}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{
        requested.dataset.name(),
        requested.operation.id(),
        @tagName(requested.implementation),
        requested.operation.denominator(),
        median_ns,
        validation.digest,
        corpus.fingerprint,
        source_cardinality,
        histogram.arrays,
        histogram.bitsets,
        histogram.runs,
        validation.serialized_bytes,
    });
}

const StableHasher = struct {
    state: u64 = 0xcbf29ce484222325,
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
