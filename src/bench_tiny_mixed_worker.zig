// SPDX-License-Identifier: MPL-2.0

//! Fresh-process timing worker for the spec 48 mixed tiny-bitmap corpus.

const std = @import("std");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const fixtures = @import("tiny_bench_fixtures.zig");
const setup = @import("bench_tiny_setup.zig");
const lifecycle = @import("bench_tiny_worker.zig");

const warmup_runs = 1;
const timed_runs = 3;

const Implementation = enum { rawr, croaring, reference };
const AllocatorKind = enum { smp, libc };
const CellKind = enum { total, band };

const RequestedTuple = struct {
    cell: CellKind,
    band: ?fixtures.MixedBand,
    implementation: Implementation,
    allocator_kind: AllocatorKind,
};

const MixedReplay = struct {
    allocator: std.mem.Allocator,
    values: []u32,
    offsets: []usize,
    cardinality_sum: u64,

    fn init(allocator: std.mem.Allocator, band: ?fixtures.MixedBand) !MixedReplay {
        var corpus = try fixtures.generateMixedCardinalityCorpus(allocator);
        defer corpus.deinit();
        try setup.validateMixedCardinalityCorpus(&corpus);

        var fixture_count: usize = 0;
        var value_count: usize = 0;
        var cardinality_sum: u64 = 0;
        for (corpus.cardinalities) |cardinality| {
            if (band) |selected| {
                if (!selected.contains(cardinality)) continue;
            }
            fixture_count += 1;
            value_count = try std.math.add(usize, value_count, cardinality);
            cardinality_sum = try std.math.add(u64, cardinality_sum, cardinality);
        }
        if (fixture_count == 0) return error.EmptyMixedReplay;

        const values = try allocator.alloc(u32, value_count);
        errdefer allocator.free(values);
        const offsets = try allocator.alloc(usize, fixture_count + 1);
        errdefer allocator.free(offsets);

        var fixture_index: usize = 0;
        var value_offset: usize = 0;
        offsets[0] = 0;
        for (corpus.cardinalities, 0..) |cardinality, corpus_index| {
            if (band) |selected| {
                if (!selected.contains(cardinality)) continue;
            }
            const next_offset = try std.math.add(usize, value_offset, cardinality);
            const fixture_values = values[value_offset..next_offset];
            try fixtures.fillSpread(fixture_values, fixtures.mixedValueSeed(corpus_index, cardinality));
            try fixtures.validateFixture(.spread, fixture_values, cardinality);
            fixture_index += 1;
            offsets[fixture_index] = next_offset;
            value_offset = next_offset;
        }
        std.debug.assert(fixture_index == fixture_count);
        std.debug.assert(value_offset == values.len);

        return .{
            .allocator = allocator,
            .values = values,
            .offsets = offsets,
            .cardinality_sum = cardinality_sum,
        };
    }

    fn deinit(self: *MixedReplay) void {
        self.allocator.free(self.offsets);
        self.allocator.free(self.values);
        self.* = undefined;
    }

    fn fixtureCount(self: *const MixedReplay) usize {
        return self.offsets.len - 1;
    }

    fn fixture(self: *const MixedReplay, index: usize) []const u32 {
        std.debug.assert(index < self.fixtureCount());
        return self.values[self.offsets[index]..self.offsets[index + 1]];
    }
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var list = false;
    var header = false;
    var cell: ?CellKind = null;
    var band: ?fixtures.MixedBand = null;
    var implementation: ?Implementation = null;
    var allocator_kind: ?AllocatorKind = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--list")) {
            list = true;
        } else if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--cell=")) {
            cell = std.meta.stringToEnum(CellKind, arg[7..]) orelse return error.UnknownCell;
        } else if (std.mem.startsWith(u8, arg, "--band=")) {
            band = parseBand(arg[7..]) orelse return error.UnknownBand;
        } else if (std.mem.startsWith(u8, arg, "--implementation=")) {
            implementation = std.meta.stringToEnum(Implementation, arg[17..]) orelse
                return error.UnknownImplementation;
        } else if (std.mem.startsWith(u8, arg, "--allocator=")) {
            allocator_kind = std.meta.stringToEnum(AllocatorKind, arg[12..]) orelse
                return error.UnknownAllocator;
        } else {
            return error.UnknownArgument;
        }
    }

    if (list) {
        if (header or cell != null or band != null or implementation != null or allocator_kind != null) {
            return error.ConflictingArguments;
        }
        try printManifest();
        return;
    }
    if (header) {
        if (cell != null or band != null or implementation != null or allocator_kind != null) {
            return error.ConflictingArguments;
        }
        printHeader();
        return;
    }

    const requested = RequestedTuple{
        .cell = cell orelse return error.MissingCell,
        .band = band,
        .implementation = implementation orelse return error.MissingImplementation,
        .allocator_kind = allocator_kind orelse return error.MissingAllocator,
    };
    try validateTuple(requested);

    var replay = try MixedReplay.init(
        std.heap.page_allocator,
        if (requested.cell == .band) requested.band else null,
    );
    defer replay.deinit();
    const median_ns = try measure(requested, &replay);

    try validateResults(requested, &replay);
    bench_time.print("MIXED_RESULT\t{s}\t{s}\t{s}\t{s}\t{d}\t{d}\n", .{
        @tagName(requested.cell),
        if (requested.band) |selected| selected.name() else "total",
        @tagName(requested.implementation),
        @tagName(requested.allocator_kind),
        replay.fixtureCount(),
        median_ns,
    });
}

fn parseBand(name: []const u8) ?fixtures.MixedBand {
    for (fixtures.mixed_bands) |band| {
        if (std.mem.eql(u8, name, band.name())) return band;
    }
    return null;
}

fn printManifest() !void {
    var corpus = try fixtures.generateMixedCardinalityCorpus(std.heap.page_allocator);
    defer corpus.deinit();
    try setup.validateMixedCardinalityCorpus(&corpus);

    bench_time.print("MIXED_META\t{d}\t{d}\t{d}\t0x{x:0>16}\n", .{
        corpus.cardinalities.len,
        corpus.median,
        corpus.p99,
        corpus.cardinality_hash,
    });
    bench_time.print("MIXED_CELL\ttotal\ttotal\t{d}\n", .{corpus.cardinalities.len});
    printTuple(.total, null, .rawr, .smp, corpus.cardinalities.len);
    printTuple(.total, null, .rawr, .libc, corpus.cardinalities.len);
    printTuple(.total, null, .croaring, .libc, corpus.cardinalities.len);
    printTuple(.total, null, .reference, .smp, corpus.cardinalities.len);
    printTuple(.total, null, .reference, .libc, corpus.cardinalities.len);

    for (fixtures.mixed_bands) |band| {
        const count = corpus.bandCount(band);
        bench_time.print("MIXED_CELL\tband\t{s}\t{d}\n", .{ band.name(), count });
        if (count == 0) continue;
        printTuple(.band, band, .rawr, .smp, count);
        printTuple(.band, band, .rawr, .libc, count);
    }
}

fn printTuple(
    cell: CellKind,
    band: ?fixtures.MixedBand,
    implementation: Implementation,
    allocator_kind: AllocatorKind,
    count: usize,
) void {
    bench_time.print("MIXED_TUPLE\t{s}\t{s}\t{s}\t{s}\t{d}\n", .{
        @tagName(cell),
        if (band) |selected| selected.name() else "total",
        @tagName(implementation),
        @tagName(allocator_kind),
        count,
    });
}

fn printHeader() void {
    bench_time.printBenchEnvironment();
    bench_time.print("# requested-cpu: native\n", .{});
    bench_time.print("# protocol: {d} warmup corpus cycle, {d} timed corpus cycles, process median\n", .{
        warmup_runs, timed_runs,
    });
    bench_time.print("# mixed corpus: spread, count={d}, seed=0x{x}\n", .{
        fixtures.mixed_count, fixtures.mixed_seed,
    });
    bench_time.print("# croaring-avx512: {s}\n", .{
        if (c.CROARING_COMPILER_SUPPORTS_AVX512 != 0) "on" else "off",
    });
}

fn validateTuple(requested: RequestedTuple) !void {
    switch (requested.cell) {
        .total => if (requested.band != null) return error.UnexpectedBand,
        .band => {
            const band = requested.band orelse return error.MissingBand;
            if (band == .zero) return error.EmptyBand;
            if (requested.implementation != .rawr) return error.UnsupportedBandImplementation;
        },
    }
    if (requested.implementation == .croaring and requested.allocator_kind != .libc) {
        return error.UnsupportedTuple;
    }
}

fn allocatorFor(kind: AllocatorKind) std.mem.Allocator {
    return switch (kind) {
        .smp => std.heap.smp_allocator,
        .libc => bench_time.cAllocator(),
    };
}

fn measure(requested: RequestedTuple, replay: *const MixedReplay) !u64 {
    for (0..warmup_runs) |_| _ = try runBatch(requested, replay);

    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| elapsed.* = try runBatch(requested, replay);
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

fn runBatch(requested: RequestedTuple, replay: *const MixedReplay) !u64 {
    var checksum: u64 = 0;
    const start = bench_time.monotonicNanos();
    for (0..replay.fixtureCount()) |fixture_index| {
        const values = replay.fixture(fixture_index);
        checksum +%= switch (requested.implementation) {
            .rawr => try lifecycle.runRawrLifecycle(allocatorFor(requested.allocator_kind), values),
            .croaring => try lifecycle.runCRoaringLifecycle(values),
            .reference => try lifecycle.runReferenceLifecycle(allocatorFor(requested.allocator_kind), values),
        };
    }
    const elapsed_ns = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(checksum);
    if (checksum != replay.cardinality_sum) return error.TimedCardinalityMismatch;
    return elapsed_ns;
}

fn validateResults(requested: RequestedTuple, replay: *const MixedReplay) !void {
    for (0..replay.fixtureCount()) |fixture_index| {
        const values = replay.fixture(fixture_index);
        switch (requested.implementation) {
            .rawr, .croaring => try setup.validateCrossImplementation(values),
            .reference => try lifecycle.validateReference(values),
        }
    }
}
