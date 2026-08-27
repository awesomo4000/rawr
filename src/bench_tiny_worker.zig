// SPDX-License-Identifier: MPL-2.0

//! Fresh-process timing worker for the spec 48 tiny-bitmap sweep.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const fixtures = @import("tiny_bench_fixtures.zig");
const setup = @import("bench_tiny_setup.zig");

const RoaringBitmap = rawr.RoaringBitmap;
const warmup_runs = 1;
const timed_runs = 3;
const batch_count = fixtures.sweep_iterations;

const Implementation = enum { rawr, croaring, reference };
const AllocatorKind = enum { smp, libc };

const RequestedTuple = struct {
    shape: fixtures.Shape,
    cardinality: u32,
    implementation: Implementation,
    allocator_kind: AllocatorKind,
};

pub fn main(init: std.process.Init) !void {
    // Argument parsing and fixture generation must not precondition either tested allocator.
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var list = false;
    var header = false;
    var shape: ?fixtures.Shape = null;
    var cardinality: ?u32 = null;
    var implementation: ?Implementation = null;
    var allocator_kind: ?AllocatorKind = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--list")) {
            list = true;
        } else if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--shape=")) {
            shape = parseShape(arg[8..]) orelse return error.UnknownShape;
        } else if (std.mem.startsWith(u8, arg, "--cardinality=")) {
            cardinality = try std.fmt.parseInt(u32, arg[14..], 10);
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
        if (header or shape != null or cardinality != null or implementation != null or allocator_kind != null) {
            return error.ConflictingArguments;
        }
        printManifest();
        return;
    }
    if (header) {
        if (shape != null or cardinality != null or implementation != null or allocator_kind != null) {
            return error.ConflictingArguments;
        }
        printHeader();
        return;
    }

    const requested = RequestedTuple{
        .shape = shape orelse return error.MissingShape,
        .cardinality = cardinality orelse return error.MissingCardinality,
        .implementation = implementation orelse return error.MissingImplementation,
        .allocator_kind = allocator_kind orelse return error.MissingAllocator,
    };
    try validateTuple(requested);

    var pool = try fixtures.generateSweepPool(std.heap.page_allocator, requested.shape, requested.cardinality);
    defer pool.deinit();
    const median_ns = try measure(requested, &pool);

    // Full-set and cross-format validation deliberately follows timing.
    try validateResults(requested, &pool);
    bench_time.print("RESULT\t{s}\t{d}\t{s}\t{s}\t{d}\t{d}\n", .{
        requested.shape.name(),
        requested.cardinality,
        @tagName(requested.implementation),
        @tagName(requested.allocator_kind),
        batch_count,
        median_ns,
    });
}

fn parseShape(name: []const u8) ?fixtures.Shape {
    for (fixtures.shapes) |shape| {
        if (std.mem.eql(u8, name, shape.name())) return shape;
    }
    return null;
}

fn printManifest() void {
    for (fixtures.shapes) |shape| {
        for (fixtures.sweep_cardinalities) |cardinality| {
            const fixture_count: usize = if (cardinality == 0) 1 else fixtures.sweep_pool_size;
            bench_time.print("ROW\t{s}\t{d}\t{d}\t{d}\n", .{
                shape.name(), cardinality, fixture_count, batch_count,
            });
            printTuple(shape, cardinality, .rawr, .smp);
            printTuple(shape, cardinality, .rawr, .libc);
            printTuple(shape, cardinality, .croaring, .libc);
            printTuple(shape, cardinality, .reference, .smp);
            printTuple(shape, cardinality, .reference, .libc);
        }
    }
}

fn printTuple(
    shape: fixtures.Shape,
    cardinality: u32,
    implementation: Implementation,
    allocator_kind: AllocatorKind,
) void {
    bench_time.print("TUPLE\t{s}\t{d}\t{s}\t{s}\t{d}\n", .{
        shape.name(), cardinality, @tagName(implementation), @tagName(allocator_kind), batch_count,
    });
}

fn printHeader() void {
    bench_time.printBenchEnvironment();
    bench_time.print("# requested-cpu: native\n", .{});
    bench_time.print("# protocol: {d} warmup batch, {d} timed batches, process median\n", .{
        warmup_runs, timed_runs,
    });
    bench_time.print("# batch: {d} complete bitmap lifecycles\n", .{batch_count});
    bench_time.print("# croaring-avx512: {s}\n", .{
        if (c.CROARING_COMPILER_SUPPORTS_AVX512 != 0) "on" else "off",
    });
}

fn validateTuple(requested: RequestedTuple) !void {
    var cardinality_found = false;
    for (fixtures.sweep_cardinalities) |cardinality| {
        if (cardinality == requested.cardinality) cardinality_found = true;
    }
    if (!cardinality_found) return error.CardinalityOutsideSweep;
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

fn measure(requested: RequestedTuple, pool: *const fixtures.FixturePool) !u64 {
    for (0..warmup_runs) |_| _ = try runBatch(requested, pool);

    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| elapsed.* = try runBatch(requested, pool);
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

fn runBatch(requested: RequestedTuple, pool: *const fixtures.FixturePool) !u64 {
    if (batch_count % pool.fixture_count != 0) return error.PartialFixtureCycle;
    const cycles = batch_count / pool.fixture_count;
    var checksum: u64 = 0;

    const start = bench_time.monotonicNanos();
    for (0..cycles) |_| {
        for (0..pool.fixture_count) |fixture_index| {
            const values = pool.fixture(fixture_index);
            checksum +%= switch (requested.implementation) {
                .rawr => try runRawrLifecycle(allocatorFor(requested.allocator_kind), values),
                .croaring => try runCRoaringLifecycle(values),
                .reference => try runReferenceLifecycle(allocatorFor(requested.allocator_kind), values),
            };
        }
    }
    const elapsed_ns = bench_time.monotonicNanos() - start;
    std.mem.doNotOptimizeAway(checksum);

    const expected = @as(u64, batch_count) * requested.cardinality;
    if (checksum != expected) return error.TimedCardinalityMismatch;
    return elapsed_ns;
}

pub fn runRawrLifecycle(allocator: std.mem.Allocator, values: []const u32) !u64 {
    var source = try RoaringBitmap.init(allocator);
    errdefer source.deinit();
    for (values) |value| _ = try source.add(value);

    const serialized_size = source.serializedSizeInBytes();
    const bytes = try allocator.alloc(u8, serialized_size);
    errdefer allocator.free(bytes);
    var writer = std.Io.Writer.fixed(bytes);
    try source.serializeToWriter(&writer);
    std.mem.doNotOptimizeAway(bytes);

    var decoded = try RoaringBitmap.deserialize(allocator, bytes);
    errdefer decoded.deinit();
    std.mem.doNotOptimizeAway(&decoded);
    const cardinality = decoded.cardinality();

    decoded.deinit();
    allocator.free(bytes);
    source.deinit();
    return cardinality;
}

pub fn runCRoaringLifecycle(values: []const u32) !u64 {
    const source = c.roaring_bitmap_create() orelse return error.CRoaringAllocFailed;
    errdefer c.roaring_bitmap_free(source);
    c.roaring_bitmap_add_many(source, values.len, values.ptr);

    const serialized_size = c.roaring_bitmap_portable_size_in_bytes(source);
    const allocator = bench_time.cAllocator();
    const bytes = try allocator.alloc(u8, serialized_size);
    errdefer allocator.free(bytes);
    if (c.roaring_bitmap_portable_serialize(source, @ptrCast(bytes.ptr)) != serialized_size) {
        return error.CRoaringSerializeFailed;
    }
    std.mem.doNotOptimizeAway(bytes);

    const decoded = c.roaring_bitmap_portable_deserialize_safe(@ptrCast(bytes.ptr), bytes.len) orelse
        return error.CRoaringDeserializeFailed;
    std.mem.doNotOptimizeAway(decoded);
    const cardinality = c.roaring_bitmap_get_cardinality(decoded);

    c.roaring_bitmap_free(decoded);
    allocator.free(bytes);
    c.roaring_bitmap_free(source);
    return cardinality;
}

pub fn runReferenceLifecycle(allocator: std.mem.Allocator, values: []const u32) !u64 {
    const source = try allocator.dupe(u32, values);
    errdefer allocator.free(source);

    const serialized_size = 4 + values.len * @sizeOf(u32);
    const bytes = try allocator.alloc(u8, serialized_size);
    errdefer allocator.free(bytes);
    encodeReference(bytes, source);
    std.mem.doNotOptimizeAway(bytes);

    const decoded = try decodeReference(allocator, bytes);
    errdefer allocator.free(decoded);
    std.mem.doNotOptimizeAway(decoded);
    const cardinality = decoded.len;

    allocator.free(decoded);
    allocator.free(bytes);
    allocator.free(source);
    return cardinality;
}

fn encodeReference(bytes: []u8, values: []const u32) void {
    std.mem.writeInt(u32, bytes[0..4], @intCast(values.len), .little);
    for (values, 0..) |value, index| {
        std.mem.writeInt(u32, bytes[4 + index * 4 ..][0..4], value, .little);
    }
}

fn decodeReference(allocator: std.mem.Allocator, bytes: []const u8) ![]u32 {
    if (bytes.len < 4) return error.ReferenceTooShort;
    const count = std.mem.readInt(u32, bytes[0..4], .little);
    const expected_len = 4 + try std.math.mul(usize, count, @sizeOf(u32));
    if (bytes.len != expected_len) return error.ReferenceLengthMismatch;
    const values = try allocator.alloc(u32, count);
    errdefer allocator.free(values);
    for (values, 0..) |*value, index| {
        value.* = std.mem.readInt(u32, bytes[4 + index * 4 ..][0..4], .little);
    }
    return values;
}

fn validateResults(requested: RequestedTuple, pool: *const fixtures.FixturePool) !void {
    for (0..pool.fixture_count) |fixture_index| {
        const values = pool.fixture(fixture_index);
        switch (requested.implementation) {
            .rawr, .croaring => try setup.validateCrossImplementation(values),
            .reference => try validateReference(values),
        }
    }
}

pub fn validateReference(values: []const u32) !void {
    const allocator = std.heap.page_allocator;
    const bytes = try allocator.alloc(u8, 4 + values.len * @sizeOf(u32));
    defer allocator.free(bytes);
    encodeReference(bytes, values);
    const decoded = try decodeReference(allocator, bytes);
    defer allocator.free(decoded);
    if (!std.mem.eql(u32, values, decoded)) return error.ReferenceValueMismatch;
}
