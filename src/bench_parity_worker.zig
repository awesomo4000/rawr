// SPDX-License-Identifier: MPL-2.0

//! Fresh-process CRoaring parity benchmark worker.
//!
//! The shell controller discovers rows and tuples through `--list`, then starts
//! one worker process for exactly one `(row, implementation, allocator)` tuple.

const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");

const RoaringBitmap = rawr.RoaringBitmap;

const warmup_runs = 3;
const timed_runs = 21;
const sparse_value_count = 500_000;
const cardinality_value_count = 1_000_000;

const Implementation = enum {
    rawr,
    croaring,
};

const AllocatorKind = enum {
    none,
    smp,
    libc,
    arena,
};

const ReportingUnit = enum {
    ms,
    ns_per_op,

    fn name(self: ReportingUnit) []const u8 {
        return switch (self) {
            .ms => "ms",
            .ns_per_op => "ns/op",
        };
    }
};

const AllocationClass = enum {
    allocating,
    non_allocating,
};

const Operation = enum {
    sparse_and,
    cardinality,
};

const ValidationOracle = enum {
    portable_bytes,
    cardinality,
};

const TupleKey = struct {
    implementation: Implementation,
    allocator: AllocatorKind,
};

const Reference = struct {
    row_id: []const u8,
    variant: TupleKey,
};

const Variant = struct {
    implementation: Implementation,
    allocator: AllocatorKind,
};

const ManifestRow = struct {
    id: []const u8,
    display_name: []const u8,
    corpus: []const u8,
    seed: u64,
    rawr_operation: []const u8,
    croaring_operation: []const u8,
    allocation_class: AllocationClass,
    variants: []const Variant,
    reference: ?Reference = null,
    setup_boundary: []const u8,
    teardown_boundary: []const u8,
    validation_oracle: ValidationOracle,
    batch_count: usize,
    reporting_unit: ReportingUnit,
    operation: Operation,
};

const sparse_and_variants = [_]Variant{
    .{ .implementation = .rawr, .allocator = .smp },
    .{ .implementation = .rawr, .allocator = .libc },
    .{ .implementation = .croaring, .allocator = .libc },
};

const cardinality_variants = [_]Variant{
    .{ .implementation = .rawr, .allocator = .none },
    .{ .implementation = .croaring, .allocator = .none },
};

const manifest = [_]ManifestRow{
    .{
        .id = "sparse-and",
        .display_name = "bitwiseAnd (sparse)",
        .corpus = "500000 deterministic sorted/deduplicated u32 values split into overlapping halves",
        .seed = 54321,
        .rawr_operation = "RoaringBitmap.bitwiseAnd",
        .croaring_operation = "roaring_bitmap_and",
        .allocation_class = .allocating,
        .variants = &sparse_and_variants,
        .setup_boundary = "input construction outside timing; result construction inside timing",
        .teardown_boundary = "result deinit/free inside timing",
        .validation_oracle = .portable_bytes,
        .batch_count = 1,
        .reporting_unit = .ms,
        .operation = .sparse_and,
    },
    .{
        .id = "cardinality",
        .display_name = "cardinality",
        .corpus = "bitmap built from 1000000 deterministic random u32 values",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.cardinality",
        .croaring_operation = "roaring_bitmap_get_cardinality",
        .allocation_class = .non_allocating,
        .variants = &cardinality_variants,
        .setup_boundary = "bitmap construction outside timing; batched cardinality calls inside timing",
        .teardown_boundary = "bitmap deinit/free outside timing",
        .validation_oracle = .cardinality,
        // This proves normalized ns/op reporting. Spec 22-03 performs the final
        // cross-host >=1 ms batch calibration for all tiny rows.
        .batch_count = 1024,
        .reporting_unit = .ns_per_op,
        .operation = .cardinality,
    },
};

var sparse_values: [sparse_value_count]u32 = undefined;
var cardinality_values: [cardinality_value_count]u32 = undefined;

const RawrSparseInputs = struct {
    a: RoaringBitmap,
    b: RoaringBitmap,

    fn deinit(self: *RawrSparseInputs) void {
        self.a.deinit();
        self.b.deinit();
    }
};

const CRoaringSparseInputs = struct {
    a: *c.roaring_bitmap_t,
    b: *c.roaring_bitmap_t,

    fn deinit(self: *CRoaringSparseInputs) void {
        c.roaring_bitmap_free(self.a);
        c.roaring_bitmap_free(self.b);
    }
};

const RequestedTuple = struct {
    row: *const ManifestRow,
    variant: Variant,
};

pub fn main(init: std.process.Init) !void {
    try validateManifest();

    // Argument parsing must not precondition either allocator under test.
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var list = false;
    var header = false;
    var row_id: ?[]const u8 = null;
    var implementation: ?Implementation = null;
    var allocator: ?AllocatorKind = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--list")) {
            list = true;
        } else if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--row=")) {
            row_id = arg[6..];
        } else if (std.mem.startsWith(u8, arg, "--implementation=")) {
            implementation = parseImplementation(arg[17..]) orelse return error.UnknownImplementation;
        } else if (std.mem.startsWith(u8, arg, "--allocator=")) {
            allocator = parseAllocator(arg[12..]) orelse return error.UnknownAllocator;
        } else {
            return error.UnknownArgument;
        }
    }

    if (list) {
        if (header or row_id != null or implementation != null or allocator != null) return error.ConflictingArguments;
        printManifest();
        return;
    }
    if (header) {
        if (row_id != null or implementation != null or allocator != null) return error.ConflictingArguments;
        printHeader();
        return;
    }

    const requested = try resolveTuple(
        row_id orelse return error.MissingRow,
        implementation orelse return error.MissingImplementation,
        allocator orelse return error.MissingAllocator,
    );
    const median_ns = try runTuple(requested);

    // RESULT is emitted only after the untimed validation in runTuple succeeds.
    bench_time.print("RESULT\t{s}\t{s}\t{s}\t{s}\t{d}\t{d}\n", .{
        requested.row.id,
        @tagName(requested.variant.implementation),
        @tagName(requested.variant.allocator),
        requested.row.reporting_unit.name(),
        requested.row.batch_count,
        median_ns,
    });
}

fn validateManifest() !void {
    for (&manifest, 0..) |*row, i| {
        if (row.id.len == 0 or row.variants.len == 0 or row.batch_count == 0) return error.InvalidManifestRow;
        if (hasProtocolDelimiter(row.id) or
            hasProtocolDelimiter(row.display_name) or
            hasProtocolDelimiter(row.corpus) or
            hasProtocolDelimiter(row.rawr_operation) or
            hasProtocolDelimiter(row.croaring_operation) or
            hasProtocolDelimiter(row.setup_boundary) or
            hasProtocolDelimiter(row.teardown_boundary))
        {
            return error.InvalidManifestText;
        }

        for (manifest[i + 1 ..]) |other| {
            if (std.mem.eql(u8, row.id, other.id)) return error.DuplicateManifestRow;
        }
        for (row.variants, 0..) |variant, variant_index| {
            for (row.variants[variant_index + 1 ..]) |other| {
                if (variant.implementation == other.implementation and variant.allocator == other.allocator) {
                    return error.DuplicateManifestVariant;
                }
            }
        }

        if (row.reference) |reference| {
            if (!manifestHasTuple(reference.row_id, reference.variant)) return error.InvalidManifestReference;
        } else if (hasRawrVariant(row) and findCRoaringVariant(row) == null) {
            return error.MissingCRoaringReference;
        }
    }
}

fn hasProtocolDelimiter(text: []const u8) bool {
    return std.mem.indexOfAny(u8, text, "\t\r\n") != null;
}

fn hasRawrVariant(row: *const ManifestRow) bool {
    for (row.variants) |variant| {
        if (variant.implementation == .rawr) return true;
    }
    return false;
}

fn manifestHasTuple(row_id: []const u8, key: TupleKey) bool {
    for (&manifest) |*row| {
        if (!std.mem.eql(u8, row.id, row_id)) continue;
        for (row.variants) |variant| {
            if (variant.implementation == key.implementation and variant.allocator == key.allocator) return true;
        }
        return false;
    }
    return false;
}

fn printHeader() void {
    bench_time.printBenchEnvironment();
    bench_time.print("# requested-cpu: native\n", .{});
    bench_time.print("# protocol: {d}w/{d}t median\n", .{ warmup_runs, timed_runs });
    bench_time.print("# croaring-avx512: {s}\n", .{
        if (c.CROARING_COMPILER_SUPPORTS_AVX512 != 0) "on" else "off",
    });
}

fn printManifest() void {
    for (&manifest) |*row| {
        if (row.reference) |reference| {
            bench_time.print("ROW\t{s}\t{s}\t{s}\t{d}\t{s}\t{s}\t{s}\t{s}\t{d}\t{s}\t{s}\t{s}\t{s}\t{s}-{s}\n", .{
                row.id,
                row.display_name,
                row.corpus,
                row.seed,
                row.rawr_operation,
                row.croaring_operation,
                @tagName(row.allocation_class),
                row.reporting_unit.name(),
                row.batch_count,
                row.setup_boundary,
                row.teardown_boundary,
                @tagName(row.validation_oracle),
                reference.row_id,
                @tagName(reference.variant.implementation),
                @tagName(reference.variant.allocator),
            });
        } else {
            bench_time.print("ROW\t{s}\t{s}\t{s}\t{d}\t{s}\t{s}\t{s}\t{s}\t{d}\t{s}\t{s}\t{s}\t-\t-\n", .{
                row.id,
                row.display_name,
                row.corpus,
                row.seed,
                row.rawr_operation,
                row.croaring_operation,
                @tagName(row.allocation_class),
                row.reporting_unit.name(),
                row.batch_count,
                row.setup_boundary,
                row.teardown_boundary,
                @tagName(row.validation_oracle),
            });
        }

        for (row.variants) |variant| {
            if (variant.implementation == .rawr) {
                const reference = row.reference orelse Reference{
                    .row_id = row.id,
                    .variant = findCRoaringVariant(row) orelse unreachable,
                };
                bench_time.print("TUPLE\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n", .{
                    row.id,
                    @tagName(variant.implementation),
                    @tagName(variant.allocator),
                    reference.row_id,
                    @tagName(reference.variant.implementation),
                    @tagName(reference.variant.allocator),
                });
            } else {
                bench_time.print("TUPLE\t{s}\t{s}\t{s}\t-\t-\t-\n", .{
                    row.id,
                    @tagName(variant.implementation),
                    @tagName(variant.allocator),
                });
            }
        }
    }
}

fn findCRoaringVariant(row: *const ManifestRow) ?TupleKey {
    for (row.variants) |variant| {
        if (variant.implementation == .croaring) {
            return .{ .implementation = variant.implementation, .allocator = variant.allocator };
        }
    }
    return null;
}

fn resolveTuple(row_id: []const u8, implementation: Implementation, allocator: AllocatorKind) !RequestedTuple {
    for (&manifest) |*row| {
        if (!std.mem.eql(u8, row.id, row_id)) continue;
        for (row.variants) |variant| {
            if (variant.implementation == implementation and variant.allocator == allocator) {
                return .{ .row = row, .variant = variant };
            }
        }
        return error.UnsupportedTuple;
    }
    return error.UnknownRow;
}

fn parseImplementation(name: []const u8) ?Implementation {
    if (std.mem.eql(u8, name, "rawr")) return .rawr;
    if (std.mem.eql(u8, name, "croaring")) return .croaring;
    return null;
}

fn parseAllocator(name: []const u8) ?AllocatorKind {
    if (std.mem.eql(u8, name, "none")) return .none;
    if (std.mem.eql(u8, name, "smp")) return .smp;
    if (std.mem.eql(u8, name, "libc")) return .libc;
    if (std.mem.eql(u8, name, "arena")) return .arena;
    return null;
}

fn runTuple(requested: RequestedTuple) !u64 {
    return switch (requested.row.operation) {
        .sparse_and => runSparseAndTuple(requested),
        .cardinality => runCardinalityTuple(requested),
    };
}

fn initSparseValues(seed: u64) usize {
    var prng = std.Random.DefaultPrng.init(seed);
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

fn buildRawrSparseInputs(sparse_len: usize) !RawrSparseInputs {
    var a = try RoaringBitmap.init(std.heap.smp_allocator);
    errdefer a.deinit();
    var b = try RoaringBitmap.init(std.heap.smp_allocator);
    errdefer b.deinit();

    const half = sparse_len / 2;
    try a.addMany(sparse_values[0..half]);
    try b.addMany(sparse_values[half / 2 .. sparse_len]);
    return .{ .a = a, .b = b };
}

fn buildCRoaringSparseInputs(sparse_len: usize) !CRoaringSparseInputs {
    const a = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(a);
    const b = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(b);

    const half = sparse_len / 2;
    c.roaring_bitmap_add_many(a, half, sparse_values[0..half].ptr);
    c.roaring_bitmap_add_many(b, sparse_len - half / 2, sparse_values[half / 2 .. sparse_len].ptr);
    return .{ .a = a, .b = b };
}

fn runSparseAndTuple(requested: RequestedTuple) !u64 {
    const sparse_len = initSparseValues(requested.row.seed);
    const median_ns = switch (requested.variant.implementation) {
        .rawr => rawr: {
            var inputs = try buildRawrSparseInputs(sparse_len);
            defer inputs.deinit();
            const allocator = switch (requested.variant.allocator) {
                .smp => std.heap.smp_allocator,
                .libc => bench_time.cAllocator(),
                else => return error.UnsupportedAllocator,
            };
            break :rawr try measure(runRawrSparseAnd, .{ &inputs, allocator }, requested.row.batch_count);
        },
        .croaring => croaring: {
            var inputs = try buildCRoaringSparseInputs(sparse_len);
            defer inputs.deinit();
            break :croaring try measure(runCRoaringSparseAnd, .{&inputs}, requested.row.batch_count);
        },
    };

    try validateSparseAnd(requested, sparse_len);
    return median_ns;
}

noinline fn runRawrSparseAnd(inputs: *const RawrSparseInputs, allocator: std.mem.Allocator) !u64 {
    var result = try inputs.a.bitwiseAnd(allocator, &inputs.b);
    defer result.deinit();
    const cardinality = result.cardinality();
    std.mem.doNotOptimizeAway(&result);
    return cardinality;
}

noinline fn runCRoaringSparseAnd(inputs: *const CRoaringSparseInputs) !u64 {
    const result = c.roaring_bitmap_and(inputs.a, inputs.b) orelse return error.OutOfMemory;
    defer c.roaring_bitmap_free(result);
    const cardinality = c.roaring_bitmap_get_cardinality(result);
    std.mem.doNotOptimizeAway(result);
    return cardinality;
}

fn validateSparseAnd(requested: RequestedTuple, sparse_len: usize) !void {
    switch (requested.variant.implementation) {
        .rawr => {
            var rawr_inputs = try buildRawrSparseInputs(sparse_len);
            defer rawr_inputs.deinit();
            const allocator = switch (requested.variant.allocator) {
                .smp => std.heap.smp_allocator,
                .libc => bench_time.cAllocator(),
                else => return error.UnsupportedAllocator,
            };
            var rawr_result = try rawr_inputs.a.bitwiseAnd(allocator, &rawr_inputs.b);
            defer rawr_result.deinit();

            var cr_inputs = try buildCRoaringSparseInputs(sparse_len);
            defer cr_inputs.deinit();
            const cr_result = c.roaring_bitmap_and(cr_inputs.a, cr_inputs.b) orelse return error.OutOfMemory;
            defer c.roaring_bitmap_free(cr_result);
            try expectPortableEqual(&rawr_result, cr_result);
        },
        .croaring => {
            var cr_inputs = try buildCRoaringSparseInputs(sparse_len);
            defer cr_inputs.deinit();
            const cr_result = c.roaring_bitmap_and(cr_inputs.a, cr_inputs.b) orelse return error.OutOfMemory;
            defer c.roaring_bitmap_free(cr_result);

            var rawr_inputs = try buildRawrSparseInputs(sparse_len);
            defer rawr_inputs.deinit();
            var rawr_result = try rawr_inputs.a.bitwiseAnd(std.heap.smp_allocator, &rawr_inputs.b);
            defer rawr_result.deinit();
            try expectPortableEqual(&rawr_result, cr_result);
        },
    }
}

fn expectPortableEqual(rawr_result: *const RoaringBitmap, cr_result: *const c.roaring_bitmap_t) !void {
    const allocator = std.heap.page_allocator;
    const rawr_bytes = try rawr_result.serialize(allocator);
    defer allocator.free(rawr_bytes);

    const cr_len = c.roaring_bitmap_portable_size_in_bytes(cr_result);
    if (rawr_bytes.len != cr_len) return error.SerializedSizeMismatch;
    const cr_bytes = try allocator.alloc(u8, cr_len);
    defer allocator.free(cr_bytes);
    if (c.roaring_bitmap_portable_serialize(cr_result, @ptrCast(cr_bytes.ptr)) != cr_len) {
        return error.SerializedSizeMismatch;
    }
    if (!std.mem.eql(u8, rawr_bytes, cr_bytes)) return error.CRoaringMismatch;
}

fn initCardinalityValues(seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    for (cardinality_values[0..]) |*value| value.* = prng.random().int(u32);
}

fn buildRawrCardinalityBitmap() !RoaringBitmap {
    var bitmap = try RoaringBitmap.init(std.heap.smp_allocator);
    errdefer bitmap.deinit();
    for (cardinality_values[0..]) |value| _ = try bitmap.add(value);
    return bitmap;
}

fn buildCRoaringCardinalityBitmap() !*c.roaring_bitmap_t {
    const bitmap = c.roaring_bitmap_create() orelse return error.OutOfMemory;
    errdefer c.roaring_bitmap_free(bitmap);
    for (cardinality_values[0..]) |value| c.roaring_bitmap_add(bitmap, value);
    return bitmap;
}

fn runCardinalityTuple(requested: RequestedTuple) !u64 {
    initCardinalityValues(requested.row.seed);
    const median_ns = switch (requested.variant.implementation) {
        .rawr => rawr: {
            var bitmap = try buildRawrCardinalityBitmap();
            defer bitmap.deinit();
            break :rawr try measure(runRawrCardinality, .{&bitmap}, requested.row.batch_count);
        },
        .croaring => croaring: {
            const bitmap = try buildCRoaringCardinalityBitmap();
            defer c.roaring_bitmap_free(bitmap);
            break :croaring try measure(runCRoaringCardinality, .{bitmap}, requested.row.batch_count);
        },
    };

    try validateCardinality(requested);
    return median_ns;
}

noinline fn runRawrCardinality(bitmap: *const RoaringBitmap) !u64 {
    return bitmap.cardinality();
}

noinline fn runCRoaringCardinality(bitmap: *const c.roaring_bitmap_t) !u64 {
    return c.roaring_bitmap_get_cardinality(bitmap);
}

fn validateCardinality(requested: RequestedTuple) !void {
    switch (requested.variant.implementation) {
        .rawr => {
            var rawr_bitmap = try buildRawrCardinalityBitmap();
            defer rawr_bitmap.deinit();
            const cr_bitmap = try buildCRoaringCardinalityBitmap();
            defer c.roaring_bitmap_free(cr_bitmap);
            if (rawr_bitmap.cardinality() != c.roaring_bitmap_get_cardinality(cr_bitmap)) {
                return error.CardinalityMismatch;
            }
        },
        .croaring => {
            const cr_bitmap = try buildCRoaringCardinalityBitmap();
            defer c.roaring_bitmap_free(cr_bitmap);
            var rawr_bitmap = try buildRawrCardinalityBitmap();
            defer rawr_bitmap.deinit();
            if (rawr_bitmap.cardinality() != c.roaring_bitmap_get_cardinality(cr_bitmap)) {
                return error.CardinalityMismatch;
            }
        },
    }
}

fn measure(comptime operation: anytype, args: anytype, batch_count: usize) !u64 {
    for (0..warmup_runs) |_| try runBatch(operation, args, batch_count);

    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| {
        const start = bench_time.monotonicNanos();
        try runBatch(operation, args, batch_count);
        elapsed.* = bench_time.monotonicNanos() - start;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

fn runBatch(comptime operation: anytype, args: anytype, batch_count: usize) !void {
    var checksum: u64 = 0;
    for (0..batch_count) |_| checksum +%= try @call(.auto, operation, args);
    std.mem.doNotOptimizeAway(checksum);
}
