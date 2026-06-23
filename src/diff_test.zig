const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");
const test_gen = rawr.test_gen;

const Allocator = std.mem.Allocator;
const RoaringBitmap = rawr.RoaringBitmap;
const Profile = test_gen.Profile;

const PRINT_PASSES = false;
const RANDOM_SEED: u64 = 0xD1FF_7E57_0001;
const RANDOM_ITERS: usize = 1000;
const RANDOM_MAX_CHUNKS: usize = 3;
const PRINT_RANDOM_PROGRESS = false;
const ORACLE_IDENTITY_SEED: u64 = 0x1D3A_71AF_0001;
const ORACLE_IDENTITY_ITERS: usize = 50;

const BinaryOp = enum {
    bitwise_or,
    bitwise_and,
    bitwise_xor,
    bitwise_difference,

    fn name(self: BinaryOp) []const u8 {
        return switch (self) {
            .bitwise_or => "or",
            .bitwise_and => "and",
            .bitwise_xor => "xor",
            .bitwise_difference => "andnot",
        };
    }
};

const MatrixCase = struct {
    name: []const u8,
    a: ?Profile,
    b: ?Profile,
};

const Range = struct {
    start: u32,
    end: u32,
};

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        try runSparseDenseOrCase(allocator);
        try runOperationMatrix(allocator);
        try runJaccardEmptyCase(allocator);
        try runFlipCases(allocator);
        try runRangeCases(allocator);
        try runNwayCases(allocator);
        try runTransitionCases(allocator);
        try runOracleAnchoredIdentities(allocator);
        try runRandomizedLoop(allocator);
    }

    if (gpa.deinit() != .ok) {
        return error.MemoryLeak;
    }
}

fn runSparseDenseOrCase(allocator: Allocator) !void {
    var prng = std.Random.DefaultPrng.init(0x01_03_00_01);
    const rng = prng.random();

    const a_chunks = [_]test_gen.ChunkProfile{
        .{ .key = 42, .profile = .sparse },
    };
    var a = try test_gen.build(allocator, rng, &a_chunks, false);
    defer a.deinit();

    const b_chunks = [_]test_gen.ChunkProfile{
        .{ .key = 42, .profile = .dense },
    };
    var b = try test_gen.build(allocator, rng, &b_chunks, false);
    defer b.deinit();

    const oracle_a = try buildOracle(a.values, false);
    defer c.roaring_bitmap_free(oracle_a);
    const oracle_b = try buildOracle(b.values, false);
    defer c.roaring_bitmap_free(oracle_b);

    var rawr_result = try a.bm.bitwiseOr(allocator, &b.bm);
    defer rawr_result.deinit();

    const oracle_result = c.roaring_bitmap_or(oracle_a, oracle_b) orelse return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(oracle_result);

    try assertAgree(allocator, "or sparse|dense", &rawr_result, oracle_result);
}

fn runOperationMatrix(allocator: Allocator) !void {
    const cases = [_]MatrixCase{
        .{ .name = "sparse_sparse", .a = .sparse, .b = .sparse },
        .{ .name = "sparse_dense", .a = .sparse, .b = .dense },
        .{ .name = "dense_sparse", .a = .dense, .b = .sparse },
        .{ .name = "dense_dense", .a = .dense, .b = .dense },
        .{ .name = "sparse_runs", .a = .sparse, .b = .runs },
        .{ .name = "runs_sparse", .a = .runs, .b = .sparse },
        .{ .name = "dense_runs", .a = .dense, .b = .runs },
        .{ .name = "runs_dense", .a = .runs, .b = .dense },
        .{ .name = "runs_runs", .a = .runs, .b = .runs },
        .{ .name = "full_sparse", .a = .full, .b = .sparse },
        .{ .name = "x_empty", .a = .dense, .b = null },
        .{ .name = "empty_x", .a = null, .b = .dense },
    };

    for (&[_]bool{ false, true }) |run_optimize| {
        for (cases) |case| {
            try runMatrixCase(allocator, case, run_optimize);
        }
    }
}

fn runJaccardEmptyCase(allocator: Allocator) !void {
    var a = try RoaringBitmap.init(allocator);
    defer a.deinit();
    var b = try RoaringBitmap.init(allocator);
    defer b.deinit();

    const empty_values = [_]u32{};
    const oracle_a = try buildOracle(&empty_values, false);
    defer c.roaring_bitmap_free(oracle_a);
    const oracle_b = try buildOracle(&empty_values, false);
    defer c.roaring_bitmap_free(oracle_b);

    try expectEqualFloat(
        "jaccard_empty_empty",
        "a_b",
        "jaccardIndex",
        a.jaccardIndex(&b),
        c.roaring_bitmap_jaccard_index(oracle_a, oracle_b),
    );
}

fn runFlipCases(allocator: Allocator) !void {
    for (&[_]bool{ false, true }) |run_optimize| {
        var prng = std.Random.DefaultPrng.init(if (run_optimize) 0xF11F_0001 else 0xF11F_0000);
        const rng = prng.random();

        for (&[_]Profile{ .sparse, .dense, .runs }) |profile| {
            const chunks = [_]test_gen.ChunkProfile{.{ .key = 77, .profile = profile }};
            var generated = try test_gen.build(allocator, rng, &chunks, run_optimize);
            defer generated.deinit();

            const oracle = try buildOracle(generated.values, run_optimize);
            defer c.roaring_bitmap_free(oracle);

            var name_buf: [96]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "flip:within:{s}:runopt={}", .{ @tagName(profile), run_optimize });
            try assertFlipAgree(allocator, name, &generated.bm, oracle, (77 << 16) + 10, (77 << 16) + 2000);
        }

        {
            const chunks = [_]test_gen.ChunkProfile{.{ .key = 77, .profile = .full }};
            var generated = try test_gen.build(allocator, rng, &chunks, run_optimize);
            defer generated.deinit();

            const oracle = try buildOracle(generated.values, run_optimize);
            defer c.roaring_bitmap_free(oracle);
            try assertFlipAgree(allocator, "flip:full-populated", &generated.bm, oracle, 77 << 16, (77 << 16) + 65_535);
        }

        {
            var bm = try RoaringBitmap.init(allocator);
            defer bm.deinit();
            const empty_values = [_]u32{};
            const oracle = try buildOracle(&empty_values, run_optimize);
            defer c.roaring_bitmap_free(oracle);
            try assertFlipAgree(allocator, "flip:full-empty", &bm, oracle, 78 << 16, (78 << 16) + 65_535);
        }

        {
            const chunks = [_]test_gen.ChunkProfile{
                .{ .key = 76, .profile = .sparse },
                .{ .key = 78, .profile = .dense },
            };
            var generated = try test_gen.build(allocator, rng, &chunks, run_optimize);
            defer generated.deinit();

            const oracle = try buildOracle(generated.values, run_optimize);
            defer c.roaring_bitmap_free(oracle);
            try assertFlipAgree(allocator, "flip:cross-chunk", &generated.bm, oracle, (76 << 16) + 50_000, (79 << 16) + 100);
        }

        {
            const values = try valuesFromRanges(allocator, &.{
                .{ .start = 0, .end = 10 },
                .{ .start = 65_530, .end = 65_540 },
            });
            defer allocator.free(values);

            var bm = try buildRawrFromValues(allocator, values, run_optimize);
            defer bm.deinit();
            const oracle = try buildOracle(values, run_optimize);
            defer c.roaring_bitmap_free(oracle);

            try assertFlipAgree(allocator, "flip:single-zero", &bm, oracle, 0, 0);
            try assertFlipAgree(allocator, "flip:boundary", &bm, oracle, 65_535, 65_536);
            try assertFlipAgree(allocator, "flip:empty-range", &bm, oracle, 100, 99);
        }
    }
}

fn runRangeCases(allocator: Allocator) !void {
    for (&[_]bool{ false, true }) |run_optimize| {
        var prng = std.Random.DefaultPrng.init(if (run_optimize) 0xA11E_0001 else 0xA11E_0000);
        const rng = prng.random();

        for (&[_]Profile{ .sparse, .dense, .runs }) |profile| {
            const chunks = [_]test_gen.ChunkProfile{.{ .key = 77, .profile = profile }};
            var generated = try test_gen.build(allocator, rng, &chunks, run_optimize);
            defer generated.deinit();

            const oracle = try buildOracle(generated.values, run_optimize);
            defer c.roaring_bitmap_free(oracle);

            var name_buf: [96]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "range:within:{s}:runopt={}", .{ @tagName(profile), run_optimize });
            try assertRangeOpsAgree(allocator, name, &generated.bm, oracle, (77 << 16) + 10, (77 << 16) + 2000);
        }

        {
            const chunks = [_]test_gen.ChunkProfile{.{ .key = 77, .profile = .full }};
            var generated = try test_gen.build(allocator, rng, &chunks, run_optimize);
            defer generated.deinit();

            const oracle = try buildOracle(generated.values, run_optimize);
            defer c.roaring_bitmap_free(oracle);
            try assertRangeOpsAgree(allocator, "range:full-populated", &generated.bm, oracle, 77 << 16, (77 << 16) + 65_535);
        }

        {
            var bm = try RoaringBitmap.init(allocator);
            defer bm.deinit();
            const empty_values = [_]u32{};
            const oracle = try buildOracle(&empty_values, run_optimize);
            defer c.roaring_bitmap_free(oracle);
            try assertRangeOpsAgree(allocator, "range:full-empty", &bm, oracle, 78 << 16, (78 << 16) + 65_535);
        }

        {
            const chunks = [_]test_gen.ChunkProfile{
                .{ .key = 76, .profile = .sparse },
                .{ .key = 78, .profile = .dense },
            };
            var generated = try test_gen.build(allocator, rng, &chunks, run_optimize);
            defer generated.deinit();

            const oracle = try buildOracle(generated.values, run_optimize);
            defer c.roaring_bitmap_free(oracle);
            try assertRangeOpsAgree(allocator, "range:cross-chunk", &generated.bm, oracle, (76 << 16) + 50_000, (79 << 16) + 100);
        }

        {
            const values = try valuesFromRanges(allocator, &.{
                .{ .start = 0, .end = 10 },
                .{ .start = 65_530, .end = 65_540 },
                .{ .start = std.math.maxInt(u32) - 10, .end = std.math.maxInt(u32) },
            });
            defer allocator.free(values);

            var bm = try buildRawrFromValues(allocator, values, run_optimize);
            defer bm.deinit();
            const oracle = try buildOracle(values, run_optimize);
            defer c.roaring_bitmap_free(oracle);

            try assertRangeOpsAgree(allocator, "range:single-zero", &bm, oracle, 0, 0);
            try assertRangeOpsAgree(allocator, "range:boundary", &bm, oracle, 65_535, 65_536);
            try assertRangeOpsAgree(allocator, "range:max-boundary", &bm, oracle, std.math.maxInt(u32) - 10, std.math.maxInt(u32));
            try assertRangeOpsAgree(allocator, "range:empty-range", &bm, oracle, 100, 99);
        }
    }
}

const ManyOp = enum {
    bor,
    xor,

    fn name(self: ManyOp) []const u8 {
        return switch (self) {
            .bor => "orMany",
            .xor => "xorMany",
        };
    }
};

fn runNwayCases(allocator: Allocator) !void {
    try runNwayPureRawrEdges(allocator);
    try runNwayGeneratedOracleCases(allocator);
    try runNwayHeterogeneousSameKeyCase(allocator);
}

fn runNwayPureRawrEdges(allocator: Allocator) !void {
    var empty_or = try RoaringBitmap.orMany(allocator, &.{});
    defer empty_or.deinit();
    try expectRawrEmpty("orMany:empty-list", &empty_or);

    var empty_xor = try RoaringBitmap.xorMany(allocator, &.{});
    defer empty_xor.deinit();
    try expectRawrEmpty("xorMany:empty-list", &empty_xor);

    var prng = std.Random.DefaultPrng.init(0x0A11_0A11);
    const rng = prng.random();

    var a = try test_gen.randomMixed(allocator, rng, 4, true);
    defer a.deinit();
    var b = try test_gen.randomMixed(allocator, rng, 4, false);
    defer b.deinit();
    var gen_c = try test_gen.randomMixed(allocator, rng, 4, true);
    defer gen_c.deinit();

    {
        var result = try RoaringBitmap.orMany(allocator, &.{&a.bm});
        defer result.deinit();
        try expectRawrEqual("orMany:single", &result, &a.bm);
        _ = try result.add(std.math.maxInt(u32));
        if (a.bm.contains(std.math.maxInt(u32))) return error.NwayAliasedInput;
    }

    {
        var result = try RoaringBitmap.xorMany(allocator, &.{&a.bm});
        defer result.deinit();
        try expectRawrEqual("xorMany:single", &result, &a.bm);
        _ = try result.add(std.math.maxInt(u32));
        if (a.bm.contains(std.math.maxInt(u32))) return error.NwayAliasedInput;
    }

    {
        var many = try RoaringBitmap.orMany(allocator, &.{ &a.bm, &b.bm });
        defer many.deinit();
        var two_way = try a.bm.bitwiseOr(allocator, &b.bm);
        defer two_way.deinit();
        try expectRawrEqual("orMany:two-way-cross-check", &many, &two_way);
    }

    {
        var many = try RoaringBitmap.xorMany(allocator, &.{ &a.bm, &b.bm });
        defer many.deinit();
        var two_way = try a.bm.bitwiseXor(allocator, &b.bm);
        defer two_way.deinit();
        try expectRawrEqual("xorMany:two-way-cross-check", &many, &two_way);
    }

    {
        var many = try RoaringBitmap.orMany(allocator, &.{ &a.bm, &b.bm, &gen_c.bm });
        defer many.deinit();
        var ab = try a.bm.bitwiseOr(allocator, &b.bm);
        defer ab.deinit();
        var folded = try ab.bitwiseOr(allocator, &gen_c.bm);
        defer folded.deinit();
        try expectRawrEqual("orMany:three-way-cross-check", &many, &folded);
    }

    {
        var owned = try RoaringBitmap.orManyOwned(allocator, &.{ &a.bm, &b.bm });
        defer owned.deinit();
        var expected = try a.bm.bitwiseOr(allocator, &b.bm);
        defer expected.deinit();
        try expectRawrEqual("orManyOwned:two-way", &owned.bitmap, &expected);
    }

    {
        var owned = try RoaringBitmap.xorManyOwned(allocator, &.{ &a.bm, &b.bm });
        defer owned.deinit();
        var expected = try a.bm.bitwiseXor(allocator, &b.bm);
        defer expected.deinit();
        try expectRawrEqual("xorManyOwned:two-way", &owned.bitmap, &expected);
    }
}

fn runNwayGeneratedOracleCases(allocator: Allocator) !void {
    const ns = [_]usize{ 1, 2, 3, 8, 32 };

    for (&[_]bool{ false, true }) |run_optimize| {
        var prng = std.Random.DefaultPrng.init(if (run_optimize) 0x0B0A_0001 else 0x0B0A_0000);
        const rng = prng.random();

        for (ns) |n| {
            var generated = try allocator.alloc(test_gen.Generated, n);
            defer allocator.free(generated);
            var generated_len: usize = 0;
            defer {
                for (generated[0..generated_len]) |*gen| gen.deinit();
            }

            var rawr_inputs = try allocator.alloc(*const RoaringBitmap, n);
            defer allocator.free(rawr_inputs);
            var oracle_inputs = try allocator.alloc(*c.roaring_bitmap_t, n);
            defer allocator.free(oracle_inputs);
            var oracle_len: usize = 0;
            defer {
                for (oracle_inputs[0..oracle_len]) |oracle| {
                    c.roaring_bitmap_free(oracle);
                }
            }

            for (0..n) |i| {
                generated[i] = try test_gen.randomMixed(allocator, rng, 6, run_optimize);
                generated_len += 1;
                rawr_inputs[i] = &generated[i].bm;
                oracle_inputs[i] = try buildOracle(generated[i].values, run_optimize);
                oracle_len += 1;
            }

            try assertManyAgree(allocator, .bor, rawr_inputs, oracle_inputs, run_optimize, n);
            try assertManyAgree(allocator, .xor, rawr_inputs, oracle_inputs, run_optimize, n);
        }

        try runNwayWithEmptyInputs(allocator, rng, run_optimize);
    }
}

fn runNwayWithEmptyInputs(allocator: Allocator, rng: std.Random, run_optimize: bool) !void {
    var empty = try RoaringBitmap.init(allocator);
    defer empty.deinit();
    const empty_values = [_]u32{};

    var a = try test_gen.randomMixed(allocator, rng, 4, run_optimize);
    defer a.deinit();
    var b = try test_gen.randomMixed(allocator, rng, 4, run_optimize);
    defer b.deinit();

    const rawr_inputs = [_]*const RoaringBitmap{ &empty, &a.bm, &b.bm };

    const oracle_empty = try buildOracle(&empty_values, run_optimize);
    defer c.roaring_bitmap_free(oracle_empty);
    const oracle_a = try buildOracle(a.values, run_optimize);
    defer c.roaring_bitmap_free(oracle_a);
    const oracle_b = try buildOracle(b.values, run_optimize);
    defer c.roaring_bitmap_free(oracle_b);
    const oracle_inputs = [_]*c.roaring_bitmap_t{ oracle_empty, oracle_a, oracle_b };

    try assertManyAgree(allocator, .bor, &rawr_inputs, &oracle_inputs, run_optimize, rawr_inputs.len);
    try assertManyAgree(allocator, .xor, &rawr_inputs, &oracle_inputs, run_optimize, rawr_inputs.len);
}

fn runNwayHeterogeneousSameKeyCase(allocator: Allocator) !void {
    var prng = std.Random.DefaultPrng.init(0x0A_B17E_57);
    const rng = prng.random();
    const key: u16 = 123;

    var array_gen = try test_gen.build(allocator, rng, &.{.{ .key = key, .profile = .sparse }}, false);
    defer array_gen.deinit();
    var bitset_gen = try test_gen.build(allocator, rng, &.{.{ .key = key, .profile = .dense }}, false);
    defer bitset_gen.deinit();
    var run_gen = try test_gen.build(allocator, rng, &.{.{ .key = key, .profile = .runs }}, true);
    defer run_gen.deinit();

    const rawr_inputs = [_]*const RoaringBitmap{ &array_gen.bm, &bitset_gen.bm, &run_gen.bm };

    const oracle_array = try buildOracle(array_gen.values, false);
    defer c.roaring_bitmap_free(oracle_array);
    const oracle_bitset = try buildOracle(bitset_gen.values, false);
    defer c.roaring_bitmap_free(oracle_bitset);
    const oracle_run = try buildOracle(run_gen.values, true);
    defer c.roaring_bitmap_free(oracle_run);
    const oracle_inputs = [_]*c.roaring_bitmap_t{ oracle_array, oracle_bitset, oracle_run };

    try assertManyAgree(allocator, .bor, &rawr_inputs, &oracle_inputs, true, rawr_inputs.len);
    try assertManyAgree(allocator, .xor, &rawr_inputs, &oracle_inputs, true, rawr_inputs.len);
}

fn runTransitionCases(allocator: Allocator) !void {
    try runPromotionCase(allocator);
    try runDemotionCase(allocator);
    try runEmptyOutCase(allocator);
    try runBoundarySplitCase(allocator);
}

fn runRandomizedLoop(allocator: Allocator) !void {
    std.debug.print("random difftest seed=0x{x}, iters={d}, max_chunks={d}\n", .{
        RANDOM_SEED,
        RANDOM_ITERS,
        RANDOM_MAX_CHUNKS,
    });

    var prng = std.Random.DefaultPrng.init(RANDOM_SEED);
    const rng = prng.random();

    for (0..RANDOM_ITERS) |i| {
        const run_optimize = (i % 2) == 1;
        if (PRINT_RANDOM_PROGRESS) {
            std.debug.print("random iteration {d}, run_optimize={}\n", .{ i, run_optimize });
        }
        runRandomIteration(allocator, rng, i, run_optimize) catch |err| {
            std.debug.print("FAIL: random iteration {d}, seed=0x{x}, run_optimize={} -> {s}\n", .{
                i,
                RANDOM_SEED,
                run_optimize,
                @errorName(err),
            });
            return err;
        };
    }
}

fn runOracleAnchoredIdentities(allocator: Allocator) !void {
    var prng = std.Random.DefaultPrng.init(ORACLE_IDENTITY_SEED);
    const rng = prng.random();

    for (0..ORACLE_IDENTITY_ITERS) |i| {
        const run_optimize = (i % 2) == 1;
        try runDistributivityIdentity(allocator, rng, i, run_optimize);
        try runXorDecompositionIdentity(allocator, rng, i, run_optimize);
    }
}

fn runDistributivityIdentity(allocator: Allocator, rng: std.Random, iteration: usize, run_optimize: bool) !void {
    var a = try test_gen.randomMixed(allocator, rng, RANDOM_MAX_CHUNKS, run_optimize);
    defer a.deinit();
    var b = try test_gen.randomMixed(allocator, rng, RANDOM_MAX_CHUNKS, run_optimize);
    defer b.deinit();
    var gen_c = try test_gen.randomMixed(allocator, rng, RANDOM_MAX_CHUNKS, run_optimize);
    defer gen_c.deinit();

    const oracle_a = try buildOracle(a.values, run_optimize);
    defer c.roaring_bitmap_free(oracle_a);
    const oracle_b = try buildOracle(b.values, run_optimize);
    defer c.roaring_bitmap_free(oracle_b);
    const oracle_c = try buildOracle(gen_c.values, run_optimize);
    defer c.roaring_bitmap_free(oracle_c);

    var rawr_b_or_c = try b.bm.bitwiseOr(allocator, &gen_c.bm);
    defer rawr_b_or_c.deinit();
    var rawr_lhs = try a.bm.bitwiseAnd(allocator, &rawr_b_or_c);
    defer rawr_lhs.deinit();

    const oracle_b_or_c = try oracleAllocatingOp(.bitwise_or, oracle_b, oracle_c);
    defer c.roaring_bitmap_free(oracle_b_or_c);
    const oracle_lhs = try oracleAllocatingOp(.bitwise_and, oracle_a, oracle_b_or_c);
    defer c.roaring_bitmap_free(oracle_lhs);

    var name_buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "identity:distributivity:{d}", .{iteration});
    try assertAgree(allocator, name, &rawr_lhs, oracle_lhs);
}

fn runXorDecompositionIdentity(allocator: Allocator, rng: std.Random, iteration: usize, run_optimize: bool) !void {
    var a = try test_gen.randomMixed(allocator, rng, RANDOM_MAX_CHUNKS, run_optimize);
    defer a.deinit();
    var b = try test_gen.randomMixed(allocator, rng, RANDOM_MAX_CHUNKS, run_optimize);
    defer b.deinit();

    const oracle_a = try buildOracle(a.values, run_optimize);
    defer c.roaring_bitmap_free(oracle_a);
    const oracle_b = try buildOracle(b.values, run_optimize);
    defer c.roaring_bitmap_free(oracle_b);

    var rawr_xor = try a.bm.bitwiseXor(allocator, &b.bm);
    defer rawr_xor.deinit();

    const oracle_a_minus_b = try oracleAllocatingOp(.bitwise_difference, oracle_a, oracle_b);
    defer c.roaring_bitmap_free(oracle_a_minus_b);
    const oracle_b_minus_a = try oracleAllocatingOp(.bitwise_difference, oracle_b, oracle_a);
    defer c.roaring_bitmap_free(oracle_b_minus_a);
    const oracle_result = try oracleAllocatingOp(.bitwise_or, oracle_a_minus_b, oracle_b_minus_a);
    defer c.roaring_bitmap_free(oracle_result);

    var name_buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "identity:xor-decomposition:{d}", .{iteration});
    try assertAgree(allocator, name, &rawr_xor, oracle_result);
}

fn runRandomIteration(allocator: Allocator, rng: std.Random, iteration: usize, run_optimize: bool) !void {
    var a = try test_gen.randomMixed(allocator, rng, RANDOM_MAX_CHUNKS, run_optimize);
    defer a.deinit();
    var b = try test_gen.randomMixed(allocator, rng, RANDOM_MAX_CHUNKS, run_optimize);
    defer b.deinit();

    if (PRINT_RANDOM_PROGRESS) {
        printBitmapSummary("A", &a.bm);
        printBitmapSummary("B", &b.bm);
    }

    const oracle_a = try buildOracle(a.values, run_optimize);
    defer c.roaring_bitmap_free(oracle_a);
    const oracle_b = try buildOracle(b.values, run_optimize);
    defer c.roaring_bitmap_free(oracle_b);

    var case_name_buf: [64]u8 = undefined;
    const case_name = try std.fmt.bufPrint(&case_name_buf, "random:{d}", .{iteration});
    try runOrderedChecks(allocator, case_name, "a_b", &a.bm, &b.bm, oracle_a, oracle_b);
    try runOrderedChecks(allocator, case_name, "b_a", &b.bm, &a.bm, oracle_b, oracle_a);

    const flip_lo = @as(u32, @intCast((iteration % 3) * 65_536)) + rng.uintLessThan(u32, 65_536);
    const flip_len = rng.uintLessThan(u32, 150_000);
    const flip_hi = if (iteration % 10 == 0 and flip_lo > 0) flip_lo - 1 else flip_lo +| flip_len;
    var flip_name_buf: [80]u8 = undefined;
    const flip_name = try std.fmt.bufPrint(&flip_name_buf, "random:{d}:flip", .{iteration});
    try assertFlipAgree(allocator, flip_name, &a.bm, oracle_a, flip_lo, flip_hi);

    const range_lo = @as(u32, @intCast((iteration % 3) * 65_536)) + rng.uintLessThan(u32, 65_536);
    const range_len = rng.uintLessThan(u32, 150_000);
    const range_hi = if (iteration % 10 == 1 and range_lo > 0) range_lo - 1 else range_lo +| range_len;
    var range_name_buf: [80]u8 = undefined;
    const range_name = try std.fmt.bufPrint(&range_name_buf, "random:{d}:range", .{iteration});
    try assertRangeOpsAgree(allocator, range_name, &a.bm, oracle_a, range_lo, range_hi);
}

fn logPass(comptime fmt: []const u8, args: anytype) void {
    if (PRINT_PASSES) {
        std.debug.print(fmt, args);
    }
}

fn printBitmapSummary(label: []const u8, bm: *RoaringBitmap) void {
    std.debug.print("{s}: containers={d}, cardinality={d}", .{ label, bm.size, bm.cardinality() });
    for (bm.keys[0..bm.size], bm.containers[0..bm.size]) |key, container| {
        std.debug.print(" [{d}:{s}]", .{ key, @tagName(container.getType()) });
    }
    std.debug.print("\n", .{});
}

fn runPromotionCase(allocator: Allocator) !void {
    const values_a = try valuesFromRanges(allocator, &.{.{ .start = 0, .end = 2047 }});
    defer allocator.free(values_a);
    const values_b = try valuesFromRanges(allocator, &.{.{ .start = 2048, .end = 4096 }});
    defer allocator.free(values_b);

    var a = try buildRawrFromValues(allocator, values_a, false);
    defer a.deinit();
    var b = try buildRawrFromValues(allocator, values_b, false);
    defer b.deinit();
    const oracle_a = try buildOracle(values_a, false);
    defer c.roaring_bitmap_free(oracle_a);
    const oracle_b = try buildOracle(values_b, false);
    defer c.roaring_bitmap_free(oracle_b);

    var rawr_result = try a.bitwiseOr(allocator, &b);
    defer rawr_result.deinit();
    try assertSingleContainerType("transition:promotion", &rawr_result, .bitset);

    const oracle_result = try oracleAllocatingOp(.bitwise_or, oracle_a, oracle_b);
    defer c.roaring_bitmap_free(oracle_result);
    try assertAgree(allocator, "transition:promotion", &rawr_result, oracle_result);
}

fn runDemotionCase(allocator: Allocator) !void {
    const values_a = try valuesFromRanges(allocator, &.{.{ .start = 0, .end = 4999 }});
    defer allocator.free(values_a);
    const values_b = try valuesFromRanges(allocator, &.{.{ .start = 4900, .end = 9899 }});
    defer allocator.free(values_b);

    var a = try buildRawrFromValues(allocator, values_a, false);
    defer a.deinit();
    var b = try buildRawrFromValues(allocator, values_b, false);
    defer b.deinit();
    const oracle_a = try buildOracle(values_a, false);
    defer c.roaring_bitmap_free(oracle_a);
    const oracle_b = try buildOracle(values_b, false);
    defer c.roaring_bitmap_free(oracle_b);

    var rawr_result = try a.bitwiseAnd(allocator, &b);
    defer rawr_result.deinit();
    try assertSingleContainerType("transition:demotion", &rawr_result, .array);

    const oracle_result = try oracleAllocatingOp(.bitwise_and, oracle_a, oracle_b);
    defer c.roaring_bitmap_free(oracle_result);
    try assertAgree(allocator, "transition:demotion", &rawr_result, oracle_result);
}

fn runEmptyOutCase(allocator: Allocator) !void {
    const chunk1_start = @as(u32, 1) << 16;
    const chunk2_start = @as(u32, 2) << 16;

    const values_a = try valuesFromRanges(allocator, &.{
        .{ .start = 10, .end = 19 },
        .{ .start = chunk1_start + 100, .end = chunk1_start + 199 },
        .{ .start = chunk2_start + 300, .end = chunk2_start + 319 },
    });
    defer allocator.free(values_a);
    const values_b = try valuesFromRanges(allocator, &.{
        .{ .start = chunk1_start + 100, .end = chunk1_start + 199 },
    });
    defer allocator.free(values_b);

    var a = try buildRawrFromValues(allocator, values_a, false);
    defer a.deinit();
    var b = try buildRawrFromValues(allocator, values_b, false);
    defer b.deinit();
    const oracle_a = try buildOracle(values_a, false);
    defer c.roaring_bitmap_free(oracle_a);
    const oracle_b = try buildOracle(values_b, false);
    defer c.roaring_bitmap_free(oracle_b);

    var rawr_result = try a.bitwiseDifference(allocator, &b);
    defer rawr_result.deinit();
    try assertKeys("transition:empty-out", &rawr_result, &.{ 0, 2 });

    const oracle_result = try oracleAllocatingOp(.bitwise_difference, oracle_a, oracle_b);
    defer c.roaring_bitmap_free(oracle_result);
    try assertAgree(allocator, "transition:empty-out", &rawr_result, oracle_result);
}

fn runBoundarySplitCase(allocator: Allocator) !void {
    const values_a = try valuesFromRanges(allocator, &.{.{ .start = 0, .end = 65_535 }});
    defer allocator.free(values_a);
    const values_b = try valuesFromRanges(allocator, &.{.{ .start = 12_345, .end = 12_345 }});
    defer allocator.free(values_b);

    var a = try buildRawrFromValues(allocator, values_a, true);
    defer a.deinit();
    var b = try buildRawrFromValues(allocator, values_b, true);
    defer b.deinit();
    const oracle_a = try buildOracle(values_a, true);
    defer c.roaring_bitmap_free(oracle_a);
    const oracle_b = try buildOracle(values_b, true);
    defer c.roaring_bitmap_free(oracle_b);

    var rawr_result = try a.bitwiseDifference(allocator, &b);
    defer rawr_result.deinit();

    const oracle_result = try oracleAllocatingOp(.bitwise_difference, oracle_a, oracle_b);
    defer c.roaring_bitmap_free(oracle_result);
    try assertAgree(allocator, "transition:run-boundary", &rawr_result, oracle_result);
}

fn runMatrixCase(allocator: Allocator, case: MatrixCase, run_optimize: bool) !void {
    var prng = std.Random.DefaultPrng.init(caseSeed(case, run_optimize));
    const rng = prng.random();
    const key: u16 = 77;

    var a = try buildOperand(allocator, rng, key, case.a, run_optimize);
    defer a.deinit();
    var b = try buildOperand(allocator, rng, key, case.b, run_optimize);
    defer b.deinit();

    const oracle_a = try buildOracle(a.values, run_optimize);
    defer c.roaring_bitmap_free(oracle_a);
    const oracle_b = try buildOracle(b.values, run_optimize);
    defer c.roaring_bitmap_free(oracle_b);

    try runOrderedChecks(allocator, case.name, "a_b", &a.bm, &b.bm, oracle_a, oracle_b);
    try runOrderedChecks(allocator, case.name, "b_a", &b.bm, &a.bm, oracle_b, oracle_a);
}

fn buildOperand(
    allocator: Allocator,
    rng: std.Random,
    key: u16,
    profile: ?Profile,
    run_optimize: bool,
) !test_gen.Generated {
    if (profile) |p| {
        const chunks = [_]test_gen.ChunkProfile{
            .{ .key = key, .profile = p },
        };
        return test_gen.build(allocator, rng, &chunks, run_optimize);
    }

    return test_gen.build(allocator, rng, &.{}, run_optimize);
}

fn valuesFromRanges(allocator: Allocator, ranges: []const Range) ![]u32 {
    var values = std.array_list.Managed(u32).init(allocator);
    defer values.deinit();

    for (ranges) |range| {
        var value = range.start;
        while (true) {
            try values.append(value);
            if (value == range.end) break;
            value += 1;
        }
    }

    return values.toOwnedSlice();
}

fn buildRawrFromValues(allocator: Allocator, values: []const u32, run_optimize: bool) !RoaringBitmap {
    var bm = try RoaringBitmap.init(allocator);
    errdefer bm.deinit();

    for (values) |value| {
        _ = try bm.add(value);
    }
    if (run_optimize) {
        _ = try bm.runOptimize();
    }

    return bm;
}

fn assertSingleContainerType(
    name: []const u8,
    bm: *const RoaringBitmap,
    expected: rawr.TaggedPtr.ContainerType,
) !void {
    if (bm.size != 1) {
        std.debug.print("FAIL: {s} - expected 1 container, got {d}\n", .{ name, bm.size });
        return error.ContainerShapeMismatch;
    }
    const actual = bm.containers[0].getType();
    if (actual != expected) {
        std.debug.print("FAIL: {s} - expected container type {s}, got {s}\n", .{
            name,
            @tagName(expected),
            @tagName(actual),
        });
        return error.ContainerShapeMismatch;
    }
}

fn assertKeys(name: []const u8, bm: *const RoaringBitmap, expected: []const u16) !void {
    if (bm.size != expected.len) {
        std.debug.print("FAIL: {s} - expected {d} containers, got {d}\n", .{ name, expected.len, bm.size });
        return error.ContainerShapeMismatch;
    }
    for (expected, 0..) |key, i| {
        if (bm.keys[i] != key) {
            std.debug.print("FAIL: {s} - key[{d}] expected {d}, got {d}\n", .{ name, i, key, bm.keys[i] });
            return error.ContainerShapeMismatch;
        }
    }
}

fn runOrderedChecks(
    allocator: Allocator,
    case_name: []const u8,
    order_name: []const u8,
    a: *RoaringBitmap,
    b: *RoaringBitmap,
    oracle_a: *c.roaring_bitmap_t,
    oracle_b: *c.roaring_bitmap_t,
) !void {
    if (PRINT_RANDOM_PROGRESS and std.mem.startsWith(u8, case_name, "random:")) {
        std.debug.print("{s}:{s}:predicates\n", .{ case_name, order_name });
    }
    try assertPredicatesAgree(case_name, order_name, a, b, oracle_a, oracle_b);

    const ops = [_]BinaryOp{ .bitwise_or, .bitwise_and, .bitwise_xor, .bitwise_difference };
    for (ops) |op| {
        if (PRINT_RANDOM_PROGRESS and std.mem.startsWith(u8, case_name, "random:")) {
            std.debug.print("{s}:{s}:{s}:alloc\n", .{ case_name, order_name, op.name() });
        }
        try assertAllocatingOpAgree(allocator, case_name, order_name, op, a, b, oracle_a, oracle_b);
        if (PRINT_RANDOM_PROGRESS and std.mem.startsWith(u8, case_name, "random:")) {
            std.debug.print("{s}:{s}:{s}:inplace\n", .{ case_name, order_name, op.name() });
        }
        try assertInPlaceOpAgree(allocator, case_name, order_name, op, a, b, oracle_a, oracle_b);
    }
}

fn assertAllocatingOpAgree(
    allocator: Allocator,
    case_name: []const u8,
    order_name: []const u8,
    op: BinaryOp,
    a: *RoaringBitmap,
    b: *RoaringBitmap,
    oracle_a: *c.roaring_bitmap_t,
    oracle_b: *c.roaring_bitmap_t,
) !void {
    if (PRINT_RANDOM_PROGRESS and std.mem.startsWith(u8, case_name, "random:")) {
        std.debug.print("{s}:{s}:{s}:alloc:rawr\n", .{ case_name, order_name, op.name() });
    }
    var rawr_result = try rawrAllocatingOp(allocator, op, a, b);
    defer rawr_result.deinit();

    if (PRINT_RANDOM_PROGRESS and std.mem.startsWith(u8, case_name, "random:")) {
        std.debug.print("{s}:{s}:{s}:alloc:oracle\n", .{ case_name, order_name, op.name() });
    }
    const oracle_result = try oracleAllocatingOp(op, oracle_a, oracle_b);
    defer c.roaring_bitmap_free(oracle_result);

    var name_buf: [128]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "{s}:{s}:{s}:alloc", .{ case_name, order_name, op.name() });
    if (PRINT_RANDOM_PROGRESS and std.mem.startsWith(u8, case_name, "random:")) {
        std.debug.print("{s}:{s}:{s}:alloc:assert\n", .{ case_name, order_name, op.name() });
    }
    try assertAgree(allocator, name, &rawr_result, oracle_result);
}

fn assertInPlaceOpAgree(
    allocator: Allocator,
    case_name: []const u8,
    order_name: []const u8,
    op: BinaryOp,
    a: *RoaringBitmap,
    b: *RoaringBitmap,
    oracle_a: *c.roaring_bitmap_t,
    oracle_b: *c.roaring_bitmap_t,
) !void {
    var rawr_allocating = try rawrAllocatingOp(allocator, op, a, b);
    defer rawr_allocating.deinit();

    var rawr_in_place = try a.clone(allocator);
    defer rawr_in_place.deinit();
    try rawrInPlaceOp(op, &rawr_in_place, b);

    if (!rawr_in_place.equals(&rawr_allocating)) {
        std.debug.print("FAIL: {s}:{s}:{s}:inplace - rawr in-place result differs from allocating result\n", .{
            case_name,
            order_name,
            op.name(),
        });
        return error.InPlaceMismatch;
    }

    const oracle_in_place = c.roaring_bitmap_copy(oracle_a) orelse return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(oracle_in_place);
    oracleInPlaceOp(op, oracle_in_place, oracle_b);

    var name_buf: [128]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "{s}:{s}:{s}:inplace", .{ case_name, order_name, op.name() });
    try assertAgree(allocator, name, &rawr_in_place, oracle_in_place);
}

fn assertFlipAgree(
    allocator: Allocator,
    name: []const u8,
    bm: *RoaringBitmap,
    oracle: *c.roaring_bitmap_t,
    lo: u32,
    hi: u32,
) !void {
    var rawr_result = try bm.flip(allocator, lo, hi);
    defer rawr_result.deinit();

    const oracle_result = c.roaring_bitmap_flip_closed(oracle, lo, hi) orelse return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(oracle_result);

    var alloc_name_buf: [128]u8 = undefined;
    const alloc_name = try std.fmt.bufPrint(&alloc_name_buf, "{s}:alloc", .{name});
    try assertSameValues(allocator, alloc_name, &rawr_result, oracle_result);

    var rawr_in_place = try bm.clone(allocator);
    defer rawr_in_place.deinit();
    try rawr_in_place.flipInplace(lo, hi);

    if (!rawr_in_place.equals(&rawr_result)) {
        std.debug.print("FAIL: {s}:inplace - rawr in-place result differs from allocating result\n", .{name});
        return error.InPlaceMismatch;
    }

    const oracle_in_place = c.roaring_bitmap_copy(oracle) orelse return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(oracle_in_place);
    c.roaring_bitmap_flip_inplace_closed(oracle_in_place, lo, hi);

    var inplace_name_buf: [128]u8 = undefined;
    const inplace_name = try std.fmt.bufPrint(&inplace_name_buf, "{s}:inplace", .{name});
    try assertSameValues(allocator, inplace_name, &rawr_in_place, oracle_in_place);
}

fn assertRangeOpsAgree(
    allocator: Allocator,
    name: []const u8,
    bm: *RoaringBitmap,
    oracle: *c.roaring_bitmap_t,
    lo: u32,
    hi: u32,
) !void {
    const rawr_cardinality = bm.rangeCardinality(lo, hi);
    const oracle_cardinality = c.roaring_bitmap_range_cardinality_closed(oracle, lo, hi);
    try expectEqualScalar(name, "range", "rangeCardinality", rawr_cardinality, oracle_cardinality);

    const rawr_contains = bm.containsRange(lo, hi);
    const oracle_contains = c.roaring_bitmap_contains_range_closed(oracle, lo, hi);
    try expectEqualBool(name, "range", "containsRange", rawr_contains, oracle_contains);

    const rawr_intersects = bm.intersectsRange(lo, hi);
    const oracle_intersects = c.roaring_bitmap_intersect_with_range(oracle, @as(u64, lo), @as(u64, hi) + 1);
    try expectEqualBool(name, "range", "intersectsRange", rawr_intersects, oracle_intersects);

    if (lo <= hi) {
        const range_size = @as(u64, hi) - lo + 1;
        if (rawr_contains != (rawr_cardinality == range_size)) {
            std.debug.print("FAIL: {s}:containsRange rawr cross-check failed: contains={} card={d} range={d}\n", .{
                name,
                rawr_contains,
                rawr_cardinality,
                range_size,
            });
            return error.PredicateMismatch;
        }
    } else if (!rawr_contains or rawr_cardinality != 0) {
        std.debug.print("FAIL: {s}:empty range cross-check failed: contains={} card={d}\n", .{
            name,
            rawr_contains,
            rawr_cardinality,
        });
        return error.PredicateMismatch;
    }

    if (rawr_intersects != (rawr_cardinality > 0)) {
        std.debug.print("FAIL: {s}:intersectsRange rawr cross-check failed: intersects={} card={d}\n", .{
            name,
            rawr_intersects,
            rawr_cardinality,
        });
        return error.PredicateMismatch;
    }

    var rawr_removed = try bm.clone(allocator);
    defer rawr_removed.deinit();
    const before = rawr_removed.cardinality();
    const removed = try rawr_removed.removeRange(lo, hi);
    const after = rawr_removed.cardinality();
    if (removed != before - after) {
        std.debug.print("FAIL: {s}:removeRange count differs: removed={d} before-after={d}\n", .{
            name,
            removed,
            before - after,
        });
        return error.PredicateMismatch;
    }

    const oracle_removed = c.roaring_bitmap_copy(oracle) orelse return error.CRoaringAllocFailed;
    defer c.roaring_bitmap_free(oracle_removed);
    c.roaring_bitmap_remove_range_closed(oracle_removed, lo, hi);

    var remove_name_buf: [128]u8 = undefined;
    const remove_name = try std.fmt.bufPrint(&remove_name_buf, "{s}:removeRange", .{name});
    try assertSameValues(allocator, remove_name, &rawr_removed, oracle_removed);
}

fn rawrAllocatingOp(allocator: Allocator, op: BinaryOp, a: *RoaringBitmap, b: *RoaringBitmap) !RoaringBitmap {
    return switch (op) {
        .bitwise_or => a.bitwiseOr(allocator, b),
        .bitwise_and => a.bitwiseAnd(allocator, b),
        .bitwise_xor => a.bitwiseXor(allocator, b),
        .bitwise_difference => a.bitwiseDifference(allocator, b),
    };
}

fn rawrInPlaceOp(op: BinaryOp, a: *RoaringBitmap, b: *RoaringBitmap) !void {
    return switch (op) {
        .bitwise_or => a.bitwiseOrInPlace(b),
        .bitwise_and => a.bitwiseAndInPlace(b),
        .bitwise_xor => a.bitwiseXorInPlace(b),
        .bitwise_difference => a.bitwiseDifferenceInPlace(b),
    };
}

fn oracleAllocatingOp(op: BinaryOp, a: *c.roaring_bitmap_t, b: *c.roaring_bitmap_t) !*c.roaring_bitmap_t {
    return switch (op) {
        .bitwise_or => c.roaring_bitmap_or(a, b) orelse error.CRoaringAllocFailed,
        .bitwise_and => c.roaring_bitmap_and(a, b) orelse error.CRoaringAllocFailed,
        .bitwise_xor => c.roaring_bitmap_xor(a, b) orelse error.CRoaringAllocFailed,
        .bitwise_difference => c.roaring_bitmap_andnot(a, b) orelse error.CRoaringAllocFailed,
    };
}

fn oracleInPlaceOp(op: BinaryOp, a: *c.roaring_bitmap_t, b: *c.roaring_bitmap_t) void {
    switch (op) {
        .bitwise_or => c.roaring_bitmap_or_inplace(a, b),
        .bitwise_and => c.roaring_bitmap_and_inplace(a, b),
        .bitwise_xor => c.roaring_bitmap_xor_inplace(a, b),
        .bitwise_difference => c.roaring_bitmap_andnot_inplace(a, b),
    }
}

fn assertPredicatesAgree(
    case_name: []const u8,
    order_name: []const u8,
    a: *RoaringBitmap,
    b: *RoaringBitmap,
    oracle_a: *c.roaring_bitmap_t,
    oracle_b: *c.roaring_bitmap_t,
) !void {
    try expectEqualScalar(case_name, order_name, "andCardinality", a.andCardinality(b), c.roaring_bitmap_and_cardinality(oracle_a, oracle_b));
    try expectEqualScalar(case_name, order_name, "orCardinality", a.orCardinality(b), c.roaring_bitmap_or_cardinality(oracle_a, oracle_b));
    try expectEqualScalar(case_name, order_name, "xorCardinality", a.xorCardinality(b), c.roaring_bitmap_xor_cardinality(oracle_a, oracle_b));
    try expectEqualScalar(case_name, order_name, "differenceCardinality", a.differenceCardinality(b), c.roaring_bitmap_andnot_cardinality(oracle_a, oracle_b));
    try expectEqualFloat(case_name, order_name, "jaccardIndex", a.jaccardIndex(b), c.roaring_bitmap_jaccard_index(oracle_a, oracle_b));
    try expectEqualBool(case_name, order_name, "intersects", a.intersects(b), c.roaring_bitmap_intersect(oracle_a, oracle_b));
    try expectEqualBool(case_name, order_name, "isSubsetOf", a.isSubsetOf(b), c.roaring_bitmap_is_subset(oracle_a, oracle_b));
    try expectEqualBool(case_name, order_name, "isStrictSubsetOf", a.isStrictSubsetOf(b), c.roaring_bitmap_is_strict_subset(oracle_a, oracle_b));
    try expectEqualBool(case_name, order_name, "equals", a.equals(b), c.roaring_bitmap_equals(oracle_a, oracle_b));
    try expectEqualScalar(case_name, order_name, "cardinality(a)", a.cardinality(), c.roaring_bitmap_get_cardinality(oracle_a));
    try expectEqualScalar(case_name, order_name, "cardinality(b)", b.cardinality(), c.roaring_bitmap_get_cardinality(oracle_b));
    try assertPositionalsAgree(case_name, order_name, "a", a, oracle_a);
    try assertPositionalsAgree(case_name, order_name, "b", b, oracle_b);
    try assertMinMaxAgree(case_name, order_name, "a", a, oracle_a);
    try assertMinMaxAgree(case_name, order_name, "b", b, oracle_b);
}

fn assertPositionalsAgree(
    case_name: []const u8,
    order_name: []const u8,
    operand_name: []const u8,
    bm: *RoaringBitmap,
    oracle: *c.roaring_bitmap_t,
) !void {
    var probes: [32]u32 = undefined;
    var probe_count: usize = 0;

    addProbe(&probes, &probe_count, 0);
    addProbe(&probes, &probe_count, 65_535);
    addProbe(&probes, &probe_count, 65_536);
    addProbe(&probes, &probe_count, std.math.maxInt(u32));

    if (bm.minimum()) |min| {
        addProbe(&probes, &probe_count, min);
        if (min > 0) addProbe(&probes, &probe_count, min - 1);
    }
    if (bm.maximum()) |max| {
        addProbe(&probes, &probe_count, max);
        if (max < std.math.maxInt(u32)) addProbe(&probes, &probe_count, max + 1);
    }

    var it = bm.iterator();
    var present_seen: usize = 0;
    while (present_seen < 8) : (present_seen += 1) {
        const value = it.next() orelse break;
        addProbe(&probes, &probe_count, value);
        if (value < std.math.maxInt(u32)) addProbe(&probes, &probe_count, value + 1);
    }

    std.mem.sort(u32, probes[0..probe_count], {}, std.sort.asc(u32));
    probe_count = dedupeSortedProbes(probes[0..probe_count]);

    var rawr_ranks: [32]u64 = undefined;
    bm.rankMany(probes[0..probe_count], rawr_ranks[0..probe_count]);

    for (probes[0..probe_count], rawr_ranks[0..probe_count]) |probe, rank_many_value| {
        try expectEqualScalar(case_name, order_name, "rankMany", rank_many_value, c.roaring_bitmap_rank(oracle, probe));
        try expectEqualScalar(case_name, order_name, "rank", bm.rank(probe), c.roaring_bitmap_rank(oracle, probe));

        const rawr_index = bm.getIndex(probe);
        const oracle_index = c.roaring_bitmap_get_index(oracle, probe);
        if (oracle_index < 0) {
            if (rawr_index != null) {
                std.debug.print("FAIL: {s}:{s}:getIndex({s},{d}) differs: rawr={?d} croaring=-1\n", .{
                    case_name,
                    order_name,
                    operand_name,
                    probe,
                    rawr_index,
                });
                return error.PredicateMismatch;
            }
        } else if (rawr_index == null or rawr_index.? != @as(u64, @intCast(oracle_index))) {
            std.debug.print("FAIL: {s}:{s}:getIndex({s},{d}) differs: rawr={?d} croaring={d}\n", .{
                case_name,
                order_name,
                operand_name,
                probe,
                rawr_index,
                oracle_index,
            });
            return error.PredicateMismatch;
        }
    }

    const card = bm.cardinality();
    var ranks: [8]u64 = undefined;
    var rank_count: usize = 0;
    addRankProbe(&ranks, &rank_count, 0);
    if (card > 1) addRankProbe(&ranks, &rank_count, 1);
    if (card > 0) {
        addRankProbe(&ranks, &rank_count, card / 2);
        addRankProbe(&ranks, &rank_count, card - 1);
    }
    addRankProbe(&ranks, &rank_count, card);
    addRankProbe(&ranks, &rank_count, card + 1);
    addRankProbe(&ranks, &rank_count, std.math.maxInt(u64));

    for (ranks[0..rank_count]) |rank_probe| {
        const rawr_value = bm.select(rank_probe);
        if (rank_probe <= std.math.maxInt(u32)) {
            var oracle_value: u32 = undefined;
            const oracle_ok = c.roaring_bitmap_select(oracle, @intCast(rank_probe), &oracle_value);
            if (!oracle_ok) {
                if (rawr_value != null) {
                    std.debug.print("FAIL: {s}:{s}:select({s},{d}) differs: rawr={?d} croaring=null\n", .{
                        case_name,
                        order_name,
                        operand_name,
                        rank_probe,
                        rawr_value,
                    });
                    return error.PredicateMismatch;
                }
            } else if (rawr_value == null or rawr_value.? != oracle_value) {
                std.debug.print("FAIL: {s}:{s}:select({s},{d}) differs: rawr={?d} croaring={d}\n", .{
                    case_name,
                    order_name,
                    operand_name,
                    rank_probe,
                    rawr_value,
                    oracle_value,
                });
                return error.PredicateMismatch;
            }
        } else if (rawr_value != null) {
            std.debug.print("FAIL: {s}:{s}:select({s},{d}) expected null for oversized rank, got {?d}\n", .{
                case_name,
                order_name,
                operand_name,
                rank_probe,
                rawr_value,
            });
            return error.PredicateMismatch;
        }

        if (rawr_value) |value| {
            try expectEqualScalar(case_name, order_name, "rank(select(k))", bm.rank(value), rank_probe + 1);
            const selected_again = bm.select(bm.rank(value) - 1);
            if (selected_again == null or selected_again.? != value) {
                std.debug.print("FAIL: {s}:{s}:select(rank(v)-1)({s},{d}) got {?d}\n", .{
                    case_name,
                    order_name,
                    operand_name,
                    value,
                    selected_again,
                });
                return error.PredicateMismatch;
            }
        }
    }
}

fn addProbe(probes: *[32]u32, count: *usize, value: u32) void {
    if (count.* == probes.len) return;
    probes[count.*] = value;
    count.* += 1;
}

fn dedupeSortedProbes(probes: []u32) usize {
    if (probes.len == 0) return 0;
    var write_idx: usize = 1;
    for (probes[1..]) |probe| {
        if (probe != probes[write_idx - 1]) {
            probes[write_idx] = probe;
            write_idx += 1;
        }
    }
    return write_idx;
}

fn addRankProbe(probes: *[8]u64, count: *usize, value: u64) void {
    if (count.* == probes.len) return;
    for (probes[0..count.*]) |existing| {
        if (existing == value) return;
    }
    probes[count.*] = value;
    count.* += 1;
}

fn assertMinMaxAgree(
    case_name: []const u8,
    order_name: []const u8,
    operand_name: []const u8,
    bm: *const RoaringBitmap,
    oracle: *c.roaring_bitmap_t,
) !void {
    if (c.roaring_bitmap_is_empty(oracle)) {
        if (bm.minimum() != null or bm.maximum() != null) {
            std.debug.print("FAIL: {s}:{s}:minmax({s}) - rawr non-null min/max for empty bitmap\n", .{
                case_name,
                order_name,
                operand_name,
            });
            return error.PredicateMismatch;
        }
        return;
    }

    try expectEqualOptionalScalar(case_name, order_name, "minimum", bm.minimum(), c.roaring_bitmap_minimum(oracle));
    try expectEqualOptionalScalar(case_name, order_name, "maximum", bm.maximum(), c.roaring_bitmap_maximum(oracle));
}

fn expectEqualBool(
    case_name: []const u8,
    order_name: []const u8,
    predicate_name: []const u8,
    rawr_value: bool,
    oracle_value: bool,
) !void {
    if (rawr_value != oracle_value) {
        std.debug.print("FAIL: {s}:{s}:{s} differs: rawr={} croaring={}\n", .{
            case_name,
            order_name,
            predicate_name,
            rawr_value,
            oracle_value,
        });
        return error.PredicateMismatch;
    }
}

fn expectEqualFloat(
    case_name: []const u8,
    order_name: []const u8,
    predicate_name: []const u8,
    rawr_value: f64,
    oracle_value: f64,
) !void {
    if (std.math.isNan(rawr_value) and std.math.isNan(oracle_value)) return;
    if (rawr_value == oracle_value) return;

    const diff = @abs(rawr_value - oracle_value);
    if (diff <= 1e-12) return;

    std.debug.print("FAIL: {s}:{s}:{s} differs: rawr={d} croaring={d}\n", .{
        case_name,
        order_name,
        predicate_name,
        rawr_value,
        oracle_value,
    });
    return error.PredicateMismatch;
}

fn expectEqualScalar(
    case_name: []const u8,
    order_name: []const u8,
    predicate_name: []const u8,
    rawr_value: u64,
    oracle_value: u64,
) !void {
    if (rawr_value != oracle_value) {
        std.debug.print("FAIL: {s}:{s}:{s} differs: rawr={d} croaring={d}\n", .{
            case_name,
            order_name,
            predicate_name,
            rawr_value,
            oracle_value,
        });
        return error.PredicateMismatch;
    }
}

fn expectEqualOptionalScalar(
    case_name: []const u8,
    order_name: []const u8,
    predicate_name: []const u8,
    rawr_value: ?u32,
    oracle_value: u32,
) !void {
    if (rawr_value == null or rawr_value.? != oracle_value) {
        std.debug.print("FAIL: {s}:{s}:{s} differs: rawr={?d} croaring={d}\n", .{
            case_name,
            order_name,
            predicate_name,
            rawr_value,
            oracle_value,
        });
        return error.PredicateMismatch;
    }
}

fn caseSeed(case: MatrixCase, run_optimize: bool) u64 {
    var seed: u64 = if (run_optimize) 0x9E37_79B9 else 0x51ED_0000;
    for (case.name) |byte| {
        seed = (seed *% 131) +% byte;
    }
    return seed;
}

fn buildOracle(values: []const u32, run_optimize: bool) !*c.roaring_bitmap_t {
    const oracle = c.roaring_bitmap_create() orelse return error.CRoaringAllocFailed;
    errdefer c.roaring_bitmap_free(oracle);

    for (values) |value| {
        c.roaring_bitmap_add(oracle, value);
    }
    if (run_optimize) {
        _ = c.roaring_bitmap_run_optimize(oracle);
    }

    return oracle;
}

fn buildManyOracle(op: ManyOp, inputs: []const *c.roaring_bitmap_t) !*c.roaring_bitmap_t {
    std.debug.assert(inputs.len > 0);
    return switch (op) {
        .bor => c.roaring_bitmap_or_many(inputs.len, @ptrCast(@constCast(inputs.ptr))) orelse error.CRoaringAllocFailed,
        .xor => c.roaring_bitmap_xor_many(inputs.len, @ptrCast(@constCast(inputs.ptr))) orelse error.CRoaringAllocFailed,
    };
}

fn assertManyAgree(
    allocator: Allocator,
    op: ManyOp,
    rawr_inputs: []const *const RoaringBitmap,
    oracle_inputs: []const *c.roaring_bitmap_t,
    run_optimize: bool,
    n: usize,
) !void {
    var rawr_result = switch (op) {
        .bor => try RoaringBitmap.orMany(allocator, rawr_inputs),
        .xor => try RoaringBitmap.xorMany(allocator, rawr_inputs),
    };
    defer rawr_result.deinit();

    const oracle_result = try buildManyOracle(op, oracle_inputs);
    defer c.roaring_bitmap_free(oracle_result);

    var name_buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "{s}:n={d}:runopt={}", .{ op.name(), n, run_optimize });
    try assertSameValues(allocator, name, &rawr_result, oracle_result);
}

fn expectRawrEqual(name: []const u8, a: *const RoaringBitmap, b: *const RoaringBitmap) !void {
    if (!a.equals(b)) {
        std.debug.print("FAIL: {s} - rawr bitmaps differ\n", .{name});
        return error.RawrMismatch;
    }
}

fn expectRawrEmpty(name: []const u8, bm: *const RoaringBitmap) !void {
    if (!bm.isEmpty()) {
        std.debug.print("FAIL: {s} - expected empty bitmap\n", .{name});
        return error.RawrMismatch;
    }
}

fn assertAgree(
    allocator: Allocator,
    name: []const u8,
    rawr_bm: *RoaringBitmap,
    oracle: *c.roaring_bitmap_t,
) !void {
    const comparable_oracle = if (rawrHasRunContainers(rawr_bm)) blk: {
        const copy = c.roaring_bitmap_copy(oracle) orelse return error.CRoaringAllocFailed;
        _ = c.roaring_bitmap_run_optimize(copy);
        break :blk copy;
    } else oracle;
    defer if (comparable_oracle != oracle) c.roaring_bitmap_free(comparable_oracle);

    const rawr_cardinality = rawr_bm.cardinality();
    const oracle_cardinality = c.roaring_bitmap_get_cardinality(comparable_oracle);
    if (rawr_cardinality != oracle_cardinality) {
        std.debug.print("FAIL: {s} - cardinality differs: rawr={d} croaring={d}\n", .{
            name,
            rawr_cardinality,
            oracle_cardinality,
        });
        return error.CardinalityMismatch;
    }

    const rawr_bytes = try rawr_bm.serialize(allocator);
    defer allocator.free(rawr_bytes);

    const oracle_size = c.roaring_bitmap_portable_size_in_bytes(comparable_oracle);
    const oracle_bytes = try allocator.alloc(u8, oracle_size);
    defer allocator.free(oracle_bytes);
    _ = c.roaring_bitmap_portable_serialize(comparable_oracle, @ptrCast(oracle_bytes.ptr));

    if (!std.mem.eql(u8, rawr_bytes, oracle_bytes)) {
        printByteMismatch(name, rawr_bytes, oracle_bytes);
        return error.ByteMismatch;
    }

    var iter = rawr_bm.iterator();
    while (iter.next()) |value| {
        if (!c.roaring_bitmap_contains(comparable_oracle, value)) {
            std.debug.print("FAIL: {s} - CRoaring missing rawr value {d}\n", .{ name, value });
            return error.MissingValue;
        }
    }

    try assertAbsentSamplesAgree(name, rawr_bm, comparable_oracle);
    try assertCrossDeserializeAgree(allocator, name, rawr_bm, comparable_oracle, rawr_bytes, oracle_bytes);

    logPass("  PASS: {s} ({d} values, {d} bytes)\n", .{
        name,
        rawr_cardinality,
        rawr_bytes.len,
    });
}

fn assertSameValues(
    allocator: Allocator,
    name: []const u8,
    rawr_bm: *RoaringBitmap,
    oracle: *c.roaring_bitmap_t,
) !void {
    const rawr_cardinality = rawr_bm.cardinality();
    const oracle_cardinality = c.roaring_bitmap_get_cardinality(oracle);
    if (rawr_cardinality != oracle_cardinality) {
        std.debug.print("FAIL: {s} - cardinality differs: rawr={d} croaring={d}\n", .{
            name,
            rawr_cardinality,
            oracle_cardinality,
        });
        return error.CardinalityMismatch;
    }

    var iter = rawr_bm.iterator();
    while (iter.next()) |value| {
        if (!c.roaring_bitmap_contains(oracle, value)) {
            std.debug.print("FAIL: {s} - CRoaring missing rawr value {d}\n", .{ name, value });
            return error.MissingValue;
        }
    }

    try assertAbsentSamplesAgree(name, rawr_bm, oracle);

    const rawr_bytes = try rawr_bm.serialize(allocator);
    defer allocator.free(rawr_bytes);
    const oracle_size = c.roaring_bitmap_portable_size_in_bytes(oracle);
    const oracle_bytes = try allocator.alloc(u8, oracle_size);
    defer allocator.free(oracle_bytes);
    _ = c.roaring_bitmap_portable_serialize(oracle, @ptrCast(oracle_bytes.ptr));

    try assertCrossDeserializeAgree(allocator, name, rawr_bm, oracle, rawr_bytes, oracle_bytes);
}

fn rawrHasRunContainers(rawr_bm: *const RoaringBitmap) bool {
    for (rawr_bm.containers[0..rawr_bm.size]) |container| {
        if (container.getType() == rawr.TaggedPtr.ContainerType.run) return true;
    }
    return false;
}

fn assertAbsentSamplesAgree(name: []const u8, rawr_bm: *const RoaringBitmap, oracle: *c.roaring_bitmap_t) !void {
    const candidates = [_]u32{
        0,
        1,
        2,
        65_535,
        65_536,
        65_537,
        0x0002_FFFF,
        0x7FFF_FFFF,
        0xDEAD_BEEF,
        0xFFFF_FFFF,
    };

    for (candidates) |value| {
        const rawr_contains = rawr_bm.contains(value);
        const oracle_contains = c.roaring_bitmap_contains(oracle, value);
        if (rawr_contains != oracle_contains) {
            std.debug.print("FAIL: {s} - contains({d}) differs: rawr={} croaring={}\n", .{
                name,
                value,
                rawr_contains,
                oracle_contains,
            });
            return error.ContentMismatch;
        }
    }
}

fn assertCrossDeserializeAgree(
    allocator: Allocator,
    name: []const u8,
    rawr_bm: *RoaringBitmap,
    oracle: *c.roaring_bitmap_t,
    rawr_bytes: []const u8,
    oracle_bytes: []const u8,
) !void {
    const cr_from_rawr = c.roaring_bitmap_portable_deserialize_safe(@ptrCast(rawr_bytes.ptr), rawr_bytes.len) orelse {
        std.debug.print("FAIL: {s} - CRoaring failed to deserialize rawr bytes\n", .{name});
        return error.CRoaringDeserializeFailed;
    };
    defer c.roaring_bitmap_free(cr_from_rawr);

    if (c.roaring_bitmap_get_cardinality(cr_from_rawr) != rawr_bm.cardinality()) {
        std.debug.print("FAIL: {s} - CRoaring cardinality changed after rawr-byte deserialize\n", .{name});
        return error.CardinalityMismatch;
    }
    if (!c.roaring_bitmap_equals(cr_from_rawr, oracle)) {
        std.debug.print("FAIL: {s} - CRoaring content changed after rawr-byte deserialize\n", .{name});
        return error.ContentMismatch;
    }

    var rawr_from_cr = RoaringBitmap.deserialize(allocator, oracle_bytes) catch |err| {
        std.debug.print("FAIL: {s} - rawr failed to deserialize CRoaring bytes: {s}\n", .{ name, @errorName(err) });
        return error.RawrDeserializeFailed;
    };
    defer rawr_from_cr.deinit();

    if (rawr_from_cr.cardinality() != c.roaring_bitmap_get_cardinality(oracle)) {
        std.debug.print("FAIL: {s} - rawr cardinality changed after CRoaring-byte deserialize\n", .{name});
        return error.CardinalityMismatch;
    }
    if (!rawr_from_cr.equals(rawr_bm)) {
        std.debug.print("FAIL: {s} - rawr content changed after CRoaring-byte deserialize\n", .{name});
        return error.ContentMismatch;
    }
}

fn printByteMismatch(name: []const u8, rawr_bytes: []const u8, oracle_bytes: []const u8) void {
    std.debug.print("FAIL: {s} - bytes differ! rawr={d} bytes, croaring={d} bytes\n", .{
        name,
        rawr_bytes.len,
        oracle_bytes.len,
    });

    const min_len = @min(rawr_bytes.len, oracle_bytes.len);
    for (0..min_len) |i| {
        if (rawr_bytes[i] != oracle_bytes[i]) {
            std.debug.print("  First difference at byte {d}: rawr=0x{x:0>2} cr=0x{x:0>2}\n", .{
                i,
                rawr_bytes[i],
                oracle_bytes[i],
            });
            return;
        }
    }
}
