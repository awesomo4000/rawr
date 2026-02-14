const std = @import("std");
const RoaringBitmap = @import("rawr").RoaringBitmap;

const WARMUP_RUNS = 3;
const BENCH_RUNS = 21;
const N_VALUES = 1_000_000;

// --- Allocator registry ---

const AllocChoice = enum { c, smp, arena, fba };

fn nameToChoice(name: []const u8) ?AllocChoice {
    if (std.mem.eql(u8, name, "c")) return .c;
    if (std.mem.eql(u8, name, "smp")) return .smp;
    if (std.mem.eql(u8, name, "arena")) return .arena;
    if (std.mem.eql(u8, name, "fba")) return .fba;
    return null;
}

fn choiceToName(choice: AllocChoice) []const u8 {
    return switch (choice) {
        .c => "c",
        .smp => "smp",
        .arena => "arena",
        .fba => "fba",
    };
}

// --- Managed allocator context ---

const AllocContext = struct {
    allocator: std.mem.Allocator,
    arena: ?*std.heap.ArenaAllocator = null,
    fba: ?*std.heap.FixedBufferAllocator = null,
    fba_buf: ?[]u8 = null,

    fn init(choice: AllocChoice) AllocContext {
        switch (choice) {
            .c => return .{ .allocator = std.heap.c_allocator },
            .smp => return .{ .allocator = std.heap.smp_allocator },
            .arena => {
                const arena = std.heap.page_allocator.create(std.heap.ArenaAllocator) catch @panic("OOM");
                arena.* = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                return .{
                    .allocator = arena.allocator(),
                    .arena = arena,
                };
            },
            .fba => {
                // 256MB buffer for sparse bitmaps
                const buf = std.heap.page_allocator.alloc(u8, 256 * 1024 * 1024) catch @panic("OOM");
                const fba = std.heap.page_allocator.create(std.heap.FixedBufferAllocator) catch @panic("OOM");
                fba.* = std.heap.FixedBufferAllocator.init(buf);
                return .{
                    .allocator = fba.allocator(),
                    .fba = fba,
                    .fba_buf = buf,
                };
            },
        }
    }

    fn deinit(self: *AllocContext) void {
        if (self.arena) |a| {
            a.deinit();
            std.heap.page_allocator.destroy(a);
        }
        if (self.fba) |f| {
            std.heap.page_allocator.destroy(f);
        }
        if (self.fba_buf) |buf| {
            std.heap.page_allocator.free(buf);
        }
    }

    fn reset(self: *AllocContext) void {
        if (self.arena) |a| _ = a.reset(.retain_capacity);
        if (self.fba) |f| f.reset();
    }
};

// --- Test data ---

var random_values: [N_VALUES]u32 = undefined;
var serialized_data: ?[]u8 = null;

fn initTestData() void {
    var rng = std.Random.DefaultPrng.init(12345);
    const random = rng.random();
    for (&random_values) |*v| {
        v.* = random.int(u32);
    }

    // Create serialized data for deserialize benchmark
    var bm = RoaringBitmap.init(std.heap.c_allocator) catch @panic("OOM");
    defer bm.deinit();
    for (random_values[0 .. N_VALUES / 2]) |v| {
        _ = bm.add(v) catch @panic("OOM");
    }
    serialized_data = bm.serialize(std.heap.c_allocator) catch @panic("OOM");
}

fn buildSparseBitmaps(alloc: std.mem.Allocator) struct { a: RoaringBitmap, b: RoaringBitmap } {
    var a = RoaringBitmap.init(alloc) catch @panic("OOM");
    var b = RoaringBitmap.init(alloc) catch @panic("OOM");
    for (random_values, 0..) |v, i| {
        if (i % 2 == 0) {
            _ = a.add(v) catch @panic("OOM");
        } else {
            _ = b.add(v) catch @panic("OOM");
        }
    }
    return .{ .a = a, .b = b };
}

// --- Benchmark timing ---

const BenchResult = struct {
    median_ns: u64,
};

fn benchmark(comptime func: anytype, args: anytype) BenchResult {
    var times: [BENCH_RUNS]u64 = undefined;

    // Warmup
    for (0..WARMUP_RUNS) |_| {
        @call(.auto, func, args);
    }

    // Timed runs
    for (&times) |*t| {
        const start = std.time.nanoTimestamp();
        @call(.auto, func, args);
        const end = std.time.nanoTimestamp();
        t.* = @intCast(end - start);
    }

    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return .{ .median_ns = times[BENCH_RUNS / 2] };
}

// --- Benchmark functions ---

// Pre-allocated FBA buffer for output (64MB)
var fba_out_buf: ?[]u8 = null;

fn ensureFbaOutBuf() void {
    if (fba_out_buf == null) {
        fba_out_buf = std.heap.page_allocator.alloc(u8, 64 * 1024 * 1024) catch @panic("OOM");
    }
}

fn benchBitwiseAnd(input_a: *const RoaringBitmap, input_b: *const RoaringBitmap, out_choice: AllocChoice) void {
    switch (out_choice) {
        .c => {
            var result = input_a.bitwiseAnd(std.heap.c_allocator, input_b) catch unreachable;
            defer result.deinit();
            std.mem.doNotOptimizeAway(&result);
        },
        .smp => {
            var result = input_a.bitwiseAnd(std.heap.smp_allocator, input_b) catch unreachable;
            defer result.deinit();
            std.mem.doNotOptimizeAway(&result);
        },
        .arena => {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            var result = input_a.bitwiseAnd(arena.allocator(), input_b) catch unreachable;
            std.mem.doNotOptimizeAway(&result);
        },
        .fba => {
            ensureFbaOutBuf();
            var fba = std.heap.FixedBufferAllocator.init(fba_out_buf.?);
            var result = input_a.bitwiseAnd(fba.allocator(), input_b) catch unreachable;
            std.mem.doNotOptimizeAway(&result);
        },
    }
}

fn benchBitwiseOr(input_a: *const RoaringBitmap, input_b: *const RoaringBitmap, out_choice: AllocChoice) void {
    switch (out_choice) {
        .c => {
            var result = input_a.bitwiseOr(std.heap.c_allocator, input_b) catch unreachable;
            defer result.deinit();
            std.mem.doNotOptimizeAway(&result);
        },
        .smp => {
            var result = input_a.bitwiseOr(std.heap.smp_allocator, input_b) catch unreachable;
            defer result.deinit();
            std.mem.doNotOptimizeAway(&result);
        },
        .arena => {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            var result = input_a.bitwiseOr(arena.allocator(), input_b) catch unreachable;
            std.mem.doNotOptimizeAway(&result);
        },
        .fba => {
            ensureFbaOutBuf();
            var fba = std.heap.FixedBufferAllocator.init(fba_out_buf.?);
            var result = input_a.bitwiseOr(fba.allocator(), input_b) catch unreachable;
            std.mem.doNotOptimizeAway(&result);
        },
    }
}

fn benchDeserialize(out_choice: AllocChoice) void {
    const data = serialized_data.?;
    switch (out_choice) {
        .c => {
            var result = RoaringBitmap.deserialize(std.heap.c_allocator, data) catch unreachable;
            defer result.deinit();
            std.mem.doNotOptimizeAway(&result);
        },
        .smp => {
            var result = RoaringBitmap.deserialize(std.heap.smp_allocator, data) catch unreachable;
            defer result.deinit();
            std.mem.doNotOptimizeAway(&result);
        },
        .arena => {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            var result = RoaringBitmap.deserialize(arena.allocator(), data) catch unreachable;
            std.mem.doNotOptimizeAway(&result);
        },
        .fba => {
            ensureFbaOutBuf();
            var fba = std.heap.FixedBufferAllocator.init(fba_out_buf.?);
            var result = RoaringBitmap.deserialize(fba.allocator(), data) catch unreachable;
            std.mem.doNotOptimizeAway(&result);
        },
    }
}

// --- Output formatting ---

fn printHeader() void {
    std.debug.print("Allocator Matrix Benchmark\n", .{});
    std.debug.print("==========================\n", .{});
    std.debug.print("{d} warmup, {d} timed runs (median)\n\n", .{ WARMUP_RUNS, BENCH_RUNS });
}

fn printSingleResult(op_name: []const u8, input_name: []const u8, output_name: []const u8, ms: f64) void {
    std.debug.print("{s:<30} input={s:<6} output={s:<6} {d:>8.2} ms\n", .{ op_name, input_name, output_name, ms });
}

fn printMatrixHeader(op_name: []const u8) void {
    std.debug.print("\n{s}\n", .{op_name});
    std.debug.print("{s:>14}", .{""});
    inline for (std.meta.fields(AllocChoice)) |f| {
        std.debug.print(" {s:>8}", .{f.name});
    }
    std.debug.print("\n", .{});
}

fn printMatrixRow(input_name: []const u8, values: [4]f64) void {
    std.debug.print("INPUT: {s:<6}", .{input_name});
    for (values) |v| {
        if (v < 0) {
            std.debug.print(" {s:>8}", .{"N/A"});
        } else {
            std.debug.print(" {d:>8.2}", .{v});
        }
    }
    std.debug.print("\n", .{});
}

// --- Main ---

pub fn main() !void {
    var args = std.process.args();
    _ = args.skip(); // program name

    var input_choice: AllocChoice = .smp;
    var output_choice: AllocChoice = .smp;
    var run_matrix = false;
    var run_and = true;
    var run_or = true;
    var run_deser = true;

    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--input=")) {
            const name = arg[8..];
            input_choice = nameToChoice(name) orelse {
                std.debug.print("Unknown allocator: {s}\n", .{name});
                return;
            };
        } else if (std.mem.startsWith(u8, arg, "--output=")) {
            const name = arg[9..];
            output_choice = nameToChoice(name) orelse {
                std.debug.print("Unknown allocator: {s}\n", .{name});
                return;
            };
        } else if (std.mem.eql(u8, arg, "--matrix")) {
            run_matrix = true;
        } else if (std.mem.startsWith(u8, arg, "--ops=")) {
            const ops = arg[6..];
            run_and = false;
            run_or = false;
            run_deser = false;
            var it = std.mem.splitScalar(u8, ops, ',');
            while (it.next()) |op| {
                if (std.mem.eql(u8, op, "and")) run_and = true;
                if (std.mem.eql(u8, op, "or")) run_or = true;
                if (std.mem.eql(u8, op, "deser")) run_deser = true;
                if (std.mem.eql(u8, op, "all")) {
                    run_and = true;
                    run_or = true;
                    run_deser = true;
                }
            }
        }
    }

    std.debug.print("Initializing test data...\n", .{});
    initTestData();

    if (run_matrix) {
        runMatrix(run_and, run_or, run_deser);
    } else {
        runSingle(input_choice, output_choice, run_and, run_or, run_deser);
    }
}

fn runSingle(input_choice: AllocChoice, output_choice: AllocChoice, run_and: bool, run_or: bool, run_deser: bool) void {
    printHeader();
    std.debug.print("input={s}, output={s}\n\n", .{ choiceToName(input_choice), choiceToName(output_choice) });

    // Build input bitmaps
    var input_ctx = AllocContext.init(input_choice);
    defer input_ctx.deinit();
    const bitmaps = buildSparseBitmaps(input_ctx.allocator);
    var a = bitmaps.a;
    var b = bitmaps.b;
    defer a.deinit();
    defer b.deinit();

    if (run_and) {
        const r = benchmark(benchBitwiseAnd, .{ &a, &b, output_choice });
        const ms = @as(f64, @floatFromInt(r.median_ns)) / 1_000_000.0;
        printSingleResult("bitwiseAnd (sparse)", choiceToName(input_choice), choiceToName(output_choice), ms);
    }

    if (run_or) {
        const r = benchmark(benchBitwiseOr, .{ &a, &b, output_choice });
        const ms = @as(f64, @floatFromInt(r.median_ns)) / 1_000_000.0;
        printSingleResult("bitwiseOr (sparse)", choiceToName(input_choice), choiceToName(output_choice), ms);
    }

    if (run_deser) {
        const r = benchmark(benchDeserialize, .{output_choice});
        const ms = @as(f64, @floatFromInt(r.median_ns)) / 1_000_000.0;
        printSingleResult("deserialize", "N/A", choiceToName(output_choice), ms);
    }
}

fn runMatrix(run_and: bool, run_or: bool, run_deser: bool) void {
    printHeader();

    const choices = [_]AllocChoice{ .c, .smp, .arena, .fba };

    if (run_and) {
        printMatrixHeader("bitwiseAnd sparse (ms)");
        for (choices) |input_choice| {
            var input_ctx = AllocContext.init(input_choice);
            defer input_ctx.deinit();
            const bitmaps = buildSparseBitmaps(input_ctx.allocator);
            var a = bitmaps.a;
            var b = bitmaps.b;
            defer a.deinit();
            defer b.deinit();

            var row: [4]f64 = undefined;
            for (choices, 0..) |output_choice, i| {
                const r = benchmark(benchBitwiseAnd, .{ &a, &b, output_choice });
                row[i] = @as(f64, @floatFromInt(r.median_ns)) / 1_000_000.0;
            }
            printMatrixRow(choiceToName(input_choice), row);
        }
    }

    if (run_or) {
        printMatrixHeader("bitwiseOr sparse (ms)");
        for (choices) |input_choice| {
            var input_ctx = AllocContext.init(input_choice);
            defer input_ctx.deinit();
            const bitmaps = buildSparseBitmaps(input_ctx.allocator);
            var a = bitmaps.a;
            var b = bitmaps.b;
            defer a.deinit();
            defer b.deinit();

            var row: [4]f64 = undefined;
            for (choices, 0..) |output_choice, i| {
                const r = benchmark(benchBitwiseOr, .{ &a, &b, output_choice });
                row[i] = @as(f64, @floatFromInt(r.median_ns)) / 1_000_000.0;
            }
            printMatrixRow(choiceToName(input_choice), row);
        }
    }

    if (run_deser) {
        printMatrixHeader("deserialize (ms)");
        var row: [4]f64 = undefined;
        for (choices, 0..) |output_choice, i| {
            const r = benchmark(benchDeserialize, .{output_choice});
            row[i] = @as(f64, @floatFromInt(r.median_ns)) / 1_000_000.0;
        }
        printMatrixRow("(N/A)", row);
    }

    std.debug.print("\nDone.\n", .{});
}
