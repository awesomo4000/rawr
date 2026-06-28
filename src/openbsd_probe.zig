const std = @import("std");
const builtin = @import("builtin");
const rawr = @import("rawr");
const c = @import("c");
const bench_time = @import("bench_time.zig");

const RoaringBitmap = rawr.RoaringBitmap;

const MAX_VALUES = 1_000_000;
const DEFAULT_WARMUP_RUNS = 3;
const DEFAULT_BENCH_RUNS = 21;
const MAX_BENCH_RUNS = 64;

const AllocatorChoice = enum {
    openbsd_c,
    std_c,
    smp,
};

const CallMode = enum {
    auto,
    never_inline,
};

const HarnessMode = enum {
    direct,
    noinline_harness,
};

const Case = enum(u8) {
    @"00_allocator_smoke" = 0,
    @"01_rawr_init_loop" = 1,
    @"02_rawr_add_random_once" = 2,
    @"03_rawr_add_sequential_once" = 3,
    @"04_rawr_add_random_bench" = 4,
    @"05_rawr_add_sequential_bench" = 5,
    @"06_rawr_contains_hit_bench" = 6,
    @"07_rawr_contains_miss_bench" = 7,
    @"08_croaring_add_random_bench" = 8,
    @"99_all" = 99,
};

const Config = struct {
    case: Case = .@"04_rawr_add_random_bench",
    allocator_choice: AllocatorChoice = .openbsd_c,
    call_mode: CallMode = .auto,
    harness_mode: HarnessMode = .direct,
    values_len: usize = MAX_VALUES,
    warmup_runs: usize = DEFAULT_WARMUP_RUNS,
    bench_runs: usize = DEFAULT_BENCH_RUNS,
    iterations: usize = 10_000,
    trace: bool = true,
};

const BenchResult = struct {
    median_ns: u64,
    p25_ns: u64,
    p75_ns: u64,
};

var random_values: [MAX_VALUES]u32 = undefined;
var sequential_values: [MAX_VALUES]u32 = undefined;
var active_allocator: std.mem.Allocator = undefined;
var active_values_len: usize = MAX_VALUES;
var rawr_contains_bm: ?RoaringBitmap = null;

pub fn main(init: std.process.Init) !void {
    var cfg = Config{};
    if (!try parseArgs(init, &cfg)) return;

    if (cfg.values_len == 0 or cfg.values_len > MAX_VALUES) {
        bench_time.print("invalid --values; expected 1..{d}\n", .{MAX_VALUES});
        return;
    }
    if (cfg.bench_runs == 0 or cfg.bench_runs > MAX_BENCH_RUNS) {
        bench_time.print("invalid --runs; expected 1..{d}\n", .{MAX_BENCH_RUNS});
        return;
    }

    active_allocator = allocatorFor(cfg.allocator_choice);
    active_values_len = cfg.values_len;

    bench_time.print("OpenBSD probe\n", .{});
    bench_time.print("target={s} allocator={s} case={s} call={s} harness={s}\n", .{
        @tagName(builtin.os.tag),
        @tagName(cfg.allocator_choice),
        @tagName(cfg.case),
        @tagName(cfg.call_mode),
        @tagName(cfg.harness_mode),
    });
    bench_time.print("values={d} warmup={d} runs={d} iterations={d}\n", .{
        cfg.values_len,
        cfg.warmup_runs,
        cfg.bench_runs,
        cfg.iterations,
    });

    trace(&cfg, "init test data");
    initTestData(cfg.values_len);

    runCase(&cfg, cfg.case);

    cleanup();
    trace(&cfg, "done");
}

fn runCase(cfg: *const Config, selected: Case) void {
    printCaseHeader(selected);

    switch (selected) {
        .@"00_allocator_smoke" => allocatorSmoke(cfg),
        .@"01_rawr_init_loop" => rawrInitLoop(cfg),
        .@"02_rawr_add_random_once" => runOnce(cfg, "rawr add random once", benchRawrAddRandom),
        .@"03_rawr_add_sequential_once" => runOnce(cfg, "rawr add sequential once", benchRawrAddSequential),
        .@"04_rawr_add_random_bench" => runBenchmark(cfg, "rawr add random", benchRawrAddRandom),
        .@"05_rawr_add_sequential_bench" => runBenchmark(cfg, "rawr add sequential", benchRawrAddSequential),
        .@"06_rawr_contains_hit_bench" => {
            initRawrContainsBm(cfg);
            runBenchmark(cfg, "rawr contains hit", benchRawrContainsHit);
        },
        .@"07_rawr_contains_miss_bench" => {
            initRawrContainsBm(cfg);
            runBenchmark(cfg, "rawr contains miss", benchRawrContainsMiss);
        },
        .@"08_croaring_add_random_bench" => runBenchmark(cfg, "CRoaring add random", benchCRoaringAddRandom),
        .@"99_all" => runAllCases(cfg),
    }
}

fn runAllCases(cfg: *const Config) void {
    runCase(cfg, .@"00_allocator_smoke");
    runCase(cfg, .@"01_rawr_init_loop");
    runCase(cfg, .@"02_rawr_add_random_once");
    runCase(cfg, .@"03_rawr_add_sequential_once");
    runCase(cfg, .@"04_rawr_add_random_bench");
    runCase(cfg, .@"05_rawr_add_sequential_bench");
    runCase(cfg, .@"06_rawr_contains_hit_bench");
    runCase(cfg, .@"07_rawr_contains_miss_bench");
    runCase(cfg, .@"08_croaring_add_random_bench");
}

fn printCaseHeader(selected: Case) void {
    bench_time.print("\nCASE {d:0>2} {s}\n", .{ @intFromEnum(selected), caseDisplayName(selected) });
}

fn caseDisplayName(selected: Case) []const u8 {
    const name = @tagName(selected);
    if (name.len > 3 and name[2] == '_') return name[3..];
    return name;
}

fn parseArgs(init: std.process.Init, cfg: *Config) !bool {
    var args = try init.minimal.args.iterateAllocator(std.heap.smp_allocator);
    defer args.deinit();
    _ = args.skip();

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return false;
        } else if (std.mem.startsWith(u8, arg, "--case=")) {
            cfg.case = parseCase(arg[7..]) orelse {
                bench_time.print("unknown --case={s}\n", .{arg[7..]});
                printUsage();
                return false;
            };
        } else if (std.mem.startsWith(u8, arg, "--allocator=")) {
            cfg.allocator_choice = parseEnum(AllocatorChoice, arg[12..]) orelse {
                bench_time.print("unknown --allocator={s}\n", .{arg[12..]});
                printUsage();
                return false;
            };
        } else if (std.mem.startsWith(u8, arg, "--call=")) {
            cfg.call_mode = parseEnum(CallMode, arg[7..]) orelse {
                bench_time.print("unknown --call={s}\n", .{arg[7..]});
                printUsage();
                return false;
            };
        } else if (std.mem.startsWith(u8, arg, "--harness=")) {
            cfg.harness_mode = parseEnum(HarnessMode, arg[10..]) orelse {
                bench_time.print("unknown --harness={s}\n", .{arg[10..]});
                printUsage();
                return false;
            };
        } else if (std.mem.startsWith(u8, arg, "--values=")) {
            cfg.values_len = try std.fmt.parseInt(usize, arg[9..], 10);
        } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
            cfg.warmup_runs = try std.fmt.parseInt(usize, arg[9..], 10);
        } else if (std.mem.startsWith(u8, arg, "--runs=")) {
            cfg.bench_runs = try std.fmt.parseInt(usize, arg[7..], 10);
        } else if (std.mem.startsWith(u8, arg, "--iterations=")) {
            cfg.iterations = try std.fmt.parseInt(usize, arg[13..], 10);
        } else if (std.mem.eql(u8, arg, "--no-trace")) {
            cfg.trace = false;
        } else {
            bench_time.print("unknown argument: {s}\n", .{arg});
            printUsage();
            return false;
        }
    }

    return true;
}

fn parseCase(name: []const u8) ?Case {
    if (parseEnum(Case, name)) |case| return case;

    const number = std.fmt.parseInt(u8, name, 10) catch return null;
    return switch (number) {
        0 => .@"00_allocator_smoke",
        1 => .@"01_rawr_init_loop",
        2 => .@"02_rawr_add_random_once",
        3 => .@"03_rawr_add_sequential_once",
        4 => .@"04_rawr_add_random_bench",
        5 => .@"05_rawr_add_sequential_bench",
        6 => .@"06_rawr_contains_hit_bench",
        7 => .@"07_rawr_contains_miss_bench",
        8 => .@"08_croaring_add_random_bench",
        99 => .@"99_all",
        else => null,
    };
}

fn parseEnum(comptime E: type, name: []const u8) ?E {
    inline for (std.meta.fields(E)) |field| {
        if (std.mem.eql(u8, name, field.name)) {
            return @enumFromInt(field.value);
        }
    }
    return null;
}

fn printUsage() void {
    bench_time.print(
        \\usage: openbsd_probe [options]
        \\
        \\  --case=00_allocator_smoke|01_rawr_init_loop
        \\         |02_rawr_add_random_once|03_rawr_add_sequential_once
        \\         |04_rawr_add_random_bench|05_rawr_add_sequential_bench
        \\         |06_rawr_contains_hit_bench|07_rawr_contains_miss_bench
        \\         |08_croaring_add_random_bench|99_all
        \\         Bare numbers also work: --case=0, --case=04, --case=99.
        \\  --allocator=openbsd_c|std_c|smp
        \\  --call=auto|never_inline
        \\  --harness=direct|noinline_harness
        \\  --values=N        default 1000000, max 1000000
        \\  --warmup=N        default 3
        \\  --runs=N          default 21, max 64
        \\  --iterations=N    default 10000 for rawr_init_loop
        \\  --no-trace
        \\
    , .{});
}

fn allocatorFor(choice: AllocatorChoice) std.mem.Allocator {
    return switch (choice) {
        .openbsd_c => bench_time.cAllocator(),
        .std_c => std.heap.c_allocator,
        .smp => std.heap.smp_allocator,
    };
}

fn trace(cfg: *const Config, comptime message: []const u8) void {
    if (cfg.trace) bench_time.print("TRACE: " ++ message ++ "\n", .{});
}

fn initTestData(values_len: usize) void {
    var prng = std.Random.DefaultPrng.init(12345);
    for (0..values_len) |i| {
        random_values[i] = prng.random().int(u32);
        sequential_values[i] = @intCast(i);
    }
}

fn allocatorSmoke(cfg: *const Config) void {
    trace(cfg, "allocator smoke u16[4]");
    const keys = active_allocator.alloc(u16, 4) catch unreachable;
    defer active_allocator.free(keys);
    keys[0] = 0x1234;
    std.mem.doNotOptimizeAway(keys.ptr);

    trace(cfg, "allocator smoke usize[4]");
    const containers = active_allocator.alloc(usize, 4) catch unreachable;
    defer active_allocator.free(containers);
    containers[0] = 0xabcdef;
    std.mem.doNotOptimizeAway(containers.ptr);

    bench_time.print("allocator smoke ok\n", .{});
}

fn rawrInitLoop(cfg: *const Config) void {
    trace(cfg, "rawr init loop begin");
    for (0..cfg.iterations) |i| {
        var bm = RoaringBitmap.init(active_allocator) catch unreachable;
        defer bm.deinit();
        if (cfg.trace and i != 0 and i % 1000 == 0) {
            bench_time.print("TRACE: rawr init loop iteration {d}\n", .{i});
        }
        std.mem.doNotOptimizeAway(&bm);
    }
    bench_time.print("rawr init loop ok iterations={d}\n", .{cfg.iterations});
}

fn runOnce(cfg: *const Config, comptime name: []const u8, comptime func: anytype) void {
    trace(cfg, name ++ " begin");
    callBenchmarkTarget(func, .{}, cfg.call_mode);
    bench_time.print("{s} ok\n", .{name});
}

fn runBenchmark(cfg: *const Config, comptime name: []const u8, comptime func: anytype) void {
    trace(cfg, name ++ " benchmark begin");
    const result = benchmark(func, .{}, cfg);
    printResult(name, result);
}

fn benchmark(comptime func: anytype, args: anytype, cfg: *const Config) BenchResult {
    return switch (cfg.harness_mode) {
        .direct => benchmarkInline(func, args, cfg),
        .noinline_harness => benchmarkNoInline(func, args, cfg),
    };
}

fn benchmarkInline(comptime func: anytype, args: anytype, cfg: *const Config) BenchResult {
    return benchmarkImpl(func, args, cfg);
}

noinline fn benchmarkNoInline(comptime func: anytype, args: anytype, cfg: *const Config) BenchResult {
    return benchmarkImpl(func, args, cfg);
}

fn benchmarkImpl(comptime func: anytype, args: anytype, cfg: *const Config) BenchResult {
    var times: [MAX_BENCH_RUNS]u64 = undefined;

    for (0..cfg.warmup_runs) |i| {
        if (cfg.trace) bench_time.print("TRACE: warmup {d}\n", .{i});
        callBenchmarkTarget(func, args, cfg.call_mode);
    }

    for (0..cfg.bench_runs) |i| {
        if (cfg.trace) bench_time.print("TRACE: timed {d}\n", .{i});
        const start = bench_time.monotonicNanos();
        callBenchmarkTarget(func, args, cfg.call_mode);
        times[i] = bench_time.monotonicNanos() - start;
    }

    const run_times = times[0..cfg.bench_runs];
    std.mem.sort(u64, run_times, {}, std.sort.asc(u64));

    return .{
        .p25_ns = run_times[cfg.bench_runs / 4],
        .median_ns = run_times[cfg.bench_runs / 2],
        .p75_ns = run_times[(3 * cfg.bench_runs) / 4],
    };
}

fn callBenchmarkTarget(comptime func: anytype, args: anytype, mode: CallMode) void {
    switch (mode) {
        .auto => _ = @call(.auto, func, args),
        .never_inline => _ = @call(.never_inline, func, args),
    }
}

fn printResult(name: []const u8, result: BenchResult) void {
    bench_time.print("{s}: median={d:.3} ms p25={d:.3} ms p75={d:.3} ms\n", .{
        name,
        nsToMs(result.median_ns),
        nsToMs(result.p25_ns),
        nsToMs(result.p75_ns),
    });
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn benchRawrAddRandom() void {
    var bm = RoaringBitmap.init(active_allocator) catch unreachable;
    defer bm.deinit();
    for (random_values[0..active_values_len]) |value| {
        _ = bm.add(value) catch unreachable;
    }
    std.mem.doNotOptimizeAway(&bm);
}

fn benchRawrAddSequential() void {
    var bm = RoaringBitmap.init(active_allocator) catch unreachable;
    defer bm.deinit();
    for (sequential_values[0..active_values_len]) |value| {
        _ = bm.add(value) catch unreachable;
    }
    std.mem.doNotOptimizeAway(&bm);
}

fn initRawrContainsBm(cfg: *const Config) void {
    if (rawr_contains_bm != null) return;

    trace(cfg, "init rawr contains bitmap");
    var bm = RoaringBitmap.init(active_allocator) catch unreachable;
    for (random_values[0..active_values_len]) |value| {
        _ = bm.add(value) catch unreachable;
    }
    rawr_contains_bm = bm;
}

fn benchRawrContainsHit() void {
    const bm = &rawr_contains_bm.?;
    var hits: u32 = 0;
    for (random_values[0..active_values_len]) |value| {
        if (bm.contains(value)) hits += 1;
    }
    std.mem.doNotOptimizeAway(hits);
}

fn benchRawrContainsMiss() void {
    const bm = &rawr_contains_bm.?;
    var hits: u32 = 0;
    for (random_values[0..active_values_len]) |value| {
        if (bm.contains(value | 0x80000000)) hits += 1;
    }
    std.mem.doNotOptimizeAway(hits);
}

fn benchCRoaringAddRandom() void {
    const bm = c.roaring_bitmap_create() orelse unreachable;
    defer c.roaring_bitmap_free(bm);
    for (random_values[0..active_values_len]) |value| {
        c.roaring_bitmap_add(bm, value);
    }
    std.mem.doNotOptimizeAway(bm);
}

fn cleanup() void {
    if (rawr_contains_bm) |*bm| {
        bm.deinit();
        rawr_contains_bm = null;
    }
}
