const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library module (exposed for package consumers)
    const lib_mod = b.addModule("rawr", .{
        .root_source_file = b.path("src/roaring.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Static library
    const lib = b.addLibrary(.{
        .name = "rawr",
        .root_module = lib_mod,
        .linkage = .static,
    });
    b.installArtifact(lib);

    // Tests
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/roaring.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);

    // Benchmark executable (always ReleaseFast, including the library)
    const bench_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/roaring.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mod.addImport("rawr", bench_lib_mod);
    bench_mod.link_libc = true;
    addBenchmarkPlatformShim(b, bench_mod, target);

    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = bench_mod,
    });
    b.installArtifact(bench_exe);

    const bench_step = b.step("bench", "Build benchmarks");
    bench_step.dependOn(&b.addInstallArtifact(bench_exe, .{}).step);

    // CRoaring validation executable
    const validate_mod = b.createModule(.{
        .root_source_file = b.path("src/validate_croaring.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    validate_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, validate_mod, target);
    addTranslatedCImport(b, validate_mod, .{
        .header = "vendor/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .target = target,
        .optimize = .ReleaseFast,
    });

    const validate_exe = b.addExecutable(.{
        .name = "validate_croaring",
        .root_module = validate_mod,
    });
    b.installArtifact(validate_exe);

    const validate_step = b.step("validate", "Run CRoaring interop validation");
    const run_validate = b.addRunArtifact(validate_exe);
    validate_step.dependOn(&run_validate.step);

    // Differential tests against CRoaring
    const difftest_mod = b.createModule(.{
        .root_source_file = b.path("src/diff_test.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    difftest_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, difftest_mod, target);
    addTranslatedCImport(b, difftest_mod, .{
        .header = "vendor/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .target = target,
        .optimize = .ReleaseFast,
    });

    const difftest_exe = b.addExecutable(.{
        .name = "diff_test",
        .root_module = difftest_mod,
    });
    b.installArtifact(difftest_exe);

    const difftest_step = b.step("difftest", "Differential tests vs CRoaring");
    const run_difftest = b.addRunArtifact(difftest_exe);
    difftest_step.dependOn(&run_difftest.step);

    // CRoaring benchmark comparison
    const bench_cr_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_croaring.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_cr_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_cr_mod, target);
    addTranslatedCImport(b, bench_cr_mod, .{
        .header = "vendor/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_cr_exe = b.addExecutable(.{
        .name = "bench_croaring",
        .root_module = bench_cr_mod,
    });
    b.installArtifact(bench_cr_exe);

    const bench_cr_step = b.step("bench-compare", "Build CRoaring comparison benchmarks");
    bench_cr_step.dependOn(&b.addInstallArtifact(bench_cr_exe, .{}).step);

    // Allocator matrix benchmark
    const bench_alloc_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_allocators.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_alloc_mod.addImport("rawr", bench_lib_mod);
    bench_alloc_mod.link_libc = true;
    addBenchmarkPlatformShim(b, bench_alloc_mod, target);

    const bench_alloc_exe = b.addExecutable(.{
        .name = "bench_alloc",
        .root_module = bench_alloc_mod,
    });
    b.installArtifact(bench_alloc_exe);

    const bench_alloc_step = b.step("bench-alloc", "Build allocator matrix benchmark");
    bench_alloc_step.dependOn(&b.addInstallArtifact(bench_alloc_exe, .{}).step);

    const openbsd_repros_step = b.step("openbsd-repros", "Build OpenBSD runtime repro programs");
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_00_empty",
        .root = "src/openbsd_repro/00_empty.zig",
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_01_c_no_args",
        .root = "src/openbsd_repro/01_c_no_args.zig",
        .repro_c = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_02_c_puts_zstring",
        .root = "src/openbsd_repro/02_c_puts_zstring.zig",
        .repro_c = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_03_c_write_literal",
        .root = "src/openbsd_repro/03_c_write_literal.zig",
        .repro_c = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_04_c_write_stack",
        .root = "src/openbsd_repro/04_c_write_stack.zig",
        .repro_c = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_05_c_write_global",
        .root = "src/openbsd_repro/05_c_write_global.zig",
        .repro_c = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_06_std_debug_print_no_libc",
        .root = "src/openbsd_repro/06_std_debug_print.zig",
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_07_std_debug_print_libc",
        .root = "src/openbsd_repro/06_std_debug_print.zig",
        .link_libc = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_08_std_fmt_bufprint",
        .root = "src/openbsd_repro/08_std_fmt_bufprint.zig",
        .repro_c = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_09_io_writer_fixed",
        .root = "src/openbsd_repro/09_io_writer_fixed.zig",
        .repro_c = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_10_std_c_gettimeofday",
        .root = "src/openbsd_repro/10_std_c_gettimeofday.zig",
        .repro_c = true,
        .link_libc = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_11_std_c_clock_gettime",
        .root = "src/openbsd_repro/11_std_c_clock_gettime.zig",
        .repro_c = true,
        .link_libc = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_12_std_heap_c_allocator",
        .root = "src/openbsd_repro/12_std_heap_c_allocator.zig",
        .repro_c = true,
        .link_libc = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_13_c_malloc_shim",
        .root = "src/openbsd_repro/13_c_malloc_shim.zig",
        .repro_c = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_14_bench_time_print",
        .root = "src/openbsd_repro/14_bench_time_print.zig",
        .bench_shim = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_15_rawr_basic",
        .root = "src/openbsd_repro/15_rawr_basic.zig",
        .rawr = true,
        .bench_shim = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_16_croaring_basic",
        .root = "src/openbsd_repro/16_croaring_basic.zig",
        .rawr = true,
        .repro_c = true,
        .croaring = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_17_process_init",
        .root = "src/openbsd_repro/17_process_init.zig",
        .repro_c = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_18_bench_time_print_with_croaring",
        .root = "src/openbsd_repro/18_bench_time_print_with_croaring.zig",
        .bench_shim = true,
        .croaring = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_19_bench_time_print_with_rawr_and_croaring",
        .root = "src/openbsd_repro/19_bench_time_print_with_rawr_and_croaring.zig",
        .rawr = true,
        .bench_shim = true,
        .croaring = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_20_direct_bench_shim_write_with_croaring",
        .root = "src/openbsd_repro/20_direct_bench_shim_write_with_croaring.zig",
        .bench_shim = true,
        .bench_shim_c = true,
        .croaring = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_21_direct_bench_shim_write_with_rawr_and_croaring",
        .root = "src/openbsd_repro/21_direct_bench_shim_write_with_rawr_and_croaring.zig",
        .rawr = true,
        .bench_shim = true,
        .bench_shim_c = true,
        .croaring = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_22_bench_croaring_imports_no_globals",
        .root = "src/openbsd_repro/22_bench_croaring_imports_no_globals.zig",
        .rawr = true,
        .bench_shim = true,
        .croaring = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_23_bench_croaring_shape_print_only",
        .root = "src/openbsd_repro/23_bench_croaring_shape_print_only.zig",
        .rawr = true,
        .bench_shim = true,
        .croaring = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_24_relative_bench_time_print",
        .root = "src/openbsd_repro_24_relative_bench_time_print.zig",
        .bench_shim_c = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_25_bench_croaring_header_shape",
        .root = "src/openbsd_repro_25_bench_croaring_header_shape.zig",
        .rawr = true,
        .bench_shim_c = true,
        .croaring = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_26_call_bench_croaring_main",
        .root = "src/openbsd_repro_26_call_bench_croaring_main.zig",
        .rawr = true,
        .bench_shim_c = true,
        .croaring = true,
    });
    addOpenBsdRepro(b, target, openbsd_repros_step, .{
        .name = "openbsd_repro_27_bench_croaring_exact_root",
        .root = "src/bench_croaring.zig",
        .rawr = true,
        .bench_shim_c = true,
        .croaring = true,
    });

    // Tarball
    const tarball_step = b.step("tarball", "Create source tarball from git HEAD");
    const tarball_cmd = b.addSystemCommand(&.{
        "git", "archive", "--format=tar.gz", "--prefix=rawr/", "HEAD", "-o", "rawr.tar.gz",
    });
    tarball_step.dependOn(&tarball_cmd.step);
}

const OpenBsdReproOptions = struct {
    name: []const u8,
    root: []const u8,
    rawr: bool = false,
    repro_c: bool = false,
    bench_shim: bool = false,
    bench_shim_c: bool = false,
    croaring: bool = false,
    link_libc: bool = false,
};

fn addOpenBsdRepro(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    step: *std.Build.Step,
    opts: OpenBsdReproOptions,
) void {
    const bench_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/roaring.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const mod = b.createModule(.{
        .root_source_file = b.path(opts.root),
        .target = target,
        .optimize = .ReleaseFast,
    });

    if (opts.rawr) {
        mod.addImport("rawr", bench_lib_mod);
    }
    if (opts.bench_shim) {
        const bench_time_mod = b.createModule(.{
            .root_source_file = b.path("src/bench_time.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        mod.addImport("bench_time", bench_time_mod);
    }
    if (opts.repro_c) {
        mod.addCSourceFile(.{
            .file = b.path("src/openbsd_repro/repro_c.c"),
            .flags = &.{ "-std=c11", "-O0", "-g" },
        });
        mod.link_libc = true;
    }
    if (opts.bench_shim) {
        addBenchmarkPlatformShim(b, mod, target);
    }
    if (opts.bench_shim_c and !(opts.bench_shim and target.result.os.tag == .openbsd)) {
        addBenchmarkOpenBsdShimC(b, mod);
    }
    if (opts.link_libc) {
        mod.link_libc = true;
    }
    if (opts.croaring) {
        addTranslatedCImport(b, mod, .{
            .header = "vendor/croaring_wrapper.h",
            .include_dir = "vendor/",
            .c_source = "vendor/roaring.c",
            .target = target,
            .optimize = .ReleaseFast,
        });
    }

    const exe = b.addExecutable(.{
        .name = opts.name,
        .root_module = mod,
    });
    step.dependOn(&b.addInstallArtifact(exe, .{}).step);
}

fn addBenchmarkPlatformShim(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    if (target.result.os.tag == .openbsd) {
        addBenchmarkOpenBsdShimC(b, mod);
    }
}

fn addBenchmarkOpenBsdShimC(b: *std.Build, mod: *std.Build.Module) void {
    mod.addCSourceFile(.{
        .file = b.path("src/bench_openbsd.c"),
        .flags = &.{ "-std=c11", "-O2" },
    });
    mod.link_libc = true;
}

fn addTranslatedCImport(b: *std.Build, mod: *std.Build.Module, opts: struct {
    import_name: []const u8 = "c",
    header: []const u8,
    include_dir: []const u8,
    c_source: ?[]const u8 = null,
    c_flags: []const []const u8 = &.{ "-std=c11", "-O3", "-DNDEBUG" },
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
}) void {
    mod.addIncludePath(b.path(opts.include_dir));

    if (opts.target.result.os.tag == .freebsd) {
        mod.addCMacro("bswap64", "__builtin_bswap64");
    }
    if (isBsdTarget(opts.target.result)) {
        mod.addCMacro("CROARING_COMPILER_SUPPORTS_AVX512", "0");
    }

    if (opts.c_source) |c_source| {
        mod.addCSourceFile(.{
            .file = b.path(c_source),
            .flags = opts.c_flags,
        });
    }
    mod.link_libc = true;

    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path(opts.header),
        .target = opts.target,
        .optimize = opts.optimize,
    });
    translate_c.addIncludePath(b.path(opts.include_dir));
    if (opts.target.result.os.tag == .freebsd) {
        translate_c.defineCMacro("bswap64", "__builtin_bswap64");
    }
    if (isBsdTarget(opts.target.result)) {
        translate_c.defineCMacro("CROARING_COMPILER_SUPPORTS_AVX512", "0");
    }

    mod.addImport(opts.import_name, translate_c.createModule());
}

fn isBsdTarget(target: std.Target) bool {
    return switch (target.os.tag) {
        .dragonfly, .freebsd, .netbsd, .openbsd => true,
        else => false,
    };
}
