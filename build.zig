// SPDX-License-Identifier: MPL-2.0

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const croaring_avx512 = b.option(
        bool,
        "croaring-avx512",
        "Enable AVX512 in the vendored CRoaring reference build",
    ) orelse false;

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

    const roaring64_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/roaring64_tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_roaring64_tests = b.addRunArtifact(roaring64_tests);
    const test64_step = b.step("test64", "Run Roaring64 unit tests");
    test64_step.dependOn(&run_roaring64_tests.step);

    // Benchmark executable (always ReleaseFast, including the library)
    const bench_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/roaring.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const validation_optimize = if (optimize == .Debug) .ReleaseFast else optimize;
    const validation_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/roaring.zig"),
        .target = target,
        .optimize = validation_optimize,
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
    const bench_step = b.step("bench", "Build benchmarks");
    bench_step.dependOn(&b.addInstallArtifact(bench_exe, .{}).step);

    // Standalone array-intersection kernel benchmark.
    const bench_aa_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_aa.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_aa_mod.link_libc = true;
    addBenchmarkPlatformShim(b, bench_aa_mod, target);

    const bench_aa_exe = b.addExecutable(.{
        .name = "bench_aa",
        .root_module = bench_aa_mod,
    });
    const bench_aa_step = b.step("bench-aa", "Build array-intersection kernel benchmarks");
    bench_aa_step.dependOn(&b.addInstallArtifact(bench_aa_exe, .{}).step);

    // Isolated single-allocation container prototype benchmark.
    const bench_proto_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_single_alloc.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_proto_mod.link_libc = true;
    addBenchmarkPlatformShim(b, bench_proto_mod, target);

    const bench_proto_exe = b.addExecutable(.{
        .name = "bench_single_alloc",
        .root_module = bench_proto_mod,
    });
    const bench_proto_step = b.step("bench-proto", "Build single-allocation prototype benchmark");
    bench_proto_step.dependOn(&b.addInstallArtifact(bench_proto_exe, .{}).step);

    // Transient-bitset arena Phase A experiment harness.
    const bench_transient_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_transient_arena.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_transient_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_transient_mod, target);
    addTranslatedCImport(b, bench_transient_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_transient_exe = b.addExecutable(.{
        .name = "bench_transient_arena",
        .root_module = bench_transient_mod,
    });
    const bench_transient_step = b.step(
        "bench-transient",
        "Build transient-bitset arena experiment harness",
    );
    bench_transient_step.dependOn(&b.addInstallArtifact(bench_transient_exe, .{}).step);

    // Consuming in-place OR benchmark and allocation-attribution harness.
    const bench_consuming_or_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_consuming_or.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_consuming_or_mod.addImport("rawr", bench_lib_mod);
    bench_consuming_or_mod.link_libc = true;
    addBenchmarkPlatformShim(b, bench_consuming_or_mod, target);

    const bench_consuming_or_exe = b.addExecutable(.{
        .name = "bench_consuming_or",
        .root_module = bench_consuming_or_mod,
    });
    const bench_consuming_or_step = b.step(
        "bench-consuming-or",
        "Build consuming in-place OR benchmark",
    );
    bench_consuming_or_step.dependOn(&b.addInstallArtifact(bench_consuming_or_exe, .{}).step);

    // Lazy-OR construction attribution harness.
    const bench_lazy_attr_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_lazy_or_attribution.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_lazy_attr_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_lazy_attr_mod, target);
    addTranslatedCImport(b, bench_lazy_attr_mod, .{
        .header = "tools/croaring_lazy_attribution.h",
        .include_dir = "tools/",
        .c_source = "tools/croaring_lazy_attribution.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_lazy_attr_exe = b.addExecutable(.{
        .name = "bench_lazy_or_attribution",
        .root_module = bench_lazy_attr_mod,
    });
    const bench_lazy_attr_step = b.step(
        "bench-lazy-or-attribution",
        "Build lazy-OR construction attribution benchmark",
    );
    bench_lazy_attr_step.dependOn(&b.addInstallArtifact(bench_lazy_attr_exe, .{}).step);

    // Focused parity board for operations whose broad-harness numbers need isolation.
    const bench_parity_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_parity_isolated.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_parity_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_parity_mod, target);
    addTranslatedCImport(b, bench_parity_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_parity_exe = b.addExecutable(.{
        .name = "bench_parity_isolated",
        .root_module = bench_parity_mod,
    });
    const bench_parity_step = b.step(
        "bench-parity-isolated",
        "Build focused CRoaring parity benchmark",
    );
    bench_parity_step.dependOn(&b.addInstallArtifact(bench_parity_exe, .{}).step);

    // Manifest-backed fresh-process parity worker used by run-compare-bench.sh.
    const bench_parity_worker_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_parity_worker.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_parity_worker_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_parity_worker_mod, target);
    addTranslatedCImport(b, bench_parity_worker_mod, .{
        .header = "tools/croaring_iterate_diag.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{"tools/croaring_iterate_diag.c"},
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_parity_worker_exe = b.addExecutable(.{
        .name = "bench_parity_worker",
        .root_module = bench_parity_worker_mod,
    });
    const bench_parity_worker_step = b.step(
        "bench-parity-worker",
        "Build manifest-backed CRoaring parity worker",
    );
    bench_parity_worker_step.dependOn(&b.addInstallArtifact(bench_parity_worker_exe, .{}).step);

    // Fresh-process four-path iteration diagnosis harness.
    const iterate_diag_optimize = if (optimize == .Debug) .ReleaseFast else optimize;
    const iterate_diag_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/roaring.zig"),
        .target = target,
        .optimize = iterate_diag_optimize,
    });
    const bench_iterate_diag_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_iterate_diag.zig"),
        .target = target,
        .optimize = iterate_diag_optimize,
    });
    bench_iterate_diag_mod.addImport("rawr", iterate_diag_lib_mod);
    addBenchmarkPlatformShim(b, bench_iterate_diag_mod, target);
    addTranslatedCImport(b, bench_iterate_diag_mod, .{
        .header = "tools/croaring_iterate_diag.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{"tools/croaring_iterate_diag.c"},
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = iterate_diag_optimize,
    });

    const bench_iterate_diag_exe = b.addExecutable(.{
        .name = "bench_iterate_diag",
        .root_module = bench_iterate_diag_mod,
    });
    const bench_iterate_diag_step = b.step(
        "bench-iterate-diag",
        "Build four-path iteration diagnosis benchmark",
    );
    bench_iterate_diag_step.dependOn(&b.addInstallArtifact(bench_iterate_diag_exe, .{}).step);

    // Focused skewed array-cardinality diagnosis harness.
    const bench_and_card_diag_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_and_cardinality_diag.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    addBenchmarkPlatformShim(b, bench_and_card_diag_mod, target);
    addTranslatedCImport(b, bench_and_card_diag_mod, .{
        .header = "tools/croaring_and_cardinality_diag.h",
        .include_dir = "tools/",
        .c_source = "tools/croaring_and_cardinality_diag.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_and_card_diag_exe = b.addExecutable(.{
        .name = "bench_and_cardinality_diag",
        .root_module = bench_and_card_diag_mod,
    });
    const bench_and_card_diag_step = b.step(
        "bench-and-cardinality-diag",
        "Build skewed andCardinality diagnosis benchmark",
    );
    bench_and_card_diag_step.dependOn(&b.addInstallArtifact(bench_and_card_diag_exe, .{}).step);

    // CRoaring validation executable
    const validate_mod = b.createModule(.{
        .root_source_file = b.path("src/validate_croaring.zig"),
        .target = target,
        .optimize = validation_optimize,
    });
    validate_mod.addImport("rawr", validation_lib_mod);
    addBenchmarkPlatformShim(b, validate_mod, target);
    addTranslatedCImport(b, validate_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = validation_optimize,
    });

    const validate_exe = b.addExecutable(.{
        .name = "validate_croaring",
        .root_module = validate_mod,
    });
    const validate_step = b.step("validate", "Run CRoaring interop validation");
    const run_validate = b.addRunArtifact(validate_exe);
    validate_step.dependOn(&run_validate.step);

    const validate64_mod = b.createModule(.{
        .root_source_file = b.path("src/validate_roaring64.zig"),
        .target = target,
        .optimize = validation_optimize,
    });
    validate64_mod.addImport("rawr", validation_lib_mod);
    addBenchmarkPlatformShim(b, validate64_mod, target);
    addTranslatedCImport(b, validate64_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = validation_optimize,
    });

    const validate64_exe = b.addExecutable(.{
        .name = "validate_roaring64",
        .root_module = validate64_mod,
    });
    const validate64_step = b.step("validate64", "Run CRoaring roaring64 interop validation");
    const run_validate64 = b.addRunArtifact(validate64_exe);
    validate64_step.dependOn(&run_validate64.step);

    // Differential tests against CRoaring
    const difftest_mod = b.createModule(.{
        .root_source_file = b.path("src/diff_test.zig"),
        .target = target,
        .optimize = validation_optimize,
    });
    difftest_mod.addImport("rawr", validation_lib_mod);
    addBenchmarkPlatformShim(b, difftest_mod, target);
    addTranslatedCImport(b, difftest_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = validation_optimize,
    });

    const difftest_exe = b.addExecutable(.{
        .name = "diff_test",
        .root_module = difftest_mod,
    });
    const difftest_step = b.step("difftest", "Differential tests vs CRoaring");
    const run_difftest = b.addRunArtifact(difftest_exe);
    difftest_step.dependOn(&run_difftest.step);

    const difftest64_mod = b.createModule(.{
        .root_source_file = b.path("src/diff_test64.zig"),
        .target = target,
        .optimize = validation_optimize,
    });
    difftest64_mod.addImport("rawr", validation_lib_mod);
    addBenchmarkPlatformShim(b, difftest64_mod, target);
    addTranslatedCImport(b, difftest64_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = validation_optimize,
    });

    const difftest64_exe = b.addExecutable(.{
        .name = "diff_test64",
        .root_module = difftest64_mod,
    });
    const difftest64_step = b.step("difftest64", "Differential tests vs CRoaring roaring64");
    const run_difftest64 = b.addRunArtifact(difftest64_exe);
    difftest64_step.dependOn(&run_difftest64.step);

    // CRoaring benchmark comparison
    const bench_cr_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_croaring.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_cr_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_cr_mod, target);
    addTranslatedCImport(b, bench_cr_mod, .{
        .header = "tools/croaring_iterate_diag.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{"tools/croaring_iterate_diag.c"},
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_cr_exe = b.addExecutable(.{
        .name = "bench_croaring",
        .root_module = bench_cr_mod,
    });
    const bench_cr_step = b.step("bench-compare", "Build CRoaring screening dashboard");
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
    const bench_alloc_step = b.step("bench-alloc", "Build allocator matrix benchmark");
    bench_alloc_step.dependOn(&b.addInstallArtifact(bench_alloc_exe, .{}).step);

    // Tarball
    const tarball_step = b.step("tarball", "Create source tarball from git HEAD");
    const tarball_cmd = b.addSystemCommand(&.{
        "git", "archive", "--format=tar.gz", "--prefix=rawr/", "HEAD", "-o", "rawr.tar.gz",
    });
    tarball_step.dependOn(&tarball_cmd.step);
}

fn addBenchmarkPlatformShim(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    if (target.result.os.tag == .openbsd) {
        mod.addCSourceFile(.{
            .file = b.path("src/bench_openbsd.c"),
            .flags = &.{ "-std=c11", "-O2" },
        });
        mod.link_libc = true;
    }
}

fn addTranslatedCImport(b: *std.Build, mod: *std.Build.Module, opts: struct {
    import_name: []const u8 = "c",
    header: []const u8,
    include_dir: []const u8,
    c_source: ?[]const u8 = null,
    extra_c_sources: []const []const u8 = &.{},
    c_flags: []const []const u8 = &.{ "-std=c11", "-O3", "-DNDEBUG" },
    croaring_avx512: bool = false,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
}) void {
    mod.addIncludePath(b.path(opts.include_dir));

    if (opts.target.result.os.tag == .freebsd) {
        mod.addCMacro("bswap64", "__builtin_bswap64");
    }
    if (!opts.croaring_avx512) {
        mod.addCMacro("CROARING_COMPILER_SUPPORTS_AVX512", "0");
    }

    if (opts.c_source) |c_source| {
        mod.addCSourceFile(.{
            .file = b.path(c_source),
            .flags = croaringCFlags(b, opts.c_flags, opts.croaring_avx512),
        });
    }
    for (opts.extra_c_sources) |c_source| {
        mod.addCSourceFile(.{
            .file = b.path(c_source),
            .flags = croaringCFlags(b, opts.c_flags, opts.croaring_avx512),
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
    if (!opts.croaring_avx512) {
        translate_c.defineCMacro("CROARING_COMPILER_SUPPORTS_AVX512", "0");
    }

    mod.addImport(opts.import_name, translate_c.createModule());
}

fn croaringCFlags(b: *std.Build, base: []const []const u8, croaring_avx512: bool) []const []const u8 {
    if (croaring_avx512) return base;

    const flags = b.allocator.alloc([]const u8, base.len + 1) catch @panic("OOM");
    @memcpy(flags[0..base.len], base);
    flags[base.len] = "-DCROARING_COMPILER_SUPPORTS_AVX512=0";
    return flags;
}
