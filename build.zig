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
