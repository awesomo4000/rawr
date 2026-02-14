const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library module
    const lib_mod = b.createModule(.{
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
    validate_mod.addIncludePath(b.path("vendor/"));
    validate_mod.addCSourceFile(.{
        .file = b.path("vendor/roaring.c"),
        .flags = &.{ "-std=c11", "-O3", "-DNDEBUG" },
    });
    validate_mod.link_libc = true;

    const validate_exe = b.addExecutable(.{
        .name = "validate_croaring",
        .root_module = validate_mod,
    });
    b.installArtifact(validate_exe);

    const validate_step = b.step("validate", "Run CRoaring interop validation");
    const run_validate = b.addRunArtifact(validate_exe);
    validate_step.dependOn(&run_validate.step);

    // CRoaring benchmark comparison
    const bench_cr_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_croaring.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_cr_mod.addImport("rawr", bench_lib_mod);
    bench_cr_mod.addIncludePath(b.path("vendor/"));
    bench_cr_mod.addCSourceFile(.{
        .file = b.path("vendor/roaring.c"),
        .flags = &.{ "-std=c11", "-O3", "-DNDEBUG" },
    });
    bench_cr_mod.link_libc = true;

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
