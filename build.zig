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

    addCheck32Step(b);
    addCheckDocsStep(b);
    addCheckPackageStep(b);
    addRealdataCorpusSteps(b, target);
    addCrossWidthFixtureSteps(b, lib_mod, target);

    // Benchmark executable (always ReleaseFast, including the library)
    const bench_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/roaring.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const range_bench_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/range_bench_root.zig"),
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
        .header = "tools/croaring_bench_diag.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{
            "tools/croaring_iterate_diag.c",
            "tools/croaring_select_diag.c",
        },
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

    // Fresh-process real-data comparison worker used by run-realdata-bench.sh.
    const bench_realdata_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_realdata.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_realdata_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_realdata_mod, target);
    addTranslatedCImport(b, bench_realdata_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_realdata_exe = b.addExecutable(.{
        .name = "bench_realdata",
        .root_module = bench_realdata_mod,
    });
    const bench_realdata_step = b.step(
        "bench-realdata",
        "Build the pinned real-data comparison worker",
    );
    bench_realdata_step.dependOn(&b.addInstallArtifact(bench_realdata_exe, .{}).step);

    // Fresh-process array OR/ANDNOT attribution worker used by
    // run-array-attribution.sh. Kept separate from the canonical 21-row manifest.
    const bench_array_attr_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_array_attribution.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_array_attr_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_array_attr_mod, target);
    addTranslatedCImport(b, bench_array_attr_mod, .{
        .header = "tools/croaring_array_attribution.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{"tools/croaring_array_attribution.c"},
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_array_attr_exe = b.addExecutable(.{
        .name = "bench_array_attribution",
        .root_module = bench_array_attr_mod,
    });
    const bench_array_attr_step = b.step(
        "bench-array-attribution",
        "Build the real-data array OR/ANDNOT attribution worker",
    );
    bench_array_attr_step.dependOn(&b.addInstallArtifact(bench_array_attr_exe, .{}).step);

    // Spec 48 fixture, lifecycle, and allocation-accounting setup checker.
    const bench_tiny_setup_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_tiny_setup.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_tiny_setup_mod.addImport("rawr", bench_lib_mod);
    addTranslatedCImport(b, bench_tiny_setup_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_tiny_setup_exe = b.addExecutable(.{
        .name = "bench_tiny_setup",
        .root_module = bench_tiny_setup_mod,
    });
    const bench_tiny_setup_step = b.step(
        "bench-tiny-setup",
        "Build the spec 48 tiny-bitmap setup checker",
    );
    bench_tiny_setup_step.dependOn(&b.addInstallArtifact(bench_tiny_setup_exe, .{}).step);

    const run_tiny_setup = b.addRunArtifact(bench_tiny_setup_exe);
    run_tiny_setup.addArg("check");
    const check_tiny_setup_step = b.step(
        "check-tiny-setup",
        "Validate spec 48 fixtures, lifecycle, and accounting",
    );
    check_tiny_setup_step.dependOn(&run_tiny_setup.step);

    const bench_tiny_worker_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_tiny_worker.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_tiny_worker_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_tiny_worker_mod, target);
    addTranslatedCImport(b, bench_tiny_worker_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_tiny_worker_exe = b.addExecutable(.{
        .name = "bench_tiny_worker",
        .root_module = bench_tiny_worker_mod,
    });
    const bench_tiny_worker_step = b.step(
        "bench-tiny-worker",
        "Build the spec 48 tiny-bitmap timing worker",
    );
    bench_tiny_worker_step.dependOn(&b.addInstallArtifact(bench_tiny_worker_exe, .{}).step);

    const bench_tiny_mixed_worker_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_tiny_mixed_worker.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_tiny_mixed_worker_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_tiny_mixed_worker_mod, target);
    addTranslatedCImport(b, bench_tiny_mixed_worker_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_tiny_mixed_worker_exe = b.addExecutable(.{
        .name = "bench_tiny_mixed_worker",
        .root_module = bench_tiny_mixed_worker_mod,
    });
    const bench_tiny_mixed_worker_step = b.step(
        "bench-tiny-mixed-worker",
        "Build the spec 48 mixed-corpus timing worker",
    );
    bench_tiny_mixed_worker_step.dependOn(&b.addInstallArtifact(bench_tiny_mixed_worker_exe, .{}).step);

    // Fresh-process lazy-OR page-residency diagnosis worker.
    const bench_lazy_residency_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_lazy_or_residency.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_lazy_residency_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_lazy_residency_mod, target);
    addTranslatedCImport(b, bench_lazy_residency_mod, .{
        .header = "tools/bench_residency_diag.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{
            "tools/croaring_iterate_diag.c",
            "tools/croaring_select_diag.c",
            "tools/bench_residency_diag.c",
        },
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_lazy_residency_exe = b.addExecutable(.{
        .name = "bench_lazy_or_residency",
        .root_module = bench_lazy_residency_mod,
    });
    const bench_lazy_residency_step = b.step(
        "bench-lazy-residency",
        "Build lazy-OR page-residency diagnosis worker",
    );
    bench_lazy_residency_step.dependOn(
        &b.addInstallArtifact(bench_lazy_residency_exe, .{}).step,
    );

    // Fresh-process lazy-OR allocator cost attribution worker.
    const bench_lazy_allocator_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_lazy_or_allocator.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_lazy_allocator_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_lazy_allocator_mod, target);
    addTranslatedCImport(b, bench_lazy_allocator_mod, .{
        .header = "tools/bench_residency_diag.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{
            "tools/croaring_iterate_diag.c",
            "tools/croaring_select_diag.c",
            "tools/bench_residency_diag.c",
        },
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_lazy_allocator_exe = b.addExecutable(.{
        .name = "bench_lazy_or_allocator",
        .root_module = bench_lazy_allocator_mod,
    });
    const bench_lazy_allocator_step = b.step(
        "bench-lazy-allocator",
        "Build lazy-OR allocator cost attribution worker",
    );
    bench_lazy_allocator_step.dependOn(
        &b.addInstallArtifact(bench_lazy_allocator_exe, .{}).step,
    );

    // Standalone SMP allocator address-order diagnosis worker. It intentionally
    // imports neither rawr nor CRoaring so allocator layout is the only variable.
    const bench_smp_layout_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_smp_layout.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_smp_layout_mod.link_libc = true;

    const bench_smp_layout_exe = b.addExecutable(.{
        .name = "bench_smp_layout",
        .root_module = bench_smp_layout_mod,
    });
    const bench_smp_layout_step = b.step(
        "bench-smp-layout",
        "Build standalone SMP allocator address-order diagnosis worker",
    );
    bench_smp_layout_step.dependOn(
        &b.addInstallArtifact(bench_smp_layout_exe, .{}).step,
    );

    // Address-sorted repair/teardown diagnosis. This is benchmark-only and
    // leaves the production bitmap implementation unchanged.
    const bench_address_sorted_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_address_sorted.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_address_sorted_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_address_sorted_mod, target);
    addTranslatedCImport(b, bench_address_sorted_mod, .{
        .header = "tools/croaring_address_sorted.h",
        .include_dir = "tools/",
        .c_source = "tools/croaring_address_sorted.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_address_sorted_exe = b.addExecutable(.{
        .name = "bench_address_sorted",
        .root_module = bench_address_sorted_mod,
    });
    const bench_address_sorted_step = b.step(
        "bench-address-sorted",
        "Build address-sorted repair and teardown diagnosis worker",
    );
    bench_address_sorted_step.dependOn(
        &b.addInstallArtifact(bench_address_sorted_exe, .{}).step,
    );

    // Deferred demote-free ordering diagnosis. This imports the canonical
    // sparse parity corpus but keeps all candidate repair code benchmark-only.
    const bench_free_order_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_free_order.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_free_order_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_free_order_mod, target);
    addTranslatedCImport(b, bench_free_order_mod, .{
        .header = "tools/croaring_bench_diag.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{
            "tools/croaring_iterate_diag.c",
            "tools/croaring_select_diag.c",
            "tools/bench_peak_rss.c",
        },
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_free_order_exe = b.addExecutable(.{
        .name = "bench_free_order",
        .root_module = bench_free_order_mod,
    });
    const bench_free_order_step = b.step(
        "bench-free-order",
        "Build deferred demote-free ordering diagnosis worker",
    );
    bench_free_order_step.dependOn(
        &b.addInstallArtifact(bench_free_order_exe, .{}).step,
    );

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

    // Fresh-process select call-boundary and cost-attribution harness.
    const select_diag_optimize = if (optimize == .Debug) .ReleaseFast else optimize;
    const select_diag_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/roaring.zig"),
        .target = target,
        .optimize = select_diag_optimize,
    });
    const bench_select_diag_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_select_diag.zig"),
        .target = target,
        .optimize = select_diag_optimize,
    });
    bench_select_diag_mod.addImport("rawr", select_diag_lib_mod);
    addBenchmarkPlatformShim(b, bench_select_diag_mod, target);
    addTranslatedCImport(b, bench_select_diag_mod, .{
        .header = "tools/croaring_select_diag.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{"tools/croaring_select_diag.c"},
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = select_diag_optimize,
    });

    const bench_select_diag_exe = b.addExecutable(.{
        .name = "bench_select_diag",
        .root_module = bench_select_diag_mod,
    });
    const bench_select_diag_step = b.step(
        "bench-select-diag",
        "Build select call-boundary diagnosis benchmark",
    );
    bench_select_diag_step.dependOn(&b.addInstallArtifact(bench_select_diag_exe, .{}).step);

    // Compact ArrayContainer header replica diagnostic.
    const bench_compact_array_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_compact_header_array.zig"),
        .target = target,
        .optimize = select_diag_optimize,
    });
    addBenchmarkPlatformShim(b, bench_compact_array_mod, target);
    const bench_compact_array_exe = b.addExecutable(.{
        .name = "bench_compact_header_array",
        .root_module = bench_compact_array_mod,
    });
    const bench_compact_array_step = b.step(
        "bench-compact-header-array",
        "Build the compact ArrayContainer header diagnostic",
    );
    bench_compact_array_step.dependOn(
        &b.addInstallArtifact(bench_compact_array_exe, .{}).step,
    );

    // Compact RunContainer header replica diagnostic.
    const bench_compact_run_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_compact_header_run.zig"),
        .target = target,
        .optimize = select_diag_optimize,
    });
    addBenchmarkPlatformShim(b, bench_compact_run_mod, target);
    const bench_compact_run_exe = b.addExecutable(.{
        .name = "bench_compact_header_run",
        .root_module = bench_compact_run_mod,
    });
    const bench_compact_run_step = b.step(
        "bench-compact-header-run",
        "Build the compact RunContainer header diagnostic",
    );
    bench_compact_run_step.dependOn(
        &b.addInstallArtifact(bench_compact_run_exe, .{}).step,
    );

    // Rawr-only canonical full rows for compact-header three-way candidates.
    const bench_compact_full_rows_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_compact_header_full_rows.zig"),
        .target = target,
        .optimize = select_diag_optimize,
    });
    bench_compact_full_rows_mod.addImport("rawr", select_diag_lib_mod);
    addBenchmarkPlatformShim(b, bench_compact_full_rows_mod, target);
    const bench_compact_full_rows_exe = b.addExecutable(.{
        .name = "bench_compact_header_full_rows",
        .root_module = bench_compact_full_rows_mod,
    });
    const bench_compact_full_rows_step = b.step(
        "bench-compact-header-full-rows",
        "Build compact-header canonical full-row diagnostic",
    );
    bench_compact_full_rows_step.dependOn(
        &b.addInstallArtifact(bench_compact_full_rows_exe, .{}).step,
    );

    // orMany source-attribution and word-major fusion diagnostic.
    const bench_or_many_fusion_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_or_many_fusion.zig"),
        .target = target,
        .optimize = select_diag_optimize,
    });
    bench_or_many_fusion_mod.addImport("rawr", select_diag_lib_mod);
    addBenchmarkPlatformShim(b, bench_or_many_fusion_mod, target);
    addTranslatedCImport(b, bench_or_many_fusion_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = select_diag_optimize,
    });
    const bench_or_many_fusion_exe = b.addExecutable(.{
        .name = "bench_or_many_fusion",
        .root_module = bench_or_many_fusion_mod,
    });
    const bench_or_many_fusion_step = b.step(
        "bench-or-many-fusion",
        "Build the orMany fusion diagnostic",
    );
    bench_or_many_fusion_step.dependOn(
        &b.addInstallArtifact(bench_or_many_fusion_exe, .{}).step,
    );

    // select container-walk kernel matrix and prefix ceiling.
    const bench_select_kernel_matrix_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_select_kernel_matrix.zig"),
        .target = target,
        .optimize = select_diag_optimize,
    });
    bench_select_kernel_matrix_mod.addImport("rawr", select_diag_lib_mod);
    addBenchmarkPlatformShim(b, bench_select_kernel_matrix_mod, target);
    addTranslatedCImport(b, bench_select_kernel_matrix_mod, .{
        .header = "tools/croaring_select_diag.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{"tools/croaring_select_diag.c"},
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = select_diag_optimize,
    });
    const bench_select_kernel_matrix_exe = b.addExecutable(.{
        .name = "bench_select_kernel_matrix",
        .root_module = bench_select_kernel_matrix_mod,
    });
    const bench_select_kernel_matrix_step = b.step(
        "bench-select-kernel-matrix",
        "Build the select kernel matrix diagnostic",
    );
    bench_select_kernel_matrix_step.dependOn(
        &b.addInstallArtifact(bench_select_kernel_matrix_exe, .{}).step,
    );

    // Fresh-process fixed-buffer serialization diagnosis harness.
    const serialize_diag_optimize = if (optimize == .Debug) .ReleaseFast else optimize;
    const serialize_diag_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/serialize_bench_root.zig"),
        .target = target,
        .optimize = serialize_diag_optimize,
    });
    const bench_serialize_diag_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_serialize_diag.zig"),
        .target = target,
        .optimize = serialize_diag_optimize,
    });
    bench_serialize_diag_mod.addImport("rawr", serialize_diag_lib_mod);
    addBenchmarkPlatformShim(b, bench_serialize_diag_mod, target);
    addTranslatedCImport(b, bench_serialize_diag_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = serialize_diag_optimize,
    });

    const bench_serialize_diag_exe = b.addExecutable(.{
        .name = "bench_serialize_diag",
        .root_module = bench_serialize_diag_mod,
    });
    const bench_serialize_diag_step = b.step(
        "bench-serialize-diag",
        "Build fixed-buffer serialization diagnosis benchmark",
    );
    bench_serialize_diag_step.dependOn(&b.addInstallArtifact(bench_serialize_diag_exe, .{}).step);

    // Fresh-process component attribution for architecture-specific parity rows.
    const m4_cluster_diag_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_m4_cluster_diag.zig"),
        .target = target,
        .optimize = select_diag_optimize,
    });
    m4_cluster_diag_mod.addImport("rawr", select_diag_lib_mod);
    addBenchmarkPlatformShim(b, m4_cluster_diag_mod, target);
    const bench_m4_cluster_diag_exe = b.addExecutable(.{
        .name = "bench_m4_cluster_diag",
        .root_module = m4_cluster_diag_mod,
    });
    const bench_m4_cluster_diag_step = b.step(
        "bench-m4-cluster-diag",
        "Build architecture-specific parity component diagnosis benchmark",
    );
    bench_m4_cluster_diag_step.dependOn(&b.addInstallArtifact(bench_m4_cluster_diag_exe, .{}).step);

    // Fresh-process dense result-construction diagnosis harness.
    const dense_result_diag_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/dense_result_bench_root.zig"),
        .target = target,
        .optimize = select_diag_optimize,
    });
    const bench_dense_result_diag_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_dense_result_diag.zig"),
        .target = target,
        .optimize = select_diag_optimize,
    });
    bench_dense_result_diag_mod.addImport("rawr", dense_result_diag_lib_mod);
    addBenchmarkPlatformShim(b, bench_dense_result_diag_mod, target);
    addTranslatedCImport(b, bench_dense_result_diag_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "vendor/",
        .c_source = "vendor/roaring.c",
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = select_diag_optimize,
    });

    const bench_dense_result_diag_exe = b.addExecutable(.{
        .name = "bench_dense_result_diag",
        .root_module = bench_dense_result_diag_mod,
    });
    const bench_dense_result_diag_step = b.step(
        "bench-dense-result-diag",
        "Build dense result-construction diagnosis benchmark",
    );
    bench_dense_result_diag_step.dependOn(&b.addInstallArtifact(bench_dense_result_diag_exe, .{}).step);

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
        .header = "tools/croaring_bench_diag.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{
            "tools/croaring_iterate_diag.c",
            "tools/croaring_select_diag.c",
        },
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

    // Focused allocation probe for the direct-range strategy work.
    const bench_range_alloc_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_range_alloc.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_range_alloc_mod.addImport("rawr_range_bench", range_bench_lib_mod);

    const bench_range_alloc_exe = b.addExecutable(.{
        .name = "bench_range_alloc",
        .root_module = bench_range_alloc_mod,
    });
    const bench_range_alloc_step = b.step(
        "bench-range-alloc",
        "Build the range-operation allocation probe",
    );
    bench_range_alloc_step.dependOn(&b.addInstallArtifact(bench_range_alloc_exe, .{}).step);

    // Clone-vs-removeRange attribution diagnostic; intentionally outside the parity manifest.
    const bench_range_attrib_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_range_attrib.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_range_attrib_mod.addImport("rawr", bench_lib_mod);
    addBenchmarkPlatformShim(b, bench_range_attrib_mod, target);
    addTranslatedCImport(b, bench_range_attrib_mod, .{
        .header = "tools/croaring_range_attrib.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{"tools/croaring_range_attrib.c"},
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_range_attrib_exe = b.addExecutable(.{
        .name = "bench_range_attrib",
        .root_module = bench_range_attrib_mod,
    });
    const bench_range_attrib_step = b.step(
        "bench-range-attrib",
        "Build the clone and removeRange attribution diagnostic",
    );
    bench_range_attrib_step.dependOn(&b.addInstallArtifact(bench_range_attrib_exe, .{}).step);

    // Fused copy-with-range-removed construction and allocation diagnostic.
    const remove_range_copy_diag_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/remove_range_copy_bench_root.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_remove_range_copy_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_remove_range_copy.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_remove_range_copy_mod.addImport("rawr", remove_range_copy_diag_lib_mod);
    addBenchmarkPlatformShim(b, bench_remove_range_copy_mod, target);
    addTranslatedCImport(b, bench_remove_range_copy_mod, .{
        .header = "tools/croaring_wrapper.h",
        .include_dir = "tools/",
        .c_source = "vendor/roaring.c",
        .extra_c_sources = &.{"tools/croaring_range_attrib.c"},
        .croaring_avx512 = croaring_avx512,
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_remove_range_copy_exe = b.addExecutable(.{
        .name = "bench_remove_range_copy",
        .root_module = bench_remove_range_copy_mod,
    });
    const bench_remove_range_copy_step = b.step(
        "bench-remove-range-copy",
        "Build the fused removeRangeCopy diagnostic",
    );
    bench_remove_range_copy_step.dependOn(
        &b.addInstallArtifact(bench_remove_range_copy_exe, .{}).step,
    );

    // Tarball
    const tarball_step = b.step("tarball", "Create source tarball from git HEAD");
    const tarball_cmd = b.addSystemCommand(&.{
        "git", "archive", "--format=tar.gz", "--prefix=rawr/", "HEAD", "-o", "rawr.tar.gz",
    });
    tarball_step.dependOn(&tarball_cmd.step);
}

fn addRealdataCorpusSteps(b: *std.Build, target: std.Build.ResolvedTarget) void {
    const corpus_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/realdata_corpus.zig"),
            .target = target,
            .optimize = .ReleaseSafe,
        }),
    });
    const run_corpus_tests = b.addRunArtifact(corpus_tests);
    const test_step = b.step(
        "check-realdata-loader",
        "Test the deterministic external real-data corpus loader",
    );
    test_step.dependOn(&run_corpus_tests.step);

    const checker = b.addExecutable(.{
        .name = "realdata_corpus_check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/check_realdata_corpus.zig"),
            .target = target,
            .optimize = .ReleaseSafe,
        }),
    });
    const checker_step = b.step(
        "realdata-corpus-check",
        "Build the external real-data corpus checker",
    );
    checker_step.dependOn(&b.addInstallArtifact(checker, .{}).step);
}

fn addCheck32Step(b: *std.Build) void {
    const step = b.step("check-32", "Compile the public API probe for supported 32-bit targets");
    const targets = [_]struct {
        name: []const u8,
        query: std.Target.Query,
    }{
        .{
            .name = "wasm32-freestanding",
            .query = .{ .cpu_arch = .wasm32, .os_tag = .freestanding },
        },
        .{
            .name = "x86-linux-musl",
            .query = .{ .cpu_arch = .x86, .os_tag = .linux, .abi = .musl },
        },
        .{
            .name = "arm-linux-musleabi",
            .query = .{ .cpu_arch = .arm, .os_tag = .linux, .abi = .musleabi },
        },
        .{
            .name = "riscv32-linux",
            .query = .{ .cpu_arch = .riscv32, .os_tag = .linux },
        },
        .{
            .name = "x86-linux-baseline",
            .query = .{ .cpu_arch = .x86, .cpu_model = .baseline, .os_tag = .linux },
        },
    };

    for (targets) |entry| {
        const resolved = b.resolveTargetQuery(entry.query);
        const rawr_mod = b.createModule(.{
            .root_source_file = b.path("src/roaring.zig"),
            .target = resolved,
            .optimize = .ReleaseSafe,
        });
        const probe_mod = b.createModule(.{
            .root_source_file = b.path("tools/check_32_api.zig"),
            .target = resolved,
            .optimize = .ReleaseSafe,
        });
        probe_mod.addImport("rawr", rawr_mod);

        const object = b.addObject(.{
            .name = b.fmt("rawr-check-32-{s}", .{entry.name}),
            .root_module = probe_mod,
        });
        step.dependOn(&object.step);
    }
}

fn addCheckDocsStep(b: *std.Build) void {
    const rawr_mod = b.createModule(.{
        .root_source_file = b.path("src/roaring.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
    });
    const check_mod = b.createModule(.{
        .root_source_file = b.path("check_docs.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
    });
    check_mod.addImport("rawr", rawr_mod);

    const check_exe = b.addExecutable(.{
        .name = "check_docs",
        .root_module = check_mod,
    });
    const run_check = b.addRunArtifact(check_exe);
    const step = b.step("check-docs", "Check stable public method documentation coverage");
    step.dependOn(&run_check.step);
}

fn addCheckPackageStep(b: *std.Build) void {
    const check_mod = b.createModule(.{
        .root_source_file = b.path("check_package.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
    });
    const check_exe = b.addExecutable(.{
        .name = "check_package",
        .root_module = check_mod,
    });
    const run_check = b.addRunArtifact(check_exe);
    run_check.addArg(b.graph.zig_exe);

    const step = b.step("check-package", "Build and run an allowlist-only package consumer");
    step.dependOn(&run_check.step);
}

fn addCrossWidthFixtureSteps(
    b: *std.Build,
    rawr_mod: *std.Build.Module,
    target: std.Build.ResolvedTarget,
) void {
    const fixture_mod = b.createModule(.{
        .root_source_file = b.path("tools/cross_width_fixture.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    fixture_mod.addImport("rawr", rawr_mod);

    const fixture_exe = b.addExecutable(.{
        .name = "cross_width_fixture",
        .root_module = fixture_mod,
    });
    const fixture_step = b.step(
        "cross-width-fixture",
        "Build the deterministic cross-width serialization fixture tool",
    );
    fixture_step.dependOn(&b.addInstallArtifact(fixture_exe, .{}).step);

    const produce = b.addRunArtifact(fixture_exe);
    produce.addArg("produce");
    const fixture_file = produce.addOutputFileArg("rawr-cross-width.bin");

    const verify = b.addRunArtifact(fixture_exe);
    verify.addArg("verify");
    verify.addFileArg(fixture_file);

    const check_step = b.step(
        "check-cross-width-64",
        "Produce and verify serialization fixtures on the host target",
    );
    check_step.dependOn(&verify.step);
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
