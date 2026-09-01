// SPDX-License-Identifier: MPL-2.0

//! Rebuilds a consumer from exactly the files allowed by build.zig.zon.

const std = @import("std");
const manifest = @import("build.zig.zon");

const package_build =
    \\// SPDX-License-Identifier: MPL-2.0
    \\const std = @import("std");
    \\pub fn build(b: *std.Build) void {
    \\    const target = b.standardTargetOptions(.{});
    \\    const run_consumer = b.option(bool, "run-consumer", "Run the package consumer") orelse false;
    \\    const dep = b.dependency("rawr", .{ .target = target, .optimize = .ReleaseSafe });
    \\    const exe = b.addExecutable(.{
    \\        .name = "rawr-package-consumer",
    \\        .root_module = b.createModule(.{
    \\            .root_source_file = b.path("main.zig"),
    \\            .target = target,
    \\            .optimize = .ReleaseSafe,
    \\        }),
    \\    });
    \\    exe.root_module.addImport("rawr", dep.module("rawr"));
    \\    const check = b.step("check", "Build or run the package consumer");
    \\    if (run_consumer) {
    \\        const run = b.addRunArtifact(exe);
    \\        check.dependOn(&run.step);
    \\    } else {
    \\        check.dependOn(&exe.step);
    \\    }
    \\}
;

const consumer_source =
    \\// SPDX-License-Identifier: MPL-2.0
    \\const std = @import("std");
    \\const rawr = @import("rawr");
    \\pub fn main() !void {
    \\    var bitmap = try rawr.RoaringBitmap.init(std.heap.smp_allocator);
    \\    defer bitmap.deinit();
    \\    _ = try bitmap.add(42);
    \\    if (!bitmap.contains(42) or bitmap.rank(42) != 1) return error.BitmapCheckFailed;
    \\    var bitmap64 = try rawr.Roaring64Bitmap.init(std.heap.smp_allocator);
    \\    defer bitmap64.deinit();
    \\    _ = try bitmap64.add((@as(u64, 1) << 32) | 42);
    \\    if (bitmap64.cardinality() != 1) return error.Bitmap64CheckFailed;
    \\}
;

const consumer_manifest =
    \\.{
    \\    .name = .rawr_package_consumer,
    \\    .version = "0.0.0",
    \\    .fingerprint = 0xcdd6879e110e90cf,
    \\    .minimum_zig_version = "0.16.0",
    \\    .dependencies = .{
    \\        .rawr = .{ .path = "../package" },
    \\    },
    \\    .paths = .{ "build.zig", "build.zig.zon", "main.zig" },
    \\}
;

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();
    _ = args.skip();

    const zig_exe = args.next() orelse return error.MissingZigExecutable;
    var target: ?[]const u8 = null;
    var cpu: ?[]const u8 = null;
    var run_consumer = true;
    var scratch_suffix: ?[]const u8 = null;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--target")) {
            if (target != null) return error.DuplicateTarget;
            target = args.next() orelse return error.MissingTarget;
        } else if (std.mem.eql(u8, arg, "--cpu")) {
            if (cpu != null) return error.DuplicateCpu;
            cpu = args.next() orelse return error.MissingCpu;
        } else if (std.mem.eql(u8, arg, "--build-only")) {
            run_consumer = false;
        } else if (std.mem.eql(u8, arg, "--scratch-suffix")) {
            if (scratch_suffix != null) return error.DuplicateScratchSuffix;
            scratch_suffix = args.next() orelse return error.MissingScratchSuffix;
            if (!validScratchSuffix(scratch_suffix.?)) return error.InvalidScratchSuffix;
        } else {
            return error.UnknownArgument;
        }
    }
    if (run_consumer and (target != null or cpu != null)) return error.RunTargetOverride;
    if (cpu != null and target == null) return error.CpuWithoutTarget;

    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const scratch = if (scratch_suffix) |suffix|
        try std.fmt.allocPrint(allocator, ".zig-cache/check-package-{s}", .{suffix})
    else
        try allocator.dupe(u8, ".zig-cache/check-package");
    defer allocator.free(scratch);
    try cwd.deleteTree(io, scratch);
    defer cwd.deleteTree(io, scratch) catch {};

    const package_path = try std.fmt.allocPrint(allocator, "{s}/package", .{scratch});
    defer allocator.free(package_path);
    const consumer_path = try std.fmt.allocPrint(allocator, "{s}/consumer", .{scratch});
    defer allocator.free(consumer_path);

    const package_dir = try cwd.createDirPathOpen(io, package_path, .{});
    defer package_dir.close(io);
    const consumer_dir = try cwd.createDirPathOpen(io, consumer_path, .{});
    defer consumer_dir.close(io);

    inline for (manifest.paths) |path| {
        try cwd.copyFile(path, package_dir, path, io, .{ .make_path = true });
    }

    try consumer_dir.writeFile(io, .{ .sub_path = "build.zig", .data = package_build });
    try consumer_dir.writeFile(io, .{ .sub_path = "main.zig", .data = consumer_source });
    try consumer_dir.writeFile(io, .{ .sub_path = "build.zig.zon", .data = consumer_manifest });

    var argv: [8][]const u8 = undefined;
    var argc: usize = 0;
    argv[argc] = zig_exe;
    argc += 1;
    argv[argc] = "build";
    argc += 1;
    argv[argc] = "check";
    argc += 1;
    argv[argc] = if (run_consumer) "-Drun-consumer=true" else "-Drun-consumer=false";
    argc += 1;
    const target_arg = if (target) |value|
        try std.fmt.allocPrint(allocator, "-Dtarget={s}", .{value})
    else
        null;
    defer if (target_arg) |value| allocator.free(value);
    if (target_arg) |value| {
        argv[argc] = value;
        argc += 1;
    }
    const cpu_arg = if (cpu) |value|
        try std.fmt.allocPrint(allocator, "-Dcpu={s}", .{value})
    else
        null;
    defer if (cpu_arg) |value| allocator.free(value);
    if (cpu_arg) |value| {
        argv[argc] = value;
        argc += 1;
    }

    const result = try std.process.run(allocator, io, .{
        .argv = argv[0..argc],
        .cwd = .{ .dir = consumer_dir },
        .stdout_limit = .limited(1024 * 1024),
        .stderr_limit = .limited(1024 * 1024),
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    const success = switch (result.term) {
        .exited => |code| code == 0,
        else => false,
    };
    if (!success) {
        std.debug.print("check-package: consumer failed\n{s}{s}", .{ result.stdout, result.stderr });
        return error.PackageConsumerFailed;
    }
    if (run_consumer) {
        std.debug.print("check-package: OK ({d} allowlisted files)\n", .{manifest.paths.len});
    } else {
        std.debug.print("check-package: OK ({d} allowlisted files, cross-target build)\n", .{
            manifest.paths.len,
        });
    }
}

fn validScratchSuffix(suffix: []const u8) bool {
    if (suffix.len == 0) return false;
    for (suffix) |byte| {
        if (!std.ascii.isAlphanumeric(byte) and byte != '-' and byte != '_') return false;
    }
    return true;
}
