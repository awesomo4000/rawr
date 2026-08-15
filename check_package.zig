// SPDX-License-Identifier: MPL-2.0

//! Rebuilds a consumer from exactly the files allowed by build.zig.zon.

const std = @import("std");
const manifest = @import("build.zig.zon");

const package_build =
    \\// SPDX-License-Identifier: MPL-2.0
    \\const std = @import("std");
    \\pub fn build(b: *std.Build) void {
    \\    const dep = b.dependency("rawr", .{});
    \\    const exe = b.addExecutable(.{
    \\        .name = "rawr-package-consumer",
    \\        .root_module = b.createModule(.{
    \\            .root_source_file = b.path("main.zig"),
    \\            .target = b.graph.host,
    \\            .optimize = .ReleaseSafe,
    \\        }),
    \\    });
    \\    exe.root_module.addImport("rawr", dep.module("rawr"));
    \\    const run = b.addRunArtifact(exe);
    \\    b.step("check", "Build and run the package consumer").dependOn(&run.step);
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
    if (args.next() != null) return error.TooManyArguments;

    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const scratch = ".zig-cache/check-package";
    try cwd.deleteTree(io, scratch);
    defer cwd.deleteTree(io, scratch) catch {};

    const package_dir = try cwd.createDirPathOpen(io, scratch ++ "/package", .{});
    defer package_dir.close(io);
    const consumer_dir = try cwd.createDirPathOpen(io, scratch ++ "/consumer", .{});
    defer consumer_dir.close(io);

    inline for (manifest.paths) |path| {
        try cwd.copyFile(path, package_dir, path, io, .{ .make_path = true });
    }

    try consumer_dir.writeFile(io, .{ .sub_path = "build.zig", .data = package_build });
    try consumer_dir.writeFile(io, .{ .sub_path = "main.zig", .data = consumer_source });
    try consumer_dir.writeFile(io, .{ .sub_path = "build.zig.zon", .data = consumer_manifest });

    const result = try std.process.run(allocator, io, .{
        .argv = &.{ zig_exe, "build", "check" },
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
    std.debug.print("check-package: OK ({d} allowlisted files)\n", .{manifest.paths.len});
}
