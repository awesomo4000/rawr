// SPDX-License-Identifier: MPL-2.0

//! Runs independently invocable portability steps without hiding later cells.

const std = @import("std");

const Mode = enum {
    matrix,
    matrix_with_options,
    options_only,
    expect_broken,
};

const StepResult = struct {
    passed: bool,
    stdout: []u8,
    stderr: []u8,

    fn deinit(self: *StepResult, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.stderr);
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();
    _ = args.skip();

    const mode_text = args.next() orelse return error.MissingMode;
    const mode = std.meta.stringToEnum(Mode, mode_text) orelse return error.InvalidMode;
    const zig_exe = args.next() orelse return error.MissingZigExecutable;
    const expected_cells_text = args.next() orelse return error.MissingExpectedCellCount;
    const expected_cells = try std.fmt.parseInt(usize, expected_cells_text, 10);

    var option_failures: usize = 0;
    if (mode == .matrix_with_options or mode == .options_only) {
        option_failures = try runBuildOptionChecks(allocator, init.io, zig_exe);
    }
    if (mode == .options_only) {
        if (expected_cells != 0) return error.CellCountMismatch;
        if (option_failures != 0) return error.BuildOptionCheckFailed;
        return;
    }

    var cells: usize = 0;
    var broken: usize = 0;
    var not_targetable: usize = 0;
    while (args.next()) |cell| {
        cells += 1;
        const control = try runNamedStep(allocator, init.io, zig_exe, cell, "control");
        const probe = try runNamedStep(allocator, init.io, zig_exe, cell, "probe");
        const package = try runNamedStep(allocator, init.io, zig_exe, cell, "package");

        const status: []const u8 = if (!control) blk: {
            not_targetable += 1;
            break :blk "not-targetable";
        } else if (!probe or !package) blk: {
            broken += 1;
            break :blk "broken";
        } else "compiles";

        std.debug.print(
            "PORTABILITY cell={s} control={s} probe={s} package={s} status={s}\n",
            .{ cell, resultName(control), resultName(probe), resultName(package), status },
        );
    }
    if (cells != expected_cells) {
        std.debug.print(
            "PORTABILITY-ERROR expected-cells={d} actual-cells={d}\n",
            .{ expected_cells, cells },
        );
        return error.CellCountMismatch;
    }

    std.debug.print(
        "PORTABILITY total={d} broken={d} not-targetable={d} option-failures={d}\n",
        .{ cells, broken, not_targetable, option_failures },
    );

    if (mode == .expect_broken) {
        if (broken == 0 or option_failures != 0) return error.ExpectedBrokenCell;
        return;
    }
    if (broken != 0 or option_failures != 0) return error.PortabilityCheckFailed;
}

fn runNamedStep(
    allocator: std.mem.Allocator,
    io: std.Io,
    zig_exe: []const u8,
    cell: []const u8,
    phase: []const u8,
) !bool {
    const step = try std.fmt.allocPrint(allocator, "check-portability-{s}-{s}", .{ cell, phase });
    defer allocator.free(step);

    var result = try runCommand(allocator, io, &.{ zig_exe, "build", step, "--summary", "none" });
    defer result.deinit(allocator);
    if (!result.passed) {
        std.debug.print(
            "PORTABILITY-ERROR cell={s} phase={s}\n{s}{s}",
            .{ cell, phase, result.stdout, result.stderr },
        );
    }
    return result.passed;
}

fn runBuildOptionChecks(allocator: std.mem.Allocator, io: std.Io, zig_exe: []const u8) !usize {
    const checks = [_]struct {
        name: []const u8,
        argv_tail: []const []const u8,
    }{
        .{
            .name = "croaring-avx512=false",
            .argv_tail = &.{
                "build",
                "bench-parity-worker",
                "-Dtarget=aarch64-linux-gnu",
                "-Dcpu=baseline",
                "-Dcroaring-avx512=false",
                "--summary",
                "none",
            },
        },
        .{
            .name = "croaring-avx512=true",
            .argv_tail = &.{
                "build",
                "bench-parity-worker",
                "-Dtarget=x86_64-linux-gnu",
                "-Dcpu=x86_64_v4+evex512",
                "-Dcroaring-avx512=true",
                "--summary",
                "none",
            },
        },
    };

    var failures: usize = 0;
    for (checks) |check| {
        var argv: [9][]const u8 = undefined;
        argv[0] = zig_exe;
        @memcpy(argv[1 .. check.argv_tail.len + 1], check.argv_tail);
        var result = try runCommand(allocator, io, argv[0 .. check.argv_tail.len + 1]);
        defer result.deinit(allocator);
        if (!result.passed) {
            failures += 1;
            std.debug.print("OPTION name={s} result=fail\n{s}{s}", .{
                check.name,
                result.stdout,
                result.stderr,
            });
        } else {
            std.debug.print("OPTION name={s} result=pass\n", .{check.name});
        }
    }
    return failures;
}

fn runCommand(
    allocator: std.mem.Allocator,
    io: std.Io,
    argv: []const []const u8,
) !StepResult {
    const result = try std.process.run(allocator, io, .{
        .argv = argv,
        .stdout_limit = .limited(4 * 1024 * 1024),
        .stderr_limit = .limited(4 * 1024 * 1024),
    });
    return .{
        .passed = switch (result.term) {
            .exited => |code| code == 0,
            else => false,
        },
        .stdout = result.stdout,
        .stderr = result.stderr,
    };
}

fn resultName(passed: bool) []const u8 {
    return if (passed) "pass" else "fail";
}
