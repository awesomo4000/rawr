// SPDX-License-Identifier: MPL-2.0

//! Checks direct public methods on rawr's stable types against the guarded
//! Quick Reference in API.md. Nested public types and constants are outside
//! this guard's scope.

const std = @import("std");
const rawr = @import("rawr");

const api_doc = @embedFile("API.md");
const root_source = @embedFile("src/roaring.zig");
const region_begin = "<!-- check-docs:begin -->";
const region_end = "<!-- check-docs:end -->";

const StableExport = struct {
    name: []const u8,
    value: type,
};

const stable_exports = [_]StableExport{
    .{ .name = "RoaringBitmap", .value = rawr.RoaringBitmap },
    .{ .name = "Roaring64Bitmap", .value = rawr.Roaring64Bitmap },
    .{ .name = "OwnedBitmap", .value = rawr.OwnedBitmap },
    .{ .name = "FrozenBitmap", .value = rawr.FrozenBitmap },
    .{ .name = "Frozen64Bitmap", .value = rawr.Frozen64Bitmap },
    .{ .name = "ValidateError", .value = rawr.ValidateError },
};

const InternalExport = struct {
    name: []const u8,
    reason: []const u8,
};

const internal_exports = [_]InternalExport{
    .{ .name = "ArrayContainer", .reason = "validation, benchmarks, and differential tooling" },
    .{ .name = "BitsetContainer", .reason = "validation, benchmarks, and differential tooling" },
    .{ .name = "RunContainer", .reason = "validation, benchmarks, and differential tooling" },
    .{ .name = "Container", .reason = "validation, benchmarks, and differential tooling" },
    .{ .name = "TaggedPtr", .reason = "validation, benchmarks, and differential tooling" },
    .{ .name = "container_ops", .reason = "validation, benchmarks, and differential tooling" },
    .{ .name = "optimize", .reason = "validation, benchmarks, and differential tooling" },
    .{ .name = "test_gen", .reason = "test data generation" },
    .{ .name = "roaring64_test_gen", .reason = "64-bit test data generation" },
    .{ .name = "roaring64_test_support", .reason = "64-bit test support" },
    .{ .name = "lazy_construction", .reason = "lazy-OR construction benchmark diagnostics" },
};

const MethodOmission = struct {
    type_name: []const u8,
    method_name: []const u8,
    reason: []const u8,
};

const method_omissions = [_]MethodOmission{};

comptime {
    for (internal_exports) |entry| {
        if (entry.reason.len == 0) @compileError("internal export reasons must not be empty");
    }
    for (method_omissions) |entry| {
        if (entry.reason.len == 0) @compileError("method omission reasons must not be empty");
    }
}

pub fn main() !void {
    const region = guardedRegion() catch |err| {
        std.debug.print("check-docs: malformed guarded region: {s}\n", .{@errorName(err)});
        return error.DocumentationCheckFailed;
    };

    var failures: usize = 0;
    var method_count: usize = 0;
    inline for (stable_exports) |entry| {
        if (@typeInfo(entry.value) != .@"struct") continue;
        inline for (@typeInfo(entry.value).@"struct".decls) |decl| {
            if (@typeInfo(@TypeOf(@field(entry.value, decl.name))) != .@"fn") continue;
            if (comptime isOmitted(entry.name, decl.name)) continue;
            method_count += 1;

            var token_buffer: [256]u8 = undefined;
            const token = try std.fmt.bufPrint(&token_buffer, "`{s}.{s}`", .{ entry.name, decl.name });
            if (std.mem.indexOf(u8, region, token) == null) {
                std.debug.print("check-docs: missing {s}\n", .{token});
                failures += 1;
            }
        }
    }

    failures += try checkRootExportClassification();
    if (failures != 0) {
        std.debug.print("check-docs: {d} failure(s)\n", .{failures});
        return error.DocumentationCheckFailed;
    }

    std.debug.print(
        "check-docs: OK ({d} direct public methods; nested types and constants are outside scope)\n",
        .{method_count},
    );
}

fn guardedRegion() ![]const u8 {
    const begin = std.mem.indexOf(u8, api_doc, region_begin) orelse return error.MissingBeginMarker;
    const content_start = begin + region_begin.len;
    if (std.mem.indexOfPos(u8, api_doc, content_start, region_begin) != null) {
        return error.DuplicateBeginMarker;
    }
    const end = std.mem.indexOfPos(u8, api_doc, content_start, region_end) orelse {
        return error.MissingEndMarker;
    };
    if (std.mem.indexOfPos(u8, api_doc, end + region_end.len, region_end) != null) {
        return error.DuplicateEndMarker;
    }
    return api_doc[content_start..end];
}

fn isOmitted(comptime type_name: []const u8, comptime method_name: []const u8) bool {
    inline for (method_omissions) |entry| {
        if (std.mem.eql(u8, entry.type_name, type_name) and
            std.mem.eql(u8, entry.method_name, method_name)) return true;
    }
    return false;
}

fn checkRootExportClassification() !usize {
    const allocator = std.heap.smp_allocator;
    var ast = try std.zig.Ast.parse(allocator, root_source, .zig);
    defer ast.deinit(allocator);

    if (ast.errors.len != 0) {
        std.debug.print("check-docs: failed to parse src/roaring.zig\n", .{});
        return 1;
    }

    var failures: usize = 0;
    for (ast.rootDecls()) |node| {
        const name = publicDeclName(ast, node) orelse continue;
        if (!isClassifiedRootExport(name)) {
            std.debug.print("check-docs: unclassified root export `{s}`\n", .{name});
            failures += 1;
        }
    }
    return failures;
}

fn publicDeclName(ast: std.zig.Ast, node: std.zig.Ast.Node.Index) ?[]const u8 {
    if (ast.fullVarDecl(node)) |decl| {
        if (decl.visib_token == null) return null;
        return ast.tokenSlice(decl.ast.mut_token + 1);
    }

    var buffer: [1]std.zig.Ast.Node.Index = undefined;
    if (ast.fullFnProto(&buffer, node)) |decl| {
        if (decl.visib_token == null) return null;
        const name_token = decl.name_token orelse return null;
        return ast.tokenSlice(name_token);
    }
    return null;
}

fn isClassifiedRootExport(name: []const u8) bool {
    inline for (stable_exports) |entry| {
        if (std.mem.eql(u8, name, entry.name)) return true;
    }
    inline for (internal_exports) |entry| {
        if (std.mem.eql(u8, name, entry.name)) return true;
    }
    return false;
}
