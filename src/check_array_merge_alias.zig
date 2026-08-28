// SPDX-License-Identifier: MPL-2.0

//! Child process for exercising the array merge output-alias assertions.

const std = @import("std");
const rawr = @import("rawr");

const Operation = enum {
    bitwise_or,
    difference,

    fn parse(value: []const u8) ?Operation {
        if (std.mem.eql(u8, value, "union")) return .bitwise_or;
        if (std.mem.eql(u8, value, "difference")) return .difference;
        return null;
    }
};

const Case = enum {
    same_a,
    same_b,
    head,
    tail,
    inside,
    adjacent,
    separate,

    fn parse(value: []const u8) ?Case {
        inline for (std.meta.fields(Case)) |field| {
            if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
        }
        return null;
    }
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    const operation = Operation.parse(args.next() orelse return error.MissingOperation) orelse
        return error.UnknownOperation;
    const case = Case.parse(args.next() orelse return error.MissingCase) orelse
        return error.UnknownCase;
    if (args.next() != null) return error.UnexpectedArgument;

    var storage = [_]u16{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    var other = [_]u16{ 1, 3, 5, 7 };
    var separate_output: [16]u16 = undefined;

    var a: []const u16 = undefined;
    var b: []const u16 = &.{};
    var output: []u16 = undefined;
    switch (case) {
        .same_a => {
            a = storage[0..4];
            output = storage[0..4];
        },
        .same_b => {
            a = other[0..2];
            b = storage[0..4];
            output = storage[0..4];
        },
        .head => {
            a = storage[4..8];
            output = storage[2..6];
        },
        .tail => {
            a = storage[2..6];
            output = storage[4..8];
        },
        .inside => {
            a = storage[1..9];
            output = storage[3..7];
        },
        .adjacent => {
            a = storage[0..4];
            output = storage[4..8];
        },
        .separate => {
            a = &other;
            output = &separate_output;
        },
    }

    const count = switch (operation) {
        .bitwise_or => rawr.container_ops.arrayUnionWrite(a, b, output),
        .difference => rawr.container_ops.arrayDifferenceWrite(a, b, output),
    };

    if (case == .adjacent or case == .separate) {
        if (count != a.len or !std.mem.eql(u16, output[0..count], a)) {
            return error.ControlResultMismatch;
        }
    }
}
