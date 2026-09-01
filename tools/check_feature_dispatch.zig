// SPDX-License-Identifier: MPL-2.0

//! Compile-only assertion for baseline array-intersection dispatch.

const std = @import("std");
const array_kernels = @import("array_kernels");

comptime {
    assertScalarRegistry(array_kernels.write_bench_kernels);
    assertScalarRegistry(array_kernels.card_bench_kernels);
}

export fn rawrCheckBaselineArrayDispatch() void {}

fn assertScalarRegistry(comptime kernels: anytype) void {
    const expected = [_][]const u8{ "dispatch", "gallop", "merge" };
    if (kernels.len != expected.len) {
        @compileError("baseline target selected a SIMD array-intersection kernel");
    }
    inline for (kernels, expected) |kernel, name| {
        if (!std.mem.eql(u8, kernel.name, name)) {
            @compileError("baseline target selected a SIMD array-intersection kernel");
        }
    }
}
