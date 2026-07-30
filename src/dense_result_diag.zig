// SPDX-License-Identifier: MPL-2.0

//! Repository-only dense set-operation construction variants for spec 29-00.

const std = @import("std");
const bitmap_mod = @import("bitmap.zig");
const container_mod = @import("container.zig");
const ops = @import("container_ops.zig");
const RunContainer = @import("run_container.zig").RunContainer;

const RoaringBitmap = bitmap_mod.RoaringBitmap;
const Container = container_mod.Container;

pub const Operation = enum {
    band,
    bor,
};

pub const Cell = enum {
    baseline,
    a,
    b,
    c,
    a_c,
    b_c,

    pub fn usesKernelIdentity(self: Cell) bool {
        return self == .a or self == .a_c;
    }

    pub fn usesBitmapIdentity(self: Cell) bool {
        return self == .b or self == .b_c;
    }

    pub fn usesPresizing(self: Cell) bool {
        return self == .c or self == .a_c or self == .b_c;
    }

    pub fn identityEnabled(self: Cell) bool {
        return self.usesKernelIdentity() or self.usesBitmapIdentity();
    }
};

pub const Diagnostics = struct {
    identity_hits: u32 = 0,
    scratch_constructions: u32 = 0,
    scratch_reservations: u32 = 0,
    scratch_requested_bytes: u64 = 0,
};

pub fn merge(
    allocator: std.mem.Allocator,
    a: *const RoaringBitmap,
    b: *const RoaringBitmap,
    comptime operation: Operation,
    comptime cell: Cell,
    diagnostics: ?*Diagnostics,
) !RoaringBitmap {
    return switch (operation) {
        .band => mergeAnd(allocator, a, b, cell, diagnostics),
        .bor => mergeOr(allocator, a, b, cell, diagnostics),
    };
}

fn mergeAnd(
    allocator: std.mem.Allocator,
    a: *const RoaringBitmap,
    b: *const RoaringBitmap,
    comptime cell: Cell,
    diagnostics: ?*Diagnostics,
) !RoaringBitmap {
    const capacity: u32 = if (cell.usesPresizing()) @min(a.size, b.size) else 4;
    var result = try RoaringBitmap.initCapacity(allocator, capacity);
    errdefer result.deinit();

    var scratch_buffer: [8448]u8 = undefined;
    var scratch = std.heap.FixedBufferAllocator.init(&scratch_buffer);
    var i: usize = 0;
    var j: usize = 0;
    while (i < a.size and j < b.size) {
        const key_a = a.keys[i];
        const key_b = b.keys[j];
        if (key_a < key_b) {
            i += 1;
        } else if (key_a > key_b) {
            j += 1;
        } else {
            const container_a = Container.fromTagged(a.containers[i]);
            const container_b = Container.fromTagged(b.containers[j]);
            if (cell.usesBitmapIdentity() and hasFullRun(container_a, container_b)) {
                const selected = selectBitmapIdentity(.band, container_a, container_b);
                const cloned = try selected.clone(allocator);
                noteIdentity(diagnostics);
                try appendOwned(&result, key_a, cloned);
            } else {
                try appendScratchIntersection(
                    &result,
                    allocator,
                    key_a,
                    container_a,
                    container_b,
                    &scratch,
                    cell.usesKernelIdentity(),
                    diagnostics,
                );
            }
            i += 1;
            j += 1;
        }
    }
    result.cached_cardinality = -1;
    return result;
}

fn mergeOr(
    allocator: std.mem.Allocator,
    a: *const RoaringBitmap,
    b: *const RoaringBitmap,
    comptime cell: Cell,
    diagnostics: ?*Diagnostics,
) !RoaringBitmap {
    const sum = @as(u64, a.size) + b.size;
    const capacity: u32 = if (cell.usesPresizing()) @intCast(@min(sum, 65536)) else 4;
    var result = try RoaringBitmap.initCapacity(allocator, capacity);
    errdefer result.deinit();

    var i: usize = 0;
    var j: usize = 0;
    while (i < a.size and j < b.size) {
        const key_a = a.keys[i];
        const key_b = b.keys[j];
        if (key_a < key_b) {
            try appendClone(&result, allocator, key_a, Container.fromTagged(a.containers[i]));
            i += 1;
        } else if (key_a > key_b) {
            try appendClone(&result, allocator, key_b, Container.fromTagged(b.containers[j]));
            j += 1;
        } else {
            const container_a = Container.fromTagged(a.containers[i]);
            const container_b = Container.fromTagged(b.containers[j]);
            const merged = if (cell.usesBitmapIdentity() and hasFullRun(container_a, container_b)) blk: {
                noteIdentity(diagnostics);
                break :blk try selectBitmapIdentity(.bor, container_a, container_b).clone(allocator);
            } else if (cell.usesKernelIdentity())
                try unionWithKernelIdentity(allocator, container_a, container_b, diagnostics)
            else
                try ops.containerUnion(allocator, container_a, container_b);
            try appendOwned(&result, key_a, merged);
            i += 1;
            j += 1;
        }
    }
    while (i < a.size) : (i += 1) {
        try appendClone(&result, allocator, a.keys[i], Container.fromTagged(a.containers[i]));
    }
    while (j < b.size) : (j += 1) {
        try appendClone(&result, allocator, b.keys[j], Container.fromTagged(b.containers[j]));
    }
    result.cached_cardinality = -1;
    return result;
}

fn appendScratchIntersection(
    result: *RoaringBitmap,
    allocator: std.mem.Allocator,
    key: u16,
    a: Container,
    b: Container,
    scratch: *std.heap.FixedBufferAllocator,
    kernel_identity: bool,
    diagnostics: ?*Diagnostics,
) !void {
    const scratch_allocator = scratch.allocator();
    noteScratch(diagnostics, a, b);
    const intersection = if (kernel_identity)
        intersectWithKernelIdentity(scratch_allocator, a, b, diagnostics)
    else
        ops.containerIntersection(scratch_allocator, a, b);
    const temporary = intersection catch {
        scratch.reset();
        const persistent = if (kernel_identity)
            try intersectWithKernelIdentity(allocator, a, b, diagnostics)
        else
            try ops.containerIntersection(allocator, a, b);
        if (persistent.getCardinality() != 0) {
            try appendOwned(result, key, persistent);
        } else {
            persistent.deinit(allocator);
        }
        return;
    };

    if (temporary.getCardinality() != 0) {
        const persistent = try temporary.clone(allocator);
        try appendOwned(result, key, persistent);
    }
    scratch.reset();
}

fn intersectWithKernelIdentity(
    allocator: std.mem.Allocator,
    a: Container,
    b: Container,
    diagnostics: ?*Diagnostics,
) !Container {
    if (hasFullRun(a, b)) {
        noteIdentity(diagnostics);
        const selected = selectBitmapIdentity(.band, a, b);
        return copyRunWithBaselineCapacity(allocator, a, b, selected);
    }
    return ops.containerIntersection(allocator, a, b);
}

fn unionWithKernelIdentity(
    allocator: std.mem.Allocator,
    a: Container,
    b: Container,
    diagnostics: ?*Diagnostics,
) !Container {
    if (hasFullRun(a, b)) {
        noteIdentity(diagnostics);
        const selected = selectBitmapIdentity(.bor, a, b);
        return copyRunWithBaselineCapacity(allocator, a, b, selected);
    }
    return ops.containerUnion(allocator, a, b);
}

fn copyRunWithBaselineCapacity(
    allocator: std.mem.Allocator,
    a: Container,
    b: Container,
    selected: Container,
) !Container {
    const run_a = a.run;
    const run_b = b.run;
    const source = selected.run;
    const requested: u16 = @intCast(@min(@as(usize, run_a.n_runs) + run_b.n_runs, 65535));
    const result = try RunContainer.init(allocator, requested);
    @memcpy(result.runs[0..source.n_runs], source.runs[0..source.n_runs]);
    result.n_runs = source.n_runs;
    result.cardinality = -1;
    return .{ .run = result };
}

fn appendClone(
    result: *RoaringBitmap,
    allocator: std.mem.Allocator,
    key: u16,
    source: Container,
) !void {
    const cloned = try source.clone(allocator);
    try appendOwned(result, key, cloned);
}

fn appendOwned(result: *RoaringBitmap, key: u16, container: Container) !void {
    errdefer container.deinit(result.allocator);
    try result.ensureTotalCapacity(result.size + 1);
    result.keys[result.size] = key;
    result.containers[result.size] = container.toTagged();
    result.size += 1;
}

pub fn isFullRun(container: Container) bool {
    return switch (container) {
        .run => |run| run.n_runs == 1 and run.runs[0].start == 0 and run.runs[0].length == std.math.maxInt(u16),
        else => false,
    };
}

fn hasFullRun(a: Container, b: Container) bool {
    return isFullRun(a) or isFullRun(b);
}

fn selectBitmapIdentity(operation: Operation, a: Container, b: Container) Container {
    return switch (operation) {
        .band => if (isFullRun(a)) b else a,
        .bor => if (isFullRun(a)) a else b,
    };
}

fn noteIdentity(diagnostics: ?*Diagnostics) void {
    if (diagnostics) |stats| stats.identity_hits += 1;
}

fn noteScratch(diagnostics: ?*Diagnostics, a: Container, b: Container) void {
    const stats = diagnostics orelse return;
    stats.scratch_constructions += 1;
    stats.scratch_reservations += 2;
    const run_a = a.run;
    const run_b = b.run;
    const capacity = @max(@as(usize, 4), @min(@as(usize, run_a.n_runs) + run_b.n_runs, 65535));
    stats.scratch_requested_bytes += @sizeOf(RunContainer) + capacity * @sizeOf(RunContainer.RunPair);
}
