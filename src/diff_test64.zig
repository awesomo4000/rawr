const std = @import("std");
const rawr = @import("rawr");
const c = @import("c");

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        var rbm = try rawr.Roaring64Bitmap.init(allocator);
        defer rbm.deinit();

        const cr = c.roaring64_bitmap_create() orelse return error.CRoaringAllocFailed;
        defer c.roaring64_bitmap_free(cr);

        try expectEmptyAgreement(&rbm, cr);
    }

    if (gpa.deinit() != .ok) return error.MemoryLeak;
    std.debug.print("difftest64: OK\n", .{});
}

fn expectEmptyAgreement(rbm: *const rawr.Roaring64Bitmap, cr: *c.roaring64_bitmap_t) !void {
    try std.testing.expect(rbm.isEmpty());
    try std.testing.expect(c.roaring64_bitmap_is_empty(cr));
    try std.testing.expectEqual(@as(u64, 0), rbm.cardinality());
    try std.testing.expectEqual(@as(u64, 0), @as(u64, @intCast(c.roaring64_bitmap_get_cardinality(cr))));
}
