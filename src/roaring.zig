//! Rawr: A high-performance Roaring Bitmap implementation in Zig.
//!
//! Roaring bitmaps partition 32-bit integers into chunks of 2^16 values.
//! Each chunk uses the optimal container type based on cardinality.

pub const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
pub const FrozenBitmap = @import("frozen.zig").FrozenBitmap;
pub const ArrayContainer = @import("array_container.zig").ArrayContainer;
pub const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
pub const RunContainer = @import("run_container.zig").RunContainer;
pub const Container = @import("container.zig").Container;
pub const TaggedPtr = @import("container.zig").TaggedPtr;
pub const container_ops = @import("container_ops.zig");

test {
    _ = @import("array_container.zig");
    _ = @import("bitset_container.zig");
    _ = @import("run_container.zig");
    _ = @import("container.zig");
    _ = @import("container_ops.zig");
    _ = @import("bitmap.zig");
    _ = @import("frozen.zig");
    _ = @import("compare.zig");
    _ = @import("property_tests.zig");
}
