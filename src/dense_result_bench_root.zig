// SPDX-License-Identifier: MPL-2.0

//! Repository-only module root for dense result construction diagnostics.

pub const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
pub const Container = @import("container.zig").Container;
pub const dense_result_diag = @import("dense_result_diag.zig");
