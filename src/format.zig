/// RoaringFormatSpec serialization constants.
/// These are shared between RoaringBitmap and FrozenBitmap.

pub const SERIAL_COOKIE_NO_RUNCONTAINER: u32 = 12346;
pub const SERIAL_COOKIE: u32 = 12347;
pub const NO_OFFSET_THRESHOLD: u32 = 4; // Containers below this don't use offset header
