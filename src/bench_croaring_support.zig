// SPDX-License-Identifier: MPL-2.0

const bench_time = @import("bench_time.zig");
const c = @import("c");

pub fn main() void {
    const support: c_int = c.rawr_croaring_hardware_support();
    bench_time.print("CROARING_SUPPORT\t{d}\tavx2={s}\tavx512={s}\n", .{
        support,
        if (support & c.RAWR_CROARING_SUPPORTS_AVX2 != 0) "on" else "off",
        if (support & c.RAWR_CROARING_SUPPORTS_AVX512 != 0) "on" else "off",
    });
}
