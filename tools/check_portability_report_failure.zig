// SPDX-License-Identifier: MPL-2.0

//! Seeded compile failure used to verify matrix reporting continues afterward.

comptime {
    @compileError("seeded mid-matrix portability failure");
}

export fn rawrPortabilitySeededFailure() void {}
