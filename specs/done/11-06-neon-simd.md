<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 11-06: NEON `tbl` array kernels (aarch64)

Final chunk of [array-kernel performance](11-array-kernel-perf.md). Same algorithm
and 4 KB shuffle table as [11-05](11-05-x86-simd.md); only two primitives differ.
There's no existing NEON array kernel to port from (CRoaring's non-x64 balanced
path is scalar), so this is a from-scratch implementation for ARM targets (Apple
Silicon, ARM BSDs, Windows-on-ARM). **Measured goal:** beat the branchless-merge
baseline at balanced ratios on ARM — confirmed by the bench, not assumed.
Behavior-preserving.

**Dependency order:** after [11-05](11-05-x86-simd.md) — reuses its shuffle table,
block loop, headroom rule, and scalar tails. Structure `array_simd.zig` generic over
`(shuffle, movemask)`, instantiated per-arch at comptime, so this chunk only adds the
two aarch64 primitives.

## Gating — target feature, not arch alone

`builtin.cpu.arch == .aarch64` does **not** guarantee Advanced SIMD on every generic
aarch64 target; gate on the `.neon` feature (fall back to the merge path if absent):

```zig
const HAS_NEON = builtin.cpu.arch == .aarch64 and
    std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon);
```

## Primitive 1 — dynamic byte shuffle (`tbl`)

`@shuffle` needs a comptime mask, so inline asm:

```zig
inline fn tbl(v: @Vector(16, u8), m: @Vector(16, u8)) @Vector(16, u8) {
    return asm ("tbl %[out].16b, { %[t].16b }, %[m].16b"
        : [out] "=w" (-> @Vector(16, u8)), : [t] "w" (v), [m] "w" (m));
}
```

Out-of-range indices (the 0xFF table padding) yield 0 on `tbl` — fine, those lanes
are past `@popCount(mask)` and never counted.

## Primitive 2 — movemask

NEON has no `pmovmskb`:
1. Try `const mask: u8 = @bitCast(matches);` and inspect codegen — recent LLVM lowers
   i1×8 bitcasts on aarch64 acceptably (shift-accumulate / `addv`).
2. If codegen is poor: the `shrn.8b v, v, #4` narrow-to-GPR trick — behind the same
   `inline fn movemask8` seam so it's swappable.

Everything else (block loop with `block_end_a`/`block_end_b`, advance logic, local
scratch headroom, scalar tails, shuffle table) is shared with 11-05.

## Acceptance

- Differential suites green on an aarch64 host — Apple Silicon qualifies
  (`zig build test test64 validate validate64 difftest difftest64`, ReleaseSafe +
  ReleaseFast).
- Kernel bench vs branchless merge **2–4× at balanced ratios** (predicted — 11-00's
  `bench-aa` on the aarch64 host produces the number).
- Record the **gallop-vs-merge crossover on aarch64** (csel codegen may shift it from
  the x86 value).
- Compile-check matrix (from 11-05) covers the ARM targets; **runtime tests on an
  Apple Silicon host.**
