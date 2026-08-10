<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 40: 32-bit target support

Make rawr build and work on 32-bit targets. Surfaced as a **pre-existing** blocker during spec 39-00,
where the 32-bit cross-build failed in `TaggedPtr` before reaching the benchmark under test.

**This is a portability spec, not a parity one.** No board rows, no ratio gates, no performance
requirements on 32-bit. Success is **"compiles and is correct,"** not "is fast."

## The blocker — diagnosed and fix verified

The **entire** 32-bit compile failure is one struct in `src/container.zig`:

```zig
pub const TaggedPtr = packed struct(u64) {   // ← fixed 64-bit backing
    tag: ContainerType,
    addr: u62,                                // ← fixed 62-bit address field
    ...
    return @ptrFromInt(@as(u64, self.addr) << 2);   // ← ×3, u64 into a usize parameter
```

Error on every 32-bit target:

```
error: expected type 'usize', found 'u64'
    return @ptrFromInt(@as(u64, self.addr) << 2);
note: unsigned 32-bit int cannot represent all possible unsigned 64-bit values
```

**Verified fix** (compiled during spec authoring, see Evidence):

```zig
pub const TaggedPtr = packed struct(usize) {
    tag: ContainerType,
    addr: Addr,

    /// Pointer bits minus the 2 tag bits: 62 on 64-bit, 30 on 32-bit.
    pub const Addr = std.meta.Int(.unsigned, @bitSizeOf(usize) - 2);
    ...
    return @ptrFromInt(@as(usize, self.addr) << 2);   // ×3
```

### Evidence (already gathered — do not re-derive)

With that patch applied to a scratch copy, `zig build-obj` succeeded on:

| target | before | after |
|---|---|---|
| `x86-linux` (i386) | **error** | **OK** |
| `wasm32-freestanding` | **error** | **OK** |
| `arm-linux-musleabi` | — | **OK** |
| `riscv32-linux` | — | **OK** |
| `x86_64-linux` (control) | OK | **OK** — unaffected |

A probe exercising the **full public API** — `add`, `addRange`, `remove`, `contains`, `cardinality`,
`rank`, `select`, `minimum`, `maximum`, `bitwiseAnd`, `bitwiseOr`, `lazyOr`, `clone`, `runOptimize`,
`shrinkToFit`, `serialize`, `deserialize` — **compiled clean on wasm32, i386 and arm32**. A separate
probe exercising **`Roaring64Bitmap`** (u64 *values* on a 32-bit target) also compiled clean.

**`TaggedPtr` is the only compile blocker.** No other pointer-width assumption exists: the only other
`@intFromPtr` use (`counting_allocator.zig`) is a width-agnostic comparison, and `TaggedPtr` is the only
fixed-width `packed struct`.

## What compiling does NOT prove

**This is where the actual work is.** A clean cross-compile says nothing about runtime correctness:

- The **address field narrows from 62 to 30 bits**. Sound only because the low 2 bits are tag and
  pointers are ≥4-byte aligned — but that must be *asserted*, not assumed (below).
- Arithmetic that is fine at 64-bit `usize` may **overflow or truncate at 32-bit** in ways the type
  checker accepts (e.g. `@as(usize, x) * 4` in size computations).
- SIMD paths use `@Vector(8, u16)` (128-bit) — valid on wasm32 SIMD128 and i386 SSE2, but unexercised.
- **Nothing has been executed on a 32-bit target.**

## Alignment headroom shrinks — assert it at comptime

The 2 tag bits require **≥4-byte alignment**. On 64-bit, containers are 8-byte aligned (they hold
pointers), leaving a spare bit. On 32-bit, `@alignOf` drops to **4** — still sufficient, but **exactly at
the limit with zero headroom**.

Today this is only a **runtime** `std.debug.assert(raw & 0x3 == 0)`. Add a **comptime** assertion that
each container type is at least 4-byte aligned, so a future field-layout change that drops alignment
fails at compile time on 32-bit rather than corrupting pointers at runtime:

```zig
comptime {
    for (.{ ArrayContainer, BitsetContainer, RunContainer }) |T|
        std.debug.assert(@alignOf(T) >= 4);
}
```

## Cross-width serialization must be proven, not assumed

The portable format looks width-independent — `serialize.zig` uses `usize` only for **local size/offset
arithmetic**, and all stored fields are fixed-width (the `* 4` / `* 2` byte accounting confirms it). But
this is exactly the kind of property that breaks silently.

**Required test: cross-width round-trip.** Serialize on 64-bit → deserialize on 32-bit, **and the
reverse**, asserting byte-identity and set equality. This is the highest-value new test in the spec and
it is what makes 32-bit support *meaningful* rather than merely compiling.

## Address space is a real limitation — document it

A 32-bit process has ~2–4 GB usable. A worst-case dense bitmap is 65,536 containers × 8 KB = **512 MB of
payload** before overhead, so large bitmaps are feasible but **allocation failure is far more likely**
than on 64-bit. **Document this as a known limitation**; do not present 32-bit as equivalent.

## Test execution story — the real deliverable

Compiling is cheap; **running** is the work. Pick and pin one:

| option | notes |
|---|---|
| **`wasmtime` / `node` on wasm32-wasi** | likely the cheapest — no emulator, Zig targets wasi well |
| **`qemu-user`** (i386 / arm / riscv32) | closest to a real 32-bit ABI; needs toolchain setup |
| real 32-bit hardware | unnecessary |

Whichever is chosen, the **existing test suite plus `difftest` must run and pass** on it.

## CI guard

A **cross-compile check costs seconds and needs no emulator** — it would have caught this before it sat
latent. Add `zig build-obj`/`build-lib` for **wasm32 and one of i386/arm32** to CI as a build-only gate,
independent of whether test *execution* on 32-bit is wired up.

## Out of scope

- **Performance on 32-bit.** No board rows, no ratio gates, no allocator work. The campaign's parity
  board remains 64-bit only.
- Any change to the container model, tag scheme, or serialized format.
- `SmpAllocator` behaviour (32-bit allocator characteristics are not investigated here).

## Acceptance

- `TaggedPtr` made width-generic; **`zig build` succeeds for wasm32 and at least one of
  i386 / arm32 / riscv32**; 64-bit unaffected (byte-identical behaviour, board unmoved).
- **Comptime alignment assertion** added for all container types.
- **Cross-width serialization round-trip** test passing in both directions (byte-identity + set equality).
- Full test suite **and `difftest` execute and pass on a 32-bit target** via the chosen execution story.
- **CI cross-compile guard** added.
- Address-space limitation documented in the README / allocator guidance.
- `zig build test`, `ReleaseSafe`, `ReleaseFast` green on 64-bit; canonical board unchanged.

## Chunk plan (confirm at review)

- **`40-00`** — the fix + comptime alignment assert + CI cross-compile guard + cross-width round-trip
  test (runnable entirely on existing 64-bit hosts). Delivers "compiles everywhere, format proven
  portable, regression-guarded" **without** needing an emulator.
- **`40-01`** — 32-bit **test execution** (wasmtime or qemu-user), suite + `difftest` green, limitation
  documented.

## Estimate

**S** for `40-00` — the fix is 4 lines and already verified; the round-trip test and CI guard are small.
**M** for `40-01` — dominated by toolchain/runner setup, not by code.
