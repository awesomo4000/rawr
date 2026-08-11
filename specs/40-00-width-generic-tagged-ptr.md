<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 40-00: Width-generic `TaggedPtr`, guard, and fixtures

Toplevel: [40-32-bit-support.md](40-32-bit-support.md).

**Runs entirely on existing 64-bit hosts — no emulator, no 32-bit execution.** Delivers "compiles
everywhere, decode centralized, format fixtures ready, regression-guarded." No behaviour change on
64-bit.

## 1. Width-generic `TaggedPtr` (`src/container.zig`)

```zig
pub const TaggedPtr = packed struct(usize) {
    tag: ContainerType,
    addr: Addr,

    /// Pointer bits minus the 2 tag bits: 62 on 64-bit, 30 on 32-bit.
    pub const Addr = std.meta.Int(.unsigned, @bitSizeOf(usize) - 2);

    /// The decoded address, tag bits removed. The ONLY place this shift lives.
    pub fn rawAddr(self: TaggedPtr) usize {
        return @as(usize, self.addr) << 2;
    }
```

- `getArray` / `getBitset` / `getRun` become `@ptrFromInt(self.rawAddr())`.
- **`src/bench_lazy_or_attribution.zig:170` routed through `rawAddr()`** — after this there is
  **exactly one decode site repository-wide**, so the class of bug cannot recur.
- **Update the stale comment at `src/container.zig:7`**: it says pointers are "at least 8-byte aligned";
  the invariant is **≥4-byte**.

**64-bit is identity by construction** — `usize == u64`, `Addr == u62`. No behaviour change.

## 2. Comptime invariants — `@compileError`, not `assert`

```zig
comptime {
    if (@bitSizeOf(TaggedPtr) != @bitSizeOf(usize))
        @compileError("TaggedPtr must be exactly pointer-width");
    if (@sizeOf(TaggedPtr) != @sizeOf(usize))
        @compileError("TaggedPtr must be exactly pointer-sized");
    if (@bitSizeOf(usize) < 4)
        @compileError("target pointer width leaves no room for 2 tag bits");
    for (.{ ArrayContainer, BitsetContainer, RunContainer }) |T| {
        if (@alignOf(T) < 4)
            @compileError(@typeName(T) ++ " must be >=4-byte aligned: TaggedPtr steals 2 low bits");
    }
}
```

Descriptive messages so a future layout or target change explains itself rather than tripping an
opaque assert.

## 3. `zig build check-32` — required

**The repository has no CI** (verified: no `.github/`, no other config). **Introducing GitHub Actions is
explicitly out of scope.** Add a local build step; a future CI job invokes it.

**It must compile an in-tree API probe, not just the module.** Zig is lazily analyzed — this exact bug
survived `zig build-lib src/roaring.zig -target wasm32-freestanding` **silently**, because nothing
referenced `getArray`. A "module builds" guard would have shipped it.

**Probe shape — exported root function so lazy analysis cannot skip it:**

```zig
export fn rawrCheck32Api() void { ... }
```

**Surface it must reference** (no file I/O, so it builds freestanding; compile-only, never executed):

- **`RoaringBitmap`:** `init`, `add`, `addRange`, `remove`, `contains`, `cardinality`, `rank`, `select`,
  `minimum`, `maximum`, `bitwiseAnd`, `bitwiseOr`, `lazyOr`, `repairAfterLazy`,
  `repairAfterLazyWithOptions`, `clone`, `runOptimize`, `shrinkToFit`, `serialize`, `deserialize`,
  `serializedSizeInBytes`, `deinit`.
- **`Roaring64Bitmap`:** `init`, `add`, `addRange`, `remove`, `contains`, `cardinality`, `rank`,
  `select`, `minimum`, `maximum`, `bitwiseAnd`, `bitwiseOr`, `bitwiseDifference`, `clone`,
  `serializedSizeInBytes`, `serialize`, **`deserialize`**, **`deserializeSafe`**, `deinit` —
  **both serialization directions**.

**Breadth matrix** (pointer-width coverage, *not* SIMD-lowering — all 32-bit targets are scalar since
`array_simd.zig` gates on `x86_64`/`aarch64`): `wasm32-freestanding`, `x86-linux-musl`,
`arm-linux-musleabi`, `riscv32-linux`, `x86-linux -mcpu=baseline`.

## 4. Cross-width fixtures + producer/consumer protocol

`40-00` **builds and proves the 64→64 path**; `40-01` executes both directions.

**Corpus — both bitmap types:**

- **`RoaringBitmap`:** all three container types, empty, single-container, chunk-boundary cases.
- **`Roaring64Bitmap`:** **spanning multiple high-32-bit buckets**, plus empty and single-bucket.

**Width-independent generation — the trap this exists to avoid.** A `usize`-dependent draw produces a
*different corpus* on 32- and 64-bit, and the failure would present as a serialization bug rather than a
generator bug.

- **`RoaringBitmap`:** `u32` values.
- **`Roaring64Bitmap`:** `u64` built from fixed `u32` halves — `(@as(u64, high) << 32) | low`, `high`
  spanning several buckets.
- **Fixed-width PRNG only** (`int(u32)`, `uintLessThan(u32, n)`), pinned seed — **never** `int(usize)`.
- **Checked-in corpus hash**, asserted at both ends, so width-dependent drift fails loudly at generation.

**Protocol:** file format, invocation, comparison rules — byte-reproducible so producer and consumer can
run on different hosts at different times. The fixture executable is **separate from the probe** (it needs
file I/O and cannot target freestanding).

## 5. 64-bit non-regression smoke

Identity by construction at 64 bits (already machine-checked in §2), so a full board run is not
warranted.

- **Host:** M4. **Rows:** `clone (dense)`, `bitwiseAnd (dense)`, `select (dense)`, `lazyOr+repair`.
- **Protocol:** five fresh-process medians + full ranges, **≤5% per row**, spec-28 layout exception.

## Acceptance

- `TaggedPtr` width-generic; **`rawAddr()` the only decode site**, with
  `bench_lazy_or_attribution.zig:170` routed through it; `container.zig:7` comment corrected to 4-byte.
- Comptime `@compileError` invariants in place.
- **`zig build check-32` added**, compiling the **exported** probe across the breadth matrix, covering
  both listed surfaces including **both** Roaring64 deserialize paths.
- Fixture corpus + protocol built, width-independent per the rules above, corpus hash checked in,
  **64→64 round-trip proven**.
- 64-bit smoke shows no regression; `zig build test`, `difftest`, `ReleaseSafe`, `ReleaseFast` green.
- **No 32-bit execution required in this chunk.**

## Estimate

**S** — the fix is 4 lines and pre-verified; the probe, guard, and fixture protocol are small.
