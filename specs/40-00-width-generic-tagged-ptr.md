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

    /// The decoded address, tag bits removed. Route ALL address decoding through this —
    /// do not re-derive `.addr << 2` elsewhere.
    pub fn rawAddr(self: TaggedPtr) usize {
        return @as(usize, self.addr) << 2;
    }
```

- `getArray` / `getBitset` / `getRun` become `@ptrFromInt(self.rawAddr())`.
- **`src/bench_lazy_or_attribution.zig:170` routed through `rawAddr()`** — this removes **all current
  duplicate decode sites**, leaving `rawAddr()` as the single one.
- **Recurrence is caught mechanically — corrected after measurement.** This spec originally claimed
  centralization "does not make recurrence impossible" and leaned on the manual audit for that job.
  **Measured (see Verification record): `check-32` fails on a newly introduced `@as(u64, self.addr) << 2`
  even with the struct left correctly width-generic**, because a `u64` will not coerce to `usize` /
  `@ptrFromInt` on a 32-bit target. So every *width-breaking* recurrence is a hard compile error. What the
  guard does **not** catch is a duplicated but *correct* `@as(usize, …) << 2` — untidy, not a 32-bit break.
  The manual audit therefore polices duplication, not portability.
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

**Allocator — pin it, "no file I/O" is not sufficient.** `smp_allocator`, `page_allocator` and
`c_allocator` can each fail to build or link on `wasm32-freestanding` for reasons unrelated to pointer
width, which would make the guard fail for the wrong cause (or force dropping the freestanding target).
**Use a `FixedBufferAllocator` over local or static storage** — no OS, no libc, no syscalls.

**Surface it must reference** (no file I/O either, so it builds freestanding; compile-only, never
executed):

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

- `TaggedPtr` width-generic; `bench_lazy_or_attribution.zig:170` routed through `rawAddr()`;
  `container.zig:7` comment corrected to 4-byte.
- **Verified by search, not assertion — with a command that actually matches.** A narrow pattern like
  `\.addr\s*(<<|\*)\s*(2|4)` **does not work**: the real decode is `@as(usize, self.addr) << 2`, so a
  `)` sits between `.addr` and `<<`. That pattern returns **zero matches against the current source** —
  i.e. it would pass with the defect fully present. Use a **broad review command and read the output**:

  ```sh
  rg -n '\.addr\b|@ptrFromInt|rawAddr\(' src --glob '*.zig'
  ```

  **Record the command and its output**, and confirm by eye that **`rawAddr()` contains the only decode**.
  Remaining `.addr` uses must be **encode or compare only**.

  Known, acceptable hits at time of writing: `range_strategy_tests.zig:363` (**compare**);
  `container.zig:38` and `bench_lazy_or_attribution.zig:164` (**encode** — `.addr = @truncate(raw >> 2)`).
  Note `:164` is a *second copy of the encoding*, but `@truncate` is width-safe (`raw >> 2` truncated to
  `Addr`), so it is **not** a 32-bit break and is permitted under the encode-only rule — routing it through
  a shared init helper is optional tidying, not required here.
- Comptime `@compileError` invariants in place.
- **`zig build check-32` added**, compiling the **exported** probe across the breadth matrix, covering
  both listed surfaces including **both** Roaring64 deserialize paths.
- Fixture corpus + protocol built, width-independent per the rules above, corpus hash checked in,
  **64→64 round-trip proven**.
- 64-bit smoke shows no regression; **all four suites green on 64-bit — `test`, `difftest`, `test64`,
  `difftest64`** (this chunk changes production code and `TaggedPtr` sits beneath `Roaring64Bitmap`, so the
  64-suites are not optional here), plus `ReleaseSafe` and `ReleaseFast`.
- **No 32-bit execution required in this chunk.**

## Verification record — implemented, reviewed, ACCEPTED

Implementation delivered width-generic `TaggedPtr` with centralized decode and comptime layout checks
(`src/container.zig`), the exported probe and five-target `check-32` matrix (`build.zig`,
`tools/check_32_api.zig`), deterministic fixtures with pinned corpus hash `0x1e4d9768fabb6ac5`
(`tools/cross_width_fixture.zig`), and the protocol doc (`docs/cross-width-fixtures.md`).

**Reviewed independently, not accepted on report.** Sources were copied to a scratch tree and the guards
were run against *deliberately reintroduced defects* — the umbrella's standing question, applied to the
guard itself: *would this check fail if the defect were present?*

| Control | Result |
| --- | --- |
| **A — original defect restored** (`packed struct(u64)`, `addr: u62`, `u64` decode) | **check-32 FAILS.** Both the coercion error *and* the `@compileError("TaggedPtr must be exactly pointer-width")` fire. |
| **B — new duplicate `u64` decode site, struct left correct** | **check-32 FAILS.** Establishes the recurrence result recorded in §1. |
| **C — module-only guard, defect present, comptime block removed** | `zig build-lib src/roaring.zig -target wasm32-freestanding` → **exit 0, silent.** Confirms empirically that the probe requirement in §3 was load-bearing; a "module builds" guard would have shipped the bug. |
| **D — corpus drift**, one fixture value changed | **FAILS** with `corpus hash mismatch expected=0x1e4d9768fabb6ac5 actual=0x2b1505eb2a91fa48` → `UnexpectedCorpusHash`. |
| Restored tree | `check-32` and `check-cross-width-64` both clean. |

Control C is the one worth keeping: the justification for the exported probe was previously an argument,
and is now a measurement.

**Decode audit** (broad command per Acceptance) returns exactly one decode — `container.zig:47` inside
`rawAddr()`. All remaining hits are encode (`container.zig:41`, `bench_lazy_or_attribution.zig:164`) or
compare (`range_strategy_tests.zig:363`), as predicted. One additional hit,
`bench_lazy_or_residency.zig:325` (`@ptrFromInt(address)`, a volatile byte probe), is **not** a tag decode
— audited and cleared. Probe surface covers every function listed in §3, including both `Roaring64Bitmap`
deserialize paths. Error output shows the probe forcing analysis through real call paths
(`referenced by: select: src/bitmap.zig:1025`), not just touching signatures.

**Nit, no action required:** in `generate64`, the per-bucket `0` and `maxInt(u32)` values are appended
*after* the random draws and are not dedup-checked against them, so a colliding draw yields a duplicate
entry. Generation stays deterministic and `add` is idempotent, so corpus and hash are unaffected.

**Not yet satisfied at review time:** changes were uncommitted. Reported suites (all four × `ReleaseSafe`
/ `ReleaseFast`) and the M4 smoke (clone +0.26%, dense AND +0.35%, select −0.08%, lazyOr+repair −0.26%,
ranges overlapping, inside the 5% gate) are accepted as reported — consistent with identity-by-
construction at 64 bits.

**40-01 is unblocked.**

## Estimate

**S** — the fix is 4 lines and pre-verified; the probe, guard, and fixture protocol are small.
