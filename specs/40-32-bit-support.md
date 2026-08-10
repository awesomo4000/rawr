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

**`TaggedPtr` is the only compile blocker in the LIBRARY.** But it is **not the only fixed-width pointer
reconstruction in the repository** — my original survey filtered `src/bench*` out and missed one:

- **`src/bench_lazy_or_attribution.zig:170`** — `@ptrFromInt(@as(u64, tagged.addr) << 2)`, an independent
  reconstruction of the same encoding in diagnostic tooling.

**PINNED — centralize the decode rather than patching the copy.** Converting that one line to `usize`
fixes today's break but leaves the *class* open: a third raw decode could appear and regress 32-bit
silently, because **`check-32` compiles the API probe — it does not compile every diagnostic**.

So: **add a single type-agnostic accessor on `TaggedPtr`** and route every raw-address reconstruction
through it —

```zig
/// The decoded address, with the 2 tag bits removed. The ONLY place this shift lives.
pub fn rawAddr(self: TaggedPtr) usize {
    return @as(usize, self.addr) << 2;
}
```

`getArray` / `getBitset` / `getRun` become `@ptrFromInt(self.rawAddr())`, and
**`bench_lazy_or_attribution.zig:170` calls `rawAddr()`** instead of re-deriving the shift. **After this
there is exactly one decode site in the repository**, so the regression cannot recur regardless of what
`check-32` compiles.

**No alternative path.** Centralization is the pinned approach — patching the copy while leaving a second
decode site in the tree is not an accepted fallback.

The remaining `@intFromPtr` use (`counting_allocator.zig`) is a width-agnostic comparison and is fine.

## What compiling does NOT prove

**This is where the actual work is.** A clean cross-compile says nothing about runtime correctness:

- The **address field narrows from 62 to 30 bits**. Sound only because the low 2 bits are tag and
  pointers are ≥4-byte aligned — but that must be *asserted*, not assumed (below).
- Arithmetic that is fine at 64-bit `usize` may **overflow or truncate at 32-bit** in ways the type
  checker accepts (e.g. `@as(usize, x) * 4` in size computations).
- **SIMD is not a 32-bit variable at all.** `array_simd.zig` gates dispatch on
  **`arch == .x86_64`** (`has_x86_simd`) and **`arch == .aarch64`** (`has_neon`), so **every current
  32-bit target — x86, arm32, riscv32, wasm32 — takes the scalar array-intersection path regardless of
  CPU features.** There is only one lowering to validate, and **the executable 32-bit test already
  covers it**. (Two earlier drafts were wrong here: first claiming i386 guarantees SSE2, then claiming
  the matrix exercises both a scalar and a feature-bearing 32-bit lowering. Neither holds.)
- **Nothing has been executed on a 32-bit target.**

## Alignment headroom shrinks — assert it at comptime

The 2 tag bits require **≥4-byte alignment**. On 64-bit, containers are 8-byte aligned (they hold
pointers), leaving a spare bit. On 32-bit, `@alignOf` drops to **4** — still sufficient, but **exactly at
the limit with zero headroom**.

Today this is only a **runtime** `std.debug.assert(raw & 0x3 == 0)`. Add **comptime invariants** so a
future layout or target change fails at compile time rather than corrupting pointers at runtime — and use
**descriptive `@compileError`**, not `std.debug.assert`, so the failure explains itself:

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

**Also assert 64-bit layout is unchanged** by this spec, since `usize == u64` and `Addr == u62` there —
the representation is identical *by construction*, and that should be machine-checked rather than argued.

## Cross-width serialization must be proven, not assumed

The portable format looks width-independent — `serialize.zig` uses `usize` only for **local size/offset
arithmetic**, and all stored fields are fixed-width (the `* 4` / `* 2` byte accounting confirms it). But
this is exactly the kind of property that breaks silently.

**Required test: cross-width round-trip** — serialize on 64-bit → deserialize on 32-bit **and the
reverse**, asserting byte-identity and set equality. Highest-value new test in the spec; it is what makes
32-bit support *meaningful* rather than merely compiling.

**It CANNOT be completed in `40-00`** — a genuine 64→32→64 round trip requires **executing 32-bit code**,
which `40-00` deliberately does not do. So it splits:

| chunk | responsibility |
|---|---|
| **`40-00`** | build the **deterministic fixture corpus** (pinned seeds/shapes, incl. all three container types, empty, single-container, and chunk-boundary cases) and the **producer/consumer protocol** — file format, invocation, comparison rules. Prove the 64-bit→64-bit path with it. |
| **`40-01`** | **execute both directions** under the pinned 32-bit runtime and compare. |

Fixtures must be **byte-reproducible** so producer and consumer can be run on different hosts at
different times — and that requires **width-independent generation, which is easy to get wrong**:

- **Never generate with `usize`-dependent operations.** `random().int(usize)`, `uintLessThan(usize, n)`,
  or any length/index typed `usize` produce **different corpora on 32-bit and 64-bit**, silently making
  the "same" fixture different on the two ends — which would look like a serialization bug.
- **Pin: values are `u32`, and the generator uses only fixed-width PRNG calls** (`int(u32)`,
  `uintLessThan(u32, n)`) with a pinned seed.
- **Belt and braces: check in an expected hash (or the bytes) of the generated corpus**, asserted on both
  ends, so any width-dependent drift fails loudly at generation time rather than being misdiagnosed as a
  format defect downstream.

## Address space is a real limitation — document it

A 32-bit process has ~2–4 GB usable. A worst-case dense bitmap is 65,536 containers × 8 KB = **512 MB of
payload** before overhead, so large bitmaps are feasible but **allocation failure is far more likely**
than on 64-bit. **Document this as a known limitation**; do not present 32-bit as equivalent.

## Test execution — ONE pinned target and runner

Compiling is cheap; **running** is the work, and leaving the runner open is not acceptable because
**`difftest` compiles and runs vendored CRoaring (C)**.

**PRIMARY (pinned): statically-linked `x86-linux-musl`, executed NATIVELY — no emulator.** Verified on the
actual Zen 4 / WSL2 host: the kernel runs static i386 binaries directly
(`ELF 32-bit LSB executable, Intel 80386, statically linked` → `exit=0`). Static musl avoids a 32-bit
sysroot, and C compiles and links normally, which is what `difftest` needs.

**This still exercises everything the spec cares about** — the real 32-bit ABI, `usize` width, pointer
layout, musl, and CRoaring's portable path — and it is *preferable* to emulation: faster, fewer moving
parts, nothing to install.

**Explicitly NOT assumed: wasm32.** The compile evidence in this spec is `wasm32-freestanding`, which
**does not establish that `wasm32-wasi` can build and run the C-backed differential test.** wasm32 may be
added later as a *compile-only* target; it is not the execution vehicle.

### What the runner does and does NOT require (read before installing anything)

- **Nothing to install.** No QEMU, no Linux distro, no VM, no rootfs, no `debootstrap`.
- **No sysroot and no multilib** — because the binary is **statically linked musl**. That is the whole
  reason musl was chosen. **Do not switch the target to `-gnu`**: a dynamically linked 32-bit binary would
  need a 32-bit loader and libc present, which is where sysroot/multilib pain begins.
- **Zig supplies the C toolchain.** It ships musl and compiles the vendored CRoaring itself; `build.zig`
  threads `target` through to every artifact, so `-Dtarget=x86-linux-musl` reaches the C compilation too.
  No system cross-compiler needed.

**Do NOT pass `-fqemu` — it would not do what an earlier draft of this spec claimed.** Zig 0.16 treats
`x86` targets as **natively executable on an `x86_64` Linux host**: its runner returns `.native` **before**
considering QEMU, so `-fqemu` does not force `qemu-i386` for this host/target pair. Earlier text asserting
that Zig would invoke `qemu-i386`, and the accompanying package-name / symlink / `binfmt_misc` discussion,
was **wrong and has been removed**.

**If native execution ever proves insufficient**, QEMU must be wired **explicitly** — a custom build step
prefixing the test artifact with `qemu-i386`. **`-fqemu` will not do it on this host/target pair.**

**Expected CRoaring behaviour on 32-bit (not a defect).** `vendor/roaring.h` sets `CROARING_IS_X64` only
under `__x86_64__` / `_M_X64`; on `x86-linux-musl` **neither is defined**, so the SIMD/intrinsics block is
skipped and CRoaring takes its **portable C path by design**. That path is less travelled than the x64
one, which is exactly why **`difftest` — not unit tests alone — must be part of the preflight.**

### Execution host and preflight — NOT blocked; runnable as soon as the fix lands

**Host (pinned): the Zen 4 / WSL2 machine.** Zig is present but **not on `PATH`** — use the explicit path
**`/home/alr/.zvm/0.16.0/zig`** (0.16.0). **Nothing else needs installing.**

```sh
/home/alr/.zvm/0.16.0/zig build test     -Dtarget=x86-linux-musl
/home/alr/.zvm/0.16.0/zig build difftest -Dtarget=x86-linux-musl
```

**Both must pass.** If `difftest` cannot be made to run, report it and re-pick the runner — **do NOT
silently reduce acceptance to unit tests only.** `difftest` is the reason the vehicle had to link C at all,
so losing it invalidates the choice rather than merely narrowing coverage.

**`40-01` is no longer blocked on tooling** — the earlier QEMU-installation blocker does not exist. It can
preflight **immediately after the `TaggedPtr` fix lands**.

**Fallback if native execution proves insufficient:** `wasm32-wasi` + `wasmtime` (Zig compiles C to wasi,
so `difftest` is not automatically disqualified), or explicit QEMU via a custom build step as described
above.

## Also fix: the stale alignment comment

`src/container.zig:7` currently reads:

```zig
/// Low 2 bits encode the container type (pointers are at least 8-byte aligned).
```

That was true when the type was 64-bit-only. **The invariant this spec establishes is ≥4-byte
alignment** (which is what 2 tag bits actually require, and what 32-bit containers provide). Update the
comment to say **4**, so it matches the new `@compileError` invariant rather than contradicting it.

## Acceptance

- `TaggedPtr` made width-generic; **`zig build` succeeds for wasm32 and at least one of
  i386 / arm32 / riscv32**; 64-bit unaffected (byte-identical behaviour, board unmoved).
- **Comptime alignment assertion** added for all container types.
- **Fixture corpus + producer/consumer protocol** defined and byte-reproducible (`40-00`), generated with
  **`u32` values and fixed-width PRNG calls only** — no `usize`-dependent operations — with a **checked-in
  corpus hash** asserted on both ends; **cross-width round-trip executed in BOTH directions** (`40-01`),
  byte-identity + set equality.
- Full test suite **and `difftest` execute and pass** under **static `x86-linux-musl`, natively** (no
  emulator); if `difftest` cannot run there, that is reported and the runner re-picked — **not** silently
  downgraded to unit tests.
- **Decode centralized:** `TaggedPtr.rawAddr()` added, the three getters and
  `bench_lazy_or_attribution.zig:170` all routed through it — **exactly one decode site repository-wide**.
- **`zig build check-32` added — required**, compiling an **in-tree, no-I/O API probe exported as a root
  function** (`export fn rawrCheck32Api() void`) so lazy analysis cannot skip it — not merely the module,
  which is shallow enough to have missed this very bug — across the compile-only **breadth matrix**,
  exercising the listed `RoaringBitmap` surface plus `Roaring64Bitmap`. **GitHub Actions explicitly out of
  scope.**
- **`src/container.zig:7` comment corrected** from 8-byte to 4-byte alignment.
- **Runner preflight completed on the WSL2 host** — Zig invoked at **`/home/alr/.zvm/0.16.0/zig`** with
  **`-Dtarget=x86-linux-musl`** (no `-fqemu`, no install), and **both `test` and `difftest` confirmed
  running**.
- **64-bit focused smoke** (M4; clone / dense-AND / select / lazyOr+repair; ≤5%, five fresh processes)
  shows no regression.
- Address-space limitation documented in the README / allocator guidance.
- `zig build test`, `ReleaseSafe`, `ReleaseFast` green on 64-bit; canonical board unchanged.

## Chunk plan (confirm at review)

- **`40-00`** — width-generic `TaggedPtr`; **comptime invariants** (`@compileError`); **centralized
  `rawAddr()` decode** with `bench_lazy_or_attribution.zig:170` routed through it; the **compile-only
  breadth matrix**; **deterministic width-independent serialization fixtures + producer/consumer
  protocol**; **`zig build check-32` (required)**; corrected `container.zig:7` comment; 64-bit focused
  smoke. **All runnable on existing 64-bit hosts — no emulator.**
- **`40-01`** — **native 32-bit execution** (static `x86-linux-musl`, no emulator, preflighted for both
  `test` and `difftest`); **actual 32-bit unit + differential execution**; **bidirectional cross-width
  fixture exchange**; address-space limitation documented. **Unblocked once the `TaggedPtr` fix lands.**

## Estimate

**S** for `40-00` — the fix is 4 lines and already verified; the fixture protocol and build guard are
small.
**M** for `40-01` — dominated by toolchain/runner setup, not by code.
