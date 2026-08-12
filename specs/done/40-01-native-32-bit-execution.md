<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 40-01: Native 32-bit execution and cross-width exchange

Toplevel: [40-32-bit-support.md](40-32-bit-support.md). Gated on: [40-00](40-00-width-generic-tagged-ptr.md).

**Not blocked on tooling.** Runs as soon as `40-00`'s `TaggedPtr` fix lands — **no QEMU, no install, no
distro, no rootfs**.

## Runner — native, no emulator

**Verified on the actual Zen 4 / WSL2 host:** the kernel runs static i386 binaries directly
(`ELF 32-bit LSB executable, Intel 80386, statically linked` → `exit=0`).

**Do NOT pass `-fqemu`.** Zig 0.16 treats `x86` as **natively executable on an `x86_64` Linux host** — its
runner returns `.native` **before** considering QEMU, so `-fqemu` does not force `qemu-i386` for this
host/target pair. If native execution ever proves insufficient, QEMU must be wired **explicitly** via a
custom build step prefixing the artifact with `qemu-i386`; the flag will not do it.

Zig 0.16.0 is present but **not on `PATH`** on the pinned host; resolve its installation path before
running these commands:

```sh
zig build test       -Dtarget=x86-linux-musl
zig build difftest   -Dtarget=x86-linux-musl
zig build test64     -Dtarget=x86-linux-musl
zig build difftest64 -Dtarget=x86-linux-musl
```

**All four must pass.** `test64` and `difftest64` are **separate build steps** — `difftest` does **not**
cover `Roaring64Bitmap`, and 64-bit-value support is part of the claim. **Static musl is load-bearing:**
do not switch to `-gnu`, which would need a 32-bit loader and libc and reintroduce sysroot/multilib.

**If any suite cannot be made to run, report it and re-pick the runner — do NOT silently reduce
acceptance to unit tests only.** The differential steps are why the vehicle had to link C at all.

**Expected, not a defect:** `vendor/roaring.h` sets `CROARING_IS_X64` only under `__x86_64__`/`_M_X64`,
neither defined here, so CRoaring takes its **portable C path by design**. That path is less travelled
than the x64 one — which is exactly why the differential steps, not unit tests alone, are required.

**Fallback if native proves insufficient:** `wasm32-wasi` + `wasmtime` (Zig compiles C to wasi, so
`difftest` is not automatically disqualified), or explicit QEMU as above.

## Cross-width exchange — both directions

Using `40-00`'s corpus and protocol:

- **64-bit producer → 32-bit consumer**, and **32-bit producer → 64-bit consumer**.
- Assert **byte-identity** of serialized output and **set equality** after deserialization.
- Covers **both** `RoaringBitmap` and `Roaring64Bitmap` (the latter spanning multiple high-32-bit
  buckets).
- Confirm the **checked-in corpus hash matches on both widths** — a mismatch means width-dependent
  *generation*, not a format defect, and must be diagnosed as such.

## Documentation

- **Address-space limitation.** A 32-bit process has ~2–4 GB usable; a worst-case dense bitmap is
  65,536 containers × 8 KB = **512 MB of payload** before overhead. Large bitmaps are feasible but
  **allocation failure is far more likely** than on 64-bit. Document as a known limitation — **do not
  present 32-bit as equivalent**.
- Record the supported/tested 32-bit targets and the exact commands above.

## Acceptance

- **All four suites — `test`, `difftest`, `test64`, `difftest64` — execute and pass** natively under
  static `x86-linux-musl`, with no emulator.
- **Bidirectional cross-width round-trip** passes for both bitmap types: byte-identity + set equality;
  corpus hash agrees across widths.
- Address-space limitation documented; supported targets and commands recorded.
- 64-bit unaffected — board unchanged, and **all four 64-bit suites green: `test`, `difftest`, `test64`,
  `difftest64`** (runtime-discovered fixes in this chunk may touch `Roaring64Bitmap` code), plus
  `ReleaseSafe` and `ReleaseFast`.

## Verification record — implemented, reviewed, ACCEPTED

All four suites pass natively under static `x86-linux-musl` on the Zen 4 / WSL2 host, no emulator, as
specified. Cross-width exchange passed in both directions.

**Independently reproduced, not accepted on report:** building the fixture tool on an aarch64 host and
running `produce` yields SHA-256
`813ba1bc467ff67ef6849357f9ac5c7b88d049b42aed8460ceb85f26cbadb171` — **byte-identical to the i386
fixture**, on a third architecture neither side of the original exchange used. `verify` also passes.
This is the strongest available evidence that serialization is genuinely width- and
endianness-neutral rather than coincidentally matching between two hosts.

### Runtime findings — the guard was sound, the *surface list* was not

Execution surfaced five defects `40-00` did not: four whole-struct `@as(u64, @bitCast(tp))` `TaggedPtr`
identity comparisons (`bitmap.zig` ×4, `bench_consuming_or.zig` ×1, now `TaggedPtr.eql`) and one 64-bit-only
`RunContainer` size assertion (now `@sizeOf(usize) + 8`, still pinning 16 bytes on 64-bit and preserving the
spec-32 header result).

Two controls establish where the gap actually was:

| Control | Result |
| --- | --- |
| **E — revert one `@bitCast` compare, current probe** | **check-32 FAILS**: `@bitCast size mismatch: destination type 'u64' has 64 bits but source type 'container.TaggedPtr' has 32 bits`. |
| **F — same defect, `40-00`-era probe (in-place ops removed)** | **compiles clean, exit 0.** |

**So `check-32`'s coverage is exactly the surface the probe enumerates, and nothing more.** The mechanism
was never weak; the §3 list in `40-00` omitted the in-place operations, so four real defects sat in
unguarded code. The probe now covers `bitwiseOrInPlace`, `bitwiseOrInPlaceConsume`,
`bitwiseDifferenceInPlace`, and `bitwiseXorInPlace`.

**Standing maintenance rule, adopted:** *adding public API means adding it to the probe.* An unlisted
function is unguarded on 32-bit, and the failure mode is silence — exactly the lazy-analysis behaviour
control C in `40-00` demonstrated.

**Second audit-command miss, recorded honestly.** `40-00`'s acceptance command
(`rg -n '\.addr\b|@ptrFromInt|rawAddr\('`) **could not have found these** — they never touch `.addr`, they
`@bitCast` the whole struct. That command had already been corrected once, for a pattern that matched
nothing. Both misses share a cause: a hand-written pattern encodes what its author expected the defect to
look like. The compile matrix, which requires no such guess, found all four. **Weight the executable guard
over the text search**; the audit remains useful for duplication, not for portability.

## Estimate

**M** — **not** dominated by runner setup (native execution needs no install). The work is **runtime
debugging of whatever 32-bit failures the four suites surface** and the **bidirectional fixture
exchange**, including the Roaring64 corpus.
