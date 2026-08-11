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

Zig is present but **not on `PATH`** — use the explicit path:

```sh
/home/alr/.zvm/0.16.0/zig build test       -Dtarget=x86-linux-musl
/home/alr/.zvm/0.16.0/zig build difftest   -Dtarget=x86-linux-musl
/home/alr/.zvm/0.16.0/zig build test64     -Dtarget=x86-linux-musl
/home/alr/.zvm/0.16.0/zig build difftest64 -Dtarget=x86-linux-musl
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
  static `x86-linux-musl`, no emulator, at the explicit Zig path.
- **Bidirectional cross-width round-trip** passes for both bitmap types: byte-identity + set equality;
  corpus hash agrees across widths.
- Address-space limitation documented; supported targets and commands recorded.
- 64-bit unaffected — board unchanged, `zig build test` / `difftest` / `ReleaseSafe` / `ReleaseFast`
  green.

## Estimate

**M** — **not** dominated by runner setup (native execution needs no install). The work is **runtime
debugging of whatever 32-bit failures the four suites surface** and the **bidirectional fixture
exchange**, including the Roaring64 corpus.
