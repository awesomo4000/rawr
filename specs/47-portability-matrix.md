<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 47: Portability matrix — arch × OS verification

**Goal.** Establish, with evidence, what rawr actually supports across
{**aarch64**, **x86_64**} × {**Linux**, **macOS**, **Windows**, **FreeBSD**, **NetBSD**, **OpenBSD**} —
and record the result honestly rather than assuming.

**VM/host provisioning is out of scope** — owner-handled. This spec defines *what to verify and what
counts as passing*.

## 1. The finding that shapes this spec

**The shipped library *sources* contain no OS-conditional code. The shipped `build.zig` does.**

An earlier version of this section said all OS-conditional code lives in the benchmark harness and that
`build.zig`'s `addBenchmarkPlatformShim` does not ship. **That contradicted this spec's own Tier 1 table**,
which correctly lists `build.zig` as shipped. Re-verified 08/31:

- **Sources: clean.** Every one of the **33** files in `build.zig.zon`'s `.paths`, **including the test
  files that ship**, is free of `builtin.os` and `os.tag`. *(An earlier check excluded test files; they
  are in `.paths`, so they are Tier 1 and had to be included. The answer did not change.)*
- **`build.zig`: three OS-conditional sites, and it ships.** `addBenchmarkPlatformShim` branches on
  **OpenBSD** (`:1297`) and adds `src/bench_openbsd.c` — **a file that is not in `.paths`**. Two further
  branches add a `bswap64` macro on **FreeBSD** (`:1319`, `:1345`) in the CRoaring translate-c setup.

**This reverses the priority this spec assigned to the BSDs.** They are not exotic low-stakes cells: on
the current evidence **OpenBSD and FreeBSD are the only OS values that flip a branch in shipped code**,
and one of those branches names a file a consumer would not receive. Whether the consumer path ever
reaches them is **unverified** — `addBenchmarkPlatformShim` appears to be invoked only from benchmark and
difftest modules, but "appears to be" is not the standard this campaign uses.

**`check-package` on OpenBSD and FreeBSD is the check that settles it**, because it builds and runs a
consumer from the allowlist alone: if the consumer path needed `src/bench_openbsd.c`, it would fail on a
missing file. Those two cells therefore rank **above** Windows for Tier 1 risk, even though Windows is the
larger unknown overall.

**Arch-conditional code is confined to `array_simd.zig`:** `has_x86_simd` requires `x86_64` + AVX,
`has_neon` requires `aarch64` + NEON. Everything else takes scalar paths — already proven by `check-32`
across four 32-bit targets.

**So portability splits into two tiers with very different stakes**, and conflating them would
overstate the problem:

| Tier | What it is | Ships to users? | Risk |
| --- | --- | --- | --- |
| **1 — the library** | the `.paths` allowlist: `src/*.zig` + `build.zig` | **yes** | std behaviour differences, arch SIMD gating |
| **2 — dev tooling** | benches, `difftest`, `validate_croaring`, fixtures | **no** | OS-specific clocks, CRoaring C build, libc linkage |

**A Tier 2 gap is not a user-facing portability defect.** It limits *our* ability to test on that
platform, which matters, but it must not be reported as "rawr does not support X".

## 2. Known-good baseline

Two cells are already continuously exercised by the parity campaign and need no new work:

- **macOS / aarch64** (M4) — full suites, benches, board.
- **Linux / x86_64 under WSL2** — full suites, benches, board, plus native 32-bit `x86-linux-musl`.

**WSL2 is not a native-Linux support claim.** It is a Linux kernel under a Windows host, and `52-00`
Part B demonstrated it producing a **2.47x different result for identical code** against a historical run.
That finding is about timing, not correctness, so this cell is genuine evidence for *buildability and
test-passing* — but a README line saying "Linux/x86_64 verified" would be resting on a virtualized
environment. **Record it as WSL2 specifically** until native Linux runs, which `52-00` Part A will supply.

That leaves **10 of the 12 runtime cells** (2 arches × 6 OS families) genuinely unverified, plus native
Linux/x86_64 as a qualified eleventh.

**Note the two counts differ and should not be conflated:** the §3 *compile* matrix has **16 cells**
(2 arches × 8 target triples, since Linux and Windows each have two ABIs), while the §4 *runtime* matrix
has **12** (2 arches × 6 OS families). The evidence table (§6) is keyed by compile target; runtime status
attaches to the OS family.

## 3. Tier 1 — compile matrix first (cheap, catches most)

**Generalize the existing `check-32` mechanism.** `tools/check_32_api.zig` is already an exported probe
covering the full public surface of all five stable types — it exists, it works, and spec 40-01
established that its enumerated surface *is* the guard boundary. Reuse it verbatim.

Add **`zig build check-portability`**: compile that probe for every target in the matrix.

- No execution, no host needed — this is pure cross-compilation and runs on any dev machine.
- Targets: `{aarch64, x86_64}` × `{linux-gnu, linux-musl, macos, windows-gnu, **windows-msvc**, freebsd,
  netbsd, openbsd}` — **record which of these Zig 0.16 cannot target at all**, since that is itself a
  finding. Include **both Windows ABIs**: `gnu` and `msvc` differ in libc and linking, and a consumer on
  Windows may be using either.
- **`check-portability` does NOT replace `check-32`.** They cover **different axes** — `check-32` is the
  *pointer-width* matrix (wasm32, x86, arm, riscv32), this is the *arch × OS* matrix at 64-bit. Merging
  them would quietly drop 32-bit coverage. Keep both steps.
- **This is the single highest-value step.** Most portability breakage is a compile error, and this finds
  it without a single VM.

### 3.1 Build options are part of the buildable surface

**Exercise every documented `-D` build option**, at minimum compiling with each of its values on the dev
host. There is currently exactly **one**: `-Dcroaring-avx512`.

That single option **did not compile** until `52-00` fixed it — the C amalgamation auto-detected the macro
while translate-c processed rawr's wrapper without seeing the definition. **100% of the documented build
option surface was broken, and nothing caught it**, because every check in the tree exercises default
values only. A documented option that does not build is a buildability defect of exactly the kind this
spec exists to find, and the cost of covering it is one extra compile per value.

### 3.2 Cover the feature-gated fallbacks

`array_simd.zig` gates on **features, not just architecture**: `has_x86_simd` requires `avx` **and**
`ssse3`, `has_neon` requires the NEON bit. A target lacking them takes scalar paths silently.

**Add baseline-feature cells** — `x86_64` with no AVX and `aarch64` with no NEON — and confirm they
compile and pass. This is cheap cross-compilation and it records a fact consumers need: **a generically
built rawr gets no SIMD at all**, which is the same compile-time-versus-runtime dispatch question raised
in [spec 52 §6](52-x86-64-parity.md). The matrix should state it rather than leave it implicit.

## 4. Tier 1 — runtime verification, per available host

For each host the owner can provide:

1. `zig build test`
2. `zig build test64`
3. **`zig build check-package`** — the allowlist-only consumer builds *and runs* (spec 41). This is the
   closest proxy for "a real user on this platform can use rawr".

`difftest` / `difftest64` are **Tier 2** — they link CRoaring. Run them where they work; their absence is
a testing gap, not a library defect.

## 5. Specific risks to check, not assume

- **Windows** — the largest unknown; no OS-conditional library code exists but nothing has ever run
  there. Check: `std.heap.smp_allocator` availability and behaviour, 64-byte `alignedAlloc`, and whether
  `check-package`'s generated consumer project builds under the Windows shell/path rules.
- **BSDs — now the top Tier 1 risk, per §1.** OpenBSD and FreeBSD are the only OS values that branch in
  the shipped `build.zig`, and the OpenBSD branch names `src/bench_openbsd.c`, which is not in `.paths`.
  Run `check-package` there first. Separately, OpenBSD has a bench shim and a custom
  `openbsd_c_allocator` (`bench_time.zig:427-442`): evidence of **prior platform-specific work**, but
  whether it was forced by a defect or chosen as a preference is **not established by its existence**.
  Establish which. NetBSD is untried and branches nowhere, so it is the lower-risk BSD.
- **There is an OpenBSD stash in the working tree.** Examine it before starting — it may already contain
  findings, and rediscovering them would be waste.
- **aarch64 on Linux/BSD/Windows** — NEON gating is on `builtin.cpu.arch == .aarch64` **plus** the NEON
  feature bit, so a target without the feature silently takes scalar paths. Confirm which path each
  aarch64 cell actually takes, and record it.
- **`bench_time.zig`** already special-cases Windows and OpenBSD; other BSDs fall to the POSIX path.
  Confirm rather than assume.

## 6. Deliverable — an evidence table, not a claim

For every cell, record one of:

| Status | Meaning |
| --- | --- |
| **verified** | compiles **and** the §4 runtime set passes on real hardware/VM |
| **compiles** | Tier 1 cross-compile passes; not executed |
| **tooling-gap** | library fine, Tier 2 (difftest/bench) unavailable — say *why* |
| **broken** | with the actual error |
| **not targetable** | Zig 0.16 cannot target it |

Record it in `docs/` (repo-only). **Then update `README.md`'s support statement to match the evidence** —
that is the point of the exercise. Today the README claims 32-bit targets it *does* verify via
`check-32`; the arch/OS matrix currently has no such statement, and it should not acquire an optimistic
one.

**Language discipline, per spec 41:** the README states *what is tested*, with no performance claims and
no implied guarantees for untested cells. "Compiles" and "verified" are different words and must stay
different.

## 7. Acceptance

- `zig build check-portability` added, reusing `tools/check_32_api.zig`, covering the matrix; targets Zig
  cannot build are **recorded, not silently skipped**.
- **Every documented `-D` option compiled with each of its values** (§3.1) — currently just
  `-Dcroaring-avx512`, whose broken state went unnoticed because nothing exercised non-default values.
- **Baseline-feature cells covered** (§3.2): `x86_64` without AVX and `aarch64` without NEON compile and
  pass, and the evidence table records that a generically built rawr takes scalar paths.
- **`check-package` run on OpenBSD and FreeBSD before other unverified cells**, since §1 shows they are
  the only OS values that branch in shipped code.
- Runtime set (§4) run on every host the owner provides; results recorded per cell.
- Evidence table complete in `docs/`, every cell carrying one of the §6 statuses.
- `README.md` support statement matches the table — **verified vs compiles distinguished**, and the
  Linux/x86_64 cell recorded as **WSL2** unless native Linux has run.
- Any **Tier 1** breakage either fixed or recorded as a known limitation with its error.
- **Tier 2 gaps recorded as testing gaps, never as user-facing unsupported platforms.**
- The two known-good cells still pass: suites, `check-32`, `check-docs`, `check-package` **green**.
  *(Green, not "no regression" — performance is out of scope per §8, so there is no timing claim here.)*
  `check-docs` and `check-package` are **host-local** checks, run on the dev machine; they are not part of
  the per-target matrix.

## 8. Out of scope

- VM/host provisioning (owner-handled).
- Making CRoaring build everywhere — Tier 2, and it is vendor code.
- Performance on new platforms. This spec is about **correctness and buildability**; the parity board
  remains M4 + Zen 4 only.
- New SIMD paths for any architecture.

## 9. Estimate

**S/M** — §3 is small and mostly mechanical; the work is the runtime passes and honest recording.
