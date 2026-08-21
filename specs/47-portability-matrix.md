<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 47: Portability matrix — arch × OS verification

**Goal.** Establish, with evidence, what rawr actually supports across
{**aarch64**, **x86_64**} × {**Linux**, **macOS**, **Windows**, **FreeBSD**, **NetBSD**, **OpenBSD**} —
and record the result honestly rather than assuming.

**VM/host provisioning is out of scope** — owner-handled. This spec defines *what to verify and what
counts as passing*.

## 1. The finding that shapes this spec

**All OS-conditional code lives in the benchmark harness, not the shipped library.** Verified:
`bench_time.zig`, `bench_croaring.zig`, `bench_smp_layout.zig`, `bench_lazy_or_residency.zig`, and
`build.zig`'s `addBenchmarkPlatformShim` (which has an **OpenBSD** C shim and links libc). Nothing in
`bitmap.zig`, `container.zig`, `serialize.zig`, `roaring64.zig`, or the frozen types branches on
`builtin.os.tag`.

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
- **Linux / x86_64** (WSL2) — full suites, benches, board, plus native 32-bit `x86-linux-musl`.

That leaves **10 cells** genuinely unverified.

## 3. Tier 1 — compile matrix first (cheap, catches most)

**Generalize the existing `check-32` mechanism.** `tools/check_32_api.zig` is already an exported probe
covering the full public surface of all five stable types — it exists, it works, and spec 40-01
established that its enumerated surface *is* the guard boundary. Reuse it verbatim.

Add **`zig build check-portability`**: compile that probe for every target in the matrix.

- No execution, no host needed — this is pure cross-compilation and runs on any dev machine.
- Targets: `{aarch64, x86_64}` × `{linux-gnu, linux-musl, macos, windows-gnu, freebsd, netbsd, openbsd}`,
  plus whatever Zig 0.16 actually supports — **record which targets Zig cannot target at all**, since
  that is itself a finding.
- **This is the single highest-value step.** Most portability breakage is a compile error, and this finds
  it without a single VM.

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
- **BSDs** — **OpenBSD already has a bench shim and a custom `openbsd_c_allocator`
  (`bench_time.zig:427-442`)**, which means someone hit real problems there before. Whether that was
  Tier 1 or Tier 2 should be established, not guessed. FreeBSD and NetBSD are untried.
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
- Runtime set (§4) run on every host the owner provides; results recorded per cell.
- Evidence table complete in `docs/`, every cell carrying one of the §6 statuses.
- `README.md` support statement matches the table — **verified vs compiles distinguished**.
- Any **Tier 1** breakage either fixed or recorded as a known limitation with its error.
- **Tier 2 gaps recorded as testing gaps, never as user-facing unsupported platforms.**
- No regression on the two known-good cells; existing suites, `check-32`, `check-docs`, `check-package`
  all still green.

## 8. Out of scope

- VM/host provisioning (owner-handled).
- Making CRoaring build everywhere — Tier 2, and it is vendor code.
- Performance on new platforms. This spec is about **correctness and buildability**; the parity board
  remains M4 + Zen 4 only.
- New SIMD paths for any architecture.

## 9. Estimate

**S/M** — §3 is small and mostly mechanical; the work is the runtime passes and honest recording.
