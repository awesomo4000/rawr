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

- **Sources: clean.** All **26 `src/*.zig` entries** in `build.zig.zon`'s `.paths`, **including the test
  files that ship**, are free of `builtin.os` and `os.tag`. *(An earlier check excluded test files; they
  are in `.paths`, so they are Tier 1 and had to be included. The answer did not change.)* The `.paths`
  allowlist holds **33** entries in total — those 26 sources plus `build.zig`, `build.zig.zon`, and five
  docs/asset files. **Saying "all 33 are clean" contradicts the next bullet**, since `build.zig` is one of
  the 33 and is not clean. The sources are audited as a set; `build.zig` is audited separately.
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
missing file.

**Stated precisely:** call-site inspection confirms the OpenBSD and FreeBSD helpers attach only to
repository development modules, **never to `lib_mod`**. So they are **the highest known
package-integrity risk** — the shipped build graph declares them and a consumer resolves that graph —
**not** categorically above Windows for overall Tier 1 portability risk. Windows remains the larger
unknown. *(An earlier draft made the broader claim, which the call sites do not support.)*

**Explicit target-feature predicates live in `array_simd.zig`** — `has_x86_simd` requires `x86_64` + AVX
+ SSSE3, `has_neon` requires `aarch64` + NEON — and `array_kernels.zig` consumes them to select the
**array-intersection kernels**. *(An earlier draft added "everything else takes scalar paths". **False**,
and it contradicts §3.2: other code still uses portable `@Vector` with no feature gate, notably
`simdBitsetOp`.)* What the predicates gate is the array-intersection path, nothing wider. Scalar
correctness across targets is separately proven by `check-32` on four 32-bit targets.

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

That leaves **10 of the 12 host families** genuinely unverified, plus native Linux/x86_64 as a qualified
eleventh — and rather more than 10 *cells*, since the unverified families include ABI pairs (§6).

**The 12 is provisioning context only, not an evidence unit.** 2 arches × 6 OS families = **12 hosts to
provision**, but **evidence and `verified` status attach independently to all 16 ABI-specific target
cells** (§6), because Linux and Windows each carry two ABIs and running one never verifies its sibling.
The §3 compile matrix and the §6 evidence table are both keyed by **target triple**. *(An earlier draft
attached runtime status to the OS family, which §6 correctly forbids.)*

## 3. Tier 1 — compile matrix first (cheap, catches most)

**Generalize the existing `check-32` mechanism** — but not verbatim, and not alone.

**The probe does not cover all five stable types.** `tools/check_32_api.zig` never references
`OwnedBitmap`, so a portability defect confined to its six methods would pass unnoticed. Spec 40-01
established that the probe's **enumerated surface is the guard boundary**, which is exactly why the gap
matters: an unenumerated type is invisible, not implicitly covered. **Extend the probe to `OwnedBitmap`
and add a seeded failure control** proving the added surface actually trips the guard. Until then, this
spec may not describe the probe as covering the full public surface.

**Compiling the probe does not exercise the shipped `build.zig`.** `check-32` compiles source modules
directly and bypasses the dependency/package path — which is precisely where §1's OpenBSD and FreeBSD
branches live. A matrix built only on the probe would reprioritise the BSDs and then never test the thing
that made them the priority. **Add a per-target, allowlist-only package consumer that passes the target
into `b.dependency`**, so the shipped build script is executed for every cell. Native `check-package`
remains its runtime counterpart.

**Report every cell.** A single aggregate step cannot: one failing or non-targetable target stops it, and
the remaining cells go unrecorded while looking merely absent. **Require independently invocable
per-target substeps, or a controller that runs each cell and records pass, failure, or not-targetable
without silently skipping the rest.** The deliverable is an evidence table, and a table with unexplained
holes is the failure mode.

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

**Exercise every documented `-D` build option through a step that actually reaches the affected code.**
There is currently exactly **one**: `-Dcroaring-avx512`.

**A plain `zig build -Dcroaring-avx512=true` would pass with the defect present.** It builds the library
and need not instantiate translate-c, and translate-c was the broken path. The check must name a
**CRoaring-backed step** — `bench-parity-worker` or equivalent. **`true` also panics deliberately on
non-x86_64** (`build.zig:13`), so that value runs only on an x86_64 target while `false` runs everywhere.
Pin both commands exactly rather than describing them.

That single option **did not compile** until `52-00` fixed it — the C amalgamation auto-detected the macro
while translate-c processed rawr's wrapper without seeing the definition. **100% of the documented build
option surface was broken, and nothing caught it**, because every check in the tree exercises default
values only. A documented option that does not build is a buildability defect of exactly the kind this
spec exists to find, and the cost of covering it is one extra compile per value.

### 3.2 Cover the feature-gated fallbacks

`array_simd.zig` gates on **features, not just architecture**: `has_x86_simd` requires `avx` **and**
`ssse3`, `has_neon` requires the NEON bit. A target lacking them takes scalar paths silently.

**Add baseline-feature cells** — `x86_64` with no AVX and `aarch64` with no NEON — which **compile and
assert the expected dispatch**. They are cross-compile-only, so "pass" is not available to them and the
assertion is the deliverable.

**The fact to record is narrower than an earlier draft claimed.** That draft said a generically built rawr
"gets no SIMD at all". **False.** The gates select only the explicit array-intersection kernels
(`array_simd.zig` defines `has_x86_simd`/`has_neon`; `array_kernels.zig:145` consumes them). **Bitset
container operations are vectorized unconditionally** — `simdBitsetOp` (`bitset_container.zig:213`) uses
`@Vector(8, u64)` with no feature gate at all, and its lowering is LLVM's choice. The correct statement is
that a baseline target **takes the scalar array-intersection path**, and it connects to the
compile-time-versus-runtime dispatch question in [spec 52 §6](52-x86-64-parity.md).

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
- **The OpenBSD stash was examined and is superseded.** Its changes landed in `29c51d7`; it holds only
  Tier 2 benchmark fixes for `ReleaseFast` copying large arrays onto OpenBSD's 4 MB stack. **No library or
  package finding.** Recorded so it is not re-inspected.
- **aarch64 on Linux/BSD/Windows** — NEON gating is on `builtin.cpu.arch == .aarch64` **plus** the NEON
  feature bit, so a target without the feature silently takes scalar paths. Confirm which path each
  aarch64 cell actually takes, and record it.
- **`bench_time.zig`** already special-cases Windows and OpenBSD; other BSDs fall to the POSIX path.
  Confirm rather than assume.

## 6. Deliverable — an evidence table, not a claim

For every cell, record **one status in each tier** — the two are independent, not alternatives:

**Two independent columns, because the tiers are not alternatives.** An earlier draft required exactly
one status per cell, which is wrong: a target can be Tier 1 **verified** *and* carry a Tier 2 tooling gap
at the same time, and forcing a choice would either hide a working library or hide a testing hole.

**Tier 1 — the shipped library:**

| Status | Meaning |
| --- | --- |
| **verified** | compiles **and** the §4 runtime set passes on real hardware/VM |
| **compiles** | Tier 1 cross-compile passes; not executed |
| **broken** | with the actual error |
| **not targetable** | Zig 0.16 cannot resolve or build a **minimal control program** for the target. See [`47-01` §1](47-01-compile-matrix.md) — a probe or package-consumer failure is `broken`, even when the diagnostic reads like a target limitation, so a rawr defect cannot be filed here |

**Tier 2 — dev tooling:**

| Status | Meaning |
| --- | --- |
| **passes** | **`zig build difftest` and `zig build difftest64`** both run and pass on this cell |
| **gap** | unavailable — **say why** |
| **not-run** | not attempted |

**Status attaches to a target triple, never to an OS family.** §3 distinguishes `linux-gnu` from
`linux-musl` and `windows-gnu` from `windows-msvc` at compile time, so runtime status must keep that
distinction: **running Windows GNU does not make Windows MSVC `verified`**, and Linux GNU says nothing
about musl. An executed cell is `verified`; its unexecuted ABI siblings stay `compiles`. *(An earlier
draft collapsed runtime status to the OS family, which would have promoted untested ABIs for free.)*

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
- **Probe extended to `OwnedBitmap` with a seeded failure control** proving the added surface trips the
  guard (§3).
- **Per-target allowlist-only package consumer** exercising the shipped `build.zig` for every cell, and
  **every cell reported** — pass, failure, or not-targetable — with no silent skipping (§3).
- **Every documented `-D` option exercised through a CRoaring-backed step with each of its values**
  (§3.1), `true` on x86_64 only, both commands pinned. Its broken state went unnoticed because nothing
  exercised non-default values.
- **Baseline-feature cells covered** (§3.2): `x86_64` without AVX and `aarch64` without NEON **compile and
  assert the expected dispatch**, and the table records that they take the **scalar array-intersection
  path** — not that they lack SIMD, since bitset operations stay vectorized.
- **Every status recorded per target triple**, with ABI siblings never promoted by association (§6).
- **`check-package` run on OpenBSD and FreeBSD before other unverified cells**, since §1 shows they are
  the only OS values that branch in shipped code.
- Runtime set (§4) run on every host the owner provides; results recorded per cell.
- **Two tables complete in `docs/`** ([`47-02` §2](47-02-runtime-and-evidence.md)): the 16 target-triple
  cells each carrying **both** a Tier 1 and a Tier 2 status, and a separate **feature-dispatch table keyed
  by target triple + CPU profile** for the baseline cells. **No cell forced to a single status**, and no
  baseline cell folded into the triple-keyed table.
- `README.md` support statement matches the table — **verified vs compiles distinguished**, and the
  Linux/x86_64 cell recorded as **WSL2** unless native Linux has run.
- Any **Tier 1** breakage either fixed or recorded as a known limitation with its error.
- **Tier 2 gaps recorded as testing gaps, never as user-facing unsupported platforms.**
- The two known-good cells still pass: suites, `check-32`, `check-docs`, `check-package` **green**.
  *(Green, not "no regression" — performance is out of scope per §8, so there is no timing claim here.)*
  `check-docs` and `check-package` are **host-local** checks, run on the dev machine; they are not part of
  the per-target matrix.

## 7.1 Chunking

- **[47-00](47-00-portability-machinery.md)** — build the checks and **exercise all five falsification
  controls**. The toplevel review found four proposed guards that would pass with their defect present, so
  the controls are the deliverable as much as the checks are. No matrix run.
- **[47-01](47-01-compile-matrix.md)** — run the 16 target cells plus the two baseline-feature cells, BSD
  and Windows first. Breakage **fixed or recorded, labelled which**. No cell called `verified`.
- **[47-02](47-02-runtime-and-evidence.md)** — runtime cells on whatever hosts exist, the evidence table
  keyed by **target triple**, and the README. **Unprovisioned cells stay `compiles` and the chunk
  completes** — this spec makes useful progress without every host.

**`52-00` Part A is not a dependency.** Its only interaction is upgrading the Linux/x86_64 cell from WSL2
to native when it runs, which is a one-line change in `47-02`.

## 8. Out of scope

- VM/host provisioning (owner-handled).
- Making CRoaring build everywhere — Tier 2, and it is vendor code.
- Performance on new platforms. This spec is about **correctness and buildability**; the parity board
  remains M4 + Zen 4 only.
- New SIMD paths for any architecture.

## 9. Estimate

**S/M** — §3 is small and mostly mechanical; the work is the runtime passes and honest recording.
