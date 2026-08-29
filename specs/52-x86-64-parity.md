<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 52: The x86_64 gap

**Target.** The canonical SMP board splits by architecture. On the clean-`b3ab49f` boards of 08/28,
**aarch64 (M4) has 5 of 27 SMP rows** over the 1.10x gate and **x86_64 (Zen 4) has 13 of 27**, several
near or above 2.7x. On the candidate boards the counts are 3 and 13.

**Diagnosis first. No kernel written until attribution earns it.** An earlier draft of this spec proposed
a vector-width explanation. §1 records why that was wrong, because the way it was wrong is instructive.

## 1. What an earlier draft got wrong

### 1.1 rawr is not uniformly 128-bit

The first draft claimed every production vector in rawr is 128-bit. **False.** `bitset_container.zig`,
`container_ops.zig` and `bitmap.zig` all use `@Vector(VEC_SIZE, u64)` with `VEC_SIZE = 8` — a **512-bit
logical vector**. Only `array_simd.zig` is 128-bit.

The claim came from a search for `@Vector\([0-9]+`, which **cannot match a named constant** and so could
not find the wide vectors it was used to rule out. This is the same failure this campaign has now hit
three times: spec 51's original hypothesis found a symbol without checking reachability, and spec 47's
first grep excluded the files that would have contradicted it. **A search that cannot express the defect
is not evidence of its absence.**

**More importantly, source width does not determine emitted width.** LLVM legalizes a 512-bit logical
vector however it chooses for the target: it may split it into narrower vector operations, scalarize it
entirely, or select a different sequence altogether — vector `@popCount` is a particular case where the
lowering is not obvious. **Predicting an instruction count from the source width is the same error in a
smaller form.** Only production disassembly is authoritative, and this spec may not restate a width claim
without one.

### 1.2 CRoaring is not scalar on aarch64

The first draft said CRoaring runs scalar on aarch64. **True only for array intersection.** Its bitset
container operations use NEON under `CROARING_USENEON` (`roaring.c:7990`). So the comfortable story —
rawr vectorized against an unvectorized reference on aarch64 — is wrong for the bitset cluster, which is
most of the failing rows.

**Architecture claims must be made per cluster and backed by disassembly**, never as one statement about
"CRoaring on ARM".

### 1.3 The array-intersection framing overstated the evidence

Both kernels are 128-bit and both use a 256-entry shuffle-mask table. CRoaring uses `_mm_cmpestrm` **and
`_mm_cmpistrm`**, and its dispatch is AVX2-gated even though the instructions are 128-bit. rawr's
`compareAny` (`array_simd.zig:33`) unrolls eight lane-splat compares and eight ORs.

That is a real difference, but **comparing the two shipped kernels cannot isolate it** — the surrounding
loops differ too. Attribution needs a same-loop alternative arm, or disassembly plus instruction
accounting.

**Spec 15 was parked, not executed.** It establishes that `vpshufb` cannot compact across 128-bit lanes,
which rules out **one** AVX2 compaction strategy. It does not establish that every AVX2 approach is a
dead end, and this spec may not claim that it does.

## 2. The allocator evidence, which reframes the whole target

Most board rows run under **both** allocators (not all — see §4 for the variants that do not pair). The
first draft never cross-referenced them. Doing so splits the failing rows immediately:

| Zen 4 clean-HEAD row | SMP | libc |
| --- | ---: | ---: |
| `toArrayAlloc (1M values)` | **2.914x** | **1.043x** |
| `serialize` | **2.771x** | **0.805x** |

**The two largest gaps are allocator-localized.** Same operation, same code, different result allocator.
That points at SMP allocation or allocator conditioning on this host — **not** at conversion or
serialization code, and not at vector width. The cluster the first draft labelled "unknown mechanism"
has a strong candidate, and it is not a kernel.

### 2.1 The spec 28 `serialize` discrepancy is open, not resolved

A previous version of this section claimed spec 28 had measured the libc row and that the contradiction
dissolved. **That is wrong.** Spec 28 measured **Zen 4 SMP**: `1.035 → 0.824 ms`, reaching **0.81x** with
rawr ahead. Today's libc figure of `0.805x` is a **coincidence**, and treating a numeric near-match as an
identification was exactly the reasoning error this campaign keeps having to correct.

So the position is: **Zen 4 SMP `serialize` was 0.81x and is now 2.771x, and production serialization has
not changed since `2ba714a`.** That is a real discrepancy — cross-harness, board-definition, or
allocator-state — and it is one of the strongest signals on the board precisely because the code is
constant. **Do not report that nothing regressed.** `52-00` must reconcile it.

This does not weaken §2's split. `toArrayAlloc` and `serialize` are still allocator-localized *today*.
But `serialize` additionally has a history in which SMP was fine, so it is not a static allocator story.

**One row runs the other way and needs its own account:** `bitwiseAnd (array balanced)` is **2.688x under
SMP and 9.138x under libc**. Whatever is happening there, "libc exonerates the code" does not apply.

**A second localization comes free from the board.** Of four intersection rows, only the balanced array
case fails:

| Zen 4 | SMP | libc |
| --- | ---: | ---: |
| `bitwiseAnd (sparse)` | 0.601x | 1.391x |
| `bitwiseAnd (array skewed)` | 0.663x | 0.964x |
| `bitwiseAnd (dense)` | 1.691x | 1.668x |
| `bitwiseAnd (array balanced)` | **2.688x** | **9.138x** |

**Under SMP** rawr is faster on both the sparse and skewed rows, where CRoaring falls back to galloping
(the sparse libc figure is 1.391x, so this is an SMP statement, not a general one). The gap is specific to the
**balanced path where the vector kernel actually runs**.

## 3. Complete inventory — all 13 Zen 4 SMP rows over the gate

Clean-`b3ab49f` board, `parity-20260828-134456`.

| row | SMP | libc | classification |
| --- | ---: | ---: | --- |
| `toArrayAlloc (1M values)` | 2.914 | 1.043 | **allocator-localized** — new attribution |
| `serialize` | 2.771 | 0.805 | **allocator-localized today**, but SMP was **0.81x** in spec 28 with the code unchanged — **open discrepancy**, see §2.1 |
| `bitwiseAnd (array balanced)` | 2.688 | 9.138 | **new attribution** — balanced vector path, libc worse |
| `lazyOr repair (sparse)` | 1.753 | 1.338 | **previously diagnosed** (38/39); opt-in remedy exists |
| `bitwiseAnd (dense)` | 1.691 | 1.668 | new attribution — bitset word loop |
| `clone (dense)` | 1.633 | 1.617 | **previously diagnosed** (27) — analysed residual on M4, open here |
| `addRange (1M)` | 1.414 | 1.526 | new attribution |
| `flip wide range (dense)` | 1.381 | 1.532 | new attribution |
| `bitwiseOr (dense)` | 1.322 | 1.404 | new attribution — bitset word loop |
| `lazyOr+repair (pre-adoption baseline)` | 1.293 | 3.927 | **reference variant, not a shipped row** |
| `addMany (sequential 1M)` | 1.220 | 1.160 | new attribution |
| `lazyOr+repair (sparse)` | 1.211 | 1.199 | **previously diagnosed** (39-01); opt-in remedy exists |
| `lazyOr+repair (sparse, descending frees)` | 1.114 | — | **this is the opt-in remedy row** |

**The unexplained set is far smaller than thirteen.** One row is a baseline reference variant, three are
already diagnosed with an opt-in remedy shipped, and one is a documented analysed residual. **Do not
relabel known allocator, clone and repair findings as generic SIMD-width problems** — that is precisely
what the first draft's framing would have done.

`add (sequential 1M)` sits on the gate at 1.097 clean / 1.112 candidate and is listed here only so its
borderline status is on the record.

## 4. Stage 0: establish whether the host is representative

The x86_64 host runs **WSL2**. Spec 36 already returned inconclusive on it. Given §2, the two largest
rows are allocator-shaped, and allocator behaviour is exactly what a different kernel and page-management
path would change.

**Run the canonical board on native Linux x86_64 on the same physical machine.**

**Honest scope of the comparison.** Native boot does **not** change one variable. It preserves the CPU
and changes the kernel, libc and userland, the scheduler, page management, and possibly power and
frequency configuration. It is the *closest available* comparison, not a clean one, and the record must
say so. A cloud instance changes the hardware as well; Windows additionally changes CRoaring's C
toolchain and needs MSYS2 or Git Bash inside the pipeline that produces the numbers, and rawr's Windows
support is unverified because [spec 47](47-portability-matrix.md) was never run.

**Pin the configuration, or the comparison means nothing:**

- source commit, Zig version, optimization mode (`ReleaseFast`), and CPU configuration (`-Dcpu=native`);
- the spec 22 process protocol: fresh process per cell, warmup then timed, **≥5 process medians with full
  ranges**;
- **every existing manifest variant**, with the SMP/libc comparison reported **where both exist**. The
  board does not pair them universally and this spec must not pretend otherwise:
  `lazy-or-repair-descending` is **SMP-only**; the default/non-allocating and arena rows have no SMP/libc
  pair; the pre-adoption rows carry rawr SMP and libc but reference **another row's** CRoaring tuple, so
  their libc ratio is not a like-for-like comparison. A libc descending-free diagnostic, if wanted, is a
  **benchmark-only tuple to be added**, not something the canonical board already contains.
- **CRoaring's dispatch reported per row, by the weaker of the two available methods, and labelled as
  such.** The canonical worker currently reports only the compile-time AVX-512 setting, so `52-00` will
  report `croaring_hardware_support()` once per host and **map each row to its source-gated expected
  path**. That is sufficient here, because the purpose is to confirm the reference did not change branch
  between two environments running the same binary. **It is not branch observation and may not be
  described as one.** Instrumented benchmark-only C wrappers that report the branch actually taken —
  which `51-00` built and proved out — belong in Stage 1, where the branch is the thing under study.

**Verdict rule, per row, pre-registered:**

| condition | verdict |
| --- | --- |
| `rawr_min / croaring_max > 1.10` | **gap survives** |
| `rawr_max / croaring_min <= 1.10` | **gap closes** |
| otherwise | **inconclusive** — rerun once, then report as inconclusive |

Applied to the **15 unique rows**: all 13 of §3 plus the `bitwiseAnd (sparse)` and `bitwiseAnd (array
skewed)` controls (dense and balanced intersection are already among the 13), **under each allocator
variant that exists for that row**. "Collapse" is not a
verdict; this table is.

**Stage 0 decides what the campaign is about.** If the allocator-localized rows close on native Linux,
those gaps are **environment-conditioned** and the campaign is mostly about kernels. **That is the only
claim the experiment supports.** It cannot isolate WSL2 from the kernel, libc, the scheduler, or page
behaviour, because native boot changes all of them together. "Host artifact" names a cause the design
cannot identify and must not be used.

## 5. What must not regress — named rows, not categories

**Owner constraint: intersection must not regress.** Stated as exact rows, on **both hosts** and under
**each allocator variant that exists for the row**:

- `bitwiseAnd (sparse)`
- `bitwiseAnd (dense)`
- `bitwiseAnd (array balanced)`
- `bitwiseAnd (array skewed)`

`array skewed` and `sparse` matter most here: rawr is currently **faster** than CRoaring on both, so they
are the rows a specialized balanced-path kernel could most easily damage.

**aarch64 is a negative control and must not move.** The risk is not the kernel — per-arch work is
comptime-dispatched and leaves the aarch64 path alone. The risks are the two this campaign has already
been bitten by:

- **Whole-binary layout movement** (spec 28). Every aarch64 comparison uses the **paired exact-HEAD
  protocol** from `51-02`: clean-HEAD and candidate boards in the same session on the same machine.
  Separated ranges across different runs are not evidence of a code effect — that produced a phantom 6.2%
  regression in `51-02` which was then withdrawn.
- **Shared structure changes** (spec 32's Array-header NO-GO). Nothing here may change a container layout
  or shared struct to serve x86_64.

**Gate every host and every row, never an aggregate** — spec 35's dual stop-gate.

## 6. The dispatch fork — settle before any kernel

rawr uses **compile-time** feature detection (`builtin.cpu.features`); CRoaring uses **runtime** dispatch.
Under `-Dcpu=native` the board is fair either way, but **a library distributed for generic x86_64 gets
nothing from a compile-time-only specialized path**.

This is a distribution decision, not a performance one, and it shapes every kernel written afterwards.
Note that `_mm_cmpestrm` and `_mm_cmpistrm` are **SSE4.2**, above rawr's current `avx + ssse3` floor, so
even the array-row candidate raises it.

## 7. Out of scope

- **Adding a 32-bit limitation.** rawr ships 32-bit support (spec 40).
- **Any aarch64 kernel change.** aarch64 is **outside this x86_64-focused campaign** and appears only as
  a negative control. It is not claimed to be at parity — the clean board has five SMP rows over the
  gate, which is separate work.
- **Re-deriving the lazy-OR repair rows.** Specs 38 and 39 diagnosed them and shipped an opt-in remedy;
  this campaign records their status rather than reopening them.

## 8. Chunking

- **[`52-00`](52-00-host-validation.md) — Stage 0, host validation.** Two parts. **Part A**: the canonical
  board on native Linux x86_64 *and* WSL2, same machine, same commit, **both measured fresh** — the 08/28
  boards may not be reused, since assuming this environment has not drifted is assuming the answer.
  **Part B**: reconcile the §2.1 `serialize` discrepancy by running `2ba714a` and `HEAD` in the current
  environment. **Part B needs no new hardware and can start immediately.** No production change.
- **Stage 1 attribution is deliberately unwritten.** Its arms depend on which rows survive Stage 0 and on
  whether the allocator-localized cluster is real. Writing them now would be guessing — the same reason
  `51`'s Stage 2 stayed unwritten until `51-00` reported.
