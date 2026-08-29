<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 52: The x86_64 gap

**Target.** The canonical SMP board splits sharply by architecture. On aarch64 (M4), **3 of 27 rows**
exceed the 1.10x gate, and two of those three are `pre-adoption baseline` reference variants — the only
shipped row over the gate is `lazyOr construction (sparse)` at 1.223x, a residual specs 43–46 accepted
deliberately. On x86_64 (Zen 4), **13 of 27 rows** exceed it, several near or above 2.7x.

**Diagnosis first. No kernel written until attribution earns it.** This is the discipline spec 51
established, and §1 shows the obvious explanation is wrong for the most important row.

## 1. Three clusters, three candidate mechanisms, none established

| Zen 4 SMP row | rawr | CRoaring | ratio |
| --- | ---: | ---: | ---: |
| `toArrayAlloc (1M values)` | 3.278 [3.237, 3.375] ms | 1.125 [1.073, 1.184] | **2.914x** |
| `serialize` | 2.004 [1.975, 2.043] ms | 0.723 [0.699, 0.762] | **2.771x** |
| `bitwiseAnd (array balanced)` | 152,770 ns/op | 56,833 | **2.688x** |
| `lazyOr repair (sparse)` | 15.449 ms | 8.716 | 1.772x |
| `bitwiseAnd (dense)` | 230.187 ns/op | 136.126 | 1.691x |
| `clone (dense)` | — | — | 1.593x |
| `addRange (1M)` | — | — | 1.426x |
| `flip wide range (dense)` | — | — | 1.381x |
| `bitwiseOr (dense)` | — | — | 1.353x |

Ranges are non-overlapping and the same rows appear at the same magnitudes on a clean-HEAD board, so
these are resolved gaps rather than run noise or anything spec 51-02 introduced.

### 1.1 What the source establishes

**Every vector in rawr's production code is 128-bit** — `@Vector(8, u16)` and `@Vector(16, u8)`, nothing
wider anywhere. rawr's x86 gate requires `avx` and `ssse3` and **never requests AVX2**
(`array_simd.zig:6`). CRoaring carries **70 AVX2-gated sites**, all behind `CROARING_IS_X64`, which is
defined only for `__x86_64__` / `_M_X64`.

So on aarch64, 128 bits is the full NEON width and CRoaring runs scalar — rawr competes at the hardware
ceiling against an unvectorized reference. On x86_64 the same rawr code meets a reference using twice the
width in 70 places. **That is a structural fact about the two sources. It is not a measured cause of any
row**, and naming a mechanism is not the same as measuring it.

### 1.2 Where that story does not hold — the array rows

**CRoaring's array intersect is also 128-bit.** `intersect_vector16` (`roaring.c:822`) uses `__m128i`
throughout: `_mm_cmpestrm` for the all-pairs compare, then `_mm_shuffle_epi8` against a 256-entry shuffle
table. rawr's `intersectWriteSimd` uses **the same width and the same shuffle-table algorithm**, with one
difference — `compareAny` (`array_simd.zig:33`) unrolls **eight lane-splat compares and eight ORs** where
CRoaring issues **one `_mm_cmpestrm`**.

So `bitwiseAnd (array balanced)` at 2.688x is **not a width gap**. Both sides are 128-bit. The candidate
is instruction selection at equal width, which is the same shape spec 51 found in the scalar merge:
portable code expressing in many instructions what a specialized one does in one.

**Going wider is not the fix here, and spec 15 already established why.** `vpshufb` cannot compact across
128-bit lanes, so AVX2 is a dead end for intersection compaction; the enabler would be AVX-512
`vpcompressw`. **Do not propose AVX2 for the array rows.**

### 1.3 The bulk-memory rows are a third thing

`toArrayAlloc` and `serialize` are bulk extraction and bulk write. Neither is obviously a vector-width
problem, and one observation actively contradicts a general "x86_64 is slow" story: **`deserialize` runs
at 0.4621x — rawr nearly 2.2x *faster* than CRoaring on the same host and the same board.** A mechanism
that made rawr broadly slow on x86_64 would not produce that.

**One contradiction must be resolved rather than carried.** Spec 28 recorded Zen 4 `serialize` at
**0.81x**. It now reads **2.771x**. Either the row regressed since and nobody noticed, the row definition
moved, or spec 28's figure did not hold. Any of those changes what this campaign is chasing.

### 1.4 Summary of what Stage 1 must tell apart

| cluster | rows | candidate mechanism | status |
| --- | --- | --- | --- |
| **array kernels** | `bitwiseAnd (array balanced)` | instruction selection at equal 128-bit width | candidate |
| **dense / bitset word loops** | dense AND/OR, clone, flip, addRange | rawr 128-bit against CRoaring AVX2 | candidate |
| **bulk memory** | `toArrayAlloc`, `serialize` | unknown — not obviously vector width | **no candidate** |

Three clusters may have three causes. **They must not be treated as one phenomenon**, which is the error
spec 51 §1.3 caught for OR and ANDNOT.

## 2. Stage 0: establish whether the host is representative

**The x86_64 host runs WSL2.** Spec 36 already returned inconclusive on it once, and the two largest rows
are bulk-memory shaped — exactly where a virtualization layer and a different page-management path would
show. Nothing in §1 can be trusted as a statement about x86_64 until this is settled.

**Run the canonical board on native Linux x86_64 on the same physical machine** — dual boot or live USB,
not a cloud instance and not Windows. The reasoning is experimental, not preferential:

- **Same machine changes exactly one variable.** A cloud instance changes hardware and OS together, so a
  difference cannot be attributed to either. Windows changes OS, userland, allocator page behaviour and
  CRoaring's C toolchain at once, and a null result there would not distinguish WSL2 from
  Windows-versus-Linux.
- **The harness runs unchanged.** It is bash, awk, sort, curl and unzip. Windows would need MSYS2 or Git
  Bash, placing ported `sort` and `awk` inside the pipeline that produces the canonical numbers — a
  compatibility layer added to remove a virtualization layer.
- **Linux x86_64 is the deployment target**, so the numbers mean something after the experiment.

**Windows is not excluded from the project** — it is a Tier 1 target in [spec 47](47-portability-matrix.md)
and rawr's support there is currently **unverified**, since 47 was never run. That is portability
validation and it is a different job from establishing a measurement baseline. Do not conflate them.

**Stage 0 decides whether Stage 1 happens at all.** If `serialize` and `toArrayAlloc` collapse on native
Linux, two clusters disappear and the campaign is only about kernels.

## 3. Stage 1: attribute before building

Same structure that worked in `51-00`: layers chosen so subtraction is valid, arms whose meaning is
checked directly rather than assumed, and per-row verdicts rather than an aggregate.

**Per cluster, the question differs**, so one arm set will not serve all three. At minimum Stage 1 must
separate, for each affected row:

- **kernel time** from **allocation and assembly**, as `51-00` did with its Layer A and Layer B split;
- for the array rows, **rawr's `compareAny` against CRoaring's `_mm_cmpestrm`** at equal width, replaying
  the shipped kernels;
- for the bulk-memory rows, **whether any vector code is on the path at all** before assuming width or
  instruction selection is involved.

**`bitwiseAnd (array balanced)` is the priority row** and also the cheapest probe: rawr already has a SIMD
kernel there, so a gap cannot be explained by a missing one.

## 4. What must not regress

**`bitwiseAnd` is the protected row.** Owner constraint: an x86_64 change that regresses it is not worth
having.

**aarch64 must not move.** The mechanism is structural — width and instruction selection are comptime
arch dispatches, and the aarch64 path keeps `@Vector(8, u16)` and its existing NEON kernels untouched.
The real risk is not the kernel:

- **Whole-binary layout movement.** Spec 28 established that adding code moves untouched rows with
  instruction-identical bodies. Every aarch64 board comparison uses the **paired exact-HEAD protocol**
  recorded in `51-02` — clean-HEAD and candidate boards run in the same session on the same machine.
  Separated ranges across different runs are not evidence of a code effect, which is how a 6.2% phantom
  regression was reported and then withdrawn in `51-02`.
- **Shared structure changes.** Spec 32's Array-header NO-GO helped one path and hurt another. Nothing in
  this campaign may change a container layout or a shared struct to serve x86_64.

**Gate every host on every row.** Spec 35's dual stop-gate: an aggregate that improves while a binding
constraint regresses is a failure, not a win.

## 5. The dispatch fork — decide before writing a kernel

rawr uses **compile-time** feature detection (`builtin.cpu.features`). CRoaring uses **runtime** dispatch.

Under `-Dcpu=native` the board is a fair comparison either way, and rawr genuinely has only 128-bit code
even then. But **a library distributed for generic x86_64 gets nothing from a compile-time-only wider
path**, because the generic baseline has no AVX2 and no SSE4.2.

This is a distribution decision, not a performance one, and it changes the shape of every kernel written
afterwards. **Settle it before Stage 2, not during.** Note that `_mm_cmpestrm` is **SSE4.2**, above
rawr's current `avx + ssse3` floor, so even the array-row candidate raises this question.

## 6. Out of scope

- **AVX2 for array intersection compaction** — spec 15 established `vpshufb` cannot compact across
  128-bit lanes. Any wider array kernel is an AVX-512 `vpcompressw` question, and that remains parked as
  [spec 15](todo/15-avx512-array-kernel.md).
- **Adding a 32-bit limitation.** rawr ships 32-bit support (spec 40); per-arch work must not regress it.
- **Any aarch64 kernel change.** aarch64 is at parity and is out of scope except as a negative control.

## 7. Chunking

- **`52-00` — Stage 0, host validation.** Canonical board on native Linux x86_64, same machine. Resolves
  the spec 28 `serialize` contradiction. **No production change.** Decides whether the rest happens.
- **Stage 1 attribution is deliberately unwritten.** Its arms depend on which clusters survive Stage 0,
  and writing them now would be guessing — the same reason `51`'s Stage 2 was left unwritten until
  `51-00` reported.
