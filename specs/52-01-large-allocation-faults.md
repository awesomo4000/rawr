<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 52-01: Why SMP loses on single large allocations

Toplevel: [52-x86-64-parity.md](52-x86-64-parity.md).

**Diagnosis only. No production change, no allocator change, no default change.**

## Implementation outcome, 10/06/2026

Completed on Linux x86_64 under KVM and Linux aarch64. The durable report,
commands, full timing ranges, worker hashes, controls, and validation results
are in [large-allocation-faults.md](../docs/large-allocation-faults.md).

- The installed Zig 0.16.0 sources give a 32,768-byte pooled ceiling on both
  hosts at alignment 1. The bare 32,768/32,769 pair has separated timing ranges
  on both. Traces change from zero mappings to one map/unmap per cycle, with
  9/3 minor faults at 32,769 bytes on x86_64/aarch64.
- At a fixed 4 MiB, direct PageAllocator and SMP have overlapping full-write
  ranges and the same 1,024/256 minor faults per cycle. No-write retains
  mapping but removes payload faults. Retained buffers remove timed mapping
  and faults; x86_64 still has a separated SMP/libc residual. Overlap on
  aarch64 is not a claim of identical cost.
- The prescribed glibc tunables induce one map/unmap per cycle at 4 MiB.
  Full-write libc rises from 1.730 [1.556,1.929] ms to
  4.107 [3.724,4.141] ms, with 1,025 minor faults instead of zero. This supports
  release/re-touch cost, not a quantitative attribution to fault handling.
- Production replay preserves canonical preparation and batching, with
  validation and allocation observation after timing. Actual requests are
  2,524,082 bytes/alignment 1 for serialize and 3,999,572 bytes/alignment 4
  for toArrayAlloc. The x86_64 serialize split is present; toArrayAlloc remains
  unresolved after one paired retry. On aarch64 serialize is absent after
  its paired retry and toArrayAlloc is absent on its first run under the
  greater-than-10% range rule. Neither absence means equal cost.
- Historical WSL2 transfer remains unverified. No board, production code,
  public API, or default allocator changed. One output allocation excludes
  spec 37's many-output-buffer ordering explanation, not every layout effect.

All acceptance items below were exercised. The campaigns contain 140 cells
and 720 authoritative timing processes including retries, plus disposable
calibration and separate trace processes. Positive exact-count tracing,
malformed-marker rejection, the pooled negative case, and worker-protocol
mutation controls passed. Disassembly confirms the fill survives on both
architectures as a call to `compiler_rt.memset`.

Validation passed: normal build; both unit suites on both hosts; both
differential suites and check-32 on x86_64; check-docs and check-package on
both; and the 18-cell check-portability matrix on aarch64. The package
allowlist remains 33 files. A separate ReleaseSafe bare build passed write,
retained, and no-write smoke cases. This outcome is committed with the
implementation before review handoff.

## 1. What this is about

`SmpAllocator` is **bimodal** against libc, not slower. **These are rawr/CRoaring wall-time ratios from
the clean-`b3ab49f` Zen 4 board of 08/28 under WSL2** — historical, and the `SMP/libc` column divides two
such ratios, so it is a shape indicator rather than a measurement:

| row | SMP ratio | libc ratio | SMP/libc |
| --- | ---: | ---: | ---: |
| `serialize` | 2.771x | 0.805x | **3.44** |
| `toArrayAlloc (1M values)` | 2.914x | 1.043x | **2.79** |
| `bitwiseOr (sparse)` | 0.555x | 1.450x | 0.38 |
| `deserialize` | 0.462x | 1.320x | 0.35 |
| `bitwiseAnd (array balanced)` | 2.688x | 9.138x | 0.29 |
| `lazyOr construction` | 0.241x | 1.012x | 0.24 |

**This is not an allocator-selection question.** Switching the default to libc would break five rows to
fix two. The question is **which allocation shape SMP serves badly**.

**§6 requires fresh reproduction on the measured host before any of this is explained.** The table is
WSL2 and the work runs on a KVM guest; a bare experiment that does not reproduce the split describes that
host's allocator behaviour and **cannot explain the WSL2 numbers**.

### 1.1 What the single allocation does and does not rule out

Spec 37's mechanism was **address order across 16,364 separate 8KB output buffers**, where sorting the
identical buffers recovered nearly all of the cost. Both losing rows make **one** allocation:

- `serialize` — `allocator.alloc(u8, size_bytes)` (`serialize.zig:153`);
- `toArrayAlloc` — `allocator.alloc(u32, total)` (`bitmap.zig:793`).

**One output allocation excludes spec 37's many-output-buffer *ordering* mechanism. It does not exclude
every layout or residency effect**, and an earlier draft overstated this. What is ruled out is
specifically the ordering pathology; other address- or residency-sensitive explanations remain open.

### 1.2 The allocation route is source-established; only its cost is hypothesis

From Zig 0.16 `std/heap/SmpAllocator.zig`:

```
min_class        = log2(@sizeOf(usize))            = 3
slab_len         = @max(page_size_max, 64 * 1024)  = 65536
size_class_count = log2(slab_len) - min_class      = 13
sizeClassIndex(len, alignment)
                 = @max(@bitSizeOf(usize) - @clz(len - 1), @intFromEnum(alignment), min_class) - min_class
alloc: if (class >= size_class_count) return PageAllocator.map(len, alignment);
```

**Pooled classes stop at 32 KiB.** `32768` is class 12 and pooled; **`32769` is class 13 and already
takes `PageAllocator.map`**, whose free unmaps. *(An earlier draft put the boundary at 64 KiB. Wrong —
and the sweep it specified would have shown a threshold while locating it in the wrong place.)*

**Alignment participates in the class**, so a high-alignment request can leave the pool at a smaller
length. **Pin the alignment used in every cell** and state it.

**Re-derive the boundary from the Zig actually under test** rather than trusting these constants;
`slab_len` depends on `page_size_max`, which is target-dependent.

**Hypothesis:** mapping and unmapping per iteration makes the kernel re-establish and re-zero the pages
each time, and that dominates. **The route is established by source. The timing consequence is what this
chunk tests.**

### 1.3 What glibc actually does

An earlier draft said glibc "retains blocks above its mmap threshold". **That is backwards.**
Mmap-backed allocations **are** returned on free. The candidate explanation for glibc's speed here is
that the default mmap threshold **adapts dynamically** — after observing frees of mapped blocks it raises
the threshold, moving subsequent allocations of that size into the reusable heap, which is not unmapped.

**Adaptation is a candidate explanation, not a measured one.** Nothing in this chunk has established
that it is why glibc is fast on these rows, and an earlier draft asserted it. It is the leading
hypothesis for the glibc arm, to be tested by §4.4.

**It has a direct consequence for §4.4:** setting an explicit threshold **disables the adaptation**, so
the induction arm changes two things at once — threshold value *and* adaptation. It must therefore
**record the tunables in force and verify the mapping behaviour actually obtained**, not infer it from the
setting.

References: GNU allocator documentation and the glibc memory-allocation tunables page.

## 2. Two layers

**Layer 1 — bare reproducer, zero rawr and zero CRoaring code.** Allocator, one `alloc`, payload write,
`free`, in a loop. Spec 37's shape, and what made spec 37 decisive: if a bare loop shows the split, the
mechanism belongs to the allocator.

*(`c_allocator` is normally discouraged in rawr paths because it hides leaks. Here the allocator **is**
the subject, as in spec 37, and the scope is this reproducer.)*

**Layer 2 — the real rows.** Replay the production `serialize` and `toArrayAlloc` paths under both
allocators, **reporting each path's actual requested length and alignment** rather than assuming.

**Layer 2 must preserve the canonical harness order and its batching**: corpus initialisation as the
canonical worker does it, **validation after timing, never before**, and **canonical batching left
unchanged** unless a different batching is measured as its own labelled diagnostic. Spec 35's 1.52 ms artifact came from validation
preconditioning SMP ahead of the timed cell, and reproducing that here would manufacture exactly the
effect under study.

## 3. Instrumentation and measurement boundaries

**Fault counts.** Reuse spec 36's counter (`bench_lazy_or_residency.zig:217`) and **carry its source tag**
into the report. **Counter semantics are not uniform:** on Linux it reports `getrusage` `ru_minflt`; on
Darwin the available counters are Mach faults and page-ins, which are **not** equivalent to minor/major
faults. Report the tag per host and do not compare across hosts without naming the difference.

**Derive a page-touch prediction, and call it that.** It is **not** an exact fault count. Use:

```text
pages = ceil(((address % page_size) + length) / page_size)
```

**Compute it; do not reason about it informally.** An earlier draft asserted that a 32 KiB + 1 payload
spans 9 base pages aligned and 10 unaligned. **Wrong** — at 4 KiB pages it spans **exactly 9 at every
offset**, since `ceil((4095 + 32769) / 4096) = 9`. That is the one size in the sweep where the span does
*not* vary with offset, so it was the worst available example of a point about ranges. Other sizes do
vary; evaluate the formula per cell rather than quoting a remembered figure.

A separate example: **a 4 MiB payload on M4's 16 KiB pages predicts 256 touches**, where a bare
bytes-over-4-KiB division would have said 1024.

**Record the page size, the observed allocation alignment, and any huge-page policy in force** per host,
and state the prediction as touches rather than faults.

**Syscall counts must be region-aware, which `strace -c` cannot provide.** An earlier draft specified it
and then asked for per-region counts; `-c` aggregates process startup, corpus initialisation, warmup and
the timed region into one total, so it cannot answer the question it was given.

Use an **event trace with explicit phase markers and a parser**: trace `mmap`, `munmap` **and `write`**,
have the worker emit marker writes to a dedicated fd at each phase transition, and segment the event
stream between markers. Any other region-aware method is acceptable if it produces per-phase counts.

**Controls for the segmentation — a negative case alone is vacuous.** Requiring only "zero mappings in a
pooled cell" is passed by a parser that always returns zero. Required:

- **Positive attribution.** Perform **known** `mmap`/`munmap` operations **before, inside and after** the
  marked region and assert the parser attributes each to the correct segment, by exact count.
- **Malformed input rejected.** Missing, duplicated or out-of-order markers must fail rather than produce
  a plausible segmentation.
- **Negative case retained**, but read correctly: a mapping observed inside a pooled cell is a **reason to
  investigate**, not automatic proof the parser is broken — the allocator may genuinely have mapped.

**Tracing must not appear in timing runs** — its overhead contaminates exactly what is measured. Report
counts and timings from different runs and say so.

**Region and protocol.** Count and time **only the operation region**: `alloc`, payload write, `free`.
Distinguish warmup from timed iterations and report both counts. Spec 22 protocol: fresh process per
cell, warmup then timed, **≥5 process medians with full ranges**.

**The operation is pinned here, not left to implementation.** An earlier draft listed these as choices,
which would have let materially different experiments all satisfy acceptance:

| choice | pinned value |
| --- | --- |
| payload write | **`@memset(buf, 0xA5)`** — full touch of every page. A strided touch would under-touch and confound the §3 page prediction. |
| optimizer barrier | **`std.mem.doNotOptimizeAway`** applied to the buffer after the write, inside the timed region |
| write survival | **verify in the emitted code that the `@memset` was not elided**; a loop that compiled its writes away would measure nothing and pass every other check |
| batch size | **derived, not guessed**: enough `alloc`/write/`free` cycles per timed iteration that the iteration exceeds **1 ms**, with the resulting count **recorded per cell**. **Calibrate in disposable processes and pass the fixed count into fresh measurement workers** — calibrating inside the measurement worker runs allocate/free cycles that can raise glibc's adaptive threshold before the recorded warmup, preconditioning the very arm under study. |
| retained buffer (§4.3) | **pre-touched** with one full `@memset` outside the timed region, so the first iteration does not carry the faults |
| induction tunables (§4.4) | **`MALLOC_MMAP_THRESHOLD_=16384`, `MALLOC_TRIM_THRESHOLD_=0`, `MALLOC_TOP_PAD_=0`**, recorded verbatim in the artifact |

## 4. Controls — and what each can and cannot establish

**Separate two claims throughout: "mapping churn observed" and "fault handling explains the timing."**
An earlier draft conflated them, and three of its controls claimed more than they could.

**4.1 Size sweep across the real boundary.** Sizes **4 KiB, 16 KiB, 32 KiB, 32 KiB + 1, 64 KiB, 256 KiB,
1 MiB, 4 MiB, 16 MiB**, at a pinned alignment, with the boundary re-derived per §1.2. **The 32 KiB and
32 KiB + 1 pair is the decisive one** — adjacent lengths on either side of the pooled ceiling.

**A missing timing discontinuity does not refute the route, and it does not clear mapping either.** The
route is established by source and no timing result can overturn it. But an earlier draft then concluded
that flat adjacent cells mean "mapping is not where the cost lives" — **also too strong**. Two cells that
time alike at 32 KiB establish only **no resolved discontinuity at that size**; mapping and re-touch could
still dominate at 4 MiB, where there are 128 times as many pages to establish.

**Apply the §6 range discipline here too: overlapping ranges are not evidence of equal cost.** The
conclusion available from a flat pair is local to that size, and the sweep's larger cells are where a
mapping cost would show if it exists.

**4.2 Allocation and free without a payload write.** This separates the two claims: it retains mapping
churn while removing the payload touch.

**A gap that survives here is allocator bookkeeping plus mapping cost** — not "kernel-side cost", as an
earlier draft put it, since SMP's own class selection and freelist work are still running. **A gap that
disappears implicates the payload write and the page touches it forces.** Neither outcome is a failure,
and §4.5 is what apportions between them.

**4.3 Retained buffer.** Allocate once, reuse, free at the end. **A win here does not isolate fault
cost** — it removes allocation, mapping, unmapping *and* repeated faults together. Read it only as an
upper bound on the total cost of per-iteration churn, and interpret it against §4.2.

**4.4 Induce the behaviour in glibc.** Force mapping with the §3 tunables, **recording them verbatim and
verifying the mapping behaviour obtained** rather than inferring it from the setting.

**What a positive result establishes is bounded.** If glibc then maps per iteration and slows down, that
is **evidence that per-iteration release and re-touch carries real cost** — reproduced in the control arm,
which beats observing it once in the suspect. It is **not** a quantitative attribution to fault handling,
because the tunables change mapping overhead and disable threshold adaptation at the same time. **Do not
require an identical slowdown**; the arms are not otherwise matched. **Linux and glibc only.**

**4.5 Faults versus time, at one fixed size, with three named arms.** Spec 36 refuted first-touch for
lazy-OR because **40 faults could not explain 2.426 ms**. Apply that test — but **not by observing that
both grow with bytes**, since write bandwidth grows with bytes too and co-scaling across §4.1 establishes
nothing.

**At a fixed length and alignment SMP takes exactly one route**, so the comparison cannot come from SMP
alone; an earlier draft asked for pooled-versus-mapped at fixed size, which is not available. The arms at
one fixed size above the ceiling are:

| arm | what it supplies |
| --- | --- |
| **SMP** | maps per iteration, plus SMP's own bookkeeping |
| **`PageAllocator` directly** | maps per iteration with **no SMP bookkeeping** — isolates map/unmap and page touches from SMP's handling of them |
| **libc** | the fast reference under default tunables. **Whether it is heap-reused is an observed outcome to report, not a property assumed here** — §4.4 is what establishes its mapping behaviour. |

**`SMP` against `PageAllocator` is the arm that says whether the cost is mapping itself or SMP's handling
of it.** Read all three against §4.2. A count that does not account for the timing is **contributing or
incidental, and must be reported as such**.

**No control here requires the gap to vanish or to match exactly.** **Partial explanation and
inconclusive are reportable outcomes**, and a residual gap does not refute a contributing mechanism.

## 5. Hosts

**The x86_64 Linux KVM guest is the primary host.** §4.4 needs Linux and glibc and runs there only.

**Second host: the aarch64 Linux machine**, not the aarch64 macOS machine. An earlier draft said "the
aarch64 host" ambiguously. Linux keeps the counter semantics comparable with the primary host; macOS would
introduce the Mach-counter difference of §3 on top of everything else. **Running the macOS host is
optional and, if done, its counters must be labelled as Mach faults and page-ins, not minor faults.**

**No board run, and no bare-metal host is required.** The split is measured within one machine and one
run; see `52-00` §A.0.

## 6. Reproduce before explaining

**The §1 table is WSL2 and historical. Run Layer 2 on the primary host first**, and apply this
**independently to `serialize` and to `toArrayAlloc`** — they are different allocation sizes and may not
behave alike.

**Range rule per row**, matching `52-00` §A.4 rather than inventing a second convention:

| condition | outcome |
| --- | --- |
| `SMP_min / libc_max > 1.10` | **split present** on this host |
| `SMP_max / libc_min <= 1.10` | **split absent** on this host |
| otherwise | **unresolved** — rerun once, then report as unresolved |

| outcome | what may be claimed |
| --- | --- |
| present | explain the **KVM** result. **Transfer of the mechanism to the historical WSL2 figures is unverified** — the WSL2 numbers were not re-measured and remain a separate claim. |
| absent | Layer 1 still characterises **that host's** allocator behaviour; the WSL2 result is **not explained** and stays open |
| unresolved | no explanation of either host; report the ranges |

**Even a clean reproduction explains the host it was measured on.** An earlier draft let a KVM result
stand in for the WSL2 one. **Do not explain a gap the measured host does not show, and do not transfer an
explanation across hosts without measuring there.**

## 7. What this chunk may not conclude

- **No fix, no allocator change, no default change.** SMP wins 3–4x on five rows; no switch is on the
  table and this chunk does not evaluate one.
- **No claim that SMP is "slow".** If it holds, the finding is a **route threshold at the pooled size
  ceiling**, named by shape rather than by allocator.
- **No performance claim for any newly measured host.**

## Acceptance

- Boundary **re-derived from the Zig under test** and reported, with the pooled ceiling and the
  **alignment pinned per cell**.
- Layer 1 bare reproducer and Layer 2 production replay, both allocators, with **Layer 2 reporting actual
  length and alignment** and **preserving canonical corpus initialisation and validation-after-timing**.
- §3 operation **pinned as specified** — `@memset(buf, 0xA5)`, `doNotOptimizeAway`, derived batch size
  recorded per cell, pre-touched retained buffer, verbatim tunables — and **the emitted code checked to
  confirm the `@memset` was not elided**.
- **Syscall counts collected by a region-aware event trace with phase markers**, not `strace -c`, in
  non-authoritative runs, with **all three segmentation controls**: positive attribution of known
  mappings before, inside and after the region by exact count; malformed-marker rejection; and the
  pooled-cell negative case.
- **Batch counts calibrated in disposable processes** and passed as fixed values into fresh measurement
  workers; **Layer 2 canonical batching unchanged** unless measured as its own labelled diagnostic.
- **Page-touch prediction computed per cell** from `ceil(((address % page_size) + length) / page_size)` —
  labelled touches, not faults, and evaluated rather than quoted — with page size, observed alignment,
  huge-page policy and the **counter source tag** recorded per host.
- §4.1 sweep complete including **32 KiB and 32 KiB + 1**, with a flat pair reported as **no resolved
  discontinuity at that size** — neither refuting the source-established route nor clearing mapping cost
  at larger sizes — and **overlapping ranges never read as equal cost**.
- §4.2, §4.3 and §4.4 run; **§4.5 run at one fixed size with all three arms — SMP, `PageAllocator`,
  libc** — and not inferred from co-scaling.
- **§4.4 reports the tunables verbatim and the mapping behaviour verified**, with its conclusion limited
  to release/re-touch cost rather than quantitative fault attribution.
- **Mapping churn and fault-explains-timing reported as separate conclusions**, with partial or
  inconclusive outcomes stated as such rather than forced.
- §6 applied **per row** to `serialize` and `toArrayAlloc` under the stated range rule, with
  **unresolved** available as an outcome, and **no transfer of a KVM explanation to the WSL2 figures**.
- Spec 37's **ordering** mechanism recorded as excluded for this shape, with other layout and residency
  effects **not** claimed excluded.
- Second host is the **aarch64 Linux** machine; any macOS run labelled with Mach counter semantics.
- No production change; existing suites plus `check-32`, `check-docs`, `check-package`,
  `check-portability` green.

## Estimate

**S/M** — the reproducer is small and the fault counter exists. The controls and the measurement pinning
are the work, and §4.2 plus §4.4 are what would settle it.
