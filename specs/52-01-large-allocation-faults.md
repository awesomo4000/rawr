<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 52-01: Why SMP loses on single large allocations

Toplevel: [52-x86-64-parity.md](52-x86-64-parity.md).

**Diagnosis only. No production change, no allocator change, no default change.**

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
Mmap-backed allocations **are** returned on free. What makes glibc fast here is that the default mmap
threshold **adapts dynamically** — after observing frees of mapped blocks it raises the threshold, moving
subsequent allocations of that size into the reusable heap, which is not unmapped.

**That has a direct consequence for §4.4:** setting an explicit threshold **disables the adaptation**, so
the induction arm changes two things at once. It must therefore **record the tunables in force and verify
the mapping behaviour actually obtained**, not infer it from the setting.

References: GNU allocator documentation and the glibc memory-allocation tunables page.

## 2. Two layers

**Layer 1 — bare reproducer, zero rawr and zero CRoaring code.** Allocator, one `alloc`, payload write,
`free`, in a loop. Spec 37's shape, and what made spec 37 decisive: if a bare loop shows the split, the
mechanism belongs to the allocator.

*(`c_allocator` is normally discouraged in rawr paths because it hides leaks. Here the allocator **is**
the subject, as in spec 37, and the scope is this reproducer.)*

**Layer 2 — the real rows.** Replay the production `serialize` and `toArrayAlloc` paths under both
allocators, **reporting each path's actual requested length and alignment** rather than assuming.

**Layer 2 must preserve the canonical harness order**: corpus initialisation as the canonical worker does
it, and **validation after timing, never before**. Spec 35's 1.52 ms artifact came from validation
preconditioning SMP ahead of the timed cell, and reproducing that here would manufacture exactly the
effect under study.

## 3. Instrumentation and measurement boundaries

**Fault counts.** Reuse spec 36's counter (`bench_lazy_or_residency.zig:217`) and **carry its source tag**
into the report. **Counter semantics are not uniform:** on Linux it reports `getrusage` `ru_minflt`; on
Darwin the available counters are Mach faults and page-ins, which are **not** equivalent to minor/major
faults. Report the tag per host and do not compare across hosts without naming the difference.

**Derive predicted counts, do not assert them.** Expected faults = payload bytes ÷ page size. An earlier
draft wrote "~1024 faults", which assumes 4 MiB over 4 KiB pages; **M4 uses 16 KiB pages**, giving 256 for
the same payload. **Record the page size and any huge-page policy in force** per host.

**Syscall counts.** Pin the method: count `mmap`/`munmap` by tracing (`strace -c -f -e trace=mmap,munmap`
or the platform equivalent) in **separate, non-authoritative runs**. **Tracing must not appear in timing
runs** — its overhead would contaminate exactly what is being measured. Report the two from different runs
and say so.

**Region and protocol.** Count and time **only the operation region**: `alloc`, payload write, `free`.
Distinguish warmup from timed iterations and report both counts. Spec 22 protocol: fresh process per
cell, warmup then timed, **≥5 process medians with full ranges**.

**Pin the operation itself**, since the whole result depends on it: the exact payload write (`@memset`
versus a strided touch), the **optimizer barrier** preventing its elision, the **batch size** per timed
iteration, and for §4.3 whether the retained buffer is **pre-touched** before timing begins.

## 4. Controls — and what each can and cannot establish

**Separate two claims throughout: "mapping churn observed" and "fault handling explains the timing."**
An earlier draft conflated them, and three of its controls claimed more than they could.

**4.1 Size sweep across the real boundary.** Sizes **4 KiB, 16 KiB, 32 KiB, 32 KiB + 1, 64 KiB, 256 KiB,
1 MiB, 4 MiB, 16 MiB**, at a pinned alignment, with the boundary re-derived per §1.2. **The 32 KiB and
32 KiB + 1 pair is the decisive one** — adjacent lengths on either side of the pooled ceiling. A split
that does not track the source-established route refutes §1.2.

**4.2 Allocation and free without a payload write.** This is the control that separates the two claims:
it retains mapping churn while removing the payload touch. **A gap that survives here is mapping and
kernel-side cost; a gap that disappears implicates the payload write and its faults.** Neither outcome is
a failure.

**4.3 Retained buffer.** Allocate once, reuse, free at the end. **A win here does not isolate fault
cost** — it removes allocation, mapping, unmapping *and* repeated faults together. Read it only as an
upper bound on the total cost of per-iteration churn, and interpret it against §4.2.

**4.4 Induce the behaviour in glibc.** Force mapping with explicit tunables, per §1.3 **recording the
tunables and verifying the mapping behaviour obtained**. If glibc then shows the fault counts and a
slowdown, the mechanism is confirmed by reproduction in the control arm, which is stronger evidence than
observing it once in the suspect. **Do not require an identical slowdown** — the tunables also disable
threshold adaptation, so the arms are not otherwise matched. **Linux and glibc only**; state that scope.

**4.5 Faults versus time, at fixed size.** Spec 36 refuted first-touch for lazy-OR because **40 faults
could not explain 2.426 ms**. Apply that test, but **not by observing that both grow with bytes** — write
bandwidth grows with bytes too, so co-scaling across §4.1 establishes nothing. **Compare at fixed payload
size** between the pooled and mapped routes, and against §4.2. A count that does not account for the
timing is **contributing or incidental, and must be reported as such**.

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

**The §1 table is WSL2 and historical. Run Layer 2 on the primary host first.**

| outcome | what may be claimed |
| --- | --- |
| split reproduces on KVM | proceed to explain it, and state that the WSL2 figures were not re-measured |
| split does not reproduce | Layer 1 still characterises **that host's** allocator behaviour; **the WSL2 result is not explained** and stays open |

**Do not explain a gap that the measured host does not show.**

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
- §3 boundaries pinned and recorded: operation region only, warmup separated from timed, payload write and
  optimizer barrier specified, batch size stated, retained-buffer pre-touch stated.
- **Syscall counts collected in non-authoritative runs**, with tracing absent from timing runs and both
  provenances stated.
- **Predicted fault counts derived from payload bytes and the host page size**, with page size and
  huge-page policy recorded, and the **counter source tag per host**.
- §4.1 sweep complete including **32 KiB and 32 KiB + 1**; §4.2, §4.3 and §4.4 run; §4.5 evaluated **at
  fixed size**, not from co-scaling.
- **§4.4 reports the tunables in force and the mapping behaviour verified**, not inferred.
- **Mapping churn and fault-explains-timing reported as separate conclusions**, with partial or
  inconclusive outcomes stated as such rather than forced.
- §6 applied: Layer 2 reproduction on the primary host **before** any explanation of the historical
  figures, and the WSL2 result left open if it does not reproduce.
- Spec 37's **ordering** mechanism recorded as excluded for this shape, with other layout and residency
  effects **not** claimed excluded.
- Second host is the **aarch64 Linux** machine; any macOS run labelled with Mach counter semantics.
- No production change; existing suites plus `check-32`, `check-docs`, `check-package`,
  `check-portability` green.

## Estimate

**S/M** — the reproducer is small and the fault counter exists. The controls and the measurement pinning
are the work, and §4.2 plus §4.4 are what would settle it.
