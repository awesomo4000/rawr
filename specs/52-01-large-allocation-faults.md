<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 52-01: Why SMP loses on single large allocations

Toplevel: [52-x86-64-parity.md](52-x86-64-parity.md).

**Diagnosis only. No production change, no allocator change, no default change.**

## 1. What this is about

`SmpAllocator` is **bimodal** against libc on the x86_64 board, not slower:

| row | SMP | libc | SMP/libc |
| --- | ---: | ---: | ---: |
| `serialize` | 2.771 | 0.805 | **3.44** |
| `toArrayAlloc (1M values)` | 2.914 | 1.043 | **2.79** |
| `bitwiseOr (sparse)` | 0.555 | 1.450 | 0.38 |
| `deserialize` | 0.462 | 1.320 | 0.35 |
| `bitwiseAnd (array balanced)` | 2.688 | 9.138 | 0.29 |
| `lazyOr construction` | 0.241 | 1.012 | 0.24 |

**So this is not an allocator-selection question.** Switching the default to libc would break five rows to
fix two. The question is **which allocation shape SMP serves badly**.

### 1.1 Spec 37's mechanism is ruled out, not assumed forward

Spec 37 established that SMP is **order-sensitive**: its cost came from the *address order* returned
across **16,364 separate 8KB buffers** traversed in allocation sequence, and address-sorting the identical
buffers recovered nearly all of it.

**That cannot be the cause here.** Both losing rows make **one allocation**:

- `serialize` — `allocator.alloc(u8, size_bytes)` for the whole output buffer (`serialize.zig:153`);
- `toArrayAlloc` — `allocator.alloc(u32, total)`, **4 MB** at 1M values (`bitmap.zig:793`).

**With one allocation there is no traversal order to get wrong.** Carrying spec 37's pathology forward
would have been the same mistake spec 51 made when it assumed a vectorized reference it had not checked.

### 1.2 The hypothesis under test

`SmpAllocator` uses **64KB slabs**. A request larger than a slab cannot be served from one, so it is
expected to go to the OS and to return its pages on free. The benchmarks allocate and free **per
iteration**, so every iteration would re-fault and the kernel would re-zero the whole buffer. glibc
retains blocks above its mmap threshold rather than returning them, handing back the same warm pages.

**This is a hypothesis with a numeric prediction**, which is what makes it worth testing:
**~1024 minor faults per iteration for a 4 MB buffer under SMP, near zero under libc.**

It also predicts the bimodality without appealing to hardware: SMP should win where allocations are small
and slab-served and lose only where one allocation exceeds the slab.

## 2. Two layers

**Layer 1 — bare reproducer, zero rawr and zero CRoaring code.** Allocator, one large `alloc`, write the
buffer, `free`, in a loop. This is spec 37's shape, and it is what made spec 37 decisive: if a bare
reproducer shows the split, the mechanism belongs to the allocator and nothing else is implicated.

*(Using `c_allocator` directly is normally discouraged in rawr paths because it hides leaks. Here the
allocator **is** the subject, exactly as in spec 37, and the scope is this reproducer only.)*

**Layer 2 — the real rows.** Replay the production `serialize` and `toArrayAlloc` paths under both
allocators with the same instrumentation, and **report the allocation size each one actually requests**
rather than assuming. Without Layer 2 this chunk would explain a synthetic case and assume it is the same
thing.

## 3. Instrumentation

**Reuse spec 36's fault counter.** It already reads `getrusage` → `ru_minflt` and reports *which* source
it used (`RAWR_RESIDENCY_FAULT_LINUX_RUSAGE`, `bench_lazy_or_residency.zig:217`). Do not rebuild it, and
**carry the source tag into this report** — a fault count whose provenance is unstated is not evidence.

Per cell, report:

- **minor and major faults per iteration**;
- **`mmap` and `munmap` call counts**, which test "goes to the OS per allocation" directly rather than by
  inference;
- wall time under the spec 22 protocol: fresh process per cell, warmup then timed, **≥5 process medians
  with full ranges**.

## 4. Controls — each can refute the hypothesis

**4.1 Size sweep across the slab boundary.** Sizes **16 KB, 32 KB, 64 KB, 128 KB, 256 KB, 1 MB, 4 MB,
16 MB**. If the mechanism is "exceeds the 64KB slab", the fault and time split must **appear at or above
the boundary and be absent below it**. **A split with no size threshold refutes the size-class
hypothesis** and this is the central falsification.

**4.2 Retained buffer.** Allocate once, reuse across all iterations, free at the end. If the cost is
re-faulting after release, the gap must **vanish for both allocators**. If it persists, the cause is not
allocate/free churn and §1.2 is wrong.

**4.3 Induce the defect in the fast arm.** glibc's mmap threshold is tunable:
`MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=0` forces it to mmap and release large blocks.
**Under that setting libc should acquire the same fault count and the same slowdown.** This is the
strongest arm in the design — confirming a mechanism by reproducing it in the control is far better
evidence than observing it once in the suspect. **Linux and glibc only**; state that scope rather than
implying portability.

**4.4 The faults must account for the time.** Spec 36 refuted first-touch for lazy-OR precisely because
**40 faults could not explain 2.426 ms**. Apply the same test here: report faults per iteration *and*
time, and check they move **together** across the §4.1 sweep. **A large fault count that does not track
the timing is incidental, not causal**, and must be reported as such.

## 5. Hosts

**The x86_64 Linux KVM guest is the primary host** and is sufficient. §4.3 needs Linux and glibc, so that
arm runs there only.

**Run Layers 1 and 2 on the aarch64 host as well.** `getrusage` provides `ru_minflt` there too, and the
bimodality is an x86_64 observation that has never been checked on aarch64 — if SMP shows the same
size-threshold behaviour on both, the finding is about the allocator's design rather than one platform.
**Report the fault-source tag per host**, since the mechanism differs.

**No board run, and no bare-metal host is required.** The split is measured within one machine and one
run, so the host class does not threaten it; see `52-00` §A.0.

## 6. What this chunk may not conclude

- **No fix, no allocator change, no default change.** SMP wins 3–4x on five rows; a switch is not on the
  table and this chunk does not evaluate one.
- **No claim that SMP is "slow".** The finding, if it holds, is a **size threshold**, and the honest
  statement names the shape rather than the allocator.
- **No performance claim for any newly measured host.**

## Acceptance

- Layer 1 bare reproducer and Layer 2 production replay, both allocators, per §2, with **Layer 2
  reporting each path's actual requested allocation size**.
- Faults, `mmap`/`munmap` counts, and timing per §3, with the **fault-source tag recorded per host**.
- **§4.1 size sweep complete**, with the threshold located or its **absence reported as refuting §1.2**.
- **§4.2 retained-buffer control run**, with the gap vanishing or §1.2 reported as wrong.
- **§4.3 induction control run on Linux/glibc**, stating whether forcing libc to mmap reproduces both the
  fault count and the slowdown.
- **§4.4 fault-versus-time check stated explicitly**, including the case where counts do not track timing.
- Spec 37's order mechanism **recorded as ruled out for this shape**, with the one-allocation reason, so it
  is not reintroduced.
- Both hosts for Layers 1 and 2; §4.3 Linux only and scoped as such.
- No production change; existing suites plus `check-32`, `check-docs`, `check-package`, `check-portability`
  green.

## Estimate

**S/M** — the reproducer is small and the fault counter already exists. The controls are most of the work,
and §4.3 is the one that would settle it.
