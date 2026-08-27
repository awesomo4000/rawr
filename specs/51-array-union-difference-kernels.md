<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 51: Diagnose the array union and difference gap

**Target.** The one real-data gap that reproduced on both hosts (spec 50-02): `wikileaks-noquotes`
pairwise OR at **2.586x** (M4) / **2.682x** (Zen 4), ANDNOT at **2.005x** / **4.073x**.

**Diagnosis first. No production change until a prototype earns it.**

## 1. What the source actually shows

*(An earlier draft of this spec claimed "CRoaring vectorizes all three" and proposed missing SIMD as the
cause. **That is wrong on M4**, and M4 is where the gap was found.)*

### 1.1 CRoaring's vectorized union and difference are x86-only

`fast_union_uint16` and the ANDNOT dispatch are gated on **`CROARING_IS_X64`**, which `roaring.h:143`
defines only under `__x86_64__` or `_M_X64`. **On M4 (aarch64) CRoaring runs scalar `union_uint16` and
`difference_uint16`.**

M4 still shows OR **2.586x** and ANDNOT **2.005x**. **So missing SIMD cannot explain the M4 gap**, and the
hypothesis splits by host:

| host | what rawr is losing to |
| --- | --- |
| **M4** | CRoaring's **scalar** union/difference. So: algorithm, codegen, or surrounding work |
| **Zen 4** | the same scalar difference, **possibly compounded** by CRoaring's AVX2 path |

### 1.2 rawr's union does work CRoaring's does not

`arrayUnionArray` (`container_ops.zig:559`) ends with `arrayToArrayOrRun`, which calls
`countRunsInArray` — **a full linear scan of the merged result** — and may then allocate and convert to a
run container. CRoaring's array union returns an array directly.

**That is a per-union extra pass entirely outside the merge kernel**, and it scales with result size,
which fits the corpus pattern: `wikileaks` has median cardinality 280, `uscensus2000` has 2.

### 1.3 But the scan does not cover ANDNOT

`arrayDifferenceArray` (`:996`) returns `.{ .array = result }` **directly, with no run scan**. Only
array-union among the array-array operations carries `arrayToArrayOrRun`.

**ANDNOT shows 2.005x / 4.073x anyway.** So there are at least two things to explain, and OR and ANDNOT
must not be treated as one phenomenon.

### 1.4 Candidate causes, none established

- rawr's eager **run scan and possible run conversion** (union only);
- **scalar merge algorithm or codegen** differences against CRoaring's scalar path (both operations);
- **CRoaring AVX2** on Zen 4 only;
- **top-level work**: unmatched-container cloning, allocation, result sizing.

Stage 1 exists to tell these apart. The corpus histograms from 50-01 (`wikileaks`: 1,892 arrays, 0
bitsets, 0 runs; median card 280) say the answer lives in the array-array path, not in bitset or run
handling.

## 2. Relation to the parked kernel work

Two prior kernel efforts are adjacent but not precedent either way:

- **Spec 34 (NO-GO)** unrolled an *existing* kernel for `select`. The row closed later through Run-header
  locality instead.
- **Spec 15 (parked)** asks whether *wider* SIMD beats the existing 128-bit intersect.

Both concern an existing kernel. **This spec starts from a measured 2-4x gap with no established cause**,
and §1 shows the cause may not be a kernel at all. If Stage 1 points at the run scan, spec 15's
per-arch-cost argument never comes into play.

## 3. Stage 1: attribute the gap before building anything

**Do not start with SIMD, or with any kernel.** Establish where the time goes.

### 3.1 Two measurement layers, so subtraction is valid

*(An earlier draft listed four arms that mixed allocation levels: "merge kernel only" sounded
allocation-free while "merge + post-processing" allocates, and the CRoaring arm did not say. Nothing
could be cleanly subtracted from anything.)*

**Layer A — kernel replay, every output buffer preallocated.** No allocation inside any timed region.

| arm | what it runs |
| --- | --- |
| A1 | rawr scalar merge |
| A2 | CRoaring **scalar** (`union_uint16` / `difference_uint16`) |
| A3 | CRoaring **production-selected** path for this build and host |

**Layer B — matched-container replay, production allocation and teardown included.**

| arm | what it runs |
| --- | --- |
| B1 | rawr as it ships today |
| B2 | CRoaring as it ships today |
| B3 | rawr **without normalization** — merge and allocate, skip `arrayToArrayOrRun` |

**Each question is then one subtraction:**

| question | answer |
| --- | --- |
| rawr scalar vs CRoaring scalar | **A1 − A2** |
| CRoaring AVX2 uplift (Zen 4) | **A2 − A3** |
| **normalization cost** (union only) | **B1 − B3** |
| **rawr** allocation and assembly | **B3 − A1** |
| **CRoaring** allocation and assembly | **B2 − A3** |

**These five terms fully apportion `matched_delta`. The identity is exact:**

```
B1 - B2 =   (B1 - B3)     normalization
          + (B3 - A1)     rawr allocation/assembly
          + (A1 - A2)     scalar-kernel difference
          + (A2 - A3)     CRoaring production/AVX2 uplift
          - (B2 - A3)     CRoaring allocation/assembly
```

*(Verified algebraically: the first four collapse to `B1 - A3`, and subtracting `B2 - A3` gives
`B1 - B2`.)*

**The identity is arithmetic, not evidence.** It holds for any values, so a residual check would pass
unconditionally and could not detect a mislabeled arm. `51-00` §3 validates arm meaning directly instead:
identical pair and input counts, matching semantic digests, zero timed allocations in Layer A, and a zero
normalization counter in B3.

*(An earlier draft omitted `B2 - A3`, so the terms did not close and CRoaring's own allocation cost was
silently attributed elsewhere.)*

*(An earlier draft gave allocation as `B1 − A1`. Wrong: that also contains normalization, so it answers
neither question. B3 is merge plus production allocation **without** normalization, which is exactly the
allocation layer.)*

**`B1 − B3` is normalization cost, not pure run-scan cost.** It includes any run-container conversion and
its allocation. **Report the conversion count**; only when it is zero does this subtraction isolate the
scan itself.

**On M4, A2 and A3 must agree algorithmically** (§1.1): **A3 must report which scalar path it selected**.
Timing need not match — A3 goes through the `fast_union_uint16` wrapper, which adds dispatch and may
reorder operands by size, so differing ranges do **not** invalidate the run. Selecting a *vectorized* path
on M4 would.

**On Zen 4, `croaring_hardware_support()` is necessary but not sufficient.** A3 must report **the branch
it actually selected, per operation**. `array_container_andnot` (`roaring.c:6874`) takes
`difference_vector16` only when AVX2 support **and** `(out != array_1) && (out != array_2)` both hold, so
an aliased output buffer yields the scalar path while support still reports true. Union carries no such
condition. See `51-00` §2 and §5 for the buffer requirement and the branch-selection guard.

### 3.2 Account for pairs that never reach the merge

Count and report, per operation:

- **matched array-array pairs** that run the merge;
- **union pairs taking the `max_card > 4096` bitset path** (`container_ops.zig:512`) — these never execute
  the merge at all;
- **unmatched containers, reported by behaviour rather than as one number**, because the two operations
  differ:
  - **OR clones unmatched containers from both sides**;
  - **ANDNOT clones unmatched left containers and skips unmatched right containers entirely.**

A kernel change cannot help work that does not run the kernel, and the proportions decide whether the
exercise is worth starting.

### 3.3 Call boundary

**Batch all pairs behind a single call boundary on both sides.** A Zig call per pair against a linked-C
call per pair would reintroduce a dispatch artifact rather than measure the operation.

### 3.4 Gate — absolute deltas, pre-registered

*(An earlier draft asked for kernel and end-to-end **ratios** to agree "within a stated tolerance", with
no tolerance stated. Equal ratios are not an attribution test.)*

Replay **every eligible pair exactly once** and compare **absolute deltas**:

```
endtoend_delta  = rawr_endtoend_time - croaring_endtoend_time
matched_delta   = B1 - B2
explained_share = matched_delta / endtoend_delta
```

**`matched_delta` is what the matched-container path accounts for; A1/A2/A3 and B3 then apportion it**
into kernel, AVX2, normalization, and allocation per the §3.1 table.

**Pre-registered requirement: `explained_share` ≥ 0.70**, decided by the interval rule in `51-00` §3.1
(`S_min >= 0.70` passes, `S_max < 0.70` fails, between is inconclusive), not by informal range overlap.

**Evaluated independently per operation and per host.** §1.3 established OR and ANDNOT may be separate
phenomena, so a single combined verdict would hide exactly the distinction this stage exists to draw. OR
may pass on both hosts while ANDNOT fails, or either may pass on one host only.

- An `explained_share` **< 0.70** for an operation means its gap lives outside the matched-container
  path. **Stop for that operation.** The lever is top-level cloning, allocation, or result sizing, and that is a different spec.
- If **B1 − B3** accounts for a large share on union, **normalization is the lever, not a kernel**. Call
  it *scan* cost only when the reported run-conversion count is zero; otherwise it is scan plus
  conversion plus that conversion's allocation.

### 3.5 Reporting

Report **ns per pair** and **ns per input element**, not only per output element: an empty ANDNOT result
makes per-output undefined. Report the **input-size distribution**, since it decides whether any
vectorized kernel could help at all.

## 4. Stage 2: prototype, gated

**Only after Stage 1 attributes the gap, and the fix follows the attribution.** The order below is
cheapest-first, and stops as soon as a gate is met.

- **If normalization is the lever, address it first** — deferring or removing the eager
  `arrayToArrayOrRun`. That is not a kernel change and costs no per-arch code.
- Otherwise implement **scalar-improved** union and difference (galloping when sizes are skewed, matching
  what `intersectWriteGallop` already does for AND). **A scalar win needs no new per-arch code and should
  be tried before SIMD.**
- **Baseline and candidate must be arms in the same binary**, per spec 28: comparing across builds would
  charge layout movement to the change.
- Then, and only if scalar is insufficient, a vectorized `unionWrite` / `differenceWrite` pair.

**Cost to weigh explicitly:** a SIMD union plus difference means **four new per-arch kernels** (x86 and
NEON for each), roughly doubling `array_simd.zig`. zroar declined per-arch paths entirely; spec 15 has
stayed parked over the same cost. That is a real maintenance and correctness surface, and the decision
belongs to the owner.

## 5. Measurement

- **Canonical real-data protocol from spec 50**: one process per cell, 5 processes, 1 warmup + 7 timed
  cycles, aggregate medians, semantic digests, cross-host audit.
- **Both hosts.** The gap is one of the few findings that held on both, and it must be shown to close on
  both.
- **All three corpora.** `uscensus2000` and `census1881` are the controls: a change that helps
  `wikileaks` while hurting them is not a win.
- **Parity board unaffected** and re-run to confirm it. The board's array rows are synthetic and a kernel
  change touches them.

## 6. Gates

- **Stage 1:** `explained_share = (B1 − B2) / endtoend_delta` clears **`S_min >= 0.70`** under the
  `51-00` §3.1 interval rule, **evaluated independently per operation and per host**. An operation that
  fails stops there.
- **Stage 2, also per operation:** the operation that passed Stage 1 improves materially on the hosts
  where it passed, with non-overlapping ranges, and neither control corpus regresses beyond 5%.
  **OR and ANDNOT are adopted or rejected separately** — a working run-scan removal for OR is adoptable
  with ANDNOT still unexplained, and the reverse holds too.
- **If an operation passes attribution on only one host, the other host is a no-regression control** for
  Stage 2: the change must not make it worse, even though it is not expected to help there.
- **No parity-board row regresses** beyond the spec-28 layout tolerance.
- Correctness unchanged: semantic digests still match CRoaring across all 42 cells.
- **Any production kernel additionally needs direct randomized and edge-case differential tests** —
  empty inputs, single element, full 4096, disjoint, identical, adjacent-value runs. The 42 digests cover
  three corpora and would not exercise those.

## 7. Out of scope

- The other two 50-02 findings, each of which deserves its own spec:
  - **`census1881` serialize + deserialize**, 1.640x M4 / 2.824x Zen 4, consistent on both hosts;
  - **`toArray` on Zen 4**, 2.546x and 2.237x, against 0.918x / 0.837x on M4.
- Wider SIMD for the existing intersect kernel (spec 15).
- Bitset or run kernels. The corpus that shows this gap has neither.

## 8. Estimate

**M** — Stage 1 is a focused microbenchmark reusing the spec 50 harness: two measurement layers (six arms
total), pair accounting, and per-operation attribution. Stage 2 depends on whether the run scan or a scalar improvement suffices.

## 9. Stage 1 outcome (08/27/2026)

**Both operations pass attribution on both hosts.** Matched arrays explain OR at 1.048 point share on M4
and 1.078 on Zen 4, and ANDNOT at 0.981 and 1.027. All four interval lower bounds exceed 0.70. See
[`51-00`](51-00-attribution.md) for the complete arms, ranges, and artifacts.

The result does not support one shared lever:

- **OR:** eager normalization/conversion and the scalar merge are both material. Normalization is the
  largest Zen 4 term; scalar merge is the largest M4 term. Of 930 array-array unions, 864 convert to runs,
  so the normalization term is not a pure scan.
- **ANDNOT:** scalar merge is the dominant term on both hosts. CRoaring's AVX2 path adds a separate Zen 4
  advantage; M4 has the same scalar-only conclusion without it.

Stage 2 therefore needs operation-specific candidates.

### Revising the cheapest-first order — scalar goes first

This toplevel pre-registered "if normalization is the lever, address it first." That ordering was written
before the terms were known, and the measured terms argue against it:

- The **scalar merge is the only term present and material in all four cells** (0.226–0.322 ms each). It is
  the sole dominant term for ANDNOT on both hosts, where normalization is zero. **Scalar work is required
  regardless of what happens to normalization**, so doing it first cannot be wasted.
- **Normalization carries a representation trade that the scalar merge does not.** 864 of 930 union
  results genuinely convert to runs — they are not spurious conversions. Deferring or removing the eager
  conversion leaves array containers that are larger when serialized and worse-suited to whatever operation
  consumes them next. **That cost lands outside the OR row**, so a normalization candidate must be gated on
  the wider board and on serialized size, not on this corpus's OR timing alone.

**Order for Stage 2:** scalar algorithm and codegen diagnosis first, covering both operations at once;
normalization second, as a separately gated candidate with its own downstream measurements. SIMD stays
last and applies only to ANDNOT — the Zen 4 union result shows a vectorized union losing to its own scalar
arm on this corpus, so there is no evidence to justify writing one.

## 10. Chunking

- **[51-00](51-00-attribution.md)** — Stage 1 attribution. Diagnosis only. **Accepted.**
- **[51-01](51-01-scalar-merge-candidates.md)** — scalar merge candidates measured as Layer A arms. Still
  no production change. Covers both operations, since the scalar term is material in all four cells.
- **`51-02` productize is unwritten**, and stays that way until `51-01` reports which candidate wins and
  whether the C3 arm shows a residual. A residual means the diagnosis is incomplete and productizing
  would be premature.
- **Normalization is a separate later chunk**, gated on the wider board and serialized size per §9, not
  on this corpus.
