<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 51-00: Stage 1, attribute the OR/ANDNOT gap

Toplevel: [51-array-union-difference-kernels.md](51-array-union-difference-kernels.md).

**Diagnosis only. No production change, no kernel written.** Stage 2 stays unwritten until this reports,
because its content depends on what the attribution shows.

## 1. What is being attributed

`wikileaks-noquotes` pairwise OR at **2.586x** (M4) / **2.682x** (Zen 4), ANDNOT at **2.005x** /
**4.073x**, from spec 50-02.

**OR and ANDNOT are treated as separate phenomena throughout.** `arrayUnionArray` carries an eager
`arrayToArrayOrRun`; `arrayDifferenceArray` does not, yet both show a gap.

## 2. Arms

**Layer A, kernel replay, every output buffer preallocated**, no allocation in any timed region.

**Every Layer A output buffer must be distinct from both input buffers.** `array_container_andnot`
(`roaring.c:6874`) selects `difference_vector16` only when
`(croaring_hardware_support() & ROARING_SUPPORTS_AVX2) && (out != array_1) && (out != array_2)`. Its own
comment notes that `out` *may* alias `array_1`, so aliasing is legal and easy to do by accident — and an
aliased harness would take the scalar path on Zen 4 while reporting AVX2 as supported. A3 would then equal
A2 and the uplift term would read as zero for the wrong reason.

| arm | runs |
| --- | --- |
| A1 | rawr scalar merge |
| A2 | CRoaring scalar (`union_uint16` / `difference_uint16`) |
| A3 | CRoaring production-selected path for this build and host |

**Layer B, matched-container replay, production allocation and teardown included:**

| arm | runs |
| --- | --- |
| B1 | rawr as it ships |
| B2 | CRoaring as it ships |
| B3 | rawr merge and allocate, **skipping** `arrayToArrayOrRun` |

**Batch all pairs behind one call boundary on both sides.** A Zig call per pair against a linked-C call
per pair measures dispatch, not the operation.

## 3. Apportionment

```
endtoend_delta  = rawr_endtoend_time - croaring_endtoend_time
matched_delta   = B1 - B2
explained_share = matched_delta / endtoend_delta
```

`matched_delta` decomposes exactly:

```
B1 - B2 =   (B1 - B3)     normalization
          + (B3 - A1)     rawr allocation/assembly
          + (A1 - A2)     scalar-kernel difference
          + (A2 - A3)     CRoaring production/AVX2 uplift
          - (B2 - A3)     CRoaring allocation/assembly
```

**The identity is arithmetic, not evidence.** The five terms telescope for *any* values, so a residual
check would always pass and validates nothing. *(An earlier draft required it as a gate.)*

**Arm meaning is validated directly instead:**

| check | catches |
| --- | --- |
| **identical pair and input-element counts** across every arm | an arm replaying a different workload |
| **semantic result digests match** across arms and against CRoaring | an arm computing a different answer |
| **zero allocations inside any Layer A timed region** | A-layer arms secretly allocating, which would make `B3 - A1` wrong |
| **normalization counter is zero in B3** | B3 still running `arrayToArrayOrRun`, which would collapse `B1 - B3` to noise |

Each is a property an arm can actually violate.

**`B1 - B3` is normalization, not scan cost.** It includes run conversion and that conversion's
allocation. **Report the run-conversion count**; only a count of zero makes it scan cost.

### 3.1 How the numbers are computed

- Every term comes from **aggregate medians**: median of the ≥5 fresh-process medians, each process
  running 1 warmup and 7 timed cycles.
- **`explained_share` is computed from those aggregate medians**, never as a median or mean of per-process
  shares.
- **Ranges validate by interval arithmetic, pinned exactly.** With `E1`/`E2` the rawr and CRoaring
  end-to-end companion cells (§3.2):

  ```
  N = [B1_min - B2_max,  B1_max - B2_min]     matched_delta interval
  D = [E1_min - E2_max,  E1_max - E2_min]     endtoend_delta interval
  S = [N_min / D_max,    N_max / D_min]       explained_share interval
  ```

  **Check the sign of both intervals before dividing.** The quotient form above is only valid for a
  non-negative numerator and a strictly positive denominator:

  | condition | outcome |
  | --- | --- |
  | `D_min <= 0` | **report as undefined**, do not divide — a denominator spanning zero makes the share meaningless |
  | `N_max <= 0` | **attribution fails** — the matched-container path does not account for the gap at all |
  | `N_min < 0 < N_max` | **inconclusive, rerun** — the sign of the numerator is not established |
  | `N_min >= 0` and `D_min > 0` | compute `S` and apply the threshold below |

  Then:

  - **Pass** if `S_min >= 0.70`
  - **Fail** if `S_max < 0.70`
  - **Otherwise inconclusive: rerun the cell.**

### 3.2 The end-to-end denominator is measured here, not imported

`endtoend_delta` comes from **rawr and CRoaring end-to-end companion cells run in this chunk**, under the
**same binary, corpus, and process protocol** as the six arms.

**Do not import the spec 50-02 numbers.** They came from a different binary, and spec 28 established that
code changes move untouched rows. A denominator from one build with a numerator from another is not a
share of anything.

## 4. Pair accounting

Per operation, count and report:

- matched array-array pairs that run the merge;
- **union pairs taking the `max_card > 4096` bitset path** (`container_ops.zig:512`), which never reach
  the merge;
- unmatched containers **by behaviour**: OR clones from both sides; ANDNOT clones unmatched left and
  skips unmatched right entirely.

A kernel change cannot help work that never runs the kernel.

## 5. Host requirements

**M4:** A2 and A3 must **agree algorithmically**. A3 reports which scalar path it selected. Timing need
not match, since A3 goes through the `fast_union_uint16` wrapper with dispatch and operand reordering.
**A vectorized selection on M4 invalidates the run.**

**Zen 4:** hardware support is necessary but **not sufficient**.

- **A3 reports the branch it actually selected, per operation** — vectorized or scalar — not the
  capability bits.
- **ANDNOT requires both AVX2 support and non-aliasing outputs** (`roaring.c:6874`). Union has no
  aliasing condition; the two operations must be reported separately rather than assumed to match.

**Branch-selection guard:** the run **fails** if a reported path disagrees with the conditions that
should produce it — a vectorized branch reported on M4, a vectorized ANDNOT reported with aliased
buffers, or a scalar branch reported on Zen 4 with AVX2 support and distinct buffers. Reporting the
branch without checking it against its preconditions would let a mislabeled arm through.

## 6. Reporting

**ns per pair** and **ns per input element**, not only per output element: an empty ANDNOT result makes
per-output undefined. Report the **input-size distribution**, which decides whether a vectorized kernel
could help at all.

## 7. Gate

**`S_min >= 0.70` per the §3.1 interval rule, evaluated independently per operation and per host.**
`S_max < 0.70` fails; anything between is inconclusive and reruns.

- Below 0.70 for an operation on a host: its gap lives outside the matched-container path. **Stop for that
  operation**; the lever is top-level cloning, allocation, or result sizing, and that is a different spec.
- At or above: report which term dominates. That determines what Stage 2 is even about, and it may be
  different for OR and ANDNOT, or for the two hosts.

## Acceptance

- Six arms per §2, batched behind one call boundary.
- Apportionment per §3, with the **run-conversion count reported**.
- **Arm-meaning checks all pass** (§3): identical pair and input counts, matching semantic digests, zero
  timed allocations in Layer A, zero normalization counter in B3.
- `explained_share` from aggregate medians; **interval rule applied exactly as §3.1**, including the
  **sign checks on `N` and `D` before dividing**, with inconclusive cells rerun.
- **End-to-end companion cells measured in this chunk** under the same binary, corpus, and protocol; no
  spec 50-02 numbers imported.
- Pair accounting per §4, unmatched reported by behaviour.
- §5 host requirements met and stated in the artifacts, including **A3's selected branch per operation**
  and the **branch-selection guard passing**.
- **Every Layer A output buffer distinct from both inputs**, verified rather than assumed.
- §6 reporting complete.
- **Verdict per operation per host**, with the dominant term named where the gate passes.
- **No production change. No kernel written. Stage 2 remains unwritten.**
- Existing suites and checks green; no board row moves.

## Outcome (08/27/2026)

**GO for Stage 2 on both operations and both hosts.** The matched array-array path accounts for at least
70% of the end-to-end gap in every cell under the pre-registered interval rule:

| host | operation | end-to-end delta | matched delta | point share | share interval | verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| M4 | OR | 0.477 ms | 0.500 ms | 1.048 | [0.738, 1.471] | PASS |
| M4 | ANDNOT | 0.270 ms | 0.265 ms | 0.981 | [0.792, 1.187] | PASS |
| Zen 4 | OR | 0.480 ms | 0.517 ms | 1.078 | [0.788, 1.322] | PASS |
| Zen 4 | ANDNOT | 0.377 ms | 0.387 ms | 1.027 | [0.857, 1.110] | PASS |

A point share above 1.0 is not an overclaim: it means work outside the matched path partially offsets the
matched-path disadvantage. The interval, not the point estimate, decides the gate. The first M4 ANDNOT
run was inconclusive and was rerun as required; the table is the decisive rerun.

The terms identify two different Stage 2 targets:

| host | operation | normalization | rawr alloc/assembly | scalar | CR production/AVX2 | CR alloc/assembly |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M4 | OR | +0.224 ms | +0.012 ms | **+0.300 ms** | +0.011 ms | +0.047 ms |
| M4 | ANDNOT | -0.005 ms | -0.015 ms | **+0.322 ms** | -0.007 ms | +0.030 ms |
| Zen 4 | OR | **+0.376 ms** | +0.012 ms | +0.226 ms | -0.080 ms | +0.017 ms |
| Zen 4 | ANDNOT | +0.018 ms | +0.017 ms | **+0.263 ms** | **+0.109 ms** | +0.019 ms |

- **OR:** normalization is material on both hosts and dominant on Zen 4; the scalar merge is dominant on
  M4 and still material on Zen 4. All 930 pairs use the array merge and 864 results convert to runs, so
  `B1 - B3` is scan plus conversion and conversion allocation, not a pure scan. Zen 4 selected AVX2, but
  its production wrapper was slower than the direct scalar arm for this corpus; AVX2 is not the OR lever.
- **ANDNOT:** normalization is zero within noise and the scalar merge dominates on both hosts. Zen 4's
  selected AVX2 path adds a separate 0.109 ms advantage for CRoaring. Stage 2 should start with the scalar
  algorithm/codegen difference, then consider per-architecture SIMD only if scalar work is insufficient.

Every operation replays 930 matched array pairs and 245,589 input elements, with no bitset-path or
matched-other pairs. Input sizes are `[min=2, p50=128, p90=683, p99=1320, max=2603]`; unmatched counts are
961 left and 944 right. The M4 A3 arms selected scalar; the Zen 4 A3 arms selected AVX2 with distinct
outputs. Branch guards passed.

The cross-host audit matched all semantic digests, the corpus fingerprint, pair accounting, counters,
output guards, and size quantiles across 16 tuples and 80 processes per host. Debug, ReleaseSafe, and
ReleaseFast tests passed, `check-package` remained at 33 allowlisted files, the spec 50 protocol controls
passed, and the full 42-row canonical M4 parity board completed. No public API, default behavior, or
kernel changed; the production scalar loops were only extracted into internal inline hooks so Layer A
executes the exact shipped loops.

Artifacts:

- M4: `misc/array-attribution-20260827-100146-summary.txt`
- Zen 4: `misc/array-attribution-20260827-153342-summary.txt`, copied to the aarch64 host with a
  `-zen4-` suffix for audit

## Verification record — implemented, reviewed, ACCEPTED

**Re-derived independently, not taken on report.** The five-term apportionment closes to the stated
matched delta in all four cells (0.500, 0.265, 0.517, 0.387/0.388 with rounding), and all four interval
lower bounds clear 0.70 with margin (0.738, 0.792, 0.788, 0.857).

**Two results act as controls even though they were not designed as controls:**

- **ANDNOT normalization is zero within noise on both hosts** (-0.005 and +0.018). That matches the source
  reading that the array difference path returns its array directly and never calls the run-normalization
  helper, while the union path does. A material term there would have meant either the source reading or
  the arm wiring was wrong. Neither was.
- **A3 branch selection split by host** — scalar on aarch64, AVX2 on x86_64. This is the reading that
  corrected the original Stage 1 hypothesis, and the guard confirms it in the running binary rather than
  by grepping for a symbol whose reachability was never checked.

**The cross-cell pattern is not visible in any single row.** The scalar merge is the only term that is
present and material in **all four** cells (0.226–0.322 ms; 1.111 ms summed). Normalization is
**OR-only** (0.600 ms summed). The AVX2 advantage is **Zen 4 ANDNOT only** (0.109 ms). One scalar change
can therefore move every cell; the other two levers each move a subset.

**Zen 4 OR is the sharpest single finding.** CRoaring's own vectorized union was *slower* than its scalar
arm on this corpus (-0.080 ms), so rawr trails there while the reference runs its worse path. That rules
out a vectorized union as the OR lever without anyone having to write one first.

The first M4 ANDNOT cell came back inconclusive and was rerun, as §3.1 requires. The rule fired on real
data rather than sitting unexercised.

## Estimate

**M** — six arms, exact apportionment, and pair accounting, reusing the spec 50 harness and corpus.
