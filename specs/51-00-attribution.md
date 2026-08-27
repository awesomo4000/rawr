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

## Estimate

**M** — six arms, exact apportionment, and pair accounting, reusing the spec 50 harness and corpus.
