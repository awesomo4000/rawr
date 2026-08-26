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

**Layer A, kernel replay, every output buffer preallocated**, no allocation in any timed region:

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

**Report the residual.** It must be zero up to measurement noise; a non-zero residual means an arm is not
measuring what its name says.

**`B1 - B3` is normalization, not scan cost.** It includes run conversion and that conversion's
allocation. **Report the run-conversion count**; only a count of zero makes it scan cost.

### 3.1 How the numbers are computed

- Every term comes from **aggregate medians**: median of the ≥5 fresh-process medians, each process
  running 1 warmup and 7 timed cycles.
- **`explained_share` is computed from those aggregate medians**, never as a median or mean of per-process
  shares.
- **Ranges validate rather than compute.** Report the full range for every arm. The `explained_share`
  verdict holds only if recomputing it at the range endpoints keeps it on the same side of 0.70. If the
  endpoints straddle the threshold, the result is **inconclusive** and the cell is rerun.

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

**Zen 4:** report that AVX2 was selected at runtime via `croaring_hardware_support()`, not inferred from
architecture or build flags.

## 6. Reporting

**ns per pair** and **ns per input element**, not only per output element: an empty ANDNOT result makes
per-output undefined. Report the **input-size distribution**, which decides whether a vectorized kernel
could help at all.

## 7. Gate

**`explained_share` ≥ 0.70, evaluated independently per operation and per host**, validated by ranges per
§3.1.

- Below 0.70 for an operation on a host: its gap lives outside the matched-container path. **Stop for that
  operation**; the lever is top-level cloning, allocation, or result sizing, and that is a different spec.
- At or above: report which term dominates. That determines what Stage 2 is even about, and it may be
  different for OR and ANDNOT, or for the two hosts.

## Acceptance

- Six arms per §2, batched behind one call boundary.
- Apportionment per §3 with the **residual reported** and the **run-conversion count reported**.
- `explained_share` from aggregate medians, range-validated per §3.1; straddling endpoints reported as
  inconclusive.
- Pair accounting per §4, unmatched reported by behaviour.
- §5 host requirements met and stated in the artifacts.
- §6 reporting complete.
- **Verdict per operation per host**, with the dominant term named where the gate passes.
- **No production change. No kernel written. Stage 2 remains unwritten.**
- Existing suites and checks green; no board row moves.

## Estimate

**M** — six arms, exact apportionment, and pair accounting, reusing the spec 50 harness and corpus.
