<!-- SPDX-License-Identifier: MPL-2.0 -->

# CRoaring parity measurement

## Verdict

The all-in-one `bench_croaring` executable is useful for broad screening, but its rawr-SMP
set-operation timings are sensitive to timing protocol and prior allocator history. It
must not be the sole basis for selecting optimization targets. Performance decisions
should use a focused executable in a fresh process, report five-process median and range,
and show both rawr's default SMP allocator and libc when the timed operation allocates.

The remaining broad sparse-AND and sparse-OR gaps are harness artifacts for rawr's default
allocator: both operations beat CRoaring when isolated. Their allocation-matched libc
variants remain slower than CRoaring and are valid, narrower allocator/code-path targets.
The skewed `andCardinality` gap survives isolation and is the clearest general parity
target because the timed operation does not allocate.

## Broad-harness residual

The controlled matrix recreates the sparse lazy-OR corpus and keeps all broad benchmark
groups linked behind a runtime selector. Every process validates repaired rawr-SMP and
rawr-libc results against CRoaring portable bytes before timing. Results below are the
median of five independent process medians with the full range in brackets.

Environment: Zig 0.16.0, `ReleaseFast`, native Apple M4, macOS aarch64.

| Condition | Protocol | rawr SMP ms | rawr libc ms | CRoaring ms |
| --- | --- | ---: | ---: | ---: |
| Focused executable | 2/9 | 5.980 [5.906, 11.643] | 3.933 [3.851, 7.357] | 3.626 [3.510, 6.844] |
| Broad binary, target-only init | 2/9 | 6.046 [5.939, 11.681] | 3.800 [3.786, 7.074] | 3.547 [3.369, 7.009] |
| Broad binary, target-only init | 3/21 | 4.438 [4.335, 8.154] | 3.849 [3.794, 4.039] | 3.502 [3.468, 4.124] |
| Broad binary, full init, target first | 3/21 | 4.340 [3.906, 5.175] | 4.191 [4.006, 7.357] | 3.740 [3.520, 4.106] |
| Broad binary, full init, target last | 3/21 | 7.909 [7.425, 15.832] | 4.171 [4.058, 7.076] | 3.796 [3.686, 7.121] |
| Full init plus allocator-only prime | 3/21 | 5.369 [5.264, 5.979] | 4.200 [4.111, 5.471] | 3.788 [3.648, 4.131] |
| Full init plus cache-touch-only prime | 3/21 | 4.297 [3.918, 6.652] | 4.179 [4.027, 7.726] | 3.731 [3.627, 4.365] |

The comparisons isolate these effects:

- **Code layout:** focused versus broad target-only at 2/9 is effectively unchanged for
  all three variants. Keeping the broad functions linked does not create the residual.
- **Timing protocol:** changing only 2/9 to 3/21 lowers rawr-SMP by 1.608 ms (27%). It has
  no material effect on rawr-libc or CRoaring. More warmup and sampling reaches a different
  steady allocator state.
- **Unrelated initialization:** full initialization with the target first changes rawr-SMP
  by -0.098 ms. The small libc and CRoaring increases are below the 10% materiality gate.
- **Execution history:** moving the target last raises rawr-SMP by 3.569 ms (82%), with
  non-overlapping five-process ranges. Rawr-libc and CRoaring are unchanged.
- **Allocator versus cache:** a synthetic allocator-only prime recreates 1.029 ms of the
  execution-history increase, while a cache-only corpus walk recreates none. The remaining
  2.540 ms is named the **workload-specific SMP allocator-history residual**: the synthetic
  size-class prime bounds but does not reproduce the complete allocation/free sequence of
  the earlier groups.

The rawr-SMP target-last median, 7.909 ms, is close to the anchored historical 8.375 ms;
CRoaring similarly reproduces at 3.796 versus 3.832 ms. The historical rawr-libc result of
7.574 ms does not reproduce: the controlled target-last median is 4.171 ms even after the
earlier rawr-libc benchmark operations and compile-time allocator specialization are
restored in their historical order. That 3.403 ms discrepancy is retained as a
**historical libc harness residual**, not attributed to production rawr.

The practical cause of the reproducible broad distortion is process-global SMP allocator
state, amplified by the benchmark protocol and prior allocation-heavy groups. Per-group
allocator reset is not available for `std.heap.smp_allocator`; fresh-process focused runs
are the correct control.

## Corrected parity board

The focused board recreates the exact broad sparse and skewed-array corpora. Sparse result
bytes from rawr-SMP and rawr-libc are compared with CRoaring portable bytes, and skewed
cardinality is compared directly, before any timing is accepted.

| Target | Variant | Broad ms | Isolated ms | Isolated ratio to CRoaring | Read |
| --- | --- | ---: | ---: | ---: | --- |
| Sparse AND | rawr SMP | 0.881 | 0.618 [0.589, 0.622] | 0.91x | Broad gap is an artifact; rawr is faster isolated. |
| Sparse AND | rawr libc | 1.242 | 0.967 [0.956, 0.995] | 1.42x | Smaller but real allocation-matched gap. |
| Sparse AND | CRoaring | 0.687 | 0.679 [0.659, 0.691] | 1.00x | Stable across harnesses. |
| Sparse OR | rawr SMP | 2.827 | 1.789 [1.727, 1.834] | 0.75x | Broad gap is an artifact; rawr is faster isolated. |
| Sparse OR | rawr libc | 4.539 | 3.237 [3.162, 3.258] | 1.36x | Smaller but real allocation-matched gap. |
| Sparse OR | CRoaring | 2.450 | 2.378 [2.343, 2.442] | 1.00x | Stable across harnesses. |
| Skewed `andCardinality` | rawr | 0.020 | 0.019 [0.018, 0.019] | 1.46x | Real non-allocating algorithm/kernel gap. |
| Skewed `andCardinality` | CRoaring | 0.014 | 0.013 [0.012, 0.014] | 1.00x | Stable across harnesses. |

Ratios use the isolated five-process medians. The broad columns above use the anchored
medians from commit `0599cae`; their original ranges remain recorded in spec `20a-01`.

## Skewed `andCardinality` attribution

The focused diagnosis separates the original full-API case from direct array-container
kernel calls. It uses the exact 200-container bitmap shape from the parity board (180
matching keys, `32x4096` all-hit arrays), validates both implementations' container types
and cardinalities before timing, and batches the direct calls above one millisecond. The
direct sweep covers all-hit, disjoint, and deterministic half-hit distributions across
the ratio, fixed-ratio, extreme-small-side, and dispatch-boundary cases in spec 21.

Environment: Zig 0.16.0, `ReleaseFast`, native Apple M4, macOS aarch64. Results are the
median of five independent process medians with the full process range in brackets.

| Layer | rawr | CRoaring | rawr / CRoaring |
| --- | ---: | ---: | ---: |
| Full API, original corpus | 0.020 [0.020, 0.021] ms | 0.014 [0.013, 0.014] ms | 1.43x |
| Forced gallop, per matching array pair | 75.561 [74.157, 80.810] ns | 42.968 [40.191, 45.288] ns | 1.76x |
| Normal array dispatch, per matching pair | 77.758 [73.486, 80.749] ns | 42.449 [40.313, 44.891] ns | 1.83x |

The full-API difference is 6.000 microseconds. The direct dispatch difference is 35.309 ns
per matching pair, or 6.356 microseconds across 180 matching containers. The resulting
-0.356 microsecond **full-API measurement residual** is smaller than the full-API clock and
process ranges. Amortizing the full time over the matching pairs leaves 33.353 ns/pair of
rawr traversal and 35.329 ns/pair of CRoaring traversal after subtracting direct dispatch.
Those figures also include the unmatched-key work, but show that surrounding traversal is
at parity. The measurable gap is therefore in the galloping cardinality kernel; dispatch
adds no material cost at the original point.

The forced-gallop result generalizes. Representative rawr/CRoaring ratios are 2.09x at
`8x1024` all-hit, 1.62x at `16x2048` all-hit, 1.76x at `32x4096` all-hit, and 1.24x-1.40x
for their mixed counterparts. Disjoint ratios are larger, but the absolute CRoaring times
are only about 5-6 ns and rawr remains below 29 ns in the measured matrix. The one-element
mixed case is effectively parity. This is not specific to the original cardinalities or
all-hit distribution.

The boundary rows confirm dispatch selection rather than implicating it in the original
case. Rawr changes from NEON to gallop at its inclusive 40:1 threshold; CRoaring changes to
gallop only above its strict 64:1 threshold. Forced cross-checks show that the best boundary
choice depends on overlap distribution, but both implementations already select gallop at
the original 128:1 point.

Source and generated-code inspection identifies a concrete implementation difference for
the follow-up fix spec. Rawr applies the reusable `gallopSearch` lower-bound routine for
every small-side value and then performs a separate equality check. CRoaring uses a fused
two-array state machine with cached current values and an inlined `advanceUntil` primitive
that has immediate-next and exact-match exits. The M4 disassembly preserves that structure;
the benchmark establishes its throughput advantage, though it does not assign the cost to
one individual instruction.

### Fused-kernel result

Commit `5dae1e1` replaced the count-only generic-search loop with the fused cached-cursor
state machine while leaving write kernels, dispatch, and thresholds unchanged. On the M4,
the original `32x4096` all-hit forced-gallop result moved from 75.561 ns/container to
43.426 ns/container, equal to CRoaring's 43.426 ns/container median. The focused parity
board reports both full APIs at 0.014 ms median (rawr range 0.014-0.015 ms, CRoaring range
0.013-0.014 ms).

The shared kernel was also validated natively on Windows x86-64 / Zen 4. At the original
point rawr measured 41.958 [39.086, 42.123] ns/container versus CRoaring at
42.755 [42.742, 43.051] ns/container. The full API measured 0.009 [0.008, 0.009] ms for
rawr and 0.008 [0.008, 0.008] ms for CRoaring. The direct generalization matrix remained
neutral-to-better across all-hit, disjoint, and mixed distributions on both hosts.

Unit tests, the explicit empty/singleton/reversed-argument kernel cases, `bench_aa`,
CRoaring differential validation, and full `ReleaseSafe` and `ReleaseFast` builds passed
on the M4 and Zen 4 hosts. The skewed cardinality gap is closed at the kernel level; the
remaining sub-microsecond full-API difference on Zen 4 is at the benchmark clock scale.

## Accurate harness rollout

The spec 22 harness replaces process-shared parity measurements incrementally. Its worker
owns a machine-readable row manifest, exposes rows and tuples through `--list`, and accepts
exactly one `(row, implementation, allocator)` tuple per process. Each process uses a fixed
3-warmup/21-timed protocol, validates only after timing, and emits one structured result after
validation succeeds. The controller runs five independent processes per tuple and reports the
median of process medians with the full process range.

The initial architecture pilot covered sparse AND (rawr SMP, rawr libc, and CRoaring libc) and
cardinality (rawr and CRoaring). On the Apple M4 pilot, sparse AND measured 0.581
[0.577, 0.612] ms for rawr SMP, 0.931 [0.917, 0.935] ms for rawr libc, and 0.690
[0.680, 0.752] ms for CRoaring. Cardinality is normalized to `ns/op`; its final cross-host
batch calibration belongs to spec 22-03. The same controller passed under Windows Git Bash on
Zen 4, where sparse AND measured 0.780 [0.778, 0.787] ms for rawr SMP, 2.102
[2.054, 2.164] ms for rawr libc, and 1.554 [1.522, 1.867] ms for CRoaring. These two rows prove
the worker and aggregation protocol.

Spec 22-01 extends that architecture to all 38 rows published by `bench_croaring`. The manifest
now records the exact corpus, operation pair, allocation class, timing boundaries, validation
oracle, and remaining allocator/calibration work for every row. Bitmap-producing operations
validate portable bytes; query, scalar, and array operations validate their exact outputs. The
38-row table is functionally complete, but allocator side-by-side completion and tiny-operation
calibration remain assigned to specs 22-02 and 22-03.

During the rollout, run the complete functional table explicitly:

```sh
./scripts/run-compare-bench.sh --parity
```

Without `--parity`, the script retains the existing broad screening-dashboard behavior
until the complete manifest becomes canonical in spec 22-04.

## Recommendation

Keep `bench_croaring` as a broad regression dashboard, but qualify performance gaps in a
focused fresh-process executable before opening an optimization spec. For allocating
operations, report default rawr-SMP and allocation-matched rawr-libc side by side. For
non-allocating operations, report a single rawr result. Do not combine allocator variants
or unrelated operation groups in one process when the number will drive parity work.

Based on the corrected board, prioritize skewed `andCardinality` for general CRoaring
parity. Sparse AND and OR need no default-allocator optimization spec; their remaining libc
gaps are relevant only to an explicitly allocation-matched investigation.

## Reproduction

Build and run the controlled five-process matrix:

```sh
./scripts/run-bench-lazy-context.sh
```

Build and run the focused corrected parity board:

```sh
./scripts/run-bench-parity-isolated.sh
```

Build and run the five-process skewed `andCardinality` diagnosis:

```sh
./scripts/run-bench-and-cardinality-diag.sh
```

These scripts build native `ReleaseFast` executables, retain individual process output under
`misc/`, and write an aggregate summary. The recorded runs are:

- `misc/lazy-context-20260722-204248-summary.txt`
- `misc/parity-isolated-20260722-184152-summary.txt`
- `misc/and-cardinality-diag-20260723-011446-summary.txt`
- `misc/and-cardinality-diag-20260723-014315-summary.txt` (M4, fused kernel)
- `misc/parity-isolated-20260723-014621-summary.txt` (M4, fused kernel)
- `misc/and-cardinality-diag-20260723-192153-summary.txt` (Zen 4, fused kernel)
