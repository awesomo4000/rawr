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

Both scripts build native `ReleaseFast` executables, retain individual process output under
`misc/`, and write an aggregate summary. The recorded runs are:

- `misc/lazy-context-20260722-204248-summary.txt`
- `misc/parity-isolated-20260722-184152-summary.txt`
