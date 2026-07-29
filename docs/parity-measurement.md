<!-- SPDX-License-Identifier: MPL-2.0 -->

# CRoaring parity measurement

## Verdict

The canonical parity runner is `./scripts/run-compare-bench.sh`. It measures every tuple in
fresh processes, reports the median and full range across five process medians, and shows both
rawr's default SMP allocator and libc when the timed operation allocates. The all-in-one
`bench_croaring` executable is retained only as a quick screening dashboard; its rawr-SMP
set-operation timings are sensitive to timing protocol and prior allocator history and are not
authoritative parity measurements.

The remaining broad sparse-AND and sparse-OR gaps are harness artifacts for rawr's default
allocator: both operations beat CRoaring when isolated. Their allocation-matched libc
variants remain slower than CRoaring and are valid, narrower allocator/code-path targets.
The corrected cross-machine board puts skewed `andCardinality` at parity. Remaining gaps are
operation- and architecture-specific and should be selected from the canonical per-host tables.

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
batch calibration belongs to spec 22-03. Cardinality uses independently calibrated batch counts:
Rawr caches the bitmap-wide total, while CRoaring scans and sums its containers on each call.
Both retain the same corpus and public operation, exceed the per-sample timing floor, and are
normalized to `ns/op`. The calibrated counts are 524288 calls for Rawr and 64 calls for
CRoaring. Their median timed samples were 1.312 ms and 2.254 ms on the M4, and 1.693 ms and
3.737 ms on Zen 4, respectively. A shared count based on Rawr had made each CRoaring sample
take about 19 seconds without improving the normalized comparison. The same controller passed
under Windows Git Bash on
Zen 4, where sparse AND measured 0.780 [0.778, 0.787] ms for rawr SMP, 2.102
[2.054, 2.164] ms for rawr libc, and 1.554 [1.522, 1.867] ms for CRoaring. These two rows prove
the worker and aggregation protocol.

The canonical manifest covers 39 rows, including the standalone dense-clone row added after
spec 26, and records the exact
corpus, operation pair, allocation class, timing boundaries, validation oracle, allocator
variants, and effective batch count for every row. Bitmap-producing operations validate portable
bytes; query, scalar, and array operations validate their exact outputs. Allocator parity and
tiny-operation calibration are complete.

Run the canonical accurate parity table:

```sh
./scripts/run-compare-bench.sh
```

Run the retained broad screening dashboard only when a quick, non-authoritative signal is useful:

```sh
./scripts/run-compare-bench.sh --dashboard
```

## Fixed-buffer serialization diagnosis

Captured 07/28/2026 with Zig 0.16.0, `ReleaseFast`, and the canonical one-million-value
random corpus. The factorial runner executes each cell in five fresh processes, reports the
median and full process range, validates bytes after timing against the unchanged
`serializeToWriter()` path and CRoaring, and records output and temporary allocations
separately.

| Host | Cell | Median ms [min, max] | Ratio to CRoaring |
| --- | --- | ---: | ---: |
| Apple M4 | temp tables + Writer | 1.182 [1.179, 1.205] | 1.144x |
| Apple M4 | direct construction + Writer | 1.505 [1.425, 1.593] | 1.457x |
| Apple M4 | temp tables + direct indexing | 1.120 [1.083, 1.156] | 1.084x |
| Apple M4 | direct construction + direct indexing | 1.069 [1.046, 1.211] | 1.035x |
| Apple M4 | CRoaring | 1.033 [0.991, 1.051] | 1.000x |
| Zen 4 | temp tables + Writer | 1.236 [1.058, 2.328] | 0.803x |
| Zen 4 | direct construction + Writer | 1.323 [1.300, 2.197] | 0.860x |
| Zen 4 | temp tables + direct indexing | 1.641 [0.922, 2.168] | 1.067x |
| Zen 4 | direct construction + direct indexing | 1.233 [1.061, 1.381] | 0.802x |
| Zen 4 | CRoaring | 1.538 [0.980, 1.764] | 1.000x |

The M4 factorial attributes the useful improvement to bypassing `std.Io.Writer`; removing
the temporary tables is beneficial only once output is direct. Direct construction through
the Writer is slower because it replaces bulk table writes with per-entry Writer calls. The
Zen 4 process ranges are wider, but direct construction plus direct output is neutral to the
legacy cell and does not reproduce the M4 allocation-removal regression from spec 27.

Production `serialize()` now writes descriptors, offsets, and container data directly into
its exactly-sized owned buffer. It performs one output allocation and no temporary table
allocations; the legacy cell performs the same output allocation plus two SMP allocations
totaling 524288 bytes on this corpus. `serializeToWriter()` remains unchanged for generic
writers and serves as the byte oracle.

The canonical board moved M4 rawr-SMP serialize from 1.128 ms to 1.068 ms (5.3%) and Zen 4
rawr-SMP serialize from 1.035 ms to 0.824 ms (20.4%). The matched all-row comparison was not
uniformly within 5%: untouched rawr and CRoaring rows moved on both hosts. The largest M4
movement was `rankMany`, from 166.875 to 203.875 ns/op, despite instruction-identical rawr
`rankMany` disassembly; its function and benchmark globals moved within the monolithic worker.
This is retained as a whole-binary code-layout sensitivity, not attributed to the serializer
algorithm, and means the strict all-row regression gate did not pass cleanly.

Artifacts: `misc/serialize-diag-20260729-061550-summary.txt` (M4),
`misc/serialize-diag-20260729-061649-summary.txt` (Zen 4),
`misc/parity-20260728-203158-summary.txt` (M4 production), and
`misc/parity-20260728-204017-summary.txt` (Zen 4 production).

## Canonical cross-machine tables

Captured 07/24/2026 from the canonical runner. Each value is the median of five
independent process medians, with the full process range in brackets. A ratio below 1.0 means
rawr is faster; a ratio above 1.0 means CRoaring is faster.

### Apple M4

```text
Accurate Rawr vs CRoaring parity table
======================================
Processes per tuple: 5
# rawr bench env (compiled target)
# zig 0.16.0 | ReleaseFast | macos aarch64
# cpu: apple_m4 | features: neon
# array-intersect kernel: neon

# requested-cpu: native
# protocol: 3w/21t median
# croaring-avx512: off

operation                    variant          unit     rawr median [min,max]         unit       CR median [min,max]    ratio
---------------------------- -------- ------------ ------------------------- ------------ ------------------------- --------
add (random 1M)              smp                ms  264.115 [257.741,284.541]           ms  284.943 [284.420,287.696]    0.9269x
add (random 1M)              libc               ms  264.471 [263.396,280.098]           ms  284.943 [284.420,287.696]    0.9282x
add (sequential 1M)          smp                ms    3.241 [  3.228,  3.288]           ms    3.179 [  3.174,  3.239]      1.02x
add (sequential 1M)          libc               ms    3.278 [  3.230,  3.398]           ms    3.179 [  3.174,  3.239]     1.031x
addMany (random 1M)          smp                ms  245.073 [244.381,247.649]           ms  286.728 [286.047,289.252]    0.8547x
addMany (random 1M)          libc               ms  243.666 [242.862,244.941]           ms  286.728 [286.047,289.252]    0.8498x
addMany (sequential 1M)      smp                ms    2.184 [  2.169,  2.222]           ms    2.029 [  2.024,  2.033]     1.076x
addMany (sequential 1M)      libc               ms    2.180 [  2.161,  2.190]           ms    2.029 [  2.024,  2.033]     1.074x
addRange (1M)                smp             ns/op  239.624 [234.131,241.699]        ns/op  334.839 [328.613,344.238]    0.7156x
addRange (1M)                libc            ns/op  486.206 [484.253,500.610]        ns/op  334.839 [328.613,344.238]     1.452x
contains (hit)               default            ms   86.885 [ 86.155, 88.877]           ms   91.760 [ 89.986, 96.603]    0.9469x
contains (miss)              default            ms   83.805 [ 83.398, 84.605]           ms   81.432 [ 81.196, 82.689]     1.029x
bitwiseAnd (sparse)          smp                ms    0.600 [  0.590,  0.602]           ms    0.678 [  0.675,  0.692]     0.885x
bitwiseAnd (sparse)          libc               ms    0.962 [  0.952,  1.051]           ms    0.678 [  0.675,  0.692]     1.419x
bitwiseAnd (sparse, arena)   arena              ms    0.564 [  0.559,  0.566]           ms    0.678 [  0.675,  0.692]    0.8319x
bitwiseAnd (dense)           smp             ns/op  305.176 [300.415,316.040]        ns/op  169.189 [166.992,175.293]     1.804x
bitwiseAnd (dense)           libc            ns/op  302.856 [296.265,306.885]        ns/op  169.189 [166.992,175.293]      1.79x
bitwiseOr (sparse)           smp                ms    1.758 [  1.738,  1.760]           ms    2.390 [  2.382,  2.438]    0.7356x
bitwiseOr (sparse)           libc               ms    3.216 [  3.187,  3.259]           ms    2.390 [  2.382,  2.438]     1.346x
bitwiseOr (sparse, arena)    arena              ms    1.572 [  1.558,  1.584]           ms    2.390 [  2.382,  2.438]    0.6577x
bitwiseOr (dense)            smp             ns/op  407.593 [394.409,415.283]        ns/op  315.430 [310.547,324.219]     1.292x
bitwiseOr (dense)            libc            ns/op  507.446 [493.286,522.705]        ns/op  315.430 [310.547,324.219]     1.609x
lazyOr+repair (sparse)       smp                ms   14.735 [ 14.415, 15.446]           ms   12.643 [ 12.504, 13.172]     1.165x
lazyOr+repair (sparse)       libc               ms   12.995 [ 12.984, 13.394]           ms   12.643 [ 12.504, 13.172]     1.028x
lazyOr construction (sparse) smp                ms    5.918 [  5.772,  6.003]           ms    3.479 [  3.422,  3.625]     1.701x
lazyOr construction (sparse) libc               ms    3.808 [  3.764,  3.889]           ms    3.479 [  3.422,  3.625]     1.095x
lazyOr repair (sparse)       smp                ms    8.265 [  8.184,  8.624]           ms    8.388 [  8.152,  8.483]    0.9853x
lazyOr repair (sparse)       libc               ms    8.152 [  8.068,  8.391]           ms    8.388 [  8.152,  8.483]    0.9719x
orMany (32 mixed)            smp             ns/op 14507.812 [14171.875,14578.125]        ns/op 11476.562 [11382.812,11640.625]     1.264x
orMany (32 mixed)            libc            ns/op 15031.250 [14945.312,15195.312]        ns/op 11476.562 [11382.812,11640.625]      1.31x
orManyHeap (32 mixed)        smp             ns/op 14265.625 [14148.438,14460.938]        ns/op 27945.312 [27640.625,28164.062]    0.5105x
orManyHeap (32 mixed)        libc            ns/op 14796.875 [14757.812,14859.375]        ns/op 27945.312 [27640.625,28164.062]    0.5295x
xorMany (32 mixed)           smp             ns/op 16226.562 [16140.625,16343.750]        ns/op 29625.000 [29421.875,29703.125]    0.5477x
xorMany (32 mixed)           libc            ns/op 16867.188 [16804.688,16914.062]        ns/op 29625.000 [29421.875,29703.125]    0.5694x
bitwiseAnd (array balanced)  smp             ns/op 140562.500 [139187.500,141750.000]        ns/op 199687.500 [196562.500,199750.000]    0.7039x
bitwiseAnd (array balanced)  libc            ns/op 150812.500 [149625.000,152312.500]        ns/op 199687.500 [196562.500,199750.000]    0.7552x
andCardinality (array balanced) default         ns/op 113812.500 [112406.250,114625.000]        ns/op 186281.250 [183593.750,186812.500]     0.611x
bitwiseXor (array balanced)  smp             ns/op 211750.000 [210250.000,223250.000]        ns/op 420625.000 [410875.000,447500.000]    0.5034x
bitwiseXor (array balanced)  libc            ns/op 227000.000 [214250.000,227250.000]        ns/op 420625.000 [410875.000,447500.000]    0.5397x
bitwiseAnd (array skewed)    smp             ns/op 22156.250 [21171.875,22460.938]        ns/op 44429.688 [42726.562,45132.812]    0.4987x
bitwiseAnd (array skewed)    libc            ns/op 25289.062 [24875.000,26101.562]        ns/op 44429.688 [42726.562,45132.812]    0.5692x
andCardinality (array skewed) default         ns/op 12710.938 [12429.688,13328.125]        ns/op 13078.125 [12507.812,13890.625]    0.9719x
iterate (1M values)          default            ms    2.148 [  2.132,  2.181]           ms    3.045 [  2.995,  3.077]    0.7054x
toArray (1M values)          default            ms    0.895 [  0.887,  0.934]           ms    1.041 [  1.020,  1.063]    0.8598x
toArrayAlloc (1M values)     smp                ms    1.081 [  1.065,  1.117]           ms    1.086 [  1.066,  1.116]    0.9954x
toArrayAlloc (1M values)     libc               ms    0.899 [  0.873,  0.927]           ms    1.086 [  1.066,  1.116]    0.8278x
serialize                    smp                ms    1.118 [  1.101,  1.148]           ms    0.996 [  0.992,  1.021]     1.122x
serialize                    libc               ms    1.004 [  0.997,  1.017]           ms    0.996 [  0.992,  1.021]     1.008x
deserialize                  smp                ms    1.500 [  1.488,  1.512]           ms    2.474 [  2.416,  2.508]    0.6063x
deserialize                  libc               ms    3.029 [  2.997,  3.086]           ms    2.474 [  2.416,  2.508]     1.224x
deserialize (arena)          arena              ms    1.344 [  1.326,  1.385]           ms    2.474 [  2.416,  2.508]    0.5432x
cardinality                  default         ns/op    2.487 [  2.478,  2.520]        ns/op 35703.125 [31359.375,37906.250] 6.966e-05x
rank (dense)                 default            ms   10.583 [ 10.245, 10.817]           ms    9.246 [  9.115,  9.381]     1.145x
select (dense)               default            ms   15.385 [ 14.319, 15.682]           ms   10.168 [ 10.045, 10.356]     1.513x
rankMany (dense)             default         ns/op 179375.000 [179000.000,180625.000]        ns/op 171375.000 [165125.000,172500.000]     1.047x
rangeCardinality small (bitset) default            ms   12.391 [ 12.178, 12.526]           ms  104.211 [103.542,104.680]    0.1189x
rangeCardinality large (bitset) default            ms   63.121 [ 63.031, 63.208]           ms  122.276 [121.619,122.494]    0.5162x
flip wide range (dense)      smp             ns/op  644.653 [629.028,650.024]        ns/op  364.746 [356.689,374.878]     1.767x
flip wide range (dense)      libc            ns/op 1107.544 [1088.745,1127.197]        ns/op  364.746 [356.689,374.878]     3.036x
removeRange wide (dense)     smp             ns/op  506.470 [495.850,509.644]        ns/op  233.765 [228.027,255.005]     2.167x
removeRange wide (dense)     libc            ns/op  949.097 [940.308,992.554]        ns/op  233.765 [228.027,255.005]      4.06x
```

### AMD Zen 4

```text
Accurate Rawr vs CRoaring parity table
======================================
Processes per tuple: 5
# rawr bench env (compiled target)
# zig 0.16.0 | ReleaseFast | windows x86_64
# cpu: znver4 | features: sse2 ssse3 sse4_2 avx avx2
# array-intersect kernel: x86-simd

# requested-cpu: native
# protocol: 3w/21t median
# croaring-avx512: off

operation                    variant          unit     rawr median [min,max]         unit       CR median [min,max]    ratio
---------------------------- -------- ------------ ------------------------- ------------ ------------------------- --------
add (random 1M)              smp                ms  211.464 [201.769,286.547]           ms  311.024 [305.836,315.348]    0.6799x
add (random 1M)              libc               ms  252.455 [236.778,280.167]           ms  311.024 [305.836,315.348]    0.8117x
add (sequential 1M)          smp                ms    4.883 [  4.846,  5.012]           ms    4.665 [  4.638,  4.787]     1.047x
add (sequential 1M)          libc               ms    4.949 [  4.880,  5.095]           ms    4.665 [  4.638,  4.787]     1.061x
addMany (random 1M)          smp                ms  204.611 [200.666,320.908]           ms  327.175 [325.502,329.054]    0.6254x
addMany (random 1M)          libc               ms  274.471 [263.651,309.483]           ms  327.175 [325.502,329.054]    0.8389x
addMany (sequential 1M)      smp                ms    3.546 [  3.524,  3.669]           ms    2.772 [  2.760,  2.867]     1.279x
addMany (sequential 1M)      libc               ms    3.609 [  3.568,  3.709]           ms    2.772 [  2.760,  2.867]     1.302x
addRange (1M)                smp             ns/op  340.771 [338.281,392.773]        ns/op  824.500 [816.125,859.241]    0.4133x
addRange (1M)                libc            ns/op 1136.108 [1098.694,1146.655]        ns/op  824.500 [816.125,859.241]     1.378x
contains (hit)               default            ms   94.230 [ 93.852,136.079]           ms   98.460 [ 94.823,131.049]     0.957x
contains (miss)              default            ms   89.033 [ 88.552, 89.780]           ms   87.749 [ 86.316, 90.912]     1.015x
bitwiseAnd (sparse)          smp                ms    0.817 [  0.804,  0.844]           ms    1.585 [  1.562,  1.672]    0.5157x
bitwiseAnd (sparse)          libc               ms    2.114 [  2.085,  2.351]           ms    1.585 [  1.562,  1.672]     1.334x
bitwiseAnd (sparse, arena)   arena              ms    0.991 [  0.966,  1.016]           ms    1.585 [  1.562,  1.672]    0.6254x
bitwiseAnd (dense)           smp             ns/op  249.976 [249.121,260.608]        ns/op  447.571 [445.325,468.799]    0.5585x
bitwiseAnd (dense)           libc            ns/op  483.093 [477.905,494.373]        ns/op  447.571 [445.325,468.799]     1.079x
bitwiseOr (sparse)           smp                ms    2.127 [  2.097,  2.200]           ms    7.756 [  7.432,  8.133]    0.2742x
bitwiseOr (sparse)           libc               ms    8.673 [  8.499,  8.936]           ms    7.756 [  7.432,  8.133]     1.118x
bitwiseOr (sparse, arena)    arena              ms    2.591 [  2.549,  2.605]           ms    7.756 [  7.432,  8.133]    0.3341x
bitwiseOr (dense)            smp             ns/op  382.019 [368.433,458.032]        ns/op  824.670 [795.276,836.133]    0.4632x
bitwiseOr (dense)            libc            ns/op  989.685 [973.206,1022.351]        ns/op  824.670 [795.276,836.133]       1.2x
lazyOr+repair (sparse)       smp                ms   35.418 [ 35.257, 35.474]           ms   99.748 [ 92.705,100.904]    0.3551x
lazyOr+repair (sparse)       libc               ms   95.316 [ 94.256, 99.649]           ms   99.748 [ 92.705,100.904]    0.9556x
lazyOr construction (sparse) smp                ms   20.305 [ 20.283, 20.468]           ms   63.958 [ 63.646, 64.356]    0.3175x
lazyOr construction (sparse) libc               ms   61.784 [ 61.508, 62.231]           ms   63.958 [ 63.646, 64.356]     0.966x
lazyOr repair (sparse)       smp                ms   14.419 [ 14.212, 14.516]           ms   26.653 [ 26.384, 26.699]     0.541x
lazyOr repair (sparse)       libc               ms   27.510 [ 27.448, 27.611]           ms   26.653 [ 26.384, 26.699]     1.032x
orMany (32 mixed)            smp             ns/op 21116.406 [20811.719,21533.594]        ns/op 23290.625 [23207.031,24167.969]    0.9066x
orMany (32 mixed)            libc            ns/op 21810.938 [21554.688,22414.062]        ns/op 23290.625 [23207.031,24167.969]    0.9365x
orManyHeap (32 mixed)        smp             ns/op 20964.844 [20775.000,21914.062]        ns/op 56441.406 [56397.656,57492.188]    0.3714x
orManyHeap (32 mixed)        libc            ns/op 21827.344 [21679.688,22510.156]        ns/op 56441.406 [56397.656,57492.188]    0.3867x
xorMany (32 mixed)           smp             ns/op 14942.188 [14840.625,15561.719]        ns/op 71638.281 [70246.875,72682.812]    0.2086x
xorMany (32 mixed)           libc            ns/op 15731.250 [15467.969,16083.594]        ns/op 71638.281 [70246.875,72682.812]    0.2196x
bitwiseAnd (array balanced)  smp             ns/op 149243.750 [148756.250,154087.500]        ns/op 197268.750 [194306.250,204800.000]    0.7566x
bitwiseAnd (array balanced)  libc            ns/op 311837.500 [307631.250,323156.250]        ns/op 197268.750 [194306.250,204800.000]     1.581x
andCardinality (array balanced) default         ns/op 99968.750 [99146.875,100709.375]        ns/op 46275.000 [45615.625,47040.625]      2.16x
bitwiseXor (array balanced)  smp             ns/op 256812.500 [254925.000,262875.000]        ns/op 706750.000 [696900.000,732800.000]    0.3634x
bitwiseXor (array balanced)  libc            ns/op 522687.500 [514675.000,534237.500]        ns/op 706750.000 [696900.000,732800.000]    0.7396x
bitwiseAnd (array skewed)    smp             ns/op 14833.594 [14780.469,15251.562]        ns/op 26667.188 [26521.094,27381.250]    0.5562x
bitwiseAnd (array skewed)    libc            ns/op 23779.688 [23566.406,25166.406]        ns/op 26667.188 [26521.094,27381.250]    0.8917x
andCardinality (array skewed) default         ns/op 8489.844 [8312.500,8704.688]        ns/op 8093.750 [8048.438,8374.219]     1.049x
iterate (1M values)          default            ms    3.343 [  3.313,  3.352]           ms    3.409 [  3.371,  3.459]    0.9806x
toArray (1M values)          default            ms    1.067 [  1.056,  1.117]           ms    0.938 [  0.903,  0.950]     1.138x
toArrayAlloc (1M values)     smp                ms    1.438 [  1.363,  1.499]           ms    1.365 [  1.293,  1.472]     1.053x
toArrayAlloc (1M values)     libc               ms    1.450 [  1.392,  1.966]           ms    1.365 [  1.293,  1.472]     1.062x
serialize                    smp                ms    1.028 [  1.004,  1.073]           ms    0.953 [  0.939,  0.985]     1.078x
serialize                    libc               ms    1.044 [  1.041,  1.112]           ms    0.953 [  0.939,  0.985]     1.095x
deserialize                  smp                ms    1.729 [  1.722,  1.809]           ms    5.372 [  5.248,  6.133]    0.3219x
deserialize                  libc               ms    8.376 [  8.318,  8.447]           ms    5.372 [  5.248,  6.133]     1.559x
deserialize (arena)          arena              ms    1.789 [  1.775,  1.844]           ms    5.372 [  5.248,  6.133]     0.333x
cardinality                  default         ns/op    3.258 [  3.228,  3.362]        ns/op 56087.500 [48943.750,56675.000] 5.809e-05x
rank (dense)                 default            ms   12.012 [ 11.932, 12.131]           ms   11.296 [ 11.243, 12.260]     1.063x
select (dense)               default            ms   13.197 [ 12.551, 13.384]           ms   11.043 [ 10.966, 11.066]     1.195x
rankMany (dense)             default         ns/op 245437.500 [244337.500,255550.000]        ns/op 249575.000 [249025.000,257912.500]    0.9834x
rangeCardinality small (bitset) default            ms   12.850 [ 12.813, 12.880]           ms   48.398 [ 48.100, 49.597]    0.2655x
rangeCardinality large (bitset) default            ms   45.858 [ 45.776, 46.300]           ms   58.469 [ 58.253, 58.651]    0.7843x
flip wide range (dense)      smp             ns/op  816.345 [790.015,850.708]        ns/op 1445.679 [1405.200,1460.547]    0.5647x
flip wide range (dense)      libc            ns/op 2307.788 [2192.615,2357.837]        ns/op 1445.679 [1405.200,1460.547]     1.596x
removeRange wide (dense)     smp             ns/op  684.155 [658.545,769.983]        ns/op  634.937 [612.183,683.875]     1.078x
removeRange wide (dense)     libc            ns/op 2091.699 [2071.094,2146.216]        ns/op  634.937 [612.183,683.875]     3.294x
```


## Iteration model attribution

The original canonical `iterate` row compared rawr's idiomatic pull iterator with CRoaring's
push-style `roaring_iterate` callback API. A four-path, fresh-process diagnosis replaced that
model mismatch with symmetric measurements: rawr pull, benchmark-only rawr direct push,
CRoaring pull in a C wrapper, and CRoaring push with a C callback. Every path traversed the same
999,893-value sorted sequence and produced the same count, wrapping sum, and untimed rolling
hash. Both implementations reported exactly 65,536 array containers and no bitset or run
containers.

Results below are median nanoseconds per deduplicated value across five independent process
medians; brackets show the complete process range.

| Host | Path | ns/value | Interpretation |
| --- | --- | ---: | --- |
| Apple M4 | rawr pull | 2.126 [2.117, 2.164] | Idiomatic public pull iterator. |
| Apple M4 | rawr push diagnostic | 0.911 [0.883, 0.950] | Inline comptime sink; benchmark-only. |
| Apple M4 | CRoaring pull | 2.997 [2.974, 3.134] | Complete pull loop inside C. |
| Apple M4 | CRoaring push | 1.598 [1.583, 1.646] | C runtime callback. |
| AMD Zen 4 | rawr pull | 3.458 [3.431, 4.154] | Idiomatic public pull iterator. |
| AMD Zen 4 | rawr push diagnostic | 1.064 [1.057, 1.094] | Inline comptime sink; benchmark-only. |
| AMD Zen 4 | CRoaring pull | 3.491 [3.384, 3.593] | Complete pull loop inside C. |
| AMD Zen 4 | CRoaring push | 2.188 [2.168, 2.226] | C runtime callback. |

The like-for-like pull ratios are 0.709x on M4 and 0.991x on Zen 4, both within the 1.10x
decision gate. The original 1.52x and 1.88x gaps were benchmark-model artifacts, so no
`Iterator.next()` optimization or public push API was added. The canonical row now compares
pull with pull and validates the same iterator paths it times. The push comparison remains
diagnostic rather than a parity claim because rawr's comptime sink inlines while CRoaring uses a
runtime function pointer.

## Select attribution

The original `select` row made one Zig-to-C call per query while rawr's call could inline into
the Zig timing loop. The corrected row gives both implementations one non-inlined benchmark
boundary per query: rawr calls a `noinline` wrapper containing an inlined
`RoaringBitmap.select`, while CRoaring runs the query loop in C and calls
`roaring_bitmap_select` there. Untimed validation compares all one million results plus empty,
boundary, and out-of-range ranks. M4 disassembly confirms the rawr loop calls its wrapper once,
the wrapper contains the select body, and the C loop calls `roaring_bitmap_select` once. The
same sources and validation are used for the Zen 4 build.

The diagnosis uses the canonical seed and draw sequence. Both implementations produce eight
run containers for values `0..499999`. Query ranks span `0..499999`, average 250254.666, and
reach every container. Results are median nanoseconds per query across five independent process
medians; brackets show the full process range.

| Host | Path | Before fix ns/query | After fix ns/query |
| --- | --- | ---: | ---: |
| Apple M4 | rawr forced inline | 14.274 [13.727, 15.217] | 13.432 [13.058, 14.199] |
| Apple M4 | rawr public boundary | 15.674 [15.379, 16.011] | 14.335 [13.837, 14.939] |
| Apple M4 | CRoaring from Zig | 10.020 [9.979, 10.543] | 10.206 [9.780, 10.601] |
| Apple M4 | CRoaring loop in C | 10.318 [9.833, 10.490] | 10.347 [9.845, 10.519] |
| AMD Zen 4 | rawr forced inline | 12.847 [12.773, 14.173] | 11.526 [11.325, 12.697] |
| AMD Zen 4 | rawr public boundary | 16.589 [14.995, 18.372] | 12.725 [12.611, 14.606] |
| AMD Zen 4 | CRoaring from Zig | 11.255 [11.166, 11.519] | 11.372 [11.272, 12.004] |
| AMD Zen 4 | CRoaring loop in C | 10.963 [10.953, 11.044] | 10.961 [10.925, 11.078] |

The pre-fix M4 rawr cost split was 9.308 ns/query in the top-level container skip, 1.854 in
the target run container, and 2.856 in the named fusion/code-generation residual after the
0.256 ns/query checksum baseline. Container traversal was the dominant cost. The retained fix
specializes the top-level walk on the tagged container type, uses a `u32` remaining rank, and
indexes array containers directly. It preserves the existing bitset and run selection kernels.
An extra helper around every container and a directly integrated run loop were tested and
rejected because each regressed the measured path.

The focused post-fix public-boundary ratios are 1.385x on M4 and 1.161x on Zen 4. In the full
canonical board they are 1.513x and 1.195x respectively. The change is a reproducible
improvement, but it does not meet the 1.10x phase-2 target. The remaining dominant lever is an
indexed or cached prefix-cardinality lookup; that would add mutation-maintenance and storage
costs to every bitmap and is deferred rather than introduced for this eight-container corpus.

## M4 cluster attribution

The six rows initially grouped as a possible bitset/NEON code-generation cluster do not share a
bitset kernel. The dense inputs are built with `addRange` and contain eight and nine run
containers. Sparse lazy-OR contains 32,691 and 49,169 array containers. The `orMany` corpus has
192 input containers: 96 arrays, 48 bitsets, and 48 runs. Untimed counting uses the real output
allocator and excludes input construction.

The focused phase measurements below are medians across five fresh processes. They are rawr A/B
attributions, not additive models: independently timed allocator and traversal phases can overlap,
and their sum is not claimed to equal a full operation.

| Row | M4 attribution | Zen 4 check | Decision |
| --- | --- | --- | --- |
| Dense AND | Canonical 305.176 ns; five matching run-container operations dominate. Full construction makes 14 allocations/320 bytes. A direct container sweep is 355.469 ns, with the difference named the scratch-allocation/code-layout residual because production uses its fixed scratch allocator. | Direct sweep 131.055 ns; full diagnostic 215.723 ns; canonical rawr is 0.56x CRoaring. | No bitset or NEON lever. A fix would require changing run-result allocation/layout. |
| Dense OR | Canonical 407.593 ns. Matching run unions take 257.812 ns; unmatched run clones, top-level traversal, and result arrays make up the remaining full-operation work. The result makes 30 allocations/760 bytes. | Direct sweep 134.961 ns; full diagnostic 323.145 ns; canonical rawr is 0.46x CRoaring. | Independent run/clone path; no architecture-neutral change supported. |
| Flip | Canonical 644.653 ns. The mask-based implementation makes 77 allocations/1,887 requested bytes. Mask construction is 172.852 ns and mutation of pre-cloned inputs is 407.715 ns; the standalone clone is 371.582 ns. Their non-additive overlap is the allocator/code-layout residual. | Full diagnostic 765.479 ns while canonical rawr is 0.56x CRoaring. | Closing M4 requires replacing the clone-plus-mask algorithm, not tuning a word loop. |
| Remove range | Canonical 506.470 ns. Clone-plus-mask difference makes 70 allocations/1,552 bytes. Pre-cloned removal is 362.793 ns; the same mask and clone controls apply. | Full diagnostic 653.857 ns and canonical ratio 1.078x. | Same algorithmic rewrite as flip, with no demonstrated cross-host win. |
| Sparse lazy-OR construction | Canonical 5.918 ms and 130,994 allocations/137,172,592 requested bytes. Direct matched-container allocation and array `setList` accumulation is 1.486 ms, leaving a named 4.432 ms top-level/result-allocation residual. | Direct accumulation 14.053 ms; full diagnostic 19.632 ms; canonical rawr is 0.32x CRoaring. | This is the previously identified transient-container allocation path, not a bitset word kernel. No default allocator change is reopened. |
| `orMany` | Canonical 14.508 microseconds; focused full 14.711 and mixed-container accumulation 14.180 microseconds, leaving 0.531 microseconds for result allocation, repair, and surrounding traversal. It makes 17 allocations/49,624 bytes. | Full 21.056 and accumulation 20.038 microseconds; canonical rawr is 0.91x CRoaring. | Opposite host behavior and no isolated codegen defect; retain as a named M4 mixed-accumulation residual. |

The kernel hypotheses were tested independently even though they do not drive the dense rows.
For a 1,024-word bitset on M4, width 8 is fastest: AND with cardinality is 138.672 ns at width
8 versus 187.256 at width 4 and 353.760 at width 2. Eliding cardinality lowers width 8 to
123.779 ns, only 14.893 ns. Lazy OR is 122.314 ns at width 8 versus 151.611 and 229.004 ns;
`countWords` is 98.877 ns versus 169.922 and 326.416 ns. Zen 4 also selects width 8: AND with
cardinality is 80.420 ns and without cardinality is 81.030 ns, while production count is
32.764 ns. A per-architecture width or eager card/no-card split therefore has no supported
connection to the six canonical gaps.

M4 disassembly of `bench_m4_cluster_diag.diagnosticWordOp__anon_21532` shows each width-8
iteration as four 128-bit NEON loads, `and.16b` operations, and stores. Cardinality adds
`cnt.16b`, `udot.4s`, and `uadalp.2d`; the no-card specialization omits them. Production
`simdBitsetOp` and `simdBitsetOpLazy` inline to the same loops. CRoaring's
`CROARING_USENEON` path in `vendor/roaring.c` is likewise explicitly unrolled over four
128-bit vectors per eight words, including the same byte-count and pairwise-reduction shape.
The x86_64 assembly emitted with `x86_64_v3` uses vector OR/AND and an AVX2 nibble-lookup
software popcount; width 8 remains fastest in the Zen measurements.

Codegen inspection commands:

```sh
lldb -b -o 'target create zig-out/bin/bench_m4_cluster_diag' \
  -o 'disassemble -n bench_m4_cluster_diag.diagnosticWordOp__anon_21532' \
  -o 'disassemble -n bench_m4_cluster_diag.diagnosticWordOp__anon_21535' \
  -o 'disassemble -n bench_m4_cluster_diag.diagnosticCount__anon_21549'

zig build-exe -target x86_64-macos -OReleaseFast -mcpu x86_64_v3 \
  --dep rawr -Mroot=src/bench_m4_cluster_diag.zig \
  -OReleaseFast -Mrawr=src/roaring.zig \
  -femit-asm=/tmp/bench_m4_cluster_diag_x86.s -fno-emit-bin
```

Phase 2 is a **NO-GO**. The original grouping was a representation mistake, width 8 is already
the best measured choice on both hosts, and the rows split across run allocation, clone-plus-mask
range algorithms, transient sparse allocations, and mixed-container accumulation. No production
kernel change is justified by this diagnosis. The benchmark and evidence remain so a future
range-algorithm or run-allocation initiative can start from measured components.

## Direct range operation allocation baseline (07/25/2026)

The `bench-range-alloc` target isolates allocations performed by the three range entry points on
the canonical dense input (`addRange(0, 499999)`) and inclusive range `[100000, 650000]`. Input
construction is outside the measurement region. By-value `flip` allocates its result through the
counting allocator; the in-place rows reset counters after constructing their input.

```sh
zig build bench-range-alloc
./zig-out/bin/bench_range_alloc
```

| Legacy operation | Allocations | Frees during operation | Requested bytes | Peak measured live bytes |
| --- | ---: | ---: | ---: | ---: |
| `flip` | 77 | 65 | 1,887 | 1,179 |
| `flipInplace` | 57 | 63 | 1,447 | 1,179 |
| `removeRange` | 50 | 62 | 1,112 | 992 |

These are M4 ReleaseFast structural baselines, not timing results. The larger previously recorded
`removeRange` composition count (70 allocations / 1,552 bytes) includes cloning the input before
the in-place operation.

The retained direct implementation produces the following counts on the same probe:

| Direct operation | Allocations | Frees during operation | Requested bytes | Peak measured live bytes |
| --- | ---: | ---: | ---: | ---: |
| `flip` | 32 | 20 | 660 | 300 |
| `flipInplace` | 30 | 36 | 680 | 632 |
| `removeRange` | 2 | 14 | 40 | 440 |

The mask bitmap is gone from both in-place operations, and by-value flip constructs its result
directly instead of cloning the whole input before applying a mask.

### Cross-host direct range decision

The canonical `flip` and clone-plus-`removeRange` rows were run in five fresh processes per
strategy and implementation. Values are median nanoseconds per operation with the full rawr
process range in brackets; ratios compare direct rawr SMP with CRoaring.

| Host | Row | Legacy rawr | Direct rawr | CRoaring | Direct ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| M4 | flip | 675.415 [664.185, 705.078] | 360.229 [344.971, 368.042] | 379.272 | 0.950x |
| M4 | clone + removeRange | 518.555 [495.728, 519.287] | 426.392 [421.753, 442.627] | 231.689 | 1.840x |
| Zen 4 | flip | 1098.621 [995.288, 1243.164] | 397.205 [385.889, 406.360] | 1725.537 | 0.230x |
| Zen 4 | clone + removeRange | 763.477 [751.111, 789.001] | 290.039 [274.390, 304.541] | 705.579 | 0.411x |

Direct is the shipped implementation on every architecture. Flip reaches parity on M4 and is
substantially faster on Zen 4. The M4 remove row retains a supported 17.8% rawr improvement even
though its clone-inclusive comparison remains above CRoaring; direct removal itself is reduced to
two edge-container allocations. The production legacy composition was removed. A test-only copy
remains solely to enforce portable-byte identity across the range matrix.

The M4 broad-run regression check covered every unaffected row through deserialization, followed
by fresh five-process checks of cardinality, rank, select, rankMany, and both range-cardinality
rows. No unaffected rawr/CRoaring ratio worsened by more than 5%; the few absolute shifts above 5%
were shared by both implementations and treated as host noise.

## Clone and removeRange attribution (07/27/2026)

The canonical board now includes a standalone deep-copy row on the same eight-run-container
corpus as the clone-inclusive `removeRange` row. CRoaring copy-on-write is disabled, so both
implementations perform a deep copy. Portable bytes from each rawr clone are checked against its
source and CRoaring outside timing.

| Host | Row | rawr SMP ns/op | rawr libc ns/op | CRoaring ns/op |
| --- | --- | ---: | ---: | ---: |
| M4 | clone | 379.883 [333.984, 394.165] | 293.457 [288.940, 302.490] | 207.275 [202.881, 242.432] |
| M4 | clone + `removeRange` | 438.354 [414.429, 449.707] | 327.271 [319.702, 337.280] | 230.591 [225.830, 234.253] |
| Zen 4 | clone | 201.892 [197.192, 203.210] | 572.095 [544.751, 717.688] | 558.154 [555.200, 609.314] |
| Zen 4 | clone + `removeRange` | 251.013 [240.503, 254.980] | 642.786 [612.109, 719.495] | 612.598 [597.925, 639.758] |

The attribution diagnostic measures clone, mutation, and clone-plus-mutation bodies separately,
with destruction outside timing. A timer-only control quantifies the two clock reads in each
sample. The following values are directly measured medians; ranges are the full range across five
process medians.

| Host | Component | rawr SMP ns/op | rawr libc ns/op | CRoaring ns/op |
| --- | --- | ---: | ---: | ---: |
| M4 | timer control | 17.944 [17.090, 18.433] | 17.822 [16.724, 17.944] | 17.822 [17.700, 17.944] |
| M4 | clone body | 272.339 [242.676, 296.509] | 170.776 [162.109, 199.341] | 114.746 [112.671, 119.019] |
| M4 | remove body | 67.749 [65.918, 68.481] | 123.413 [120.361, 130.737] | 96.313 [90.576, 98.145] |
| M4 | clone + remove body | 311.401 [306.274, 317.139] | 301.392 [292.114, 320.190] | 200.073 [197.632, 207.397] |
| Zen 4 | timer control | 27.771 [27.515, 27.856] | 27.637 [27.490, 27.722] | 27.734 [27.637, 28.320] |
| Zen 4 | clone body | 154.736 [152.832, 173.499] | 407.959 [395.532, 439.380] | 320.142 [316.870, 330.518] |
| Zen 4 | remove body | 110.095 [106.799, 167.676] | 200.671 [200.134, 205.908] | 158.130 [154.138, 168.250] |
| Zen 4 | clone + remove body | 281.140 [256.421, 308.276] | 572.620 [567.542, 582.654] | 449.780 [447.937, 466.040] |

The untimed inventory is identical on both hosts: both sides have eight run containers. A rawr
clone makes 20 allocations, requests 440 bytes, and copies 48 payload bytes; CRoaring makes 18
allocations, requests 288 bytes, and copies 56 payload bytes. Payload copying is therefore not the
cause of the M4 gap. Rawr's extra top-level capacity and per-container allocation traffic are the
structural difference, and the M4 SMP/libc split shows that allocator behavior amplifies it.

After subtracting the timer control, the M4 rawr-SMP clone body is estimated at 254.395 ns versus
96.924 ns for CRoaring. The mutation body is 49.805 ns versus 78.491 ns, so rawr's direct mutation
is faster. The reduced-result teardown estimate is 144.897 ns versus 48.340 ns. The body-level
interaction residual is -10.743 ns for rawr and 6.836 ns for CRoaring. These independently
measured medians are diagnostic estimates and are not expected to add exactly. They attribute the
207.763 ns canonical composite difference primarily to clone work and reduced-result teardown,
not to `removeRange` mutation.

Zen 4 reaches the opposite default-allocator outcome: the timer-corrected rawr-SMP clone body is
126.965 ns versus 292.408 ns for CRoaring, and the canonical composite is 0.410x CRoaring. The
rawr-libc clone remains slower at an estimated 380.322 ns, confirming that allocator behavior is
architecture-specific while the allocation-count and layout difference is common. The follow-up
should be a clone-specific optimization spec targeting top-level capacity, per-container
allocation, and teardown. The direct range algorithm should remain unchanged.

Complete 39-row canonical boards passed the regression gate on M4 and Zen 4. No pre-existing row
worsened by more than 5% against a fresh post-spec-26 baseline after range-overlap reruns.

### Clone optimization follow-up (07/28/2026)

The Phase 1 experiment replaced clone's initial-capacity allocation and immediate growth with an
exact-capacity initialization. It produced the predicted inventory reduction from 20 allocations
and 440 requested bytes to 18 allocations and 400 requested bytes. It did not pass the performance
gate. On M4, two independent five-process runs measured rawr-SMP clone body at 407.715
[395.996, 421.021] and 430.420 [406.616, 451.050] ns/op, substantially worse than the 26a result
of 272.339 [242.676, 296.509] ns/op. Rawr-libc improved to approximately 132-144 ns/op, confirming
that reducing the nominal allocation count is not sufficient to predict SMP allocator behavior.
The direct-capacity change was reverted.

The required allocation-failure sweep exposed a separate correctness defect: when a later
container clone failed, the clone error path did not record already cloned containers in
`result.size`, so `errdefer` leaked them. The retained fix sets the partial size only on the error
branch before returning. Every injected allocation failure is leak-free and leaves the source
portable bytes unchanged. Its success path is neutral: the retained M4 diagnostic measured clone
body at 246.216 [230.835, 260.132] ns/op, while focused canonical reruns measured clone at
362.305 ns/op versus the 364.868 ns/op baseline and clone-plus-`removeRange` at 429.443 ns/op
versus 427.856 ns/op. `flipDirect` already initializes its exact result capacity and has no
clone-style init-then-grow waste.

The retained fix is also neutral on Zen 4. Clone body measured 158.508
[154.443, 166.699] ns/op versus 154.736 [152.832, 173.499] before the fix, with overlapping
ranges. Focused canonical reruns measured clone at 197.253 ns/op versus 198.730 ns/op and
clone-plus-`removeRange` at 250.439 ns/op versus 253.381 ns/op.

The conditional deeper-layout analysis also produced NO-GO results before implementation. Zig
0.16's SMP allocator rounds small allocations to power-of-two slots with an 8-byte minimum:

| Run capacity | Split struct + payload slots | Combined slot | Result |
| ---: | ---: | ---: | --- |
| 1-4 (minimum capacity 4) | 32 + 16 = 48 | 64 | worse |
| 8 | 32 + 32 = 64 | 64 | neutral |
| 16 | 32 + 64 = 96 | 128 | worse |
| 32 | 32 + 128 = 160 | 256 | worse |
| 64 | 32 + 256 = 288 | 512 | worse |
| 128 | 32 + 512 = 544 | 1024 | worse |

Some non-power-of-two capacities have narrow favorable windows, but the one-run target corpus and
the normal doubling sequence cross into larger classes. A clone-only combined layout would also
need allocation-mode metadata and a migration design when a mutable clone grows; converting all
run containers would require pointer-changing growth or a broader handle redesign. The failed
size-class gate makes that ownership work unjustified.

Combined top-level key/container storage fails the same gate. At capacities 1, 2, 4, 8, and 16,
the separate SMP slot totals are 16, 24, 40, 80, and 160 bytes; combined requests round to 16, 32,
64, 128, and 256 bytes. Implementing it would additionally touch `initCapacity`, `deinit`, normal
growth, `shrinkToFit`, in-place set-operation replacement, and range growth. It is rejected before
that blast radius is incurred.

Spec 27 therefore closes without a clone layout optimization. The M4 clone gap remains a documented
architecture-specific SMP allocator residual; the direct range algorithm remains unchanged. The
allocation-failure cleanup is retained as an independent correctness fix.

Complete pre/post 39-row boards passed on both hosts. The M4 after-table initially flagged
array-skewed AND/libc and select; five-process reruns reduced their ratio changes to 3.2% and 4.2%.
The Zen 4 after-table initially flagged addMany-random/libc, serialize, and clone/libc. Reruns put
addMany below 5%, kept serialize inside the baseline process ranges, and improved clone/libc
against its rerun reference. No regression is attributable to the retained change.

## Recommendation

Use the canonical runner for performance decisions. Keep `bench_croaring` only as a quick broad
regression dashboard, and confirm any signal there in the canonical table before opening an
optimization spec. For allocating operations, report default rawr-SMP and allocation-matched
rawr-libc side by side. For non-allocating operations, report a single rawr result.

Use the per-host canonical tables above when selecting optimization work. Skewed
`andCardinality` is now at parity on both reference hosts. Sparse AND and OR need no
default-allocator optimization spec; their remaining libc gaps are relevant only to an
explicitly allocation-matched investigation.

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

Build and run the four-path iteration diagnosis:

```sh
./scripts/run-bench-iterate-diag.sh
```

Build and run the select call-boundary and attribution diagnosis:

```sh
./scripts/run-bench-select-diag.sh
```

Build and run the architecture-specific component diagnosis:

```sh
./scripts/run-bench-m4-cluster-diag.sh
```

Build and run the clone/removeRange attribution diagnosis:

```sh
./scripts/run-bench-range-attrib.sh
```

These scripts build native `ReleaseFast` executables, retain individual process output under
`misc/`, and write an aggregate summary. The recorded runs are:

- `misc/lazy-context-20260722-204248-summary.txt`
- `misc/parity-isolated-20260722-184152-summary.txt`
- `misc/and-cardinality-diag-20260723-011446-summary.txt`
- `misc/and-cardinality-diag-20260723-014315-summary.txt` (M4, fused kernel)
- `misc/parity-isolated-20260723-014621-summary.txt` (M4, fused kernel)
- `misc/and-cardinality-diag-20260723-192153-summary.txt` (Zen 4, fused kernel)
- `misc/iterate-diag-20260724-094909-summary.txt` (M4)
- `misc/iterate-diag-20260724-152701-summary.txt` (Zen 4)
- `misc/select-diag-20260725-085940-summary.txt` (M4, retained fix)
- `misc/select-diag-20260725-090021-summary.txt` (Zen 4, retained fix)
- `misc/parity-20260725-090712-summary.txt` (M4, corrected canonical row)
- `misc/parity-20260725-091537-summary.txt` (Zen 4, corrected canonical row)
- `misc/m4-cluster-diag-20260725-111229-summary.txt` (M4)
- `misc/m4-cluster-diag-20260725-111417-summary.txt` (Zen 4)
- `misc/range-attrib-20260727-182905-summary.txt` (M4)
- `misc/range-attrib-20260727-183135-summary.txt` (Zen 4)
- `misc/parity-20260728-032923-summary.txt` (M4 pre-change baseline)
- `misc/parity-20260728-033704-summary.txt` (Zen 4 pre-change baseline)
- `misc/range-attrib-20260728-035544-summary.txt` (M4 direct-capacity NO-GO confirmation)
- `misc/range-attrib-20260728-041330-summary.txt` (M4 retained cleanup fix)
- `misc/range-attrib-20260728-054509-summary.txt` (Zen 4 retained cleanup fix)
- `misc/parity-20260728-060853-summary.txt` (M4 retained-fix board)
- `misc/parity-20260728-061732-summary.txt` (Zen 4 retained-fix board)
