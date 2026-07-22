<!-- SPDX-License-Identifier: MPL-2.0 -->

# Lazy-OR construction attribution

## Conclusion

The focused M4 benchmark does not find a CRoaring algorithm or loop-codegen advantage in
the shared-key lazy-OR path. With both implementations using libc allocation, rawr's
shared-key pipeline is at parity with CRoaring (2.816 ms versus 2.806 ms). The remaining
whole-operation gap is 0.359 ms, and unique-container cloning accounts for 0.355 ms of it.

Rawr with `std.heap.smp_allocator` takes 6.162 ms versus CRoaring's 3.601 ms. Changing only
the rawr result allocator to libc removes 2.202 ms, or 86% of that focused gap. The
controlled shared-key pipeline attributes at least 0.965 ms of the SMP gap to bitset
construction and accumulation. Isolated clear measurements put the memory-state-sensitive
zeroing excess between 0.507 and 2.087 ms. Header allocation, words allocation,
accumulation, and merge/append are individually at or near parity.

The dominant focused cost is therefore **SMP allocator/zeroing interaction while creating
16,364 separate 8 KB bitsets**, not accumulation or merge code. Under matched libc
allocation, the remaining actionable term is unique-container cloning.

The older all-in-one comparison harness reported a larger gap (rawr SMP 8.375 ms, rawr
libc 7.574 ms, CRoaring 3.832 ms). The focused executable does not reproduce the extra
rawr-only cost, even after an untimed lazy construction and repair immediately before each
sample. That additional cost is retained as a named **broad-harness context residual**.
The all-in-one executable touches several million-element corpora and runs earlier
allocation-heavy groups before lazy OR; this diagnosis does not establish whether its
residual comes from allocator state, executable layout, or another harness interaction.
It should not be treated as evidence for a library algorithm change.

## Corpus and validation

The benchmark recreates the sparse corpus from `bench_croaring.zig`: 500,000 deterministic
random `u32` values, sorted and deduplicated, with the left bitmap receiving the first half
and the right bitmap receiving values from the quarter point onward.

| Counter | Exact value |
| --- | ---: |
| Left chunk keys | 32,691 |
| Right chunk keys | 49,169 |
| Shared chunk keys | 16,364 |
| Left-only chunk keys | 16,327 |
| Right-only chunk keys | 32,805 |
| Lazy bitsets created | 16,364 |
| Bytes explicitly cleared | 134,053,888 |

Every shared source pair is array/array. Before timing is accepted, production rawr results
using SMP and libc are repaired and compared with each other. Their portable serialized
bytes are then compared byte-for-byte with a repaired CRoaring result.

The C attribution code is a benchmark-only translation unit. It includes the unmodified
vendored amalgamation and calls its actual internal allocation, conversion, accumulation,
clone, and append primitives. No vendored source is modified or copied into rawr-owned
code.

## Measurements

Environment: Zig 0.16.0, `ReleaseFast`, native Apple M4, macOS aarch64. Each process runs
two warmups and nine timed samples per row. The table reports the median of five independent
process medians and the range across those process medians. Setup and cleanup are outside
the timed interval, and no per-container clocks are used.

| Component | rawr SMP ms | rawr libc ms | CRoaring ms |
| --- | ---: | ---: | ---: |
| Full construction | 6.162 [5.819, 6.438] | 3.960 [3.848, 4.001] | 3.601 [3.500, 3.653] |
| Full after repair priming | 6.318 [6.204, 6.353] | 4.101 [3.973, 4.322] | 3.817 [3.624, 3.886] |
| Shared-key pipeline | 3.771 [3.289, 4.170] | 2.816 [2.710, 2.853] | 2.806 [2.793, 2.872] |
| Header allocation | 0.042 [0.042, 0.045] | 0.092 [0.089, 0.093] | 0.076 [0.069, 0.079] |
| Words allocation | 0.301 [0.280, 0.313] | 0.269 [0.262, 0.283] | 0.249 [0.247, 0.285] |
| Bitset create (combined) | 6.855 [6.739, 7.274] | 3.306 [3.111, 3.754] | 3.417 [3.289, 3.680] |
| Zero fresh allocation | 5.271 [4.775, 5.628] | 3.432 [3.301, 3.616] | 3.184 [3.166, 3.529] |
| Zero dirtied allocation | 3.765 [3.746, 4.007] | 3.564 [3.557, 3.745] | 3.258 [3.134, 3.543] |
| First accumulation | 0.574 [0.552, 0.621] | 0.566 [0.543, 0.592] | 0.625 [0.604, 0.682] |
| Second accumulation | 0.216 [0.205, 0.256] | 0.234 [0.206, 0.283] | 0.218 [0.189, 0.243] |
| Clone unique keys | 0.652 [0.629, 0.669] | 1.234 [1.196, 1.259] | 0.879 [0.875, 0.906] |
| Merge/append only | 0.073 [0.067, 0.075] | 0.072 [0.069, 0.073] | 0.100 [0.099, 0.102] |

Component microbenchmarks are bounds, not percentages forced to total 100%. In particular,
the zero rows deliberately expose two memory states and cannot be added to allocation or
pipeline rows. The combined bitset-create result being larger than the controlled pipeline
also demonstrates why isolated costs must not be summed as though cache and VM state were
identical.

For libc, the controlled differences explain the whole-operation gap closely:

- shared pipeline: rawr +0.010 ms;
- unique cloning: rawr +0.355 ms;
- merge/append: rawr -0.028 ms;
- named residual: +0.022 ms.

For SMP, shared pipeline (+0.965 ms), faster unique cloning (-0.227 ms), and faster merge
(-0.027 ms) leave a +1.850 ms **allocator-state/interleaving residual** within the focused
full operation. The whole-operation allocator substitution bounds that residual: libc
removes 2.202 ms without changing the algorithm or inputs.

## Code generation

Build command:

```sh
zig build bench-lazy-or-attribution -Dcpu=native
```

Stable probes are retained as these symbols:

- Zig: `_rawr_lazy_attr_zero_probe`, `_rawr_lazy_attr_accumulate_probe`
- C: `_rawr_cr_attr_zero_probe`, `_rawr_cr_attr_accumulate_probe`

Inspection commands on macOS:

```sh
nm -nm zig-out/bin/bench_lazy_or_attribution | \
  rg 'rawr_(lazy_attr|cr_attr)_(zero|accumulate)_probe'
otool -tvV -p _rawr_lazy_attr_zero_probe zig-out/bin/bench_lazy_or_attribution
otool -tvV -p _rawr_lazy_attr_accumulate_probe zig-out/bin/bench_lazy_or_attribution
otool -tvV zig-out/bin/bench_lazy_or_attribution | \
  rg -A20 '_rawr_cr_attr_(zero|accumulate)_probe'
```

Both 8 KB zero probes become a call/tail-call to `bzero` with length `0x2000`. Both
accumulation probes become the same scalar pattern: load a `u16`, derive the word offset,
load the word, shift a one-bit mask, OR, store, and loop. No unexpected scalar-versus-SIMD
asymmetry or extra Zig bounds check appears in the hot loop.

The preserved all-in-one benchmark and focused benchmark both call an out-of-line
`RoaringBitmap.lazyOr`; the timed wrappers do not inline the entire operation. Their
binary-specific `lazyOr` bodies differ in register assignment and global-input
specialization, but both retain the same allocator-vtable calls, `bzero`, and accumulation
structure. This inspection provides no codegen mechanism for the older broad-harness
residual.

## Reproduction

Run the five-process benchmark and aggregation with:

```sh
./scripts/run-bench-lazy-or-attribution.sh
```

The script builds `ReleaseFast` for the native CPU, saves each process output under
`misc/lazy-or-attribution-*-runN.txt`, and writes the aggregate summary beside it. The run
used for this report is `misc/lazy-or-attribution-20260722-134156-summary.txt`.

## Decision for 20-01

Do not change the lazy shared-key accumulation algorithm based on the original 2.19x
number. It is already at parity when allocation is matched, and its generated loops are
equivalent. Any follow-up should be narrowly scoped to one of these independently measured
targets:

1. decide whether the benchmark-of-record should compare CRoaring with rawr-libc as the
   allocation-matched result;
2. investigate SMP bitset allocation/zeroing as an allocator-specific concern; or
3. reduce libc unique-container clone overhead, with a separate benchmark gate.

The broad-harness context residual needs its own reproduction before it can justify a
production library change.
