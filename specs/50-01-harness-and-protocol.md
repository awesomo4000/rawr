<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 50-01: Worker, controller, operations, digests

Toplevel: [50-realdata-comparison.md](50-realdata-comparison.md).
Gated on: [50-00](50-00-fetch-and-corpus.md).

**Builds the instrument; produces no reported comparison.** A smoke run to prove it executes is fine —
its numbers are not output. The two-host comparison is `50-02`.

## 1. Worker

`src/bench_realdata.zig` + build step `bench-realdata`, wired like `bench_croaring` — same
`addTranslatedCImport` and `addBenchmarkPlatformShim`. Dev tooling; **not** in `.paths`.

**One process runs exactly one `(implementation, dataset, operation)` cell.**

| op | rawr | CRoaring | ops/cycle |
| --- | --- | --- | --- |
| successive AND / OR / ANDNOT / XOR | `bitwiseAnd` / `Or` / `Difference` / `Xor` over `i, i+1` | `roaring_bitmap_and` / `_or` / `_andnot` / `_xor` | **199** |
| total union | `orMany` | `roaring_bitmap_or_many` | **1** |
| toArray | `toArray` into caller buffer | `roaring_bitmap_to_uint32_array` | **200** |
| serialize + deserialize | `serialize` → `deserialize` | `portable_serialize` → `portable_deserialize_safe` | **200** |

## 2. Construction — pinned, because it can fake a result

Container representation strongly affects OR and ANDNOT, so this is part of the experiment:

- rawr: `RoaringBitmap.fromSorted(alloc, values)`
- CRoaring: `roaring_bitmap_create()` + `roaring_bitmap_add_many(n, values)`
- **`runOptimize` is NOT called** on either side — a decision, not an omission. A run-optimized arm is a
  separate future variant.
- Archive order preserved (`50-00` §3).

These are each library's bulk path for sorted input and **need not produce identical container types**.
If histograms differ materially, that is a **finding to report**, not something to smooth over.

## 3. Allocators

**rawr: `std.heap.smp_allocator`. CRoaring: default libc `malloc`**, no memory hooks. Cross-allocator by
construction, and **this project's canonical pairing** — rawr takes a caller-supplied allocator, so it has
no inherent one. **State it in the report header.**

## 4. Timing boundaries

- **1 warmup cycle, then 7 timed cycles**; **true median** of the 7.
- **Before timing — only the unavoidable:** corpus load and parse, bitmap construction, caller output
  buffers, the pointer array for `orMany` / `or_many`.
- **Inside timing:** result allocation and teardown **where applicable**. **`toArray` is the exception** —
  it writes into a preallocated caller buffer and allocates nothing.
- **After timing, or in a separate process:** metadata, container histograms, semantic validation. **No
  additional validation, metadata, or reporting pass before the warmup/timed protocol.**

## 5. Semantic digests — after timing, compared in the controller

**FNV-1a 64, little-endian framed** — per result: `u32` **result index**, `u64` cardinality, then each
`u32` value ascending.

**Result index** is the pair index for adjacent-pair ops, the **source index** for `toArray` and
serialize+deserialize, and **zero** for total union.

**Computed AFTER all timed cycles.** A validation pass allocates and frees the same result shapes, so
running it first would condition SMP against the measured operation.

**Controller enforces:**

1. **rawr vs CRoaring digests match** — mismatch fails the run;
2. **repeat consistency within each implementation** across its ≥5 processes — an implementation
   disagreeing with itself means nondeterminism, and no timing from that cell is interpretable;
3. **corpus fingerprint identical** across every implementation and process for the dataset;
4. **source cardinality total identical**, and **container histogram identical within each implementation
   across repetitions** — these catch **setup nondeterminism** that no digest would reveal.

   **Source cardinality must be read from the constructed bitmap objects**, not from the parsed input
   count. The parsed count would agree even if construction dropped or duplicated values, which is
   precisely the failure this check exists to catch.

**Serialized bytes are reported, not required equal** — equivalent sets have multiple valid portable
encodings (spec 46-00). Each implementation deserializes and semantically validates its own output.

## 6. Controller and reporting

**`scripts/run-realdata-bench.sh`**, and its **worker manifest is the source of truth**: it enumerates
every `(implementation, dataset, operation)` cell, and the controller **validates the exact expected row
and process counts** before reporting. Without that, an omitted operation yields a plausible-looking
partial report rather than a failure.

- **≥5 processes** per cell; report **median of process medians** and **full range**.
- **Ratios from aggregates:** `median(rawr process medians) / median(CRoaring process medians)` —
  **never** a median or mean of per-run ratios.
- Report total cycle time **and** time ÷ denominator; denominators from §1.
- Header carries host, `ReleaseFast`, native CPU, allocator pairing, corpus fingerprint, container
  histograms.

## Acceptance

- Worker per §1, one cell per process; all seven operations on both sides.
- Construction per §2; `runOptimize` not called; histograms reported.
- Timing boundaries per §4, including the `toArray` exception.
- Digests per §5, **computed after timing**, with all four controller checks enforced and each
  demonstrated to fail on a seeded violation.
- Reporting per §6, ratios from aggregate medians.
- `bench-realdata` builds; not in `.paths`.
- No board row moves; existing suites and checks green.
- **No comparison result reported** — that is `50-02`.

## Verification record — implemented, reviewed, ACCEPTED

Added `src/bench_realdata.zig`, `scripts/run-realdata-bench.sh`,
`scripts/check-realdata-protocol.sh`, `scripts/validate-realdata-results.awk`.

**Checked here, not taken on report:**

| item | result |
| --- | --- |
| manifest shape | 63 lines = **21 ROW + 42 TUPLE** ✓ |
| controller count guard | `run-realdata-bench.sh:49` rejects anything but 21 rows / 42 tuples; `expected_processes = tuple_count * runs` ✓ |
| construction pinned | header prints `rawr=fromSorted, CRoaring=create+add_many, runOptimize=off` ✓ |
| ordering: timing then validation | `measureRawr` → `validateRawr` → `metadata()`; histogram computed after timing ✓ |
| `toArray` exception | output buffer allocated before the timed region; other ops allocate their validation buffer after ✓ |
| source cardinality | read from constructed bitmaps via `sources.metadata()`, compared against `corpus.total_values` ✓ |
| package | `check-package: OK (33 allowlisted files)`, `bench_realdata` not in `.paths` ✓ |

**All six seeded controller violations fire the intended guard** (`check-realdata-protocol.sh`).
`expect_failure` asserts the *expected* error name, so a run that failed for the wrong reason would not
pass:

```
caught cross-digest: DigestCrossImplementationMismatch
caught repeat-digest: DigestRepeatMismatch
caught fingerprint:  CorpusFingerprintMismatch
caught cardinality:  SourceCardinalityMismatch
caught histogram:    HistogramRepeatMismatch
caught process-count: ProcessCountMismatch
```

### The construction confound is resolved, and in rawr's favour

`50-02` §3 flagged unequal construction paths as a **live confound**. The histograms say they are not:

| dataset | arrays | bitsets | runs | both sides agree |
| --- | ---: | ---: | ---: | --- |
| uscensus2000 | 2221 | 0 | 0 | **yes** |
| census1881 | 1459 | 5 | 0 | **yes** |
| wikileaks-noquotes | 1892 | 0 | 0 | **yes** |

`fromSorted` and `create + add_many` produce **identical container representations** on all three corpora,
and both sides produce the same semantic digest. So the OR/ANDNOT comparison starts from the same state,
and **one of the three scratch-run confounds is now measured away** rather than argued away. `50-02` should
record this instead of repeating the warning.

### Accepted limitation

The six controls are all **controller-level**. The **timing-boundary properties** — validation after
timing, `toArray` buffer preallocated, no metadata pass before the warmup/timed protocol — are properties
of code order with no runtime guard that could catch a regression. They were verified by reading. Worth
knowing that a future edit could reorder them silently.

**No comparison result recorded**, per scope.

## Estimate

**M** — worker/controller split, cross-process enforcement, and the digest protocol.
