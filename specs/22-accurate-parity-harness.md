<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 22: Accurate CRoaring parity benchmark harness

Rebuild the parity benchmark so **every published row is measured correctly**, and audit
them all — not patch individual suspicious ones. `run-compare-bench.sh` becomes the
canonical accurate runner producing the familiar full comparison table; the all-in-one
`bench_croaring` is retained only as a quick screening dashboard.

## Why

The parity effort (specs 16–21) repeatedly found the broad `bench_croaring` numbers were
**process-context artifacts, not real gaps**. Spec 20a proved it with a controlled matrix:
the harness's **rawr-SMP** rows carried *process-global `smp_allocator` history* (earlier
allocation-heavy groups) plus *warmup-protocol sensitivity* (2/9 → 3/21 moved rawr-SMP 27%),
while CRoaring (libc) escaped it — making the ratios **structurally asymmetric**. Two of the
three "gaps" (sparse AND, sparse OR) were phantoms; rawr is actually faster isolated. We could
only trust the real board by building **focused fresh-process** executables
(`bench_parity_isolated`, the and-cardinality diagnosis).

So the correct number for *any* row already required the isolated method. The lesson is not
"fix the three rows" — it is that an in-process, group-sequenced harness cannot be trusted for
parity, and **every row deserves the isolated treatment**. This spec operationalizes the
`docs/parity-measurement.md` recommendation into the standard runner.

## Target design (the eight requirements)

1. **Keep the existing benchmark workloads** — same corpora and operations. This changes *how*
   we measure, not *what*.
2. **Fresh process per operation or tightly related group.** `std.heap.smp_allocator` is a
   global singleton with no per-group reset (nor does libc), so process isolation is the only
   clean control against allocator-history bleed.
3. **Aggregate ≥ 5 process-level medians** — median of per-process medians, with range/IQR.
4. **Validate results outside the timed region** — correctness (result equal to the CRoaring
   oracle, or byte-identical where applicable) confirmed *before* a timing is accepted, never
   in the hot path.
5. **Compare allocating operations under matched allocator conditions** — report rawr-**SMP**
   (default) and rawr-**libc** side by side against CRoaring-libc; non-allocating ops report a
   single rawr number.
6. **Batch very fast operations and report `ns/op`** instead of `0.00 ms` — sub-clock ops
   (e.g. `addRange`, `cardinality`, `flip`, `removeRange`) batched above clock resolution.
7. **Identical data, semantics, CPU features, and timing boundaries** for rawr and CRoaring —
   same inputs, same `-Dcpu=native`, same warmup/timed protocol, same construct-vs-timed
   boundaries. Mind the known interop equivalences so both compute the same thing (`addRange`
   inclusive vs CRoaring exclusive → `end + 1`; `bitwiseDifference` ⇄ `andnot`).
8. **Retain `bench_croaring` only as a quick screening dashboard** — explicitly *not* the
   parity oracle.

**Outcome:** `run-compare-bench.sh` is the canonical accurate runner and prints the familiar
full comparison table, every row trustworthy for target selection.

## Proposed chunk plan (confirm at review)

- **`22-00` Harness architecture** — the fresh-process runner + ≥5-process aggregation +
  validation-outside-timing framework + the output-table format. The foundation the rest build
  on; generalizes `bench_parity_isolated` / `run-bench-parity-isolated.sh` into the standard
  runner.
- **`22-01` Workload audit** — enumerate **every** published `bench_croaring` row, port each to
  the isolated harness, and confirm or correct its real number. The "audit every row"
  deliverable; produces the trustworthy full table.
- **`22-02` Allocator parity** — matched-allocator handling: SMP and libc side by side for
  allocating ops (identical SMP-built inputs, only the result allocator varied), single rawr
  number for non-allocating ops.
- **`22-03` Tiny-operation calibration** — batching + `ns/op` for the sub-clock ops that
  currently read `0.00 ms`; define batch sizes that clear clock resolution.
- **`22-04` Cross-machine validation** — run the canonical table on the M4 and the x86-64 /
  Zen 4 host, identical features/protocol, so every row is validated on both architectures.

## Constraints

- **Benchmark-only** — no production/library change, no committed vendored-source change.
- **Do not silently change what is measured** — corpora and operation semantics stay as they
  are; only the measurement method changes. Any corpus that must change to enable batching or
  validation is called out explicitly.
- Record target / CPU / features in the header (spec 14 style); results committed to a durable
  artifact, not left in ignored `misc/`.

## Acceptance (toplevel; per-chunk gates in the chunks)

- `run-compare-bench.sh` produces the familiar full comparison table where **every row** is
  fresh-process, ≥5 process medians + range, validated outside timing, allocator-labeled
  (SMP/libc for allocating ops), and `ns/op` for tiny ops — with identical data / semantics /
  CPU features / timing boundaries for rawr and CRoaring.
- **Every published row audited**; the corrected numbers match the isolated method (e.g. sparse
  AND/OR show rawr at/ahead on default SMP), and no row remains that is a known process-context
  artifact.
- `bench_croaring` retained but explicitly demoted to a screening dashboard.
- The canonical table is cross-validated on M4 and Zen 4.

## Estimate

L. A harness rebuild plus a full row-by-row audit and cross-machine validation — five chunks,
sequenced with `22-00` first and the rest layered on the framework.
