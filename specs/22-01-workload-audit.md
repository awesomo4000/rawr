<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 22-01: Workload audit — populate the manifest, port every row

Second chunk of [accurate parity harness](22-accurate-parity-harness.md). Uses `22-00`'s
schema and framework to bring **every published `bench_croaring` row** onto the isolated
harness. Produces the **complete functional table** — final trustworthiness is reached after
`22-02` (allocator completion) and `22-03` (tiny-op calibration).

## Gate

- `22-00` complete: manifest schema + per-tuple fresh-process runner + validation framework +
  output format, proven on the two pilot rows.

## Deliverables

- **Populate the manifest for every published row** using `22-00`'s schema — enumerate the full
  `bench_croaring` inventory: add / addMany (random + sequential), contains (hit/miss),
  bitwiseAnd/Or sparse (+ arena), lazyOr+repair, orMany / orManyHeap / xorMany, the
  array-balanced/skewed AND / andCardinality / xor rows, iterate / toArray / toArrayAlloc,
  serialize / deserialize (+ arena), cardinality, rank / select / rankMany, rangeCardinality,
  flip, removeRange — each with its corpus/seed, matched op pair, allocating classification,
  oracle, and timing boundaries.
- **Port each row to the isolated harness** and confirm or correct its real number; every row
  runs per-tuple in fresh processes with validation outside timing.
- Produce the **complete functional table** — every row present, validated, and measured; the
  allocator side-by-side completeness (`22-02`) and `ns/op` calibration (`22-03`) finish it.

## Sanity anchors, not gates

The prior isolated results (e.g. sparse AND/OR showing rawr at/ahead on default SMP;
`andCardinality` at parity after spec 21) are **sanity anchors** to catch a mis-port. A
correct harness is **allowed to report a changed number** — correctness is never gated on rawr
matching a previous figure.

## Acceptance

- The manifest is **complete** — every published `bench_croaring` row has an entry.
- Every row is ported, runs per-tuple in ≥5 fresh processes, and is validated against its
  oracle outside the timed region; the complete functional table is produced.
- Rows that need allocator side-by-side or `ns/op` are **flagged in the manifest** for `22-02` /
  `22-03` (not left silently `0.00 ms` or single-allocator where the manifest says otherwise).
- **Benchmark-only:** no production/library or vendored-source change; build green under
  `ReleaseSafe` and `ReleaseFast`.
