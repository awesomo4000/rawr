<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 22-01: Workload audit — populate the manifest, port every row

Second chunk of [accurate parity harness](22-accurate-parity-harness.md). Uses `22-00`'s
schema and framework to bring **every published `bench_croaring` row** onto the isolated
harness. Produces the **complete functional table** — final trustworthiness is reached after
`22-02` (allocator completion) and `22-03` (tiny-op calibration).

## Gate

- `22-00` complete: manifest schema + per-tuple fresh-process runner + validation framework +
  output format, proven on the two pilot rows.

## The explicit inventory — all 38 dashboard rows

Acceptance is anchored to this exact list so "every row" is mechanically checkable (the current
dashboard publishes **38 rows**; lazy OR is **three** rows, range-cardinality is **two**, and
dense AND/OR and `addRange` are present):

1. `add (random 1M)` · 2. `add (sequential 1M)` · 3. `addMany (random 1M)` ·
4. `addMany (sequential 1M)` · 5. `addRange (1M)` · 6. `contains (hit)` · 7. `contains (miss)` ·
8. `bitwiseAnd (sparse)` · 9. `bitwiseAnd (sparse, arena)` · 10. `bitwiseAnd (dense)` ·
11. `bitwiseOr (sparse)` · 12. `bitwiseOr (sparse, arena)` · 13. `bitwiseOr (dense)` ·
14. `lazyOr+repair (sparse)` combined · 15. `lazyOr construction (sparse)` ·
16. `lazyOr repair (sparse)` · 17. `orMany (32 mixed)` · 18. `orManyHeap (32 mixed)` ·
19. `xorMany (32 mixed)` · 20. `bitwiseAnd (array balanced)` ·
21. `andCardinality (array balanced)` · 22. `bitwiseXor (array balanced)` ·
23. `bitwiseAnd (array skewed)` · 24. `andCardinality (array skewed)` · 25. `iterate (1M values)` ·
26. `toArray (1M values)` · 27. `toArrayAlloc (1M values)` · 28. `serialize` · 29. `deserialize` ·
30. `deserialize (arena)` · 31. `cardinality` · 32. `rank (dense)` · 33. `select (dense)` ·
34. `rankMany (dense)` · 35. `rangeCardinality small (bitset)` ·
36. `rangeCardinality large (bitset)` · 37. `flip wide range (dense)` ·
38. `removeRange wide (dense)`.

If the live dashboard differs from this list, reconcile against the dashboard (it is the
source) and record the corrected count — but the manifest must cover **exactly** the published
set, not a subset.

## Deliverables

- **Populate the manifest for all 38 rows** using `22-00`'s schema — each with its corpus/seed,
  matched op pair, allocating classification, allocator variants, oracle, and timing boundaries.
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

- The manifest is **complete** — **`--list` yields exactly the 38 rows above** (row-count check
  is mechanical); every published `bench_croaring` row has an entry.
- Every row is ported, runs per-tuple in ≥5 fresh processes, and is validated against its
  oracle outside the timed region; the complete functional table is produced.
- Rows that need allocator side-by-side or `ns/op` are **flagged in the manifest** for `22-02` /
  `22-03` (not left silently `0.00 ms` or single-allocator where the manifest says otherwise).
- **Benchmark-only:** no production/library or vendored-source change; build green under
  `ReleaseSafe` and `ReleaseFast`.

## Validation

- `zig build test`; `zig build -Doptimize=ReleaseSafe`; `zig build -Doptimize=ReleaseFast`
- **row-count check:** the worker `--list` mode emits exactly **38** rows matching the inventory
  above
- `scripts/run-compare-bench.sh` produces the complete functional table; every row shows a
  validated result (no unhandled/oracle-failing rows)

## Checklist

- [ ] Manifest populated for all 38 rows (each: corpus/seed, matched op pair, allocating class,
      allocator variants, oracle, timing boundaries)
- [ ] `--list` emits exactly 38 rows matching the inventory
- [ ] Every row ported; runs per-tuple in ≥5 fresh processes; validated outside timing
- [ ] Complete functional table produced
- [ ] Rows needing allocator side-by-side / `ns/op` flagged for `22-02` / `22-03`
- [ ] `zig build test`, ReleaseSafe, ReleaseFast all green; benchmark-only
