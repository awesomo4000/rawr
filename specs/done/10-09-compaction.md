# Spec 10-09: `Roaring64Bitmap` runOptimize + shrinkToFit

Phase-2 parity chunk of [64-bit Roaring](10-roaring64.md). Two representation-
compaction ops, grouped because both just delegate per sub-bitmap and are tested
together.

## Features

| rawr 64-bit (new) | CRoaring | Semantics |
|---|---|---|
| `runOptimize() bool` | `roaring64_bitmap_run_optimize` | convert eligible sub-containers to RUN |
| `shrinkToFit() usize` | `roaring64_bitmap_shrink_to_fit` | release excess capacity; return bytes freed |

## `runOptimize` — return semantics (careful)

CRoaring's `run_optimize` returns **whether the result contains at least one run
container**, *not* whether anything changed. Decide rawr's contract explicitly and
state it in the doc comment. **Recommended:** match CRoaring — return "has ≥1 run
container after optimization" so the `difftest64` bool comparison against
`roaring64_bitmap_run_optimize` is a straight equality. (If instead we want
"changed?", we cannot oracle it against CRoaring's bool — so prefer the has-run
semantics for a clean oracle.)

Implementation: call `bm.runOptimize()` on every sub-bitmap; OR their "has-run"
results (or re-scan container types) to produce the aggregate bool. Invalidate
nothing cardinality-wise (run-optimize is cardinality-preserving).

## `shrinkToFit`

Release over-allocation: shrink the bucket array to `size`, and call each
sub-bitmap's shrink path. Return the total bytes reclaimed (sum of bucket-array
delta + per-sub-bitmap deltas) to mirror CRoaring's `size_t` return. Exact byte
accounting need not match CRoaring's number (allocators differ) — see oracle note.

## Wrapper decls

```c
bool roaring64_bitmap_run_optimize(roaring64_bitmap_t *r);   // already added in 10-04
size_t roaring64_bitmap_shrink_to_fit(roaring64_bitmap_t *r);
```

## Tests / oracle

- `runOptimize`: build run-friendly bitmaps (via `addRange`), assert the returned
  bool equals `roaring64_bitmap_run_optimize` on the parallel oracle, and assert
  set-equality is preserved (cardinality + membership unchanged, `equals` original).
- `shrinkToFit`: assert it's cardinality/membership-preserving and idempotent
  (second call frees 0). **Do not** assert the byte count equals CRoaring's — the
  allocators differ; only assert rawr's own invariants (no data loss, capacity ==
  size after). Run-optimize agreement is the CRoaring-oracled part.

## Acceptance

- `runOptimize` returns the has-run bool (documented), delegates per sub-bitmap,
  preserves the set; oracled bool-equal vs CRoaring in `difftest64`.
- `shrinkToFit` reclaims excess, preserves the set, is idempotent; rawr-only
  assertions (no CRoaring byte-count comparison).
- Green; no 32-bit regression.
