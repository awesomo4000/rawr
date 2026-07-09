# Spec 10-16: `Roaring64Bitmap` validate

Phase-2 parity chunk of [64-bit Roaring](10-roaring64.md). Structural
self-validation, mirroring 32-bit `RoaringBitmap.validate` and CRoaring's
`internal_validate`.

## Feature

| rawr 64-bit (new) | CRoaring | Semantics |
|---|---|---|
| `validate() ValidateError!void` | `roaring64_bitmap_internal_validate` | assert all structural invariants hold |

Returns a typed error on the first violation (mirror `RoaringBitmap.ValidateError`
style); `void` on success.

## Invariants to check

The 64-bit frame's invariants, then delegate the per-sub-bitmap checks:

1. **Keys strictly ascending & unique** — `buckets[i].hi < buckets[i+1].hi` for
   all `i` (the sorted-slice invariant). Violations → `UnsortedKeys` /
   `DuplicateKeys`.
2. **No empty buckets** — every `buckets[0..size].bm` is non-empty (the prune
   invariant). Violation → `EmptyBucket`.
3. **`size <= capacity`**, `buckets.len == capacity`. Violation → structural error.
4. **Cardinality cache consistency** — if `cached_cardinality` is non-null, it
   equals `Σ bucket.bm.cardinality()`. Violation → `CardinalityMismatch`.
5. **Each sub-bitmap valid** — delegate `bucket.bm.validate()` and propagate its
   `ValidateError` (this covers all the 32-bit container invariants for free).

Add a `Roaring64ValidateError` set (or reuse/extend `RoaringBitmap.ValidateError`
with the frame-level additions). `*const`, allocation-free.

## Wrapper decl

```c
bool roaring64_bitmap_internal_validate(const roaring64_bitmap_t *r, const char **reason);
```

## Tests / oracle

- Inline positive: validate passes on empties, single/many buckets, run-bearing,
  post-set-op, post-removeRange results.
- Inline negative (construct malformed internals in-test): unsorted keys,
  duplicate keys, a deliberately-empty bucket, a wrong `cached_cardinality` →
  assert the specific `ValidateError`.
- `difftest64`/`validate64`: every bitmap produced in the agreement loops calls
  `validate()` and must pass (a cheap invariant sweep piggybacking on the existing
  corpora). Optionally cross-check that CRoaring's `internal_validate` also passes
  on the oracle for the same inputs.

## Acceptance

- `validate` checks the frame invariants (sorted/unique keys, non-empty buckets,
  size/capacity, cache consistency) and delegates per sub-bitmap; typed errors.
- Wired into the differential corpora as an always-on invariant check.
- Green; no 32-bit regression.
