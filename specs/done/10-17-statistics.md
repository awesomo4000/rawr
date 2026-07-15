<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 10-17: `Roaring64Bitmap` statistics

Phase-2 parity chunk of [64-bit Roaring](10-roaring64.md). Aggregate introspection
over the container mix, mirroring CRoaring's `statistics`.

## Feature

| rawr 64-bit (new) | CRoaring | Semantics |
|---|---|---|
| `statistics() Statistics` | `roaring64_bitmap_statistics` | counts/sizes of containers by type + totals |

`*const`, allocation-free. Returns a `Statistics` struct.

## The `Statistics` struct

Mirror the fields CRoaring's `roaring64_statistics_t` exposes so the values are
directly comparable. At minimum:

- `n_containers: u64` (total sub-containers across all buckets)
- `n_array_containers` / `n_run_containers` / `n_bitset_containers: u64`
- `n_values_array_containers` / `_run_` / `_bitset_: u64` (cardinality by type)
- `n_bytes_array_containers` / `_run_` / `_bitset_: u64` (bytes by type)
- `n_buckets: u64` — **rawr-only** (number of high-key buckets). CRoaring's
  `roaring64_statistics_t` has **no** such field; include it for rawr introspection
  but **do not oracle it**.
- `cardinality: u64` (total)
- `min_value` / `max_value: u64`

Confirm the exact `roaring64_statistics_t` field set against `vendor/roaring.h`
and match names/units where practical so `difftest64` can compare the shared
fields field-by-field. `n_buckets` has no CRoaring counterpart and is excluded
from the oracle comparison.

## Implementation

Walk every bucket's sub-bitmap containers, tallying by `container.getType()`
(array/run/bitset), accumulating counts, per-type cardinalities, and per-type byte
sizes (reuse the container size accounting from `serializedSizeInBytes` / the
32-bit stats if one exists). `n_buckets = size`. `min_value`/`max_value` from
`minimum()`/`maximum()`.

## Wrapper decls

```c
typedef struct roaring64_statistics_s { /* fields per vendor/roaring.h */ } roaring64_statistics_t;
void roaring64_bitmap_statistics(const roaring64_bitmap_t *r, roaring64_statistics_t *stat);
```

## Tests / oracle

- Inline: known bitmaps with a controlled container mix (force array/run/bitset
  via cardinality + `addRange` + `runOptimize`); assert exact counts/cardinalities.
- `difftest64`: compare rawr `statistics()` field-by-field against
  `roaring64_bitmap_statistics` over the corpus, for the fields whose definitions
  match. **Byte-size fields may differ** if rawr's container layout differs from
  CRoaring's — compare only the layout-agnostic counts (container/value counts,
  cardinality, min/max) against the oracle; assert byte fields against rawr's own
  `serializedSizeInBytes`-derived expectations, not CRoaring's.

## Acceptance

- `Statistics` struct populated by a single container walk; `n_buckets` included.
- Count/cardinality/min/max fields oracled against CRoaring; byte fields checked
  against rawr's own accounting (documented divergence if layouts differ).
- Green; no 32-bit regression.
