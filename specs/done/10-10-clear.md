<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 10-10: `Roaring64Bitmap` clear

Phase-2 parity chunk of [64-bit Roaring](10-roaring64.md). Reset a bitmap to
empty without freeing its backing allocation (so it can be refilled cheaply).

## Feature

| rawr 64-bit (new) | CRoaring | Semantics |
|---|---|---|
| `clear() void` | `roaring64_bitmap_clear` | remove all values; keep the object usable |

## Implementation

Deinit every sub-bitmap (`bucket.bm.deinit()` for `buckets[0..size]`), set
`size = 0`, set `cached_cardinality = 0`. **Keep the bucket-array allocation**
(do not free `buckets`) — that's the whole point vs `deinit` + `init`. After
`clear`, the bitmap is a valid empty `Roaring64Bitmap` ready for `add`.

Trivial, but assert the invariant: `capacity` is unchanged, `buckets` pointer is
retained, and the object is immediately reusable.

## Wrapper decl

```c
void roaring64_bitmap_clear(roaring64_bitmap_t *r);
```

## Tests / oracle

- Inline: fill, `clear`, assert `isEmpty` / `size == 0` / `cardinality() == 0` /
  `capacity` unchanged; then `add` again and confirm it works (no leak under the
  checking GPA).
- `difftest64`: light — clear both rawr and oracle mid-iteration, assert
  agreement (empty on both). Not a heavy path.

## Acceptance

- `clear` empties in place, retains capacity, leaves a reusable bitmap, no leak.
- Green; no 32-bit regression.
