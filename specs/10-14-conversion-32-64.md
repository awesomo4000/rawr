# Spec 10-14: `Roaring64Bitmap` ↔ `RoaringBitmap` conversion

Phase-2 parity chunk of [64-bit Roaring](10-roaring64.md). Bridge between the
32-bit and 64-bit types.

## Features

| rawr (new) | CRoaring | Semantics |
|---|---|---|
| `fromRoaring32(allocator, r32) !Self` | `roaring64_bitmap_move_from_roaring32` | build a 64-bit bitmap from a 32-bit one (all values under high-key `0`) |
| `toRoaring32(allocator) !?RoaringBitmap` | — (rawr convenience) | extract the `hi == 0` sub-bitmap as a standalone 32-bit bitmap; `null` if any value has `hi != 0` |

`fromRoaring32` is the parity item (CRoaring has it). `toRoaring32` is the natural
rawr inverse — include it since it's trivial given the bucket model, but it's
rawr-only (no oracle) so mark it as such.

## Implementation

- **`fromRoaring32`** — a 32-bit bitmap's values all have high-32 = `0`. Clone the
  32-bit bitmap into a single bucket `{ hi = 0, bm = r32.clone() }` (if non-empty).
  CRoaring's `move_from_roaring32` *consumes* the source; rawr's version **clones**
  (rawr has no move-semantics convention here) — document that it does not take
  ownership of `r32`. Empty `r32` → empty `Roaring64Bitmap`.
- **`toRoaring32`** — if any bucket has `hi != 0`, return `null` (values exceed the
  32-bit domain). Otherwise clone the `hi == 0` bucket's sub-bitmap (or return an
  empty 32-bit bitmap if there are no buckets).

## Wrapper decl

```c
roaring64_bitmap_t *roaring64_bitmap_move_from_roaring32(roaring_bitmap_t *r);
```

## Tests / oracle

- Inline: round-trip `fromRoaring32(x).toRoaring32()` equals `x`; `fromRoaring32`
  of empty; `toRoaring32` returns `null` when a `hi != 0` value is present;
  ownership — assert the source `r32` is still valid/unmodified after
  `fromRoaring32` (clone, not move).
- `difftest64`/`validate64`: build a `RoaringBitmap`, convert via `fromRoaring32`,
  and oracle against `roaring64_bitmap_move_from_roaring32` on a **copy** of the
  equivalent CRoaring 32-bit bitmap (since CRoaring's variant consumes its input,
  feed it a copy).

## Acceptance

- `fromRoaring32` clones a 32-bit bitmap into the `hi == 0` bucket (documented
  non-consuming); `toRoaring32` extracts or returns `null`.
- Oracled against `move_from_roaring32` (on a copy); green; no 32-bit regression.
