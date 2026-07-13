# Spec 12: Capacity management API (issue #3)

Small, self-contained, user-requested ([issue #3](https://github.com/awesomo4000/rawr/issues/3)):
let callers pre-size a bitmap to cut allocation churn, and reclaim / retain
capacity on demand. Behavior-preserving for existing code (new API only, plus one
rename of a just-landed method). Applies to **both** `RoaringBitmap` and
`Roaring64Bitmap` for symmetry.

## Status going in — a third is already done

- **Shrink** (issue ask #2) — `RoaringBitmap.shrinkToFit` already exists
  (`bitmap.zig:241`, landed with 10-09's compaction work) and
  `Roaring64Bitmap.shrinkToFit` too. This spec only **verifies** it satisfies the
  request; no new shrink code.
- **Clear-retaining** (issue comment) — `Roaring64Bitmap.clear()` shipped in 10-10;
  `RoaringBitmap` has no equivalent yet.
- **Pre-size on construction** (issue ask #1) — neither type has it. This is the
  main new work.

## Naming — Zig std convention

Match `std.ArrayList` vocabulary so the API reads idiomatically:

| method | replaces / adds | on |
|---|---|---|
| `initCapacity(allocator, container_capacity) !Self` | new | both |
| `ensureTotalCapacity(container_capacity) !void` | new (public wrapper over the internal `ensureCapacity`) | both |
| `clearRetainingCapacity() void` | **rename** `Roaring64Bitmap.clear` → this; **add** to `RoaringBitmap` | both |
| `shrinkToFit() !usize` | already exists | both (verify) |

Rename `Roaring64Bitmap.clear()` → `clearRetainingCapacity()` (it just landed in
10-10, unreleased, so a clean rename — update its call sites in `diff_test64` /
`10-10` tests). Do **not** keep `clear` as an alias; one name.

`clearAndFree` (drop to minimal allocation) is **out of scope** — the issue only
asks to retain capacity; add later if requested.

## Semantics — reserve by *container count*, not element count

The issue asks to pre-size by "exact starting size (or total elements)." **Per-
element preallocation is not meaningful in a Roaring structure** and the API must
not pretend otherwise (see the doc section — this is the crux of the answer to the
reporter):

- A bitmap is a sorted array of up to 65 536 *containers*, one per distinct high-16
  chunk of the value space. The churn the reporter is hitting is the **geometric
  regrowth of that top-level `keys` + `containers` array** (starts at 4, doubles) as
  a bitmap comes to span many chunks.
- `initCapacity(n)` / `ensureTotalCapacity(n)` preallocate that top-level array to
  hold `n` containers — directly killing the regrowth churn. `n` ≈ the number of
  distinct 16-bit high chunks you expect to touch (for `Roaring64Bitmap`, `n` = the
  number of high-32 buckets).
- You **cannot** preallocate per-container storage by element count: values map to
  containers unpredictably, and a container's type (array/bitset/run) — hence its
  storage — is decided dynamically. So the reserve granularity is containers, and
  the doc says so plainly.

## Implementation

**`RoaringBitmap`:**
- `initCapacity(allocator, cap: u32) !Self` — like `init`, but allocate `keys` and
  `containers` to `@max(cap, 1)` (or `INITIAL_CAPACITY` floor if you prefer a
  minimum) instead of the hardcoded `INITIAL_CAPACITY = 4`. `size = 0`,
  `cached_cardinality = 0`.
- `ensureTotalCapacity(self, cap: u32) !void` — public wrapper over the existing
  internal `ensureCapacity` (grow the top-level arrays to hold `cap` containers;
  no-op if already ≥). Does not touch `size` or contents.
- `clearRetainingCapacity(self) void` — deinit every live container, `size = 0`,
  `cached_cardinality = 0`, **keep** the `keys`/`containers` allocation. (Mirror of
  the existing `Roaring64Bitmap.clear` body.)

**`Roaring64Bitmap`:**
- `initCapacity(allocator, cap: u32) !Self` — allocate the `buckets` array to `cap`.
- `ensureTotalCapacity(self, cap: u32) !void` — public wrapper over its internal
  `ensureCapacity`.
- `clearRetainingCapacity` — the renamed `clear`.

All new methods `!`-return only where they allocate (`initCapacity`,
`ensureTotalCapacity`); `clearRetainingCapacity` is infallible `void`.

## Docs — new "How rawr bitmaps allocate" section (required)

Add to `API.md` (near Construction / Footguns) — this is both the issue answer and
the permanent project doc. Content:

> ### How rawr bitmaps allocate
>
> A Roaring bitmap is a sorted array of **containers**, one per distinct high-16
> chunk of the value space (up to 65 536 of them). There are two independent
> allocation axes:
>
> 1. **The container index** — the top-level `keys` + `containers` arrays. These
>    start small (4 entries) and grow geometrically as your data comes to span more
>    16-bit chunks.
> 2. **Each container's own storage** — array / bitset / run, sized and typed
>    dynamically as you insert.
>
> To cut allocation churn when you know you'll touch many chunks, pre-size the
> container index:
>
> ```zig
> // Expecting values spread across ~1000 distinct high-16 chunks:
> var bm = try RoaringBitmap.initCapacity(allocator, 1000);
> // or, on an existing bitmap:
> try bm.ensureTotalCapacity(1000);
> ```
>
> Reserve by **container count** (≈ the number of distinct high-16 chunks, or for
> `Roaring64Bitmap` the number of high-32 buckets), *not* by element count — values
> map to containers unpredictably and container storage is chosen dynamically, so an
> element count can't be turned into a precise allocation.
>
> - `shrinkToFit()` releases unused capacity on both axes (returns approximate bytes
>   freed) once you're done inserting.
> - `clearRetainingCapacity()` empties the bitmap but keeps both axes' allocations,
>   so you can refill without re-allocating.

## Tests

- `initCapacity(n)`: construct, assert `capacity >= n` and `size == 0`; insert into
  `n` distinct chunks and assert **no reallocation of the top-level array** occurred
  (e.g. capture the `containers.ptr` and assert it's unchanged after the inserts).
- `ensureTotalCapacity`: grows to request, no-op when already large enough, contents
  preserved.
- `clearRetainingCapacity`: fill → clear → assert empty + capacity retained
  (`containers.ptr` unchanged) → refill works, leak-free under the checking GPA.
- `shrinkToFit` after `clearRetainingCapacity` returns capacity to minimal (existing
  behavior; add a coverage test tying the two together).
- Same set for `Roaring64Bitmap` (bucket-array capacity).

## Acceptance

- `initCapacity` + `ensureTotalCapacity` + `clearRetainingCapacity` on both types;
  `Roaring64Bitmap.clear` renamed (no alias); `shrinkToFit` verified as the shrink
  answer.
- `API.md` carries the "How rawr bitmaps allocate" section.
- `zig build test test64` green; no regression. No container-ABI change (independent
  of spec 11 / 11-04).
