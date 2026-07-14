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
| `ensureTotalCapacity(container_capacity) !void` | new public method (32-bit: rename of the already-public `ensureCapacity` + undocumented alias; 64-bit: rename of its *private* `ensureCapacity`, **no alias**) | both |
| `clearRetainingCapacity() void` | **rename** `Roaring64Bitmap.clear` → this; **add** to `RoaringBitmap` | both |
| `shrinkToFit() !usize` | already exists | both (verify) |

Rename `Roaring64Bitmap.clear()` → `clearRetainingCapacity()` (it just landed in
10-10, unreleased, so a clean rename — update its call sites in `diff_test64` /
`10-10` tests). Do **not** keep `clear` as an alias; one name.

`clearAndFree` (drop to minimal allocation) is **out of scope** — the issue only
asks to retain capacity; add later if requested.

**`ensureCapacity` is already public** on `RoaringBitmap` (not internal, as an
earlier draft implied). Since external users may already call it: **rename to the
Zig-idiomatic `ensureTotalCapacity`**, and keep `ensureCapacity` as a thin,
undocumented compatibility alias delegating to it (migrate rawr's own internal
callers to the new name). **`Roaring64Bitmap`'s `ensureCapacity` is *private*** —
there's no external API to preserve, so just rename it to `ensureTotalCapacity`
(now public) and add **no alias**. Alias lives on the 32-bit type only.

**Precise `clearRetainingCapacity` semantics** (an earlier draft overstated this):
it retains **only the container index** — the top-level `keys`/`containers`
(or `buckets`) array. It **deinitializes every live container**, so per-container
array/run/bitset storage *is* freed. "Retaining capacity" = axis 1 only, never
axis 2. The docs must say this exactly (see below).

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

**Capacity rule (one rule, both types): allocate the *exact* requested capacity,
including zero.** No `@max(cap, 1)`, no floor. `initCapacity(allocator, 0)` yields
a valid empty bitmap with zero-length index arrays (Zig `alloc(T, 0)` is fine; the
first insert grows via `ensureTotalCapacity`). Normal `init()` keeps requesting
`INITIAL_CAPACITY` (4) — implement `init` as `initCapacity(allocator, INITIAL_CAPACITY)`
so there's a single allocation path.

**`RoaringBitmap`:**
- `initCapacity(allocator, cap: u32) !Self` — allocate `keys` and `containers` to
  exactly `cap` (errdefer-free the first if the second fails). `size = 0`,
  `cached_cardinality = 0`.
- `ensureTotalCapacity(self, cap: u32) !void` — the renamed `ensureCapacity`: grow
  the top-level arrays to hold `cap` containers, no-op if already ≥. Does not touch
  `size` or contents. **Must be OOM-safe — see below.** Keep `ensureCapacity` as a
  compat alias calling this.
- `clearRetainingCapacity(self) void` — deinit every live container, `size = 0`,
  `cached_cardinality = 0`, **keep** the `keys`/`containers` allocation (container
  index only; per-container storage is freed by the deinits).

**`Roaring64Bitmap`:**
- `initCapacity(allocator, cap: u32) !Self` — allocate `buckets` to exactly `cap`;
  `init` delegates to `initCapacity(allocator, INITIAL_CAPACITY)`.
- `ensureTotalCapacity(self, cap: u32) !void` — renamed from its **private**
  `ensureCapacity` (no alias — nothing public to preserve), OOM-safe.
- `clearRetainingCapacity` — the renamed `clear` (retains the bucket array; sub-
  bitmaps are deinit'd).

**OOM-safety fix (required — this is a real latent bug the public API exposes):**
the current 32-bit growth path (`bitmap.zig:226`) frees/replaces `keys` *before*
allocating the new `containers` array — if that second alloc fails, the bitmap is
left inconsistent (freed `keys`, stale `containers`). Since reservation is now a
deliberate public entry point, `ensureTotalCapacity` must **allocate both
replacement arrays first, only mutate `self` once both succeed**, then free the old
ones:

```zig
const new_keys = try allocator.alloc(u16, new_cap);
errdefer allocator.free(new_keys);
const new_containers = try allocator.alloc(TaggedPtr, new_cap);   // if THIS fails, new_keys is freed by errdefer, self untouched
@memcpy(new_keys[0..self.size], self.keys[0..self.size]);
@memcpy(new_containers[0..self.size], self.containers[0..self.size]);
allocator.free(self.keys[0..self.capacity]);
allocator.free(self.containers[0..self.capacity]);
self.keys = new_keys; self.containers = new_containers; self.capacity = new_cap;
```

Apply the same allocate-both-before-mutate discipline to the 64-bit bucket growth
if it has the same shape. Add an **allocation-failure test** using
`std.testing.FailingAllocator` (or `std.testing.checkAllAllocationFailures`) to fail
the Nth allocation, asserting the bitmap is unchanged and still usable after a
failed `ensureTotalCapacity`.

**Overflow-safe doubling (required):** both types currently grow via `capacity * 2`,
which **overflows** for a large `u32` capacity — now reachable because the public
`initCapacity`/`ensureTotalCapacity` let callers set capacity anywhere in `u32`.
Use **saturating** doubling and clamp to the request:

```zig
const new_cap = @max(self.capacity *| 2, needed);   // `*|` saturates at maxInt(u32)
```

`needed` is itself a `u32` (≤ `maxInt(u32)`), so `new_cap` never exceeds the type.
(A container index can't need more than 65 536 entries in practice, but the
arithmetic must be total regardless of what a caller passes.)

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
> - `clearRetainingCapacity()` empties the bitmap and **keeps the container index**
>   (axis 1) so you can refill without re-growing it. The per-container storage
>   (axis 2) of the cleared containers is freed — retaining it would mean holding
>   onto storage for containers that no longer exist.

## Tests

- `initCapacity(n)`: construct, assert `capacity == n` (exact) and `size == 0`;
  insert into `n` distinct chunks and assert **no reallocation of the top-level
  array** occurred (capture `containers.ptr`, assert unchanged after the inserts).
- `initCapacity(0)`: valid empty bitmap; first insert grows cleanly; leak-free.
- `ensureTotalCapacity`: grows to request, no-op when already large enough, contents
  preserved. On `RoaringBitmap` only, the `ensureCapacity` alias still compiles and
  behaves identically (no alias test on `Roaring64Bitmap` — it has none).
- Overflow-safe growth: `ensureTotalCapacity` with a very large `cap` does not
  overflow the doubling (saturates); a normal small grow is unaffected.
- **Allocation-failure**: a failing allocator that fails the *second* index alloc
  during a grow — assert `ensureTotalCapacity` returns `error.OutOfMemory` and the
  bitmap is **unchanged and still usable** (old `keys`/`containers`/`size` intact,
  subsequent ops work). This is the regression test for the OOM-safety fix.
- `clearRetainingCapacity`: fill → clear → assert empty + index retained
  (`containers.ptr` unchanged) → refill works, leak-free under the checking GPA.
- `shrinkToFit` after `clearRetainingCapacity` returns capacity to minimal (existing
  behavior; add a coverage test tying the two together).
- Same set for `Roaring64Bitmap` (bucket-array capacity).

## Acceptance

- `initCapacity` (exact cap, incl. 0) + `ensureTotalCapacity` (OOM-safe) +
  `clearRetainingCapacity` on both types. `Roaring64Bitmap.clear` renamed (no
  alias); on `RoaringBitmap` only, `ensureCapacity` retained as an undocumented
  compat alias for `ensureTotalCapacity` (64-bit's was private → renamed, no alias).
  `shrinkToFit` verified as the shrink answer.
- `ensureTotalCapacity` allocates both replacement arrays before mutating `self`
  and uses saturating (`*|`) doubling; the allocation-failure test passes (bitmap
  unchanged on OOM).
- `API.md` carries the "How rawr bitmaps allocate" section, with
  `clearRetainingCapacity` documented as retaining the container index only.
- `zig build test test64` green; no regression. No container-ABI change (independent
  of spec 11 / 11-04).
