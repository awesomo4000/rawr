# Spec 10-18: `Roaring64Bitmap` frozen (zero-copy read-only view)

Final Phase-2 parity chunk of [64-bit Roaring](10-roaring64.md), and the **large**
one — likely its own mini-phase if it gets hairy. A zero-copy, read-only view over
a serialized buffer, mirroring rawr's existing `FrozenBitmap`.

## Up-front decision — which frozen format?

CRoaring's frozen format is **memory-layout-specific** (a flat, alignment-sensitive
image for `mmap`/zero-copy), and it is **not** the portable format. rawr's existing
`FrozenBitmap` is *its own* read-only view shape, not byte-compatible with
CRoaring's frozen image. **Decide and document up front which target:**

- **(A) rawr-native frozen64 (recommended v1)** — a `Frozen64Bitmap` that is a
  read-only view over rawr's *own* frozen layout (extend rawr's `FrozenBitmap`
  approach to the 64-bit frame: a header of `{count, [hi, offset]...}` over frozen
  32-bit sub-views). rawr↔rawr only. No CRoaring frozen interop claimed. This is
  the tractable, in-house-consistent option and matches how 32-bit frozen works.
- **(B) CRoaring frozen interop** — target `roaring64_bitmap_frozen_{serialize,
  view}`'s exact byte layout. Much larger: alignment rules, endianness, and the
  ART/bucket image are CRoaring-internal and version-sensitive. **Not recommended**
  unless cross-process `mmap` interop with CRoaring is an actual requirement.

**Default to (A).** If (B) is ever wanted, split it into its own chunk — do not
let it block the rest of parity.

## Feature (option A)

| rawr 64-bit (new) | mirrors | Semantics |
|---|---|---|
| `frozenSizeInBytes() !usize` | `FrozenBitmap` | size of the rawr frozen64 image |
| `frozenSerialize(buf) !void` | `FrozenBitmap` | write the frozen64 image into `buf` |
| `Frozen64Bitmap.view(bytes) !Frozen64Bitmap` | `FrozenBitmap` | **truly zero-copy** read-only view (no allocator) |
| read-only ops on the view | — | `contains`, `cardinality`, `minimum`, `maximum`, `iterator`, `rank`, `select`, `getIndex` (no mutation) |

**`frozenSizeInBytes` returns `!usize`** — same overflow policy as
`serializedSizeInBytes` (10-04 / toplevel): a 64-bit frozen image can exceed
`maxInt(usize)`; accumulate with checked arithmetic and return `error.Overflow`.
`frozenSerialize` propagates it.

**`view` takes no allocator and allocates nothing** (see layout below) — the
signature `view(bytes) !Frozen64Bitmap` is honored literally; the only errors are
malformed/misaligned input, not OOM.

## Implementation (option A)

- Frozen64 image = frame header `{ u64 count, then an offset/key table: per bucket
  `{ u32 hi, u64 offset }` , then the frozen 32-bit sub-images }`, where each
  sub-image is the existing 32-bit `FrozenBitmap` layout. Respect the same
  alignment `FrozenBitmap` requires for its container arrays (propagate per-bucket;
  pad the offset table / sub-images so each 32-bit image starts at its required
  alignment).
- **The offset+key table lives *in the image*, so `view` is truly zero-copy** —
  `Frozen64Bitmap` holds only the borrowed byte slice and locates a bucket's `hi`
  and sub-image start by indexing into the borrowed table (binary search on `hi`).
  It constructs a 32-bit `FrozenBitmap` sub-view on demand from the borrowed
  bytes; **no owned bucket index, no allocation.** This is why `view` needs no
  allocator — the "parsed index" is the in-image table, not a heap structure.
- Read-only ops delegate to the per-bucket `FrozenBitmap` (constructed on the fly
  from the borrowed table) exactly as the mutable type delegates to
  `RoaringBitmap`. `view` validates the table bounds/alignment against `bytes.len`
  and errors on malformed input — it never allocates.

## Wrapper decls (only if option B is chosen)

```c
size_t roaring64_bitmap_frozen_size_in_bytes(const roaring64_bitmap_t *r);
void roaring64_bitmap_frozen_serialize(const roaring64_bitmap_t *r, char *buf);
const roaring64_bitmap_t *roaring64_bitmap_frozen_view(const char *buf, size_t length);
```

Under option A these are **not** used (no CRoaring frozen oracle); validate rawr's
frozen64 round-trip against rawr's own mutable type instead.

## Tests / oracle

- Inline: `frozenSerialize` → `Frozen64Bitmap.view` → assert every read-only op
  agrees with the source mutable bitmap (contains/cardinality/min/max/iterator/
  rank/select/getIndex) across empty / single / many buckets / run-bearing;
  alignment respected (no misaligned loads under UBSan/safe build); the view
  borrows (no leak, source buffer owns the memory).
- Option A has **no CRoaring frozen oracle** — the bar is rawr-mutable ⇄
  rawr-frozen64 agreement. Document the no-interop scope in API.md alongside the
  32-bit frozen note.

## Acceptance

- Decision (A vs B) recorded; default **A** (rawr-native frozen64, no CRoaring
  frozen interop) unless cross-process interop is required.
- `Frozen64Bitmap` zero-copy read-only view with the read-only op surface, built
  on the existing 32-bit `FrozenBitmap`; alignment honored.
- rawr mutable ⇄ frozen64 round-trip validated; interop scope documented.
- Green; no 32-bit regression. **If option B or the layout turns hairy, promote to
  its own mini-phase rather than blocking parity closeout.**
