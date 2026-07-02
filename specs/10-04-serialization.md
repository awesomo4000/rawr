# Spec 10-04: `Roaring64Bitmap` serialization (CRoaring portable-64)

Fourth piece of [64-bit Roaring](10-roaring64.md). Portable serialize /
deserialize, plus the `validate64` round-trip path that proves byte-level interop
with CRoaring's 64-bit format.

## Interop scope (decided in the toplevel)

Target **CRoaring's roaring64 portable format only**
(`roaring64_bitmap_portable_serialize` / `_deserialize` / `_size_in_bytes`).
**No** claim of interop with Java `Roaring64NavigableMap` or Java
`Roaring64Bitmap` — those layouts differ and we cannot test them here. The docs
must say exactly this.

## Format

CRoaring's portable 64-bit layout (confirm field widths/endianness against
`vendor/roaring.h` / the CRoaring serialization source during implementation):

- a `uint64_t` count of buckets (high-32 keys), then
- for each bucket, in ascending key order: the `uint32_t` high key followed by a
  standard **32-bit portable** roaring bitmap (exactly the bytes rawr's existing
  `serialize` already produces for a `RoaringBitmap`).

This is why the chunk is small: the per-bucket payload is the already-validated
32-bit portable encoding from `src/serialize.zig`. The 64-bit layer is only the
count + key framing around each sub-bitmap.

## Task 0 — Wrapper decls

Append to `vendor/croaring_wrapper.h`:

```c
size_t roaring64_bitmap_portable_size_in_bytes(const roaring64_bitmap_t*);
size_t roaring64_bitmap_portable_serialize(const roaring64_bitmap_t*, char *buf);
roaring64_bitmap_t *roaring64_bitmap_portable_deserialize_safe(const char *buf, size_t maxbytes);
```

## Task 1 — `serializedSizeInBytes` / `serialize` / `serializeToWriter`

- `serializedSizeInBytes(self) usize` — `8` (count) + Σ over buckets of
  `4` (key) + `bucket.bm.serializedSizeInBytes()`.
- `serializeToWriter(self, writer) !void` — write the `u64` count, then for each
  bucket the `u32` key and the sub-bitmap via the existing
  `RoaringBitmap.serializeToWriter`. Match the byte order CRoaring uses (the
  32-bit path is already correct; replicate its key/count endianness).
- `serialize(self, allocator) ![]u8` — allocate `serializedSizeInBytes`, write via
  a fixed buffer.

Place the framing logic in `src/serialize.zig` next to the 32-bit functions (or a
sibling), reusing the 32-bit writer for the payload — do not duplicate the
container-encoding logic.

## Task 2 — `deserialize` / `deserializeSafe`

- `deserialize(allocator, data) !Self` — read count, then for each bucket read the
  `u32` key and call `RoaringBitmap.deserialize` on the remaining slice,
  advancing by each sub-bitmap's consumed length. Keys must be **strictly
  ascending**; sub-bitmaps must be non-empty (mirror the 32-bit validate
  posture).
- `deserializeSafe(allocator, data) !Self` — bounds-checked variant routing the
  per-bucket payload through `RoaringBitmap.deserializeSafe`; reject truncated
  input, a count that overruns the buffer, out-of-order/duplicate keys, and empty
  sub-bitmaps. Reuse the hardening posture from specs 04/05 — no new attack
  surface beyond the count+key frame, since the payloads go through the
  already-hardened 32-bit path.

(`deserializeFromReader` / owned / frozen 64-bit variants are out of scope —
deferred per the toplevel.)

## Task 3 — `validate64` round-trip path

Extend `src/validate_roaring64.zig` (from 10-01) with the serialization bar,
mirroring `src/validate_croaring.zig`:

1. Build a `Roaring64Bitmap` from a generated 64-bit corpus.
2. **rawr → CRoaring:** `rawr.serialize` → `roaring64_bitmap_portable_deserialize_safe`
   → assert cardinality + full membership + min/max agree.
3. **CRoaring → rawr:** build the equivalent CRoaring `roaring64`,
   `roaring64_bitmap_portable_serialize` → `rawr.deserialize` → assert `equals`
   the original rawr bitmap.
4. **rawr → rawr:** `serialize` → `deserialize` → `equals` (self round-trip).
5. Assert `serializedSizeInBytes` equals the actual bytes written **and**
   `roaring64_bitmap_portable_size_in_bytes` for the same set.

Cover the empty bitmap, single-bucket, many-buckets, and run-containing
sub-bitmaps (so the 32-bit RUN payload path is exercised under the 64-bit frame).

## Task 4 — Docs

Document the interop scope in the public API docs (wherever the 32-bit
serialization interop is documented): rawr's 64-bit portable format round-trips
with **CRoaring `roaring64`**; Java 64-bit variants are **not** supported/tested.

## Acceptance

- `serialize`/`serializedSizeInBytes`/`deserialize`/`deserializeSafe` implemented
  on `Roaring64Bitmap`, reusing the 32-bit payload encoder/decoder.
- Inline tests: self round-trip across empty / single-bucket / many-bucket /
  run-bearing inputs; `serializedSizeInBytes` matches bytes written;
  `deserializeSafe` rejects truncated input, count overrun, non-ascending keys,
  and empty sub-bitmaps.
- `validate64` passes all five round-trip assertions above against CRoaring
  `roaring64`, including the size-agreement check.
- Docs state the CRoaring-only interop scope; no Java-compat claim.
- `zig build test64 validate64` green; no 32-bit regression.
