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

CRoaring's portable 64-bit layout, confirmed from `vendor/roaring.c`:

- a `uint64_t` count of buckets (high-32 keys), then
- for each bucket, in ascending key order: the `uint32_t` high key followed by a
  standard **32-bit portable** roaring bitmap (exactly the bytes rawr's existing
  `serialize` already produces for a `RoaringBitmap`).

Endianness is little-endian for the count and key fields — the same assumption
rawr's current 32-bit serialization already makes; reuse the existing byte-order
helpers rather than introducing a new convention.

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

- `serializedSizeInBytes(self) !usize` — `8` (count) + Σ over buckets of
  `4` (key) + `bucket.bm.serializedSizeInBytes()`. **Returns `!usize`** (a
  signature divergence from the 32-bit method) and yields `error.Overflow` when
  the running sum exceeds `maxInt(usize)`, per the toplevel overflow policy —
  accumulate with checked arithmetic.
- `serializeToWriter(self, writer) !void` — write the `u64` count, then for each
  bucket the `u32` key and the sub-bitmap via the existing
  `RoaringBitmap.serializeToWriter`. Match the byte order CRoaring uses (the
  32-bit path is already correct; replicate its key/count endianness).
- `serialize(self, allocator) ![]u8` — allocate `try serializedSizeInBytes()`,
  write via a fixed buffer; propagates `error.Overflow`.

Place the framing logic in `src/serialize.zig` next to the 32-bit functions (or a
sibling), reusing the 32-bit writer for the payload — do not duplicate the
container-encoding logic.

## Task 2 — Prerequisite: per-sub-bitmap consumed length

The existing 32-bit `RoaringBitmap.deserialize(data)` does **not** report how many
bytes it consumed, but a 64-bit frame packs multiple portable bitmaps back-to-back
— so the 64-bit parser must know where each sub-bitmap ends to find the next key.
Pick one (implement in `src/serialize.zig` alongside the 32-bit code):

- **(a) header-driven size helper** — `portableSizeInBytes(data) !usize` that
  reads the portable header (cookie, container count, the per-container
  type/cardinality descriptors, and the offset table when the run-flag/threshold
  says one is present) to compute the exact serialized length of the *leading*
  bitmap without materializing it. Then `deserialize` slices `data[0..len]`, parses
  it, and advances by `len`. This mirrors what `serializedSizeInBytes` computes,
  but from serialized bytes rather than a live bitmap.

  > **RUN containers need a data peek, not just the header.** A RUN container's
  > run count (`n_runs`) is **not** in the descriptor block — it's the first `u16`
  > of that container's own data, and its byte size is `2 + 4 * n_runs`. So the
  > helper cannot size a bitmap from headers/descriptors/offsets alone: for each
  > container flagged RUN it must read that leading `u16` from the container's data
  > region. This is exactly what CRoaring's `ra_portable_deserialize_size` does.
  > Array/bitset container sizes *are* fixed by their cardinality descriptor; only
  > RUN needs the peek. Bounds-check the peek in the `Safe` path (a RUN `n_runs`
  > must not push the computed length past the buffer).
- **(b) counting-reader parse** — a reader wrapper over the byte slice that tracks
  its cursor, and a `deserialize`-from-reader path that consumes exactly one
  portable bitmap and exposes the new cursor. (Note the existing
  `deserializeFromReader` takes `data_len` up front — which is the unknown here —
  so this route still needs the header to bound each sub-bitmap, i.e. it reduces
  to (a) unless the reader parse is genuinely length-free.)

Prefer **(a)** — it's a small, testable, `*const` byte-length function and keeps
the 64-bit parser a plain slice-and-advance loop. The helper is **bounds-checked
in both paths, not only `Safe`**: a truncated slice or a length (including a RUN
`n_runs` peek) that would read past the buffer must return an error, never read
out of bounds — even plain `deserialize` must fail cleanly on truncated input
rather than over-read. The `Safe` vs plain distinction is about how the *32-bit
sub-bitmap payload* is validated (`deserializeSafe` vs `deserialize`), not about
whether the frame bounds are checked — the frame is always checked.

## Task 3 — `deserialize` / `deserializeSafe`

- `deserialize(allocator, data) !Self` — read count, then for each bucket read the
  `u32` key, compute the sub-bitmap length via the Task 2 helper, parse that slice
  with `RoaringBitmap.deserialize`, and advance. Keys must be **strictly
  ascending**; sub-bitmaps must be non-empty (mirror the 32-bit validate
  posture).
- `deserializeSafe(allocator, data) !Self` — bounds-checked variant routing the
  per-bucket payload through `RoaringBitmap.deserializeSafe`; reject truncated
  input, a count that overruns the buffer, out-of-order/duplicate keys, and empty
  sub-bitmaps. Reuse the hardening posture from specs 04/05 — no new attack
  surface beyond the count+key frame, since the payloads go through the
  already-hardened 32-bit path.

The count field is `u64` on the wire but `size` is `u32`; **reject any count
`> maxInt(u32)`** before allocating (matches CRoaring and the `size: u32` field).
This is a cheap early rejection independent of the buffer-overrun check.

(`deserializeFromReader` / owned / frozen 64-bit variants are out of scope —
deferred per the toplevel.)

## Task 4 — `validate64` round-trip path

Extend `src/validate_roaring64.zig` (from 10-01) with the serialization bar,
mirroring `src/validate_croaring.zig`:

1. Build a `Roaring64Bitmap` from a generated 64-bit corpus.
2. **rawr → CRoaring:** `rawr.serialize` → `roaring64_bitmap_portable_deserialize_safe`
   → assert cardinality + full membership + min/max agree.
3. **CRoaring → rawr:** build the equivalent CRoaring `roaring64`,
   `roaring64_bitmap_portable_serialize` → `rawr.deserialize` → assert `equals`
   the original rawr bitmap.
4. **rawr → rawr:** `serialize` → `deserialize` → `equals` (self round-trip).
5. Assert `serializedSizeInBytes` equals the actual bytes written. Compare against
   `roaring64_bitmap_portable_size_in_bytes` **only after the run-container caveat
   below** — otherwise sizes legitimately differ.

**Run-container caveat (same as the 32-bit difftest).** rawr's `addRange` and
`runOptimize` produce RUN containers, so rawr's serialized bytes encode runs while
a freshly-built CRoaring oracle does not — byte-length and byte-equality then
differ even though the *sets* are identical. Before any byte-level comparison
(size or bytes), **clone the CRoaring oracle and call
`roaring64_bitmap_run_optimize` on it** so both sides agree on run encoding; or
drop the byte comparison for that case and rely on set-equality + cross-deserialize
(`equals`) instead. Steps 2–4 (set/membership/`equals`) are unaffected and are the
primary bar; step 5's byte/size check is the one that needs the run-optimize.

Add the wrapper decl:
```c
bool roaring64_bitmap_run_optimize(roaring64_bitmap_t *r);
```

Cover the empty bitmap, single-bucket, many-buckets, and run-containing
sub-bitmaps (so the 32-bit RUN payload path is exercised under the 64-bit frame).

## Task 5 — Docs

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
