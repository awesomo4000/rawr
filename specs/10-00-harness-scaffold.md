# Spec 10-00: `Roaring64Bitmap` harness scaffold

Zeroth piece of [64-bit Roaring](10-roaring64.md). Pure plumbing: stand up an
**empty** `Roaring64Bitmap` type and the three 64-bit build steps, prove they
wire up and go green, and extend the CRoaring wrapper header so later chunks can
bind `roaring64_*`. **No real features** — the whole point is a small first
handoff that de-risks the rig before any behavior lands in 10-01.

## Deliverable

A minimal `src/roaring64.zig`, three new `build.zig` steps, a new
`validate_roaring64.zig` + `diff_test64.zig`, and the `roaring64_*` lifecycle
decls in `vendor/croaring_wrapper.h`. After this chunk
`zig build test64 validate64 difftest64` is green but exercises essentially
nothing.

## Task 1 — Empty type

`src/roaring64.zig` exporting `pub const Roaring64Bitmap` with **only** the
backing structure and lifecycle — enough to construct, destroy, and report
emptiness. Mirror `RoaringBitmap`'s conventions (`src/bitmap.zig`): store the
allocator, cache cardinality.

```zig
pub const Roaring64Bitmap = struct {
    const Bucket = struct { hi: u32, bm: RoaringBitmap };
    buckets: []Bucket,
    size: u32,
    capacity: u32,
    allocator: std.mem.Allocator,
    cached_cardinality: i64 = 0,

    pub fn init(allocator) !Self
    pub fn deinit(self) void
    pub fn isEmpty(self) bool       // size == 0
    pub fn cardinality(self) u64    // 0 for now
};
```

Re-export from `src/roaring.zig` (`pub const Roaring64Bitmap =
@import("roaring64.zig").Roaring64Bitmap;`) and import it so its inline tests
fold into the default `test` step. One trivial inline test: `init` → `isEmpty` →
`deinit`, leak-free under a checking GPA.

(`add`/`contains`/`remove`/`iterator`/etc. arrive in 10-01. Do **not** build them
here.)

## Task 2 — Wrapper header: lifecycle decls

Add the `roaring64` lifecycle/identity decls to `vendor/croaring_wrapper.h` —
already compiled in `vendor/roaring.c`, this only declares them for translate-c.
Later chunks append their own (set ops in 10-02, positional/range in 10-03,
portable serialize in 10-04):

```c
typedef struct roaring64_bitmap_s roaring64_bitmap_t;
roaring64_bitmap_t *roaring64_bitmap_create(void);
void roaring64_bitmap_free(roaring64_bitmap_t *r);  // NB: non-const, unlike 32-bit roaring_bitmap_free
roaring64_bitmap_t *roaring64_bitmap_copy(const roaring64_bitmap_t *r);
void roaring64_bitmap_add(roaring64_bitmap_t *r, uint64_t x);
uint64_t roaring64_bitmap_get_cardinality(const roaring64_bitmap_t *r);
bool roaring64_bitmap_is_empty(const roaring64_bitmap_t *r);
```

## Task 3 — Three build steps

In `build.zig`, mirror the existing `test` / `validate` / `difftest` wiring:

- **`test64`** — `addTest` rooted at `src/roaring64.zig` (or a dedicated
  `roaring64_tests.zig`), run step `test64`.
- **`validate64`** — executable rooted at a new `src/validate_roaring64.zig`,
  built with the CRoaring import via the same `addTranslatedCImport` pattern as
  `validate`, run step `validate64`. **Stub body:** create an empty CRoaring
  `roaring64` and an empty `Roaring64Bitmap`, assert both report cardinality `0`
  / empty, free both, print OK. Proves the `roaring64_*` binding links and runs.
- **`difftest64`** — executable rooted at a new `src/diff_test64.zig`, run step
  `difftest64`. **Stub body:** same trivial empty-vs-empty agreement check.
  Real generators/assertions arrive from 10-01 on.

The stubs exist so every later chunk has a place to hang its assertions and so we
catch build-system / translate-c wiring problems now, in isolation, instead of
tangled with first-feature bugs.

## Acceptance

- `Roaring64Bitmap` constructs/destroys cleanly, re-exported from `roaring.zig`,
  inline lifecycle test passes under `test64` and `test`.
- `vendor/croaring_wrapper.h` declares the lifecycle `roaring64_*` functions; the
  validate64/difftest64 programs compile and **link** against them (proves the
  amalgam already provides the symbols — no amalgam change needed).
- `zig build test64 validate64 difftest64` all green (trivial bodies).
- No regression: `zig build test validate difftest` still green.
