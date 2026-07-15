<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 01-01: Extend the CRoaring wrapper

Chunk of [`01-differential-testing.md`](01-differential-testing.md). **First and
enabling** — every later chunk assumes these declarations exist. Low risk, ~30 min.

## Context

CRoaring is the oracle for the differential suite, reached via the translated
`vendor/croaring_wrapper.h` binding. Zig 0.16 uses build-system `translate-c`
(`b.addTranslateC`), imported as `const c = @import("c");` — there is **no
`@cImport`**. The header is already wired into the `validate` and `bench-compare`
builds via the `addTranslatedCImport` helper in `build.zig`.

`vendor/croaring_wrapper.h` already exposes creation, basic ops, all four set ops
+ their in-place forms, `run_optimize`, and portable serialization.

## Task

Add these six declarations to `vendor/croaring_wrapper.h` — they are the ones the
differential harness needs and are **currently missing**:

```c
bool roaring_bitmap_equals(const roaring_bitmap_t*, const roaring_bitmap_t*);
bool roaring_bitmap_is_subset(const roaring_bitmap_t*, const roaring_bitmap_t*);
uint64_t roaring_bitmap_and_cardinality(const roaring_bitmap_t*, const roaring_bitmap_t*);
bool roaring_bitmap_intersect(const roaring_bitmap_t*, const roaring_bitmap_t*);
uint32_t roaring_bitmap_minimum(const roaring_bitmap_t*);
uint32_t roaring_bitmap_maximum(const roaring_bitmap_t*);
```

`roaring_bitmap_andnot` and `roaring_bitmap_andnot_inplace` are **already present**
— do not re-add them.

Keep the wrapper minimal: do **not** vendor the whole CRoaring header. This stays
within the spirit of the `00-04` interop chunk (single imported header, minimal C
surface).

## Acceptance criteria

1. The six declarations are present in `vendor/croaring_wrapper.h`, matching the
   real CRoaring signatures.
2. `zig build validate` and `zig build bench-compare` still build with no other
   change — confirming the `translate-c` step picks up the new symbols (they
   become `c.roaring_bitmap_*` automatically).
3. No new C surface beyond these six; `andnot`/`andnot_inplace` untouched.

## Dependencies

None. Blocks: every chunk that calls the oracle (01-03 onward).
