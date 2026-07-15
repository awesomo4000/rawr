<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 01-08: Round-trip and addRange coverage (extend existing)

Chunk of [`01-differential-testing.md`](01-differential-testing.md). Extends the
existing `validate_croaring.zig` to drive the generator profiles and to cover
`addRange` across container types.

## Dependencies

- **01-02** (`test_gen` profiles / `randomMixed`).
- **01-01** (wrapper) — already linked by the `validate` build.

## Carry-over facts

- `validate_croaring.zig` already does build→serialize→cross-deserialize well,
  including a first-divergence-byte reporter (`validateRoundTrip`) and the
  inclusive/exclusive range handling (`validateRangeRoundTrip`).
- rawr `addRange(start, end)` is **inclusive** both ends; CRoaring
  `roaring_bitmap_add_range(min, max)` is **exclusive** on max → always pass
  `@as(u64, end) + 1`. Preserve this convention.

## Task

Extend `validate_croaring.zig`'s fixture list to drive the generator profiles
rather than hand-rolled arrays:

1. Round-trip a `randomMixed` bitmap for each profile combination, both
   run-optimized and not.
2. **Keep** the existing hand-picked boundary fixtures (chunk boundaries,
   `NO_OFFSET_THRESHOLD` container counts 3/4/5) — those are valuable and targeted.
3. Add `addRange` differential cases that span container types:
   - a range that produces a **run**,
   - a range > 4096 that produces a **bitset**,
   - a range crossing several chunk boundaries.
   Compare bytes against `roaring_bitmap_add_range` (remember the `+1`
   exclusive-end convention).

## Acceptance criteria

1. `validate_croaring.zig` round-trips `randomMixed` bitmaps across profiles,
   run-optimized and not, with byte-identical cross-deserialization.
2. Existing boundary fixtures retained.
3. The three `addRange` cross-container cases pass against the oracle with the
   correct `+1` convention.
4. `zig build validate` passes.
