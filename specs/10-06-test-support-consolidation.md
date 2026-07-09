# Spec 10-06: Roaring64 test-support consolidation

Cleanup chunk. By the end of v1 (10-00 … 10-05) the 64-bit test code accumulated
real duplication across the two harness executables (`validate_roaring64.zig`,
`diff_test64.zig`) and the inline/property tests (`roaring64.zig`,
`roaring64_property_tests.zig`). Morty already consolidated the **generator**
into `roaring64_test_gen.zig`; this chunk finishes the job for the **oracle,
assertion, and fixture helpers**. Pure test-code refactor — no behavior change to
`Roaring64Bitmap`, no new features.

## Why roaring64-only (not a general 32+64 module)

The 32-bit harnesses (`diff_test.zig` ~1700 lines, `validate_croaring.zig`) are
stable, done, and **type-differentiated** — every helper is typed against
`RoaringBitmap` / `roaring_bitmap_t` and the 32-bit CRoaring function set. A
shared 32+64 module would require comptime-generic abstraction over *both* bitmap
types *and* both CRoaring API surfaces (`roaring_bitmap_*` vs `roaring64_bitmap_*`),
which means reworking stable, passing 32-bit test code for marginal gain and real
regression risk. **Out of scope.** Keep 10-06 confined to the 64-bit test files.
(If a 32-bit test refactor is ever independently wanted, that's its own chunk.)

## The duplication being removed

Exact counts at end of v1:

- **`rawrHasRunContainers` / `roaring64HasRunContainers` — 3 copies** (`diff_test64`,
  `validate_roaring64`, `roaring64.zig`). *(A 4th, 32-bit-typed copy lives in
  `diff_test.zig` — leave it; see scope note above.)*
- **Oracle-comparison family — 2 copies each** across `validate_roaring64` ↔
  `diff_test64`: `buildCRoaring`, `assertAgreement`, `assertPositionalAgreement`,
  `applyAddRange`, `applyRemoveRange`, `assertRangeAgreement`,
  `cContainsRangeClosed`, and the serialization cross-check
  (`assertSerializationAgreement` ≈ `validateSerializationCase`).
- **Fixture builders — 2 copies**: `buildFrame` ≈ `buildRoaring64Frame`
  (`roaring64_property_tests` ↔ `roaring64.zig`); the malformed-frame battery
  (`roaring64.zig` inline test ↔ `roaring64_property_tests`).
- **Own generator not migrated**: `validate_roaring64` still carries its own
  `fillGeneratedCorpus` instead of using the shared `roaring64_test_gen`.

## Design — split by the CRoaring dependency boundary

The single hard constraint: **`roaring64.zig`'s inline tests and
`roaring64_property_tests.zig` do NOT link CRoaring** (no `c` module wired), but
the two harness executables do. So the shared helpers split into two modules:

### 1. `roaring64_test_support.zig` — CRoaring-free, shared by everyone

Importable by `roaring64.zig` tests, `roaring64_property_tests.zig`, **and** both
harnesses. Must not `@import("c")`. Where it needs the bitmap type, make it
**generic** (`fn hasRunContainers(bm: anytype) bool`, `fn fromValues(comptime
Bitmap: type, ...)`) — mirroring `roaring64_test_gen.toBitmap`'s existing
`comptime Bitmap` pattern — so the module does not `@import("roaring64.zig")` and
no import cycle forms.

Moves here:
- `hasRunContainers(bm: anytype) bool` (the 3 → 1 collapse).
- `buildFrame(allocator, keys, sub_bitmap) ![]u8` (the raw 64-bit frame builder).
- `fromValues(comptime Bitmap, allocator, values)` (the `roaring64FromValues`
  helper, if not already covered by `roaring64_test_gen`).
- `expectSerializationRoundTrip(bm)` (rawr→bytes→rawr plain+safe, size check).
- `expectMalformedFramesRejected(allocator, bm)` — the malformed-frame battery
  (empty input, truncation sweep, count > maxInt(u32), non-ascending keys, empty
  sub-bitmap) as one callable, so `roaring64.zig` and `roaring64_property_tests`
  both invoke it instead of open-coding it twice.

### 2. `roaring64_oracle.zig` — CRoaring-dependent, harness-only

Imported by `validate_roaring64.zig` and `diff_test64.zig` only (both wire `c`).
`@import("c")` lives here.

Moves here:
- `buildCRoaring(values) !*c.roaring64_bitmap_t`.
- `assertAgreement(allocator, rbm, cr, probes)` — cardinality / empty / contains /
  min-max (with the empty-sentinel special case) / toArray / iterator.
- `assertPositionalAgreement(rbm, cr, probes)` — rank / getIndex / select.
- `assertRangeAgreement(rbm, cr, lo, hi)` + `cContainsRangeClosed` (the `hi+1`
  overflow special-case) + `applyAddRange` / `applyRemoveRange`.
- `assertSerializationAgreement(allocator, rbm, cr)` — the single canonical copy,
  including the **run-bearing byte-identity relaxation** (run-optimize the oracle;
  for run-bearing rawr bitmaps fall back to cross-deserialize `equals`, per 10-04
  / 10-05). This is the highest-value dedup: it's the most intricate helper and
  currently exists as two hand-kept-in-sync copies.

## Migration steps

1. Create the two modules above; export `roaring64_test_support` from
   `roaring.zig` alongside `roaring64_test_gen` (keep `roaring64_oracle` out of
   the public `roaring.zig` surface — it's harness-internal and `c`-dependent).
2. Rewrite `validate_roaring64.zig` and `diff_test64.zig` to import both modules;
   delete their local copies of every helper listed above.
3. Migrate `validate_roaring64.zig` onto `roaring64_test_gen` (delete its private
   `fillGeneratedCorpus`; use `build`/`randomMixed`/edge profiles).
4. Rewrite `roaring64.zig`'s serialization/malformed inline tests and
   `roaring64_property_tests.zig`'s frame/malformed tests to call the shared
   `roaring64_test_support` helpers; delete the open-coded duplicates.
5. Confirm the wiring: `roaring64_oracle.zig` needs the same
   `addTranslatedCImport` + `addBenchmarkPlatformShim` treatment its importers
   already have — but since it's imported *by* the harness modules (not a separate
   executable), it inherits their `c` import; verify it resolves `@import("c")`
   through the importing module.

## Acceptance

- No `Roaring64Bitmap` source change; only test files + two new support modules.
- `rawrHasRunContainers`/`roaring64HasRunContainers` exists in **one** place for
  the 64-bit code (generic `hasRunContainers`); the serialization cross-check and
  the oracle-comparison family each exist once.
- `validate_roaring64` uses `roaring64_test_gen`; no private `fillGeneratedCorpus`
  remains in the 64-bit harnesses.
- The malformed-frame battery and frame builder each exist once, called from both
  the inline and property tests.
- `zig build test test64 validate64 difftest64` all green, unchanged behavior —
  same assertions run, same seeds, just sourced from shared helpers. No 32-bit
  test file touched.
