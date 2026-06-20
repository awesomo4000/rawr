# Spec 00-02: Library And Unit Test Compatibility

## Goal

Fix any remaining pure Zig library or unit-test compilation failures after the
serialization API migration, while keeping exported library behavior stable.

Benoit's chunking note: keep this separate from `00-01` because serialization is
the known first blocker, and this chunk captures unknown fallout without
expanding the first change.

## Scope

Primary files:

- `src/*.zig` library modules
- `src/*_tests.zig`
- `src/property_tests.zig`

Out of scope:

- Benchmark executable entrypoints
- CRoaring translate-c migration
- Metadata/docs updates

## Implementation Notes

Run `zig build test` after `00-01` and address the next pure-library failures in
small patches.

Likely areas to check if failures appear:

- stdlib renames from the Zig 0.16.0 cheatsheet
- generic reader/writer pointer receivers
- runtime vector indexing restrictions
- packed struct/union restrictions
- `std.mem` find/index rename fallout

Do not refactor algorithms unless the compiler requires a small structural
change. Preserve public API names exported from `src/roaring.zig`.

## Dependencies

- `00-01` complete

## Validation

```bash
zig build test
```

## Checklist

- [x] Run `zig build test` after `00-01`
- [x] Fix remaining pure library/test Zig 0.16.0 compile failures
- [x] Preserve public API behavior
- [x] `zig build test` passes
