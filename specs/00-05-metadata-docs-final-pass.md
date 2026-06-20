# Spec 00-05: Metadata, Docs, And Final Validation

## Goal

Update package metadata and user-facing documentation after the codebase builds
cleanly on Zig 0.16.0, then run the complete validation matrix.

## Scope

Primary files:

- `build.zig.zon`
- `README.md`
- `API.md`, only if it references old Zig I/O or allocator guidance that changed
- `CLAUDE.md`, only if keeping it in sync is still desired after `AGENTS.md`
- `specs/00-upgrade-to-zig-0.16.md`

This chunk is also for integration cleanup. Avoid new feature work and unrelated
refactors.

## Implementation Notes

Update `build.zig.zon`:

```zig
.minimum_zig_version = "0.16.0",
```

Update build instructions in README from Zig 0.15.2+ to Zig 0.16.0.

Check docs for stale references:

- `std.io writer` wording should become a generic Zig writer or `std.Io.Writer`.
- Examples should still compile in spirit under Zig 0.16.0.
- Allocator guidance should remain unchanged unless Zig 0.16.0 behavior proves otherwise.

Do not rewrite broad docs unrelated to the version upgrade.

## Dependencies

- `00-01` through `00-04` complete

## Required Commands

```bash
zig build
zig build test
zig build validate
zig build bench
zig build bench-compare
zig build bench-alloc
```

Also run optimized test builds if practical:

```bash
zig build test -Doptimize=ReleaseFast
zig build test -Doptimize=ReleaseSafe
```

## Compatibility Checks

- Serialization tests still pass.
- `validate` confirms CRoaring round-trip compatibility.
- Public API docs still match exported names from `src/roaring.zig`.
- Benchmark executables build without deprecated API usage.
- No new broad formatting churn.

## Checklist

- [x] Update `minimum_zig_version`
- [x] Update README version requirement
- [x] Remove stale `std.io` documentation references
- [x] Keep API examples aligned with Zig 0.16.0
- [x] `zig build` passes
- [x] `zig build test` passes
- [x] `zig build validate` passes
- [x] `zig build bench` builds
- [x] `zig build bench-compare` builds
- [x] `zig build bench-alloc` builds
- [x] Optimized test builds run or skipped with a documented reason
- [x] Main spec checklist updated
