# Spec 00-04: CRoaring Interop Via Translate-C

## Goal

Replace deprecated `@cImport` usage with build-system `translate-c` imports for
the CRoaring validation and comparison benchmark targets.

## Scope

Primary files:

- `build.zig`
- `src/validate_croaring.zig`
- `src/bench_croaring.zig`
- `vendor/croaring_wrapper.h`, only if include structure requires a tiny change

## Current State

Baseline full-repo grep:

```bash
rg -n "@cImport" .
```

Current matches:

- `src/validate_croaring.zig`
- `src/bench_croaring.zig`
- `vendor/croaring_wrapper.h` comment text only
- `docs/roaring-zig-architecture.md` stale documentation example
- `specs/00-upgrade-to-zig-0.16.md` and this spec
- older specs under `specs/todo/` and `specs/done/`

`src/validate_croaring.zig` and `src/bench_croaring.zig` both do:

```zig
const c = @cImport(@cInclude("croaring_wrapper.h"));
```

The current `build.zig` adds the include path and C source directly to each
module. Keep those C compile/link settings, but provide translated C bindings
as a module import named `c`.

## Implementation Notes

In `build.zig`, create a translate-c step for the wrapper header using the same
target/optimize settings as the executable module. Add `vendor/` as an include
path to the translate-c step if required by the API.

Then replace source imports with:

```zig
const c = @import("c");
```

Avoid widening the C surface area. Keep `croaring_wrapper.h` as the only header
rawr imports from Zig.

CRoaring interop should stay late in the migration because it is isolated to
validation/benchmark targets and should not block the pure Zig library upgrade.

## Dependencies

- `00-03` complete

## Validation

```bash
zig build validate
zig build bench-compare
rg -n "@cImport" .
```

The final full-repo grep must show no remaining source call sites. Any remaining
matches must be intentional migration/spec history or documentation references,
not active Zig code. Prefer updating stale docs/comments so the only matches are
in specs that explicitly discuss the migration.

## Checklist

- [ ] Add translate-c module import for CRoaring validation
- [ ] Add translate-c module import for CRoaring benchmark comparison
- [ ] Replace `@cImport` in Zig source
- [ ] `rg -n "@cImport" .` has been reviewed across the whole repo
- [ ] Full-repo grep confirms zero remaining active Zig `@cImport` call sites
- [ ] Keep C source compile flags and libc linkage intact
- [ ] `zig build validate` passes
- [ ] `zig build bench-compare` builds
