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
```

## Checklist

- [ ] Add translate-c module import for CRoaring validation
- [ ] Add translate-c module import for CRoaring benchmark comparison
- [ ] Replace `@cImport` in Zig source
- [ ] Keep C source compile flags and libc linkage intact
- [ ] `zig build validate` passes
- [ ] `zig build bench-compare` builds
