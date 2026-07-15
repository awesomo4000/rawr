<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 00-04: CRoaring Interop Via Translate-C

## Goal

Replace deprecated legacy C import builtin usage with build-system `translate-c` imports for
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
rg -n '@''cImport' .
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
const c = @import("c");
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
rg -n '@''cImport' .
```

The final full-repo grep must show no remaining matches. The quoted shell
command above searches for the deprecated builtin without placing its literal
spelling in this spec.

## Checklist

- [x] Add translate-c module import for CRoaring validation
- [x] Add translate-c module import for CRoaring benchmark comparison
- [x] Replace legacy C import builtin usage in Zig source
- [x] `rg -n '@''cImport' .` confirms zero remaining matches across the whole repo
- [x] Keep C source compile flags and libc linkage intact
- [x] `zig build validate` passes
- [x] `zig build bench-compare` builds
