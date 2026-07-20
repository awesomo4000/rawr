<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 10-19: `hasRunContainers` as a bitmap method (dedup)

**Refactor / cleanup — parked for the post-parity refactoring pass**, not part of
the parity feature work. No behavior change; pure de-duplication.

## Problem

The predicate *"does this bitmap contain any run-encoded container?"* (walk the
container list, check each 2-bit type tag for `.run`) is open-coded in **five**
places — over the 3+ threshold, and it's really a missing method rather than a
helper:

| # | location | scope |
|---|---|---|
| 1 | `serialize.zig` `hasRunContainers(bm: *const RoaringBitmap)` | production, 32-bit — picks `SERIAL_COOKIE` vs `SERIAL_COOKIE_NO_RUNCONTAINER` |
| 2 | `roaring64.zig` `hasRunContainers(self: *const Self)` | production, 64-bit — `runOptimize` return value |
| 3 | `roaring64_test_support.zig` `hasRunContainers(bm: anytype)` | test, 64-bit (generic) |
| 4 | `diff_test.zig` `rawrHasRunContainers(*const RoaringBitmap)` | test, 32-bit |
| 5 | `validate_croaring.zig` `bitmapHasRunContainers(*const RoaringBitmap)` | test, 32-bit |

Two consumers need it: serialization (cookie selection) and the test/oracle layer
(knowing when byte-identity comparison vs CRoaring is valid — the run-bearing
relaxation).

## Change

Make it the bitmap types' own API:

- **`RoaringBitmap.hasRunContainers(self: *const Self) bool`** — the canonical
  32-bit method (move the `serialize.zig` loop onto the type). Copies #1, #4, #5
  become `bm.hasRunContainers()`.
- **`Roaring64Bitmap.hasRunContainers(self: *const Self) bool`** — one level up
  (walk buckets, `bucket.bm.hasRunContainers()`), replacing the private copy #2.
- **Delete `roaring64_test_support.hasRunContainers`** (#3) — callers use
  `bm.hasRunContainers()` on whichever type they hold. (If a generic call site
  remains, `anytype` + `bm.hasRunContainers()` still works since both types now
  expose the method.)

All five open-coded loops collapse to method calls. Two methods, each of which is
just the type's own API.

## Constraints

- **Zero behavior change.** The logic is identical everywhere; this only relocates
  it. The existing 32-bit serialization round-trip tests + the run-bearing
  differential/serialization paths already cover both consumers — they must stay
  green unchanged.
- **Touches stable 32-bit production** (`serialize.zig`). That's the reason this is
  a deliberate refactor-pass item, not a drive-by edit: verify the serialization
  cookie selection is byte-identical before/after (the 32-bit `validate`/`difftest`
  suites confirm it).
- Keep it a plain `*const` method, allocation-free.

## Acceptance

- `hasRunContainers` exists as a public method on both `RoaringBitmap` and
  `Roaring64Bitmap`; the five open-coded copies are gone (serialize + both
  harnesses + roaring64 production + test-support).
- `roaring64_test_support` no longer defines `hasRunContainers`.
- `zig build test test64 validate validate64 difftest difftest64` all green with
  no behavior change (serialization output byte-identical).
