<!-- SPDX-License-Identifier: MPL-2.0 -->

# API-design pass — running notes

Scratchpad of topics for the upcoming ergonomics/clarity pass: making rawr's API
easy to use, clear, and hard to misuse. Not a spec — a place to accumulate ideas
as they surface, to be shaped into actual spec(s) when the pass starts. Feature
parity and performance work are complete (see
[`07-parity-inventory.md`](done/07-parity-inventory.md)).

## Topics surfaced so far

### Arena as a first-class perf lever
rawr's `*Owned` / arena path is meaningfully faster for allocation-heavy,
short-lived results (e.g. arena `bitwiseAnd`/`bitwiseOr` on sparse). The allocator
story is a strength worth making *explicit and ergonomic* — guide users toward an
arena for build-once/throw-away result bitmaps.
- **Caveat to design around:** arena is a trap for ops that allocate-then-discard
  intermediates. E.g. `lazyOr+repair` on sparse inflates ~65k tiny arrays to 8 KB
  bitsets then demotes them back; an arena never reclaims the dead bitsets →
  ~512 MB peak for a ~4 MB result. So "use an arena" is not universal advice; the
  API/docs should steer arena toward the clean-win ops and away from
  inflate-deflate ones.

### `lazyOr`/`lazyXor` usage guidance + the lazy-state footgun
- 2-way lazy is a non-scenario: for a 2-way union/xor use eager `bitwiseOr`/
  `bitwiseXor` (faster, lower memory). Lazy is for **n-way dense** chains, and
  `orMany`/`xorMany` already use the internal k-way fold — so the public
  `lazyOr`/`lazyXor` are rarely the right call directly.
- The lazy result is in an invalid state until `repairAfterLazy` — a real footgun.
  Design question: make this harder to misuse (a distinct "lazy/dirty" type that
  only exposes `repair()` → returns the usable bitmap? builder pattern?).

### Other themes to explore (flesh out when the pass starts)
- Naming consistency across the surface (`bitwiseOr`/`orMany`/`lazyOr`;
  `fromSorted`/`fromSlice`/`addMany` family; `*Owned` suffix convention).
- The error-union ergonomics (`!T` everywhere) — is there a cleaner story for the
  common infallible/owned paths?
- Allocator handling — explicit-allocator vs owned-allocator (`OwnedBitmap`)
  consistency; when each is the right pattern.
- Iterator / builder ergonomics.
- The `*Owned` API surface — which ops should have it, and is it discoverable.

## Disposition of leftover perf items (parked, low value)
- `orMany` ~2 µs residual — profile before any spec (the workspace guess regressed
  it).
- `lazyOr+repair (sparse)` ~3 ms gap — non-scenario; arena would cut time but blow
  memory (above). No fix; usage guidance instead.
