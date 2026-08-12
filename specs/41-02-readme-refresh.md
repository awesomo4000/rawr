<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 41-02: `README.md` — remove performance claims, refresh stale inventories

Toplevel: [41-documentation-parity.md](41-documentation-parity.md). Independent of `41-00`/`41-01`;
may land in any order. No production code changes.

## 1. Remove or neutralize every performance claim

**No numbers, ratios, comparisons against the reference C library, or "fast"/"faster"/"fastest"
framing.** A README benchmark claim is read as universal, while every number we hold is scoped to one
harness, three allocators, and two architectures — and it goes stale silently on each later commit.
Measurements belong in `docs/` and the specs, where their scope travels with them.

Verified present:

| Line | Text | Action |
| --- | --- | --- |
| 103 | "**Faster** for deserialize and set operations" | remove/neutralize |
| 142 | "`std.heap.c_allocator` was roughly **1.3-1.8x slower** than alternatives" | remove |
| 148 | "Recommended allocators, **fastest to most flexible**" | re-word |
| 150–152 | Speed column: **`Fastest` / `Fast` / `Good`** | re-cast as characteristics |
| 144 | relative link to `docs/parity-measurement.md` | **remove — broken for consumers** |

**Keep the allocator guidance itself** — it is genuinely useful. Re-cast the Speed column as
**workload/lifetime characteristics**, matching what `41-01` does to `API.md`'s Allocator Guide.

**On the link:** `docs/` is **not** in `.paths`, so for anyone who received only the package that link
resolves to nothing. Removal is cleanest and consistent with the no-claims direction. An absolute GitHub
URL or adding the file to `.paths` would also work, but both re-introduce measurements into the shipped
set.

*(Not in scope: the "high-performance" phrasing lives in `src/roaring.zig`'s doc comment, not the
README.)*

## 2. Bitmap-types table (line ~92) — half the library is missing

Currently lists only `RoaringBitmap`, `OwnedBitmap`, `FrozenBitmap`. **Add `Roaring64Bitmap` and
`Frozen64Bitmap`** — as written, the table omits the entire 64-bit half, which is also the half `API.md`
under-documents. A reader has no entry point to it from either document.

## 3. Project structure (line ~205)

Stale well beyond `tools/`. Missing at least: `roaring64.zig`, `frozen64.zig`, the `roaring64_*`
test/support files, `range_ops.zig`, `array_kernels.zig`, `array_simd.zig`, and under `tools/` the new
`check_32_api.zig` and `cross_width_fixture.zig`.

**Also remove the volatile claim at line 186** — "~9400 lines of Zig across 18 source files". Both
numbers are wrong the next time a file is added, and neither tells a reader anything actionable. Same
class of problem as the performance numbers: a measurement pinned into prose that nothing keeps honest.

## 4. Verify the 32-bit section (line ~28)

Added by spec 40. Confirm the target list and commands match what `check-32` actually builds, and that no
absolute or user-specific paths remain.

## Acceptance

- All five §1 claim sites removed or neutralized; allocator guidance retained as characteristics.
- `docs/parity-measurement.md` link removed; **no relative link from `README.md` to a path outside
  `.paths`**.
- Bitmap-types table includes `Roaring64Bitmap` and `Frozen64Bitmap`.
- Project structure refreshed; line-count/file-count claim removed.
- 32-bit section verified against `check-32`.
- **No performance claim remains in `README.md`** — verify by reading, and record the check.
- No production code changed; **all four suites green — `test`, `difftest`, `test64`, `difftest64`.**

## Estimate

**S** — needs care rather than volume.
