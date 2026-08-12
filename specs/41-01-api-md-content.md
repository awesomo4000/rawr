<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 41-01: `API.md` content — prose, `Roaring64Bitmap`, and contracts

Toplevel: [41-documentation-parity.md](41-documentation-parity.md). Gated on:
[41-00](41-00-check-docs-guard.md).

**Coverage is already satisfied** — `41-00` populated the guarded Quick Reference, so `check-docs` is
green going in. This chunk adds **explanation**, not coverage. No production code changes.

## 1. Work list

**The authoritative list is `41-00`'s recorded first-run inventory**, not the counts below. Those came
from bare-name matching and are a floor.

**`Roaring64Bitmap` — needs a section of its own.** `API.md` mentions it six times in passing and gives
it **no section**, while `OwnedBitmap`, `FrozenBitmap`, and `Frozen64Bitmap` each have one — for the type
carrying the entire 64-bit value-range claim. Cover: construction, the `*Bulk` family (`addBulk`,
`containsBulk`, `removeBulk`), `fromRange`, `fromSortedSlice`, the `fromRoaring32`/`toRoaring32`
conversion pair, `clone`, `flipInPlace`, `statistics`, and a pointer to the portable-format note at
line ~305.

**`RoaringBitmap`** — `bitwiseOrInPlaceConsume`, `removeRangeCopy`, `repairAfterLazyWithOptions`,
`clone`.

**`FrozenBitmap`** (spec 42) — `rank`, `select`, `getIndex`, `minimum`, `maximum`.

Document these in the existing topical sections (Construction, Mutation, Set Operations, Extraction),
**not** an appendix.

## 2. Behavioural contracts

**Describe the contract, not the benchmark.** No ratios or board numbers in `API.md`. *"Frees in
descending order, which some allocators reward"* is usable guidance; *"1.033x on M4"* rots on the next
commit and is scoped to a harness the reader does not have.

- **`bitwiseOrInPlaceConsume`** — **ownership**, the important one. It consumes its right operand;
  state precisely what the caller may and may not do with that operand afterwards. A correctness
  contract, not a performance note.
- **`repairAfterLazyWithOptions`** — opt-in; the default `repairAfterLazy` is unchanged; it changes free
  order only; any benefit is **allocator-dependent**; measure on your own allocator and workload. Claim
  no speedup.
- **`removeRangeCopy`** — constructs only the surviving containers rather than cloning and discarding.
- **`FrozenBitmap.minimum`/`maximum`** — array and run are direct reads; **bitset scans up to 1,024
  words**. **Do not write "O(1)".** `rank`/`select`/`getIndex` are **O(containers + one container
  probe)** — the frozen descriptor holds per-container cardinalities and **no prefix sums**.

## 3. Neutralize the Allocator Guide (line ~393)

The table currently ranks and advises on performance:

| Row | Current text | Problem |
| --- | --- | --- |
| `OwnedBitmap` helpers | "**Fast** temporary read-only results." | speed claim |
| `std.heap.c_allocator` | "**Avoid** for rawr's many small allocations." | perf-derived directive, and contradicted by our own measurements — some allocation-heavy operations favour libc |

Re-cast the column as **workload and lifetime characteristics**, matching the direction `41-02` applies
to `README.md`. Keep the guide — allocator choice genuinely matters — but describe *when a shape fits*,
not which is faster. The `c_allocator` row in particular should describe interop and allocation-pattern
trade-offs rather than issue an avoid-directive the board does not support.

## Acceptance

- Every function in `41-00`'s recorded inventory documented in prose, in its topical section.
- `Roaring64Bitmap` section present and peer to the other type sections.
- §2 contracts stated, with **no ratios or board numbers anywhere in `API.md`**.
- Allocator Guide neutralized per §3 — no "Fast", no avoid-directive.
- **`zig build check-docs` still passes with an empty allow-list.** Prose must not disturb the guarded
  region; if a Quick Reference entry changes, the region stays complete.
- No production code changed; **all four suites green — `test`, `difftest`, `test64`, `difftest64`.**

## Estimate

**M** — the `Roaring64Bitmap` section is the bulk.
