<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 42: Frozen rank/select — close the layering inversion

**Question that prompted this.** `Frozen64Bitmap` exposes `rank`, `select`, `minimum`, `maximum`, and
`getIndex` while `FrozenBitmap` does not — yet the 64-bit type is layered on the 32-bit one. How does the
upper layer have operations its own foundation lacks?

**Answer: it does not have them, it re-derives them by brute force.** `src/frozen64.zig` carries three
private helpers — `frozen32Rank`, `frozen32Select`, `frozen32GetIndex` — each of which constructs a
`FrozenBitmap.Iterator` over a sub-view and walks it one value at a time:

```zig
fn frozen32Rank(sub: *const FrozenBitmap, value: u32) u64 {
    var count: u64 = 0;
    var it = sub.iterator();
    while (it.next()) |current| {
        if (current > value) break;
        count += 1;
    }
    return count;
}
```

`minimum` and `maximum` are the same shape inline: `maximum` (`frozen64.zig:69`) iterates the **entire**
final bucket and keeps the last value seen.

So the layering is not inverted — the **work** is. These are O(values) scans standing in for operations
that are O(log n) plus one container probe everywhere else in the library.

## 1. The metadata is already present

This is not a case where the frozen format lacks what the operations need. `FrozenBitmap` already has, as
private methods:

| Member | Line | Provides |
| --- | --- | --- |
| `getCardinality(idx)` | `frozen.zig:137` | per-container cardinality — lets `rank`/`select` skip whole containers |
| `findKey(key)` | `frozen.zig:149` | binary search over the key array |
| `isRunContainer(idx)` | `frozen.zig:143` | container-type discrimination |
| `getContainerDataOffset` / `getContainerSize` | `frozen.zig:171` / `187` | direct container payload access |
| `binarySearchArray` | `frozen.zig:234` | already used by `contains` |

`FrozenBitmap.contains` is already binary-search-based. Rank and select were simply never written; the
64-bit layer needed them, could not call them, and open-coded scans instead.

## 2. Deliverable

- **Implement `rank`, `select`, `minimum`, `maximum`, `getIndex` as public methods on `FrozenBitmap`,**
  container-aware, mirroring the `RoaringBitmap` implementations rather than inventing new logic.
  `minimum`/`maximum` read the first/last container directly — neither requires a scan.
- **`Frozen64Bitmap` delegates.** Delete `frozen32Rank`, `frozen32Select`, `frozen32GetIndex`. The
  bucket-level skipping in `frozen64.zig` (already cardinality-based and correct) stays; only the
  within-bucket work changes.
- **Behaviour must not change.** Same results for every input, including empty bitmaps, absent values,
  out-of-range `select`, and the run-container path.

## 3. Correctness before speed

The frozen types read a serialized image directly, so a wrong offset is a **memory-safety** bug, not a
wrong answer. `FrozenBitmap` already carries `validateContainerBounds` and `checkedContainerSize`; any new
container-payload access must go through the same checked accessors.

**Differential requirement, and it is the point of the chunk:** for every corpus bitmap, and across all
three container types, assert new `FrozenBitmap.rank/select/minimum/maximum/getIndex` **equal
`RoaringBitmap`'s** for the same value set — and equal the current linear implementations, which are slow
but are the trusted reference. Keep the old helpers alive under test until the differential passes, then
delete them.

Include: empty bitmap, single container of each type, values below/above all elements, `select(0)`,
`select(cardinality-1)`, `select(cardinality)` → null, and a run container whose runs span a
container boundary.

## 4. Ordering against spec 41

**Land 42 before `41-01` if convenient.** `41-01` has to document these five methods either way; if 42
lands first it documents one honest cost class instead of two, and `FrozenBitmap` gains five documented
methods. They are otherwise independent — 41 may proceed without 42, noting the cost asymmetry per
spec 41 §4.

## 5. Out of scope

- Adding mutation to the frozen types — they are read-only views by design.
- Changing the frozen serialization format. This spec reads the existing image; it does not alter it.
- `Frozen64Bitmap`'s bucket-level loops in `cardinality`/`rank`/`select` — they are already
  cardinality-driven and correct.
- Any parity-board row. Frozen types are not on the board, and **no board row may move**; if one does,
  that is an unintended coupling and stops the chunk.

## 6. Acceptance

- `FrozenBitmap` exposes `rank`, `select`, `minimum`, `maximum`, `getIndex`, container-aware, using the
  existing checked accessors.
- `frozen32Rank`, `frozen32Select`, `frozen32GetIndex` **deleted**; `Frozen64Bitmap` delegates and its
  observable behaviour is unchanged.
- Differential equality against `RoaringBitmap` **and** against the pre-change linear implementations,
  over all three container types plus the edge cases in §3.
- **Negative control:** perturb one new implementation (e.g. off-by-one in the rank accumulation) and
  confirm the differential test **fails and names the case**. A differential that passes a seeded defect
  is not evidence.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `ReleaseSafe` and
  `ReleaseFast`. `zig build check-32` still passes; new public methods on a public type must be added to
  the `check-32` probe (spec 40-01: the probe's enumerated surface **is** the guard boundary, and
  omissions fail silently).
- No parity-board row moves.

## 7. Chunking sketch

Not chunked — pending review of this toplevel. Plausibly single-chunk (**S/M**): the implementations are
small and modelled on existing `RoaringBitmap` code, with the differential harness the larger half. Split
into 42-00 (`FrozenBitmap` methods + differential) and 42-01 (`Frozen64Bitmap` delegation + helper
deletion) only if the differential proves bulky.
