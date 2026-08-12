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

So the layering is not inverted — the **work** is. These walk individual values where the rest of the
library walks containers.

## 1. What the format does and does not provide

**Corrected from the first draft, which overstated this.** The frozen descriptor stores per-container
key/cardinality pairs (`frozen.zig:54`) — it has **no prefix sums**. So `rank`, `getIndex`, and `select`
must still scan the preceding containers to accumulate a running count. The gain is not a search:

> **O(values) → O(containers + one container probe).**

That is the same shape `RoaringBitmap` already has — it also scans containers to accumulate rank. This
spec brings `FrozenBitmap` to the existing standard; it does not beat it. Prefix sums would be a format
change and are out of scope (§5).

The metadata needed for that shape is already present as private members of `FrozenBitmap`:

| Member | Line | Provides |
| --- | --- | --- |
| `getCardinality(idx)` | `frozen.zig:137` | per-container cardinality — accumulate, and skip whole containers |
| `findKey(key)` | `frozen.zig:149` | binary search over the key array |
| `isRunContainer(idx)` | `frozen.zig:143` | container-type discrimination |
| `getContainerDataOffset` / `getContainerSize` | `frozen.zig:171` / `187` | container payload location |
| `binarySearchArray` | `frozen.zig:234` | already used by `contains` |

`FrozenBitmap.contains` is already binary-search-based. Rank and select were simply never written; the
64-bit layer needed them, could not call them, and open-coded scans instead.

## 2. Deliverable

- **Implement `rank`, `select`, `minimum`, `maximum`, `getIndex` as public methods on `FrozenBitmap`,**
  container-aware, mirroring the `RoaringBitmap` implementations rather than inventing new logic.
- **`minimum`/`maximum` read the first/last container — but the cost is per-container-type:** array and
  run are direct reads of the first/last element; **bitset still scans up to 1,024 words** for the first
  or last set bit. They stop scanning *values*; they are not universally constant-time. Document it that
  way, and do not write "O(1)".
- **`Frozen64Bitmap` delegates.** Delete the production `frozen32Rank`, `frozen32Select`,
  `frozen32GetIndex`. The bucket-level skipping in `frozen64.zig` is already cardinality-driven and
  correct — only the within-bucket work changes.
- **Behaviour must not change.** Same results for every input, including empty bitmaps, absent values,
  and out-of-range `select`.

## 3. Safety — validated at `init`, infallible at query

The frozen types read a serialized image directly, so a bad offset is a **memory-safety** bug, not a wrong
answer. But the existing design already places that burden correctly, and the first draft of this spec got
the mechanism wrong:

`validateContainerBounds` / `checkedContainerSize` run during **`init`**. Once `init` succeeds the image is
immutable and validated, which is exactly why the query methods are infallible (`contains` returns `bool`,
`cardinality` returns `u64`). **Re-checking per query would force error-union return types and change the
public API — that is not wanted.**

So the requirement is: new methods use the **already-validated offsets** and read via the existing
little-endian helpers. **No pointer-casting of serialized bytes**, no fresh bounds arithmetic invented
alongside the validated accessors. New methods stay infallible.

## 4. Differential correctness — the substantial part

For every corpus bitmap, and across all three container types, assert new
`FrozenBitmap.rank/select/minimum/maximum/getIndex` **equal `RoaringBitmap`'s** for the same value set,
**and** equal the iterator-based linear results.

**The linear oracle is test-only.** The production helpers in `frozen64.zig` are deleted per §2, but
equivalent iterator-based implementations must be **retained in test code** as the trusted reference —
they are slow and obviously correct, which is precisely what an oracle should be. Deleting them outright
would discard the only independent check.

**Failures must be case-labelled** — which bitmap, which container type, which operation, which input. An
assertion that reports only "expected X, found Y" makes a differential over a corpus nearly unusable.

Cases to include:

- empty bitmap; single container of each of the three types
- values below all elements, above all elements, and absent from a populated container
- `select(0)`, `select(cardinality - 1)`, `select(cardinality)` → null
- **a range crossing `65535 → 65536`**, which produces **two** run containers — the boundary case that
  matters. *(The first draft asked for "a run spanning a container boundary", which cannot exist: runs
  live inside a single 16-bit-keyed container.)*
- **a single container holding multiple disjoint runs**

## 5. Out of scope

- Adding prefix sums or otherwise changing the frozen serialization format.
- Adding mutation to the frozen types — they are read-only views by design.
- `Frozen64Bitmap`'s bucket-level loops in `cardinality`/`rank`/`select` — already correct.
- Any change to mutable-bitmap or benchmark code paths.

## 6. Acceptance

- `FrozenBitmap` exposes `rank`, `select`, `minimum`, `maximum`, `getIndex`, container-aware, infallible,
  reading through validated offsets and little-endian helpers per §3.
- Production `frozen32Rank` / `frozen32Select` / `frozen32GetIndex` **deleted**; `Frozen64Bitmap`
  delegates; observable behaviour unchanged. Iterator-based equivalents retained **test-only** as oracle.
- Differential equality against `RoaringBitmap` **and** the linear oracle, over all three container types
  plus every §4 case, with case-labelled failures.
- **Negative control:** seed an off-by-one in the rank accumulation and confirm the differential **fails
  and names the case**. A differential that passes a seeded defect is not evidence.
- **Structural gate in place of a speed gate.** This is a complexity change, not a tuned optimization, and
  frozen types are absent from the parity board — so rather than stand up a new timing harness, require
  that the production `rank`/`select`/`minimum`/`maximum`/`getIndex` paths on both frozen types
  **construct no `Iterator` and perform no value-at-a-time traversal**. Verify by reading the
  implementations and record the confirmation. A before/after timing on a dense multi-container image may
  be recorded as evidence, but is **not** a gate.
- **`check-32` probe extended to the frozen surface.** Instantiate **both** `FrozenBitmap` and
  `Frozen64Bitmap` and call their full query surfaces — not merely the five new `FrozenBitmap` methods,
  or the new `Frozen64Bitmap` **delegation paths** stay outside the guard. Per spec 40-01, the probe's
  enumerated surface **is** the guard boundary and omissions fail silently; that finding cost four real
  defects, so this is not optional tidying.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `ReleaseSafe` and
  `ReleaseFast`.
- **No mutable-bitmap or benchmark path changed; a canonical board run is not required.** *(Replaces the
  first draft's "no parity-board row moves", which was not operationally defined — literal zero movement
  is impossible under measurement noise, and these methods are not on the board. If any mutable or
  benchmark path does end up touched, that assumption is void: pin the host, use the five-process
  protocol, and state a tolerance.)*

## 7. Ordering against spec 41

**Land 42 before `41-01` if convenient.** `41-01` documents these five methods either way; if 42 lands
first it documents one honest cost class instead of two, and `FrozenBitmap` gains five documented methods.
Otherwise independent — 41 may proceed, noting the cost asymmetry per spec 41 §4.

## 8. Chunking

**Single chunk — [42-00](42-00-frozen-rank-select.md), cleared and ready to implement.** The
implementations are small and modelled on existing `RoaringBitmap` code; the differential and
negative-control coverage are the substantial part. **S/M.**

`42-00` additionally folds in the review note to extend the existing `Frozen64Bitmap` round-trip test
(`roaring64.zig:2203`) with absent-value and bucket-boundary probes — it already exercises every
delegated method, so it is the natural companion to the helper deletion.
