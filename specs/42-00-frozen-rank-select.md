<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 42-00: Frozen rank/select — implementation and differential

Toplevel: [42-frozen-rank-select.md](42-frozen-rank-select.md). **Single chunk** — the whole of spec 42.

No format change, no mutable-bitmap change, no benchmark change.

## 1. Implement on `FrozenBitmap` (`src/frozen.zig`)

Five new **public**, **infallible** methods, container-aware, modelled on the existing `RoaringBitmap`
implementations:

`rank`, `select`, `minimum`, `maximum`, `getIndex`

**Cost shape — state it correctly.** The descriptor holds per-container key/cardinality pairs
(`frozen.zig:54`) and **no prefix sums**, so `rank`/`getIndex`/`select` accumulate across preceding
containers: **O(containers + one container probe)**, not a search. This matches what `RoaringBitmap`
already does. It is a move *to* the existing standard, not past it.

**`minimum`/`maximum` are per-container-type, not constant-time.** Array and run: direct read of the
first/last element. **Bitset: scans up to 1,024 words** for the first/last set bit. They stop scanning
*values*. **Do not write "O(1)".**

**Safety.** `validateContainerBounds` / `checkedContainerSize` run once inside `init`; the image is
immutable and validated thereafter, which is why queries are infallible (`contains` → `bool`,
`cardinality` → `u64`). New methods therefore:

- use the **already-validated** offsets and the existing little-endian read helpers,
- **do not pointer-cast serialized bytes**,
- **do not** invent fresh bounds arithmetic alongside the validated accessors,
- **stay infallible** — re-checking per query would force error-union returns and change the public API.

## 2. Delegate from `Frozen64Bitmap` (`src/frozen64.zig`)

- **All five methods delegate** — `rank`, `select`, `getIndex`, **and `minimum`/`maximum`**. The latter
  two are **not** covered by deleting the three helpers: they are iterator-based **inline** in
  `frozen64.zig:61-79`, where `maximum` walks the entire final bucket keeping the last value seen. They
  must be rewritten to call `FrozenBitmap`'s new methods too, or the structural gate fails on exactly
  the worst offender.
- **Delete** production `frozen32Rank`, `frozen32Select`, `frozen32GetIndex`.
- The bucket-level loops stay — already cardinality-driven and correct. Only within-bucket work changes.
- Observable behaviour unchanged.

## 3. Differential correctness — the substantial part

Assert new `FrozenBitmap.rank/select/minimum/maximum/getIndex` equal **`RoaringBitmap`'s** for the same
value set **and** equal the **iterator-based linear results**.

**The linear implementations survive as test-only oracles.** They are deleted from production per §2 but
must be retained in test code — slow and obviously correct is exactly what an oracle should be. Deleting
them outright discards the only independent check.

**Failures must be case-labelled** — which bitmap, which container type, which operation, which input.
`expected X, found Y` alone is close to unusable across a corpus.

Cases:

- empty bitmap; single container of each of the three types
- values below all elements, above all elements, and absent from a populated container
- `select(0)`, `select(cardinality - 1)`, `select(cardinality)` → null
- **a range crossing `65535 → 65536`**, producing **two** run containers
- **a single container holding multiple disjoint runs**

## 4. Extend the existing `Frozen64Bitmap` round-trip test

`src/roaring64.zig:2203` (`"Roaring64Bitmap frozen64 round-trip read-only operations"`) already exercises
every delegated method, so it is the natural home for the deletion's coverage — but it currently probes
`rank`/`getIndex`/`select` **only at values that are present**, by iterating `toArrayAlloc`. Absence is
tested for `contains` alone.

Extend it to cover — **noting that `select` takes a rank, not a value**, so absent-value and
absent-bucket probing does not apply to it:

**Value-indexed — `rank` and `getIndex` only:**

- **Absent values**, not just `contains`.
- **Probes around high-32-bit bucket boundaries** — `(1 << 32) - 1`, `1 << 32`, `(2 << 32)`.
- **A value in an absent bucket.** The fixture populates buckets 0, 1, and 3; **bucket 2 is a hole**, so
  probing it exercises the `hi > target_hi` early-return branch that the present-values loop never
  reaches.

**Rank-indexed — `select`:**

- Ranks **immediately before, at, and immediately after each cumulative bucket boundary** — the
  equivalent stress, since what varies for `select` is which bucket the running count lands in.
- `select(cardinality)` → null.

## 5. Guard — extend `check-32`

Instantiate **both** `FrozenBitmap` and `Frozen64Bitmap` in `tools/check_32_api.zig` and call their
**full query surfaces** — not merely the five new `FrozenBitmap` methods, or the new `Frozen64Bitmap`
**delegation paths** stay outside the guard.

**The two take different byte sources — they are not interchangeable:**

- `FrozenBitmap.init` consumes **portable serialized** bytes (`serialize` / `serializedSizeInBytes`).
- `Frozen64Bitmap.view` consumes bytes from **`frozenSerialize`** (sized by `frozenSizeInBytes`) — rawr's
  own frozen64 image, **not** the portable Roaring64 serialization.

Feeding portable Roaring64 bytes to `view` is the easy mistake here, and since the probe is compile-only
it would never surface at runtime.

Per spec 40-01, the probe's enumerated surface **is** the guard boundary and omissions fail **silently**;
that finding cost four real defects. This is not optional tidying.

## Acceptance

- Five methods public, infallible, container-aware on `FrozenBitmap`, per §1's safety rules.
- Production `frozen32Rank` / `frozen32Select` / `frozen32GetIndex` deleted; `Frozen64Bitmap` delegates;
  behaviour unchanged. Iterator-based equivalents retained **test-only** as oracles.
- Differential passes against `RoaringBitmap` **and** the linear oracle, over all three container types
  and every §3 case, with **case-labelled** failures.
- **Negative control:** seed an off-by-one in the rank accumulation; confirm the differential **fails and
  names the case**. Record the output. A differential that passes a seeded defect is not evidence.
- §4 extensions present and passing, including the absent-bucket probe.
- **`check-32` passes with both frozen types and their full query surfaces in the probe.**
- **Structural gate (replaces a timing gate):** production `rank`/`select`/`minimum`/`maximum`/`getIndex`
  on **both** frozen types construct **no `Iterator`** and perform **no value-at-a-time traversal**.
  Verify by reading the implementations; record the confirmation. A before/after timing may be recorded
  as evidence but is **not** a gate.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `ReleaseSafe` and
  `ReleaseFast`.
- **No mutable-bitmap or benchmark path changed; no canonical board run required.** If any such path is
  touched, that exemption is void: pin the host, use the five-process protocol, state a tolerance.

## Estimate

**S/M** — implementations are small and modelled on existing code; the differential and negative-control
coverage are the bulk.
