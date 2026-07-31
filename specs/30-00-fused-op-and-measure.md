<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 30-00: `removeRangeCopy` — implement, verify, measure

Build the fused copy-with-range-removed path and its correctness surface, then measure the three
cells on both hosts. This chunk **adds the public `removeRangeCopy` library operation**; what
`30-01` changes is the **canonical parity-row** (manifest), not the library API. **No canonical
parity row changes in this chunk** — the diagnostic is a named benchmark-local artifact.

Toplevel: [30-fused-remove-range-copy.md](30-fused-remove-range-copy.md).

## The operation

Add `removeRangeCopy(self: *const Self, allocator, lo, hi) !Self` — produces a modified copy with
`[lo, hi]` removed, **preserving the source**:

1. Determine each container's disposition from a cheap key-range scan.
2. Per container: **fully outside `[lo, hi]`** → `clone` into the result; **fully covered** →
   **skip** (allocate nothing); **boundary** → build the **difference container directly** into the
   result.
3. **Never mutate `self`.**
4. **Ownership / cleanup:** result is independently owned and independently mutable, holds **no**
   source containers; on any mid-loop allocation failure the partial result is fully deinited (set
   `result.size` before returning the error so `errdefer` sees partial containers — spec-27/
   `3e27675` clone-leak discipline).

The in-place `removeRange` primitive is **unchanged**; this op is additive (**one** public
operation).

**API exposure — exactly one public API (measure two policies without two APIs):**

- **Public `removeRangeCopy`** initially calls the **normal-growth** implementation.
- A **shared internal helper** takes the **capacity policy** parameter (normal-growth vs
  exact-presize).
- The **repository-only diagnostic module** invokes the helper with **both** policies to produce the
  fused-default and fused-presized cells.
- No `removeRangeCopyPresized` public API — `30-01` selects the production policy **inside**
  `removeRangeCopy`.

## Top-level capacity is a variable (two fused shapes)

- **fused-default** — normal top-level growth (default clone capacity policy).
- **fused-presized** — reserve the **exact final container count**: `survivors + (boundary results
  whose surviving cardinality > 0)`. A boundary container can come out **empty** (every present
  value in its chunk lies inside a partial-chunk removal); exclude those from the count via a
  **pre-scan using range cardinality** (or an equivalent exact method). Do **not** assume every
  boundary survives. On full-source removal this collapses to a **zero-capacity** result.

Both shapes are built and measured here; neither is chosen until `30-01`.

## Corpus assertion (assert before any timing)

Canonical `remove_range` corpus (from `bench_parity_worker.zig`, **not** the broad
`bench_croaring.zig`): source = `addRange(0, 499999)` → **8 run containers** (keys 0–7); remove
`[100000, 650000]`. Assert the partition:

- **survive untouched (1):** key 0;
- **boundary → partial diff (1):** key 1 (keep `[65536, 99999]`, single-run result);
- **fully removed → deleted (6):** keys 2–7;
- **result: 2 containers.**

Fusion **constructs 2** result containers where the naive clone builds **8**.

## Allocation accounting (report; container instances ≠ allocator calls)

Report, **for each rawr cell** (baseline `clone`+`removeRange`, fused-default, fused-presized), five
figures kept distinct:

- container constructions / clones (instances),
- actual **allocator calls** (the 8-container clone was ~**20** calls, not 8),
- **frees during construction** (before result teardown),
- requested bytes,
- **result-teardown frees**.

**CRoaring: timing-only** unless allocator-call accounting via memory hooks is added — **decide
here and record which**; only claim a rawr-vs-CRoaring allocator-call comparison if actually
instrumented. "Zero frees during construction" is **corpus-specific** (this 8→2 shape; ordinary
growth can free top-level buffers on other shapes); result teardown still frees the 2 owned
containers.

## Measurement (three cells, both hosts, SMP, canonical protocol)

| cell | doomed-container skip | top-level capacity |
|---|---|---|
| baseline | — (clone + removeRange) | current clone default |
| fused-default | ✓ | normal top-level growth |
| fused-presized | ✓ | exact final container count (non-empty boundaries only) |

- **Both hosts (M4, Zen 4), SMP**, canonical protocol (3 warmup / 21 timed, five process medians +
  full range, `batch_count = 8192`), vs **one CRoaring reference per host**.
- **Timing boundary (pinned, both sides):** the gated number includes **copy + range removal +
  result teardown**; source construction is **outside** timing. rawr = `removeRangeCopy` + result
  deinit; CRoaring = `roaring_bitmap_copy` + `roaring_bitmap_remove_range_closed` + result free,
  **copy-on-write disabled**. An optional construction/teardown split is diagnostic-only; the gated
  number includes canonical teardown on both sides.
- Named, committed diagnostic artifact; **no canonical manifest row touched.**

## Correctness (byte-identity + differential + failure injection)

`removeRangeCopy(self)` serializes **byte-identical** to `clone(self)`-then-`removeRange`, and
matches CRoaring set-parity, across at least:

- **`lo > hi`** → independent clone (no removal);
- **empty source**;
- **range entirely before or after** all set bits (no-op copy);
- **full-source removal** → an **empty result usable by `add`** (adding grows correctly); the
  *zero top-level capacity* assertion applies **only to fused-presized**, not fused-default;
- **`0` and `maxInt(u32)` boundaries**;
- **single-container** and **exact chunk-boundary** ranges;
- **boundary container that empties** (surviving cardinality 0 — must not be allocated/counted);
- **different source and result allocators**;
- a boundary diff producing the **same container type** the in-place path would.

**Cardinality / cache parity:**

- **Bitmap-level:** a known `cached_cardinality` is adjusted **exactly as clone-then-remove**; an
  unknown source cardinality **stays unknown**.
- **Per-container:** a cloned survivor keeps its source's cached/unknown (`-1`) state; a boundary
  diff sets the state the in-place path would. Assert **both** cached and unknown source states.

**Ownership / source invariants (assert on success and every injected failure):**

- **source serialization unchanged** — source untouched;
- **result owns no source containers**, remains independently mutable;
- on OOM the result is valid or cleanly errored, **no leak**.

## Acceptance (Phase 1 GO)

- Corpus + partition asserted (8 source → 2 result: survive 1 / boundary 1 / deleted 6).
- Five allocation figures reported for each rawr cell; CRoaring method (timing-only vs hooked)
  recorded.
- Three cells timed on M4/Zen 4 SMP with the pinned boundary; named diagnostic committed.
- Byte-identity + differential + failure-injection green; source-preservation proven on success and
  every injected failure.
- **No canonical manifest row changed.**
- `zig build test`; `zig build difftest` green; **`ReleaseSafe` and `ReleaseFast` both green**
  (this chunk introduces the public operation and its OOM-cleanup path — validate both build modes
  before committing Phase 1, not deferred to `30-01`); diagnostic section of
  `docs/parity-measurement.md` updated.
