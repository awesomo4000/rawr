<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 30: Fused copy-with-range-removed (`removeRangeCopy`)

Close the biggest M4 row on the board: **removeRange (wide, dense) 1.932x** (rawr *ahead* on
Zen 4 — 0.411x — a hard no-regress gate). **Parity is a hard requirement:** the row closes only
at **≤ 1.10x**; anything above stays open for a further lever.

**This is not a range-algorithm fix — the algorithm already wins.** 26a exonerated the mutation
body (rawr 49.8 ns vs CRoaring 78.5 ns); the 1.932x is the **copy+remove workflow**: the canonical
row clones the input to get a fresh copy, then removes. The naive clone dutifully copies **every**
container — including the ones the wide range is about to delete — then frees them. The lever is a
**fused operation that produces the modified copy directly**, never allocating the doomed
containers.

## Why this is the right lever family

The removeRange, `clone`, and dense-AND M4 gaps share a root: `std.heap.smp_allocator` is slower
than libc `malloc` for many small per-container allocations on M4 (arch-specific — Zen 4 rawr
wins). The allocator-replacement track is closed (spec 18) and capacity re-tuning is NO-GO
(spec 27, exact-capacity clone *regressed* M4 SMP). The one family the evidence still favors is
**reducing allocation demand** (fewer clones) — and the wide copy+remove workflow is where demand
is most wasteful.

Critically, **CRoaring pays the same waste**: `roaring_bitmap_copy` copies all 8 containers, then
`roaring_bitmap_remove_range_closed` frees the 6 covered ones. So rawr constructing **2 result
containers instead of cloning 8** is how "fewer allocations" **offsets** rawr's slower-per-
allocation SMP cost — the mechanism by which this can reach parity, not just narrow the gap.

## Pinned corpus (assert before any timing)

**Canonical row** (`bench_parity_worker.zig`, `remove_range`): source = `addRange(0, 499999)` →
**8 run containers** (keys 0–7); operation removes `[100000, 650000]`.

> Pinned from the **canonical parity worker**, not the broad `bench_croaring.zig` dashboard (which
> uses `0..999999` → 16 containers). Confirm any number here against the canonical runner.

Partition:

- **survive untouched (1):** key 0 (`present [0, 65535]`, entirely below the removal);
- **boundary → partial diff (1):** key 1 (`present [65536, 131071]`, keep `[65536, 99999]`);
- **fully removed → deleted (6):** keys 2, 3, 4, 5, 6, 7;
- **result: 2 containers** (key 0 survivor + key 1 boundary; the boundary result is a single-run
  container).

The `30-00` diagnostic must assert this exact inventory (a drift invalidates the attribution).
Fusion **constructs 2 result containers** where the naive clone builds **8** — the survivor cloned,
the boundary built, the 6 doomed containers never allocated.

## Allocation accounting (container instances ≠ allocator calls)

Container instances and allocator calls are **different units** — a run-container clone allocates
its **struct and run payload separately**, and top-level key/container arrays plus growth add
further calls (the 8-container clone was measured at **20 allocator calls**, not 8). The diagnostic
must therefore report, per side, **separately**:

- **container constructions / clones** (instances),
- **actual allocator calls**,
- **frees during construction** (before result teardown),
- **requested bytes**,
- **result-teardown frees**.

**"Zero frees" applies only during fused construction** — the fused op allocates nothing for the
doomed containers, so it frees nothing mid-build. **Destroying the returned result still frees its
2 owned containers** (the same teardown the baseline pays). Do not claim zero frees for the whole
workflow.

## The fused operation

Add a **new owned-result path** (proposed `removeRangeCopy(self: *const Self, allocator, lo, hi)
!Self`) that produces a modified copy with the range removed, **preserving the source**:

1. Determine each container's disposition from a cheap key-range scan.
2. Per container:
   - **fully outside `[lo, hi]`** → `clone` the container into the result (untouched survivor);
   - **fully covered** → **skip** (allocate nothing);
   - **boundary** → build the **difference container directly** into the result.
3. **Never mutate `self`.**
4. **Ownership / cleanup:** result is independently owned and independently mutable, holds **no**
   source containers; on any mid-loop allocation failure the partially built result is fully
   deinited (set `result.size` before returning the error so `errdefer` sees the partial
   containers — the spec-27/`3e27675` clone-leak discipline).

Top-level result capacity is a **separate variable**, measured in `30-00` (below) — **not** baked
into this path. The existing in-place `removeRange` stays unchanged (a separate, exonerated
primitive still used by mutate-in-place callers). This op is **additive**.

## `30-00` diagnostic cells — separate fusion from pre-sizing

Spec 27 showed exact pre-sizing can **regress M4 SMP**. If the fused path is measured only at exact
capacity, a pre-sizing regression could **conceal** a successful clone-elimination result.
Therefore measure three cells (both hosts, SMP, canonical protocol):

| cell | doomed-container skip | top-level capacity |
|---|---|---|
| baseline | — (clone + removeRange) | current clone default |
| fused-default | ✓ | normal top-level growth |
| fused-presized | ✓ | exact / upper-bound reserve |

**Ship whichever fused shape wins independently** — fusion and pre-sizing are decided on their own
numbers, per host.

## Timing boundary (pin for both sides)

The canonical gated number includes **creation of the modified copy + range removal + result
destruction**; source construction is **outside** timing. Preserve that boundary on both sides:

- **rawr:** `removeRangeCopy` **+ result deinit**;
- **CRoaring:** `roaring_bitmap_copy` + `roaring_bitmap_remove_range_closed` **+ result free**,
  **copy-on-write disabled**.

An optional construction/teardown split is useful diagnostically, but **the gated number must
include canonical teardown** (no moving teardown out of timing on either side).

## Measurement legitimacy (mandatory — the crux)

Both sides produce a modified copy while **preserving the source**, so the comparison is
apples-to-apples. **Forbidden:** moving rawr's clone/teardown outside the timed region while
CRoaring's copy/free stays inside — that is a measurement artifact, not an optimization.

## Row rename (on adoption)

Once adopted, the manifest must name the rawr operation **`removeRangeCopy`** ("copy with range
removed"). Leaving it named `removeRange` would imply the **mutating primitive** got faster — but
that primitive is unchanged and already faster than CRoaring (26a). The current
`rawr_operation = "RoaringBitmap.clone plus removeRange"` label updates to the fused op.

## Correctness (pin explicitly; byte-identity + differential + failure injection)

`removeRangeCopy(self)` must serialize **byte-identical** to `clone(self)`-then-`removeRange`, and
match CRoaring set-parity, across at least:

- **`lo > hi`** → returns an independent clone (no removal);
- **empty source**;
- **range entirely before or after** all set bits (no-op copy);
- **full-source removal** → **zero-capacity result**, then **`add`** into it (growth from zero);
- **`0` and `maxInt(u32)` boundaries**;
- **single-container** and **exact chunk-boundary** ranges;
- **different source and result allocators**;
- **cached and unknown (`-1`) cardinality** states on the source containers;
- a boundary diff producing the **same container type** the in-place path would.

Ownership/source invariants (assert on success **and every injected failure**):

- **source serialization unchanged** — source untouched;
- **result owns no source containers** and remains **independently mutable**;
- on OOM the result is valid or cleanly errored, **no leak**.

## Constraints / gates

- **Zen 4 no-regress (hard):** rawr is ahead (0.411x); the change stays within noise (≤ 5%, rerun
  on overlap) — it should only help.
- **Spec-27 M4 SMP gate:** the allocation reduction and any pre-sizing are **measured on M4 SMP,
  per the canonical protocol, before shipping** — not assumed from the count; fusion and pre-sizing
  ship independently.
- **Board gate + tightened layout exception (spec 28):** no canonical row worsens > 5% vs a fresh
  pre-change baseline, both hosts; layout classification requires **both** stable focused timing
  *and* instruction-identical disassembly.

## Scope

- **Closes the removeRange workflow row only.** It does **not** touch the standalone `clone (dense)`
  1.764x row — a full copy has no doomed containers to skip — nor dense-AND. Those remain
  documented allocator residuals.

## Acceptance

- **Phase 1 GO:** corpus inventory asserted (8 source → 2 result; survive 1 / boundary 1 /
  deleted 6); the five allocation-accounting figures reported per side per fused cell; the three
  cells timed on M4/Zen 4 SMP with the pinned boundary; byte-identity + differential +
  failure-injection green; no canonical row changed.
- **Phase 2 GO — hard:** the canonical **removeRange (wide) row reaches ≤ 1.10x on M4 SMP** (via
  legitimate copy-vs-copy, winning fused shape only), **Zen 4 not regressed**, board gate held
  (layout exception), row renamed to `removeRangeCopy`. **Anything above 1.10x is a partial result
  and the row stays open** — a residual would be the shared M4 SMP per-container-clone cost, which
  reopens with the next lever.
- `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` both hosts;
  `ReleaseSafe` / `ReleaseFast` green; `docs/parity-measurement.md` updated.

## Proposed chunk plan (confirm at review)

- **`30-00`** — implement `removeRangeCopy` with full correctness (byte-identity, differential,
  allocation-failure injection, source-preservation); assert the corpus + five-figure allocation
  accounting; time the three cells (baseline / fused-default / fused-presized) on M4/Zen 4 SMP with
  the pinned boundary in a named diagnostic; **no canonical row changed**.
- **`30-01`** — adopt the winning fused shape into the canonical parity row (legitimate
  copy-vs-copy), rename the row to `removeRangeCopy`, full board gate, ship on M4/Zen 4 SMP
  numbers; hard ≤ 1.10x acceptance (row stays open above it).

## Estimate

M for `30-00` (new path + full correctness surface + failure injection + three-cell measurement).
S–M for `30-01` (row wiring + rename + board gate).
