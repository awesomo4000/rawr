<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 35: Headerless transient lazy bitsets (E3)

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md) (Wave 3 — the
finale). Close the **last material M4 gap**: **lazyOr construction 1.663x** (post-`d7d357b`
canonical board). **lazyOr + repair 1.178x is downstream of the same phase** — repair-alone is
**1.049x** (fine), so closing construction is expected to close the combined row too; both are
gated here. Once these close, **the full M4 board is at or under parity.**

**Parity is a hard requirement** — rows close at ≤ 1.10x; a partial is adopted by owner judgement
(spec-30 policy) and the row stays open.

## Post-Wave-1 baseline (pinned)

This spec measures against the fresh post-`d7d357b` board (Wave 1 shipped: compact Run header,
word-major orMany). M4: lazyOr construction **5.746 ms vs 3.456 ms = 1.663x**; lazyOr+repair
**14.612 vs 12.403 = 1.178x**; lazyOr repair-alone **8.315 vs 7.928 = 1.049x**. Zen 4 reference
numbers are captured fresh in `35-00` (the no-regress gate needs them). The **compact Array header
was NO-GO here** (spec 32 made lazy-OR *worse*) — that lever is closed; this spec attacks the
**transient bitset accumulators**, which spec 32 never touched.

## Verified structural facts (function names authoritative; mirror line numbers may lag `d7d357b`)

In `lazyMergeTwo` (`bitmap.zig`, ~`:2210` pre-Wave-1):

- For every **matched key**, `use_lazy_bitset = (op == .xor) or bitset_conversion or either side is
  a bitset`. The canonical row passes `bitset_conversion = true`, so **every matched pair takes the
  bitset path** regardless of cardinality.
- That path does **`BitsetContainer.init(allocator)`** = **2 allocator calls** (16 B header `create`
  + 8 KB aligned `words`) **plus an 8 KB `@memset` zero-fill**, then accumulates both sides into it.
- **Unmatched keys** are `cloneContainer`ed (the thousands-of-array-clones share; spec 17/18 closed
  the arena/allocator levers for those, and spec 32's Array-header NO-GO closed the header lever).
- `repairAfterLazy` then, per bitset: `computeCardinality()` (full 1024-word scan); **card ≤ 4096 →
  `bitsetToArray` + `bc.deinit` — the header AND the 8 KB words both die**; card 0 → both die;
  only card > 4096 survives.
- `BitsetContainer.words` is **already a separate allocation** from the header — so a survivor path
  that "allocate header late, adopt words" requires **no copy**; adoption is pointer assembly.

**Corpus shape (assert in `35-00`, don't trust this estimate):** the sparse corpus (500 k random
u32 across the full space; `a = [0..half]`, `b = [half/2..]`) gives roughly **~16 k matched keys**,
each holding tiny arrays (~7–15 values) — so essentially **every transient bitset demotes** at
repair. On this corpus the eliminated-vs-deferred split (umbrella) collapses to the good case:
**headers (and words) are born-to-die**, E3's exact target. `35-00` must pin the actual counts:
matched keys, transient bitsets created, demoted vs surviving, per-key cardinalities.

## Attribution first — including a like-for-like check on CRoaring (iterate's lesson)

Before building anything, `35-00` attributes the 1.663x across:

1. **Transient-bitset lifecycle** — header create, 8 KB words alloc, 8 KB zero-fill, accumulate,
   repair scan, demote copy, header free, words free — per matched key, summed.
2. **Unmatched clone traffic** — the known residual share (closed levers; measured for share, not
   as a target).
3. **Top-level assembly** — `initCapacity(min(a.size+b.size, 65536))` etc.

**Like-for-like verification (mandatory):** instrument/count what **CRoaring actually materializes
per matched key** under `roaring_bitmap_lazy_or(..., bitsetconversion=true)` on this same corpus —
per-key container types during its lazy construction. **If CRoaring thresholds tiny array/array
pairs** (does *not* build an 8 KB bitset for a ~15-value pair even with `bitsetconversion=true`),
then a large part of the 1.663x is a **semantics mismatch** — rawr doing strictly more work than the
comparison — the same shape as the iterate pull-vs-push phantom. That finding would go to the owner
as an explicit decision: is rawr's `bitset_conversion=true` = "always bitset" pinned **API
contract**, or is the lazy accumulator strategy an **internal detail** free to threshold like
CRoaring? **The spec does not pre-decide this**; it requires the count, and the owner decides on
the numbers. (The umbrella's "must produce bitsets" pin was written assuming CRoaring behaves the
same — if attribution falsifies that assumption, the pin is re-examined, not silently kept or
dropped.)

## The lever (L1) — never allocate a header that is going to die

For the transient accumulator on the lazy path:

- Allocate **only the aligned 8 KB words** (one allocator call instead of two; still zeroed —
  correctness requires a zeroed accumulator for OR).
- Track the transient state via the **`reserved = 0b11` tag** as a **transient lazy-bitset tag** —
  which is free in the enum but **NOT operationally free**: the `Container` union has no member for
  it and generic paths return false/zero, skip deallocation, or treat it as unreachable. The
  production chunk requires the **complete dispatch/lifecycle inventory** (umbrella): repeated lazy
  ops on an unrepaired bitmap, clone/move of an unrepaired bitmap, repair failure, deinit-before-
  repair, serialization attempts, and generic queries (contains / cardinality / rank / select /
  iterate) — each with defined, tested behavior, no default fall-through.
- **Repair:** compute cardinality directly from the words; **demote → free the words, no header was
  ever allocated** (the eliminated case); **survive → allocate the 16 B header and adopt the words**
  (deferred case — no copy, words are already a separate allocation).

## Eliminated vs deferred (the load-bearing accounting — umbrella, verbatim obligation)

The diagnostic reports all five, and **the gate is the COMBINED construction+repair row**:

1. headers **permanently eliminated** (demotion),
2. headers **deferred** to repair (survivors),
3. **construction-only** allocation reduction,
4. **full construction+repair** allocation reduction,
5. **repair regression** from allocating surviving headers there.

On the canonical sparse corpus, `35-00`'s pinned counts are expected to show ~all-eliminated /
~zero-deferred — but the accounting must hold for bitset-heavy corpora too (a dense control where
survivors dominate), so the deferred path's repair cost is measured, not assumed benign.

## Numeric stop-gate (before touching the container union)

Benchmark-only prototype first. Pin the bar: **permanently-eliminated header calls × measured
per-call SMP cost, plus the removed header frees, must project the combined construction+repair row
to ≤ 1.10x** — or show a required focused-time improvement that does. If the header-call arithmetic
cannot get there (e.g. the 8 KB zero-fill + repair scan dominate, not the 16 B create), **stop
before changing the container union** and report what *does* dominate — that attribution then
drives whatever follows (possibly the owner decision from the like-for-like check above).

## Measurement discipline

- Canonical protocol: **3 warmup / 21 timed, five process medians + full range**, fresh-process,
  **M4 and Zen 4**, one CRoaring reference per host; E3-owned diagnostic module (shared
  `build.zig`/runner/docs edits are implementer-owned).
- **Accounting per cell:** allocations, frees, requested bytes, effective SMP-class bytes,
  teardown — container instances ≠ allocator calls.
- **Construction and repair measured separately AND combined** (the gate is combined; the split is
  the attribution).
- **Both-flag correctness before performance** (spec-32 discipline): the transient-tag build passes
  `zig build test`, `zig build difftest`, `ReleaseSafe`, `ReleaseFast` before numbers are accepted.
- **Zen 4 policy (spec 30):** within-noise passes (repeated focused timing + range overlap); a real
  regression fails by default, adoptable only via explicit owner exception.
- **Board gate + spec-28 layout exception** on production adoption; one architecture-neutral shape.

## Correctness (production chunk)

- **Output invariants:** post-repair result has the same container kinds / cardinalities / values
  as baseline lazyOr+repair, identical portable bytes where serialize is valid; CRoaring
  set-parity differential.
- **The full transient-tag lifecycle inventory** (above) tested explicitly — including
  deinit-before-repair (words freed, no header leak) and repair-failure mid-way (partially repaired
  bitmap remains deinit-able, no leak).
- **OOM / failure injection** on: words allocation, survivor header allocation at repair, demote
  array allocation — valid-or-cleanly-errored, inputs untouched, no leak (build-then-commit).
- `xorMany`/eager paths untouched; `bitset_conversion=false` path untouched.

## Acceptance

- **Phase 1 GO (35-00):** corpus counts pinned (matched keys, transient bitsets, demote/survive
  split); the three-way attribution reported; **CRoaring like-for-like materialization counts
  reported**; headerless prototype measured benchmark-only with the five-figure
  eliminated/deferred accounting, both hosts; stop-gate arithmetic explicit. No production change.
- **Phase 2 GO — hard (35-01):** **lazyOr construction AND lazyOr+repair reach ≤ 1.10x on M4 SMP**,
  Zen 4 within noise, full lifecycle inventory + invariants + failure injection green, board gate
  held. Partial adoption per spec-30 policy (owner judgement, row stays open). If the like-for-like
  check reveals a semantics mismatch instead, the owner decision is recorded and this spec's scope
  is re-cut accordingly — **not** silently absorbed.
- `zig build test`; `zig build difftest`; `ReleaseSafe`/`ReleaseFast`; canonical
  `run-compare-bench.sh` both hosts on adoption; `docs/parity-measurement.md` updated.

## Proposed chunk plan (confirm at review)

- **`35-00`** — attribution (lifecycle / clone-share / assembly) + **CRoaring like-for-like
  materialization count** + pinned corpus counts + benchmark-only headerless prototype with
  eliminated/deferred accounting + stop-gate arithmetic, both hosts. No production change.
- **`35-01`** — production transient-tag migration (conditional on `35-00` GO): complete dispatch/
  lifecycle inventory, invariants, failure injection, board gate, ship on both-host numbers.

## Estimate

M for `35-00` (attribution + instrumented CRoaring count + prototype, two hosts). M–L for `35-01`
(container-union change with the full lifecycle inventory) — if it runs.
