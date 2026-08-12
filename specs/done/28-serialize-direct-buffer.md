<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 28: Serialize — direct fixed-buffer writer

Close the persistent `serialize` gap — **M4 SMP 1.14x**, **Zen 4 SMP 1.08x**: primarily an M4
gap now, with Zen 4 an important **regression gate** (not a second target). The structural
difference is verified in source, and this is also the **cheap probe** of the "drop temp arrays
+ write directly" pattern that dense AND/OR want next — on a small, no-public-API surface, before
we bet the bigger rewrites on it.

> **Outcome (2026-07-29, commit `2ba714a`) — GO; target met, and the lever proved safe.**
> `serialize()` now writes directly into its fixed buffer (no temp descriptor/offset tables);
> `serializeToWriter()` unchanged for generic writers; the four diagnostic cells call production
> so the winning cell can't drift. **M4 SMP 1.128 → 1.068 ms (5.3% faster, 1.05x — meets the
> ≤1.10x gate); Zen 4 SMP 1.035 → 0.824 ms (20.4% faster, 0.81x — rawr ahead).** Byte-identity
> (legacy oracle + round-trip + CRoaring) + threshold-boundary coverage green. **Dual payoff:
> the drop-temp-arrays + direct-write lever is proven *safe* on M4 SMP here — no spec-27-style
> regression — de-risking that pattern for future work.**
>
> **Caveat (measurement, not a regression):** the strict all-row 5% board gate was **not clean**
> — adding the diagnostic/direct code shifted **whole-binary layout**, moving *untouched* rawr
> **and** CRoaring rows; M4 `rankMany` slowed with **instruction-identical disassembly**.
> Documented, not misrepresented as a serializer regression. Implication: sub-~1.2x M4 ratios now
> carry a **layout-noise floor** — `rankMany`'s 1.194x is contaminated by this, not a real gap.

**Diagnosis first — because spec 27 is the hard prior.** Removing allocations *regressed* the
M4-SMP clone body (20→18 allocs, ~50% slower). So we do **not** assume the temp arrays are the
cost or that removing them wins on M4 SMP; we measure which component it is, then fix that.

## What is verified in source

`serialize()` (`src/serialize.zig:150`) allocates the exact output buffer, wraps it in
`std.Io.Writer.fixed`, and calls `serializeToWriter`. `serializeToWriter` (`:160`) then:

- allocates a **temporary descriptor array** `desc_buf` = `size × 2` u16 (`:194`), fills it
  (key + cardinality−1 per container), and **copies it into the output** via `writer.writeAll`;
- allocates a **temporary offset array** `offset_buf` = `size` u32 (`:220`), fills it with
  absolute offsets, and copies it in.

So two heap allocations + two copies on top of the output allocation. The reference writes both
tables **directly into the destination buffer**. Because `serialize()` owns a fixed buffer and
knows the exact layout (cookie + optional run-bitset + descriptor + offsets + container data), it
can compute each table's byte position and write in place — no `desc_buf`/`offset_buf`.

## Phase 1 — Diagnosis (benchmark-only, both hosts, no preselected cause)

Attribute the M4 1.14x among: **temporary-array allocation** (`desc_buf` + `offset_buf`
alloc/free); **the copy** of each temp table into the output; and **the `std.Io.Writer.fixed`
abstraction** itself (per-`writeAll` bounds/state vs a raw indexed store — an open question,
a real suspect independent of the allocations).

**Measure the SMP path — libc is a conditional control, not a routine column.** The target is
rawr's default **SMP** path and the gap is M4 SMP, so Phase 1 measures the **four rawr-SMP cells
on M4 and Zen 4** against **one CRoaring reference per host**. rawr-**libc** is *not* a routine
variant here: for serialize it isn't even whole-op libc — only the output buffer would use libc
while `desc_buf`/`offset_buf` stay on `bm.allocator` (SMP) (`serialize.zig:194`; row wiring
`bench_croaring.zig:730`) — so it answers nothing cleanly. Run **one M4 rawr-libc control only
if** an SMP cell's cost is ambiguous between allocation and other work; it is **not part of the
acceptance gate or routine reporting**. The construction lever is read directly from the
**allocation counts + bytes** (output buffer vs temp tables, reported separately per cell), no
libc timing run needed.

**A/B — a true `construction × output` factorial (temp/direct × Writer/direct-index),
interactions treated as interactions (not additive %):**

1. **temp tables + Writer** — the current implementation;
2. **direct construction + Writer** — table entries **streamed through the Writer** as they are
   computed, **no temp tables** (this cell naturally makes more `Writer` calls — that is inherent
   to the cell, note it);
3. **temp tables + direct index** — temp tables still allocated, copied into the buffer via
   **direct indexed stores** (no Writer);
4. **direct construction + direct index** — in-place table entries **and** direct indexing (the
   candidate shipping shape).

Untimed allocation/byte counters; each of the four rawr-SMP cells a fresh process, plus one
CRoaring reference per host; M4 and Zen 4; no nested timers. The 2×2 cleanly separates the
**construction** lever (temp allocation — carries the spec-27 M4-SMP risk) from the **output**
lever (Writer abstraction — allocator-independent) and their interaction.

Phase 1 stands alone: "which cell wins, per host, and why" is the deliverable.

## Phase 2 — Fix (conditional on the attribution)

Add a **direct fixed-buffer path for `serialize()`** implementing the components Phase 1 named:

- Write the descriptor and offset tables **directly into the output buffer** at their computed
  positions (indexed stores, no temp arrays) if temp-array alloc/copy dominates;
- Bypass the `Writer` abstraction for the fixed-buffer case (raw indexed stores) if that
  dominates;
- **`serializeToWriter` stays exactly as-is** for generic writers — they cannot be indexed, so
  the temp-array path remains their (correct) implementation. No public API change; the by-value
  `serialize()` internally routes to the direct path.

## Constraints / gates

- **Byte-identical output — non-negotiable, with an in-repo legacy oracle.** `serialize()`
  produces a wire format. The direct path's bytes must be **byte-identical to the unchanged
  `serializeToWriter()` output** (the in-repository legacy oracle — checked *before* CRoaring and
  round-trip), and then both **rawr `deserialize`** and **CRoaring** must read it (roundtrip + `roaring_bitmap_portable_deserialize`
  equality). This is the portable-byte contract; a differential across container-type mixes
  (array/bitset/run, run and no-run formats, the empty bitmap, and **run-format container counts
  immediately below and exactly at `NO_OFFSET_THRESHOLD`** — the branch that decides whether the
  offset table is written) stays green.
- **Cursor invariant:** the direct path asserts its **final write position equals `buf.len`** —
  a mismatch is a layout bug caught immediately, not a silently truncated/over-run buffer.
- **Spec-27 SMP gate.** Measure **M4 SMP** explicitly. If the direct path removes allocations but
  regresses M4-SMP serialize (the clone trap), that is a NO-GO for the allocation part — record
  it; the Writer-bypass part may still stand on its own numbers.
- **Both hosts, no regression.** serialize reaches **≤ 1.10x on both M4 and Zen 4** — or a
  material improvement (> 5% with range support) — with no canonical row worsening > 5% vs a
  fresh pre-change baseline run from the latest committed head immediately before the after-run,
  both hosts.
- **Error semantics unchanged:** `serialize()` returns the owned buffer or frees it on error
  (`errdefer`); the direct path allocates only the output buffer.

## Acceptance

- **Phase 1 GO:** the M4 1.14x attributed to the construction lever / output lever / interaction,
  per host, from the four rawr-SMP cells vs the CRoaring reference (libc control only if an SMP
  cell needs allocator attribution).
- **Phase 2 GO (if attempted):** serialize ≤ 1.10x (or material improvement) on both hosts,
  byte-identical output, roundtrip + CRoaring differential green, M4-SMP not regressed, board
  gate held. A documented partial (Writer-bypass shipped, alloc-removal NO-GO on M4 SMP, or vice
  versa) is acceptable — each sub-change ships on its own numbers.
- `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` both hosts;
  `ReleaseSafe` / `ReleaseFast` green.

## NO-GO

- Phase 1 shows the gap is not in a movable component (e.g. it is the shared
  `getCardinality()`-per-container work both sides do, or measurement variance) → document and
  stop.

## Estimate

S for Phase 1 (counters + A/B on the existing harness). Phase 2 is S–M — a direct-indexed
fixed-buffer path plus the byte-identity differential; `serializeToWriter` untouched.
