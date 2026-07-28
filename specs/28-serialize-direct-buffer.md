<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 28: Serialize — direct fixed-buffer writer

Close the persistent both-host `serialize` gap (~**1.14x**, and it is on *both* architectures,
not an M4-only residual). The structural difference is verified in source, and this is also the
**cheap probe** of the "drop temp arrays + write directly" pattern that dense AND/OR want next —
on a small, no-public-API surface, before we bet the bigger rewrites on it.

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

Attribute the ~1.14x among the candidates, on the canonical `serialize` corpus:

- **temporary-array allocation** (`desc_buf` + `offset_buf` alloc/free);
- **the copy** of each temp table into the output;
- **the `std.Io.Writer.fixed` abstraction** itself (per-`writeAll` bounds/state vs a raw indexed
  store) — Morty's open question, and a real suspect independent of the allocations.

Method per the campaign discipline: untimed allocation/byte counters; A/B variants
(direct-indexed-write vs Writer; temp-arrays vs in-place) measured in fresh processes;
**rawr-SMP and rawr-libc** both, plus CRoaring, on M4 and Zen 4. If the cost is the Writer
abstraction rather than the allocations, the fix is a direct indexed writer, not (only) removing
temp arrays — and that matters because removing allocations alone might replay spec 27's M4-SMP
regression.

Phase 1 stands alone: "which component is the 1.14x, per host" is the deliverable.

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

- **Byte-identical output — non-negotiable.** `serialize()` produces a wire format. The direct
  path's bytes must be **identical to the current output** for every corpus, and both **rawr
  `deserialize`** and **CRoaring** must read it (roundtrip + `roaring_bitmap_portable_deserialize`
  equality). This is the portable-byte contract; a differential across container-type mixes
  (array/bitset/run, run and no-run formats, the `NO_OFFSET_THRESHOLD` boundary, empty bitmap)
  stays green.
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

- **Phase 1 GO:** the 1.14x attributed to temp-array alloc / copy / Writer abstraction, per host,
  with counters + A/B, on rawr-SMP + rawr-libc + CRoaring.
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
