<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 25-00: M4 cluster — per-row decomposition + attribution

First chunk of [M4 bitset-kernel codegen cluster](25-m4-bitset-codegen.md). **Diagnosis only,
benchmark-only.** Deliverable: each of the six M4-slow rows attributed to a **component** and a
cause (or a named residual), and a statement of **whether any subset shares one cause** — the
input that decides `25-01`'s lever, or a NO-GO.

## The six rows and their paths

removeRange (2.313x M4), dense AND (1.946x), flip (1.771x), lazyOr-construction (1.739x), orMany
(1.257x), dense OR (1.247x) — all rawr *at/ahead* on Zen 4. Paths differ (attribute dynamically):
eager AND/OR → `simdBitsetOp` (inline `@popCount`); lazy accumulation → `simdBitsetOpLazy` (no
popcount); sparse construction → scalar `setList`; orMany → lazy + `repairAfterLazy`; flip →
`toggleRange`, removeRange → `clearRange` (mask-based: clone + mask + op + recompute + maybe
demote).

## Full-operation decomposition (no nested timers)

Split each row's M4 time across **allocation/init, clone/copy, top-level traversal, word kernel,
cardinality/repair, representation conversion**. These ops are hundreds of ns, so:

- **Counters** (allocations, clones, words touched, conversions) collected **untimed**.
- Each phase measured via an **isolated fresh-process benchmark variant** or a **matched full-row
  A/B variant** (e.g. inline-popcount on/off; clone on/off) — **never a nested timer** inside the
  canonical operation.
- Compare each phase to the **corresponding CRoaring phase where accessible**; otherwise report a
  **rawr A/B attribution + a named residual** (an A/B delta is not proof of the CRoaring gap).

## Kernel/codegen hypotheses (test, not assume)

1. **Inline `@popCount` — eager dense AND/OR only:** `simdBitsetOp` with vs without the inline
   popcount, both hosts (wide-u64 `@popCount` has no native aarch64 instruction).
2. **`VEC_SIZE = 8` width on NEON:** sweep VEC_SIZE (2/4/8) for the `simdBitsetOp` /
   `simdBitsetOpLazy` / `countWords` loops, both hosts.
3. **Mask-based range ops:** attribute flip/removeRange M4 cost to word kernel vs clone vs mask
   build vs cardinality recompute.
4. **Codegen inspection:** aarch64 (M4) vs x86 (Zen 4) disassembly of the implicated kernels, and
   vs CRoaring's aarch64 bitset paths (`_nocard` / explicit NEON where rawr is not?). Record build
   command, symbol, asm.

## Measurement

Current canonical parity harness, both hosts, 3w/21t median, ≥5 fresh processes, per-path process
isolation; absolute medians + ranges; ns-level where applicable.

## Acceptance

- For each of the six rows: **supported attribution where evidence exists** (component + cause),
  **every remaining residual quantified and named** (no speculative cause to fill the checklist),
  and **enough attribution to choose a `25-01` fix or an explicit NO-GO**.
- Whether any subset shares a single cause is **stated, not assumed**, on both hosts.
- Codegen inspection (command / symbol / asm) recorded in `docs/parity-measurement.md`.
- **No production behavior or public API change during diagnosis.** Because the kernels under
  test (`simdBitsetOp` / `simdBitsetOpLazy` / `countWords`) are private production source,
  internal refactoring — or exposing them to the benchmark and gating the A/B variants
  (inline-popcount on/off, `VEC_SIZE` sweep) — **is allowed provided the default production path
  and public API stay unchanged**. (Prefer exercising the real kernels; if a benchmark-local
  replica is used instead, disassembly must prove it matches production codegen.)
- **Validation:** `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast` green;
  and **re-run the six canonical rows on both hosts** after diagnostics to confirm the changes
  did not alter their normal shape.

## Result to record (feeds `25-01`)

The dominant component + cause per row and the shared-cause verdict — decides whether `25-01`
does a card/no-card split, a per-arch `VEC_SIZE`, a range-op component fix, or is a documented
NO-GO.
