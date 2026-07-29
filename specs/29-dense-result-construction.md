<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 29: Dense set-op result construction (AND / OR)

Close the two biggest real M4 gaps: **bitwiseAnd dense 1.911x**, **bitwiseOr dense 1.167x**
(rawr *ahead* on Zen 4 — 0.56x / 0.46x — so Zen 4 is a hard no-regress gate). **Parity is a hard
requirement:** the row closes only at **≤ 1.10x**; anything above that is a partial result and the
row stays open until a further lever closes it. This is a multi-lever investigation of
dense-result construction — the full-run kernel skip alone cannot explain 1.9x (one run per
matching pair on this corpus), so the structural suspects are the scratch-and-clone sequence and
grow-from-4 top-level storage.

## Verified structural differences

Both dense ops route through `twoWayAllocatingMerge` (`bitwiseAnd`/`bitwiseOr` at
`src/bitmap.zig:921`/`:927`):

- **Top-level result starts at `INITIAL_CAPACITY = 4` and grows** (`bitmap.zig:42`); CRoaring
  **pre-sizes** — `min(a.size, b.size)` for AND, `a.size + b.size` for OR. Dense **OR** undergoes
  **multiple growth cycles** CRoaring does not.
- Dense **AND** intersects each matched pair in a **scratch `FixedBufferAllocator`** then
  **clones** each non-empty result into the real allocator (the scratch-then-clone sequence in
  `twoWayAllocatingMerge`, currently ~`bitmap.zig:1850` and ~`:1930`) — every kept container is
  built twice.
- `runIntersectRun` / `runUnionRun` (`container_ops.zig:742` / `:608`) **lack the full-run
  identity fast paths** CRoaring has (full-run ∩ X = X; full-run ∪ X = full-run).
- Both implementations still allocate independently-owned result containers — a full-run identity
  still produces an **owned** container: AND clones the other operand's container; **OR clones the
  full-run operand or constructs a new canonical full-run container** (it cannot *keep* an input
  container — the result is independently owned).

## Pinned corpus inventory (assert before any timing)

The `29-00` diagnostic must **assert this structure holds** before timing (a corpus drift would
invalidate the attribution):

- **left operand: 8 run containers; right operand: 9 run containers**;
- **5 matched keys**, all **run/run** pairs, **each with at least one full run**;
- **AND output: 5 containers; OR output: 12 containers**;
- the full-run **identity branch fires exactly 5 times** (both AND and OR).

## Phase 1 — Diagnostic cells (SMP, both hosts, no preselected cause)

Levers are **not** an orthogonal 2×2 — **B subsumes A** (B bypasses `runIntersectRun`/
`runUnionRun` entirely, so A+B ≡ B on this corpus). The cells are therefore:

| cell | full-run kernel branch (A) | bitmap-level full-run identity (B) | pre-sized top-level (C) |
|---|---|---|---|
| baseline | — | — | — |
| A | ✓ | — | — |
| B | — | ✓ | — |
| C | — | — | ✓ |
| A+C | ✓ | — | ✓ |
| B+C | — | ✓ | ✓ |

- **A — full-run kernel identity branches** in `runIntersectRun`/`runUnionRun`: an early
  identity return inside the kernel. **Does not change allocation shape** (still one owned result
  per pair) — but is **not** immune to code-layout effects.
- **B — bitmap-level full-run identity**: on a matched full-run pair, skip scratch/kernel and
  directly produce the owned identity result (AND → clone the other operand's container; OR →
  clone the full-run / construct a canonical full-run). **Changes allocation/execution shape for
  AND** (bypasses the scratch-then-clone) — so B carries the spec-27 measurement obligation and is
  layout-affected.
- **C — pre-sized top-level storage** (formulas below), removing the grow-from-4 cycles.
- Measured **for AND and OR separately** (their dominant lever likely differs — OR's multi-growth
  vs AND's scratch-clone).

**Additional AND lever, explicitly deferred:** "construct non-empty results directly in the real
allocator" (scratch-bypass for *non-full* runs) is **not** in this matrix — on this corpus B
already bypasses scratch for every matched pair, so it is unnecessary. If a later, non-full-run
corpus is added, it becomes lever **D** then; not now.

**Measurement discipline:** construction and teardown measured **separately** (26a matched-
boundary; teardown subtraction-derived, **diagnostic not gated**; no nested timers); allocation
counts + bytes per cell (scratch / clone / top-level / payloads separately). Four+ rawr-SMP cells
per op, fresh process each, M4 and Zen 4, vs **one CRoaring reference per host**; rawr-libc a
**conditional control** only. **`29-00` must not change the production default** — benchmark-local
variants or gated internal helpers only; the shipping path changes in later chunks after
attribution.

## Phase 2 — Fix (conditional, per lever, on its own numbers)

- **A** is allocation-shape-neutral but layout-affected — ships on its measured M4/Zen 4 numbers.
- **B and C both change allocation shape** — **neither is assumed; both carry the spec-27
  measurement obligation** (spec 27 showed exact-capacity pre-sizing *regressed* clone on M4 SMP
  despite fewer allocations). C's shape differs from clone's (dense **OR** does *multiple* growth
  cycles), so it may help OR even though it hurt clone — **test independently for AND and OR, drop
  on any host where it regresses.**
- Ship the winning combination; each lever ships only on its own SMP numbers.

### Safe capacity formulas (lever C)

- **AND:** `min(self.size, other.size)`.
- **OR:** `min(self.size + other.size, 65536)` (clamp — the key space is 16-bit).
- Tests must include **empty and disjoint results** (min-capacity / zero-overlap) and
  **mutation after a zero-capacity result** (adding to a bitmap whose op produced 0 containers).

## Constraints / gates

- **Representation-identical output** (spec 26): results **byte-identical** (via `serialize`) to
  the current implementation **and** CRoaring set-parity — a full-run identity must produce the
  *same container type* the merge would. Differential across container-type mixes, full/partial
  runs, empty/disjoint results, and chunk-boundary cases stays green.
- **Error semantics — build-then-commit, leak-free.** Exhaustive allocation-failure injection on
  the changed paths; on OOM the result is valid or cleanly errored, inputs untouched, no leak.
- **Zen 4 no-regress (hard):** rawr is ahead on both dense ops on Zen 4; the shipped change stays
  within noise (≤ 5%, rerun on overlap).
- **Board gate + tightened layout exception (spec 28).** No canonical row worsens > 5% vs a fresh
  pre-change baseline, both hosts. An untouched row's movement is classified as **layout (not a
  regression) only when BOTH:** its **focused before/after timing is stable** *and* its relevant
  **disassembly is instruction-identical**. Otherwise it is a **failed regression gate** — CRoaring
  moving too is *not* sufficient on its own.

## Acceptance

- **Phase 1 GO:** corpus inventory asserted; the M4 dense-AND and dense-OR gaps attributed across
  the cells (lever × construction/teardown/allocation), per host, on the SMP path; no production
  default changed.
- **Phase 2 GO — hard:** **bitwiseAnd dense and bitwiseOr dense reach ≤ 1.10x on M4 SMP**, with
  **Zen 4 not regressed**, byte-identical output, differential + failure-injection green, board
  gate held (tightened layout exception). **Anything above 1.10x is a partial result and the row
  stays open** — parity is hard-required; an attributed-but-material residual **reopens with the
  next lever, it does not close the row.**
- `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` both hosts;
  `ReleaseSafe` / `ReleaseFast` green; `docs/parity-measurement.md` updated.

## Proposed chunk plan (confirm at review)

- **`29-00`** — the six-cell diagnostic (A/B/C/A+C/B+C vs baseline, construction/teardown/alloc,
  SMP both hosts), corpus inventory asserted, no production default changed → the attribution.
- **`29-01`** — lever **A** (full-run kernel identity branches) if implicated; byte-identity +
  failure injection.
- **`29-02`** — levers **B and/or C** (bitmap-level full-run identity, pre-sizing), each gated on
  `29-00` and measured against the spec-27 SMP trap independently for AND and OR.

## Estimate

M for `29-00` (six cells across two ops, two hosts, construction/teardown split). `29-01` S–M.
`29-02` M (allocation-shape changes with the spec-27 gate per op).
