<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 29: Dense set-op result construction (AND / OR) factorial

Close the two biggest real M4 gaps: **bitwiseAnd dense 1.911x**, **bitwiseOr dense 1.167x**
(rawr *ahead* on Zen 4 — 0.56x / 0.46x — so Zen 4 is a hard no-regress gate). Parity is a hard
requirement; this is a factorial investigation of dense-result construction, **not** a single
lever. The full-run kernel skip alone cannot explain 1.9x — only one run per matching pair on
this corpus — so the structural suspects are the scratch-and-clone sequence and grow-from-4
top-level storage.

## Verified structural differences

Both dense ops route through `twoWayAllocatingMerge` (`bitwiseAnd`/`bitwiseOr` at
`src/bitmap.zig:921`/`:927`):

- **Top-level result starts at `INITIAL_CAPACITY = 4` and grows** (`bitmap.zig:42`); CRoaring
  **pre-sizes** — `min(a.size, b.size)` for AND, `a.size + b.size` for OR. Dense **OR** therefore
  undergoes **multiple growth cycles** that CRoaring does not.
- Dense **AND** intersects each pair in a **scratch `FixedBufferAllocator`** (8448 B) then
  **clones** each non-empty result into the real allocator (`:1385`, `:1433`) — every kept
  container is built twice.
- `runIntersectRun` / `runUnionRun` (`container_ops.zig:742` / `:608`) **lack the full-run
  identity fast paths** CRoaring has (full-run ∩ X = X; full-run ∪ X = full-run).
- Both implementations must still allocate independently-owned result containers — a full-run
  *identity* still produces an owned clone of the identity operand.

## Phase 1 — Factorial diagnosis (SMP, both hosts, no preselected cause)

Levers, measured **independently and in combination**, for AND and OR separately (they may
differ — OR's multi-growth vs AND's scratch-clone):

- **A — full-run kernel identity branches** in `runIntersectRun`/`runUnionRun` (allocator-
  independent, both-host, pure CRoaring catch-up).
- **B — bitmap-level full-run identity**: when a matched container is a full run, **directly
  clone/keep the identity operand** (AND → the other operand; OR → the full run), bypassing
  scratch and the kernel entirely for that pair.
- **C — pre-sized top-level result storage** (`min(a,b)` AND / `a+b` OR), removing the
  grow-from-4 cycles.
- **D — combinations** (A+B, A+C, B+C, A+B+C).

**Measurement discipline (the whole campaign's lessons):**
- **Construction and teardown measured separately** (26a matched-boundary; teardown is
  subtraction-derived, **diagnostic not gated**; no nested timers). **Allocation counts + bytes**
  per cell — scratch, clone, top-level, container payloads separately.
- **SMP path is the target:** four+ rawr-SMP cells per op, fresh process each, on M4 and Zen 4,
  vs **one CRoaring reference per host**. rawr-libc is a **conditional control** only (run if an
  SMP cell's cost is ambiguous between allocation and other work) — not routine, not gated.
- Interactions treated as interactions, not additive percentages.

Output: which lever(s) carry the M4 gap for AND and for OR, with construction/teardown/allocation
attribution.

## Phase 2 — Fix (conditional, per lever, on its own numbers)

Ship the winning combination; each lever ships only on its own measured numbers.

- **A / B (full-run identities)** are allocator-independent — **no spec-27 or layout risk**; the
  safest levers.
- **C (pre-sizing) MUST be measured, not assumed.** Spec 27 showed **exact-capacity pre-sizing
  regressed clone on M4 SMP** despite fewer allocations. But the shape differs — dense **OR**
  does *multiple* growth cycles where clone did one, so pre-sizing may help OR even though it hurt
  clone. Test it independently for AND and OR; drop it on the host(s) where it regresses.
- **Bypassing scratch-and-clone** (via B for full-run pairs, and/or constructing non-empty AND
  results directly in the real allocator) is the structural lever for AND — but "construct
  directly in the real allocator" is itself an allocation-shape change subject to the same
  spec-27 measurement, so it is a cell, not an assumption.

## Constraints / gates

- **Representation-identical output** (as spec 26): results must be **byte-identical** (via
  `serialize`) to the current implementation's output **and** CRoaring set-parity — a full-run
  identity clone must produce the *same* container type the merge would. Differential across
  container-type mixes, full/partial runs, empty results, and chunk-boundary cases stays green.
- **Error semantics — build-then-commit, leak-free.** Exhaustive allocation-failure injection on
  the changed paths; on OOM the result is valid or cleanly errored, inputs untouched, no leak.
- **Zen 4 no-regress (hard):** rawr is ahead on both dense ops on Zen 4; the shipped change stays
  within noise (≤ 5%, rerun on overlap) there.
- **Board gate with the layout-noise floor (spec 28):** no canonical row worsens > 5% vs a fresh
  pre-change baseline, both hosts — but a moved *untouched* row where **CRoaring also moved** and
  disassembly is instruction-identical is **layout, not a regression** (document, don't chase).

## Acceptance

- **Phase 1 GO:** the M4 dense-AND and dense-OR gaps attributed across the factorial
  (lever × construction/teardown/allocation), per host, on the SMP path.
- **Phase 2 GO:** **bitwiseAnd dense and bitwiseOr dense reach ≤ ~1.10x on M4 SMP** — or a
  statistically supported improvement with rationale where a component is intrinsic — with **Zen 4
  not regressed**, byte-identical output, differential + failure-injection green, and the board
  gate held (layout-floor tolerance). A documented partial (some levers shipped, others NO-GO on
  M4 SMP) is acceptable *only with the residual attributed* — parity is a hard requirement, so a
  material residual reopens with the next lever, it does not close the row.
- `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` both hosts;
  `ReleaseSafe` / `ReleaseFast` green; `docs/parity-measurement.md` updated.

## Proposed chunk plan (confirm at review)

- **`29-00`** — factorial diagnostic (levers A/B/C + combinations, construction/teardown/alloc,
  SMP both hosts) → the attribution.
- **`29-01`** — full-run identity fast paths (A/B), the allocator-independent levers, if the
  attribution implicates them; byte-identity + failure injection.
- **`29-02`** — pre-sizing / scratch-bypass (C and the AND direct-construction), gated on `29-00`
  and each measured against the spec-27 SMP trap independently for AND and OR.

## Estimate

M for `29-00` (factorial across two ops, two hosts, construction/teardown split). `29-01` S–M
(kernel identity branches). `29-02` M (allocation-shape changes with the spec-27 measurement gate
per op).
