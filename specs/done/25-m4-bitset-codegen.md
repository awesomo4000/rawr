<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 25: M4 bitset-kernel codegen cluster — diagnosis-first

Six default-SMP rows are slow on **M4 only** while rawr is at/ahead on Zen 4 — a signature that
**points to aarch64/NEON codegen** as the leading hypothesis rather than an algorithm gap, but
the actual cause is what this spec determines:

| op | M4 | Zen 4 |
|---|---:|---:|
| removeRange wide (dense) | 2.313x | 1.08x |
| bitwiseAnd (dense) | 1.946x | 0.56x |
| flip wide range (dense) | 1.771x | 0.56x |
| lazyOr construction (sparse) | 1.739x | 0.32x |
| orMany (32 mixed) | 1.257x | 0.91x |
| bitwiseOr (dense) | 1.247x | 0.46x |

> **Outcome (2026-07-25) — NO-GO; hypothesis disproven, no production change.** The `25-00`
> decomposition falsified the shared-SIMD/codegen theory: the affected rows are dominated by
> **run containers, allocation-heavy sparse paths, and mixed-container accumulation** — not the
> bitset word kernels — and **width-8 SIMD was already optimal** on both hosts, so no speculative
> production change was made. The M4-only rows stand as **attributed, documented residuals**
> (correct, and at/ahead on x86). Diagnostic harness + full attribution recorded in
> [`docs/parity-measurement.md`](../../docs/parity-measurement.md). Exactly the outcome the
> staged design exists to produce cheaply.

**Diagnosis-first, no preselected cause** — but the goal is high-leverage: find whether these
share **one** NEON codegen root cause, because a single fix could move all six rows. (Per the
running lesson each row still gets a like-for-like check; the cross-arch signature — rawr
*ahead* on Zen 4 — argues these are real M4 gaps, not benchmark artifacts.)

## Candidate pathways (attribute per row — these are NOT one shared kernel)

The rows take **different** paths, so inline popcount alone cannot explain the cluster —
attribute each **dynamically** before grouping:

- **eager dense AND/OR** → `simdBitsetOp` (`src/bitset_container.zig:213`): a `@Vector(VEC_SIZE=8,
  u64)` word loop that **also computes cardinality inline** (`card_vec += @popCount(result)`).
- **lazy bitset accumulation** (in lazy construction / n-way) → `simdBitsetOpLazy` — **no inline
  popcount**.
- **sparse lazy construction** → may use **scalar `setList`** for array containers (not a bitset
  word loop at all).
- **`orMany`** → lazy accumulation **+ `repairAfterLazy`**, where cardinality is computed
  **separately** in repair.
- **flip** → `toggleRange`; **removeRange** → `clearRange` — both reached via **mask-based bitmap
  operations** (clone + mask + XOR/difference + recompute + maybe demote), **not** direct
  top-level range fills.

So the inline-popcount hypothesis applies **only to eager dense AND/OR**. Whether the six rows
share *any* single cause is a Phase-1 question, not an assumption.

## Phase 1 — Diagnosis (benchmark-only; both hosts)

### First: per-row full-operation decomposition (not just the word kernel)

Every one of the six rows **allocates and does work beyond the named kernel**, so a fast isolated
NEON kernel would not explain the canonical gap. For each row, dynamically attribute its path
(above) and split the M4 time across components:

- **allocation / init**, **clone / copy**, **top-level container traversal**, **word kernel**,
  **cardinality / repair**, **representation conversion** (demote/promote).

Concretely: `flip`/`removeRange` clone + build a mask + XOR/difference + recompute cardinality +
maybe demote; lazy construction adds top-level merge + many container allocations; `orMany` adds
cursor scanning + allocation + accumulation + repair.

**Method — no nested timers.** These ops are only hundreds of nanoseconds, so a clock call per
phase would materially distort them. Instead:
- **Counters** (allocations, clones, words touched, conversions) collected **untimed**.
- Each phase measured through an **isolated fresh-process benchmark variant** or a **matched
  full-row A/B variant** (e.g. with vs without the inline popcount, with vs without the clone) —
  **never a nested timer inside the canonical operation**.
- Compare each phase to the **corresponding CRoaring phase where accessible**; where it is not,
  report the result as a **rawr A/B attribution plus a named unexplained residual** — an A/B delta
  is **not** by itself proof of the CRoaring parity gap.

### Then: kernel/codegen hypotheses (test, not assume)

1. **Inline vector `@popCount` — eager dense AND/OR only.** `simdBitsetOp` computes cardinality
   per op via wide-u64 `@popCount`, which has **no native aarch64 instruction** (NEON `cnt` is
   byte-wise + reductions) and may lower far worse on M4. CRoaring has `_nocard` variants. **Test:**
   `simdBitsetOp` with vs without the inline popcount, both hosts. (Does **not** apply to the lazy /
   `setList` / repair paths.)
2. **`VEC_SIZE = 8` (512-bit) width on NEON** — the `simdBitsetOp` / `simdBitsetOpLazy` /
   `countWords` loops split 512-bit into **4×128-bit** NEON ops; a width tuned for x86 may be a
   poor aarch64 choice. **Test:** sweep VEC_SIZE (2/4/8) on both hosts.
3. **Mask-based range ops** — `flip`→`toggleRange`, `removeRange`→`clearRange` reached via
   clone+mask+op: attribute the M4 cost to the word kernel vs the clone vs the mask build vs the
   cardinality recompute (per the decomposition) before assuming a NEON kernel.
4. **Codegen inspection** — aarch64 (M4) vs x86 (Zen 4) disassembly of the implicated kernels, and
   vs CRoaring's aarch64 bitset paths (explicit NEON intrinsics / `_nocard` where rawr is not?).
   Record exact build command, symbol, and the relevant asm.

Attribute the M4 gap per row to a component + cause, and report **whether any subset shares a
single cause** (do not assume the six do). Measure on the current canonical parity harness, both hosts,
3w/21t median, ≥5 fresh processes, per-path process isolation; absolute medians + ranges with a
named residual.

Phase 1 stands alone: "do these six share a NEON codegen cause, and which kernel/construct is
it" is the deliverable.

## Phase 2 — Fix (conditional; lever follows the attribution)

- **If inline popcount dominates:** split rawr's bitset ops into **card / no-card variants**
  (compute cardinality only when the caller needs it, matching CRoaring), or defer cardinality —
  correctness of the `-1` cached-cardinality invariant preserved. One change, multiple rows.
- **If width dominates:** a **per-arch `VEC_SIZE`** (comptime), or restructure so aarch64 lowers
  cleanly.
- **If a mask-based range-op component dominates** (`toggleRange`/`clearRange` word kernel, or the
  clone/mask/recompute around it): fix the implicated component — vectorize the aarch64 word loop,
  avoid the clone, or defer the cardinality recompute.
- Rows that don't share the cause fall back to their own component attribution.

## Constraints

- **Must not regress Zen 4** (where rawr is at/ahead on these rows) — a per-arch fix or a change
  that's neutral-to-better on x86; verify on both hosts.
- **Correctness:** bitset op results **and** cardinality stay correct — differential green,
  including cardinality after every op, and the cache-invalidation invariant. A card/no-card
  split must not leave a stale cached cardinality.
- **Phase 1 is diagnosis with no production behavior / public API change.** The kernels under
  test (`simdBitsetOp` / `simdBitsetOpLazy` / `countWords`) are private production source, so
  **behavior-neutral internal refactoring** — or exposing them to the benchmark and gating the
  A/B variants (popcount on/off, `VEC_SIZE` sweep) — is allowed **provided the default production
  path and public API stay unchanged** (else a benchmark-local replica with disassembly proving
  it matches production codegen). Phase 2 is a production kernel change, differential-covered.

## Acceptance

- **Phase 1 GO:** for each of the six rows — **supported attribution where the evidence exists**
  (component: alloc / clone / traversal / word kernel / cardinality-repair / conversion, + cause),
  **every remaining residual quantified and named** (do **not** invent a speculative cause to fill
  the checklist), and **enough attribution to choose a production fix or an explicit NO-GO**;
  whether any subset shares a single cause stated (not assumed), on both hosts, codegen inspection
  recorded in `docs/parity-measurement.md`.
- **Phase 2 GO (if attempted):** **rows addressed by the selected fix** reach **≤ 1.10x on M4** —
  or retain a **statistically supported improvement with rationale** — with **no regression on
  Zen 4** (and no other canonical row worsening >5% vs the **latest committed corrected parity
  baseline**, rerun on range overlap), differential green including cardinality. **Remaining rows**
  not addressed by the fix may close as **documented no-clean-fix residuals**.
- Validation: `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` on both
  hosts; `ReleaseSafe` / `ReleaseFast` green.

## NO-GO

- The rows do **not** share a cause and each is a small independent M4 quirk with no clean fix →
  record the attribution and stop; the ops remain correct and are at/ahead on x86.

## Estimate

M for Phase 1 (six per-row decompositions across two architectures + assembly inspection; a popcount-elided diagnostic + a VEC_SIZE sweep + aarch64/x86 codegen
inspection on the existing harness). Phase 2 depends on the cause — a card/no-card split is M
(touches the bitset op surface + callers), a per-arch width is S.
