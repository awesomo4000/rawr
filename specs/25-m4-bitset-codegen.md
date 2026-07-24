<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 25: M4 bitset-kernel codegen cluster — diagnosis-first

Six default-SMP rows are slow on **M4 only** while rawr is at/ahead on Zen 4 — the classic
signature of an **aarch64/NEON codegen** issue, not an algorithm gap:

| op | M4 | Zen 4 |
|---|---:|---:|
| removeRange wide (dense) | 2.313x | 1.08x |
| bitwiseAnd (dense) | 1.946x | 0.56x |
| flip wide range (dense) | 1.771x | 0.56x |
| lazyOr construction (sparse) | 1.739x | 0.32x |
| orMany (32 mixed) | 1.257x | 0.91x |
| bitwiseOr (dense) | 1.247x | 0.46x |

**Diagnosis-first, no preselected cause** — but the goal is high-leverage: find whether these
share **one** NEON codegen root cause, because a single fix could move all six rows. (Per the
running lesson each row still gets a like-for-like check; the cross-arch signature — rawr
*ahead* on Zen 4 — argues these are real M4 gaps, not benchmark artifacts.)

## The shared pathway (confirmed)

All the bitwise-dense/lazy/n-way rows funnel through **`simdBitsetOp`**
(`src/bitset_container.zig:213`): a `@Vector(VEC_SIZE, u64)` word loop with **`VEC_SIZE = 8`**
(512-bit) that **also computes cardinality inline** — `card_vec += @popCount(result)` on every
iteration. `bitwiseAnd/Or` (dense), `orMany`, and `lazyOr construction` all use it. `flip`/
`removeRange` (wide) use the `setRange`/`clearRange` word-range fill loops
(`:97`, `:130`). So there are two candidate kernels, and the AND/OR/lazy/n-way cluster shares
the first.

## Phase 1 — Diagnosis (benchmark-only; both hosts)

Hypotheses to **test, not assume** (in rough suspicion order):

1. **Inline vector `@popCount` (cardinality-per-op).** rawr computes cardinality on *every*
   bitset op; CRoaring has `_nocard` bitset variants and skips it where not needed. `@popCount`
   on a wide u64 `@Vector` has **no native aarch64 instruction** (NEON `cnt` is byte-wise +
   reductions), so it may lower far worse on M4 than on x86. **Test:** measure `simdBitsetOp`
   with vs without the inline popcount, on both hosts — if the M4 gap largely closes without it,
   this is the cause.
2. **`VEC_SIZE = 8` (512-bit) pessimal on NEON.** 512-bit splits into **4×128-bit** NEON ops per
   iteration; a width tuned for x86 AVX2 may be a poor aarch64 choice. **Test:** sweep VEC_SIZE
   (2/4/8) on both hosts and compare.
3. **`setRange`/`clearRange` word-range fills** (flip/removeRange) not vectorized on aarch64 —
   the middle-word `for` fills should become NEON stores/memset; confirm they do.
4. **Codegen inspection.** Disassemble `simdBitsetOp` and the range fills on **aarch64 (M4)** vs
   **x86 (Zen 4)**, and compare against CRoaring's aarch64 bitset kernels (does CRoaring use
   explicit NEON intrinsics or `_nocard` paths where rawr does not?). Record exact build command,
   symbol, and the relevant asm.

Attribute the M4 gap per op to (1)/(2)/(3)/other, and report **whether the AND/OR/lazy/n-way
rows share a single cause**. Measure on the canonical spec-22 harness, both hosts, 3w/21t median,
≥5 fresh processes, per-path process isolation; absolute medians + ranges with a named residual.

Phase 1 stands alone: "do these six share a NEON codegen cause, and which kernel/construct is
it" is the deliverable.

## Phase 2 — Fix (conditional; lever follows the attribution)

- **If inline popcount dominates:** split rawr's bitset ops into **card / no-card variants**
  (compute cardinality only when the caller needs it, matching CRoaring), or defer cardinality —
  correctness of the `-1` cached-cardinality invariant preserved. One change, multiple rows.
- **If width dominates:** a **per-arch `VEC_SIZE`** (comptime), or restructure so aarch64 lowers
  cleanly.
- **If the range fills dominate:** vectorize `setRange`/`clearRange` (or lower to memset) on
  aarch64.
- Rows that don't share the cause fall back to their own attribution.

## Constraints

- **Must not regress Zen 4** (where rawr is at/ahead on these rows) — a per-arch fix or a change
  that's neutral-to-better on x86; verify on both hosts.
- **Correctness:** bitset op results **and** cardinality stay correct — differential green,
  including cardinality after every op, and the cache-invalidation invariant. A card/no-card
  split must not leave a stale cached cardinality.
- Benchmark-only for Phase 1 (measurement + a popcount-elided diagnostic variant); Phase 2 is a
  production kernel change, differential-covered.

## Acceptance

- **Phase 1 GO:** each of the six rows' M4 gap attributed (popcount / width / range-fill / other),
  and whether the AND/OR/lazy/n-way subset shares one cause, on both hosts, with the codegen
  inspection recorded in `docs/parity-measurement.md`.
- **Phase 2 GO (if attempted):** the affected rows reach **≤ 1.10x on M4** with **no regression
  on Zen 4** (and no other canonical row worsening >5% vs the committed spec-22 baseline, rerun on
  range overlap), differential green including cardinality.
- Validation: `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` on both
  hosts; `ReleaseSafe` / `ReleaseFast` green.

## NO-GO

- The rows do **not** share a cause and each is a small independent M4 quirk with no clean fix →
  record the attribution and stop; the ops remain correct and are at/ahead on x86.

## Estimate

S–M for Phase 1 (a popcount-elided diagnostic + a VEC_SIZE sweep + aarch64/x86 codegen
inspection on the existing harness). Phase 2 depends on the cause — a card/no-card split is M
(touches the bitset op surface + callers), a per-arch width is S.
