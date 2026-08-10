<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 39-01: Descending demote frees — production

Toplevel: [39-descending-free-order.md](39-descending-free-order.md). Gated on:
[39-00](39-00-free-order-measurement.md).

**BLOCKED until two things happen:** (1) `39-00` reports a full-cycle win, and (2) the **scope decision**
below is made. Everything else here is already pinned.

## Precondition — the scope decision (owner call)

**An opt-in cannot close a default-path row.** Pick one; the rest of this chunk follows from it.

| | (A) optional opt-in variant *(recommended)* | (B) default adoption |
|---|---|---|
| **API** | `repairAfterLazyWithOptions(...)` added; `repairAfterLazy()` unchanged | default path changes |
| **Runtime gate** | **none** — activation entirely caller-controlled | **demotion prepass REQUIRED** before mutation, so the mechanism can decline cheaply for small inputs |
| **Board** | reported as a **separate variant row** (precedent: `bitwiseAnd (sparse, arena)`) | canonical row may be claimed |
| **Requires** | nothing further | **`39-00` must show libc is unharmed on the repair-demote path** |

**Option 2 (unconditional deferral, gating only the reordering) is REJECTED** either way — it imposes the
deferral cost and raised memory peak on every call regardless of benefit, and makes `D-key` a **silent
floor** for below-gate callers.

**Reporting rule, non-negotiable:** while the API is opt-in, results are a **variant row, never the
canonical row.**

## Change

**Deferred descending free in `repairAfterLazy`, option (a):** keep the conversion pass in **key order**
(preserving result allocation order and key order exactly as today) while **collecting old bitset
pointers**; then run a **separate descending free pass** using the rung `39-00` selected.

- Reorder mechanism: **the cheapest rung `39-00` found sufficient** — no re-litigating the ladder here.
- Scratch: **`self.size` pointers, upper bound, no prepass** (unless (B), which adds the prepass),
  **allocated from `self.allocator`**; failure **falls back to interleaved before any mutation**.
- Portability: `@bitSizeOf(usize) − @clz(span)`, **never hardcoded 64**; `span == 0` / `n <= 1` early
  return; **must compile on 32-bit targets** even though gates run on M4 and Zen 4.

## API, libc policy, and benchmark gate — ALL CONDITIONAL ON A/B

These three follow from the scope decision and **must not be stated unconditionally.** An earlier draft
described A's API and libc policy as if they held under B; they do not.

| axis | under **(A) opt-in variant** | under **(B) default** |
|---|---|---|
| **new API** | **ADD `repairAfterLazyWithOptions(...)`** — the only new public entry point | no new entry point required; behaviour moves into the existing one |
| **`repairAfterLazy()`** | **UNCHANGED** | **CHANGES** — it becomes the deferred descending path (gated by the demotion prepass) |
| **runtime gate** | **none** — caller-controlled | **demotion prepass**, so the mechanism can decline cheaply below threshold |
| **benchmark gate** | a **separate opt-in variant row** (precedent: `bitwiseAnd (sparse, arena)`); the canonical row is **not** claimed | **the canonical row itself is gated** — no variant row; the canonical number is the result |
| **libc position** | **excluded from the default path** by not opting in; opting in remains possible and documented — a **contract, not a mechanism** | **libc is necessarily ON the default path.** Admissible **only after `39-00` proves libc unharmed on the repair-demote path**; if it regresses, (B) is unavailable |
| **caller assertion** | documented: *"my allocator benefits from descending free order"* — **not** *"I am on M4"* | n/a — no caller assertion exists |

**Common to both:** `deinit`, `clearRetainingCapacity`, `Roaring64Bitmap`, `OwnedBitmap` are **EXCLUDED**.
`OwnedBitmap` on principle — **its arena owns teardown**, so container-level free order is irrelevant; the
other three to keep the first shipped surface minimal, revisitable on their own evidence.

**Common to both:** allocator **detection stays rejected** (opaque `Allocator` vtable; comparing against
`smp_allocator.vtable` breaks for every wrapped allocator). Under (A) that is why the opt-in is
caller-declared; under (B) it is why the mechanism must be **safe for all allocators** rather than
conditionally enabled.

**Reminder on framing:** the effect is **SMP-specific, and Zen 4 gains MORE in absolute terms**
(−3.577 vs −2.086 ms in `38-00`) — so neither position should be described as an M4 fix.

## NEW GUARANTEE this chunk introduces — partial-repair invariant

**This does not exist today.** `repairAfterLazy` commits `self.size = @intCast(write_idx)` **only after the
complete loop**, so a mid-loop allocation failure currently leaves compacted/overwritten entries with the
**old** `size`. Spec 35 designed the in-place partial-commit invariant but **35-01 never shipped**.

`39-01` **introduces** it, and its scope explicitly includes **tail compaction and final state commit**:

- Repaired **prefix** `[0, write_idx)` stands.
- **Untouched tail compacted behind it**; those entries keep their existing valid containers.
- **`self.size` committed**; **cardinality left unknown**; **no dangling entries**.
- Every collected bitset freed **exactly once** via `errdefer` over the scratch list — the scratch is the
  **sole owner** of bitsets no longer reachable through `self.containers`.
- **A failed repair may be retried** (remaining transients are still in the tail).
- **Do not describe this as preserving existing behaviour.**

## Gates

- **Full-cycle** (construction + repair + result teardown) improvement on steady-state `lazyOr+repair`,
  with **no injected noise** — reported against the **canonical-equivalent opt-in variant** under **(A)**,
  or against **the canonical row itself** under **(B)**.
- **Peak RSS reported** — deferral raises the temporary live peak (~134 MB held simultaneously).
- Size gate per the scope decision: **(A) none / (B) prepass**, threshold from `39-00`'s crossover
  candidates, chosen deliberately (the M4/Zen spread was 16× at rung 4).
- **Board gate + spec-28 layout exception**, both hosts. **Zen 4 policy (spec 30):** within-noise passes; a
  real regression needs an explicit owner exception.
- **One architecture-neutral shape.**

## Correctness

- Repair results **byte-identical** to today — container kinds, cardinalities, values, key order.
- Every container freed **exactly once**; shadow-bitmap verified; no leak, no double-free.
- Reorder is a **permutation**.
- **Failure injection:** scratch allocation (falls back pre-mutation), and **first / middle / last**
  collected-bitset positions; plus the new partial-repair invariant verified at each, including
  **retry-after-failed-repair**.
- `zig build test`, `zig build difftest`, `ReleaseSafe`, `ReleaseFast`; canonical
  `run-compare-bench.sh` both hosts.

## Acceptance

- Scope position **recorded**, and **API / libc policy / benchmark gate all shipped per that position's
  row of the conditional table** — not a blend. Under (A): new options entry point, `repairAfterLazy()`
  untouched, variant-row reporting, libc excluded by non-opt-in. Under (B): `repairAfterLazy()` changes,
  prepass gate, canonical row claimed, **libc proven unharmed** on the repair-demote path.
- Selected rung implemented; portability asserted; scratch from `self.allocator` with pre-mutation
  fallback.
- **Partial-repair invariant implemented and tested** — documented as new behaviour.
- Full-cycle win demonstrated with no injected noise — on the **canonical-equivalent opt-in variant under
  (A)**, or on **the canonical row itself under (B)**; peak RSS reported; board gate held; Zen 4 within
  noise or explicitly excepted.
- Correctness surface green.
- `docs/parity-measurement.md` updated; the pathology doc cross-referenced.

## Estimate

M — the reorder itself is small. The work is the deferred-free restructure, the **new** partial-repair
invariant with positional failure injection, the opt-in surface, and (under (B)) the prepass.
