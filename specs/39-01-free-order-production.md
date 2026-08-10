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

## API surface — PINNED

| decision | |
|---|---|
| **ADD** | `repairAfterLazyWithOptions(...)` — the only new public entry point *(under (B), the default path changes instead)* |
| **UNCHANGED** | `repairAfterLazy()` |
| **EXCLUDED** | `deinit`, `clearRetainingCapacity`, `Roaring64Bitmap`, `OwnedBitmap` |

`OwnedBitmap` is excluded on principle — **its arena owns teardown**, so container-level free order is
irrelevant. The other three are excluded to keep the first shipped surface minimal; they may be revisited
on their own evidence, not this spec's.

**Document what the caller asserts:** *"my allocator benefits from descending free order"* — **not** *"I am
on M4."* The effect is **SMP-specific, and Zen 4 gains more in absolute terms** (−3.577 vs −2.086 ms in
`38-00`).

**libc-on-M4 is excluded from the DEFAULT path**, not made impossible — opting in remains available and
that is documented. With detection rejected (opaque `Allocator` vtable; comparing against
`smp_allocator.vtable` breaks for every wrapped allocator), this is a **contract, not a mechanism**.

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
  reported against the **canonical-equivalent opt-in variant** with **no injected noise**.
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

- Scope position **recorded**; API shipped accordingly; reporting rule honoured (variant row under (A)).
- Selected rung implemented; portability asserted; scratch from `self.allocator` with pre-mutation
  fallback.
- **Partial-repair invariant implemented and tested** — documented as new behaviour.
- Full-cycle win demonstrated on the no-noise canonical-equivalent variant; peak RSS reported; board gate
  held; Zen 4 within noise or explicitly excepted.
- Correctness surface green.
- `docs/parity-measurement.md` updated; the pathology doc cross-referenced.

## Estimate

M — the reorder itself is small. The work is the deferred-free restructure, the **new** partial-repair
invariant with positional failure injection, the opt-in surface, and (under (B)) the prepass.
