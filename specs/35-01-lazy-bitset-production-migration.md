<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 35-01: `.lazy_bitset` production migration

Toplevel: [35-headerless-transient-lazy-bitsets.md](35-headerless-transient-lazy-bitsets.md) (E3).
Ship the headerless transient accumulator in production. **Gated on `35-00`'s stop-gate projecting
the combined construction+repair row to ≤ 1.10x.** This is the risky chunk: it changes the core
container discrimination.

## 1. Tag rename + payload (the enforcement mechanism)

```zig
// ContainerType
lazy_bitset = 0b11,                  // was: reserved
// Container union
lazy_bitset: *align(64) [1024]u64,   // was: reserved: void  (fromTagged discarded the pointer)
```

**Rename, do not reuse `.reserved`** — the rename makes every unhandled switch arm a **compile
error**, so dangerous existing arms (`.reserved => {}` / `=> false` / `=> null`) cannot silently
survive. This converts the dispatch inventory below into a **compiler-checked invariant**.

## 2. Lever

On the lazy path, allocate **only the aligned 8 KB words** (zeroed — required for OR **and** XOR);
no header. **Repair:** compute cardinality from the words; **demote → free words, no header ever
allocated**; **survive → allocate the 16 B header and adopt the words** (no copy — words are already
a separate allocation).

## 3. Activation scope (all must be covered)

Transient containers arise exactly where `use_lazy_bitset` is true
(`op == .xor or bitset_conversion or either side is a bitset`):

| entry point | takes transient path when |
|---|---|
| `lazyOr(…, true)` | always — the target row |
| `lazyOr(…, false)` | when either matched side is already a bitset |
| `lazyOrInPlace(…)` | as `lazyOr` |
| `lazyXor(…)` | always (`lazyMergeTwo(.xor, …, true)`) |
| `lazyXorInPlace(…)` | as `lazyXor` |

XOR-specific behavior (`lazyToggle` / `lazyXorWith`, the `card == 0` drop rule) is covered
explicitly, not just the sparse OR row.

## 4. Pinned behavior contract

A transient behaves as an **unknown-cardinality bitset** for read-only queries (`contains`,
`getCardinality` from words, `rank`, `select`, `minimum`/`maximum`, iteration, `toArray`,
equality/subset), and:

- **clone** → a **normal owned unknown-cardinality bitset** (allocate header, copy words,
  cardinality `-1`); never aliases, never propagates the transient representation.
- **repeated lazy operations** → a transient can be accumulated into again without materializing a
  header.
- **serialization → `error.UnrepairedLazyResult` at EVERY writing entry point**: `serialize`,
  `serializeToWriter`, `OwnedBitmap.serialize`, plus any other container-walking writer
  (frozen / `roaring64`). All already return error unions.
- **`serializedSizeInBytes` → keep the `usize` signature**; compute the true (post-repair-equivalent)
  size by scanning the transient's words. **No panic, no `!usize` API break.**
- **`validate` → add `UnrepairedLazyResult` to `ValidateError` and return it on ANY `.lazy_bitset`**;
  normal validation must **pass after `repairAfterLazy`**. That pairing machine-checks "no transient
  survives repair." Transient pointer/alignment checks live in an **internal helper** — **no
  repaired-state field is added to any bitmap.**
- **eager set ops and the many-ops never PRODUCE transients, but their INPUT dispatch is updated**
  per this consume/reject policy — `orMany`, `orManyHeap`, `xorMany`, `orManyOwned`, `xorManyOwned`
  all take `[]const *const Self` and **can receive an unrepaired bitmap**. Each arm is a real
  unknown-cardinality-bitset implementation or an explicit documented rejection; **pinned per op**.
- **deinit** → frees the words; no header to free.

## 5. Dispatch inventory (compiler-driven)

Work the **compile errors** from the rename to completion — the mirror shows **~98 tag-switch sites
across 17 files**, production being `bitmap.zig`, `container.zig`, `container_ops.zig`, `compare.zig`,
`optimize.zig`, `serialize.zig`, `roaring64.zig`. Explicitly include `validate`, conversion /
optimization (`runOptimize` / `optimize`), range and set operations, `toArray`, equality/subset,
minimum/maximum, and clearing. **No default fall-through; no `unreachable` on any path a transient
can actually reach.**

## 6. Repair transactional strategy (pick ONE, pin it)

`repairAfterLazy` mutates in place and **frees containers before committing `self.size`**; E3 adds a
failure point (the **deferred survivor header** allocation). Choose:

- **(a) Two-phase replacement** — build repaired key/container arrays beside the originals, commit
  only on full success; or
- **(b) Explicit rollback bookkeeping** — an undo log sufficient to restore a consistently
  deinit-able bitmap on failure at any point.

**Post-failure invariant (explicit):** the bitmap is **valid and deinit-able, no leak, no
double-free**; contents are either fully repaired or the documented partial state — never dangling.

## 7. Failure injection (positional)

At minimum: **first / middle / last demotion** position; **first / middle / last survivor-header**
allocation position; words allocation during construction; demote-array allocation. Each →
valid-or-cleanly-errored, **inputs untouched**, no leak.

## 8. Gates

- **Output invariants:** post-repair results match baseline lazyOr+repair in container kinds,
  cardinalities, and values; identical portable bytes where serialize is valid; CRoaring set-parity
  differential.
- **Dense survivor control:** construction, repair-only, combined — **one-sided
  `candidate / baseline ≤ 1.05`**, both hosts, with process-range analysis.
- **Zen 4 policy (spec 30):** within noise passes (repeated focused timing + range overlap); a real
  regression fails by default, adoptable only via explicit owner exception.
- **Board gate + spec-28 layout exception**; **one architecture-neutral shape**.

## Acceptance

- **lazyOr construction AND lazyOr+repair reach ≤ 1.10x on M4 SMP**; dense survivor control within
  the one-sided gate on both hosts; Zen 4 within noise.
- Tag renamed with the **compiler-enforced inventory complete**; the pinned clone / serialize /
  `serializedSizeInBytes` / `validate` / eager+many-op dispatch behaviors implemented and tested;
  activation-table coverage (incl. `lazyXor` / `lazyXorInPlace`) green; transactional repair with
  positional failure injection green; board gate held.
- Partial adoption per spec-30 policy (owner judgement; row stays open above 1.10x).
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast`; canonical
  `run-compare-bench.sh` both hosts; `docs/parity-measurement.md` updated.
