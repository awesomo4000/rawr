# Spec 01-05: Container-transition and edge cases

Chunk of [`01-differential-testing.md`](01-differential-testing.md). The
depth payload: four hand-built cases that target container promotion/demotion and
the ghost-container invariant — the logic in `optimize.zig` / container
promotion-demotion under operations, which is currently untested. The
empty-out-one-of-many case is the highest-value missing test in the whole spec.

## Dependencies

- **01-03** (`assertAgree`, `buildOracle`).
- **01-02** (generator) — though several cases are easier hand-built directly.

## Carry-over facts

- Container type thresholds: array → bitset promotion when cardinality exceeds
  4096; the inverse demotion happens when an op drops a bitset below that.
- Run containers only after `runOptimize`.
- rawr `bitwiseDifference` ⇄ CRoaring `andnot`. Allocator/probes/byte-diff per 01-03.

## Task — the four transition cases (each verified via `assertAgree`)

1. **Promotion.** Two `sparse` arrays in the same chunk whose **union** exceeds
   4096 → result must become a **bitset**. Construct A and B so their union
   crosses the threshold (and ideally each alone stays an array). `OR`, then
   `assertAgree` against the oracle. Optionally inspect the result container tag
   to confirm it is a bitset.

2. **Demotion.** Two `dense` bitsets whose **intersection** drops below 4096 →
   result should become an **array**. Construct B to share few elements with A.
   `AND`, then `assertAgree`. Optionally confirm the result container is an array.

3. **Empty-out-one-of-many (ghost-container invariant).** A has containers in
   chunks {0, 1, 2}; B equals A's chunk 1 exactly. `A andnot B` must **drop**
   chunk 1 entirely while chunks 0 and 2 survive. Assert the result has exactly
   **2 containers** and byte-matches the oracle. This is the highest-value case —
   it catches stale/ghost containers left behind after an op empties one.

4. **Run boundary.** A `full` chunk (all 65536, run-optimized) `andnot` a single
   value → verify the run splits correctly. Run-optimize on. `assertAgree`.

## Acceptance criteria

1. All four cases run and pass via `assertAgree` against CRoaring.
2. Promotion produces a bitset result; demotion produces an array result
   (assert the container tag, not just byte-equality, so a wrong-type-but-
   coincidentally-equal result is still caught).
3. Empty-out case: result has exactly 2 containers, chunk 1 gone, chunks 0 and 2
   intact, byte-identical to oracle.
4. Run-boundary case passes with `runOptimize` on.
5. Leak-checking allocator clean.
