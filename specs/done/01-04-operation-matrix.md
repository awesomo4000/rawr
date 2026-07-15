<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 01-04: The differential operation matrix

Chunk of [`01-differential-testing.md`](01-differential-testing.md). The breadth
payload: every operation, against the oracle, on every container-type pairing.

## Dependencies

- **01-03** (`assertAgree`, `buildOracle`, the difftest build).
- **01-02** (generator profiles).

## Carry-over facts

- rawr `bitwiseDifference` ⇄ CRoaring `andnot`.
- Run containers only after `runOptimize` — run every pairing **both
  run-optimized and not** (see 01-02 profile-type table).
- Probes/allocator/byte-diff conventions per 01-03.

## Task 1 — Producing operations (use `assertAgree` on the result)

For each, run on the same pair `(A, B)` in rawr and CRoaring, then `assertAgree`:

| rawr op                     | CRoaring op                       |
|-----------------------------|-----------------------------------|
| `bitwiseOr`                 | `roaring_bitmap_or`               |
| `bitwiseAnd`                | `roaring_bitmap_and`              |
| `bitwiseXor`                | `roaring_bitmap_xor`              |
| `bitwiseDifference`         | `roaring_bitmap_andnot`           |
| `bitwiseOrInPlace`          | `roaring_bitmap_or_inplace`       |
| `bitwiseAndInPlace`         | `roaring_bitmap_and_inplace`      |
| `bitwiseXorInPlace`         | `roaring_bitmap_xor_inplace`      |
| `bitwiseDifferenceInPlace`  | `roaring_bitmap_andnot_inplace`   |

For in-place ops: clone A first (`A.clone()` / `roaring_bitmap_copy`), mutate the
clone, then `assertAgree`. **Also** assert the in-place result equals the
allocating result (e.g. `bitwiseXorInPlace(clone, B)` ⇄ `bitwiseXor(A, B)`) — this
catches bugs where the allocating path is correct but the in-place path diverges.
Required for **all four** ops (currently only OR has this cross-check).

## Task 2 — Non-producing predicates (compare scalar/bool directly)

| rawr                | CRoaring                         |
|---------------------|----------------------------------|
| `andCardinality`    | `roaring_bitmap_and_cardinality` |
| `intersects`        | `roaring_bitmap_intersect`       |
| `isSubsetOf`        | `roaring_bitmap_is_subset`       |
| `equals`            | `roaring_bitmap_equals`          |
| `cardinality`       | `roaring_bitmap_get_cardinality` |
| `minimum`/`maximum` | `roaring_bitmap_minimum/maximum` |

## Task 3 — The 9-pair matrix (deterministic, not randomized)

Force each operand-type pairing in the **same chunk** (so containers actually
meet), run with `run_optimize` **off and on**:

```
(sparse,  sparse)   -> array  X array
(sparse,  dense)    -> array  X bitset
(dense,   sparse)   -> bitset X array     (asymmetric — test both orders!)
(dense,   dense)    -> bitset X bitset
(sparse,  runs)     -> array  X run
(runs,    sparse)   -> run    X array
(dense,   runs)     -> bitset X run
(runs,    dense)    -> run    X bitset
(runs,    runs)     -> run    X run
(full,    sparse)   -> full-chunk edge
(X,       empty)    -> every op against an empty operand
(empty,   X)        -> empty on the left
```

Order matters for ANDNOT/difference (non-commutative) — test **both** `A op B`
and `B op A`. AND/OR/XOR are commutative; still test both orders once to catch
order-dependent container-allocation bugs.

## Acceptance criteria

1. All 8 producing ops differentially checked via `assertAgree` on all 9
   container-pair combos + the full-chunk and empty-operand edges, run-optimized
   and not.
2. Both orderings tested for the non-commutative ops (difference/andnot), and at
   least once for the commutative ones.
3. In-place == allocating asserted for **all four** ops.
4. All predicates in Task 2 differentially checked.
5. `zig build difftest` runs the full matrix with zero failures; leak-checking
   allocator clean.

## Note

Container **transition** cases (promotion/demotion/empty-out/run-boundary) are
**not** here — they are 01-05.
