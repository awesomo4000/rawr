# Spec 07-06b: Close the n-way `orMany` perf gap (+ heap variants)

Follow-on to [`07-06`](07-parity-inventory.md). Lazy fold brought `orMany` to
**1.25×** and `xorMany` to **0.55×** vs CRoaring. This chunk chases the residual
`orMany` gap and adds the `*_many_heap` parity APIs. We're already at parity, so
this is **optional polish** — but it's a clean, contained piece.

## Profile first (Task 0) — don't optimize blind

We've twice found the real perf cause only by looking, so **profile before
building.** The `orMany` hot path has three suspected per-distinct-key costs:

1. **Allocator + memset churn** — every multi-input key does
   `BitsetContainer.init` (alloc + `@memset` 8 KB) and a free; small-union keys
   then pay a repair demotion (alloc array + copy + free the bitset).
2. **Three full O(N) cursor scans per key** — `nextManyKey` + `foldManyKey`'s
   count loop + its accumulate loop, each walking all N cursors.
3. Repair walking the whole result again.

Measure which dominates (allocation count + a `perf`/timing pass on a 32-mixed
`orMany`). Implement the lever that matches:
- churn dominates → **Task 1 (workspace)** is the big win, likely makes Task 2
  unnecessary for perf;
- scan dominates → **Task 2 (heap)** is the move.

Record the finding in the commit. Expectation (to be confirmed, not assumed):
churn (#1) dominates.

## Task 1 — Reusable bitset workspace (the likely fix)

Replace per-key bitset allocation with **one scratch bitset reused across the
whole fold:**

- Allocate a single `BitsetContainer` workspace once in `manyMerge`.
- For each **multi-input** key: accumulate the same-key containers into the
  workspace with the raw lazy ops, **tracking the touched word range**
  `[min_word, max_word]`.
- **Materialize the output container directly** from the workspace: popcount the
  touched words for the cardinality; if `≤ ArrayContainer.MAX_CARDINALITY` build
  an **array** straight from the set bits; else **clone** the workspace words into
  an owned bitset; drop the key if the count is 0 (xor case).
- **Clear only the touched words** (`@memset(words[min..max+1], 0)`) before the
  next key — not the whole 8 KB.
- Single-input keys still clone as-is.

This removes the per-key alloc+zero+free (#1) **and** the repair demotion churn
(the output is right-typed at fold time), so the n-way path no longer needs a
separate `repairAfterLazy` pass — accumulate `cached_cardinality` as you go.
(Caveat: a bitset operand OR'd in touches all 1024 words, so those keys still pay
a full clear; array/run-heavy keys get cheap partial clears. The alloc/free
elimination is the win regardless.)

`repairAfterLazy` and the public `lazyOr`/`lazyXor` APIs from `07-06` are
**unchanged** — this only reworks the internal n-way fold.

## Task 2 — Heap k-way cursor + `*_many_heap` parity APIs

Addresses the cursor-scan cost (#2) and lands the remaining parity surface:

```c
roaring_bitmap_t* roaring_bitmap_or_many_heap(uint32_t number, const roaring_bitmap_t** rs);
roaring_bitmap_t* roaring_bitmap_xor_many_heap(uint32_t number, const roaring_bitmap_t** rs);
```

- Replace `nextManyKey`'s linear min-scan with a **binary heap** of cursors keyed
  by current head key, so each step yields the min key and its matching cursors in
  O(log N) without scanning non-matching inputs. Fuse the count/accumulate passes
  while you're there.
- Expose `orManyHeap`/`xorManyHeap` (rawr names) for parity. CRoaring's heap
  variant is a balanced pairwise merge tuned for **widely varying input sizes**;
  if rawr's k-way + heap already covers that case, `orManyHeap` can be a thin
  alias of `orMany` — but verify against the oracle and bench the varying-size
  case before deciding they're equivalent.

Only build Task 2 if Task 0 says the scan matters, **or** purely for the
`*_many_heap` parity API (in which case a thin alias may suffice).

## Task 3 — Tests

- **Behavior-preserving:** the `07-05`/`07-06` `orMany`/`xorMany` differential
  tests must still pass unchanged — the workspace rewrite changes performance, not
  results.
- **`*_many_heap` differential:** `orManyHeap`/`xorManyHeap` vs
  `roaring_bitmap_or_many_heap`/`xor_many_heap` by `assertSameValues`, across
  varied N and profiles, including a **widely-varying-input-size** case (the
  scenario the heap variant exists for).
- **`validate()` / `serialize`** on workspace-built results (the output is built
  directly now, so confirm the invariants hold without the repair pass).
- Leak check across the fold (the single workspace is freed once).

## Task 4 — Benchmark

Re-run the `07-06` n-way benches against the committed baseline (`orMany` 1.25×,
`xorMany` 0.55×) and record the after. Add an `orManyHeap` row if Task 2 lands.
Target: `orMany` ≤ ~1.0×. Note if the workspace alone gets there (Task 2 then
becomes parity-only).

## Acceptance criteria

1. Task 0 profile finding recorded in the commit; the implemented lever matches it.
2. `orMany`/`xorMany` results unchanged (`07-05`/`07-06` differential tests pass);
   workspace-built outputs pass `validate()` and `serialize`.
3. `orMany` bench improved from 1.25× toward ≤ ~1.0×, recorded before/after;
   `xorMany` not regressed.
4. If built: `orManyHeap`/`xorManyHeap` match `*_many_heap` oracles incl. a
   varying-size case; the single workspace is leak-free.
5. `zig build test`, `validate`, `difftest`, `bench-compare` pass.

## Notes

- `repairAfterLazy` and the public `lazyOr`/`lazyXor` stay as-is.
- Mark `or_many_heap`/`xor_many_heap` ✅ in the
  [inventory](07-parity-inventory.md) when done (or note them as thin aliases if
  that's the conclusion).
