# Spec 07-05: N-way unions (`orMany` / `xorMany`)

Fifth piece of the [CRoaring parity effort](07-parity-inventory.md). Combine an
arbitrary list of bitmaps in one call. This is the last **perf-headline** parity
feature — n-way throughput is a CRoaring strength — but the deepest optimization
(lazy union + repair) is deliberately split into **`07-06`**. This chunk lands the
API, a correct **k-way merge** implementation, differential coverage, and a bench
that documents where we stand before lazy.

## Features

| rawr (new) | CRoaring | Semantics |
|---|---|---|
| `orMany(allocator, bitmaps)` | `roaring_bitmap_or_many` | union of all input bitmaps → new bitmap |
| `xorMany(allocator, bitmaps)` | `roaring_bitmap_xor_many` | symmetric difference of all (value present in an odd number of inputs) |

```zig
pub fn orMany(allocator: std.mem.Allocator, bitmaps: []const *const Self) !Self
pub fn xorMany(allocator: std.mem.Allocator, bitmaps: []const *const Self) !Self
```

Both produce a fresh bitmap (allocating). The `*Owned` variants are
**receiverless** too (unlike the binary `*Owned` ops, which take a receiver) —
they take the backing allocator and the list, and return an arena-backed
`OwnedBitmap`:

```zig
pub fn orManyOwned(backing: std.mem.Allocator, bitmaps: []const *const Self) !OwnedBitmap
pub fn xorManyOwned(backing: std.mem.Allocator, bitmaps: []const *const Self) !OwnedBitmap
``` **Out of scope here (→ `07-06`):** the
`*_many_heap` variants and the lazy/`repair_after_lazy` machinery — they're the
perf layer, built on top of this chunk's k-way structure.

## Edge cases

- empty list (`bitmaps.len == 0`) → empty bitmap.
- single element → a clone of it (don't alias the input).
- inputs are `*const` and must not be mutated.

## Task 0 — Wrapper decls

```c
roaring_bitmap_t* roaring_bitmap_or_many(size_t number, const roaring_bitmap_t** rs);
roaring_bitmap_t* roaring_bitmap_xor_many(size_t number, const roaring_bitmap_t** rs);
```

## Task 1 — `orMany` via k-way merge

The naive approach (clone `bitmaps[0]`, fold `bitwiseOrInPlace` over the rest) is
correct but re-normalizes and re-walks the whole accumulator N times. Implement
the **k-way merge** instead — build each output container exactly once:

- Keep a cursor per input bitmap into its `keys`/`containers` arrays.
- Repeatedly find the **minimum key** across the live cursor heads. For `07-05` a
  linear scan over the N heads is fine (a binary heap is the `or_many_heap`
  optimization — defer to `07-06`); note the O(N·distinct_keys) cost in a comment.
- For that key, **union all same-key containers** from the inputs that have it
  into one result container (fold over them — see the mutating-vs-not decision
  below), append it to the result, and advance those cursors.

This reuses the existing container union ops; the new code is the k-way key
merge. The per-key fold is exactly where `07-06`'s lazy path will later avoid
intermediate normalization — keep the per-key union isolated in a small helper so
lazy can swap it.

**Use the non-mutating `containerUnion`, not `containerUnionInPlace`.** This is a
firm decision, not a choice:
- `containerUnionInPlace` has an inconsistent contract — some branches mutate `a`
  and return the same pointer, others **allocate a new container and leave the
  old `a` for the caller to free** (e.g. `arrayUnionBitsetInPlace`,
  `arrayUnionArrayInPlace` on overflow, and the `run` branches which aren't
  actually in-place). Threading "free `a` iff the returned pointer ≠ `a`" through
  the fold is leak/double-free prone.
- The inputs are `*const`. The genuinely-in-place branches would **corrupt an
  input** if an input container were ever passed as the mutated `a`.
- `containerUnion` always returns a **fresh** container and never touches its
  args, giving uniform ownership.

Per-key fold with the non-mutating op:

```
acc = clone(first same-key container)   // owned accumulator
for each subsequent same-key input b (const):
    const next = try containerUnion(acc, b)   // always fresh
    acc.deinit(allocator)                     // free previous accumulator
    acc = next
append acc
```

The per-step alloc+free is exactly what `07-06`'s lazy path removes (accumulate
into the widest form, normalize once) — leaving the in-place optimization to lazy
keeps this chunk simple and the seam clean.

**Ownership discipline (in the per-key helper):**
- A key present in **only one** input → the result must **clone** that container,
  never append/alias the input's container (inputs are `*const` and the result
  owns its containers).
- The multi-input fold owns `acc` and each `next`; free the previous `acc` each
  step (above) and `errdefer acc.deinit(allocator)` so a mid-fold failure doesn't
  leak.

## Task 2 — `xorMany`

Same k-way merge skeleton, but **xor-accumulate** the same-key containers: a key
present in only one input contributes that container (cloned — same ownership
rule as Task 1, never alias the input); a key in several inputs folds
`containerXor` across them with the same clone-accumulator / free-previous pattern
as Task 1 (associative/commutative, so order within a key doesn't matter).
`containerXor` is already non-mutating (there is no `containerXorInPlace` at the
container level), so this matches Task 1's model directly. Drop a container that
folds to empty (cardinality 0) — same ghost-container discipline as `bitwiseXor`.
Free intermediates on every error path.

## Task 3 — Differential checks (`diff_test.zig`)

n-way results may pick a different *valid* container representation than CRoaring's
`or_many`/`xor_many` (same situation as flip/removeRange), so compare by
**semantic equality** (`assertSameValues`), not byte-identical `assertAgree`. (Try
byte-identity first if you like; if it diverges, semantic is the contract.)

- Build **N generated bitmaps** for several `N >= 1` (e.g. 1, 2, 3, 8, ~32) using
  the mixed generator across profiles, both `run_optimize` states. `orMany` /
  `xorMany` them in rawr; build the same N oracles and call the CRoaring
  counterpart; assert `assertSameValues`.
- **Hand-built heterogeneous-per-key case (explicit, not generator-based):**
  construct three inputs where the **same high key** is an **array** in one, a
  **bitset** in the second, and a **run** in the third (e.g. build each so that
  chunk lands in the intended container type, run-optimizing the third). `orMany`
  and `xorMany` over the three and assert against the oracle. This is the fold
  behavior that matters most; pin it deterministically rather than hoping the
  generator produces it.
- **Cross-check vs the 2-way op (pure rawr):** `orMany([a,b])` equals
  `a.bitwiseOr(b)`; `xorMany([a,b])` equals `a.bitwiseXor(b)`. And
  `orMany([a,b,c])` equals `a.bitwiseOr(b).bitwiseOr(c)` — cheap, catches fold
  bugs without the oracle.
- **Edge cases:** empty list and single element are tested **pure-rawr, without
  the oracle** — for `bitmaps.len == 0` assert the rawr result `isEmpty()`; for a
  single element assert it equals a clone and is independent (mutate the result,
  confirm the input is unchanged). Reserve the CRoaring oracle for `N >= 1`:
  calling `roaring_bitmap_or_many(0, ptr)` still requires a valid (even if
  dummy/`undefined`-but-readable) `const roaring_bitmap_t **` argument, so it's
  simpler to just not route `N == 0` through the oracle. Also test inputs that
  contain empty bitmaps in the list (`N >= 1`) against the oracle.

## Task 4 — Benchmark vs CRoaring

Extend `bench_croaring.zig`: bench **both** `orMany` (vs `roaring_bitmap_or_many`)
and `xorMany` (vs `roaring_bitmap_xor_many`) over a list of many bitmaps (e.g.
16–64 mixed-profile bitmaps). Record both ratios.

**Expectation/framing:** the k-way merge with eager per-key normalization will
likely still trail CRoaring, whose `or_many` is **lazy fold + repair internally**
(and `or_many_heap` is a separate balanced-merge path). That gap is **expected
and is exactly what `07-06` closes** — don't treat a >1.0× here as a failure;
record it as the baseline the lazy work will improve. (Keep both benches so
`07-06` can show the before/after.)

## Acceptance criteria

1. `orMany` / `xorMany` (+ receiverless `*Owned`) exist with the signatures and
   edge-case behavior above; inputs never mutated; single-element result is an
   independent clone.
2. Implemented as a k-way merge (each output container built once) using the
   **non-mutating** `containerUnion`/`containerXor` over a cloned accumulator (not
   `containerUnionInPlace`), with the per-key union isolated in a helper for
   `07-06` to swap; single-input keys cloned (never aliased), the previous
   accumulator freed each fold step, intermediates freed on error paths.
3. Match CRoaring `or_many` / `xor_many` by `assertSameValues` across varied
   `N >= 1`, profiles, and both run-optimized states — including the **hand-built
   array/bitset/run same-key** case — plus the pure-rawr 2-way cross-checks; empty
   list and single-element verified pure-rawr (no oracle).
4. **Both** `orMany` and `xorMany` benched vs their CRoaring counterparts with the
   ratios recorded (gap acceptable; `07-06` closes it).
5. No leaks (intermediate containers freed); `zig build test`, `validate`,
   `difftest` pass.

## Notes

- Deliberately omits `or_many_heap`/`xor_many_heap` and lazy/repair → `07-06`.
- The per-key union helper is the seam lazy will use (accumulate into the widest
  form, normalize once at the end) — keep it clean.
- Mark `or_many`/`xor_many` ✅ in the [inventory](07-parity-inventory.md) when
  done (leave the `_heap`/lazy rows for `07-06`).
