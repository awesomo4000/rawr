# Spec 10-05: `Roaring64Bitmap` property + differential tests

Final piece of [64-bit Roaring](10-roaring64.md). Hardens the layer with
algebraic property tests and a randomized differential loop against CRoaring
`roaring64`, mirroring the 32-bit `src/property_tests.zig` and
`src/diff_test.zig`. By this chunk all v1 features (10-01 … 10-04) exist; this is
the confidence pass that lets us call the feature done.

## Task 1 — 64-bit generator

A generator that produces `Roaring64Bitmap`s exercising the **64-bit-specific**
structure that 32-bit generators can't reach:

- values clustered into several distinct high-32 keys (multi-bucket), including
  adjacent and far-apart keys;
- sub-bitmaps that land in each container type (array / bitset / run) so the
  64-bit frame is tested over every 32-bit payload shape;
- edge keys: `hi == 0`, `hi == 0xFFFFFFFF`, single-bucket, empty;
- values near `u64` boundaries (`0`, `u64` max, `0xFFFFFFFF`/`0x100000000`
  straddle).

Seed-logged and reproducible, same posture as `runRandomizedLoop` in
`diff_test.zig` (print `seed=0x…` so failures replay). Use a leak-checking GPA,
**not `c_allocator`**, per the rawr harness rule.

## Task 2 — Property tests (`roaring64_property_tests.zig`, run under `test64`)

Lift the 32-bit algebraic laws to 64-bit (no CRoaring oracle needed — these are
self-consistency properties):

- commutativity of ∪/∩/⊕;
- associativity of ∪/∩;
- distributivity `A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)`;
- identity `A ∪ ∅ = A`; idempotence `A ∪ A = A`, `A ∩ A = A`;
- complement `A − A = ∅`; self-xor `A ⊕ A = ∅` (**both must prune to a truly
  empty bitmap — `size == 0`**, the core invariant under algebraic cancellation);
- cardinality law `|A ∪ B| + |A ∩ B| = |A| + |B|`;
- subset/difference: `(A ∩ B) ⊆ A`, `(A − B) ⊆ A`;
- xor decomposition `A ⊕ B = (A − B) ∪ (B − A)`;
- absorption `A ∪ (A ∩ B) = A`;
- **positional round-trip:** for every element, `select(getIndex(v)) == v` and
  `rank(v) == getIndex(v) + 1`;
- **serialize round-trip:** `deserialize(serialize(A)).equals(A)`.

## Task 3 — Differential loop (`difftest64`)

Extend the `difftest64` program (built up across 10-01 … 10-04) into a randomized
loop mirroring `diff_test.zig`'s `runRandomizedLoop` / `runOperationMatrix`:

- for each generated pair/triple, build the parallel CRoaring `roaring64` bitmap
  (via `add_many`) and assert agreement on the **full v1 surface**:
  - membership, cardinality, min/max, `toArray`;
  - the 4 set ops (out-of-place + in-place) + 4 cardinality variants +
    intersects/subset/strict-subset/equals;
  - rank/select/getIndex (scalar);
  - addRange/removeRange/rangeCardinality/containsRange across key boundaries;
  - portable serialize round-trip both directions (rawr↔CRoaring).
- an `assertAgree64` helper (the analog of the 32-bit `assertAgree`) that
  compares a rawr `Roaring64Bitmap` to a CRoaring `roaring64_bitmap_t` by
  cardinality + element-wise membership. **Serialized-byte equality only after
  run-optimizing the oracle** (rawr `addRange`/`runOptimize` emit RUN containers
  the fresh oracle lacks — clone + `roaring64_bitmap_run_optimize` the CRoaring
  side first, per the 10-04 caveat), otherwise compare by cross-deserialize
  `equals` rather than raw bytes.
- iteration count tunable; default to a few thousand iters so the loop stays in
  the "fast" tier (seconds), per the harness layout.
- **Never call CRoaring `xor_inplace` / `andnot_inplace` with identical pointers**
  (it forbids `r1 == r2`; see 10-02). The oracle path for any in-place op uses two
  distinct CRoaring bitmaps; the self-aliased `A ⊕ A` / `A − A` cases are asserted
  rawr-only (result is empty), per the property pass in Task 2.

## Task 4 — Malformed-input smoke (deserialize)

A small fixed battery feeding crafted/truncated 64-bit buffers to
`deserializeSafe` — count overrun, non-ascending keys, truncated sub-bitmap,
empty sub-bitmap, zero-length input — asserting graceful errors (no panic, no
leak). Mirrors spec 01-09 / the 32-bit malformed-smoke posture; the per-bucket
payload hardening is already covered by the 32-bit path, so this only needs to
exercise the count+key frame.

## Acceptance

- 64-bit generator produces multi-bucket, multi-container-type, edge-key corpora,
  seed-logged and reproducible.
- All property tests pass under `test64` (and therefore under `test`, via the
  `roaring64.zig` import), including the positional and serialize round-trips and
  the **empty-after-cancellation prune** checks.
- `difftest64` randomized loop passes over its default iteration budget with the
  full v1 surface asserted against CRoaring `roaring64`; failures print a
  replayable seed.
- Malformed-input smoke passes (graceful errors, leak-free under the checking
  GPA).
- `zig build test test64 validate64 difftest64` all green; no 32-bit regression.
- Toplevel spec 10 umbrella acceptance satisfied → 64-bit v1 done; deferred
  parity items (flip, lazy, rankMany, jaccard, owned/frozen, benchmarks) recorded
  for a later pass.
