# Spec 01-07: Property tests over the mixed generator + oracle-anchored identities

Chunk of [`01-differential-testing.md`](01-differential-testing.md). Splits into
two parts because of a **build constraint**: `property_tests.zig` runs inside
`zig build test` (pulled in via `src/roaring.zig`), and **that build does not link
CRoaring**. So the oracle cannot be called from `property_tests.zig`. Part A is
pure rawr there; Part B (oracle-anchored) lives in `diff_test.zig`.

## Dependencies

- **01-02** (`test_gen.randomMixed`) — for Part A and B.
- **01-03** (`assertAgree`, `buildOracle`) — for Part B.

## Part A — `property_tests.zig` (pure rawr, no oracle)

1. Swap `randomBitmap` for `test_gen.randomMixed` so the existing algebraic
   identities (commutativity, associativity, distributivity, De Morgan, xor
   decomposition, absorption, etc.) run over **bitset/run** containers instead of
   only arrays. This alone is a large coverage win — the identities currently
   never see a non-array container.
2. Increase iteration counts modestly (e.g. 50 → 200) now that each iteration
   covers more container variety. No AFL-scale volume.

**Do not import `c` in `property_tests.zig`.** `test_gen.zig` stays pure rawr so it
imports cleanly into the unit-test build; keep it that way.

## Part B — `diff_test.zig` (oracle available)

3. Add oracle-anchored versions of the most bug-revealing identities. The blind
   spot in Part A: an identity computed two ways in rawr (`A∩(B∪C)` vs
   `(A∩B)∪(A∩C)`) can pass even if **both** sides share the same bug. Anchoring at
   least one side to CRoaring removes that blind spot. For each anchored identity,
   compute the left side in rawr and assert it byte-matches the CRoaring result of
   the equivalent operation via `assertAgree`. A handful of the highest-value
   identities is enough (**distributivity, xor decomposition**) — they don't all
   need anchoring, since the 01-04 matrix already pins every individual op to the
   oracle.

## Division of labor

Part A proves rawr is *internally consistent* over all container types; the 01-04
matrix + Part B prove rawr *agrees with CRoaring*.

## Acceptance criteria

1. **Part A:** `property_tests.zig` runs its algebraic identities over
   `test_gen.randomMixed` (exercising bitset/run), iteration counts bumped
   modestly, no `c` import, passes under `zig build test` with a leak-checking
   allocator.
2. **Part B:** a handful of highest-value identities (distributivity, xor
   decomposition) are oracle-anchored in `diff_test.zig` via `assertAgree`, and
   pass under `zig build difftest`.
