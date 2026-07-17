<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 11-02: Branchless array→bitset conversion loops (portable)

Chunk of [array-kernel performance](11-array-kernel-perf.md). Replace per-element
`BitsetContainer.add` loops (run during every array-overflow conversion inside
union/xor) with raw word ops + a single cardinality repair. Behavior-preserving.

**Dependency order:** after [11-00](11-00-kernel-extraction-bench.md) — add the
array→bitset conversion case to `bench_aa` and measure against its baseline (work is
sequential, so depending on 11-00 is simpler than a throwaway microbench). No
dependency on 11-01.

## Exact call-site list

Convert each; confirm exhaustiveness by grepping per-element `bc.add(` /
`contains`-then-`remove`/`add` loops over `BitsetContainer` and record additions in
the PR:

- `arrayXorArray` overflow path — `container_ops.zig:1315`
- `arrayUnionArray` overflow path — `container_ops.zig:541`
- `bitsetUnionArrayInPlace` — `container_ops.zig:417`
- `bitsetXorArrayInPlace` — `container_ops.zig:489`
- `arrayUnionBitsetInPlace`
- `arrayUnionBitset`
- `arrayUnionRun`
- `arrayXorBitset`
- `arrayXorRun`
- `arrayToBitset`
- `ArrayContainer.unionInPlace` overflow conversion

Any site that turns out **not** to contain the per-element bitset-fill loop is
excluded with a one-line reason in the PR; the rest are converted.

## Root cause + replacement

`BitsetContainer.add` carries a bounds shift, a `was_absent` test, and a
`cardinality >= 0` branch **per element**; the XOR path additionally does `contains`
→ branch → `remove`/`add`.

```zig
// union: array elements into bitset words
for (ac.values[0..ac.cardinality]) |v|
    bc.words[v >> 6] |= @as(u64, 1) << @truncate(v);
bc.invalidateCardinality();

// xor: toggling needs no membership test
for (values) |v|
    bc.words[v >> 6] ^= @as(u64, 1) << @truncate(v);
bc.invalidateCardinality();
```

## Cardinality postcondition (explicit)

`internalValidate` **rejects a bitset with unknown cardinality (`-1`)**, so a
container returned from a *normal* (non-lazy) union/xor must carry a **valid**
cardinality: compute it once with the vectorized `computeCardinality()`
(`countWords`) before returning. **Only explicitly lazy APIs** (contract already
says "repair required before use") may leave `-1`. State this in each converted
function's doc comment.

**Semantic hazard:** the loops leave `cardinality == -1` transiently. Every reader
on the affected paths must use the container `getCardinality()` accessor (computes
on `-1`; named to avoid ambiguity with bitmap-level cardinality APIs), never a
direct `.cardinality` field read. Audit array-demotion comparisons
(`<= MAX_CARDINALITY`) and serialization sizing on these paths and route any direct
read through the accessor or an explicit `computeCardinality()`.

Optional (measure): CRoaring's `bitset_set_list` tracks the previous word to avoid
the read-modify-write chain; skip unless the plain loop disappoints.

## Under-threshold demotion (union invariant — required)

`arrayUnionArray` decides "go to bitset" from the **upper bound**
`a.cardinality + b.cardinality`. For overlapping inputs the **actual** union can be
`≤ 4096` (`MAX_CARDINALITY`) — e.g. union of two identical 3000-element arrays is
3000. Returning a bitset there produces a container `internalValidate` **rejects**
(a bitset must have `> 4096` elements) and whose portable serialization is invalid.
The conversion rewrite touches exactly these paths, so it must fix the invariant:

- **After computing the actual cardinality**, a **non-lazy** bitset result with
  `getCardinality() <= MAX_CARDINALITY` is **demoted back to an array** before
  returning.
- **`ArrayContainer.unionInPlace`**: if its temporary bitset ends `≤ 4096`, copy the
  result back into the array and **remain an array** — never return the invalid
  bitset.
- **Where the exact cardinality is already known** (e.g. `arrayToBitset` from a
  unique array), **assign it directly** instead of always scanning the full 8 KB with
  `countWords`.
- **Regression tests:** duplicate-heavy `arrayUnionArray` (identical arrays,
  heavy-overlap arrays) whose true union is `≤ 4096` — assert the result is a **valid
  array** (passes `internalValidate`, round-trips through serialization), plus the
  same for `unionInPlace`.

## Acceptance

- Differential suites green (`zig build test test64 validate validate64 difftest
  difftest64`); `internalValidate` accepts all non-lazy results (no `-1` escapes).
- Perf: add the **4096-element array→bitset union conversion** case to `bench_aa`
  and **record the pre-change baseline first** (direct word writes + a full 8 KB
  `countWords` scan may not beat the per-element loop by 2× in every scenario).
  Initial gate = **no regression + a measured improvement**; 2× is a target to
  confirm, not an assumed result.
