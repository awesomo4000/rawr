<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 10-20: `Roaring64Bitmap` internals refactor

**Refactor / prettification — the post-parity pass.** No behavior change, no new
API. Collapse the two structural duplications that accumulated across v1 + Phase-2.
Pairs with [10-19](10-19-has-run-containers-method.md) (the `hasRunContainers`
method dedup); this one is the 64-bit-internal production cleanup.

Do this **after** parity (10-07 … 10-18) is landed and green — which it is — so the
full set of call sites exists and the refactor is measured against a complete,
passing suite.

## Target 1 — key-span range decomposition (6 call sites)

Every range op open-codes the same two things: (a) for a bucket covering part of
`[lo, hi]`, compute its low-32 sub-range, and (b) walk the covered buckets. The
`(start_low, end_low)` computation

```zig
const start_low: u32 = if (bucket.hi == start_hi) lowBits(lo) else 0;
const end_low: u32 = if (bucket.hi == end_hi) lowBits(hi) else std.math.maxInt(u32);
```

appears verbatim in **six** methods: `addRange`, `removeRange`, `rangeCardinality`,
`containsRange`, `intersectsRange`, `flipInPlace`. Well over the 3+ threshold.

**Extract:** a small helper

```zig
fn bucketRangeBounds(bucket_hi: u32, start_hi: u32, end_hi: u32, lo: u64, hi: u64) struct { start_low: u32, end_low: u32 }
```

(or two inline helpers) and replace all six inlined copies with a call. Pure
computation, no state — trivially safe.

**Walk shape — leave as-is (do not over-abstract).** There are two distinct walk
shapes and they should *stay* distinct:
- **create-missing** key-cursor walk (`addRange`, `flipInPlace`, `containsRange`):
  iterates keys `lo_key..hi_key` as `u64`, find-or-creating buckets.
- **skip-missing** idx-cursor walk (`removeRange`, `rangeCardinality`,
  `intersectsRange`): `lowerBound(start_hi)` then advances over existing buckets.

Unifying these two into one comptime-parameterized iterator would tangle the
create-vs-skip and prune-vs-read logic for little gain — **only the bounds
computation is extracted**, not the walk. State this so the refactor doesn't
sprawl.

## Target 2 — two-way merge skeleton (unify the 2 identical)

`twoWayAllocatingMerge` and `twoWayCardinality` are the **same** merge skeleton
(`while i<a and j<b { compare hi → left-only / right-only / both }` + two drain
loops), differing only in what each case emits — and they already delegate the
per-case work to tiny helpers (`appendLeftOnly` vs `leftOnlyCardinality`, etc.).

**Decision (now safe to make):** unify them behind one comptime-parameterized
driver. The earlier reason to wait — "lazy ops will add more merge-walk callers" —
is **void**: CRoaring `roaring64` has no lazy ops, so no new callers are coming.
With exactly two identical instances and no growth, dedup them into a single
`twoWayMerge(comptime op, comptime Sink, ...)`-style driver where the sink decides
allocate-a-bucket vs accumulate-a-count.

**Leave `intersects` and `isSubsetOf` standalone.** They reuse the skeleton but
have early-exit / asymmetric control flow; folding them in hurts readability.
Two-of-four unified is the right cut.

## Constraints

- **Zero behavior change.** Same results, same pruning, same cardinality-cache
  handling, same OOM-rollback semantics. The existing `test64` / `validate64` /
  `difftest64` suites (property laws, randomized differential, serialization) are
  the safety net — they must stay green with no edits.
- Production 64-bit code only (`roaring64.zig`); no spec-behavior or API change.
- Keep helpers `inline`/private; no public surface added.

## Acceptance

- The `(start_low, end_low)` computation exists in **one** place; all six range
  methods call it.
- `twoWayAllocatingMerge` + `twoWayCardinality` share one driver; `intersects` /
  `isSubsetOf` untouched.
- `zig build test test64 validate64 difftest64` green, unchanged behavior (same
  seeds, same assertions pass). No diff in any oracle result.
