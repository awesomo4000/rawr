<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 48-00: Harness, fixtures, references, accounting

Toplevel: [48-tiny-bitmap-cost-measurement.md](48-tiny-bitmap-cost-measurement.md).

**No reported measurements.** This chunk builds and validates the instrument. Every number in `48-01`
and `48-02` depends on the fixtures and accounting being right, so they are verified **before** any
result exists to be influenced by.

## 1. Deliverables

- **`Mtiny` lifecycle** exactly per toplevel §2.1, including object lifetimes across the deserialize
  (source and byte buffer stay live) — that is what makes checkpoint 4 mean peak concurrent residency.
- **Fixture pools** per §2 and §2.0.1: three shapes, 1,024 sets per (shape, cardinality), **pool size 1 at
  cardinality 0**, per-fixture variation as tabulated, `shape_id` as assigned.
- **Mixed corpus** per §7.1, with its **two separate PRNG streams** — Zipf draws cardinalities only,
  values seeded per bitmap from `corpus_index` and `cardinality`.
- **Both plain-list references** per §5 — byte reference and heap-owned reference. **Neither is a floor**;
  they bound a simple alternative.
- **Untimed accounting pass**: allocation/free counts, requested/live/peak bytes, allocation-size
  histogram, all five §6 checkpoints.
- **CRoaring accounting** via its **memory hooks**, including the **caller-owned serialization buffer**.
- **Validation** per §9, outside timing.

## 2. Correctness gates — these are the point of the chunk

- **Every fixture pool hashed and the hash checked in**, for all three shapes at every sweep cardinality
  *and* the mixed corpus. Assertion at generation.
- **Realized Zipf quantiles asserted**: median in `[1,2]`, p99 in `[1000, 20000]`. `s = 1.48` was computed
  against this band, so **a failure here means the sampler is wrong, not that the exponent needs tuning**
  (§7.1).
- **Teardown checkpoint proves zero** — every lifecycle returns to zero live bytes.
- **Validation passes**: deserialized cardinality and full value set match the input, and rawr↔CRoaring
  cross-deserialization round-trips.
- **Determinism check:** generate each pool **twice in separate processes** and confirm identical hashes.
  **This catches nondeterminism only** — uninitialised memory, address- or thread-dependent behaviour.
  *(An earlier draft claimed it catches a shared-PRNG-stream mistake. It does not: two deterministic
  processes produce the same output, including the same **wrong** output.)*

- **Shared-stream guard.** Check in **two** hashes: the **cardinality sequence on its own**, and the
  **full corpus**.

  *(An earlier draft claimed any value draw from the Zipf stream changes the cardinality sequence. Too
  broad: if the generator produces **all** cardinalities first and values afterwards, sharing the stream
  leaves the cardinality sequence untouched — only the values differ.)*

  So the two hashes catch different sharing patterns:

  | Sharing pattern | cardinality hash | full-corpus hash |
  | --- | --- | --- |
  | **interleaved** (per bitmap: draw cardinality, then values) | **fails** | fails |
  | **sequential** (all cardinalities, then all values) | unchanged | **fails** |

  **Mutation test:** deliberately wire value generation to share the Zipf stream and confirm the guard
  fires — **passing when EITHER hash fails**. The cardinality-only hash is retained as the isolation
  check that tells the two patterns apart. A guard that has never been seen to fail is not known to
  work.

### 2.2 Structural assertions — hashes bless whatever was generated

A checked-in hash freezes the corpus; it does not establish the corpus is *correct*. Validation (§1)
proves serialization preserves the generated set — **not** that the generated set matches the spec.

Assert **before** accepting any hash:

| Shape | Invariant |
| --- | --- |
| **localized** | exactly **one** distinct high key across the set |
| **one-per-container** | exactly **`cardinality`** distinct high keys |
| **spread** | **sorted ascending, unique, every value < 10,000,000** |
| all shapes | set size **equals the requested cardinality** |
| all nonzero pools | **1,024 pairwise-distinct sets** |

An initially wrong generator whose output got hashed would otherwise be locked in and look reproducible
forever.

## 3. Explicitly not in this chunk

- No timed cells, no curves, no crossovers, no projection.
- **No results reported.** A smoke run to prove the harness executes is fine; its numbers are not output.
- No design conclusions.

## Acceptance

- `Mtiny` implemented per **toplevel §2.1** with correct object lifetimes; batching in **whole pool cycles**
  (102,400 = 100 × 1,024).
- All fixture pools and the mixed corpus generated, hashed, hashes checked in.
- **Two-process determinism check passes** for every pool (nondeterminism only — see §2).
- **Both hashes checked in** — cardinality sequence and full corpus — and the **shared-stream mutation
  test fires on at least one of them**, for both the interleaved and sequential sharing patterns.
- **§2.2 structural assertions pass for every pool before its hash is accepted.**
- Zipf quantiles asserted and reported.
- Both plain-list references implemented per §5, described as references and not floors.
- Accounting pass produces all §6 checkpoints, out of band from any timing.
- CRoaring hooks capture allocations **including the caller-owned buffer**.
- Validation green outside timing.
- No board row moves; all four suites plus `check-32`, `check-docs`, `check-package` green.
- **No measurement results claimed.**

## Verification record — implemented, reviewed, ACCEPTED

Verified independently in the working tree, not taken on report.

| Item | Result |
| --- | --- |
| Production library untouched | ✓ — only `build.zig`, `tools/croaring_wrapper.h` (dev-only, **not** in `.paths`), and three new files |
| Both hashes pinned | ✓ — **36 sweep pool hashes** (3 shapes × 12 cardinalities) plus `expected_mixed_cardinality_hash` and `expected_mixed_full_hash` |
| Pool size 1 at cardinality 0 | ✓ — `if (cardinality == 0) 1 else sweep_pool_size` |
| Whole pool cycles | ✓ — `sweep_iterations = sweep_pool_size * 100` = **102,400** |
| Per-fixture reseed with `shape_id` | ✓ — `sweepValueSeed(shape, cardinality, fixture_index)` |
| Structural assertions run **before** hashing | ✓ — `validateFixture` is called per fixture inside `generateSweepPool`, before the pool is returned |
| All six §2.2 invariants present | ✓ — `CardinalityMismatch`, `NotSortedUnique`, `LocalizedTopologyMismatch`, `SpreadValueOutOfRange`, `OnePerContainerTopologyMismatch`, `DuplicateFixture` |

**`zig build check-tiny-setup` output, run here:**

```
quantiles: median=2 p99=4961
mutation interleaved: caught cardinality=true full=true
mutation sequential: caught cardinality=false full=true
tiny setup: OK
```

**The mutation result matches §2's predicted table exactly** — and the sequential row is the important
one: `cardinality=false, full=true`. That **empirically confirms the correction** to this spec. The
original single-hash design would have let sequential stream-sharing through undetected; only the
full-corpus hash catches it. The guard is now demonstrated, not assumed.

**Quantile note, so nobody later "fixes" it:** realized p99 is **4,961** against the **4,935** computed
by exact inverse-CDF during spec review. These differ by 0.5% because one is a *theoretical distribution
quantile* and the other a *realized sample quantile* at n=100,000. Both sit inside the asserted
`[1000, 20000]` band; the difference is sampling variation, not an error.

**Independent structural-guard control (not built into the harness).** The mutation test covers the PRNG
guards; the §2.2 structural assertions had none. I seeded a defect — changed `fillLocalized`'s stride from
`*7` to `*70000` so values spill across containers — and the guard fired with the correct named error:

```
error: LocalizedTopologyMismatch
```

**Follow-up LANDED** (committed `abdf517`): `verifyStructuralMutationGuards` seeds a defect for **all six**
named errors — `CardinalityMismatch`, `NotSortedUnique`, `LocalizedTopologyMismatch`,
`SpreadValueOutOfRange`, `OnePerContainerTopologyMismatch`, `DuplicateFixture` — wired into
`check-tiny-setup` as `mutation_structural`. Broader than the single case verified during review; every
structural guard is now self-verifying rather than depending on a reviewer checking it once.

Also noted: `verifyPinnedHashesAndFixtures` rejects a zero hash (`error.UnpinnedSweepHash` /
`UnpinnedMixedHash`), which catches the "table left unfilled" mode a hash comparison alone would not.

No timing claims made, per the chunk's scope.

## Estimate

**M** — the fixtures and the CRoaring accounting are the work.
