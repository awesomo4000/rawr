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
- **Reproducibility check:** generate each pool **twice in separate processes** and confirm identical
  hashes. This is what catches a shared-PRNG-stream mistake, which is otherwise invisible — a single
  extra rejection in `spread` would shift every subsequent cardinality, and a hash produced once would
  simply bless it.

## 3. Explicitly not in this chunk

- No timed cells, no curves, no crossovers, no projection.
- **No results reported.** A smoke run to prove the harness executes is fine; its numbers are not output.
- No design conclusions.

## Acceptance

- `Mtiny` implemented per §2.1 with correct object lifetimes; batching in **whole pool cycles**
  (102,400 = 100 × 1,024).
- All fixture pools and the mixed corpus generated, hashed, hashes checked in.
- **Two-process reproducibility check passes** for every pool.
- Zipf quantiles asserted and reported.
- Both plain-list references implemented per §5, described as references and not floors.
- Accounting pass produces all §6 checkpoints, out of band from any timing.
- CRoaring hooks capture allocations **including the caller-owned buffer**.
- Validation green outside timing.
- No board row moves; all four suites plus `check-32`, `check-docs`, `check-package` green.
- **No measurement results claimed.**

## Estimate

**M** — the fixtures and the CRoaring accounting are the work.
