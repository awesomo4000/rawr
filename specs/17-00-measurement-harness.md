<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 17-00: Phase A measurement harness + corpora

First chunk of the [transient-bitset arena](17-transient-bitset-arena.md) Phase A.
**Infrastructure only — no arena yet.** It builds the corpora and the measurement
harness that chunks `17-01` (A1) and `17-02` (A2) plug into, so their go/no-go numbers
are apples-to-apples and reproducible. Ships as benchmark-only code; changes no
production path.

## Deliverables

### Corpora (fixed, deterministic)

1. **Sparse 2-way** — reuse spec 16's sparse corpus verbatim (500k random `u32` dedup,
   the same two bitmaps `a`/`b`), so A1 is directly comparable to the spec-16 numbers.
2. **Sparse-heavy n-way** — an n-way input set with **many shared keys whose summed
   input cardinality per key is ≤ `ArrayContainer.MAX_CARDINALITY` (4096)**, so the
   guaranteed-demote eligibility fires and the transient path is actually exercised.
   Enough shared keys and operands that construction/repair/alloc/memory are stable, not
   sub-`0.01 ms` noise.
3. **Dense n-way control** — an n-way input set whose shared keys sum **above 4096**, so
   every key bypasses the arena. This is the no-regression control.

All corpora are seeded and byte-identical across variants and runs; document the seed,
operand counts, key-sharing structure, and per-key cardinality distribution.

### Harness

- **Construction/repair split** (spec 16 method): time lazy construction and repair
  separately, plus a combined figure. **Teardown (arena bulk free / result deinit) is
  inside the combined timed region** so free cost is never hidden.
- **Timing protocol:** five independent process runs, median + range/IQR per phase,
  identical setup/teardown applied to every variant, warmup excluded, result destruction
  excluded from *construction-only* and prepared outside the *repair-only* sample.
- **Counting allocator wrapper** over the child allocator that records, per variant:
  child-allocator call count, total requested bytes, **effective SMP size-class bytes**
  (the rounded slot, not the logical request), and **actual peak live (size-class)
  bytes**. `queryCapacity()` and requested bytes are reported as diagnostics but are
  distinct from the size-class peak that the Phase-A 110% memory gate is judged on.
- **Variant registry:** the harness runs each experiment across the allocator variants
  the later chunks register (baseline current-path, `ArenaAllocator`, exactly sized
  `FixedBufferAllocator`) and emits one row per (experiment, variant, phase).
- **Value-parity checks** wired in but exercised by the later chunks: a byte-identical
  comparison against rawr's current path, and a logical set/cardinality comparison
  against the CRoaring oracle.

## Acceptance

- The three corpora build deterministically; re-running produces byte-identical inputs.
  The sparse-heavy n-way corpus is shown (by the eligibility count the harness reports)
  to contain a substantial majority of guaranteed-demote keys; the dense control is
  shown to contain ~none.
- The harness runs a trivial placeholder variant end-to-end and emits the full metric
  row set (construction / repair / combined times with median+range; child calls;
  requested; SMP-class; peak) with teardown inside the combined timing.
- Counting-allocator numbers are validated against a hand-computed case (a known small
  input's alloc count and size-class bytes match).
- Benchmark-only: no production `lazyOr` / `manyMerge` / `repairAfterLazy` code changes;
  full build green under `ReleaseSafe` and `ReleaseFast`.

## Out of scope

No arena, no eligibility prepass, no A1/A2 logic — those are `17-01` and `17-02`. This
chunk is done when a placeholder can be measured and reported through the full pipeline.
