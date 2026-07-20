<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 13-00: Single-alloc prototype + D2 decision

First chunk of [single-allocation container layout](13-single-alloc-containers.md).
**Measurement, not implementation.** It answers two questions with numbers before
any production layout is written:

- **Q1 (whole-spec go/no-go):** does one allocation per array container (vs today's
  two) actually reduce allocation count and yield a net time win on allocation-heavy
  workloads — enough to justify the L-effort ABI change?
- **Q2 (D2):** stored-slice vs derived-accessor — is the pointer-chase win of derived
  accessors big enough to justify the repo-wide `.values`/`.runs` migration, or do we
  keep stored slices?

Output: both answers recorded back into the umbrella (D2 marked resolved with
numbers; go/no-go stated). Prototype code is throwaway; only the counting-allocator
util and the winning layout (as a seed for 13-03) survive.

## Isolation

The prototype must **not** touch production containers or the bitmap. Build it as a
standalone module + bench (e.g. `src/proto_single_alloc.zig` / a `bench-proto` step),
so `test`/`validate`/`difftest` are unaffected and there's no behavior change to the
library from this chunk.

## What to build

Three `ArrayContainer` variants exercised under one harness:

1. **Baseline** — the *current* two-alloc `ArrayContainer` (header via
   `allocator.create`, values via `alignedAlloc`). The number everything is compared
   against.
2. **Single-alloc, stored-slice** — one aligned block; `Header { cardinality: u16,
   capacity: u16, values: []Elem }` co-located, `values` points into the block.
   `dataOffset() = alignForward(@sizeOf(Header), 16)`; block =
   `dataOffset() + cap*@sizeOf(u16)`. **Refresh `values` ptr *and* length after every
   capacity change** (in-place resize *and* move), per the umbrella D2 rule.
3. **Single-alloc, derived-accessor** — one aligned block; `Header { cardinality:
   u16, capacity: u16 }`, `fn values(self) [*]u16 = @ptrCast(self) + dataOffset()`,
   never stored.

Each variant needs just enough to run the workloads: `init`, `add` (with grow via
`resize`-then-move), a read pass (membership / iterate / sum), `clone`, `deinit`.
Array alignment 16, block alignment ≥ 4 (trivially satisfied at 16). Use the shared
`dataOffset()`/`blockSize()` math from the umbrella.

## Counting-allocator harness (keep this — reused in 13-06)

A real, reusable `std.mem.Allocator` wrapper that tallies **alloc/free call counts
and total bytes** (wrapping any backing allocator). This is the tool that proves the
allocation-count claim; make it a proper test util, not throwaway.

## Fixed workloads (pin everything — reproducible + decisive)

State and hold constant: **N, container shapes, seed, and target/CPU** (reuse spec
14's env header to stamp zig/mode/os/arch/cpu on the output).

- **N = 10,000** array containers, fixed-seed `std.Random.DefaultPrng`.
- **Shapes:** a fixed mix — small (cardinality 1–64), medium (256–1024), near-threshold
  (≈4000) — so both tiny and large blocks are represented.
- **Workloads, each timed (median of ≥9) and allocation-counted:**
  - **build** — construct N containers from scratch (the alloc-count headline: expect
    baseline ≈ 2N allocs, single-alloc ≈ N).
  - **clone** — clone all N.
  - **read pass** — membership probes + full iterate + cardinality sum over all N (the
    pointer-chase workload that separates stored-slice from derived-accessor).
- Also record the **real accessor-migration count** for D2: grep `.values`/`.runs`,
  then hand-filter to the actual `ArrayContainer`/`RunContainer` field accesses (the
  ~207/~160 grep is a rough upper bound).

## Decision criteria

- **Q1 go/no-go (whole spec):** the build/clone allocation count must drop ~2N→N
  (near-certain by construction — confirm it), **and** single-alloc must be **no
  slower, ideally faster** than baseline on build/clone/read under the fixed config.
  If halving allocations yields no measurable time win on allocation-heavy workloads,
  that's a signal to **park the whole spec** — record the numbers and stop.
- **Q2 (D2):** pick **derived-accessor only if** its read-pass advantage over
  stored-slice is **clearly beyond run-to-run noise** *and* worth the measured
  migration count. Otherwise **default to stored-slice** (keeps every call site,
  avoids the repo-wide refactor). Record the chosen option + the numbers behind it.

## Deliverable

- A results table: {baseline, single-alloc stored-slice, single-alloc derived} ×
  {build, clone, read} for **alloc count, bytes, time**, plus the env header.
- The real accessor-migration count.
- **Edit the umbrella:** mark D2 **resolved** (chosen option + one-line numeric
  justification) and record the Q1 go/no-go outcome. If go, 13-01+ proceed; if no-go,
  the spec parks with the evidence attached.

## Acceptance

- Three variants + counting-allocator util + fixed workloads build and run under the
  isolated `bench-proto` step; no production container/bitmap change; `zig build test`
  unaffected.
- Baseline and both single-alloc variants measured; results table + accessor-migration
  count recorded.
- **D2 decided with numbers and written back into the umbrella; Q1 go/no-go stated.**
- The counting allocator lands as a reusable test util (for 13-06).
