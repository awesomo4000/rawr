<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 22-02: Allocator parity — manifest-driven allocator scope

Third chunk of [accurate parity harness](22-accurate-parity-harness.md). Completes the
allocator handling on the functional table so every allocating row is measured under the
correct, matched allocator conditions.

## Gate

- `22-01` complete: the functional table exists with every row ported and manifest entries
  flagged for allocator handling.

## Deliverables — allocator scope follows the manifest (per row)

- **Set ops on prebuilt inputs** (`bitwiseAnd/Or`, array AND/OR/xor, `lazyOr+repair`, n-way):
  inputs are **SMP-built and shared**; vary only the **result allocator** — report rawr-**SMP**
  and rawr-**libc** side by side against CRoaring-libc.
- **Construction ops** (`add`, `addMany`, `deserialize`) allocate **throughout**, so the whole
  operation runs under the variant's allocator (rawr-SMP and rawr-libc each end to end).
- **Distinct-boundary ops** (`serialize`, `toArrayAlloc`, `flip`, `removeRange`): the manifest
  names exactly which allocation is under test; measure under that boundary.
- **Arena variants** (`sparse AND arena`, `sparse OR arena`, `deserialize arena`): supplemental
  **rawr-arena** rows. CRoaring has no arena mode, so the arena row's CRoaring column **reuses
  the standard CRoaring baseline** (labeled), showing the arena's effect on rawr against the
  same reference.
- **Non-allocating ops** (`contains`, `iterate`, `toArray`, `cardinality`, rank/select,
  `andCardinality`, rangeCardinality): a **single rawr number** vs CRoaring — no SMP/libc split.

## Acceptance

- Every allocating row reports **rawr-SMP and rawr-libc** against CRoaring-libc, with the
  allocator scope (result-only / whole-op / named-boundary) matching its manifest entry.
- Arena rows present and labeled, CRoaring baseline reused as specified.
- Non-allocating rows report a single rawr number.
- All rows still validated outside timing; **validated logical outputs remain unchanged** —
  timing results are *expected* to change once the allocator scope is corrected, and that is
  fine; only the computed set/cardinality must stay identical.
- **Benchmark-only:** no production/library or vendored-source change; build green under
  `ReleaseSafe` and `ReleaseFast`.

## Validation

- `zig build test`; `zig build -Doptimize=ReleaseSafe`; `zig build -Doptimize=ReleaseFast`
- `scripts/run-compare-bench.sh` shows rawr-SMP and rawr-libc columns for every allocating row,
  arena rows labeled with the reused CRoaring baseline, and a single rawr number for
  non-allocating rows
- each row's logical output still validates against its oracle
