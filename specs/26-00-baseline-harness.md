<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 26-00: Baseline representation harness + strategy seam

First chunk of [direct range ops](26-direct-range-ops.md). **Infrastructure only — no
production behavior change.** Builds the equality harness, allocation instrumentation, and the
comptime strategy seam that `26-01`/`26-02` implement against and `26-03` gates on.

## Deliverables

- **Portable-byte equality harness (direct vs legacy).** For `flip`, `flipInplace`, and
  `removeRange`: run legacy and direct implementations on identical inputs and assert their
  **portable serializations are byte-identical**, plus **CRoaring set parity** as the
  independent oracle. Matrix (both implementations, all cases): range within one chunk;
  spanning chunk boundaries; whole-universe `[0, maxInt(u32)]`; empty bitmap; range over
  missing chunks; `lo`/`hi` at multiples of 65536 and ±1; full chunks at either range edge
  (interior-path check); all three container types on the edges, including conversions and
  drop-to-empty; `lo > hi` (in-place no-op; by-value flip returns an independent clone).
- **Comptime strategy seam.** A build option / comptime flag selecting
  `legacy` / `direct` (default: current behavior until `26-03` flips it) so **both
  implementations run on any host** — the harness runs the full matrix under both. This is the
  test seam the per-arch selection uses later if needed.
- **Allocation-count instrumentation** for the three ops (reuse the counting-allocator
  machinery): allocations per op under legacy recorded as the baseline the direct paths must
  collapse (the mask-bitmap and whole-clone allocations).
- **Overflow-safe iteration checks** in the harness corpus (the `[0, maxInt(u32)]` case
  exercises the no-`u16`-increment-through-65535 requirement).

## Acceptance

- Harness runs the full matrix; with only legacy present, legacy-vs-legacy passes trivially and
  the CRoaring parity oracle is green (validates the harness itself).
- Strategy seam builds and selects; `--help`/build docs name the flag.
- Legacy allocation baselines recorded for all three ops.
- No production behavior/API change; `zig build test`, `zig build difftest`, `ReleaseSafe`,
  `ReleaseFast` green.

## Checklist

- [ ] Byte-equality harness over the full case matrix, dual-implementation capable
- [ ] CRoaring set-parity oracle wired in
- [ ] Comptime strategy flag (`legacy`/`direct`), default = current behavior
- [ ] Allocation-count baselines recorded (mask/clone allocations visible)
- [ ] `[0, maxInt(u32)]` case present
- [ ] test / difftest / ReleaseSafe / ReleaseFast green; benchmark/test-only change
