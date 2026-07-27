<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 26-00: Baseline representation harness + strategy seam

First chunk of [direct range ops](26-direct-range-ops.md). **Infrastructure only — no
production behavior change.** Builds the equality harness, allocation instrumentation, and the
comptime strategy seam that `26-01`/`26-02` implement against and `26-03` gates on.

## Deliverables

- **Portable-byte equality harness (direct vs legacy).** For `flip`, `flipInplace`, and
  `removeRange`: run legacy and direct implementations on identical inputs and assert their
  **portable serializations are byte-identical**, plus **CRoaring set parity** as the
  independent oracle. **Dependency split:** the direct-vs-legacy byte comparison lives in
  **pure-Zig unit tests** (no C dependency — `zig test src/<file>.zig` must keep working); the
  **CRoaring oracle lives in `diff_test`** (or another repository-only validation executable),
  never in the library's unit tests. Matrix (both implementations, all cases): range within one chunk;
  spanning chunk boundaries; whole-universe `[0, maxInt(u32)]`; empty bitmap; range over
  missing chunks; `lo`/`hi` at multiples of 65536 and ±1; full chunks at either range edge
  (interior-path check); all three container types on the edges, including conversions and
  drop-to-empty; `lo > hi` (in-place no-op; by-value flip returns an independent clone).
- **Comptime strategy seam — internal, not a module contract.** An internal **`RangeStrategy`
  comptime parameter** on private helpers selects `legacy` / `direct`; the public methods choose
  the shipped default internally. Benchmark/test **build options may feed that parameter**, but
  the library itself must **not** require a generated `build_options` import — direct
  `zig test src/<file>.zig` keeps working, and no new public API is added. Default: current
  behavior until `26-03` flips it. Both implementations run on any host through this seam — the
  harness runs the full matrix under both, and it is the mechanism per-arch selection uses later
  if needed.
- **Allocation-count instrumentation** for the three ops (reuse the counting-allocator
  machinery): allocations per op under legacy recorded as the baseline the direct paths must
  collapse (the mask-bitmap and whole-clone allocations). **Baselines are recorded in
  `docs/parity-measurement.md`** (a named `misc/` summary may hold the raw run), not left only
  in ignored output.
- **Overflow-safe iteration checks** in the harness corpus (the `[0, maxInt(u32)]` case
  exercises the no-`u16`-increment-through-65535 requirement).
- **Non-enumerating oracle for huge-cardinality cases.** Flipping `[0, maxInt(u32)]` on an
  empty bitmap yields 2³² values — `assertSameValues`-style per-value iteration is effectively
  non-terminating there. For such cases the CRoaring parity check must **never materialize or
  iterate the full result**: **serialize rawr, deserialize with CRoaring, compare via
  `roaring_bitmap_equals`** against the CRoaring oracle; additionally compare **cardinality**
  and **selected boundary/sample membership** (0, 65535/65536 seams, `maxInt(u32)`, a few
  interior probes). Do the **reverse** (CRoaring → rawr deserialize + equality) where practical.

## Acceptance

- Harness runs the full matrix; with only legacy present, legacy-vs-legacy passes trivially and
  the CRoaring parity oracle is green (validates the harness itself).
- Strategy seam builds and selects; `--help`/build docs name the flag.
- Legacy allocation baselines recorded for all three ops.
- No production behavior/API change; `zig build test`, `zig build difftest`, `ReleaseSafe`,
  `ReleaseFast` green.

## Checklist

- [ ] Byte-equality harness over the full case matrix, dual-implementation capable, pure-Zig
      (no C dependency in unit tests; `zig test src/<file>.zig` works)
- [ ] CRoaring set-parity oracle wired into `diff_test` (repo-only executable)
- [ ] Internal `RangeStrategy` comptime parameter, default = current behavior; no generated
      build_options import required by the library; no new public API
- [ ] Allocation-count baselines recorded (mask/clone allocations visible)
- [ ] `[0, maxInt(u32)]` case present, with the **non-enumerating** oracle path
      (serialize → CRoaring deserialize → `roaring_bitmap_equals` + cardinality + sample probes)
- [ ] test / difftest / ReleaseSafe / ReleaseFast green; benchmark/test-only change
