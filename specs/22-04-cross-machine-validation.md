<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 22-04: Cross-machine validation + canonical runner

Fifth chunk of [accurate parity harness](22-accurate-parity-harness.md). Run the trustworthy
canonical table on both reference architectures and make `run-compare-bench.sh` the canonical
accurate runner.

## Gate

- `22-02` and `22-03` complete: the full table is allocator-complete and tiny-op-calibrated
  (trustworthy).

## Deliverables

- **Run the canonical table on both hosts** — Apple M4 and the x86-64 / Zen 4 host — with the
  **same requested target/protocol** (`-Dcpu=native`, same warmup/timed protocol), and **each
  implementation's effective feature configuration recorded** per host (`croaring-avx512=on/off`
  and rawr's `-Dcpu=native` features). Every row validated outside timing on each host.
- **`run-compare-bench.sh` becomes the canonical runner** — produces the familiar full
  comparison table, and works under **macOS Bash** (M4) and **Windows Git Bash** (Zen 4).
- **Commit the canonical table** to the named durable artifact (e.g. `docs/parity-measurement.md`
  and/or a committed summary path), per host, not left in ignored `misc/`.
- **Demote `bench_croaring`** in docs to a quick **screening dashboard** only — the canonical
  runner is the parity oracle.

## Acceptance

- The canonical table runs and validates on **both M4 and Zen 4**, each with its effective
  feature configuration recorded in the header.
- `run-compare-bench.sh` is canonical, portable across macOS Bash and Windows Git Bash, and
  emits the familiar full comparison table.
- The per-host canonical tables are committed to the named artifact.
- `bench_croaring` is documented as a screening dashboard, not the parity oracle.
- **Benchmark-only:** no production/library or vendored-source change; build green under
  `ReleaseSafe` and `ReleaseFast` on both hosts.

## Result

The accurate, cross-validated parity board becomes the standing record — every row measured
per-tuple in fresh processes, validated, allocator-matched, `ns/op`-calibrated, and confirmed
on two architectures. Future parity claims are read off this, not the screening dashboard.
