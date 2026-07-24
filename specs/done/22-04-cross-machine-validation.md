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
- **Commit the canonical table** to **`docs/parity-measurement.md`** (per host section), not
  left in ignored `misc/`.
- **Demote `bench_croaring`** in docs to a quick **screening dashboard** only — the canonical
  runner is the parity oracle.

## Acceptance

- The canonical table runs and validates on **both M4 and Zen 4**, each with its effective
  feature configuration recorded in the header.
- `run-compare-bench.sh` is canonical, portable across macOS Bash and Windows Git Bash, and
  emits the familiar full comparison table.
- The per-host canonical tables are committed to **`docs/parity-measurement.md`**.
- `bench_croaring` is documented as a screening dashboard, not the parity oracle.
- **Benchmark-only:** no production/library or vendored-source change; build green under
  `ReleaseSafe` and `ReleaseFast` on both hosts.

## Validation

- On **each** host (M4, Zen 4): `zig build test`; `zig build -Doptimize=ReleaseSafe`;
  `zig build -Doptimize=ReleaseFast`
- `scripts/run-compare-bench.sh` (macOS Bash on M4, Windows Git Bash on Zen 4) emits the full
  38-row table with the header recording that host's effective feature config
  (`croaring-avx512=on/off`, rawr `-Dcpu=native` features)
- both per-host tables committed to `docs/parity-measurement.md`; `bench_croaring` demoted to a
  screening dashboard in the docs

## Checklist

- [ ] Full 38-row table runs and validates on **both** M4 and Zen 4
- [ ] Each host's header records its effective feature config (`croaring-avx512=on/off`, rawr
      `-Dcpu=native`)
- [ ] `run-compare-bench.sh` canonical and portable (macOS Bash + Windows Git Bash)
- [ ] Per-host tables committed to `docs/parity-measurement.md`
- [ ] `bench_croaring` documented as a screening dashboard, not the oracle
- [ ] `zig build test`, ReleaseSafe, ReleaseFast green on both hosts; benchmark-only

## Result

The accurate, cross-validated parity board becomes the standing record — every row measured
per-tuple in fresh processes, validated, allocator-matched, `ns/op`-calibrated, and confirmed
on two architectures. Future parity claims are read off this, not the screening dashboard.
