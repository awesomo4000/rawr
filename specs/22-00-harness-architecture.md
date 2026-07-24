<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 22-00: Harness architecture + manifest schema

First chunk of [accurate parity harness](22-accurate-parity-harness.md). **The foundation.**
Define the row-manifest **schema** and build the measurement framework the other chunks run
on, proven end-to-end on **two representative pilot rows**. No full audit here.

## Deliverables

### Manifest schema

Define the row-manifest record (the contract `22-01` populates and the harness executes) with
these fields per row:

- **stable row ID** + display name;
- **corpus / seed**;
- **rawr operation** and **CRoaring operation** (the matched pair);
- **allocating / non-allocating** classification;
- **allocator variants** (rawr SMP, rawr libc, CRoaring, rawr arena where applicable);
- **setup / teardown timing boundaries** (what is inside vs outside the timed region);
- **validation oracle** (byte-identity vs logical-equality, against what);
- **batch count + reporting unit** (`ms` or `ns/op`).

### Measurement framework

- **One fresh process per `(row, implementation, allocator)` tuple** — the runner spawns a
  separate process for each, so rawr-SMP / rawr-libc / CRoaring never share process or
  allocator state.
- **Two-level aggregation.** Each process runs a **pinned internal protocol**; the controller
  runs **≥ 5 independent processes per tuple** and aggregates the **median of per-process
  medians + full min/max range** (range, not IQR — matching `run-bench-parity-isolated.sh`).
- **Pin the internal timing protocol — not an implementation choice.** Fix the warmup count,
  timed-sample count, and median calculation (**3 warmup / 21 timed / median** unless a
  documented reason changes it), because earlier work proved 2/9 vs 3/21 materially changes SMP
  results. **Print it in the header** (e.g. `protocol=3w/21t median`), identical across every
  tuple and both hosts.
- **The manifest is the executable source of truth.** One **machine-readable registry**
  consumed by the worker; the controller enumerates rows via a **`--list`** (or equivalent)
  mode; **no duplicated row list hardcoded in the shell script**. The worker emits **structured
  output** per measurement containing: row ID, implementation, allocator, unit, batch count,
  median.
- **Validate without contaminating timing** — either a **separate validation process**, or
  **time first then validate untimed** before emitting the row. Never validate (which may
  allocate) before the timed region.
- **Output-table format** — the familiar full comparison table; header records target / CPU /
  effective features including **`croaring-avx512=on/off`** and rawr's `-Dcpu=native` features.
- **Portable runner** — the driving script works under **macOS Bash** and **Windows Git Bash**.

### Two pilot rows (prove the framework)

Implement the schema + framework end-to-end on **one allocating row** (e.g. `bitwiseAnd
(sparse)`, with SMP + libc) and **one tiny row** (e.g. `cardinality`, batched → ns/op) — enough
to exercise per-tuple isolation, validation-outside-timing, both reporting units, and the
header. The complete inventory is `22-01`.

## Acceptance

- Manifest schema defined and documented (the field set above), with the two pilot rows
  expressed as manifest entries.
- Runner spawns **one fresh process per tuple**, ≥5 processes each, aggregating median + full
  min/max range; validation runs outside the timed region.
- Both pilot rows produce correct table rows (validated against their oracle) with the header
  reporting target/CPU + `croaring-avx512=on/off`.
- Script runs under macOS Bash and Windows Git Bash.
- **Benchmark-only:** no production/library or vendored-source change; full build green under
  `ReleaseSafe` and `ReleaseFast`.

## Validation

- `zig build test`
- `zig build -Doptimize=ReleaseSafe` and `zig build -Doptimize=ReleaseFast`
- worker `--list` mode enumerates rows (here: the two pilot rows) — the controller consumes
  this, and no row list is hardcoded in the shell
- `scripts/run-compare-bench.sh` (or the pilot invocation) runs the two pilot rows per-tuple in
  fresh processes and emits the table + header (`protocol=…`, `croaring-avx512=on/off`)
- confirm on macOS Bash and Windows Git Bash

## Result to record

The manifest schema and the working framework — `22-01`..`22-04` build on it. Findings and the
canonical table land in the durable artifact **`docs/parity-measurement.md`**, not ignored
`misc/`.
