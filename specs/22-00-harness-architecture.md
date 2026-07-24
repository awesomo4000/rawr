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
- **≥ 5 independent processes per tuple**, aggregated to the **median of per-process medians +
  full min/max range** (range, not IQR — matching `run-bench-parity-isolated.sh`).
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

## Result to record

The manifest schema and the working framework — `22-01`..`22-04` build on it. Findings and the
canonical table land in the named durable artifact (e.g. `docs/parity-measurement.md`),
not ignored `misc/`.
