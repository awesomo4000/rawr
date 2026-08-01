<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 34: `select` container-skip kernel matrix (E4)

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md) (Wave 1). Close
**select (dense) 1.486x** — a **compute/branch** gap with **no allocation** (`select` allocates
nothing). The **top-level cardinality walk** (skip containers until the one holding the nth element)
dominates.

**Parity is a hard requirement** — closes at ≤ 1.10x; a partial is adopted by owner judgement
(spec-30 policy) and stays open.

## Kernel matrix (both hosts, canonical corpus)

1. **Current scalar walk** (baseline).
2. **2-container unrolled** walk.
3. **4-container unrolled** walk.
4. **Homogeneous-run specialization** — see the caveat below.
5. **Precomputed prefix-cardinality lookup** — **ceiling experiment only** (not a shippable shape
   as-is; bounds the maximum recoverable).
6. **rawr vs CRoaring disassembly + branch counts** on the canonical corpus.

## Homogeneous-run caveat (avoid repeating a rejected experiment)

An earlier **integrated run loop already regressed** (`docs/parity-measurement.md`). Cell 4 must
either **explain how homogeneous-run dispatch differs** from that integrated loop, or be **retained
explicitly as a control**, not a presumed candidate. **Prefix cardinalities (cell 5) are the
strongest ceiling experiment.**

## Tooling

- **Disassembly and focused timing are mandatory** on both hosts.
- **Branch-counter collection is best-effort where host tooling permits** — Apple M4 branch counters
  may not be reachable through the same tooling as Zen 4; a missing M4 branch count does not block
  the experiment.

## Decision rule

- If **unrolling or homogeneous dispatch** closes the gap → **ship, no storage change.**
- If **only prefix cardinalities** (the ceiling) close it, choose **explicitly** between:
  - **(a) optional caller-owned `RankSelectIndex`** — helps indexed users but **does not close the
    base row**; or
  - **(b) maintained bitmap metadata** — must pay **mutation and memory gates across the whole
    board**.
  - **Do not add a permanent index until the ceiling experiment proves it recovers the full 1.486x
    gap.**

## Measurement / gates

- **Both hosts, SMP, canonical protocol** (3 warmup / 21 timed, five process medians + full range),
  vs one CRoaring reference per host. E4 owns its own bench module (no shared-file edits).
- **No allocation** in `select` — accounting is timing + branch/disasm, not alloc counts.
- **Board gate + spec-28 layout exception** on any production adoption; **Zen 4 policy** (spec 30);
  **one architecture-neutral shape**.
- **Rebaseline note:** `select` walks container representations E1 may change — if E1 adopts first
  (Wave 2), **re-measure E4 after rebasing** onto the accepted E1 state before its board gate.

## Correctness

- **Representation-identical result** — `select(n)` returns the identical value to the baseline for
  all valid n, and the identical error/behavior at the boundary (n = cardinality, n = 0, empty
  bitmap); CRoaring differential across container-type mixes.

## Acceptance

- **Phase 1 GO:** the matrix run on both hosts with mandatory disasm + focused timing (best-effort
  branch counts); the ceiling (prefix cardinalities) established; a shippable shape identified **or**
  the explicit index decision framed with evidence. No production change.
- **Phase 2 (if a storage-free shape wins):** `select` closes to **≤ 1.10x M4 SMP** (or a beneficial
  partial adopted by owner judgement, row stays open), Zen 4 within noise, identical results, board
  gate held. **An index (a/b) is out of scope for this spec** unless the ceiling proves it recovers
  the full gap and a follow-up spec takes the board-wide mutation/memory gates.
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast` green; canonical
  `run-compare-bench.sh` both hosts on adoption; `docs/parity-measurement.md` updated.

## Proposed chunk plan (confirm at review)

- **`34-00`** — the kernel matrix (cells 1–6) both hosts, ceiling established, disasm/branch
  evidence; no production change. Decides shippable-shape vs index-decision.
- **`34-01`** — production kernel (conditional on `34-00`, only if a storage-free shape wins):
  identity, board gate, ship. (An index path, if ever justified, is a separate follow-up spec.)

## Estimate

M for `34-00` (six-cell matrix + ceiling + disasm, two hosts). S for `34-01` (storage-free kernel +
board gate) if applicable.
