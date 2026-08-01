<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 34: `select` container-skip kernel matrix (E4)

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md) (Wave 1). Close
**select (dense) 1.486x** — a **compute/branch** gap with **no allocation** (`select` allocates
nothing). The **top-level cardinality walk** (skip containers until the one holding the nth element)
dominates.

**Parity is a hard requirement** — closes at ≤ 1.10x; a partial is adopted by owner judgement
(spec-30 policy) and stays open.

## Canonical corpus (pin invariants; assert before timing)

The `select` row: **1,000,000 deterministic select probes over dense range `0..499999`**
(`bench_parity_worker.zig`) → **8 Run containers** (keys 0–7). Assert before timing:

- **1,000,000 seeded queries** (state the seed), **values `0..499999`**, **8 Run containers**;
- the **query distribution** (e.g. uniform over `[0, cardinality)` at the pinned seed) — fixed so
  cells are comparable.

## Kernel matrix (both hosts)

1. **Current scalar walk** (baseline).
2. **2-container unrolled** walk — **shippable candidate**.
3. **4-container unrolled** walk — **shippable candidate**.
4. **Homogeneous-run specialization** — **CONTROL, not a presumed candidate.** A prior **integrated
   run loop already regressed** (`docs/parity-measurement.md`); we cannot yet articulate a concrete
   algorithm that provably differs, so cell 4 is measured to **confirm/deny that regression on this
   corpus**, not to ship. It is promoted to a candidate only if a concrete differing algorithm
   emerges *and* it wins without regressing the mixed-container controls below.
5. **Precomputed prefix-cardinality lookup** — **ceiling experiment only** (fully defined below);
   bounds the maximum recoverable, not shippable as-is.
6. **rawr vs CRoaring disassembly + branch counts** on the canonical corpus.

## Prefix-cardinality ceiling (fully defined)

- **Built outside timing** — the prefix table is constructed before the timed region (it is a
  ceiling, not a shippable maintained index); its **build cost and memory footprint are reported
  separately**.
- **Prefix convention** — `prefix[i] = sum of cardinalities of containers 0..i-1` (so container `i`
  holds selection indices `[prefix[i], prefix[i+1])`); state the exact half-open convention.
- **Lookup algorithm** — binary search `prefix` for the container, then the in-container select on
  the remainder `n - prefix[container]`.
- **Same boundary as baseline** — the timed region uses the **same `noinline` full-`select`
  boundary** as the baseline cell (no inlining advantage that the baseline lacks), so the ceiling is
  a like-for-like upper bound.

## Mixed-container performance controls (before shipping an all-Run winner)

The canonical corpus is **all-Run**. Before adopting any unrolled/specialized kernel, run **control
cells on Array, Bitset, and mixed-container bitmaps, and on a non-multiple-of-four container count**,
to ensure the winner does not regress non-Run or non-aligned shapes. A kernel that wins all-Run but
regresses these controls is **not** architecture-neutral and does not ship.

## Tooling

- **Disassembly and focused timing are mandatory** on both hosts.
- **Branch-counter collection is best-effort where host tooling permits** — Apple M4 branch counters
  may not be reachable through the same tooling as Zen 4; a missing M4 branch count does not block
  the experiment.

## Decision rule

- If an **unrolled walk** (cell 2/3) closes the gap **without regressing the mixed-container
  controls** → **ship, no storage change.**
- If **only prefix cardinalities** (the ceiling) close it, `34-00` decides **only**: (i)
  storage-free **NO-GO**, and (ii) **whether a separate index spec is justified** (i.e. the ceiling
  recovers the full 1.486x gap and the row cannot close without stored metadata). **`34-00` does NOT
  choose index architecture** — the choice between an optional caller-owned `RankSelectIndex` (helps
  indexed users, does not close the base row) and maintained bitmap metadata (pays board-wide
  mutation + memory gates) belongs to that **follow-up index spec**, which is out of scope here.
- **Do not add a permanent index in this spec** regardless of the ceiling result.

## Measurement / gates

- **Both hosts, SMP, canonical protocol** (3 warmup / 21 timed, five process medians + full range),
  vs one CRoaring reference per host. E4 owns its own bench module (no shared-file edits).
- **No allocation** in `select` — accounting is timing + branch/disasm, not alloc counts.
- **Board gate + spec-28 layout exception** on any production adoption; **Zen 4 policy** (spec 30);
  **one architecture-neutral shape**.
- **Rebaseline note:** `select` walks container representations E1 may change — if E1 adopts first
  (Wave 2), **re-measure E4 after rebasing** onto the accepted E1 state before its board gate.

## Correctness

- **Identical result** — `select(n)` returns the identical value to the baseline for all valid n;
  CRoaring differential across container-type mixes.
- **Boundary validation (corrected)** — **`n = 0` is VALID for a nonempty bitmap** (returns the
  smallest element), not an error. Test: **`0`**, **each prefix boundary** (the first index of each
  container), **`cardinality - 1`** (last valid), **`cardinality`** (first out-of-range),
  **empty bitmap**, **`maxInt(u32)`**, and **values above `cardinality`** — each matching the
  baseline's value or error exactly.

## Acceptance

- **Phase 1 GO:** corpus invariants asserted; the matrix run on both hosts with mandatory disasm +
  focused timing (best-effort branch counts); the ceiling (prefix cardinalities) established with its
  build cost + footprint; mixed-container controls run. Output is **storage-free GO/NO-GO** and, if
  NO-GO, **whether a separate index spec is justified** — **not** an index architecture choice. No
  production change.
- **Phase 2 (if a storage-free shape wins):** `select` closes to **≤ 1.10x M4 SMP** (or a beneficial
  partial adopted by owner judgement, row stays open), Zen 4 within noise, identical results, board
  gate held. **An index (a/b) is out of scope for this spec** unless the ceiling proves it recovers
  the full gap and a follow-up spec takes the board-wide mutation/memory gates.
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast` green; canonical
  `run-compare-bench.sh` both hosts on adoption; `docs/parity-measurement.md` updated.

## Proposed chunk plan (confirm at review)

- **`34-00`** — corpus invariants + the kernel matrix (cells 1–6) both hosts, ceiling established
  (build cost + footprint), mixed-container controls, disasm/branch evidence; no production change.
  Decides **storage-free GO/NO-GO** and **whether an index spec is justified** (not its
  architecture).
- **`34-01`** — production kernel (conditional on `34-00`, only if a storage-free shape wins):
  identity, board gate, ship. (An index path, if ever justified, is a separate follow-up spec.)

## Estimate

M for `34-00` (six-cell matrix + ceiling + disasm, two hosts). S for `34-01` (storage-free kernel +
board gate) if applicable.
