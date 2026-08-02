<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 34: `select` container-skip kernel matrix (E4)

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md) (Wave 1). Close
**select (dense) 1.486x** — a **compute/branch** gap with **no allocation** (`select` allocates
nothing). The **top-level cardinality walk** (skip containers until the one holding the nth element)
dominates.

**Parity is a hard requirement** — closes at ≤ 1.10x; a partial is adopted by owner judgement
(spec-30 policy) and stays open.

## Canonical corpus (pin invariants; assert before timing)

The bitmap under `select` is the canonical dense `a = addRange(0, 499999)` → **8 Run containers**
(keys 0–7) (`initRawrDenseBitmaps`). `select(k: u64) ?u32`. Assert before timing:

- **1,000,000 queries**, each **`uintLessThan(u32, 500_000)`**, from
  **`std.Random.DefaultPrng.init(12345)`** (`initTestData`), drawn as the **3rd** per-iteration value
  (order per `i`: `int(u32)` → `uintLessThan(500_000)` → **`uintLessThan(500_000)` ← select** → range
  draws). This is the exact existing draw — not a re-parameterized uniform.
- **8 Run containers** over `0..499999`; assert the container inventory.

## Kernel matrix (both hosts)

1. **Current scalar walk** (baseline). The **`noinline` full-`select` boundary** used here is the
   fixed measurement boundary for **all** cells.
2. **2-container unrolled walk** — **shippable candidate.** A **full-`select` implementation** that
   advances the top-level cardinality walk in **2-container groups**, with **identical dispatch and
   cardinality behavior** to the scalar walk and a **scalar tail** for the remainder; same `noinline`
   boundary.
3. **4-container unrolled walk** — **shippable candidate.** As cell 2 but **4-container groups** +
   scalar tail.
4. **Homogeneous-run integrated loop** — **historical control only, defined as the exact previously
   rejected integrated-run implementation** (`docs/parity-measurement.md`). Re-run **solely** to
   confirm/deny that regression on this corpus; **not a shipping candidate** in this spec.
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

The canonical corpus is **all-Run**. Before adopting any unrolled kernel, run pinned control cells to
ensure the winner does not regress non-Run or non-aligned shapes:

- **Array control** — a bitmap whose containers are **array** containers, built from
  `std.Random.DefaultPrng.init(54321)` sparse values (`initSparseValues`); select the same 1M
  `uintLessThan` queries clamped to its cardinality.
- **Bitset control** — a bitmap of **bitset** containers (e.g. the canonical `bitset_range` corpus,
  values `0..60000` step 3 per chunk → bitset), same query protocol.
- **Mixed control** — array + bitset + run containers in one bitmap (the `orMany`-style mixed shape).
- **Non-multiple-of-four container count** — a bitmap with a container count **not** divisible by 4
  (e.g. **5** or **7** containers) to exercise the unrolled tail.
- **Acceptable regression threshold: ≤ 5%** (board noise) on each control at the pinned seed. A
  kernel that wins all-Run but exceeds 5% on any control is **not** architecture-neutral and does
  **not** ship.

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
- **Boundary validation (`?u32` value / null)** — `select(k: u64)` returns the value or **`null`**
  when out of range (never an error). **`k = 0` is VALID for a nonempty bitmap** (returns the
  smallest element). Test each returns the **identical `?u32`** to baseline: **`0`**, **each prefix
  boundary** (first index of each container), **`cardinality - 1`** (last valid → value),
  **`cardinality`** (first out-of-range → `null`), **empty bitmap** (any k → `null`),
  **`maxInt(u32) + 1`** and **`maxInt(u64)`** (→ `null`).

## Acceptance

- **Phase 1 GO:** corpus invariants asserted; the matrix run on both hosts with mandatory disasm +
  focused timing (best-effort branch counts); the ceiling (prefix cardinalities) established with its
  build cost + footprint; mixed-container controls run. Output is **storage-free GO/NO-GO** and, if
  NO-GO, **whether a separate index spec is justified** — **not** an index architecture choice. No
  production change.
- **Phase 2 (if a storage-free shape wins):** `select` closes to **≤ 1.10x M4 SMP** (or a beneficial
  partial adopted by owner judgement, row stays open), Zen 4 within noise, identical results, board
  gate held. **An index is ALWAYS out of scope for this spec** — a winning ceiling does not authorize
  building one here; it only **authorizes a separate follow-up index spec** (which then takes the
  board-wide mutation/memory gates and the a/b architecture choice).
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
