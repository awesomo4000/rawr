<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 34-00: select kernel matrix diagnostic

Toplevel: [34-select-kernel-matrix.md](34-select-kernel-matrix.md) (E4). Run the `select` kernel
matrix + ceiling + controls, both hosts. **No production change.** Decides **storage-free GO/NO-GO**
and **whether a separate index spec is justified** — **not** index architecture.

## Corpus (assert before timing)

Bitmap under select: canonical dense `a = addRange(0, 499999)` → **8 Run containers** (keys 0–7).
`select(k: u64) ?u32`. Queries: **1M** from `std.Random.DefaultPrng.init(12345)`, each
`uintLessThan(u32, 500_000)`, the **3rd** per-iteration draw (`int(u32)` → `uintLessThan(500_000)` →
**select** → range draws). Assert the 8-container inventory.

## Matrix (both hosts; cells 1–5)

1. **Scalar walk** (baseline) — its **`noinline` full-`select` boundary** is the fixed boundary for
   all cells.
2. **2-container unrolled** — full-`select`, identical dispatch/cardinality behavior, scalar tail,
   same boundary. **Shippable candidate.**
3. **4-container unrolled** — as (2), 4-container groups. **Shippable candidate.**
4. **Prefix-cardinality ceiling** — **ceiling only.** Built **outside timing** (report build cost +
   footprint); `prefix[i] = Σ cardinalities of containers 0..i-1` (half-open); lookup = binary-search
   `prefix` then in-container select on `n - prefix[container]`; **same `noinline` boundary** as
   baseline.
5. **rawr vs CRoaring disassembly + branch counts** (disasm + focused timing mandatory both hosts;
   branch counters best-effort — M4 may lack them).

*(No homogeneous-run cell — the prior integrated-run regression is documented, not recreated.)*

## Mixed-container controls (before any all-Run winner ships)

Pinned, all 8 containers except the tail; **re-seed `DefaultPrng.init(12345)` independently per
control**, 1M `uintLessThan(u64, cardinality)` queries:

- **8 Array** — keys 0–7, contiguous low `0..2047`, cardinality 2048.
- **8 Bitset** — keys 0–7, even low `0..11998` (card 6000, multi-container).
- **8 Mixed** — `[array, bitset, run, array, bitset, run, array, bitset]` (run keys =
  `addRange(base, base+12000)`).
- **7 Run tail** — 8-Run build minus key 7 (count 7, exercises the unrolled tail).
- **Threshold: ≤ 5%** regression on each control, else not architecture-neutral.

## Correctness

- `select(n)` identical value to baseline for all valid n; CRoaring differential across type mixes.
- **Boundary (`?u32` / null):** `k = 0` valid (nonempty → smallest); test `0`, each prefix boundary,
  `cardinality - 1` (value), `cardinality` (null), empty (null), `maxInt(u32) + 1` (null),
  `maxInt(u64)` (null).

## Acceptance

- Corpus + controls asserted; cells 1–5 both hosts; ceiling established with build cost + footprint;
  disasm + focused timing done (best-effort branch counts).
- Output: **storage-free GO/NO-GO**, and if NO-GO **whether an index spec is justified** (ceiling
  recovers the full 1.486x gap and the row cannot close without stored metadata) — **not** an index
  architecture choice. No production change.
- `zig build test`; `zig build difftest` green; diagnostic section of `docs/parity-measurement.md`
  updated.
