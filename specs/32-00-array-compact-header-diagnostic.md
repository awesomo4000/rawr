<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 32-00: Array compact-header diagnostic

Toplevel: [32-compact-container-headers.md](32-compact-container-headers.md) (E1). Prototype the
compact **`ArrayContainer`** header (24 B → 16 B, 32-byte → 16-byte SMP slot, payload untouched) and
measure it. **Benchmark-only; no production default changed.** Produces the **Array GO/NO-GO**.

## Prototype (two tiers — see toplevel "Candidate execution mechanism")

- **Container-level replica cells** → **standalone replica struct** in an E1-owned diagnostic file
  (e.g. `bench_compact_header_array.zig`, so this runs concurrently with `32-01` — else serialize).
  Not an edit to `bench_single_alloc.zig`. No bitmap, no production edit.
- **Full-bitmap GO row** (`lazyOr(...,true)`) → **compile-time layout selection** switching
  `ArrayContainer` to the compact header (default build unchanged), built and measured in an
  **isolated diagnostic worktree** vs the committed baseline executable. **Do NOT duplicate bitmap
  ops** in the diagnostic.
- Compact `ArrayContainer`: `values` becomes `[*]align(32) u16` (8 B) + `cardinality: u16` +
  `capacity: u16` = 16 B. `cardinality` bounds reads; `capacity` controls growth/dealloc; use sites
  reconstruct a temporary slice (`ptr[0..cardinality]` / `ptr[0..capacity]`) so `ReleaseSafe` bounds
  checks hold.

## Corpus (pinned, assert before timing)

Canonical **sparse** corpus: `std.Random.DefaultPrng.init(54321)`, `500_000` values `int(u32)`,
sorted + deduped to `sparse_len` (`initSparseValues`); `a = sparse_values[0..half]`,
`b = sparse_values[half/2..]`, `half = sparse_len/2` (overlapping quarter-to-end) — the array-container
population the sparse 2-way lazy-OR merge clones.

## Cells (both hosts, SMP, canonical 3 warmup / 21 timed, five process medians + full range)

- **Container-level replicas:** reserved build, growth, clone, deinit, membership, iteration.
- **Isolated attribution cell:** clone the array-container population (counts header cost) —
  **attribution only.**
- **Full-bitmap GO row:** the actual canonical **`lazyOr(a, b, /*bitset_conversion=*/true)`**
  construction path, **reporting how many unmatched Array headers it clones** — the row Array E1 must
  move.

## Accounting + asserts (per cell)

- allocations, frees, requested bytes, **effective SMP-class bytes** (host class accounting),
  teardown — kept distinct.
- **`@sizeOf` 24→16** (compile-time); **`@alignOf` ≥ 4** asserted separately (tag fits).
- header now in the **16-byte class** (host accounting); payload **requested length / alignment /
  SMP class unchanged per case** (not assumed power-of-two).
- `ReleaseFast` for timing cells; `ReleaseSafe` for the correctness/bounds pass.

## Acceptance

- Corpus fingerprint asserted; all cells run both hosts; the five accounting figures reported per
  cell; 16 B header + 16-byte class + unchanged payload asserted.
- **Array GO requires movement in the full-bitmap `lazyOr(...,true)` row**, not the isolated clone
  cell alone. Record GO/NO-GO with both hosts' numbers.
- Named, committed diagnostic artifact; **no production default changed** (compile-time layout flag
  defaults off; full-row candidate lives in a worktree, unmerged).
- `zig build test` green; **both `ReleaseSafe` and `ReleaseFast` diagnostic runs green** — this chunk
  depends specifically on the reconstructed-slice `ReleaseSafe` bounds behavior, so run both modes;
  diagnostic section of `docs/parity-measurement.md` updated.
