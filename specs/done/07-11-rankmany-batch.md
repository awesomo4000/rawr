<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 07-11: Batch `rankMany` (single-sweep per container)

Post-umbrella perf fix. **Behavior-preserving** — no API/result change. Closes the
`rankMany (dense)` ~1.50× gap, the last genuine non-noise perf gap on the board.

## Root cause (confirmed in code)

rawr's `rankMany` cursors correctly at the *bitmap* level (advances
`container_idx`, accumulates `prior`), but for each probe it calls the
**single-probe** `containerRank` → `bitsetRank`, which popcounts **from word 0** to
the probe. For K probes landing in the same bitset container, that's **K
independent from-word-0 scans**.

CRoaring's `rank_many` batches at the *container* level (`container_rank_many`): it
consumes all consecutive probes in a container in **one forward word-sweep** with a
running popcount, emitting each rank as it passes. **O(words + K)** vs rawr's
**O(K × words)**.

(Note: this is why vectorizing `bitsetRank` regressed `rankMany` earlier — the
problem was never scalar-vs-vector popcount, it was re-scanning K times. The fix is
to not re-scan, not to make each re-scan faster. Leave `bitsetRank` as-is.)

## Task 1 — Per-container batch rank

Add `containerRankMany` mirroring CRoaring's `container_rank_many`, dispatching to
per-type batch helpers. Suggested shape:

```zig
/// Rank a run of consecutive probes (sorted ascending, all with this container's
/// high key) in a single pass. `base` is the rank accumulated from prior
/// containers. Writes results to `out`; returns the number of probes consumed.
pub fn containerRankMany(c: Container, base: u64, lows: []const u16, out: []u64) usize
```

(`lows` = the low-16 of the run of probes whose high key matches this container;
the bitmap-level caller slices them — see Task 2. Pass low-16 to keep it container-
local.)

Per type, single sweep with a running in-container count:
- **bitset** (`bitsetRankMany`): walk words once. Keep `running` (full-word
  popcount so far) and `word_idx`. For each probe `low` (ascending): advance
  `while word_idx < low>>6 : running += @popCount(words[word_idx]); word_idx += 1`,
  then `out = base + running + @popCount(words[word_idx] & mask≤bit)`. The partial
  masked popcount is recomputed per probe (cheap, one word); the full-word
  `running` accumulates once per word. **Mind the bit-63 mask** (don't shift by
  64 — reuse the `bitRangeMask`/`bitsetRank` masking).
- **array** (`arrayRankMany`): probes ascending, so walk the array with a single
  advancing index — for each probe, advance the cursor past values `≤ low`, emit
  `base + cursor`. No per-probe binary search.
- **run** (`runRankMany`): walk runs once with a running cardinality; for each
  probe, advance to the run containing/after it and emit accordingly.

Keep the single-probe `containerRank`/`bitsetRank`/etc. **unchanged** — they're
still used by `rank()` and `getIndex()`.

## Task 2 — Rewrite `rankMany` to use it

Keep the bitmap-level cursor (advance `container_idx`, accumulate `prior` over
fully-passed containers). When the run of probes matches the current container's
key, hand the whole run to `containerRankMany` in one call and skip past the
consumed probes — instead of looping `containerRank` per probe. Mirror CRoaring's
loop structure (advance container on `xhigh > key`, batch on `==`, emit `prior` on
`<`). Preserve the existing debug-asserts (`out.len == values.len`, sorted input).

## Verification

- **Behavior-preserving — no test changes.** The existing `rankMany` differential
  test (vs repeated `roaring_bitmap_rank`) and the rank/select consistency checks
  must stay green; that's the gate.
- **Bench:** re-run `rankMany (dense)` — expect it to drop from ~1.50× toward
  parity. Record before/after. Confirm `rank`/`select` rows unregressed (their
  single-probe paths are untouched).

## Acceptance criteria

1. `containerRankMany` + `bitsetRankMany`/`arrayRankMany`/`runRankMany` exist,
   single-sweep, correct at the bit-63 boundary.
2. `rankMany` uses the batch path (no per-probe `containerRank`); single-probe
   `rank`/`getIndex` unchanged.
3. `zig build test`/`validate`/`difftest` pass **unchanged**.
4. `rankMany (dense)` benched to ~parity; before/after recorded; `rank`/`select`
   unregressed.
5. No leaks.

## Notes

- Independent of the `orMany` residual — different subsystem, no shared code.
- After this, the board's only remaining >1.0× items are measurement noise and
  the unsolved `orMany` residual (which needs a profiling pass before any spec).
