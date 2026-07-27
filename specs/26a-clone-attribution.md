<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 26a: Clone-vs-mutation attribution + standalone clone row

Discovered at the close of [spec 26](done/26-direct-range-ops.md). Direct `removeRange` is
near-free (2 edge-container allocations, 0.411x Zen 4), yet the canonical M4 row still reads
**1.840x** — and that row is **clone-inclusive on both sides** (`clone + removeRange` vs
`roaring_bitmap_copy + remove_range_closed`; the op is destructive, so each rep clones first).
So the remaining number conflates two operations, and the near-free mutation makes **rawr's
clone the prime suspect on M4** — a hypothesis to measure, not assume. Meanwhile the board has
**no standalone clone row** at all (a gap first flagged in the spec-18 review), even though
`clone` is a real public operation.

**Diagnosis + board completion, no preselected cause.** A clone optimization, if warranted, is
a separate follow-up spec written around the attribution.

## Deliverables

### 1. Attribute the 1.840x: clone vs mutation (both hosts)

**A naive row delta is not mutation time.** Both canonical rows include destruction inside
timing, and the full clone and the range-reduced clone have **different teardown costs** — so
`(clone + remove + reduced-deinit) − (clone + full-deinit)` conflates mutation with the teardown
difference and can even go negative. Instead, add **matched-boundary diagnostic measurements
with internal timing** (the internal-timing precedent from the lazy-phase rows), per
implementation, on the same wide-dense corpus:

- **clone body only** — deinit outside timing;
- **removeRange body only** — one fresh clone created (untimed) before each timed invocation,
  deinitialized (untimed) afterward;
- **clone + removeRange body only** — deinit outside timing.

No pre-cloned pool is needed (each invocation's clone is created immediately before it), and no
nested timers. The **canonical clone and composite rows stay unchanged** for end-to-end
reporting; these diagnostics attribute, they do not replace. Report all three per host with
ranges and a **named residual** — the decomposition distinguishes **clone work vs mutation work
vs teardown vs a non-additive residual**, not just clone-vs-mutation.

**Representation inventory before any cause is named.** The wide-dense corpus is
`addRange`-built, which produces **run containers in rawr — not 8 KB bitsets** — so no component
hypothesis is assumed in advance. An **untimed inventory** records, for both implementations
where measurable: container types/counts, allocation counts/bytes, and **copied payload bytes**
(representation-neutral). Only then is the dominant component named — allocator traffic,
payload copying, or per-container overhead — via the counters + A/B methodology.

### 2. Standalone `clone` canonical row (manifest 38 → 39)

Add `clone (dense)` to the canonical board: rawr `clone` (SMP + libc — it is an allocating op)
vs `roaring_bitmap_copy`, on the wide-dense corpus so it composes with the `clone + removeRange`
row by inspection. Explicitly:

- **the row-count checks change 38 → 39** — update all **active executable assertions, scripts,
  and current documentation** (`--list`, `validateManifest`, runner checks, the current tables'
  prose); **completed specs and historical 38-row results stay historical**. This is the spec-23
  lesson about manifest-count changes, called out so it cannot be missed;
- validation: portable-byte identity of the clone against its source (rawr) and CRoaring set
  parity; validation outside timing per the standing rules;
- note in the manifest that CRoaring `copy` is measured with COW disabled (our build), so both
  sides deep-copy — the comparison is honest.

### 3. Row-shape verdict

Recommendation to confirm by review: **keep** the `clone + removeRange` row as-is (it reflects
real destructive-op usage and avoids the pre-cloned-pool contamination), and read it **alongside**
the new `clone` row — the pair makes the subtraction visible on the board itself. Re-scoping the
row to mutation-only is **rejected** unless the review finds a contamination-free way to measure
it directly.

## Conditional follow-up (not this spec)

- **Clone dominates M4** → a clone-optimization spec written around the named component
  (allocator traffic / payload copying / per-container overhead), with the usual gates:
  canonical harness, both hosts,
  Zen 4 no-regress (clone-only Zen 4 number becomes the baseline), correctness byte-identical.
- **Mutation dominates** (unexpected) → revisit direct `removeRange`'s edge work with that
  evidence.
- **Neither dominates / intrinsic** → document the residual; `removeRange` stays closed.

## Constraints / measurement

- Canonical protocol throughout: fresh process per `(row, implementation, allocator)` tuple,
  3w/21t median, ≥5 processes, median + full range, both hosts; baseline of record = the
  post-spec-26 canonical tables in `docs/parity-measurement.md`.
- Benchmark-only: no production/library change in this spec; the clone row and count-check
  updates are harness/manifest changes.
- Results + attribution recorded in `docs/parity-measurement.md`; no regression > 5% on any
  existing canonical row (the new row must not perturb others — it is additive).

## Acceptance

- The three matched-boundary diagnostics (clone body / removeRange body with per-invocation
  untimed clone / clone+remove body) measured both implementations × both hosts; the M4 1.840x
  attributed across **clone work / mutation work / teardown / non-additive residual**; the
  untimed representation + allocation inventory recorded **before** any component is named.
- `clone (dense)` row live on the canonical board; **every 38-row count check updated to 39**;
  row validated (byte identity vs source + CRoaring parity) outside timing.
- Row-shape verdict recorded (keep clone-inclusive + standalone pair, unless review overturns).
- `zig build test`; `zig build difftest`; canonical `run-compare-bench.sh` on both hosts;
  `ReleaseSafe` / `ReleaseFast` green.

## Estimate

S. One new canonical row + count-check updates + a clone-only measurement and a written
attribution. The conditional clone optimization, if any, is its own spec.
